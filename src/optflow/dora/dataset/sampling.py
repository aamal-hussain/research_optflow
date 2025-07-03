"""Sampling for Dora. Lifted directly from Douglas' implementation since it is
better than the original.
"""

import logging
from typing import Sequence

import fpsample
import numpy as np
import point_cloud_utils as pcu


def perform_farthest_point_sampling(
    points: np.ndarray,
    num_samples: int,
    kdtree_height: int = 7,
) -> np.ndarray:
    """Perform farthest point sampling on points.

    Args:
        points: points to sample from
        num_samples: number of samples to take
        kdtree_height: height of the kd-tree to use for sampling.

    Returns:
        indices: indices of the sampled points
    """
    return fpsample.bucket_fps_kdline_sampling(points, num_samples, h=kdtree_height)


def sample_points_on_faces(
    vertices: np.ndarray,
    faces: np.ndarray,
    n_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample points uniformly on a mesh surface.

    Args:
        vertices: numpy array of shape (n_vertices, 3) containing vertex coordinates
        faces: numpy array of shape (n_faces, 3) containing vertex indices for each face
        n_samples: number of points to sample

    Returns:
        numpy array of shape (n_samples, 3) containing sampled points
        numpy array of shape (n_samples,) containing face indices for each sampled point
    """
    face_indices, barycentric_coordinates = pcu.sample_mesh_random(vertices, faces, n_samples)
    points = pcu.interpolate_barycentric_coords(
        faces,
        face_indices,
        barycentric_coordinates,
        vertices,
    )
    return points, face_indices


def sample_points_on_edges(
    verts: np.ndarray,
    edges: np.ndarray,
    num_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample points uniformly on edges of a mesh.

    Args:
        verts: numpy array of shape (n_vertices, 3) containing vertex coordinates
        edges: numpy array of shape (n_edges, 2) containing vertex indices for each edge
        num_samples: number of points to sample
    Returns:
        numpy array of shape (num_samples, 3) containing sampled points
        numpy array of shape (num_samples,) containing edge indices for each sampled point
    """
    start_points = verts[edges[:, 0]]
    end_points = verts[edges[:, 1]]

    # Calculate edge lengths
    edge_vectors = end_points - start_points
    edge_lengths = np.linalg.norm(edge_vectors, axis=1)
    total_length = np.sum(edge_lengths)

    # Calculate probability distribution based on edge lengths
    edge_probabilities = edge_lengths / total_length

    # Randomly select edges based on their probabilities
    edge_indices = np.random.choice(np.arange(len(edges)), size=num_samples, p=edge_probabilities)

    # Generate random parameters for interpolation (uniform along each edge)
    random_params = np.random.random((num_samples, 1))

    # Get the start and end points of the selected edges
    selected_starts = start_points[edge_indices]
    selected_ends = end_points[edge_indices]

    # Interpolate points along edges using the random parameters
    sampled_points = selected_starts + random_params * (selected_ends - selected_starts)

    return sampled_points, edge_indices


def sample_coarse_points_and_normals(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_normals: np.ndarray,
    num_samples: int,
    oversample_factor: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Process a mesh to detect sharp edges and sample points on the mesh surface.

    This function generates the surface points (fps_sharp_surface and fps_coarse_surface).

    Args:
        vertices: numpy array of shape (n_vertices, 3) containing vertex coordinates
        faces: numpy array of shape (n_faces, 3) containing vertex indices for each face
        face_normals: numpy array of shape (n_faces, 3) containing normal vectors for each face
        num_samples: number of points to sample
        oversample_factor: factor to oversample the points (default: 1)

    Returns:
        coarse_points: sampled points on the mesh surface
        coarse_normals: normals of the sampled points
    """
    points, face_indices = sample_points_on_faces(vertices, faces, num_samples * oversample_factor)
    normals = face_normals[face_indices]
    indices = perform_farthest_point_sampling(points, num_samples)
    points, normals = points[indices], normals[indices]
    return points, normals


def get_edges_and_adjacent_faces(faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Get edges and adjacent faces from a mesh.

    Args:
        faces: numpy array of shape (n_faces, 3) containing vertex indices for each face

    Returns:
        edges: numpy array of shape (n_edges, 2) containing vertex indices for each edge
        adjacent_faces: numpy array of shape (n_edges, 2) containing face indices for each edge
    """
    face_indices = np.arange(len(faces))
    sorted_faces = np.sort(faces, axis=1)
    edges = np.vstack(
        (
            sorted_faces[:, [0, 1]],
            sorted_faces[:, [0, 2]],
            sorted_faces[:, [1, 2]],
        )
    )
    face_indices = np.tile(face_indices, 3)

    idx = np.lexsort(edges[:, ::-1].T, axis=0)
    edges = edges[idx]
    face_indices = face_indices[idx]

    closed_edge_idx = np.where(np.all(np.diff(edges, axis=0) == 0, axis=1))[0]

    edges = edges[closed_edge_idx]
    adjacent_faces = np.stack(
        (face_indices[closed_edge_idx], face_indices[closed_edge_idx + 1]), axis=1
    )

    return edges, adjacent_faces


def get_sharp_edges_with_normals(
    faces: np.ndarray,
    face_normals: np.ndarray,
    minimum_sharp_edge_angle: float,
    epsilon: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Get sharp edges and their normals from a mesh.

    Args:
        faces: numpy array of shape (n_faces, 3) containing vertex indices for each face
        face_normals: numpy array of shape (n_faces, 3) containing normal vectors for each face
        minimum_sharp_edge_angle: minimum threshold for sharpness in degrees
        epsilon: threshold for keeping edges when we average the adjacent face normals, avoiding
                the edge cases of colinear or antiparallel normals
    Returns:
        sharp_edges: numpy array of shape (n_sharp_edges, 2) containing vertex indices for each
                    sharp edge
        sharp_edge_normals: numpy array of shape (n_sharp_edges, 3) containing normal vectors for
                    each sharp edge
    """
    edges, adjacent_faces = get_edges_and_adjacent_faces(faces)
    edge_normals = face_normals[adjacent_faces]
    edge_cos_angles = np.einsum("ij,ij->i", edge_normals[:, 0], edge_normals[:, 1])
    edge_angles = np.rad2deg(np.arccos(np.clip(edge_cos_angles, -1.0, 1.0)))
    mean_edge_normals = edge_normals.mean(axis=1)
    mean_edge_normal_lengths = np.linalg.norm(mean_edge_normals, axis=-1)
    mean_edge_normals /= (
        mean_edge_normal_lengths[:, np.newaxis] + 1e-6
    )  # Add small value to stabilise division
    edge_is_sharp = (edge_angles >= minimum_sharp_edge_angle) & (mean_edge_normal_lengths > epsilon)
    sharp_edges = edges[edge_is_sharp]
    sharp_edge_normals = mean_edge_normals[edge_is_sharp]
    return sharp_edges, sharp_edge_normals


def sample_sharp_points_and_normals(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_normals: np.ndarray,
    num_samples: int,
    minimum_sharp_edge_angle: float,
    oversample_factor: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample points on sharp edges of a mesh.

    Args:
        vertices: numpy array of shape (n_vertices, 3) containing vertex coordinates
        faces: numpy array of shape (n_faces, 3) containing vertex indices for each face
        face_normals: numpy array of shape (n_faces, 3) containing normal vectors for each face
        num_samples: number of points to sample
        minimum_sharp_edge_angle: minimum threshold for sharpness in degrees
        oversample_factor: factor to oversample the points (default: 1)

    Returns:
        points: sampled points on the sharp edges
        normals: normals of the sampled points
    """
    sharp_edges, sharp_edge_normals = get_sharp_edges_with_normals(
        faces, face_normals, minimum_sharp_edge_angle
    )
    if len(sharp_edges) == 0:
        return sample_coarse_points_and_normals(
            vertices, faces, face_normals, num_samples, oversample_factor
        )

    points, sampled_indices = sample_points_on_edges(
        vertices,
        sharp_edges,
        num_samples * oversample_factor,
    )
    normals = sharp_edge_normals[sampled_indices]

    if points.shape[0] > num_samples:
        indices = perform_farthest_point_sampling(points, num_samples)
        points, normals = points[indices], normals[indices]

    return points, normals


def sample_points_and_signed_distance(
    vertices: np.ndarray,
    faces: np.ndarray,
    num_samples: int,
    starting_points: np.ndarray | None = None,
    standard_deviations: Sequence[float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample points and compute signed distance to a mesh.

    Args:
        vertices: numpy array of shape (n_vertices, 3) containing vertex coordinates
        faces: numpy array of shape (n_faces, 3) containing vertex indices for each face
        num_samples: number of points to sample
        starting_points: optional starting points for sampling
        standard_deviations: optional standard deviations for sampling
    Returns:
        points: sampled points
        signed_distance: signed distance to the mesh
    """
    if starting_points is not None:
        indices = np.random.choice(starting_points.shape[0], size=num_samples)
        points = starting_points[indices]
    else:
        points = np.random.uniform(-1.05, 1.05, (num_samples, 3))

    if standard_deviations is not None:
        standard_deviation = np.random.choice(standard_deviations, size=(num_samples, 1))
        points += standard_deviation * np.random.standard_normal(size=(num_samples, 3))

    signed_distance, _, _ = pcu.signed_distance_to_mesh(
        points,
        vertices.astype(points.dtype),
        faces,
    )
    return points, signed_distance
