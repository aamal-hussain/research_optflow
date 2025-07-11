"""SAP Implementation, taken directly from the SAP repo."""

import numpy as np
import torch
import torch.fft
import torch.nn as nn

from optflow.utils.sap_utils import (
    fftfreqs,
    grid_interp,
    img,
    point_rasterize,
    spec_gaussian_filter,
)


class DPSR(nn.Module):
    def __init__(self, res, sig=10, scale=True, shift=True):
        """:param res: tuple of output field resolution. eg., (128,128)
        :param sig: degree of gaussian smoothing
        """
        super(DPSR, self).__init__()
        self.res = res
        self.sig = sig
        self.dim = len(res)
        self.denom = np.prod(res)
        G = spec_gaussian_filter(res=res, sig=sig).float()

        self.omega = fftfreqs(res, dtype=torch.float32)
        self.scale = scale
        self.shift = shift
        self.register_buffer("G", G)

    def forward(self, V, N):
        """:param V: (batch, nv, 2 or 3) tensor for point cloud coordinates
        :param N: (batch, nv, 2 or 3) tensor for point normals
        :return phi: (batch, res, res, ...) tensor of output indicator function field
        """
        assert V.shape == N.shape  # [b, nv, ndims]
        ras_p = point_rasterize(V, N, self.res)  # [b, n_dim, dim0, dim1, dim2]

        ras_s = torch.fft.rfftn(ras_p, dim=(2, 3, 4))
        ras_s = ras_s.permute(*tuple([0] + list(range(2, self.dim + 1)) + [self.dim + 1, 1]))
        N_ = ras_s[..., None] * self.G  # [b, dim0, dim1, dim2/2+1, n_dim, 1]

        omega = fftfreqs(self.res, dtype=torch.float32).unsqueeze(
            -1
        )  # [dim0, dim1, dim2/2+1, n_dim, 1]
        omega *= 2 * np.pi  # normalize frequencies
        omega = omega.to(V.device)

        DivN = torch.sum(-img(torch.view_as_real(N_[..., 0])) * omega, dim=-2)

        Lap = -torch.sum(omega**2, -2)  # [dim0, dim1, dim2/2+1, 1]
        Phi = DivN / (Lap + 1e-6)  # [b, dim0, dim1, dim2/2+1, 2]
        Phi = Phi.permute(
            *tuple([list(range(1, self.dim + 2)) + [0]])
        )  # [dim0, dim1, dim2/2+1, 2, b]
        Phi[tuple([0] * self.dim)] = 0
        Phi = Phi.permute(
            *tuple([[self.dim + 1] + list(range(self.dim + 1))])
        )  # [b, dim0, dim1, dim2/2+1, 2]

        phi = torch.fft.irfftn(torch.view_as_complex(Phi), s=self.res, dim=(1, 2, 3))

        if self.shift or self.scale:
            # ensure values at points are zero
            fv = grid_interp(phi.unsqueeze(-1), V, batched=True).squeeze(-1)  # [b, nv]
            if self.shift:  # offset points to have mean of 0
                offset = torch.mean(fv, dim=-1)  # [b,]
                phi -= offset.view(*tuple([-1] + [1] * self.dim))

            phi = phi.permute(*tuple([list(range(1, self.dim + 1)) + [0]]))
            fv0 = phi[tuple([0] * self.dim)]  # [b,]
            phi = phi.permute(*tuple([[self.dim] + list(range(self.dim))]))

            if self.scale:
                phi = -phi / torch.abs(fv0.view(*tuple([-1] + [1] * self.dim))) * 0.5
        return phi


if __name__ == "__main__":
    model = DPSR((128, 128, 128), sig=0).to("cuda")

    from pyvista.examples import download_bunny

    mesh = download_bunny()
    mesh = mesh.clean(absolute=False, tolerance=1e-8)

    verts = np.array(mesh.points)
    verts = verts / 1.2 + 0.5
    normals = np.array(mesh.point_normals)

    vals = model(
        torch.from_numpy(verts.astype(np.float32))[None].to(device="cuda"),
        torch.from_numpy(normals.astype(np.float32))[None].to(device="cuda"),
    )

    print(vals.shape)
