"""SAP Utils stolen from SAP."""

import numpy as np
import torch


def fftfreqs(res, dtype=torch.float32, exact=True):
    n_dims = len(res)
    freqs = []
    for dim in range(n_dims - 1):
        r_ = res[dim]
        freq = np.fft.fftfreq(r_, d=1 / r_)
        freqs.append(torch.tensor(freq, dtype=dtype))
    r_ = res[-1]
    if exact:
        freqs.append(torch.tensor(np.fft.rfftfreq(r_, d=1 / r_), dtype=dtype))
    else:
        freqs.append(torch.tensor(np.fft.rfftfreq(r_, d=1 / r_)[:-1], dtype=dtype))
    omega = torch.meshgrid(freqs)
    omega = list(omega)
    omega = torch.stack(omega, dim=-1)

    return omega


def img(x, deg=1):  # imaginary of tensor (assume last dim: real/imag)
    deg %= 4
    if deg == 0:
        res = x
    elif deg == 1:
        res = x[..., [1, 0]]
        res[..., 0] = -res[..., 0]
    elif deg == 2:
        res = -x
    elif deg == 3:
        res = x[..., [1, 0]]
        res[..., 1] = -res[..., 1]
    return res


def spec_gaussian_filter(res, sig):
    omega = fftfreqs(res, dtype=torch.float64)  # [dim0, dim1, dim2, d]
    dis = torch.sqrt(torch.sum(omega**2, dim=-1))
    filter_ = torch.exp(-0.5 * ((sig * 2 * dis / res[0]) ** 2)).unsqueeze(-1).unsqueeze(-1)
    filter_.requires_grad = False

    return filter_


def grid_interp(grid, pts, batched=True):
    if not batched:
        grid = grid.unsqueeze(0)
        pts = pts.unsqueeze(0)
    dim = pts.shape[-1]
    bs = grid.shape[0]
    size = torch.tensor(grid.shape[1:-1]).to(grid.device).type(pts.dtype)
    cubesize = 1.0 / size

    ind0 = torch.floor(pts / cubesize).long()  # (batch, num_points, dim)
    ind1 = torch.fmod(torch.ceil(pts / cubesize), size).long()  # periodic wrap-around
    ind01 = torch.stack((ind0, ind1), dim=0)  # (2, batch, num_points, dim)
    tmp = torch.tensor([0, 1], dtype=torch.long)
    com_ = torch.stack(torch.meshgrid(tuple([tmp] * dim)), dim=-1).view(-1, dim)
    dim_ = torch.arange(dim).repeat(com_.shape[0], 1)  # (2**dim, dim)
    ind_ = ind01[com_, ..., dim_]  # (2**dim, dim, batch, num_points)
    ind_n = ind_.permute(2, 3, 0, 1)  # (batch, num_points, 2**dim, dim)
    ind_b = (
        torch.arange(bs).expand(ind_n.shape[1], ind_n.shape[2], bs).permute(2, 0, 1)
    )  # (batch, num_points, 2**dim)
    # latent code on neighbor nodes
    if dim == 2:
        lat = grid.clone()[
            ind_b, ind_n[..., 0], ind_n[..., 1]
        ]  # (batch, num_points, 2**dim, in_features)
    else:
        lat = grid.clone()[
            ind_b, ind_n[..., 0], ind_n[..., 1], ind_n[..., 2]
        ]  # (batch, num_points, 2**dim, in_features)

    # weights of neighboring nodes
    xyz0 = ind0.type(cubesize.dtype) * cubesize  # (batch, num_points, dim)
    xyz1 = (ind0.type(cubesize.dtype) + 1) * cubesize  # (batch, num_points, dim)
    xyz01 = torch.stack((xyz0, xyz1), dim=0)  # (2, batch, num_points, dim)
    pos_ = xyz01[1 - com_, ..., dim_].permute(2, 3, 0, 1)  # (batch, num_points, 2**dim, dim)
    pos_ = pos_.type(pts.dtype)
    dxyz_ = torch.abs(pts.unsqueeze(-2) - pos_) / cubesize  # (batch, num_points, 2**dim, dim)
    weights = torch.prod(dxyz_, dim=-1, keepdim=False)  # (batch, num_points, 2**dim)
    query_values = torch.sum(
        lat * weights.unsqueeze(-1), dim=-2
    )  # (batch, num_points, in_features)
    if not batched:
        query_values = query_values.squeeze(0)

    return query_values


def scatter_to_grid(inds, vals, size):
    dims = inds.shape[1]
    assert inds.shape[0] == vals.shape[0]
    assert len(size) == dims
    dev = vals.device
    # result = torch.zeros(*size).view(-1).to(dev).type(vals.dtype)  # flatten
    # # flatten inds
    result = torch.zeros(*size, device=dev).view(-1).type(vals.dtype)  # flatten
    # flatten inds
    fac = [np.prod(size[i + 1 :]) for i in range(len(size) - 1)] + [1]
    fac = torch.tensor(fac, device=dev).type(inds.dtype)
    inds_fold = torch.sum(inds * fac, dim=-1)  # [#values,]
    result.scatter_add_(0, inds_fold, vals)
    result = result.view(*size)
    return result


def point_rasterize(pts, vals, size):
    dim = pts.shape[-1]
    assert pts.shape[:2] == vals.shape[:2]
    assert pts.shape[2] == dim
    size_list = list(size)
    size = torch.tensor(size).to(pts.device).float()
    cubesize = 1.0 / size
    bs = pts.shape[0]
    nf = vals.shape[-1]
    npts = pts.shape[1]
    dev = pts.device

    ind0 = torch.floor(pts / cubesize).long()  # (batch, num_points, dim)
    ind1 = torch.fmod(torch.ceil(pts / cubesize), size).long()  # periodic wrap-around
    ind01 = torch.stack((ind0, ind1), dim=0)  # (2, batch, num_points, dim)
    tmp = torch.tensor([0, 1], dtype=torch.long)
    com_ = torch.stack(torch.meshgrid(tuple([tmp] * dim)), dim=-1).view(-1, dim)
    dim_ = torch.arange(dim).repeat(com_.shape[0], 1)  # (2**dim, dim)
    ind_ = ind01[com_, ..., dim_]  # (2**dim, dim, batch, num_points)
    ind_n = ind_.permute(2, 3, 0, 1)  # (batch, num_points, 2**dim, dim)
    # ind_b = torch.arange(bs).expand(ind_n.shape[1], ind_n.shape[2], bs).permute(2, 0, 1)
    ind_b = (
        torch.arange(bs, device=dev).expand(ind_n.shape[1], ind_n.shape[2], bs).permute(2, 0, 1)
    )  # (batch, num_points, 2**dim)

    # weights of neighboring nodes
    xyz0 = ind0.type(cubesize.dtype) * cubesize  # (batch, num_points, dim)
    xyz1 = (ind0.type(cubesize.dtype) + 1) * cubesize  # (batch, num_points, dim)
    xyz01 = torch.stack((xyz0, xyz1), dim=0)  # (2, batch, num_points, dim)
    pos_ = xyz01[1 - com_, ..., dim_].permute(2, 3, 0, 1)  # (batch, num_points, 2**dim, dim)
    pos_ = pos_.type(pts.dtype)
    dxyz_ = torch.abs(pts.unsqueeze(-2) - pos_) / cubesize  # (batch, num_points, 2**dim, dim)
    weights = torch.prod(dxyz_, dim=-1, keepdim=False)  # (batch, num_points, 2**dim)

    ind_b = ind_b.unsqueeze(-1).unsqueeze(-1)  # (batch, num_points, 2**dim, 1, 1)
    ind_n = ind_n.unsqueeze(-2)  # (batch, num_points, 2**dim, 1, dim)
    ind_f = torch.arange(nf, device=dev).view(1, 1, 1, nf, 1)  # (1, 1, 1, nf, 1)
    # ind_f = torch.arange(nf).view(1, 1, 1, nf, 1)  # (1, 1, 1, nf, 1)

    ind_b = ind_b.expand(bs, npts, 2**dim, nf, 1)
    ind_n = ind_n.expand(bs, npts, 2**dim, nf, dim).to(dev)
    ind_f = ind_f.expand(bs, npts, 2**dim, nf, 1)
    inds = torch.cat([ind_b, ind_f, ind_n], dim=-1)  # (batch, num_points, 2**dim, nf, 1+1+dim)

    # weighted values
    vals = weights.unsqueeze(-1) * vals.unsqueeze(-2)  # (batch, num_points, 2**dim, nf)

    inds = inds.view(-1, dim + 2).permute(1, 0).long()  # (1+dim+1, bs*npts*2**dim*nf)
    vals = vals.reshape(-1)  # (bs*npts*2**dim*nf)
    raster = scatter_to_grid(inds.permute(1, 0), vals, [bs, nf] + size_list)

    return raster  # [batch, nf, res, res, res]
