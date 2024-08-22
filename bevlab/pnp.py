# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
This file contains Efficient PnP algorithm for Perspective-n-Points problem.
It finds a camera position (defined by rotation `R` and translation `T`) that
minimizes re-projection error between the given 3D points `x` and
the corresponding uncalibrated 2D points `y`.
"""

import warnings
from typing import NamedTuple, Optional, Tuple, Union, List

import torch
import torch.nn.functional as F
#from pytorch3d.common.compat import eigh
#from pytorch3d.ops import points_alignment, utils as oputil

def eigh(A: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:  # pragma: no cover
    """
    Like torch.linalg.eigh, assuming the argument is a symmetric real matrix.
    """
    if hasattr(torch, "linalg") and hasattr(torch.linalg, "eigh"):
        return torch.linalg.eigh(A)
    return torch.symeig(A, eigenvectors=True)


# named tuples for inputs/outputs
class SimilarityTransform(NamedTuple):
    R: torch.Tensor
    T: torch.Tensor
    s: torch.Tensor

def is_pointclouds(pcl: Union[torch.Tensor, "Pointclouds"]) -> bool:
    """Checks whether the input `pcl` is an instance of `Pointclouds`
    by checking the existence of `points_padded` and `num_points_per_cloud`
    functions.
    """
    return hasattr(pcl, "points_padded") and hasattr(pcl, "num_points_per_cloud")


def convert_pointclouds_to_tensor(pcl: Union[torch.Tensor, "Pointclouds"]):
    """
    If `type(pcl)==Pointclouds`, converts a `pcl` object to a
    padded representation and returns it together with the number of points
    per batch. Otherwise, returns the input itself with the number of points
    set to the size of the second dimension of `pcl`.
    """
    if is_pointclouds(pcl):
        X = pcl.points_padded()  # type: ignore
        num_points = pcl.num_points_per_cloud()  # type: ignore
    elif torch.is_tensor(pcl):
        X = pcl
        num_points = X.shape[1] * torch.ones(  # type: ignore
            # pyre-fixme[16]: Item `Pointclouds` of `Union[Pointclouds, Tensor]` has
            #  no attribute `shape`.
            X.shape[0],
            device=X.device,
            dtype=torch.int64,
        )
    else:
        raise ValueError(
            "The inputs X, Y should be either Pointclouds objects or tensors."
        )
    return X, num_points


AMBIGUOUS_ROT_SINGULAR_THR = 1e-15
def corresponding_points_alignment(
    X: Union[torch.Tensor, "Pointclouds"],
    Y: Union[torch.Tensor, "Pointclouds"],
    weights: Union[torch.Tensor, List[torch.Tensor], None] = None,
    estimate_scale: bool = False,
    allow_reflection: bool = False,
    eps: float = 1e-9,
) -> SimilarityTransform:
    """
    Finds a similarity transformation (rotation `R`, translation `T`
    and optionally scale `s`)  between two given sets of corresponding
    `d`-dimensional points `X` and `Y` such that:
    `s[i] X[i] R[i] + T[i] = Y[i]`,
    for all batch indexes `i` in the least squares sense.
    The algorithm is also known as Umeyama [1].
    Args:
        **X**: Batch of `d`-dimensional points of shape `(minibatch, num_point, d)`
            or a `Pointclouds` object.
        **Y**: Batch of `d`-dimensional points of shape `(minibatch, num_point, d)`
            or a `Pointclouds` object.
        **weights**: Batch of non-negative weights of
            shape `(minibatch, num_point)` or list of `minibatch` 1-dimensional
            tensors that may have different shapes; in that case, the length of
            i-th tensor should be equal to the number of points in X_i and Y_i.
            Passing `None` means uniform weights.
        **estimate_scale**: If `True`, also estimates a scaling component `s`
            of the transformation. Otherwise assumes an identity
            scale and returns a tensor of ones.
        **allow_reflection**: If `True`, allows the algorithm to return `R`
            which is orthonormal but has determinant==-1.
        **eps**: A scalar for clamping to avoid dividing by zero. Active for the
            code that estimates the output scale `s`.
    Returns:
        3-element named tuple `SimilarityTransform` containing
        - **R**: Batch of orthonormal matrices of shape `(minibatch, d, d)`.
        - **T**: Batch of translations of shape `(minibatch, d)`.
        - **s**: batch of scaling factors of shape `(minibatch, )`.
    References:
        [1] Shinji Umeyama: Least-Suqares Estimation of
        Transformation Parameters Between Two Point Patterns
    """

    # make sure we convert input Pointclouds structures to tensors
    Xt, num_points = convert_pointclouds_to_tensor(X)
    Yt, num_points_Y = convert_pointclouds_to_tensor(Y)

    if (Xt.shape != Yt.shape) or (num_points != num_points_Y).any():
        raise ValueError(
            "Point sets X and Y have to have the same \
            number of batches, points and dimensions."
        )
    if weights is not None:
        if isinstance(weights, list):
            if any(np != w.shape[0] for np, w in zip(num_points, weights)):
                raise ValueError(
                    "number of weights should equal to the "
                    + "number of points in the point cloud."
                )
            weights = [w[..., None] for w in weights]
            weights = strutil.list_to_padded(weights)[..., 0]

        if Xt.shape[:2] != weights.shape:
            raise ValueError("weights should have the same first two dimensions as X.")

    b, n, dim = Xt.shape

    if (num_points < Xt.shape[1]).any() or (num_points < Yt.shape[1]).any():
        # in case we got Pointclouds as input, mask the unused entries in Xc, Yc
        mask = (
            torch.arange(n, dtype=torch.int64, device=Xt.device)[None]
            < num_points[:, None]
        ).type_as(Xt)
        weights = mask if weights is None else mask * weights.type_as(Xt)

    # compute the centroids of the point sets
    Xmu = wmean(Xt, weight=weights, eps=eps)
    Ymu = wmean(Yt, weight=weights, eps=eps)

    # mean-center the point sets
    Xc = Xt - Xmu
    Yc = Yt - Ymu

    total_weight = torch.clamp(num_points, 1)
    # special handling for heterogeneous point clouds and/or input weights
    if weights is not None:
        Xc *= weights[:, :, None]
        Yc *= weights[:, :, None]
        total_weight = torch.clamp(weights.sum(1), eps)

    if (num_points < (dim + 1)).any():
        warnings.warn(
            "The size of one of the point clouds is <= dim+1. "
            + "corresponding_points_alignment cannot return a unique rotation."
        )

    # compute the covariance XYcov between the point sets Xc, Yc
    XYcov = torch.bmm(Xc.transpose(2, 1), Yc)
    XYcov = XYcov / total_weight[:, None, None]

    # decompose the covariance matrix XYcov
    U, S, V = torch.svd(XYcov)

    # catch ambiguous rotation by checking the magnitude of singular values
    if (S.abs() <= AMBIGUOUS_ROT_SINGULAR_THR).any() and not (
        num_points < (dim + 1)
    ).any():
        warnings.warn(
            "Excessively low rank of "
            + "cross-correlation between aligned point clouds. "
            + "corresponding_points_alignment cannot return a unique rotation."
        )

    # identity matrix used for fixing reflections
    E = torch.eye(dim, dtype=XYcov.dtype, device=XYcov.device)[None].repeat(b, 1, 1)

    if not allow_reflection:
        # reflection test:
        #   checks whether the estimated rotation has det==1,
        #   if not, finds the nearest rotation s.t. det==1 by
        #   flipping the sign of the last singular vector U
        R_test = torch.bmm(U, V.transpose(2, 1))
        E[:, -1, -1] = torch.det(R_test)

    # find the rotation matrix by composing U and V again
    R = torch.bmm(torch.bmm(U, E), V.transpose(2, 1))

    if estimate_scale:
        # estimate the scaling component of the transformation
        trace_ES = (torch.diagonal(E, dim1=1, dim2=2) * S).sum(1)
        Xcov = (Xc * Xc).sum((1, 2)) / total_weight

        # the scaling component
        s = trace_ES / torch.clamp(Xcov, eps)

        # translation component
        T = Ymu[:, 0, :] - s[:, None] * torch.bmm(Xmu, R)[:, 0, :]
    else:
        # translation component
        T = Ymu[:, 0, :] - torch.bmm(Xmu, R)[:, 0, :]

        # unit scaling since we do not estimate scale
        s = T.new_ones(b)

    return SimilarityTransform(R, T, s)

def wmean(
    x: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    dim: Union[int, Tuple[int]] = -2,
    keepdim: bool = True,
    eps: float = 1e-9,
) -> torch.Tensor:
    """
    Finds the mean of the input tensor across the specified dimension.
    If the `weight` argument is provided, computes weighted mean.
    Args:
        x: tensor of shape `(*, D)`, where D is assumed to be spatial;
        weights: if given, non-negative tensor of shape `(*,)`. It must be
            broadcastable to `x.shape[:-1]`. Note that the weights for
            the last (spatial) dimension are assumed same;
        dim: dimension(s) in `x` to average over;
        keepdim: tells whether to keep the resulting singleton dimension.
        eps: minimum clamping value in the denominator.
    Returns:
        the mean tensor:
        * if `weights` is None => `mean(x, dim)`,
        * otherwise => `sum(x*w, dim) / max{sum(w, dim), eps}`.
    """
    args = {"dim": dim, "keepdim": keepdim}

    if weight is None:
        return x.mean(**args)

    if any(
        xd != wd and xd != 1 and wd != 1
        for xd, wd in zip(x.shape[-2::-1], weight.shape[::-1])
    ):
        raise ValueError("wmean: weights are not compatible with the tensor")

    return (x * weight[..., None]).sum(**args) / weight[..., None].sum(**args).clamp(
        eps
    )



class EpnpSolution(NamedTuple):
    x_cam: torch.Tensor
    R: torch.Tensor
    T: torch.Tensor
    err_2d: torch.Tensor
    err_3d: torch.Tensor


def _define_control_points(x, weight, storage_opts=None):
    """
    Returns control points that define barycentric coordinates
    Args:
        x: Batch of 3-dimensional points of shape `(minibatch, num_points, 3)`.
        weight: Batch of non-negative weights of
            shape `(minibatch, num_point)`. `None` means equal weights.
        storage_opts: dict of keyword arguments to the tensor constructor.
    """
    storage_opts = storage_opts or {}
    x_mean = wmean(x, weight)
    c_world = F.pad(torch.eye(3, **storage_opts), (0, 0, 0, 1), value=0.0).expand_as(
        x[:, :4, :]
    )
    return c_world + x_mean


def _compute_alphas(x, c_world):
    """
    Computes barycentric coordinates of x in the frame c_world.
    Args:
        x: Batch of 3-dimensional points of shape `(minibatch, num_points, 3)`.
        c_world: control points in world coordinates.
    """
    x = F.pad(x, (0, 1), value=1.0)
    c = F.pad(c_world, (0, 1), value=1.0)
    return torch.matmul(x, torch.inverse(c))  # B x N x 4


def _build_M(y, alphas, weight):
    """Returns the matrix defining the reprojection equations.
    Args:
        y: projected points in camera coordinates of size B x N x 2
        alphas: barycentric coordinates of size B x N x 4
        weight: Batch of non-negative weights of
            shape `(minibatch, num_point)`. `None` means equal weights.
    """
    bs, n, _ = y.size()

    # prepend t with the column of v's
    def prepad(t, v):
        return F.pad(t, (1, 0), value=v)

    if weight is not None:
        # weight the alphas in order to get a correctly weighted version of M
        alphas = alphas * weight[:, :, None]

    # outer left-multiply by alphas
    def lm_alphas(t):
        return torch.matmul(alphas[..., None], t).reshape(bs, n, 12)

    M = torch.cat(
        (
            lm_alphas(
                prepad(prepad(-y[:, :, 0, None, None], 0.0), 1.0)
            ),  # u constraints
            lm_alphas(
                prepad(prepad(-y[:, :, 1, None, None], 1.0), 0.0)
            ),  # v constraints
        ),
        dim=-1,
    ).reshape(bs, -1, 12)

    return M


def _null_space(m, kernel_dim):
    """Finds the null space (kernel) basis of the matrix
    Args:
        m: the batch of input matrices, B x N x 12
        kernel_dim: number of dimensions to approximate the kernel
    Returns:
        * a batch of null space basis vectors
            of size B x 4 x 3 x kernel_dim
        * a batch of spectral values where near-0s correspond to actual
            kernel vectors, of size B x kernel_dim
    """
    mTm = torch.bmm(m.transpose(1, 2), m)
    s, v = eigh(mTm)
    return v[:, :, :kernel_dim].reshape(-1, 4, 3, kernel_dim), s[:, :kernel_dim]


def _reproj_error(y_hat, y, weight, eps=1e-9):
    """Projects estimated 3D points and computes the reprojection error
    Args:
        y_hat: a batch of predicted 2D points in homogeneous coordinates
        y: a batch of ground-truth 2D points
        weight: Batch of non-negative weights of
            shape `(minibatch, num_point)`. `None` means equal weights.
    Returns:
        Optionally weighted RMSE of difference between y and y_hat.
    """
    y_hat = y_hat / torch.clamp(y_hat[..., 2:], eps)
    dist = ((y - y_hat[..., :2]) ** 2).sum(dim=-1, keepdim=True) ** 0.5
    return wmean(dist, weight)[:, 0, 0]


def _algebraic_error(x_w_rotated, x_cam, weight):
    """Computes the residual of Umeyama in 3D.
    Args:
        x_w_rotated: The given 3D points rotated with the predicted camera.
        x_cam: the lifted 2D points y
        weight: Batch of non-negative weights of
            shape `(minibatch, num_point)`. `None` means equal weights.
    Returns:
        Optionally weighted MSE of difference between x_w_rotated and x_cam.
    """
    dist = ((x_w_rotated - x_cam) ** 2).sum(dim=-1, keepdim=True)
    return wmean(dist, weight)[:, 0, 0]


def _compute_norm_sign_scaling_factor(c_cam, alphas, x_world, y, weight, eps=1e-9):
    """Given a solution, adjusts the scale and flip
    Args:
        c_cam: control points in camera coordinates
        alphas: barycentric coordinates of the points
        x_world: Batch of 3-dimensional points of shape `(minibatch, num_points, 3)`.
        y: Batch of 2-dimensional points of shape `(minibatch, num_points, 2)`.
        weights: Batch of non-negative weights of
            shape `(minibatch, num_point)`. `None` means equal weights.
        eps: epsilon to threshold negative `z` values
    """
    # position of reference points in camera coordinates
    x_cam = torch.matmul(alphas, c_cam)

    x_cam = x_cam * (1.0 - 2.0 * (wmean(x_cam[..., 2:], weight) < 0).float())
    if torch.any(x_cam[..., 2:] < -eps):
        neg_rate = wmean((x_cam[..., 2:] < 0).float(), weight, dim=(0, 1)).item()
        warnings.warn("\nEPnP: %2.2f%% points have z<0." % (neg_rate * 100.0))

    R, T, s = corresponding_points_alignment(
        x_world, x_cam, weight, estimate_scale=True
    )
    s = s.clamp(eps)
    x_cam = x_cam / s[:, None, None]
    T = T / s[:, None]
    x_w_rotated = torch.matmul(x_world, R) + T[:, None, :]
    # x_w_rotated = torch.matmul(R, x_world.permuate(0,2,1)).permute(0,2,1) + T[:, None, :]
    err_2d = _reproj_error(x_w_rotated, y, weight)
    err_3d = _algebraic_error(x_w_rotated, x_cam, weight)

    return EpnpSolution(x_cam, R, T, err_2d, err_3d)


def _gen_pairs(input, dim=-2, reducer=lambda a, b: ((a - b) ** 2).sum(dim=-1)):
    """Generates all pairs of different rows and then applies the reducer
    Args:
        input: a tensor
        dim: a dimension to generate pairs across
        reducer: a function of generated pair of rows to apply (beyond just concat)
    Returns:
        for default args, for A x B x C input, will output A x (B choose 2)
    """
    n = input.size()[dim]
    range = torch.arange(n)
    idx = torch.combinations(range).to(input).long()
    left = input.index_select(dim, idx[:, 0])
    right = input.index_select(dim, idx[:, 1])
    return reducer(left, right)


def _kernel_vec_distances(v):
    """Computes the coefficients for linearization of the quadratic system
        to match all pairwise distances between 4 control points (dim=1).
        The last dimension corresponds to the coefficients for quadratic terms
        Bij = Bi * Bj, where Bi and Bj correspond to kernel vectors.
    Arg:
        v: tensor of B x 4 x 3 x D, where D is dim(kernel), usually 4
    Returns:
        a tensor of B x 6 x [(D choose 2) + D];
        for D=4, the last dim means [B11 B22 B33 B44 B12 B13 B14 B23 B24 B34].
    """
    dv = _gen_pairs(v, dim=-3, reducer=lambda a, b: a - b)  # B x 6 x 3 x D

    # we should take dot-product of all (i,j), i < j, with coeff 2
    rows_2ij = 2.0 * _gen_pairs(dv, dim=-1, reducer=lambda a, b: (a * b).sum(dim=-2))
    # this should produce B x 6 x (D choose 2) tensor

    # we should take dot-product of all (i,i)
    rows_ii = (dv ** 2).sum(dim=-2)
    # this should produce B x 6 x D tensor

    return torch.cat((rows_ii, rows_2ij), dim=-1)


def _solve_lstsq_subcols(rhs, lhs, lhs_col_idx):
    """Solves an over-determined linear system for selected LHS columns.
        A batched version of `torch.lstsq`.
    Args:
        rhs: right-hand side vectors
        lhs: left-hand side matrices
        lhs_col_idx: a slice of columns in lhs
    Returns:
        a least-squares solution for lhs * X = rhs
    """
    lhs = lhs.index_select(-1, torch.tensor(lhs_col_idx, device=lhs.device).long())
    return torch.matmul(torch.pinverse(lhs), rhs[:, :, None])


def _binary_sign(t):
    return (t >= 0).to(t) * 2.0 - 1.0


def _find_null_space_coords_1(kernel_dsts, cw_dst, eps=1e-9):
    """Solves case 1 from the paper [1]; solve for 4 coefficients:
       [B11 B22 B33 B44 B12 B13 B14 B23 B24 B34]
         ^               ^   ^   ^
    Args:
        kernel_dsts: distances between kernel vectors
        cw_dst: distances between control points
    Returns:
        coefficients to weight kernel vectors
    [1] Moreno-Noguer, F., Lepetit, V., & Fua, P. (2009).
    EPnP: An Accurate O(n) solution to the PnP problem.
    International Journal of Computer Vision.
    https://www.epfl.ch/labs/cvlab/software/multi-view-stereo/epnp/
    """
    beta = _solve_lstsq_subcols(cw_dst, kernel_dsts, [0, 4, 5, 6])

    beta = beta * _binary_sign(beta[:, :1, :])
    return beta / torch.clamp(beta[:, :1, :] ** 0.5, eps)


def _find_null_space_coords_2(kernel_dsts, cw_dst):
    """Solves case 2 from the paper; solve for 3 coefficients:
        [B11 B22 B33 B44 B12 B13 B14 B23 B24 B34]
          ^   ^           ^
    Args:
        kernel_dsts: distances between kernel vectors
        cw_dst: distances between control points
    Returns:
        coefficients to weight kernel vectors
    [1] Moreno-Noguer, F., Lepetit, V., & Fua, P. (2009).
    EPnP: An Accurate O(n) solution to the PnP problem.
    International Journal of Computer Vision.
    https://www.epfl.ch/labs/cvlab/software/multi-view-stereo/epnp/
    """
    beta = _solve_lstsq_subcols(cw_dst, kernel_dsts, [0, 4, 1])

    coord_0 = (beta[:, :1, :].abs() ** 0.5) * _binary_sign(beta[:, 1:2, :])
    coord_1 = (beta[:, 2:3, :].abs() ** 0.5) * (
        (beta[:, :1, :] >= 0) == (beta[:, 2:3, :] >= 0)
    ).float()

    return torch.cat((coord_0, coord_1, torch.zeros_like(beta[:, :2, :])), dim=1)


def _find_null_space_coords_3(kernel_dsts, cw_dst, eps=1e-9):
    """Solves case 3 from the paper; solve for 5 coefficients:
        [B11 B22 B33 B44 B12 B13 B14 B23 B24 B34]
          ^   ^           ^   ^       ^
    Args:
        kernel_dsts: distances between kernel vectors
        cw_dst: distances between control points
    Returns:
        coefficients to weight kernel vectors
    [1] Moreno-Noguer, F., Lepetit, V., & Fua, P. (2009).
    EPnP: An Accurate O(n) solution to the PnP problem.
    International Journal of Computer Vision.
    https://www.epfl.ch/labs/cvlab/software/multi-view-stereo/epnp/
    """
    beta = _solve_lstsq_subcols(cw_dst, kernel_dsts, [0, 4, 1, 5, 7])

    coord_0 = (beta[:, :1, :].abs() ** 0.5) * _binary_sign(beta[:, 1:2, :])
    coord_1 = (beta[:, 2:3, :].abs() ** 0.5) * (
        (beta[:, :1, :] >= 0) == (beta[:, 2:3, :] >= 0)
    ).float()
    coord_2 = beta[:, 3:4, :] / torch.clamp(coord_0[:, :1, :], eps)

    return torch.cat(
        (coord_0, coord_1, coord_2, torch.zeros_like(beta[:, :1, :])), dim=1
    )


def efficient_pnp(
    x: torch.Tensor,
    y: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    skip_quadratic_eq: bool = False,
) -> EpnpSolution:
    """
    Implements Efficient PnP algorithm [1] for Perspective-n-Points problem:
    finds a camera position (defined by rotation `R` and translation `T`) that
    minimizes re-projection error between the given 3D points `x` and
    the corresponding uncalibrated 2D points `y`, i.e. solves

    `y[i] = Proj(x[i] R[i] + T[i])`

    in the least-squares sense, where `i` are indices within the batch, and
    `Proj` is the perspective projection operator: `Proj([x y z]) = [x/z y/z]`.
    In the noise-less case, 4 points are enough to find the solution as long
    as they are not co-planar.

    Args:
        x: Batch of 3-dimensional points of shape `(minibatch, num_points, 3)`.
        y: Batch of 2-dimensional points of shape `(minibatch, num_points, 2)`.
        weights: Batch of non-negative weights of
            shape `(minibatch, num_point)`. `None` means equal weights.
        skip_quadratic_eq: If True, assumes the solution space for the
            linear system is one-dimensional, i.e. takes the scaled eigenvector
            that corresponds to the smallest eigenvalue as a solution.
            If False, finds the candidate coordinates in the potentially
            4D null space by approximately solving the systems of quadratic
            equations. The best candidate is chosen by examining the 2D
            re-projection error. While this option finds a better solution,
            especially when the number of points is small or perspective
            distortions are low (the points are far away), it may be more
            difficult to back-propagate through.

    Returns:
        `EpnpSolution` namedtuple containing elements:
        **x_cam**: Batch of transformed points `x` that is used to find
            the camera parameters, of shape `(minibatch, num_points, 3)`.
            In the general (noisy) case, they are not exactly equal to
            `x[i] R[i] + T[i]` but are some affine transform of `x[i]`s.
        **R**: Batch of rotation matrices of shape `(minibatch, 3, 3)`.
        **T**: Batch of translation vectors of shape `(minibatch, 3)`.
        **err_2d**: Batch of mean 2D re-projection errors of shape
            `(minibatch,)`. Specifically, if `yhat` is the re-projection for
            the `i`-th batch element, it returns `sum_j norm(yhat_j - y_j)`
            where `j` iterates over points and `norm` denotes the L2 norm.
        **err_3d**: Batch of mean algebraic errors of shape `(minibatch,)`.
            Specifically, those are squared distances between `x_world` and
            estimated points on the rays defined by `y`.

    [1] Moreno-Noguer, F., Lepetit, V., & Fua, P. (2009).
    EPnP: An Accurate O(n) solution to the PnP problem.
    International Journal of Computer Vision.
    https://www.epfl.ch/labs/cvlab/software/multi-view-stereo/epnp/
    """
    # define control points in a world coordinate system (centered on the 3d
    # points centroid); 4 x 3
    # TODO: more stable when initialised with the center and eigenvectors!
    c_world = _define_control_points(
        x.detach(), weights, storage_opts={"dtype": x.dtype, "device": x.device}
    )

    # find the linear combination of the control points to represent the 3d points
    alphas = _compute_alphas(x, c_world)

    M = _build_M(y, alphas, weights)

    # Compute kernel M
    kernel, spectrum = _null_space(M, 4)

    c_world_distances = _gen_pairs(c_world)
    kernel_dsts = _kernel_vec_distances(kernel)

    betas = (
        []
        if skip_quadratic_eq
        else [
            fnsc(kernel_dsts, c_world_distances)
            for fnsc in [
                _find_null_space_coords_1,
                _find_null_space_coords_2,
                _find_null_space_coords_3,
            ]
        ]
    )

    c_cam_variants = [kernel] + [
        torch.matmul(kernel, beta[:, None, :, :]) for beta in betas
    ]

    solutions = [
        _compute_norm_sign_scaling_factor(c_cam[..., 0], alphas, x, y, weights)
        for c_cam in c_cam_variants
    ]

    sol_zipped = EpnpSolution(*(torch.stack(list(col)) for col in zip(*solutions)))
    best = torch.argmin(sol_zipped.err_2d, dim=0)

    def gather1d(source, idx):
        # reduces the dim=1 by picking the slices in a 1D tensor idx
        # in other words, it is batched index_select.
        return source.gather(
            0,
            idx.reshape(1, -1, *([1] * (len(source.shape) - 2))).expand_as(source[:1]),
        )[0]

    return EpnpSolution(*[gather1d(sol_col, best) for sol_col in sol_zipped])

if __name__ == '__main__':
    pass
    x = torch.randn((8,100,3))
    y = torch.randn((8,100,2))
    _, R, t, _, _ = efficient_pnp(x,y)
    print(R.shape)
    print(t.shape)