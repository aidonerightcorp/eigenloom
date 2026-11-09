"""Dual-mode (pure Python stdlib + automatic NumPy acceleration) linear algebra and safetensors codec."""

from __future__ import annotations

import json
import math
import struct
from typing import Any

from eigenloom_core.config_loader import get_manifold_spec

try:
    import numpy as _np  # type: ignore[import-not-found]

    HAS_NUMPY = True
except ImportError:
    _np = None  # type: ignore[assignment]
    HAS_NUMPY = False

Matrix = list[list[float]]
Vector = list[float]


def _use_numpy(force_stdlib: bool = False) -> bool:
    return bool(HAS_NUMPY and not force_stdlib)


def zeros_matrix(rows: int, cols: int) -> Matrix:
    return [[0.0 for _ in range(cols)] for _ in range(rows)]


def identity_matrix(dim: int) -> Matrix:
    return [
        [1.0 if r == c else 0.0 for c in range(dim)]
        for r in range(dim)
    ]


def copy_matrix(mat: Matrix) -> Matrix:
    return [list(row) for row in mat]


def transpose(mat: Matrix) -> Matrix:
    if not mat:
        return []
    rows = len(mat)
    cols = len(mat[0])
    return [[float(mat[r][c]) for r in range(rows)] for c in range(cols)]


def vec_dot(vec_a: Vector, vec_b: Vector) -> float:
    return float(sum(float(a) * float(b) for a, b in zip(vec_a, vec_b)))


def vec_norm(vec: Vector) -> float:
    return math.sqrt(max(0.0, vec_dot(vec, vec)))


def l2_normalize(vec: Vector, eps: float | None = None) -> Vector:
    floor = eps if eps is not None else get_manifold_spec().c_float("pico")
    nrm = vec_norm(vec)
    if nrm <= floor:
        return [0.0 for _ in vec]
    inv = 1.0 / nrm
    return [float(x) * inv for x in vec]


def frobenius_norm(mat: Matrix) -> float:
    total = 0.0
    for row in mat:
        for val in row:
            fv = float(val)
            total += fv * fv
    return math.sqrt(max(0.0, total))


def matvec(mat: Matrix, vec: Vector, *, force_stdlib: bool = False) -> Vector:
    if not mat:
        return []
    if _use_numpy(force_stdlib):
        arr = _np.asarray(mat, dtype=float)
        v = _np.asarray(vec, dtype=float)
        return [float(x) for x in (arr @ v).tolist()]
    return [vec_dot(row, vec) for row in mat]


def matmul(mat_a: Matrix, mat_b: Matrix, *, force_stdlib: bool = False) -> Matrix:
    if not mat_a or not mat_b:
        return []
    if _use_numpy(force_stdlib):
        arr_a = _np.asarray(mat_a, dtype=float)
        arr_b = _np.asarray(mat_b, dtype=float)
        return [[float(v) for v in row] for row in (arr_a @ arr_b).tolist()]
    bt = transpose(mat_b)
    return [[vec_dot(row_a, col_b) for col_b in bt] for row_a in mat_a]


def add_scaled(mat_a: Matrix, mat_b: Matrix, scale: float = 1.0) -> Matrix:
    return [
        [float(a) + scale * float(b) for a, b in zip(row_a, row_b)]
        for row_a, row_b in zip(mat_a, mat_b)
    ]


def outer_product(vec_u: Vector, vec_v: Vector, scale: float = 1.0) -> Matrix:
    return [[scale * float(u) * float(v) for v in vec_v] for u in vec_u]


def softmax(logits: Vector, temperature: float = 1.0) -> Vector:
    if not logits:
        return []
    spec = get_manifold_spec()
    temp = max(float(temperature), spec.c_float("micro"))
    scaled = [float(x) / temp for x in logits]
    peak = max(scaled)
    exps = [math.exp(x - peak) for x in scaled]
    denom = sum(exps)
    if denom <= spec.c_float("pico"):
        uniform = 1.0 / float(len(logits))
        return [uniform for _ in logits]
    return [val / denom for val in exps]


def activate_vector(vec: Vector, family: str) -> Vector:
    spec = get_manifold_spec()
    half = spec.c_float("half")
    c1 = spec.c_float("gelu_c1")
    c2 = spec.c_float("gelu_c2")
    three = spec.c_int("three")
    if family == "relu":
        return [val if val > 0.0 else 0.0 for val in vec]
    if family == "silu":
        out: Vector = []
        for val in vec:
            clamped = max(-spec.c_float("twenty"), min(spec.c_float("twenty"), val))
            out.append(val / (1.0 + math.exp(-clamped)))
        return out
    if family == "gelu":
        return [
            half * val * (1.0 + math.tanh(c1 * (val + c2 * (val ** three))))
            for val in vec
        ]
    return [float(val) for val in vec]


def linear_solve(mat_a: Matrix, rhs_b: Matrix, *, force_stdlib: bool = False) -> Matrix:
    """Solve A * X = B with partial-pivoted Gaussian elimination (or NumPy solve)."""
    spec = get_manifold_spec()
    eps = spec.c_float("nano")
    n = len(mat_a)
    if n == 0:
        return []
    if _use_numpy(force_stdlib):
        arr_a = _np.asarray(mat_a, dtype=float) + _np.eye(n, dtype=float) * eps
        arr_b = _np.asarray(rhs_b, dtype=float)
        sol = _np.linalg.solve(arr_a, arr_b)
        return [[float(v) for v in row] for row in sol.tolist()]

    cols_b = len(rhs_b[0]) if rhs_b else 0
    aug = [
        [float(mat_a[i][j]) + (eps if i == j else 0.0) for j in range(n)]
        + [float(rhs_b[i][k]) for k in range(cols_b)]
        for i in range(n)
    ]
    total_cols = n + cols_b
    for col in range(n):
        pivot_row = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if pivot_row != col:
            aug[col], aug[pivot_row] = aug[pivot_row], aug[col]
        pivot_val = aug[col][col]
        if abs(pivot_val) <= spec.c_float("pico"):
            pivot_val = eps
        inv_pivot = 1.0 / pivot_val
        for j in range(col, total_cols):
            aug[col][j] *= inv_pivot
        for r in range(n):
            if r == col:
                continue
            factor = aug[r][col]
            if factor == 0.0:
                continue
            for j in range(col, total_cols):
                aug[r][j] -= factor * aug[col][j]
    return [row[n:] for row in aug]


def ridge_solve(
    design_x: Matrix,
    target_y: Matrix,
    ridge_lambda: float,
    *,
    force_stdlib: bool = False,
) -> Matrix:
    """Fit W in R^{d_out x d_in} minimizing ||X W^T - Y||_F^2 + lambda ||W||_F^2."""
    if not design_x or not target_y:
        return []
    xt = transpose(design_x)
    xtx = matmul(xt, design_x, force_stdlib=force_stdlib)
    dim_in = len(xtx)
    reg = max(float(ridge_lambda), get_manifold_spec().c_float("nano"))
    for idx in range(dim_in):
        xtx[idx][idx] += reg
    xty = matmul(xt, target_y, force_stdlib=force_stdlib)
    wt = linear_solve(xtx, xty, force_stdlib=force_stdlib)
    return transpose(wt)


def woodbury_rank1_update(
    a_inv: Matrix,
    feature_x: Vector,
    damping: float = 1.0,
    *,
    force_stdlib: bool = False,
) -> Matrix:
    """Exact Sherman-Morrison-Woodbury rank-1 inverse covariance update."""
    spec = get_manifold_spec()
    ax = matvec(a_inv, feature_x, force_stdlib=force_stdlib)
    quad = vec_dot(feature_x, ax)
    denom = max(float(damping) + quad, spec.c_float("nano"))
    correction = outer_product(ax, ax, scale=1.0 / denom)
    return add_scaled(a_inv, correction, scale=-1.0)


def _jacobi_symmetric_eigen(sym: Matrix) -> tuple[Vector, Matrix]:
    """Cyclic Jacobi eigendecomposition for real symmetric matrix S = V diag(eig) V^T."""
    spec = get_manifold_spec()
    n = len(sym)
    if n == 0:
        return [], []
    a = copy_matrix(sym)
    v = identity_matrix(n)
    max_sweeps = spec.c_int("jacobi_max_sweeps")
    tol = spec.c_float("pico")
    two = float(spec.c_int("two"))

    for _ in range(max_sweeps):
        max_off = 0.0
        for p in range(n):
            for q in range(p + 1, n):
                apq = abs(a[p][q])
                if apq > max_off:
                    max_off = apq
        if max_off <= tol:
            break
        for p in range(n):
            for q in range(p + 1, n):
                apq = a[p][q]
                if abs(apq) <= tol:
                    continue
                tau = (a[q][q] - a[p][p]) / (two * apq)
                if tau >= 0.0:
                    t = 1.0 / (tau + math.sqrt(1.0 + tau * tau))
                else:
                    t = -1.0 / (-tau + math.sqrt(1.0 + tau * tau))
                c = 1.0 / math.sqrt(1.0 + t * t)
                s = t * c
                app = a[p][p]
                aqq = a[q][q]
                a[p][p] = app - t * apq
                a[q][q] = aqq + t * apq
                a[p][q] = 0.0
                a[q][p] = 0.0
                for k in range(n):
                    if k != p and k != q:
                        akp = a[k][p]
                        akq = a[k][q]
                        a[k][p] = c * akp - s * akq
                        a[p][k] = a[k][p]
                        a[k][q] = s * akp + c * akq
                        a[q][k] = a[k][q]
                for k in range(n):
                    vkp = v[k][p]
                    vkq = v[k][q]
                    v[k][p] = c * vkp - s * vkq
                    v[k][q] = s * vkp + c * vkq

    eigs = [max(0.0, a[i][i]) for i in range(n)]
    order = sorted(range(n), key=lambda idx: eigs[idx], reverse=True)
    sorted_eigs = [eigs[idx] for idx in order]
    sorted_v = [[v[r][idx] for idx in order] for r in range(n)]
    return sorted_eigs, sorted_v


def _gram_schmidt_columns(cols: list[Vector], dim: int) -> list[Vector]:
    spec = get_manifold_spec()
    tol = spec.c_float("nano")
    basis: list[Vector] = []
    for vec in cols:
        w = list(vec)
        for b in basis:
            proj = vec_dot(w, b)
            w = [wi - proj * bi for wi, bi in zip(w, b)]
        nrm = vec_norm(w)
        if nrm > tol:
            basis.append([wi / nrm for wi in w])
    # Complete orthonormal basis if needed
    for e_idx in range(dim):
        if len(basis) >= len(cols):
            break
        cand = [1.0 if i == e_idx else 0.0 for i in range(dim)]
        for b in basis:
            proj = vec_dot(cand, b)
            cand = [ci - proj * bi for ci, bi in zip(cand, b)]
        nrm = vec_norm(cand)
        if nrm > tol:
            basis.append([ci / nrm for ci in cand])
    return basis[: len(cols)]


def svd_compact(
    mat: Matrix,
    *,
    force_stdlib: bool = False,
) -> tuple[Matrix, Vector, Matrix]:
    """Compute compact SVD (U, S, Vt) such that U @ diag(S) @ Vt == mat."""
    if not mat or not mat[0]:
        return [], [], []
    if _use_numpy(force_stdlib):
        arr = _np.asarray(mat, dtype=float)
        u_arr, s_arr, vt_arr = _np.linalg.svd(arr, full_matrices=False)
        return (
            [[float(x) for x in row] for row in u_arr.tolist()],
            [float(x) for x in s_arr.tolist()],
            [[float(x) for x in row] for row in vt_arr.tolist()],
        )

    spec = get_manifold_spec()
    tol = spec.c_float("nano")
    rows = len(mat)
    cols = len(mat[0])
    rank = min(rows, cols)
    mt = transpose(mat)

    if rows >= cols:
        gram = matmul(mt, mat, force_stdlib=True)
        eigs, v_mat = _jacobi_symmetric_eigen(gram)
        s_vals = [math.sqrt(max(0.0, ev)) for ev in eigs[:rank]]
        v_cols = [[v_mat[r][k] for r in range(cols)] for k in range(rank)]
        raw_u_cols: list[Vector] = []
        for k in range(rank):
            av = matvec(mat, v_cols[k], force_stdlib=True)
            if s_vals[k] > tol:
                raw_u_cols.append([val / s_vals[k] for val in av])
            else:
                raw_u_cols.append([0.0 for _ in range(rows)])
        u_cols = _gram_schmidt_columns(raw_u_cols, rows)
        u_mat = [[u_cols[k][r] for k in range(rank)] for r in range(rows)]
        vt_mat = [list(v_cols[k]) for k in range(rank)]
        return u_mat, s_vals, vt_mat

    gram = matmul(mat, mt, force_stdlib=True)
    eigs, u_full = _jacobi_symmetric_eigen(gram)
    s_vals = [math.sqrt(max(0.0, ev)) for ev in eigs[:rank]]
    u_cols = [[u_full[r][k] for r in range(rows)] for k in range(rank)]
    raw_v_cols: list[Vector] = []
    for k in range(rank):
        atu = matvec(mt, u_cols[k], force_stdlib=True)
        if s_vals[k] > tol:
            raw_v_cols.append([val / s_vals[k] for val in atu])
        else:
            raw_v_cols.append([0.0 for _ in range(cols)])
    v_cols = _gram_schmidt_columns(raw_v_cols, cols)
    u_mat = [[u_cols[k][r] for k in range(rank)] for r in range(rows)]
    vt_mat = [list(v_cols[k]) for k in range(rank)]
    return u_mat, s_vals, vt_mat


def orthogonal_procrustes(
    source_mat: Matrix,
    target_mat: Matrix,
    *,
    force_stdlib: bool = False,
) -> Matrix:
    """Return orthogonal matrix Omega minimizing ||Omega @ source_mat - target_mat||_F."""
    cross = matmul(target_mat, transpose(source_mat), force_stdlib=force_stdlib)
    u_mat, _, vt_mat = svd_compact(cross, force_stdlib=force_stdlib)
    return matmul(u_mat, vt_mat, force_stdlib=force_stdlib)


def safetensors_dumps(
    tensors: dict[str, Matrix],
    metadata: dict[str, str] | None = None,
) -> bytes:
    """Serialize 2D float32 matrices into standard HuggingFace .safetensors format."""
    spec = get_manifold_spec()
    align = spec.c_int("safetensors_alignment_bytes")
    f32_bytes = spec.c_int("float32_byte_width")

    header: dict[str, Any] = {}
    if metadata:
        header["__metadata__"] = {str(k): str(v) for k, v in metadata.items()}

    body_chunks: list[bytes] = []
    offset = 0
    for name in sorted(tensors.keys()):
        mat = tensors[name]
        rows = len(mat)
        cols = len(mat[0]) if rows > 0 else 0
        flat = [float(val) for row in mat for val in row]
        chunk = struct.pack(f"<{len(flat)}f", *flat) if flat else b""
        byte_len = len(flat) * f32_bytes
        header[name] = {
            "dtype": "F32",
            "shape": [rows, cols],
            "data_offsets": [offset, offset + byte_len],
        }
        body_chunks.append(chunk)
        offset += byte_len

    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    pad = (align - (len(header_bytes) % align)) % align
    if pad > 0:
        header_bytes = header_bytes + (b" " * pad)

    prefix = struct.pack("<Q", len(header_bytes))
    return prefix + header_bytes + b"".join(body_chunks)


def safetensors_loads(blob: bytes) -> tuple[dict[str, Matrix], dict[str, str]]:
    """Deserialize a .safetensors binary buffer back into 2D matrices and metadata."""
    spec = get_manifold_spec()
    align = spec.c_int("safetensors_alignment_bytes")
    f32_bytes = spec.c_int("float32_byte_width")
    if len(blob) < align:
        raise ValueError("Invalid safetensors buffer: shorter than header size prefix")

    (header_len,) = struct.unpack("<Q", blob[:align])
    header_end = align + header_len
    header = json.loads(blob[align:header_end].decode("utf-8").strip())
    data_buf = blob[header_end:]

    meta = {
        str(k): str(v)
        for k, v in header.pop("__metadata__", {}).items()
    }
    tensors: dict[str, Matrix] = {}
    for name, info in header.items():
        shape = info["shape"]
        rows = int(shape[0]) if shape else 0
        cols = int(shape[1]) if len(shape) > 1 else 0
        start, end = int(info["data_offsets"][0]), int(info["data_offsets"][1])
        count = (end - start) // f32_bytes
        vals = list(struct.unpack(f"<{count}f", data_buf[start:end])) if count > 0 else []
        mat: Matrix = []
        for r in range(rows):
            mat.append([float(vals[r * cols + c]) for c in range(cols)])
        tensors[name] = mat
    return tensors, meta
