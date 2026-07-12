import logging
import os
import time

import torch
import triton
import triton.language as tl

from .convrot import build_hadamard as _build_hadamard


_LOGGED_NATIVE_PATH = False
_TARGET_CHECKED = False
_IS_GFX1103_TARGET = False
_TARGET_DESCRIPTION = "uninitialized"
_PROFILE_ENABLED = os.environ.get("COMFY_INT4_PROFILE", "0").lower() in ("1", "true", "yes", "on")
_PROFILE_FAILED = False
_PROFILE_COUNTS: dict[tuple, int] = {}


@triton.jit
def _round_to_nearest_even(x):
    lower = tl.floor(x)
    fraction = x - lower
    lower_int = lower.to(tl.int32)
    tie = tl.where((lower_int & 1) != 0, lower + 1.0, lower)
    return tl.where(fraction > 0.5, lower + 1.0, tl.where(fraction < 0.5, lower, tie))


@triton.jit
def _convrot_matmul_kernel(
    x_ptr,
    h_ptr,
    out_ptr,
    group_rows,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    x_ptrs = x_ptr + offs_m[:, None] * GROUP_SIZE + offs_k[None, :]
    h_ptrs = h_ptr + offs_k[:, None] * GROUP_SIZE + offs_n[None, :]
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for k_base in range(0, GROUP_SIZE, BLOCK_K):
        x = tl.load(x_ptrs, mask=(offs_m[:, None] < group_rows) & (offs_k[None, :] + k_base < GROUP_SIZE), other=0.0)
        h = tl.load(h_ptrs, mask=(offs_k[:, None] + k_base < GROUP_SIZE) & (offs_n[None, :] < GROUP_SIZE), other=0.0)
        accumulator += tl.dot(x, h)
        x_ptrs += BLOCK_K
        h_ptrs += BLOCK_K * GROUP_SIZE
    tl.store(
        out_ptr + offs_m[:, None] * GROUP_SIZE + offs_n[None, :],
        accumulator,
        mask=(offs_m[:, None] < group_rows) & (offs_n[None, :] < GROUP_SIZE),
    )


@triton.jit
def _quantize_i4_rowwise_kernel(x_ptr, q_ptr, scale_ptr, k_size, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    values = tl.load(x_ptr + row * k_size + offsets, mask=offsets < k_size, other=0.0).to(tl.float32)
    absmax = tl.max(tl.abs(values), axis=0)
    scale = tl.maximum(absmax / 7.0, 1e-10)
    tl.store(scale_ptr + row, scale)

    pair_offsets = tl.arange(0, BLOCK_SIZE // 2)
    even_offsets = pair_offsets * 2
    odd_offsets = even_offsets + 1
    even = tl.load(x_ptr + row * k_size + even_offsets, mask=even_offsets < k_size, other=0.0).to(tl.float32) / scale
    odd = tl.load(x_ptr + row * k_size + odd_offsets, mask=odd_offsets < k_size, other=0.0).to(tl.float32) / scale
    even = _round_to_nearest_even(even)
    odd = _round_to_nearest_even(odd)
    even = tl.minimum(tl.maximum(even, -7.0), 7.0).to(tl.int32)
    odd = tl.minimum(tl.maximum(odd, -7.0), 7.0).to(tl.int32)
    packed = (even & 0xF) | ((odd & 0xF) << 4)
    tl.store(q_ptr + row * (k_size // 2) + pair_offsets, packed.to(tl.int8), mask=pair_offsets < k_size // 2)


@triton.jit
def _wmma_i4_accumulate(
    a0,
    a1,
    b0,
    b1,
    c0,
    c1,
    c2,
    c3,
    c4,
    c5,
    c6,
    c7,
):
    return tl.inline_asm_elementwise(
        asm=(
            "v_mov_b32 v8, $8\n\t"
            "v_mov_b32 v9, $9\n\t"
            "v_mov_b32 v10, $10\n\t"
            "v_mov_b32 v11, $11\n\t"
            "v_mov_b32 v16, $12\n\t"
            "v_mov_b32 v17, $13\n\t"
            "v_mov_b32 v18, $14\n\t"
            "v_mov_b32 v19, $15\n\t"
            "v_mov_b32 v20, $16\n\t"
            "v_mov_b32 v21, $17\n\t"
            "v_mov_b32 v22, $18\n\t"
            "v_mov_b32 v23, $19\n\t"
            "v_wmma_i32_16x16x16_iu4 v[24:31], v[8:9], v[10:11], v[16:23] neg_lo:[1,1,0]\n\t"
            "v_mov_b32 $0, v24\n\t"
            "v_mov_b32 $1, v25\n\t"
            "v_mov_b32 $2, v26\n\t"
            "v_mov_b32 $3, v27\n\t"
            "v_mov_b32 $4, v28\n\t"
            "v_mov_b32 $5, v29\n\t"
            "v_mov_b32 $6, v30\n\t"
            "v_mov_b32 $7, v31"
        ),
        constraints=(
            "=v,=v,=v,=v,=v,=v,=v,=v,"
            "v,v,v,v,v,v,v,v,v,v,v,v,"
            "~{v8},~{v9},~{v10},~{v11},"
            "~{v16},~{v17},~{v18},~{v19},~{v20},~{v21},~{v22},~{v23},"
            "~{v24},~{v25},~{v26},~{v27},~{v28},~{v29},~{v30},~{v31}"
        ),
        args=[a0, a1, b0, b1, c0, c1, c2, c3, c4, c5, c6, c7],
        dtype=(tl.int32,) * 8,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _load_i4_word(ptr, byte_offset, row_mask):
    x0 = tl.load(ptr + byte_offset, mask=row_mask, other=0).to(tl.uint8).to(tl.uint32)
    x1 = tl.load(ptr + byte_offset + 1, mask=row_mask, other=0).to(tl.uint8).to(tl.uint32)
    x2 = tl.load(ptr + byte_offset + 2, mask=row_mask, other=0).to(tl.uint8).to(tl.uint32)
    x3 = tl.load(ptr + byte_offset + 3, mask=row_mask, other=0).to(tl.uint8).to(tl.uint32)
    return (x0 | (x1 << 8) | (x2 << 16) | (x3 << 24)).to(tl.int32, bitcast=True)


@triton.jit
def _w4a4_gemm_kernel(
    a_ptr,
    w_ptr,
    a_scale_ptr,
    w_scale_ptr,
    bias_ptr,
    out_ptr,
    m_size,
    n_size,
    k_size,
    stride_am,
    stride_wn,
    stride_om,
    n_start,
    HAS_BIAS: tl.constexpr,
):
    tile_m = tl.program_id(0) * 16
    tile_n = n_start + tl.program_id(1) * 16
    wave_lane = tl.arange(0, 32)
    lane = wave_lane % 16
    lane_group = wave_lane // 16
    row_a = tile_m + lane
    row_w = tile_n + lane
    mask_a = row_a < m_size
    mask_w = row_w < n_size

    c0 = tl.zeros((32,), tl.int32)
    c1 = tl.zeros((32,), tl.int32)
    c2 = tl.zeros((32,), tl.int32)
    c3 = tl.zeros((32,), tl.int32)
    c4 = tl.zeros((32,), tl.int32)
    c5 = tl.zeros((32,), tl.int32)
    c6 = tl.zeros((32,), tl.int32)
    c7 = tl.zeros((32,), tl.int32)

    for k_base in range(0, k_size, 16):
        byte_k = k_base // 2
        a_byte = row_a * stride_am + byte_k
        w_byte = row_w * stride_wn + byte_k
        a0 = _load_i4_word(a_ptr, a_byte, mask_a)
        a1 = _load_i4_word(a_ptr, a_byte + 4, mask_a)
        b0 = _load_i4_word(w_ptr, w_byte, mask_w)
        b1 = _load_i4_word(w_ptr, w_byte + 4, mask_w)
        c0, c1, c2, c3, c4, c5, c6, c7 = _wmma_i4_accumulate(
            a0,
            a1,
            b0,
            b1,
            c0,
            c1,
            c2,
            c3,
            c4,
            c5,
            c6,
            c7,
        )

    # Convert metadata in registers.  Keeping the source tensor in its stored
    # dtype avoids a separate eager dtype-conversion kernel on every forward.
    w_scales = tl.load(w_scale_ptr + tile_n + lane, mask=mask_w, other=0.0).to(tl.float32)
    if HAS_BIAS:
        bias = tl.load(bias_ptr + tile_n + lane, mask=mask_w, other=0.0).to(tl.float32)
    else:
        bias = tl.zeros((32,), tl.float32)

    accumulators = (c0, c1, c2, c3, c4, c5, c6, c7)
    for element in tl.static_range(8):
        row_in_tile = element * 2 + lane_group
        row = tile_m + row_in_tile
        col = tile_n + lane
        a_scale = tl.load(a_scale_ptr + row, mask=row < m_size, other=0.0).to(tl.float32)
        value = accumulators[element].to(tl.float32)
        value *= a_scale * w_scales
        value += bias
        tl.store(out_ptr + row * stride_om + col, value, mask=(row < m_size) & (col < n_size))


@triton.jit
def _w4a4_gemm_32x32_kernel(
    a_ptr,
    w_ptr,
    a_scale_ptr,
    w_scale_ptr,
    bias_ptr,
    out_ptr,
    m_size,
    n_size,
    k_size,
    stride_am,
    stride_wn,
    stride_om,
    n_start,
    HAS_BIAS: tl.constexpr,
):
    tile_m = tl.program_id(0) * 32
    tile_n = n_start + tl.program_id(1) * 32
    wave_lane = tl.arange(0, 32)
    lane = wave_lane % 16
    lane_group = wave_lane // 16

    row_a0 = tile_m + lane
    row_a1 = tile_m + 16 + lane
    row_w0 = tile_n + lane
    row_w1 = tile_n + 16 + lane
    mask_a0 = row_a0 < m_size
    mask_a1 = row_a1 < m_size
    mask_w0 = row_w0 < n_size
    mask_w1 = row_w1 < n_size

    tl0 = tl.zeros((32,), tl.int32)
    tl1 = tl.zeros((32,), tl.int32)
    tl2 = tl.zeros((32,), tl.int32)
    tl3 = tl.zeros((32,), tl.int32)
    tl4 = tl.zeros((32,), tl.int32)
    tl5 = tl.zeros((32,), tl.int32)
    tl6 = tl.zeros((32,), tl.int32)
    tl7 = tl.zeros((32,), tl.int32)
    tr0 = tl.zeros((32,), tl.int32)
    tr1 = tl.zeros((32,), tl.int32)
    tr2 = tl.zeros((32,), tl.int32)
    tr3 = tl.zeros((32,), tl.int32)
    tr4 = tl.zeros((32,), tl.int32)
    tr5 = tl.zeros((32,), tl.int32)
    tr6 = tl.zeros((32,), tl.int32)
    tr7 = tl.zeros((32,), tl.int32)
    bl0 = tl.zeros((32,), tl.int32)
    bl1 = tl.zeros((32,), tl.int32)
    bl2 = tl.zeros((32,), tl.int32)
    bl3 = tl.zeros((32,), tl.int32)
    bl4 = tl.zeros((32,), tl.int32)
    bl5 = tl.zeros((32,), tl.int32)
    bl6 = tl.zeros((32,), tl.int32)
    bl7 = tl.zeros((32,), tl.int32)
    br0 = tl.zeros((32,), tl.int32)
    br1 = tl.zeros((32,), tl.int32)
    br2 = tl.zeros((32,), tl.int32)
    br3 = tl.zeros((32,), tl.int32)
    br4 = tl.zeros((32,), tl.int32)
    br5 = tl.zeros((32,), tl.int32)
    br6 = tl.zeros((32,), tl.int32)
    br7 = tl.zeros((32,), tl.int32)

    for k_base in range(0, k_size, 16):
        byte_k = k_base // 2
        a0_byte = row_a0 * stride_am + byte_k
        a1_byte = row_a1 * stride_am + byte_k
        w0_byte = row_w0 * stride_wn + byte_k
        w1_byte = row_w1 * stride_wn + byte_k
        at0 = _load_i4_word(a_ptr, a0_byte, mask_a0)
        at1 = _load_i4_word(a_ptr, a0_byte + 4, mask_a0)
        ab0 = _load_i4_word(a_ptr, a1_byte, mask_a1)
        ab1 = _load_i4_word(a_ptr, a1_byte + 4, mask_a1)
        wl0 = _load_i4_word(w_ptr, w0_byte, mask_w0)
        wl1 = _load_i4_word(w_ptr, w0_byte + 4, mask_w0)
        wr0 = _load_i4_word(w_ptr, w1_byte, mask_w1)
        wr1 = _load_i4_word(w_ptr, w1_byte + 4, mask_w1)
        tl0, tl1, tl2, tl3, tl4, tl5, tl6, tl7 = _wmma_i4_accumulate(
            at0, at1, wl0, wl1, tl0, tl1, tl2, tl3, tl4, tl5, tl6, tl7
        )
        tr0, tr1, tr2, tr3, tr4, tr5, tr6, tr7 = _wmma_i4_accumulate(
            at0, at1, wr0, wr1, tr0, tr1, tr2, tr3, tr4, tr5, tr6, tr7
        )
        bl0, bl1, bl2, bl3, bl4, bl5, bl6, bl7 = _wmma_i4_accumulate(
            ab0, ab1, wl0, wl1, bl0, bl1, bl2, bl3, bl4, bl5, bl6, bl7
        )
        br0, br1, br2, br3, br4, br5, br6, br7 = _wmma_i4_accumulate(
            ab0, ab1, wr0, wr1, br0, br1, br2, br3, br4, br5, br6, br7
        )

    # Scale/bias may be FP16, BF16, or FP32 in the checkpoint.  Load the
    # original dtype and promote in registers instead of materializing FP32
    # copies through PyTorch eager operations.
    w_scale0 = tl.load(w_scale_ptr + row_w0, mask=mask_w0, other=0.0).to(tl.float32)
    w_scale1 = tl.load(w_scale_ptr + row_w1, mask=mask_w1, other=0.0).to(tl.float32)
    if HAS_BIAS:
        bias0 = tl.load(bias_ptr + row_w0, mask=mask_w0, other=0.0).to(tl.float32)
        bias1 = tl.load(bias_ptr + row_w1, mask=mask_w1, other=0.0).to(tl.float32)
    else:
        bias0 = tl.zeros((32,), tl.float32)
        bias1 = tl.zeros((32,), tl.float32)

    top_left = (tl0, tl1, tl2, tl3, tl4, tl5, tl6, tl7)
    top_right = (tr0, tr1, tr2, tr3, tr4, tr5, tr6, tr7)
    bottom_left = (bl0, bl1, bl2, bl3, bl4, bl5, bl6, bl7)
    bottom_right = (br0, br1, br2, br3, br4, br5, br6, br7)
    for element in tl.static_range(8):
        row_in_tile = element * 2 + lane_group
        row0 = tile_m + row_in_tile
        row1 = tile_m + 16 + row_in_tile
        col0 = tile_n + lane
        col1 = tile_n + 16 + lane
        a_scale0 = tl.load(a_scale_ptr + row0, mask=row0 < m_size, other=0.0).to(tl.float32)
        a_scale1 = tl.load(a_scale_ptr + row1, mask=row1 < m_size, other=0.0).to(tl.float32)
        value_tl = top_left[element].to(tl.float32) * a_scale0 * w_scale0 + bias0
        value_tr = top_right[element].to(tl.float32) * a_scale0 * w_scale1 + bias1
        value_bl = bottom_left[element].to(tl.float32) * a_scale1 * w_scale0 + bias0
        value_br = bottom_right[element].to(tl.float32) * a_scale1 * w_scale1 + bias1
        tl.store(out_ptr + row0 * stride_om + col0, value_tl, mask=(row0 < m_size) & (col0 < n_size))
        tl.store(out_ptr + row0 * stride_om + col1, value_tr, mask=(row0 < m_size) & (col1 < n_size))
        tl.store(out_ptr + row1 * stride_om + col0, value_bl, mask=(row1 < m_size) & (col0 < n_size))
        tl.store(out_ptr + row1 * stride_om + col1, value_br, mask=(row1 < m_size) & (col1 < n_size))


@triton.jit
def _w4a4_dot_gemm_kernel(
    a_ptr,
    w_ptr,
    a_scale_ptr,
    w_scale_ptr,
    bias_ptr,
    out_ptr,
    m_size,
    n_size,
    stride_am,
    stride_wn,
    stride_om,
    K_SIZE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Multi-wave signed W4A4 GEMM lowered by Triton to RDNA3 IU4 WMMA."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k_packed = tl.arange(0, BLOCK_K // 2)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), tl.int32)
    for k_base in range(0, K_SIZE, BLOCK_K):
        packed_k = k_base // 2
        a_bytes = tl.load(
            a_ptr
            + offs_m[:, None] * stride_am
            + packed_k
            + offs_k_packed[None, :],
            mask=offs_m[:, None] < m_size,
            other=0,
        )
        w_bytes = tl.load(
            w_ptr
            + offs_n[:, None] * stride_wn
            + packed_k
            + offs_k_packed[None, :],
            mask=offs_n[:, None] < n_size,
            other=0,
        )

        # tl.join adds a minor dimension; reshape restores low/high nibbles in
        # their original K order without ever materializing an INT8 matrix.
        a_i4 = tl.reshape(
            tl.join((a_bytes & 0xF).to(tl.int4), (a_bytes >> 4).to(tl.int4)),
            (BLOCK_M, BLOCK_K),
        )
        w_i4 = tl.reshape(
            tl.join((w_bytes & 0xF).to(tl.int4), (w_bytes >> 4).to(tl.int4)),
            (BLOCK_N, BLOCK_K),
        )
        accumulator += tl.dot(a_i4, tl.trans(w_i4), out_dtype=tl.int32)

    a_scales = tl.load(
        a_scale_ptr + offs_m, mask=offs_m < m_size, other=0.0
    ).to(tl.float32)
    w_scales = tl.load(
        w_scale_ptr + offs_n, mask=offs_n < n_size, other=0.0
    ).to(tl.float32)
    values = accumulator.to(tl.float32) * a_scales[:, None] * w_scales[None, :]
    if HAS_BIAS:
        biases = tl.load(
            bias_ptr + offs_n, mask=offs_n < n_size, other=0.0
        ).to(tl.float32)
        values += biases[None, :]

    output_offsets = offs_m[:, None] * stride_om + offs_n[None, :]
    output_mask = (offs_m[:, None] < m_size) & (offs_n[None, :] < n_size)
    tl.store(out_ptr + output_offsets, values, mask=output_mask)


def triton_w4a4_gemm(
    activation: torch.Tensor,
    weight: torch.Tensor,
    activation_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    output_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    if activation.ndim != 2 or weight.ndim != 2:
        raise ValueError("W4A4 GEMM expects 2D packed activation and weight tensors")
    if activation.dtype != torch.int8 or weight.dtype != torch.int8:
        raise TypeError("W4A4 GEMM expects packed tensors stored as int8")
    if activation.shape[1] != weight.shape[1]:
        raise ValueError("W4A4 activation and weight K dimensions do not match")
    m_size = activation.shape[0]
    n_size = weight.shape[0]
    k_size = activation.shape[1] * 2
    if k_size % 64 != 0:
        raise ValueError(f"W4A4 GEMM requires K divisible by 64, got {k_size}")
    if not activation.is_contiguous() or not weight.is_contiguous():
        raise ValueError("W4A4 GEMM expects contiguous packed tensors")
    supported_metadata_dtypes = (torch.float16, torch.bfloat16, torch.float32)
    if activation_scale.dtype not in supported_metadata_dtypes:
        raise TypeError(f"Unsupported activation scale dtype {activation_scale.dtype}")
    if weight_scale.dtype not in supported_metadata_dtypes:
        raise TypeError(f"Unsupported weight scale dtype {weight_scale.dtype}")
    if activation_scale.device != activation.device or weight_scale.device != activation.device:
        raise ValueError("W4A4 scales must already be on the activation device")
    if not activation_scale.is_contiguous() or not weight_scale.is_contiguous():
        raise ValueError("W4A4 scales must be contiguous; the native hot path will not copy them")
    if activation_scale.numel() != m_size or weight_scale.numel() != n_size:
        raise ValueError("W4A4 scale shapes do not match activation or weight rows")
    if bias is not None:
        if bias.dtype not in supported_metadata_dtypes:
            raise TypeError(f"Unsupported W4A4 bias dtype {bias.dtype}")
        if bias.device != activation.device:
            raise ValueError("W4A4 bias must already be on the activation device")
        if not bias.is_contiguous():
            raise ValueError("W4A4 bias must be contiguous; the native hot path will not copy it")
        if bias.numel() != n_size:
            raise ValueError("W4A4 bias shape does not match weight rows")
    if output_dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(f"W4A4 output must be FP16 or BF16, got {output_dtype}")

    output = torch.empty((m_size, n_size), device=activation.device, dtype=output_dtype)
    block_m, block_n, block_k, num_warps = 64, 128, 64, 8
    grid = (triton.cdiv(m_size, block_m), triton.cdiv(n_size, block_n))
    _w4a4_dot_gemm_kernel[grid](
        activation,
        weight,
        activation_scale,
        weight_scale,
        bias if bias is not None else weight_scale,
        output,
        m_size,
        n_size,
        stride_am=activation.stride(0),
        stride_wn=weight.stride(0),
        stride_om=output.stride(0),
        K_SIZE=k_size,
        HAS_BIAS=bias is not None,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=num_warps,
        num_stages=1,
    )
    return output


def _select_w4a4_tile(m_size: int, n_size: int) -> int:
    """Choose between the two precompiled WMMA layouts without autotuning.

    A 32x32 program executes four 16x16 WMMA operations.  It is faster for
    dense tiles because it reuses A and W loads, but can do substantially more
    work on narrow/tail-heavy matrices.  Keep the larger tile when its WMMA
    count is within 25% of the 16x16 layout.
    """
    if min(m_size, n_size) <= 16:
        return 16
    wmma_16 = triton.cdiv(m_size, 16) * triton.cdiv(n_size, 16)
    wmma_32 = 4 * triton.cdiv(m_size, 32) * triton.cdiv(n_size, 32)
    return 32 if wmma_32 * 4 <= wmma_16 * 5 else 16


def _select_dispatch_n_chunk(m_size: int, n_size: int, k_size: int, tile_size: int) -> int:
    """Bound a single gfx1103 compute dispatch without changing IU4 math.

    Profiling the 1024x1024 Krea2 workload showed individual native GEMM
    dispatches as long as 1.96 seconds.  The same iGPU also drives DWM, so
    those monolithic dispatches make the desktop unresponsive and approach
    Windows TDR limits.  Split only genuinely large GEMMs; small/text paths
    retain one launch.  1024 is divisible by both supported tile sizes.
    """
    work = m_size * n_size * k_size
    if work < 100_000_000_000 or n_size <= 1024:
        return n_size
    return min(n_size, 1024 - (1024 % tile_size))


def _profile_this_shape(key: tuple) -> bool:
    if not _PROFILE_ENABLED or _PROFILE_FAILED:
        return False
    count = _PROFILE_COUNTS.get(key, 0)
    _PROFILE_COUNTS[key] = count + 1
    # The first call may include Triton compilation.  Sample the second call
    # once so the diagnostic reflects the steady-state device timeline.
    return count == 1


def is_gfx1103_target() -> bool:
    global _TARGET_CHECKED, _IS_GFX1103_TARGET, _TARGET_DESCRIPTION
    if _TARGET_CHECKED:
        return _IS_GFX1103_TARGET
    try:
        target = triton.runtime.driver.active.get_current_target()
    except Exception:
        # Triton's driver may not be initialized during module import.  Do not
        # cache that transient failure; the first real GPU forward may succeed.
        return False
    _IS_GFX1103_TARGET = (
        target.backend == "hip" and target.arch == "gfx1103" and target.warp_size == 32
    )
    _TARGET_DESCRIPTION = (
        f"backend={target.backend!r}, arch={target.arch!r}, warp_size={target.warp_size}"
    )
    _TARGET_CHECKED = True
    return _IS_GFX1103_TARGET


def _require_gfx11_hip_target() -> None:
    if not is_gfx1103_target():
        raise RuntimeError(
            "Native INT4 ConvRot requires the gfx1103 HIP Triton target with wave32; "
            f"got {_TARGET_DESCRIPTION}"
        )


def triton_rotate_activation(
    x: torch.Tensor,
    group_size: int,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    _require_gfx11_hip_target()
    if x.ndim != 2 or not x.is_contiguous():
        raise ValueError("ConvRot expects a contiguous 2D activation tensor")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(f"ConvRot expects FP16, BF16, or FP32 activation, got {x.dtype}")
    if output_dtype is None:
        output_dtype = x.dtype
    if output_dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(f"ConvRot output must be FP16 or BF16, got {output_dtype}")
    if group_size not in (16, 64, 256):
        raise ValueError(f"ConvRot group size must be 16, 64, or 256, got {group_size}")
    if x.shape[1] % group_size != 0:
        raise ValueError(
            f"Activation K={x.shape[1]} must be divisible by ConvRot group size {group_size}"
        )

    grouped = x.reshape(-1, group_size)
    hadamard = _build_hadamard(group_size, device=x.device, dtype=x.dtype).contiguous()
    output = torch.empty(grouped.shape, device=x.device, dtype=output_dtype)
    block_n = 16 if group_size == 16 else 32
    block_k = 16 if group_size == 16 else 32
    grid = (triton.cdiv(grouped.shape[0], 16), triton.cdiv(group_size, block_n))
    _convrot_matmul_kernel[grid](
        grouped,
        hadamard,
        output,
        grouped.shape[0],
        GROUP_SIZE=group_size,
        BLOCK_M=16,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=4,
    )
    return output.reshape_as(x)


def triton_quantize_int4_rowwise(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    _require_gfx11_hip_target()
    if x.ndim != 2 or not x.is_contiguous():
        raise ValueError("INT4 activation quantization expects a contiguous 2D tensor")
    if x.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(f"INT4 activation quantization expects FP16 or BF16, got {x.dtype}")
    rows, k_size = x.shape
    if k_size % 64 != 0:
        raise ValueError(f"INT4 activation quantization requires K divisible by 64, got {k_size}")

    block_size = triton.next_power_of_2(k_size)
    packed = torch.empty((rows, k_size // 2), device=x.device, dtype=torch.int8)
    scales = torch.empty((rows,), device=x.device, dtype=torch.float32)
    num_warps = 8 if block_size >= 2048 else 4
    _quantize_i4_rowwise_kernel[(rows,)](
        x,
        packed,
        scales,
        k_size,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return packed, scales


def triton_convrot_w4a4_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    convrot_groupsize: int = 256,
    quant_group_size: int = 64,
    linear_dtype: str = "int4",
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    global _LOGGED_NATIVE_PATH, _PROFILE_FAILED
    _require_gfx11_hip_target()
    if not _LOGGED_NATIVE_PATH:
        logging.info(
            "INT4 Fast: using native gfx1103 Triton ConvRot W4A4 "
            "(v_wmma_i32_16x16x16_iu4, no eager fallback)."
        )
        _LOGGED_NATIVE_PATH = True
    if linear_dtype != "int4":
        raise ValueError(f"Native W4A4 kernel only supports linear_dtype='int4', got {linear_dtype!r}")
    if quant_group_size != 64:
        raise ValueError(f"Native W4A4 kernel requires quant_group_size=64, got {quant_group_size}")
    if x.device != weight.device:
        raise ValueError(f"Activation and packed weight must be on the same device, got {x.device} and {weight.device}")
    if not x.is_contiguous():
        raise ValueError("Native W4A4 activation must be contiguous; the hot path will not make an eager copy")
    if not weight.is_contiguous():
        raise ValueError("Native W4A4 packed weight must be contiguous; the hot path will not make an eager copy")
    if output_dtype is None:
        output_dtype = x.dtype
    if output_dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(f"Native W4A4 output must be FP16 or BF16, got {output_dtype}")

    original_shape = x.shape
    x_2d = x.reshape(-1, original_shape[-1])
    profile_key = (
        x_2d.shape[0],
        weight.shape[0],
        x_2d.shape[1],
        convrot_groupsize,
        x.dtype,
        output_dtype,
        weight_scale.dtype,
        None if bias is None else bias.dtype,
    )
    profile_this = _profile_this_shape(profile_key)
    events = None
    cpu_times = None
    if profile_this:
        try:
            events = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
            events[0].record()
            cpu_times = [time.perf_counter()]
        except Exception as exc:
            _PROFILE_FAILED = True
            logging.warning("INT4 Fast profiler disabled after event setup failed: %s", exc)

    rotated = triton_rotate_activation(x_2d, convrot_groupsize, output_dtype=output_dtype)
    if events is not None:
        cpu_times.append(time.perf_counter())
        events[1].record()
    packed_activation, activation_scale = triton_quantize_int4_rowwise(rotated)
    if events is not None:
        cpu_times.append(time.perf_counter())
        events[2].record()
    del rotated
    output = triton_w4a4_gemm(
        packed_activation,
        weight,
        activation_scale,
        weight_scale,
        bias,
        output_dtype=output_dtype,
    )
    if events is not None:
        cpu_times.append(time.perf_counter())
        events[3].record()
        try:
            events[3].synchronize()
            gpu_ms = [events[index].elapsed_time(events[index + 1]) for index in range(3)]
            cpu_ms = [
                (cpu_times[index + 1] - cpu_times[index]) * 1000.0
                for index in range(3)
            ]
            logging.info(
                "INT4 Fast profile M=%d N=%d K=%d group=%d tile=64x128x64/8w: "
                "GPU rotate=%.3fms quant=%.3fms gemm=%.3fms; "
                "CPU submit rotate=%.3fms quant=%.3fms gemm=%.3fms",
                x_2d.shape[0],
                weight.shape[0],
                x_2d.shape[1],
                convrot_groupsize,
                *gpu_ms,
                *cpu_ms,
            )
        except Exception as exc:
            _PROFILE_FAILED = True
            logging.warning("INT4 Fast profiler disabled after timing failed: %s", exc)
    return output.reshape(*original_shape[:-1], weight.shape[0])
