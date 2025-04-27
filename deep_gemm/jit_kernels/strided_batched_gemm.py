import torch
from typing import Tuple

from .gemm import get_best_configs
from .tuner import jit_tuner
from .utils import get_col_major_tma_aligned_tensor, get_num_sms

# C++ code templates
includes = ('"deep_gemm/fp8_gemm.cuh"', )
template = """
using namespace deep_gemm;

// Templated args from Python JIT call
constexpr auto N = {N}, K = {K};
constexpr auto BLOCK_M = {BLOCK_M};
constexpr auto BLOCK_N = {BLOCK_N};
constexpr auto kNumStages = {NUM_STAGES};
constexpr auto kNumTMAMulticast = {NUM_TMA_MULTICAST};

// Make a templated grouped GEMM
using GemmType = Gemm<N, K, BLOCK_M, BLOCK_N, 128, {NUM_GROUPS}, kNumStages, kNumTMAMulticast, GemmType::{GEMM_TYPE}>;

// Launch kernel
auto tma_a_desc = GemmType::make_2d_tma_a_desc(lhs, m * {NUM_GROUPS}, ld_lhs_in_bytes);
auto tma_b_desc = GemmType::make_2d_tma_b_desc(rhs, ld_rhs_in_bytes);
auto tma_scales_a_desc = GemmType::make_2d_tma_scales_a_desc(lhs_scales, m);

// Will not be used for strided batched GEMM for now.
auto tma_d_desc = GemmType::make_2d_tma_d_desc(out, m, ld_out_in_bytes);

GemmType::run(out, rhs_scales, m, tma_a_desc, tma_b_desc, tma_scales_a_desc, tma_d_desc,
              ld_lhs_in_bytes / 1, stride_lhs, ld_rhs_in_bytes / 1, stride_rhs, ld_out_in_bytes / 2,
              stride_out, stream, num_sms, smem_size);
"""

def strided_batched_gemm_fp8_fp8_bf16_nt(lhs: Tuple[torch.Tensor, torch.Tensor],
                                                   rhs: Tuple[torch.Tensor, torch.Tensor],
                                                   out: torch.Tensor, 
                                                   ld_lhs: int, ld_rhs: int, ld_out: int, 
                                                   stride_lhs: int, stride_rhs: int, stride_out: int) -> None:
    """
    Current implementation notes:
      - For the input `lhs`, it must satisfy stride_lhs % ld_lhs == 0 and stride_lhs >= ld_lhs.
      - For the input `rhs`, it must satisfy stride_rhs % ld_rhs == 0 and stride_rhs >= ld_rhs.
      - For the output tensor (`out`), there are no such requirements on stride_out.
    """
    lhs, lhs_scales = lhs
    rhs, rhs_scales = rhs
    num_groups, m, k = lhs.shape
    num_groups_, n, k_ = rhs.shape
    m_, n_ = out.shape

    # Type and shape checks
    assert num_groups == num_groups_
    assert m == m_ and n * num_groups == n_ and k == k_
    # assert lhs_scales.shape == (num_groups_, m, (k + 127) // 128)
    assert rhs_scales.shape == (num_groups_, (n + 127) // 128, (k + 127) // 128)
    assert lhs.dtype == torch.float8_e4m3fn and lhs_scales.dtype == torch.float32
    assert rhs.dtype == torch.float8_e4m3fn and rhs_scales.dtype == torch.float32
    assert out.dtype == torch.bfloat16
    assert lhs.is_contiguous() and rhs.is_contiguous()
    assert out.is_contiguous()

    # # LHS scales must be transposed for TMA load, but not for RHS scales
    lhs_scales = get_col_major_tma_aligned_tensor(lhs_scales)
    assert rhs_scales.is_contiguous()

    # Auto-tuning with compilation
    global includes, template
    num_sms = get_num_sms()
    block_m, block_n, num_stages, num_tma_multicast, smem_size = get_best_configs(m, n, k, num_groups_, num_sms)
    args = (lhs, ld_lhs * 1, stride_lhs, lhs_scales, rhs, ld_rhs * 1, stride_rhs, rhs_scales, out, ld_out * 2,
            stride_out, m, torch.cuda.current_stream(), num_sms, smem_size)
    runtime = jit_tuner.compile_and_tune(
        name='m_grouped_gemm_fp8_fp8_bf16_nt',
        keys={'N': n, 'K': k, 'BLOCK_M': block_m, 'BLOCK_N': block_n, 'NUM_GROUPS': num_groups_,
              'NUM_STAGES': num_stages, 'NUM_TMA_MULTICAST': num_tma_multicast, 'GEMM_TYPE': 'StridedBatched'},
        space=(),
        includes=includes,
        arg_defs=(('lhs', torch.float8_e4m3fn), ('ld_lhs_in_bytes', int), ('stride_lhs', int), ('lhs_scales', torch.float),
                  ('rhs', torch.float8_e4m3fn), ('ld_rhs_in_bytes', int), ('stride_rhs', int), ('rhs_scales', torch.float),
                  ('out', torch.bfloat16), ('ld_out_in_bytes', int), ('stride_out', int), ('m', int), 
                  ('stream', torch.cuda.Stream), ('num_sms', int), ('smem_size', int)),
        template=template,
        args=args
    )

    # Run the kernel
    runtime(*args)
