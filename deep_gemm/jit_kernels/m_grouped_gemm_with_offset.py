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
auto tma_a_desc = GemmType::make_2d_tma_a_desc(lhs, max_m_total);
auto tma_b_desc = GemmType::make_2d_tma_b_desc(rhs);
auto tma_scales_a_desc = GemmType::make_tma_scales_a_offset_desc(lhs_scales, max_m_padded_4_total);
auto tma_d_desc = GemmType::make_2d_tma_d_desc(out, max_m_total);
GemmType::run(out, rhs_scales, token_offset, scale_offset,
              tma_a_desc, tma_b_desc, tma_scales_a_desc, tma_d_desc,
              stream, num_sms, smem_size);
"""

template_swapAB = """
using namespace deep_gemm;

// Templated args from Python JIT call
constexpr auto M = {M}, K = {K};
constexpr auto BLOCK_M = {BLOCK_M};
constexpr auto BLOCK_N = {BLOCK_N};
constexpr auto kNumStages = {NUM_STAGES};
constexpr auto kNumTMAMulticast = {NUM_TMA_MULTICAST};

// Make a templated grouped GEMM
using GemmType = GemmSwapAB<M, K, BLOCK_M, BLOCK_N, 128, {NUM_GROUPS}, kNumStages, kNumTMAMulticast, GemmType::{GEMM_TYPE}>;

// Launch kernel
auto tma_a_desc = GemmType::make_2d_tma_a_desc(lhs); // a是左值,weight
auto tma_b_desc = GemmType::make_2d_tma_b_desc(rhs, max_n_total);  // b是右值,act
auto tma_scales_b_desc = GemmType::make_tma_scales_b_offset_desc(rhs_scales, max_n_padded_4_total);  // act scales
auto tma_d_desc = GemmType::make_2d_tma_d_desc(out, max_n_total);
GemmType::run(out, lhs_scales, token_offset, scale_offset,
              tma_a_desc, tma_b_desc, tma_scales_b_desc, tma_d_desc,
              stream, num_sms, smem_size);
"""

def m_grouped_gemm_fp8_fp8_bf16_nt_offset(lhs: Tuple[torch.Tensor, torch.Tensor],
                                          rhs: Tuple[torch.Tensor, torch.Tensor],
                                          out: torch.Tensor, token_offset: torch.Tensor,
                                          scale_offset: torch.Tensor, expected_m: int,
                                          max_m_total: int) -> None:
    lhs, lhs_scales = lhs
    rhs, rhs_scales = rhs
    m, k = lhs.shape
    num_groups_, n, k_ = rhs.shape
    m_, n_ = out.shape
    num_groups___ = token_offset.numel() - 1
    max_m_padded_4_total = lhs_scales.shape[0]

    # Type and shape checks
    assert num_groups_ == num_groups___
    assert m == m_ and n == n_ and k == k_
    assert expected_m > 0 and m > 0 and n > 0 and k > 0 and num_groups_ > 0
    # assert lhs_scales.shape == (num_groups_, m, (k + 127) // 128)
    assert rhs_scales.shape == (num_groups_, (n + 127) // 128, (k + 127) // 128)
    assert lhs.dtype == torch.float8_e4m3fn and lhs_scales.dtype == torch.float32
    assert rhs.dtype == torch.float8_e4m3fn and rhs_scales.dtype == torch.float32
    assert out.dtype == torch.bfloat16
    assert token_offset.dtype == torch.int64
    assert lhs.is_contiguous() and rhs.is_contiguous()
    assert out.is_contiguous() and token_offset.is_contiguous()

    # # LHS scales must be transposed for TMA load, but not for RHS scales
    lhs_scales = get_col_major_tma_aligned_tensor(lhs_scales)
    assert rhs_scales.is_contiguous()

    # Auto-tuning with compilation
    global includes, template
    num_sms = get_num_sms()
    # print(f'num_sms: {num_sms}')
    block_m, block_n, num_stages, num_tma_multicast, smem_size = get_best_configs(expected_m, n, k, num_groups_, num_sms)
    # print(f'block_m: {block_m}, block_n: {block_n}, num_stages: {num_stages}, num_tma_multicast: {num_tma_multicast}, smem_size: {smem_size}')
    args = (lhs, lhs_scales, rhs, rhs_scales, out,
            token_offset, scale_offset, torch.cuda.current_stream(), num_sms, smem_size, max_m_total, max_m_padded_4_total)
    runtime = jit_tuner.compile_and_tune(
        name='m_grouped_gemm_fp8_fp8_bf16_nt',
        keys={'N': n, 'K': k, 'BLOCK_M': block_m, 'BLOCK_N': block_n, 'NUM_GROUPS': num_groups_,
              'NUM_STAGES': num_stages, 'NUM_TMA_MULTICAST': num_tma_multicast, 'GEMM_TYPE': 'GroupedWithOffset'},
        space=(),
        includes=includes,
        arg_defs=(('lhs', torch.float8_e4m3fn), ('lhs_scales', torch.float),
                  ('rhs', torch.float8_e4m3fn), ('rhs_scales', torch.float),
                  ('out', torch.bfloat16),
                  ('token_offset', torch.int64), ('scale_offset', torch.int64),
                  ('stream', torch.cuda.Stream), ('num_sms', int), ('smem_size', int),
                  ('max_m_total', int), ('max_m_padded_4_total', int)),
        template=template,
        args=args
    )

    # Run the kernel
    runtime(*args)

def m_grouped_gemm_fp8_fp8_bf16_nt_offset_swapAB(lhs: Tuple[torch.Tensor, torch.Tensor],
                                          rhs: Tuple[torch.Tensor, torch.Tensor],
                                          out: torch.Tensor, token_offset: torch.Tensor,
                                          scale_offset: torch.Tensor, expected_n: int,
                                          max_n_total: int) -> None:
    lhs, lhs_scales = lhs  # weight
    rhs, rhs_scales = rhs  # act
    num_groups_, m, k = lhs.shape
    n, k_ = rhs.shape
    n_, m_ = out.shape  # torch里是row-major，cutlass里保持row-major
    num_groups___ = token_offset.numel() - 1
    max_n_padded_4_total = rhs_scales.shape[0]
    # Type and shape checks
    assert num_groups_ == num_groups___
    assert m == m_ and n == n_ and k == k_
    assert expected_n > 0 and m > 0 and n > 0 and k > 0 and num_groups_ > 0
    # assert lhs_scales.shape == (num_groups_, m, (k + 127) // 128)
    assert lhs_scales.shape == (num_groups_, (m + 127) // 128, (k + 127) // 128)
    assert lhs.dtype == torch.float8_e4m3fn and lhs_scales.dtype == torch.float32
    assert rhs.dtype == torch.float8_e4m3fn and rhs_scales.dtype == torch.float32
    assert out.dtype == torch.bfloat16
    assert token_offset.dtype == torch.int64
    assert lhs.is_contiguous() and rhs.is_contiguous()
    assert (out.is_contiguous()) and token_offset.is_contiguous()

    # # RHS scales must be transposed for TMA load, but not for LHS scales
    rhs_scales = get_col_major_tma_aligned_tensor(rhs_scales)
    assert lhs_scales.is_contiguous()

    # Auto-tuning with compilation
    global includes, template_swapAB
    num_sms = get_num_sms()
    # print(f'num_sms: {num_sms}')
    # TODO: num_tma_multicast在num_groups_ == 1时，结果可能是有问题的，现在blocks在m方向上连续排列
    block_m, block_n, num_stages, num_tma_multicast, smem_size = get_best_configs(m, expected_n, k, num_groups_, num_sms, swapAB=True)
    # print(f'block_m: {block_m}, block_n: {block_n}, num_stages: {num_stages}, num_tma_multicast: {num_tma_multicast}, smem_size: {smem_size}')
    args = (lhs, lhs_scales, rhs, rhs_scales, out,
            token_offset, scale_offset, torch.cuda.current_stream(), num_sms, smem_size, max_n_total, max_n_padded_4_total) # max_n_total是act的总token数量
    runtime = jit_tuner.compile_and_tune(
        name='m_grouped_gemm_fp8_fp8_bf16_nt_swapAB',
        keys={'M': m, 'K': k, 'BLOCK_M': block_m, 'BLOCK_N': block_n, 'NUM_GROUPS': num_groups_,
              'NUM_STAGES': num_stages, 'NUM_TMA_MULTICAST': num_tma_multicast, 'GEMM_TYPE': 'GroupedWithOffset'},
        space=(),
        includes=includes,
        arg_defs=(('lhs', torch.float8_e4m3fn), ('lhs_scales', torch.float),
                  ('rhs', torch.float8_e4m3fn), ('rhs_scales', torch.float),
                  ('out', torch.bfloat16),
                  ('token_offset', torch.int64), ('scale_offset', torch.int64),
                  ('stream', torch.cuda.Stream), ('num_sms', int), ('smem_size', int),
                  ('max_n_total', int), ('max_n_padded_4_total', int)),
        template=template_swapAB,
        args=args
    )

    # Run the kernel
    runtime(*args)
