import random
import torch
from typing import Tuple

import deep_gemm
from deep_gemm import bench_kineto, calc_diff, ceil_div, get_col_major_tma_aligned_tensor


def per_token_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2 and x.size(1) % 128 == 0
    m, n = x.shape
    x_view = x.view(m, -1, 128)
    x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
    return (x_view * (448.0 / x_amax.unsqueeze(2))).to(torch.float8_e4m3fn).view(m, n), (x_amax / 448.0).view(m, -1)

# act scale是m-major，需要16bytes对齐，也就是4对齐；act是k-major，k足够大，不需要再显式对齐
def per_token_cast_to_fp8_padded(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2 and x.size(1) % 128 == 0
    m, n = x.shape
    aligned_m = ceil_div(m, 4) * 4
    x_view = x.view(m, -1, 128)
    x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
    scales_padded = torch.zeros((aligned_m, x_amax.shape[-1]), dtype=x_amax.dtype, device=x_amax.device)
    scales_padded[:m, :] = (x_amax / 448.0).view(m, -1)
    return (x_view * (448.0 / x_amax.unsqueeze(2))).to(torch.float8_e4m3fn).view(m, n), scales_padded


def per_block_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    m, n = x.shape
    x_padded = torch.zeros((ceil_div(m, 128) * 128, ceil_div(n, 128) * 128), dtype=x.dtype, device=x.device)
    x_padded[:m, :n] = x
    x_view = x_padded.view(-1, 128, x_padded.size(1) // 128, 128)
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    x_scaled = (x_view * (448.0 / x_amax)).to(torch.float8_e4m3fn)
    return x_scaled.view_as(x_padded)[:m, :n].contiguous(), (x_amax / 448.0).view(x_view.size(0), x_view.size(2))


def construct(m: int, k: int, n: int) -> \
        Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
    x = torch.randn((m, k), device='cuda', dtype=torch.bfloat16)
    y = torch.randn((n, k), device='cuda', dtype=torch.bfloat16)
    out = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)
    ref_out = x @ y.t()

    x_fp8, y_fp8 = per_token_cast_to_fp8(x), per_block_cast_to_fp8(y)
    # Transpose earlier so that the testing will not trigger transposing kernels
    x_fp8 = (x_fp8[0], get_col_major_tma_aligned_tensor(x_fp8[1]))
    return x_fp8, y_fp8, out, ref_out


def construct_grouped(num_groups: int, m: int, k: int, n: int, input_type: str = 'general') -> \
        Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
    seed = 42
    torch.manual_seed(seed)  # PyTorch CPU
    torch.cuda.manual_seed(seed)  # PyTorch GPU单卡
    x = torch.randn((num_groups, m, k), device='cuda', dtype=torch.bfloat16)
    y = torch.randn((num_groups, n, k), device='cuda', dtype=torch.bfloat16)
    out = torch.empty((num_groups, m, n), device='cuda', dtype=torch.bfloat16)
    ref_out = torch.einsum('gmk,gnk->gmn', x, y)

    assert m % 4 == 0 or input_type == 'offset', f'TMA alignment error: {m}'
    aligned_m = ceil_div(m, 4) * 4
    x_fp8 = (torch.empty_like(x, dtype=torch.float8_e4m3fn), torch.empty((num_groups, aligned_m, k // 128), device='cuda', dtype=torch.float))
    y_fp8 = (torch.empty_like(y, dtype=torch.float8_e4m3fn), torch.empty((num_groups, (n + 127) // 128, k // 128), device='cuda', dtype=torch.float))
    for i in range(num_groups):
        x_fp8[0][i], x_fp8[1][i] = per_token_cast_to_fp8_padded(x[i])
        y_fp8[0][i], y_fp8[1][i] = per_block_cast_to_fp8(y[i])

    # For non-masked input, we must merge the group and M dims
    if input_type == 'contiguous':
        x_fp8 = (x_fp8[0].view(-1, k), per_token_cast_to_fp8(x.view(-1, k))[1])
        out, ref_out = out.view(-1, n), ref_out.view(-1, n)

    # Transpose earlier so that the testing will not trigger transposing kernels
    x_fp8 = (x_fp8[0], get_col_major_tma_aligned_tensor(x_fp8[1]))
    return x_fp8, y_fp8, out, ref_out


def test_gemm() -> None:
    print('Testing GEMM:')
    # for m in (64, 128, 4096):
    for m in (1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256):
        # for k, n in [(7168, 2112), (1536, 24576), (512, 32768), (16384, 7168), (7168, 4096), (2048, 7168)]:
        for k, n in [(7168, 18432)]:
            x_fp8, y_fp8, out, ref_out = construct(m, k, n)
            deep_gemm.gemm_fp8_fp8_bf16_nt(x_fp8, y_fp8, out)
            diff = calc_diff(out, ref_out)
            assert diff < 0.001, f'{m=}, {k=}, {n=}, {diff:.5f}'

            # noinspection PyShadowingNames
            def test_func():
                # Construct new tensors every time to avoid L2 cache acceleration
                x_fp8, y_fp8, out, ref_out = construct(m, k, n)
                deep_gemm.gemm_fp8_fp8_bf16_nt(x_fp8, y_fp8, out)

            t = bench_kineto(test_func, 'fp8_gemm', suppress_kineto_output=True)
            print(f' > Performance (m={m:5}, n={n:5}, k={k:5}): {t * 1e6:4.0f} us | '
                  f'throughput: {2 * m * n * k / t / 1e12:4.0f} TFLOPS, '
                  f'{(m * k + k * n + m * n * 2) / 1e9 / t:4.0f} GB/s')
    print()

def test_gemm_swapAB() -> None:
    print('Testing GEMM (swapAB):')
    # for m in (64, 128, 4096):
    for m in (1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256):
        # for k, n in [(7168, 2112), (1536, 24576), (512, 32768), (16384, 7168), (7168, 4096), (2048, 7168)]:
        for k, n in [(7168, 18432)]:
            x_fp8, y_fp8, out, ref_out = construct(m, k, n)
            deep_gemm.gemm_fp8_fp8_bf16_nt_swapAB(y_fp8, x_fp8, out)
            diff = calc_diff(out, ref_out)
            assert diff < 0.001, f'{m=}, {k=}, {n=}, {diff:.5f}'

            # noinspection PyShadowingNames
            def test_func():
                # Construct new tensors every time to avoid L2 cache acceleration
                x_fp8, y_fp8, out, ref_out = construct(m, k, n)
                deep_gemm.gemm_fp8_fp8_bf16_nt_swapAB(y_fp8, x_fp8, out)

            t = bench_kineto(test_func, 'fp8_gemm_kernel_swapAB', suppress_kineto_output=True)
            print(f' > Performance (m={m:5}, n={n:5}, k={k:5}): {t * 1e6:4.0f} us | '
                  f'throughput: {2 * m * n * k / t / 1e12:4.0f} TFLOPS, '
                  f'{(m * k + k * n + m * n * 2) / 1e9 / t:4.0f} GB/s')
    print()


def test_m_grouped_gemm_contiguous() -> None:
    print('Testing grouped contiguous GEMM:')

    for num_groups, m, k, n in ((4, 8192, 7168, 4096), (4, 8192, 2048, 7168), (8, 4096, 7168, 4096), (8, 4096, 2048, 7168)):
        # TODO: make a stronger test
        x_fp8, y_fp8, out, ref_out = construct_grouped(num_groups, m, k, n, input_type='contiguous')
        m_indices = torch.arange(0, num_groups, device='cuda', dtype=torch.int)
        m_indices = m_indices.unsqueeze(-1).expand(num_groups, m).contiguous().view(-1)
        deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(x_fp8, y_fp8, out, m_indices)
        diff = calc_diff(out, ref_out)
        assert diff < 0.001, f'm={m * num_groups}, {k=}, {n=}, {diff:.5f}'

        # noinspection PyShadowingNames
        def test_func():
            # Construct new tensors every time to avoid L2 cache acceleration
            x_fp8, y_fp8, out, ref_out = construct_grouped(num_groups, m, k, n, input_type='contiguous')
            m_indices = torch.arange(0, num_groups, device='cuda', dtype=torch.int)
            m_indices = m_indices.unsqueeze(-1).expand(num_groups, m).contiguous().view(-1)
            deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(x_fp8, y_fp8, out, m_indices)

        t = bench_kineto(test_func, 'fp8_gemm', suppress_kineto_output=True)
        print(f' > Performance ({num_groups=}, m_per_group={m:4}, n={n:4}, k={k:4}): {t * 1e6:4.0f} us | '
              f'throughput: {2 * num_groups * m * n * k / t / 1e12:4.0f} TFLOPS, '
              f'{(num_groups * (m * k + k * n + m * n * 2)) / 1e9 / t:4.0f} GB/s')
    print()


def test_m_grouped_gemm_masked() -> None:
    print('Testing grouped masked GEMM:')

    for num_groups, m in ((1, 1024), (2, 512), (4, 256)):
        for k, n in ((7168, 4096), (2048, 7168), ):
            # Test correctness
            masked_m_candidates = list(filter(lambda candidate: candidate <= m, (64, 128, 192, 256, 320, 384)))
            for i in range(10):
                x_fp8, y_fp8, out, ref_out = construct_grouped(num_groups, m, k, n, input_type='masked')
                masked_m = torch.empty((num_groups, ), device='cuda', dtype=torch.int)
                for j in range(num_groups):
                    masked_m[j] = random.choice(masked_m_candidates)
                expected_m = min(int(masked_m.float().mean()) + 1, m)
                deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_masked(x_fp8, y_fp8, out, masked_m, expected_m)
                for j in range(num_groups):
                    diff = calc_diff(out[j, :masked_m[j].item()], ref_out[j, :masked_m[j].item()])
                    assert diff < 0.001, f'{m=}, {k=}, {n=}, {j=}, masked_m={masked_m[j]}, {num_groups=}, {diff:.5f}'

            # noinspection PyShadowingNames
            def test_func():
                # Construct new tensors every time to avoid L2 cache acceleration
                x_fp8, y_fp8, out, ref_out = construct_grouped(num_groups, m, k, n, input_type='masked')
                masked_m = torch.ones((num_groups, ), device='cuda', dtype=torch.int) * m
                deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_masked(x_fp8, y_fp8, out, masked_m, m)

            # Test performance with fixed shapes
            t = bench_kineto(test_func, 'fp8_gemm', suppress_kineto_output=True)
            print(f' > Performance ({num_groups=}, m_per_group={m:4}, n={n:4}, k={k:4}): {t * 1e6:4.0f} us | '
                  f'throughput: {2 * num_groups * m * n * k / t / 1e12:4.0f} TFLOPS, '
                  f'{(num_groups * (m * k + k * n + m * n * 2)) / 1e9 / t:4.0f} GB/s')
    print()


def change_to_offset_layout(x_fp8, out, ref_out, offset) -> torch.Tensor:
    x_list = []
    x_scale_list = []
    out_list = []
    ref_out_list = []

    for i in range(x_fp8[0].shape[0]):
        m = x_fp8[0].shape[1]
        x_list.append(x_fp8[0][i][0:m])
        x_scale_list.append(x_fp8[1][i])
        out_list.append(out[i][0:m])
        ref_out_list.append(ref_out[i][0:m])
        offset[i + 1] = offset[i] + m

    ret_x = torch.cat(x_list)
    ret_x_scale = torch.cat(x_scale_list)
    ret_out = torch.cat(out_list)
    ret_ref_out = torch.cat(ref_out_list)

    return (ret_x, ret_x_scale), ret_out, ret_ref_out, offset

def change_to_strided_batched_layout(x_fp8, out, ref_out) -> torch.Tensor:
    out_list = []
    ref_out_list = []

    for i in range(x_fp8[0].shape[0]):
        out_list.append(out[i])
        ref_out_list.append(ref_out[i])

    ret_out = torch.cat(out_list, dim=1)
    ret_ref_out = torch.cat(ref_out_list, dim=1)
    return ret_out, ret_ref_out

def test_m_grouped_gemm_offset() -> None:
    print('Testing grouped GEMM with token offset:')
    for num_groups, m in ([(2, 2), (2, 4), (2, 8), (2, 16), (2, 24), (2, 32), (2, 40), (2, 48), (2, 56), (2, 64), (2, 72), (2, 80), (2, 88), (2, 96), (2, 104), (2, 112), (2, 120), (2, 128), (2, 136), (2, 144), (2, 152), (2, 160), (2, 168), (2, 176), (2, 184), (2, 192), (2, 200), (2, 208), (2, 216), (2, 224), (2, 232), (2, 240), (2, 248), (2, 256),
                            (4, 2), (4, 4), (4, 8), (4, 16), (4, 24), (4, 32), (4, 40), (4, 48), (4, 56), (4, 64), (4, 72), (4, 80), (4, 88), (4, 96), (4, 104), (4, 112), (4, 120), (4, 128), (4, 136), (4, 144), (4, 152), (4, 160), (4, 168), (4, 176), (4, 184), (4, 192), (4, 200), (4, 208), (4, 216), (4, 224), (4, 232), (4, 240), (4, 248), (4, 256),
                            (8, 2), (8, 4), (8, 8), (8, 16), (8, 24), (8, 32), (8, 40), (8, 48), (8, 56), (8, 64), (8, 72), (8, 80), (8, 88), (8, 96), (8, 104), (8, 112), (8, 120), (8, 128), (8, 136), (8, 144), (8, 152), (8, 160), (8, 168), (8, 176), (8, 184), (8, 192), (8, 200), (8, 208), (8, 216), (8, 224), (8, 232), (8, 240), (8, 248), (8, 256),
                            ]):
        for k, n in ([(2048, 7168)]):
            offset = torch.tensor([i * m for i in range(num_groups + 1)], device='cuda', dtype=torch.int64)
            aligned_m = ceil_div(m, 4) * 4
            scales_offset = torch.tensor([i * aligned_m for i in range(num_groups + 1)], device='cuda', dtype=torch.int64)
            x_fp8, y_fp8, out, ref_out = construct_grouped(num_groups, m, k, n, input_type='offset')
            x_fp8, out, ref_out, offset = change_to_offset_layout(x_fp8, out, ref_out, offset)
            deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_offset(x_fp8, y_fp8, out, offset, scales_offset, m, m * num_groups)
            for j in range(num_groups):
                diff = calc_diff(out, ref_out)
                assert diff < 0.001, f'm={m * num_groups}, {k=}, {n=}, {diff:.5f}'

            # noinspection PyShadowingNames
            def test_func():
                # Construct new tensors every time to avoid L2 cache acceleration
                offset = torch.tensor([i * m for i in range(num_groups + 1)], device='cuda', dtype=torch.int64)
                aligned_m = ceil_div(m, 4) * 4
                scales_offset = torch.tensor([i * aligned_m for i in range(num_groups + 1)], device='cuda', dtype=torch.int64)
                x_fp8, y_fp8, out, ref_out = construct_grouped(num_groups, m, k, n, input_type='offset')
                x_fp8, out, ref_out, offset = change_to_offset_layout(x_fp8, out, ref_out, offset)
                deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_offset(x_fp8, y_fp8, out, offset, scales_offset, m, m * num_groups)

            # Test performance with fixed shapes
            t = bench_kineto(test_func, 'fp8_gemm', suppress_kineto_output=True)
            print(f' > Performance ({num_groups=}, m_per_group={m:4}, n={n:4}, k={k:4}): {t * 1e6:4.0f} us | '
                f'throughput: {2 * num_groups * m * n * k / t / 1e12:4.0f} TFLOPS, '
                f'{(num_groups * (m * k + k * n + m * n * 2)) / 1e9 / t:4.0f} GB/s')

    print()

def test_m_grouped_gemm_offset_swapAB() -> None:
    print('Testing grouped GEMM with token offset (swapAB):')
    for num_groups, m in ([(2, 2), (2, 4), (2, 8), (2, 16), (2, 24), (2, 32), (2, 40), (2, 48), (2, 56), (2, 64), (2, 72), (2, 80), (2, 88), (2, 96), (2, 104), (2, 112), (2, 120), (2, 128), (2, 136), (2, 144), (2, 152), (2, 160), (2, 168), (2, 176), (2, 184), (2, 192), (2, 200), (2, 208), (2, 216), (2, 224), (2, 232), (2, 240), (2, 248), (2, 256),
                            (4, 2), (4, 4), (4, 8), (4, 16), (4, 24), (4, 32), (4, 40), (4, 48), (4, 56), (4, 64), (4, 72), (4, 80), (4, 88), (4, 96), (4, 104), (4, 112), (4, 120), (4, 128), (4, 136), (4, 144), (4, 152), (4, 160), (4, 168), (4, 176), (4, 184), (4, 192), (4, 200), (4, 208), (4, 216), (4, 224), (4, 232), (4, 240), (4, 248), (4, 256),
                            (8, 2), (8, 4), (8, 8), (8, 16), (8, 24), (8, 32), (8, 40), (8, 48), (8, 56), (8, 64), (8, 72), (8, 80), (8, 88), (8, 96), (8, 104), (8, 112), (8, 120), (8, 128), (8, 136), (8, 144), (8, 152), (8, 160), (8, 168), (8, 176), (8, 184), (8, 192), (8, 200), (8, 208), (8, 216), (8, 224), (8, 232), (8, 240), (8, 248), (8, 256),
                            ]):
        for k, n in ([(2048, 7168),]):
            offset = torch.tensor([i * m for i in range(num_groups + 1)], device='cuda', dtype=torch.int64)
            aligned_m = ceil_div(m, 4) * 4
            scales_offset = torch.tensor([i * aligned_m for i in range(num_groups + 1)], device='cuda', dtype=torch.int64)
            x_fp8, y_fp8, out, ref_out = construct_grouped(num_groups, m, k, n, input_type='offset')
            x_fp8, out, ref_out, offset = change_to_offset_layout(x_fp8, out, ref_out, offset)
            deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_offset_swapAB(y_fp8, x_fp8, out, offset, scales_offset, m, m * num_groups)
            
            for j in range(num_groups):
                diff = calc_diff(out, ref_out)
                assert diff < 0.001, f'm={m * num_groups}, {k=}, {n=}, {diff:.5f}'

            # noinspection PyShadowingNames
            def test_func():
                # Construct new tensors every time to avoid L2 cache acceleration
                offset = torch.tensor([i * m for i in range(num_groups + 1)], device='cuda', dtype=torch.int64)
                aligned_m = ceil_div(m, 4) * 4
                scales_offset = torch.tensor([i * aligned_m for i in range(num_groups + 1)], device='cuda', dtype=torch.int64)
                x_fp8, y_fp8, out, ref_out = construct_grouped(num_groups, m, k, n, input_type='offset')
                x_fp8, out, ref_out, offset = change_to_offset_layout(x_fp8, out, ref_out, offset)
                deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_offset_swapAB(y_fp8, x_fp8, out, offset, scales_offset, m, m * num_groups)
            
            # Test performance with fixed shapes
            t = bench_kineto(test_func, 'fp8_gemm_kernel_swapAB', suppress_kineto_output=True)
            print(f' > Performance ({num_groups=}, m_per_group={m:4}, n={n:4}, k={k:4}): {t * 1e6:4.0f} us | '
                  f'throughput: {2 * num_groups * m * n * k / t / 1e12:4.0f} TFLOPS, '
                  f'{(num_groups * (m * k + k * n + m * n * 2)) / 1e9 / t:4.0f} GB/s')

    print()

def test_strided_batched_gemm() -> None:
    print('Testing strided batched GEMM:')
    for num_groups, m in ([(1, 1024), (2, 512), (4, 256)]):
        for k, n in ([(7168, 4096), (2048, 7168)]):
            for i in range(1):
                x_fp8, y_fp8, out, ref_out = construct_grouped(num_groups, m, k, n)
                out, ref_out = change_to_strided_batched_layout(x_fp8, out, ref_out)
                deep_gemm.strided_batched_gemm_fp8_fp8_bf16_nt(x_fp8, y_fp8, out, k, k, num_groups * n, m * k, n * k, n)
                for j in range(num_groups):
                    diff = calc_diff(out, ref_out)
                    assert diff < 0.001, f'm={m * num_groups}, {k=}, {n=}, {diff:.5f}'

                # noinspection PyShadowingNames
                def test_func():
                    # Construct new tensors every time to avoid L2 cache acceleration
                    x_fp8, y_fp8, out, ref_out = construct_grouped(num_groups, m, k, n)
                    out, ref_out = change_to_strided_batched_layout(x_fp8, out, ref_out)
                    deep_gemm.strided_batched_gemm_fp8_fp8_bf16_nt(x_fp8, y_fp8, out, k, k, num_groups * n, m * k, n * k, n)

                # Test performance with fixed shapes
                t = bench_kineto(test_func, 'fp8_gemm', suppress_kineto_output=True)
                print(f' > Performance ({num_groups=}, m_per_group={m:4}, n={n:4}, k={k:4}): {t * 1e6:4.0f} us | '
                    f'throughput: {2 * num_groups * m * n * k / t / 1e12:4.0f} TFLOPS, '
                    f'{(num_groups * (m * k + k * n + m * n * 2)) / 1e9 / t:4.0f} GB/s')

    print()


if __name__ == '__main__':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(0)
    random.seed(0)

    print('Library path:')
    print(f' > {deep_gemm.__path__}\n')

    # test_gemm()
    # test_gemm_swapAB()
    # test_m_grouped_gemm_contiguous()
    # test_m_grouped_gemm_masked()
    test_m_grouped_gemm_offset()
    test_m_grouped_gemm_offset_swapAB()
    # test_strided_batched_gemm()