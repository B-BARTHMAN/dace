import dace
import numpy as np
import pytest
from dace.transformation.blockedfp.blockedfp import BlockedFP

N = dace.symbol("N")
M = dace.symbol("M")
L = dace.symbol("L")

@dace.program
def gemm_kernel(alpha: dace.float64, beta: dace.float64, 
                C: dace.float64[16 * N, 16 * L], 
                A: dace.float64[16 * N, 16 * M], 
                B: dace.float64[16 * M, 16 * L]):
    C[:] = alpha * A @ B + beta * C

# Fixture to compile once
@pytest.fixture(scope="module")
def compiled_gemm_kernel():
    sdfg = gemm_kernel.to_sdfg()
    BlockedFP().apply_pass(sdfg, {})
    sdfg.expand_library_nodes()
    sdfg.validate()
    return sdfg.compile()

# Parametrized test
@pytest.mark.parametrize("k", [1, 2, 4, 8])         # matrix size multipliers
@pytest.mark.parametrize("alpha", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("beta", [0.0, 0.5, 1.0])
def test_gemm_kernel(compiled_gemm_kernel, k, alpha, beta):
    n = 16 * k
    m = 16 * k
    l = 16 * k

    # Random matrices
    A = np.random.uniform(-1, 1, (n, m))
    B = np.random.uniform(-1, 1, (m, l))
    C = np.random.uniform(-1, 1, (n, l))

    C_orig = C.copy()  # store original C for expected computation
    expected = alpha * (A @ B) + beta * C_orig

    compiled_gemm_kernel(
        alpha=alpha, beta=beta, C=C, A=A, B=B,
        N=n // 16, M=m // 16, L=l // 16
    )

    error = np.linalg.norm(C - expected) / (n * l)
    assert error < 1e-12, f"Failed with k={k}, alpha={alpha}, beta={beta}, error={error:.3e}"

x = gemm_kernel.to_sdfg()
BlockedFP().apply_pass(x, {})
x.view()