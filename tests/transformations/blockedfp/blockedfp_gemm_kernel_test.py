import dace
import numpy as np
import pytest
from dace.transformation.blockedfp.blockedfp import BlockedFP

# Symbols
N = dace.symbol("N")
M = dace.symbol("M")
L = dace.symbol("L")
S = dace.symbol("S")


@dace.program
def gemm_kernel(alpha: dace.float64,
                beta: dace.float64,
                A: dace.float64[S * N, S * M],
                B: dace.float64[S * M, S * L],
                C: dace.float64[S * N, S * L]) -> None:
    C[:] = alpha * (A @ B) + beta * C


# Compile once per block size
@pytest.fixture(scope="module", params=[2, 4, 8, 16])
def compiled_gemm_kernel(request):
    s = request.param

    sdfg = gemm_kernel.to_sdfg()
    BlockedFP(s).apply_pass(sdfg, {})
    sdfg.expand_library_nodes()
    sdfg.validate()

    return sdfg.compile(), s


# Parametrized test
@pytest.mark.parametrize("k", [1, 2, 4, 8])
@pytest.mark.parametrize("alpha", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("beta", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("scale", [1, 2, 4, 8, 16, 32])
def test_gemm_kernel(compiled_gemm_kernel, k, alpha, beta, scale):
    compiled_fn, s = compiled_gemm_kernel

    n = k
    m = k
    l = k

    n_size = n * s
    m_size = m * s
    l_size = l * s

    A = np.random.uniform(-scale, scale, (n_size, m_size))
    B = np.random.uniform(-scale, scale, (m_size, l_size))
    C = np.random.uniform(-scale, scale, (n_size, l_size))

    C_orig = C.copy()
    expected = alpha * (A @ B) + beta * C_orig

    compiled_fn(
        alpha=alpha,
        beta=beta,
        A=A,
        B=B,
        C=C,
        N=n,
        M=m,
        L=l,
        S=s,
    )

    error = np.linalg.norm(C - expected) / (n_size * l_size)

    assert error < 1e-3, (
        f"Failed with k={k}, alpha={alpha}, beta={beta}, "
        f"scale={scale}, blocksize={s}, error={error}"
    )
