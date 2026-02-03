import dace
import numpy as np
from dace.transformation.blockedfp.blockedfp import BlockedFP
import pytest

N = dace.symbol("N")
M = dace.symbol("M")
L = dace.symbol("L")
S = dace.symbol("S")

@dace.program
def gemm(A: dace.float64[S * N, S * M], B: dace.float64[S * M, S * L], C: dace.float64[S * N, S * L]) -> None:
    C = A @ B

# Fixture compiles SDFG once
@pytest.fixture(scope="module", params=[2, 4, 8, 16])
def compiled_gemm(request):
    s = request.param
    
    sdfg = gemm.to_sdfg()
    BlockedFP(s).apply_pass(sdfg, {})
    sdfg.expand_library_nodes()
    sdfg.validate()
    return sdfg.compile(), s

# Parametrized test
@pytest.mark.parametrize("n", [1, 2, 8, 100])
@pytest.mark.parametrize("m", [1, 2, 8, 100])
@pytest.mark.parametrize("l", [1, 2, 8, 100])
@pytest.mark.parametrize("i", [1, 2, 4, 8, 16, 32, 64, 128, 256])
def test_gemm(compiled_gemm, n, m, l, i):
    compiled_fn, s = compiled_gemm
    
    n_size = n * s
    m_size = m * s
    l_size = l * s
    

    A = np.random.uniform(-i, i, (n_size, m_size))
    B = np.random.uniform(-i, i, (m_size, l_size))
    C = np.zeros((n_size, l_size))

    expected = A @ B

    compiled_fn(
        A=A, B=B, C=C, 
        N=n,
        M=m,
        L=l,
        S=s,
    )

    error = np.linalg.norm(C - expected) / (n_size * l_size)

    # Include parameters in assert message
    assert error < 1e-3, f"Failed with n={n}, m={m}, l={l}, scale={i}, error={error}, blocksize={s}"