import dace
import numpy as np
import pytest
from dace.transformation.blockedfp.blockedfp import blocked_fp

N = dace.symbol("N")

@dace.program
def vec_add(A: dace.float64[16*N], B: dace.float64[16*N], C: dace.float64[16*N]):
    for i in dace.map[0:16*N]:
        C[i] = A[i] + B[i]

@dace.program
def vec_mult(A: dace.float64[16*N], B: dace.float64[16*N], C: dace.float64[16*N]):
    for i in dace.map[0:16*N]:
        C[i] = A[i] * B[i]
        
@pytest.mark.parametrize("N", [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
@pytest.mark.parametrize("bound", [1e0, 1e1, 1e2, 1e3, 1e4, 1e5])
def test_vec_add(N, bound):
    size = 16 * N
    
    A = np.random.uniform(-bound, bound, size)
    B = np.random.uniform(-bound, bound, size)
    C = np.zeros_like(A)
    expected = A + B
    
    # Run the transform
    sdfg = vec_add.to_sdfg()
    blocked_fp(sdfg, ["A", "B", "C"])
    sdfg.validate()
    compiled = sdfg.compile()
    
    # Run the compiled version
    compiled(A=A, B=B, C=C, N=N)
    
    assert np.linalg.norm(C - expected) < 1e-3

@pytest.mark.parametrize("N", [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
@pytest.mark.parametrize("bound", [1e0, 1e1, 1e2, 1e3, 1e4, 1e5])
def test_vec_cwisemult(N, bound):
    size = 16 * N
    
    A = np.random.uniform(-bound, bound, size)
    B = np.random.uniform(-bound, bound, size)
    C = np.zeros_like(A)
    expected = A * B
    
    # Run the transform
    sdfg = vec_mult.to_sdfg()
    blocked_fp(sdfg, ["A", "B", "C"])
    sdfg.validate()
    compiled = sdfg.compile()
    
    # Run the compiled version
    compiled(A=A, B=B, C=C, N=N)
    
    assert np.linalg.norm(C - expected) < 1e-3