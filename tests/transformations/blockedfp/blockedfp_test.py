import dace
import numpy as np

from dace.transformation.blockedfp.blockedfp import BlockedFP

N = dace.symbol('N')

def vec_add(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N]):
        for i in dace.map[0:N]:
            C[i] = A[i] + B[i]

def test_vec_add():
    # Get sdfg and transform
    sdfg = dace.program(vec_add).to_sdfg()
    BlockedFP(["A", "B", "C"]).apply_pass(sdfg, {})
    
    # Validate and compile
    sdfg.validate()
    compiled = sdfg.compile()
    sdfg.view()
    
    # Create arrays to test
    A = np.arange(64, dtype=np.float64)
    B = np.arange(63, -1, -1, dtype=np.float64)
    C = np.zeros_like(A)
    
    A_copy = np.copy(A)
    B_copy = np.copy(B)
    C_copy = np.copy(C)
    
    # Compute Expected result and actual result
    # vec_add(A, B, C)
    # print(C)
    compiled(A_copy, B_copy, C_copy, N=64)
    
    print(C_copy)
    


test_vec_add()