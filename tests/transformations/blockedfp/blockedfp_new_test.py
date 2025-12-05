import dace
import numpy as np

from dace.transformation.blockedfp.blockedfp_new import BlockedFPNew

M = dace.symbol("M")
N = dace.symbol("N")
K = dace.symbol("K")

def visualize_program(prgm, arrays):
    
    sdfg: dace.SDFG = prgm.to_sdfg()
    sdfg.view()
    
    # BlockedFPNew(arrays).apply_pass(sdfg, {})
    # sdfg.expand_library_nodes()
                
    # sdfg.simplify()
    
    sdfg.validate()
    sdfg.view()
    compiled = sdfg.compile()

@dace.program
def vecadd(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N]):
    for i in dace.map[0:N]:
        C[i] = A[i] + B[i]

@dace.program
def matadd(A: dace.float64[M,N], B: dace.float64[M,N], C: dace.float64[M,N]):
    for i, j in dace.map[0:M, 0:N]:
        C[i, j] = A[i, j] + B[i, j]

# Represents single read multiple write
@dace.program
def shiftright0(A: dace.float64[N], B: dace.float64[N]):
    for i in dace.map[0:(N-1)]:
        B[i+1] = A[i] + 1

# Represents single read multiple write
@dace.program
def shiftright1(A: dace.float64[N], B: dace.float64[N]):
    for i in dace.map[1:N]:
        B[i] = A[i-1] + 1

# Represents single write multiple read
@dace.program
def shiftleft0(A: dace.float64[N], B: dace.float64[N]):
    for i in dace.map[0:(N-1)]:
        B[i] = A[i+1] + 1

# Represents single write multiple read
@dace.program
def shiftleft1(A: dace.float64[N], B: dace.float64[N]):
    for i in dace.map[1:N]:
        B[i-1] = A[i] + 1

#visualize_program(matadd, ["A", "B", "C"])
visualize_program(vecadd, ["A", "B", "C"])
#visualize_program(shiftright0, ["A", "B"])
#visualize_program(shiftright1, ["A", "B"])
#visualize_program(shiftleft0, ["A", "B"])
#visualize_program(shiftleft1, ["A", "B"])

# Map-Reduce version of matrix multiplication
@dace.program
def matmul(A: dace.float64[M, K], B: dace.float64[K, N], C: dace.float64[M, N]):
    tmp = np.ndarray([M, N, K], dtype=A.dtype)

    # Multiply every pair of values to a large 3D temporary array
    for i, j, k in dace.map[0:M, 0:N, 0:K]:
        with dace.tasklet:
            in_A << A[i, k]
            in_B << B[k, j]
            out >> tmp[i, j, k]

            out = in_A * in_B

    # Sum last dimension of temporary array to obtain resulting matrix
    dace.reduce(lambda a, b: a + b, tmp, C, axis=2, identity=0)

@dace.program
def cast_in_bias(orig: dace.float64[16*N], bias: dace.float64[N]):
    for i in dace.map[0:N]:
        for j in dace.map[0:16]:
            bias[i] += orig[16 * i + j] / 16
            
@dace.program
def cast_in_scale(orig: dace.float64[16*N], scale: dace.float64[N], bias: dace.float64[N]):
    for i in dace.map[0:N]:
        diff = dace.define_local([16], dace.float64)
        for j in dace.map[0:16]:
            diff[j] = abs(orig[i * 16 + j] - bias[i])
        scale[i] = dace.reduce(lambda a, b: a if a > b else b, diff)
        
#sdfg = matmul.to_sdfg()
#sdfg.expand_library_nodes()
#sdfg.simplify()
#sdfg.view()
