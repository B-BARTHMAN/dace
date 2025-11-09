import dace

from dace.transformation.blockedfp.blockedfp import BlockedFP

def visualize_program(prgm, arrays):
    
    sdfg: dace.SDFG = prgm.to_sdfg()
    sdfg.view()
    
    BlockedFP(arrays, [16]).apply_pass(sdfg, {})
    sdfg.validate()
    
    sdfg.view()
    
N = dace.symbol("N")

@dace.program
def vadd(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N]):
    for i in dace.map[0:N] @ dace.ScheduleType.Sequential:
        C[i] = 0.5 * (A[i] + B[i])

@dace.program
def write_program(A: dace.float64[N], B: dace.float64[N]):
    for i in dace.map[0:N-1] @ dace.ScheduleType.Sequential:
        A[i] = B[i+1]

@dace.program
def vmult(A: dace.float64[N], B: dace.float64[N], C: dace.float64):
    for i in dace.map[0:N//2] @ dace.ScheduleType.Sequential:
        C[i] = 0.5*(A[i] * B[i] + 0.5 * C[i])

visualize_program(vmult, ["A", "B", "C"])