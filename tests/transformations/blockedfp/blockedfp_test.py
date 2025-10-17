import dace

from dace.transformation.blockedfp.blockedfp import BlockedFP

N = dace.symbol("N")

@dace.program
def vadd(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N]):
    for i in dace.map[0:N] @ dace.ScheduleType.Sequential:
        C[i] = 0.5 * (A[i] + B[i])

sdfg = vadd.to_sdfg()
BlockedFP("A", [16]).apply_pass(sdfg,{})
sdfg.validate()
sdfg.view()