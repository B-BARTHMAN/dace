import dace
from typing import List
from dace.transformation.blockedfp.cast_in_out import cast_in_out
from dace.transformation.blockedfp.auxiliary_arrays import add_auxiliary_arrays
from dace.transformation.blockedfp.transform_state import blockedfp_transform_state
from dace.transformation.passes.split_tasklets import SplitTasklets
import numpy as np

from samples.optimization.matmul import matmul


def blocked_fp(sdfg: dace.SDFG, array_names: List[str], blocking_factor: int = 16) -> None:
    """Apply blocked floating-point transformation to selected arrays."""
    sdfg.expand_library_nodes()
    
    SplitTasklets().apply_pass(sdfg, {})
    
    add_auxiliary_arrays(sdfg, array_names, blocking_factor)
    
    main_state = cast_in_out(sdfg, array_names, blocking_factor)
    
    blockedfp_transform_state(sdfg, main_state, array_names, blocking_factor)

N = dace.symbol("N")
@dace.program
def vec_add(A: dace.float64[16*N], B: dace.float64[16*N], C: dace.float64[16*N]):
    for i in dace.map[0:16*N]:
        C[i] = A[i] + B[i]

sdfg = vec_add.to_sdfg()
sdfg.expand_library_nodes()
blocked_fp(sdfg, ["A", "B", "C"])
sdfg.view()