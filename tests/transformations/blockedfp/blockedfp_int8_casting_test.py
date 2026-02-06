import dace
import pytest

from dace.transformation.blockedfp.libraries import BFPCastinNodeInt8
from dace.transformation.blockedfp.libraries import BFPCastoutNodeInt8

import numpy as np

#@pytest.fixture(scope="module")
def built_sdfg() -> dace.CompiledSDFG:
    N = dace.symbol("N")
    S = dace.symbol("S")
    
    sdfg = dace.SDFG("CASTING_TEST")
    sdfg.add_symbol("N", stype=int)
    sdfg.add_symbol("S", stype=int)
    
    sdfg.add_array(
        name="array",
        shape=(N * S,),
        dtype=dace.float64
    )
    
    state = sdfg.add_state("casting", is_start_state=True)
    
    bias = state.add_transient(
        name="bias",
        shape=(N,),
        dtype=dace.float64
    )
    scale = state.add_transient(
        name="scale",
        shape=(N,),
        dtype=dace.int16
    )
    ints = state.add_transient(
        name="ints",
        shape=(N * S,),
        dtype=dace.int16
    )
    
    access_in = state.add_access("array")
    access_out= state.add_access("array")
    
    castin = BFPCastinNodeInt8("castin", symbol_mapping={"N":"N", "S":"S"})
    castout = BFPCastoutNodeInt8("castout", symbol_mapping={"N":"N", "S":"S"})
    state.add_nodes_from([castin, castout])
    
    state.add_edge(access_in, None, castin, "array", sdfg.make_array_memlet("array"))
    state.add_edge(castin, "bias", bias, None, sdfg.make_array_memlet("bias"))
    state.add_edge(castin, "scale", scale, None, sdfg.make_array_memlet("scale"))
    state.add_edge(castin, "ints", ints, None, sdfg.make_array_memlet("ints"))
    state.add_edge(bias, None, castout, "bias", sdfg.make_array_memlet("bias"))
    state.add_edge(scale, None, castout, "scale", sdfg.make_array_memlet("scale"))
    state.add_edge(ints, None, castout, "ints", sdfg.make_array_memlet("ints"))
    state.add_edge(castout, "array", access_out, None, sdfg.make_array_memlet("array"))
    
    sdfg.expand_library_nodes()
    sdfg.validate()
    
    return sdfg.compile()

@pytest.mark.parametrize("k", [1, 2, 4, 8, 16, 32, 64, 128, 256])         # matrix size multipliers
@pytest.mark.parametrize("n", [1, 2, 4, 8, 16, 32])
@pytest.mark.parametrize("s", [2, 4, 8, 16, 32])
def test_gemm_kernel(built_sdfg, k, n, s):
    A = np.random.uniform(-k, k, (n*s,))
    expected = A.copy()
    
    built_sdfg(
        array=A, N=n, S=s
    )
    
    assert np.linalg.norm(A - expected) < 10

compiled = built_sdfg()
compiled.sdfg.view()
A = np.random.uniform(0, 1, (1*4,))
expected = A.copy()
compiled(array=A, N=1, S=4)
print(A, expected)