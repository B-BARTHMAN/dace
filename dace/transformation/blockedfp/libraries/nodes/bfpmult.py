import dace
from dace.transformation.transformation import ExpandTransformation

@dace.program
def bfpmult(a_scale: dace.float64[1], a_bias: dace.float64[1], a_fp: dace.float32[16], b_scale: dace.float64[1], b_bias: dace.float64[1], b_fp: dace.float32[16], out_scale: dace.float64[1], out_bias: dace.float64[1], out_fp: dace.float32[16]):
    # Dequantize
    a_val = a_scale[0] * a_fp[:] + a_bias[0]
    b_val = b_scale[0] * b_fp[:] + b_bias[0]

    # Add
    out_val = a_val[:] * b_val[:]

    # Compute output bias and scale (block floating-point normalization)
    bias_val = 0.
    for i in dace.map[0:16] @ dace.ScheduleType.Sequential:
        bias_val += out_val[i] / 16.0
    scale_val = 1.
    for i in dace.map[0:16] @ dace.ScheduleType.Sequential:
        diff = out_val[i] - bias_val
        diff = abs(diff)
        scale_val = max(scale_val, diff)

    # Write back to outputs (important!)
    out_bias[0] = bias_val
    out_scale[0] = scale_val

    # Quantize back
    out_fp[:] = (out_val[:] - bias_val) / scale_val

@dace.library.expansion
class ExpandBFPMultNode(ExpandTransformation):
    environments = []
    
    @staticmethod
    def expansion(node: "BFPMultNode", parent_state: dace.SDFGState, parent_sdfg: dace.SDFGState) -> dace.nodes.Tasklet:
        
        a_scale = parent_sdfg.arrays[next(e.data.data for e in parent_state.in_edges(node) if e.dst_conn == "a_scale")]
        a_bias  = parent_sdfg.arrays[next(e.data.data for e in parent_state.in_edges(node) if e.dst_conn == "a_bias")]
        a_fp    = parent_sdfg.arrays[next(e.data.data for e in parent_state.in_edges(node) if e.dst_conn == "a_fp")]

        b_scale = parent_sdfg.arrays[next(e.data.data for e in parent_state.in_edges(node) if e.dst_conn == "b_scale")]
        b_bias  = parent_sdfg.arrays[next(e.data.data for e in parent_state.in_edges(node) if e.dst_conn == "b_bias")]
        b_fp    = parent_sdfg.arrays[next(e.data.data for e in parent_state.in_edges(node) if e.dst_conn == "b_fp")]

        out_scale = parent_sdfg.arrays[next(e.data.data for e in parent_state.out_edges(node) if e.src_conn == "out_scale")]
        out_bias  = parent_sdfg.arrays[next(e.data.data for e in parent_state.out_edges(node) if e.src_conn == "out_bias")]
        out_fp    = parent_sdfg.arrays[next(e.data.data for e in parent_state.out_edges(node) if e.src_conn == "out_fp")]
        
        assert all([a_scale, a_bias, a_fp, b_scale, b_bias, b_fp, out_scale, out_bias, out_fp])
        
        sdfg = bfpmult.to_sdfg(
            a_scale, a_bias, a_fp, b_scale, b_bias, b_fp, out_scale, out_bias, out_fp
        )
        sdfg.simplify()
        return sdfg
        

@dace.library.node
class BFPMultNode(dace.sdfg.nodes.LibraryNode):
    
    implementations = {
        "pure": ExpandBFPMultNode,
    }
    default_implementation = 'pure'
    
    def __init__(self, name):
        super().__init__(name, inputs={"a_scale", "a_bias", "a_fp", "b_scale", "b_bias", "b_fp"}, outputs={"out_scale", "out_bias", "out_fp"})