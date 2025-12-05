import dace
from typing import Tuple, List
import dace.transformation.pass_pipeline as ppl

# ------------------------------
# Helpers
# ------------------------------

def make_map_ranges(shape: Tuple[int, ...], start: int = 0) -> dict:
    """Return a dictionary mapping iteration variables to ranges."""
    return {f"i{d + start}": f"0:{size}" for d, size in enumerate(shape)}


def make_linear_index(strides: Tuple[int, ...]) -> str:
    """Build linear index string for 1D access: i0*s0 + i1*s1 + ..."""
    return "+".join([f"i{d}*{strides[d]}" for d in range(len(strides))])


def add_cast_tasklet(
    sdfg: dace.SDFG,
    state: dace.SDFGState,
    name: str,
    fp_to_original: bool
):
    """
    Adds a map + tasklet to perform casting.

    fp_to_original = False → original → fp
    fp_to_original = True  → fp → original
    """
    shape = sdfg.arrays[f"{name}_fp"].shape
    strides = sdfg.arrays[f"{name}_fp"].strides
    dims = len(shape)
    half_dims = dims // 2

    # Access nodes
    orig_acc = state.add_access(name)
    fp_acc = state.add_access(f"{name}_fp")
    scale_acc = state.add_access(f"{name}_scale")
    bias_acc = state.add_access(f"{name}_bias")

    # Map
    map_entry, map_exit = state.add_map(
        f"{'cast_out' if fp_to_original else 'cast_in'}_{name}_fp",
        ndrange=make_map_ranges(shape)
    )

    # Tasklet
    if fp_to_original:
        code = "original = fp * scale + bias;"
        inputs = ["fp", "bias", "scale"]
        outputs = ["original"]
    else:
        code = f"fp = static_cast<{dace.float32.ctype}>((original - bias) / scale);"
        inputs = ["original", "bias", "scale"]
        outputs = ["fp"]
        

    tasklet = state.add_tasklet(
        name=f"compute_{'orig' if fp_to_original else 'fp'}_{name}",
        inputs=inputs,
        outputs=outputs,
        code=code,
        language=dace.dtypes.Language.CPP
    )

    # Connect memlets
    fp_subset = dace.subsets.Indices([f"i{d}" for d in range(dims)])
    hb_subset = dace.subsets.Indices([f"i{d}" for d in range(half_dims)])
    orig_index = dace.subsets.Indices([make_linear_index(strides)])

    if fp_to_original:
        # Inputs
        state.add_edge(fp_acc, None, map_entry, "IN_fp", sdfg.make_array_memlet(f"{name}_fp"))
        state.add_edge(scale_acc, None, map_entry, "IN_scale", sdfg.make_array_memlet(f"{name}_scale"))
        state.add_edge(bias_acc, None, map_entry, "IN_bias", sdfg.make_array_memlet(f"{name}_bias"))
        map_entry.add_in_connector("IN_fp")
        map_entry.add_in_connector("IN_scale")
        map_entry.add_in_connector("IN_bias")

        # Map → Tasklet
        state.add_edge(map_entry, "OUT_fp", tasklet, "fp", dace.Memlet(data=f"{name}_fp", subset=fp_subset))
        state.add_edge(map_entry, "OUT_scale", tasklet, "scale", dace.Memlet(data=f"{name}_scale", subset=hb_subset))
        state.add_edge(map_entry, "OUT_bias", tasklet, "bias", dace.Memlet(data=f"{name}_bias", subset=hb_subset))
        map_entry.add_out_connector("OUT_fp")
        map_entry.add_out_connector("OUT_scale")
        map_entry.add_out_connector("OUT_bias")

        # Tasklet → Map Exit → Original Access
        state.add_edge(tasklet, "original", map_exit, "IN_original", dace.Memlet(data=name, subset=orig_index))
        state.add_edge(map_exit, "OUT_original", orig_acc, None, sdfg.make_array_memlet(name))
        map_exit.add_in_connector("IN_original")
        map_exit.add_out_connector("OUT_original")

    else:
        # Inputs
        state.add_edge(orig_acc, None, map_entry, "IN_original", sdfg.make_array_memlet(name))
        state.add_edge(scale_acc, None, map_entry, "IN_scale", sdfg.make_array_memlet(f"{name}_scale"))
        state.add_edge(bias_acc, None, map_entry, "IN_bias", sdfg.make_array_memlet(f"{name}_bias"))
        map_entry.add_in_connector("IN_original")
        map_entry.add_in_connector("IN_scale")
        map_entry.add_in_connector("IN_bias")

        # Map → Tasklet
        state.add_edge(map_entry, "OUT_original", tasklet, "original", dace.Memlet(data=name, subset=orig_index))
        state.add_edge(map_entry, "OUT_scale", tasklet, "scale", dace.Memlet(data=f"{name}_scale", subset=hb_subset))
        state.add_edge(map_entry, "OUT_bias", tasklet, "bias", dace.Memlet(data=f"{name}_bias", subset=hb_subset))
        map_entry.add_out_connector("OUT_original")
        map_entry.add_out_connector("OUT_scale")
        map_entry.add_out_connector("OUT_bias")

        # Tasklet → Map Exit → FP Access
        state.add_edge(tasklet, "fp", map_exit, "IN_fp", dace.Memlet(data=f"{name}_fp", subset=fp_subset))
        state.add_edge(map_exit, "OUT_fp", fp_acc, None, sdfg.make_array_memlet(f"{name}_fp"))
        map_exit.add_in_connector("IN_fp")
        map_exit.add_out_connector("OUT_fp")

def calculate_scale(sdfg: dace.SDFG,
    state: dace.SDFGState,
    name: str):
    
    shape = sdfg.arrays[f"{name}_fp"].shape
    strides = sdfg.arrays[f"{name}_fp"].strides
    dims = len(shape)
    half_dims = dims // 2
    
    # Access nodes
    orig_access = state.add_access(name)
    bias_access = state.add_access(f"{name}_bias")
    scale_access = state.add_access(f"{name}_scale")
    sdfg.add_array(f"{name}_scale_result", (1,), dtype=dace.float64, transient=True)
    diff_access = state.add_transient(f"{name}_diff", shape[half_dims:], dtype=dace.float64)
    scale_result_access = state.add_access(f"{name}_scale_result")
    
    # Map
    outer_map_entry, outer_map_exit = state.add_map(
        f"cast_in_{name}_scale",
        ndrange=make_map_ranges(shape[:half_dims])
    )
    
    # AbsSub Tasklet
    _, inner_map_entry, inner_map_exit = state.add_mapped_tasklet(
        name= f"calculate_diff_{name}",
        map_ranges= make_map_ranges(shape[half_dims:], half_dims),
        inputs={
            "original": dace.Memlet(data=name, subset=dace.subsets.Indices([make_linear_index(strides)])),
            "bias": dace.Memlet(data=f"{name}_bias", subset=dace.subsets.Indices([f"i{dim}" for dim in range(half_dims)]))
        },
        code="diff = abs(original-bias);",
        outputs={
            "diff": dace.Memlet(data=f"{name}_diff", subset=dace.subsets.Indices([f"i{dim + half_dims}" for dim in range(half_dims)]))
        }
    )
    
    # Add connectors to maps
    for edge in state.out_edges(inner_map_entry):
        connector = edge.dst_conn
        edge.src_conn = f"OUT_{connector}"
        inner_map_entry.add_in_connector(f"IN_{connector}")
        inner_map_entry.add_out_connector(f"OUT_{connector}")
    for edge in state.in_edges(inner_map_exit):
        connector = edge.src_conn
        edge.dst_conn = f"IN_{connector}"
        inner_map_exit.add_out_connector(f"OUT_{connector}")
        inner_map_exit.add_in_connector(f"IN_{connector}")
    
    # Connect inner map to transient
    state.add_edge(inner_map_exit, "OUT_diff", diff_access, None, sdfg.make_array_memlet(f"{name}_diff"))
    
    # Connect Outer map
    state.add_edge(outer_map_entry, "OUT_bias", inner_map_entry, "IN_bias", dace.Memlet(
        data=f"{name}_bias", subset=dace.subsets.Indices([f"i{dim}" for dim in range(half_dims)])
    ))
    state.add_edge(outer_map_entry, "OUT_original", inner_map_entry, "IN_original", dace.Memlet(
        data=name, subset=dace.subsets.Range([(f"i{dim}", f"i{dim}+{shape[half_dims + dim]-1}",1) for dim in range(half_dims)])
    ))
    state.add_edge(bias_access, None, outer_map_entry, "IN_bias", sdfg.make_array_memlet(f"{name}_bias"))
    state.add_edge(orig_access, None, outer_map_entry, "IN_original", sdfg.make_array_memlet(name))
    outer_map_entry.add_in_connector("IN_bias")
    outer_map_entry.add_in_connector("IN_original")
    outer_map_entry.add_out_connector("OUT_bias")
    outer_map_entry.add_out_connector("OUT_original")
    
    # Reduce Tasklet
    _, inner_map_entry2, inner_map_exit2 = state.add_mapped_tasklet(
        name= f"reduce_diff_{name}",
        map_ranges= make_map_ranges(shape[half_dims:], half_dims),
        inputs={
            "diff": dace.Memlet(data=f"{name}_diff", subset=dace.subsets.Indices([f"i{dim + half_dims}" for dim in range(half_dims)]))
        },
        code="result = diff;",
        outputs={
            "result": dace.Memlet(expr=f"{name}_scale_result[0]")
        }
    )
    
    # Add connectors to maps
    for edge in state.out_edges(inner_map_entry2):
        connector = edge.dst_conn
        edge.src_conn = f"OUT_{connector}"
        inner_map_entry2.add_in_connector(f"IN_{connector}")
        inner_map_entry2.add_out_connector(f"OUT_{connector}")
    for edge in state.in_edges(inner_map_exit2):
        connector = edge.src_conn
        edge.dst_conn = f"IN_{connector}"
        inner_map_exit2.add_out_connector(f"OUT_{connector}")
        inner_map_exit2.add_in_connector(f"IN_{connector}")
        edge.data.wcr = "(lambda a, b: (a if (a > b) else b))"
    
    # connect in transient
    state.add_edge(diff_access, None, inner_map_entry2, "IN_diff", sdfg.make_array_memlet(f"{name}_diff"))
    state.add_edge(inner_map_exit2, "OUT_result", scale_result_access, None, sdfg.make_array_memlet(f"{name}_scale_result")).data.wcr = "(lambda a, b: (a if (a > b) else b))"
    
    # Assign Tasklet
    tasklet = state.add_tasklet(
        f"assign_{name}_scale",
        inputs=["scale"],
        outputs=["result"],
        code="result = scale;"
    )
    state.add_edge(scale_result_access, None, tasklet, "scale", sdfg.make_array_memlet(f"{name}_scale_result"))
    state.add_edge(tasklet, "result", outer_map_exit, "IN_scale", 
                   dace.Memlet(data=f"{name}_scale", subset=dace.subsets.Indices([f"i{dim}" for dim in range(half_dims)]))
                   )
    state.add_edge(outer_map_exit, "OUT_scale", scale_access, None, sdfg.make_array_memlet(f"{name}_scale"))
    outer_map_exit.add_in_connector("IN_scale")
    outer_map_exit.add_out_connector("OUT_scale")

def calculate_bias(sdfg: dace.SDFG,
    state: dace.SDFGState,
    name: str):
    
    shape = sdfg.arrays[name].shape
    shape_fp = sdfg.arrays[f"{name}_fp"].shape
    strides = sdfg.arrays[f"{name}_fp"].strides
    dims = len(shape_fp)
    half_dims = dims // 2
    
    # Access nodes
    orig_access = state.add_access(name)
    bias_access = state.add_access(f"{name}_bias")
    
    # Connect map
    map_entry, map_exit = state.add_map(f"cast_in_{name}_bias_map", ndrange={
        f"i{dim}": f"0:int_ceil({size},{shape_fp[dim + half_dims]})" for dim, size in enumerate(shape)
    })
    state.add_edge(orig_access, None, map_entry, "IN_original", sdfg.make_array_memlet(name))
    state.add_edge(map_exit, "OUT_bias", bias_access, None, sdfg.make_array_memlet(f"{name}_bias"))
    map_entry.add_in_connector("IN_original")
    map_entry.add_out_connector("OUT_original")
    map_exit.add_in_connector("IN_bias")
    map_exit.add_out_connector("OUT_bias")
    
    block_size = 1
    for dim in shape_fp[half_dims:]:
        block_size *= dim
    
    _, inner_map_entry, inner_map_exit = state.add_mapped_tasklet(
        name="div",
        map_ranges=make_map_ranges(shape_fp[half_dims:], half_dims),
        inputs={
            "original": dace.Memlet(data=name, subset=dace.subsets.Indices([make_linear_index(strides)]))
        },
        code=f"bias = original / ({block_size})",
        outputs={
            "bias": dace.Memlet(data=f"{name}_bias", subset=dace.subsets.Indices([f"i{dim}" for dim in range(half_dims)]))
        }
    )
    
    for edge in state.in_edges(inner_map_exit):
        edge.dst_conn = "IN_bias"
        edge.data.wcr = "(lambda x, y: (x+y))"
        inner_map_exit.add_in_connector("IN_bias")
        inner_map_exit.add_out_connector("OUT_bias")
    for edge in state.out_edges(inner_map_entry):
        edge.src_conn = "OUT_original"
        inner_map_entry.add_in_connector("IN_original")
        inner_map_entry.add_out_connector("OUT_original")
    
    state.add_edge(inner_map_exit, "OUT_bias", map_exit, "IN_bias", dace.Memlet(data=f"{name}_bias", subset=dace.subsets.Indices([f"i{dim}" for dim in range(half_dims)]))).data.wcr = "(lambda x, y: (x+y))"
    state.add_edge(map_entry, "OUT_original", inner_map_entry, "IN_original", dace.Memlet(
        data=name,
        subset=dace.subsets.Range([(f"i{dim}", f"i{dim}+{shape_fp[dim + half_dims]-1}",1) for dim in range(half_dims)])
    ))
    
def init_bias(sdfg: dace.SDFG,
    state: dace.SDFGState,
    name: str):
    
    shape_bias = sdfg.arrays[f"{name}_bias"].shape
    dims = len(shape_bias)
    
    # Access nodes
    bias_access = state.add_access(f"{name}_bias")
    
    _, map_entry, map_exit = state.add_mapped_tasklet(
        name=f"init_{name}_bias_map",
        map_ranges={
            f"i{dim}" : f"0:{size}" for dim, size in enumerate(shape_bias)
        },
        inputs = {},
        code="_out = 0.0;",
        outputs={
            "_out" : dace.Memlet(data=f"{name}_bias", subset=dace.subsets.Indices([f"i{dim}" for dim in range(dims)]))
        }
    )
    
    # Connect map
    for edge in state.in_edges(map_exit):
        edge.dst_conn = "IN_bias"
        map_exit.add_in_connector("IN_bias")
        map_exit.add_out_connector("OUT_bias")
    
    state.add_edge(map_exit, "OUT_bias", bias_access, None, sdfg.make_array_memlet(f"{name}_bias"))
  

# ------------------------------
# CastInOutPass
# ------------------------------

class CastInOutPass(ppl.Pass):

    def __init__(self, names: List[str]):
        self.__names = names

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Everything

    def should_reapply(self, _) -> bool:
        return False

    def apply_pass(self, sdfg: dace.SDFG, _) -> None:
        start_state = sdfg.start_state

        # Cast in/out states for fp
        cast_in_fp_state = sdfg.add_state_before(start_state, "cast_in_fp")
        cast_out_fp_state = sdfg.add_state_after(start_state, "cast_out_fp")
        
        # Cast in state for scale
        cast_in_scale_state = sdfg.add_state_before(cast_in_fp_state, "cast_in_scale")
        # Cast in state for bias
        cast_in_bias_state = sdfg.add_state_before(cast_in_scale_state, "cast_in_bias")
        # Init state for bias
        init_bias_state = sdfg.add_state_before(cast_in_bias_state, "init_bias")

        # Cast in (original → fp)
        for name in self.__names:
            add_cast_tasklet(sdfg, cast_in_fp_state, name, fp_to_original=False)

        # Cast out (fp → original)
        for name in self.__names:
            add_cast_tasklet(sdfg, cast_out_fp_state, name, fp_to_original=True)
        
        # Cast in (bias → scale)
        for name in self.__names:
            calculate_scale(sdfg, cast_in_scale_state, name)
        
        # Cast in (orig → bias)
        for name in self.__names:
            calculate_bias(sdfg, cast_in_bias_state, name) 
            
        # Init bias (bias → 0)
        for name in self.__names:
            init_bias(sdfg, init_bias_state, name)
        
        # Quickly init scaleresult
        state2 = sdfg.add_state_before(init_bias_state, "init_scale_result")
        for name in self.__names:
            access = state2.add_access(f"{name}_scale_result")
            tasklet = state2.add_tasklet(
                f"{name}_scale_result_init",
                inputs={},
                outputs={"_out"},
                code="_out = 1.0;"
            )
            state2.add_edge(tasklet, "_out", access, None, sdfg.make_array_memlet(f"{name}_scale_result"))