import dace
from typing import List, Tuple, Union, Dict

def make_map_ranges(
    shape: Tuple[Union[int, dace.symbolic.symbol, dace.symbolic.SymExpr]],
    variable: str = "i"
) -> Dict[str, str]:
    """Create map ranges from a multidimensional shape."""
    return {
        f"{variable}{dim}": f"0:{val}"
        for dim, val in enumerate(shape)
    }

def make_block_map_ranges(
    shape: Tuple[Union[int, dace.symbolic.symbol, dace.symbolic.SymExpr]],
    variable_loop: str = "i",
    variable_block: str = "j",
    blocking_factor: int = 16
) -> Dict[str, str]:
    """Create map ranges to iterate over blocks from a multidimensional shape."""
    result = {
        f"{variable_loop}{dim}": f"0:({val} / {blocking_factor})"#f"{variable_loop}{dim}": f"0:int_floor({val}, {blocking_factor})"
        for dim, val in enumerate(shape)
    }
    for dim in range(len(shape)):
        result[f"{variable_block}{dim}"] = f"0:{blocking_factor}"
    return result

def make_block_indices(
    indices: Tuple[Union[int, dace.symbolic.symbol, dace.symbolic.SymExpr]],
    blocking_factor: int = 16
) -> Tuple[dace.symbolic.SymExpr]:
    """Convert indices to block indices."""
    return tuple(
        dace.symbolic.SymExpr(f"({val} / {blocking_factor})")#dace.symbolic.SymExpr(f"int_floor({val}, {blocking_factor})")
        for val in indices
    )

def cast_in_out(
    sdfg: dace.SDFG,
    array_names: List[str],
    blocking_factor: int = 16
) -> dace.SDFGState:
    """Insert state(s) responsible for casting inputs."""
    if len(sdfg.states()) != 1:
        raise ValueError("The sdfg has more or less than 1 state!")

    main_state = sdfg.start_state

    compute_fp_state = sdfg.add_state_before(main_state, "compute_fp")
    cast_in_fp(sdfg, compute_fp_state, array_names, blocking_factor)
    
    compute_scale_state = sdfg.add_state_before(compute_fp_state, "compute_scale")
    cast_in_scale(sdfg, compute_scale_state, array_names, blocking_factor)
    
    compute_bias_state = sdfg.add_state_before(compute_scale_state, "compute_bias")
    cast_in_bias(sdfg, compute_bias_state, array_names, blocking_factor)
    
    compute_init_state = sdfg.add_state_before(compute_bias_state, "compute_init")
    cast_in_init(sdfg, compute_init_state, array_names)
    
    compute_out_state = sdfg.add_state_after(main_state, "cast_out")
    cast_out(sdfg, compute_out_state, array_names, blocking_factor)
    
    return main_state

def cast_in_fp(
    sdfg: dace.SDFG,
    state: dace.SDFGState,
    array_names: List[str],
    blocking_factor: int = 16
) -> None:
    """Add mapped tasklets that compute full-precision values."""
    for name in array_names:
        # Access nodes
        access_scale = state.add_access(f"{name}_scale")
        access_bias = state.add_access(f"{name}_bias")
        access_fp = state.add_access(f"{name}_fp")
        access_input = state.add_access(name)

        # Map iteration space
        map_ranges = make_map_ranges(sdfg.arrays[name].shape)
        map_indices = tuple(map_ranges.keys())

        # Blocked indexing
        block_subset = dace.subsets.Indices(
            make_block_indices(map_indices, blocking_factor)
        )

        # Input / output descriptors
        inputs: Dict[str, Tuple[dace.Memlet, str, dace.Memlet, dace.nodes.AccessNode]] = {
            "scale": (
                dace.Memlet(data=f"{name}_scale", subset=block_subset),
                f"{name}_scale",
                sdfg.make_array_memlet(f"{name}_scale"),
                access_scale
            ),
            "bias": (
                dace.Memlet(data=f"{name}_bias", subset=block_subset),
                f"{name}_bias",
                sdfg.make_array_memlet(f"{name}_bias"),
                access_bias
            ),
            "base": (
                dace.Memlet(data=name, subset=dace.subsets.Indices(map_indices)),
                name,
                sdfg.make_array_memlet(name),
                access_input
            )
        }

        outputs: Dict[str, Tuple[dace.Memlet, str, dace.Memlet, dace.nodes.AccessNode]] = {
            "fp": (
                dace.Memlet(data=f"{name}_fp", subset=dace.subsets.Indices(map_indices)),
                f"{name}_fp",
                sdfg.make_array_memlet(f"{name}_fp"),
                access_fp
            )
        }

        # Mapped tasklet
        tasklet, map_entry, map_exit = state.add_mapped_tasklet(
            name=f"cast_in_fp_{name}",
            map_ranges=map_ranges,
            inputs={k: mem for k, (mem, _, _, _) in inputs.items()},
            code="fp = (base - bias) / scale",
            outputs={k: mem for k, (mem, _, _, _) in outputs.items()}
        )

        # Wire inputs
        for k, (_, connector, mem, access) in inputs.items():
            for edge in state.in_edges(tasklet):
                if edge.dst_conn == k:
                    map_entry.add_in_connector(f"IN_{connector}")
                    map_entry.add_out_connector(f"OUT_{connector}")
                    edge.src_conn = f"OUT_{connector}"
                    state.add_edge(access, None, map_entry, f"IN_{connector}", mem)

        # Wire outputs
        for k, (_, connector, mem, access) in outputs.items():
            for edge in state.out_edges(tasklet):
                if edge.src_conn == k:
                    map_exit.add_in_connector(f"IN_{connector}")
                    map_exit.add_out_connector(f"OUT_{connector}")
                    edge.dst_conn = f"IN_{connector}"
                    state.add_edge(map_exit, f"OUT_{connector}", access, None, mem)

def cast_in_scale(
    sdfg: dace.SDFG,
    state: dace.SDFGState,
    array_names: List[str],
    blocking_factor: int = 16
) -> None:
    for name in array_names:
        # Access nodes
        access_scale = state.add_access(f"{name}_scale")
        access_bias = state.add_access(f"{name}_bias")
        access_input = state.add_access(name)
        
        # Map iteration space
        map_ranges = make_map_ranges(sdfg.arrays[name].shape)
        map_indices = tuple(map_ranges.keys())
        
        # Blocked indexing
        block_subset = dace.subsets.Indices(
            make_block_indices(map_indices, blocking_factor)
        )

        # Input / output descriptors
        inputs: Dict[str, Tuple[dace.Memlet, str, dace.Memlet, dace.nodes.AccessNode]] = {
            "bias": (
                dace.Memlet(data=f"{name}_bias", subset=block_subset),
                f"{name}_bias",
                sdfg.make_array_memlet(f"{name}_bias"),
                access_bias
            ),
            "base": (
                dace.Memlet(data=name, subset=dace.subsets.Indices(map_indices)),
                name,
                sdfg.make_array_memlet(name),
                access_input
            )
        }

        outputs: Dict[str, Tuple[dace.Memlet, str, dace.Memlet, dace.nodes.AccessNode]] = {
            "scale": (
                dace.Memlet(data=f"{name}_scale", subset=block_subset, wcr="(lambda a, b: (a if (a > b) else b))"),
                f"{name}_scale",
                sdfg.make_array_memlet(f"{name}_scale"),
                access_scale
            ),
        }
        outputs["scale"][2].wcr = "(lambda a, b: (a if (a > b) else b))"
        
        # Mapped tasklet
        tasklet, map_entry, map_exit = state.add_mapped_tasklet(
            name=f"cast_in_scale_{name}",
            map_ranges=map_ranges,
            inputs={k: mem for k, (mem, _, _, _) in inputs.items()},
            code="scale = abs(base - bias)",
            outputs={k: mem for k, (mem, _, _, _) in outputs.items()}
        )
        
        # Wire inputs
        for k, (_, connector, mem, access) in inputs.items():
            for edge in state.in_edges(tasklet):
                if edge.dst_conn == k:
                    map_entry.add_in_connector(f"IN_{connector}")
                    map_entry.add_out_connector(f"OUT_{connector}")
                    edge.src_conn = f"OUT_{connector}"
                    state.add_edge(access, None, map_entry, f"IN_{connector}", mem)

        # Wire outputs
        for k, (_, connector, mem, access) in outputs.items():
            for edge in state.out_edges(tasklet):
                if edge.src_conn == k:
                    map_exit.add_in_connector(f"IN_{connector}")
                    map_exit.add_out_connector(f"OUT_{connector}")
                    edge.dst_conn = f"IN_{connector}"
                    state.add_edge(map_exit, f"OUT_{connector}", access, None, mem)

def cast_in_bias(
    sdfg: dace.SDFG,
    state: dace.SDFGState,
    array_names: List[str],
    blocking_factor: int = 16
) -> None:
    for name in array_names:
        # Access nodes
        access_bias = state.add_access(f"{name}_bias")
        access_input = state.add_access(name)
        
        # Map iteration space
        map_ranges = make_block_map_ranges(sdfg.arrays[name].shape)
        map_indices = tuple(
            dace.symbolic.SymExpr(f"{blocking_factor} * i{dim} + j{dim}")
            for dim in range(len(sdfg.arrays[name].shape))
        )
        
        # Blocked indexing
        block_subset = dace.subsets.Indices(
            tuple(
                dace.symbolic.SymExpr(f"i{dim}")
                for dim, _ in enumerate(sdfg.arrays[name].shape)
            )
        )
        
        # Input / output descriptors
        inputs: Dict[str, Tuple[dace.Memlet, str, dace.Memlet, dace.nodes.AccessNode]] = {
            "base": (
                dace.Memlet(data=name, subset=dace.subsets.Indices(map_indices)),
                name,
                sdfg.make_array_memlet(name),
                access_input
            )
        }

        outputs: Dict[str, Tuple[dace.Memlet, str, dace.Memlet, dace.nodes.AccessNode]] = {
            "bias": (
                dace.Memlet(data=f"{name}_bias", subset=block_subset, wcr="(lambda a, b: a + b)"),
                f"{name}_bias",
                sdfg.make_array_memlet(f"{name}_bias"),
                access_bias
            ),
        }
        outputs["bias"][2].wcr = "(lambda a, b: a + b)"
        
        # Calculate the number of elements in a block:
        block_size = blocking_factor ** len(sdfg.arrays[name].shape)
        
        # Mapped tasklet
        tasklet, map_entry, map_exit = state.add_mapped_tasklet(
            name=f"cast_in_bias_{name}",
            map_ranges=map_ranges,
            inputs={k: mem for k, (mem, _, _, _) in inputs.items()},
            code=f"bias = base / float({block_size})",
            outputs={k: mem for k, (mem, _, _, _) in outputs.items()}
        )
        
        # Wire inputs
        for k, (_, connector, mem, access) in inputs.items():
            for edge in state.in_edges(tasklet):
                if edge.dst_conn == k:
                    map_entry.add_in_connector(f"IN_{connector}")
                    map_entry.add_out_connector(f"OUT_{connector}")
                    edge.src_conn = f"OUT_{connector}"
                    state.add_edge(access, None, map_entry, f"IN_{connector}", mem)

        # Wire outputs
        for k, (_, connector, mem, access) in outputs.items():
            for edge in state.out_edges(tasklet):
                if edge.src_conn == k:
                    map_exit.add_in_connector(f"IN_{connector}")
                    map_exit.add_out_connector(f"OUT_{connector}")
                    edge.dst_conn = f"IN_{connector}"
                    state.add_edge(map_exit, f"OUT_{connector}", access, None, mem)

def cast_in_init(
    sdfg: dace.SDFG,
    state: dace.SDFGState,
    array_names: List[str]
) -> None:
    for name in array_names:
        # Access nodes
        access_scale = state.add_access(f"{name}_scale")
        access_bias = state.add_access(f"{name}_bias")
        
        # Map iteration space
        map_ranges = make_map_ranges(sdfg.arrays[f"{name}_bias"].shape)
        map_indices = tuple(map_ranges.keys())
        
        # Blocked indexing
        block_subset = dace.subsets.Indices(
            map_indices
        )

        # Input / output descriptors
        outputs: Dict[str, Tuple[dace.Memlet, str, dace.Memlet, dace.nodes.AccessNode]] = {
            "bias": (
                dace.Memlet(data=f"{name}_bias", subset=block_subset),
                f"{name}_bias",
                sdfg.make_array_memlet(f"{name}_bias"),
                access_bias
            ),
            "scale": (
                dace.Memlet(data=f"{name}_scale", subset=block_subset),
                f"{name}_scale",
                sdfg.make_array_memlet(f"{name}_scale"),
                access_scale
            ),
        }
        
        # Mapped tasklet
        tasklet, _, map_exit = state.add_mapped_tasklet(
            name=f"cast_init_{name}",
            map_ranges=map_ranges,
            inputs={},
            code="scale = 1.0; bias = 0.0;",
            outputs={k: mem for k, (mem, _, _, _) in outputs.items()}
        )

        # Wire outputs
        for k, (_, connector, mem, access) in outputs.items():
            for edge in state.out_edges(tasklet):
                if edge.src_conn == k:
                    map_exit.add_in_connector(f"IN_{connector}")
                    map_exit.add_out_connector(f"OUT_{connector}")
                    edge.dst_conn = f"IN_{connector}"
                    state.add_edge(map_exit, f"OUT_{connector}", access, None, mem)

def cast_out(
    sdfg: dace.SDFG,
    state: dace.SDFGState,
    array_names: List[str],
    blocking_factor: int = 16
) -> None:
    for name in array_names:
        # Access nodes
        access_scale = state.add_access(f"{name}_scale")
        access_bias = state.add_access(f"{name}_bias")
        access_fp = state.add_access(f"{name}_fp")
        access_input = state.add_access(name)

        # Map iteration space
        map_ranges = make_map_ranges(sdfg.arrays[name].shape)
        map_indices = tuple(map_ranges.keys())

        # Blocked indexing
        block_subset = dace.subsets.Indices(
            make_block_indices(map_indices, blocking_factor)
        )

        # Input / output descriptors
        inputs: Dict[str, Tuple[dace.Memlet, str, dace.Memlet, dace.nodes.AccessNode]] = {
            "scale": (
                dace.Memlet(data=f"{name}_scale", subset=block_subset),
                f"{name}_scale",
                sdfg.make_array_memlet(f"{name}_scale"),
                access_scale
            ),
            "bias": (
                dace.Memlet(data=f"{name}_bias", subset=block_subset),
                f"{name}_bias",
                sdfg.make_array_memlet(f"{name}_bias"),
                access_bias
            ),
            "fp": (
                dace.Memlet(data=f"{name}_fp", subset=dace.subsets.Indices(map_indices)),
                f"{name}_fp",
                sdfg.make_array_memlet(f"{name}_fp"),
                access_fp
            )
        }

        outputs: Dict[str, Tuple[dace.Memlet, str, dace.Memlet, dace.nodes.AccessNode]] = {
             "base": (
                dace.Memlet(data=name, subset=dace.subsets.Indices(map_indices)),
                name,
                sdfg.make_array_memlet(name),
                access_input
            )
        }

        # Mapped tasklet
        tasklet, map_entry, map_exit = state.add_mapped_tasklet(
            name=f"cast_out_{name}",
            map_ranges=map_ranges,
            inputs={k: mem for k, (mem, _, _, _) in inputs.items()},
            code="base = fp * scale + bias",
            outputs={k: mem for k, (mem, _, _, _) in outputs.items()}
        )

        # Wire inputs
        for k, (_, connector, mem, access) in inputs.items():
            for edge in state.in_edges(tasklet):
                if edge.dst_conn == k:
                    map_entry.add_in_connector(f"IN_{connector}")
                    map_entry.add_out_connector(f"OUT_{connector}")
                    edge.src_conn = f"OUT_{connector}"
                    state.add_edge(access, None, map_entry, f"IN_{connector}", mem)

        # Wire outputs
        for k, (_, connector, mem, access) in outputs.items():
            for edge in state.out_edges(tasklet):
                if edge.src_conn == k:
                    map_exit.add_in_connector(f"IN_{connector}")
                    map_exit.add_out_connector(f"OUT_{connector}")
                    edge.dst_conn = f"IN_{connector}"
                    state.add_edge(map_exit, f"OUT_{connector}", access, None, mem)
