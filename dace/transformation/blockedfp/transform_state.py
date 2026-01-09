import dace
import dace.sdfg.tasklet_utils as tutil
import dace.transformation as xf
from dace.sdfg.utils import node_path_graph
from dace.transformation.dataflow import MapTiling

import dace.transformation.blockedfp.libraries as bfplib

from typing import List, Dict, Tuple

def blockedfp_transform_state(
    sdfg: dace.SDFG, 
    state: dace.SDFGState, 
    array_names: List[str],
    blocking_factor: int = 16) -> None:
    
    # Map Tiling
    tile_maps(sdfg, state, array_names, blocking_factor)
    
    sdfg.apply_transformations(ExtendMapTransform, options={"names": array_names, "blocking_factor": blocking_factor}, validate=False)
    
    # Add new arrays for all transient arrays
    #blockedfp_extend_transient(sdfg, state, blocking_factor)
    
    """
    # Add new nodes
    access_nodes: Dict[dace.nodes.AccessNode, Tuple[dace.nodes.AccessNode, dace.nodes.AccessNode, dace.nodes.AccessNode]] = {}
    library_nodes: Dict[dace.nodes.Tasklet, dace.nodes.LibraryNode] = {}
    for node in state.nodes():
        if isinstance(node, dace.nodes.AccessNode):
            blockedfp_extend_transient(sdfg, state, node, 16)
            access_nodes[node] = add_access(state, node)
        elif isinstance(node, dace.nodes.Tasklet):
            library_nodes[node] = add_lib(state, node)
    """


class ExtendMapTransform(xf.SingleStateTransformation):
    
    outer_map_entry = xf.PatternNode(dace.nodes.MapEntry)
    inner_map_entry = xf.PatternNode(dace.nodes.MapEntry)
    
    def __init__(self, names: List[str], blocking_factor: int = 16):
        self.__names = names
        self.__blocking_factor = blocking_factor
    
    @classmethod
    def expressions(cls):
        return [node_path_graph(cls.outer_map_entry, cls.inner_map_entry)]
    
    def can_be_applied(self, graph: dace.SDFGState, expr_index: int, sdfg: dace.SDFG, permissive = False) -> bool:
        # Check if map reads from one of our arrays
        for edge in graph.in_edges(self.outer_map_entry):
            if edge.data.data in self.__names:
                return True
        # Check if map writes to one of our arrays
        for edge in graph.out_edges(graph.exit_node(self.outer_map_entry)):
            if edge.data.data in self.__names:
                return True
        
        return False
    
    def apply(self, state: dace.SDFGState, sdfg: dace.SDFG) -> None:
        indices = self.inner_map_entry.map.params
        inner_range = self.inner_map_entry.map.range
        inner_map = dict(zip(indices, inner_range))
        inner_map_block = dict(zip(indices, inner_range))
        print(inner_range)
        
        # Extend all access nodes and tasklets
        access_nodes: Dict[dace.nodes.AccessNode, Tuple[dace.nodes.AccessNode, dace.nodes.AccessNode, dace.nodes.AccessNode]] = {}
        tasklet_nodes: Dict[dace.nodes.Tasklet, dace.nodes.LibraryNode] = {}
        for node in state.all_nodes_between(self.outer_map_entry, state.exit_node(self.outer_map_entry)):
            if isinstance(node, dace.nodes.AccessNode):
                access_nodes[node] = extend_access(sdfg, state, node, self.__blocking_factor, len(indices))
            elif isinstance(node, dace.nodes.Tasklet):
                tasklet_nodes[node] = add_lib(state, node)
        
        # Connect the libnodes
        for tasklet in state.all_nodes_between(self.inner_map_entry, state.exit_node(self.inner_map_entry)):
            if not isinstance(tasklet, dace.nodes.Tasklet):
                continue
            
            libnode = tasklet_nodes[tasklet]
            
            prefix = "a"
            for edge in state.in_edges(tasklet):
                ## Access Nodes
                if isinstance(edge.src, dace.nodes.AccessNode):
                    fp, scale, bias = access_nodes[edge.src]
                    state.add_edge(fp, None, libnode, f"{prefix}_fp", sdfg.make_array_memlet(fp.data))
                    state.add_edge(scale, None, libnode, f"{prefix}_scale", sdfg.make_array_memlet(scale.data))
                    state.add_edge(bias, None, libnode, f"{prefix}_bias", sdfg.make_array_memlet(bias.data))
                # Map Entry
                elif isinstance(edge.src, dace.nodes.MapEntry):
                    if not edge.src == self.inner_map_entry:
                        raise ValueError("Weird connection happened")
                    
                    subset = dace.subsets.Range([
                        inner_map[str(r[0])] for r in edge.data.subset.ranges
                    ])
                    
                    state.add_edge(self.outer_map_entry, f"OUT_{edge.data.data}_fp", libnode, f"{prefix}_fp", dace.Memlet(
                        data=f"{edge.data.data}_fp",
                        subset=subset
                        ))
                    state.add_edge(self.outer_map_entry, f"OUT_{edge.data.data}_scale", libnode, f"{prefix}_scale", dace.Memlet())
                    state.add_edge(self.outer_map_entry, f"OUT_{edge.data.data}_bias", libnode, f"{prefix}_bias", dace.Memlet())
                prefix = chr(ord(prefix) + 1)
            
            for edge in state.out_edges(tasklet):
                if isinstance(edge.dst, dace.nodes.AccessNode):
                    fp, scale, bias = access_nodes[edge.dst]
                    state.add_edge(libnode, f"out_fp", fp, None, sdfg.make_array_memlet(fp.data))
                    state.add_edge(libnode, f"out_scale", scale, None, sdfg.make_array_memlet(scale.data))
                    state.add_edge(libnode, f"out_bias", bias, None, sdfg.make_array_memlet(bias.data))


def tile_maps(
    sdfg: dace.SDFG,
    state: dace.SDFGState,
    array_names: List[str],
    blocking_factor: int = 16
) -> None:
    for map_entry in [n for n in state.nodes() if isinstance(n, dace.nodes.MapEntry)]:
        
        # Get the corresponding exit node
        map_exit = state.exit_node(map_entry)
        
        # primitive check if this map reads from or writes to one of our desired arrays
        tile = any(edge.data.data in array_names for edge in state.in_edges(map_entry)) \
        or any(edge.data.data in array_names for edge in state.out_edges(map_exit))
        
        if not tile:
            continue
        
        MapTiling.apply_to(
            sdfg=sdfg,
            options={
                "tile_sizes": [blocking_factor],
                "divides_evenly": True,
                "tile_trivial": False,
                "skew": False
            },
            map_entry=map_entry
        )

def extend_access(
    sdfg: dace.SDFG,
    state: dace.SDFGState,
    access: dace.nodes.AccessNode,
    blocking_factor: int = 16,
    dimensions: int = 1
) -> Tuple[dace.nodes.AccessNode, dace.nodes.AccessNode, dace.nodes.AccessNode]:
    name = access.data
    array = sdfg.arrays[name]
    
    block_shape = tuple(blocking_factor for i in range(dimensions))
    
    sdfg.add_array(
        name=f"{name}_fp",
        shape=block_shape,
        dtype=dace.dtypes.float32,
        storage=array.storage,
        location=array.location,
        transient=True
    )
    sdfg.add_array(
            name=f"{name}_bias",
            shape=(1,),
            dtype=array.dtype,
            storage=array.storage,
            location=array.location,
            transient=True
        )
    sdfg.add_array(
            name=f"{name}_scale",
            shape=(1,),
            dtype=array.dtype,
            storage=array.storage,
            location=array.location,
            transient=True
        )
    
    access_fp = state.add_access(f"{name}_fp")
    access_scale = state.add_access(f"{name}_scale")
    access_bias = state.add_access(f"{name}_bias")
    
    return access_fp, access_scale, access_bias

def add_lib(
    state: dace.SDFGState,
    tasklet: dace.nodes.Tasklet
) -> dace.nodes.LibraryNode:
    # Get classification of this tasklet
    classification = tutil.classify_tasklet(state, tasklet)
        
    # Find corresponding library node and add it
    if classification["type"] is tutil.TaskletType.ARRAY_ARRAY or classification["type"] is tutil.TaskletType.ARRAY_SCALAR or classification["type"] is tutil.TaskletType.SCALAR_ARRAY or classification["type"] is tutil.TaskletType.SCALAR_SCALAR:
        match classification["op"]:
            case '+':
                libnode = bfplib.BFPAddNode(tasklet.name)
    elif classification["type"] is tutil.TaskletType.ARRAY_SCALAR_ASSIGNMENT or classification["type"] is tutil.TaskletType.ARRAY_ARRAY_ASSIGNMENT:
        libnode = bfplib.BFPAssignScalarNode(tasklet.name)
    else:
        raise ValueError(f"WHAT THE FUCK AM I SEEING, {classification['type']}")
    
    # Add this libnode
    state.add_node(libnode)
    return libnode