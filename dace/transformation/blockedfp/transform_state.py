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
    
    sdfg.apply_transformations(ExtendMapTransform, options={"names": array_names, "blocking_factor": blocking_factor}, validate=True)


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
                prefix = chr(ord(prefix) + 1)
            
            for edge in state.out_edges(tasklet):
                ## Access Nodes
                if isinstance(edge.dst, dace.nodes.AccessNode):
                    fp, scale, bias = access_nodes[edge.dst]
                    state.add_edge(libnode, f"out_fp", fp, None, sdfg.make_array_memlet(fp.data))
                    state.add_edge(libnode, f"out_scale", scale, None, sdfg.make_array_memlet(scale.data))
                    state.add_edge(libnode, f"out_bias", bias, None, sdfg.make_array_memlet(bias.data))
        
        # Connect Inner Maps Entry
        prefix = 'a'
        for edge_in in state.in_edges(self.inner_map_entry):
            for edge_out in state.out_edges(self.inner_map_entry):
                # Check if edges match
                if edge_in.data.data != edge_out.data.data:
                    continue
                # Check if this edge is a blockedfp
                if edge_in.data.data not in self.__names:
                    continue
                
                # Check if it writes to a tasklet
                if not isinstance(edge_out.dst, dace.nodes.Tasklet):
                    raise ValueError("Inner map is not writing to a tasklet")
                libnode = tasklet_nodes[edge_out.dst]
                
                # Get block access
                subset: dace.subsets.Range = edge_in.data.subset
                block_subset = dace.subsets.Range([
                    [
                        dace.symbolic.SymExpr(f"int_ceil({tup[0]}, {self.__blocking_factor})"),
                        dace.symbolic.SymExpr(f"int_ceil({tup[1]}, {self.__blocking_factor})"),
                        1
                    ] for tup in subset.ranges
                ])
                
                # Add connections
                state.add_edge(edge_in.src, f"{edge_in.src_conn}_fp", libnode, f"{prefix}_fp", dace.Memlet(data=f"{edge_in.data.data}_fp", subset=subset))
                state.add_edge(edge_in.src, f"{edge_in.src_conn}_bias", libnode, f"{prefix}_bias", dace.Memlet(data=f"{edge_in.data.data}_bias", subset=block_subset))
                state.add_edge(edge_in.src, f"{edge_in.src_conn}_scale", libnode, f"{prefix}_scale", dace.Memlet(data=f"{edge_in.data.data}_scale", subset=block_subset))
                
                prefix = chr(ord(prefix) + 1)
        
        # Connect Inner Maps Exit
        for edge_in in state.in_edges(state.exit_node(self.inner_map_entry)):
            for edge_out in state.out_edges(state.exit_node(self.inner_map_entry)):
                # Check if edges match
                if edge_in.data.data != edge_out.data.data:
                    continue
                # Check if this edge is a blockedfp
                if edge_in.data.data not in self.__names:
                    continue
                
                # Check if it writes from a tasklet
                if not isinstance(edge_in.src, dace.nodes.Tasklet):
                    raise ValueError("Inner map is not writing from a tasklet")
                libnode = tasklet_nodes[edge_in.src]
                
                # Get block access
                subset: dace.subsets.Range = edge_out.data.subset
                block_subset = dace.subsets.Range([
                    [
                        dace.symbolic.SymExpr(f"int_ceil({tup[0]}, {self.__blocking_factor})"),
                        dace.symbolic.SymExpr(f"int_ceil({tup[1]}, {self.__blocking_factor})"),
                        1
                    ] for tup in subset.ranges
                ])
                
                # Add connections
                state.add_edge(libnode, "out_fp", edge_out.dst, f"{edge_out.dst_conn}_fp", dace.Memlet(data=f"{edge_out.data.data}_fp", subset=subset))
                state.add_edge(libnode, "out_bias", edge_out.dst, f"{edge_out.dst_conn}_bias", dace.Memlet(data=f"{edge_out.data.data}_bias", subset=block_subset))
                state.add_edge(libnode, "out_scale", edge_out.dst, f"{edge_out.dst_conn}_scale", dace.Memlet(data=f"{edge_out.data.data}_scale", subset=block_subset))
        
        # Delete connectors from outer maps
        for conn in list(self.outer_map_entry.in_connectors):
            self.outer_map_entry.remove_in_connector(conn)
        for conn in list(self.outer_map_entry.out_connectors):
            self.outer_map_entry.remove_out_connector(conn)
        for conn in list(state.exit_node(self.outer_map_entry).in_connectors):
            state.exit_node(self.outer_map_entry).remove_in_connector(conn)
        for conn in list(state.exit_node(self.outer_map_entry).out_connectors):
            state.exit_node(self.outer_map_entry).remove_out_connector(conn)
        
        # Add connectors back to map entry
        for out_edge in state.out_edges(self.outer_map_entry):
            if out_edge.data.data.endswith("_fp") or out_edge.data.data.endswith("_bias") or out_edge.data.data.endswith("_scale"):
                conn = out_edge.src_conn
                self.outer_map_entry.add_out_connector(conn)
                self.outer_map_entry.add_in_connector(conn.replace("OUT", "IN"))
        
        # Add connectors back to map exit
        for in_edge in state.in_edges(state.exit_node(self.outer_map_entry)):
            if in_edge.data.data.endswith("_fp") or in_edge.data.data.endswith("_bias") or in_edge.data.data.endswith("_scale"):
                conn = in_edge.dst_conn
                state.exit_node(self.outer_map_entry).add_in_connector(conn)
                state.exit_node(self.outer_map_entry).add_out_connector(conn.replace("IN", "OUT"))
        
        # Rewire Input
        nodes_to_remove = list(state.all_nodes_between(self.inner_map_entry, state.exit_node(self.inner_map_entry)))
        for in_edge in state.in_edges(self.outer_map_entry):
            # Check if input is from access node
            if not isinstance(in_edge.src, dace.nodes.AccessNode):
                raise ValueError("Map doesn't read from access node")
            
            subset: dace.subsets.Range = in_edge.data.subset
            
            access_fp = state.add_access(f"{in_edge.data.data}_fp")
            access_bias = state.add_access(f"{in_edge.data.data}_bias")
            access_scale = state.add_access(f"{in_edge.data.data}_scale")
            state.add_edge(access_fp, None, self.outer_map_entry, f"IN_{in_edge.data.data}_fp", dace.Memlet(data=f"{in_edge.data.data}_fp", subset=subset))
            state.add_edge(access_bias, None, self.outer_map_entry, f"IN_{in_edge.data.data}_bias", sdfg.make_array_memlet(f"{in_edge.data.data}_bias"))
            state.add_edge(access_scale, None, self.outer_map_entry, f"IN_{in_edge.data.data}_scale", sdfg.make_array_memlet(f"{in_edge.data.data}_scale"))
            
            nodes_to_remove.append(in_edge.src)
        
        # Rewire Output
        for out_edge in state.out_edges(state.exit_node(self.outer_map_entry)):
            # Check if output is to a access node
            if not isinstance(out_edge.dst, dace.nodes.AccessNode):
                raise ValueError("Map doesn't write to access node")
            
            subset: dace.subsets.Range = out_edge.data.subset
            
            access_fp = state.add_access(f"{out_edge.data.data}_fp")
            access_bias = state.add_access(f"{out_edge.data.data}_bias")
            access_scale = state.add_access(f"{out_edge.data.data}_scale")
            state.add_edge(state.exit_node(self.outer_map_entry), f"OUT_{out_edge.data.data}_fp", access_fp, None, dace.Memlet(data=f"{out_edge.data.data}_fp", subset=subset))
            state.add_edge(state.exit_node(self.outer_map_entry), f"OUT_{out_edge.data.data}_bias", access_bias, None, sdfg.make_array_memlet(f"{out_edge.data.data}_bias"))
            state.add_edge(state.exit_node(self.outer_map_entry), f"OUT_{out_edge.data.data}_scale", access_scale, None, sdfg.make_array_memlet(f"{out_edge.data.data}_scale"))
            
            nodes_to_remove.append(out_edge.dst)
        
        # Delete Inner Maps
        nodes_to_remove.append(self.inner_map_entry)
        nodes_to_remove.append(state.exit_node(self.inner_map_entry))
        state.remove_nodes_from(nodes_to_remove)
        
                
                
            
            

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