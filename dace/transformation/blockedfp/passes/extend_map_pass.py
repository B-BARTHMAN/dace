import dace
import dace.transformation.pass_pipeline as ppl
from dace.transformation.blockedfp.passes.change_fp_type_pass import ChangeFPType

import dace.transformation as xf
from dataclasses import dataclass
from dace.sdfg.state import StateSubgraphView
from dace.sdfg.utils import node_path_graph
from typing import Dict, Any, Set, List
import dace.sdfg.tasklet_utils as tutil

@dataclass(unsafe_hash=True)
class ExtendMapPass(ppl.Pass):
    
    def __init__(self, names: List[str]):
        self._names = names
    
    # this pass has to run after the add scale/bias step
    def depends_on(self) -> Set[ppl.Pass]:
        return {ChangeFPType}
    
    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Everything
    
    def should_reapply(self, _) -> bool:
        return False
    
    def apply_pass(self, sdfg: dace.SDFG, _: Dict[str, Any]):
        
        for state in sdfg.states():
            for node in state.nodes():
                
                if isinstance(node, dace.nodes.MapEntry):
                    self._handle_map_entry(node, state, sdfg)
                    map_exit = state.exit_node(node)
                    self._handle_map_exit(map_exit, state, sdfg)
    
    def _handle_map_entry(self, map_entry: dace.nodes.MapEntry, state: dace.SDFGState, sdfg: dace.SDFG) -> None:
        
        for edge in state.in_edges(map_entry):
            if edge.data.data in self._names:
                
                # add in connectors
                for con in [f"{edge.dst_conn}_scale", f"{edge.dst_conn}_bias"]:
                    if not con in map_entry.in_connectors:
                        map_entry.add_in_connector(con)
                
                # connect if it is connected to direct access
                if isinstance(edge.src, dace.nodes.AccessNode):
                    scale_access = state.add_access(f"{edge.data.data}_scale")
                    bias_access = state.add_access(f"{edge.data.data}_bias")
                    
                    state.add_edge(scale_access, None, map_entry, f"{edge.dst_conn}_scale", sdfg.make_array_memlet(f"{edge.data.data}_scale"))
                    state.add_edge(bias_access, None, map_entry, f"{edge.dst_conn}_bias", sdfg.make_array_memlet(f"{edge.data.data}_bias"))
                
                # connect if inner map
                elif isinstance(edge.src, dace.nodes.MapEntry):
                    
                    # add out connectors to outer map
                    for con in [f"{edge.src_conn}_scale", f"{edge.src_conn}_bias"]:
                        if not con in edge.src.out_connectors:
                            edge.src.add_out_connector(con)
                    
                    # calculate memlets
                    subset = dace.subsets.Range(edge.data.subset.ranges[:len(edge.data.subset.ranges)//2])
                    scale_memlet = dace.Memlet(data=f"{edge.data.data}_scale", subset=subset)
                    bias_memlet = dace.Memlet(data=f"{edge.data.data}_bias", subset=subset)
                    
                    state.add_edge(edge.src, f"{edge.src_conn}_scale", map_entry, f"{edge.dst_conn}_scale", scale_memlet)
                    state.add_edge(edge.src, f"{edge.src_conn}_bias", map_entry, f"{edge.dst_conn}_bias", bias_memlet)

    def _handle_map_exit(self, map_exit: dace.nodes.MapExit, state: dace.SDFGState, sdfg: dace.SDFG) -> None:
        
        for edge in state.out_edges(map_exit):
            if edge.data.data in self._names:
                
                # add out connectors
                for con in [f"{edge.src_conn}_scale", f"{edge.src_conn}_bias"]:
                    if not con in map_exit.out_connectors:
                        map_exit.add_out_connector(con)
                
                # connect if it is connected to direct access
                if isinstance(edge.dst, dace.nodes.AccessNode):
                    scale_access = state.add_access(f"{edge.data.data}_scale")
                    bias_access = state.add_access(f"{edge.data.data}_bias")
                    
                    state.add_edge(map_exit, f"{edge.src_conn}_scale", scale_access, None, sdfg.make_array_memlet(f"{edge.data.data}_scale"))
                    state.add_edge(map_exit, f"{edge.src_conn}_bias", bias_access, None, sdfg.make_array_memlet(f"{edge.data.data}_bias"))
                
                # connect if inner map
                elif isinstance(edge.dst, dace.nodes.MapExit):
                    
                    # add in connectors to outer map
                    for con in [f"{edge.dst_conn}_scale", f"{edge.dst_conn}_bias"]:
                        if not con in edge.dst.in_connectors:
                            edge.dst.add_in_connector(con)
                    
                    # calculate memlets
                    subset = dace.subsets.Range(edge.data.subset.ranges[:len(edge.data.subset.ranges)//2])
                    scale_memlet = dace.Memlet(data=f"{edge.data.data}_scale", subset=subset)
                    bias_memlet = dace.Memlet(data=f"{edge.data.data}_bias", subset=subset)
                    
                    state.add_edge(map_exit, f"{edge.src_conn}_scale", edge.dst, f"{edge.dst_conn}_scale", scale_memlet)
                    state.add_edge(map_exit, f"{edge.src_conn}_bias", edge.dst, f"{edge.dst_conn}_bias", bias_memlet)