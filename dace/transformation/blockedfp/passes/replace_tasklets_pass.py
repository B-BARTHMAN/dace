import dace
import dace.transformation.pass_pipeline as ppl
import dace.transformation as xf
from dace.transformation.blockedfp.libraries.bfpadd import BFPAddNode
from dace.transformation.blockedfp.passes.extend_tasklets_pass import ExtendTaskletsPass
from dace.sdfg.graph import MultiConnectorEdge
from dace.sdfg.state import StateSubgraphView
from dace.sdfg.utils import node_path_graph
from dataclasses import dataclass
from typing import Dict, Any, List, Set

@dataclass(unsafe_hash=True)
class ReplaceTaskletsPass(ppl.Pass):
    
    def __init__(self, name: str):
        self._name = name
    
    # this pass has to run after the add scale/bias step
    def depends_on(self) -> Set[ppl.Pass]:
        return {ExtendTaskletsPass}
    
    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Everything
    
    def should_reapply(self, _) -> bool:
        return False

    def apply_pass(self, sdfg: dace.SDFG, _: Dict[str, Any]) -> None:
        sdfg.apply_transformations(ReplaceTaskletsTransform, options={"name": self._name}, validate=False)

class ReplaceTaskletsTransform(xf.SingleStateTransformation):
    
    tasklet = xf.PatternNode(dace.nodes.Tasklet)
    
    def __init__(self, name: str):
        self._name = name
    
    @classmethod
    def expressions(cls) -> List[StateSubgraphView]:
        return [node_path_graph(cls.tasklet)]
    
    def can_be_applied(self, graph: dace.SDFGState, expr_index: int, sdfg: dace.SDFG, permissive = False) -> bool:
        # very primitive for now, just check if the necessary Memlets feed into it
        available_names = [edge.data.data for edge in graph.in_edges(self.tasklet)]
        
        for name in [f"{self._name}", f"{self._name}_scale", f"{self._name}_bias"]:
            if name not in available_names:
                return False
        
        return True
    
    def apply(self, graph: dace.SDFGState, sdfg: dace.SDFG) -> None:
        
        libnode = self._create_libnode()
        self._replace_tasklet(libnode, graph)
    
    def _create_libnode(self) -> dace.nodes.LibraryNode:
        # primitive  check what type of op it is
        code = self.tasklet.code.as_string
        if '+' in code:
            return BFPAddNode("bfpadd")
        else:
            raise ValueError(f"Not supported opperation was encountered: {code}")
    
    def _replace_tasklet(self, libnode: dace.nodes.LibraryNode, state: dace.SDFGState) -> None:
        
        # add libnode to the state
        state.add_node(libnode)
        
        # connect old inputs to libnode
        for edge in state.in_edges(self.tasklet):
            lib_in_connector = self._get_libnode_in_connector(edge)
            state.add_edge(edge.src, edge.src_conn, libnode, lib_in_connector, edge.data)
        
        # connect old output to libnode
        for edge in state.out_edges(self.tasklet):
            state.add_edge(libnode, "_out", edge.dst, edge.dst_conn, edge.data)
        
        # delete old tasklet
        state.remove_node(self.tasklet)
        
        libnode.expand(state)
        
    
    # helper method
    def _get_libnode_in_connector(self, edge: MultiConnectorEdge[dace.Memlet]) -> str:
        if edge.data.data == f"{self._name}":
            return "_fp"
        elif edge.data.data == f"{self._name}_scale":
            return "_scale"
        elif edge.data.data == f"{self._name}_bias":
            return "_bias"
        else:
            return "_other"