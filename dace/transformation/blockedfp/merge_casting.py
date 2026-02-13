import dace
import dace.transformation as xf
from dace.sdfg import nodes, graph as gr

from dace.transformation.blockedfp.libraries import BFPCastinNode, BFPCastoutNode, BFPCastinNodeInt8, BFPCastoutNodeInt8

class MergeCastingTransform(xf.SingleStateTransformation):

    castout = xf.PatternNode(BFPCastoutNode)
    castin = xf.PatternNode(BFPCastinNode)
    
    castout_int8 = xf.PatternNode(BFPCastoutNodeInt8)
    castin_int8 = xf.PatternNode(BFPCastinNodeInt8)
    
    access_node_middle = xf.PatternNode(nodes.AccessNode)
    
    access_a_out = xf.PatternNode(nodes.AccessNode)
    access_b_out = xf.PatternNode(nodes.AccessNode)
    access_c_out = xf.PatternNode(nodes.AccessNode)
    access_a_in = xf.PatternNode(nodes.AccessNode)
    access_b_in = xf.PatternNode(nodes.AccessNode)
    access_c_in = xf.PatternNode(nodes.AccessNode)

    @classmethod
    def expressions(cls):
        state0 = gr.OrderedMultiDiConnectorGraph()
        state0.add_nedge(cls.access_a_out, cls.castout, dace.Memlet())
        state0.add_nedge(cls.access_b_out, cls.castout, dace.Memlet())
        state0.add_nedge(cls.access_c_out, cls.castout, dace.Memlet())
        state0.add_nedge(cls.castout, cls.access_node_middle, dace.Memlet())
        state0.add_nedge(cls.access_node_middle, cls.castin, dace.Memlet())
        state0.add_nedge(cls.castin, cls.access_a_in, dace.Memlet())
        state0.add_nedge(cls.castin, cls.access_b_in, dace.Memlet())
        state0.add_nedge(cls.castin, cls.access_c_in, dace.Memlet())
        
        state1 = gr.OrderedMultiDiConnectorGraph()
        state1.add_nedge(cls.access_a_out, cls.castout_int8, dace.Memlet())
        state1.add_nedge(cls.access_b_out, cls.castout_int8, dace.Memlet())
        state1.add_nedge(cls.access_c_out, cls.castout_int8, dace.Memlet())
        state1.add_nedge(cls.castout_int8, cls.access_node_middle, dace.Memlet())
        state1.add_nedge(cls.access_node_middle, cls.castin_int8, dace.Memlet())
        state1.add_nedge(cls.castin_int8, cls.access_a_in, dace.Memlet())
        state1.add_nedge(cls.castin_int8, cls.access_b_in, dace.Memlet())
        state1.add_nedge(cls.castin_int8, cls.access_c_in, dace.Memlet())
        return [state0, state1]
    
    def can_be_applied(self, graph, expr_index, sdfg, permissive = False):
        if self.access_a_in.data != self.access_a_out.data:
            return False
        if self.access_b_in.data != self.access_b_out.data:
            return False
        if self.access_c_in.data != self.access_c_out.data:
            return False
        return True
    
    def apply(self, graph: dace.SDFGState, sdfg: dace.SDFG):
        for edge in graph.out_edges(self.access_a_in):
            graph.add_edge(self.access_a_out, None, edge.dst, edge.dst_conn, edge.data)
        for edge in graph.out_edges(self.access_b_in):
            graph.add_edge(self.access_b_out, None, edge.dst, edge.dst_conn, edge.data)
        for edge in graph.out_edges(self.access_c_in):
            graph.add_edge(self.access_c_out, None, edge.dst, edge.dst_conn, edge.data)
        
        graph.remove_nodes_from(
            [self.castout, self.access_node_middle, self.castin, self.access_a_in, self.access_b_in, self.access_c_in]
        )