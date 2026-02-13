import dace.transformation.pass_pipeline as ppl

from dace.transformation.blockedfp.blockedfp_gemm import BlockedFPGEMMTransform
from dace.transformation.blockedfp.merge_casting import MergeCastingTransform

class BlockedFP(ppl.Pass):
    
    def __init__(self, blocking_factor: int = 16, use_int8: bool = False):
        self._blocking_factor = blocking_factor
        self._use_int8 = use_int8
        super().__init__()
    
    def modifies(self):
        return ppl.Modifies.Everything
    
    def apply_pass(self, sdfg, pipeline_results):
        
        sdfg.apply_transformations_once_everywhere(BlockedFPGEMMTransform, options={"blocking_factor": self._blocking_factor, "use_int8": self._use_int8}, validate=True)
        sdfg.apply_transformations_once_everywhere(MergeCastingTransform, options={})