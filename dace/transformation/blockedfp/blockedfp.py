import dace.transformation.pass_pipeline as ppl

from dace.transformation.blockedfp.auxiliary_arrays import add_auxiliary_arrays
from dace.transformation.blockedfp.cast_in_out import cast_in_out
from dace.transformation.blockedfp.blockedfp_gemm import BlockedFPGEMMTransform

class BlockedFP(ppl.Pass):
    
    def __init__(self, blocking_factor: int = 16):
        self._blocking_factor = blocking_factor
        super().__init__()
    
    def modifies(self):
        return ppl.Modifies.Everything
    
    def apply_pass(self, sdfg, pipeline_results):
        
        sdfg.apply_transformations(BlockedFPGEMMTransform, options={"blocking_factor": self._blocking_factor}, validate=True)