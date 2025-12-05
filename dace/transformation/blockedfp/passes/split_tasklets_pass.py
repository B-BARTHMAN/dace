import dace
import dace.transformation.pass_pipeline as ppl
from typing import List

from dace.transformation.passes.split_tasklets import SplitTasklets

class SplitTaskletsPass(ppl.Pass):
    
    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Everything
    
    def should_reapply(self, _) -> bool:
        return False
    
    def apply_pass(self, sdfg: dace.SDFG, _) -> None:
        
        SplitTasklets().apply_pass(sdfg, {})