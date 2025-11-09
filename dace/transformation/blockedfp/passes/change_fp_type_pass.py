import dace
import dace.transformation.pass_pipeline as ppl
from dace.transformation.blockedfp.passes.add_scale_bias_pass import AddScaleBias
from dataclasses import dataclass
from typing import Dict, Any, Set, List
from dace.sdfg.fp_utils.change_fp_types import change_fptype

@dataclass(unsafe_hash=True)
class ChangeFPType(ppl.Pass):
    
    def __init__(self, names: List[str]):
        self._names = names
    
    # this pass has to run after the blocking step
    def depends_on(self) -> Set[ppl.Pass]:
        return {AddScaleBias}

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Descriptors

    def should_reapply(self, _) -> bool:
        return False
    
    def apply_pass(self, sdfg: dace.SDFG, _: Dict[str, Any]) -> None:
        
        for name in self._names:
            change_fptype(
                sdfg=sdfg, 
                src_fptype=dace.float64,
                dst_fptype=dace.float16,
                cast_in_and_out_data=False,
                arrays_to_replace=set(name))
    