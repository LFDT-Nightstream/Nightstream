import tests.AxiomAudit
import NightstreamFPrime.Spec.Nebula.RunMemory

open NightstreamFPrime.Spec.Nebula.MemoryTrace

#audit_axioms eventValuesBounded_of_binds
#audit_axioms accessAll_valid
#audit_axioms Reorders.mem_order
#audit_axioms runMemory_of_executes
#audit_axioms runMemory_of_slotOrder_executes
