import Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.FixedOneCanonical

/-!
Compile-time surface regression for the canonical fixed-one active carrier.

| Stage path | Guarded surface | Assurance status |
|---|---|---|
| `fprime.fixed_one.carrier.slot` | sole typed slot | model-level computation |
| `fprime.fixed_one.carrier.fresh` | structure-free fresh body | model-level carrier |
| `fprime.fixed_one.carrier.input` | counter/structure-free active input | model-level carrier |
| `fprime.fixed_one.projection.obligations` | three obligations iff full active obligations | model-level equivalence |
| `fprime.fixed_one.projection.relation` | canonical iff active relation | model-level soundness and completeness |
| `fprime.fixed_one.projection.raw` | conditional raw round-trip | model-level projection; not Rust refinement |
-/

open Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.FixedOneCanonical

#check selected
#check fin_eq_selected
#check Fresh.toStatement
#check Fresh.erase
#check Fresh.erase_toStatement
#check Fresh.toStatement_erase
#check Input.toActive
#check Input.erase
#check Input.erase_toActive
#check Input.priorSlot_derived
#check Input.expectedStructure_derived
#check Input.toActive_erase_of_authority
#check dispatch_derived
#check Obligations
#check Obligations.toActive
#check Obligations.ofActive
#check obligations_iff_active
#check Holds
#check holds_iff_active
#check holds_projection_iff
