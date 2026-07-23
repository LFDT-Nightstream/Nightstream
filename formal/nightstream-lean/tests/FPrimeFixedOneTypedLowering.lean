import Nightstream.Implementation.Lowering.FPrimeFixedOne

/-!
Focused surface regression for obligation 9 of the paper-authoritative
fixed-one `F'` verifier.

Assurance tier: model-proved.  The checks cover artifact-independent
primitive/block/program semantics, total abstract emission receipts,
definitional four-way cost conservation, and the exact fixed-one step and
terminal programs.  They do not claim a selected physical encoding or
Rust/R1CS conformance.
-/

open Nightstream.Implementation.Lowering.Typed

#check Primitive.exec_eq_some_iff_holds
#check Primitive.sound
#check Primitive.complete
#check Block.exec_eq_some_iff_holds
#check Block.sound
#check Block.complete
#check Program.exec_eq_some_iff_holds
#check Program.sound
#check Program.complete
#check Program.flattened_conservation
#check Program.cost_eq_receipt_event_cost

#check Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.accepts_iff_fixedOne
#check Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.accepts_iff_transition
#check Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.accepts_iff_fixedOne
#check Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.accepts_iff_transition
