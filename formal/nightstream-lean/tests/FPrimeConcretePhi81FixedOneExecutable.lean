import Nightstream.Protocol.FPrime.ConcretePhi81.ActiveEvaluator.FixedOneCanonical

/-!
Compile-time regression for the payload-minimal fixed-one F-prime verifier.

| Stage path | Guarded surface | Failure caught |
|---|---|---|
| `fprime.fixed_one.carrier` | payload-only input and canonical materializers | prover-owned structure/stage/selection fields |
| `fprime.fixed_one.context` | exact canonical NIFS context projection | semantic/executable context drift |
| `fprime.fixed_one.outer` | two retained outer equation families | reintroduced redundant checks |
| `fprime.fixed_one.nifs` | direct exact raw-message NIFS checker | abstract callback authority |
| `fprime.fixed_one.run` | fail-closed canonical output and exact physical characterization | caller-supplied output |
| `fprime.fixed_one.semantic` | conditional soundness and honest completeness | unconditional or vacuous semantic claim |
-/

open Nightstream.Protocol.FPrime.ConcretePhi81.ActiveEvaluator.FixedOneCanonical

#check Input
#check Input.system
#check Input.slot
#check Input.toSemantic
#check Input.toActive
#check Input.toActive_fresh_stage
#check Input.toActive_priorPc
#check nifsContext
#check nifsContext_materialize
#check Certificate
#check OuterChecks
#check outerCheck
#check outerCheck_eq_true_iff
#check Accepted
#check check
#check check_eq_true_iff_accepted
#check PhysicalChecks
#check run
#check run_eq_some_iff_physicalChecks
#check SoundnessClosure
#check run_sound_of_closure
#check exists_run_and_holds_or_samplerShortfall
