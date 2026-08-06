import Nightstream.Protocol.FPrime.Frozen

/-!
Focused surface regression for the paper-only one-slot executable verifiers.

Assurance tier: model-level. This checks only the direct fixed-one step and
terminal interfaces and their equality with independent Construction-2
transitions. It does not assert Rust/R1CS conformance or a global
arithmetization lower bound.
-/

open Nightstream.Protocol.FPrime.Frozen.HyperNova

#check FixedOne.Step.eval_eq_generic
#check FixedOne.Step.accepts_iff_transition
#check FixedOne.Terminal.eval_eq_generic
#check FixedOne.Terminal.accepts_iff_transition
