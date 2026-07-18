import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Selection.AggregateExactness
import tests.Axioms.Support

/-! Fail-closed dependency ownership for model-level selection substitution. -/

open Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Selection

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Selection.currentSelectionBlock_iff_aggregate' depends on axioms: [Quot.sound] -/
#guard_msgs in
#audit_axioms currentSelectionBlock_iff_aggregate
