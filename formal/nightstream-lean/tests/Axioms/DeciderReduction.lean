import Nightstream.Assurance.DeciderReduction
import tests.Axioms.Support

/-! Axiom gate for the concrete F' decider reduction. -/

/-- info: 'Nightstream.Assurance.DeciderReduction.sound_or_bad' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.DeciderReduction.sound_or_bad

/-- info: 'Nightstream.Assurance.DeciderReduction.verifierReductionTarget' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.DeciderReduction.verifierReductionTarget
