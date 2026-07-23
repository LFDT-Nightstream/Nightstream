import Nightstream.SuperNeo.Folding.PiRLC.PaperWeakFiniteUniform
import tests.Axioms.Support

/-!
Fail-closed dependency guards for the finite-uniform operational paper
`Pi_RLC` weak reduction. The exact sets below were recorded from a focused
kernel probe.
-/

open Nightstream.SuperNeo.Folding.PiRLC.PaperWeakFiniteUniform

/-- info: 'Nightstream.SuperNeo.Folding.PiRLC.PaperWeakFiniteUniform.theorem10Contract' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms theorem10Contract

/-- info: 'Nightstream.SuperNeo.Folding.PiRLC.PaperWeakFiniteUniform.operationalGame' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms operationalGame

/-- info: 'Nightstream.SuperNeo.Folding.PiRLC.PaperWeakFiniteUniform.paperWeak' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms paperWeak
