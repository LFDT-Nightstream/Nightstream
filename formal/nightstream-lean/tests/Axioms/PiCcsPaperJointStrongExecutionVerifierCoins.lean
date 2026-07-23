import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.VerifierCoins
import tests.Axioms.Support

/-!
Fail-closed dependency probes for finite causal `Pi_CCS` verifier coins. The
expected sets were recorded from a focused build.
-/

open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.VerifierCoins

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.VerifierCoins.support_nodup' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms support_nodup

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.VerifierCoins.support_nonempty' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms support_nonempty

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.VerifierCoins.support_cardinality_pow' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms support_cardinality_pow

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.VerifierCoins.mem_support_iff' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms mem_support_iff

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.VerifierCoins.toPublicCoins_alpha_coordinates' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms toPublicCoins_alpha_coordinates

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.VerifierCoins.toPublicCoins_gamma' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms toPublicCoins_gamma

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.VerifierCoins.toPublicCoins_round_coordinates' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms toPublicCoins_round_coordinates

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.VerifierCoins.alphaWord_marginal' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms alphaWord_marginal

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.VerifierCoins.gamma_marginal' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms gamma_marginal

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.VerifierCoins.roundWord_marginal' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms roundWord_marginal
