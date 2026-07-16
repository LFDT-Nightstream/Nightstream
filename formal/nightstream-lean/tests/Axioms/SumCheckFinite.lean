import Nightstream.SuperNeo.SumCheck.VerifierCertificate
import Nightstream.SuperNeo.SumCheck.HypercubeTruth
import tests.Axioms.Support

/-!
Fail-closed trusted-dependency audit for the finite verifier-visible SumCheck
layer.

Owns: trusted-dependency expectations for the finite polynomial encoding,
executable claimed-chain verifier, completeness, symbolic projection, and the
canonical finite hypercube truth-path construction.

Does not own: the generic symbolic SumCheck audit, transcript replay, PiCCS,
root counting, or any implementation/R1CS theorem.

| Audited theorem | Guarantee |
|---|---|
| `Message.canonicalCheck_eq_true_iff` | raw shape check exactly matches canonical coefficients |
| `check_eq_true_iff_accepted` | executable replay exactly matches the finite relation |
| `complete_of_canonical_chain` | every accepted finite chain executes successfully |
| `Chain.messages_length_eq_challenges_length` | accepted finite chains consume messages and challenges in lockstep |
| `HypercubeTruth.semanticGhosts_honest` | expected rounds and terminal are derived from one explicit polynomial |
| `accepted_implies_symbolicAccepted_and_truthPath` | finite replay projects into the symbolic model with honest ghosts |
-/

open Nightstream.SuperNeo.SumCheck.Finite

/-- info: 'Nightstream.SuperNeo.SumCheck.Finite.Message.canonicalCheck_eq_true_iff' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Message.canonicalCheck_eq_true_iff

/-- info: 'Nightstream.SuperNeo.SumCheck.Finite.check_eq_true_iff_accepted' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms check_eq_true_iff_accepted

/-- info: 'Nightstream.SuperNeo.SumCheck.Finite.complete_of_canonical_chain' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms complete_of_canonical_chain

/-- info: 'Nightstream.SuperNeo.SumCheck.Finite.Chain.messages_length_eq_challenges_length' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Chain.messages_length_eq_challenges_length

/-- info: 'Nightstream.SuperNeo.SumCheck.Finite.HypercubeTruth.semanticGhosts_honest' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms HypercubeTruth.semanticGhosts_honest

/-- info: 'Nightstream.SuperNeo.SumCheck.Finite.accepted_implies_symbolicAccepted_and_truthPath' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms accepted_implies_symbolicAccepted_and_truthPath
