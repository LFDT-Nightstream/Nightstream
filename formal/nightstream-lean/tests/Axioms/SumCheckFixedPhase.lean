import Nightstream.SuperNeo.SumCheck.FixedPhase
import tests.Axioms.Support

/-!
Fail-closed dependency gate for the ghost-free fixed-layout SumCheck verifier.

| Audited theorem | Guarantee |
|---|---|
| `check_eq_true_iff_accepted` | executable replay exactly matches the logical chain |
| `exists_honest_certificate` | representable semantic rounds produce certificate-only messages |
| `expectedRoundsRepresentable_of_honest` | honest messages discharge the fixed-degree semantic premise |
| `complete` | the true initial sum and honest fixed rounds are accepted |
| `false_acceptance_implies_bad_challenge` | false acceptance deterministically exposes a semantic collision |
-/

open Nightstream.SuperNeo.SumCheck.Finite.FixedPhase

/-- info: 'Nightstream.SuperNeo.SumCheck.Finite.FixedPhase.check_eq_true_iff_accepted' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms check_eq_true_iff_accepted

/-- info: 'Nightstream.SuperNeo.SumCheck.Finite.FixedPhase.exists_honest_certificate' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms exists_honest_certificate

/-- info: 'Nightstream.SuperNeo.SumCheck.Finite.FixedPhase.expectedRoundsRepresentable_of_honest' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms expectedRoundsRepresentable_of_honest

/-- info: 'Nightstream.SuperNeo.SumCheck.Finite.FixedPhase.complete' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms complete

/-- info: 'Nightstream.SuperNeo.SumCheck.Finite.FixedPhase.false_acceptance_implies_bad_challenge' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms false_acceptance_implies_bad_challenge
