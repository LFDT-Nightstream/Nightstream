import tests.SumCheckFiniteRejection
import tests.Axioms.Support

/-!
Fail-closed trusted-dependency audit for the finite SumCheck rejection
witnesses.

Owns: axiom expectations for the negative-witness set that discharges
`SUM-FINITE-CERT`'s Section-8 counterexample obligation, plus the positive
control the negatives are measured against.

Does not own: the finite certificate semantics themselves
(`tests/Axioms/SumCheckFinite.lean`), symbolic SumCheck, root counting, or
production integration.

The witnesses use kernel `decide`, so none of them may acquire
`Lean.trustCompiler`. This guard exists so that a later switch to
`native_decide` breaks the build rather than silently widening the trusted
base of the promotion evidence.
-/

namespace NightstreamTests.Axioms.SumCheckFiniteRejection

open NightstreamTests.Axioms

/-! Positive control. -/

/-- info: 'NightstreamTests.SumCheckFiniteRejection.honest_accepted' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms NightstreamTests.SumCheckFiniteRejection.honest_accepted

/-- info: 'NightstreamTests.SumCheckFiniteRejection.honest_chain' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms NightstreamTests.SumCheckFiniteRejection.honest_chain

/-! One rejection per `checkChain` branch. -/

/-- info: 'NightstreamTests.SumCheckFiniteRejection.emptyMessage_rejected' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms NightstreamTests.SumCheckFiniteRejection.emptyMessage_rejected

/-- info: 'NightstreamTests.SumCheckFiniteRejection.trailingZero_rejected' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms NightstreamTests.SumCheckFiniteRejection.trailingZero_rejected

/-- info: 'NightstreamTests.SumCheckFiniteRejection.degreeTwo_accepted_at_cap_two' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms NightstreamTests.SumCheckFiniteRejection.degreeTwo_accepted_at_cap_two

/-- info: 'NightstreamTests.SumCheckFiniteRejection.degreeAboveCap_rejected' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms NightstreamTests.SumCheckFiniteRejection.degreeAboveCap_rejected

/-- info: 'NightstreamTests.SumCheckFiniteRejection.brokenInitialClaim_rejected' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms NightstreamTests.SumCheckFiniteRejection.brokenInitialClaim_rejected

/-- info: 'NightstreamTests.SumCheckFiniteRejection.brokenTerminal_rejected' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms NightstreamTests.SumCheckFiniteRejection.brokenTerminal_rejected

/-- info: 'NightstreamTests.SumCheckFiniteRejection.missingChallenge_rejected' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms NightstreamTests.SumCheckFiniteRejection.missingChallenge_rejected

/-- info: 'NightstreamTests.SumCheckFiniteRejection.extraChallenge_rejected' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms NightstreamTests.SumCheckFiniteRejection.extraChallenge_rejected

/-! Length discipline. -/

/-- info: 'NightstreamTests.SumCheckFiniteRejection.honest_lockstep' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms NightstreamTests.SumCheckFiniteRejection.honest_lockstep

end NightstreamTests.Axioms.SumCheckFiniteRejection
