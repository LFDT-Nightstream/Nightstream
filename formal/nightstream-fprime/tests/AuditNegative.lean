/-
Provenance: adapted from dcSpark/ProtogaLattice `formal/tests/Meta/AuditNegative.lean` at commit
96b57a1c801b.
-/
import tests.AxiomAudit

/-!
# What `#audit_axioms` rejects

Rejections of `#audit_axioms` (`tests/AxiomAudit.lean`), each pinned with `#guard_msgs`, and one
acceptance so that the rejections are not vacuous.

The fixtures need tokens that `scripts/check-boundaries.sh` bans everywhere else. Each line
carrying one ends in the marker `-- audit-negative fixture`; the script exempts marked lines of
this file only, and fails if any module imports this file, so nothing declared here reaches the
library or another test. -/

theorem AuditNegativeShadowed.target : True := trivial

namespace AuditNegative

/-! Acceptance. The closure of `clean` reaches `Nat.add`, whose compiled body is overridden in
`Init`; ambient overrides are not reported. -/

theorem clean : (1 : Nat) + 1 = 2 := rfl

/-- info: AuditNegative.clean: [] -/
#guard_msgs in #audit_axioms AuditNegative.clean

/-! A short name resolves through the enclosing namespace. -/

/-- error: clean is not fully qualified; it resolves to AuditNegative.clean. Write `#audit_axioms AuditNegative.clean`. -/
#guard_msgs in #audit_axioms clean

/-! A full name that the enclosing namespace captures: `AuditNegativeShadowed.target` names the
root declaration above, but here it resolves to this namespace's copy. -/

theorem AuditNegativeShadowed.target : True := trivial

/-- error: AuditNegativeShadowed.target resolves here to AuditNegative.AuditNegativeShadowed.target, a different declaration; audit it outside the namespace or `open` that captures it -/
#guard_msgs in #audit_axioms AuditNegativeShadowed.target

/-! An unfinished proof. -/

/-- warning: declaration uses `sorry` -/ -- audit-negative fixture
#guard_msgs in set_option warningAsError false in
theorem usesSorry : (1 : Nat) = 2 := sorry -- audit-negative fixture

/-- error: AuditNegative.usesSorry depends on disallowed axioms: [sorryAx] -/
#guard_msgs in #audit_axioms AuditNegative.usesSorry

/-! A locally declared axiom. -/

axiom localAxiom : (1 : Nat) = 2 -- audit-negative fixture

theorem usesLocalAxiom : (1 : Nat) = 2 := localAxiom

/-- error: AuditNegative.usesLocalAxiom depends on disallowed axioms: [AuditNegative.localAxiom] -/
#guard_msgs in #audit_axioms AuditNegative.usesLocalAxiom

/-! A compiled-body override two steps away: the statement mentions `wrapper`, whose body mentions
`overridden`, whose compiled code is `fastImpl` while the kernel reduces `overridden` itself. -/

def fastImpl (n : Nat) : Nat := n + 1

@[implemented_by fastImpl] def overridden (n : Nat) : Nat := n

def wrapper (n : Nat) : Nat := overridden n

theorem reachesOverride : wrapper 1 = 1 := rfl

/-- error: AuditNegative.reachesOverride reaches compiled-body overrides outside `AxiomAudit.ambientRoots`: AuditNegative.overridden (@[implemented_by]) -/
#guard_msgs in #audit_axioms AuditNegative.reachesOverride

/-! The other override attribute, reached directly. -/

@[extern "nightstream_audit_negative"] def externed (n : Nat) : Nat := n

theorem reachesExtern : externed 1 = 1 := rfl

/-- error: AuditNegative.reachesExtern reaches compiled-body overrides outside `AxiomAudit.ambientRoots`: AuditNegative.externed (@[extern]) -/
#guard_msgs in #audit_axioms AuditNegative.reachesExtern

end AuditNegative
