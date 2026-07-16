import SuperNeo.FoldingProtocol.FiatShamirReroute
import SuperNeo.FPrimeRecursiveVerifier.Authority

/-!
Owns: the post-`Pi_RLC` authority reduction from the checked RLC parent to the
derived `Pi_DEC` statement, while retaining child validation.

Does not own: concrete child payload validation, Poseidon2 transcript replay,
or Rust/R1CS refinement.

Emits constraints: no.

Authority boundary: the validated RLC parent is transcript authority; DEC
children remain validation inputs but their statement is not an independent
authority check.

| Obligation | Lean owner | Guarantee |
|---|---|---|
| Parent plan | `minimalPostRlc_accepts_iff_target` | RLC parent acceptance implies the post-RLC target |
| DEC erasure | `decStatement_redundant` | Proves the separate DEC statement redundant |
| Child validation | `PostRlcCheckedMinimal` | Keeps concrete validation as an independent premise |
-/

namespace SuperNeo.FPrimeRecursiveVerifier

universe u

/-- Checks visible at the theorem-level post-`Π_RLC` boundary. -/
inductive PostRlcCheck where
  | rlcParent
  | decStatement
deriving Repr, DecidableEq

/-- Paper target after `Π_RLC`: parent validity together with `Π_DEC`. -/
def PostRlcTarget (ctx : SuperNeo.ProtocolTargetContext) : Prop :=
  SuperNeo.piRLCWeakStatement ctx ∧
    SuperNeo.piDECKnowledgeStatement ctx

/-- Interpret theorem-level post-`Π_RLC` checks. -/
def postRlcCheckSemantics :
    PostRlcCheck → SuperNeo.ProtocolTargetContext → Prop
  | .rlcParent => SuperNeo.piRLCWeakStatement
  | .decStatement => SuperNeo.piDECKnowledgeStatement

/-- Legacy plan that treats the derived DEC statement as a separate check. -/
def fullPostRlcChecks : Finset PostRlcCheck :=
  { .rlcParent, .decStatement }

/-- Minimal theorem-level authority plan: only the RLC parent is authoritative. -/
def minimalPostRlcChecks : Finset PostRlcCheck :=
  { .rlcParent }

theorem minimalPostRlc_accepts_iff_target
    (ctx : SuperNeo.ProtocolTargetContext) :
    Accepts postRlcCheckSemantics minimalPostRlcChecks ctx ↔
      PostRlcTarget ctx := by
  constructor
  · intro hAccepts
    have hParent : SuperNeo.piRLCWeakStatement ctx := by
      exact hAccepts .rlcParent (by simp [minimalPostRlcChecks])
    exact ⟨hParent, SuperNeo.piDEC_of_weak hParent⟩
  · intro hTarget check hCheck
    have hOnly : check = .rlcParent := by
      simpa [minimalPostRlcChecks] using hCheck
    subst check
    exact hTarget.1

/-- Certified minimal theorem-level post-RLC plan. -/
def minimalPostRlcPlan :
    CertifiedPlan postRlcCheckSemantics PostRlcTarget where
  checks := minimalPostRlcChecks
  sound := by
    intro ctx hAccepts
    exact (minimalPostRlc_accepts_iff_target ctx).mp hAccepts
  complete := by
    intro ctx hTarget
    exact (minimalPostRlc_accepts_iff_target ctx).mpr hTarget

/-- The full theorem-level plan is also exact. -/
def fullPostRlcPlan :
    CertifiedPlan postRlcCheckSemantics PostRlcTarget where
  checks := fullPostRlcChecks
  sound := by
    intro ctx hAccepts
    exact ⟨
      hAccepts .rlcParent (by simp [fullPostRlcChecks]),
      hAccepts .decStatement (by simp [fullPostRlcChecks])⟩
  complete := by
    intro ctx hTarget check hCheck
    cases check with
    | rlcParent => exact hTarget.1
    | decStatement => exact hTarget.2

/-- `Π_DEC` is a proved-redundant theorem-level check once the parent is valid. -/
theorem decStatement_redundant :
    Redundant postRlcCheckSemantics fullPostRlcChecks .decStatement := by
  intro ctx hWithout
  have hParent : SuperNeo.piRLCWeakStatement ctx := by
    apply hWithout .rlcParent
    simp [fullPostRlcChecks]
  exact SuperNeo.piDEC_of_weak hParent

theorem erase_decStatement_eq_minimal :
    fullPostRlcChecks.erase .decStatement = minimalPostRlcChecks := by
  decide

/-- Mechanical construction of the smaller certificate from the full plan. -/
def fullPostRlcPlanWithoutDec :
    CertifiedPlan postRlcCheckSemantics PostRlcTarget :=
  fullPostRlcPlan.eraseRedundant .decStatement decStatement_redundant

theorem fullPostRlcPlanWithoutDec_checks :
    fullPostRlcPlanWithoutDec.checks = minimalPostRlcChecks := by
  exact erase_decStatement_eq_minimal

/--
Operational payload for the same boundary. `childrenValidate` is deliberately
not derived from the parent theorem and therefore remains a circuit obligation.
-/
structure PostRlcStep (Children : Type u) where
  ctx : SuperNeo.ProtocolTargetContext
  children : Children

/-- Full semantic target including checked child material. -/
def PostRlcCheckedTarget
    {Children : Type u}
    (childrenValidate : SuperNeo.ProtocolTargetContext → Children → Prop)
    (step : PostRlcStep Children) : Prop :=
  SuperNeo.piRLCWeakStatement step.ctx ∧
    childrenValidate step.ctx step.children ∧
    SuperNeo.piDECKnowledgeStatement step.ctx

/-- Minimal operational checks: RLC parent validity and child recomposition. -/
def PostRlcCheckedMinimal
    {Children : Type u}
    (childrenValidate : SuperNeo.ProtocolTargetContext → Children → Prop)
    (step : PostRlcStep Children) : Prop :=
  SuperNeo.piRLCWeakStatement step.ctx ∧
    childrenValidate step.ctx step.children

/-- Parent-derived DEC removes no child-validation obligation. -/
theorem postRlcCheckedMinimal_iff_target
    {Children : Type u}
    (childrenValidate : SuperNeo.ProtocolTargetContext → Children → Prop)
    (step : PostRlcStep Children) :
    PostRlcCheckedMinimal childrenValidate step ↔
      PostRlcCheckedTarget childrenValidate step := by
  constructor
  · rintro ⟨hParent, hChildren⟩
    exact ⟨hParent, hChildren, SuperNeo.piDEC_of_weak hParent⟩
  · intro hTarget
    exact ⟨hTarget.1, hTarget.2.1⟩

/-- Build the generic authority model with the concrete SuperNeo RLC parent predicate. -/
def superNeoParentAuthorityModel
    {Children : Type u} {Digest Challenge : Type}
    (digestParent : SuperNeo.ProtocolTargetContext → Digest)
    (digestChildren : Children → Digest)
    (squeezeChallenge : Digest → Challenge)
    (childrenValidate : SuperNeo.ProtocolTargetContext → Children → Prop)
    (continuation : Challenge → Prop) :
    ParentAuthorityModel
      SuperNeo.ProtocolTargetContext Children Digest Challenge :=
  { digestParent := digestParent
    digestChildren := digestChildren
    squeezeChallenge := squeezeChallenge
    parentValid := SuperNeo.piRLCWeakStatement
    childrenValidateAgainstParent := childrenValidate
    continuation := continuation }

end SuperNeo.FPrimeRecursiveVerifier
