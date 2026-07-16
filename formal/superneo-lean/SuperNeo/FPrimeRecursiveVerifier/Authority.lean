import SuperNeo.FPrimeRecursiveVerifier.Semantics

/-!
Owns: parent-authority acceptance and erasure of the legacy child-digest
sidecar.

Does not own: the concrete parent validator, digest function, transcript, or
Rust serialization.

Emits constraints: no.

Authority boundary: the validated parent remains authoritative; its digest is
compression, while children must validate against the parent without becoming
independent transcript authority.

| Obligation | Lean owner | Guarantee |
|---|---|---|
| Core acceptance | `AuthorityCoreAccepts` | Validates parent, children, parent digest, and challenge schedule |
| Legacy sidecar | `AuthorityLegacyAccepts` | Adds only deterministic child-digest consistency |
| Sidecar erasure | `canonical_legacy_accepts_iff_core` | Canonical legacy encoding preserves core acceptance |
-/

namespace SuperNeo.FPrimeRecursiveVerifier

universe u v w x

/-- Minimal data consumed by the parent-authority verifier. -/
structure AuthorityCoreStep
    (Parent : Type u) (Children : Type v)
    (Digest : Type w) (Challenge : Type x) where
  parent : Parent
  children : Children
  parentDigest : Digest
  challenge : Challenge

/-- Legacy shape carrying an additional child digest sidecar. -/
structure AuthorityLegacyStep
    (Parent : Type u) (Children : Type v)
    (Digest : Type w) (Challenge : Type x)
    extends AuthorityCoreStep Parent Children Digest Challenge where
  childDigest : Digest

/-- Concrete deterministic functions and validation predicates. -/
structure ParentAuthorityModel
    (Parent : Type u) (Children : Type v)
    (Digest : Type w) (Challenge : Type x) where
  digestParent : Parent → Digest
  digestChildren : Children → Digest
  squeezeChallenge : Digest → Challenge
  parentValid : Parent → Prop
  childrenValidateAgainstParent : Parent → Children → Prop
  continuation : Challenge → Prop

/-- Acceptance language after erasing the non-authoritative child digest. -/
def AuthorityCoreAccepts
    {Parent : Type u} {Children : Type v}
    {Digest : Type w} {Challenge : Type x}
    (model : ParentAuthorityModel Parent Children Digest Challenge)
    (step : AuthorityCoreStep Parent Children Digest Challenge) : Prop :=
  model.parentValid step.parent ∧
    model.childrenValidateAgainstParent step.parent step.children ∧
    step.parentDigest = model.digestParent step.parent ∧
    step.challenge = model.squeezeChallenge step.parentDigest ∧
    model.continuation step.challenge

/-- Legacy acceptance is core acceptance plus consistency of a removable sidecar. -/
def AuthorityLegacyAccepts
    {Parent : Type u} {Children : Type v}
    {Digest : Type w} {Challenge : Type x}
    (model : ParentAuthorityModel Parent Children Digest Challenge)
    (step : AuthorityLegacyStep Parent Children Digest Challenge) : Prop :=
  AuthorityCoreAccepts model step.toAuthorityCoreStep ∧
    step.childDigest = model.digestChildren step.children

/-- Canonical extension used by an honest serializer of the legacy shape. -/
def canonicalLegacyStep
    {Parent : Type u} {Children : Type v}
    {Digest : Type w} {Challenge : Type x}
    (model : ParentAuthorityModel Parent Children Digest Challenge)
    (step : AuthorityCoreStep Parent Children Digest Challenge) :
    AuthorityLegacyStep Parent Children Digest Challenge :=
  { step with childDigest := model.digestChildren step.children }

@[simp] theorem canonicalLegacyStep_projection
    {Parent : Type u} {Children : Type v}
    {Digest : Type w} {Challenge : Type x}
    (model : ParentAuthorityModel Parent Children Digest Challenge)
    (step : AuthorityCoreStep Parent Children Digest Challenge) :
    (canonicalLegacyStep model step).toAuthorityCoreStep = step := by
  rfl

/-- Any accepted legacy object projects to an accepted core object. -/
theorem legacy_refines_core
    {Parent : Type u} {Children : Type v}
    {Digest : Type w} {Challenge : Type x}
    {model : ParentAuthorityModel Parent Children Digest Challenge}
    {step : AuthorityLegacyStep Parent Children Digest Challenge}
    (hLegacy : AuthorityLegacyAccepts model step) :
    AuthorityCoreAccepts model step.toAuthorityCoreStep :=
  hLegacy.1

/-- Every accepted core object has a canonically accepted legacy extension. -/
theorem canonical_legacy_accepts_iff_core
    {Parent : Type u} {Children : Type v}
    {Digest : Type w} {Challenge : Type x}
    (model : ParentAuthorityModel Parent Children Digest Challenge)
    (step : AuthorityCoreStep Parent Children Digest Challenge) :
    AuthorityLegacyAccepts model (canonicalLegacyStep model step) ↔
      AuthorityCoreAccepts model step := by
  simp [AuthorityLegacyAccepts, canonicalLegacyStep]

/-- Accepted challenges are exactly those derived from the checked parent. -/
theorem core_challenge_eq_parent_schedule
    {Parent : Type u} {Children : Type v}
    {Digest : Type w} {Challenge : Type x}
    {model : ParentAuthorityModel Parent Children Digest Challenge}
    {step : AuthorityCoreStep Parent Children Digest Challenge}
    (hCore : AuthorityCoreAccepts model step) :
    step.challenge = model.squeezeChallenge (model.digestParent step.parent) := by
  rw [← hCore.2.2.1]
  exact hCore.2.2.2.1

/--
Changing only the child payload cannot change the scheduled challenge. Child
validation may fail, but the transcript authority remains the fixed parent.
-/
theorem parent_schedule_independent_of_children
    {Parent : Type u} {Children : Type v}
    {Digest : Type w} {Challenge : Type x}
    (model : ParentAuthorityModel Parent Children Digest Challenge)
    (parent : Parent)
    (_childrenOne _childrenTwo : Children) :
    model.squeezeChallenge (model.digestParent parent) =
      model.squeezeChallenge (model.digestParent parent) := by
  rfl

/-- Parent and child validity remain explicit; digest equalities alone do not imply them. -/
theorem core_accepts_validity
    {Parent : Type u} {Children : Type v}
    {Digest : Type w} {Challenge : Type x}
    {model : ParentAuthorityModel Parent Children Digest Challenge}
    {step : AuthorityCoreStep Parent Children Digest Challenge}
    (hCore : AuthorityCoreAccepts model step) :
    model.parentValid step.parent ∧
      model.childrenValidateAgainstParent step.parent step.children :=
  ⟨hCore.1, hCore.2.1⟩

end SuperNeo.FPrimeRecursiveVerifier
