import SuperNeo.FoldingProtocol.FiatShamirReroute

/-!
Contract interface for `SuperNeo.FiatShamirReroute`.

Spec: `specs/FiatShamirReroute.spec.md`
-/

namespace SuperNeo

universe u v w x

namespace FiatShamirRerouteInterface

/-! ## Core Surfaces -/

/-- [Role: Theorem-Target] Curated re-export of `ParentAuthorityFS`. -/
abbrev ParentAuthorityFS := SuperNeo.ParentAuthorityFS

/-- [Role: Theorem-Target] Challenge derived from parent authority only. -/
abbrev parentChallenge
    {Parent Digest Challenge : Type}
    (fs : SuperNeo.ParentAuthorityFS Parent Digest Challenge)
    (parent : Parent) : Challenge :=
  fs.challenge parent

/--
[Role: Theorem-Target] Different DEC-child payloads cannot alter a challenge
whose schedule depends only on the parent authority.
-/
theorem challenge_independent_of_children
    {Parent Children Digest Challenge : Type}
    (fs : SuperNeo.ParentAuthorityFS Parent Digest Challenge)
    (parent : Parent)
    (children₁ children₂ : Children) :
    fs.challenge parent = fs.challenge parent :=
  SuperNeo.ParentAuthorityFS.challenge_independent_of_children
    fs parent children₁ children₂

/-- [Role: Theorem-Target] Curated re-export of the `Π_RLC` parent authority statement. -/
abbrev rlcParentAuthorityStatement := SuperNeo.rlcParentAuthorityStatement

/-- [Role: Theorem-Target] Curated re-export of the downstream `Π_DEC` validation statement. -/
abbrev decValidationStatement := SuperNeo.decValidationStatement

/-- [Role: Theorem-Target] Curated re-export of the parent-authority reroute statement. -/
abbrev rlcParentAuthorityWithDecValidation :=
  SuperNeo.rlcParentAuthorityWithDecValidation

/--
[Role: Theorem-Target] The downstream DEC validation follows from the RLC
parent authority statement.
-/
theorem rlcParentAuthorityWithDecValidation_of_parent
    {ctx : ProtocolTargetContext} :
    rlcParentAuthorityStatement ctx →
      rlcParentAuthorityWithDecValidation ctx :=
  SuperNeo.rlcParentAuthorityWithDecValidation_of_parent

/--
[Role: Theorem-Target] The parent-authority reroute is theorem-equivalent to
the `Π_RLC` weak parent statement.
-/
theorem rlcParentAuthorityWithDecValidation_iff_parent
    (ctx : ProtocolTargetContext) :
    rlcParentAuthorityWithDecValidation ctx ↔
      rlcParentAuthorityStatement ctx :=
  SuperNeo.rlcParentAuthorityWithDecValidation_iff_parent ctx

/-- [Role: Theorem-Target] Curated re-export of the generic verifier shape. -/
abbrev parentAuthorityVerifierAccepts
    {Parent : Type u} {Children : Type x}
    {Digest : Type v} {Challenge : Type w}
    (fs : SuperNeo.ParentAuthorityFS Parent Digest Challenge)
    (parentValid : Parent → Prop)
    (childrenValidateAgainstParent : Parent → Children → Prop)
    (continuation : Challenge → Prop)
    (parent : Parent) (children : Children) : Prop :=
  SuperNeo.parentAuthorityVerifierAccepts
    fs parentValid childrenValidateAgainstParent continuation
    parent children

/--
[Role: Theorem-Target] Accepted parent-authority executions run their
continuation at the parent-derived challenge.
-/
theorem parentAuthorityVerifier_continuation
    {Parent Children Digest Challenge : Type}
    {fs : SuperNeo.ParentAuthorityFS Parent Digest Challenge}
    {parentValid : Parent → Prop}
    {childrenValidateAgainstParent : Parent → Children → Prop}
    {continuation : Challenge → Prop}
    {parent : Parent} {children : Children} :
    parentAuthorityVerifierAccepts
        fs parentValid childrenValidateAgainstParent continuation
        parent children →
      continuation (fs.challenge parent) :=
  SuperNeo.parentAuthorityVerifier_continuation

/-! ## SuperNeo-Specialized Reroute Surface -/

/--
[Role: Theorem-Target] Specialized verifier shape where the
`ProtocolTargetContext` is the transcript-authoritative parent.
-/
abbrev rlcParentAuthorityVerifierAccepts
    {Children : Type x} {Digest : Type v} {Challenge : Type w}
    (fs : SuperNeo.ParentAuthorityFS ProtocolTargetContext Digest Challenge)
    (childrenValidateAgainstParent : ProtocolTargetContext → Children → Prop)
    (continuation : Challenge → Prop)
    (ctx : ProtocolTargetContext) (children : Children) : Prop :=
  SuperNeo.rlcParentAuthorityVerifierAccepts
    fs childrenValidateAgainstParent continuation ctx children

/--
[Role: Theorem-Target] If concrete child validation implies the DEC statement,
then accepted rerouted executions establish the parent-authority statement and
run the continuation at the parent-derived challenge.
-/
theorem rlcParentAuthorityVerifier_sound_from_checked_children
    {Children : Type x} {Digest : Type v} {Challenge : Type w}
    {fs : SuperNeo.ParentAuthorityFS ProtocolTargetContext Digest Challenge}
    {childrenValidateAgainstParent : ProtocolTargetContext → Children → Prop}
    {continuation : Challenge → Prop}
    {ctx : ProtocolTargetContext} {children : Children} :
    (childrenValidateAgainstParent ctx children →
        decValidationStatement ctx) →
      rlcParentAuthorityVerifierAccepts
        fs childrenValidateAgainstParent continuation ctx children →
        rlcParentAuthorityWithDecValidation ctx ∧ continuation (fs.challenge ctx) :=
  SuperNeo.rlcParentAuthorityVerifier_sound_from_checked_children

/--
[Role: Theorem-Target] Accepted rerouted executions are sound from the RLC parent
authority alone, while still checking the child predicate in the verifier shape.
-/
theorem rlcParentAuthorityVerifier_sound_from_parent
    {Children : Type x} {Digest : Type v} {Challenge : Type w}
    {fs : SuperNeo.ParentAuthorityFS ProtocolTargetContext Digest Challenge}
    {childrenValidateAgainstParent : ProtocolTargetContext → Children → Prop}
    {continuation : Challenge → Prop}
    {ctx : ProtocolTargetContext} {children : Children} :
      rlcParentAuthorityVerifierAccepts
        fs childrenValidateAgainstParent continuation ctx children →
        rlcParentAuthorityWithDecValidation ctx ∧ continuation (fs.challenge ctx) :=
  SuperNeo.rlcParentAuthorityVerifier_sound_from_parent

/--
[Role: Theorem-Target] Exact no-payload theorem-level reroute equivalence:
parent authority plus DEC validation is equivalent to the accepted rerouted
verifier shape.
-/
theorem rlcParentAuthorityVerifier_exact_iff
    {Digest : Type v} {Challenge : Type w}
    (fs : SuperNeo.ParentAuthorityFS ProtocolTargetContext Digest Challenge)
    (continuation : Challenge → Prop)
    (ctx : ProtocolTargetContext) :
    parentAuthorityVerifierAccepts
        fs rlcParentAuthorityStatement
        (fun ctx (_ : Unit) => decValidationStatement ctx)
        continuation ctx () ↔
      rlcParentAuthorityWithDecValidation ctx ∧ continuation (fs.challenge ctx) :=
  SuperNeo.rlcParentAuthorityVerifier_exact_iff fs continuation ctx

end FiatShamirRerouteInterface

end SuperNeo
