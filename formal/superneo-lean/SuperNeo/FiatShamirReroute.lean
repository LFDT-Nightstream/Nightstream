import SuperNeo.PiDEC

/-!
Parent-authority Fiat-Shamir reroute lemma.

This module owns the deterministic proof boundary for using the `Π_RLC` parent
as transcript authority while checking `Π_DEC` outputs as downstream witnesses.
It does not model a random oracle; it proves that, once the challenge schedule
is defined from the parent, child payloads cannot influence that challenge, and
that the existing theorem surface derives the DEC validation statement from the
RLC weak statement.
-/

namespace SuperNeo

universe u v w x

/--
Deterministic Fiat-Shamir schedule whose challenge is derived from the parent
authority object only.
-/
structure ParentAuthorityFS
    (Parent : Type u) (Digest : Type v) (Challenge : Type w) where
  digestParent : Parent → Digest
  squeezeChallenge : Digest → Challenge

namespace ParentAuthorityFS

/-- The challenge obtained from the parent authority. -/
def challenge
    {Parent : Type u} {Digest : Type v} {Challenge : Type w}
    (fs : ParentAuthorityFS Parent Digest Challenge)
    (parent : Parent) : Challenge :=
  fs.squeezeChallenge (fs.digestParent parent)

/--
If the parent authority is fixed, the challenge is fixed.  In particular, two
different DEC-child payloads cannot change the challenge in a parent-authority
schedule because children are not an input to `challenge`.
-/
theorem challenge_independent_of_children
    {Parent : Type u} {Children : Type x}
    {Digest : Type v} {Challenge : Type w}
    (fs : ParentAuthorityFS Parent Digest Challenge)
    (parent : Parent)
    (_children₁ _children₂ : Children) :
    fs.challenge parent = fs.challenge parent := by
  rfl

end ParentAuthorityFS

/-- The `Π_RLC` parent authority statement used by the rerouted transcript. -/
def rlcParentAuthorityStatement (ctx : ProtocolTargetContext) : Prop :=
  piRLCWeakStatement ctx

/-- The downstream `Π_DEC` validation statement checked against the parent. -/
def decValidationStatement (ctx : ProtocolTargetContext) : Prop :=
  piDECKnowledgeStatement ctx

/--
Parent-authority reroute statement: the transcript authority is the `Π_RLC`
weak statement, and `Π_DEC` remains checked as a downstream validation.
-/
def rlcParentAuthorityWithDecValidation (ctx : ProtocolTargetContext) : Prop :=
  rlcParentAuthorityStatement ctx ∧ decValidationStatement ctx

/--
Existing `Π_DEC` theorem surface closes the downstream validation from the
`Π_RLC` parent authority.
-/
theorem rlcParentAuthorityWithDecValidation_of_parent
    {ctx : ProtocolTargetContext}
    (hParent : rlcParentAuthorityStatement ctx) :
    rlcParentAuthorityWithDecValidation ctx := by
  exact ⟨hParent, piDEC_of_weak hParent⟩

/-- Recover the transcript-authoritative `Π_RLC` statement from the reroute. -/
theorem rlcParentAuthority_of_reroute
    {ctx : ProtocolTargetContext}
    (h : rlcParentAuthorityWithDecValidation ctx) :
    rlcParentAuthorityStatement ctx := by
  exact h.1

/--
The parent-authority reroute carries no extra theorem-level authority beyond
the `Π_RLC` weak statement: `Π_DEC` validation is derivable from that parent
statement and therefore can be checked downstream without being an independent
Fiat-Shamir input.
-/
theorem rlcParentAuthorityWithDecValidation_iff_parent
    (ctx : ProtocolTargetContext) :
    rlcParentAuthorityWithDecValidation ctx ↔
      rlcParentAuthorityStatement ctx := by
  constructor
  · intro h
    exact rlcParentAuthority_of_reroute h
  · intro hParent
    exact rlcParentAuthorityWithDecValidation_of_parent hParent

/--
Generic verifier shape for a parent-authority schedule: parent validity and
child validation are both checked, but the continuation sees only the challenge
derived from the parent.
-/
def parentAuthorityVerifierAccepts
    {Parent : Type u} {Children : Type x}
    {Digest : Type v} {Challenge : Type w}
    (fs : ParentAuthorityFS Parent Digest Challenge)
    (parentValid : Parent → Prop)
    (childrenValidateAgainstParent : Parent → Children → Prop)
    (continuation : Challenge → Prop)
    (parent : Parent) (children : Children) : Prop :=
  parentValid parent ∧
    childrenValidateAgainstParent parent children ∧
      continuation (fs.challenge parent)

/--
Projection from the generic verifier shape: accepted executions always run the
continuation at the parent-derived challenge.
-/
theorem parentAuthorityVerifier_continuation
    {Parent : Type u} {Children : Type x}
    {Digest : Type v} {Challenge : Type w}
    {fs : ParentAuthorityFS Parent Digest Challenge}
    {parentValid : Parent → Prop}
    {childrenValidateAgainstParent : Parent → Children → Prop}
    {continuation : Challenge → Prop}
    {parent : Parent} {children : Children}
    (h :
      parentAuthorityVerifierAccepts
        fs parentValid childrenValidateAgainstParent continuation
        parent children) :
    continuation (fs.challenge parent) := by
  exact h.2.2

/--
Specialized verifier shape for the SuperNeo reroute: the `ProtocolTargetContext`
is the parent authority, child material is checked against that parent, and the
continuation receives only the parent-derived challenge.
-/
def rlcParentAuthorityVerifierAccepts
    {Children : Type x} {Digest : Type v} {Challenge : Type w}
    (fs : ParentAuthorityFS ProtocolTargetContext Digest Challenge)
    (childrenValidateAgainstParent : ProtocolTargetContext → Children → Prop)
    (continuation : Challenge → Prop)
    (ctx : ProtocolTargetContext) (children : Children) : Prop :=
  parentAuthorityVerifierAccepts
    fs rlcParentAuthorityStatement childrenValidateAgainstParent continuation
    ctx children

/--
If the specialized verifier accepts and the child-validation predicate is known
to imply the `Π_DEC` statement, then the rerouted parent-authority statement and
the parent-derived continuation both hold.
-/
theorem rlcParentAuthorityVerifier_sound_from_checked_children
    {Children : Type x} {Digest : Type v} {Challenge : Type w}
    {fs : ParentAuthorityFS ProtocolTargetContext Digest Challenge}
    {childrenValidateAgainstParent : ProtocolTargetContext → Children → Prop}
    {continuation : Challenge → Prop}
    {ctx : ProtocolTargetContext} {children : Children}
    (hChildren :
      childrenValidateAgainstParent ctx children →
        decValidationStatement ctx)
    (h :
      rlcParentAuthorityVerifierAccepts
        fs childrenValidateAgainstParent continuation ctx children) :
    rlcParentAuthorityWithDecValidation ctx ∧ continuation (fs.challenge ctx) := by
  exact ⟨⟨h.1, hChildren h.2.1⟩, h.2.2⟩

/--
Using the existing theorem surface, accepted specialized executions are sound
from the `Π_RLC` parent authority alone: `Π_DEC` validation is derivable from the
parent weak statement, while the child predicate is still checked by the verifier
shape.
-/
theorem rlcParentAuthorityVerifier_sound_from_parent
    {Children : Type x} {Digest : Type v} {Challenge : Type w}
    {fs : ParentAuthorityFS ProtocolTargetContext Digest Challenge}
    {childrenValidateAgainstParent : ProtocolTargetContext → Children → Prop}
    {continuation : Challenge → Prop}
    {ctx : ProtocolTargetContext} {children : Children}
    (h :
      rlcParentAuthorityVerifierAccepts
        fs childrenValidateAgainstParent continuation ctx children) :
    rlcParentAuthorityWithDecValidation ctx ∧ continuation (fs.challenge ctx) := by
  exact rlcParentAuthorityVerifier_sound_from_checked_children
    (fun _ => piDEC_of_weak h.1) h

/--
Exact theorem-level reroute with no separate child payload: parent authority plus
the already-derived DEC validation is equivalent to the rerouted statement, and
the continuation is still evaluated at the parent-derived challenge.
-/
theorem rlcParentAuthorityVerifier_exact_iff
    {Digest : Type v} {Challenge : Type w}
    (fs : ParentAuthorityFS ProtocolTargetContext Digest Challenge)
    (continuation : Challenge → Prop)
    (ctx : ProtocolTargetContext) :
    parentAuthorityVerifierAccepts
        fs rlcParentAuthorityStatement
        (fun ctx (_ : Unit) => decValidationStatement ctx)
        continuation ctx () ↔
      rlcParentAuthorityWithDecValidation ctx ∧ continuation (fs.challenge ctx) := by
  constructor
  · intro h
    exact rlcParentAuthorityVerifier_sound_from_checked_children
      (fun hDec => hDec) h
  · intro h
    exact ⟨h.1.1, h.1.2, h.2⟩

end SuperNeo
