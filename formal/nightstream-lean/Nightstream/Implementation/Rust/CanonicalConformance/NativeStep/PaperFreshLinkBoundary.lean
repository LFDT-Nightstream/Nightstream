import Nightstream.Protocol.FPrime.Step

/-!
Contract: exact paper-equality boundary of the native-step fresh-link
interface.

Owns:
- the factorization required to turn a native binary fresh-link predicate into
  HyperNova's unary `freshPublic`/`encodeInstance` equality;
- a finite countermodel admitted by the current abstract `Step.Semantics`
  interface but impossible to express by such an equality.

Does not own: the production fresh-link implementation, a claim that the
production relation has this countermodel, Rust-source refinement, codecs,
R1CS, or the delayed outgoing-link discharge.  A concrete production bridge
can close this boundary by proving its actual fresh link has the stated
factorization.

Emits constraints: no.
-/

namespace Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.PaperFreshLinkBoundary

open Nightstream.Protocol.FPrime

universe uDigest uFresh uEncoded

/-- HyperNova Construction 2 compares two independently computed unary
encodings.  This is the exact factorization a native binary link predicate
must provide before it can instantiate that paper check. -/
def EqualityFactorization
    {Digest : Type uDigest}
    {Fresh : Type uFresh}
    {Encoded : Type uEncoded}
    (link : Digest -> Fresh -> Bool)
    (freshPublic : Fresh -> Encoded)
    (encodeInstance : Digest -> Encoded) : Prop :=
  forall digest fresh,
    link digest fresh = true <->
      freshPublic fresh = encodeInstance digest

/-- A four-entry binary relation whose true fibers overlap without being
equal.  Equality of unary encodings cannot have this shape. -/
def overlappingLink (digest fresh : Bool) : Bool :=
  digest || fresh

/-- No choice of encoded carrier or unary maps can express `overlappingLink`
as the paper's public-input equality. -/
theorem overlappingLink_not_equalityFactorized
    {Encoded : Type uEncoded}
    (freshPublic : Bool -> Encoded)
    (encodeInstance : Bool -> Encoded) :
    Not (EqualityFactorization overlappingLink freshPublic encodeInstance) := by
  intro factorized
  have falseTrue :
      freshPublic true = encodeInstance false :=
    (factorized false true).1 (by decide)
  have trueFalse :
      freshPublic false = encodeInstance true :=
    (factorized true false).1 (by decide)
  have trueTrue :
      freshPublic true = encodeInstance true :=
    (factorized true true).1 (by decide)
  have falseFalse :
      freshPublic false = encodeInstance false :=
    trueFalse.trans (trueTrue.symm.trans falseTrue)
  have rejected :
      overlappingLink false false = true :=
    (factorized false false).2 falseFalse
  exact Bool.noConfusion rejected

/-- A complete inhabitant of the current native lifecycle interface whose
fresh-link field is the non-factorizing relation above.  The other fields are
irrelevant to the obstruction and are fixed transparently. -/
def counterSemantics :
    Step.Semantics Bool Bool Bool Bool Unit Unit where
  emptyRunning := false
  initialNebula := none
  runningDigest := id
  chunkDigest := fun _ fresh => fresh.any id
  freshLink := overlappingLink
  nifsVerify := fun _ running _ _ => some running
  applicationStep := fun _ _ _ => true
  nebulaVerify := fun _ _ _ => true

/-- Exact current-interface obstruction: `Step.Semantics` does not by itself
supply the paper-required unary encoding factorization. -/
theorem currentInterface_admits_nonFactorizingFreshLink
    {Encoded : Type uEncoded}
    (freshPublic : Bool -> Encoded)
    (encodeInstance : Bool -> Encoded) :
    Not (EqualityFactorization counterSemantics.freshLink
      freshPublic encodeInstance) := by
  exact overlappingLink_not_equalityFactorized freshPublic encodeInstance

end Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.PaperFreshLinkBoundary
