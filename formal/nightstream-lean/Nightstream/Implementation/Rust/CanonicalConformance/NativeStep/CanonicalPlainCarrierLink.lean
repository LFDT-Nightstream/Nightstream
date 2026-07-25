import Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.CanonicalPublicInputLink

/-!
Contract: typed plain-profile realization of the production 270-coordinate
fresh-public carrier check.

Owns:
- the exact plain carrier split into one affine coordinate, 256 independently
  checked field coordinates in lane/bit order, and thirteen padding
  coordinates;
- the verifier-owned `m_in = 270` check;
- exact equality factorization through
  `[1 | enc_inst(digest) | 0^13]`.

Does not own: proof that Rust's variable-length `Vec` loops implement the raw
checker, lifecycle call-site refinement, the Nebula public suffix, R1CS rows,
or proof that a native boundary receipt invokes this check.

Emits constraints: no.
-/

namespace Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.CanonicalPlainCarrierLink

open Nightstream.Implementation.Encoding.FPrime
open Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.PaperFreshLinkBoundary

def logicalWidth : Nat := 257

def carrierWidth : Nat := 270

def paddingWidth : Nat := carrierWidth - logicalWidth

theorem paddingWidth_eq_thirteen : paddingWidth = 13 := by
  decide

/-- The field-valued `enc_inst` body before it has passed the verifier's
zero/one comparisons. Unlike `EncInst`, this type can express malformed
non-binary coordinates. -/
abbrev RawBody := Fin 4 -> Fin 64 -> Nat

/-- Plain SuperNeo carrier in its verifier-owned coordinate groups. -/
structure Carrier where
  one : Nat
  body : RawBody
  padding : Fin paddingWidth -> Nat

/-- Flatten the typed groups in the exact production order: affine one,
lane-major little-endian bits, then carrier padding. -/
def Carrier.coordinates (carrier : Carrier) : List Nat :=
  [carrier.one] ++
    (List.ofFn fun lane : Fin 4 =>
      List.ofFn fun bit : Fin 64 => carrier.body lane bit).flatten ++
    List.ofFn carrier.padding

set_option maxRecDepth 8192 in
theorem Carrier.coordinates_length (carrier : Carrier) :
    carrier.coordinates.length = carrierWidth := by
  simp [Carrier.coordinates, carrierWidth, paddingWidth, logicalWidth]

theorem Carrier.eq_of_fields
    (left right : Carrier)
    (one : left.one = right.one)
    (body : left.body = right.body)
    (padding : left.padding = right.padding) :
    left = right := by
  cases left
  cases right
  cases one
  cases body
  cases padding
  rfl

/-- Claim metadata and the complete typed 270-coordinate carrier. -/
structure Claim where
  mIn : Nat
  x : Carrier

theorem Claim.eq_of_fields
    (left right : Claim)
    (mIn : left.mIn = right.mIn)
    (x : left.x = right.x) :
    left = right := by
  cases left
  cases right
  cases mIn
  cases x
  rfl

/-- Untrusted source container before the verifier checks `x.len()`. -/
structure RawClaim where
  mIn : Nat
  x : List Nat

theorem RawClaim.eq_of_fields
    (left right : RawClaim)
    (mIn : left.mIn = right.mIn)
    (x : left.x = right.x) :
    left = right := by
  cases left
  cases right
  cases mIn
  cases x
  rfl

def encodedBit (digest : Digest) (lane : Fin 4) (bit : Fin 64) : Nat :=
  if (encodeEncInst digest lane).getLsbD bit.val then 1 else 0

def encodeCarrier (digest : Digest) : Carrier where
  one := 1
  body := encodedBit digest
  padding := fun _ => 0

def encodeClaim (digest : Digest) : Claim where
  mIn := carrierWidth
  x := encodeCarrier digest

def encodeRawClaim (digest : Digest) : RawClaim where
  mIn := carrierWidth
  x := (encodeCarrier digest).coordinates

/-- Complete the paper's logical 257-coordinate input with exactly the
thirteen verifier-fixed zero coordinates of the plain SuperNeo carrier. -/
def completeCarrier (input : PublicInput) : Carrier where
  one := input.one
  body := fun lane bit =>
    if (input.body lane).getLsbD bit.val then 1 else 0
  padding := fun _ => 0

def completeClaim (input : PublicInput) : Claim where
  mIn := carrierWidth
  x := completeCarrier input

theorem completeClaim_encodePublicInput
    (digest : Digest) :
    completeClaim (encodePublicInput digest) = encodeClaim digest := by
  rfl

/-- Source-shaped plain-profile checks: exact `m_in`, affine one, all 256
link coordinates, and every verifier-fixed carrier padding coordinate. -/
def check (digest : Digest) (claim : Claim) : Bool :=
  decide (claim.mIn = carrierWidth) &&
    decide (claim.x.one = 1) &&
    decide (forall lane bit, claim.x.body lane bit = encodedBit digest lane bit) &&
    decide (forall padding, claim.x.padding padding = 0)

/-- Executable aggregate semantics of the production plain-profile shape and
coordinate loops over an untrusted variable-length source vector. Proving the
Rust source implements this function remains a separate refinement theorem. -/
def rawCheck (digest : Digest) (claim : RawClaim) : Bool :=
  decide (claim.mIn = carrierWidth) &&
    decide (claim.x.length = carrierWidth) &&
    decide (claim.x = (encodeCarrier digest).coordinates)

theorem check_eq_true_iff
    (digest : Digest)
    (claim : Claim) :
    check digest claim = true <->
      claim = encodeClaim digest := by
  constructor
  · intro accepted
    simp only [check, Bool.and_eq_true, decide_eq_true_eq] at accepted
    rcases accepted with ⟨⟨⟨mIn, one⟩, body⟩, paddingEqual⟩
    apply Claim.eq_of_fields
    · exact mIn
    · apply Carrier.eq_of_fields
      · exact one
      · exact funext fun lane => funext fun bit =>
          body lane bit
      · exact funext fun padding =>
          paddingEqual padding
  · intro equal
    subst claim
    simp [check, encodeClaim, encodeCarrier]

/-- The complete typed plain carrier has the same equality shape as
HyperNova's recursive public-input check. -/
theorem equalityFactorization :
    EqualityFactorization check id encodeClaim := by
  intro digest claim
  exact check_eq_true_iff digest claim

theorem rawCheck_eq_true_iff
    (digest : Digest)
    (claim : RawClaim) :
    rawCheck digest claim = true <->
      claim = encodeRawClaim digest := by
  constructor
  · intro accepted
    simp only [rawCheck, Bool.and_eq_true, decide_eq_true_eq] at accepted
    rcases accepted with ⟨⟨mIn, _length⟩, coordinates⟩
    apply RawClaim.eq_of_fields
    · exact mIn
    · exact coordinates
  · intro equal
    subst claim
    simp [rawCheck, encodeRawClaim, Carrier.coordinates_length]

/-- The raw vector model is neither a digest nor authority: it accepts exactly
when it serializes one typed carrier accepted by the coordinate-level check. -/
theorem rawCheck_reduces_to_typedCarrier
    (digest : Digest)
    (raw : RawClaim) :
    rawCheck digest raw = true <->
      exists claim,
        check digest claim = true /\
          raw.mIn = claim.mIn /\
          raw.x = claim.x.coordinates := by
  constructor
  · intro accepted
    refine ⟨encodeClaim digest, ?_, ?_, ?_⟩
    · exact (check_eq_true_iff digest _).2 rfl
    · have rawEqual :=
        (rawCheck_eq_true_iff digest raw).1 accepted
      simpa [encodeRawClaim, encodeClaim] using
        congrArg RawClaim.mIn rawEqual
    · have rawEqual :=
        (rawCheck_eq_true_iff digest raw).1 accepted
      simpa [encodeRawClaim, encodeClaim] using
        congrArg RawClaim.x rawEqual
  · rintro ⟨claim, accepted, mIn, coordinates⟩
    have claimEqual :
        claim = encodeClaim digest :=
      (check_eq_true_iff digest claim).1 accepted
    subst claim
    apply (rawCheck_eq_true_iff digest raw).2
    apply RawClaim.eq_of_fields
    · exact mIn
    · exact coordinates

/-- Registered implementation refinement: the accepted 270-coordinate plain
carrier is exactly the zero completion of a logical input accepted by the
paper-owned 257-coordinate equality check. -/
theorem check_reduces_to_logicalPaperLink
    (digest : Digest)
    (claim : Claim) :
    check digest claim = true <->
      exists logical,
        CanonicalPublicInputLink.check digest logical = true /\
          claim = completeClaim logical := by
  constructor
  · intro accepted
    refine ⟨encodePublicInput digest, ?_, ?_⟩
    · exact
        (CanonicalPublicInputLink.check_eq_true_iff digest _).2 rfl
    · rw [completeClaim_encodePublicInput]
      exact (check_eq_true_iff digest claim).1 accepted
  · rintro ⟨logical, logicalAccepted, equal⟩
    have logicalEqual :
        logical = encodePublicInput digest :=
      (CanonicalPublicInputLink.check_eq_true_iff digest logical).1
        logicalAccepted
    subst logical
    rw [completeClaim_encodePublicInput] at equal
    exact (check_eq_true_iff digest claim).2 equal

end Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.CanonicalPlainCarrierLink
