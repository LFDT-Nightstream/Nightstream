import Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.CanonicalPlainCarrierLink

/-!
Contract: source-shaped list semantics for the shared native plain-profile
public-link predicate.

Owns:
- the exact affine-one/body/padding list split used by the Rust helper;
- a universal equivalence between that split check and the raw canonical
  carrier checker;
- pointwise and batch reduction through the typed carrier to the logical
  HyperNova public-input equality.

Does not own: a proof about compiled Rust, translation of Rust syntax or MIR,
the optional Nebula suffix, terminal nonemptiness, R1CS rows, or NIFS.

Emits constraints: no.
-/

namespace Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.CanonicalPlainCarrierSource

open Nightstream.Implementation.Encoding.FPrime
open Nightstream.Implementation.Rust.CanonicalConformance.NativeStep
open CanonicalPlainCarrierLink

def bodyCoordinates (digest : Digest) : List Nat :=
  (List.ofFn fun lane : Fin 4 =>
    List.ofFn fun bit : Fin 64 => encodedBit digest lane bit).flatten

def paddingCoordinates : List Nat :=
  List.ofFn fun _ : Fin paddingWidth => 0

set_option maxRecDepth 8192 in
theorem bodyCoordinates_length (digest : Digest) :
    (bodyCoordinates digest).length = 256 := by
  simp [bodyCoordinates]

theorem paddingCoordinates_length :
    paddingCoordinates.length = 13 := by
  simp [paddingCoordinates, paddingWidth, carrierWidth, logicalWidth]

theorem encoded_coordinates_eq_segments (digest : Digest) :
    (encodeCarrier digest).coordinates =
      1 :: bodyCoordinates digest ++ paddingCoordinates := by
  rfl

/-- Boolean semantics of the shared Rust helper on the plain profile. Shape is
checked before the three verifier-owned segments are compared. -/
def sourceCheck (digest : Digest) (claim : RawClaim) : Bool :=
  decide (claim.mIn = carrierWidth) &&
    decide (claim.x.length = carrierWidth) &&
    match claim.x with
    | [] => false
    | one :: tail =>
        decide (one = 1) &&
          decide (tail.take 256 = bodyCoordinates digest) &&
          decide (tail.drop 256 = paddingCoordinates)

theorem sourceCheck_eq_true_iff
    (digest : Digest)
    (claim : RawClaim) :
    sourceCheck digest claim = true <->
      claim = encodeRawClaim digest := by
  cases claim with
  | mk mIn coordinates =>
      cases coordinates with
      | nil =>
          simp [sourceCheck, encodeRawClaim, encoded_coordinates_eq_segments]
      | cons one tail =>
          constructor
          · intro accepted
            simp only [sourceCheck, Bool.and_eq_true, decide_eq_true_eq] at accepted
            rcases accepted with
              ⟨⟨mInEqual, _length⟩, ⟨oneEqual, takeEqual⟩, dropEqual⟩
            apply RawClaim.eq_of_fields
            · exact mInEqual
            · simp only [encodeRawClaim, encoded_coordinates_eq_segments]
              rw [oneEqual]
              congr 1
              calc
                tail = tail.take 256 ++ tail.drop 256 :=
                  (List.take_append_drop 256 tail).symm
                _ = bodyCoordinates digest ++ paddingCoordinates := by
                  rw [takeEqual, dropEqual]
          · intro equal
            have mInEqual :
                mIn = carrierWidth :=
              congrArg RawClaim.mIn equal
            have coordinatesEqual :
                one :: tail =
                  1 :: bodyCoordinates digest ++ paddingCoordinates := by
              simpa [encodeRawClaim, encoded_coordinates_eq_segments] using
                congrArg RawClaim.x equal
            have oneEqual : one = 1 :=
              List.cons.inj coordinatesEqual |>.1
            have tailEqual :
                tail = bodyCoordinates digest ++ paddingCoordinates :=
              List.cons.inj coordinatesEqual |>.2
            subst one
            subst tail
            simp only [sourceCheck, Bool.and_eq_true, decide_eq_true_eq]
            constructor
            · constructor
              · exact mInEqual
              · simp [bodyCoordinates_length, paddingCoordinates_length,
                  carrierWidth]
            · constructor
              · exact ⟨trivial, by
                  rw [← bodyCoordinates_length digest]
                  simp⟩
              · rw [← bodyCoordinates_length digest]
                simp

theorem sourceCheck_eq_rawCheck
    (digest : Digest)
    (claim : RawClaim) :
    sourceCheck digest claim = true <->
      rawCheck digest claim = true := by
  rw [sourceCheck_eq_true_iff, rawCheck_eq_true_iff]

/-- Complete refinement chain for one plain native source value. -/
theorem sourceCheck_reduces_to_logicalPaperLink
    (digest : Digest)
    (raw : RawClaim) :
    sourceCheck digest raw = true <->
      exists typed logical,
        check digest typed = true /\
          CanonicalPublicInputLink.check digest logical = true /\
          raw.mIn = typed.mIn /\
          raw.x = typed.x.coordinates /\
          typed = completeClaim logical := by
  constructor
  · intro accepted
    refine
      ⟨encodeClaim digest, encodePublicInput digest, ?_, ?_, ?_, ?_, ?_⟩
    · exact (check_eq_true_iff digest _).2 rfl
    · exact
        (CanonicalPublicInputLink.check_eq_true_iff digest _).2 rfl
    · have rawEqual :=
        (sourceCheck_eq_true_iff digest raw).1 accepted
      simpa [encodeRawClaim, encodeClaim] using
        congrArg RawClaim.mIn rawEqual
    · have rawEqual :=
        (sourceCheck_eq_true_iff digest raw).1 accepted
      simpa [encodeRawClaim, encodeClaim] using
        congrArg RawClaim.x rawEqual
    · exact (completeClaim_encodePublicInput digest).symm
  · rintro
      ⟨typed, logical, typedAccepted, logicalAccepted, mIn, coordinates,
        completed⟩
    have typedEqual :
        typed = encodeClaim digest :=
      (check_eq_true_iff digest typed).1 typedAccepted
    have logicalEqual :
        logical = encodePublicInput digest :=
      (CanonicalPublicInputLink.check_eq_true_iff digest logical).1
        logicalAccepted
    subst typed
    subst logical
    apply (sourceCheck_eq_true_iff digest raw).2
    apply RawClaim.eq_of_fields
    · exact mIn
    · exact coordinates

def sourceBatchCheck (digest : Digest) (claims : List RawClaim) : Bool :=
  claims.all (sourceCheck digest)

theorem sourceBatchCheck_reduces_to_logicalPaperLink
    (digest : Digest)
    (claims : List RawClaim) :
    sourceBatchCheck digest claims = true <->
      forall raw, raw ∈ claims ->
        exists typed logical,
          check digest typed = true /\
            CanonicalPublicInputLink.check digest logical = true /\
            raw.mIn = typed.mIn /\
            raw.x = typed.x.coordinates /\
            typed = completeClaim logical := by
  rw [sourceBatchCheck, List.all_eq_true]
  constructor
  · intro accepted raw member
    exact
      (sourceCheck_reduces_to_logicalPaperLink digest raw).1
        (accepted raw member)
  · intro reduced raw member
    exact
      (sourceCheck_reduces_to_logicalPaperLink digest raw).2
        (reduced raw member)

end Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.CanonicalPlainCarrierSource
