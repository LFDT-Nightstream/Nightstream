import Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.CanonicalPlainCarrierSource

/-!
Contract: lossless serialization of the typed production plain carrier into
the source-shaped raw claim.

Assurance tier: model-level representation refinement.

Owns:
- exact reads of the affine, lane-major body, and padding coordinates;
- injectivity of the complete 270-coordinate carrier serialization;
- injectivity of typed-claim serialization, including `m_in`;
- equivalence of the typed and source-shaped executable checks on serialized
  claims.

Does not own: Rust syntax or compiled semantics, host shape validation, R1CS
rows, producer-column placement, NIFS, or the optional application suffix.

Emits constraints: no.
-/

namespace Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.CanonicalPlainCarrierSerialization

open CanonicalPlainCarrierLink
open CanonicalPlainCarrierSource

private theorem getD_ofFn
    {Item : Type}
    {count : Nat}
    (items : Fin count -> Item)
    (index : Fin count)
    (default : Item) :
    (List.ofFn items).getD index.val default = items index := by
  rw [List.getD_eq_getElem?_getD,
    List.getElem?_eq_getElem (by simp), List.getElem_ofFn]
  simp

private theorem getD_flatten_ofFn_ofFn
    {Item : Type}
    {outer inner : Nat}
    (items : Fin outer -> Fin inner -> Item)
    (outerIndex : Fin outer)
    (innerIndex : Fin inner)
    (default : Item) :
    ((List.ofFn fun outerPosition =>
        List.ofFn fun innerPosition =>
          items outerPosition innerPosition).flatten).getD
      (outerIndex.val * inner + innerIndex.val) default =
        items outerIndex innerIndex := by
  induction outer with
  | zero => exact Fin.elim0 outerIndex
  | succ outer inductionHypothesis =>
      refine Fin.cases ?_ (fun index => ?_) outerIndex
      · simp only [List.ofFn_succ, List.flatten_cons, Fin.val_zero,
          Nat.zero_mul, Nat.zero_add]
        simp only [List.getD_eq_getElem?_getD]
        rw [List.getElem?_append_left (by simp)]
        change (List.ofFn (items 0)).getD innerIndex.val default = _
        exact getD_ofFn _ innerIndex default
      · simp only [List.ofFn_succ, List.flatten_cons, Fin.val_succ]
        simp only [List.getD_eq_getElem?_getD]
        rw [List.getElem?_append_right (by
          simp only [List.length_ofFn]
          rw [Nat.add_mul, Nat.one_mul]
          omega)]
        simp only [List.length_ofFn]
        change ((List.ofFn fun outerPosition =>
          List.ofFn fun innerPosition =>
            items outerPosition.succ innerPosition).flatten).getD
          ((index.val + 1) * inner + innerIndex.val - inner) default = _
        have indexArithmetic :
            (index.val + 1) * inner + innerIndex.val - inner =
              index.val * inner + innerIndex.val := by
          rw [Nat.add_mul, Nat.one_mul]
          omega
        rw [indexArithmetic]
        exact inductionHypothesis
          (fun outerPosition innerPosition =>
            items outerPosition.succ innerPosition)
          index

set_option maxRecDepth 8192 in
private theorem bodyCoordinates_length
    (body : RawBody) :
    (List.ofFn fun lane : Fin 4 =>
      List.ofFn fun bit : Fin 64 => body lane bit).flatten.length = 256 := by
  simp

theorem Carrier.coordinates_getD_one
    (carrier : Carrier) :
    carrier.coordinates.getD 0 0 = carrier.one := by
  rfl

set_option maxRecDepth 8192 in
theorem Carrier.coordinates_getD_body
    (carrier : Carrier)
    (lane : Fin 4)
    (bit : Fin 64) :
    carrier.coordinates.getD
        (1 + lane.val * 64 + bit.val) 0 =
      carrier.body lane bit := by
  have indexEqual :
      1 + lane.val * 64 + bit.val =
        (lane.val * 64 + bit.val) + 1 := by
    omega
  rw [indexEqual]
  simp only [Carrier.coordinates, List.singleton_append,
    List.getD_eq_getElem?_getD]
  rw [List.getElem?_append_left (by
    simp only [List.length_cons]
    rw [bodyCoordinates_length]
    omega)]
  simp only [List.getElem?_cons_succ]
  simpa only [List.getD_eq_getElem?_getD] using
    getD_flatten_ofFn_ofFn carrier.body lane bit 0

theorem Carrier.coordinates_getD_padding
    (carrier : Carrier)
    (padding : Fin paddingWidth) :
    carrier.coordinates.getD (257 + padding.val) 0 =
      carrier.padding padding := by
  have indexEqual :
      257 + padding.val = (256 + padding.val) + 1 := by
    omega
  rw [indexEqual]
  simp only [Carrier.coordinates, List.singleton_append,
    List.getD_eq_getElem?_getD]
  rw [List.getElem?_append_right (by
    simp only [List.length_cons]
    rw [bodyCoordinates_length]
    omega)]
  simp only [List.length_cons]
  rw [bodyCoordinates_length]
  have subtract :
      256 + padding.val + 1 - (256 + 1) = padding.val := by
    omega
  rw [subtract]
  simpa only [List.getD_eq_getElem?_getD] using
    getD_ofFn carrier.padding padding 0

/-- Every typed carrier field is retained by the exact 270-coordinate list. -/
theorem Carrier.coordinates_injective :
    Function.Injective Carrier.coordinates := by
  intro left right equal
  apply Carrier.eq_of_fields
  · have read := congrArg (fun values => values.getD 0 0) equal
    simpa only [Carrier.coordinates_getD_one] using read
  · funext lane bit
    have read := congrArg
      (fun values =>
        values.getD (1 + lane.val * 64 + bit.val) 0)
      equal
    simpa only [Carrier.coordinates_getD_body] using read
  · funext padding
    have read := congrArg
      (fun values => values.getD (257 + padding.val) 0)
      equal
    simpa only [Carrier.coordinates_getD_padding] using read

/-- Complete raw representation of a typed claim. -/
def serializeClaim (claim : Claim) : RawClaim where
  mIn := claim.mIn
  x := claim.x.coordinates

theorem serializeClaim_encodeClaim
    (digest : Nightstream.Implementation.Encoding.FPrime.Digest) :
    serializeClaim (encodeClaim digest) = encodeRawClaim digest := by
  rfl

theorem serializeClaim_injective :
    Function.Injective serializeClaim := by
  intro left right equal
  apply Claim.eq_of_fields
  · exact congrArg RawClaim.mIn equal
  · apply Carrier.coordinates_injective
    exact congrArg RawClaim.x equal

/-- On losslessly serialized inputs, the raw source predicate and typed
coordinate predicate are extensionally identical. -/
theorem sourceCheck_serializeClaim_iff_check
    (digest : Nightstream.Implementation.Encoding.FPrime.Digest)
    (claim : Claim) :
    sourceCheck digest (serializeClaim claim) = true <->
      check digest claim = true := by
  rw [sourceCheck_eq_true_iff, check_eq_true_iff]
  constructor
  · intro equal
    apply serializeClaim_injective
    simpa only [serializeClaim_encodeClaim] using equal
  · intro equal
    subst claim
    exact serializeClaim_encodeClaim digest

end Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.CanonicalPlainCarrierSerialization
