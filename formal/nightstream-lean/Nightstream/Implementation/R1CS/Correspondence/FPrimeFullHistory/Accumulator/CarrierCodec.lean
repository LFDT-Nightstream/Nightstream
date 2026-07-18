import Nightstream.Protocol.FPrime.ConcretePhi81.AccumulatorBinding
import Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra.Commitment

/-!
Exact field codec for the two reduced Phi81 accumulator carriers.

Assurance tier: model-level representation refinement.

Owns: one lossless field order for the point-plus-ordered-child-commitments
carrier and the point-plus-parent-commitment carrier; injectivity of both
encoders; a length-checked right inverse for flat commitment fields; exact
fixed-profile payload lengths; and generic hash-scheme instantiations whose
serialization-collision branch is impossible.

Does not own: a domain tag, Poseidon2 padding or collision resistance, Ajtai
opening binding, canonical-child extraction, Rust serialization, R1CS wires,
constraint costs, or row removal.

Emits constraints: no.

Authority boundary: dimensions and arity are type-level verifier inputs and
are not redundantly serialized. The fixed-profile formulas use the independent
Phi81 parameters `k = 14`, commitment width `18`, and ring degree `54`. They
count carrier fields only; a concrete hash preimage must add and bind its own
domain separation.

| Stage path | Mathematical obligation | Authority class | Field order / formula | Lean owner |
|---|---|---|---|---|
| `fprime.accumulator.codec.point` | retain every extension-field point coordinate | computed | coordinate-major `(c0, c1)`; `2 * rowVariables` | `encodePoint_injective` |
| `fprime.accumulator.codec.commitment` | retain every Ajtai commitment coefficient | computed | row-major, then coefficient-major; `verifierRows * 54` | `encodeCommitment_injective` |
| `fprime.accumulator.codec.commitment.decode` | checked flat fields round-trip without padding or truncation | derived | requires exact `verifierRows * 54` length | `encodeCommitment_decodeCommitmentOfLength` |
| `fprime.accumulator.codec.children` | retain exact child index order | computed | child-major commitment blocks | `encodeChildren_injective` |
| `fprime.accumulator.codec.commitment_family` | losslessly encode point plus fourteen child commitments | computed | `2 * rowVariables + 13_608` | `encodeCommitmentFamily_injective` |
| `fprime.accumulator.codec.canonical_parent` | losslessly encode point plus one parent commitment | computed | `2 * rowVariables + 972` | `encodeCanonicalParent_injective` |
| `fprime.accumulator.codec.scheme` | instantiate the generic domain-separated hash interface without an encoding failure | derived | exact encoder above | `commitmentFamilyScheme_no_encodingCollision`, `canonicalParentScheme_no_encodingCollision` |
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistory.Accumulator.CarrierCodec

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.CanonicalParentAuthority
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.ChildCommitmentAuthority

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

private theorem ofFn_injective
    {Item : Type}
    {count : Nat} :
    Function.Injective (List.ofFn : (Fin count -> Item) -> List Item) := by
  intro left right same
  funext index
  have reads := congrArg
    (fun values => values.getD index.val (left index)) same
  simpa [getD_ofFn] using reads

private def encodeBlocks {Item : Type}
    (encode : Item -> List F) : List Item -> List F
  | [] => []
  | item :: items => encode item ++ encodeBlocks encode items

private theorem encodeBlocks_eq_flatten_map
    {Item : Type}
    (encode : Item -> List F)
    (items : List Item) :
    encodeBlocks encode items = (items.map encode).flatten := by
  induction items with
  | nil => rfl
  | cons item items inductionHypothesis =>
      simp [encodeBlocks, inductionHypothesis]

private theorem encodeBlocks_length
    {Item : Type}
    (encode : Item -> List F)
    (blockWidth : Nat)
    (blockLength : forall item, (encode item).length = blockWidth)
    (items : List Item) :
    (encodeBlocks encode items).length = items.length * blockWidth := by
  induction items with
  | nil => simp [encodeBlocks]
  | cons item items inductionHypothesis =>
      simp [encodeBlocks, blockLength, inductionHypothesis, Nat.succ_mul,
        Nat.add_comm]

private theorem encodeBlocks_injective
    {Item : Type}
    (encode : Item -> List F)
    (blockWidth : Nat)
    (blockWidthPositive : 0 < blockWidth)
    (blockLength : forall item, (encode item).length = blockWidth)
    (blockInjective : Function.Injective encode) :
    Function.Injective (encodeBlocks encode) := by
  intro left
  induction left with
  | nil =>
      intro right same
      cases right with
      | nil => rfl
      | cons item items =>
          have lengths := congrArg List.length same
          rw [encodeBlocks_length encode blockWidth blockLength []] at lengths
          rw [encodeBlocks_length encode blockWidth blockLength
            (item :: items)] at lengths
          simp only [List.length_nil, Nat.zero_mul, List.length_cons,
            Nat.succ_mul] at lengths
          omega
  | cons leftHead leftTail inductionHypothesis =>
      intro right same
      cases right with
      | nil =>
          have lengths := congrArg List.length same
          rw [encodeBlocks_length encode blockWidth blockLength
            (leftHead :: leftTail)] at lengths
          rw [encodeBlocks_length encode blockWidth blockLength []] at lengths
          simp only [List.length_cons, Nat.succ_mul, List.length_nil,
            Nat.zero_mul] at lengths
          omega
      | cons rightHead rightTail =>
          have heads := congrArg (List.take blockWidth) same
          have tails := congrArg (List.drop blockWidth) same
          simp only [encodeBlocks] at heads tails
          have headFields : encode leftHead = encode rightHead := by
            simpa [blockLength] using heads
          have tailFields :
              encodeBlocks encode leftTail = encodeBlocks encode rightTail := by
            simpa [blockLength] using tails
          cases blockInjective headFields
          cases inductionHypothesis tailFields
          rfl

/-! ## Point fields -/

def encodeK (value : K) : List F :=
  [value.c0, value.c1]

@[simp] theorem encodeK_length (value : K) :
    (encodeK value).length = 2 := by
  rfl

theorem encodeK_injective : Function.Injective encodeK := by
  intro left right same
  cases left with
  | mk leftC0 leftC1 =>
      cases right with
      | mk rightC0 rightC1 =>
          have fields : leftC0 = rightC0 /\ leftC1 = rightC1 := by
            simpa [encodeK] using same
          cases fields.1
          cases fields.2
          rfl

def encodePoint {shape : Shape} (point : Point shape) : List F :=
  encodeBlocks encodeK point.coordinates

def pointFieldCount (shape : Shape) : Nat :=
  2 * shape.rowVariables

theorem encodePoint_eq_flatten_map {shape : Shape} (point : Point shape) :
    encodePoint point = (point.coordinates.map encodeK).flatten := by
  exact encodeBlocks_eq_flatten_map encodeK point.coordinates

@[simp] theorem encodePoint_length {shape : Shape} (point : Point shape) :
    (encodePoint point).length = pointFieldCount shape := by
  rw [encodePoint, encodeBlocks_length encodeK 2 encodeK_length]
  rw [point.dimension]
  simp [pointFieldCount, Nat.mul_comm]

theorem encodePoint_injective {shape : Shape} :
    Function.Injective (encodePoint : Point shape -> List F) := by
  intro left right same
  have coordinates : left.coordinates = right.coordinates :=
    encodeBlocks_injective encodeK 2 (by decide) encodeK_length
      encodeK_injective same
  cases left
  cases right
  cases coordinates
  rfl

/-! ## Commitment fields -/

def encodeRing (value : RingF) : List F :=
  List.ofFn value

@[simp] theorem encodeRing_length (value : RingF) :
    (encodeRing value).length = ringDegree := by
  simp [encodeRing]

theorem encodeRing_injective : Function.Injective encodeRing := by
  exact ofFn_injective

def encodeCommitment {verifierRows : Nat}
    (commitment : Commitment.Value verifierRows) : List F :=
  encodeBlocks encodeRing (List.ofFn commitment)

def commitmentFieldCount (verifierRows : Nat) : Nat :=
  verifierRows * ringDegree

@[simp] theorem encodeCommitment_length {verifierRows : Nat}
    (commitment : Commitment.Value verifierRows) :
    (encodeCommitment commitment).length =
      commitmentFieldCount verifierRows := by
  rw [encodeCommitment,
    encodeBlocks_length encodeRing ringDegree encodeRing_length]
  simp [commitmentFieldCount]

theorem encodeCommitment_injective {verifierRows : Nat} :
    Function.Injective
      (encodeCommitment : Commitment.Value verifierRows -> List F) := by
  intro left right same
  apply ofFn_injective
  exact encodeBlocks_injective encodeRing ringDegree (by decide)
    encodeRing_length encodeRing_injective same

/-- Row-major index of one coefficient in a flat commitment carrier. -/
def commitmentIndex {verifierRows : Nat}
    (row : Fin verifierRows) (coefficient : Fin ringDegree) :
    Fin (verifierRows * ringDegree) :=
  ⟨row.val * ringDegree + coefficient.val, by
    have rowNext : row.val + 1 <= verifierRows :=
      Nat.succ_le_of_lt row.isLt
    have coefficientLt := coefficient.isLt
    have scaled : (row.val + 1) * ringDegree <=
        verifierRows * ringDegree :=
      Nat.mul_le_mul_right ringDegree rowNext
    exact Nat.lt_of_lt_of_le
      (show row.val * ringDegree + coefficient.val <
          row.val * ringDegree + ringDegree from
        Nat.add_lt_add_left coefficientLt _)
      (by simpa [Nat.add_mul] using scaled)⟩

/-- Total typed decoder once the exact flat field count has been checked. -/
def decodeCommitmentOfLength {verifierRows : Nat}
    (fields : List F)
    (length : fields.length = verifierRows * ringDegree) :
    Commitment.Value verifierRows :=
  fun row coefficient =>
    fields.get (Fin.cast length.symm (commitmentIndex row coefficient))

/-- The checked row-major decoder is a right inverse of the canonical
commitment encoder. -/
theorem encodeCommitment_decodeCommitmentOfLength
    {verifierRows : Nat}
    (fields : List F)
    (length : fields.length = verifierRows * ringDegree) :
    encodeCommitment (decodeCommitmentOfLength fields length) = fields := by
  apply List.ext_get
  · simp [commitmentFieldCount, length]
  · intro index leftLt rightLt
    have indexLt : index < verifierRows * ringDegree := by
      simpa [commitmentFieldCount] using leftLt
    let row : Fin verifierRows :=
      ⟨index / ringDegree,
        (Nat.div_lt_iff_lt_mul (by decide : 0 < ringDegree)).2 indexLt⟩
    let coefficient : Fin ringDegree :=
      ⟨index % ringDegree, Nat.mod_lt _ (by decide)⟩
    have indexEq :
        row.val * ringDegree + coefficient.val = index := by
      simpa [row, coefficient, Nat.mul_comm] using
        Nat.div_add_mod index ringDegree
    have decodedAt := getD_flatten_ofFn_ofFn
      (decodeCommitmentOfLength fields length) row coefficient 0
    rw [indexEq] at decodedAt
    calc
      (encodeCommitment
          (decodeCommitmentOfLength fields length))[index] =
          (encodeCommitment
            (decodeCommitmentOfLength fields length)).getD index 0 := by
        rw [← List.getElem_eq_getD]
      _ = decodeCommitmentOfLength fields length row coefficient :=
        by
          simpa [encodeCommitment, encodeBlocks_eq_flatten_map,
            List.map_ofFn, encodeRing] using decodedAt
      _ = fields[index] := by
        simp [decodeCommitmentOfLength, commitmentIndex, row, coefficient,
          indexEq]

private def encodeCommitmentList {verifierRows : Nat}
    (commitments : List (Commitment.Value verifierRows)) : List F :=
  encodeBlocks encodeCommitment commitments

private theorem encodeCommitmentList_length {verifierRows : Nat}
    (commitments : List (Commitment.Value verifierRows)) :
    (encodeCommitmentList commitments).length =
      commitments.length * commitmentFieldCount verifierRows := by
  exact encodeBlocks_length encodeCommitment
    (commitmentFieldCount verifierRows) encodeCommitment_length commitments

def encodeChildren {verifierRows count : Nat}
    (children : Fin count -> Commitment.Value verifierRows) : List F :=
  encodeCommitmentList (List.ofFn children)

@[simp] theorem encodeChildren_length {verifierRows count : Nat}
    (children : Fin count -> Commitment.Value verifierRows) :
    (encodeChildren children).length =
      count * commitmentFieldCount verifierRows := by
  simp [encodeChildren, encodeCommitmentList_length]

theorem encodeChildren_injective {verifierRows count : Nat} :
    Function.Injective
      (encodeChildren :
        (Fin count -> Commitment.Value verifierRows) -> List F) := by
  cases verifierRows with
  | zero =>
      intro left right _same
      funext child row
      exact Fin.elim0 row
  | succ rows =>
      intro left right same
      apply ofFn_injective
      exact encodeBlocks_injective encodeCommitment
        (commitmentFieldCount (rows + 1)) (by
          simp [commitmentFieldCount, ringDegree])
        encodeCommitment_length encodeCommitment_injective same

/-- Pointwise checked commitment decoding preserves child order and exposes
the exact concatenated flat fields. -/
theorem encodeChildren_decodeCommitmentOfLength
    {verifierRows count : Nat}
    (fields : Fin count -> List F)
    (length : forall child,
      (fields child).length = verifierRows * ringDegree) :
    encodeChildren
        (fun child => decodeCommitmentOfLength (fields child) (length child)) =
      (List.ofFn fields).flatten := by
  rw [encodeChildren, encodeCommitmentList, encodeBlocks_eq_flatten_map,
    List.map_ofFn]
  apply congrArg List.flatten
  apply congrArg List.ofFn
  funext child
  exact encodeCommitment_decodeCommitmentOfLength
    (fields child) (length child)

/-! ## Reduced carrier encoders -/

def encodeCommitmentFamily
    {shape : Shape}
    {verifierRows count : Nat}
    (payload :
      CommitmentFamilyPayload shape (Commitment.Value verifierRows) count) :
    List F :=
  encodePoint payload.point ++ encodeChildren payload.children

def commitmentFamilyFieldCount
    (shape : Shape) (verifierRows count : Nat) : Nat :=
  pointFieldCount shape + count * commitmentFieldCount verifierRows

@[simp] theorem encodeCommitmentFamily_length
    {shape : Shape}
    {verifierRows count : Nat}
    (payload :
      CommitmentFamilyPayload shape (Commitment.Value verifierRows) count) :
    (encodeCommitmentFamily payload).length =
      commitmentFamilyFieldCount shape verifierRows count := by
  simp [encodeCommitmentFamily, commitmentFamilyFieldCount]

theorem encodeCommitmentFamily_injective
    {shape : Shape}
    {verifierRows count : Nat} :
    Function.Injective
      (encodeCommitmentFamily :
        CommitmentFamilyPayload shape (Commitment.Value verifierRows) count ->
          List F) := by
  intro left right same
  have pointFields := congrArg (List.take (pointFieldCount shape)) same
  have childFields := congrArg (List.drop (pointFieldCount shape)) same
  have pointEq : left.point = right.point := by
    apply encodePoint_injective
    simpa [encodeCommitmentFamily, encodePoint_length] using pointFields
  have childrenEq : left.children = right.children := by
    apply encodeChildren_injective
    simpa [encodeCommitmentFamily, encodePoint_length] using childFields
  cases left
  cases right
  cases pointEq
  cases childrenEq
  rfl

def encodeCanonicalParent
    {shape : Shape}
    {verifierRows : Nat}
    (payload : CanonicalParentPayload shape (Commitment.Value verifierRows)) :
    List F :=
  encodePoint payload.point ++ encodeCommitment payload.commitment

def canonicalParentFieldCount
    (shape : Shape) (verifierRows : Nat) : Nat :=
  pointFieldCount shape + commitmentFieldCount verifierRows

@[simp] theorem encodeCanonicalParent_length
    {shape : Shape}
    {verifierRows : Nat}
    (payload : CanonicalParentPayload shape (Commitment.Value verifierRows)) :
    (encodeCanonicalParent payload).length =
      canonicalParentFieldCount shape verifierRows := by
  simp [encodeCanonicalParent, canonicalParentFieldCount]

theorem encodeCanonicalParent_injective
    {shape : Shape}
    {verifierRows : Nat} :
    Function.Injective
      (encodeCanonicalParent :
        CanonicalParentPayload shape (Commitment.Value verifierRows) ->
          List F) := by
  intro left right same
  have pointFields := congrArg (List.take (pointFieldCount shape)) same
  have commitmentFields :=
    congrArg (List.drop (pointFieldCount shape)) same
  have pointEq : left.point = right.point := by
    apply encodePoint_injective
    simpa [encodeCanonicalParent, encodePoint_length] using pointFields
  have commitmentEq : left.commitment = right.commitment := by
    apply encodeCommitment_injective
    simpa [encodeCanonicalParent, encodePoint_length] using commitmentFields
  cases left
  cases right
  cases pointEq
  cases commitmentEq
  rfl

/-! ## Generic binding-scheme instantiations -/

open Nightstream.Protocol.FPrime.AccumulatorBinding

/-- The hash remains a caller-supplied, constructor-domain-separated
Poseidon2 boundary. This definition fixes only the claim encoder. -/
def commitmentFamilyScheme
    {shape : Shape}
    {verifierRows count : Nat}
    {Digest : Type}
    (hash : Message (List F) Digest -> Digest) :
    Scheme
      (CommitmentFamilyPayload shape (Commitment.Value verifierRows) count)
      (List F) Digest where
  encodeClaim := encodeCommitmentFamily
  hash := hash

/-- The hash remains a caller-supplied, constructor-domain-separated
Poseidon2 boundary. This definition fixes only the claim encoder. -/
def canonicalParentScheme
    {shape : Shape}
    {verifierRows : Nat}
    {Digest : Type}
    (hash : Message (List F) Digest -> Digest) :
    Scheme
      (CanonicalParentPayload shape (Commitment.Value verifierRows))
      (List F) Digest where
  encodeClaim := encodeCanonicalParent
  hash := hash

theorem commitmentFamilyScheme_no_encodingCollision
    {shape : Shape}
    {verifierRows count : Nat}
    {Digest : Type}
    (hash : Message (List F) Digest -> Digest) :
    ¬ EncodingCollision
      (commitmentFamilyScheme
        (shape := shape) (verifierRows := verifierRows) (count := count)
        hash) := by
  intro collision
  rcases collision with ⟨left, right, different, sameEncoding⟩
  exact different (encodeCommitmentFamily_injective sameEncoding)

theorem canonicalParentScheme_no_encodingCollision
    {shape : Shape}
    {verifierRows : Nat}
    {Digest : Type}
    (hash : Message (List F) Digest -> Digest) :
    ¬ EncodingCollision
      (canonicalParentScheme
        (shape := shape) (verifierRows := verifierRows) hash) := by
  intro collision
  rcases collision with ⟨left, right, different, sameEncoding⟩
  exact different (encodeCanonicalParent_injective sameEncoding)

/-- Equal claim digests for the ordered-child carrier reduce directly to
exact payload equality or a hash collision; serialization cannot explain the
equality. -/
theorem commitmentFamily_claim_eq_or_hashCollision
    {shape : Shape}
    {verifierRows count : Nat}
    {Digest : Type}
    (hash : Message (List F) Digest -> Digest)
    (left right :
      CommitmentFamilyPayload shape (Commitment.Value verifierRows) count)
    (sameDigest :
      claimDigest (commitmentFamilyScheme hash) left =
        claimDigest (commitmentFamilyScheme hash) right) :
    left = right \/ HashCollision
      (commitmentFamilyScheme
        (shape := shape) (verifierRows := verifierRows) (count := count)
        hash) := by
  rcases claim_eq_or_failure (commitmentFamilyScheme hash)
      left right sameDigest with exactClaim | failure
  · exact Or.inl exactClaim
  · cases failure with
    | encoding collision =>
        exact False.elim
          (commitmentFamilyScheme_no_encodingCollision hash collision)
    | hash collision => exact Or.inr collision

/-- Equal claim digests for the canonical-parent carrier reduce directly to
exact payload equality or a hash collision; serialization cannot explain the
equality. -/
theorem canonicalParent_claim_eq_or_hashCollision
    {shape : Shape}
    {verifierRows : Nat}
    {Digest : Type}
    (hash : Message (List F) Digest -> Digest)
    (left right :
      CanonicalParentPayload shape (Commitment.Value verifierRows))
    (sameDigest :
      claimDigest (canonicalParentScheme hash) left =
        claimDigest (canonicalParentScheme hash) right) :
    left = right \/ HashCollision
      (canonicalParentScheme
        (shape := shape) (verifierRows := verifierRows) hash) := by
  rcases claim_eq_or_failure (canonicalParentScheme hash)
      left right sameDigest with exactClaim | failure
  · exact Or.inl exactClaim
  · cases failure with
    | encoding collision =>
        exact False.elim
          (canonicalParentScheme_no_encodingCollision hash collision)
    | hash collision => exact Or.inr collision

/-! ## Fixed Phi81 profile counts -/

abbrev FixedCommitment :=
  Commitment.Value productionProfile.commitmentWidth

abbrev FixedCommitmentFamilyPayload (shape : Shape) :=
  CommitmentFamilyPayload shape FixedCommitment productionGlobalParams.k

abbrev FixedCanonicalParentPayload (shape : Shape) :=
  CanonicalParentPayload shape FixedCommitment

theorem fixed_commitment_field_count :
    commitmentFieldCount productionProfile.commitmentWidth = 972 := by
  decide

theorem fixed_commitment_family_field_count (shape : Shape) :
    commitmentFamilyFieldCount shape productionProfile.commitmentWidth
      productionGlobalParams.k = 2 * shape.rowVariables + 13608 := by
  simp [commitmentFamilyFieldCount, pointFieldCount, commitmentFieldCount,
    productionProfile, productionGlobalParams, ringDegree]

theorem fixed_canonical_parent_field_count (shape : Shape) :
    canonicalParentFieldCount shape productionProfile.commitmentWidth =
      2 * shape.rowVariables + 972 := by
  simp [canonicalParentFieldCount, pointFieldCount, commitmentFieldCount,
    productionProfile, ringDegree]

end Nightstream.Implementation.R1CS.FPrimeFullHistory.Accumulator.CarrierCodec
