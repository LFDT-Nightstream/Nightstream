import SuperNeo.FPrimeRecursiveVerifier.PiRlcAlgebra.Semantics.ProjectionBindingShape

/-!
Owns: exact version-one field framing and serialized preimage lengths for the
fixed Pi_RLC projection-binding profiles.

Does not own: semantic carrier authority, a generated fixed-F-prime Rust
conformance artifact, SIS binding security, Poseidon2 refinement, or R1CS rows.

Emits constraints: no. This file lowers typed semantic family shapes into exact
serializer lengths.

Authority boundary: serialization length is accounting only. It does not make
the resulting digest authoritative and does not permit row removal.

| Predicate/theorem | Rust stage | Guarantee | Assumptions | Permits row removal? |
|---|---|---|---|---|
| `plainFixedProfile_serialized_length` | `nifs.pi_rlc.verify.projection_binding.{domain,combined,quotient,sis_digest}` | The modeled plain profile serializes to exactly 3,616 fields in production family order | `PlainFixedProfileShape`; Rust conformance artifact required | No |
| `counterfactualAllAdvSameXProfile_serialized_length` | diagnostic same-X serializer | Adding all adv material while retaining five X lanes gives 6,889 fields | `CounterfactualAllAdvSameXProfileShape` | No - not a Nebula production profile |

| Serialized family | Fixed arity | Fields per item, including framing | Plain contribution |
|---|---:|---:|---:|
| Domain | 1 | 8 | 8 |
| Combined commitment | 1 | 6 + 972 | 978 |
| Commitment quotients | 18 | 6 + 53 | 1,062 |
| X combined/quotient pairs | 5 | 60 + 60 | 600 |
| `y_ring` limb pairs | 6 | 61 + 60 | 726 |
| `y_zcol` limb pairs | 2 | 61 + 60 | 242 |
| Plain total | | | 3,616 |
| Counterfactual all-adv leaf addition | 3 | 7 + 4 | 33 |
| Counterfactual all-adv quotient addition | 54 | 7 + 53 | 3,240 |
| Counterfactual same-X total | | | 6,889 |

The numeric results are derived from the typed semantic profile and the exact
serializer in `PiRlcChallenge.Transcript.ProjectionPrefix`. They do not use a
full-history artifact as count evidence. Nebula extends the public input and
therefore has a different X-family arity; 6,889 must not be used as its cost.
-/

namespace SuperNeo.FPrimeRecursiveVerifier.PiRlcAlgebra

open PiRlcChallenge

private theorem taggedProjectionFields_length
    (label : String) (values : List SuperNeo.F) :
    (taggedProjectionFields label values).length =
      (packedBytesWithLength (utf8Bytes label)).length + 1 +
        values.length := by
  simp [taggedProjectionFields]
  omega

private theorem taggedFamily_length
    (label : String) (fieldLength : Nat)
    {families : List (List SuperNeo.F)}
    (shape : families.Forall (fun fields => fields.length = fieldLength)) :
    (families.flatMap (taggedProjectionFields label)).length =
      families.length *
        ((packedBytesWithLength (utf8Bytes label)).length + 1 +
          fieldLength) := by
  induction families with
  | nil => simp
  | cons head tail inductionHypothesis =>
      simp only [List.forall_cons] at shape
      rw [List.flatMap_cons, List.length_append,
        taggedProjectionFields_length,
        shape.1, inductionHypothesis shape.2]
      simp only [List.length_cons, Nat.add_mul, one_mul]
      omega

private theorem laneFamily_length
    (combinedLabel quotientLabel : String)
    {bindings : List ProjectionLaneBinding}
    (shape : bindings.Forall ProjectionLaneShape) :
    (serializeLaneBindings combinedLabel quotientLabel bindings).length =
      bindings.length *
        ((packedBytesWithLength (utf8Bytes combinedLabel)).length + 1 +
            SuperNeo.d +
          ((packedBytesWithLength (utf8Bytes quotientLabel)).length + 1 +
            (SuperNeo.d - 1))) := by
  induction bindings with
  | nil => simp [serializeLaneBindings]
  | cons head tail inductionHypothesis =>
      simp only [List.forall_cons] at shape
      rw [show serializeLaneBindings combinedLabel quotientLabel
          (head :: tail) =
        taggedProjectionFields combinedLabel head.combined ++
          taggedProjectionFields quotientLabel head.quotient ++
          serializeLaneBindings combinedLabel quotientLabel tail by
            rfl]
      rw [List.length_append, List.length_append,
        taggedProjectionFields_length, taggedProjectionFields_length,
        shape.1.combined, shape.1.quotient,
        inductionHypothesis shape.2]
      simp only [List.length_cons, Nat.add_mul, one_mul]
      omega

/-!
These small concrete string reductions intentionally use kernel evaluation,
not `native_decide`; the exported count theorems therefore do not trust the
native compiler.
-/

private theorem byteArrayLoop_length
    (bytes : ByteArray) (index : Nat) (accumulator : List UInt8) :
    (ByteArray.toList.loop bytes index accumulator).length =
      accumulator.length + (bytes.size - index) := by
  fun_induction ByteArray.toList.loop bytes index accumulator with
  | case1 index accumulator index_lt inductionHypothesis =>
      rw [inductionHypothesis]
      simp only [List.length_cons]
      omega
  | case2 index accumulator index_ge =>
      rw [List.length_reverse]
      omega

private theorem byteArrayToList_length (bytes : ByteArray) :
    bytes.toList.length = bytes.size := by
  rw [ByteArray.toList, byteArrayLoop_length]
  simp

private theorem utf8Bytes_length (value : String) :
    (utf8Bytes value).length = value.utf8ByteSize := by
  rw [utf8Bytes, List.length_map, byteArrayToList_length]
  exact String.size_toByteArray

private theorem packedBytesWithLength_length (bytes : List Nat) :
    (packedBytesWithLength bytes).length = 1 + (bytes.length + 6) / 7 := by
  simp [packedBytesWithLength, Nat.add_comm]

private theorem projectionDomain_length :
    (packedBytesWithLength (utf8Bytes projectionBindingDomain)).length = 8 := by
  rw [packedBytesWithLength_length, utf8Bytes_length]
  rfl

private theorem combinedCommitmentLabel_length :
    (packedBytesWithLength (utf8Bytes combinedCommitmentLabel)).length = 5 := by
  rw [packedBytesWithLength_length, utf8Bytes_length]
  rfl

private theorem commitmentQuotientLabel_length :
    (packedBytesWithLength (utf8Bytes commitmentQuotientLabel)).length = 5 := by
  rw [packedBytesWithLength_length, utf8Bytes_length]
  rfl

private theorem combinedAdvLabel_length :
    (packedBytesWithLength (utf8Bytes combinedAdvLabel)).length = 6 := by
  rw [packedBytesWithLength_length, utf8Bytes_length]
  rfl

private theorem advQuotientLabel_length :
    (packedBytesWithLength (utf8Bytes advQuotientLabel)).length = 6 := by
  rw [packedBytesWithLength_length, utf8Bytes_length]
  rfl

private theorem combinedXLabel_length :
    (packedBytesWithLength (utf8Bytes combinedXLabel)).length = 5 := by
  rw [packedBytesWithLength_length, utf8Bytes_length]
  rfl

private theorem xQuotientLabel_length :
    (packedBytesWithLength (utf8Bytes xQuotientLabel)).length = 6 := by
  rw [packedBytesWithLength_length, utf8Bytes_length]
  rfl

private theorem combinedYRingLabel_length :
    (packedBytesWithLength (utf8Bytes combinedYRingLabel)).length = 6 := by
  rw [packedBytesWithLength_length, utf8Bytes_length]
  rfl

private theorem yRingQuotientLabel_length :
    (packedBytesWithLength (utf8Bytes yRingQuotientLabel)).length = 6 := by
  rw [packedBytesWithLength_length, utf8Bytes_length]
  rfl

private theorem combinedYZcolLabel_length :
    (packedBytesWithLength (utf8Bytes combinedYZcolLabel)).length = 6 := by
  rw [packedBytesWithLength_length, utf8Bytes_length]
  rfl

private theorem yZcolQuotientLabel_length :
    (packedBytesWithLength (utf8Bytes yZcolQuotientLabel)).length = 6 := by
  rw [packedBytesWithLength_length, utf8Bytes_length]
  rfl

/-- The fixed plain SIS preimage contains exactly 3,616 base-field values. -/
theorem plainFixedProfile_serialized_length
    {profile : FixedProjectionProfile}
    (shape : PlainFixedProfileShape profile) :
    (serializeProjectionBinding profile.material).length = 3616 := by
  have commitmentQuotients :
      (profile.material.commitmentQuotients.flatMap
        (taggedProjectionFields commitmentQuotientLabel)).length =
        18 * (5 + 1 + 53) := by
    have quotientShape :
        profile.material.commitmentQuotients.Forall
          (fun fields => fields.length = 53) := by
      simpa [ProjectionQuotientShape, SuperNeo.d] using
        shape.families.commitment_quotients.quotients
    rw [taggedFamily_length commitmentQuotientLabel 53 quotientShape,
      plainFixedProfile_commitmentQuotientCount shape,
      commitmentQuotientLabel_length]
  have xBindings :
      (serializeLaneBindings combinedXLabel xQuotientLabel
        profile.material.x).length =
        5 * ((5 + 1 + 54) + (6 + 1 + 53)) := by
    rw [laneFamily_length combinedXLabel xQuotientLabel]
    rw [plainFixedProfile_xCount shape, combinedXLabel_length,
      xQuotientLabel_length, SuperNeo.d]
    exact shape.families.x.lanes
  have yRingBindings :
      (serializeLaneBindings combinedYRingLabel yRingQuotientLabel
        profile.material.yRing).length =
        6 * ((6 + 1 + 54) + (6 + 1 + 53)) := by
    rw [laneFamily_length combinedYRingLabel yRingQuotientLabel]
    rw [plainFixedProfile_yRingCount shape, combinedYRingLabel_length,
      yRingQuotientLabel_length, SuperNeo.d]
    exact paddedProjectionLaneFamily_bindingShapes
      shape.families.y_ring
  have yZcolBindings :
      (serializeLaneBindings combinedYZcolLabel yZcolQuotientLabel
        profile.material.yZcol).length =
        2 * ((6 + 1 + 54) + (6 + 1 + 53)) := by
    rw [laneFamily_length combinedYZcolLabel yZcolQuotientLabel]
    rw [plainFixedProfile_yZcolCount shape, combinedYZcolLabel_length,
      yZcolQuotientLabel_length, SuperNeo.d]
    exact paddedProjectionLaneFamily_bindingShapes
      shape.families.y_zcol
  rw [serializeProjectionBinding, List.length_append,
    List.length_append, List.length_append, List.length_append,
    List.length_append, List.length_append, List.length_append,
    projectionDomain_length,
    taggedProjectionFields_length, combinedCommitmentLabel_length,
    shape.families.combined_commitment, commitmentQuotients,
    shape.adv.leaves, shape.adv.quotients, xBindings, yRingBindings,
    yZcolBindings]
  decide

/-!
Counterfactual only: if all three adv coordinates are added without enlarging
the plain five-lane X family, the preimage has 6,889 fields. Nebula does not
instantiate this premise.
-/
theorem counterfactualAllAdvSameXProfile_serialized_length
    {profile : FixedProjectionProfile}
    (shape : CounterfactualAllAdvSameXProfileShape profile) :
    (serializeProjectionBinding profile.material).length = 6889 := by
  have families : FixedProjectionFamiliesShape profile := shape.families
  have commitmentQuotients :
      (profile.material.commitmentQuotients.flatMap
        (taggedProjectionFields commitmentQuotientLabel)).length =
        18 * (5 + 1 + 53) := by
    have quotientShape :
        profile.material.commitmentQuotients.Forall
          (fun fields => fields.length = 53) := by
      simpa [ProjectionQuotientShape, SuperNeo.d] using
        families.commitment_quotients.quotients
    rw [taggedFamily_length commitmentQuotientLabel 53 quotientShape,
      families.commitment_quotients.count_eq, commitmentLanes,
      commitmentQuotientLabel_length]
  have advLeaves :
      (profile.material.combinedAdvLeaves.flatMap
        (taggedProjectionFields combinedAdvLabel)).length =
        3 * (6 + 1 + 4) := by
    rw [taggedFamily_length combinedAdvLabel 4 shape.adv.leaves,
      shape.adv.leaf_count, combinedAdvLabel_length]
  have advQuotients :
      (profile.material.advQuotients.flatMap
        (taggedProjectionFields advQuotientLabel)).length =
        54 * (6 + 1 + 53) := by
    have quotientShape :
        profile.material.advQuotients.Forall
          (fun fields => fields.length = 53) := by
      simpa [ProjectionQuotientShape, SuperNeo.d] using
        shape.adv.quotient_family.quotients
    rw [taggedFamily_length advQuotientLabel 53 quotientShape,
      counterfactualAllAdvSameXProfile_advQuotientCount shape,
      advQuotientLabel_length]
  have xBindings :
      (serializeLaneBindings combinedXLabel xQuotientLabel
        profile.material.x).length =
        5 * ((5 + 1 + 54) + (6 + 1 + 53)) := by
    rw [laneFamily_length combinedXLabel xQuotientLabel]
    rw [show profile.material.x.length = 5 by
      simpa [activeXColumns] using families.x.count_eq,
      combinedXLabel_length, xQuotientLabel_length, SuperNeo.d]
    exact families.x.lanes
  have yRingBindings :
      (serializeLaneBindings combinedYRingLabel yRingQuotientLabel
        profile.material.yRing).length =
        6 * ((6 + 1 + 54) + (6 + 1 + 53)) := by
    rw [laneFamily_length combinedYRingLabel yRingQuotientLabel]
    rw [show profile.material.yRing.length = 6 by
      simpa [yRingRows, extensionLimbs] using
        families.y_ring.binding_count_eq,
      combinedYRingLabel_length, yRingQuotientLabel_length, SuperNeo.d]
    exact paddedProjectionLaneFamily_bindingShapes families.y_ring
  have yZcolBindings :
      (serializeLaneBindings combinedYZcolLabel yZcolQuotientLabel
        profile.material.yZcol).length =
        2 * ((6 + 1 + 54) + (6 + 1 + 53)) := by
    rw [laneFamily_length combinedYZcolLabel yZcolQuotientLabel]
    rw [show profile.material.yZcol.length = 2 by
      simpa [extensionLimbs] using
        families.y_zcol.binding_count_eq,
      combinedYZcolLabel_length, yZcolQuotientLabel_length, SuperNeo.d]
    exact paddedProjectionLaneFamily_bindingShapes families.y_zcol
  rw [serializeProjectionBinding, List.length_append,
    List.length_append, List.length_append, List.length_append,
    List.length_append, List.length_append, List.length_append,
    projectionDomain_length,
    taggedProjectionFields_length, combinedCommitmentLabel_length,
    families.combined_commitment, commitmentQuotients, advLeaves,
    advQuotients, xBindings, yRingBindings, yZcolBindings]
  decide

end SuperNeo.FPrimeRecursiveVerifier.PiRlcAlgebra
