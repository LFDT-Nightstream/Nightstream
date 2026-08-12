import Nightstream.Implementation.NebulaV2.ProductionMemoryCarryRows

/-!
Contract: lossless field-native encoding of one production memory carry.

The seven bounded control values and the 52 challenge, product, and digest
limbs are emitted directly in the normative carry-tag order. The resulting
59-field image replaces the 3,433-bit carry only in a distinct field-native
production profile. It does not change the reference V2 codec.

The encoding is injective before hashing, and every canonical carry field is
a canonical Goldilocks representative. A digest of this list is compression,
not authority.

Does not own carry-transition rows, state-hash rows, absolute columns,
external bytes, Rust refinement, or a verifier key.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.NebulaV2.ProductionMemoryCarryFields

open Nightstream.Implementation.NebulaV2.MemoryCarryCodec
open Nightstream.Implementation.R1CS
open Nightstream.Protocol.NebulaV2
open Nightstream.Protocol.NebulaV2.FPrime
open Nightstream.SuperNeo.Concrete

def encodeFor (tags : List FieldTag) (value : Value) : List Nat :=
  tags.map value.fieldValue

def encode (value : Value) : List Nat := encodeFor schema value

theorem schema_length_exact : schema.length = 59 := by
  decide

theorem encode_length (value : Value) : (encode value).length = 59 := by
  simp [encode, encodeFor, schema_length_exact]

/-- The schema tag at one native carry coordinate. The cast is justified by
the protocol-owned exact 59-field schema length. -/
def tagAt (index : Fin 59) : FieldTag :=
  schema.get (Fin.cast schema_length_exact.symm index)

/-- Every native carry coordinate is exactly the matching tagged carry
value. This theorem is the structural bridge used to share memory-transition
columns with the F-prime state hash. -/
theorem encode_get (value : Value) (index : Fin 59) :
    (encode value).get (Fin.cast (encode_length value).symm index) =
      value.fieldValue (tagAt index) := by
  simp [encode, encodeFor, tagAt]

private theorem encodeFor_equal_at_member
    {left right : Value} {tags : List FieldTag}
    (equal : encodeFor tags left = encodeFor tags right)
    {tag : FieldTag} (member : tag ∈ tags) :
    left.fieldValue tag = right.fieldValue tag := by
  induction tags with
  | nil => simp at member
  | cons head tail inductionHypothesis =>
      simp only [encodeFor, List.map_cons, List.cons.injEq] at equal
      rcases List.mem_cons.mp member with tagEqual | tailMember
      · subst tag
        exact equal.1
      · exact inductionHypothesis equal.2 tailMember

/-- The complete native carry image determines the exact typed carry. -/
theorem encode_injective : Function.Injective encode := by
  intro left right equal
  apply Value.fieldValue_injective
  funext tag
  exact encodeFor_equal_at_member equal tag.mem_schema

private theorem challengeField_lt_goldilocks
    (value : Value) (repetition coordinate limb : Fin 2) :
    value.fieldValue (.challenge repetition coordinate limb) <
      goldilocksP := by
  fin_cases coordinate <;> fin_cases limb <;>
    simpa [Value.fieldValue, MemoryClaimCodec.challengeValue,
      MemoryClaimCodec.kLimbValue, goldilocksP, goldilocksModulus] using
      (if coordinate = 0 then
        (if limb = 0 then
          (value.challenges repetition).gamma1.c0.isLt
        else (value.challenges repetition).gamma1.c1.isLt)
      else
        (if limb = 0 then
          (value.challenges repetition).gamma2.c0.isLt
        else (value.challenges repetition).gamma2.c1.isLt))

private theorem productField_lt_goldilocks
    (value : Value) (repetition : Fin 2)
    (role : MemoryClaimCodec.ProductRole) (limb : Fin 2) :
    value.fieldValue (.product repetition role limb) < goldilocksP := by
  cases role <;> fin_cases limb <;>
    simp [Value.fieldValue, MemoryClaimCodec.productValue,
      MemoryClaimCodec.kLimbValue, goldilocksP, goldilocksModulus]

private theorem rootField_lt_goldilocks
    (value : Value) (source : RootSource) (lane : Fin 4) :
    value.fieldValue (.root source lane) < goldilocksP := by
  cases source with
  | memory =>
      simpa [Value.fieldValue, rootSourceValue, goldilocksP,
        ShiftedTernary41V1.modulus] using
        (value.memoryRoot.lanes lane).property
  | precommit role =>
      cases role with
      | operations =>
          simpa [Value.fieldValue, rootSourceValue,
            MemoryClaimCodec.rootValue, goldilocksP,
            ShiftedTernary41V1.modulus] using
            (value.dPre.operations.lanes lane).property
      | initialSnapshot =>
          simpa [Value.fieldValue, rootSourceValue,
            MemoryClaimCodec.rootValue, goldilocksP,
            ShiftedTernary41V1.modulus] using
            (value.dPre.initialSnapshot.lanes lane).property
      | finalSnapshot =>
          simpa [Value.fieldValue, rootSourceValue,
            MemoryClaimCodec.rootValue, goldilocksP,
            ShiftedTernary41V1.modulus] using
            (value.dPre.finalSnapshot.lanes lane).property
  | seen role =>
      cases role with
      | operations =>
          simpa [Value.fieldValue, rootSourceValue,
            MemoryClaimCodec.rootValue, goldilocksP,
            ShiftedTernary41V1.modulus] using
            (value.dSeen.operations.lanes lane).property
      | initialSnapshot =>
          simpa [Value.fieldValue, rootSourceValue,
            MemoryClaimCodec.rootValue, goldilocksP,
            ShiftedTernary41V1.modulus] using
            (value.dSeen.initialSnapshot.lanes lane).property
      | finalSnapshot =>
          simpa [Value.fieldValue, rootSourceValue,
            MemoryClaimCodec.rootValue, goldilocksP,
            ShiftedTernary41V1.modulus] using
            (value.dSeen.finalSnapshot.lanes lane).property

/-- Every field of a canonical carry is safe to absorb as one Goldilocks
field. No 64-bit integer reduction occurs. -/
theorem encode_fields_canonical
    {headers : ChainHeaders Digest.Value} {value : Value}
    (canonical : value.Canonical headers)
    {field : Nat} (member : field ∈ encode value) :
    field < goldilocksP := by
  simp only [encode, encodeFor, List.mem_map] at member
  obtain ⟨tag, _, rfl⟩ := member
  cases tag with
  | challenge repetition coordinate limb =>
      exact challengeField_lt_goldilocks value repetition coordinate limb
  | product repetition role limb =>
      exact productField_lt_goldilocks value repetition role limb
  | root source lane => exact rootField_lt_goldilocks value source lane
  | phase =>
      cases phaseExact : value.phase <;>
        simp [Value.fieldValue, phaseValue, phaseExact, goldilocksP]
  | segmentIndex =>
      exact canonical.segmentIndex.trans
        (by norm_num [MemoryWireGeometry.segmentIndexBits, goldilocksP])
  | stepIndex =>
      exact canonical.stepIndex.trans
        (by norm_num [Lifecycle.claimsPerSegment, goldilocksP])
  | globalTimestamp =>
      exact canonical.globalTimestamp.trans
        (by norm_num [MemoryWireGeometry.timestampBits, goldilocksP])
  | segmentStartTimestamp =>
      exact canonical.segmentStartTimestamp.trans
        (by norm_num [MemoryWireGeometry.timestampBits, goldilocksP])
  | segmentActiveAccessCount =>
      exact canonical.segmentActiveAccessCount.trans
        (by norm_num [MemoryWireGeometry.segmentActiveAccessCountBits,
          goldilocksP])
  | segmentEndTimestamp =>
      exact canonical.segmentEndTimestamp.trans
        (by norm_num [MemoryWireGeometry.timestampBits, goldilocksP])

end Nightstream.Implementation.NebulaV2.ProductionMemoryCarryFields
