import Nightstream.Implementation.Encoding.FPrime
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls.AffineMap
import Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.FixedOneCanonicalAdapter

/-!
Contract: concrete Goldilocks codecs for the production four-lane F-prime
digest and the compact fixed-one adapter encoding.

Assurance tier: model-level.

Owns:
- the exact four-lane digest coordinate order;
- a canonical presence-tagged optional digest with zeroed absent payload;
- the adapter encoding `(optional digest, linked bit)`;
- the exact affine coordinate map for the adapter's `encodeInstance`.

Does not own: byte serialization, Rust memory layout, state/fresh/witness
codecs, the nonlinear `freshPublic` map, physical rows, generated artifacts,
or compiled-Rust semantics.

Emits constraints: no.
-/

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionDigestCodecs

open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls
open Nightstream.Implementation.Lowering.Goldilocks

namespace Native

abbrev Digest := Nightstream.Implementation.Encoding.FPrime.Digest

abbrev Encoded :=
  Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.FixedOneCanonicalAdapter.Encoded
    Digest

end Native

private def index0 : Fin 4 := ⟨0, by decide⟩
private def index1 : Fin 4 := ⟨1, by decide⟩
private def index2 : Fin 4 := ⟨2, by decide⟩
private def index3 : Fin 4 := ⟨3, by decide⟩

private def fieldOfLane
    (lane : Nightstream.Implementation.Encoding.FPrime.Lane) : Field :=
  ⟨lane.val, by
    simpa [
      Nightstream.Implementation.R1CS.goldilocksP,
      Nightstream.SuperNeo.Concrete.goldilocksModulus
    ] using lane.2⟩

private def laneOfField
    (field : Field) : Nightstream.Implementation.Encoding.FPrime.Lane :=
  ⟨field.val, field.isLt⟩

@[simp] private theorem fieldOfLane_laneOfField (field : Field) :
    fieldOfLane (laneOfField field) = field := by
  apply Fin.ext
  rfl

@[simp] private theorem laneOfField_fieldOfLane
    (lane : Nightstream.Implementation.Encoding.FPrime.Lane) :
    laneOfField (fieldOfLane lane) = lane := by
  apply Subtype.ext
  rfl

private theorem fin4_cases (index : Fin 4) :
    index = index0 ∨ index = index1 ∨ index = index2 ∨ index = index3 := by
  rcases index with ⟨value, bound⟩
  simp only [index0, index1, index2, index3, Fin.mk.injEq]
  omega

private def digestOfFields
    (first second third fourth : Field) : Native.Digest :=
  fun index =>
    if index = index0 then laneOfField first
    else if index = index1 then laneOfField second
    else if index = index2 then laneOfField third
    else laneOfField fourth

@[simp] private theorem digestOfFields_index0
    (first second third fourth : Field) :
    digestOfFields first second third fourth index0 = laneOfField first := by
  simp [digestOfFields]

@[simp] private theorem digestOfFields_index1
    (first second third fourth : Field) :
    digestOfFields first second third fourth index1 = laneOfField second := by
  simp [digestOfFields, index0, index1]

@[simp] private theorem digestOfFields_index2
    (first second third fourth : Field) :
    digestOfFields first second third fourth index2 = laneOfField third := by
  simp [digestOfFields, index0, index1, index2]

@[simp] private theorem digestOfFields_index3
    (first second third fourth : Field) :
    digestOfFields first second third fourth index3 = laneOfField fourth := by
  simp [digestOfFields, index0, index1, index2, index3]

private def encodeDigest (digest : Native.Digest) : List Field :=
  [ fieldOfLane (digest index0)
  , fieldOfLane (digest index1)
  , fieldOfLane (digest index2)
  , fieldOfLane (digest index3)
  ]

private def decodeDigest : List Field -> Option Native.Digest
  | [first, second, third, fourth] =>
      some (digestOfFields first second third fourth)
  | _ => none

private theorem digestOfEncodedFields (digest : Native.Digest) :
    digestOfFields
        (fieldOfLane (digest index0))
        (fieldOfLane (digest index1))
        (fieldOfLane (digest index2))
        (fieldOfLane (digest index3)) =
      digest := by
  funext index
  rcases fin4_cases index with equal | equal | equal | equal
  · subst index
    simp
  · subst index
    simp
  · subst index
    simp
  · subst index
    simp

/-- Canonical production digest as four Goldilocks lanes in lane order. -/
def digestCodec : Codec Native.Digest where
  width := 4
  Admissible := fun _ => True
  encode := encodeDigest
  decode := decodeDigest
  encode_length := by
    intro digest
    rfl
  decode_encode := by
    intro digest admissible
    simp only [encodeDigest, decodeDigest, Option.some.injEq]
    exact digestOfEncodedFields digest
  encode_decode := by
    intro coordinates digest decoded
    cases coordinates with
    | nil =>
        simp [decodeDigest] at decoded
    | cons first tail =>
        cases tail with
        | nil =>
            simp [decodeDigest] at decoded
        | cons second tail =>
            cases tail with
            | nil =>
                simp [decodeDigest] at decoded
            | cons third tail =>
                cases tail with
                | nil =>
                    simp [decodeDigest] at decoded
                | cons fourth tail =>
                    cases tail with
                    | nil =>
                        simp only [decodeDigest, Option.some.injEq] at decoded
                        subst digest
                        exact ⟨True.intro, by
                          simp [encodeDigest]⟩
                    | cons fifth tail =>
                        simp [decodeDigest] at decoded

/-- One production digest lane interpreted as its identical Goldilocks field
coordinate. -/
def laneCoordinate
    (lane : Nightstream.Implementation.Encoding.FPrime.Lane) : Field :=
  fieldOfLane lane

/-- The digest codec owns the exact Rust `[F; 4]` lane order. -/
theorem digestCodec_encode_exact (digest : Native.Digest) :
    digestCodec.encode digest =
      [ laneCoordinate (digest 0)
      , laneCoordinate (digest 1)
      , laneCoordinate (digest 2)
      , laneCoordinate (digest 3)
      ] := by
  rfl

/-- Every canonical four-lane production digest round-trips. -/
theorem digestCodec_roundtrip (digest : Native.Digest) :
    digestCodec.decode (digestCodec.encode digest) = some digest :=
  digestCodec.decode_encode digest True.intro

private def encodeOptionalDigest : Option Native.Digest -> List Field
  | none => [0, 0, 0, 0, 0]
  | some digest => 1 :: digestCodec.encode digest

private def decodeOptionalDigest : List Field -> Option (Option Native.Digest)
  | [present, first, second, third, fourth] =>
      if present = 0 then
        if first = 0 ∧ second = 0 ∧ third = 0 ∧ fourth = 0 then
          some none
        else
          none
      else if present = 1 then
        some (some (digestOfFields first second third fourth))
      else
        none
  | _ => none

private theorem field_zero_ne_one : (0 : Field) ≠ 1 := by
  decide

private theorem field_one_ne_zero : (1 : Field) ≠ 0 := by
  decide

private theorem goldilocksModulus_ne_one :
    Nightstream.SuperNeo.Concrete.goldilocksModulus ≠ 1 := by
  decide

private theorem decodeOptionalDigest_present
    (first second third fourth : Field) :
    decodeOptionalDigest [1, first, second, third, fourth] =
      some (some (digestOfFields first second third fourth)) := by
  simp [decodeOptionalDigest, goldilocksModulus_ne_one]

/-- Canonical rejecting digest: one presence bit followed by four digest
lanes. The absent representation has an all-zero payload. -/
def optionalDigestCodec : Codec (Option Native.Digest) where
  width := 5
  Admissible := fun _ => True
  encode := encodeOptionalDigest
  decode := decodeOptionalDigest
  encode_length := by
    intro digest
    cases digest <;> rfl
  decode_encode := by
    intro digest admissible
    cases digest with
    | none =>
        simp [encodeOptionalDigest, decodeOptionalDigest]
    | some digest =>
        change
          decodeOptionalDigest
              [ 1
              , fieldOfLane (digest index0)
              , fieldOfLane (digest index1)
              , fieldOfLane (digest index2)
              , fieldOfLane (digest index3)
              ] =
            some (some digest)
        rw [decodeOptionalDigest_present]
        exact congrArg
          (fun value : Native.Digest =>
            (some (some value) : Option (Option Native.Digest)))
          (digestOfEncodedFields digest)
  encode_decode := by
    intro coordinates digest decoded
    cases coordinates with
    | nil =>
        simp [decodeOptionalDigest] at decoded
    | cons present tail =>
        cases tail with
        | nil =>
            simp [decodeOptionalDigest] at decoded
        | cons first tail =>
            cases tail with
            | nil =>
                simp [decodeOptionalDigest] at decoded
            | cons second tail =>
                cases tail with
                | nil =>
                    simp [decodeOptionalDigest] at decoded
                | cons third tail =>
                    cases tail with
                    | nil =>
                        simp [decodeOptionalDigest] at decoded
                    | cons fourth tail =>
                        cases tail with
                        | cons fifth tail =>
                            simp [decodeOptionalDigest] at decoded
                        | nil =>
                            by_cases absent : present = 0
                            · by_cases zeroPayload :
                                first = 0 ∧ second = 0 ∧
                                  third = 0 ∧ fourth = 0
                              · subst present
                                rcases zeroPayload with
                                  ⟨rfl, rfl, rfl, rfl⟩
                                have normalized :
                                    some none = some digest := by
                                  simpa [decodeOptionalDigest] using decoded
                                have digestEq : none = digest :=
                                  Option.some.inj normalized
                                cases digestEq
                                exact ⟨True.intro, rfl⟩
                              · simp [decodeOptionalDigest, absent,
                                  zeroPayload] at decoded
                            · by_cases presentOne : present = 1
                              ·
                                subst present
                                have digestEq :
                                    some
                                        (digestOfFields first second third
                                          fourth) =
                                      digest :=
                                  Option.some.inj <| by
                                    rw [decodeOptionalDigest_present] at decoded
                                    exact decoded
                                cases digestEq
                                exact ⟨True.intro, by
                                  simp [encodeOptionalDigest, digestCodec,
                                    encodeDigest]⟩
                              · simp [decodeOptionalDigest, absent,
                                  presentOne] at decoded

/-- The sole rejecting optional-digest encoding is the all-zero presence and
payload vector. -/
@[simp] theorem optionalDigestCodec_encode_none :
    optionalDigestCodec.encode none = [0, 0, 0, 0, 0] :=
  rfl

/-- An accepted optional digest has presence one followed by the exact
four-lane digest encoding. -/
@[simp] theorem optionalDigestCodec_encode_some (digest : Native.Digest) :
    optionalDigestCodec.encode (some digest) =
      1 :: digestCodec.encode digest :=
  rfl

/-- Every rejecting optional digest round-trips, including the unique
all-zero absent representation. -/
theorem optionalDigestCodec_roundtrip (digest : Option Native.Digest) :
    optionalDigestCodec.decode (optionalDigestCodec.encode digest) =
      some digest :=
  optionalDigestCodec.decode_encode digest True.intro

private theorem boolCodec_encode_true :
    boolCodec.encode true = [1] := rfl

private theorem boolCodec_decode_false :
    boolCodec.decode [0] = some false := rfl

private theorem boolCodec_decode_true :
    boolCodec.decode [1] = some true := by
  rw [← boolCodec_encode_true]
  exact boolCodec.decode_encode true True.intro

private def encodeAdapterEncoded (encoded : Native.Encoded) : List Field :=
  optionalDigestCodec.encode encoded.digest ++
    boolCodec.encode encoded.linked

private def decodeAdapterEncoded : List Field -> Option Native.Encoded
  | [present, first, second, third, fourth, linked] =>
      match
          optionalDigestCodec.decode
            [present, first, second, third, fourth],
          boolCodec.decode [linked] with
      | some digest, some linked =>
          some { digest := digest, linked := linked }
      | _, _ => none
  | _ => none

/-- Exact compact adapter encoding: canonical optional digest followed by one
canonical linked bit. -/
def adapterEncodedCodec : Codec Native.Encoded where
  width := 6
  Admissible := fun _ => True
  encode := encodeAdapterEncoded
  decode := decodeAdapterEncoded
  encode_length := by
    intro encoded
    rw [encodeAdapterEncoded, List.length_append,
      optionalDigestCodec.encode_length, boolCodec.encode_length]
    change 5 + 1 = 6
    rfl
  decode_encode := by
    intro encoded admissible
    cases encoded with
    | mk digest linked =>
        cases digest with
        | none =>
            cases linked <;>
              rfl
        | some digest =>
            have digestDecoded :=
              optionalDigestCodec.decode_encode (some digest) True.intro
            have digestEncoding :
                optionalDigestCodec.encode (some digest) =
                  [ 1
                  , fieldOfLane (digest index0)
                  , fieldOfLane (digest index1)
                  , fieldOfLane (digest index2)
                  , fieldOfLane (digest index3)
                  ] :=
              rfl
            rw [digestEncoding] at digestDecoded
            cases linked with
            | false =>
                change
                  decodeAdapterEncoded
                      ([ 1
                       , fieldOfLane (digest index0)
                       , fieldOfLane (digest index1)
                       , fieldOfLane (digest index2)
                       , fieldOfLane (digest index3)
                       ] ++ [0]) =
                    some { digest := some digest, linked := false }
                simp only [List.cons_append, List.nil_append,
                  decodeAdapterEncoded, digestDecoded,
                  boolCodec_decode_false]
            | true =>
                change
                  decodeAdapterEncoded
                      ([ 1
                       , fieldOfLane (digest index0)
                       , fieldOfLane (digest index1)
                       , fieldOfLane (digest index2)
                       , fieldOfLane (digest index3)
                       ] ++ [1]) =
                    some { digest := some digest, linked := true }
                simp only [List.cons_append, List.nil_append,
                  decodeAdapterEncoded, digestDecoded,
                  boolCodec_decode_true]
  encode_decode := by
    intro coordinates encoded decoded
    cases coordinates with
    | nil =>
        simp [decodeAdapterEncoded] at decoded
    | cons present tail =>
        cases tail with
        | nil =>
            simp [decodeAdapterEncoded] at decoded
        | cons first tail =>
            cases tail with
            | nil =>
                simp [decodeAdapterEncoded] at decoded
            | cons second tail =>
                cases tail with
                | nil =>
                    simp [decodeAdapterEncoded] at decoded
                | cons third tail =>
                    cases tail with
                    | nil =>
                        simp [decodeAdapterEncoded] at decoded
                    | cons fourth tail =>
                        cases tail with
                        | nil =>
                            simp [decodeAdapterEncoded] at decoded
                        | cons linked tail =>
                            cases tail with
                            | cons extra tail =>
                                simp [decodeAdapterEncoded] at decoded
                            | nil =>
                                generalize digestDecoded :
                                  optionalDigestCodec.decode
                                    [present, first, second, third, fourth] =
                                      digestResult
                                generalize linkedDecoded :
                                  boolCodec.decode [linked] = linkedResult
                                cases digestResult with
                                | none =>
                                    simp [decodeAdapterEncoded,
                                      digestDecoded] at decoded
                                | some digest =>
                                    cases linkedResult with
                                    | none =>
                                        simp [decodeAdapterEncoded,
                                          digestDecoded, linkedDecoded] at decoded
                                    | some linkedValue =>
                                        simp only [decodeAdapterEncoded,
                                          digestDecoded, linkedDecoded,
                                          Option.some.injEq] at decoded
                                        subst encoded
                                        have digestEncoding :=
                                          (optionalDigestCodec.encode_decode
                                            [present, first, second, third,
                                              fourth]
                                            digest digestDecoded).2
                                        have linkedEncoding :=
                                          (boolCodec.encode_decode [linked]
                                            linkedValue linkedDecoded).2
                                        exact ⟨True.intro, by
                                          simp only [encodeAdapterEncoded]
                                          rw [digestEncoding,
                                            linkedEncoding]
                                          rfl⟩

/-- Every compact adapter output round-trips through exactly six
coordinates. -/
theorem adapterEncodedCodec_roundtrip (encoded : Native.Encoded) :
    adapterEncodedCodec.decode (adapterEncodedCodec.encode encoded) =
      some encoded :=
  adapterEncodedCodec.decode_encode encoded True.intro

private def copyCoordinate : Nat -> AffineCoordinate
  | 0 => { constant := 0, coefficients := [1, 0, 0, 0, 0] }
  | 1 => { constant := 0, coefficients := [0, 1, 0, 0, 0] }
  | 2 => { constant := 0, coefficients := [0, 0, 1, 0, 0] }
  | 3 => { constant := 0, coefficients := [0, 0, 0, 1, 0] }
  | 4 => { constant := 0, coefficients := [0, 0, 0, 0, 1] }
  | _ => { constant := 0, coefficients := [0, 0, 0, 0, 0] }

private def trueCoordinate : AffineCoordinate where
  constant := 1
  coefficients := [0, 0, 0, 0, 0]

/-- The compact adapter's `encodeInstance` copies the optional digest's five
coordinates and appends the verifier-fixed true bit. -/
def encodeInstanceAffineMap :
    AffineEncodingMap optionalDigestCodec adapterEncodedCodec
      Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.FixedOneCanonicalAdapter.encodeInstance where
  coordinates :=
    [ copyCoordinate 0
    , copyCoordinate 1
    , copyCoordinate 2
    , copyCoordinate 3
    , copyCoordinate 4
    , trueCoordinate
    ]
  coordinateCount := rfl
  coefficientCounts := by
    intro coordinate member
    simp only [List.mem_cons, List.not_mem_nil, or_false] at member
    rcases member with rfl | rfl | rfl | rfl | rfl | rfl
    · rfl
    · rfl
    · rfl
    · rfl
    · rfl
    · rfl
  outputAdmissible := by
    intro digest admissible
    exact True.intro
  encode_eq := by
    intro digest admissible
    cases digest with
    | none =>
        simp [adapterEncodedCodec, encodeAdapterEncoded,
          Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.FixedOneCanonicalAdapter.encodeInstance,
          optionalDigestCodec, encodeOptionalDigest,
          copyCoordinate, trueCoordinate, AffineCoordinate.eval, fieldDot,
          boolCodec_encode_true, Fin.zero_mul, Fin.one_mul,
          Fin.add_zero]
    | some digest =>
        simp [adapterEncodedCodec, encodeAdapterEncoded,
          Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.FixedOneCanonicalAdapter.encodeInstance,
          optionalDigestCodec, encodeOptionalDigest, digestCodec, encodeDigest,
          copyCoordinate, trueCoordinate, AffineCoordinate.eval,
          fieldDot, boolCodec_encode_true, Fin.zero_mul, Fin.one_mul,
          Fin.zero_add, Fin.add_zero]

/-- Exact coordinate equation supplied to the affine direct-call recipe:
the five optional-digest coordinates are copied and the linked coordinate is
the verifier-fixed one. -/
theorem encodeInstance_coordinates_exact (digest : Option Native.Digest) :
    adapterEncodedCodec.encode
        (Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.FixedOneCanonicalAdapter.encodeInstance
          digest) =
      encodeInstanceAffineMap.coordinates.map (fun coordinate =>
        coordinate.eval (optionalDigestCodec.encode digest)) :=
  encodeInstanceAffineMap.encode_eq digest True.intro

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionDigestCodecs
