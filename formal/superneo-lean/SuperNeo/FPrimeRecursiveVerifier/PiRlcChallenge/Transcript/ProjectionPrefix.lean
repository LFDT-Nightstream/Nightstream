import SuperNeo.FPrimeRecursiveVerifier.PiRlcChallenge.Transcript.Schedule

/-!
Owns: the ordered transcript prefix from the Pi_CCS output digest through the
Pi_RLC projection-binding digest and the two-field `beta` squeeze.

Does not own: the concrete Poseidon2 permutation, the SIS implementation or
its binding reduction, Pi_DEC verification, child-opening authority, or an
exact Rust/R1CS trace refinement.

Emits constraints: no. This file states executable prefix semantics and the
precise dependency boundary needed by the delayed NC projection argument.

Authority boundary: the Pi_RLC parent commitment, parent `y_zcol`, and all
projection quotients are serialized into the SIS input before `beta`. Pi_DEC
children are not part of this prefix. Consequently the parent material is
fixed up to a typed SIS-binding collision, while child recomposition requires
an independent opening-binding argument; transcript order alone cannot fix it.

| Predicate/theorem | Rust stage | Guarantee | Assumptions | Permits row removal? |
|---|---|---|---|---|
| `serializeProjectionBinding` | `projection_binding.{domain,combined,quotient,sis_digest}` | Exact family order of the SIS preimage | Concrete field lists supplied by caller | No - Rust serialization refinement open |
| `betaPrefixResult_eq_of_bound_digests` | Pi_CCS output bind, rho schedule, projection digest bind, `transcript_beta` | Equal incoming cursor and equal bound digests give the same complete pre-beta result | Supplied `Poseidon2Core` and SIS digest | No - concrete Poseidon2/Rust trace open |
| `parentYZcol_eq_or_typedSisCollision` | projection SIS binding | Equal binding digests fix parent `y_zcol`, or expose a typed collision | Typed material-to-digest function | No - serialization injectivity and MSIS reduction open |
| `betaPrefix_fixes_parentYZcol_or_collision` | complete pre-beta prefix | Same public output digest and projection digest fix both beta and parent `y_zcol`, modulo a typed SIS collision | Equal incoming cursor | No - Fiat-Shamir and concrete binding reductions open |
| `betaWithChildren_eq` | Pi_DEC child handoff | The model makes `beta` independent of Pi_DEC children | Model-level post-beta boundary; Rust trace refinement open | No |
| `piDecChildren_not_fixed_by_betaPrefix` | Pi_DEC child handoff | Distinct child messages have the same modeled pre-child beta prefix | Two distinct child values | No - identifies a missing refinement gate |

This is an abstract deterministic-schedule theorem, not production transcript
closure. A concrete refinement must still prove that the Rust labels, packed
bytes, SIS preimage, Poseidon2 calls, and source columns instantiate these
definitions. The typed collision must also be reduced through serialization
injectivity to the production SIS/MSIS binding event.
-/

namespace SuperNeo.FPrimeRecursiveVerifier.PiRlcChallenge

/-! ## Exact byte and field framing -/

/-- Convert a UTF-8 protocol label to canonical byte values. -/
def utf8Bytes (value : String) : List Nat :=
  value.toUTF8.toList.map UInt8.toNat

/-- One little-endian seven-byte limb, matching `pack_bytes_as_fields`. -/
def packedByteLimb (bytes : List Nat) (limb : Nat) : F :=
  F.ofNat <| (List.range 7).foldl
    (fun total offset =>
      total + bytes.getD (7 * limb + offset) 0 * 256 ^ offset) 0

/-- Packed bytes with their byte-length header. -/
def packedBytesWithLength (bytes : List Nat) : List F :=
  F.ofNat bytes.length ::
    (List.range ((bytes.length + 6) / 7)).map (packedByteLimb bytes)

/-- Native `absorb_packed_bytes_with_len`. -/
def appendPackedBytes
    (core : Poseidon2Core) (cursor : SpongeCursor) (bytes : List Nat) :
    SpongeCursor :=
  let afterLength := absorbElem core cursor (F.ofNat bytes.length)
  match (List.range ((bytes.length + 6) / 7)).map (packedByteLimb bytes) with
  | [] => afterLength
  | limbs => absorbSlice core afterLength limbs

/-- Native labeled `append_fields`: packed label, field count, then fields. -/
def appendLabeledFields
    (core : Poseidon2Core) (cursor : SpongeCursor)
    (label : String) (values : List F) : SpongeCursor :=
  appendFieldsRaw core
    (appendPackedBytes core cursor (utf8Bytes label)) values

/-- Native `append_message`: packed label followed by packed message bytes. -/
def appendMessageBytes
    (core : Poseidon2Core) (cursor : SpongeCursor)
    (label message : String) : SpongeCursor :=
  appendPackedBytes core
    (appendPackedBytes core cursor (utf8Bytes label))
    (utf8Bytes message)

/-! ## Projection-binding material -/

/-- One combined polynomial and its matching division quotient. -/
structure ProjectionLaneBinding where
  combined : List F
  quotient : List F
deriving DecidableEq, Repr

/--
Typed form of the exact projection SIS preimage families. Lists preserve the
production iteration order within each family. `yZcol` has two entries in the
fixed profile, one for each base-field coefficient of the extension field.
-/
structure ProjectionBindingMaterial where
  combinedCommitment : List F
  commitmentQuotients : List (List F)
  combinedAdvLeaves : List (List F)
  advQuotients : List (List F)
  x : List ProjectionLaneBinding
  yRing : List ProjectionLaneBinding
  yZcol : List ProjectionLaneBinding
deriving DecidableEq, Repr

def projectionBindingDomain : String :=
  "neo.fold.clean/pi_rlc/projection_binding/v1"

def combinedCommitmentLabel : String := "pi_rlc/projection_combined_c"
def commitmentQuotientLabel : String := "pi_rlc/projection_quotients"
def combinedAdvLabel : String := "pi_rlc/projection_combined_adv"
def advQuotientLabel : String := "pi_rlc/projection_adv_quotients"
def combinedXLabel : String := "pi_rlc/projection_combined_x"
def xQuotientLabel : String := "pi_rlc/projection_x_quotients"
def combinedYRingLabel : String := "pi_rlc/projection_combined_y_ring"
def yRingQuotientLabel : String := "pi_rlc/projection_y_ring_quotients"
def combinedYZcolLabel : String := "pi_rlc/projection_combined_y_zcol"
def yZcolQuotientLabel : String := "pi_rlc/projection_y_zcol_quotients"

/-- `append_projection_binding`: packed label, field count, and fields. -/
def taggedProjectionFields (label : String) (values : List F) : List F :=
  packedBytesWithLength (utf8Bytes label) ++
    F.ofNat values.length :: values

def serializeLaneBindings
    (combinedLabel quotientLabel : String)
    (bindings : List ProjectionLaneBinding) : List F :=
  bindings.flatMap fun binding =>
    taggedProjectionFields combinedLabel binding.combined ++
      taggedProjectionFields quotientLabel binding.quotient

/--
Exact family order in native `projection_schedule` and circuit
`projection/binding.rs`: domain, commitment, commitment quotients, optional
adv leaves and quotients, X pairs, y_ring pairs, then y_zcol pairs.
-/
def serializeProjectionBinding
    (material : ProjectionBindingMaterial) : List F :=
  packedBytesWithLength (utf8Bytes projectionBindingDomain) ++
    taggedProjectionFields combinedCommitmentLabel
      material.combinedCommitment ++
    material.commitmentQuotients.flatMap
      (taggedProjectionFields commitmentQuotientLabel) ++
    material.combinedAdvLeaves.flatMap
      (taggedProjectionFields combinedAdvLabel) ++
    material.advQuotients.flatMap
      (taggedProjectionFields advQuotientLabel) ++
    serializeLaneBindings combinedXLabel xQuotientLabel material.x ++
    serializeLaneBindings combinedYRingLabel yRingQuotientLabel
      material.yRing ++
    serializeLaneBindings combinedYZcolLabel yZcolQuotientLabel
      material.yZcol

/-! ## Squeeze-after-binding schedule -/

/-- Four Goldilocks lanes used by both production binding digests. -/
abbrev Digest4 := Fin 4 -> F

/-- Two Goldilocks lanes interpreted as one extension-field `beta`. -/
abbrev Beta2 := Fin 2 -> F

def digestFields (digest : Digest4) : List F :=
  List.ofFn digest

structure BetaPrefixInput where
  piCcsOutputDigest : Digest4
  projection : ProjectionBindingMaterial

structure BetaPrefixResult where
  cursor : SpongeCursor
  beta : Beta2

def piRlcInputClaimsDigestLabel : String :=
  "pi_rlc/input_claims_digest"

def piRlcProjectionBindingDigestLabel : String :=
  "pi_rlc/projection_binding_digest"

def piRlcProjectionBetaLabel : String :=
  "pi_rlc/projection_beta"

def challengeLabelDomain : String := "chal/label"

/-- One native two-field squeeze after the challenge-label message. -/
def squeezeBetaPair
    (core : Poseidon2Core) (cursor : SpongeCursor) : BetaPrefixResult :=
  let squeezed := permuteCursor core (absorbElem core cursor F.one)
  { cursor := squeezed
    beta := fun index => squeezed.state
      ⟨index.val, Nat.lt_trans index.isLt (by decide)⟩ }

/-- Native `challenge_fields("pi_rlc/projection_beta", 2)`. -/
def challengeProjectionBeta
    (core : Poseidon2Core) (cursor : SpongeCursor) : BetaPrefixResult :=
  squeezeBetaPair core <|
    appendMessageBytes core cursor challengeLabelDomain
      piRlcProjectionBetaLabel

/--
Modeled production-order cursor from the pre-Pi_RLC NIFS cursor to beta:
bind the Pi_CCS output digest, run all fifteen rho samples, bind the projection
SIS digest, then append the beta challenge label and squeeze two fields.

This definition fixes the count to fifteen. A separate relation-shape theorem
must prove that the accepted Rust relation has exactly fifteen Pi_CCS outputs.
-/
def betaPrefixResult
    (core : Poseidon2Core)
    (sisDigest : List F -> Digest4)
    (initial : SpongeCursor)
    (input : BetaPrefixInput) : BetaPrefixResult :=
  let afterOutputs := appendLabeledFields core initial
    piRlcInputClaimsDigestLabel (digestFields input.piCcsOutputDigest)
  let afterRhos := (fixedRhoSchedule core afterOutputs).cursor
  let projectionDigest := sisDigest
    (serializeProjectionBinding input.projection)
  let afterProjection := appendLabeledFields core afterRhos
    piRlcProjectionBindingDigestLabel (digestFields projectionDigest)
  challengeProjectionBeta core afterProjection

/-- The result is a pure function of the complete pre-beta input. -/
theorem betaPrefixResult_deterministic
    (core : Poseidon2Core)
    (sisDigest : List F -> Digest4)
    (initial : SpongeCursor)
    {left right : BetaPrefixInput}
    (hInput : left = right) :
    betaPrefixResult core sisDigest initial left =
      betaPrefixResult core sisDigest initial right := by
  subst right
  rfl

/--
The complete pre-beta state depends on projection material only through the
SIS digest absorbed before the beta label. This is the deterministic prefix
theorem; it does not turn digest equality into a binding theorem.
-/
theorem betaPrefixResult_eq_of_bound_digests
    (core : Poseidon2Core)
    (sisDigest : List F -> Digest4)
    (initial : SpongeCursor)
    (left right : BetaPrefixInput)
    (hOutputs : left.piCcsOutputDigest = right.piCcsOutputDigest)
    (hProjection :
      sisDigest (serializeProjectionBinding left.projection) =
        sisDigest (serializeProjectionBinding right.projection)) :
    betaPrefixResult core sisDigest initial left =
      betaPrefixResult core sisDigest initial right := by
  simp only [betaPrefixResult]
  rw [hOutputs, hProjection]

/-! ## What the prefix does and does not fix -/

/--
Typed collision in the projection binding. A concrete security theorem must
add injectivity of `serializeProjectionBinding` and reduce this event to the
production SIS/MSIS assumption.
-/
structure TypedSisBindingCollision
    (sisDigest : List F -> Digest4)
    (left right : ProjectionBindingMaterial) : Prop where
  different : left ≠ right
  sameDigest :
    sisDigest (serializeProjectionBinding left) =
      sisDigest (serializeProjectionBinding right)

/--
Equal pre-beta SIS digests fix the parent `y_zcol` material unless the typed
binding map collides. This is the exact deterministic premise the delayed
projection proof may consume; it is not yet a concrete MSIS reduction.
-/
theorem parentYZcol_eq_or_typedSisCollision
    (sisDigest : List F -> Digest4)
    (left right : ProjectionBindingMaterial)
    (hDigest :
      sisDigest (serializeProjectionBinding left) =
        sisDigest (serializeProjectionBinding right)) :
    left.yZcol = right.yZcol ∨
      TypedSisBindingCollision sisDigest left right := by
  by_cases hYZcol : left.yZcol = right.yZcol
  · exact Or.inl hYZcol
  · apply Or.inr
    refine ⟨?_, hDigest⟩
    intro hMaterial
    exact hYZcol (congrArg ProjectionBindingMaterial.yZcol hMaterial)

/--
Exact model-level prefix-fixation statement for the Pi_RLC parent: two runs
with the same incoming cursor and the same bound public digests have the same
beta result, and their parent `y_zcol` values agree unless the typed SIS map
collides. Concrete Fiat-Shamir and SIS/MSIS reductions remain separate.
-/
theorem betaPrefix_fixes_parentYZcol_or_collision
    (core : Poseidon2Core)
    (sisDigest : List F -> Digest4)
    (initial : SpongeCursor)
    (left right : BetaPrefixInput)
    (hOutputs : left.piCcsOutputDigest = right.piCcsOutputDigest)
    (hProjection :
      sisDigest (serializeProjectionBinding left.projection) =
        sisDigest (serializeProjectionBinding right.projection)) :
    betaPrefixResult core sisDigest initial left =
        betaPrefixResult core sisDigest initial right ∧
      (left.projection.yZcol = right.projection.yZcol ∨
        TypedSisBindingCollision sisDigest left.projection
          right.projection) := by
  exact ⟨
    betaPrefixResult_eq_of_bound_digests core sisDigest initial
      left right hOutputs hProjection,
    parentYZcol_eq_or_typedSisCollision sisDigest left.projection
      right.projection hProjection⟩

/-- Make the current post-beta Pi_DEC child dependency explicit. -/
def betaWithChildren
    {Children : Type}
    (core : Poseidon2Core)
    (sisDigest : List F -> Digest4)
    (initial : SpongeCursor)
    (input : BetaPrefixInput)
    (_children : Children) : Beta2 :=
  (betaPrefixResult core sisDigest initial input).beta

/-- Pi_DEC children do not affect the already-sampled beta. -/
theorem betaWithChildren_eq
    {Children : Type}
    (core : Poseidon2Core)
    (sisDigest : List F -> Digest4)
    (initial : SpongeCursor)
    (input : BetaPrefixInput)
    (left right : Children) :
    betaWithChildren core sisDigest initial input left =
      betaWithChildren core sisDigest initial input right := by
  rfl

/--
Negative prefix-fixation theorem: whenever two distinct child messages exist,
the current beta prefix cannot distinguish them. Child recomposition therefore
needs the independent Pi_DEC opening-binding dichotomy; it cannot be justified
by claiming that the children were absorbed before beta.
-/
theorem piDecChildren_not_fixed_by_betaPrefix
    {Children : Type}
    (core : Poseidon2Core)
    (sisDigest : List F -> Digest4)
    (initial : SpongeCursor)
    (input : BetaPrefixInput)
    (left right : Children)
    (different : left ≠ right) :
    left ≠ right ∧
      betaWithChildren core sisDigest initial input left =
        betaWithChildren core sisDigest initial input right := by
  exact ⟨different,
    betaWithChildren_eq core sisDigest initial input left right⟩

end SuperNeo.FPrimeRecursiveVerifier.PiRlcChallenge
