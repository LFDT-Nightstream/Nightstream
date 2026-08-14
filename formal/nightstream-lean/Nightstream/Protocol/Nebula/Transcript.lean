import Nightstream.Protocol.Nebula.ProductState
import Nightstream.Protocol.Nebula.Profile
import Nightstream.Protocol.Nebula.FPrime

/-!
Contract: complete ordered memory-challenge transcript frame for Nebula V2.

Assurance tier: model-level and cryptographic-reduction boundary.

Owns every authority field absorbed before the two challenge-pair repetitions,
their exact order and tags, and four domain-separated challenge coordinates.

Does not own Poseidon2, Fiat-Shamir unpredictability, random-oracle extraction,
query bounds, or a concrete circuit transcript.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Protocol.Nebula.Transcript

open Nightstream.Protocol.Nebula.FPrime
open Nightstream.Protocol.Nebula.Lifecycle

inductive DigestTag where
  | verifierKey
  | applicationRelation
  | program
  | memoryPlan
  | laneLayout
  | priorState
  | runningAccumulator
  | operationsRoot
  | initialSnapshotRoot
  | finalSnapshotRoot
deriving DecidableEq, Repr

inductive NatTag where
  | segmentIndex
  | segmentStartTimestamp
  | activeAccessCount
  | segmentEndTimestamp
  | checkedStepCount
  | operationSlots
  | scanSlots
deriving DecidableEq, Repr

inductive Atom (Digest : Type) where
  | domain
  | profile (identity : Profile.Identity)
  | digest (tag : DigestTag) (value : Digest)
  | nat (tag : NatTag) (value : Nat)
deriving DecidableEq, Repr

/-- Number of fresh claims in one fixed 1,088-step memory segment. Selected
profiles separately prove a positive checked-step factor that divides 1,088. -/
def checkedStepCount (profile : Profile.Identity) : Nat :=
  claimsPerSegment / profile.checkedStepsPerFreshClaim

/-- Complete pre-squeeze data. The end timestamp is included independently so
the frame cannot rely on an implicit arithmetic convention. -/
structure Frame (Digest : Type) where
  profile : Profile.Identity
  verifierKeyDigest : Digest
  applicationRelationDigest : Digest
  programDigest : Digest
  memoryPlanDigest : Digest
  laneLayoutDigest : Digest
  priorStateDigest : Digest
  runningAccumulatorDigest : Digest
  segmentIndex : Nat
  segmentStartTimestamp : Nat
  activeAccessCount : Nat
  segmentEndTimestamp : Nat
  roots : Roots Digest
deriving DecidableEq, Repr

/-- Normative absorption order. Domain and field tags are part of each atom. -/
def encode {Digest : Type} (frame : Frame Digest) : List (Atom Digest) :=
  [ .domain
  , .profile frame.profile
  , .digest .verifierKey frame.verifierKeyDigest
  , .digest .applicationRelation frame.applicationRelationDigest
  , .digest .program frame.programDigest
  , .digest .memoryPlan frame.memoryPlanDigest
  , .digest .laneLayout frame.laneLayoutDigest
  , .digest .priorState frame.priorStateDigest
  , .digest .runningAccumulator frame.runningAccumulatorDigest
  , .nat .segmentIndex frame.segmentIndex
  , .nat .segmentStartTimestamp frame.segmentStartTimestamp
  , .nat .activeAccessCount frame.activeAccessCount
  , .nat .segmentEndTimestamp frame.segmentEndTimestamp
  , .nat .checkedStepCount (checkedStepCount frame.profile)
  , .nat .operationSlots 63
  , .nat .scanSlots 64
  , .digest .operationsRoot frame.roots.operations
  , .digest .initialSnapshotRoot frame.roots.initialSnapshot
  , .digest .finalSnapshotRoot frame.roots.finalSnapshot
  ]

theorem encode_length {Digest : Type} (frame : Frame Digest) :
    (encode frame).length = 19 :=
  rfl

theorem encode_injective
    {Digest : Type} [DecidableEq Digest] :
    Function.Injective (encode : Frame Digest → List (Atom Digest)) := by
  intro left right equal
  cases left
  cases right
  simp_all [encode]
  apply Roots.ext <;> tauto

/-- One oracle call index for each coordinate, in this exact order:
`gamma[0].gamma1`, `gamma[0].gamma2`, `gamma[1].gamma1`,
`gamma[1].gamma2`. -/
def coordinateIndex (repetition coordinate : Fin 2) : Fin 4 :=
  ⟨2 * repetition.val + coordinate.val, by
    have repetitionBound := repetition.isLt
    have coordinateBound := coordinate.isLt
    omega⟩

/-- The four repetition/coordinate positions have four distinct oracle
indices. This prevents structural challenge reuse. It does not prove that a
random oracle returns different field values. -/
theorem coordinateIndex_injective :
    Function.Injective
      (fun position : Fin 2 × Fin 2 =>
        coordinateIndex position.1 position.2) := by
  intro left right equal
  have values :
      2 * left.1.val + left.2.val =
        2 * right.1.val + right.2.val :=
    congrArg Fin.val equal
  apply Prod.ext
  · apply Fin.ext
    omega
  · apply Fin.ext
    omega

theorem coordinateIndex_ne
    {left right : Fin 2 × Fin 2} (different : left ≠ right) :
    coordinateIndex left.1 left.2 ≠
      coordinateIndex right.1 right.2 :=
  fun equal => different (coordinateIndex_injective equal)

def coordinatePosition (index : Fin 4) : Fin 2 × Fin 2 :=
  (⟨index.val / 2, by omega⟩, ⟨index.val % 2, by omega⟩)

/-- Exact bijection between the two repetition bits plus two coordinate bits
and the four transcript squeeze indices. -/
def coordinateEquiv : Fin 2 × Fin 2 ≃ Fin 4 where
  toFun position := coordinateIndex position.1 position.2
  invFun := coordinatePosition
  left_inv := by
    intro position
    apply Prod.ext
    · apply Fin.ext
      simp [coordinateIndex, coordinatePosition]
      omega
    · apply Fin.ext
      simp [coordinateIndex, coordinatePosition]
      omega
  right_inv := by
    intro index
    apply Fin.ext
    simp [coordinateIndex, coordinatePosition]
    omega

abbrev Oracle (Digest ChallengeField : Type) :=
  List (Atom Digest) → Fin 4 → ChallengeField

def derive
    {Digest ChallengeField : Type}
    (oracle : Oracle Digest ChallengeField)
    (frame : Frame Digest) : ProductState.Challenges ChallengeField :=
  fun repetition =>
    { gamma1 := oracle (encode frame) (coordinateIndex repetition 0)
      gamma2 := oracle (encode frame) (coordinateIndex repetition 1) }

@[simp]
theorem derive_gamma1
    {Digest ChallengeField : Type}
    (oracle : Oracle Digest ChallengeField)
    (frame : Frame Digest) (repetition : Fin 2) :
    (derive oracle frame repetition).gamma1 =
      oracle (encode frame) (coordinateIndex repetition 0) :=
  rfl

@[simp]
theorem derive_gamma2
    {Digest ChallengeField : Type}
    (oracle : Oracle Digest ChallengeField)
    (frame : Frame Digest) (repetition : Fin 2) :
    (derive oracle frame repetition).gamma2 =
      oracle (encode frame) (coordinateIndex repetition 1) :=
  rfl

/-- Structural frame binding does not imply unpredictability. A constant
oracle is an explicit countermodel and must be excluded by the ROM theorem,
not by a tautological data-flow lemma. -/
def constantOracle
    {Digest ChallengeField : Type} (value : ChallengeField) :
    Oracle Digest ChallengeField :=
  fun _frame _coordinate => value

theorem constant_oracle_ignores_distinct_frames
    {Digest ChallengeField : Type}
    (value : ChallengeField) (left right : Frame Digest) :
    derive (constantOracle value) left = derive (constantOracle value) right := by
  rfl

end Nightstream.Protocol.Nebula.Transcript
