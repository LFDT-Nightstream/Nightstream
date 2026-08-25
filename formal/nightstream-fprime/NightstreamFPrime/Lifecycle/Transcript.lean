import NightstreamFPrime.Lifecycle.XOut
import NightstreamFPrime.Spec.Phi81StrongSet

/-!
Owns the Stage 1 Fiat–Shamir transcript over the Poseidon2 sponge: duplex
absorb and squeeze, the Π_CCS oracle (statement absorb, one absorb per
sum-check round, labelled `α`/`γ`/`r′` squeezes), absorption of the complete
Π_CCS output, and the fail-closed Π_RLC challenge sampler into the strong
set `𝓒 = {coefficients in {−2,…,2}}`. The sampler consumes eight complete
four-lane Poseidon2 digests per scalar, exposes two little-endian 16-bit
candidates per lane, rejects candidate `65535`, and returns no batch unless
all 54 coefficients of every scalar exist. The absorb order is the paper's
(SuperNeo B.1): every challenge is squeezed only after the data it must depend
on has been absorbed. All parity-surface definitions are computable.
-/

namespace NightstreamFPrime.Lifecycle.Transcript

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Lifecycle

abbrev State := Poseidon2.State

/-- Absorb a word list in `rate`-sized chunks, permuting after each chunk. -/
def absorb (s : State) (xs : List F) : State :=
  let chunks := (List.range ((xs.length + Poseidon2.rate - 1) / Poseidon2.rate)).map
    (fun c => (xs.drop (c * Poseidon2.rate)).take Poseidon2.rate)
  chunks.foldl Poseidon2.absorbBlock s

/-- Absorb a self-delimiting block: length prefix, then the words. -/
def absorbBlock (s : State) (xs : List F) : State := absorb s (block xs)

/-- Fold a typed list of self-delimiting blocks through the transcript. -/
def absorbBlocks (state : State) (blocks : List (List F)) : State :=
  blocks.foldl absorbBlock state

@[simp] theorem absorbBlocks_append (state : State)
    (left right : List (List F)) :
    absorbBlocks state (left ++ right) =
      absorbBlocks (absorbBlocks state left) right := by
  simp [absorbBlocks, List.foldl_append]

/-- Squeeze one field word (lane 0), then permute. -/
def squeezeF (s : State) : F × State := (s.getD 0 0, Poseidon2.permute s)

/-- Squeeze one extension element from two successive words. -/
def squeezeK (s : State) : K × State :=
  let (c0, s) := squeezeF s
  let (c1, s) := squeezeF s
  (⟨c0, c1⟩, s)

def squeezeKs : Nat → State → List K × State
  | 0, s => ([], s)
  | n + 1, s =>
    let (k, s) := squeezeK s
    let (ks, s) := squeezeKs n s
    (k :: ks, s)

/-! ## Π_CCS oracle -/

def initialState : State := Poseidon2.zeroState

/-- ASCII bytes of `Nightstream/SuperNeo/NIFS/v1`, retained for the complete
NIFS transcript after PiCCS. -/
def domainTagBytes : List Nat :=
  [78, 105, 103, 104, 116, 115, 116, 114, 101, 97, 109, 47,
    83, 117, 112, 101, 114, 78, 101, 111, 47, 78, 73, 70, 83, 47, 118, 49]

/-- Domain tag absorbed before every protocol transcript. -/
def domainTag : List F := domainTagBytes.map Poseidon2.ofNat

@[simp] theorem domainTag_length : domainTag.length = 28 := by
  simp [domainTag, domainTagBytes]

/-- ASCII bytes of `Nightstream/SuperNeo/PiCCS/digest-only/v1_1`. This tag
selects the owner-approved committed-statement schedule. -/
def piCcsDigestDomainTagBytes : List Nat :=
  [78, 105, 103, 104, 116, 115, 116, 114, 101, 97, 109, 47,
    83, 117, 112, 101, 114, 78, 101, 111, 47, 80, 105, 67, 67, 83, 47,
    100, 105, 103, 101, 115, 116, 45, 111, 110, 108, 121, 47, 118, 49,
    95, 49]

/-- Domain tag for the sole digest-only PiCCS statement schedule. -/
def piCcsDigestDomainTag : List F :=
  piCcsDigestDomainTagBytes.map Poseidon2.ofNat

@[simp] theorem piCcsDigestDomainTag_length :
    piCcsDigestDomainTag.length = 43 := by
  simp [piCcsDigestDomainTag, piCcsDigestDomainTagBytes]

def serializeMessage (m : SumCheck.Finite.Message K) : List F :=
  m.coefficients.flatMap serializeK

/-- The two verifier-input blocks of v1.1 Π_CCS: prior point, then Pad
claims in `I_K` order followed by matrix claims in `I_A` order. -/
def verifierInputBlocks
    (input : ProtocolPolynomial.VerifierInput K productionShape) :
    List (List F) :=
  [serializePoint input.priorPoint,
    (canonicalPadCoordinates productionShape).flatMap
        (fun coordinate => serializeK (input.claimedPadCoefficient coordinate)) ++
      (canonicalMatrixCoordinates productionShape).flatMap
        (fun coordinate => serializeK (input.claimedMatrixCoefficient coordinate))]

/-- Absorb the verifier input from its one canonical block list. The
constraint polynomial is key data bound through the verifier-key digest. -/
def absorbVerifierInput (state : State)
    (input : ProtocolPolynomial.VerifierInput K productionShape) : State :=
  absorbBlocks state (verifierInputBlocks input)

/-- Label words keep `α`, `γ`, and round squeezes in distinct domains. -/
def labelWord : FiatShamir.ChallengeLabel productionShape → List F
  | .alpha c => [natWord 1, natWord c.val]
  | .gamma => [natWord 2]
  | .sumcheck r => [natWord 3, natWord r.val]

def piCcsOracle :
    NightstreamFPrime.Spec.Folding.PiCCS.TranscriptReplay.Oracle
      K State productionShape where
  transcript :=
    { initialState := fun statement => statement.priorState
      absorbRound := fun s round m =>
        absorbBlock s (natWord round.val :: serializeMessage m)
      squeeze := fun s label => squeezeK (absorb s (labelWord label)) }

/-! ## Π_RLC challenge sampler -/

namespace PiRlcSampler

open NightstreamFPrime.Spec.Sampling
open NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcSampler
open ProductionAlphabet
open ProductionSchedule
open ProductionStrongSet

/-- Two little-endian 16-bit candidates from each of the four rate lanes. -/
def digestChunks (state : State) : Fin chunksPerDigest → Chunk :=
  fun position =>
    let lane := position.val / 2
    let part := position.val % 2
    ⟨((state.getD lane 0).val / (2 ^ (16 * part))) % chunkModulus,
      Nat.mod_lt _ (by decide)⟩

/-- One complete digest step. The current four rate lanes are the digest;
the successor state is the next Poseidon2 permutation. -/
def digestBlock (state : State) (_counter : Nat) :
    State × (Fin chunksPerDigest → Chunk) :=
  (Poseidon2.permute state, digestChunks state)

/-- Preserve the established scalar domain separator `[4, coordinate]`. -/
def enterScalar (state : State) (coordinate : Nat) : State :=
  absorb state [natWord 4, natWord coordinate]

/-- Concrete additive-sponge instantiation of the fixed eight-block schedule. -/
def machine : ProductionSchedule.Machine State where
  enterScalar := enterScalar
  digestBlock := digestBlock

def specification : Specification State Chunk Coefficient Scalar :=
  ProductionSchedule.specification machine assembleCoefficients

/-- Convert a successful exact-length coefficient list to its scalar carrier.
The default branch is unreachable for every successful bounded sample. -/
def scalarOfList (coefficients : List Coefficient) : Scalar :=
  fun position => coefficients.getD position.val ⟨2, by decide⟩

/-- One indexed scalar sample from the fixed 64-candidate source. -/
def sampleScalar (initial : State) (coordinate : Nat) : Option Scalar :=
  let source := sourceAt specification initial coordinate
  (FirstAccepted.boundedSample verifier coefficientCount
    (FirstAccepted.streamPrefix source.stream candidateBound)).map scalarOfList

/-- One ring challenge, or explicit failure when fewer than 54 candidates
are accepted. -/
def sampleRingChallenge (initial : State) (coordinate : Nat) : Option RingF :=
  (sampleScalar initial coordinate).map Phi81StrongSet.embedScalar

/-- A successful scalar sample has exactly 54 accepted coefficients. -/
theorem sampleScalar_success_length
    {initial : State} {coordinate : Nat} {coefficients : List Coefficient}
    (success : FirstAccepted.boundedSample verifier coefficientCount
      (FirstAccepted.streamPrefix
        (sourceAt specification initial coordinate).stream candidateBound) =
        some coefficients) :
    coefficients.length = coefficientCount :=
  FirstAccepted.bounded_success_length success

/-- Failure is exactly bounded rejection-sampler shortfall. -/
theorem sampleScalar_eq_none_iff_shortfall
    (initial : State) (coordinate : Nat) :
    sampleScalar initial coordinate = none ↔
      ShortfallAt specification candidateBound initial coordinate := by
  unfold sampleScalar ShortfallAt
  change
    Option.map scalarOfList
        (FirstAccepted.boundedSample verifier coefficientCount
          (FirstAccepted.streamPrefix
            (sourceAt specification initial coordinate).stream
            candidateBound)) = none ↔
      FirstAccepted.Shortfall verifier coefficientCount
        (FirstAccepted.streamPrefix
          (sourceAt specification initial coordinate).stream candidateBound)
  rw [Option.map_eq_none_iff]
  exact FirstAccepted.boundedSample_eq_none_iff_shortfall

/-- Every successful concrete ring challenge is in the production strong
set. No membership claim exists on shortfall. -/
theorem sampleRingChallenge_member
    {initial : State} {coordinate : Nat} {challenge : RingF}
    (success : sampleRingChallenge initial coordinate = some challenge) :
    Phi81StrongSet.ProductionMember challenge := by
  unfold sampleRingChallenge at success
  cases sampled : sampleScalar initial coordinate with
  | none => simp [sampled] at success
  | some scalar =>
      simp only [sampled, Option.map_some, Option.some.injEq] at success
      subst challenge
      exact ⟨scalar, rfl⟩

/-- Successful fixed-size batch. Its `Fin` domain prevents a fallback value
when the verifier indexes the `K+k` challenge vector. -/
structure Batch (count : Nat) where
  challenges : Fin count → RingF
  finalState : State

/-- Sample `ρ₁,…,ρ_count` in exact order. Any shortfall rejects the
whole batch. The final state always follows every fixed digest block. -/
def sampleBatch (initial : State) : (count : Nat) → Option (Batch count)
  | 0 => some ⟨Fin.elim0, stateAt specification initial 0⟩
  | count + 1 =>
      match sampleBatch initial count, sampleRingChallenge initial count with
      | some priorBatch, some challenge =>
          some ⟨Fin.lastCases challenge priorBatch.challenges,
            stateAt specification initial (count + 1)⟩
      | _, _ => none

/-- Public computable batch entrypoint. -/
def piRlcChallengesWithState (initial : State) (count : Nat) :
    Option (Batch count) :=
  sampleBatch initial count

def piRlcChallenges (initial : State) (count : Nat) :
    Option (Fin count → RingF) :=
  (piRlcChallengesWithState initial count).map Batch.challenges

/-- The successful batch state is the state after every fixed block. -/
theorem piRlcChallengesWithState_finalState
    {initial : State} {count : Nat} {batch : Batch count}
    (success : piRlcChallengesWithState initial count = some batch) :
    batch.finalState = stateAt specification initial count := by
  induction count with
  | zero =>
      simp [piRlcChallengesWithState, sampleBatch] at success
      subst batch
      rfl
  | succ count inductionHypothesis =>
      rw [piRlcChallengesWithState, sampleBatch] at success
      cases priorEq : sampleBatch initial count with
      | none => simp [priorEq] at success
      | some priorBatch =>
          cases challengeEq : sampleRingChallenge initial count with
          | none => simp [priorEq, challengeEq] at success
          | some challenge =>
              simp [priorEq, challengeEq] at success
              subst batch
              rfl

/-- Every indexed value in a successful batch is a strong-set challenge. -/
theorem piRlcChallenges_member
    {initial : State} {count : Nat} {batch : Batch count}
    (success : piRlcChallengesWithState initial count = some batch)
    (index : Fin count) :
    Phi81StrongSet.ProductionMember (batch.challenges index) := by
  induction count with
  | zero => exact Fin.elim0 index
  | succ count inductionHypothesis =>
      rw [piRlcChallengesWithState, sampleBatch] at success
      cases priorEq : sampleBatch initial count with
      | none => simp [priorEq] at success
      | some priorBatch =>
          cases challengeEq : sampleRingChallenge initial count with
          | none => simp [priorEq, challengeEq] at success
          | some challenge =>
              simp [priorEq, challengeEq] at success
              subst batch
              refine Fin.lastCases ?_ (fun prior => ?_) index
              · simpa using sampleRingChallenge_member challengeEq
              ·
                have priorSuccess :
                    piRlcChallengesWithState initial count = some priorBatch := by
                  simpa [piRlcChallengesWithState] using priorEq
                simpa using inductionHypothesis priorSuccess prior

end PiRlcSampler

end NightstreamFPrime.Lifecycle.Transcript
