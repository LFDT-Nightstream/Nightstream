import NightstreamFPrime.Lifecycle.XOut
import NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcSampler.Transcript

/-!
Owns the Stage 1 Fiat–Shamir transcript over the Poseidon2 sponge: duplex
absorb and squeeze, the Π_CCS oracle (statement absorb, one absorb per
sum-check round, labelled `α`/`γ`/`r′` squeezes), absorption of the complete
Π_CCS output, and the total Π_RLC challenge sampler into the strong set
`𝓒 = {coefficients in {−2,…,2}}`. Each scalar uses one four-field Poseidon2
window, interpreted in base Goldilocks and reduced modulo `5^54`. The absorb order is the paper's
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
  initialState_is_prior := by
    intro statement
    rfl

/-! ## Π_RLC challenge sampler -/

namespace PiRlcSampler

/-- Enter the established scalar domain `[4, coordinate]`. -/
def enterScalar (state : State) (coordinate : Nat) : State :=
  Spec.Folding.Nifs.NonInteractive.PiRlcSampler.Transcript.enter state coordinate

/-- One total challenge from the verifier-owned transcript schedule. -/
def sampleRingChallenge (initial : State) (coordinate : Nat) : RingF :=
  Spec.Folding.Nifs.NonInteractive.PiRlcSampler.Transcript.challengeAt initial coordinate

structure Batch (count : Nat) where
  challenges : Fin count → RingF
  finalState : State

/-- The coefficients of `sampleRingChallenge initial coordinate`. The transcript state, draw
and scalar are values here. `sampleRingChallenge` is a function of the lane, so each lane read
replays the whole transcript schedule up to the coordinate. -/
private def challengeCoefficients (initial : State) (coordinate : Nat) : Array F :=
  let entered := Spec.Folding.Nifs.NonInteractive.PiRlcSampler.Transcript.enter
    (Spec.Folding.Nifs.NonInteractive.PiRlcSampler.Transcript.stateAt initial coordinate) coordinate
  let reduced := Spec.Folding.Nifs.NonInteractive.PiRlcSampler.reduce
    (Spec.Folding.Nifs.NonInteractive.PiRlcSampler.drawIndex
      (Spec.Folding.Nifs.NonInteractive.PiRlcSampler.Transcript.block entered))
  let digits := Array.ofFn (Spec.Folding.Nifs.NonInteractive.PiRlcSampler.scalarIndex.symm reduced)
  Array.ofFn fun position : Fin ringDegree =>
    Phi81StrongSet.embedCoefficient (digits[(Phi81StrongSet.scalarPosition position).val]'(by
      simp only [digits, Array.size_ofFn]
      exact (Phi81StrongSet.scalarPosition position).isLt))

/-- Compute each coefficient once; later ring operations only read the batch. -/
def piRlcChallengesWithState (initial : State) (count : Nat) : Batch count :=
  let values := Array.ofFn fun index : Fin count => challengeCoefficients initial index.val
  { challenges := fun index lane =>
      (values[index.val]'(by simp only [values, Array.size_ofFn]; exact index.isLt))[lane.val]'(by
        simp only [values, Array.getElem_ofFn, challengeCoefficients, Array.size_ofFn]
        exact lane.isLt)
    finalState := Spec.Folding.Nifs.NonInteractive.PiRlcSampler.Transcript.stateAt initial count }

@[simp] theorem piRlcChallengesWithState_challenges (initial : State) (count : Nat) :
    (piRlcChallengesWithState initial count).challenges =
      fun index => sampleRingChallenge initial index.val := by
  funext index lane
  simp only [piRlcChallengesWithState, challengeCoefficients, Array.getElem_ofFn]
  rfl

def piRlcChallenges (initial : State) (count : Nat) : Fin count → RingF :=
  (piRlcChallengesWithState initial count).challenges

theorem piRlcChallengesWithState_finalState (initial : State) (count : Nat) :
    (piRlcChallengesWithState initial count).finalState = Spec.Folding.Nifs.NonInteractive.PiRlcSampler.Transcript.stateAt initial count := rfl

theorem piRlcChallenges_member (initial : State) (count : Nat) (index : Fin count) :
    Phi81StrongSet.ProductionMember (piRlcChallenges initial count index) := by
  rw [piRlcChallenges, piRlcChallengesWithState_challenges]
  exact Spec.Folding.Nifs.NonInteractive.PiRlcSampler.Transcript.challengeAt_member initial index.val

end PiRlcSampler

end NightstreamFPrime.Lifecycle.Transcript
