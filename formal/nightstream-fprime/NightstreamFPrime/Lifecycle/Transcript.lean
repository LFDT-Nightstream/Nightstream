import NightstreamFPrime.Lifecycle.XOut
import NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcSampler.Transcript

/-!
SuperNeo v1.2 Section 7.3 public-coin absorb order.
Rust parity surface: every definition is computable.
-/

namespace NightstreamFPrime.Lifecycle.Transcript

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Lifecycle

abbrev State := Poseidon2.State

def absorb (s : State) (xs : List F) : State :=
  let chunks := (List.range ((xs.length + Poseidon2.rate - 1) / Poseidon2.rate)).map
    (fun c => (xs.drop (c * Poseidon2.rate)).take Poseidon2.rate)
  chunks.foldl Poseidon2.absorbBlock s

def initialState : State := Poseidon2.zeroState

def foldDomainBytes : List Nat :=
  "Nightstream/SuperNeo/fold/v2".toList.map Char.toNat

def foldDomainChunk : List F :=
  (List.range Poseidon2.rate).map fun word =>
    packBytes ((foldDomainBytes.drop (8 * word)).take 8)

@[simp] theorem foldDomainChunk_length : foldDomainChunk.length = 12 := by
  simp [foldDomainChunk, Poseidon2.rate]

def foldInitialState : State := absorb initialState foldDomainChunk

def serializeMessage (m : SumCheck.Finite.Message K) : List F :=
  m.coefficients.flatMap serializeK

def verifierInputBlocks
    (input : ProtocolPolynomial.VerifierInput K productionShape) :
    List (List F) :=
  [serializePoint input.priorPoint,
    (canonicalPadCoordinates productionShape).flatMap
        (fun coordinate => serializeK (input.claimedPadCoefficient coordinate)) ++
      (canonicalMatrixCoordinates productionShape).flatMap
        (fun coordinate => serializeK (input.claimedMatrixCoefficient coordinate))]

abbrev coinsPerChunk : Nat := Poseidon2.rate / 2

@[simp] theorem coinsPerChunk_eq : coinsPerChunk = 6 := rfl

def coinPosition : FiatShamir.ChallengeLabel productionShape → Nat
  | .alpha coordinate => coordinate.val
  | .gamma => productionShape.cubeVariables
  | .sumcheck _ => 0

def zeroChunk : List F := List.replicate Poseidon2.rate 0

def squeezeAt (s : State) (position : Nat) : K × State :=
  (⟨s.getD (2 * (position % coinsPerChunk)) 0,
      s.getD (2 * (position % coinsPerChunk) + 1) 0⟩,
    if position % coinsPerChunk = coinsPerChunk - 1 then absorb s zeroChunk else s)

def piCcsOracle :
    NightstreamFPrime.Spec.Folding.PiCCS.TranscriptReplay.Oracle
      K State productionShape where
  transcript :=
    { initialState := fun statement => statement.priorState
      absorbRound := fun s _round m => absorb s (serializeMessage m)
      squeeze := fun s label => squeezeAt s (coinPosition label) }
  initialState_is_prior := by
    intro statement
    rfl

namespace PiRlcSampler

def enterScalar (state : State) (coordinate : Nat) : State :=
  Spec.Folding.Nifs.NonInteractive.PiRlcSampler.Transcript.enter state coordinate

def sampleRingChallenge (initial : State) (coordinate : Nat) : RingF :=
  Spec.Folding.Nifs.NonInteractive.PiRlcSampler.Transcript.challengeAt initial coordinate

structure Batch (count : Nat) where
  challenges : Fin count → RingF
  finalState : State

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
