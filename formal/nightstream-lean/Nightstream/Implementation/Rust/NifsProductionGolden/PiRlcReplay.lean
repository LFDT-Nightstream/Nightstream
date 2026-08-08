import Nightstream.Implementation.Rust.NifsProductionGolden.CertifiedDuplex
import Nightstream.Implementation.Rust.NifsProductionGolden.Decode
import Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler.ProductionStrongSet

/-!
Exact production transcript handoff and bounded `Pi_RLC` scalar sampling.

This module includes the `digest32` query binding words `[0x104, 32]`, the
framed `pi_rlc/input_claims_digest` label, and the four 16-candidate digest
blocks used by Rust. Every permutation is checked by `CertifiedDuplex`.
-/

set_option autoImplicit false
set_option maxRecDepth 10000

namespace Nightstream.Implementation.Rust.NifsProductionGolden.PiRlcReplay

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler
open Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler.ProductionAlphabet
open Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler.ProductionStrongSet
open Nightstream.SuperNeo.Sampling
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Implementation.Rust.NifsProductionGolden
open Nightstream.Implementation.Rust.NifsProductionGolden.CertifiedDuplex
open Nightstream.Implementation.Rust.PiCcsExecution
open Nightstream.Implementation.Rust.PiCcsExecution.CachedDuplex

abbrev TranscriptState :=
  Nightstream.Implementation.R1CS.Canonical.Poseidon2Duplex.State

def digestLanes (state : TranscriptState) : List Nat :=
  List.ofFn fun lane : Fin 4 =>
    state.lanes ⟨lane.val, by
      have laneLt := lane.isLt
      change lane.val < 8
      omega⟩

/-- Canonical Rust `digest32`: squeeze, read four lanes, then bind the digest
query tag and byte length back into the same transcript. -/
def referenceDigest32 (state : TranscriptState) : List Nat × TranscriptState :=
  let squeezed := CachedDuplex.gate Poseidon2CanonicalConstants.selected state
  let rebound := CachedDuplex.absorbList Poseidon2CanonicalConstants.selected
    [0x104, 32] squeezed
  (digestLanes squeezed, rebound)

def digest32? (receipt : ProductionReceipt)
    (state : ReplayState) : Option (List Nat × ReplayState) := do
  let squeezed <- CertifiedDuplex.gate? receipt state
  let rebound <- CertifiedDuplex.absorbFields? receipt [0x104, 32] squeezed
  some (digestLanes squeezed.transcript, rebound)

theorem digest32?_sound (receipt : ProductionReceipt)
    (state outputState : ReplayState) (digest : List Nat)
    (accepted : digest32? receipt state = some (digest, outputState)) :
    (digest, outputState.transcript) = referenceDigest32 state.transcript := by
  cases squeezedEq : CertifiedDuplex.gate? receipt state with
  | none => simp [digest32?, squeezedEq] at accepted
  | some squeezed =>
    cases reboundEq : CertifiedDuplex.absorbFields? receipt [0x104, 32]
        squeezed with
    | none => simp [digest32?, squeezedEq, reboundEq] at accepted
    | some rebound =>
      have acceptedEq : (digestLanes squeezed.transcript, rebound) =
          (digest, outputState) :=
        Option.some.inj (by
          simpa [digest32?, squeezedEq, reboundEq] using accepted)
      have squeezeSound := CertifiedDuplex.gate?_sound receipt state squeezed
        squeezedEq
      have reboundSound := CertifiedDuplex.absorbFields?_sound receipt
        [0x104, 32] squeezed rebound reboundEq
      cases acceptedEq
      unfold referenceDigest32
      rw [<- squeezeSound]
      dsimp only
      rw [<- reboundSound]

def appendRawPair (state : TranscriptState) (first second : Nat) :
    TranscriptState :=
  CachedDuplex.absorbList Poseidon2CanonicalConstants.selected
    [2, first, second] state

def appendRawPair? (receipt : ProductionReceipt) (first second : Nat)
    (state : ReplayState) : Option ReplayState :=
  CertifiedDuplex.absorbFields? receipt [2, first, second] state

theorem appendRawPair?_sound (receipt : ProductionReceipt)
    (first second : Nat) (state outputState : ReplayState)
    (accepted : appendRawPair? receipt first second state = some outputState) :
    outputState.transcript = appendRawPair state.transcript first second := by
  exact CertifiedDuplex.absorbFields?_sound receipt [2, first, second]
    state outputState accepted

/-- Exact field framing of `append_fields(b"pi_rlc/input_claims_digest", d)`.
The four middle values are the 7-byte little-endian label limbs. -/
def outputDigestBindingFields (digest : List Nat) : List Nat :=
  [2, 26, 13338641331874160, 27970976485502569,
    28252447032566124, 500152231785, 4] ++ digest

def bindOutputDigest (digest : List Nat) (state : TranscriptState) :
    TranscriptState :=
  CachedDuplex.absorbList Poseidon2CanonicalConstants.selected
    (outputDigestBindingFields digest) state

def bindOutputDigest? (receipt : ProductionReceipt) (digest : List Nat)
    (state : ReplayState) : Option ReplayState :=
  CertifiedDuplex.absorbFields? receipt (outputDigestBindingFields digest) state

theorem bindOutputDigest?_sound (receipt : ProductionReceipt)
    (digest : List Nat) (state outputState : ReplayState)
    (accepted : bindOutputDigest? receipt digest state = some outputState) :
    outputState.transcript = bindOutputDigest digest state.transcript := by
  exact CertifiedDuplex.absorbFields?_sound receipt
    (outputDigestBindingFields digest) state outputState accepted

def snapshotMatches (state : ReplayState)
    (snapshot : RawTranscriptSnapshot) : Bool :=
  decide (List.ofFn state.transcript.lanes = snapshot.lanes) &&
    decide (state.transcript.absorbed = snapshot.absorbed)

theorem snapshotMatches_sound (state : ReplayState)
    (snapshot : RawTranscriptSnapshot)
    (checked : snapshotMatches state snapshot = true) :
    state.transcript = decodeSnapshot snapshot := by
  have components :
      List.ofFn state.transcript.lanes = snapshot.lanes /\
        state.transcript.absorbed = snapshot.absorbed := by
    simpa only [snapshotMatches, Bool.and_eq_true, decide_eq_true_eq]
      using checked
  cases state with
  | mk transcript nextTrace =>
    cases transcript with
    | mk lanes absorbed =>
      simp only at components
      have lanesEq : lanes =
          (decodeSnapshot snapshot).lanes := by
        funext lane
        have atLane := congrArg (fun values : List Nat =>
          values.getD lane.val 0) components.1
        simpa [decodeSnapshot] using atLane
      have absorbedEq : absorbed =
          (decodeSnapshot snapshot).absorbed := by
        simpa [decodeSnapshot] using components.2
      cases lanesEq
      cases absorbedEq
      rfl

def foldDigestMatches (receipt : ProductionReceipt) (digest : List Nat) : Bool :=
  receipt.piRlcInputs.all fun claim => decide (claim.foldDigest = digest)

structure HandoffResult where
  foldDigest : List Nat
  rhoState : ReplayState

def handoff? (receipt : ProductionReceipt) (piCcsFinal : TranscriptState) :
    Option HandoffResult := do
  let start := CertifiedDuplex.initial piCcsFinal receipt.piCcsPermutationCount
  let digested <- digest32? receipt start
  if foldDigestMatches receipt digested.1 then
    let bound <- bindOutputDigest? receipt receipt.piCcsOutputsDigest digested.2
    if snapshotMatches bound receipt.rhoStart &&
        bound.nextTrace = receipt.rhoStartPermutationCount then
      some { foldDigest := digested.1, rhoState := bound }
    else
      none
  else
    none

theorem handoff?_sound (receipt : ProductionReceipt)
    (piCcsFinal : TranscriptState)
    (result : HandoffResult)
    (accepted : handoff? receipt piCcsFinal = some result) :
    let referenceDigest := referenceDigest32 piCcsFinal
    result.foldDigest = referenceDigest.1 /\
      result.rhoState.transcript =
        bindOutputDigest receipt.piCcsOutputsDigest referenceDigest.2 /\
      result.rhoState.transcript = decodeSnapshot receipt.rhoStart /\
      receipt.piRlcInputs.all
        (fun claim => decide (claim.foldDigest = result.foldDigest)) = true := by
  unfold handoff? at accepted
  cases digestedEq : digest32? receipt
      (CertifiedDuplex.initial piCcsFinal receipt.piCcsPermutationCount) with
  | none => simp [digestedEq] at accepted
  | some digested =>
    have accepted0 :
        (if foldDigestMatches receipt digested.1 then
          do
            let bound <- bindOutputDigest? receipt receipt.piCcsOutputsDigest
              digested.2
            if snapshotMatches bound receipt.rhoStart &&
                bound.nextTrace = receipt.rhoStartPermutationCount then
              some { foldDigest := digested.1, rhoState := bound }
            else
              none
        else none) = some result := by
      simpa [digestedEq] using accepted
    split at accepted0
    · rename_i digestChecked
      cases boundEq : bindOutputDigest? receipt receipt.piCcsOutputsDigest
          digested.2 with
      | none => simp [digestedEq, digestChecked, boundEq] at accepted
      | some bound =>
        have accepted' :
            (if snapshotMatches bound receipt.rhoStart &&
                bound.nextTrace = receipt.rhoStartPermutationCount then
              some { foldDigest := digested.1, rhoState := bound }
            else none) = some result := by
          simpa [digestChecked, boundEq] using accepted0
        split at accepted'
        · rename_i finalChecked
          cases Option.some.inj accepted'
          have digestSound := digest32?_sound receipt
            (CertifiedDuplex.initial piCcsFinal receipt.piCcsPermutationCount)
            digested.2 digested.1 (by simpa using digestedEq)
          have bindingSound := bindOutputDigest?_sound receipt
            receipt.piCcsOutputsDigest digested.2 bound boundEq
          have finalComponents :
              snapshotMatches bound receipt.rhoStart = true /\
                decide (bound.nextTrace = receipt.rhoStartPermutationCount) =
                  true := by
            simpa only [Bool.and_eq_true] using finalChecked
          have snapshotChecked : snapshotMatches bound receipt.rhoStart = true :=
            finalComponents.1
          have snapshotSound := snapshotMatches_sound bound receipt.rhoStart
            snapshotChecked
          dsimp only [CertifiedDuplex.initial] at digestSound
          unfold foldDigestMatches at digestChecked
          dsimp only
          constructor
          · exact congrArg Prod.fst digestSound
          constructor
          · have digestStateSound : digested.2.transcript =
                (referenceDigest32 piCcsFinal).2 := by
              exact congrArg Prod.snd digestSound
            rw [bindingSound, digestStateSound]
          constructor
          · exact snapshotSound
          · exact digestChecked
        · contradiction
    · contradiction

def laneChunk (lane part : Nat) : Chunk :=
  ⟨(lane / (2 ^ (16 * part))) % chunkModulus,
    Nat.mod_lt _ (by decide)⟩

def digestChunks (lanes : List Nat) : List Chunk :=
  (List.range chunksPerDigest).map fun position =>
    laneChunk (lanes.getD (position / 4) 0) (position % 4)

def referenceBlocks : Nat -> Nat -> TranscriptState ->
    List Chunk × TranscriptState
  | _, 0, state => ([], state)
  | counter, count + 1, state =>
      let entered := appendRawPair state 1 counter
      let digested := referenceDigest32 entered
      let rest := referenceBlocks (counter + 1) count digested.2
      (digestChunks digested.1 ++ rest.1, rest.2)

def blocks? (receipt : ProductionReceipt) :
    Nat -> Nat -> ReplayState -> Option (List Chunk × ReplayState)
  | _, 0, state => some ([], state)
  | counter, count + 1, state => do
      let entered <- appendRawPair? receipt 1 counter state
      let digested <- digest32? receipt entered
      let rest <- blocks? receipt (counter + 1) count digested.2
      some (digestChunks digested.1 ++ rest.1, rest.2)

theorem blocks?_sound (receipt : ProductionReceipt) :
    forall counter count state chunks outputState,
      blocks? receipt counter count state = some (chunks, outputState) ->
        (chunks, outputState.transcript) =
          referenceBlocks counter count state.transcript := by
  intro counter count
  induction count generalizing counter with
  | zero =>
      intro state chunks outputState accepted
      simp only [blocks?] at accepted
      cases accepted
      rfl
  | succ count inductionHypothesis =>
      intro state chunks outputState accepted
      cases enteredEq : appendRawPair? receipt 1 counter state with
      | none => simp [blocks?, enteredEq] at accepted
      | some entered =>
        cases digestedEq : digest32? receipt entered with
        | none => simp [blocks?, enteredEq, digestedEq] at accepted
        | some digested =>
          cases restEq : blocks? receipt (counter + 1) count digested.2 with
          | none => simp [blocks?, enteredEq, digestedEq, restEq] at accepted
          | some rest =>
            have acceptedEq :
                (digestChunks digested.1 ++ rest.1, rest.2) =
                  (chunks, outputState) :=
              Option.some.inj (by
                simpa [blocks?, enteredEq, digestedEq, restEq] using accepted)
            have enteredSound := appendRawPair?_sound receipt 1 counter
              state entered enteredEq
            have digestSound := digest32?_sound receipt entered
              digested.2 digested.1 (by simpa using digestedEq)
            have restSound := inductionHypothesis (counter + 1)
              digested.2 rest.1 rest.2 (by simpa using restEq)
            cases acceptedEq
            unfold referenceBlocks
            dsimp only
            rw [<- enteredSound]
            rw [<- digestSound]
            dsimp only
            rw [<- restSound]

def scalarOfList? (values : List Coefficient) : Option Scalar :=
  if length : values.length = coefficientCount then
    some fun position => values.get
      ⟨position.val, by simpa [length] using position.isLt⟩
  else
    none

def referenceSample? (state : TranscriptState) :
    Option (Scalar × TranscriptState) :=
  let entered := appendRawPair state 0 0
  let blocks := referenceBlocks 0 digestRounds entered
  match FirstAccepted.boundedSample verifier coefficientCount blocks.1 with
  | none => none
  | some values =>
      match scalarOfList? values with
      | none => none
      | some scalar => some (scalar, blocks.2)

def sample? (receipt : ProductionReceipt)
    (state : ReplayState) : Option (Scalar × ReplayState) := do
  let entered <- appendRawPair? receipt 0 0 state
  let blocks <- blocks? receipt 0 digestRounds entered
  let values <- FirstAccepted.boundedSample verifier coefficientCount blocks.1
  let scalar <- scalarOfList? values
  if blocks.2.nextTrace = receipt.poseidonPermutationTraces.length then
    some (scalar, blocks.2)
  else
    none

theorem sample?_sound (receipt : ProductionReceipt)
    (state outputState : ReplayState) (scalar : Scalar)
    (accepted : sample? receipt state = some (scalar, outputState)) :
    referenceSample? state.transcript = some (scalar, outputState.transcript) := by
  unfold sample? at accepted
  cases enteredEq : appendRawPair? receipt 0 0 state with
  | none => simp [enteredEq] at accepted
  | some entered =>
    cases blocksEq : blocks? receipt 0 digestRounds entered with
    | none => simp [enteredEq, blocksEq] at accepted
    | some blocks =>
      cases valuesEq : FirstAccepted.boundedSample verifier coefficientCount
          blocks.1 with
      | none => simp [enteredEq, blocksEq, valuesEq] at accepted
      | some values =>
        cases scalarEq : scalarOfList? values with
        | none => simp [enteredEq, blocksEq, valuesEq, scalarEq] at accepted
        | some sampled =>
          have accepted' :
              (if blocks.2.nextTrace = receipt.poseidonPermutationTraces.length
                then some (sampled, blocks.2) else none) =
                some (scalar, outputState) := by
            simpa [enteredEq, blocksEq, valuesEq, scalarEq] using accepted
          split at accepted'
          · have pairEq := Option.some.inj accepted'
            have enteredSound := appendRawPair?_sound receipt 0 0 state
              entered enteredEq
            have blocksSound := blocks?_sound receipt 0 digestRounds entered
              blocks.1 blocks.2 (by simpa using blocksEq)
            cases pairEq
            unfold referenceSample?
            dsimp only
            rw [<- enteredSound]
            rw [<- blocksSound]
            dsimp only
            rw [valuesEq]
            simp only
            rw [scalarEq]
          · contradiction

def sampleFromReceipt? (receipt : ProductionReceipt) :
    Option (Scalar × ReplayState) :=
  sample? receipt <| CertifiedDuplex.initial
    (decodeSnapshot receipt.rhoStart) receipt.rhoStartPermutationCount

end Nightstream.Implementation.Rust.NifsProductionGolden.PiRlcReplay
