import Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalSelectorSound
import Nightstream.SuperNeo.Sampling.FirstAccepted

/-!
Contract: honest completeness for the Lean-owned 54-of-64 `Pi_RLC`
selector.

The selected source index is computed from the verifier-owned candidate list.
It is not a prover-supplied route.  The finite search returns the source of the
requested accepted position together with its exact accepted-prefix count.

This file owns the selector witness and its row satisfaction.  Candidate-row
honesty and whole-sampler witness threading are separate responsibilities.
-/

set_option autoImplicit false
set_option maxRecDepth 100000

namespace Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalSelectorHonest

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalSelector
open Nightstream.SuperNeo.Sampling
open Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler

universe uCandidate uSymbol

/-- Source index of the requested accepted candidate.  The fallback on a
short list is unreachable under `Enough`; keeping the function total makes the
honest assignment executable without a choice principle. -/
def nthAcceptedIndex
    {Candidate : Type uCandidate} {Symbol : Type uSymbol}
    (verifier : FirstAccepted.Verifier Candidate Symbol) :
    List Candidate → Nat → Nat
  | [], _ => 0
  | head :: tail, position =>
      if verifier.accepts head then
        match position with
        | 0 => 0
        | next + 1 => 1 + nthAcceptedIndex verifier tail next
      else
        1 + nthAcceptedIndex verifier tail position

/-- The finite search returns an accepted source with exactly `position`
accepted predecessors. -/
theorem nthAcceptedIndex_spec
    {Candidate : Type uCandidate} {Symbol : Type uSymbol}
    (verifier : FirstAccepted.Verifier Candidate Symbol)
    (candidates : List Candidate) (position : Nat)
    (available :
      position < FirstAccepted.acceptedCount verifier candidates) :
    let index := nthAcceptedIndex verifier candidates position
    ∃ bounded : index < candidates.length,
      verifier.accepts (candidates[index]'bounded) = true ∧
      FirstAccepted.acceptedCount verifier (candidates.take index) =
        position := by
  induction candidates generalizing position with
  | nil =>
      simp [FirstAccepted.acceptedCount,
        FirstAccepted.acceptedCandidates] at available
  | cons head tail inductionHypothesis =>
      cases decision : verifier.accepts head
      · have tailAvailable :
          position < FirstAccepted.acceptedCount verifier tail := by
          simpa [FirstAccepted.acceptedCount,
            FirstAccepted.acceptedCandidates, decision] using available
        have tailSpec := inductionHypothesis position tailAvailable
        rcases tailSpec with ⟨tailBounded, tailAccepted, tailBefore⟩
        refine ⟨by
          simp only [nthAcceptedIndex, decision, Bool.false_eq_true,
            ↓reduceIte, List.length_cons]
          omega, ?_, ?_⟩
        · simpa [nthAcceptedIndex, decision, Nat.one_add] using tailAccepted
        · simpa [nthAcceptedIndex, decision, FirstAccepted.acceptedCount,
            FirstAccepted.acceptedCandidates, Nat.one_add] using tailBefore
      · cases position with
        | zero =>
            simp [nthAcceptedIndex, decision, FirstAccepted.acceptedCount,
              FirstAccepted.acceptedCandidates]
        | succ position =>
            have tailAvailable :
                position < FirstAccepted.acceptedCount verifier tail := by
              simpa [FirstAccepted.acceptedCount,
                FirstAccepted.acceptedCandidates, decision] using available
            have tailSpec := inductionHypothesis position tailAvailable
            rcases tailSpec with ⟨tailBounded, tailAccepted, tailBefore⟩
            refine ⟨by
              simp only [nthAcceptedIndex, decision, ↓reduceIte,
                List.length_cons]
              omega, ?_, ?_⟩
            · simpa [nthAcceptedIndex, decision, Nat.one_add] using tailAccepted
            · simpa [nthAcceptedIndex, decision,
                FirstAccepted.acceptedCount,
                FirstAccepted.acceptedCandidates, Nat.one_add] using tailBefore

/-- Filtering cannot create candidates. -/
theorem acceptedCount_le_length
    {Candidate : Type uCandidate} {Symbol : Type uSymbol}
    (verifier : FirstAccepted.Verifier Candidate Symbol)
    (candidates : List Candidate) :
    FirstAccepted.acceptedCount verifier candidates ≤ candidates.length := by
  simpa [FirstAccepted.acceptedCount,
    FirstAccepted.acceptedCandidates] using
      List.length_filter_le verifier.accepts candidates

/-- With 54 accepts among 64 candidates, the source of output `position` is
inside the exact eleven-candidate window beginning at `position`. -/
theorem nthAcceptedIndex_in_window
    {Candidate : Type uCandidate} {Symbol : Type uSymbol}
    (verifier : FirstAccepted.Verifier Candidate Symbol)
    (candidates : List Candidate)
    (lengthExact : candidates.length = 64)
    (enough : FirstAccepted.Enough verifier 54 candidates)
    (position : Nat) (positionLt : position < 54) :
    let index := nthAcceptedIndex verifier candidates position
    position ≤ index ∧ index < position + 11 := by
  dsimp only
  let index := nthAcceptedIndex verifier candidates position
  change position ≤ index ∧ index < position + 11
  have available :
      position < FirstAccepted.acceptedCount verifier candidates := by
    unfold FirstAccepted.Enough at enough
    omega
  obtain ⟨indexBounded, _, before⟩ :=
    nthAcceptedIndex_spec verifier candidates position available
  have before' :
      FirstAccepted.acceptedCount verifier (candidates.take index) =
        position := by
    simpa [index] using before
  have takeLength :
      (candidates.take index).length = index := by
    rw [List.length_take]
    exact Nat.min_eq_left (Nat.le_of_lt indexBounded)
  have positionLe : position ≤ index := by
    calc
      position =
          FirstAccepted.acceptedCount verifier (candidates.take index) :=
        before'.symm
      _ ≤ (candidates.take index).length :=
        acceptedCount_le_length verifier _
      _ = index := takeLength
  have countDecomposition :
      FirstAccepted.acceptedCount verifier candidates =
        FirstAccepted.acceptedCount verifier (candidates.take index) +
          FirstAccepted.acceptedCount verifier (candidates.drop index) := by
    calc
      FirstAccepted.acceptedCount verifier candidates =
          FirstAccepted.acceptedCount verifier
            (candidates.take index ++ candidates.drop index) :=
        congrArg (FirstAccepted.acceptedCount verifier)
          (List.take_append_drop index candidates).symm
      _ = FirstAccepted.acceptedCount verifier (candidates.take index) +
          FirstAccepted.acceptedCount verifier (candidates.drop index) := by
        unfold FirstAccepted.acceptedCount
        rw [FirstAccepted.acceptedCandidates_append]
        exact List.length_append
  have suffixBound :
      FirstAccepted.acceptedCount verifier (candidates.drop index) ≤
        (candidates.drop index).length :=
    acceptedCount_le_length verifier _
  have lengthDecomposition :
      candidates.length =
        (candidates.take index).length + (candidates.drop index).length := by
    calc
      candidates.length =
          (candidates.take index ++ candidates.drop index).length :=
        congrArg List.length (List.take_append_drop index candidates).symm
      _ = (candidates.take index).length +
          (candidates.drop index).length := List.length_append
  unfold FirstAccepted.Enough at enough
  refine ⟨positionLe, ?_⟩
  omega

/-! ## Production selector source interface -/

/-- Authoritative candidate values already materialized by the candidate
recipe.  Every field is a value identity; no row acceptance or desired selector
conclusion is carried here. -/
structure SourcesMatch
    (duplexBase u64Base candidateBase : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    {count : Nat} (coordinate : Fin count)
    (assignment : Nat → Nat)
    (candidates : List ProductionAlphabet.Chunk) : Prop where
  lengthExact : candidates.length = PiRlcCanonicalCandidates.candidatesPerScalar
  accept :
    ∀ candidate : Fin PiRlcCanonicalCandidates.candidatesPerScalar,
      assignment
          (acceptSource duplexBase u64Base candidateBase initialBuilder
            coordinate candidate) =
        if ProductionAlphabet.verifier.accepts
            (candidates.get
              ⟨candidate.val, by simpa [lengthExact] using candidate.isLt⟩)
          then 1 else 0
  symbol :
    ∀ candidate : Fin PiRlcCanonicalCandidates.candidatesPerScalar,
      assignment
          (symbolSource duplexBase u64Base candidateBase initialBuilder
            coordinate candidate) =
        (ProductionAlphabet.verifier.symbol
          (candidates.get
            ⟨candidate.val, by simpa [lengthExact] using candidate.isLt⟩)).val
  prefixExact :
    ∀ candidate : Fin PiRlcCanonicalCandidates.candidatesPerScalar,
      lcEval assignment
          (prefixSource duplexBase u64Base candidateBase initialBuilder
            coordinate candidate) =
        FirstAccepted.acceptedCount ProductionAlphabet.verifier
          (candidates.take candidate.val)
  finalCount :
    assignment
        (finalCountSource duplexBase u64Base candidateBase initialBuilder
          coordinate) =
      FirstAccepted.acceptedCount ProductionAlphabet.verifier candidates

def selectedIndex
    (candidates : List ProductionAlphabet.Chunk)
    (position : Fin outputCount) : Nat :=
  nthAcceptedIndex ProductionAlphabet.verifier candidates position.val

def selectedOffset
    (candidates : List ProductionAlphabet.Chunk)
    (lengthExact :
      candidates.length = PiRlcCanonicalCandidates.candidatesPerScalar)
    (enough :
      FirstAccepted.Enough ProductionAlphabet.verifier outputCount candidates)
    (position : Fin outputCount) : Fin selectionWindow :=
  ⟨selectedIndex candidates position - position.val, by
    have window :=
      nthAcceptedIndex_in_window ProductionAlphabet.verifier candidates
        (by simpa [PiRlcCanonicalCandidates.candidatesPerScalar] using
          lengthExact)
        (by simpa [outputCount] using enough)
        position.val (by simpa [outputCount] using position.isLt)
    simp only [selectedIndex]
    simp only [selectionWindow]
    omega⟩

theorem selectedOffset_source
    (candidates : List ProductionAlphabet.Chunk)
    (lengthExact :
      candidates.length = PiRlcCanonicalCandidates.candidatesPerScalar)
    (enough :
      FirstAccepted.Enough ProductionAlphabet.verifier outputCount candidates)
    (position : Fin outputCount) :
    position.val + (selectedOffset candidates lengthExact enough position).val =
      selectedIndex candidates position := by
  have window :=
    nthAcceptedIndex_in_window ProductionAlphabet.verifier candidates
      (by simpa [PiRlcCanonicalCandidates.candidatesPerScalar] using
        lengthExact)
      (by simpa [outputCount] using enough)
      position.val (by simpa [outputCount] using position.isLt)
  simp only [selectedOffset, selectedIndex]
  omega

def selectedCandidate
    (candidates : List ProductionAlphabet.Chunk)
    (lengthExact :
      candidates.length = PiRlcCanonicalCandidates.candidatesPerScalar)
    (enough :
      FirstAccepted.Enough ProductionAlphabet.verifier outputCount candidates)
    (position : Fin outputCount) :
    Fin PiRlcCanonicalCandidates.candidatesPerScalar :=
  candidateAt position (selectedOffset candidates lengthExact enough position)

theorem selectedCandidate_val
    (candidates : List ProductionAlphabet.Chunk)
    (lengthExact :
      candidates.length = PiRlcCanonicalCandidates.candidatesPerScalar)
    (enough :
      FirstAccepted.Enough ProductionAlphabet.verifier outputCount candidates)
    (position : Fin outputCount) :
    (selectedCandidate candidates lengthExact enough position).val =
      selectedIndex candidates position := by
  exact selectedOffset_source candidates lengthExact enough position

theorem selectedCandidate_spec
    (candidates : List ProductionAlphabet.Chunk)
    (lengthExact :
      candidates.length = PiRlcCanonicalCandidates.candidatesPerScalar)
    (enough :
      FirstAccepted.Enough ProductionAlphabet.verifier outputCount candidates)
    (position : Fin outputCount) :
    ProductionAlphabet.verifier.accepts
        (candidates.get
          ⟨(selectedCandidate candidates lengthExact enough position).val,
            by simpa [lengthExact] using
              (selectedCandidate candidates lengthExact enough position).isLt⟩) =
      true ∧
    FirstAccepted.acceptedCount ProductionAlphabet.verifier
        (candidates.take
          (selectedCandidate candidates lengthExact enough position).val) =
      position.val := by
  have available :
      position.val <
        FirstAccepted.acceptedCount ProductionAlphabet.verifier candidates := by
    unfold FirstAccepted.Enough at enough
    have positionLt := position.isLt
    simp only [outputCount] at enough positionLt
    omega
  obtain ⟨bounded, accepted, before⟩ :=
    nthAcceptedIndex_spec ProductionAlphabet.verifier candidates position.val
      available
  have value :=
    selectedCandidate_val candidates lengthExact enough position
  constructor
  · simpa [value] using accepted
  · simpa [value] using before

def slackValue (candidates : List ProductionAlphabet.Chunk) : Nat :=
  FirstAccepted.acceptedCount ProductionAlphabet.verifier candidates -
    outputCount

def slackBit (candidates : List ProductionAlphabet.Chunk)
    (offset : Nat) : Nat :=
  slackValue candidates / 2 ^ offset % 2

def selectorValue
    (candidates : List ProductionAlphabet.Chunk)
    (lengthExact :
      candidates.length = PiRlcCanonicalCandidates.candidatesPerScalar)
    (enough :
      FirstAccepted.Enough ProductionAlphabet.verifier outputCount candidates)
    (position : Fin outputCount) (offset : Fin selectionWindow) : Nat :=
  if offset = selectedOffset candidates lengthExact enough position then 1
  else 0

/-! ## Concrete selector assignment -/

def scalarLocalValue
    (duplexBase u64Base candidateBase : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    {count : Nat} (coordinate : Fin count)
    (assignment : Nat → Nat)
    (candidates : List ProductionAlphabet.Chunk)
    (lengthExact :
      candidates.length = PiRlcCanonicalCandidates.candidatesPerScalar)
    (enough :
      FirstAccepted.Enough ProductionAlphabet.verifier outputCount candidates)
    (localOffset : Nat) : Nat :=
  if localOffset = 0 then
    slackValue candidates
  else if localOffset < 5 then
    slackBit candidates (localOffset - 1)
  else
    let positionNat := (localOffset - 5) / positionAuxiliaryCount
    let position : Fin outputCount :=
      ⟨positionNat % outputCount, Nat.mod_lt _ (by decide)⟩
    let within := (localOffset - 5) % positionAuxiliaryCount
    if within < selectionWindow then
      let offset : Fin selectionWindow :=
        ⟨within % selectionWindow, Nat.mod_lt _ (by decide)⟩
      selectorValue candidates lengthExact enough position offset
    else if within = 44 then
      (assignment
          (symbolSource duplexBase u64Base candidateBase initialBuilder
            coordinate
            (selectedCandidate candidates lengthExact enough position)) +
        (goldilocksP - 2)) % goldilocksP
    else
      let productIndex := within - 11
      let offset : Fin selectionWindow :=
        ⟨(productIndex / 3) % selectionWindow,
          Nat.mod_lt _ (by decide)⟩
      let selected :=
        selectorValue candidates lengthExact enough position offset
      let candidate := candidateAt position offset
      match productIndex % 3 with
      | 0 =>
          selected *
            assignment
              (symbolSource duplexBase u64Base candidateBase initialBuilder
                coordinate candidate) % goldilocksP
      | 1 =>
          selected *
            assignment
              (acceptSource duplexBase u64Base candidateBase initialBuilder
                coordinate candidate) % goldilocksP
      | _ =>
          selected *
            lcEval assignment
              (prefixSource duplexBase u64Base candidateBase initialBuilder
                coordinate candidate) % goldilocksP

/-- Honest assignment for one scalar selector block.  Exactly the contiguous
2,435-column selector allocation is overwritten. -/
def scalarWitness
    (duplexBase u64Base candidateBase selectorBase : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    {count : Nat} (coordinate : Fin count)
    (assignment : Nat → Nat)
    (candidates : List ProductionAlphabet.Chunk)
    (lengthExact :
      candidates.length = PiRlcCanonicalCandidates.candidatesPerScalar)
    (enough :
      FirstAccepted.Enough ProductionAlphabet.verifier outputCount candidates) :
    Nat → Nat :=
  fun column =>
    if below : column < scalarBase selectorBase coordinate then
      assignment column
    else if within :
        column <
          scalarBase selectorBase coordinate + scalarAuxiliaryCount then
      scalarLocalValue duplexBase u64Base candidateBase initialBuilder
        coordinate assignment candidates lengthExact enough
        (column - scalarBase selectorBase coordinate)
    else
      assignment column

theorem scalarWitness_before
    (duplexBase u64Base candidateBase selectorBase : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    {count : Nat} (coordinate : Fin count)
    (assignment : Nat → Nat)
    (candidates : List ProductionAlphabet.Chunk)
    (lengthExact :
      candidates.length = PiRlcCanonicalCandidates.candidatesPerScalar)
    (enough :
      FirstAccepted.Enough ProductionAlphabet.verifier outputCount candidates)
    {column : Nat} (before : column < scalarBase selectorBase coordinate) :
    scalarWitness duplexBase u64Base candidateBase selectorBase initialBuilder
      coordinate assignment candidates lengthExact enough column =
        assignment column := by
  simp [scalarWitness, before]

@[simp] theorem scalarWitness_slack
    (duplexBase u64Base candidateBase selectorBase : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    {count : Nat} (coordinate : Fin count)
    (assignment : Nat → Nat)
    (candidates : List ProductionAlphabet.Chunk)
    (lengthExact :
      candidates.length = PiRlcCanonicalCandidates.candidatesPerScalar)
    (enough :
      FirstAccepted.Enough ProductionAlphabet.verifier outputCount candidates) :
    scalarWitness duplexBase u64Base candidateBase selectorBase initialBuilder
        coordinate assignment candidates lengthExact enough
        (slackColumn selectorBase coordinate) =
      slackValue candidates := by
  unfold scalarWitness slackColumn
  rw [dif_neg (by omega), dif_pos (by
    simp only [scalarAuxiliaryCount, outputCount, positionAuxiliaryCount]
    omega)]
  simp [scalarLocalValue]

@[simp] theorem scalarWitness_slackBit
    (duplexBase u64Base candidateBase selectorBase : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    {count : Nat} (coordinate : Fin count)
    (assignment : Nat → Nat)
    (candidates : List ProductionAlphabet.Chunk)
    (lengthExact :
      candidates.length = PiRlcCanonicalCandidates.candidatesPerScalar)
    (enough :
      FirstAccepted.Enough ProductionAlphabet.verifier outputCount candidates)
    {offset : Nat} (offsetLt : offset < slackBitCount) :
    scalarWitness duplexBase u64Base candidateBase selectorBase initialBuilder
        coordinate assignment candidates lengthExact enough
        (slackBitColumn selectorBase coordinate offset) =
      slackBit candidates offset := by
  unfold scalarWitness slackBitColumn
  rw [dif_neg (by omega), dif_pos (by
    simp only [scalarAuxiliaryCount, outputCount, positionAuxiliaryCount,
      slackBitCount] at offsetLt ⊢
    omega)]
  unfold scalarLocalValue
  rw [if_neg (by
    simp only [slackBitCount] at offsetLt
    omega)]
  rw [if_pos (by
    simp only [slackBitCount] at offsetLt
    omega)]
  congr 1
  omega

private theorem position_block_div
    (position within : Nat) (withinLt : within < positionAuxiliaryCount) :
    (position * positionAuxiliaryCount + within) /
        positionAuxiliaryCount =
      position := by
  rw [Nat.mul_comm position positionAuxiliaryCount,
    Nat.mul_add_div (by decide : 0 < positionAuxiliaryCount),
    Nat.div_eq_of_lt withinLt, Nat.add_zero]

private theorem position_block_mod
    (position within : Nat) (withinLt : within < positionAuxiliaryCount) :
    (position * positionAuxiliaryCount + within) %
        positionAuxiliaryCount =
      within :=
  Nat.mul_add_mod_of_lt withinLt

@[simp] theorem scalarWitness_selector
    (duplexBase u64Base candidateBase selectorBase : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    {count : Nat} (coordinate : Fin count)
    (assignment : Nat → Nat)
    (candidates : List ProductionAlphabet.Chunk)
    (lengthExact :
      candidates.length = PiRlcCanonicalCandidates.candidatesPerScalar)
    (enough :
      FirstAccepted.Enough ProductionAlphabet.verifier outputCount candidates)
    (position : Fin outputCount) (offset : Fin selectionWindow) :
    scalarWitness duplexBase u64Base candidateBase selectorBase initialBuilder
        coordinate assignment candidates lengthExact enough
        (selectorColumn selectorBase coordinate position offset) =
      selectorValue candidates lengthExact enough position offset := by
  have positionLt := position.isLt
  have offsetLt := offset.isLt
  simp only [outputCount] at positionLt
  simp only [selectionWindow] at offsetLt
  unfold scalarWitness
  rw [dif_neg (by
    unfold selectorColumn positionBase
    omega)]
  rw [dif_pos (by
    unfold selectorColumn positionBase
    simp only [scalarAuxiliaryCount, outputCount, positionAuxiliaryCount]
    omega)]
  have localEq :
      selectorColumn selectorBase coordinate position offset -
          scalarBase selectorBase coordinate =
        5 + position.val * positionAuxiliaryCount + offset.val := by
    unfold selectorColumn positionBase
    omega
  rw [localEq]
  unfold scalarLocalValue
  rw [if_neg (by omega), if_neg (by omega)]
  have withinLt : offset.val < positionAuxiliaryCount := by
    simp only [positionAuxiliaryCount]
    omega
  have divEq :
      (5 + position.val * positionAuxiliaryCount + offset.val - 5) /
          positionAuxiliaryCount = position.val := by
    rw [show
      5 + position.val * positionAuxiliaryCount + offset.val - 5 =
        position.val * positionAuxiliaryCount + offset.val by omega]
    exact position_block_div position.val offset.val withinLt
  have modEq :
      (5 + position.val * positionAuxiliaryCount + offset.val - 5) %
          positionAuxiliaryCount = offset.val := by
    rw [show
      5 + position.val * positionAuxiliaryCount + offset.val - 5 =
        position.val * positionAuxiliaryCount + offset.val by omega]
    exact position_block_mod position.val offset.val withinLt
  simp only [divEq, modEq, Nat.mod_eq_of_lt position.isLt,
    Nat.mod_eq_of_lt offset.isLt]
  rw [if_pos offset.isLt]

@[simp] theorem scalarWitness_symbolProduct
    (duplexBase u64Base candidateBase selectorBase : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    {count : Nat} (coordinate : Fin count)
    (assignment : Nat → Nat)
    (candidates : List ProductionAlphabet.Chunk)
    (lengthExact :
      candidates.length = PiRlcCanonicalCandidates.candidatesPerScalar)
    (enough :
      FirstAccepted.Enough ProductionAlphabet.verifier outputCount candidates)
    (position : Fin outputCount) (offset : Fin selectionWindow) :
    scalarWitness duplexBase u64Base candidateBase selectorBase initialBuilder
        coordinate assignment candidates lengthExact enough
        (symbolProductColumn selectorBase coordinate position offset) =
      selectorValue candidates lengthExact enough position offset *
        assignment
          (symbolSource duplexBase u64Base candidateBase initialBuilder
            coordinate (candidateAt position offset)) % goldilocksP := by
  have positionLt := position.isLt
  have offsetLt := offset.isLt
  simp only [outputCount] at positionLt
  simp only [selectionWindow] at offsetLt
  unfold scalarWitness
  rw [dif_neg (by
    unfold symbolProductColumn positionBase
    omega)]
  rw [dif_pos (by
    unfold symbolProductColumn positionBase
    simp only [scalarAuxiliaryCount, outputCount, positionAuxiliaryCount]
    omega)]
  have localEq :
      symbolProductColumn selectorBase coordinate position offset -
          scalarBase selectorBase coordinate =
        5 + position.val * positionAuxiliaryCount +
          (11 + 3 * offset.val) := by
    unfold symbolProductColumn positionBase
    omega
  rw [localEq]
  unfold scalarLocalValue
  rw [if_neg (by omega), if_neg (by omega)]
  have withinLt :
      11 + 3 * offset.val < positionAuxiliaryCount := by
    simp only [positionAuxiliaryCount]
    omega
  have divEq :
      (5 + position.val * positionAuxiliaryCount +
            (11 + 3 * offset.val) - 5) /
          positionAuxiliaryCount =
        position.val := by
    rw [show
      5 + position.val * positionAuxiliaryCount +
            (11 + 3 * offset.val) - 5 =
        position.val * positionAuxiliaryCount +
          (11 + 3 * offset.val) by omega]
    exact position_block_div position.val _ withinLt
  have modEq :
      (5 + position.val * positionAuxiliaryCount +
            (11 + 3 * offset.val) - 5) %
          positionAuxiliaryCount =
        11 + 3 * offset.val := by
    rw [show
      5 + position.val * positionAuxiliaryCount +
            (11 + 3 * offset.val) - 5 =
        position.val * positionAuxiliaryCount +
          (11 + 3 * offset.val) by omega]
    exact position_block_mod position.val _ withinLt
  simp only [divEq, modEq, Nat.mod_eq_of_lt position.isLt]
  rw [if_neg (by
    simp only [selectionWindow]
    omega)]
  rw [if_neg (by omega)]
  have productEq : 11 + 3 * offset.val - 11 = 3 * offset.val := by
    omega
  simp only [productEq]
  have offsetDiv :
      3 * offset.val / 3 % selectionWindow = offset.val := by
    have divided : 3 * offset.val / 3 = offset.val := by
      omega
    rw [divided, Nat.mod_eq_of_lt offset.isLt]
  simp only [Nat.mul_mod_right, offsetDiv]

@[simp] theorem scalarWitness_acceptProduct
    (duplexBase u64Base candidateBase selectorBase : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    {count : Nat} (coordinate : Fin count)
    (assignment : Nat → Nat)
    (candidates : List ProductionAlphabet.Chunk)
    (lengthExact :
      candidates.length = PiRlcCanonicalCandidates.candidatesPerScalar)
    (enough :
      FirstAccepted.Enough ProductionAlphabet.verifier outputCount candidates)
    (position : Fin outputCount) (offset : Fin selectionWindow) :
    scalarWitness duplexBase u64Base candidateBase selectorBase initialBuilder
        coordinate assignment candidates lengthExact enough
        (acceptProductColumn selectorBase coordinate position offset) =
      selectorValue candidates lengthExact enough position offset *
        assignment
          (acceptSource duplexBase u64Base candidateBase initialBuilder
            coordinate (candidateAt position offset)) % goldilocksP := by
  have positionLt := position.isLt
  have offsetLt := offset.isLt
  simp only [outputCount] at positionLt
  simp only [selectionWindow] at offsetLt
  unfold scalarWitness
  rw [dif_neg (by
    unfold acceptProductColumn symbolProductColumn positionBase
    omega)]
  rw [dif_pos (by
    unfold acceptProductColumn symbolProductColumn positionBase
    simp only [scalarAuxiliaryCount, outputCount, positionAuxiliaryCount]
    omega)]
  have localEq :
      acceptProductColumn selectorBase coordinate position offset -
          scalarBase selectorBase coordinate =
        5 + position.val * positionAuxiliaryCount +
          (12 + 3 * offset.val) := by
    unfold acceptProductColumn symbolProductColumn positionBase
    omega
  rw [localEq]
  unfold scalarLocalValue
  rw [if_neg (by omega), if_neg (by omega)]
  have withinLt :
      12 + 3 * offset.val < positionAuxiliaryCount := by
    simp only [positionAuxiliaryCount]
    omega
  have divEq :
      (5 + position.val * positionAuxiliaryCount +
            (12 + 3 * offset.val) - 5) /
          positionAuxiliaryCount =
        position.val := by
    rw [show
      5 + position.val * positionAuxiliaryCount +
            (12 + 3 * offset.val) - 5 =
        position.val * positionAuxiliaryCount +
          (12 + 3 * offset.val) by omega]
    exact position_block_div position.val _ withinLt
  have modEq :
      (5 + position.val * positionAuxiliaryCount +
            (12 + 3 * offset.val) - 5) %
          positionAuxiliaryCount =
        12 + 3 * offset.val := by
    rw [show
      5 + position.val * positionAuxiliaryCount +
            (12 + 3 * offset.val) - 5 =
        position.val * positionAuxiliaryCount +
          (12 + 3 * offset.val) by omega]
    exact position_block_mod position.val _ withinLt
  simp only [divEq, modEq, Nat.mod_eq_of_lt position.isLt]
  rw [if_neg (by
    simp only [selectionWindow]
    omega)]
  rw [if_neg (by omega)]
  have productEq : 12 + 3 * offset.val - 11 =
      3 * offset.val + 1 := by
    omega
  simp only [productEq]
  have offsetDiv :
      (3 * offset.val + 1) / 3 % selectionWindow = offset.val := by
    have divided : (3 * offset.val + 1) / 3 = offset.val := by
      omega
    rw [divided, Nat.mod_eq_of_lt offset.isLt]
  have branch : (3 * offset.val + 1) % 3 = 1 := by
    omega
  simp only [branch, offsetDiv]

@[simp] theorem scalarWitness_prefixProduct
    (duplexBase u64Base candidateBase selectorBase : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    {count : Nat} (coordinate : Fin count)
    (assignment : Nat → Nat)
    (candidates : List ProductionAlphabet.Chunk)
    (lengthExact :
      candidates.length = PiRlcCanonicalCandidates.candidatesPerScalar)
    (enough :
      FirstAccepted.Enough ProductionAlphabet.verifier outputCount candidates)
    (position : Fin outputCount) (offset : Fin selectionWindow) :
    scalarWitness duplexBase u64Base candidateBase selectorBase initialBuilder
        coordinate assignment candidates lengthExact enough
        (prefixProductColumn selectorBase coordinate position offset) =
      selectorValue candidates lengthExact enough position offset *
        lcEval assignment
          (prefixSource duplexBase u64Base candidateBase initialBuilder
            coordinate (candidateAt position offset)) % goldilocksP := by
  have positionLt := position.isLt
  have offsetLt := offset.isLt
  simp only [outputCount] at positionLt
  simp only [selectionWindow] at offsetLt
  unfold scalarWitness
  rw [dif_neg (by
    unfold prefixProductColumn symbolProductColumn positionBase
    omega)]
  rw [dif_pos (by
    unfold prefixProductColumn symbolProductColumn positionBase
    simp only [scalarAuxiliaryCount, outputCount, positionAuxiliaryCount]
    omega)]
  have localEq :
      prefixProductColumn selectorBase coordinate position offset -
          scalarBase selectorBase coordinate =
        5 + position.val * positionAuxiliaryCount +
          (13 + 3 * offset.val) := by
    unfold prefixProductColumn symbolProductColumn positionBase
    omega
  rw [localEq]
  unfold scalarLocalValue
  rw [if_neg (by omega), if_neg (by omega)]
  have withinLt :
      13 + 3 * offset.val < positionAuxiliaryCount := by
    simp only [positionAuxiliaryCount]
    omega
  have divEq :
      (5 + position.val * positionAuxiliaryCount +
            (13 + 3 * offset.val) - 5) /
          positionAuxiliaryCount =
        position.val := by
    rw [show
      5 + position.val * positionAuxiliaryCount +
            (13 + 3 * offset.val) - 5 =
        position.val * positionAuxiliaryCount +
          (13 + 3 * offset.val) by omega]
    exact position_block_div position.val _ withinLt
  have modEq :
      (5 + position.val * positionAuxiliaryCount +
            (13 + 3 * offset.val) - 5) %
          positionAuxiliaryCount =
        13 + 3 * offset.val := by
    rw [show
      5 + position.val * positionAuxiliaryCount +
            (13 + 3 * offset.val) - 5 =
        position.val * positionAuxiliaryCount +
          (13 + 3 * offset.val) by omega]
    exact position_block_mod position.val _ withinLt
  simp only [divEq, modEq, Nat.mod_eq_of_lt position.isLt]
  rw [if_neg (by
    simp only [selectionWindow]
    omega)]
  rw [if_neg (by omega)]
  have productEq : 13 + 3 * offset.val - 11 =
      3 * offset.val + 2 := by
    omega
  simp only [productEq]
  have offsetDiv :
      (3 * offset.val + 2) / 3 % selectionWindow = offset.val := by
    have divided : (3 * offset.val + 2) / 3 = offset.val := by
      omega
    rw [divided, Nat.mod_eq_of_lt offset.isLt]
  have branch : (3 * offset.val + 2) % 3 = 2 := by
    omega
  simp only [branch, offsetDiv]

@[simp] theorem scalarWitness_output
    (duplexBase u64Base candidateBase selectorBase : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    {count : Nat} (coordinate : Fin count)
    (assignment : Nat → Nat)
    (candidates : List ProductionAlphabet.Chunk)
    (lengthExact :
      candidates.length = PiRlcCanonicalCandidates.candidatesPerScalar)
    (enough :
      FirstAccepted.Enough ProductionAlphabet.verifier outputCount candidates)
    (position : Fin outputCount) :
    scalarWitness duplexBase u64Base candidateBase selectorBase initialBuilder
        coordinate assignment candidates lengthExact enough
        (outputColumn selectorBase coordinate position) =
      (assignment
          (symbolSource duplexBase u64Base candidateBase initialBuilder
            coordinate
            (selectedCandidate candidates lengthExact enough position)) +
        (goldilocksP - 2)) % goldilocksP := by
  have positionLt := position.isLt
  simp only [outputCount] at positionLt
  unfold scalarWitness
  rw [dif_neg (by
    unfold outputColumn positionBase
    omega)]
  rw [dif_pos (by
    unfold outputColumn positionBase
    simp only [scalarAuxiliaryCount, outputCount, positionAuxiliaryCount]
    omega)]
  have localEq :
      outputColumn selectorBase coordinate position -
          scalarBase selectorBase coordinate =
        5 + position.val * positionAuxiliaryCount + 44 := by
    unfold outputColumn positionBase
    omega
  rw [localEq]
  unfold scalarLocalValue
  rw [if_neg (by omega), if_neg (by omega)]
  have withinLt : 44 < positionAuxiliaryCount := by decide
  have divEq :
      (5 + position.val * positionAuxiliaryCount + 44 - 5) /
          positionAuxiliaryCount =
        position.val := by
    rw [show
      5 + position.val * positionAuxiliaryCount + 44 - 5 =
        position.val * positionAuxiliaryCount + 44 by omega]
    exact position_block_div position.val 44 withinLt
  have modEq :
      (5 + position.val * positionAuxiliaryCount + 44 - 5) %
          positionAuxiliaryCount =
        44 := by
    rw [show
      5 + position.val * positionAuxiliaryCount + 44 - 5 =
        position.val * positionAuxiliaryCount + 44 by omega]
    exact position_block_mod position.val 44 withinLt
  simp only [divEq, modEq, Nat.mod_eq_of_lt position.isLt]
  rw [if_neg (by decide)]
  simp only [if_true]

/-! ## Preservation of caller-owned candidate columns -/

/-- Every candidate value read by the selector lies before its fresh block.
This is the exact freshness contract needed by the honest witness. -/
structure SourcesBelow
    (duplexBase u64Base candidateBase selectorBase : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    {count : Nat} (coordinate : Fin count) : Prop where
  accept :
    ∀ candidate : Fin PiRlcCanonicalCandidates.candidatesPerScalar,
      acceptSource duplexBase u64Base candidateBase initialBuilder coordinate
          candidate <
        scalarBase selectorBase coordinate
  symbol :
    ∀ candidate : Fin PiRlcCanonicalCandidates.candidatesPerScalar,
      symbolSource duplexBase u64Base candidateBase initialBuilder coordinate
          candidate <
        scalarBase selectorBase coordinate
  prefixRead :
    ∀ (candidate : Fin PiRlcCanonicalCandidates.candidatesPerScalar)
      column,
      Mentions
          (prefixSource duplexBase u64Base candidateBase initialBuilder
            coordinate candidate)
          column →
        column < scalarBase selectorBase coordinate
  finalCount :
    finalCountSource duplexBase u64Base candidateBase initialBuilder
        coordinate <
      scalarBase selectorBase coordinate

theorem scalarWitness_constant
    (duplexBase u64Base candidateBase selectorBase : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    {count : Nat} (coordinate : Fin count)
    (assignment : Nat → Nat)
    (candidates : List ProductionAlphabet.Chunk)
    (lengthExact :
      candidates.length = PiRlcCanonicalCandidates.candidatesPerScalar)
    (enough :
      FirstAccepted.Enough ProductionAlphabet.verifier outputCount candidates)
    (positive : 0 < scalarBase selectorBase coordinate)
    (constantWire : assignment 0 = 1) :
    scalarWitness duplexBase u64Base candidateBase selectorBase initialBuilder
        coordinate assignment candidates lengthExact enough 0 =
      1 := by
  rw [scalarWitness_before duplexBase u64Base candidateBase selectorBase
    initialBuilder coordinate assignment candidates lengthExact enough
    positive]
  exact constantWire

theorem scalarWitness_acceptSource
    (duplexBase u64Base candidateBase selectorBase : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    {count : Nat} (coordinate : Fin count)
    (assignment : Nat → Nat)
    (candidates : List ProductionAlphabet.Chunk)
    (lengthExact :
      candidates.length = PiRlcCanonicalCandidates.candidatesPerScalar)
    (enough :
      FirstAccepted.Enough ProductionAlphabet.verifier outputCount candidates)
    (below :
      SourcesBelow duplexBase u64Base candidateBase selectorBase
        initialBuilder coordinate)
    (candidate : Fin PiRlcCanonicalCandidates.candidatesPerScalar) :
    scalarWitness duplexBase u64Base candidateBase selectorBase initialBuilder
        coordinate assignment candidates lengthExact enough
        (acceptSource duplexBase u64Base candidateBase initialBuilder
          coordinate candidate) =
      assignment
        (acceptSource duplexBase u64Base candidateBase initialBuilder
          coordinate candidate) :=
  scalarWitness_before duplexBase u64Base candidateBase selectorBase
    initialBuilder coordinate assignment candidates lengthExact enough
    (below.accept candidate)

theorem scalarWitness_symbolSource
    (duplexBase u64Base candidateBase selectorBase : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    {count : Nat} (coordinate : Fin count)
    (assignment : Nat → Nat)
    (candidates : List ProductionAlphabet.Chunk)
    (lengthExact :
      candidates.length = PiRlcCanonicalCandidates.candidatesPerScalar)
    (enough :
      FirstAccepted.Enough ProductionAlphabet.verifier outputCount candidates)
    (below :
      SourcesBelow duplexBase u64Base candidateBase selectorBase
        initialBuilder coordinate)
    (candidate : Fin PiRlcCanonicalCandidates.candidatesPerScalar) :
    scalarWitness duplexBase u64Base candidateBase selectorBase initialBuilder
        coordinate assignment candidates lengthExact enough
        (symbolSource duplexBase u64Base candidateBase initialBuilder
          coordinate candidate) =
      assignment
        (symbolSource duplexBase u64Base candidateBase initialBuilder
          coordinate candidate) :=
  scalarWitness_before duplexBase u64Base candidateBase selectorBase
    initialBuilder coordinate assignment candidates lengthExact enough
    (below.symbol candidate)

theorem scalarWitness_prefixSource
    (duplexBase u64Base candidateBase selectorBase : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    {count : Nat} (coordinate : Fin count)
    (assignment : Nat → Nat)
    (candidates : List ProductionAlphabet.Chunk)
    (lengthExact :
      candidates.length = PiRlcCanonicalCandidates.candidatesPerScalar)
    (enough :
      FirstAccepted.Enough ProductionAlphabet.verifier outputCount candidates)
    (below :
      SourcesBelow duplexBase u64Base candidateBase selectorBase
        initialBuilder coordinate)
    (candidate : Fin PiRlcCanonicalCandidates.candidatesPerScalar) :
    lcEval
        (scalarWitness duplexBase u64Base candidateBase selectorBase
          initialBuilder coordinate assignment candidates lengthExact enough)
        (prefixSource duplexBase u64Base candidateBase initialBuilder
          coordinate candidate) =
      lcEval assignment
        (prefixSource duplexBase u64Base candidateBase initialBuilder
          coordinate candidate) := by
  apply Nightstream.Implementation.R1CS.Canonical.KMulHonest.lcEval_congr
  intro column mentioned
  exact scalarWitness_before duplexBase u64Base candidateBase selectorBase
    initialBuilder coordinate assignment candidates lengthExact enough
    (below.prefixRead candidate column mentioned)

theorem scalarWitness_finalCountSource
    (duplexBase u64Base candidateBase selectorBase : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    {count : Nat} (coordinate : Fin count)
    (assignment : Nat → Nat)
    (candidates : List ProductionAlphabet.Chunk)
    (lengthExact :
      candidates.length = PiRlcCanonicalCandidates.candidatesPerScalar)
    (enough :
      FirstAccepted.Enough ProductionAlphabet.verifier outputCount candidates)
    (below :
      SourcesBelow duplexBase u64Base candidateBase selectorBase
        initialBuilder coordinate) :
    scalarWitness duplexBase u64Base candidateBase selectorBase initialBuilder
        coordinate assignment candidates lengthExact enough
        (finalCountSource duplexBase u64Base candidateBase initialBuilder
          coordinate) =
      assignment
        (finalCountSource duplexBase u64Base candidateBase initialBuilder
          coordinate) :=
  scalarWitness_before duplexBase u64Base candidateBase selectorBase
    initialBuilder coordinate assignment candidates lengthExact enough
    below.finalCount

theorem SourcesMatch.preserved
    (duplexBase u64Base candidateBase selectorBase : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    {count : Nat} (coordinate : Fin count)
    (assignment : Nat → Nat)
    (candidates : List ProductionAlphabet.Chunk)
    (sources :
      SourcesMatch duplexBase u64Base candidateBase initialBuilder coordinate
        assignment candidates)
    (enough :
      FirstAccepted.Enough ProductionAlphabet.verifier outputCount candidates)
    (below :
      SourcesBelow duplexBase u64Base candidateBase selectorBase
        initialBuilder coordinate) :
    SourcesMatch duplexBase u64Base candidateBase initialBuilder coordinate
      (scalarWitness duplexBase u64Base candidateBase selectorBase
        initialBuilder coordinate assignment candidates sources.lengthExact
          enough)
      candidates where
  lengthExact := sources.lengthExact
  accept candidate := by
    rw [scalarWitness_acceptSource duplexBase u64Base candidateBase
      selectorBase initialBuilder coordinate assignment candidates
      sources.lengthExact enough below candidate]
    exact sources.accept candidate
  symbol candidate := by
    rw [scalarWitness_symbolSource duplexBase u64Base candidateBase
      selectorBase initialBuilder coordinate assignment candidates
      sources.lengthExact enough below candidate]
    exact sources.symbol candidate
  prefixExact candidate := by
    rw [scalarWitness_prefixSource duplexBase u64Base candidateBase
      selectorBase initialBuilder coordinate assignment candidates
      sources.lengthExact enough below candidate]
    exact sources.prefixExact candidate
  finalCount := by
    rw [scalarWitness_finalCountSource duplexBase u64Base candidateBase
      selectorBase initialBuilder coordinate assignment candidates
      sources.lengthExact enough below]
    exact sources.finalCount

/-! ## Exact witness bounds -/

theorem slackValue_le_ten
    (candidates : List ProductionAlphabet.Chunk)
    (lengthExact :
      candidates.length = PiRlcCanonicalCandidates.candidatesPerScalar) :
    slackValue candidates ≤ 10 := by
  have acceptedBound :=
    acceptedCount_le_length ProductionAlphabet.verifier candidates
  have exactLength : candidates.length = 64 := by
    simpa [PiRlcCanonicalCandidates.candidatesPerScalar] using lengthExact
  rw [exactLength] at acceptedBound
  unfold slackValue
  simp only [outputCount]
  omega

theorem slackBit_le_one
    (candidates : List ProductionAlphabet.Chunk) (offset : Nat) :
    slackBit candidates offset ≤ 1 := by
  unfold slackBit
  have bounded := Nat.mod_lt (slackValue candidates / 2 ^ offset)
    (by decide : 0 < 2)
  omega

theorem selectorValue_le_one
    (candidates : List ProductionAlphabet.Chunk)
    (lengthExact :
      candidates.length = PiRlcCanonicalCandidates.candidatesPerScalar)
    (enough :
      FirstAccepted.Enough ProductionAlphabet.verifier outputCount candidates)
    (position : Fin outputCount) (offset : Fin selectionWindow) :
    selectorValue candidates lengthExact enough position offset ≤ 1 := by
  unfold selectorValue
  split <;> omega

@[simp] theorem selectorValue_selected
    (candidates : List ProductionAlphabet.Chunk)
    (lengthExact :
      candidates.length = PiRlcCanonicalCandidates.candidatesPerScalar)
    (enough :
      FirstAccepted.Enough ProductionAlphabet.verifier outputCount candidates)
    (position : Fin outputCount) :
    selectorValue candidates lengthExact enough position
        (selectedOffset candidates lengthExact enough position) =
      1 := by
  simp [selectorValue]

theorem selectorValue_unselected
    (candidates : List ProductionAlphabet.Chunk)
    (lengthExact :
      candidates.length = PiRlcCanonicalCandidates.candidatesPerScalar)
    (enough :
      FirstAccepted.Enough ProductionAlphabet.verifier outputCount candidates)
    (position : Fin outputCount) (offset : Fin selectionWindow)
    (different :
      offset ≠ selectedOffset candidates lengthExact enough position) :
    selectorValue candidates lengthExact enough position offset = 0 := by
  simp [selectorValue, different]

end Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalSelectorHonest
