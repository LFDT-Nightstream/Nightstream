import Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalSelectorSound
import Nightstream.SuperNeo.Sampling.FirstAccepted

/-!
Contract: semantic soundness of the complete Lean-owned 54-of-64 `Pi_RLC`
sampler slice.

The candidate vector is reconstructed from the canonical-u64 source bits.
Its accepted-prefix counts are defined independently through
`FirstAccepted.acceptedCount`.  The physical candidate counter and selector
rows are then refined to that independent vector.

This is a model-level canonical-encoding theorem.  It imports neither Rust
rows nor a generated artifact and makes no probability or cryptographic
security claim.
-/

set_option autoImplicit false
set_option maxRecDepth 100000

namespace Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalSamplerSound

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalCandidates
open Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalCandidatesSound
open Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalSelector
open Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalSelectorSound
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Sampling
open Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler

/-- The verifier-owned candidate reconstructed from one physical 16-bit source
slice. -/
def semanticCandidate
    (prime : EuclidPrime goldilocksP)
    (duplexBase u64Base candidateBase count : Nat)
    (initial : SymbolicDuplex.Builder)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (constantWire : assignment 0 = 1)
    (u64Satisfied :
      Satisfies (PiRlcCanonicalU64.rows duplexBase u64Base count initial)
        assignment)
    (coordinate : Fin count) (candidate : Fin candidatesPerScalar) :
    ProductionAlphabet.Chunk :=
  PiRlcCanonicalCandidateSound.candidate assignment
    (candidateLayout duplexBase u64Base candidateBase initial coordinate
      candidate)
    (sourceBitsBoolean prime duplexBase u64Base candidateBase count initial
      canonical constantWire u64Satisfied coordinate candidate)

/-- The exact 64-candidate vector in physical source order. -/
def semanticCandidates
    (prime : EuclidPrime goldilocksP)
    (duplexBase u64Base candidateBase count : Nat)
    (initial : SymbolicDuplex.Builder)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (constantWire : assignment 0 = 1)
    (u64Satisfied :
      Satisfies (PiRlcCanonicalU64.rows duplexBase u64Base count initial)
        assignment)
    (coordinate : Fin count) : List ProductionAlphabet.Chunk :=
  List.ofFn fun candidate =>
    semanticCandidate prime duplexBase u64Base candidateBase count initial
      canonical constantWire u64Satisfied coordinate candidate

@[simp] theorem semanticCandidates_length
    (prime : EuclidPrime goldilocksP)
    (duplexBase u64Base candidateBase count : Nat)
    (initial : SymbolicDuplex.Builder)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (constantWire : assignment 0 = 1)
    (u64Satisfied :
      Satisfies (PiRlcCanonicalU64.rows duplexBase u64Base count initial)
        assignment)
    (coordinate : Fin count) :
    (semanticCandidates prime duplexBase u64Base candidateBase count initial
      canonical constantWire u64Satisfied coordinate).length =
        candidatesPerScalar := by
  simp [semanticCandidates]

theorem semanticCandidates_get
    (prime : EuclidPrime goldilocksP)
    (duplexBase u64Base candidateBase count : Nat)
    (initial : SymbolicDuplex.Builder)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (constantWire : assignment 0 = 1)
    (u64Satisfied :
      Satisfies (PiRlcCanonicalU64.rows duplexBase u64Base count initial)
        assignment)
    (coordinate : Fin count) (candidate : Fin candidatesPerScalar) :
    (semanticCandidates prime duplexBase u64Base candidateBase count initial
      canonical constantWire u64Satisfied coordinate).get
        ⟨candidate.val, by
          simpa using candidate.isLt⟩ =
      semanticCandidate prime duplexBase u64Base candidateBase count initial
        canonical constantWire u64Satisfied coordinate candidate := by
  simp [semanticCandidates]

/-- Independent mathematical accepted count before `index`. -/
def prefixCount
    (prime : EuclidPrime goldilocksP)
    (duplexBase u64Base candidateBase count : Nat)
    (initial : SymbolicDuplex.Builder)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (constantWire : assignment 0 = 1)
    (u64Satisfied :
      Satisfies (PiRlcCanonicalU64.rows duplexBase u64Base count initial)
        assignment)
    (coordinate : Fin count) (index : Nat) : Nat :=
  FirstAccepted.acceptedCount ProductionAlphabet.verifier
    ((semanticCandidates prime duplexBase u64Base candidateBase count initial
      canonical constantWire u64Satisfied coordinate).take index)

theorem prefixCount_succ
    (prime : EuclidPrime goldilocksP)
    (duplexBase u64Base candidateBase count : Nat)
    (initial : SymbolicDuplex.Builder)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (constantWire : assignment 0 = 1)
    (u64Satisfied :
      Satisfies (PiRlcCanonicalU64.rows duplexBase u64Base count initial)
        assignment)
    (coordinate : Fin count) (candidate : Fin candidatesPerScalar) :
    prefixCount prime duplexBase u64Base candidateBase count initial canonical
        constantWire u64Satisfied coordinate (candidate.val + 1) =
      prefixCount prime duplexBase u64Base candidateBase count initial canonical
          constantWire u64Satisfied coordinate candidate.val +
        (if ProductionAlphabet.verifier.accepts
            (semanticCandidate prime duplexBase u64Base candidateBase count
              initial canonical constantWire u64Satisfied coordinate candidate)
          then 1 else 0) := by
  rw [prefixCount, List.take_succ_eq_append_getElem (by
    simpa using candidate.isLt)]
  cases decision : ProductionAlphabet.verifier.accepts
      (semanticCandidate prime duplexBase u64Base candidateBase count initial
        canonical constantWire u64Satisfied coordinate candidate) <;>
    simp [prefixCount, FirstAccepted.acceptedCount,
      FirstAccepted.acceptedCandidates, semanticCandidates, decision]

theorem acceptWire_refines
    (prime : EuclidPrime goldilocksP)
    (duplexBase u64Base candidateBase count : Nat)
    (initial : SymbolicDuplex.Builder)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (constantWire : assignment 0 = 1)
    (u64Satisfied :
      Satisfies (PiRlcCanonicalU64.rows duplexBase u64Base count initial)
        assignment)
    (candidateSatisfied :
      Satisfies
        (PiRlcCanonicalCandidates.rows duplexBase u64Base candidateBase count
          initial)
        assignment)
    (coordinate : Fin count) (candidate : Fin candidatesPerScalar) :
    acceptWire duplexBase u64Base candidateBase initial coordinate assignment
        candidate =
      if ProductionAlphabet.verifier.accepts
          (semanticCandidate prime duplexBase u64Base candidateBase count
            initial canonical constantWire u64Satisfied coordinate candidate)
        then 1 else 0 := by
  let bits :=
    sourceBitsBoolean prime duplexBase u64Base candidateBase count initial
      canonical constantWire u64Satisfied coordinate candidate
  have exactAcceptance :=
    PiRlcCanonicalCandidateSound.acceptance_sound prime canonical constantWire
      bits
      (satisfies_candidate duplexBase u64Base candidateBase count initial
        assignment candidateSatisfied coordinate candidate)
  simpa [acceptWire, semanticCandidate, bits] using exactAcceptance

/-- The physical accepted-prefix recurrence is exactly the independent
`FirstAccepted` accepted count on every bounded prefix. -/
theorem acceptedPrefix_eq_prefixCount
    (prime : EuclidPrime goldilocksP)
    (duplexBase u64Base candidateBase count : Nat)
    (initial : SymbolicDuplex.Builder)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (constantWire : assignment 0 = 1)
    (u64Satisfied :
      Satisfies (PiRlcCanonicalU64.rows duplexBase u64Base count initial)
        assignment)
    (candidateSatisfied :
      Satisfies
        (PiRlcCanonicalCandidates.rows duplexBase u64Base candidateBase count
          initial)
        assignment)
    (coordinate : Fin count) :
    ∀ index, index ≤ candidatesPerScalar →
      acceptedPrefix duplexBase u64Base candidateBase initial coordinate
          assignment index =
        prefixCount prime duplexBase u64Base candidateBase count initial
          canonical constantWire u64Satisfied coordinate index := by
  intro index bounded
  induction index with
  | zero =>
      simp [acceptedPrefix, prefixCount, FirstAccepted.acceptedCount,
        FirstAccepted.acceptedCandidates]
  | succ index inductionHypothesis =>
      have indexLt : index < candidatesPerScalar := by omega
      let candidate : Fin candidatesPerScalar := ⟨index, indexLt⟩
      have candidateEq : candidateOfNat index = candidate := by
        apply Fin.ext
        exact candidateOfNat_val indexLt
      rw [acceptedPrefix, inductionHypothesis (by omega)]
      rw [prefixCount_succ prime duplexBase u64Base candidateBase count initial
        canonical constantWire u64Satisfied coordinate candidate]
      rw [candidateEq]
      rw [acceptWire_refines prime duplexBase u64Base candidateBase count
        initial canonical constantWire u64Satisfied candidateSatisfied
        coordinate candidate]

/-- The selector's exact final-count equation establishes mathematical
`Enough` for the independent candidate vector. -/
theorem enough
    (prime : EuclidPrime goldilocksP)
    (duplexBase u64Base candidateBase selectorBase count : Nat)
    (initial : SymbolicDuplex.Builder)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (constantWire : assignment 0 = 1)
    (u64Satisfied :
      Satisfies (PiRlcCanonicalU64.rows duplexBase u64Base count initial)
        assignment)
    (candidateSatisfied :
      Satisfies
        (PiRlcCanonicalCandidates.rows duplexBase u64Base candidateBase count
          initial)
        assignment)
    (selectorSatisfied :
      Satisfies
        (PiRlcCanonicalSelector.rows duplexBase u64Base candidateBase
          selectorBase count initial)
        assignment)
    (coordinate : Fin count) :
    FirstAccepted.Enough ProductionAlphabet.verifier outputCount
      (semanticCandidates prime duplexBase u64Base candidateBase count initial
        canonical constantWire u64Satisfied coordinate) := by
  let last : Fin candidatesPerScalar := ⟨63, by decide⟩
  have physicalEnough :=
    PiRlcCanonicalSelectorSound.enoughAccepted prime duplexBase u64Base
      candidateBase selectorBase count initial canonical constantWire
      selectorSatisfied coordinate
  have lastRefines :=
    candidate_refines prime duplexBase u64Base candidateBase count initial
      canonical constantWire u64Satisfied candidateSatisfied coordinate last
  rcases lastRefines with ⟨_, _, cumulative⟩
  have finalEq :
      assignment
          (finalCountSource duplexBase u64Base candidateBase initial
            coordinate) =
        acceptedPrefix duplexBase u64Base candidateBase initial coordinate
          assignment candidatesPerScalar := by
    simpa [finalCountSource, candidateSourceLayout, last,
      candidatesPerScalar] using cumulative
  rw [finalEq] at physicalEnough
  have prefixEq :=
    acceptedPrefix_eq_prefixCount prime duplexBase u64Base candidateBase count
      initial canonical constantWire u64Satisfied candidateSatisfied coordinate
      candidatesPerScalar (Nat.le_refl _)
  rw [prefixEq] at physicalEnough
  unfold FirstAccepted.Enough
  simpa [prefixCount] using physicalEnough

/-- Independent first-accepted output for one physical scalar coordinate. -/
def semanticOutput
    (prime : EuclidPrime goldilocksP)
    (duplexBase u64Base candidateBase count : Nat)
    (initial : SymbolicDuplex.Builder)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (constantWire : assignment 0 = 1)
    (u64Satisfied :
      Satisfies (PiRlcCanonicalU64.rows duplexBase u64Base count initial)
        assignment)
    (coordinate : Fin count) : List ProductionAlphabet.Coefficient :=
  FirstAccepted.firstAccepted ProductionAlphabet.verifier outputCount
    (semanticCandidates prime duplexBase u64Base candidateBase count initial
      canonical constantWire u64Satisfied coordinate)

/-- Every physical output position is the canonical centered field embedding
of the symbol at the same position of the independent first-accepted list. -/
theorem output_getElem?_eq
    (prime : EuclidPrime goldilocksP)
    (duplexBase u64Base candidateBase selectorBase count : Nat)
    (initial : SymbolicDuplex.Builder)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (constantWire : assignment 0 = 1)
    (u64Satisfied :
      Satisfies (PiRlcCanonicalU64.rows duplexBase u64Base count initial)
        assignment)
    (candidateSatisfied :
      Satisfies
        (PiRlcCanonicalCandidates.rows duplexBase u64Base candidateBase count
          initial)
        assignment)
    (selectorSatisfied :
      Satisfies
        (PiRlcCanonicalSelector.rows duplexBase u64Base candidateBase
          selectorBase count initial)
        assignment)
    (coordinate : Fin count) (position : Fin outputCount) :
    Option.map (fun coefficient =>
        (Phi81StrongSet.embedCoefficient coefficient).val)
        ((semanticOutput prime duplexBase u64Base candidateBase count initial
          canonical constantWire u64Satisfied coordinate)[position.val]?) =
      some (assignment (outputColumn selectorBase coordinate position)) := by
  obtain ⟨selected, selectedRefines⟩ :=
    PiRlcCanonicalSelectorSound.position_refines prime duplexBase u64Base
      candidateBase selectorBase count initial canonical constantWire
      selectorSatisfied coordinate position
  let candidate : Fin candidatesPerScalar := candidateAt position selected
  let bits :=
    sourceBitsBoolean prime duplexBase u64Base candidateBase count initial
      canonical constantWire u64Satisfied coordinate candidate
  have localSatisfied :=
    satisfies_candidate duplexBase u64Base candidateBase count initial
      assignment candidateSatisfied coordinate candidate
  have exactAcceptance :=
    PiRlcCanonicalCandidateSound.acceptance_sound prime canonical constantWire
      bits localSatisfied
  have candidateEq :
      PiRlcCanonicalCandidateSound.candidate assignment
          (candidateLayout duplexBase u64Base candidateBase initial coordinate
            candidate)
          bits =
        semanticCandidate prime duplexBase u64Base candidateBase count initial
          canonical constantWire u64Satisfied coordinate candidate := by
    apply Fin.ext
    rfl
  rw [candidateEq] at exactAcceptance
  have physicalAccepted :
      assignment
          (PiRlcCanonicalCandidate.acceptColumn
            (candidateLayout duplexBase u64Base candidateBase initial
              coordinate candidate)) =
        1 := by
    simpa [acceptSource, candidateSourceLayout, candidate] using
      selectedRefines.accepted
  have acceptedSemantic :
      ProductionAlphabet.verifier.accepts
          (semanticCandidate prime duplexBase u64Base candidateBase count
            initial canonical constantWire u64Satisfied coordinate candidate) =
        true := by
    cases decision :
        ProductionAlphabet.verifier.accepts
          (semanticCandidate prime duplexBase u64Base candidateBase count
            initial canonical constantWire u64Satisfied coordinate candidate)
    · have exactZero :
          assignment
              (PiRlcCanonicalCandidate.acceptColumn
                (candidateLayout duplexBase u64Base candidateBase initial
                  coordinate candidate)) =
            0 := by
        simpa [decision] using exactAcceptance
      omega
    · rfl
  have exactSymbol :=
    PiRlcCanonicalCandidateSound.residue_refines_verifier prime canonical
      constantWire bits localSatisfied
  have centeredSymbol :
      assignment (outputColumn selectorBase coordinate position) =
        (assignment
            (PiRlcCanonicalCandidate.residueColumn
              (candidateLayout duplexBase u64Base candidateBase initial
                coordinate candidate)) +
          (goldilocksP - 2)) % goldilocksP := by
    simpa [symbolSource, candidateSourceLayout, candidate] using
      selectedRefines.output
  have outputEq :
      assignment (outputColumn selectorBase coordinate position) =
        (Phi81StrongSet.embedCoefficient
          (ProductionAlphabet.verifier.symbol
            (semanticCandidate prime duplexBase u64Base candidateBase count
              initial canonical constantWire u64Satisfied coordinate
                candidate))).val := by
    rw [embedCoefficient_val_eq_shift]
    exact centeredSymbol.trans
      (congrArg (fun value =>
        (value + (goldilocksP - 2)) % goldilocksP)
        (by simpa [semanticCandidate, bits] using exactSymbol))
  have physicalPrior :
      lcEval assignment
          (candidateLayout duplexBase u64Base candidateBase initial coordinate
            candidate).prior =
        position.val := by
    simpa [prefixSource, candidateSourceLayout, candidate] using
      selectedRefines.priorCount
  have priorExact :=
    prior_refines prime duplexBase u64Base candidateBase count initial
      canonical constantWire u64Satisfied candidateSatisfied coordinate
      candidate
  have physicalPrefix :
      acceptedPrefix duplexBase u64Base candidateBase initial coordinate
          assignment candidate.val =
        position.val :=
    priorExact.symm.trans physicalPrior
  have prefixExact :=
    acceptedPrefix_eq_prefixCount prime duplexBase u64Base candidateBase count
      initial canonical constantWire u64Satisfied candidateSatisfied coordinate
      candidate.val (Nat.le_of_lt candidate.isLt)
  have before :
      FirstAccepted.acceptedCount ProductionAlphabet.verifier
          ((semanticCandidates prime duplexBase u64Base candidateBase count
            initial canonical constantWire u64Satisfied coordinate).take
              candidate.val) =
        position.val := by
    change
      prefixCount prime duplexBase u64Base candidateBase count initial
          canonical constantWire u64Satisfied coordinate candidate.val =
        position.val
    rw [← prefixExact]
    exact physicalPrefix
  have indexLt :
      candidate.val <
        (semanticCandidates prime duplexBase u64Base candidateBase count
          initial canonical constantWire u64Satisfied coordinate).length := by
    simpa using candidate.isLt
  have acceptedAt :
      ProductionAlphabet.verifier.accepts
          ((semanticCandidates prime duplexBase u64Base candidateBase count
            initial canonical constantWire u64Satisfied coordinate).get
              ⟨candidate.val, indexLt⟩) =
        true := by
    simpa [semanticCandidates, candidate] using acceptedSemantic
  have selectedResult :=
    FirstAccepted.getElem?_firstAccepted_eq_symbol_of_prefix
      (verifier := ProductionAlphabet.verifier)
      (need := outputCount)
      (candidates :=
        semanticCandidates prime duplexBase u64Base candidateBase count initial
          canonical constantWire u64Satisfied coordinate)
      (index := candidate.val) (position := position.val)
      indexLt acceptedAt before position.isLt
  have selectedResult' :
      (semanticOutput prime duplexBase u64Base candidateBase count initial
        canonical constantWire u64Satisfied coordinate)[position.val]? =
        some
          (ProductionAlphabet.verifier.symbol
            (semanticCandidate prime duplexBase u64Base candidateBase count
              initial canonical constantWire u64Satisfied coordinate
              candidate)) := by
    simpa [semanticOutput, semanticCandidates, candidate] using selectedResult
  rw [selectedResult']
  simp only [Option.map_some, Option.some.injEq]
  exact outputEq.symm

/-- Physical output columns, exposed only as their canonical natural
representatives. -/
def physicalOutputValues
    (selectorBase : Nat) {count : Nat} (coordinate : Fin count)
    (assignment : Nat → Nat) : List Nat :=
  List.ofFn fun position : Fin outputCount =>
    assignment (outputColumn selectorBase coordinate position)

/-- Complete model-level selector theorem: the physical 54-coordinate output
is extensionally equal to the centered embedding of the independent
first-accepted semantic output. -/
theorem outputs_eq_embeddedFirstAccepted
    (prime : EuclidPrime goldilocksP)
    (duplexBase u64Base candidateBase selectorBase count : Nat)
    (initial : SymbolicDuplex.Builder)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (constantWire : assignment 0 = 1)
    (u64Satisfied :
      Satisfies (PiRlcCanonicalU64.rows duplexBase u64Base count initial)
        assignment)
    (candidateSatisfied :
      Satisfies
        (PiRlcCanonicalCandidates.rows duplexBase u64Base candidateBase count
          initial)
        assignment)
    (selectorSatisfied :
      Satisfies
        (PiRlcCanonicalSelector.rows duplexBase u64Base candidateBase
          selectorBase count initial)
        assignment)
    (coordinate : Fin count) :
    physicalOutputValues selectorBase coordinate assignment =
      (semanticOutput prime duplexBase u64Base candidateBase count initial
        canonical constantWire u64Satisfied coordinate).map
          (fun coefficient =>
            (Phi81StrongSet.embedCoefficient coefficient).val) := by
  have enoughSemantic :=
    enough prime duplexBase u64Base candidateBase selectorBase count initial
      canonical constantWire u64Satisfied candidateSatisfied selectorSatisfied
      coordinate
  have semanticLength :
      (semanticOutput prime duplexBase u64Base candidateBase count initial
        canonical constantWire u64Satisfied coordinate).length =
        outputCount := by
    exact FirstAccepted.firstAccepted_length_of_enough enoughSemantic
  apply List.ext_getElem
  · simp [physicalOutputValues, semanticLength]
  · intro index leftBound rightBound
    have indexLt : index < outputCount := by
      simpa [physicalOutputValues] using leftBound
    let position : Fin outputCount := ⟨index, indexLt⟩
    have atIndex :=
      output_getElem?_eq prime duplexBase u64Base candidateBase selectorBase
        count initial canonical constantWire u64Satisfied candidateSatisfied
        selectorSatisfied coordinate position
    have semanticBound :
        index <
          (semanticOutput prime duplexBase u64Base candidateBase count initial
            canonical constantWire u64Satisfied coordinate).length := by
      rw [semanticLength]
      exact indexLt
    rw [List.getElem?_eq_getElem semanticBound] at atIndex
    simp only [Option.map_some, Option.some.injEq] at atIndex
    simp only [physicalOutputValues, List.getElem_ofFn, List.getElem_map]
    exact atIndex.symm

end Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalSamplerSound
