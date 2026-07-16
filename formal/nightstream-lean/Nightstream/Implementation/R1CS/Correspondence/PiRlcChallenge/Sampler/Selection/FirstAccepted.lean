import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Selection.Position
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Refinement.PrefixCounts

/-!
End-to-end first-accepted refinement for the production `Pi_RLC` 54-of-64
selection tail.

Owns: composition of one-hot routing, accepted/prefix/symbol bindings,
verifier-derived candidate decisions, exact accepted-prefix counts, and the
independent `FirstAccepted.firstAccepted` semantics.

Does not own: coefficient-vector-to-ring assembly, subsequent scalar samplers,
Rust trace conformance, row removal, or cost totals.

Emits constraints: no.

Authority boundary: selectors are non-authoritative witnesses. An output is
accepted only after its selected source is proved to be a transcript-machine
candidate, accepted by the verifier, preceded by exactly the requested number
of accepted candidates, and decoded by the verifier-owned symbol function.

| Protocol | Phase | Constraint family | Mathematical obligation | Lean guarantee |
|---|---|---|---|---|
| `Pi_RLC` | sampler/selection | one-hot route | choose one source in the 11-candidate window | selected offset exists and is unique |
| `Pi_RLC` | sampler/selection | accept/prefix binding | accepted source with exactly `position` prior accepts | source is output `position` under generic `FirstAccepted` semantics |
| `Pi_RLC` | sampler/selection | symbol binding | copy verifier-decoded centered symbol | `outputAt_refines` |
| `Pi_RLC` | sampler/selection | complete tail | 54 outputs from 64 candidates | semantic output has exactly 54 coefficients |
-/

namespace Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Selection.FirstAcceptedRefinement

open Nightstream.Implementation.R1CS
open Nightstream.SuperNeo.Sampling
open Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler

def defaultCoefficient : ProductionAlphabet.Coefficient := ⟨0, by decide⟩

/-- Independent mathematical output of the bounded candidate prefix. -/
def semanticOutput
    (assignment : Nat -> Nat)
    (canonical : Transcript.ChunkOrder.CanonicalAssignment assignment) :
    List ProductionAlphabet.Coefficient :=
  FirstAccepted.firstAccepted ProductionAlphabet.verifier
    ProductionAlphabet.coefficientCount
    (Refinement.PrefixCounts.candidates assignment canonical)

theorem semanticOutput_length
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat -> Nat}
    (canonical : Transcript.ChunkOrder.CanonicalAssignment assignment)
    (one : assignment 0 = 1)
    (accepted :
      FPrimeFullHistoryRecursivePiRlcTranscriptRhos.Accepted assignment)
    (refinement : Refinement.OneScalar.Refines prime canonical one accepted) :
    (semanticOutput assignment canonical).length =
      ProductionAlphabet.coefficientCount := by
  exact FirstAccepted.firstAccepted_length_of_enough
    (Refinement.PrefixCounts.enoughAccepted prime canonical one accepted
      refinement)

/-- One production output position is exactly the centered field encoding of
the same position in the independent first-accepted coefficient list. -/
theorem outputAt_refines
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat -> Nat}
    (canonical : Transcript.ChunkOrder.CanonicalAssignment assignment)
    (one : assignment 0 = 1)
    (accepted :
      FPrimeFullHistoryRecursivePiRlcTranscriptRhos.Accepted assignment)
    (refinement : Refinement.OneScalar.Refines prime canonical one accepted)
    (position : Fin ProductionAlphabet.coefficientCount) :
    Refinement.TailRows.localAssignment assignment
        (SelectionRows.outputCol position.val) =
      CandidateOrder.centeredField
        ((semanticOutput assignment canonical).getD position.val
          defaultCoefficient) := by
  have localCanonical := Refinement.TailRows.canonical canonical
  have localOne := Refinement.TailRows.constantOne one
  have tailSatisfies := Refinement.TailRows.accepted_satisfies accepted
  have positionLt : position.val < SelectionRows.outputCount := by
    simpa [ProductionAlphabet.coefficientCount, SelectionRows.outputCount] using
      position.isLt
  obtain ⟨selected, selectedLt, selectedOne⟩ :=
    OneHot.exists_selectedOffset prime localCanonical localOne positionLt
      tailSatisfies
  have routed := Position.refines prime localCanonical localOne positionLt
    selectedLt selectedOne tailSatisfies
  have candidateLt : position.val + selected < SelectionRows.candidateCount := by
    have positionBound := position.isLt
    simp only [ProductionAlphabet.coefficientCount] at positionBound
    simp only [SelectionRows.selectionWindow, SelectionRows.candidateCount,
      SelectionRows.outputCount] at selectedLt
    simp only [SelectionRows.candidateCount]
    omega
  let candidate : Fin SelectionRows.candidateCount :=
    ⟨position.val + selected, candidateLt⟩
  have inputRefines := Refinement.TailInputs.candidate_refines prime canonical
    one accepted refinement candidate
  have acceptedValue :
      (if ProductionAlphabet.verifier.accepts
          (Refinement.TailInputs.machineCandidate assignment canonical candidate)
        then 1 else 0) = 1 := by
    calc
      _ = Refinement.TailRows.localAssignment assignment
          (SelectionRows.acceptCol candidate.val) := inputRefines.accept.symm
      _ = 1 := by simpa [candidate] using routed.accepted
  have machineAccepted : ProductionAlphabet.verifier.accepts
      (Refinement.TailInputs.machineCandidate assignment canonical candidate) =
        true := by
    cases decision : ProductionAlphabet.verifier.accepts
        (Refinement.TailInputs.machineCandidate assignment canonical candidate)
    · simp [decision] at acceptedValue
    · rfl
  have prefixRefines := Refinement.PrefixCounts.prefixWire_refines prime
    canonical one accepted refinement candidate.val (Nat.le_of_lt candidate.isLt)
  have prefixAtPosition :
      Refinement.PrefixCounts.prefixCount assignment canonical candidate.val =
        position.val := by
    rw [← prefixRefines]
    simpa [candidate] using routed.priorCount
  have sourceAccepted : ProductionAlphabet.verifier.accepts
      (Refinement.PrefixCounts.candidates assignment canonical)[candidate.val] =
        true := by
    simpa [Refinement.PrefixCounts.candidates] using machineAccepted
  have sourceBefore : FirstAccepted.acceptedCount ProductionAlphabet.verifier
      ((Refinement.PrefixCounts.candidates assignment canonical).take
        candidate.val) = position.val := by
    simpa [Refinement.PrefixCounts.prefixCount] using prefixAtPosition
  have semanticAt :=
    FirstAccepted.getElem?_firstAccepted_eq_symbol_of_prefix
      (verifier := ProductionAlphabet.verifier)
      (need := ProductionAlphabet.coefficientCount)
      (candidates := Refinement.PrefixCounts.candidates assignment canonical)
      (index := candidate.val) (position := position.val)
      (by simpa using candidate.isLt) sourceAccepted sourceBefore position.isLt
  have semanticAtMachine :
      (semanticOutput assignment canonical)[position.val]? =
        some (ProductionAlphabet.verifier.symbol
          (Refinement.TailInputs.machineCandidate assignment canonical
            candidate)) := by
    simpa [semanticOutput, Refinement.PrefixCounts.candidates] using semanticAt
  have semanticGetD :
      (semanticOutput assignment canonical).getD position.val
          defaultCoefficient =
        ProductionAlphabet.verifier.symbol
          (Refinement.TailInputs.machineCandidate assignment canonical
            candidate) := by
    rw [List.getD_eq_getElem?_getD, semanticAtMachine]
    rfl
  calc
    Refinement.TailRows.localAssignment assignment
        (SelectionRows.outputCol position.val) =
      Refinement.TailRows.localAssignment assignment
        (SelectionRows.symbolCol candidate.val) := by
      simpa [candidate] using routed.output
    _ = CandidateOrder.centeredField
        (ProductionAlphabet.verifier.symbol
          (Refinement.TailInputs.machineCandidate assignment canonical
            candidate)) := inputRefines.symbol
    _ = CandidateOrder.centeredField
        ((semanticOutput assignment canonical).getD position.val
          defaultCoefficient) := congrArg CandidateOrder.centeredField
            semanticGetD.symm

/-- Exact 54 production output wires in coefficient order. -/
def productionOutput (assignment : Nat -> Nat) : List Nat :=
  List.ofFn fun position : Fin ProductionAlphabet.coefficientCount =>
    Refinement.TailRows.localAssignment assignment
      (SelectionRows.outputCol position.val)

/-- Total typed view of the independently defined semantic output. The default
is unreachable once `semanticOutput_length` is established. -/
def semanticFieldOutput
    (assignment : Nat -> Nat)
    (canonical : Transcript.ChunkOrder.CanonicalAssignment assignment) :
    List Nat :=
  List.ofFn fun position : Fin ProductionAlphabet.coefficientCount =>
    CandidateOrder.centeredField
      ((semanticOutput assignment canonical).getD position.val
        defaultCoefficient)

theorem productionOutput_eq_semanticFieldOutput
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat -> Nat}
    (canonical : Transcript.ChunkOrder.CanonicalAssignment assignment)
    (one : assignment 0 = 1)
    (accepted :
      FPrimeFullHistoryRecursivePiRlcTranscriptRhos.Accepted assignment)
    (refinement : Refinement.OneScalar.Refines prime canonical one accepted) :
    productionOutput assignment = semanticFieldOutput assignment canonical := by
  apply congrArg List.ofFn
  funext position
  exact outputAt_refines prime canonical one accepted refinement position

/-- Complete one-scalar closure theorem: accepted production rows return exactly
the typed centered-field view of the independent first 54 verifier-accepted
transcript candidates in source order. `semanticOutput_length` separately proves
that every `getD` in this view is in bounds. -/
theorem accepted_refines
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat -> Nat}
    (canonical : Transcript.ChunkOrder.CanonicalAssignment assignment)
    (one : assignment 0 = 1)
    (accepted :
      FPrimeFullHistoryRecursivePiRlcTranscriptRhos.Accepted assignment) :
    productionOutput assignment = semanticFieldOutput assignment canonical := by
  let refinement :=
    Refinement.OneScalar.accepted_refines prime canonical one accepted
  exact productionOutput_eq_semanticFieldOutput prime canonical one accepted
    refinement

end Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Selection.FirstAcceptedRefinement
