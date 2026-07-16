import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Refinement.TailPrefixCounts
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Selection.Position

/-!
Profile-independent first-accepted refinement for one `Pi_RLC` 54-of-64
selection tail.

Assurance tier: implementation/R1CS correspondence. This file composes the
readable one-hot, routing, accept, prefix, and symbol rows with independently
proved field-candidate and accepted-prefix semantics. It proves the exact
first 54 verifier-accepted field chunks returned by the tail.

Owns: the independent bounded `firstAccepted` output; one selected-position
proof; and equality between all 54 readable output wires and centered symbols
of the field-derived semantic output.

Does not own: Poseidon2 transcript provenance for the field columns,
scalar-to-scalar transcript chaining, coefficient-vector-to-ring assembly,
Rust trace conformance, row removal, or cost totals.

Emits constraints: no.

Authority boundary: selectors are non-authoritative witnesses. Every selected
source must independently be a verifier-accepted canonical field chunk with
the exact requested prior accepted count. These results deliberately say
`field-derived`, not `challenge`, until Poseidon2 provenance is proved.

| Protocol | Phase | Constraint family | Input obligation | Proven result |
|---|---|---|---|---|
| `Pi_RLC` | output selection | one-hot route | readable selector and product rows | one unique source in each 11-candidate window |
| `Pi_RLC` | output selection | accept/prefix binding | field-candidate and exact prefix semantics | selected source is the requested accepted position |
| `Pi_RLC` | output selection | symbol binding | verifier-owned centered symbol | one output wire equals the selected field chunk's symbol |
| `Pi_RLC` | bounded sampler | all 54 positions | readable tail plus enough-accepted proof | complete output equals independent `firstAccepted` field output |
-/

namespace Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.TailFirstAccepted

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript
open Nightstream.SuperNeo.Sampling
open Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler

def defaultCoefficient : ProductionAlphabet.Coefficient := ⟨0, by decide⟩

/-- Independent mathematical output of the bounded field-derived prefix. -/
def semanticOutput
    (layout : TailCandidateSemantics.Layout)
    (assignment : Nat -> Nat)
    (canonical : ChunkOrder.CanonicalAssignment assignment) :
    List ProductionAlphabet.Coefficient :=
  FirstAccepted.firstAccepted ProductionAlphabet.verifier
    ProductionAlphabet.coefficientCount
    (TailPrefixCounts.candidates layout assignment canonical)

theorem semanticOutput_length
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat -> Nat}
    (canonical : ChunkOrder.CanonicalAssignment assignment)
    (one : assignment 0 = 1)
    (layout : TailCandidateSemantics.Layout)
    (lanes : ScalarLanes.Refines assignment canonical layout.lanes)
    (bindings : TailCandidateSemantics.SourceBindings layout assignment)
    (tailSatisfies : Satisfies SelectionRows.rows
      (TailCandidateSemantics.localAssignment layout assignment)) :
    (semanticOutput layout assignment canonical).length =
      ProductionAlphabet.coefficientCount := by
  exact FirstAccepted.firstAccepted_length_of_enough
    (TailPrefixCounts.enoughAccepted prime canonical one layout lanes bindings
      tailSatisfies)

/-- One readable output is exactly the centered field encoding of the same
position in the independent first-accepted field-chunk list. -/
theorem outputAt_refines
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat -> Nat}
    (canonical : ChunkOrder.CanonicalAssignment assignment)
    (one : assignment 0 = 1)
    (layout : TailCandidateSemantics.Layout)
    (lanes : ScalarLanes.Refines assignment canonical layout.lanes)
    (bindings : TailCandidateSemantics.SourceBindings layout assignment)
    (tailSatisfies : Satisfies SelectionRows.rows
      (TailCandidateSemantics.localAssignment layout assignment))
    (position : Fin ProductionAlphabet.coefficientCount) :
    TailCandidateSemantics.localAssignment layout assignment
        (SelectionRows.outputCol position.val) =
      CandidateOrder.centeredField
        ((semanticOutput layout assignment canonical).getD position.val
          defaultCoefficient) := by
  have localCanonical : forall column,
      TailCandidateSemantics.localAssignment layout assignment column <
        goldilocksP := by
    exact TailRows.canonicalAt layout.tailBitStarts layout.tailFirstAllocated
      canonical
  have localOne :
      TailCandidateSemantics.localAssignment layout assignment 0 = 1 := by
    exact TailRows.constantOneAt layout.tailBitStarts layout.tailFirstAllocated
      one
  have positionLt : position.val < SelectionRows.outputCount := by
    simpa [ProductionAlphabet.coefficientCount, SelectionRows.outputCount] using
      position.isLt
  obtain ⟨selected, selectedLt, selectedOne⟩ :=
    Selection.OneHot.exists_selectedOffset prime localCanonical localOne
      positionLt tailSatisfies
  have routed := Selection.Position.refines prime localCanonical localOne
    positionLt selectedLt selectedOne tailSatisfies
  have candidateLt : position.val + selected < SelectionRows.candidateCount := by
    have positionBound := position.isLt
    simp only [ProductionAlphabet.coefficientCount] at positionBound
    simp only [SelectionRows.selectionWindow, SelectionRows.candidateCount,
      SelectionRows.outputCount] at selectedLt
    simp only [SelectionRows.candidateCount]
    omega
  let candidate : Fin SelectionRows.candidateCount :=
    ⟨position.val + selected, candidateLt⟩
  have inputRefines :=
    TailCandidateSemantics.candidate_refines lanes bindings candidate
  have acceptedValue :
      (if ProductionAlphabet.verifier.accepts
          (TailCandidateSemantics.fieldCandidate layout assignment canonical
            candidate)
        then 1 else 0) = 1 := by
    calc
      _ = TailCandidateSemantics.localAssignment layout assignment
          (SelectionRows.acceptCol candidate.val) := inputRefines.accept.symm
      _ = 1 := by simpa [candidate] using routed.accepted
  have fieldAccepted : ProductionAlphabet.verifier.accepts
      (TailCandidateSemantics.fieldCandidate layout assignment canonical
        candidate) = true := by
    cases decision : ProductionAlphabet.verifier.accepts
        (TailCandidateSemantics.fieldCandidate layout assignment canonical
          candidate)
    · simp [decision] at acceptedValue
    · rfl
  have prefixRefines := TailPrefixCounts.prefixWire_refines canonical one layout
    lanes bindings tailSatisfies candidate.val (Nat.le_of_lt candidate.isLt)
  have prefixAtPosition :
      TailPrefixCounts.prefixCount layout assignment canonical candidate.val =
        position.val := by
    rw [← prefixRefines]
    simpa [candidate] using routed.priorCount
  have sourceAccepted : ProductionAlphabet.verifier.accepts
      (TailPrefixCounts.candidates layout assignment canonical)[candidate.val] =
        true := by
    simpa [TailPrefixCounts.candidates] using fieldAccepted
  have sourceBefore : FirstAccepted.acceptedCount ProductionAlphabet.verifier
      ((TailPrefixCounts.candidates layout assignment canonical).take
        candidate.val) = position.val := by
    simpa [TailPrefixCounts.prefixCount] using prefixAtPosition
  have semanticAt :=
    FirstAccepted.getElem?_firstAccepted_eq_symbol_of_prefix
      (verifier := ProductionAlphabet.verifier)
      (need := ProductionAlphabet.coefficientCount)
      (candidates := TailPrefixCounts.candidates layout assignment canonical)
      (index := candidate.val) (position := position.val)
      (by simpa using candidate.isLt) sourceAccepted sourceBefore position.isLt
  have semanticAtField :
      (semanticOutput layout assignment canonical)[position.val]? =
        some (ProductionAlphabet.verifier.symbol
          (TailCandidateSemantics.fieldCandidate layout assignment canonical
            candidate)) := by
    simpa [semanticOutput, TailPrefixCounts.candidates] using semanticAt
  have semanticGetD :
      (semanticOutput layout assignment canonical).getD position.val
          defaultCoefficient =
        ProductionAlphabet.verifier.symbol
          (TailCandidateSemantics.fieldCandidate layout assignment canonical
            candidate) := by
    rw [List.getD_eq_getElem?_getD, semanticAtField]
    rfl
  calc
    TailCandidateSemantics.localAssignment layout assignment
        (SelectionRows.outputCol position.val) =
      TailCandidateSemantics.localAssignment layout assignment
        (SelectionRows.symbolCol candidate.val) := by
      simpa [candidate] using routed.output
    _ = CandidateOrder.centeredField
        (ProductionAlphabet.verifier.symbol
          (TailCandidateSemantics.fieldCandidate layout assignment canonical
            candidate)) := inputRefines.symbol
    _ = CandidateOrder.centeredField
        ((semanticOutput layout assignment canonical).getD position.val
          defaultCoefficient) := congrArg CandidateOrder.centeredField
            semanticGetD.symm

/-- Exact 54 readable output wires in coefficient order. -/
def productionOutput
    (layout : TailCandidateSemantics.Layout)
    (assignment : Nat -> Nat) : List Nat :=
  List.ofFn fun position : Fin ProductionAlphabet.coefficientCount =>
    TailCandidateSemantics.localAssignment layout assignment
      (SelectionRows.outputCol position.val)

/-- Total typed view of the independently defined field-derived output. The
default is unreachable once `semanticOutput_length` is established. -/
def semanticFieldOutput
    (layout : TailCandidateSemantics.Layout)
    (assignment : Nat -> Nat)
    (canonical : ChunkOrder.CanonicalAssignment assignment) : List Nat :=
  List.ofFn fun position : Fin ProductionAlphabet.coefficientCount =>
    CandidateOrder.centeredField
      ((semanticOutput layout assignment canonical).getD position.val
        defaultCoefficient)

/-- Complete profile-independent tail result. It is a field-derived sampler
theorem and intentionally stops before Poseidon2 transcript provenance. -/
theorem productionOutput_eq_semanticFieldOutput
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat -> Nat}
    (canonical : ChunkOrder.CanonicalAssignment assignment)
    (one : assignment 0 = 1)
    (layout : TailCandidateSemantics.Layout)
    (lanes : ScalarLanes.Refines assignment canonical layout.lanes)
    (bindings : TailCandidateSemantics.SourceBindings layout assignment)
    (tailSatisfies : Satisfies SelectionRows.rows
      (TailCandidateSemantics.localAssignment layout assignment)) :
    productionOutput layout assignment =
      semanticFieldOutput layout assignment canonical := by
  apply congrArg List.ofFn
  funext position
  exact outputAt_refines prime canonical one layout lanes bindings
    tailSatisfies position

end Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.TailFirstAccepted
