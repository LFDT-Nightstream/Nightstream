import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Refinement.TailInputs
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Selection.Acceptance
import Nightstream.SuperNeo.Sampling.FirstAccepted

/-!
Accepted-prefix refinement for the complete 64-candidate `Pi_RLC` sampler
chain.

Owns: the independent 64-candidate transcript vector; mathematical accepted
counts for every source prefix; and induction from the checked zero prefix plus
all verifier-owned candidate steps to every production prefix/cumulative wire.

Does not own: one-hot selection, output-symbol routing, coefficient assembly,
Rust trace conformance, row removal, or cost totals.

Emits constraints: no.

Authority boundary: accepted counts are defined by filtering the independent
transcript-machine candidate list with `ProductionAlphabet.verifier`. Production
counters appear only on the left side of refinement equalities.

| Protocol | Phase | Constraint family | Mathematical obligation | Lean guarantee |
|---|---|---|---|---|
| `Pi_RLC` | sampler/prefix semantics | candidate vector | exact block/lane/part source order | `candidates` contains exactly the 64 machine candidates |
| `Pi_RLC` | sampler/prefix semantics | one candidate step | append one verifier decision | `prefixCount_succ` |
| `Pi_RLC` | sampler/prefix semantics | complete count chain | checked zero plus 64 recurrence leaves | `prefixWire_refines` and `cumulativeWire_refines` |
-/

namespace Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.PrefixCounts

open Nightstream.Implementation.R1CS
open Nightstream.SuperNeo.Sampling
open Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler

/-- The exact bounded candidate prefix in source order. -/
def candidates
    (assignment : Nat -> Nat)
    (canonical : Transcript.ChunkOrder.CanonicalAssignment assignment) :
    List ProductionAlphabet.Chunk :=
  List.ofFn (TailInputs.machineCandidate assignment canonical)

@[simp] theorem candidates_length
    (assignment : Nat -> Nat)
    (canonical : Transcript.ChunkOrder.CanonicalAssignment assignment) :
    (candidates assignment canonical).length = SelectionRows.candidateCount := by
  simp [candidates]

theorem candidates_get
    (assignment : Nat -> Nat)
    (canonical : Transcript.ChunkOrder.CanonicalAssignment assignment)
    (candidate : Fin SelectionRows.candidateCount) :
    (candidates assignment canonical).get
        ⟨candidate.val, by simpa using candidate.isLt⟩ =
      TailInputs.machineCandidate assignment canonical candidate := by
  simp [candidates]

/-- Mathematical number of verifier-accepted candidates before `count`. -/
def prefixCount
    (assignment : Nat -> Nat)
    (canonical : Transcript.ChunkOrder.CanonicalAssignment assignment)
    (count : Nat) : Nat :=
  FirstAccepted.acceptedCount ProductionAlphabet.verifier
    ((candidates assignment canonical).take count)

theorem prefixCount_succ
    (assignment : Nat -> Nat)
    (canonical : Transcript.ChunkOrder.CanonicalAssignment assignment)
    (candidate : Fin SelectionRows.candidateCount) :
    prefixCount assignment canonical (candidate.val + 1) =
      prefixCount assignment canonical candidate.val +
        (if ProductionAlphabet.verifier.accepts
            (TailInputs.machineCandidate assignment canonical candidate)
          then 1 else 0) := by
  rw [prefixCount, List.take_succ_eq_append_getElem (by
    simpa using candidate.isLt)]
  cases decision : ProductionAlphabet.verifier.accepts
      (TailInputs.machineCandidate assignment canonical candidate) <;>
    simp [prefixCount, FirstAccepted.acceptedCount,
      FirstAccepted.acceptedCandidates, candidates, decision]

/-- Every production prefix wire, including the final `prefixCol 64` alias of
candidate 63's cumulative output, equals the mathematical accepted count of the
same source prefix. -/
theorem prefixWire_refines
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat -> Nat}
    (canonical : Transcript.ChunkOrder.CanonicalAssignment assignment)
    (one : assignment 0 = 1)
    (accepted :
      FPrimeFullHistoryRecursivePiRlcTranscriptRhos.Accepted assignment)
    (refinement : OneScalar.Refines prime canonical one accepted) :
    ∀ count, count <= SelectionRows.candidateCount ->
      TailRows.localAssignment assignment (SelectionRows.prefixCol count) =
        prefixCount assignment canonical count := by
  intro count within
  induction count with
  | zero =>
      have zeroPrefix :=
        Selection.Initialization.zeroPrefix_eq_zero
          (TailRows.canonical canonical) (TailRows.constantOne one)
          (TailRows.accepted_satisfies accepted)
      simpa [SelectionRows.prefixCol, prefixCount,
        FirstAccepted.acceptedCount, FirstAccepted.acceptedCandidates] using
          zeroPrefix
  | succ count inductionHypothesis =>
      have countLt : count < SelectionRows.candidateCount := by omega
      let candidate : Fin SelectionRows.candidateCount := ⟨count, countLt⟩
      have step := TailInputs.cumulative_step prime canonical one accepted
        refinement candidate
      have decision :=
        (TailInputs.candidate_refines prime canonical one accepted refinement
          candidate).accept
      have previous := inductionHypothesis (by omega)
      have semanticStep := prefixCount_succ assignment canonical candidate
      change TailRows.localAssignment assignment
          (SelectionRows.cumulativeCol count) =
        prefixCount assignment canonical (count + 1)
      change TailRows.localAssignment assignment
          (SelectionRows.cumulativeCol count) =
        TailRows.localAssignment assignment
            (SelectionRows.prefixCol count) +
          TailRows.localAssignment assignment
            (SelectionRows.acceptCol count) at step
      change TailRows.localAssignment assignment
          (SelectionRows.acceptCol count) =
        if ProductionAlphabet.verifier.accepts
            (TailInputs.machineCandidate assignment canonical candidate)
          then 1 else 0 at decision
      rw [step, previous, decision]
      simpa [candidate] using semanticStep.symm

theorem cumulativeWire_refines
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat -> Nat}
    (canonical : Transcript.ChunkOrder.CanonicalAssignment assignment)
    (one : assignment 0 = 1)
    (accepted :
      FPrimeFullHistoryRecursivePiRlcTranscriptRhos.Accepted assignment)
    (refinement : OneScalar.Refines prime canonical one accepted)
    (candidate : Fin SelectionRows.candidateCount) :
    TailRows.localAssignment assignment
        (SelectionRows.cumulativeCol candidate.val) =
      prefixCount assignment canonical (candidate.val + 1) := by
  have refined := prefixWire_refines prime canonical one accepted refinement
    (candidate.val + 1) (by
      have candidateLt := candidate.isLt
      omega)
  simpa [SelectionRows.prefixCol] using refined

/-- The exact acceptance-bound rows establish mathematical `Enough` for the
independent 64-candidate vector, after the final production counter is refined
to the verifier-filtered prefix count. -/
theorem enoughAccepted
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat -> Nat}
    (canonical : Transcript.ChunkOrder.CanonicalAssignment assignment)
    (one : assignment 0 = 1)
    (accepted :
      FPrimeFullHistoryRecursivePiRlcTranscriptRhos.Accepted assignment)
    (refinement : OneScalar.Refines prime canonical one accepted) :
    FirstAccepted.Enough ProductionAlphabet.verifier
      ProductionAlphabet.coefficientCount (candidates assignment canonical) := by
  let last : Fin SelectionRows.candidateCount := ⟨63, by decide⟩
  have tailSatisfies := TailRows.accepted_satisfies accepted
  have productionEnough :=
    Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Selection.Acceptance.enoughAccepted
      prime (TailRows.canonical canonical) (TailRows.constantOne one)
      tailSatisfies
  have finalRefines := cumulativeWire_refines prime canonical one accepted
    refinement last
  rw [finalRefines] at productionEnough
  simpa [FirstAccepted.Enough, prefixCount, candidates,
    SelectionRows.outputCount, ProductionAlphabet.coefficientCount,
    SelectionRows.candidateCount, last] using productionEnough

end Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.PrefixCounts
