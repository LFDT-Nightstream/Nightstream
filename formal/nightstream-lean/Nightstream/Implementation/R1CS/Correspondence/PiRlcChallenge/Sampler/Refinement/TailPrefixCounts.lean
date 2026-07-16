import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Refinement.TailCandidateSemantics
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Selection.Acceptance
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Selection.Initialization
import Nightstream.SuperNeo.Sampling.FirstAccepted

/-!
Profile-independent accepted-prefix semantics for one bounded `Pi_RLC`
selection tail.

Assurance tier: implementation/R1CS correspondence. Given independently
proved lane semantics, explicit lane-to-tail source bindings, and satisfaction
of the readable selection rows, this file proves that every production prefix
wire is the mathematical count of verifier-accepted field chunks.

Owns: the 64 field-derived candidates in canonical source order; mathematical
prefix counts; induction over the checked zero prefix and all 64 recurrence
leaves; and the proof that the readable acceptance bound supplies 54 outputs.

Does not own: Poseidon2 transcript provenance for the field columns, one-hot
output routing, scalar-to-scalar state chaining, coefficient assembly, Rust
trace conformance, row removal, or cost totals.

Emits constraints: no.

Authority boundary: `candidates` contains canonical chunks of explicitly named
field columns. It is not called a transcript or challenge vector. The theorem
becomes a challenge-sampler result only after a separate Poseidon2 provenance
theorem identifies those field columns with verifier-owned transcript outputs.

| Protocol | Phase | Constraint family | Input obligation | Proven result |
|---|---|---|---|---|
| `Pi_RLC` | bounded sampler | candidate order | profile-independent lane address | exactly 64 field-derived candidates in block/lane/part order |
| `Pi_RLC` | accepted-prefix chain | zero prefix | readable initialization row | mathematical prefix zero is zero |
| `Pi_RLC` | accepted-prefix chain | 64 recurrence leaves | lane semantics plus explicit source bindings | every prefix/cumulative wire equals verifier-accepted count |
| `Pi_RLC` | bounded sampler | acceptance bound | readable slack and final-count rows | at least 54 of the 64 field chunks are accepted |
-/

namespace Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.TailPrefixCounts

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript
open Nightstream.SuperNeo.Sampling
open Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler

/-- The exact bounded field-derived candidate prefix in canonical source order. -/
def candidates
    (layout : TailCandidateSemantics.Layout)
    (assignment : Nat -> Nat)
    (canonical : ChunkOrder.CanonicalAssignment assignment) :
    List ProductionAlphabet.Chunk :=
  List.ofFn
    (TailCandidateSemantics.fieldCandidate layout assignment canonical)

@[simp] theorem candidates_length
    (layout : TailCandidateSemantics.Layout)
    (assignment : Nat -> Nat)
    (canonical : ChunkOrder.CanonicalAssignment assignment) :
    (candidates layout assignment canonical).length =
      SelectionRows.candidateCount := by
  simp [candidates]

theorem candidates_get
    (layout : TailCandidateSemantics.Layout)
    (assignment : Nat -> Nat)
    (canonical : ChunkOrder.CanonicalAssignment assignment)
    (candidate : Fin SelectionRows.candidateCount) :
    (candidates layout assignment canonical).get
        ⟨candidate.val, by simpa using candidate.isLt⟩ =
      TailCandidateSemantics.fieldCandidate layout assignment canonical
        candidate := by
  simp [candidates]

/-- Mathematical number of verifier-accepted field chunks before `count`. -/
def prefixCount
    (layout : TailCandidateSemantics.Layout)
    (assignment : Nat -> Nat)
    (canonical : ChunkOrder.CanonicalAssignment assignment)
    (count : Nat) : Nat :=
  FirstAccepted.acceptedCount ProductionAlphabet.verifier
    ((candidates layout assignment canonical).take count)

theorem prefixCount_succ
    (layout : TailCandidateSemantics.Layout)
    (assignment : Nat -> Nat)
    (canonical : ChunkOrder.CanonicalAssignment assignment)
    (candidate : Fin SelectionRows.candidateCount) :
    prefixCount layout assignment canonical (candidate.val + 1) =
      prefixCount layout assignment canonical candidate.val +
        (if ProductionAlphabet.verifier.accepts
            (TailCandidateSemantics.fieldCandidate layout assignment canonical
              candidate)
          then 1 else 0) := by
  rw [prefixCount, List.take_succ_eq_append_getElem (by
    simpa using candidate.isLt)]
  cases decision : ProductionAlphabet.verifier.accepts
      (TailCandidateSemantics.fieldCandidate layout assignment canonical
        candidate) <;>
    simp [prefixCount, FirstAccepted.acceptedCount,
      FirstAccepted.acceptedCandidates, candidates, decision]

/-- Every readable prefix wire, including the final `prefixCol 64` alias of
candidate 63's cumulative output, equals the mathematical accepted count of
the same field-derived prefix. -/
theorem prefixWire_refines
    {assignment : Nat -> Nat}
    (canonical : ChunkOrder.CanonicalAssignment assignment)
    (one : assignment 0 = 1)
    (layout : TailCandidateSemantics.Layout)
    (lanes : ScalarLanes.Refines assignment canonical layout.lanes)
    (bindings : TailCandidateSemantics.SourceBindings layout assignment)
    (tailSatisfies : Satisfies SelectionRows.rows
      (TailCandidateSemantics.localAssignment layout assignment)) :
    forall count, count <= SelectionRows.candidateCount ->
      TailCandidateSemantics.localAssignment layout assignment
          (SelectionRows.prefixCol count) =
        prefixCount layout assignment canonical count := by
  have localCanonical : forall column,
      TailCandidateSemantics.localAssignment layout assignment column <
        goldilocksP := by
    exact TailRows.canonicalAt layout.tailBitStarts layout.tailFirstAllocated
      canonical
  have localOne :
      TailCandidateSemantics.localAssignment layout assignment 0 = 1 := by
    exact TailRows.constantOneAt layout.tailBitStarts layout.tailFirstAllocated
      one
  intro count within
  induction count with
  | zero =>
      have zeroPrefix :=
        Selection.Initialization.zeroPrefix_eq_zero localCanonical localOne
          tailSatisfies
      simpa [SelectionRows.prefixCol, prefixCount,
        FirstAccepted.acceptedCount, FirstAccepted.acceptedCandidates] using
          zeroPrefix
  | succ count inductionHypothesis =>
      have countLt : count < SelectionRows.candidateCount := by omega
      let candidate : Fin SelectionRows.candidateCount := ⟨count, countLt⟩
      have step :=
        TailCandidateSemantics.cumulative_step lanes bindings candidate
      have decision :=
        (TailCandidateSemantics.candidate_refines lanes bindings
          candidate).accept
      have previous := inductionHypothesis (by omega)
      have semanticStep :=
        prefixCount_succ layout assignment canonical candidate
      change TailCandidateSemantics.localAssignment layout assignment
          (SelectionRows.cumulativeCol count) =
        prefixCount layout assignment canonical (count + 1)
      change TailCandidateSemantics.localAssignment layout assignment
          (SelectionRows.cumulativeCol count) =
        TailCandidateSemantics.localAssignment layout assignment
            (SelectionRows.prefixCol count) +
          TailCandidateSemantics.localAssignment layout assignment
            (SelectionRows.acceptCol count) at step
      change TailCandidateSemantics.localAssignment layout assignment
          (SelectionRows.acceptCol count) =
        if ProductionAlphabet.verifier.accepts
            (TailCandidateSemantics.fieldCandidate layout assignment canonical
              candidate)
          then 1 else 0 at decision
      rw [step, previous, decision]
      simpa [candidate] using semanticStep.symm

theorem cumulativeWire_refines
    {assignment : Nat -> Nat}
    (canonical : ChunkOrder.CanonicalAssignment assignment)
    (one : assignment 0 = 1)
    (layout : TailCandidateSemantics.Layout)
    (lanes : ScalarLanes.Refines assignment canonical layout.lanes)
    (bindings : TailCandidateSemantics.SourceBindings layout assignment)
    (tailSatisfies : Satisfies SelectionRows.rows
      (TailCandidateSemantics.localAssignment layout assignment))
    (candidate : Fin SelectionRows.candidateCount) :
    TailCandidateSemantics.localAssignment layout assignment
        (SelectionRows.cumulativeCol candidate.val) =
      prefixCount layout assignment canonical (candidate.val + 1) := by
  have refined := prefixWire_refines canonical one layout lanes bindings
    tailSatisfies (candidate.val + 1) (by
      have candidateLt := candidate.isLt
      omega)
  simpa [SelectionRows.prefixCol] using refined

/-- The exact acceptance-bound rows establish mathematical `Enough` for the
64 field-derived candidates. This remains conditional on later transcript
provenance for the underlying field columns. -/
theorem enoughAccepted
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat -> Nat}
    (canonical : ChunkOrder.CanonicalAssignment assignment)
    (one : assignment 0 = 1)
    (layout : TailCandidateSemantics.Layout)
    (lanes : ScalarLanes.Refines assignment canonical layout.lanes)
    (bindings : TailCandidateSemantics.SourceBindings layout assignment)
    (tailSatisfies : Satisfies SelectionRows.rows
      (TailCandidateSemantics.localAssignment layout assignment)) :
    FirstAccepted.Enough ProductionAlphabet.verifier
      ProductionAlphabet.coefficientCount
      (candidates layout assignment canonical) := by
  have localCanonical : forall column,
      TailCandidateSemantics.localAssignment layout assignment column <
        goldilocksP := by
    exact TailRows.canonicalAt layout.tailBitStarts layout.tailFirstAllocated
      canonical
  have localOne :
      TailCandidateSemantics.localAssignment layout assignment 0 = 1 := by
    exact TailRows.constantOneAt layout.tailBitStarts layout.tailFirstAllocated
      one
  let last : Fin SelectionRows.candidateCount := ⟨63, by decide⟩
  have productionEnough :=
    Selection.Acceptance.enoughAccepted prime localCanonical localOne
      tailSatisfies
  have finalRefines := cumulativeWire_refines canonical one layout lanes
    bindings tailSatisfies last
  rw [finalRefines] at productionEnough
  simpa [FirstAccepted.Enough, prefixCount, candidates,
    SelectionRows.outputCount, ProductionAlphabet.coefficientCount,
    SelectionRows.candidateCount, last] using productionEnough

end Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.TailPrefixCounts
