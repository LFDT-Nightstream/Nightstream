import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Refinement.OneScalarRows

/-!
Exact local view of the production 54-of-64 selection tail.

Owns: transport of the exact mapped 2,599-row production tail back to the
readable `SelectionRows` hierarchy, plus canonicality and constant-one
transport for that local view.

Does not own: semantic interpretation of selectors, proof of first-accepted
selection, lane-input correspondence, coefficient assembly, Rust conformance,
row removal, or cost totals.

Emits constraints: no.

Authority boundary: `SelectionRows.rows` is the readable schema and the owner
artifact is only an implementation object. This file proves exact structural
correspondence between them; it does not infer sampler correctness from row
shape alone.

| Protocol | Phase | Constraint family | Production object | Lean guarantee |
|---|---|---|---|---|
| `Pi_RLC` | sampler/tail | all 2,599 rows | exact owner tail piece | accepted production rows satisfy the readable local hierarchy |
| `Pi_RLC` | sampler/tail | local columns | `tailColumnMap` | local canonicality and constant one are inherited from the production assignment |
-/

namespace Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.TailRows

open Nightstream.Implementation.R1CS

/-- Local readable-tail assignment for an arbitrary profile layout. -/
def localAssignmentAt
    (bitStarts : List Nat) (firstAllocated : Nat)
    (assignment : Nat -> Nat) : Nat -> Nat :=
  Relabel.assignment
    (AlphabetSamplingResidualTemplate.tailColumnMap
      bitStarts firstAllocated)
    assignment

/-- Recursive-profile compatibility view. New profile-independent proofs use
`localAssignmentAt` directly. -/
def localAssignment (assignment : Nat -> Nat) : Nat -> Nat :=
  localAssignmentAt OneScalarRows.tailBitStarts
    OneScalarRows.tailFirstAllocated assignment

@[simp] theorem localAssignmentAt_zero
    (bitStarts : List Nat) (firstAllocated : Nat)
    (assignment : Nat -> Nat) :
    localAssignmentAt bitStarts firstAllocated assignment 0 = assignment 0 := by
  simp [localAssignmentAt, AlphabetSamplingResidualTemplate.tailColumnMap,
    AlphabetSamplingResidualTemplate.tailInputColumns,
    AlphabetSamplingResidualTemplate.chunkBases,
    Relabel.assignment, Relabel.column]

@[simp] theorem localAssignment_zero
    (assignment : Nat -> Nat) :
    localAssignment assignment 0 = assignment 0 := by
  simp [localAssignment, AlphabetSamplingResidualTemplate.tailColumnMap,
    AlphabetSamplingResidualTemplate.tailInputColumns,
    AlphabetSamplingResidualTemplate.chunkBases,
    OneScalarRows.tailBitStarts, Relabel.assignment, Relabel.column]

theorem canonical
    {assignment : Nat -> Nat}
    (canonical : ∀ column, assignment column < goldilocksP) :
    ∀ column, localAssignment assignment column < goldilocksP :=
  Relabel.canonical canonical

theorem canonicalAt
    {assignment : Nat -> Nat}
    (bitStarts : List Nat) (firstAllocated : Nat)
    (canonical : ∀ column, assignment column < goldilocksP) :
    ∀ column,
      localAssignmentAt bitStarts firstAllocated assignment column <
        goldilocksP :=
  Relabel.canonical canonical

theorem constantOne
    {assignment : Nat -> Nat}
    (one : assignment 0 = 1) :
    localAssignment assignment 0 = 1 := by
  simpa using one

theorem constantOneAt
    {assignment : Nat -> Nat}
    (bitStarts : List Nat) (firstAllocated : Nat)
    (one : assignment 0 = 1) :
    localAssignmentAt bitStarts firstAllocated assignment 0 = 1 := by
  simpa using one

/-- Any exact mapped tail satisfying the readable template refines to the
local `SelectionRows` hierarchy, independently of a generated owner. -/
theorem satisfyingRows_refine
    {assignment : Nat -> Nat}
    (bitStarts : List Nat) (firstAllocated : Nat)
    (satisfies : Satisfies
      (AlphabetSamplingResidualTemplate.tailRows bitStarts firstAllocated)
      assignment) :
    Satisfies SelectionRows.rows
      (localAssignmentAt bitStarts firstAllocated assignment) := by
  apply (Relabel.satisfies_mapped_iff SelectionRows.rows
    (AlphabetSamplingResidualTemplate.tailColumnMap bitStarts firstAllocated)
    assignment).mp
  simpa [AlphabetSamplingResidualTemplate.tailRows,
    SelectionRows.rows_eq_generated] using satisfies

/-- Exact owner acceptance implies the readable local tail schema. The
kernel-checked `SelectionRows.rows_eq_generated` equality is the only bridge
from generated row material to the named family tree. -/
theorem accepted_satisfies
    {assignment : Nat -> Nat}
    (accepted :
      FPrimeFullHistoryRecursivePiRlcTranscriptRhos.Accepted assignment) :
    Satisfies SelectionRows.rows (localAssignment assignment) := by
  exact satisfyingRows_refine OneScalarRows.tailBitStarts
    OneScalarRows.tailFirstAllocated
    (OneScalarRows.accepted_tailRows accepted)

end Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.TailRows
