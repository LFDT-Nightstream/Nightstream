import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Refinement.TailRows
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Refinement.Terminal.ScalarRows

/-!
Readable 54-of-64 selection-tail rows for every terminal `Pi_RLC` scalar.

Assurance tier: implementation/R1CS correspondence. This file transports one
exact terminal owner tail into the profile-independent `SelectionRows` schema.
It proves row-family placement, not challenge provenance or selection meaning.

Owns: the terminal scalar-indexed local tail assignment; canonicality and
constant-one transport; and exact satisfaction of the readable 2,599-row tail.

Does not own: rejection/symbol semantics of the 64 candidate leaves,
first-accepted selection, the requirement that at least 54 candidates are
accepted, Poseidon2 transcript provenance, coefficient assembly, Rust trace
conformance, row removal, or cost totals.

Emits constraints: no.

Authority boundary: the terminal owner supplies only an exact mapped row
artifact. `SelectionRows` remains the readable mathematical row hierarchy, and
later semantic theorems must interpret it without treating owner layout or a
legacy measurement as protocol authority.

| Protocol | Phase | Constraint family | Terminal input | Proven result |
|---|---|---|---|---|
| `Pi_RLC` | scalar `rho` | tail column view | `rho : Fin 15` | exact relabeling into one readable tail assignment |
| `Pi_RLC` | sampler selection | all 2,599 rows | accepted terminal tail piece | satisfaction of the complete `SelectionRows` hierarchy |
| `Pi_RLC` | sampler selection | field discipline | canonical assignment and column zero | local canonicality and constant one |
-/

namespace Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.Terminal.TailRows

open Nightstream.Implementation.R1CS

/-- Local readable-tail assignment for one terminal scalar coordinate. -/
def localAssignment
    (assignment : Nat -> Nat) (rho : Fin ScalarRows.scalarCount) : Nat -> Nat :=
  PiRlcChallenge.Sampler.Refinement.TailRows.localAssignmentAt
    (ScalarRows.tailBitStarts rho) (ScalarRows.tailFirstAllocated rho)
    assignment

@[simp] theorem localAssignment_zero
    (assignment : Nat -> Nat) (rho : Fin ScalarRows.scalarCount) :
    localAssignment assignment rho 0 = assignment 0 := by
  exact PiRlcChallenge.Sampler.Refinement.TailRows.localAssignmentAt_zero
    (ScalarRows.tailBitStarts rho) (ScalarRows.tailFirstAllocated rho)
    assignment

/-- Canonicality of the terminal owner assignment is inherited by one local
readable tail view. -/
theorem canonical
    {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (rho : Fin ScalarRows.scalarCount) :
    forall column, localAssignment assignment rho column < goldilocksP :=
  PiRlcChallenge.Sampler.Refinement.TailRows.canonicalAt
    (ScalarRows.tailBitStarts rho) (ScalarRows.tailFirstAllocated rho)
    canonical

/-- Column zero remains the verifier's constant one in every terminal tail
view. -/
theorem constantOne
    {assignment : Nat -> Nat}
    (one : assignment 0 = 1)
    (rho : Fin ScalarRows.scalarCount) :
    localAssignment assignment rho 0 = 1 := by
  exact PiRlcChallenge.Sampler.Refinement.TailRows.constantOneAt
    (ScalarRows.tailBitStarts rho) (ScalarRows.tailFirstAllocated rho) one

/-- Exact terminal-owner acceptance exposes one complete readable selection
tail. This is structural correspondence only; semantic selection is proved in
a separate layer. -/
theorem accepted_satisfies
    {assignment : Nat -> Nat}
    (accepted :
      FPrimeFullHistoryTerminalPiRlcTranscriptRhos.Accepted assignment)
    (rho : Fin ScalarRows.scalarCount) :
    Satisfies PiRlcChallenge.Sampler.SelectionRows.rows
      (localAssignment assignment rho) := by
  exact PiRlcChallenge.Sampler.Refinement.TailRows.satisfyingRows_refine
    (ScalarRows.tailBitStarts rho) (ScalarRows.tailFirstAllocated rho)
    (ScalarRows.accepted_tailRows accepted rho)

end Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.Terminal.TailRows
