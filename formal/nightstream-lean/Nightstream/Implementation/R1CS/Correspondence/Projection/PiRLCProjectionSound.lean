import Nightstream.Implementation.R1CS.Artifacts.Projection
import Nightstream.Implementation.R1CS.Core.ProjectionLengths
import Nightstream.Implementation.R1CS.Correspondence.Projection.ProjectionSound

/-!
Contract: artifact-level soundness for the exact 714-row production PiRLC
projection primitive exported from Rust.

No golden witness appears in the main theorem.  Any canonical assignment that
satisfies the exported rows is accepted by the complete bounded-polynomial
projection predicate.
-/

namespace Nightstream.Implementation.R1CS.PiRLCProjection

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.CheckedProgram
open Nightstream.Implementation.R1CS.ProjectionProgram
open Nightstream.SuperNeo.ProjectionCheck

set_option maxRecDepth 262144

theorem projectionTrace_layout : projectionTrace.LayoutValid := by
  native_decide

theorem projectionTrace_pairsNonempty : projectionTrace.pairs ≠ [] := by
  native_decide

theorem projectionTrace_pairWidths : ∀ pair ∈ projectionTrace.pairs,
    pair.rhoColumns.length = 54 ∧ pair.inputColumns.length = 54 := by
  native_decide

theorem projectionIdentity_wellFormed (assignment : Nat → Nat) :
    (projectionTrace.identity assignment).WellFormed := by
  exact projectionTrace.identity_wellFormed_of_widths assignment
    projectionTrace_layout projectionTrace_pairsNonempty
    projectionTrace_pairWidths

/-- Satisfaction of the exact Rust-emitted row program implies acceptance of
the complete coefficient identity at the sampled extension-field point. -/
theorem exactRows_imply_batchAccepted
    {assignment : Nat → Nat}
    (assignmentCanonical : ∀ column, assignment column < goldilocksP)
    (constantOne : assignment 0 = 1)
    (satisfies : Satisfies rows assignment) :
    BatchAccepted K.ops [projectionTrace.identity assignment] := by
  have programDefinitionsHold := definitionsHold_of_satisfies
    definitions_canonical assignmentCanonical constantOne satisfies
  have traceDefinitionsHold : DefinitionsHold assignment
      projectionTrace.definitions := by
    intro definition member
    exact programDefinitionsHold definition
      (trace_definitions_are_exact_program_definitions definition member)
  have traceChecksHold : Satisfies projectionTrace.checks assignment := by
    rw [trace_checks_are_exact_program_checks]
    exact checksSatisfy_of_satisfies satisfies
  have evaluation := projectionTrace.evaluation_sound assignment constantOne
    projectionTrace_layout traceDefinitionsHold traceChecksHold
  intro identity member
  simp only [List.mem_singleton] at member
  subst identity
  exact ⟨projectionIdentity_wellFormed assignment, evaluation⟩

end Nightstream.Implementation.R1CS.PiRLCProjection
