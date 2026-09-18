import Nightstream.Implementation.R1CS.Core.ProjectionLengths
import Nightstream.Implementation.R1CS.Correspondence.Projection.ProjectionSound

/-!
Contract: lift the per-identity PiRLC projection theorem across a complete
production census, including traces that share ladder and rho definitions.
-/

namespace Nightstream.Implementation.R1CS.ProjectionProgram

open Nightstream.Implementation.R1CS
open Nightstream.SuperNeo.ProjectionCheck

/-- A complete list of exact projection traces is accepted whenever all trace
definitions and checks hold.  Repeated shared definitions are harmless:
membership, rather than execution order, supplies each per-trace premise. -/
theorem ProjectionTrace.census_batchAccepted
    (traces : List ProjectionTrace) (assignment : Nat → Nat)
    (constantOne : assignment 0 = 1)
    (layouts : ∀ trace ∈ traces, trace.LayoutValid)
    (pairsNonempty : ∀ trace ∈ traces, trace.pairs ≠ [])
    (pairWidths : ∀ trace ∈ traces, ∀ pair ∈ trace.pairs,
      pair.rhoColumns.length = 54 ∧ pair.inputColumns.length = 54)
    (definitionsHold : DefinitionsHold assignment
      (traces.flatMap ProjectionTrace.definitions))
    (checksHold : Satisfies
      (traces.flatMap ProjectionTrace.checks) assignment) :
    BatchAccepted K.ops (BatchIdentity traces assignment) := by
  intro identity identityMember
  rcases List.mem_map.mp identityMember with ⟨trace, traceMember, rfl⟩
  have traceDefinitionsHold : DefinitionsHold assignment trace.definitions := by
    intro definition definitionMember
    apply definitionsHold definition
    exact List.mem_flatMap.mpr ⟨trace, traceMember, definitionMember⟩
  have traceChecksHold : Satisfies trace.checks assignment := by
    intro row rowMember
    apply checksHold row
    exact List.mem_flatMap.mpr ⟨trace, traceMember, rowMember⟩
  refine ⟨trace.identity_wellFormed_of_widths assignment
      (layouts trace traceMember) (pairsNonempty trace traceMember)
      (pairWidths trace traceMember), ?_⟩
  exact trace.evaluation_sound assignment constantOne
    (layouts trace traceMember) traceDefinitionsHold traceChecksHold

end Nightstream.Implementation.R1CS.ProjectionProgram
