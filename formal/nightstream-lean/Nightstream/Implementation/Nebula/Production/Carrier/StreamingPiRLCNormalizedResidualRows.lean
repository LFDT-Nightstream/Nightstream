import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiRLCNormalizedCarryRows
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPiRLCFamilyBodyResidualRetained

/-!
Contract: exact normalized low-norm image of the production PiRLC additive
input-residual rows.

Assurance tier: model-level.

Owns the 108 retained equality-row images, their direct radix-seven source
decoding, and the same-assignment implication to the concrete residual
transition when the local commitment output and state columns are placed.

Does not own the family overlay that computes the local commitment, state
placement, selector authority, the stored Rust matrices, the Rust witness
encoder, replay rows, recursive orchestration, or cryptographic security.

Emits constraints: no. It specifies and proves the arithmetic meaning of the
existing normalized product-row recipe.
-/

set_option autoImplicit false
set_option maxRecDepth 100000
set_option maxHeartbeats 4000000

namespace Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedResidualRows

open Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputBinding
open Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputBindingSetup
open Nightstream.Implementation.Nebula.ProductionStreamingPiRlcAuthority
open Nightstream.Implementation.R1CS
open Nightstream.SuperNeo.Concrete

namespace Normalized

private abbrev sourceLayout :=
  Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilyCompleteRows.layout

private abbrev residualLayout :=
  Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputFamilyRows.residualLayout
    sourceLayout.input

abbrev Arm :=
  Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedCarryRows.Normalized.Arm

abbrev finalColumns :=
  Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedCarryRows.Normalized.finalColumns

theorem finalColumns_positive : 0 < finalColumns :=
  Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedCarryRows.Normalized.finalColumns_positive

abbrev selectorColumn :=
  Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedCarryRows.Normalized.selectorColumn

abbrev numericAssignment :=
  Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedCarryRows.Normalized.numericAssignment

abbrev equalityImage :=
  Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedCarryRows.Normalized.equalityImage

private theorem numericAssignment_one
    (assignment : Fin finalColumns → F)
    (constantOne : assignment ⟨0, finalColumns_positive⟩ = 1) :
    numericAssignment assignment 0 = 1 :=
  Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedCarryRows.Normalized.numericAssignment_one
    assignment constantOne

/-- Exact final image of one additive residual equation. -/
def residualImage (output : Fin (shape.rows * shape.degree)) :
    Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedAlgebraRows.Normalized.RowImage :=
  equalityImage
    [(residualLayout.beforeResidual output, 1)]
    [(residualLayout.phaseBinding output, 1),
      (residualLayout.afterResidual output, 1)]

/-- Exact acceptance predicate for all retained residual-row occurrences. -/
structure ProductionAccepted
    (arm : Arm) (assignment : Fin finalColumns → F) : Prop where
  selectorOne : assignment (selectorColumn arm) = 1
  residual : ∀ output, (residualImage output).Accepted
    (assignment (selectorColumn arm)) assignment

def productionRowCount : Nat := shape.rows * shape.degree

theorem productionRowCount_exact : productionRowCount = 108 := by
  decide

private theorem residual_row_holds
    {arm : Arm} {assignment : Fin finalColumns → F}
    (constantOne : assignment ⟨0, finalColumns_positive⟩ = 1)
    (accepted : ProductionAccepted arm assignment)
    (output : Fin (shape.rows * shape.degree)) :
    RowHolds (numericAssignment assignment)
      (Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputResidualRows.residualRow
        residualLayout output) := by
  apply (Nightstream.Implementation.R1CS.Canonical.KEquality.equalityRow_iff
    (numericAssignment assignment)
    [(residualLayout.beforeResidual output, 1)]
    [(residualLayout.phaseBinding output, 1),
      (residualLayout.afterResidual output, 1)]
    (numericAssignment_one assignment constantOne)).mpr
  exact
    (Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedCarryRows.Normalized.equalityImage_accepted_iff
      _ _ assignment constantOne).mp
      (by simpa [accepted.selectorOne] using accepted.residual output)

/-- Accepted normalized rows imply the complete source residual-row list on
the same decoded assignment. -/
theorem productionAccepted_implies_source_rows
    (arm : Arm) (assignment : Fin finalColumns → F)
    (constantOne : assignment ⟨0, finalColumns_positive⟩ = 1)
    (accepted : ProductionAccepted arm assignment) :
    Satisfies
      (Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputResidualRows.rows
        residualLayout)
      (numericAssignment assignment) := by
  intro row member
  rcases List.mem_ofFn.mp member with ⟨output, rfl⟩
  exact residual_row_holds constantOne accepted output

/-- The family-state view of the two residual vectors. -/
def StateColumnsPlaced
    (assignment : Fin finalColumns → F)
    (before after : FamilyState) : Prop :=
  Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputFamilyRows.ResidualsPlaced
    sourceLayout.input (numericAssignment assignment)
    before.inputResidual after.inputResidual

/-- The family overlay must place its authoritative local commitment in the
same phase-binding columns read by the retained residual rows. -/
def PhaseBindingPlaced
    (setup : InputBindingSetup)
    (family : Nightstream.Implementation.Nebula.ProductionStreamingPiRlcAuthority.Family)
    (inputs :
      Nightstream.Implementation.Nebula.ProductionStreamingPiRlcAuthority.Source →
        Nightstream.SuperNeo.Concrete.RingF)
    (assignment : Fin finalColumns → F) : Prop :=
  ∀ output,
    numericAssignment assignment (residualLayout.phaseBinding output) =
      (concretePhaseBinding setup family inputs output).val

/-- Accepted normalized residual rows and exact placement of the overlay and
state columns imply the semantic additive residual transition. -/
theorem productionAccepted_implies_transition
    (arm : Arm) (assignment : Fin finalColumns → F)
    (constantOne : assignment ⟨0, finalColumns_positive⟩ = 1)
    (setup : InputBindingSetup)
    (family : Nightstream.Implementation.Nebula.ProductionStreamingPiRlcAuthority.Family)
    (inputs :
      Nightstream.Implementation.Nebula.ProductionStreamingPiRlcAuthority.Source →
        Nightstream.SuperNeo.Concrete.RingF)
    (before after : FamilyState)
    (statePlaced : StateColumnsPlaced assignment before after)
    (phasePlaced : PhaseBindingPlaced setup family inputs assignment)
    (accepted : ProductionAccepted arm assignment) :
    ConcreteResidualTransition setup before.inputResidual after.inputResidual
      family inputs := by
  have columnsPlaced :
      Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputResidualRows.ColumnsPlaced
        residualLayout (numericAssignment assignment)
        before.inputResidual (concretePhaseBinding setup family inputs)
        after.inputResidual :=
    ⟨statePlaced.1, phasePlaced, statePlaced.2⟩
  exact
    Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputResidualRows.rows_imply_concreteResidualTransition
      (numericAssignment_one assignment constantOne) columnsPlaced
      (productionAccepted_implies_source_rows arm assignment constantOne accepted)
      setup family inputs rfl

/-- The semantic constants are the exact Rust-conformant receipt geometry. -/
theorem receipt_geometry_exact :
    Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedCarryRows.Normalized.sourceColumns =
        Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyBodyResidualRetained.audit.localColumns /\
      finalColumns =
        Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyBodyResidualRetained.audit.finalColumns /\
      productionRowCount =
        Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyBodyResidualRetained.audit.sourceRows /\
      Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyBodyResidualRetained.audit.sourceRowStart =
        144277 /\
      Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyBodyResidualRetained.audit.sourceStarts =
        [144918, 145026, 145134] /\
      Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyBodyResidualRetained.audit.finalStarts =
        [1076091, 1078575, 1081059] := by
  native_decide

end Normalized

end Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedResidualRows
