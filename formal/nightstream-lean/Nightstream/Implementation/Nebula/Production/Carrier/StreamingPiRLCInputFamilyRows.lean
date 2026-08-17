import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiRLCInputPhaseRows
import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiRLCInputResidualRows

/-!
Contract: complete ordered commitment and residual rows for one production
PiRLC input family.

Assurance tier: generated source-row soundness.

Owns the connection from the exact 918-field family commitment block to the
108-field additive residual update. The commitment output columns are the
phase columns in the residual equations, so one accepted assignment must use
the same local commitment in both row families.

Does not own the production setup identity, Poseidon2 replay, PiRLC arithmetic
rows, family-state glue, telescoping, or the terminal zero check.

Emits constraints: 114,091 R1CS rows.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputFamilyRows

open Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputBinding
open Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputBindingSetup
open Nightstream.Implementation.R1CS

private abbrev PhaseLayout :=
  Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputPhaseRows.Layout

private abbrev ResidualLayout :=
  Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputResidualRows.Layout

private abbrev phaseRows :=
  Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputPhaseRows.rows

private abbrev residualRows :=
  Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputResidualRows.rows

/-- Columns for one complete family block. The residual rows reuse the compact
commitment output columns from `phase`. -/
structure Layout where
  phase : PhaseLayout
  beforeResidual : Fin (shape.rows * shape.degree) → Nat
  afterResidual : Fin (shape.rows * shape.degree) → Nat

/-- The residual-link view of the complete family layout. -/
def residualLayout (layout : Layout) : ResidualLayout where
  beforeResidual := layout.beforeResidual
  phaseBinding := layout.phase.outputColumn
  afterResidual := layout.afterResidual

/-- Exact row order: family commitment, then additive residual update. -/
def rows
    (setup : SeededAjtai.Setup verifierRows messageColumnCount)
    (layout : Layout) (family : Family) : List Row :=
  phaseRows setup layout.phase family ++ residualRows (residualLayout layout)

theorem rows_length
    (setup : SeededAjtai.Setup verifierRows messageColumnCount)
    (layout : Layout) (family : Family) :
    (rows setup layout family).length = 114091 := by
  simp only [rows, List.length_append, phaseRows, residualRows,
    Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputPhaseRows.rows_length,
    Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputResidualRows.rows_length]

/-- The source assignment places the residual before and after values. The
phase value is derived from the family commitment rows. -/
def ResidualsPlaced
    (layout : Layout) (assignment : Nat → Nat)
    (before after : ResidualFields) : Prop :=
  (∀ output,
    assignment (layout.beforeResidual output) = (before output).val) /\
  (∀ output,
    assignment (layout.afterResidual output) = (after output).val)

private theorem phase_satisfies
    {setup : SeededAjtai.Setup verifierRows messageColumnCount}
    {layout : Layout} {family : Family} {assignment : Nat → Nat}
    (satisfies : Satisfies (rows setup layout family) assignment) :
    Satisfies (phaseRows setup layout.phase family) assignment := by
  intro row member
  exact satisfies row (List.mem_append_left _ member)

private theorem residual_satisfies
    {setup : SeededAjtai.Setup verifierRows messageColumnCount}
    {layout : Layout} {family : Family} {assignment : Nat → Nat}
    (satisfies : Satisfies (rows setup layout family) assignment) :
    Satisfies (residualRows (residualLayout layout)) assignment := by
  intro row member
  exact satisfies row (List.mem_append_right _ member)

/-- Semantic result of the complete family block. -/
structure Exact
    (setup : SeededAjtai.Setup verifierRows messageColumnCount)
    (layout : Layout) (family : Family) (assignment : Nat → Nat)
    (inputs : Source → RingF) (before after : ResidualFields) : Prop where
  phase :
    Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputPhaseRows.Exact
      setup layout.phase family assignment inputs
  transition : ConcreteResidualTransition setup before after family inputs

/-- Main soundness theorem. The same accepted commitment output is used as
the additive phase value; no claimed commitment is a premise. -/
theorem rows_sound
    {setup : SeededAjtai.Setup verifierRows messageColumnCount}
    {layout : Layout} {family : Family} {assignment : Nat → Nat}
    {inputs : Source → RingF} {before after : ResidualFields}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (inputsPlaced :
      Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputPhaseRows.InputsPlaced
        layout.phase assignment inputs)
    (residualsPlaced : ResidualsPlaced layout assignment before after)
    (satisfies : Satisfies (rows setup layout family) assignment) :
    Exact setup layout family assignment inputs before after := by
  have phaseExact :=
    Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputPhaseRows.rows_sound
      canonical one inputsPlaced
    (phase_satisfies satisfies)
  have placed :
      Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputResidualRows.ColumnsPlaced
        (residualLayout layout) assignment before
          (concretePhaseBinding setup family inputs) after := by
    refine ⟨residualsPlaced.1, ?_, residualsPlaced.2⟩
    intro output
    exact
      Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputPhaseRows.Exact.output_at
        phaseExact output
  have transition :=
    Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputResidualRows.rows_imply_concreteResidualTransition
      one placed (residual_satisfies satisfies) setup family inputs rfl
  exact ⟨phaseExact, transition⟩

end Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputFamilyRows
