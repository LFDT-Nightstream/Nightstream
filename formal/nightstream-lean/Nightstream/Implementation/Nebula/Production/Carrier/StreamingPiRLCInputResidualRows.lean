import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiRLCInputBindingSetup
import Nightstream.Implementation.R1CS.Canonical.KEquality

/-!
Contract: exact 108-row additive link for one production PiRLC input
residual update.

Assurance tier: generated source-row soundness.

Owns one Goldilocks equality row per rank-two Phi81 output field, exact row
geometry, source-column placement, soundness from accepted rows to the
field-level residual equation, and honest completeness.

Does not own the 918-field local commitment rows, the production setup
identity, Poseidon2 replay, other family-state links, normalized selective-CCS
slots, Rust assignment conformance, telescoping, or the terminal zero check.

Emits constraints: 108 linear R1CS rows.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputResidualRows

open Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputBinding
open Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputBindingSetup
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical

/-- Exact assignment columns for one additive residual update. -/
structure Layout where
  beforeResidual : Fin (shape.rows * shape.degree) → Nat
  phaseBinding : Fin (shape.rows * shape.degree) → Nat
  afterResidual : Fin (shape.rows * shape.degree) → Nat

/-- One field equation: `before = phase + after`. -/
def residualRow
    (layout : Layout) (output : Fin (shape.rows * shape.degree)) : Row :=
  KEquality.equalityRow
    [(layout.beforeResidual output, 1)]
    [(layout.phaseBinding output, 1), (layout.afterResidual output, 1)]

def rows (layout : Layout) : List Row :=
  List.ofFn (residualRow layout)

@[simp] theorem rows_length (layout : Layout) :
    (rows layout).length = 108 := by
  unfold rows
  rw [List.length_ofFn]
  exact exact_output_width

/-- The source assignment places all three exact 108-field values. -/
def ColumnsPlaced
    (layout : Layout) (assignment : Nat → Nat)
    (before phase after : ResidualFields) : Prop :=
  (∀ output,
    assignment (layout.beforeResidual output) = (before output).val) /\
  (∀ output,
    assignment (layout.phaseBinding output) = (phase output).val) /\
  (∀ output,
    assignment (layout.afterResidual output) = (after output).val)

private theorem residualRow_holds
    {layout : Layout} {assignment : Nat → Nat}
    (satisfies : Satisfies (rows layout) assignment)
    (output : Fin (shape.rows * shape.degree)) :
    RowHolds assignment (residualRow layout output) := by
  exact satisfies _ (List.mem_ofFn.mpr ⟨output, rfl⟩)

private theorem residualValue_lt (value : Nightstream.SuperNeo.Concrete.F) :
    value.val < goldilocksP := by
  simpa [goldilocksP, Nightstream.SuperNeo.Concrete.goldilocksModulus] using
    value.isLt

/-- One accepted row gives the exact Goldilocks addition equation. -/
theorem residualField_exact_of_row
    {layout : Layout} {assignment : Nat → Nat}
    {before phase after : ResidualFields}
    (one : assignment 0 = 1)
    (placed : ColumnsPlaced layout assignment before phase after)
    (satisfies : Satisfies (rows layout) assignment)
    (output : Fin (shape.rows * shape.degree)) :
    before output = phase output + after output := by
  apply Fin.ext
  change (before output).val =
    ((phase output).val + (after output).val) % goldilocksP
  have equal :=
    (KEquality.equalityRow_iff assignment
      [(layout.beforeResidual output, 1)]
      [(layout.phaseBinding output, 1),
        (layout.afterResidual output, 1)] one).mp
      (residualRow_holds satisfies output)
  simpa [lcEval, placed.1 output, placed.2.1 output, placed.2.2 output,
    Nat.mod_eq_of_lt (residualValue_lt (before output))] using equal

/-- Main soundness theorem for the complete 108-row block. -/
theorem rows_imply_addResidualFields
    {layout : Layout} {assignment : Nat → Nat}
    {before phase after : ResidualFields}
    (one : assignment 0 = 1)
    (placed : ColumnsPlaced layout assignment before phase after)
    (satisfies : Satisfies (rows layout) assignment) :
    before = addResidualFields phase after := by
  funext output
  exact residualField_exact_of_row one placed satisfies output

/-- Accepted rows and an exact local commitment derive the semantic family
residual transition. -/
theorem rows_imply_concreteResidualTransition
    {layout : Layout} {assignment : Nat → Nat}
    {before phase after : ResidualFields}
    (one : assignment 0 = 1)
    (placed : ColumnsPlaced layout assignment before phase after)
    (satisfies : Satisfies (rows layout) assignment)
    (setup : SeededAjtai.Setup verifierRows messageColumnCount)
    (family : Family) (inputs : Source → RingF)
    (phaseExact : phase = concretePhaseBinding setup family inputs) :
    ConcreteResidualTransition setup before after family inputs := by
  have update := rows_imply_addResidualFields one placed satisfies
  rw [phaseExact] at update
  exact update

/-- Honest values satisfy every residual-link row without auxiliary columns. -/
theorem rows_complete
    {layout : Layout} {assignment : Nat → Nat}
    {before phase after : ResidualFields}
    (one : assignment 0 = 1)
    (placed : ColumnsPlaced layout assignment before phase after)
    (update : before = addResidualFields phase after) :
    Satisfies (rows layout) assignment := by
  intro row member
  rcases List.mem_ofFn.mp member with ⟨output, rfl⟩
  apply (KEquality.equalityRow_iff assignment
    [(layout.beforeResidual output, 1)]
    [(layout.phaseBinding output, 1),
      (layout.afterResidual output, 1)] one).mpr
  have atOutput := congrArg Fin.val (congrFun update output)
  change (before output).val =
    ((phase output).val + (after output).val) % goldilocksP at atOutput
  simpa [addResidualFields, lcEval, placed.1 output, placed.2.1 output,
    placed.2.2 output,
    Nat.mod_eq_of_lt (residualValue_lt (before output))] using atOutput

end Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputResidualRows
