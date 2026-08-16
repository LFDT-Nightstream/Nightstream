import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.Artifact.Row.Gating
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.Artifact.Row.SelectorComposition
import Nightstream.Implementation.R1CS.Correspondence.SelectiveCcs.Rewrite.Artifact.FinalAssignment

/-!
Contract: same-assignment bridge from one retained normalized product row to
one decoded source R1CS row.

Assurance tier: model-level.

Owns the implication from exact A/B/C final-port images and an active selector
to the source row on the source assignment reconstructed from those same final
coordinates. It also transports the result through exact matrix-row action.

Does not own a generated artifact, source-row coverage, trace-eliminated value
reconstruction, selector authority, Rust conformance, or matrix equality.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.RetainedRowBridge

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Interpreter
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Row.Decoder
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Row.Boolean
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Row.SelectorComposition
open Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Ports
open Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.Decoder
open Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.FinalAssignment
open Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.SourceImage
open Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.SourceRelation

/-- Exact final images of the three source linear combinations in one
retained R1CS row. -/
def RetainedImagesMatch
    {sourceRows sourceColumns : Nat}
    (row : DecodedRow)
    (sourceRow : DecodedSourceR1csRow sourceRows sourceColumns)
    (slots : List (DecodedSourceSlot sourceColumns row.columns))
    (definitions : List (DecodedSourceDefinition sourceColumns))
    (fuel : Nat) : Prop :=
  Form.Equivalent
      (Form.ofPort (row.port Role.a.index))
      (sourceLinearForm row.columnsPositive slots definitions fuel sourceRow.a) /\
    Form.Equivalent
      (Form.ofPort (row.port Role.b.index))
      (sourceLinearForm row.columnsPositive slots definitions fuel sourceRow.b) /\
    Form.Equivalent
      (Form.ofPort (row.port Role.c.index))
      (sourceLinearForm row.columnsPositive slots definitions fuel sourceRow.c)

instance
    {sourceRows sourceColumns : Nat}
    (row : DecodedRow)
    (sourceRow : DecodedSourceR1csRow sourceRows sourceColumns)
    (slots : List (DecodedSourceSlot sourceColumns row.columns))
    (definitions : List (DecodedSourceDefinition sourceColumns))
    (fuel : Nat) :
    Decidable (RetainedImagesMatch row sourceRow slots definitions fuel) := by
  unfold RetainedImagesMatch
  infer_instance

private theorem action_eq_sourceValue
    {sourceColumns : Nat}
    (row : DecodedRow)
    (port : Fin 13)
    (value : DecodedSourceLinearCombination sourceColumns)
    (slots : List (DecodedSourceSlot sourceColumns row.columns))
    (definitions : List (DecodedSourceDefinition sourceColumns))
    (fuel : Nat)
    (imageMatch :
      Form.Equivalent
        (Form.ofPort (row.port port))
        (sourceLinearForm row.columnsPositive slots definitions fuel value))
    (assignment : Fin row.columns -> F)
    (constantOne : assignment ⟨0, row.columnsPositive⟩ = 1) :
    action (row.port port) assignment =
      linearValue value
        (sourceAssignment row.columnsPositive slots definitions fuel assignment)
        1 := by
  calc
    action (row.port port) assignment =
        Form.evaluate (Form.ofPort (row.port port)) assignment :=
      (Form.evaluate_ofPort (row.port port) assignment).symm
    _ = Form.evaluate
          (sourceLinearForm row.columnsPositive slots definitions fuel value)
          assignment :=
      Form.evaluate_congr imageMatch assignment
    _ = linearValue value
          (sourceAssignment row.columnsPositive slots definitions fuel assignment)
          1 :=
      evaluate_sourceLinearForm_eq_linearValue row.columnsPositive slots
        definitions fuel value assignment constantOne

/-- The retained row's three normalized port actions are exactly its decoded
source A, B, and C values on one reconstructed source assignment. -/
theorem actions_eq_source_values
    {sourceRows sourceColumns : Nat}
    (row : DecodedRow)
    (sourceRow : DecodedSourceR1csRow sourceRows sourceColumns)
    (slots : List (DecodedSourceSlot sourceColumns row.columns))
    (definitions : List (DecodedSourceDefinition sourceColumns))
    (fuel : Nat)
    (imageMatch : RetainedImagesMatch row sourceRow slots definitions fuel)
    (assignment : Fin row.columns -> F)
    (constantOne : assignment ⟨0, row.columnsPositive⟩ = 1) :
    action (row.port Role.a.index) assignment =
        linearValue sourceRow.a
          (sourceAssignment row.columnsPositive slots definitions fuel assignment)
          1 /\
      action (row.port Role.b.index) assignment =
        linearValue sourceRow.b
          (sourceAssignment row.columnsPositive slots definitions fuel assignment)
          1 /\
      action (row.port Role.c.index) assignment =
        linearValue sourceRow.c
          (sourceAssignment row.columnsPositive slots definitions fuel assignment)
          1 := by
  exact
    ⟨action_eq_sourceValue row Role.a.index sourceRow.a slots definitions fuel
        imageMatch.1 assignment constantOne,
      action_eq_sourceValue row Role.b.index sourceRow.b slots definitions fuel
        imageMatch.2.1 assignment constantOne,
      action_eq_sourceValue row Role.c.index sourceRow.c slots definitions fuel
        imageMatch.2.2 assignment constantOne⟩

/-- An active retained normalized row is satisfied exactly when its decoded
source row is satisfied on the source assignment reconstructed from the same
final coordinates. -/
theorem residual_zero_iff_rowHolds
    {sourceRows sourceColumns : Nat}
    (row : DecodedRow)
    (validated : ValidatedProductGateRow row)
    (sourceRow : DecodedSourceR1csRow sourceRows sourceColumns)
    (slots : List (DecodedSourceSlot sourceColumns row.columns))
    (definitions : List (DecodedSourceDefinition sourceColumns))
    (fuel : Nat)
    (imageMatch : RetainedImagesMatch row sourceRow slots definitions fuel)
    (assignment : Fin row.columns -> F)
    (constantOne : assignment ⟨0, row.columnsPositive⟩ = 1)
    (selectorOne : assignment validated.selectorColumn = 1) :
    residual row assignment = 0 <->
      Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.SourceRelation.RowHolds sourceRow
        (sourceAssignment row.columnsPositive slots definitions fuel assignment)
        1 := by
  have values := actions_eq_source_values row sourceRow slots definitions fuel
    imageMatch assignment constantOne
  rw [residual_eq_gatedSource row validated assignment, selectorOne, Fin.one_mul]
  unfold sourceResidual
  rw [values.1, values.2.1, values.2.2]
  rw [rowHolds_iff_residual_zero]
  simp only [rowResidual, Fin.sub_eq_add_neg]

/-- Exact compact-matrix action transports the retained-row equivalence to
the physical interpreted relation row. -/
theorem physical_residual_zero_iff_rowHolds
    {sourceRows sourceColumns : Nat}
    (row : DecodedRow)
    (relation : InterpretedRelation row.rows row.columns)
    (exact :
      Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Row.Gating.ExactRowAction
        row relation)
    (validated : ValidatedProductGateRow row)
    (sourceRow : DecodedSourceR1csRow sourceRows sourceColumns)
    (slots : List (DecodedSourceSlot sourceColumns row.columns))
    (definitions : List (DecodedSourceDefinition sourceColumns))
    (fuel : Nat)
    (imageMatch : RetainedImagesMatch row sourceRow slots definitions fuel)
    (assignment : Fin row.columns -> F)
    (constantOne : assignment ⟨0, row.columnsPositive⟩ = 1)
    (selectorOne : assignment validated.selectorColumn = 1) :
    Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.RowAction.residualAt
          relation assignment row.emittedRow = 0 <->
      Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.SourceRelation.RowHolds sourceRow
        (sourceAssignment row.columnsPositive slots definitions fuel assignment)
        1 := by
  rw [exact.residualAt_eq_decoded assignment]
  exact residual_zero_iff_rowHolds row validated sourceRow slots definitions
    fuel imageMatch assignment constantOne selectorOne

end Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.RetainedRowBridge
