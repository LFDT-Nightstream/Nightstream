import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.Artifact.Row.Boolean

/-!
Contract: coefficient-derived semantics of one compact Poseidon2 S-box row.

Owns: exact general-selector classification, equality with the independent
S-box point, and the active-row equivalence between zero residual and the
degree-seven output equation.

Does not own: a Rust family label, selector authority, source-to-final
assignment decoding, production row coverage, or permission to remove rows.

Emits constraints: no.

Assurance tier: model-level.
-/

set_option autoImplicit false
set_option maxRecDepth 10000

namespace Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Row.Sbox

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Row.Decoder
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Row.Boolean
open Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Components
open Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Ports
open Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Rows

/-- Exact coefficient shape of one compact S-box row. The input and output
ports can be arbitrary sparse linear combinations. Every other arithmetic
port is empty. -/
def IsSboxAt (row : DecodedRow)
    (selectorColumn : Fin row.columns) : Prop :=
  (row.port Role.generalSelector.index).terms =
      [{ column := selectorColumn, coefficient := 1 }] ∧
    ∀ port : Fin 13,
      port ≠ Role.generalSelector.index →
      port ≠ Role.sboxInput.index →
      port ≠ Role.c.index →
      (row.port port).terms = []

instance (row : DecodedRow) (selectorColumn : Fin row.columns) :
    Decidable (IsSboxAt row selectorColumn) := by
  unfold IsSboxAt
  infer_instance

structure ValidatedSboxRow (row : DecodedRow) where
  selectorColumn : Fin row.columns
  shape : IsSboxAt row selectorColumn

def validateSboxAt (row : DecodedRow)
    (selectorColumn : Fin row.columns) :
    Option (ValidatedSboxRow row) :=
  if shape : IsSboxAt row selectorColumn then
    some ⟨selectorColumn, shape⟩
  else
    none

theorem rowPoint_eq_sboxPoint
    (row : DecodedRow) (validated : ValidatedSboxRow row)
    (assignment : Fin row.columns → F) :
    rowPoint row assignment =
      sboxPoint
        (assignment validated.selectorColumn)
        (action (row.port Role.sboxInput.index) assignment)
        (action (row.port Role.c.index) assignment) := by
  funext port
  by_cases selectorPort : port = Role.generalSelector.index
  · subst port
    simp only [rowPoint, action]
    rw [validated.shape.1]
    simp [sboxPoint, sparsePoint, Role.index, Fin.one_mul]
  · by_cases inputPort : port = Role.sboxInput.index
    · subst port
      simp [rowPoint, sboxPoint, sparsePoint, Role.index, action]
    · by_cases outputPort : port = Role.c.index
      · subst port
        simp [rowPoint, sboxPoint, sparsePoint, Role.index, action]
      · simp only [rowPoint, action]
        rw [validated.shape.2 port selectorPort inputPort outputPort]
        have portNeGeneral : port ≠ (1 : Fin 13) := by
          simpa only [Role.index] using selectorPort
        have portNeInput : port ≠ (5 : Fin 13) := by
          simpa only [Role.index] using inputPort
        have portNeOutput : port ≠ (4 : Fin 13) := by
          simpa only [Role.index] using outputPort
        simp [sboxPoint, sparsePoint, Role.index, portNeGeneral,
          portNeInput, portNeOutput]

private theorem evaluate_sboxPoint_one
    (input output : F) :
    Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Semantics.evaluate
        (sboxPoint 1 input output) =
      input * input * input * input * input * input * input - output := by
  rw [evaluate_sboxPoint]
  simp [productResidual, sboxResidual, sboxPoint, sparsePoint, Role.index,
    Fin.one_mul, Fin.sub_eq_add_neg, Lean.Grind.Fin.add_comm]
  congr 1
  exact Fin.zero_add _

/-- With the selected branch fixed to one, the compact row is exactly the
seventh-power equation. -/
theorem residual_zero_iff_output_eq_sbox7
    (row : DecodedRow) (validated : ValidatedSboxRow row)
    (assignment : Fin row.columns → F)
    (selectorOne : assignment validated.selectorColumn = 1) :
    residual row assignment = 0 ↔
      action (row.port Role.c.index) assignment =
        action (row.port Role.sboxInput.index) assignment *
          action (row.port Role.sboxInput.index) assignment *
          action (row.port Role.sboxInput.index) assignment *
          action (row.port Role.sboxInput.index) assignment *
          action (row.port Role.sboxInput.index) assignment *
          action (row.port Role.sboxInput.index) assignment *
          action (row.port Role.sboxInput.index) assignment := by
  change
    Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Semantics.evaluate
        (rowPoint row assignment) = 0 ↔ _
  rw [rowPoint_eq_sboxPoint row validated assignment, selectorOne,
    evaluate_sboxPoint_one]
  rw [Lean.Grind.AddCommGroup.sub_eq_zero_iff]
  constructor <;> intro same <;> exact same.symm

end Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Row.Sbox
