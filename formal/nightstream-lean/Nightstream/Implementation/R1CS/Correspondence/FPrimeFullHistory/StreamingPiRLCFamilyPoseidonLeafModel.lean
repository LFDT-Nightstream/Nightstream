import Batteries.Data.List.Basic
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonLeafSchema
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.Artifact.Decoder
import Nightstream.Implementation.R1CS.Correspondence.SelectiveCcs.Polynomial.Rows

/-!
Contract: model-level decoder and one-row semantics for the relative production
PiRLC Poseidon2 leaf artifact.

Assurance tier: model-level.

Owns: fail-closed role and coefficient decoding, relative source and final
actions, and the implication from one active final row to one S-box equation.

Does not own: validity of the generated leaf, replay-batch coverage, equality
with an absolute Rust assignment, a complete permutation, or lifecycle
soundness.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonLeafModel

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonLeafSchema
open Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial
open Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Ports

namespace Wire

inductive SourceColumn where
  | externalA (lane : Fin 4)
  | externalB (lane : Fin 4)
  | local (offset : Fin 600)
deriving DecidableEq, Repr

structure SourceTerm where
  column : SourceColumn
  coefficient : F
deriving DecidableEq, Repr

structure SourceLinearCombination where
  constant : F
  terms : List SourceTerm
deriving DecidableEq, Repr

inductive ExplicitColumn where
  | one
  | selector
deriving DecidableEq, Repr

structure ExplicitTerm where
  column : ExplicitColumn
  coefficient : F
deriving DecidableEq, Repr

inductive Slot where
  | externalA (lane : Fin 4)
  | externalB (lane : Fin 4)
  | previousLocal (index : Fin 86)
  | local (index : Fin 86)
deriving DecidableEq, Repr

structure GeometricRun where
  slot : Slot
  initial : F
  ratio : F
deriving DecidableEq, Repr

structure Port where
  explicit : List ExplicitTerm
  geometric : List GeometricRun
deriving DecidableEq, Repr

structure Step where
  rowOffset : Fin 86
  input : SourceLinearCombination
  output : SourceLinearCombination
deriving DecidableEq, Repr

structure Row where
  rowOffset : Fin 86
  ports : List Port
  portsLength : ports.length = 13

private def decodeField :=
  Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Decoder.decodeField

def decodeSourceColumn : RawSourceColumn → Option SourceColumn
  | .externalA lane =>
      if bounded : lane < 4 then some (.externalA ⟨lane, bounded⟩) else none
  | .externalB lane =>
      if bounded : lane < 4 then some (.externalB ⟨lane, bounded⟩) else none
  | .local offset =>
      if bounded : offset < 600 then some (.local ⟨offset, bounded⟩) else none

def decodeSourceTerm (raw : RawSourceTerm) : Option SourceTerm := do
  let column ← decodeSourceColumn raw.column
  let coefficient ← decodeField raw.coefficient
  pure { column, coefficient }

def decodeSourceLinearCombination
    (raw : RawSourceLinearCombination) : Option SourceLinearCombination := do
  let constant ← decodeField raw.constant
  let terms ← raw.terms.mapM decodeSourceTerm
  pure { constant, terms }

def decodeExplicitColumn : RawExplicitColumn → ExplicitColumn
  | .one => .one
  | .selector => .selector

def decodeExplicitTerm (raw : RawExplicitTerm) : Option ExplicitTerm := do
  let coefficient ← decodeField raw.coefficient
  pure { column := decodeExplicitColumn raw.column, coefficient }

def decodeSlot : RawSlot → Option Slot
  | .externalA lane =>
      if bounded : lane < 4 then some (.externalA ⟨lane, bounded⟩) else none
  | .externalB lane =>
      if bounded : lane < 4 then some (.externalB ⟨lane, bounded⟩) else none
  | .previousLocal index =>
      if bounded : index < 86 then
        some (.previousLocal ⟨index, bounded⟩)
      else
        none
  | .local index =>
      if bounded : index < 86 then some (.local ⟨index, bounded⟩) else none

def decodeGeometricRun (raw : RawGeometricRun) : Option GeometricRun := do
  let slot ← decodeSlot raw.slot
  let initial ← decodeField raw.initial
  let ratio ← decodeField raw.ratio
  pure { slot, initial, ratio }

def decodePort (raw : RawPort) : Option Port := do
  let explicit ← raw.explicit.mapM decodeExplicitTerm
  let geometric ← raw.geometric.mapM decodeGeometricRun
  pure { explicit, geometric }

def decodeStep (raw : RawStep) : Option Step := do
  if bounded : raw.rowOffset < 86 then
    let input ← decodeSourceLinearCombination raw.input
    let output ← decodeSourceLinearCombination raw.output
    pure { rowOffset := ⟨raw.rowOffset, bounded⟩, input, output }
  else
    none

def decodeRow (raw : RawRow) : Option Row := do
  if bounded : raw.rowOffset < 86 then
    let ports ← raw.ports.mapM decodePort
    if portsLength : ports.length = 13 then
      pure { rowOffset := ⟨raw.rowOffset, bounded⟩, ports, portsLength }
    else
      none
  else
    none

def Row.port (row : Row) (index : Fin 13) : Port :=
  row.ports.get ⟨index.val, by rw [row.portsLength]; exact index.isLt⟩

end Wire

open Wire

structure SourceAssignment where
  externalA : Fin 4 → F
  externalB : Fin 4 → F
  localValue : Fin 600 → F

structure FinalAssignment where
  explicit : ExplicitColumn → F
  digit : Slot → Fin 41 → F

def emptyPort : Port where
  explicit := []
  geometric := []

def selectorPort : Port where
  explicit := [{ column := .selector, coefficient := 1 }]
  geometric := []

/-- Structural S-box classification for one relative leaf row. The input and
output ports stay arbitrary; all other arithmetic ports must be empty. -/
def IsSboxShape (row : Wire.Row) : Prop :=
  row.port Role.generalSelector.index = selectorPort ∧
    ∀ port : Fin 13,
      port ≠ Role.generalSelector.index →
      port ≠ Role.sboxInput.index →
      port ≠ Role.c.index →
      row.port port = emptyPort

instance (row : Wire.Row) : Decidable (IsSboxShape row) := by
  unfold IsSboxShape
  infer_instance

def sboxShapeCheck (row : Wire.Row) : Bool := decide (IsSboxShape row)

theorem sboxShapeCheck_sound (row : Wire.Row)
    (checked : sboxShapeCheck row = true) : IsSboxShape row := by
  exact of_decide_eq_true checked

def sourceValue (assignment : SourceAssignment) : SourceColumn → F
  | .externalA lane => assignment.externalA lane
  | .externalB lane => assignment.externalB lane
  | .local offset => assignment.localValue offset

def sum : List F → F
  | [] => 0
  | value :: tail => value + sum tail

def sourceAction (value : SourceLinearCombination)
    (assignment : SourceAssignment) : F :=
  value.constant + sum (value.terms.map fun term =>
    term.coefficient * sourceValue assignment term.column)

def geometricCoefficient (initial ratio : F) : Nat → F
  | 0 => initial
  | index + 1 => geometricCoefficient initial ratio index * ratio

def geometricAction (run : GeometricRun)
    (assignment : FinalAssignment) : F :=
  sum (List.ofFn fun index : Fin 41 =>
    geometricCoefficient run.initial run.ratio index.val *
      assignment.digit run.slot index)

def portAction (port : Port) (assignment : FinalAssignment) : F :=
  sum (port.explicit.map fun term =>
      term.coefficient * assignment.explicit term.column) +
    sum (port.geometric.map fun run => geometricAction run assignment)

/-- Exact source-to-final link for one decoded port. Constants and every
geometric digit run are related separately; no aggregate port equality is
assumed. -/
structure PortRealized
    (value : SourceLinearCombination) (port : Port)
    (source : SourceAssignment) (final : FinalAssignment) : Prop where
  constant :
    sum (port.explicit.map fun term =>
      term.coefficient * final.explicit term.column) = value.constant
  terms : List.Forall₂
    (fun term run =>
      geometricAction run final =
        term.coefficient * sourceValue source term.column)
    value.terms port.geometric

private theorem geometric_sum_eq_source_sum
    (source : SourceAssignment) (final : FinalAssignment) :
    ∀ {terms : List SourceTerm} {runs : List GeometricRun},
      List.Forall₂
          (fun term run =>
            geometricAction run final =
              term.coefficient * sourceValue source term.column)
          terms runs →
        sum (runs.map fun run => geometricAction run final) =
          sum (terms.map fun term =>
            term.coefficient * sourceValue source term.column) := by
  intro terms runs realized
  induction realized with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      simp only [List.map_cons, sum]
      rw [head, inductionHypothesis]

private theorem realized_geometric_sum
    {value : SourceLinearCombination} {port : Port}
    {source : SourceAssignment} {final : FinalAssignment}
    (realized : PortRealized value port source final) :
    sum (port.geometric.map fun run => geometricAction run final) =
      sum (value.terms.map fun term =>
        term.coefficient * sourceValue source term.column) :=
  geometric_sum_eq_source_sum source final realized.terms

theorem portRealized_action
    {value : SourceLinearCombination} {port : Port}
    {source : SourceAssignment} {final : FinalAssignment}
    (realized : PortRealized value port source final) :
    portAction port final = sourceAction value source := by
  unfold portAction sourceAction
  rw [realized.constant, realized_geometric_sum realized]

def point (row : Wire.Row) (assignment : FinalAssignment) : Fin 13 → F :=
  fun index => portAction (Wire.Row.port row index) assignment

def residual (row : Wire.Row) (assignment : FinalAssignment) : F :=
  Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Semantics.evaluate
    (point row assignment)

def IsSboxRow (row : Wire.Row) (assignment : FinalAssignment)
    (input output : F) : Prop :=
  point row assignment =
    Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Rows.sboxPoint
      1 input output

theorem isSboxRow_of_shape
    (row : Wire.Row) (assignment : FinalAssignment)
    (shape : IsSboxShape row)
    (selectorOne : assignment.explicit .selector = 1) :
    IsSboxRow row assignment
      (portAction (row.port Role.sboxInput.index) assignment)
      (portAction (row.port Role.c.index) assignment) := by
  unfold IsSboxRow
  funext port
  by_cases selectorIndex : port = Role.generalSelector.index
  · subst port
    simp only [point]
    rw [shape.1]
    simp [portAction, selectorPort, sum,
      Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Rows.sboxPoint,
      Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Rows.sparsePoint,
      Role.index, selectorOne, Fin.one_mul]
  · by_cases inputIndex : port = Role.sboxInput.index
    · subst port
      simp [point,
        Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Rows.sboxPoint,
        Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Rows.sparsePoint,
        Role.index]
    · by_cases outputIndex : port = Role.c.index
      · subst port
        simp [point,
          Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Rows.sboxPoint,
          Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Rows.sparsePoint,
          Role.index]
      · simp only [point]
        rw [shape.2 port selectorIndex inputIndex outputIndex]
        have portNeGeneral : port ≠ (1 : Fin 13) := by
          simpa only [Role.index] using selectorIndex
        have portNeInput : port ≠ (5 : Fin 13) := by
          simpa only [Role.index] using inputIndex
        have portNeOutput : port ≠ (4 : Fin 13) := by
          simpa only [Role.index] using outputIndex
        simp [emptyPort, portAction, sum,
          Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Rows.sboxPoint,
          Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Rows.sparsePoint,
          Role.index, portNeGeneral, portNeInput, portNeOutput]

/-- A decoded active row with the S-box port image enforces exactly one
seventh-power equation. -/
theorem residual_zero_iff_sbox
    (row : Wire.Row) (assignment : FinalAssignment) (input output : F)
    (shape : IsSboxRow row assignment input output) :
    residual row assignment = 0 ↔
      input * input * input * input * input * input * input = output := by
  unfold residual IsSboxRow at *
  rw [shape]
  exact
    Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Rows.evaluate_sboxPoint_one_eq_zero_iff
      input output

theorem residual_zero_iff_sbox_of_shape
    (row : Wire.Row) (assignment : FinalAssignment)
    (shape : IsSboxShape row)
    (selectorOne : assignment.explicit .selector = 1) :
    residual row assignment = 0 ↔
      portAction (row.port Role.sboxInput.index) assignment *
          portAction (row.port Role.sboxInput.index) assignment *
          portAction (row.port Role.sboxInput.index) assignment *
          portAction (row.port Role.sboxInput.index) assignment *
          portAction (row.port Role.sboxInput.index) assignment *
          portAction (row.port Role.sboxInput.index) assignment *
          portAction (row.port Role.sboxInput.index) assignment =
        portAction (row.port Role.c.index) assignment :=
  residual_zero_iff_sbox row assignment _ _
    (isSboxRow_of_shape row assignment shape selectorOne)

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonLeafModel
