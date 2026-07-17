import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.NifsPaper.PiRlc
import Nightstream.SuperNeo.Concrete.Phi81Relation.Types

/-!
Typed point refinement for production `Pi_RLC`.

Assurance tier: model-level. This file defines a checked representation
boundary; it does not prove that physical transcript rows supply the point.

Owns: exact decoding of production extension-coordinate pairs into a
dimension-checked Phi81 relation point; the soundness/completeness contract of
that decoder; and propagation of one output-point binding through the shared
input-point wiring and into the named parent claim.

Does not own: the semantic relation shape; the `Pi_CCS`-derived point; physical
transcript connectivity; point-column serialization; source authority; R1CS
rows; costs; or row removal.

Emits constraints: no.

Authority boundary: point decoding is partial. A wrong physical coordinate
count produces `none`; no caller-supplied digest, point, or unchecked length
can override the verifier-owned shape. `OutputPointBound` remains an explicit
obligation until accepted `Pi_CCS` rows are proved to determine the same
typed point.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_rlc.verify.point.decode.dimension` | physical pair count equals the semantic row dimension | checked | `decodeTypedPoint`, `bound_iff` |
| `nifs.pi_rlc.verify.point.decode.coordinates` | every pair decodes in order to one `K` coordinate | computed | `pointOfLength`, `bound_iff` |
| `nifs.pi_rlc.verify.point.output` | the physical output point equals the verifier-owned typed point | retained check | `OutputPointBound` |
| `nifs.pi_rlc.verify.point.inputs` | every input inherits the bound output point | direct dataflow | `inputPointBound_of_outputPointBound` |
| `nifs.pi_rlc.verify.point.parent` | the named parent inherits the same bound point | direct dataflow | `parentPointBound_of_outputPointBound` |
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.PointBridge

open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper
open Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc

/-- Dimension-checked point type used by the independent Phi81 relation. -/
abbrev TypedPoint (shape : Shape) :=
  Nightstream.SuperNeo.Concrete.Phi81Relation.Point shape

private theorem typedPoint_eq_of_coordinates
    {shape : Shape} (left right : TypedPoint shape)
    (coordinates : left.coordinates = right.coordinates) : left = right := by
  cases left
  cases right
  cases coordinates
  rfl

private theorem pointColumns_eq_of_r_eq
    (left right : PointColumns) (r : left.r = right.r) : left = right := by
  cases left
  cases right
  cases r
  rfl

/-- Total decoder once the verifier-owned dimension check has succeeded. -/
def pointOfLength
    (shape : Shape) (assignment : Nat -> Nat) (columns : PointColumns)
    (dimension : columns.r.length = shape.rowVariables) : TypedPoint shape where
  coordinates := decodePointColumns assignment columns
  dimension := by
    simpa [decodePointColumns, extensionValues] using dimension

@[simp] theorem pointOfLength_coordinates
    (shape : Shape) (assignment : Nat -> Nat) (columns : PointColumns)
    (dimension : columns.r.length = shape.rowVariables) :
    (pointOfLength shape assignment columns dimension).coordinates =
      decodePointColumns assignment columns := by
  rfl

/-- Checked production decoder. A shape mismatch fails instead of padding,
truncating, or accepting proof data from a prover. -/
def decodeTypedPoint
    (shape : Shape) (assignment : Nat -> Nat)
    (columns : PointColumns) : Option (TypedPoint shape) :=
  if dimension : columns.r.length = shape.rowVariables then
    some (pointOfLength shape assignment columns dimension)
  else
    none

/-- One physical point carrier is exactly bound to one typed semantic point. -/
def Bound
    (shape : Shape) (assignment : Nat -> Nat)
    (columns : PointColumns) (point : TypedPoint shape) : Prop :=
  decodeTypedPoint shape assignment columns = some point

/-- The checked decoder succeeds exactly when both dimension and coordinate
content match. This is the complete representation contract. -/
theorem bound_iff
    (shape : Shape) (assignment : Nat -> Nat)
    (columns : PointColumns) (point : TypedPoint shape) :
    Bound shape assignment columns point <->
      columns.r.length = shape.rowVariables /\
        decodePointColumns assignment columns = point.coordinates := by
  by_cases dimension : columns.r.length = shape.rowVariables
  · simp only [Bound, decodeTypedPoint, dif_pos dimension, Option.some.injEq]
    constructor
    · intro equal
      exact ⟨dimension, by
        simpa using congrArg
          (fun value : TypedPoint shape => value.coordinates) equal⟩
    · intro facts
      exact typedPoint_eq_of_coordinates _ _ facts.2
  · simp [Bound, decodeTypedPoint, dimension]

@[simp] theorem decodeTypedPoint_isSome_iff
    (shape : Shape) (assignment : Nat -> Nat) (columns : PointColumns) :
    (decodeTypedPoint shape assignment columns).isSome = true <->
      columns.r.length = shape.rowVariables := by
  simp [decodeTypedPoint]

/-- The output point is the sole point authority surface for the production
`Pi_RLC` attempt. A later transcript bridge must discharge this predicate. -/
def OutputPointBound
    {matrixCount : Nat}
    {params : Nightstream.SuperNeo.GlobalParams}
    {arity : Nightstream.SuperNeo.Folding.BatchArity params}
    (shape : Shape) (assignment : Nat -> Nat)
    (columns : BatchColumns params arity matrixCount)
    (point : TypedPoint shape) : Prop :=
  Bound shape assignment columns.outputPoint point

/-- Shared point-column wiring makes every source inherit the one bound output
point; no per-input point check remains. -/
theorem inputPointBound_of_outputPointBound
    {matrixCount : Nat}
    {params : Nightstream.SuperNeo.GlobalParams}
    {arity : Nightstream.SuperNeo.Folding.BatchArity params}
    {shape : Shape} {assignment : Nat -> Nat}
    {columns : BatchColumns params arity matrixCount}
    {tree : TraceTree arity matrixCount}
    {point : TypedPoint shape}
    (wiring : EquationWiringArtifact columns tree)
    (outputBound : OutputPointBound shape assignment columns point)
    (index : Fin arity.total) :
    Bound shape assignment (columns.inputPoints index) point := by
  rw [wiring.pointColumns index]
  exact outputBound

/-- Parent-column identity propagates the same typed point into the explicitly
named `Pi_DEC` parent claim. -/
theorem parentPointBound_of_outputPointBound
    {matrixCount : Nat}
    {params : Nightstream.SuperNeo.GlobalParams}
    {arity : Nightstream.SuperNeo.Folding.BatchArity params}
    {shape : Shape} {assignment : Nat -> Nat}
    {columns : BatchColumns params arity matrixCount}
    {point : TypedPoint shape}
    (parent : ParentArtifact columns)
    (outputBound : OutputPointBound shape assignment columns point) :
    Bound shape assignment { r := columns.parentClaim.rCols } point := by
  have pointColumns :
      ({ r := columns.parentClaim.rCols } : PointColumns) =
        columns.outputPoint := by
    exact pointColumns_eq_of_r_eq _ _ parent.r.symm
  rw [pointColumns]
  exact outputBound

end Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.PointBridge
