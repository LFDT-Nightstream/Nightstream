import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Semantics.Nc

/-!
Contract: restate the independent full-carrier NC obligation on the canonical
Phi81 block×lane representation.

Owns: semantic reuse of the canonical total block/lane-to-carrier map, its
inverse at the Split-NC boundary, and soundness/completeness of one cubic
residual per block/lane for the existing full-carrier strict-norm statement.

Does not own: Boolean padding, multilinear evaluation, output claims,
Π_RLC action compatibility, transcripts, Rust, R1CS, costs, or row removal.

Emits constraints: no.

Authority boundary: every block/lane value is read from the authoritative
complete assignment. There is no supplied sidecar, digest, polynomial, or
production witness in this definition.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_ccs.nc.block_lane.layout` | reuse `Phi81CarrierLayout.carrierColumn`; `blockCount * 54 = carrierWidth` (fixed profile: `5 * 54 = 270`) without omission or duplication | computed | `carrierColumn`, `carrierColumn_decode`, `blockCount_mul_ringDegree_eq_carrierWidth` |
| `nifs.pi_ccs.nc.block_lane.value` | each block/lane leaf is the corresponding authoritative assignment coordinate | computed | `value`, `value_decode` |
| `nifs.pi_ccs.nc.block_lane.residual.completeness` | strict norm zeros every block/lane cubic | derived | `residualsZero_of_truth` |
| `nifs.pi_ccs.nc.block_lane.residual.soundness` | every block/lane cubic zero implies strict norm on every carrier coordinate | checked | `truth_of_residualsZero` |
| `nifs.pi_ccs.nc.block_lane.residual.exact` | block/lane residuals are equivalent to full-carrier NC truth | derived | `residualsZero_iff_truth` |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Semantics.Nc.BlockLane

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources

/-- The complete carrier coordinate owned by one canonical Phi81 block/lane
pair. The completed carrier contains no partial final block. -/
def carrierColumn
    {shape : SemanticShape}
    (block : Fin (Phi81ColumnLayout.blockCount shape.carrierWidth))
    (lane : Fin ringDegree) : Fin shape.carrierWidth :=
  Phi81CarrierLayout.carrierColumn block lane

/-- Decoding an authoritative flat carrier coordinate and flattening the
resulting block/lane pair recovers that exact coordinate. -/
theorem carrierColumn_decode
    {shape : SemanticShape}
    (column : Fin shape.carrierWidth) :
    carrierColumn (Phi81ColumnLayout.decode column).1
        (Phi81ColumnLayout.decode column).2 = column := by
  apply Fin.ext
  exact Phi81ColumnLayout.flatIndex_decode column

/-- The canonical block/lane domain has exactly the complete carrier width;
there are no duplicate live cells and no hidden tail coordinates. -/
theorem blockCount_mul_ringDegree_eq_carrierWidth
    (shape : SemanticShape) :
    Phi81ColumnLayout.blockCount shape.carrierWidth * ringDegree =
      shape.carrierWidth := by
  unfold SemanticShape.carrierWidth
  rw [Phi81CarrierLayout.blockCount_carrierWidth]
  exact (Phi81CarrierLayout.carrierWidth_eq shape.logicalWidth).symm

/-- Read one canonical block/lane coefficient from the authoritative complete
assignment. -/
def value
    {shape : SemanticShape}
    (assignment : PaperLinearAlgebra.Assignment F shape.carrierWidth)
    (block : Fin (Phi81ColumnLayout.blockCount shape.carrierWidth))
    (lane : Fin ringDegree) : F :=
  assignment (carrierColumn block lane)

/-- The block/lane view recovers every authoritative flat coordinate exactly. -/
theorem value_decode
    {shape : SemanticShape}
    (assignment : PaperLinearAlgebra.Assignment F shape.carrierWidth)
    (column : Fin shape.carrierWidth) :
    value assignment (Phi81ColumnLayout.decode column).1
        (Phi81ColumnLayout.decode column).2 = assignment column := by
  unfold value
  rw [carrierColumn_decode]

/-- One strict-norm cubic residual for every source and every canonical
block/lane coefficient. -/
def ResidualsZero
    {shape : SemanticShape}
    (data : Data shape) : Prop :=
  forall source block lane,
    NormRange.cubicResidual
      (value (data.assignment source) block lane) = 0

/-- Full-carrier strict norm makes every canonical block/lane cubic vanish. -/
theorem residualsZero_of_truth
    {shape : SemanticShape}
    (data : Data shape) :
    Nc.Truth data -> ResidualsZero data := by
  intro truth source block lane
  have bounded :
      centeredMagnitude
        (value (data.assignment source) block lane) < 2 :=
    truth source (carrierColumn block lane)
  have represented := (NormRange.strictNormTwo_iff_representedRoot
    (value (data.assignment source) block lane)).mp bounded
  rcases represented with negative | zero | one
  · rw [negative]
    decide
  · rw [zero]
    rfl
  · rw [one]
    rfl

/-- If every canonical block/lane cubic vanishes, every authoritative carrier
coordinate satisfies the strict norm. -/
theorem truth_of_residualsZero
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    {shape : SemanticShape}
    (data : Data shape) :
    ResidualsZero data -> Nc.Truth data := by
  intro residuals source column
  apply (NormRange.cubicResidual_eq_zero_iff_strictNormTwo
    noZeroDivisors (data.assignment source column)).mp
  simpa [value_decode] using
    residuals source (Phi81ColumnLayout.decode column).1
      (Phi81ColumnLayout.decode column).2

/-- The canonical block×lane residual family is sound and complete for the
same independent full-carrier NC truth as the flat semantic statement. -/
theorem residualsZero_iff_truth
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    {shape : SemanticShape}
    (data : Data shape) :
    ResidualsZero data <-> Nc.Truth data :=
  ⟨truth_of_residualsZero noZeroDivisors data,
    residualsZero_of_truth data⟩

end Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Semantics.Nc.BlockLane
