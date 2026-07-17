import Nightstream.Implementation.R1CS.Correspondence.PiCcsOutputDigest.Semantics
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.ProductAlignment
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Types

/-!
Lossless diagnostic projection from a three-matrix Split-NC `Pi_CCS` output
carrier to the legacy terminal output-digest message family.

Assurance tier: model-level (diagnostic profile); not the active 13-matrix
relation, not Rust-conformant, and not security-reduced. This module names the
two fixed dimensions required by the legacy digest and proves that, under the
explicit three-matrix premise, the projection omits no coordinate.

Owns: exact source-count and matrix-count profile facts; finite-index
transport into the 15-output, three-row digest carrier; coordinate equations;
and projection injectivity.

Does not own: proof that a concrete F-prime shape has this profile; the active
13-matrix output carrier or serialization; output authority, SIS maps,
Poseidon2, transcript state, Rust, R1CS rows, costs, or row removal.

Emits constraints: no.

Authority boundary: `Profile` is proof data, not a caller-selected truncation.
Once supplied, `projectOutputs` is injective over every coordinate of that
three-matrix profile. It creates no semantic authority for those values and
cannot be instantiated by the independently specified 13-matrix relation.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_ccs.output_digest.profile.sources` | semantic source count is exactly 15 | checked profile | `Profile.sourceCount_eq` |
| `nifs.pi_ccs.output_digest.profile.matrices` | semantic matrix count is exactly 3 | checked profile | `Profile.matrixCount_eq` |
| `nifs.pi_ccs.output_digest.projection` | preserve every active `yRing` and `yZcol` coordinate | direct dataflow | `projectOutput_yRing`, `projectOutput_yZcol` |
| `nifs.pi_ccs.output_digest.projection.lossless` | equal projected payloads imply equal Split-NC claims | derived | `projectOutputs_injective` |
-/

namespace Nightstream.Implementation.R1CS.PiCcsOutputDigest.Projection.SplitNc

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.PiCCS
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier

/-- Exact semantic dimensions consumed by the terminal output digest. -/
structure Profile (shape : SemanticShape) : Prop where
  sourceCount_eq : shape.sourceCount = Semantics.outputCount
  matrixCount_eq : shape.matrixCount = Semantics.yRingRows

namespace Profile

/-- Partition alignment plus explicit terminal and matrix dimensions produces
the fixed output-digest profile without a second source-index convention. -/
def ofAlignment
    {shape : SemanticShape}
    {params : GlobalParams}
    {arity : BatchArity params}
    (alignment : SourceAlignment shape params arity)
    (total_eq : arity.total = Semantics.outputCount)
    (matrixCount_eq : shape.matrixCount = Semantics.yRingRows) :
    Profile shape where
  sourceCount_eq := alignment.total_eq_sourceCount.symm.trans total_eq
  matrixCount_eq := matrixCount_eq

end Profile

/-- Read one terminal output index as the corresponding semantic source. -/
def sourceIndex
    {shape : SemanticShape}
    (profile : Profile shape)
    (output : Fin Semantics.outputCount) : Fin shape.sourceCount :=
  Fin.cast profile.sourceCount_eq.symm output

/-- Read one semantic source as the corresponding terminal output index. -/
def outputIndex
    {shape : SemanticShape}
    (profile : Profile shape)
    (source : Fin shape.sourceCount) : Fin Semantics.outputCount :=
  Fin.cast profile.sourceCount_eq source

/-- Read one fixed digest row as the corresponding semantic matrix index. -/
def matrixIndex
    {shape : SemanticShape}
    (profile : Profile shape)
    (row : Fin Semantics.yRingRows) : Fin shape.matrixCount :=
  Fin.cast profile.matrixCount_eq.symm row

/-- Read one semantic matrix index as the corresponding fixed digest row. -/
def rowIndex
    {shape : SemanticShape}
    (profile : Profile shape)
    (matrix : Fin shape.matrixCount) : Fin Semantics.yRingRows :=
  Fin.cast profile.matrixCount_eq matrix

@[simp] theorem sourceIndex_outputIndex
    {shape : SemanticShape}
    (profile : Profile shape)
    (source : Fin shape.sourceCount) :
    sourceIndex profile (outputIndex profile source) = source := by
  apply Fin.ext
  rfl

@[simp] theorem outputIndex_sourceIndex
    {shape : SemanticShape}
    (profile : Profile shape)
    (output : Fin Semantics.outputCount) :
    outputIndex profile (sourceIndex profile output) = output := by
  apply Fin.ext
  rfl

@[simp] theorem matrixIndex_rowIndex
    {shape : SemanticShape}
    (profile : Profile shape)
    (matrix : Fin shape.matrixCount) :
    matrixIndex profile (rowIndex profile matrix) = matrix := by
  apply Fin.ext
  rfl

@[simp] theorem rowIndex_matrixIndex
    {shape : SemanticShape}
    (profile : Profile shape)
    (row : Fin Semantics.yRingRows) :
    rowIndex profile (matrixIndex profile row) = row := by
  apply Fin.ext
  rfl

/-- One semantic source projected into the exact active digest payload. -/
def projectOutput
    {shape : SemanticShape}
    (profile : Profile shape)
    (message :
      Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.OutputMessage shape)
    (output : Fin Semantics.outputCount) : Semantics.OutputMessage where
  yRing := fun row lane =>
    message.yRing (sourceIndex profile output) (matrixIndex profile row) lane
  yZcol := fun lane =>
    message.yZcol (sourceIndex profile output) lane

/-- Complete terminal digest payload in canonical source order. -/
def projectOutputs
    {shape : SemanticShape}
    (profile : Profile shape)
    (message :
      Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.OutputMessage shape) :
    Fin Semantics.outputCount -> Semantics.OutputMessage :=
  fun output => projectOutput profile message output

@[simp] theorem projectOutput_yRing
    {shape : SemanticShape}
    (profile : Profile shape)
    (message :
      Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.OutputMessage shape)
    (output : Fin Semantics.outputCount)
    (row : Fin Semantics.yRingRows)
    (lane : Fin Semantics.activeWidth) :
    (projectOutput profile message output).yRing row lane =
      message.yRing (sourceIndex profile output) (matrixIndex profile row)
        lane := by
  rfl

@[simp] theorem projectOutput_yZcol
    {shape : SemanticShape}
    (profile : Profile shape)
    (message :
      Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.OutputMessage shape)
    (output : Fin Semantics.outputCount)
    (lane : Fin Semantics.activeWidth) :
    (projectOutput profile message output).yZcol lane =
      message.yZcol (sourceIndex profile output) lane := by
  rfl

/-- The fixed-profile projection preserves the complete active Split-NC
message. No source, matrix, or Phi81 lane is discarded before serialization. -/
theorem projectOutputs_injective
    {shape : SemanticShape}
    (profile : Profile shape) :
    Function.Injective (projectOutputs profile) := by
  intro left right equal
  apply OutputClaims.Claims.ext left right
  · intro source matrix lane
    have coordinate := congrArg
      (fun outputs =>
        (outputs (outputIndex profile source)).yRing
          (rowIndex profile matrix) lane)
      equal
    simpa [projectOutputs, projectOutput] using coordinate
  · intro source lane
    have coordinate := congrArg
      (fun outputs =>
        (outputs (outputIndex profile source)).yZcol lane)
      equal
    simpa [projectOutputs, projectOutput] using coordinate

end Nightstream.Implementation.R1CS.PiCcsOutputDigest.Projection.SplitNc
