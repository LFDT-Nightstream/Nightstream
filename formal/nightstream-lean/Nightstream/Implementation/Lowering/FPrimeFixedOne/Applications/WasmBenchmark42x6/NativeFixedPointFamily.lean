import Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.NativeFixedPointSource
import Nightstream.Implementation.Lowering.Goldilocks.NativeCcsPhi81

/-!
Contract: install the compiler-produced native four-matrix relation at the
exact benchmark recursive shape.

Assurance tier: model-level.

Owns: the exact compiler-to-relation shape equality and the setup carrying the
four matrices compiled from the native Step rows.

Does not own: program stability under matrix replacement, terminal R1CS
lowering, Spartan, WHIR, Rust, or a security reduction.

Emits constraints: none.
-/

set_option autoImplicit false
set_option maxRecDepth 500000
set_option maxHeartbeats 2400000

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.NativeFixedPointFamily

open Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.NativeFixedPointCost
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.NativeFixedPointSource
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalKey
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

private theorem compiledPublicFits (template : Template) :
    ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth
        (program template).columnIds.length := by
  rw [NativeFixedPointSource.columnsExact template]
  exact dimensions.alignedPublicFitsCarrier

noncomputable def compiledShape (template : Template) : Phi81Relation.Shape :=
  NativeCcsPhi81.shape
    (program template) (rowDomain template) publicRingColumns
      (compiledPublicFits template)

noncomputable def compiledStructure (template : Template) :
    Structure (compiledShape template) :=
  NativeCcsPhi81.relation
    (program template) (valid template) (rowDomain template)
      publicRingColumns (compiledPublicFits template)

private theorem shape_ext
    (left right : Phi81Relation.Shape)
    (rows : left.rowVariables = right.rowVariables)
    (columns : left.logicalWidth = right.logicalWidth)
    (matrices : left.matrixCount = right.matrixCount)
    (publicColumns :
      left.publicRingColumns = right.publicRingColumns) :
    left = right := by
  cases left with
  | mk leftRows leftColumns leftMatrices leftPublic leftFits =>
    cases right with
    | mk rightRows rightColumns rightMatrices rightPublic rightFits =>
      simp only at rows columns matrices publicColumns
      subst rightRows
      subst rightColumns
      subst rightMatrices
      subst rightPublic
      rfl

theorem compiledShape_eq (template : Template) :
    compiledShape template = dimensions.shape := by
  apply shape_ext
  · rfl
  · exact NativeFixedPointSource.columnsExact template
  · exact NativeFixedPointCost.matrixCount_fixed.symm
  · rfl

/-- Exact compiler-produced native relation at the selected benchmark shape.
The polynomial is stated directly; only the matrix family needs dependent
shape transport. -/
noncomputable def finalSystem (template : Template) :
    Structure dimensions.shape where
  matrices := by
    rw [← compiledShape_eq template]
    exact (compiledStructure template).matrices
  constraintPolynomial := NativeCcsSelector.constraintPolynomial

@[simp] theorem finalSystem_polynomial (template : Template) :
    (finalSystem template).constraintPolynomial =
      NativeCcsSelector.constraintPolynomial := by
  rfl

noncomputable def finalSetup (template : Template) :
    RelationSetup dimensions commitmentRows :=
  template.withSystem (finalSystem template)

@[simp] theorem finalSetup_polynomial (template : Template) :
    (finalSetup template).system.constraintPolynomial =
      NativeCcsSelector.constraintPolynomial := by
  rfl

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.NativeFixedPointFamily
