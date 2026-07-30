import Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.Deployment
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.LeanCompiler.CurrentFixedPoint

/-!
Contract: expose the exact physical `nifsVerify` frame used by the
42-times-6 WASM benchmark and prove that its numeric namespace is independent
of the recursive relation matrix payload.

Assurance tier: model-level.

Owns: the benchmark deployment projections needed by current M4, stability of
the global NIFS column namespace, stability of the running operand bundle,
and the proof-independent numeric location of one canonical running point.

Does not own: semantic equality between different setup keys, full NIFS row
stability, the recursive fixed point, Rust, or generated artifacts.

Emits constraints: no new rows.
-/

set_option autoImplicit false
set_option maxRecDepth 500000
set_option maxHeartbeats 2400000

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.CurrentM4PhysicalFrame

open Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalKey
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.LeanCompiler.CurrentFixedPoint
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

/-- Benchmark deployment at one setup-selected relation system. -/
noncomputable def deploymentFor
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (setup : RelationSetup dimensions verifierRows) :=
  deployment setup

/-- Complete benchmark certificate at one setup-selected relation system. -/
noncomputable def certificate
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (setup : RelationSetup dimensions verifierRows) :=
  ConcreteNifsCanonicalCertification.complete
    setup
    (defaultRunning dimensions verifierRows)
    (machine benchmarkHashPlan)
    (terminalRelations dimensions verifierRows)
    (terminalChecks dimensions verifierRows)
    (widths setup)
    (selectedFootprints setup)
    (deploymentFor setup)

/-- Exact recursive NIFS invocation plan inside the benchmark Step program. -/
noncomputable def invokePlan
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (setup : RelationSetup dimensions verifierRows) :=
  let selected := certificate setup
  CanonicalStepPlan.recursiveNifsInvokePlan
    (ConcreteNifsPlain270Profile.selected dimensions
      (ConcreteNifsCanonicalOperationalProfile.selectedKeys setup)
      (defaultRunning dimensions verifierRows)
      (machine benchmarkHashPlan)
      (terminalRelations dimensions verifierRows)
      (terminalChecks dimensions verifierRows)
      (widths setup)
      (selectedFootprints setup))
    selected.baseProfile selected.allRecipes

/-- Selected operational NIFS profile inside the benchmark deployment. -/
noncomputable def operational
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (setup : RelationSetup dimensions verifierRows) :=
  (ConcreteNifsCanonicalCertification.nifs
    setup
    (defaultRunning dimensions verifierRows)
    (machine benchmarkHashPlan)
    (terminalRelations dimensions verifierRows)
    (terminalChecks dimensions verifierRows)
    (widths setup)
    (selectedFootprints setup)
    (deploymentFor setup)).operational

/-- Selected application profile inside the benchmark deployment. -/
noncomputable def application
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (setup : RelationSetup dimensions verifierRows) :=
  (deploymentFor setup).application.phase4.profile

/-- Complete numeric namespace addressed by the selected NIFS rows. -/
noncomputable def orderedIds
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (setup : RelationSetup dimensions verifierRows) : List ColumnId :=
  PaperNifsGlobalColumnMap.orderedIds (invokePlan setup).frame

/-- Physical IDs of the authoritative running operand. -/
noncomputable def runningOperandIds
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (setup : RelationSetup dimensions verifierRows) : List ColumnId :=
  PaperNifsCallFrame.runningOperand (invokePlan setup).frame.operands |>.ids

/-- Physical IDs of the authoritative fresh operand. -/
noncomputable def freshOperandIds
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (setup : RelationSetup dimensions verifierRows) : List ColumnId :=
  PaperNifsCallFrame.freshOperand (invokePlan setup).frame.operands |>.ids

/-- Physical IDs of the authoritative proof operand. -/
noncomputable def proofOperandIds
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (setup : RelationSetup dimensions verifierRows) : List ColumnId :=
  PaperNifsCallFrame.proofOperand (invokePlan setup).frame.operands |>.ids

/-- Physical IDs of the authoritative running output bundle. -/
noncomputable def outputIds
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (setup : RelationSetup dimensions verifierRows) : List ColumnId :=
  (invokePlan setup).frame.outputs.ids

/-- Physical running-parent point columns selected by the canonical codec. -/
noncomputable def parentPointColumns
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (setup : RelationSetup dimensions verifierRows)
    (coordinate : Fin dimensions.rowVariables) :=
  let frame := (invokePlan setup).frame
  let view := (operational setup).runningViews.parentPoint coordinate
  view.columns
    (PaperNifsCallFrame.runningOperand frame.operands)
    (PaperNifsCallFrame.running_widthsAgree frame)

/-- Numeric running-parent point columns used by the row gadgets. -/
noncomputable def parentPointNumeric
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (setup : RelationSetup dimensions verifierRows)
    (coordinate : Fin dimensions.rowVariables) :=
  let frame := (invokePlan setup).frame
  ConcreteNifsCarrierFrame.runningKLocation
    (application setup).family frame
    ((operational setup).runningViews.parentPoint coordinate) |>.numeric

private theorem kColumns_eq
    (left right :
      Nightstream.Implementation.R1CS.ProjectionProgram.KColumns)
    (c0Equal : left.c0 = right.c0)
    (c1Equal : left.c1 = right.c1) :
    left = right := by
  cases left
  cases right
  simp only at c0Equal c1Equal
  cases c0Equal
  cases c1Equal
  rfl

/-- Equal constraint polynomials give the same complete NIFS column
namespace. Relation matrix coefficients do not select physical columns. -/
theorem orderedIds_eq_of_constraintPolynomial_eq
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (template : SetupTemplate dimensions verifierRows)
    (left right : Structure dimensions.shape)
    (same :
      left.constraintPolynomial = right.constraintPolynomial) :
    orderedIds (template.withSystem left) =
      orderedIds (template.withSystem right) := by
  cases left with
  | mk leftMatrices leftPolynomial =>
      cases right with
      | mk rightMatrices rightPolynomial =>
          simp only at same
          subst rightPolynomial
          unfold orderedIds
          rfl

/-- Equal complete namespaces have the same constant-one physical column. -/
theorem one_eq_of_constraintPolynomial_eq
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (template : SetupTemplate dimensions verifierRows)
    (left right : Structure dimensions.shape)
    (same :
      left.constraintPolynomial = right.constraintPolynomial) :
    (invokePlan (template.withSystem left)).frame.one =
      (invokePlan (template.withSystem right)).frame.one := by
  have orderedEqual :=
    orderedIds_eq_of_constraintPolynomial_eq template left right same
  exact Option.some.inj
    (by
      simpa [orderedIds, PaperNifsGlobalColumnMap.orderedIds] using
        congrArg List.head? orderedEqual)

/-- Equal constraint polynomials give the same physical activation column. -/
theorem active_eq_of_constraintPolynomial_eq
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (template : SetupTemplate dimensions verifierRows)
    (left right : Structure dimensions.shape)
    (same :
      left.constraintPolynomial = right.constraintPolynomial) :
    (invokePlan (template.withSystem left)).frame.active =
      (invokePlan (template.withSystem right)).frame.active := by
  cases left with
  | mk leftMatrices polynomial =>
      cases right with
      | mk rightMatrices rightPolynomial =>
          simp only at same
          subst rightPolynomial
          rfl

/-- Equal constraint polynomials give the same physical call owner. -/
theorem owner_eq_of_constraintPolynomial_eq
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (template : SetupTemplate dimensions verifierRows)
    (left right : Structure dimensions.shape)
    (same :
      left.constraintPolynomial = right.constraintPolynomial) :
    (invokePlan (template.withSystem left)).frame.owner =
      (invokePlan (template.withSystem right)).frame.owner := by
  cases left with
  | mk leftMatrices polynomial =>
      cases right with
      | mk rightMatrices rightPolynomial =>
          simp only at same
          subst rightPolynomial
          rfl

/-- Equal constraint polynomials give the same temporary-column receipt. -/
theorem temporaryIds_eq_of_constraintPolynomial_eq
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (template : SetupTemplate dimensions verifierRows)
    (left right : Structure dimensions.shape)
    (same :
      left.constraintPolynomial = right.constraintPolynomial) :
    (invokePlan (template.withSystem left)).frame.temporaries.ids =
      (invokePlan (template.withSystem right)).frame.temporaries.ids := by
  cases left with
  | mk leftMatrices polynomial =>
      cases right with
      | mk rightMatrices rightPolynomial =>
          simp only at same
          subst rightPolynomial
          rfl

/-- Equal complete namespaces give the same numeric-to-physical column map.
This includes the out-of-range fallback to the constant-one column. -/
theorem columnMap_eq_of_constraintPolynomial_eq
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (template : SetupTemplate dimensions verifierRows)
    (left right : Structure dimensions.shape)
    (same :
      left.constraintPolynomial = right.constraintPolynomial) :
    PaperNifsGlobalColumnMap.columnMap
        (invokePlan (template.withSystem left)).frame =
      PaperNifsGlobalColumnMap.columnMap
        (invokePlan (template.withSystem right)).frame := by
  funext source
  have idsEqual :
      PaperNifsGlobalColumnMap.orderedIds
          (invokePlan (template.withSystem left)).frame =
        PaperNifsGlobalColumnMap.orderedIds
          (invokePlan (template.withSystem right)).frame := by
    simpa only [orderedIds] using
      orderedIds_eq_of_constraintPolynomial_eq template left right same
  unfold PaperNifsGlobalColumnMap.columnMap
  rw [idsEqual,
    one_eq_of_constraintPolynomial_eq template left right same]

/-- Equal constraint polynomials give the same authoritative running operand
bundle. -/
theorem runningOperandIds_eq_of_constraintPolynomial_eq
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (template : SetupTemplate dimensions verifierRows)
    (left right : Structure dimensions.shape)
    (same :
      left.constraintPolynomial = right.constraintPolynomial) :
    runningOperandIds (template.withSystem left) =
      runningOperandIds (template.withSystem right) := by
  cases left with
  | mk leftMatrices leftPolynomial =>
      cases right with
      | mk rightMatrices rightPolynomial =>
          simp only at same
          subst rightPolynomial
          unfold runningOperandIds
          rfl

/-- Equal constraint polynomials select the same physical parent-point
columns from the canonical running codec. -/
theorem parentPointColumns_eq_of_constraintPolynomial_eq
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (template : SetupTemplate dimensions verifierRows)
    (left right : Structure dimensions.shape)
    (same :
      left.constraintPolynomial = right.constraintPolynomial) :
    ∀ coordinate,
      parentPointColumns (template.withSystem left) coordinate =
        parentPointColumns (template.withSystem right) coordinate := by
  cases left with
  | mk leftMatrices leftPolynomial =>
      cases right with
      | mk rightMatrices rightPolynomial =>
          simp only at same
          subst rightPolynomial
          intro coordinate
          unfold parentPointColumns
          rfl

/-- Equal constraint polynomials give the same numeric location for every
canonical running-parent point. Dependent membership proofs are erased by
`locate_source_congr`. -/
theorem parentPointNumeric_eq_of_constraintPolynomial_eq
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (template : SetupTemplate dimensions verifierRows)
    (left right : Structure dimensions.shape)
    (same :
      left.constraintPolynomial = right.constraintPolynomial) :
    ∀ coordinate,
      parentPointNumeric (template.withSystem left) coordinate =
        parentPointNumeric (template.withSystem right) coordinate := by
  cases left with
  | mk leftMatrices leftPolynomial =>
      cases right with
      | mk rightMatrices rightPolynomial =>
          simp only at same
          subst rightPolynomial
          intro coordinate
          unfold parentPointNumeric
          unfold ConcreteNifsCarrierFrame.runningKLocation
          unfold PaperNifsGlobalColumnMap.kLocation
          apply kColumns_eq
          · apply PaperNifsGlobalColumnMap.locate_source_congr
            · exact
                orderedIds_eq_of_constraintPolynomial_eq template
                  { matrices := leftMatrices
                    constraintPolynomial := leftPolynomial }
                  { matrices := rightMatrices
                    constraintPolynomial := leftPolynomial }
                  rfl
            · exact congrArg PaperNifsCodecProjection.KColumnIds.c0
                (parentPointColumns_eq_of_constraintPolynomial_eq template
                  { matrices := leftMatrices
                    constraintPolynomial := leftPolynomial }
                  { matrices := rightMatrices
                    constraintPolynomial := leftPolynomial }
                  rfl coordinate)
          · apply PaperNifsGlobalColumnMap.locate_source_congr
            · exact
                orderedIds_eq_of_constraintPolynomial_eq template
                  { matrices := leftMatrices
                    constraintPolynomial := leftPolynomial }
                  { matrices := rightMatrices
                    constraintPolynomial := leftPolynomial }
                  rfl
            · exact congrArg PaperNifsCodecProjection.KColumnIds.c1
                (parentPointColumns_eq_of_constraintPolynomial_eq template
                  { matrices := leftMatrices
                    constraintPolynomial := leftPolynomial }
                  { matrices := rightMatrices
                    constraintPolynomial := leftPolynomial }
                  rfl coordinate)

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.CurrentM4PhysicalFrame
