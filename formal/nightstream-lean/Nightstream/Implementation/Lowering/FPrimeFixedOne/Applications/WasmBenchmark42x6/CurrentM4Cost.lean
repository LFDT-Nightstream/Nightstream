import Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.CostArithmetic
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.Deployment
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ApplicationStepCostSplit
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.LeanCompiler.CurrentFixedPoint
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.LeanCompiler.EncodingCostCount
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.Polynomial.Semantics

/-!
Contract: exact compact physical cost and recursive dimensions for the
42-times-6 WASM benchmark Step program.

Assurance tier: model-level.

Owns: the current recursive dimensions, the exact selected-NIFS static cost,
the exact complete Step cost, and the compact component proofs used to avoid
reducing the large emitted receipt payload.

Does not own: a concrete Ajtai verifier key, the recursive fixed-point
deployment, Rust equality, or a general compiler from arbitrary WASM.

Emits constraints: none. It computes the cost of existing proof-carrying
receipts.
-/

set_option autoImplicit false
set_option maxRecDepth 500000
set_option maxHeartbeats 2400000

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.CurrentM4Cost

open Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ApplicationStepCostSplit
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsPlain270Profile
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsStaticFootprint
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary
open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.LeanCompiler.CurrentFixedPoint
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.LeanCompiler
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc

/-- Exact recursive dimensions obtained from the complete benchmark Step
rows and owned columns. The thirteen legacy-to-aligned padding columns are
inserted by `Dimensions.alignedLogicalWidth`. -/
def dimensions : Dimensions where
  rowVariables := 25
  legacyLogicalWidth := 19_969_300
  matrixCount := 13
  legacyPublicFits := by decide

/-- SuperNeo commitment module rank `κ`. This is distinct from the expansion
factor `T = 216`. -/
abbrev commitmentRows : Nat :=
  productionProfile.commitmentWidth

theorem commitmentRows_exact : commitmentRows = 18 := by
  rfl

abbrev shape : SemanticShape :=
  ConcreteNifsPlain270Profile.Shape dimensions

def publicFits :
    ringDegree * publicRingColumns ≤ shape.carrierWidth :=
  ConcreteNifsPlain270Profile.publicFits dimensions

private theorem cost_extensionality
    {left right : Cost}
    (recurringRows : left.recurringRows = right.recurringRows)
    (committedColumns :
      left.committedColumns = right.committedColumns)
    (publicColumns : left.publicColumns = right.publicColumns)
    (auxiliaryColumns :
      left.auxiliaryColumns = right.auxiliaryColumns) :
    left = right := by
  cases left
  cases right
  simp_all

/-- Exact physical cost of the selected fixed-point NIFS verifier before its
committed running output is added by the typed call boundary. -/
theorem staticNifsCost_exact :
    ConcreteNifsStaticFootprint.cost shape
      Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Semantics.polynomial
      publicRingColumns commitmentRows publicFits =
        ⟨19_773_612, 0, 0, 19_720_201⟩ := by
  apply cost_extensionality <;> rfl

/-- The selected benchmark parameters at one setup-owned relation value. -/
noncomputable def parameters
    (setup :
      ConcreteNifsCanonicalKey.RelationSetup dimensions commitmentRows) :=
  ConcreteNifsPlain270Profile.selected dimensions
    (ConcreteNifsCanonicalOperationalProfile.selectedKeys setup)
    (defaultRunning dimensions commitmentRows)
    (machine benchmarkHashPlan)
    (terminalRelations dimensions commitmentRows)
    (terminalChecks dimensions commitmentRows)
    (widths setup)
    (selectedFootprints setup)

/-- The complete proof-carrying benchmark certificate at one setup-owned
relation value. -/
noncomputable def certificate
    (setup :
      ConcreteNifsCanonicalKey.RelationSetup dimensions commitmentRows) :=
  ConcreteNifsCanonicalCertification.complete
    setup
    (defaultRunning dimensions commitmentRows)
    (machine benchmarkHashPlan)
    (terminalRelations dimensions commitmentRows)
    (terminalChecks dimensions commitmentRows)
    (widths setup)
    (selectedFootprints setup)
    (deployment setup)

private theorem runningWidth_exact
    (setup :
      ConcreteNifsCanonicalKey.RelationSetup dimensions commitmentRows) :
    (parameters setup).widths.running = 40_440 := by
  rfl

private theorem freshWidth_exact
    (setup :
      ConcreteNifsCanonicalKey.RelationSetup dimensions commitmentRows) :
    (parameters setup).widths.fresh = 1_242 := by
  rfl

private theorem proofWidth_exact
    (setup :
      ConcreteNifsCanonicalKey.RelationSetup dimensions commitmentRows)
    (polynomialExact :
      setup.system.constraintPolynomial =
        Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Semantics.polynomial) :
    (parameters setup).widths.nifsProof = 81_034 := by
  unfold parameters widths proofCodec
  rw [polynomialExact]
  rfl

private theorem nifsCallCost_exact
    (setup :
      ConcreteNifsCanonicalKey.RelationSetup dimensions commitmentRows)
    (polynomialExact :
      setup.system.constraintPolynomial =
        Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Semantics.polynomial) :
    (signature (parameters setup)).callCost Call.nifsVerify =
      ⟨19_773_612, 40_440, 0, 19_720_201⟩ := by
  rw [nifsVerify_callCost_exact]
  have footprintExact :
      (parameters setup).footprints.nifsVerify =
        nifsFootprint setup := by
    rfl
  rw [footprintExact]
  unfold nifsFootprint ConcreteNifsStaticFootprint.footprint
    CallFootprint.cost
  simp only [List.map_cons, List.map_nil, Cost.sum,
    auxiliaryLayout_cost_exact, committedLayout_cost_exact]
  rw [polynomialExact, staticNifsCost_exact, runningWidth_exact]
  rfl

private theorem stepInputCost_exact
    (setup :
      ConcreteNifsCanonicalKey.RelationSetup dimensions commitmentRows)
    (polynomialExact :
      setup.system.constraintPolynomial =
        Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Semantics.polynomial) :
    (stepInputSchema (parameters setup)).cost =
      ⟨0, 122_731, 0, 0⟩ := by
  rw [stepInputSchema_cost_exact]
  rw [runningWidth_exact, freshWidth_exact,
    proofWidth_exact setup polynomialExact]
  rfl

private theorem compactCost_exact
    (selected : Parameters)
    (inputCost :
      (stepInputSchema selected).cost = ⟨0, 122_731, 0, 0⟩)
    (runningWidth : selected.widths.running = 40_440)
    (step :
      (signature selected).callCost Call.step = ⟨11, 7, 0, 4⟩)
    (iterationZero :
      (signature selected).callCost Call.iterationZero =
        ⟨3, 0, 0, 3⟩)
    (stateEqual :
      (signature selected).callCost Call.stateEqual =
        ⟨21, 0, 0, 21⟩)
    (hashPrior :
      (signature selected).callCost Call.hashPrior =
        ⟨2503, 0, 0, 2499⟩)
    (freshPublic :
      (signature selected).callCost Call.freshPublic =
        ⟨5, 0, 0, 5⟩)
    (encodeInstance :
      (signature selected).callCost Call.encodeInstance =
        ⟨5, 0, 0, 5⟩)
    (encodedEqual :
      (signature selected).callCost Call.encodedEqual =
        ⟨15, 0, 0, 15⟩)
    (nifsVerify :
      (signature selected).callCost Call.nifsVerify =
        ⟨19_773_612, 40_440, 0, 19_720_201⟩)
    (hashNext :
      (signature selected).callCost Call.hashNext =
        ⟨2503, 0, 5, 2494⟩) :
    compactStepCost selected =
      ⟨19_859_562, 244_058, 6, 19_725_249⟩ := by
  have runningJoin :
      (Ports.committedRunning selected).layout.cost +
          ⟨selected.widths.running, 0, 0, 0⟩ =
        ⟨40_440, 40_440, 0, 0⟩ := by
    simp only [Ports.committedRunning, dataPort,
      committedLayout_cost_exact]
    rw [runningWidth]
    rfl
  have bodyExact :
      stepBodyCosts selected = CostArithmetic.bodyCosts := by
    unfold stepBodyCosts CostArithmetic.bodyCosts
    rw [step, iterationZero, stateEqual, runningJoin, hashPrior,
      freshPublic, encodeInstance, encodedEqual, nifsVerify, hashNext]
    rfl
  unfold compactStepCost
  rw [inputCost, bodyExact]
  exact CostArithmetic.total

/-- Exact physical cost of the complete benchmark Step program.

The proof reduces the compact list of typed call costs. It does not evaluate
the full receipt payload. -/
theorem stepCost_exact
    (setup :
      ConcreteNifsCanonicalKey.RelationSetup dimensions commitmentRows)
    (polynomialExact :
      setup.system.constraintPolynomial =
        Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Semantics.polynomial) :
    (certificate setup).stepCost =
      ⟨19_859_562, 244_058, 6, 19_725_249⟩ := by
  rw [CompleteApplicationCertification.stepCost_eq_compact]
  apply compactCost_exact
  · exact stepInputCost_exact setup polynomialExact
  · exact runningWidth_exact setup
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact nifsCallCost_exact setup polynomialExact
  · rfl

/-- Exact complete Step encoding used by the benchmark fixed-point
construction. -/
noncomputable def encoding
    (setup :
      ConcreteNifsCanonicalKey.RelationSetup dimensions commitmentRows) :=
  (certificate setup).canonicalStep.program.toEncoding

/-- The exact receipt-derived physical cost belongs to the emitted encoding,
not to a separate estimate. -/
theorem encodingCost_exact
    (setup :
      ConcreteNifsCanonicalKey.RelationSetup dimensions commitmentRows)
    (polynomialExact :
      setup.system.constraintPolynomial =
        Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Semantics.polynomial) :
    (encoding setup).cost =
      ⟨19_859_562, 244_058, 6, 19_725_249⟩ := by
  simpa [encoding, CompleteApplicationCertification.stepCost] using
    stepCost_exact setup polynomialExact

/-- Exact current selective source-row count of the emitted benchmark Step
encoding. -/
theorem encodingRows_exact
    (setup :
      ConcreteNifsCanonicalKey.RelationSetup dimensions commitmentRows)
    (polynomialExact :
      setup.system.constraintPolynomial =
        Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Semantics.polynomial) :
    (EncodingRows.program (encoding setup)).length = 19_859_562 := by
  rw [EncodingCostCount.rows_length_eq_cost_rows,
    encodingCost_exact setup polynomialExact]

/-- Exact count of the emitted owned columns. -/
theorem encodingColumns_exact
    (setup :
      ConcreteNifsCanonicalKey.RelationSetup dimensions commitmentRows)
    (polynomialExact :
      setup.system.constraintPolynomial =
        Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Semantics.polynomial) :
    (encoding setup).columnIds.length = 19_969_313 := by
  rw [EncodingCostCount.columnIds_length_eq_cost_columns,
    encodingCost_exact setup polynomialExact]
  rfl

/-- The benchmark dimensions are an exact fixed point of the row and column
shape inferred from the emitted complete Step encoding. -/
theorem shapeFixedPoint
    (setup :
      ConcreteNifsCanonicalKey.RelationSetup dimensions commitmentRows)
    (polynomialExact :
      setup.system.constraintPolynomial =
        Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Semantics.polynomial)
    (publicWidth : 270 ≤ (encoding setup).columnIds.length) :
    CurrentFixedPoint.ShapeFixedPoint dimensions
      (encoding setup) publicWidth := by
  constructor
  · change
      25 =
        Profile.rowVariables
          (EncodingRows.program (encoding setup)).length
    rw [encodingRows_exact setup polynomialExact]
    rfl
  · change
      19_969_300 + 13 = (encoding setup).columnIds.length
    rw [encodingColumns_exact setup polynomialExact]
  · rfl

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.CurrentM4Cost
