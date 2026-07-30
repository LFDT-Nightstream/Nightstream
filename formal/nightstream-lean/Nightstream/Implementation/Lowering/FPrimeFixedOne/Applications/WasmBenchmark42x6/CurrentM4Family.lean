import Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.CurrentM4Domain
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.CurrentM4StepRowsAggregate

/-!
Contract: construct the recursive same-system fixed point and current M4
evidence for the exact Lean-owned 42-times-6 WASM benchmark Step program.

Assurance tier: model-level.

Owns: the benchmark deployment family, physical compiler-input stability, the
compiled final relation, and current recursive M4 evidence for the exact
19,859,562-row Step encoding.

Does not own: the setup verifier key, selection of this reduced test fixture
as a production deployment, MSIS security validation for the 25/19/6 domain,
Rust equality, or an end-to-end security reduction.

Emits constraints: no new rows.
-/

set_option autoImplicit false
set_option maxRecDepth 500000
set_option maxHeartbeats 2400000

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.CurrentM4Family

open Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.CurrentM4Cost
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.CurrentM4Domain
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalKey
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalCertification
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.LeanCompiler
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.LeanCompiler.CurrentFixedPoint
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.CurrentM4StepRowsAggregate
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.CurrentM4PhysicalStability

local instance stateDecidableEq : DecidableEq State :=
  Fintype.decidablePiFintype

local instance encodedDecidableEq : DecidableEq Encoded :=
  Fintype.decidablePiFintype

/-- Setup-owned authority that cannot be derived from the emitted program.

Domain coverage and row nonemptiness are not fields. They are exact
consequences of the benchmark encoding and are supplied by
`toSetupTemplate`. -/
structure Template where
  verifierKey :
    VerifierKey
      (ConcreteNifsPlain270Profile.Shape dimensions)
      publicRingColumns
      (ConcreteNifsPlain270Profile.publicFits dimensions)
      commitmentRows

namespace Template

/-- Construct the generic setup template from the sole setup-owned input. -/
noncomputable def toSetupTemplate (template : Template) :
    SetupTemplate dimensions commitmentRows where
  verifierKey := template.verifierKey
  domainCovers := currentNc_covers
  rowNonempty := CurrentM4Domain.rowNonempty

/-- Install one relation system with benchmark-derived shape facts. -/
noncomputable def withSystem
    (template : Template)
    (system : Structure dimensions.shape) :
    RelationSetup dimensions commitmentRows :=
  template.toSetupTemplate.withSystem system

@[simp] theorem withSystem_system
    (template : Template)
    (system : Structure dimensions.shape) :
    HEq (template.withSystem system).system system := by
  rfl

end Template

def matrixCountExact :
    dimensions.matrixCount =
      Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.RelationProfile.matrixCount := by
  rfl

noncomputable def seedSetup (template : Template) :
    RelationSetup dimensions commitmentRows :=
  template.withSystem (seedSystem dimensions matrixCountExact)

noncomputable def familyWidths (template : Template) : Widths :=
  widths (seedSetup template)

noncomputable def familyFootprints (template : Template) : Footprints :=
  selectedFootprints (seedSetup template)

theorem targetPolynomial_exact
    (template : Template)
    (system : Structure dimensions.shape)
    (fixed : UsesFixedPolynomial dimensions matrixCountExact system) :
    (template.withSystem system).system.constraintPolynomial =
      Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Semantics.polynomial := by
  exact fixed

noncomputable def deploymentFor
    (template : Template)
    (system : Structure dimensions.shape)
    (fixed : UsesFixedPolynomial dimensions matrixCountExact system) :
    Deployment
      (template.withSystem system)
      (defaultRunning dimensions commitmentRows)
      (machine benchmarkHashPlan)
      (terminalRelations dimensions commitmentRows)
      (terminalChecks dimensions commitmentRows)
      (familyWidths template)
      (familyFootprints template) := by
  cases system with
  | mk matrices polynomial =>
      change polynomial = fixedPolynomial dimensions matrixCountExact at fixed
      subst polynomial
      exact
        Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.deployment
          (template.withSystem
            { matrices := matrices
              constraintPolynomial :=
                fixedPolynomial dimensions matrixCountExact })

noncomputable def family (template : Template) :
    Family
      (defaultRunning dimensions commitmentRows)
      (machine benchmarkHashPlan)
      (terminalRelations dimensions commitmentRows)
      (terminalChecks dimensions commitmentRows)
      (familyWidths template)
      (familyFootprints template)
      template.toSetupTemplate where
  matrixCountExact := matrixCountExact
  deployment := deploymentFor template
  shapeFixed := by
    intro system polynomialFixed
    cases system with
    | mk matrices polynomial =>
        change
          polynomial =
            fixedPolynomial dimensions matrixCountExact at polynomialFixed
        subst polynomial
        exact
          shapeFixedPoint
            (template.withSystem
              { matrices := matrices
                constraintPolynomial :=
                  fixedPolynomial dimensions matrixCountExact })
            rfl
            (stepPublicWidth
              (defaultRunning dimensions commitmentRows)
              (machine benchmarkHashPlan)
              (terminalRelations dimensions commitmentRows)
              (terminalChecks dimensions commitmentRows)
              (familyWidths template)
              (familyFootprints template)
              template.toSetupTemplate
              { matrices := matrices
                constraintPolynomial :=
                  fixedPolynomial dimensions matrixCountExact }
              (deploymentFor template
                { matrices := matrices
                  constraintPolynomial :=
                    fixedPolynomial dimensions matrixCountExact }
                rfl))
  physicalStable := by
    intro system polynomialFixed
    cases system with
    | mk matrices polynomial =>
        change
          polynomial =
            fixedPolynomial dimensions matrixCountExact at polynomialFixed
        subst polynomial
        apply CurrentCompiler.PhysicalEncoding.ext
        · change
            (CurrentM4PhysicalStability.encoding
                (template.withSystem
                  (seedSystem dimensions matrixCountExact))).columnIds =
              (CurrentM4PhysicalStability.encoding
                (template.withSystem
                  { matrices := matrices
                    constraintPolynomial :=
                      fixedPolynomial dimensions matrixCountExact })).columnIds
          exact
            columnIds_eq_of_constraintPolynomial_eq template.toSetupTemplate
              (seedSystem dimensions matrixCountExact)
              { matrices := matrices
                constraintPolynomial :=
                  fixedPolynomial dimensions matrixCountExact }
              rfl
        · change
            (CurrentM4PhysicalStability.encoding
                (template.withSystem
                  (seedSystem dimensions matrixCountExact))).rows =
              (CurrentM4PhysicalStability.encoding
                (template.withSystem
                  { matrices := matrices
                    constraintPolynomial :=
                      fixedPolynomial dimensions matrixCountExact })).rows
          exact
            rows_eq_of_constraintPolynomial_eq template.toSetupTemplate
              (seedSystem dimensions matrixCountExact)
              { matrices := matrices
                constraintPolynomial :=
                  fixedPolynomial dimensions matrixCountExact }
              rfl
        · simp only [CurrentFixedPoint.stepEncoding,
            CurrentCompiler.PhysicalEncoding.ofEncoding,
            Nightstream.Implementation.Lowering.Goldilocks.SourceAlignment.AlignedReceiptProgram.toEncoding,
            Nightstream.Implementation.Lowering.Goldilocks.ReceiptProgram.toEncoding_one]

/-- The exact recursive setup compiled from the benchmark Step rows. -/
noncomputable def finalSetup (template : Template) :
    RelationSetup dimensions commitmentRows :=
  (family template).finalSetup
    (defaultRunning dimensions commitmentRows)
    (machine benchmarkHashPlan)
    (terminalRelations dimensions commitmentRows)
    (terminalChecks dimensions commitmentRows)
    (familyWidths template)
    (familyFootprints template)
    template.toSetupTemplate

/-- The benchmark deployment rebuilt against its exact compiled relation. -/
noncomputable def finalDeployment (template : Template) :=
  (family template).finalDeployment
    (defaultRunning dimensions commitmentRows)
    (machine benchmarkHashPlan)
    (terminalRelations dimensions commitmentRows)
    (terminalChecks dimensions commitmentRows)
    (familyWidths template)
    (familyFootprints template)
    template.toSetupTemplate

/-- Current recursive M4 evidence for the exact Lean-owned benchmark Step
encoding. The setup verifier key remains the explicit authority parameter. -/
noncomputable def m4Evidence (template : Template) :
    CurrentM4.Evidence
      (finalSetup template)
      (defaultRunning dimensions commitmentRows)
      (machine benchmarkHashPlan)
      (terminalRelations dimensions commitmentRows)
      (terminalChecks dimensions commitmentRows)
      (familyWidths template)
      (familyFootprints template)
      (finalDeployment template) :=
  (family template).m4
    (defaultRunning dimensions commitmentRows)
    (machine benchmarkHashPlan)
    (terminalRelations dimensions commitmentRows)
    (terminalChecks dimensions commitmentRows)
    (familyWidths template)
    (familyFootprints template)
    template.toSetupTemplate

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.CurrentM4Family
