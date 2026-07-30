import Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.DeploymentData
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalCertification
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsStaticFootprint

/-!
Contract: complete selected Lean deployment for the 42-times-6 WASM
integration fixture at any setup-owned relation instance.

Assurance tier: model-level.

Owns: the frame-independent selected-NIFS footprint, exact footprint
alignment, the complete deployment value, and the closed protocol manifest
for the benchmark application.

Does not own: a compiler from arbitrary WASM, a concrete Ajtai setup value,
recursive fixed-point dimensions, Rust, or generated artifacts.

Emits constraints: none. It assembles existing proof-carrying recipes.
-/

set_option autoImplicit false
set_option maxRecDepth 100000
set_option maxHeartbeats 2200000

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6

open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalKey
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsPlain270Profile
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Protocol.FPrime.ConcretePhi81
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

/-- Selected NIFS footprint derived only from the Lean-owned relation shape,
constraint polynomial, and canonical verifier serialization. -/
noncomputable def nifsFootprint
    {dimensions : Dimensions} {verifierRows : Nat}
    (setup : RelationSetup dimensions verifierRows) : CallFootprint :=
  ConcreteNifsStaticFootprint.footprint
    (Shape dimensions)
    setup.system.constraintPolynomial
    publicRingColumns verifierRows (publicFits dimensions)

/-- Complete application footprint table with the selected NIFS slot filled
by its Lean-derived physical cost. -/
noncomputable def selectedFootprints
    {dimensions : Dimensions} {verifierRows : Nat}
    (setup : RelationSetup dimensions verifierRows) : Footprints :=
  footprints
    (dimensions := dimensions)
    (verifierRows := verifierRows)
    (nifsFootprint setup)

/-- Equal selected constraint polynomials give the same benchmark codec
widths. Relation matrix coefficients and Ajtai key values do not size a
codec. -/
theorem widths_eq_of_constraintPolynomial_eq
    {dimensions : Dimensions} {verifierRows : Nat}
    (left right : RelationSetup dimensions verifierRows)
    (same :
      left.system.constraintPolynomial =
        right.system.constraintPolynomial) :
    widths left = widths right := by
  unfold widths proofCodec
  rw [same]

/-- Equal selected constraint polynomials give the same static NIFS
footprint. No matrix coefficient is used as a resource count. -/
theorem nifsFootprint_eq_of_constraintPolynomial_eq
    {dimensions : Dimensions} {verifierRows : Nat}
    (left right : RelationSetup dimensions verifierRows)
    (same :
      left.system.constraintPolynomial =
        right.system.constraintPolynomial) :
    nifsFootprint left = nifsFootprint right := by
  unfold nifsFootprint
  rw [same]

/-- The complete benchmark footprint table is stable when only the relation
matrix payload changes. -/
theorem selectedFootprints_eq_of_constraintPolynomial_eq
    {dimensions : Dimensions} {verifierRows : Nat}
    (left right : RelationSetup dimensions verifierRows)
    (same :
      left.system.constraintPolynomial =
        right.system.constraintPolynomial) :
    selectedFootprints left = selectedFootprints right := by
  unfold selectedFootprints
  rw [nifsFootprint_eq_of_constraintPolynomial_eq left right same]

/-- Complete proof-carrying deployment for the benchmark application.

The relation setup remains a parameter because SuperNeo setup owns its
matrices and Ajtai verifier key. No row count or matrix count is copied from
Rust. -/
noncomputable def deployment
    {dimensions : Dimensions} {verifierRows : Nat}
    (setup : RelationSetup dimensions verifierRows) :
    ConcreteNifsCanonicalCertification.Deployment
      setup
      (defaultRunning dimensions verifierRows)
      (machine benchmarkHashPlan)
      (terminalRelations dimensions verifierRows)
      (terminalChecks dimensions verifierRows)
      (widths setup)
      (selectedFootprints setup) where
  application :=
    canonicalApplication setup
      (defaultRunning dimensions verifierRows)
      (nifsFootprint setup)
  applicationCodecRecovery :=
    applicationCodecRecovery setup (nifsFootprint setup)
  footprintExact := {
    exact := by
      intro context runningRef freshRef proofRef frame
      change
        nifsFootprint setup =
          ConcreteNifsActivatedProgram.footprint
            (applicationProfile setup
              (defaultRunning dimensions verifierRows)
              (nifsFootprint setup))
            (ConcreteNifsCanonicalOperationalProfile.operational
              setup (defaultRunning dimensions verifierRows)
              (machine benchmarkHashPlan)
              (terminalRelations dimensions verifierRows)
              (terminalChecks dimensions verifierRows)
              (widths setup) (selectedFootprints setup)
              (canonicalApplication setup
                (defaultRunning dimensions verifierRows)
                (nifsFootprint setup)))
            frame
      exact
        ConcreteNifsStaticFootprint.footprint_eq
          setup (defaultRunning dimensions verifierRows)
          (machine benchmarkHashPlan)
          (terminalRelations dimensions verifierRows)
          (terminalChecks dimensions verifierRows)
          (widths setup) (selectedFootprints setup)
          (canonicalApplication setup
            (defaultRunning dimensions verifierRows)
            (nifsFootprint setup))
          frame
  }
  step :=
    selectedStepRecipe setup
      (defaultRunning dimensions verifierRows)
      (nifsFootprint setup)
  defaultRunningAdmissible :=
    defaultRunning_admissible dimensions verifierRows

/-- Closed proof-free manifest for the selected benchmark application and
setup-owned relation instance. -/
noncomputable def manifest
    {dimensions : Dimensions} {verifierRows : Nat}
    (setup : RelationSetup dimensions verifierRows) :=
  ConcreteNifsCanonicalCertification.manifest
    setup
    (defaultRunning dimensions verifierRows)
    (machine benchmarkHashPlan)
    (terminalRelations dimensions verifierRows)
    (terminalChecks dimensions verifierRows)
    (widths setup)
    (selectedFootprints setup)
    (deployment setup)

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6
