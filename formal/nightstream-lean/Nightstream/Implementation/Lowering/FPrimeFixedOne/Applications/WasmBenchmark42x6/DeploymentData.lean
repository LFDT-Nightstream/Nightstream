import Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.ApplicationProfile
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ApplicationCodecRecovery
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalInputRecovery

/-!
Contract: deployment-owned values for the 42-times-6 WASM integration
fixture.

Assurance tier: model-level.

Owns: one canonical zero running value, its exact codec admissibility, and
exact-width recovery for every application-owned codec.

Does not own: an Ajtai setup, the selected NIFS footprint, a complete
deployment, a recursive fixed point, Rust, or generated artifacts.

Emits constraints: none.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6

open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalKey
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalInputRecovery
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalRunningCodec
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsPlain270Profile
open Nightstream.Implementation.Lowering.Typed
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

/-- Canonical zero commitment at the selected Ajtai row count. -/
def zeroCommitment (verifierRows : Nat) :
    CommitmentValue verifierRows :=
  fun _ _ => 0

/-- Canonical zero public carrier. -/
def zeroPublicInput
    (dimensions : Dimensions) :
    PublicInput dimensions.shape :=
  fun _ => 0

/-- Canonical zero row-evaluation point. -/
def zeroPoint
    (dimensions : Dimensions) :
    Point dimensions.shape where
  coordinates := List.replicate dimensions.rowVariables K.zero
  dimension := by
    simp [Dimensions.shape]

/-- One canonical evaluation vector with the selected matrix count. -/
def zeroEvaluations
    (dimensions : Dimensions) :
    Array Evaluation :=
  Array.replicate dimensions.matrixCount ringKZero

/-- Complete zero running value at the selected relation shape.

The evaluation arrays retain the exact matrix count. They are not empty
placeholders. -/
def defaultRunning
    (dimensions : Dimensions) (verifierRows : Nat) :
    BenchmarkRunning dimensions verifierRows where
  parent := {
    commitment := zeroCommitment verifierRows
    publicInput := zeroPublicInput dimensions
    point := zeroPoint dimensions
    evaluations := zeroEvaluations dimensions
  }
  children := fun _ => {
    commitment := zeroCommitment verifierRows
    publicInput := zeroPublicInput dimensions
    point := zeroPoint dimensions
    evaluations := zeroEvaluations dimensions
  }

@[simp] theorem zeroEvaluations_size
    (dimensions : Dimensions) :
    (zeroEvaluations dimensions).size = dimensions.matrixCount := by
  simp [zeroEvaluations]

/-- The complete zero running value has the exact canonical codec shape. -/
theorem defaultRunning_admissible
    (dimensions : Dimensions) (verifierRows : Nat) :
    (runningCodec dimensions verifierRows).Admissible
      (defaultRunning dimensions verifierRows) := by
  apply
    (ConcreteNifsCanonicalRunningCodec.runningCodec_admissible_iff
      (defaultRunning dimensions verifierRows)).2
  constructor
  · exact zeroEvaluations_size dimensions
  · intro child
    exact zeroEvaluations_size dimensions

/-- Every application-owned codec recovers every exact-width field string. -/
theorem applicationCodecRecovery
    {dimensions : Dimensions} {verifierRows : Nat}
    (setup : RelationSetup dimensions verifierRows)
    (nifsFootprint : CallFootprint) :
    ApplicationCodecRecovery
      (ConcreteNifsPlain270Profile.selected dimensions
        (fun _ => ConcreteNifsCanonicalKey.selected setup)
        (defaultRunning dimensions verifierRows)
        (machine benchmarkHashPlan)
        (terminalRelations dimensions verifierRows)
        (terminalChecks dimensions verifierRows)
        (widths setup)
        (footprints
          (dimensions := dimensions)
          (verifierRows := verifierRows)
          nifsFootprint))
      (applicationProfile setup
        (defaultRunning dimensions verifierRows)
        nifsFootprint).codecs where
  state := stateCodec_exactWidthRecoverable
  witness := witnessCodec_exactWidthRecoverable
  runningWitness :=
    runningCodec_exactWidthRecoverable
      (ConcreteNifsPlain270Profile.Shape dimensions)
      publicRingColumns verifierRows (publicFits dimensions)
  freshWitness :=
    freshCodec_exactWidthRecoverable
      (ConcreteNifsPlain270Profile.Shape dimensions)
      publicRingColumns verifierRows (publicFits dimensions)

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6
