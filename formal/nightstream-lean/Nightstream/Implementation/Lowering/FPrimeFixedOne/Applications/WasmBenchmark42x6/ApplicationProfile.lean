import Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.ApplicationData
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.StepRecipe
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalOperationalProfile
import Nightstream.Implementation.R1CS.Canonical.GoldilocksField

/-!
Contract: selected Phase-4 application profile for the 42-times-6 WASM
integration fixture.

Assurance tier: model-level.

Owns: field laws, honest inversion, exact affine maps for `freshPublic` and
`encodeInstance`, exact hash binding, terminal equality checks, the selected
Phase-4 application, application-codec recovery, and the physical Step
recipe.

Does not own: a production WASM compiler, an Ajtai setup, the selected NIFS
footprint, a complete deployment, the recursive fixed point, Rust, or
generated artifacts.

Emits constraints: no new rows. It selects existing certified recipes.
-/

set_option autoImplicit false
set_option maxRecDepth 100000
set_option maxHeartbeats 1200000

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6

open Nightstream.HyperNova.Construction2.Paper
open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalKey
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalProofCodec
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalRunningCodec
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsParameters
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsPlain270Profile
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsCodecProjection
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Protocol.FPrime.ConcretePhi81
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc

private abbrev TranscriptState := Poseidon2Duplex.State

private def oneHot :
    (width : Nat) -> Fin width -> List Field
  | 0, index => Fin.elim0 index
  | width + 1, index =>
      Fin.cases
        (1 :: List.replicate width 0)
        (fun tail => 0 :: oneHot width tail)
        index

@[simp] private theorem oneHot_length
    (width : Nat) (selected : Fin width) :
    (oneHot width selected).length = width := by
  induction width with
  | zero =>
      exact Fin.elim0 selected
  | succ width inductionHypothesis =>
      refine Fin.cases ?_ (fun tail => ?_) selected
      · simp [oneHot]
      · simp [oneHot, inductionHypothesis tail]

private theorem fieldDot_replicate_zero
    (count : Nat) (coordinates : List Field) :
    fieldDot (List.replicate count 0) coordinates = 0 := by
  induction count generalizing coordinates with
  | zero =>
      simp [fieldDot]
  | succ count inductionHypothesis =>
      cases coordinates with
      | nil =>
          rfl
      | cons head tail =>
          change
            0 * head +
                fieldDot (List.replicate count 0) tail =
              0
          rw [inductionHypothesis tail]
          calc
            0 * head + 0 = 0 + 0 := by rw [Fin.zero_mul]
            _ = 0 := Fin.zero_add 0

private theorem fieldDot_oneHot
    (width : Nat)
    (selected : Fin width)
    (coordinates : List Field)
    (lengthExact : coordinates.length = width) :
    fieldDot (oneHot width selected) coordinates =
      coordinates.getD selected.val 0 := by
  induction width generalizing coordinates with
  | zero =>
      exact Fin.elim0 selected
  | succ width inductionHypothesis =>
      cases coordinates with
      | nil =>
          simp at lengthExact
      | cons head tail =>
          simp only [List.length_cons, Nat.succ.injEq] at lengthExact
          refine Fin.cases ?_ (fun tailIndex => ?_) selected
          · change
              1 * head +
                  fieldDot (List.replicate width 0) tail =
                head
            rw [fieldDot_replicate_zero]
            calc
              1 * head + 0 = head + 0 := by rw [Fin.one_mul]
              _ = head := Fin.add_zero head
          · simpa [oneHot, fieldDot] using
              inductionHypothesis tailIndex tail lengthExact

private def coordinateOfView
    {α : Type}
    (codec : Codec α)
    {value : α → Field}
    (view : FView codec value) :
    AffineCoordinate where
  constant := 0
  coefficients := oneHot codec.width view.index

@[simp] private theorem coordinateOfView_eval
    {α : Type}
    (codec : Codec α)
    {value : α → Field}
    (view : FView codec value)
    (input : α) :
    (coordinateOfView codec view).eval (codec.encode input) =
      value input := by
  unfold coordinateOfView AffineCoordinate.eval
  rw [Fin.zero_add]
  have selected :
      fieldDot
          (oneHot codec.width view.index)
          (codec.encode input) =
        (codec.encode input).getD view.index.val 0 := by
    exact fieldDot_oneHot codec.width view.index
      (codec.encode input) (codec.encode_length input)
  exact selected.trans (view.encodeValue input)

private noncomputable def digestView (coordinate : Fin 5) :
    FView digestCodec (fun digest => digest coordinate) := by
  change
    FView (Codec.finFunction 5 fieldCodec)
      (fun digest => digest coordinate)
  exact FView.finElement 5 coordinate
    ConcreteNifsCanonicalViews.fieldView

private noncomputable def encodeInstanceCoordinates : List AffineCoordinate :=
  List.ofFn fun coordinate : Fin 5 =>
    coordinateOfView digestCodec (digestView coordinate)

private noncomputable def encodeInstanceMap :
    AffineEncodingMap digestCodec encodedCodec encodeInstance where
  coordinates := encodeInstanceCoordinates
  coordinateCount := by
    simp [encodeInstanceCoordinates, encodedCodec_width]
  coefficientCounts := by
    intro coordinate member
    rcases List.mem_ofFn.mp member with ⟨index, rfl⟩
    simp [coordinateOfView, digestCodec_width]
  outputAdmissible := by
    intro digest admissible
    exact encodedCodec_admissible _
  encode_eq := by
    intro digest admissible
    change
      Codec.encodeFin fieldCodec 5 (encodeInstance digest) =
        encodeInstanceCoordinates.map
          (fun coordinate => coordinate.eval (digestCodec.encode digest))
    simp [Codec.encodeFin, fieldCodec,
      encodeInstanceCoordinates, encodeInstance]
    congr 1

private noncomputable def freshView
    {dimensions : Dimensions} {verifierRows : Nat}
    (coordinate : Fin 5) :
    FView (freshCodec dimensions verifierRows)
      (fun fresh : BenchmarkFresh dimensions verifierRows =>
        fresh.publicInput
          ⟨coordinate.val, by
            change coordinate.val < ringDegree * publicRingColumns
            simp only [ringDegree, publicRingColumns]
            omega⟩) :=
  (ConcreteNifsCanonicalViews.freshViews
    (Shape dimensions) publicRingColumns verifierRows
      (publicFits dimensions)).publicInput
        ⟨coordinate.val, by
          change coordinate.val < ringDegree * publicRingColumns
          simp only [ringDegree, publicRingColumns]
          omega⟩

private noncomputable def freshPublicCoordinates
    {dimensions : Dimensions} {verifierRows : Nat} :
    List AffineCoordinate :=
  List.ofFn fun coordinate : Fin 5 =>
    coordinateOfView (freshCodec dimensions verifierRows)
      (freshView
        (dimensions := dimensions)
        (verifierRows := verifierRows) coordinate)

private noncomputable def freshPublicMap
    {dimensions : Dimensions} {verifierRows : Nat} :
  AffineEncodingMap
      (freshCodec dimensions verifierRows) encodedCodec freshPublic where
  coordinates :=
    freshPublicCoordinates
      (dimensions := dimensions)
      (verifierRows := verifierRows)
  coordinateCount := by
    simp [freshPublicCoordinates, encodedCodec_width]
  coefficientCounts := by
    intro coordinate member
    rcases List.mem_ofFn.mp member with ⟨index, rfl⟩
    exact
      oneHot_length
        (freshCodec dimensions verifierRows).width
        (freshView
          (dimensions := dimensions)
          (verifierRows := verifierRows) index).index
  outputAdmissible := by
    intro fresh admissible
    exact encodedCodec_admissible _
  encode_eq := by
    intro fresh admissible
    change
      Codec.encodeFin fieldCodec 5 (freshPublic fresh) =
        (freshPublicCoordinates
          (dimensions := dimensions)
          (verifierRows := verifierRows)).map
            (fun coordinate =>
              coordinate.eval
                ((freshCodec dimensions verifierRows).encode fresh))
    simp [Codec.encodeFin, fieldCodec,
      freshPublicCoordinates, freshPublic]

def fieldLaws : FieldLaws where
  noZeroDivisors :=
    NormRange.baseFieldNoZeroDivisors_of_modulusEuclid
      GoldilocksField.goldilocks_euclidPrime

def inverseLaw : InverseLaw where
  inverse := fun value =>
    ⟨GoldilocksField.goldilocksInverseValue value.val,
      GoldilocksField.goldilocksInverseValue_canonical value.val⟩
  inverse_zero := by
    apply Fin.ext
    exact GoldilocksField.goldilocksInverseValue_zero
  mul_inverse_of_ne_zero := by
    intro value nonzero
    apply Fin.ext
    have valueNonzero : value.val ≠ 0 := by
      intro valueZero
      apply nonzero
      exact Fin.eq_of_val_eq valueZero
    simpa [Fin.val_mul] using
      GoldilocksField.goldilocksInverseValue_correct
        value.val value.isLt valueNonzero

attribute [local instance] stateDecidableEq encodedDecidableEq

section SelectedProfile

variable {dimensions : Dimensions} {verifierRows : Nat}
variable (setup : RelationSetup dimensions verifierRows)
variable (defaultRunning : BenchmarkRunning dimensions verifierRows)
variable (nifsFootprint : CallFootprint)

local notation "Plan" =>
  benchmarkHashPlan
    (dimensions := dimensions)
    (verifierRows := verifierRows)

local notation "BenchmarkParameters" =>
  ConcreteNifsPlain270Profile.selected dimensions
    (fun _ => ConcreteNifsCanonicalKey.selected setup)
    defaultRunning
    (machine Plan)
    (terminalRelations dimensions verifierRows)
    (terminalChecks dimensions verifierRows)
    (widths setup)
    (footprints
      (dimensions := dimensions)
      (verifierRows := verifierRows)
      nifsFootprint)

private theorem hashPrior_encoded
    (iteration : Nat)
    (z0 current : State)
    (running : BenchmarkRunning dimensions verifierRows) :
    digestCodec.encode
        (hashResult Plan iteration z0 current running) =
      Poseidon23Hash.resultCoordinates Plan
        (Poseidon23Hash.sourceCoordinates
          (parameters := BenchmarkParameters)
          (dataCodecs setup defaultRunning Plan
            (widths setup)
            (footprints
              (dimensions := dimensions)
              (verifierRows := verifierRows)
              nifsFootprint))
          false iteration z0 current running) := by
  unfold hashResult
  rw [digestOfCoordinates_encode
    (Poseidon23Hash.resultCoordinates Plan
      (hashSource iteration z0 current running))
    (Poseidon23Hash.resultCoordinates_length Plan
      (hashSource iteration z0 current running))]
  congr 1

private theorem hashNext_encoded
    (iteration : Nat)
    (z0 current : State)
    (running : BenchmarkRunning dimensions verifierRows) :
    digestCodec.encode
        (hashResult Plan (iteration + 1) z0 current running) =
      Poseidon23Hash.resultCoordinates Plan
        (Poseidon23Hash.sourceCoordinates
          (parameters := BenchmarkParameters)
          (dataCodecs setup defaultRunning Plan
            (widths setup)
            (footprints
              (dimensions := dimensions)
              (verifierRows := verifierRows)
              nifsFootprint))
          true iteration z0 current running) := by
  unfold hashResult
  rw [digestOfCoordinates_encode
    (Poseidon23Hash.resultCoordinates Plan
      (hashSource (iteration + 1) z0 current running))
    (Poseidon23Hash.resultCoordinates_length Plan
      (hashSource (iteration + 1) z0 current running))]
  congr 1
  change
    (boundedNatCodec.encode (iteration + 1)).getD 0 0 ::
        (stateCodec.encode z0 ++
          (stateCodec.encode current ++
            (runningCodec dimensions verifierRows).encode running)) =
      ((boundedNatCodec.encode iteration).getD 0 0 + 1) ::
        (stateCodec.encode z0 ++
          (stateCodec.encode current ++
            (runningCodec dimensions verifierRows).encode running))
  rw [boundedNatCodec_encode_succ]

/-- Complete Phase-4 profile for the exact 42-times-6 benchmark semantics.
The NIFS footprint remains an argument until the selected verifier program
derives it for every physical frame. -/
noncomputable def applicationProfile :
    Poseidon23ApplicationProfile BenchmarkParameters where
  toTerminalEqualityProfile := {
    toDirectProfile := {
      toProfile := {
        codecs :=
          dataCodecs setup defaultRunning Plan
            (widths setup)
            (footprints
              (dimensions := dimensions)
              (verifierRows := verifierRows)
              nifsFootprint)
        widthsExact := rfl
      }
      fieldLaws := fieldLaws
      inverseLaw := inverseLaw
      freshPublicMap := freshPublicMap
      encodeInstanceMap := encodeInstanceMap
      iterationZeroFootprint := rfl
      stateEqualFootprint := rfl
      freshPublicFootprint := rfl
      encodeInstanceFootprint := rfl
      encodedEqualFootprint := rfl
    }
    runningWidthsEqual := rfl
    freshWidthsEqual := rfl
    runningFootprint := rfl
    freshFootprint := rfl
    runningCheck_exact := by
      intro key running witness
      rfl
    freshCheck_exact := by
      intro key fresh witness
      rfl
  }
  alignmentWidth := 0
  hashPlan := Plan
  digestWidth := rfl
  digestAdmissible := digestCodec_admissible
  hashFootprint := rfl
  hashPrior_exact := by
    intro iteration z0 current running
    exact hashPrior_encoded setup defaultRunning nifsFootprint
      iteration z0 current running
  hashNext_exact := by
    intro iteration z0 current running
    exact hashNext_encoded setup defaultRunning nifsFootprint
      iteration z0 current running

/-- Phase-4 wrapper with the iteration slot proved to be source coordinate
zero. -/
noncomputable def phase4 :
    Phase4Application BenchmarkParameters where
  profile := applicationProfile setup defaultRunning nifsFootprint
  separating := {
    plan := Plan
    slotZero := rfl
  }
  separatingPlan_eq_hashPlan := rfl

/-- Exact representation and semantic bridge consumed by the physical Step
recipe. -/
noncomputable def stepProfile :
    StepProfile BenchmarkParameters where
  toProfile :=
    (applicationProfile setup defaultRunning nifsFootprint
      ).toTerminalEqualityProfile.toDirectProfile.toProfile
  stateEquiv := {
    toBenchmark := fun state => state
    fromBenchmark := fun state => state
    leftInverse := fun _ => rfl
    rightInverse := fun _ => rfl
  }
  stateEncodeExact := by
    intro state
    rfl
  stateAdmissible := stateCodec_admissible
  stateRecoverable := stateCodec_exactWidthRecoverable
  stepFootprintExact := rfl
  stepExact := by
    intro state witness
    rfl

/-- Certified physical Step program selected by the benchmark application. -/
noncomputable def selectedStepRecipe :
    CallRecipe
      (signature BenchmarkParameters)
      (applicationProfile setup defaultRunning nifsFootprint).family
      Call.step := by
  simpa [Poseidon23ApplicationProfile.family, TerminalEqualityProfile.family,
    DirectProfile.family, StepProfile.family] using
      stepRecipe BenchmarkParameters
        (stepProfile setup defaultRunning nifsFootprint)

/-- Canonical operational application boundary for the benchmark. -/
noncomputable def canonicalApplication :
    ConcreteNifsCanonicalOperationalProfile.Application
      setup defaultRunning
      (machine Plan)
      (terminalRelations dimensions verifierRows)
      (terminalChecks dimensions verifierRows)
      (widths setup)
      (footprints
        (dimensions := dimensions)
        (verifierRows := verifierRows)
        nifsFootprint) where
  phase4 := phase4 setup defaultRunning nifsFootprint
  runningCodec_exact := rfl
  freshCodec_exact := rfl
  proofCodec_exact := rfl

end SelectedProfile

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6
