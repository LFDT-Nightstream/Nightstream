import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsNativeCcsStepCompleteness
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsRustManifest
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.TerminalR1cs.Descriptor
import Nightstream.Implementation.Lowering.Goldilocks.NativeCcsManifest

/-!
Contract: proof-free native CCS manifest for fixed-one/plain/270 F-prime.

Assurance tier: model-level.

Owns:
- the native four-matrix Step program and exact receipt-derived Step cost;
- the unchanged outer Terminal program and shared profile metadata;
- the direct terminal-R1CS dimensions and exact derived cost;
- proof-free round trips, soundness, honest completeness, row count, and
  allocation conservation.

Does not own: JSON, Rust parsing, a deployment application, or a recursive
fixed-point relation.

Emits constraints: none. It serializes the constructed native CCS Step and
canonical Terminal programs.
-/

set_option autoImplicit false
set_option maxRecDepth 100000
set_option maxHeartbeats 2200000

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsNativeCcsManifest

open Nightstream.HyperNova.Construction2.Paper
open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.Goldilocks.NativeCcsManifest
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc

/-- Native Step plus unchanged Terminal proof-free program. Shared metadata
comes from the same Lean deployment, never from a Rust measurement. -/
structure Manifest where
  profile : ConcreteNifsRustManifest.ProfileIdentifier
  widths : Widths
  stepInput : List ConcreteNifsRustManifest.CodecSegment
  stepResult : List ConcreteNifsRustManifest.CodecSegment
  terminalInput : List ConcreteNifsRustManifest.CodecSegment
  stepProgram : NativeCcsManifest.Program
  terminalProgram : CanonicalManifest.Program
  terminalR1cs : TerminalR1cs.Descriptor
  stepResultColumns : List OwnedColumn
  stepSelector : ColumnId
  terminalSelector : ColumnId
  stepActivations : List ColumnId
  terminalActivations : List ColumnId
  stepCost : Cost
  terminalCost : Cost
deriving DecidableEq, Repr

private abbrev TranscriptState := Poseidon2Duplex.State

section Plain270

variable {dimensions : Dimensions}
variable {Digest AppState Witness Encoded
  RunningWitness FreshWitness : Type}
variable [DecidableEq AppState] [DecidableEq Encoded]
variable {verifierRows : Nat}
variable {keys :
  Fin 1 →
    ConcreteNifsPlain270Profile.Key dimensions TranscriptState verifierRows}
variable {defaultRunning :
  ConcreteNifsPlain270Profile.Running dimensions verifierRows}
variable {machine :
  Machine
    (ConcreteNifsPlain270Profile.Key dimensions TranscriptState verifierRows)
    Digest AppState Witness
    (ConcreteNifsPlain270Profile.Running dimensions verifierRows)
    (ConcreteNifsPlain270Profile.Fresh dimensions verifierRows)
    Encoded 1}
variable {terminalRelations :
  TerminalRelations
    (ConcreteNifsPlain270Profile.Key dimensions TranscriptState verifierRows)
    (ConcreteNifsPlain270Profile.Running dimensions verifierRows)
    RunningWitness
    (ConcreteNifsPlain270Profile.Fresh dimensions verifierRows)
    FreshWitness 1}
variable {terminalChecks :
  Nightstream.Protocol.FPrime.CanonicalTerminalVerifier.RelationChecks
    terminalRelations}
variable {widths : Widths} {footprints : Footprints}

local notation "Selected" =>
  ConcreteNifsPlain270Profile.selected dimensions keys defaultRunning machine
    terminalRelations terminalChecks widths footprints

private abbrev StepRecipeFor
    (application : ConcreteNifsPlain270Profile.Phase4Application Selected) :=
  CallRecipe (signature Selected)
    (application.profile.family Selected) Call.step

private abbrev DefaultAdmissibleFor
    (application : ConcreteNifsPlain270Profile.Phase4Application Selected) :=
  ((application.profile.family Selected).codecFor (.data .running)).Admissible
    defaultRunning

def certificate
    (application : ConcreteNifsPlain270Profile.Phase4Application Selected)
    (nifs : ConcreteNifsVerifyCallRecipe.Certification application.profile)
    (step : StepRecipeFor application)
    (defaultAdmissible : DefaultAdmissibleFor application) :
    CompleteApplicationCertification Selected :=
  ConcreteNifsNativeCcsStep.certificate
    application nifs step defaultAdmissible

def nativeStepProgram
    (application : ConcreteNifsPlain270Profile.Phase4Application Selected)
    (nifs : ConcreteNifsVerifyCallRecipe.Certification application.profile)
    (step : StepRecipeFor application)
    (defaultAdmissible : DefaultAdmissibleFor application) :=
  ConcreteNifsNativeCcsStep.program
    application nifs step defaultAdmissible

/-- Deterministic proof-free native manifest. The legacy manifest is used
only for metadata and the unchanged Terminal program. Its activated Step
program and Step cost are not copied. -/
def manifest
    (application : ConcreteNifsPlain270Profile.Phase4Application Selected)
    (nifs : ConcreteNifsVerifyCallRecipe.Certification application.profile)
    (step : StepRecipeFor application)
    (defaultAdmissible : DefaultAdmissibleFor application) : Manifest :=
  let shared :=
    ConcreteNifsRustManifest.manifest
      application nifs step defaultAdmissible
  let native :=
    nativeStepProgram application nifs step defaultAdmissible
  {
    profile := shared.profile
    widths := shared.widths
    stepInput := shared.stepInput
    stepResult := shared.stepResult
    terminalInput := shared.terminalInput
    stepProgram := NativeCcsManifest.Program.ofProgram native
    terminalProgram := shared.terminalProgram
    terminalR1cs :=
      TerminalR1cs.Descriptor.ofProgram native dimensions.rowVariables
        publicRingColumns verifierRows
    stepResultColumns := shared.stepResultColumns
    stepSelector := shared.stepSelector
    terminalSelector := shared.terminalSelector
    stepActivations := shared.stepActivations
    terminalActivations := shared.terminalActivations
    stepCost := native.cost
    terminalCost := shared.terminalCost
  }

@[simp] theorem terminalR1cs_logicalWidth
    (application : ConcreteNifsPlain270Profile.Phase4Application Selected)
    (nifs : ConcreteNifsVerifyCallRecipe.Certification application.profile)
    (step : StepRecipeFor application)
    (defaultAdmissible : DefaultAdmissibleFor application) :
    (manifest application nifs step
      defaultAdmissible).terminalR1cs.logicalWidth =
        (nativeStepProgram application nifs step
          defaultAdmissible).columnIds.length := by
  rfl

@[simp] theorem terminalR1cs_recursiveRows
    (application : ConcreteNifsPlain270Profile.Phase4Application Selected)
    (nifs : ConcreteNifsVerifyCallRecipe.Certification application.profile)
    (step : StepRecipeFor application)
    (defaultAdmissible : DefaultAdmissibleFor application) :
    (manifest application nifs step
      defaultAdmissible).terminalR1cs.recursiveRows =
        (nativeStepProgram application nifs step
          defaultAdmissible).rows.length := by
  rfl

theorem step_shape_valid
    (application : ConcreteNifsPlain270Profile.Phase4Application Selected)
    (nifs : ConcreteNifsVerifyCallRecipe.Certification application.profile)
    (step : StepRecipeFor application)
    (defaultAdmissible : DefaultAdmissibleFor application) :
    (manifest application nifs step defaultAdmissible).stepProgram.Valid :=
  NativeCcsManifest.Program.valid_ofProgram _

theorem step_roundTrip
    (application : ConcreteNifsPlain270Profile.Phase4Application Selected)
    (nifs : ConcreteNifsVerifyCallRecipe.Certification application.profile)
    (step : StepRecipeFor application)
    (defaultAdmissible : DefaultAdmissibleFor application) :
    (manifest application nifs step
      defaultAdmissible).stepProgram.decode =
        NativeCcsManifest.Program.imageOf
          (nativeStepProgram application nifs step defaultAdmissible) :=
  NativeCcsManifest.Program.decode_ofProgram _

theorem stepManifest_cost_exact
    (application : ConcreteNifsPlain270Profile.Phase4Application Selected)
    (nifs : ConcreteNifsVerifyCallRecipe.Certification application.profile)
    (step : StepRecipeFor application)
    (defaultAdmissible : DefaultAdmissibleFor application) :
    (manifest application nifs step
        defaultAdmissible).stepProgram.cost =
      (manifest application nifs step defaultAdmissible).stepCost :=
  NativeCcsManifest.Program.cost_ofProgram _

theorem stepManifest_rows_length
    (application : ConcreteNifsPlain270Profile.Phase4Application Selected)
    (nifs : ConcreteNifsVerifyCallRecipe.Certification application.profile)
    (step : StepRecipeFor application)
    (defaultAdmissible : DefaultAdmissibleFor application) :
    (manifest application nifs step
        defaultAdmissible).stepProgram.rows.length =
      (manifest application nifs step
        defaultAdmissible).stepCost.recurringRows := by
  rw [← stepManifest_cost_exact application nifs step defaultAdmissible]
  exact
    (NativeCcsManifest.Program.cost_recurringRows
      (manifest application nifs step defaultAdmissible).stepProgram).symm

theorem stepManifest_columns_exact
    (application : ConcreteNifsPlain270Profile.Phase4Application Selected)
    (nifs : ConcreteNifsVerifyCallRecipe.Certification application.profile)
    (step : StepRecipeFor application)
    (defaultAdmissible : DefaultAdmissibleFor application) :
    (manifest application nifs step
        defaultAdmissible).stepProgram.columns =
      (nativeStepProgram application nifs step
        defaultAdmissible).allocations :=
  NativeCcsManifest.Program.columns_ofProgram _

theorem stepManifest_satisfies_iff
    (application : ConcreteNifsPlain270Profile.Phase4Application Selected)
    (nifs : ConcreteNifsVerifyCallRecipe.Certification application.profile)
    (step : StepRecipeFor application)
    (defaultAdmissible : DefaultAdmissibleFor application)
    (assignment : ColumnId → F) :
    (manifest application nifs step
        defaultAdmissible).stepProgram.decode.Satisfies assignment ↔
      (nativeStepProgram application nifs step
        defaultAdmissible).Satisfies assignment :=
  NativeCcsManifest.Program.decoded_program_satisfies_iff _ _

/-- Manifest satisfaction reaches the frozen F-prime Step relation. -/
theorem sound
    (application : ConcreteNifsPlain270Profile.Phase4Application Selected)
    (nifs : ConcreteNifsVerifyCallRecipe.Certification application.profile)
    (step : StepRecipeFor application)
    (defaultAdmissible : DefaultAdmissibleFor application)
    (assignment : ColumnId → F)
    (input :
      Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.Canonical.Input
        AppState Witness
        (ConcreteNifsPlain270Profile.Running dimensions verifierRows)
        (ConcreteNifsPlain270Profile.Fresh dimensions verifierRows)
        (ConcreteNifsPlain270Profile.Proof dimensions TranscriptState
          verifierRows))
    (satisfied :
      (manifest application nifs step
        defaultAdmissible).stepProgram.decode.Satisfies assignment)
    (inputDecoded :
      Columns.Decodes
        ((certificate application nifs step defaultAdmissible
          ).baseProfile.family Selected)
        (CanonicalContexts.Step.input Selected) assignment
        (stepInputValues Selected input)) :
    ∃ output :
        Nightstream.HyperNova.Construction2.Paper.Output
          Digest AppState
          (ConcreteNifsPlain270Profile.Running dimensions verifierRows) 1,
      Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.Accepts
        Selected input output := by
  apply
    ConcreteNifsNativeCcsStep.sound
      application nifs step defaultAdmissible assignment input
  · exact
      (stepManifest_satisfies_iff application nifs step
        defaultAdmissible assignment).1 satisfied
  · exact inputDecoded

/-- Every accepted admissible Step has one assignment accepted by the
proof-free native manifest and encoding the exact input and output. -/
theorem complete
    (application : ConcreteNifsPlain270Profile.Phase4Application Selected)
    (nifs : ConcreteNifsVerifyCallRecipe.Certification application.profile)
    (step : StepRecipeFor application)
    (defaultAdmissible : DefaultAdmissibleFor application)
    (input : CanonicalStepCompleteness.StepInputFor Selected)
    (output : CanonicalStepCompleteness.StepOutputFor Selected)
    (accepted :
      Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.Accepts
        Selected input output)
    (admissible :
      CanonicalStepCompleteness.AdmissibleExecution Selected
        (certificate application nifs step defaultAdmissible).baseProfile
        input (CanonicalStepCompleteness.selectedRunning output)) :
    ∃ assignment : ColumnId → F,
      (manifest application nifs step
        defaultAdmissible).stepProgram.decode.Satisfies assignment ∧
        Columns.Encodes
          ((certificate application nifs step defaultAdmissible
            ).baseProfile.family Selected)
          (CanonicalContexts.Step.input Selected) assignment
          (stepInputValues Selected input) ∧
        Columns.Encodes
          ((certificate application nifs step defaultAdmissible
            ).baseProfile.family Selected)
          (CanonicalContexts.Step.result Selected) assignment
          (stepResultValues Selected output) := by
  rcases
      ConcreteNifsNativeCcsStepCompleteness.complete
        application nifs step defaultAdmissible
        input output accepted admissible with
    ⟨assignment, nativeSatisfied, inputEncoded, outputEncoded⟩
  exact
    ⟨assignment,
      (stepManifest_satisfies_iff application nifs step
        defaultAdmissible assignment).2 nativeSatisfied,
      inputEncoded, outputEncoded⟩

end Plain270

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsNativeCcsManifest
