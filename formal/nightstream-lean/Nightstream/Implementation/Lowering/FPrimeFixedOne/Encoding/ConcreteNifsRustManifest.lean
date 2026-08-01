import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ApplicationStepCostSplit
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCompleteApplication
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsPlain270Profile
import Nightstream.Implementation.Lowering.Goldilocks.CanonicalManifest

/-!
Proof-free manifest for the Lean-owned fixed-one/plain/270 F-prime programs.

The manifest is constructed only from the complete receipt-conserved Step and
Terminal encodings. It contains normalized sparse rows, structural owners,
allocation classes, result columns, selectors, branch activations, codec
segments, exact costs, and exact coefficient statistics.

The application-selected `step` remains a proof-carrying `CallRecipe`. No Rust
row, measured column, generated artifact, digest, or source hash enters this
construction.

Owns: the selected profile identifier; codec offsets; the concrete
fixed-one/plain/270 manifest constructor; round-trip and satisfaction
theorems; exact receipt-derived application-cost split.

Does not own: file I/O, a Rust emitter, a deployment application selection,
or equality with current Rust.

Emits constraints: none. It serializes existing canonical encodings.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsRustManifest

open Nightstream.HyperNova.Construction2.Paper
open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.Goldilocks.CanonicalManifest
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc

/-! ## Proof-free profile and codec metadata -/

inductive ProfileName where
  | fixedOnePlain270
deriving DecidableEq, Repr

/-- Protocol constants needed by a later Rust implementation. The matrix
count remains relation-selected rather than copied from Rust. -/
structure ProfileIdentifier where
  name : ProfileName
  matrixCount : Nat
  freshSourceCount : Nat
  runningSourceCount : Nat
  publicCarrierWidth : Nat
  freshLegacyWidth : Nat
  freshCompletionWidth : Nat
  runningCarrierWidth : Nat
  poseidonWidth : Nat
  poseidonRate : Nat
  poseidonCapacity : Nat
  poseidonDigestWidth : Nat
  bindingPreimageWidth : Nat
  decompositionBase : Nat
  decompositionChildren : Nat
deriving DecidableEq, Repr

def profileIdentifier
    (dimensions : Dimensions) : ProfileIdentifier where
  name := .fixedOnePlain270
  matrixCount := dimensions.matrixCount
  freshSourceCount := 1
  runningSourceCount := 14
  publicCarrierWidth := 270
  freshLegacyWidth := 257
  freshCompletionWidth := 13
  runningCarrierWidth := 270
  poseidonWidth := 8
  poseidonRate := 4
  poseidonCapacity := 4
  poseidonDigestWidth := 4
  bindingPreimageWidth := 23
  decompositionBase := 2
  decompositionChildren := 14

theorem profileIdentifier_exact
    (dimensions : Dimensions) :
    let identifier := profileIdentifier dimensions
    identifier.name = .fixedOnePlain270 ∧
      identifier.matrixCount = dimensions.matrixCount ∧
      identifier.freshSourceCount = 1 ∧
      identifier.runningSourceCount = 14 ∧
      identifier.publicCarrierWidth = 270 ∧
      identifier.freshLegacyWidth = 257 ∧
      identifier.freshCompletionWidth = 13 ∧
      identifier.runningCarrierWidth = 270 ∧
      identifier.poseidonWidth = 8 ∧
      identifier.poseidonRate = 4 ∧
      identifier.poseidonCapacity = 4 ∧
      identifier.poseidonDigestWidth = 4 ∧
      identifier.bindingPreimageWidth = 23 ∧
      identifier.decompositionBase = 2 ∧
      identifier.decompositionChildren = 14 := by
  simp [profileIdentifier]

inductive SegmentRole where
  | iteration
  | initialState
  | currentState
  | running
  | fresh
  | witness
  | nifsProof
  | nextState
  | nextRunning
  | digest
  | runningWitness
  | freshWitness
deriving DecidableEq, Repr

structure SegmentDescription where
  role : SegmentRole
  width : Nat
  ownership : Ownership
deriving DecidableEq, Repr

structure CodecSegment extends SegmentDescription where
  offset : Nat
deriving DecidableEq, Repr

def placeSegmentsFrom :
    Nat → List SegmentDescription → List CodecSegment
  | _, [] => []
  | offset, description :: rest =>
      { role := description.role
        width := description.width
        ownership := description.ownership
        offset := offset } ::
      placeSegmentsFrom (offset + description.width) rest

def ContiguousFrom : Nat → List CodecSegment → Prop
  | _, [] => True
  | offset, segment :: rest =>
      segment.offset = offset ∧
        ContiguousFrom (offset + segment.width) rest

theorem placeSegmentsFrom_contiguous
    (offset : Nat)
    (descriptions : List SegmentDescription) :
    ContiguousFrom offset (placeSegmentsFrom offset descriptions) := by
  induction descriptions generalizing offset with
  | nil =>
      trivial
  | cons description rest inductionHypothesis =>
      exact ⟨rfl, inductionHypothesis (offset + description.width)⟩

def stepInputDescriptions (widths : Widths) :
    List SegmentDescription :=
  [⟨.iteration, widths.iteration, .committedColumn⟩,
    ⟨.initialState, widths.state, .committedColumn⟩,
    ⟨.currentState, widths.state, .committedColumn⟩,
    ⟨.running, widths.running, .committedColumn⟩,
    ⟨.fresh, widths.fresh, .committedColumn⟩,
    ⟨.witness, widths.witness, .committedColumn⟩,
    ⟨.nifsProof, widths.nifsProof, .committedColumn⟩]

def stepResultDescriptions (widths : Widths) :
    List SegmentDescription :=
  [⟨.nextState, widths.state, .committedColumn⟩,
    ⟨.nextRunning, widths.running, .committedColumn⟩,
    ⟨.digest, widths.digest, .publicColumn⟩]

def terminalInputDescriptions (widths : Widths) :
    List SegmentDescription :=
  [⟨.iteration, widths.iteration, .publicColumn⟩,
    ⟨.initialState, widths.state, .publicColumn⟩,
    ⟨.currentState, widths.state, .publicColumn⟩,
    ⟨.running, widths.running, .committedColumn⟩,
    ⟨.runningWitness, widths.runningWitness, .committedColumn⟩,
    ⟨.fresh, widths.fresh, .committedColumn⟩,
    ⟨.freshWitness, widths.freshWitness, .committedColumn⟩]

def stepInputSegments (widths : Widths) : List CodecSegment :=
  placeSegmentsFrom 0 (stepInputDescriptions widths)

def stepResultSegments (widths : Widths) : List CodecSegment :=
  placeSegmentsFrom 0 (stepResultDescriptions widths)

def terminalInputSegments (widths : Widths) : List CodecSegment :=
  placeSegmentsFrom 0 (terminalInputDescriptions widths)

theorem stepInputSegments_contiguous (widths : Widths) :
    ContiguousFrom 0 (stepInputSegments widths) :=
  placeSegmentsFrom_contiguous 0 _

theorem stepResultSegments_contiguous (widths : Widths) :
    ContiguousFrom 0 (stepResultSegments widths) :=
  placeSegmentsFrom_contiguous 0 _

theorem terminalInputSegments_contiguous (widths : Widths) :
    ContiguousFrom 0 (terminalInputSegments widths) :=
  placeSegmentsFrom_contiguous 0 _

/-! ## Complete proof-free manifest -/

structure Manifest where
  profile : ProfileIdentifier
  widths : Widths
  stepInput : List CodecSegment
  stepResult : List CodecSegment
  terminalInput : List CodecSegment
  stepProgram : CanonicalManifest.Program
  terminalProgram : CanonicalManifest.Program
  stepResultColumns : List OwnedColumn
  stepSelector : ColumnId
  terminalSelector : ColumnId
  stepActivations : List ColumnId
  terminalActivations : List ColumnId
  stepCost : Cost
  terminalCost : Cost
  fixedProtocolCost : Cost
  applicationStepCost : Cost
  stepStatistics : Statistics
  terminalStatistics : Statistics
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

def completeCertificate
    (application : ConcreteNifsPlain270Profile.Phase4Application Selected)
    (nifs : ConcreteNifsVerifyCallRecipe.Certification application.profile)
    (step :
      CallRecipe (signature Selected)
        (application.profile.family Selected) Call.step)
    (defaultRunningAdmissible :
      ((application.profile.family Selected).codecFor (.data .running)).Admissible
        defaultRunning) :
    CompleteApplicationCertification Selected :=
  ConcreteNifsCompleteApplication.complete application nifs step
    defaultRunningAdmissible

/-- Deterministic proof-free image of the complete canonical Step and
Terminal programs. -/
def manifest
    (application : ConcreteNifsPlain270Profile.Phase4Application Selected)
    (nifs : ConcreteNifsVerifyCallRecipe.Certification application.profile)
    (step :
      CallRecipe (signature Selected)
        (application.profile.family Selected) Call.step)
    (defaultRunningAdmissible :
      ((application.profile.family Selected).codecFor (.data .running)).Admissible
        defaultRunning) : Manifest :=
  let certificate :=
    completeCertificate application nifs step defaultRunningAdmissible
  let stepProgram :=
    CanonicalManifest.Program.ofEncoding
      certificate.canonicalStep.encoding
  let terminalProgram :=
    CanonicalManifest.Program.ofEncoding
      certificate.canonicalTerminal.encoding
  {
    profile := profileIdentifier dimensions
    widths := widths
    stepInput := stepInputSegments widths
    stepResult := stepResultSegments widths
    terminalInput := terminalInputSegments widths
    stepProgram := stepProgram
    terminalProgram := terminalProgram
    stepResultColumns :=
      schemaOwnedColumns (CanonicalContexts.Step.result Selected)
    stepSelector :=
      CanonicalContexts.Step.selector Selected certificate.baseProfile
    terminalSelector :=
      CanonicalContexts.Terminal.selector Selected certificate.baseProfile
    stepActivations :=
      [activationColumn SourceOwners.stepBranchPath true,
        activationColumn SourceOwners.stepBranchPath false]
    terminalActivations :=
      [activationColumn SourceOwners.terminalBranchPath true,
        activationColumn SourceOwners.terminalBranchPath false]
    stepCost := certificate.stepCost
    terminalCost := certificate.terminalCost
    fixedProtocolCost :=
      _root_.Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ApplicationStepCostSplit.CompleteApplicationCertification.fixedProtocolCost
        certificate
    applicationStepCost :=
      _root_.Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ApplicationStepCostSplit.CompleteApplicationCertification.applicationStepCost
        certificate
    stepStatistics := stepProgram.statistics
    terminalStatistics := terminalProgram.statistics
  }

theorem manifest_profile_exact
    (application : ConcreteNifsPlain270Profile.Phase4Application Selected)
    (nifs : ConcreteNifsVerifyCallRecipe.Certification application.profile)
    (step :
      CallRecipe (signature Selected)
        (application.profile.family Selected) Call.step)
    (defaultRunningAdmissible :
      ((application.profile.family Selected).codecFor (.data .running)).Admissible
        defaultRunning) :
    (manifest application nifs step defaultRunningAdmissible).profile =
      profileIdentifier dimensions :=
  rfl

/-- Decoding the Step manifest recovers the normalized image of the exact
canonical Step encoding. -/
theorem step_roundTrip
    (application : ConcreteNifsPlain270Profile.Phase4Application Selected)
    (nifs : ConcreteNifsVerifyCallRecipe.Certification application.profile)
    (step :
      CallRecipe (signature Selected)
        (application.profile.family Selected) Call.step)
    (defaultRunningAdmissible :
      ((application.profile.family Selected).codecFor (.data .running)).Admissible
        defaultRunning) :
    (manifest application nifs step
      defaultRunningAdmissible).stepProgram.decode =
        CanonicalManifest.Program.imageOfEncoding
          (completeCertificate application nifs step
            defaultRunningAdmissible).canonicalStep.encoding := by
  exact CanonicalManifest.Program.decode_ofEncoding _

/-- Decoding the Terminal manifest recovers the normalized image of the exact
canonical Terminal encoding. -/
theorem terminal_roundTrip
    (application : ConcreteNifsPlain270Profile.Phase4Application Selected)
    (nifs : ConcreteNifsVerifyCallRecipe.Certification application.profile)
    (step :
      CallRecipe (signature Selected)
        (application.profile.family Selected) Call.step)
    (defaultRunningAdmissible :
      ((application.profile.family Selected).codecFor (.data .running)).Admissible
        defaultRunning) :
    (manifest application nifs step
      defaultRunningAdmissible).terminalProgram.decode =
        CanonicalManifest.Program.imageOfEncoding
          (completeCertificate application nifs step
            defaultRunningAdmissible).canonicalTerminal.encoding := by
  exact CanonicalManifest.Program.decode_ofEncoding _

theorem step_manifest_satisfies_iff
    (application : ConcreteNifsPlain270Profile.Phase4Application Selected)
    (nifs : ConcreteNifsVerifyCallRecipe.Certification application.profile)
    (step :
      CallRecipe (signature Selected)
        (application.profile.family Selected) Call.step)
    (defaultRunningAdmissible :
      ((application.profile.family Selected).codecFor (.data .running)).Admissible
        defaultRunning)
    (assignment : ColumnId → F) :
    Satisfies
        (manifest application nifs step
          defaultRunningAdmissible).stepProgram.decode.rows assignment ↔
      Satisfies
        (completeCertificate application nifs step
          defaultRunningAdmissible).canonicalStep.encoding.rows
        assignment := by
  exact CanonicalManifest.Program.decoded_satisfies_iff _ _

theorem terminal_manifest_satisfies_iff
    (application : ConcreteNifsPlain270Profile.Phase4Application Selected)
    (nifs : ConcreteNifsVerifyCallRecipe.Certification application.profile)
    (step :
      CallRecipe (signature Selected)
        (application.profile.family Selected) Call.step)
    (defaultRunningAdmissible :
      ((application.profile.family Selected).codecFor (.data .running)).Admissible
        defaultRunning)
    (assignment : ColumnId → F) :
    Satisfies
        (manifest application nifs step
          defaultRunningAdmissible).terminalProgram.decode.rows assignment ↔
      Satisfies
        (completeCertificate application nifs step
          defaultRunningAdmissible).canonicalTerminal.encoding.rows
        assignment := by
  exact CanonicalManifest.Program.decoded_satisfies_iff _ _

/-- Exact application-parametric Step cost. Both summands are receipt folds
over constructed programs. -/
theorem stepCost_eq_fixedProtocol_add_application
    (application : ConcreteNifsPlain270Profile.Phase4Application Selected)
    (nifs : ConcreteNifsVerifyCallRecipe.Certification application.profile)
    (step :
      CallRecipe (signature Selected)
        (application.profile.family Selected) Call.step)
    (defaultRunningAdmissible :
      ((application.profile.family Selected).codecFor (.data .running)).Admissible
        defaultRunning) :
    (manifest application nifs step defaultRunningAdmissible).stepCost =
      (manifest application nifs step
          defaultRunningAdmissible).fixedProtocolCost +
        (manifest application nifs step
          defaultRunningAdmissible).applicationStepCost := by
  exact
    _root_.Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ApplicationStepCostSplit.CompleteApplicationCertification.stepCost_eq_fixedProtocol_add_application
        (completeCertificate application nifs step
          defaultRunningAdmissible)

/-- The Step cost stored in the manifest is recomputed exactly from its
proof-free receipt stream. -/
theorem stepManifest_cost_exact
    (application : ConcreteNifsPlain270Profile.Phase4Application Selected)
    (nifs : ConcreteNifsVerifyCallRecipe.Certification application.profile)
    (step :
      CallRecipe (signature Selected)
        (application.profile.family Selected) Call.step)
    (defaultRunningAdmissible :
      ((application.profile.family Selected).codecFor (.data .running)).Admissible
        defaultRunning) :
    (manifest application nifs step
        defaultRunningAdmissible).stepProgram.cost =
      (manifest application nifs step
        defaultRunningAdmissible).stepCost := by
  exact CanonicalManifest.Program.cost_ofEncoding _

/-- The Terminal cost stored in the manifest is recomputed exactly from its
proof-free receipt stream. -/
theorem terminalManifest_cost_exact
    (application : ConcreteNifsPlain270Profile.Phase4Application Selected)
    (nifs : ConcreteNifsVerifyCallRecipe.Certification application.profile)
    (step :
      CallRecipe (signature Selected)
        (application.profile.family Selected) Call.step)
    (defaultRunningAdmissible :
      ((application.profile.family Selected).codecFor (.data .running)).Admissible
        defaultRunning) :
    (manifest application nifs step
        defaultRunningAdmissible).terminalProgram.cost =
      (manifest application nifs step
        defaultRunningAdmissible).terminalCost := by
  exact CanonicalManifest.Program.cost_ofEncoding _

theorem stepManifest_rows_length
    (application : ConcreteNifsPlain270Profile.Phase4Application Selected)
    (nifs : ConcreteNifsVerifyCallRecipe.Certification application.profile)
    (step :
      CallRecipe (signature Selected)
        (application.profile.family Selected) Call.step)
    (defaultRunningAdmissible :
      ((application.profile.family Selected).codecFor (.data .running)).Admissible
        defaultRunning) :
    (manifest application nifs step
        defaultRunningAdmissible).stepProgram.rows.length =
      (manifest application nifs step
        defaultRunningAdmissible).stepCost.recurringRows := by
  rw [← stepManifest_cost_exact application nifs step
      defaultRunningAdmissible]
  exact
    (CanonicalManifest.Program.cost_recurringRows
      (manifest application nifs step
        defaultRunningAdmissible).stepProgram).symm

theorem terminalManifest_rows_length
    (application : ConcreteNifsPlain270Profile.Phase4Application Selected)
    (nifs : ConcreteNifsVerifyCallRecipe.Certification application.profile)
    (step :
      CallRecipe (signature Selected)
        (application.profile.family Selected) Call.step)
    (defaultRunningAdmissible :
      ((application.profile.family Selected).codecFor (.data .running)).Admissible
        defaultRunning) :
    (manifest application nifs step
        defaultRunningAdmissible).terminalProgram.rows.length =
      (manifest application nifs step
        defaultRunningAdmissible).terminalCost.recurringRows := by
  rw [← terminalManifest_cost_exact application nifs step
      defaultRunningAdmissible]
  exact
    (CanonicalManifest.Program.cost_recurringRows
      (manifest application nifs step
        defaultRunningAdmissible).terminalProgram).symm

theorem stepResultColumns_exact
    (application : ConcreteNifsPlain270Profile.Phase4Application Selected)
    (nifs : ConcreteNifsVerifyCallRecipe.Certification application.profile)
    (step :
      CallRecipe (signature Selected)
        (application.profile.family Selected) Call.step)
    (defaultRunningAdmissible :
      ((application.profile.family Selected).codecFor (.data .running)).Admissible
        defaultRunning) :
    (manifest application nifs step
      defaultRunningAdmissible).stepResultColumns =
        schemaOwnedColumns (CanonicalContexts.Step.result Selected) :=
  rfl

theorem selectors_exact
    (application : ConcreteNifsPlain270Profile.Phase4Application Selected)
    (nifs : ConcreteNifsVerifyCallRecipe.Certification application.profile)
    (step :
      CallRecipe (signature Selected)
        (application.profile.family Selected) Call.step)
    (defaultRunningAdmissible :
      ((application.profile.family Selected).codecFor (.data .running)).Admissible
        defaultRunning) :
    let certificate :=
      completeCertificate application nifs step defaultRunningAdmissible
    (manifest application nifs step
        defaultRunningAdmissible).stepSelector =
        CanonicalContexts.Step.selector Selected certificate.baseProfile ∧
      (manifest application nifs step
        defaultRunningAdmissible).terminalSelector =
        CanonicalContexts.Terminal.selector Selected
          certificate.baseProfile := by
  exact ⟨rfl, rfl⟩

/-- The advertised Step selector occurs in the exact allocation stream
recovered from the proof-free manifest. -/
theorem stepSelector_allocated
    (application : ConcreteNifsPlain270Profile.Phase4Application Selected)
    (nifs : ConcreteNifsVerifyCallRecipe.Certification application.profile)
    (step :
      CallRecipe (signature Selected)
        (application.profile.family Selected) Call.step)
    (defaultRunningAdmissible :
      ((application.profile.family Selected).codecFor (.data .running)).Admissible
        defaultRunning) :
    (manifest application nifs step
        defaultRunningAdmissible).stepSelector ∈
      (manifest application nifs step
        defaultRunningAdmissible).stepProgram.decode.columns.map
          OwnedColumn.id := by
  simp only [manifest]
  rw [CanonicalManifest.Program.decoded_columns_eq]
  change
    CanonicalContexts.Step.selector Selected
        (completeCertificate application nifs step
          defaultRunningAdmissible).baseProfile ∈
      (completeCertificate application nifs step
        defaultRunningAdmissible).canonicalStep.encoding.columnIds
  rw [Encoding.columnIds_eq_receipt_columnIds]
  apply List.mem_flatMap.mpr
  refine ⟨
    (CanonicalStepPlan.selectorPlan.{0} Selected
      (completeCertificate application nifs step
        defaultRunningAdmissible).baseProfile
      (completeCertificate application nifs step
        defaultRunningAdmissible).allRecipes).receipt,
    ?_, ?_⟩
  · change
      (CanonicalStepPlan.selectorPlan.{0} Selected
        (completeCertificate application nifs step
          defaultRunningAdmissible).baseProfile
        (completeCertificate application nifs step
          defaultRunningAdmissible).allRecipes).receipt ∈
        CanonicalStepPlan.receipts Selected
          (completeCertificate application nifs step
            defaultRunningAdmissible).baseProfile
          (completeCertificate application nifs step
            defaultRunningAdmissible).allRecipes
          (completeCertificate application nifs step
            defaultRunningAdmissible).defaultRunningAdmissible
    apply List.mem_cons_of_mem
    apply List.mem_append_right
    rw [CanonicalStepPlan.bodyReceipts]
    exact List.mem_cons_of_mem _ (List.mem_cons_self)
  · let selectorPlan :=
      CanonicalStepPlan.selectorPlan.{0} Selected
        (completeCertificate application nifs step
          defaultRunningAdmissible).baseProfile
        (completeCertificate application nifs step
          defaultRunningAdmissible).allRecipes
    have outputCovered :=
      ReceiptScoping.PrimitivePlan.freshOutputsCoveredAfter selectorPlan []
    apply outputCovered
    exact CanonicalPrimitivePlan.bitCoordinate_mem
      (completeCertificate application nifs step
        defaultRunningAdmissible).baseProfile
      (.here (Ports.auxiliaryBit Selected))
      (instructionColumns SourceOwners.stepSelectorPath
        [Ports.auxiliaryBit Selected])
      ((completeCertificate application nifs step
        defaultRunningAdmissible).baseProfile.callOutputs_widthsAgree
          Selected .iterationZero)

/-- The advertised Terminal selector occurs in the exact allocation stream
recovered from the proof-free manifest. -/
theorem terminalSelector_allocated
    (application : ConcreteNifsPlain270Profile.Phase4Application Selected)
    (nifs : ConcreteNifsVerifyCallRecipe.Certification application.profile)
    (step :
      CallRecipe (signature Selected)
        (application.profile.family Selected) Call.step)
    (defaultRunningAdmissible :
      ((application.profile.family Selected).codecFor (.data .running)).Admissible
        defaultRunning) :
    (manifest application nifs step
        defaultRunningAdmissible).terminalSelector ∈
      (manifest application nifs step
        defaultRunningAdmissible).terminalProgram.decode.columns.map
          OwnedColumn.id := by
  simp only [manifest]
  rw [CanonicalManifest.Program.decoded_columns_eq]
  change
    CanonicalContexts.Terminal.selector Selected
        (completeCertificate application nifs step
          defaultRunningAdmissible).baseProfile ∈
      (completeCertificate application nifs step
        defaultRunningAdmissible).canonicalTerminal.encoding.columnIds
  rw [Encoding.columnIds_eq_receipt_columnIds]
  apply List.mem_flatMap.mpr
  refine ⟨
    (CanonicalTerminalPlan.selectorPlan.{0} Selected
      (completeCertificate application nifs step
        defaultRunningAdmissible).baseProfile
      (completeCertificate application nifs step
        defaultRunningAdmissible).allRecipes).receipt,
    ?_, ?_⟩
  · change
      (CanonicalTerminalPlan.selectorPlan.{0} Selected
        (completeCertificate application nifs step
          defaultRunningAdmissible).baseProfile
        (completeCertificate application nifs step
          defaultRunningAdmissible).allRecipes).receipt ∈
        CanonicalTerminalPlan.receipts Selected
          (completeCertificate application nifs step
            defaultRunningAdmissible).baseProfile
          (completeCertificate application nifs step
            defaultRunningAdmissible).allRecipes
    apply List.mem_cons_of_mem
    apply List.mem_append_right
    rw [CanonicalTerminalPlan.bodyReceipts]
    exact List.mem_cons_self
  · let selectorPlan :=
      CanonicalTerminalPlan.selectorPlan.{0} Selected
        (completeCertificate application nifs step
          defaultRunningAdmissible).baseProfile
        (completeCertificate application nifs step
          defaultRunningAdmissible).allRecipes
    have outputCovered :=
      ReceiptScoping.PrimitivePlan.freshOutputsCoveredAfter selectorPlan []
    apply outputCovered
    exact CanonicalPrimitivePlan.bitCoordinate_mem
      (completeCertificate application nifs step
        defaultRunningAdmissible).baseProfile
      (.here (Ports.auxiliaryBit Selected))
      (instructionColumns SourceOwners.terminalSelectorPath
        [Ports.auxiliaryBit Selected])
      ((completeCertificate application nifs step
        defaultRunningAdmissible).baseProfile.callOutputs_widthsAgree
          Selected .iterationZero)

theorem activations_exact
    (application : ConcreteNifsPlain270Profile.Phase4Application Selected)
    (nifs : ConcreteNifsVerifyCallRecipe.Certification application.profile)
    (step :
      CallRecipe (signature Selected)
        (application.profile.family Selected) Call.step)
    (defaultRunningAdmissible :
      ((application.profile.family Selected).codecFor (.data .running)).Admissible
        defaultRunning) :
    (manifest application nifs step
        defaultRunningAdmissible).stepActivations =
        [activationColumn SourceOwners.stepBranchPath true,
          activationColumn SourceOwners.stepBranchPath false] ∧
      (manifest application nifs step
        defaultRunningAdmissible).terminalActivations =
        [activationColumn SourceOwners.terminalBranchPath true,
          activationColumn SourceOwners.terminalBranchPath false] := by
  exact ⟨rfl, rfl⟩

/-- Both advertised Step activation columns occur in the exact allocation
stream recovered from the proof-free manifest. -/
theorem stepActivations_allocated
    (application : ConcreteNifsPlain270Profile.Phase4Application Selected)
    (nifs : ConcreteNifsVerifyCallRecipe.Certification application.profile)
    (step :
      CallRecipe (signature Selected)
        (application.profile.family Selected) Call.step)
    (defaultRunningAdmissible :
      ((application.profile.family Selected).codecFor (.data .running)).Admissible
        defaultRunning) :
    ∀ column ∈
        (manifest application nifs step
          defaultRunningAdmissible).stepActivations,
      column ∈
        (manifest application nifs step
          defaultRunningAdmissible).stepProgram.decode.columns.map
            OwnedColumn.id := by
  intro column member
  simp only [manifest] at member ⊢
  rw [CanonicalManifest.Program.decoded_columns_eq]
  simp only [List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl
  · change
      activationColumn SourceOwners.stepBranchPath true ∈
        (completeCertificate application nifs step
          defaultRunningAdmissible).canonicalStep.encoding.columnIds
    rw [Encoding.columnIds_eq_receipt_columnIds]
    apply List.mem_flatMap.mpr
    refine ⟨
      CanonicalBranchPlan.trueActivationReceipt
        SourceOwners.stepBranchPath oneColumn oneColumn
        (CanonicalContexts.Step.selector Selected
      (completeCertificate application nifs step
            defaultRunningAdmissible).baseProfile),
      ?_, ?_⟩
    · change
        CanonicalBranchPlan.trueActivationReceipt
            SourceOwners.stepBranchPath oneColumn oneColumn
            (CanonicalContexts.Step.selector Selected
              (completeCertificate application nifs step
                defaultRunningAdmissible).baseProfile) ∈
          CanonicalStepPlan.receipts Selected
            (completeCertificate application nifs step
              defaultRunningAdmissible).baseProfile
            (completeCertificate application nifs step
              defaultRunningAdmissible).allRecipes
            (completeCertificate application nifs step
              defaultRunningAdmissible).defaultRunningAdmissible
      simp [CanonicalStepPlan.receipts, CanonicalStepPlan.bodyReceipts]
    · simp [CanonicalBranchPlan.trueActivationReceipt,
        CanonicalBranchPlan.activationRecipe,
        InstructionReceipt.columnIds,
        InstructionReceipt.ofTrueActivation]
  · change
      activationColumn SourceOwners.stepBranchPath false ∈
        (completeCertificate application nifs step
          defaultRunningAdmissible).canonicalStep.encoding.columnIds
    rw [Encoding.columnIds_eq_receipt_columnIds]
    apply List.mem_flatMap.mpr
    refine ⟨
      CanonicalBranchPlan.falseActivationReceipt
        SourceOwners.stepBranchPath oneColumn oneColumn
        (CanonicalContexts.Step.selector Selected
      (completeCertificate application nifs step
            defaultRunningAdmissible).baseProfile),
      ?_, ?_⟩
    · change
        CanonicalBranchPlan.falseActivationReceipt
            SourceOwners.stepBranchPath oneColumn oneColumn
            (CanonicalContexts.Step.selector Selected
              (completeCertificate application nifs step
                defaultRunningAdmissible).baseProfile) ∈
          CanonicalStepPlan.receipts Selected
            (completeCertificate application nifs step
              defaultRunningAdmissible).baseProfile
            (completeCertificate application nifs step
              defaultRunningAdmissible).allRecipes
            (completeCertificate application nifs step
              defaultRunningAdmissible).defaultRunningAdmissible
      simp [CanonicalStepPlan.receipts, CanonicalStepPlan.bodyReceipts]
    · simp [CanonicalBranchPlan.falseActivationReceipt,
        CanonicalBranchPlan.activationRecipe,
        InstructionReceipt.columnIds,
        InstructionReceipt.ofFalseActivation]

/-- Both advertised Terminal activation columns occur in the exact allocation
stream recovered from the proof-free manifest. -/
theorem terminalActivations_allocated
    (application : ConcreteNifsPlain270Profile.Phase4Application Selected)
    (nifs : ConcreteNifsVerifyCallRecipe.Certification application.profile)
    (step :
      CallRecipe (signature Selected)
        (application.profile.family Selected) Call.step)
    (defaultRunningAdmissible :
      ((application.profile.family Selected).codecFor (.data .running)).Admissible
        defaultRunning) :
    ∀ column ∈
        (manifest application nifs step
          defaultRunningAdmissible).terminalActivations,
      column ∈
        (manifest application nifs step
          defaultRunningAdmissible).terminalProgram.decode.columns.map
            OwnedColumn.id := by
  intro column member
  simp only [manifest] at member ⊢
  rw [CanonicalManifest.Program.decoded_columns_eq]
  simp only [List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl
  · change
      activationColumn SourceOwners.terminalBranchPath true ∈
        (completeCertificate application nifs step
          defaultRunningAdmissible).canonicalTerminal.encoding.columnIds
    rw [Encoding.columnIds_eq_receipt_columnIds]
    apply List.mem_flatMap.mpr
    refine ⟨
      CanonicalBranchPlan.trueActivationReceipt
        SourceOwners.terminalBranchPath oneColumn oneColumn
        (CanonicalContexts.Terminal.selector Selected
      (completeCertificate application nifs step
            defaultRunningAdmissible).baseProfile),
      ?_, ?_⟩
    · change
        CanonicalBranchPlan.trueActivationReceipt
            SourceOwners.terminalBranchPath oneColumn oneColumn
            (CanonicalContexts.Terminal.selector Selected
              (completeCertificate application nifs step
                defaultRunningAdmissible).baseProfile) ∈
          CanonicalTerminalPlan.receipts Selected
            (completeCertificate application nifs step
              defaultRunningAdmissible).baseProfile
            (completeCertificate application nifs step
              defaultRunningAdmissible).allRecipes
      simp [CanonicalTerminalPlan.receipts,
        CanonicalTerminalPlan.bodyReceipts]
    · simp [CanonicalBranchPlan.trueActivationReceipt,
        CanonicalBranchPlan.activationRecipe,
        InstructionReceipt.columnIds,
        InstructionReceipt.ofTrueActivation]
  · change
      activationColumn SourceOwners.terminalBranchPath false ∈
        (completeCertificate application nifs step
          defaultRunningAdmissible).canonicalTerminal.encoding.columnIds
    rw [Encoding.columnIds_eq_receipt_columnIds]
    apply List.mem_flatMap.mpr
    refine ⟨
      CanonicalBranchPlan.falseActivationReceipt
        SourceOwners.terminalBranchPath oneColumn oneColumn
        (CanonicalContexts.Terminal.selector Selected
      (completeCertificate application nifs step
            defaultRunningAdmissible).baseProfile),
      ?_, ?_⟩
    · change
        CanonicalBranchPlan.falseActivationReceipt
            SourceOwners.terminalBranchPath oneColumn oneColumn
            (CanonicalContexts.Terminal.selector Selected
              (completeCertificate application nifs step
                defaultRunningAdmissible).baseProfile) ∈
          CanonicalTerminalPlan.receipts Selected
            (completeCertificate application nifs step
              defaultRunningAdmissible).baseProfile
            (completeCertificate application nifs step
              defaultRunningAdmissible).allRecipes
      simp [CanonicalTerminalPlan.receipts,
        CanonicalTerminalPlan.bodyReceipts]
    · simp [CanonicalBranchPlan.falseActivationReceipt,
        CanonicalBranchPlan.activationRecipe,
        InstructionReceipt.columnIds,
        InstructionReceipt.ofFalseActivation]

end Plain270

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsRustManifest
