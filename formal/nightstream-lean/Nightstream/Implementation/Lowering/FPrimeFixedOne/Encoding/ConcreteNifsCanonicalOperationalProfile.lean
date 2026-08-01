import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalKey
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalProfileViews
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalRunningCoverage
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOperationalProfile

/-!
Contract: construct the operational fixed-one NIFS profile from the selected
Lean key, canonical codecs, canonical transcript serialization, and
application-owned Phase-4 data.

The application owns its state, step, hash projection, and terminal
relations.  It must select the canonical running, fresh, and NIFS-proof
codecs because those three types cross the NIFS verifier boundary.  After
those three representation equalities, every field of
`ConcreteNifsOperationalProfile.Profile` is constructed here.

Owns: one actual operational profile value, all profile laws, and the exact
connection from the application codec family to the selected NIFS carriers.

Does not own: an application step recipe, verifier acceptance, a prover
message, footprint placement, Rust, or generated artifacts.

Emits constraints: none.
-/

set_option autoImplicit false
set_option maxRecDepth 100000
set_option maxHeartbeats 1200000

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalOperationalProfile

open Nightstream.HyperNova.Construction2.Paper
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalKey
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalProofCodec
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalProfileViews
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalRunningCodec
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalRunningCoverage
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalSerialization
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalViews
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOperationalProfile
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

section SelectedApplication

variable {Digest AppState Witness Encoded RunningWitness FreshWitness : Type}
variable [DecidableEq AppState] [DecidableEq Encoded]
variable {dimensions : Dimensions}
variable {verifierRows : Nat}
variable (setup : RelationSetup dimensions verifierRows)
variable (defaultRunning : Running dimensions verifierRows)
variable
  (machine :
    Machine
      (Key dimensions TranscriptState verifierRows)
      Digest AppState Witness
      (Running dimensions verifierRows)
      (Fresh dimensions verifierRows)
      Encoded 1)
variable
  (terminalRelations :
    TerminalRelations
      (Key dimensions TranscriptState verifierRows)
      (Running dimensions verifierRows)
      RunningWitness
      (Fresh dimensions verifierRows)
      FreshWitness 1)
variable
  (terminalChecks :
    Nightstream.Protocol.FPrime.CanonicalTerminalVerifier.RelationChecks
      terminalRelations)
variable (widths : Widths) (footprints : Footprints)

/-- The selected key occupies the sole fixed-one verifier-key slot. -/
noncomputable def selectedKeys :
    Fin 1 → Key dimensions TranscriptState verifierRows :=
  fun _ => ConcreteNifsCanonicalKey.selected setup

local notation "Selected" =>
  ConcreteNifsPlain270Profile.selected dimensions
    (selectedKeys setup) defaultRunning machine terminalRelations
      terminalChecks widths footprints

private abbrev ConstraintPolynomial :=
  CCSResidualTable.ConstraintPolynomial
    F (ConcreteNifsPlain270Profile.Shape dimensions).matrixCount

/-- A statement source list and its exact serialization law travel through
codec equality as one dependent value. -/
private structure StatementData
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (runningCodec :
      Codec
        (SelectedRunning shape publicRingColumns publicFits verifierRows))
    (freshCodec :
      Codec
        (SelectedFresh shape publicRingColumns publicFits verifierRows))
    (proofCodec :
      Codec
        (SelectedProof shape TranscriptState publicRingColumns publicFits
          verifierRows))
    (serialization :
      KSplitNcPoseidonSchedule.Serialization
        (Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.VerifierKey
          shape publicRingColumns publicFits verifierRows)
        (Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.StatementInput
          shape publicRingColumns publicFits verifierRows
            Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.arity)
        shape)
    (key :
      SelectedKey shape TranscriptState publicRingColumns publicFits
        verifierRows) where
  sources :
    List
      (ConcreteNifsOperationalProfile.FieldSource
        runningCodec freshCodec proofCodec)
  exact :
    ∀ running fresh proof,
      sources.map
          (fun source => (source.value running fresh proof).val) =
        serialization.statementFields
          (ConcreteNifsParameters.context key running fresh proof
            ).materialize.piCcsStatement
  length :
    sources.length =
      10 + runningCodec.width + freshCodec.width +
        shape.rowVariables * 2 +
        shape.runningCount * shape.matrixCount * ringDegree * 2

/-- An output source list, its exact serialization law, and its duplex
cursor law travel through codec equality as one dependent value. -/
private structure OutputData
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (runningCodec :
      Codec
        (SelectedRunning shape publicRingColumns publicFits verifierRows))
    (freshCodec :
      Codec
        (SelectedFresh shape publicRingColumns publicFits verifierRows))
    (proofCodec :
      Codec
        (SelectedProof shape TranscriptState publicRingColumns publicFits
          verifierRows))
    (serialization :
      KSplitNcPoseidonSchedule.Serialization
        (Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.VerifierKey
          shape publicRingColumns publicFits verifierRows)
        (Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.StatementInput
          shape publicRingColumns publicFits verifierRows
            Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.arity)
        shape) where
  sources :
    List
      (ConcreteNifsOperationalProfile.FieldSource
        runningCodec freshCodec proofCodec)
  exact :
    ∀ running fresh proof,
      sources.map
          (fun source => (source.value running fresh proof).val) =
        serialization.outputFields proof.certificate.piCcs.output
  length :
    sources.length =
      3 +
        shape.sourceCount * shape.matrixCount * ringDegree * 2 +
        shape.sourceCount * ringDegree * 2
  cursorOne :
    SymbolicDuplexCursor.after 0 (2 + sources.length) = 1

/-- Running views and their whole-codec coverage law form one dependent
value. -/
private structure RunningData
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (runningCodec :
      Codec
        (SelectedRunning shape publicRingColumns publicFits verifierRows)) where
  views : ConcreteNifsCarrierViews.RunningViews runningCodec
  coverage :
    ConcreteNifsCarrierViews.RunningCodecCoverage runningCodec views

/-- Application boundary for the three NIFS carriers.

These equations select representations.  They contain no verifier result,
acceptance proposition, transcript challenge, or row fact. -/
structure Application where
  phase4 : Phase4Application Selected
  runningCodec_exact :
    phase4.profile.codecs.running =
      ConcreteNifsCanonicalRunningCodec.runningCodec
        (ConcreteNifsPlain270Profile.Shape dimensions)
        publicRingColumns verifierRows (publicFits dimensions)
  freshCodec_exact :
    phase4.profile.codecs.fresh =
      ConcreteNifsCanonicalRunningCodec.freshCodec
        (ConcreteNifsPlain270Profile.Shape dimensions)
        publicRingColumns verifierRows (publicFits dimensions)
  proofCodec_exact :
    phase4.profile.codecs.nifsProof =
      ConcreteNifsCanonicalProofCodec.proofCodec
        (ConcreteNifsPlain270Profile.Shape dimensions)
        setup.system.constraintPolynomial 0
        publicRingColumns verifierRows (publicFits dimensions)

/-- Construct every field of the operational NIFS profile.

The proof begins by replacing the application's three selected NIFS codecs
with their canonical Lean definitions.  All remaining obligations are then
theorems about those definitions. -/
noncomputable def operational
    (application :
      Application setup defaultRunning machine terminalRelations
        terminalChecks widths footprints) :
    ConcreteNifsOperationalProfile.Profile application.phase4.profile := by
  let shape := ConcreteNifsPlain270Profile.Shape dimensions
  let polynomial : ConstraintPolynomial (dimensions := dimensions) :=
    setup.system.constraintPolynomial
  let canonicalStatementData :
      StatementData
        (ConcreteNifsCanonicalRunningCodec.runningCodec
          shape publicRingColumns verifierRows (publicFits dimensions))
        (ConcreteNifsCanonicalRunningCodec.freshCodec
          shape publicRingColumns verifierRows (publicFits dimensions))
        (ConcreteNifsCanonicalProofCodec.proofCodec
          shape polynomial 0 publicRingColumns verifierRows
            (publicFits dimensions))
        (ConcreteNifsCanonicalKey.selectedSerialization dimensions
          verifierRows)
        (ConcreteNifsCanonicalKey.selected setup) := {
    sources :=
      ConcreteNifsCanonicalSerialization.statementSources
        shape polynomial 0 publicRingColumns verifierRows
          (publicFits dimensions)
    exact :=
      ConcreteNifsCanonicalSerialization.statementSources_values
        shape polynomial 0 publicRingColumns verifierRows
          (publicFits dimensions)
        (ConcreteNifsCanonicalKey.selected setup)
    length :=
      ConcreteNifsCanonicalSerialization.statementSources_length
        shape polynomial 0 publicRingColumns verifierRows
          (publicFits dimensions)
  }
  let statementData :
      StatementData
        ((application.phase4.profile.family Selected).codecFor
          (.data .running))
        ((application.phase4.profile.family Selected).codecFor
          (.data .fresh))
        ((application.phase4.profile.family Selected).codecFor
          (.data .nifsProof))
        (ConcreteNifsCanonicalKey.selectedSerialization dimensions
          verifierRows)
        (ConcreteNifsCanonicalKey.selected setup) := by
    simpa only [Poseidon23ApplicationProfile.family,
      DirectCalls.DirectProfile.family,
      Profile.family, DataCodecs.family, application.runningCodec_exact,
      application.freshCodec_exact, application.proofCodec_exact] using
        canonicalStatementData
  let canonicalOutputData :
      OutputData
        (ConcreteNifsCanonicalRunningCodec.runningCodec
          shape publicRingColumns verifierRows (publicFits dimensions))
        (ConcreteNifsCanonicalRunningCodec.freshCodec
          shape publicRingColumns verifierRows (publicFits dimensions))
        (ConcreteNifsCanonicalProofCodec.proofCodec
          shape polynomial 0 publicRingColumns verifierRows
            (publicFits dimensions))
        (ConcreteNifsCanonicalKey.selectedSerialization dimensions
          verifierRows) := {
    sources :=
      ConcreteNifsCanonicalSerialization.outputSources
        shape polynomial 0 publicRingColumns verifierRows
          (publicFits dimensions)
    exact :=
      ConcreteNifsCanonicalSerialization.outputSources_values
        shape polynomial 0 publicRingColumns verifierRows
          (publicFits dimensions)
    length :=
      ConcreteNifsCanonicalSerialization.outputSources_length
        shape polynomial 0 publicRingColumns verifierRows
          (publicFits dimensions)
    cursorOne :=
      ConcreteNifsCanonicalSerialization.outputCursorOne
        shape polynomial 0 publicRingColumns verifierRows
          (publicFits dimensions)
  }
  let outputData :
      OutputData
        ((application.phase4.profile.family Selected).codecFor
          (.data .running))
        ((application.phase4.profile.family Selected).codecFor
          (.data .fresh))
        ((application.phase4.profile.family Selected).codecFor
          (.data .nifsProof))
        (ConcreteNifsCanonicalKey.selectedSerialization dimensions
          verifierRows) := by
    simpa only [Poseidon23ApplicationProfile.family,
      DirectCalls.DirectProfile.family,
      Profile.family, DataCodecs.family, application.runningCodec_exact,
      application.freshCodec_exact, application.proofCodec_exact] using
        canonicalOutputData
  let canonicalRunningData :
      RunningData
        (ConcreteNifsCanonicalRunningCodec.runningCodec
          shape publicRingColumns verifierRows
            (publicFits dimensions)) := {
    views :=
      ConcreteNifsCanonicalViews.runningViews
        shape publicRingColumns verifierRows (publicFits dimensions)
    coverage :=
      ConcreteNifsCanonicalRunningCoverage.coverage
        shape publicRingColumns verifierRows (publicFits dimensions)
  }
  let runningData :
      RunningData
        ((application.phase4.profile.family Selected).codecFor
          (.data .running)) := by
    simpa only [Poseidon23ApplicationProfile.family,
      DirectCalls.DirectProfile.family,
      Profile.family, DataCodecs.family,
      application.runningCodec_exact] using canonicalRunningData
  exact {
    constants := Poseidon2CanonicalConstants.selected
    serialization :=
      ConcreteNifsCanonicalKey.selectedSerialization dimensions verifierRows
    constraintPolynomial := polynomial
    priorAbsorbed := 0
    proofAdmissiblePolynomial := by
      intro proof admissible
      simp only [Poseidon23ApplicationProfile.family,
        DirectCalls.DirectProfile.family,
        Profile.family, DataCodecs.family,
        application.proofCodec_exact] at admissible
      change ProofAdmissible polynomial 0 proof at admissible
      exact admissible.constraintPolynomial_eq
    proofAdmissibleCursor := by
      intro proof admissible
      simp only [Poseidon23ApplicationProfile.family,
        DirectCalls.DirectProfile.family,
        Profile.family, DataCodecs.family,
        application.proofCodec_exact] at admissible
      change ProofAdmissible polynomial 0 proof at admissible
      exact admissible.priorAbsorbed_eq
    proofAdmissiblePriorState := by
      intro proof admissible
      simp only [Poseidon23ApplicationProfile.family,
        DirectCalls.DirectProfile.family,
        Profile.family, DataCodecs.family,
        application.proofCodec_exact] at admissible
      change ProofAdmissible polynomial 0 proof at admissible
      exact admissible.priorState_eq
    proofAdmissibleLanes := by
      intro proof admissible lane
      simp only [Poseidon23ApplicationProfile.family,
        DirectCalls.DirectProfile.family,
        Profile.family, DataCodecs.family,
        application.proofCodec_exact] at admissible
      change ProofAdmissible polynomial 0 proof at admissible
      exact admissible.priorLane_lt lane
    selectedSchedule := by
      rfl
    selectedSamplerMachine := by
      rfl
    priorLane := fun lane =>
      by
        simpa only [Poseidon23ApplicationProfile.family,
          DirectCalls.DirectProfile.family,
          Profile.family, DataCodecs.family,
          application.proofCodec_exact] using
          proofPriorLaneView shape polynomial 0
            publicRingColumns verifierRows (publicFits dimensions) lane
    statementSources := statementData.sources
    statementExact := statementData.exact
    statementLength := statementData.length
    outputSources := outputData.sources
    outputExact := outputData.exact
    outputLength := outputData.length
    outputCursorOne := outputData.cursorOne
    messageViews := by
      simpa only [Poseidon23ApplicationProfile.family,
        DirectCalls.DirectProfile.family,
        Profile.family, DataCodecs.family,
        application.proofCodec_exact] using
        ConcreteNifsCanonicalProfileViews.messageViews
          shape polynomial 0 publicRingColumns verifierRows
            (publicFits dimensions)
    samplerViews := by
      simpa only [Poseidon23ApplicationProfile.family,
        DirectCalls.DirectProfile.family,
        Profile.family, DataCodecs.family,
        application.proofCodec_exact] using
        ConcreteNifsCanonicalProfileViews.samplerViews
          shape polynomial 0 publicRingColumns verifierRows
            (publicFits dimensions)
    endpointViews := by
      simpa only [Poseidon23ApplicationProfile.family,
        DirectCalls.DirectProfile.family,
        Profile.family, DataCodecs.family,
        application.proofCodec_exact] using
        ConcreteNifsCanonicalProfileViews.endpointViews
          shape polynomial 0 publicRingColumns verifierRows
            (publicFits dimensions)
    runningViews := runningData.views
    runningCoverage := runningData.coverage
    freshViews := by
      simpa only [Poseidon23ApplicationProfile.family,
        DirectCalls.DirectProfile.family,
        Profile.family, DataCodecs.family,
        application.freshCodec_exact] using
        ConcreteNifsCanonicalViews.freshViews
          shape publicRingColumns verifierRows (publicFits dimensions)
    payloadViews := by
      simpa only [Poseidon23ApplicationProfile.family,
        DirectCalls.DirectProfile.family,
        Profile.family, DataCodecs.family,
        application.proofCodec_exact] using
        ConcreteNifsCanonicalProfileViews.payloadViews
          shape polynomial 0 publicRingColumns verifierRows
            (publicFits dimensions)
  }

@[simp] theorem operational_constants
    (application :
      Application setup defaultRunning machine terminalRelations
        terminalChecks widths footprints) :
    (operational setup defaultRunning machine terminalRelations
      terminalChecks widths footprints application).constants =
      Poseidon2CanonicalConstants.selected := by
  simp [operational]

@[simp] theorem operational_constraintPolynomial
    (application :
      Application setup defaultRunning machine terminalRelations
        terminalChecks widths footprints) :
    (operational setup defaultRunning machine terminalRelations
      terminalChecks widths footprints application).constraintPolynomial =
      setup.system.constraintPolynomial := by
  simp [operational]

@[simp] theorem operational_priorAbsorbed
    (application :
      Application setup defaultRunning machine terminalRelations
        terminalChecks widths footprints) :
    (operational setup defaultRunning machine terminalRelations
      terminalChecks widths footprints application).priorAbsorbed = 0 := by
  simp [operational]

private theorem fView_mpr_index_val
    {α : Type}
    {leftCodec rightCodec : Codec α}
    {value : α → Field}
    (codecEqual : leftCodec = rightCodec)
    (view : FView rightCodec value) :
    (Eq.mpr
        (congrArg (fun codec => FView codec value) codecEqual)
        view).index.val =
      view.index.val := by
  cases codecEqual
  rfl

/-- The physical index of each prior-duplex lane is the index selected by
the canonical proof codec. Relation matrix values do not affect this index. -/
theorem operational_priorLane_index
    (application :
      Application setup defaultRunning machine terminalRelations
        terminalChecks widths footprints)
    (lane : Fin 8) :
    ((operational setup defaultRunning machine terminalRelations
      terminalChecks widths footprints application).priorLane lane).index.val =
      (proofPriorLaneView
        (ConcreteNifsPlain270Profile.Shape dimensions)
        setup.system.constraintPolynomial 0 publicRingColumns verifierRows
        (publicFits dimensions) lane).index.val := by
  simp only [operational, Poseidon23ApplicationProfile.family,
    DirectCalls.DirectProfile.family,
    Profile.family, DataCodecs.family]
  apply fView_mpr_index_val
  exact application.proofCodec_exact

@[simp] theorem operational_statementSources_length
    (application :
      Application setup defaultRunning machine terminalRelations
        terminalChecks widths footprints) :
    (operational setup defaultRunning machine terminalRelations
      terminalChecks widths footprints application).statementSources.length =
      10 +
        (ConcreteNifsCanonicalRunningCodec.runningCodec
          (ConcreteNifsPlain270Profile.Shape dimensions)
          publicRingColumns verifierRows
          (publicFits dimensions)).width +
        (ConcreteNifsCanonicalRunningCodec.freshCodec
          (ConcreteNifsPlain270Profile.Shape dimensions)
          publicRingColumns verifierRows
          (publicFits dimensions)).width +
        (ConcreteNifsPlain270Profile.Shape dimensions).rowVariables * 2 +
          (ConcreteNifsPlain270Profile.Shape dimensions).runningCount *
          (ConcreteNifsPlain270Profile.Shape dimensions).matrixCount *
          ringDegree * 2 := by
  simpa only [Poseidon23ApplicationProfile.family,
    DirectCalls.DirectProfile.family,
    Profile.family, DataCodecs.family, application.runningCodec_exact,
    application.freshCodec_exact] using
      (operational setup defaultRunning machine terminalRelations
        terminalChecks widths footprints application).statementLength

@[simp] theorem operational_outputSources_length
    (application :
      Application setup defaultRunning machine terminalRelations
        terminalChecks widths footprints) :
    (operational setup defaultRunning machine terminalRelations
      terminalChecks widths footprints application).outputSources.length =
      3 +
        (ConcreteNifsPlain270Profile.Shape dimensions).sourceCount *
          (ConcreteNifsPlain270Profile.Shape dimensions).matrixCount *
          ringDegree * 2 +
        (ConcreteNifsPlain270Profile.Shape dimensions).sourceCount *
          ringDegree * 2 := by
  exact
    (operational setup defaultRunning machine terminalRelations
      terminalChecks widths footprints application).outputLength

end SelectedApplication

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalOperationalProfile
