import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalCertification
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalInputRecovery
import Nightstream.Implementation.Lowering.Goldilocks.SchemaRecovery

/-!
Contract: combine application-owned codec recovery with the selected
protocol-owned carrier recovery.

Owns: exact-width recovery for every semantic value in the current Terminal
input and for every non-proof semantic value in the current Step input.

Does not own: active NIFS-proof recovery, branch soundness, physical rows,
Rust, or generated artifacts.

Emits constraints: none.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalApplicationRecovery

open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalCertification
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalInputRecovery
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalKey
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalOperationalProfile
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsPlain270Profile
open Nightstream.Protocol.FPrime.ConcretePhi81
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc

private abbrev TranscriptState :=
  Nightstream.Implementation.R1CS.Canonical.Poseidon2Duplex.State

section

variable {Digest AppState Witness Encoded RunningWitness FreshWitness : Type}
variable [DecidableEq AppState] [DecidableEq Encoded]
variable {dimensions : Dimensions}
variable {verifierRows : Nat}
variable (setup : RelationSetup dimensions verifierRows)
variable (defaultRunning : Running dimensions verifierRows)
variable
  (machine :
    Nightstream.HyperNova.Construction2.Paper.Machine
      (Key dimensions TranscriptState verifierRows)
      Digest AppState Witness
      (Running dimensions verifierRows)
      (Fresh dimensions verifierRows)
      Encoded 1)
variable
  (terminalRelations :
    Nightstream.HyperNova.Construction2.Paper.TerminalRelations
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

local notation "Selected" =>
  ConcreteNifsPlain270Profile.selected dimensions
    (ConcreteNifsCanonicalOperationalProfile.selectedKeys setup)
    defaultRunning machine terminalRelations terminalChecks widths footprints

/-- Every current Terminal input coordinate vector has a semantic value. The
application supplies recovery only for its state and relation-witness codecs;
the selected running and fresh codecs are recovered by protocol theorems. -/
theorem terminalInputSchema_exactWidthRecoverable
    (deployment :
      Deployment setup defaultRunning machine terminalRelations
        terminalChecks widths footprints) :
    SchemaExactWidthRecoverable
      (deployment.application.phase4.profile.family Selected)
      (terminalInputSchema Selected) := by
  intro port member
  simp only [terminalInputSchema, List.mem_cons, List.not_mem_nil,
    or_false] at member
  rcases member with rfl | rfl | rfl | rfl | rfl | rfl | rfl
  · change boundedNatCodec.ExactWidthRecoverable
    exact Codec.boundedNatCodec_exactWidthRecoverable
  · change
      deployment.application.phase4.profile.codecs.state.ExactWidthRecoverable
    exact deployment.applicationCodecRecovery.state
  · change
      deployment.application.phase4.profile.codecs.state.ExactWidthRecoverable
    exact deployment.applicationCodecRecovery.state
  · change
      deployment.application.phase4.profile.codecs.running.ExactWidthRecoverable
    rw [deployment.application.runningCodec_exact]
    exact runningCodec_exactWidthRecoverable
      (ConcreteNifsPlain270Profile.Shape dimensions)
      publicRingColumns verifierRows
      (ConcreteNifsPlain270Profile.publicFits dimensions)
  · change
      deployment.application.phase4.profile.codecs.runningWitness.ExactWidthRecoverable
    exact deployment.applicationCodecRecovery.runningWitness
  · change
      deployment.application.phase4.profile.codecs.fresh.ExactWidthRecoverable
    rw [deployment.application.freshCodec_exact]
    exact freshCodec_exactWidthRecoverable
      (ConcreteNifsPlain270Profile.Shape dimensions)
      publicRingColumns verifierRows
      (ConcreteNifsPlain270Profile.publicFits dimensions)
  · change
      deployment.application.phase4.profile.codecs.freshWitness.ExactWidthRecoverable
    exact deployment.applicationCodecRecovery.freshWitness

/-- The exact current Terminal input columns construct one statement and one
fixed-one proof. No semantic input or decode equation is supplied. -/
theorem terminalInput_decode_exists
    (deployment :
      Deployment setup defaultRunning machine terminalRelations
        terminalChecks widths footprints)
    (assignment : ColumnId → Field) :
    ∃ statement :
        Nightstream.HyperNova.Construction2.Paper.TerminalStatement AppState,
      ∃ proof :
          Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.FixedOneTerminal.Proof
            Selected,
        Columns.Decodes
          (deployment.application.phase4.profile.family Selected)
          (CanonicalContexts.Terminal.input Selected) assignment
          (terminalInputValues Selected statement proof) := by
  rcases
      (CanonicalContexts.Terminal.input Selected).decode_exists assignment
        (CanonicalContexts.Terminal.inputWidths Selected
          deployment.application.phase4.certification.profile)
        (terminalInputSchema_exactWidthRecoverable
          setup defaultRunning machine terminalRelations terminalChecks
            widths footprints deployment) with
    ⟨values, decoded⟩
  change
    HVec (fun port => (typeSystem Selected).Value port.kind)
      (terminalInputSchema Selected) at values
  cases values with
  | cons iteration values =>
      cases values with
      | cons z0 values =>
          cases values with
          | cons zi values =>
              cases values with
              | cons running values =>
                  cases values with
                  | cons runningWitness values =>
                      cases values with
                      | cons fresh values =>
                          cases values with
                          | cons freshWitness values =>
                              cases values with
                              | nil =>
                                  let statement :
                                      Nightstream.HyperNova.Construction2.Paper.TerminalStatement
                                        AppState := {
                                    iteration := iteration
                                    z0 := z0
                                    zi := zi
                                  }
                                  let proof :
                                      Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.FixedOneTerminal.Proof
                                        Selected := {
                                    running := running
                                    runningWitness := runningWitness
                                    fresh := fresh
                                    freshWitness := freshWitness
                                  }
                                  exact
                                    ⟨statement, proof, by
                                      simpa [statement, proof,
                                        terminalInputValues] using decoded⟩

/-- One referenced Step input bundle decodes when its selected codec is
total at the exact declared width. -/
theorem stepInputRef_decode_exists
    (deployment :
      Deployment setup defaultRunning machine terminalRelations
        terminalChecks widths footprints)
    (assignment : ColumnId → Field)
    {kind : (typeSystem Selected).Kind}
    (reference :
      Ref (typeSystem Selected) (stepInputSchema Selected) kind)
    (recoverable :
      ((deployment.application.phase4.profile.family Selected).codecFor kind
        ).ExactWidthRecoverable) :
    ∃ value : (typeSystem Selected).Value kind,
      ((CanonicalContexts.Step.input Selected).toSchemaBundles.get reference
        ).Decodes
        (deployment.application.phase4.profile.family Selected)
        kind assignment value := by
  let bundle :=
    (CanonicalContexts.Step.input Selected).toSchemaBundles.get reference
  have lengthExact :
      (bundle.values assignment).length =
        ((deployment.application.phase4.profile.family Selected).codecFor kind
          ).width := by
    rw [ColumnBundle.values_length]
    have widthAtPort :=
      CanonicalContexts.Step.inputWidths Selected
        deployment.application.phase4.certification.profile
        reference.port (ref_port_mem reference)
    unfold PortWidthAgrees at widthAtPort
    rw [reference.port_sort] at widthAtPort
    simpa [Phase4Application.certification,
      ApplicationCertification.poseidon23] using widthAtPort.symm
  rcases
      Codec.decode_exists_of_exactWidthRecoverable
        recoverable (bundle.values assignment) lengthExact with
    ⟨value, decoded⟩
  exact ⟨value, decoded⟩

/-- The common and base Step path recovers exactly the four semantic inputs
that it reads. The inactive recursive proof is intentionally absent. -/
theorem stepBaseInputs_decode_exists
    (deployment :
      Deployment setup defaultRunning machine terminalRelations
        terminalChecks widths footprints)
    (assignment : ColumnId → Field) :
    ∃ iteration : Nat,
      ∃ z0 : AppState,
        ∃ zi : AppState,
          ∃ witness : Witness,
            ((CanonicalContexts.Step.input Selected).toSchemaBundles.get
                (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.InputRefs.iteration
                  Selected)).Decodes
                (deployment.application.phase4.profile.family Selected)
                (.data .nat) assignment iteration ∧
              ((CanonicalContexts.Step.input Selected).toSchemaBundles.get
                (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.InputRefs.z0
                  Selected)).Decodes
                (deployment.application.phase4.profile.family Selected)
                (.data .state) assignment z0 ∧
              ((CanonicalContexts.Step.input Selected).toSchemaBundles.get
                (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.InputRefs.zi
                  Selected)).Decodes
                (deployment.application.phase4.profile.family Selected)
                (.data .state) assignment zi ∧
              ((CanonicalContexts.Step.input Selected).toSchemaBundles.get
                (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.InputRefs.witness
                  Selected)).Decodes
                (deployment.application.phase4.profile.family Selected)
                (.data .witness) assignment witness := by
  rcases
      stepInputRef_decode_exists
        setup defaultRunning machine terminalRelations terminalChecks
          widths footprints deployment assignment
          (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.InputRefs.iteration
            Selected)
          Codec.boundedNatCodec_exactWidthRecoverable with
    ⟨iteration, iterationDecoded⟩
  rcases
      stepInputRef_decode_exists
        setup defaultRunning machine terminalRelations terminalChecks
          widths footprints deployment assignment
          (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.InputRefs.z0
            Selected)
          deployment.applicationCodecRecovery.state with
    ⟨z0, z0Decoded⟩
  rcases
      stepInputRef_decode_exists
        setup defaultRunning machine terminalRelations terminalChecks
          widths footprints deployment assignment
          (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.InputRefs.zi
            Selected)
          deployment.applicationCodecRecovery.state with
    ⟨zi, ziDecoded⟩
  rcases
      stepInputRef_decode_exists
        setup defaultRunning machine terminalRelations terminalChecks
          widths footprints deployment assignment
          (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.InputRefs.witness
            Selected)
          deployment.applicationCodecRecovery.witness with
    ⟨witness, witnessDecoded⟩
  exact
    ⟨iteration, z0, zi, witness, iterationDecoded, z0Decoded, ziDecoded,
      witnessDecoded⟩

/-- The active recursive NIFS occurrence supplies the only Step codec that is
not total at its declared width. Once its running, fresh, and proof operands
decode, the exact Step input columns construct the complete typed input. -/
theorem stepInput_decode_exists_of_recursiveOperands
    (deployment :
      Deployment setup defaultRunning machine terminalRelations
        terminalChecks widths footprints)
    (assignment : ColumnId → Field)
    (running : Running dimensions verifierRows)
    (fresh : Fresh dimensions verifierRows)
    (proof : Proof dimensions TranscriptState verifierRows)
    (runningDecoded :
      ((CanonicalContexts.Step.input Selected).toSchemaBundles.get
          (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.InputRefs.running
            Selected)).Decodes
        (deployment.application.phase4.profile.family Selected)
        (.data .running) assignment running)
    (freshDecoded :
      ((CanonicalContexts.Step.input Selected).toSchemaBundles.get
          (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.InputRefs.fresh
            Selected)).Decodes
        (deployment.application.phase4.profile.family Selected)
        (.data .fresh) assignment fresh)
    (proofDecoded :
      ((CanonicalContexts.Step.input Selected).toSchemaBundles.get
          (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.InputRefs.nifsProof
            Selected)).Decodes
        (deployment.application.phase4.profile.family Selected)
        (.data .nifsProof) assignment proof) :
    ∃ input :
        Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.Canonical.Input
          AppState Witness
          (Running dimensions verifierRows)
          (Fresh dimensions verifierRows)
          (Proof dimensions TranscriptState verifierRows),
      Columns.Decodes
        (deployment.application.phase4.profile.family Selected)
        (CanonicalContexts.Step.input Selected) assignment
        (stepInputValues Selected input) := by
  rcases
      stepInputRef_decode_exists
        setup defaultRunning machine terminalRelations terminalChecks
          widths footprints deployment assignment
          (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.InputRefs.iteration
            Selected)
          Codec.boundedNatCodec_exactWidthRecoverable with
    ⟨iteration, iterationDecoded⟩
  rcases
      stepInputRef_decode_exists
        setup defaultRunning machine terminalRelations terminalChecks
          widths footprints deployment assignment
          (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.InputRefs.z0
            Selected)
          deployment.applicationCodecRecovery.state with
    ⟨z0, z0Decoded⟩
  rcases
      stepInputRef_decode_exists
        setup defaultRunning machine terminalRelations terminalChecks
          widths footprints deployment assignment
          (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.InputRefs.zi
            Selected)
          deployment.applicationCodecRecovery.state with
    ⟨zi, ziDecoded⟩
  rcases
      stepInputRef_decode_exists
        setup defaultRunning machine terminalRelations terminalChecks
          widths footprints deployment assignment
          (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.InputRefs.witness
            Selected)
          deployment.applicationCodecRecovery.witness with
    ⟨witness, witnessDecoded⟩
  let input :
      Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.Canonical.Input
        AppState Witness
        (Running dimensions verifierRows)
        (Fresh dimensions verifierRows)
        (Proof dimensions TranscriptState verifierRows) := {
    iteration := iteration
    z0 := z0
    zi := zi
    running := fun _ => running
    fresh := fresh
    witness := witness
    nifsProof := proof
  }
  refine ⟨input, ?_⟩
  exact
    ⟨iterationDecoded,
      z0Decoded,
      ziDecoded,
      runningDecoded,
      freshDecoded,
      witnessDecoded,
      proofDecoded,
      True.intro⟩

end

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalApplicationRecovery
