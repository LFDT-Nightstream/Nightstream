import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalOperationalProfile
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalInputRecovery
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalProofRecovery
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsRawProgram

/-!
Contract: recover the selected semantic NIFS proof from the exact physical
proof bundle and the emitted proof-canonicality rows.

Owns: the canonical proof decoder at the physical `nifsVerify` boundary.
The theorem derives the empty prior duplex from row satisfaction and uses
the canonical application codec equality to construct the proof value.

Does not own: running or fresh input decoding, verifier acceptance, output
decoding, activation, Rust, or generated artifacts.

Emits constraints: no new rows. It consumes the eight rows owned by
`ConcreteNifsProofCanonicalityRows`.
-/

set_option autoImplicit false
set_option maxRecDepth 100000
set_option maxHeartbeats 1200000

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalProofPhysicalDecode

open Nightstream.HyperNova.Construction2.Paper
open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsCallFrame
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsCodecProjection
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalKey
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalProofCodec
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalProofRecovery
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalInputRecovery
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalRunningCodec
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalViews
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsParameters
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsPlain270Profile
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Protocol.FPrime.ConcretePhi81
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270
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

local notation "Selected" =>
  ConcreteNifsPlain270Profile.selected dimensions
    (ConcreteNifsCanonicalOperationalProfile.selectedKeys setup)
    defaultRunning machine terminalRelations terminalChecks widths footprints

private abbrev CanonicalApplication :=
  ConcreteNifsCanonicalOperationalProfile.Application
    setup defaultRunning machine terminalRelations terminalChecks
      widths footprints

private noncomputable abbrev FamilyFor
    (application : CanonicalApplication setup defaultRunning machine
      terminalRelations terminalChecks widths footprints) :=
  application.phase4.profile.family Selected

private abbrev FrameFor
    (application : CanonicalApplication setup defaultRunning machine
      terminalRelations terminalChecks widths footprints)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)} :=
  CallFrame (signature := signature Selected)
    (FamilyFor setup defaultRunning machine terminalRelations terminalChecks
      widths footprints application)
    Call.nifsVerify
    (Refs.cons runningRef
      (Refs.cons freshRef (Refs.cons proofRef .nil)))

private theorem canonicality_satisfied_of_rawRows
    (application : CanonicalApplication setup defaultRunning machine
      terminalRelations terminalChecks widths footprints)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor setup defaultRunning machine terminalRelations
      terminalChecks widths footprints application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (assignment : ColumnId → Field)
    (satisfied :
      RawSatisfies
        (ConcreteNifsRawProgram.rawRows application.phase4.profile
          (ConcreteNifsCanonicalOperationalProfile.operational
            setup defaultRunning machine terminalRelations terminalChecks
              widths footprints application)
          frame)
        assignment) :
    RawSatisfies
      (ConcreteNifsProofCanonicalityRows.rows application.phase4.profile
        (ConcreteNifsCanonicalOperationalProfile.operational
          setup defaultRunning machine terminalRelations terminalChecks
            widths footprints application)
        frame)
      assignment := by
  unfold ConcreteNifsRawProgram.rawRows at satisfied
  rcases
      (rawSatisfies_append_iff _ _ assignment).mp satisfied with
    ⟨prefixFive, _output⟩
  rcases
      (rawSatisfies_append_iff _ _ assignment).mp prefixFive with
    ⟨prefixFour, _piDec⟩
  rcases
      (rawSatisfies_append_iff _ _ assignment).mp prefixFour with
    ⟨prefixThree, _action⟩
  rcases
      (rawSatisfies_append_iff _ _ assignment).mp prefixThree with
    ⟨prefixTwo, _point⟩
  rcases
      (rawSatisfies_append_iff _ _ assignment).mp prefixTwo with
    ⟨prefixOne, _operational⟩
  exact
    ((rawSatisfies_append_iff _ _ assignment).mp prefixOne).1

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

private theorem operational_priorLane_index
    (application : CanonicalApplication setup defaultRunning machine
      terminalRelations terminalChecks widths footprints)
    (lane : Fin 8) :
    ((ConcreteNifsCanonicalOperationalProfile.operational
          setup defaultRunning machine terminalRelations terminalChecks
            widths footprints application).priorLane lane).index.val =
      (proofPriorLaneView
        (ConcreteNifsPlain270Profile.Shape dimensions)
        setup.system.constraintPolynomial 0 publicRingColumns verifierRows
        (publicFits dimensions) lane).index.val := by
  simp only [ConcreteNifsCanonicalOperationalProfile.operational,
    Poseidon23ApplicationProfile.family,
    TerminalEqualityProfile.family, DirectCalls.DirectProfile.family,
    Profile.family, DataCodecs.family]
  apply fView_mpr_index_val
  exact application.proofCodec_exact

private theorem coordinateId_eq_of_index_val
    {α β : Type}
    {leftCodec : Codec α}
    {rightCodec : Codec β}
    {layout : Layout}
    (bundle : ColumnBundle layout)
    (leftWidthsAgree :
      leftCodec.width = layout.owners.length)
    (rightWidthsAgree :
      rightCodec.width = layout.owners.length)
    (leftIndex : Fin leftCodec.width)
    (rightIndex : Fin rightCodec.width)
    (indexEqual : leftIndex.val = rightIndex.val) :
    coordinateId leftCodec bundle leftWidthsAgree leftIndex =
      coordinateId rightCodec bundle rightWidthsAgree rightIndex := by
  unfold coordinateId
  congr

/-- **Physical proof decoder.** Exact raw-row satisfaction constructs one
semantic proof whose canonical codec decodes the actual proof operand
columns. No proof value or decoding equation is supplied by the caller. -/
theorem proof_decodes_of_rawRows
    (application : CanonicalApplication setup defaultRunning machine
      terminalRelations terminalChecks widths footprints)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor setup defaultRunning machine terminalRelations
      terminalChecks widths footprints application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (assignment : ColumnId → Field)
    (constantWire : assignment frame.one = 1)
    (satisfied :
      RawSatisfies
        (ConcreteNifsRawProgram.rawRows application.phase4.profile
          (ConcreteNifsCanonicalOperationalProfile.operational
            setup defaultRunning machine terminalRelations terminalChecks
              widths footprints application)
          frame)
        assignment) :
    ∃ proof : SelectedProof
        (ConcreteNifsPlain270Profile.Shape dimensions)
        TranscriptState publicRingColumns (publicFits dimensions)
        verifierRows,
      (proofOperand frame.operands).Decodes
        (FamilyFor setup defaultRunning machine terminalRelations
          terminalChecks widths footprints application)
        (.data .nifsProof) assignment proof := by
  let shape := ConcreteNifsPlain270Profile.Shape dimensions
  let polynomial := setup.system.constraintPolynomial
  let profile :=
    ConcreteNifsCanonicalOperationalProfile.operational
      setup defaultRunning machine terminalRelations terminalChecks
        widths footprints application
  let proofBundle := proofOperand frame.operands
  have canonicalitySatisfied :
      RawSatisfies
        (ConcreteNifsProofCanonicalityRows.rows
          application.phase4.profile profile frame)
        assignment :=
    canonicality_satisfied_of_rawRows
      setup defaultRunning machine terminalRelations terminalChecks
        widths footprints application frame assignment satisfied
  have canonicalWidthsAgree :
      (ConcreteNifsCanonicalProofCodec.proofCodec
          shape polynomial 0 publicRingColumns verifierRows
            (publicFits dimensions)).width =
        proofRef.port.layout.owners.length := by
    calc
      (ConcreteNifsCanonicalProofCodec.proofCodec
          shape polynomial 0 publicRingColumns verifierRows
            (publicFits dimensions)).width =
          ((FamilyFor setup defaultRunning machine terminalRelations
            terminalChecks widths footprints application).codecFor
              (.data .nifsProof)).width :=
        (congrArg Codec.width application.proofCodec_exact).symm
      _ = proofRef.port.layout.owners.length :=
        proof_widthsAgree frame
  have proofWidth :
      (proofBundle.values assignment).length =
        (ConcreteNifsCanonicalProofCodec.proofCodec
          shape polynomial 0 publicRingColumns verifierRows
            (publicFits dimensions)).width := by
    rw [ColumnBundle.values_length]
    exact canonicalWidthsAgree.symm
  have priorLanesZero :
      ∀ lane : Fin 8,
        (proofBundle.values assignment).getD
            (coordinatesPriorLaneView shape polynomial
              publicRingColumns verifierRows (publicFits dimensions)
              lane).index.val
            0 =
          0 := by
    intro lane
    change
      (proofBundle.values assignment).getD
          (proofPriorLaneView shape polynomial 0
            publicRingColumns verifierRows (publicFits dimensions)
            lane).index.val
          0 =
        0
    rw [
      (proofPriorLaneView shape polynomial 0
        publicRingColumns verifierRows (publicFits dimensions)
        lane).bundle_getD_eq_value
          proofBundle canonicalWidthsAgree assignment
    ]
    have selectedColumnEqual :
        ConcreteNifsProofCanonicalityRows.priorLaneColumn
            application.phase4.profile profile frame lane =
          ((proofPriorLaneView shape polynomial 0
              publicRingColumns verifierRows (publicFits dimensions)
              lane).column proofBundle canonicalWidthsAgree).column := by
      unfold ConcreteNifsProofCanonicalityRows.priorLaneColumn
      unfold FView.column
      exact
        coordinateId_eq_of_index_val proofBundle
          (proof_widthsAgree frame) canonicalWidthsAgree
          (profile.priorLane lane).index
          (proofPriorLaneView shape polynomial 0
            publicRingColumns verifierRows (publicFits dimensions) lane).index
          (operational_priorLane_index
            setup defaultRunning machine terminalRelations terminalChecks
              widths footprints application lane)
    change
      assignment
          ((proofPriorLaneView shape polynomial 0
              publicRingColumns verifierRows (publicFits dimensions)
              lane).column proofBundle canonicalWidthsAgree).column =
        0
    rw [← selectedColumnEqual]
    exact
      ConcreteNifsProofCanonicalityRows.rows_sound
        application.phase4.profile profile frame assignment constantWire
        canonicalitySatisfied lane
  rcases
      proofCodec_decode_exists_of_priorLaneCoordinatesZero
        shape polynomial publicRingColumns verifierRows
        (publicFits dimensions)
        (proofBundle.values assignment) proofWidth priorLanesZero with
    ⟨proof, proofDecoded⟩
  refine ⟨proof, ?_⟩
  unfold ColumnBundle.Decodes
  simpa only [FamilyFor, Poseidon23ApplicationProfile.family,
    TerminalEqualityProfile.family, DirectCalls.DirectProfile.family,
    Profile.family, DataCodecs.family,
    application.proofCodec_exact] using proofDecoded

/-- Exact raw-row satisfaction constructs all three semantic NIFS operands
and proves that the actual physical operand bundles decode to them. -/
theorem operands_decode_of_rawRows
    (application : CanonicalApplication setup defaultRunning machine
      terminalRelations terminalChecks widths footprints)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor setup defaultRunning machine terminalRelations
      terminalChecks widths footprints application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (assignment : ColumnId → Field)
    (constantWire : assignment frame.one = 1)
    (satisfied :
      RawSatisfies
        (ConcreteNifsRawProgram.rawRows application.phase4.profile
          (ConcreteNifsCanonicalOperationalProfile.operational
            setup defaultRunning machine terminalRelations terminalChecks
              widths footprints application)
          frame)
        assignment) :
    ∃ running :
        SelectedRunning
          (ConcreteNifsPlain270Profile.Shape dimensions)
          publicRingColumns (publicFits dimensions) verifierRows,
      ∃ fresh :
          SelectedFresh
            (ConcreteNifsPlain270Profile.Shape dimensions)
            publicRingColumns (publicFits dimensions) verifierRows,
        ∃ proof :
            SelectedProof
              (ConcreteNifsPlain270Profile.Shape dimensions)
              TranscriptState publicRingColumns (publicFits dimensions)
              verifierRows,
          frame.operands.Decodes
            (FamilyFor setup defaultRunning machine terminalRelations
              terminalChecks widths footprints application)
            assignment
            (.cons running (.cons fresh (.cons proof .nil))) := by
  let shape := ConcreteNifsPlain270Profile.Shape dimensions
  let runningBundle := runningOperand frame.operands
  let freshBundle := freshOperand frame.operands
  have runningWidth :
      (runningBundle.values assignment).length =
        (runningCodec shape publicRingColumns verifierRows
          (publicFits dimensions)).width := by
    rw [ColumnBundle.values_length]
    calc
      runningRef.port.layout.owners.length =
          ((FamilyFor setup defaultRunning machine terminalRelations
            terminalChecks widths footprints application).codecFor
              (.data .running)).width :=
        (running_widthsAgree frame).symm
      _ = (runningCodec shape publicRingColumns verifierRows
            (publicFits dimensions)).width :=
        congrArg Codec.width application.runningCodec_exact
  have freshWidth :
      (freshBundle.values assignment).length =
        (freshCodec shape publicRingColumns verifierRows
          (publicFits dimensions)).width := by
    rw [ColumnBundle.values_length]
    calc
      freshRef.port.layout.owners.length =
          ((FamilyFor setup defaultRunning machine terminalRelations
            terminalChecks widths footprints application).codecFor
              (.data .fresh)).width :=
        (fresh_widthsAgree frame).symm
      _ = (freshCodec shape publicRingColumns verifierRows
            (publicFits dimensions)).width :=
        congrArg Codec.width application.freshCodec_exact
  rcases
      Codec.decode_exists_of_exactWidthRecoverable
        (runningCodec_exactWidthRecoverable
          shape publicRingColumns verifierRows (publicFits dimensions))
        (runningBundle.values assignment) runningWidth with
    ⟨running, runningDecodedCanonical⟩
  rcases
      Codec.decode_exists_of_exactWidthRecoverable
        (freshCodec_exactWidthRecoverable
          shape publicRingColumns verifierRows (publicFits dimensions))
        (freshBundle.values assignment) freshWidth with
    ⟨fresh, freshDecodedCanonical⟩
  rcases
      proof_decodes_of_rawRows
        setup defaultRunning machine terminalRelations terminalChecks
          widths footprints application frame assignment constantWire
          satisfied with
    ⟨proof, proofDecoded⟩
  have runningDecoded :
      runningBundle.Decodes
        (FamilyFor setup defaultRunning machine terminalRelations
          terminalChecks widths footprints application)
        (.data .running) assignment running := by
    unfold ColumnBundle.Decodes
    simpa only [FamilyFor, Poseidon23ApplicationProfile.family,
      TerminalEqualityProfile.family, DirectCalls.DirectProfile.family,
      Profile.family, DataCodecs.family,
      application.runningCodec_exact] using runningDecodedCanonical
  have freshDecoded :
      freshBundle.Decodes
        (FamilyFor setup defaultRunning machine terminalRelations
          terminalChecks widths footprints application)
        (.data .fresh) assignment fresh := by
    unfold ColumnBundle.Decodes
    simpa only [FamilyFor, Poseidon23ApplicationProfile.family,
      TerminalEqualityProfile.family, DirectCalls.DirectProfile.family,
      Profile.family, DataCodecs.family,
      application.freshCodec_exact] using freshDecodedCanonical
  refine ⟨running, fresh, proof, ?_⟩
  exact
    (decodes_iff
      (FamilyFor setup defaultRunning machine terminalRelations
        terminalChecks widths footprints application)
      assignment frame.operands running fresh proof).2
      ⟨runningDecoded, freshDecoded, proofDecoded⟩

end SelectedApplication

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalProofPhysicalDecode
