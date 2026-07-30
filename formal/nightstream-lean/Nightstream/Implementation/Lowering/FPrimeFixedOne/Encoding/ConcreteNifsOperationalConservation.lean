import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCarrierFrame
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOperationalOccurrence
import Nightstream.Implementation.R1CS.Canonical.KFixedPhaseSumCheckSupport
import Nightstream.Implementation.R1CS.Canonical.KSplitNcTranscriptPlacement

/-!
Contract: exact source-prefix conservation for the selected operational
Split-NC transcript.

The transcript begins after the five claimed-chain values.  This module
proves that every externally supplied transcript expression is either a
verifier constant, an authoritative visible codec coordinate, or one of
those ten claimed-value coordinates.  It does not classify the rows emitted by
the fixed-phase endpoint programs.

Emits constraints: none.
-/

set_option autoImplicit false
set_option maxRecDepth 100000
set_option maxHeartbeats 1200000

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOperationalConservation

open Nightstream.HyperNova.Construction2.Paper
open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsCallFrame
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsCallColumnMap
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsGlobalColumnMap
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsParameters
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOperationalProfile
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.Implementation.R1CS.Canonical.SymbolicDuplexPlacement
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.TranscriptAuthority.BlockLane

private abbrev TranscriptState := Poseidon2Duplex.State
private abbrev KColumns :=
  Nightstream.Implementation.R1CS.ProjectionProgram.KColumns

private theorem singleton_prefix
    {base column : Nat} (below : column < base) :
    ValueInPrefix base [(column, 1)] := by
  intro mentionedColumn mentioned
  have same : mentionedColumn = column := by
    simpa [Mentions] using mentioned
  omega

private theorem word_prefix
    {base value : Nat} (positive : 0 < base) :
    ValueInPrefix base (KSplitNcTranscript.word value) := by
  intro column mentioned
  have same : column = 0 := by
    simpa [KSplitNcTranscript.word, Mentions] using mentioned
  omega

section SelectedFrame

variable {shape : SemanticShape}
variable {Digest AppState Witness Encoded RunningWitness FreshWitness : Type}
variable [DecidableEq AppState] [DecidableEq Encoded]
variable {publicRingColumns verifierRows : Nat}
variable {publicFits :
  ringDegree * publicRingColumns <= shape.carrierWidth}
variable {keys : Fin 1 →
  SelectedKey shape TranscriptState publicRingColumns publicFits verifierRows}
variable {defaultRunning :
  SelectedRunning shape publicRingColumns publicFits verifierRows}
variable {machine :
  Machine
    (SelectedKey shape TranscriptState publicRingColumns publicFits
      verifierRows)
    Digest AppState Witness
    (SelectedRunning shape publicRingColumns publicFits verifierRows)
    (SelectedFresh shape publicRingColumns publicFits verifierRows)
    Encoded 1}
variable {terminalRelations :
  TerminalRelations
    (SelectedKey shape TranscriptState publicRingColumns publicFits
      verifierRows)
    (SelectedRunning shape publicRingColumns publicFits verifierRows)
    RunningWitness
    (SelectedFresh shape publicRingColumns publicFits verifierRows)
    FreshWitness 1}
variable {terminalChecks :
  Nightstream.Protocol.FPrime.CanonicalTerminalVerifier.RelationChecks
    terminalRelations}
variable {widths : Widths} {footprints : Footprints}

local notation "Selected" =>
  ConcreteNifsParameters.selected keys defaultRunning machine
    terminalRelations terminalChecks widths footprints

private abbrev FamilyFor
    (application : Poseidon23ApplicationProfile Selected) :=
  application.family Selected

private abbrev FrameFor
    (application : Poseidon23ApplicationProfile Selected)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)} :=
  CallFrame (signature := signature Selected)
    (FamilyFor application) Call.nifsVerify
    (Refs.cons runningRef
      (Refs.cons freshRef (Refs.cons proofRef .nil)))

private theorem fLocation_prefix
    (application : Poseidon23ApplicationProfile Selected)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    {typed : PaperNifsCodecProjection.FColumnId}
    (location :
      FLocation (columnMap frame) typed)
    (below : location.numeric < temporaryBase frame) :
    ValueInPrefix (temporarySource frame 10) location.carried := by
  intro column mentioned
  have same : column = location.numeric := by
    simpa [FLocation.carried, Mentions] using mentioned
  unfold temporarySource
  omega

private theorem kColumns_fields_prefix
    (application : Poseidon23ApplicationProfile Selected)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (columns : KColumns)
    (below :
      columns.c0 < temporarySource frame 10 ∧
        columns.c1 < temporarySource frame 10) :
    ∀ value ∈
        KSplitNcTranscript.carriedFields
          (KFixedPhaseSemanticOccurrence.carried columns),
      ValueInPrefix (temporarySource frame 10) value := by
  intro value member
  simp only [KSplitNcTranscript.carriedFields,
    KFixedPhaseSemanticOccurrence.carried, List.mem_cons,
    List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl
  · exact singleton_prefix below.1
  · exact singleton_prefix below.2

private theorem proofFieldLocation_below
    (application : Poseidon23ApplicationProfile Selected)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    {value :
      SelectedProof shape TranscriptState publicRingColumns publicFits
          verifierRows →
        Field}
    (view :
      PaperNifsCodecProjection.FView
        ((FamilyFor application).codecFor (.data .nifsProof)) value) :
    (ConcreteNifsOperationalOccurrence.proofFieldLocation
        (FamilyFor application) frame view).numeric <
      temporaryBase frame := by
  unfold ConcreteNifsOperationalOccurrence.proofFieldLocation
  apply fLocation_numeric_lt_temporaryBase
  exact proofOperand_mem_visible frame
    (view.column_mem (proofOperand frame.operands)
      (proof_widthsAgree frame))

private theorem proofColumns_below
    (application : Poseidon23ApplicationProfile Selected)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    {value :
      SelectedProof shape TranscriptState publicRingColumns publicFits
          verifierRows →
        Nightstream.SuperNeo.Concrete.K}
    (view :
      PaperNifsCodecProjection.KView
        ((FamilyFor application).codecFor (.data .nifsProof)) value) :
    (ConcreteNifsOperationalOccurrence.proofColumns
        (FamilyFor application) frame view).c0 <
        temporaryBase frame
      ∧
      (ConcreteNifsOperationalOccurrence.proofColumns
        (FamilyFor application) frame view).c1 <
        temporaryBase frame := by
  unfold ConcreteNifsOperationalOccurrence.proofColumns
  change
    (ConcreteNifsCarrierFrame.proofKLocation
        (FamilyFor application) frame view).numeric.c0 <
        temporaryBase frame
      ∧
      (ConcreteNifsCarrierFrame.proofKLocation
        (FamilyFor application) frame view).numeric.c1 <
        temporaryBase frame
  exact ConcreteNifsCarrierFrame.proofKLocation_numeric_lt
    (FamilyFor application) frame view

private theorem sourceExpression_prefix
    (application : Poseidon23ApplicationProfile Selected)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (source :
      FieldSource
        (shape := shape) (publicRingColumns := publicRingColumns)
        (verifierRows := verifierRows) (publicFits := publicFits)
        ((FamilyFor application).codecFor (.data .running))
        ((FamilyFor application).codecFor (.data .fresh))
        ((FamilyFor application).codecFor (.data .nifsProof))) :
    ValueInPrefix (temporarySource frame 10)
      (ConcreteNifsOperationalOccurrence.sourceExpression
        (FamilyFor application) frame source) := by
  cases source with
  | constant value =>
      unfold ConcreteNifsOperationalOccurrence.sourceExpression
      apply word_prefix
      unfold temporarySource temporaryBase visibleIds
      simp
  | running value view =>
      unfold ConcreteNifsOperationalOccurrence.sourceExpression
      apply fLocation_prefix application frame
      change
        (ConcreteNifsCarrierFrame.runningFLocation
          (FamilyFor application) frame view).numeric <
          temporaryBase frame
      exact ConcreteNifsCarrierFrame.runningFLocation_numeric_lt
        (FamilyFor application) frame view
  | fresh value view =>
      unfold ConcreteNifsOperationalOccurrence.sourceExpression
      apply fLocation_prefix application frame
      change
        (ConcreteNifsCarrierFrame.freshFLocation
          (FamilyFor application) frame view).numeric <
          temporaryBase frame
      exact ConcreteNifsCarrierFrame.freshFLocation_numeric_lt
        (FamilyFor application) frame view
  | proof value view =>
      unfold ConcreteNifsOperationalOccurrence.sourceExpression
      exact fLocation_prefix application frame _
        (proofFieldLocation_below application frame view)

private theorem proofRound_fields_prefix
    (application : Poseidon23ApplicationProfile Selected)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    {degree : Nat}
    (views :
      Fin (degree + 1) →
        Σ value :
          SelectedProof shape TranscriptState publicRingColumns publicFits
              verifierRows →
            Nightstream.SuperNeo.Concrete.K,
          PaperNifsCodecProjection.KView
            ((FamilyFor application).codecFor (.data .nifsProof)) value)
    (value : LinCombNormal.LinComb)
    (member :
      value ∈
        KSplitNcTranscript.roundFields
          (show KFixedPhaseSemanticOccurrence.RoundColumns degree from {
            coefficients :=
              List.ofFn fun slot =>
                ConcreteNifsOperationalOccurrence.proofColumns
                  (FamilyFor application) frame (views slot).2
            coefficients_length := by simp
          })) :
    ValueInPrefix (temporarySource frame 10) value := by
  unfold KSplitNcTranscript.roundFields at member
  rcases List.mem_flatMap.1 member with
    ⟨columns, columnsMember, valueMember⟩
  rcases List.mem_ofFn.1 columnsMember with ⟨slot, rfl⟩
  have visibleBelow :=
    proofColumns_below application frame (views slot).2
  apply kColumns_fields_prefix application frame _ _ value valueMember
  constructor <;> unfold temporarySource <;> omega

/-- The selected call frame satisfies the complete external-source
precondition of the generic Split-NC transcript placement theorem. -/
theorem transcriptInput_inPrefix
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef)) :
    KSplitNcTranscriptPlacement.InputInPrefix
      (ConcreteNifsOperationalOccurrence.transcriptInput
        application profile frame) := by
  constructor
  · unfold ConcreteNifsOperationalOccurrence.transcriptInput
      temporarySource temporaryBase visibleIds
    simp
  · intro lane
    apply fLocation_prefix application frame
    exact proofFieldLocation_below application frame
      (profile.priorLane lane)
  · intro value member
    simp only [ConcreteNifsOperationalOccurrence.transcriptInput] at member
    rcases List.mem_map.1 member with ⟨source, _, rfl⟩
    exact sourceExpression_prefix application frame source
  · intro value member
    simp only [ConcreteNifsOperationalOccurrence.transcriptInput] at member
    rcases List.mem_map.1 member with ⟨source, _, rfl⟩
    exact sourceExpression_prefix application frame source
  · exact
      kColumns_fields_prefix application frame
        (ConcreteNifsOperationalOccurrence.temporaryK
          (FamilyFor application) frame 0)
        (by
          constructor <;>
            simp [ConcreteNifsOperationalOccurrence.temporaryK,
              temporarySource])
  · intro round roundMember value valueMember
    rcases List.mem_ofFn.1 roundMember with ⟨roundIndex, rfl⟩
    exact proofRound_fields_prefix application frame
      (fun slot =>
        ⟨_, profile.messageViews.feRow roundIndex slot⟩)
      value valueMember
  · intro round roundMember value valueMember
    rcases List.mem_ofFn.1 roundMember with ⟨roundIndex, rfl⟩
    exact proofRound_fields_prefix application frame
      (fun slot =>
        ⟨_, profile.messageViews.feLane roundIndex slot⟩)
      value valueMember
  · intro round roundMember value valueMember
    rcases List.mem_ofFn.1 roundMember with ⟨roundIndex, rfl⟩
    exact proofRound_fields_prefix application frame
      (fun slot =>
        ⟨_, profile.messageViews.nc (Fin.castAdd _ roundIndex) slot⟩)
      value valueMember
  · intro round roundMember value valueMember
    rcases List.mem_ofFn.1 roundMember with ⟨roundIndex, rfl⟩
    exact proofRound_fields_prefix application frame
      (fun slot =>
        ⟨_, profile.messageViews.nc (Fin.natAdd _ roundIndex) slot⟩)
      value valueMember

/-- Every transcript permutation row stays below the claimed-chain numeric
base.  This is an exact source/allocation statement, not merely a span
calculation: caller inputs are proved in the prefix and local columns are
classified by the duplex's physical allocation theorem. -/
theorem transcriptRows_below_numericBase
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (row : Nightstream.Implementation.R1CS.Row)
    (member :
      row ∈
        KSplitNcTranscript.rows profile.constants
          (ConcreteNifsOperationalOccurrence.transcriptInput
            application profile frame))
    (column : Nat)
    (mentioned :
      Mentions row.a column ∨ Mentions row.b column ∨
        Mentions row.c column) :
    column <
      KSplitNcOperationalRows.numericBase
        (ConcreteNifsOperationalOccurrence.input
          application profile frame) := by
  rcases
      KSplitNcTranscriptPlacement.rows_conservation
        profile.constants
        (ConcreteNifsOperationalOccurrence.transcriptInput
          application profile frame)
        (transcriptInput_inPrefix application profile frame)
        row member column mentioned with before | inAllocation
  · unfold KSplitNcOperationalRows.numericBase
      ConcreteNifsOperationalOccurrence.input
    exact Nat.lt_add_right _ before
  · have localBelow :=
      SymbolicDuplexPhysical.temporaryColumns_lt_end
        (ConcreteNifsOperationalOccurrence.transcriptInput
          application profile frame).transcriptBase
        (KSplitNcTranscript.replay
          (ConcreteNifsOperationalOccurrence.transcriptInput
            application profile frame)).afterOutput.entries.length
        column inAllocation
    simpa only [KSplitNcOperationalRows.numericBase,
      ConcreteNifsOperationalOccurrence.input] using localBelow

private theorem feRow_to_output_extends
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (input : KSplitNcTranscript.Input polynomialInput domains) :
    SymbolicDuplexSemantics.Extends
      (KSplitNcTranscript.feRowReplay input).builder
      (KSplitNcTranscript.outputBuilder input) :=
  (KSplitNcTranscriptSemantics.feLane_extends input).trans
    ((KSplitNcTranscriptSemantics.ncEntry_extends input).trans
      ((KSplitNcTranscriptSemantics.ncBlock_extends input).trans
        ((KSplitNcTranscriptSemantics.ncLane_extends input).trans
          (KSplitNcTranscriptSemantics.output_extends input))))

private theorem feLane_to_output_extends
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (input : KSplitNcTranscript.Input polynomialInput domains) :
    SymbolicDuplexSemantics.Extends
      (KSplitNcTranscript.feLaneReplay input).builder
      (KSplitNcTranscript.outputBuilder input) :=
  (KSplitNcTranscriptSemantics.ncEntry_extends input).trans
    ((KSplitNcTranscriptSemantics.ncBlock_extends input).trans
      ((KSplitNcTranscriptSemantics.ncLane_extends input).trans
        (KSplitNcTranscriptSemantics.output_extends input)))

private theorem ncBlock_to_output_extends
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (input : KSplitNcTranscript.Input polynomialInput domains) :
    SymbolicDuplexSemantics.Extends
      (KSplitNcTranscript.ncBlockReplay input).builder
      (KSplitNcTranscript.outputBuilder input) :=
  (KSplitNcTranscriptSemantics.ncLane_extends input).trans
    (KSplitNcTranscriptSemantics.output_extends input)

theorem coreColumns_below_numericBase
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef)) :
    let transcript :=
      ConcreteNifsOperationalOccurrence.transcriptInput
        application profile frame
    KSplitNcTranscriptPlacement.CoreColumnsBelow
      (KSplitNcTranscript.coreReplay transcript)
      (KSplitNcOperationalRows.numericBase
        (ConcreteNifsOperationalOccurrence.input
          application profile frame)) := by
  let transcript :=
    ConcreteNifsOperationalOccurrence.transcriptInput
      application profile frame
  have placed :=
    (KSplitNcTranscriptPlacement.outputBuilder_invariant transcript
      (transcriptInput_inPrefix application profile frame)).1
  have bounds :=
    KSplitNcTranscriptPlacement.deriveCore_columns_below_of_extends
      transcript.transcriptBase shape
      Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.PiCcsDomains.production
      (KSplitNcTranscript.statementBuilder transcript)
      (KSplitNcTranscript.outputBuilder transcript)
      placed (KSplitNcTranscriptSemantics.core_to_output_extends transcript)
  simpa only [transcript, KSplitNcTranscript.coreReplay,
    KSplitNcOperationalRows.numericBase,
    ConcreteNifsOperationalOccurrence.input,
    KSplitNcTranscript.replay] using bounds

theorem temporaryK_below_transcriptBase
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (index : Nat) (bounded : index < 5) :
    let columns :=
      ConcreteNifsOperationalOccurrence.temporaryK
        (FamilyFor application) frame index
    columns.c0 <
        (ConcreteNifsOperationalOccurrence.transcriptInput
          application profile frame).transcriptBase ∧
    columns.c1 <
        (ConcreteNifsOperationalOccurrence.transcriptInput
          application profile frame).transcriptBase := by
  dsimp only [ConcreteNifsOperationalOccurrence.temporaryK]
  constructor
  · change temporarySource frame (2 * index) < temporarySource frame 10
    unfold temporarySource
    omega
  · change temporarySource frame (2 * index + 1) < temporarySource frame 10
    unfold temporarySource
    omega

theorem temporaryK_below_numericBase
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (index : Nat) (bounded : index < 5) :
    let columns :=
      ConcreteNifsOperationalOccurrence.temporaryK
        (FamilyFor application) frame index
    columns.c0 <
        KSplitNcOperationalRows.numericBase
          (ConcreteNifsOperationalOccurrence.input
            application profile frame) ∧
    columns.c1 <
        KSplitNcOperationalRows.numericBase
          (ConcreteNifsOperationalOccurrence.input
            application profile frame) := by
  have before :=
    temporaryK_below_transcriptBase application profile frame index bounded
  constructor
  · unfold KSplitNcOperationalRows.numericBase
      ConcreteNifsOperationalOccurrence.input
    exact Nat.lt_add_right _ before.1
  · unfold KSplitNcOperationalRows.numericBase
      ConcreteNifsOperationalOccurrence.input
    exact Nat.lt_add_right _ before.2

private theorem proofColumns_below_numericBase
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    {value :
      SelectedProof shape TranscriptState publicRingColumns publicFits
          verifierRows →
        Nightstream.SuperNeo.Concrete.K}
    (view :
      PaperNifsCodecProjection.KView
        ((FamilyFor application).codecFor (.data .nifsProof)) value) :
    let columns :=
      ConcreteNifsOperationalOccurrence.proofColumns
        (FamilyFor application) frame view
    columns.c0 <
        KSplitNcOperationalRows.numericBase
          (ConcreteNifsOperationalOccurrence.input
            application profile frame) ∧
      columns.c1 <
        KSplitNcOperationalRows.numericBase
          (ConcreteNifsOperationalOccurrence.input
            application profile frame) := by
  have visible := proofColumns_below application frame view
  have lowBefore :
      (ConcreteNifsOperationalOccurrence.proofColumns
          (FamilyFor application) frame view).c0 <
        (ConcreteNifsOperationalOccurrence.transcriptInput
          application profile frame).transcriptBase := by
    change
      (ConcreteNifsOperationalOccurrence.proofColumns
          (FamilyFor application) frame view).c0 <
        temporarySource frame 10
    unfold temporarySource
    omega
  have highBefore :
      (ConcreteNifsOperationalOccurrence.proofColumns
          (FamilyFor application) frame view).c1 <
        (ConcreteNifsOperationalOccurrence.transcriptInput
          application profile frame).transcriptBase := by
    change
      (ConcreteNifsOperationalOccurrence.proofColumns
          (FamilyFor application) frame view).c1 <
        temporarySource frame 10
    unfold temporarySource
    omega
  constructor
  · unfold KSplitNcOperationalRows.numericBase
      ConcreteNifsOperationalOccurrence.input
    exact Nat.lt_add_right _ lowBefore
  · unfold KSplitNcOperationalRows.numericBase
      ConcreteNifsOperationalOccurrence.input
    exact Nat.lt_add_right _ highBefore

theorem feRowChallenges_below_numericBase
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef)) :
    ∀ columns ∈
        (KSplitNcTranscript.feRowReplay
          (ConcreteNifsOperationalOccurrence.transcriptInput
            application profile frame)).challenges,
      columns.c0 <
          KSplitNcOperationalRows.numericBase
            (ConcreteNifsOperationalOccurrence.input
              application profile frame) ∧
        columns.c1 <
          KSplitNcOperationalRows.numericBase
            (ConcreteNifsOperationalOccurrence.input
              application profile frame) := by
  let input :=
    ConcreteNifsOperationalOccurrence.transcriptInput
      application profile frame
  have placed :=
    (KSplitNcTranscriptPlacement.outputBuilder_invariant input
      (transcriptInput_inPrefix application profile frame)).1
  have bounds :=
    KSplitNcTranscriptPlacement.replayRounds_challenges_below_of_extends
      input.transcriptBase .feRound
      (List.ofFn input.fe.rowRounds)
      (KSplitNcTranscript.feEntryBuilder input)
      (KSplitNcTranscript.outputBuilder input)
      placed (feRow_to_output_extends input)
  simpa only [input, KSplitNcTranscript.feRowReplay,
    KSplitNcOperationalRows.numericBase,
    ConcreteNifsOperationalOccurrence.input,
    KSplitNcTranscript.replay] using bounds

theorem feLaneChallenges_below_numericBase
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef)) :
    ∀ columns ∈
        (KSplitNcTranscript.feLaneReplay
          (ConcreteNifsOperationalOccurrence.transcriptInput
            application profile frame)).challenges,
      columns.c0 <
          KSplitNcOperationalRows.numericBase
            (ConcreteNifsOperationalOccurrence.input
              application profile frame) ∧
        columns.c1 <
          KSplitNcOperationalRows.numericBase
            (ConcreteNifsOperationalOccurrence.input
              application profile frame) := by
  let input :=
    ConcreteNifsOperationalOccurrence.transcriptInput
      application profile frame
  have placed :=
    (KSplitNcTranscriptPlacement.outputBuilder_invariant input
      (transcriptInput_inPrefix application profile frame)).1
  have bounds :=
    KSplitNcTranscriptPlacement.replayRounds_challenges_below_of_extends
      input.transcriptBase .feRound
      (List.ofFn input.fe.laneRounds)
      (KSplitNcTranscript.feRowReplay input).builder
      (KSplitNcTranscript.outputBuilder input)
      placed (feLane_to_output_extends input)
  simpa only [input, KSplitNcTranscript.feLaneReplay,
    KSplitNcOperationalRows.numericBase,
    ConcreteNifsOperationalOccurrence.input,
    KSplitNcTranscript.replay] using bounds

theorem ncBlockChallenges_below_numericBase
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef)) :
    ∀ columns ∈
        (KSplitNcTranscript.ncBlockReplay
          (ConcreteNifsOperationalOccurrence.transcriptInput
            application profile frame)).challenges,
      columns.c0 <
          KSplitNcOperationalRows.numericBase
            (ConcreteNifsOperationalOccurrence.input
              application profile frame) ∧
        columns.c1 <
          KSplitNcOperationalRows.numericBase
            (ConcreteNifsOperationalOccurrence.input
              application profile frame) := by
  let input :=
    ConcreteNifsOperationalOccurrence.transcriptInput
      application profile frame
  have placed :=
    (KSplitNcTranscriptPlacement.outputBuilder_invariant input
      (transcriptInput_inPrefix application profile frame)).1
  have bounds :=
    KSplitNcTranscriptPlacement.replayRounds_challenges_below_of_extends
      input.transcriptBase .ncRound
      (List.ofFn input.nc.blockRounds)
      (KSplitNcTranscript.ncEntryBuilder input)
      (KSplitNcTranscript.outputBuilder input)
      placed (ncBlock_to_output_extends input)
  simpa only [input, KSplitNcTranscript.ncBlockReplay,
    KSplitNcOperationalRows.numericBase,
    ConcreteNifsOperationalOccurrence.input,
    KSplitNcTranscript.replay] using bounds

theorem ncLaneChallenges_below_numericBase
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef)) :
    ∀ columns ∈
        (KSplitNcTranscript.ncLaneReplay
          (ConcreteNifsOperationalOccurrence.transcriptInput
            application profile frame)).challenges,
      columns.c0 <
          KSplitNcOperationalRows.numericBase
            (ConcreteNifsOperationalOccurrence.input
              application profile frame) ∧
        columns.c1 <
          KSplitNcOperationalRows.numericBase
            (ConcreteNifsOperationalOccurrence.input
              application profile frame) := by
  let input :=
    ConcreteNifsOperationalOccurrence.transcriptInput
      application profile frame
  have placed :=
    (KSplitNcTranscriptPlacement.outputBuilder_invariant input
      (transcriptInput_inPrefix application profile frame)).1
  have bounds :=
    KSplitNcTranscriptPlacement.replayRounds_challenges_below_of_extends
      input.transcriptBase .ncRound
      (List.ofFn input.nc.laneRounds)
      (KSplitNcTranscript.ncBlockReplay input).builder
      (KSplitNcTranscript.outputBuilder input)
      placed (KSplitNcTranscriptSemantics.output_extends input)
  simpa only [input, KSplitNcTranscript.ncLaneReplay,
    KSplitNcOperationalRows.numericBase,
    ConcreteNifsOperationalOccurrence.input,
    KSplitNcTranscript.replay] using bounds

private theorem feRowSource_below_numericBase
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef)) :
    let input :=
      ConcreteNifsOperationalOccurrence.input application profile frame
    (KSplitNcTranscript.numericColumns input.transcript).fe.rowSource.BelowBase
      (KSplitNcOperationalRows.numericBase input) := by
  dsimp only [KSplitNcFeRows.Columns.rowSource]
  constructor
  · exact (temporaryK_below_numericBase application profile frame 0
      (by omega)).1
  · exact (temporaryK_below_numericBase application profile frame 0
      (by omega)).2
  · intro round roundMember columns columnsMember
    simp only [KSplitNcTranscript.numericColumns,
      ConcreteNifsOperationalOccurrence.input,
      ConcreteNifsOperationalOccurrence.transcriptInput] at roundMember
    rcases List.mem_ofFn.1 roundMember with ⟨roundIndex, rfl⟩
    rcases List.mem_ofFn.1 columnsMember with ⟨slot, rfl⟩
    exact proofColumns_below_numericBase application profile frame
      (profile.messageViews.feRow roundIndex slot)
  · exact feRowChallenges_below_numericBase application profile frame
  · exact (temporaryK_below_numericBase application profile frame 1
      (by omega)).1
  · exact (temporaryK_below_numericBase application profile frame 1
      (by omega)).2

private theorem feLaneSource_below_numericBase
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef)) :
    let input :=
      ConcreteNifsOperationalOccurrence.input application profile frame
    (KSplitNcTranscript.numericColumns input.transcript).fe.laneSource.BelowBase
      (KSplitNcOperationalRows.numericBase input) := by
  dsimp only [KSplitNcFeRows.Columns.laneSource]
  constructor
  · exact (temporaryK_below_numericBase application profile frame 1
      (by omega)).1
  · exact (temporaryK_below_numericBase application profile frame 1
      (by omega)).2
  · intro round roundMember columns columnsMember
    simp only [KSplitNcTranscript.numericColumns,
      ConcreteNifsOperationalOccurrence.input,
      ConcreteNifsOperationalOccurrence.transcriptInput] at roundMember
    rcases List.mem_ofFn.1 roundMember with ⟨roundIndex, rfl⟩
    rcases List.mem_ofFn.1 columnsMember with ⟨slot, rfl⟩
    exact proofColumns_below_numericBase application profile frame
      (profile.messageViews.feLane roundIndex slot)
  · exact feLaneChallenges_below_numericBase application profile frame
  · exact (temporaryK_below_numericBase application profile frame 2
      (by omega)).1
  · exact (temporaryK_below_numericBase application profile frame 2
      (by omega)).2

private theorem ncSource_below_numericBase
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef)) :
    let input :=
      ConcreteNifsOperationalOccurrence.input application profile frame
    (KSplitNcTranscript.numericColumns input.transcript).nc.BelowBase
      (KSplitNcOperationalRows.numericBase input) := by
  constructor
  · exact (temporaryK_below_numericBase application profile frame 3
      (by omega)).1
  · exact (temporaryK_below_numericBase application profile frame 3
      (by omega)).2
  · intro round roundMember columns columnsMember
    simp only [KSplitNcTranscript.numericColumns,
      ConcreteNifsOperationalOccurrence.input,
      ConcreteNifsOperationalOccurrence.transcriptInput,
      List.mem_append] at roundMember
    rcases roundMember with blockMember | laneMember
    · rcases List.mem_ofFn.1 blockMember with ⟨roundIndex, rfl⟩
      rcases List.mem_ofFn.1 columnsMember with ⟨slot, rfl⟩
      exact proofColumns_below_numericBase application profile frame
        (profile.messageViews.nc
          (Fin.castAdd
            Folding.Nifs.ConcretePhi81.PiCcsDomains.production.nc.laneVariables
            roundIndex)
          slot)
    · rcases List.mem_ofFn.1 laneMember with ⟨roundIndex, rfl⟩
      rcases List.mem_ofFn.1 columnsMember with ⟨slot, rfl⟩
      exact proofColumns_below_numericBase application profile frame
        (profile.messageViews.nc
          (Fin.natAdd
            Folding.Nifs.ConcretePhi81.PiCcsDomains.production.nc.blockVariables
            roundIndex)
          slot)
  · intro columns member
    simp only [KSplitNcTranscript.numericColumns,
      ConcreteNifsOperationalOccurrence.input, KSplitNcTranscript.replay,
      List.mem_append] at member
    rcases member with blockMember | laneMember
    · exact ncBlockChallenges_below_numericBase application profile frame
        columns blockMember
    · exact ncLaneChallenges_below_numericBase application profile frame
        columns laneMember
  · exact (temporaryK_below_numericBase application profile frame 4
      (by omega)).1
  · exact (temporaryK_below_numericBase application profile frame 4
      (by omega)).2

/-- The FE row-phase sources consumed by the honest claimed-chain witness
all precede the selected numeric allocation.  This is the public
honest-completeness view of the placement result above; it carries no chain
equation or acceptance proposition. -/
theorem feRowSource_below
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef)) :
    let input :=
      ConcreteNifsOperationalOccurrence.input application profile frame
    (KSplitNcTranscript.numericColumns input.transcript).fe.rowSource.BelowBase
      (KSplitNcOperationalRows.numericBase input) :=
  feRowSource_below_numericBase application profile frame

/-- The FE lane-phase sources consumed by the honest claimed-chain witness
all precede the selected numeric allocation. -/
theorem feLaneSource_below
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef)) :
    let input :=
      ConcreteNifsOperationalOccurrence.input application profile frame
    (KSplitNcTranscript.numericColumns input.transcript).fe.laneSource.BelowBase
      (KSplitNcOperationalRows.numericBase input) :=
  feLaneSource_below_numericBase application profile frame

/-- The block×lane NC sources consumed by the honest claimed-chain witness
all precede the selected numeric allocation. -/
theorem ncSource_below
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef)) :
    let input :=
      ConcreteNifsOperationalOccurrence.input application profile frame
    (KSplitNcTranscript.numericColumns input.transcript).nc.BelowBase
      (KSplitNcOperationalRows.numericBase input) :=
  ncSource_below_numericBase application profile frame

private theorem source_below_mono
    {degree base nextBase : Nat}
    {source :
      KFixedPhaseSemanticOccurrence.SourceColumns degree}
    (placed : source.BelowBase base)
    (ordered : base ≤ nextBase) :
    source.BelowBase nextBase where
  currentLow := Nat.lt_of_lt_of_le placed.currentLow ordered
  currentHigh := Nat.lt_of_lt_of_le placed.currentHigh ordered
  rounds := by
    intro round roundMember columns columnsMember
    exact
      ⟨Nat.lt_of_lt_of_le
          (placed.rounds round roundMember columns columnsMember).1 ordered,
        Nat.lt_of_lt_of_le
          (placed.rounds round roundMember columns columnsMember).2 ordered⟩
  challenges := by
    intro columns member
    exact
      ⟨Nat.lt_of_lt_of_le (placed.challenges columns member).1 ordered,
        Nat.lt_of_lt_of_le (placed.challenges columns member).2 ordered⟩
  terminalLow := Nat.lt_of_lt_of_le placed.terminalLow ordered
  terminalHigh := Nat.lt_of_lt_of_le placed.terminalHigh ordered

private theorem source_chainRows_columns_below_end
    {degree base : Nat}
    (source : KFixedPhaseSemanticOccurrence.SourceColumns degree)
    (basePositive : 0 < base)
    (placed : source.BelowBase base)
    (row : Nightstream.Implementation.R1CS.Row)
    (member :
      row ∈
        KFixedPhaseSumCheck.chainRows
          (KFixedPhaseSemanticOccurrence.carried source.current)
          source.rowRounds source.rowChallenges
          (KFixedPhaseSemanticOccurrence.carried source.terminal) base)
    (column : Nat)
    (mentioned :
      Mentions row.a column ∨ Mentions row.b column ∨ Mentions row.c column) :
    column < base + source.rounds.length * (3 * degree) := by
  let physical :=
    source.physical base .prelude .prelude 0 basePositive placed
  simpa only [KFixedPhaseSemanticOccurrence.SourceColumns.rowRounds,
    List.length_map] using
    KFixedPhaseSumCheckSupport.chainRows_columns_below_end
    (KFixedPhaseSemanticOccurrence.carried source.current)
    source.rowRounds source.rowChallenges
    (KFixedPhaseSemanticOccurrence.carried source.terminal)
    base physical.basePositive physical.currentBelow
    physical.roundsBelow physical.challengesBelow physical.terminalBelow
    physical.sameLength row member column mentioned

/-- Every numeric FE/NC claimed-chain row stays below the first endpoint
frame. This binds its proof/message reads to the selected call frame and its
challenge reads to the exact transcript replay proved above. -/
theorem numericRows_below_endpointBase
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (row : Nightstream.Implementation.R1CS.Row)
    (member :
      row ∈ KSplitNcOperationalRows.numericRows
        (ConcreteNifsOperationalOccurrence.input application profile frame))
    (column : Nat)
    (mentioned :
      Mentions row.a column ∨ Mentions row.b column ∨ Mentions row.c column) :
    column <
      KSplitNcOperationalRows.endpointBase
        (ConcreteNifsOperationalOccurrence.input application profile frame) := by
  let input :=
    ConcreteNifsOperationalOccurrence.input application profile frame
  let columns := KSplitNcTranscript.numericColumns input.transcript
  change
    column <
      KSplitNcOperationalRows.numericBase input +
        (KSplitNcBlockLaneRows.cost columns).auxiliaryColumns
  have numericPositive : 0 < KSplitNcOperationalRows.numericBase input := by
    have transcriptPositive : 0 < input.transcript.transcriptBase := by
      dsimp only [input, ConcreteNifsOperationalOccurrence.input,
        ConcreteNifsOperationalOccurrence.transcriptInput]
      unfold temporarySource
      omega
    unfold KSplitNcOperationalRows.numericBase
    omega
  have rowPlaced := feRowSource_below_numericBase application profile frame
  have lanePlacedAtNumeric :=
    feLaneSource_below_numericBase application profile frame
  have ncPlacedAtNumeric :=
    ncSource_below_numericBase application profile frame
  have laneOrdered :
      KSplitNcOperationalRows.numericBase input ≤
        KSplitNcFeRows.laneBase columns.fe
          (KSplitNcOperationalRows.numericBase input) := by
    unfold KSplitNcFeRows.laneBase
    omega
  have lanePlaced :=
    source_below_mono lanePlacedAtNumeric laneOrdered
  have lanePositive :
      0 <
        KSplitNcFeRows.laneBase columns.fe
          (KSplitNcOperationalRows.numericBase input) := by
    omega
  have ncOrdered :
      KSplitNcOperationalRows.numericBase input ≤
        KSplitNcBlockLaneRows.ncBase columns
          (KSplitNcOperationalRows.numericBase input) := by
    unfold KSplitNcBlockLaneRows.ncBase
    omega
  have ncPlaced := source_below_mono ncPlacedAtNumeric ncOrdered
  have ncPositive :
      0 <
        KSplitNcBlockLaneRows.ncBase columns
          (KSplitNcOperationalRows.numericBase input) := by
    omega
  change row ∈ KSplitNcBlockLaneRows.rows columns
    (KSplitNcOperationalRows.numericBase input) at member
  simp only [KSplitNcBlockLaneRows.rows, KSplitNcFeRows.rows,
    KSplitNcNcRows.rows, List.mem_append] at member
  rcases member with (rowMember | laneMember) | ncMember
  · have below := source_chainRows_columns_below_end
      columns.fe.rowSource numericPositive rowPlaced
      row rowMember column mentioned
    simp only [KSplitNcBlockLaneRows.cost, Cost.add_auxiliaryColumns,
      KSplitNcFeRows.cost, KSplitNcNcRows.cost,
      KFixedPhaseSumCheck.chainCost,
      KSplitNcFeRows.Columns.rowSource] at below ⊢
    omega
  · have below := source_chainRows_columns_below_end
      columns.fe.laneSource lanePositive lanePlaced
      row laneMember column mentioned
    unfold KSplitNcFeRows.laneBase at below
    simp only [KSplitNcBlockLaneRows.cost, Cost.add_auxiliaryColumns,
      KSplitNcFeRows.cost, KSplitNcNcRows.cost,
      KFixedPhaseSumCheck.chainCost,
      KSplitNcFeRows.Columns.laneSource] at below ⊢
    omega
  · have below := source_chainRows_columns_below_end
      columns.nc ncPositive ncPlaced
      row ncMember column mentioned
    unfold KSplitNcBlockLaneRows.ncBase at below
    simp only [KSplitNcBlockLaneRows.cost, Cost.add_auxiliaryColumns,
      KSplitNcFeRows.cost, KSplitNcNcRows.cost,
      KFixedPhaseSumCheck.chainCost] at below ⊢
    omega

end SelectedFrame

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOperationalConservation
