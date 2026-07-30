import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOperationalProfile

/-!
Contract: construct one frame-static operational Split-NC row occurrence from
the selected `nifsVerify` call frame.

Every statement word, output word, prior-state lane, and prover-message
coefficient is either a verifier constant or a direct codec projection of one
decoded operand.  The five claimed-chain values (including the internal FE
row/lane boundary) are the first ten declared temporaries; the Poseidon2
transcript starts immediately afterward.
All numeric rows use the sole global call-column map.

This module owns only operational `PiCCS`.  Incoming running authority, the
fixed-active sampler, outgoing `PiDEC`, activation, and output materialization
are composed by the selected NIFS recipe.
-/

set_option autoImplicit false
set_option maxRecDepth 100000
set_option maxHeartbeats 1200000

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOperationalOccurrence

open Nightstream.HyperNova.Construction2.Paper
open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.Goldilocks.NumericRowBridge
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsCallFrame
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsCallColumnMap
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsGlobalColumnMap
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsCodecProjection
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsParameters
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOperationalProfile
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Implementation.R1CS.Canonical.KConcreteFixedPhaseBridge
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.KFixedPhaseSemanticOccurrence
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.TranscriptAuthority.BlockLane

private abbrev TranscriptState := Poseidon2Duplex.State
private abbrev KColumns :=
  Nightstream.Implementation.R1CS.ProjectionProgram.KColumns

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

private abbrev FamilyFor (application : Poseidon23ApplicationProfile Selected) :=
  application.family Selected

/-- Locate one projected base-field proof coordinate. -/
def proofFieldLocation
    (family : Family (typeSystem Selected))
    {context : Schema (typeSystem Selected)}
    {runningRef :
      Ref (typeSystem Selected) context (.data .running)}
    {freshRef :
      Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef :
      Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame :
      CallFrame (signature := signature Selected) family Call.nifsVerify
        (Refs.cons runningRef
          (Refs.cons freshRef (Refs.cons proofRef .nil))))
    {value :
      SelectedProof shape TranscriptState publicRingColumns publicFits
          verifierRows →
        Field}
    (view : FView (family.codecFor (.data .nifsProof)) value) :
    FLocation (columnMap frame)
      (view.column (proofOperand frame.operands)
        (proof_widthsAgree frame)) := by
  apply fLocation frame
  exact proofOperand_mem frame
    (view.column_mem (proofOperand frame.operands)
      (proof_widthsAgree frame))

/-- Direct proof-message pair in the global numeric namespace. -/
def proofColumns
    (family : Family (typeSystem Selected))
    {context : Schema (typeSystem Selected)}
    {runningRef :
      Ref (typeSystem Selected) context (.data .running)}
    {freshRef :
      Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef :
      Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame :
      CallFrame (signature := signature Selected) family Call.nifsVerify
        (Refs.cons runningRef
          (Refs.cons freshRef (Refs.cons proofRef .nil))))
    {value :
      SelectedProof shape TranscriptState publicRingColumns publicFits
          verifierRows →
        Nightstream.SuperNeo.Concrete.K}
    (view : KView (family.codecFor (.data .nifsProof)) value) : KColumns :=
  (ConcreteNifsOperationalFrame.proofLocation family frame view).numeric

/-- One verifier/computed extension temporary. -/
def temporaryK
    (family : Family (typeSystem Selected))
    {context : Schema (typeSystem Selected)}
    {runningRef :
      Ref (typeSystem Selected) context (.data .running)}
    {freshRef :
      Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef :
      Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame :
      CallFrame (signature := signature Selected) family Call.nifsVerify
        (Refs.cons runningRef
          (Refs.cons freshRef (Refs.cons proofRef .nil))))
    (index : Nat) : KColumns where
  c0 := temporarySource frame (2 * index)
  c1 := temporarySource frame (2 * index + 1)

/-- Numeric expression selected by one complete serialization source. -/
def sourceExpression
    (family : Family (typeSystem Selected))
    {context : Schema (typeSystem Selected)}
    {runningRef :
      Ref (typeSystem Selected) context (.data .running)}
    {freshRef :
      Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef :
      Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame :
      CallFrame (signature := signature Selected) family Call.nifsVerify
        (Refs.cons runningRef
          (Refs.cons freshRef (Refs.cons proofRef .nil))))
    (source :
      FieldSource
        (shape := shape) (publicRingColumns := publicRingColumns)
        (verifierRows := verifierRows) (publicFits := publicFits)
        (family.codecFor (.data .running))
        (family.codecFor (.data .fresh))
        (family.codecFor (.data .nifsProof))) : LinCombNormal.LinComb :=
  match source with
  | .constant value => KSplitNcTranscript.word value.val
  | .running _ view =>
      let typed :=
        view.column (runningOperand frame.operands)
          (running_widthsAgree frame)
      (fLocation frame typed
        (runningOperand_mem frame
          (view.column_mem (runningOperand frame.operands)
            (running_widthsAgree frame)))).carried
  | .fresh _ view =>
      let typed :=
        view.column (freshOperand frame.operands)
          (fresh_widthsAgree frame)
      (fLocation frame typed
        (freshOperand_mem frame
          (view.column_mem (freshOperand frame.operands)
            (fresh_widthsAgree frame)))).carried
  | .proof _ view =>
      (proofFieldLocation family frame view).carried

/-- Exact static transcript input constructed from one call frame. -/
def transcriptInput
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef :
      Ref (typeSystem Selected) context (.data .running)}
    {freshRef :
      Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef :
      Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame :
      CallFrame (signature := signature Selected)
        (FamilyFor application) Call.nifsVerify
        (Refs.cons runningRef
          (Refs.cons freshRef (Refs.cons proofRef .nil)))) :
    KSplitNcTranscript.Input
      (KSplitNcStaticInput.layoutInput profile.constraintPolynomial)
      Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.PiCcsDomains.production where
  transcriptBase := temporarySource frame 10
  priorLanes := fun lane =>
    (proofFieldLocation (FamilyFor application) frame
      (profile.priorLane lane)).carried
  priorAbsorbed := profile.priorAbsorbed
  statementFields :=
    profile.statementSources.map
      (sourceExpression (FamilyFor application) frame)
  outputFields :=
    profile.outputSources.map
      (sourceExpression (FamilyFor application) frame)
  fe := {
    initial := temporaryK (FamilyFor application) frame 0
    rowRounds := fun round => {
      coefficients :=
        List.ofFn fun slot =>
          proofColumns (FamilyFor application) frame
            (profile.messageViews.feRow round slot)
      coefficients_length := by simp
    }
    boundary := temporaryK (FamilyFor application) frame 1
    laneRounds := fun round => {
      coefficients :=
        List.ofFn fun slot =>
          proofColumns (FamilyFor application) frame
            (profile.messageViews.feLane round slot)
      coefficients_length := by simp
    }
    terminal := temporaryK (FamilyFor application) frame 2
  }
  nc := {
    initial := temporaryK (FamilyFor application) frame 3
    blockRounds := fun round => {
      coefficients :=
        List.ofFn fun slot =>
          proofColumns (FamilyFor application) frame
            (profile.messageViews.nc (Fin.castAdd _ round) slot)
      coefficients_length := by simp
    }
    laneRounds := fun round => {
      coefficients :=
        List.ofFn fun slot =>
          proofColumns (FamilyFor application) frame
            (profile.messageViews.nc (Fin.natAdd _ round) slot)
      coefficients_length := by simp
    }
    terminal := temporaryK (FamilyFor application) frame 4
  }

/-- Frame-static operational occurrence. -/
def input
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef :
      Ref (typeSystem Selected) context (.data .running)}
    {freshRef :
      Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef :
      Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame :
      CallFrame (signature := signature Selected)
        (FamilyFor application) Call.nifsVerify
        (Refs.cons runningRef
          (Refs.cons freshRef (Refs.cons proofRef .nil)))) :
    KSplitNcOperationalRows.Input
      (KSplitNcStaticInput.layoutInput profile.constraintPolynomial)
      Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.PiCcsDomains.production where
  transcript := transcriptInput application profile frame
  authority :=
    ConcreteNifsOperationalFrame.authorityColumns
      (FamilyFor application) frame profile.endpointViews

/-- Lean-owned raw operational rows before activation is applied. -/
def rows
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef :
      Ref (typeSystem Selected) context (.data .running)}
    {freshRef :
      Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef :
      Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame :
      CallFrame (signature := signature Selected)
        (FamilyFor application) Call.nifsVerify
        (Refs.cons runningRef
          (Refs.cons freshRef (Refs.cons proofRef .nil)))) :
    List Nightstream.Implementation.R1CS.Row :=
  KSplitNcOperationalRows.rows profile.constants
    (input application profile frame)

end SelectedFrame

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOperationalOccurrence
