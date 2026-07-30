import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsPiRlcActionAudit

/-!
Contract: exact numeric interval occupied by the selected public-parent
Phi81 action products.

The row audit classifies every non-visible dependency as a member of the
explicit product allocation.  This module retains the numeric source behind
each mapped product and proves that it lies in the exact interval beginning
at the caller-supplied action base.  It does not own the surrounding raw
program or activation suffix.

Emits constraints: none.
-/

set_option autoImplicit false
set_option maxRecDepth 100000
set_option maxHeartbeats 1200000

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsPiRlcActionConservation

open Nightstream.HyperNova.Construction2.Paper
open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsCallFrame
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsGlobalColumnMap
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsParameters
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOperationalProfile
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.Nifs
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc

private abbrev TranscriptState := Poseidon2Duplex.State

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

private theorem actionFrame_columns_before
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (productBase target : Nat)
    (values : Fin FixedActive.arity.total → Phi81RingAction.CarriedRing)
    (output : Phi81RingAction.CarriedRing)
    (targetLt :
      target <
        ConcreteNifsPiRlcActionRows.targetCount
          shape publicRingColumns verifierRows)
    (column : ColumnId)
    (member :
      column ∈ Phi81RingAction.productIds
        (ConcreteNifsPiRlcActionRows.actionFrame
          application profile frame productBase target values output)) :
    ∃ source,
      column = columnMap frame source
        ∧ source <
          productBase +
            (ConcreteNifsPiRlcActionRows.cost
              shape publicRingColumns verifierRows).auxiliaryColumns := by
  unfold Phi81RingAction.productIds at member
  rcases List.mem_flatMap.1 member with
    ⟨source, sourceMember, sourceRows⟩
  rcases List.mem_flatMap.1 sourceRows with
    ⟨left, leftMember, leftRows⟩
  rcases List.mem_map.1 leftRows with
    ⟨right, rightMember, rfl⟩
  have sourceLt := List.mem_range.1 sourceMember
  have leftLt := List.mem_range.1 leftMember
  have rightLt := List.mem_range.1 rightMember
  refine
    ⟨ConcreteNifsPiRlcActionRows.productSource
        productBase target source left right, ?_, ?_⟩
  · rfl
  · unfold ConcreteNifsPiRlcActionRows.productSource
      ConcreteNifsPiRlcActionRows.cost
      ConcreteNifsPiRlcActionRows.targetCount
      Phi81RingAction.productWidth Phi81RingAction.productOffset
    unfold ConcreteNifsPiRlcActionRows.targetCount at targetLt
    simp only [FixedActive.arity_total, ringDegree] at sourceLt leftLt rightLt targetLt ⊢
    omega

/-- Every explicitly allocated action product retains a numeric source
strictly before the end of the action interval. -/
theorem columns_before_end
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (productBase : Nat)
    (column : ColumnId)
    (member :
      column ∈ ConcreteNifsPiRlcActionRows.columns
        application profile frame productBase) :
    ∃ source,
      column = columnMap frame source
        ∧ source <
          productBase +
            (ConcreteNifsPiRlcActionRows.cost
              shape publicRingColumns verifierRows).auxiliaryColumns := by
  unfold ConcreteNifsPiRlcActionRows.columns at member
  rcases List.mem_flatMap.1 member with
    ⟨action, actionMember, columnMember⟩
  unfold ConcreteNifsPiRlcActionRows.frames at actionMember
  rcases List.mem_append.1 actionMember with firstThree | inHigh
  rcases List.mem_append.1 firstThree with firstTwo | inLow
  rcases List.mem_append.1 firstTwo with inCommitment | inPublic
  · rcases List.mem_ofFn.1 inCommitment with ⟨target, actionEq⟩
    rw [← actionEq] at columnMember
    exact actionFrame_columns_before application profile frame
      productBase target.val
      (ConcreteNifsPiRlcActionRows.commitmentValue
        application profile frame target)
      (ConcreteNifsPiRlcActionRows.commitmentOutput
        application profile frame target)
      (by
        unfold ConcreteNifsPiRlcActionRows.targetCount
        omega)
      column columnMember
  · rcases List.mem_ofFn.1 inPublic with ⟨block, actionEq⟩
    rw [← actionEq] at columnMember
    exact actionFrame_columns_before application profile frame
      productBase (verifierRows + block.val)
      (ConcreteNifsPiRlcActionRows.publicValue
        application profile frame block)
      (ConcreteNifsPiRlcActionRows.publicOutput
        application profile frame block)
      (by
        unfold ConcreteNifsPiRlcActionRows.targetCount
        omega)
      column columnMember
  · rcases List.mem_ofFn.1 inLow with ⟨matrix, actionEq⟩
    rw [← actionEq] at columnMember
    exact actionFrame_columns_before application profile frame
      productBase (verifierRows + publicRingColumns + matrix.val)
      (ConcreteNifsPiRlcActionRows.evaluationValueLow
        application profile frame matrix)
      (ConcreteNifsPiRlcActionRows.evaluationOutputLow
        application profile frame matrix)
      (by
        unfold ConcreteNifsPiRlcActionRows.targetCount
        omega)
      column columnMember
  · rcases List.mem_ofFn.1 inHigh with ⟨matrix, actionEq⟩
    rw [← actionEq] at columnMember
    exact actionFrame_columns_before application profile frame
      productBase
      (verifierRows + publicRingColumns + shape.matrixCount + matrix.val)
      (ConcreteNifsPiRlcActionRows.evaluationValueHigh
        application profile frame matrix)
      (ConcreteNifsPiRlcActionRows.evaluationOutputHigh
        application profile frame matrix)
      (by
        unfold ConcreteNifsPiRlcActionRows.targetCount
        omega)
      column columnMember

end SelectedFrame

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsPiRlcActionConservation
