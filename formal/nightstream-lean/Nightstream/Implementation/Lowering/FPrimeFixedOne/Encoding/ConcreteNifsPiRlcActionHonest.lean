import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsAllocationCoverage
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsPiRlcActionConservation
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsPiRlcActionSemantics
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.Phi81RingActionHonest

/-!
Contract: honest completion of the selected fixed-active Phi81 action slice.

The product witness is written once across the complete action allocation.
Its duplicate-freedom is derived from the call frame's actual temporary
receipt and the Lean-owned dense product layout.  Every visible input and
output coordinate is preserved.

This module does not own activation, the surrounding selected-NIFS rows,
paper-event transport, Rust, or generated artifacts.
-/

set_option autoImplicit false
set_option maxRecDepth 100000
set_option maxHeartbeats 2000000

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsPiRlcActionHonest

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
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.Phi81RingAction
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

private def actionPrefix
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef)) : Nat :=
  10 +
    (ConcreteNifsOperationalSampler.cost
      application profile frame).auxiliaryColumns

/-- The action allocation, independently enumerated as its dense suffix of
the selected call's temporary receipt. -/
def denseColumns
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef)) :
    List ColumnId :=
  (List.range
    (ConcreteNifsPiRlcActionRows.cost
      shape publicRingColumns verifierRows).auxiliaryColumns).map
    (fun offset =>
      columnMap frame
        (ConcreteNifsRawProgram.actionBase
          application profile frame + offset))

private theorem nodup_map_of_injective_on
    {α β : Type} [DecidableEq β]
    (items : List α)
    (map : α → β)
    (nodup : items.Nodup)
    (injective :
      ∀ left ∈ items, ∀ right ∈ items,
        map left = map right → left = right) :
    (items.map map).Nodup := by
  induction items with
  | nil =>
      simp
  | cons head tail inductionHypothesis =>
      rw [List.nodup_cons] at nodup
      simp only [List.map_cons, List.nodup_cons]
      constructor
      · intro member
        rcases List.mem_map.1 member with
          ⟨other, otherMember, equal⟩
        have same :=
          injective other (by simp [otherMember]) head (by simp) equal
        subst other
        exact nodup.1 otherMember
      · exact inductionHypothesis nodup.2
          (fun left leftMember right rightMember equal =>
            injective left (by simp [leftMember])
              right (by simp [rightMember]) equal)

private theorem perm_of_nodup_subset_length_eq
    {α : Type} [BEq α] [LawfulBEq α]
    {left right : List α}
    (leftNodup : left.Nodup)
    (subset : ∀ value ∈ left, value ∈ right)
    (sameLength : left.length = right.length) :
    left.Perm right := by
  induction left generalizing right with
  | nil =>
      have rightNil : right = [] := by
        cases right with
        | nil => rfl
        | cons head tail =>
            simp at sameLength
      subst right
      exact .refl []
  | cons head tail inductionHypothesis =>
      have parts := List.nodup_cons.1 leftNodup
      have headMember : head ∈ right :=
        subset head (by simp)
      have tailSubset :
          ∀ value ∈ tail, value ∈ right.erase head := by
        intro value valueMember
        rw [List.mem_erase_of_ne]
        · exact subset value (by simp [valueMember])
        · exact fun equal => parts.1 (equal ▸ valueMember)
      have erasedLength :
          tail.length = (right.erase head).length := by
        rw [List.length_erase_of_mem headMember]
        simpa using congrArg (fun length => length - 1) sameLength
      have tailPerm :=
        inductionHypothesis parts.2 tailSubset erasedLength
      exact (tailPerm.cons head).trans
        (List.perm_cons_erase headMember).symm

private theorem action_index_lt
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (fits : ConcreteNifsRawProgram.FrameFits application profile frame)
    (offset : Nat)
    (offsetLt :
      offset <
        (ConcreteNifsPiRlcActionRows.cost
          shape publicRingColumns verifierRows).auxiliaryColumns) :
    actionPrefix application profile frame + offset <
      frame.temporaries.ids.length := by
  unfold ConcreteNifsRawProgram.FrameFits at fits
  unfold ConcreteNifsRawProgram.allocationWidth at fits
  unfold actionPrefix
  omega

private theorem denseColumn_getElem?
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (fits : ConcreteNifsRawProgram.FrameFits application profile frame)
    (offset : Nat)
    (offsetLt :
      offset <
        (ConcreteNifsPiRlcActionRows.cost
          shape publicRingColumns verifierRows).auxiliaryColumns) :
    frame.temporaries.ids[
        actionPrefix application profile frame + offset]? =
      some
        (columnMap frame
          (ConcreteNifsRawProgram.actionBase
            application profile frame + offset)) := by
  have indexLt :=
    action_index_lt application profile frame fits offset offsetLt
  have sourceExact :
      ConcreteNifsRawProgram.actionBase application profile frame + offset =
        temporarySource frame
          (actionPrefix application profile frame + offset) := by
    rw [ConcreteNifsAllocationCoverage.actionBase_eq_temporarySource
      application profile frame]
    unfold temporarySource actionPrefix
    omega
  rw [sourceExact, columnMap_temporarySource frame indexLt,
    List.getElem?_eq_getElem indexLt]

/-- The independently enumerated dense action interval is duplicate-free
because it selects distinct positions of the actual call-frame temporary
receipt. -/
theorem denseColumns_nodup
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (fits : ConcreteNifsRawProgram.FrameFits application profile frame) :
    (denseColumns application profile frame).Nodup := by
  unfold denseColumns
  apply nodup_map_of_injective_on _ _ List.nodup_range
  intro left leftMember right rightMember equal
  have leftLt := List.mem_range.1 leftMember
  have rightLt := List.mem_range.1 rightMember
  have leftIndexLt :=
    action_index_lt application profile frame fits left leftLt
  have optionEqual :
      frame.temporaries.ids[
          actionPrefix application profile frame + left]? =
        frame.temporaries.ids[
          actionPrefix application profile frame + right]? := by
    rw [denseColumn_getElem? application profile frame fits left leftLt,
      denseColumn_getElem? application profile frame fits right rightLt,
      equal]
  have temporaryNodup :
      frame.temporaries.ids.Nodup :=
    (List.nodup_append.1 frame.allocationsNodup).2.1
  have indexEqual :=
    (List.getElem?_inj leftIndexLt temporaryNodup).1 optionEqual
  omega

/-- The emitter's product allocation is a permutation of the independently
derived dense temporary interval. -/
theorem denseColumns_perm_columns
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (fits : ConcreteNifsRawProgram.FrameFits application profile frame) :
    (denseColumns application profile frame).Perm
      (ConcreteNifsPiRlcActionRows.columns application profile frame
        (ConcreteNifsRawProgram.actionBase application profile frame)) := by
  apply perm_of_nodup_subset_length_eq
    (denseColumns_nodup application profile frame fits)
  · intro column member
    unfold denseColumns at member
    rcases List.mem_map.1 member with ⟨offset, offsetMember, rfl⟩
    exact ConcreteNifsPiRlcActionAudit.dense_column_mem
      application profile frame
      (ConcreteNifsRawProgram.actionBase application profile frame)
      offset (List.mem_range.1 offsetMember)
  · unfold denseColumns
    rw [List.length_map, List.length_range,
      ConcreteNifsPiRlcActionRows.columns_length]

/-- Exact action-product allocation is duplicate-free. -/
theorem columns_nodup
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (fits : ConcreteNifsRawProgram.FrameFits application profile frame) :
    (ConcreteNifsPiRlcActionRows.columns application profile frame
      (ConcreteNifsRawProgram.actionBase application profile frame)).Nodup :=
  (denseColumns_perm_columns application profile frame fits).nodup_iff.1
    (denseColumns_nodup application profile frame fits)

/-- Every action product is an actual temporary coordinate of this call
frame, never a visible input or output. -/
theorem column_mem_temporaries
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (fits : ConcreteNifsRawProgram.FrameFits application profile frame)
    (column : ColumnId)
    (member :
      column ∈
        ConcreteNifsPiRlcActionRows.columns application profile frame
          (ConcreteNifsRawProgram.actionBase
            application profile frame)) :
    column ∈ frame.temporaries.ids := by
  have denseMember :
      column ∈ denseColumns application profile frame :=
    (denseColumns_perm_columns application profile frame fits).mem_iff.mpr
      member
  unfold denseColumns at denseMember
  rcases List.mem_map.1 denseMember with
    ⟨offset, offsetMember, rfl⟩
  have offsetLt := List.mem_range.1 offsetMember
  apply List.mem_iff_getElem?.2
  exact
    ⟨actionPrefix application profile frame + offset,
      denseColumn_getElem? application profile frame fits offset offsetLt⟩

private theorem emit_sublist_flatMap
    {α β : Type}
    (emit : α → List β)
    (target : α)
    (items : List α)
    (member : target ∈ items) :
    List.Sublist (emit target) (items.flatMap emit) := by
  induction items with
  | nil =>
      simp at member
  | cons head tail inductionHypothesis =>
      rcases List.mem_cons.1 member with equal | inTail
      · subst target
        exact List.sublist_append_left (emit head) (tail.flatMap emit)
      · exact (inductionHypothesis inTail).trans
          (List.sublist_append_right (emit head) (tail.flatMap emit))

private theorem productIds_sublist_columns
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (target : Phi81RingAction.Frame FixedActive.arity.total)
    (member :
      target ∈ ConcreteNifsPiRlcActionRows.frames application profile frame
        (ConcreteNifsRawProgram.actionBase
          application profile frame)) :
    List.Sublist (Phi81RingAction.productIds target)
      (ConcreteNifsPiRlcActionRows.columns application profile frame
        (ConcreteNifsRawProgram.actionBase
          application profile frame)) := by
  unfold ConcreteNifsPiRlcActionRows.columns
  exact emit_sublist_flatMap Phi81RingAction.productIds target _ member

private theorem product_mem_columns
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (target : Phi81RingAction.Frame FixedActive.arity.total)
    (targetMember :
      target ∈ ConcreteNifsPiRlcActionRows.frames application profile frame
        (ConcreteNifsRawProgram.actionBase
          application profile frame))
    (column : ColumnId)
    (productMember : column ∈ Phi81RingAction.productIds target) :
    column ∈
      ConcreteNifsPiRlcActionRows.columns application profile frame
        (ConcreteNifsRawProgram.actionBase
          application profile frame) :=
  (productIds_sublist_columns application profile frame target targetMember).subset
    productMember

/-- Exact visibility facts inherited by any frame selected by the action
emitter. -/
structure VisibleFrame
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (target : Phi81RingAction.Frame FixedActive.arity.total) : Prop where
  one : target.one = frame.one
  challenges :
    ∀ source,
      ConcreteNifsPiRlcActionAudit.CarriedVisible application frame
        (target.challenges source)
  values :
    ∀ source,
      ConcreteNifsPiRlcActionAudit.CarriedVisible application frame
        (target.values source)
  output :
    ConcreteNifsPiRlcActionAudit.CarriedVisible application frame
      target.output

private theorem visibleFrame_of_mem
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (target : Phi81RingAction.Frame FixedActive.arity.total)
    (member :
      target ∈ ConcreteNifsPiRlcActionRows.frames application profile frame
        (ConcreteNifsRawProgram.actionBase
          application profile frame)) :
    VisibleFrame application profile frame target := by
  unfold ConcreteNifsPiRlcActionRows.frames at member
  rcases List.mem_append.1 member with firstThree | inHigh
  rcases List.mem_append.1 firstThree with firstTwo | inLow
  rcases List.mem_append.1 firstTwo with inCommitment | inPublic
  · rcases List.mem_ofFn.1 inCommitment with ⟨row, rfl⟩
    exact {
      one := rfl
      challenges :=
        ConcreteNifsPiRlcActionAudit.challenge_visible
          application profile frame
      values :=
        ConcreteNifsPiRlcActionAudit.commitmentValue_visible
          application profile frame row
      output :=
        ConcreteNifsPiRlcActionAudit.commitmentOutput_visible
          application profile frame row
    }
  · rcases List.mem_ofFn.1 inPublic with ⟨block, rfl⟩
    exact {
      one := rfl
      challenges :=
        ConcreteNifsPiRlcActionAudit.challenge_visible
          application profile frame
      values :=
        ConcreteNifsPiRlcActionAudit.publicValue_visible
          application profile frame block
      output :=
        ConcreteNifsPiRlcActionAudit.publicOutput_visible
          application profile frame block
    }
  · rcases List.mem_ofFn.1 inLow with ⟨matrix, rfl⟩
    exact {
      one := rfl
      challenges :=
        ConcreteNifsPiRlcActionAudit.challenge_visible
          application profile frame
      values :=
        ConcreteNifsPiRlcActionAudit.evaluationValueLow_visible
          application profile frame matrix
      output :=
        ConcreteNifsPiRlcActionAudit.evaluationOutputLow_visible
          application profile frame matrix
    }
  · rcases List.mem_ofFn.1 inHigh with ⟨matrix, rfl⟩
    exact {
      one := rfl
      challenges :=
        ConcreteNifsPiRlcActionAudit.challenge_visible
          application profile frame
      values :=
        ConcreteNifsPiRlcActionAudit.evaluationValueHigh_visible
          application profile frame matrix
      output :=
        ConcreteNifsPiRlcActionAudit.evaluationOutputHigh_visible
          application profile frame matrix
    }

private theorem visible_not_product
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (fits : ConcreteNifsRawProgram.FrameFits application profile frame)
    (target : Phi81RingAction.Frame FixedActive.arity.total)
    (targetMember :
      target ∈ ConcreteNifsPiRlcActionRows.frames application profile frame
        (ConcreteNifsRawProgram.actionBase
          application profile frame))
    (column : ColumnId)
    (visible : column ∈ frame.visibleIds) :
    column ∉ Phi81RingAction.productIds target := by
  intro productMember
  have inColumns :=
    product_mem_columns application profile frame target targetMember
      column productMember
  have temporary :=
    column_mem_temporaries application profile frame fits column inColumns
  exact frame.temporariesDisjointVisible column temporary visible

/-- Every selected action frame satisfies the exact freshness contract needed
by the Phi81 product witness. -/
theorem frame_wellFormed
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (fits : ConcreteNifsRawProgram.FrameFits application profile frame)
    (target : Phi81RingAction.Frame FixedActive.arity.total)
    (member :
      target ∈ ConcreteNifsPiRlcActionRows.frames application profile frame
        (ConcreteNifsRawProgram.actionBase
          application profile frame)) :
    Phi81RingAction.WellFormed target := by
  let visible := visibleFrame_of_mem application profile frame target member
  have globalNodup := columns_nodup application profile frame fits
  refine {
    productsNodup :=
      (productIds_sublist_columns application profile frame target member).nodup
        globalNodup
    oneFresh := ?_
    challengesFresh := ?_
    valuesFresh := ?_
    outputFresh := ?_
  }
  · rw [visible.one]
    apply visible_not_product application profile frame fits target member
    simp [CallFrame.visibleIds]
  · intro source lane term termMember
    apply visible_not_product application profile frame fits target member
    exact visible.challenges source lane term termMember
  · intro source lane term termMember
    apply visible_not_product application profile frame fits target member
    exact visible.values source lane term termMember
  · intro lane term termMember
    apply visible_not_product application profile frame fits target member
    exact visible.output lane term termMember

/-! ## One witness for the complete action slice -/

/-- Every product entry, in the emitter's exact frame and product order. -/
def productEntries
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (assignment : ColumnId → Field) :
    List Phi81RingAction.ProductEntry :=
  (ConcreteNifsPiRlcActionRows.frames application profile frame
      (ConcreteNifsRawProgram.actionBase application profile frame)).flatMap
    (fun target => Phi81RingAction.productEntries target assignment)

/-- Complete all selected Phi81 products at once. -/
def honestAssignment
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (assignment : ColumnId → Field) :
    ColumnId → Field :=
  Phi81RingAction.writeEntries assignment
    (productEntries application profile frame assignment)

theorem productEntries_columns
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (assignment : ColumnId → Field) :
    (productEntries application profile frame assignment).map
        Phi81RingAction.ProductEntry.column =
      ConcreteNifsPiRlcActionRows.columns application profile frame
        (ConcreteNifsRawProgram.actionBase application profile frame) := by
  unfold productEntries ConcreteNifsPiRlcActionRows.columns
  generalize
    ConcreteNifsPiRlcActionRows.frames application profile frame
        (ConcreteNifsRawProgram.actionBase application profile frame) =
      targets
  induction targets with
  | nil =>
      rfl
  | cons head tail inductionHypothesis =>
      simp only [List.flatMap_cons, List.map_append,
        Phi81RingAction.productEntries_columns, inductionHypothesis]

theorem productEntries_nodup
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (fits : ConcreteNifsRawProgram.FrameFits application profile frame)
    (assignment : ColumnId → Field) :
    ((productEntries application profile frame assignment).map
      Phi81RingAction.ProductEntry.column).Nodup := by
  rw [productEntries_columns]
  exact columns_nodup application profile frame fits

/-- Completion changes only the declared action-product allocation. -/
theorem honestAssignment_preserves_column
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (assignment : ColumnId → Field)
    (column : ColumnId)
    (notAllocated :
      column ∉
        ConcreteNifsPiRlcActionRows.columns application profile frame
          (ConcreteNifsRawProgram.actionBase application profile frame)) :
    honestAssignment application profile frame assignment column =
      assignment column := by
  unfold honestAssignment
  apply Phi81RingAction.writeEntries_of_not_mem
  rw [productEntries_columns]
  exact notAllocated

theorem honestAssignment_preserves_visible
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (fits : ConcreteNifsRawProgram.FrameFits application profile frame)
    (assignment : ColumnId → Field)
    (column : ColumnId)
    (visible : column ∈ frame.visibleIds) :
    honestAssignment application profile frame assignment column =
      assignment column := by
  apply honestAssignment_preserves_column
  intro allocated
  have temporary :=
    column_mem_temporaries application profile frame fits column allocated
  exact frame.temporariesDisjointVisible column temporary visible

private theorem eval_preserves_visible
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (fits : ConcreteNifsRawProgram.FrameFits application profile frame)
    (assignment : ColumnId → Field)
    (combination : LinearCombination)
    (visible :
      ∀ term ∈ combination, term.column ∈ frame.visibleIds) :
    combination.eval (honestAssignment application profile frame assignment) =
      combination.eval assignment := by
  induction combination with
  | nil =>
      rfl
  | cons term tail inductionHypothesis =>
      rw [LinearCombination.eval, LinearCombination.eval,
        honestAssignment_preserves_visible
          application profile frame fits assignment term.column
          (visible term (by simp))]
      rw [inductionHypothesis
        (fun item member => visible item (by simp [member]))]

theorem honestAssignment_preserves_carried
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (fits : ConcreteNifsRawProgram.FrameFits application profile frame)
    (assignment : ColumnId → Field)
    (value : CarriedRing)
    (visible :
      ConcreteNifsPiRlcActionAudit.CarriedVisible application frame value) :
    Phi81RingAction.decoded
        (honestAssignment application profile frame assignment) value =
      Phi81RingAction.decoded assignment value := by
  funext lane
  exact eval_preserves_visible application profile frame fits assignment
    (value lane) (visible lane)

theorem honestAssignment_changesOnly
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (assignment : ColumnId → Field) :
    ChangesOnly
      (ConcreteNifsPiRlcActionRows.columns application profile frame
        (ConcreteNifsRawProgram.actionBase application profile frame))
      assignment
      (honestAssignment application profile frame assignment) := by
  intro column notAllocated
  exact honestAssignment_preserves_column application profile frame
    assignment column notAllocated

/-- Completing the action products preserves the complete numeric prefix
owned by the operational ΠCCS occurrence, sampler, and challenge bindings. -/
theorem honestAssignment_preserves_before_actionBase
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (fits : ConcreteNifsRawProgram.FrameFits application profile frame)
    (assignment : ColumnId → Field)
    (source : Nat)
    (before :
      source <
        ConcreteNifsRawProgram.actionBase application profile frame) :
    honestAssignment application profile frame assignment
        (columnMap frame source) =
      assignment (columnMap frame source) := by
  apply honestAssignment_preserves_column
  intro allocated
  have denseMember :
      columnMap frame source ∈ denseColumns application profile frame :=
    (denseColumns_perm_columns application profile frame fits).mem_iff.mpr
      allocated
  rcases List.mem_map.1 denseMember with
    ⟨offset, offsetMember, mappedEqual⟩
  have offsetLt := List.mem_range.1 offsetMember
  by_cases beforeTemporaries : source < temporaryBase frame
  · have visible :=
      columnMap_before_temporaryBase frame source beforeTemporaries
    have temporary :=
      column_mem_temporaries application profile frame fits
        (columnMap frame source) allocated
    exact frame.temporariesDisjointVisible
      (columnMap frame source) temporary visible
  · let sourceIndex := source - temporaryBase frame
    have sourceGe : temporaryBase frame ≤ source :=
      Nat.le_of_not_gt beforeTemporaries
    have sourceExact :
        temporarySource frame sourceIndex = source := by
      simp only [temporarySource, sourceIndex]
      omega
    have sourceIndexLtPrefix :
        sourceIndex < actionPrefix application profile frame := by
      rw [ConcreteNifsAllocationCoverage.actionBase_eq_temporarySource
        application profile frame] at before
      simp only [temporarySource, actionPrefix] at before ⊢
      omega
    have sourceIndexLt :
        sourceIndex < frame.temporaries.ids.length := by
      unfold ConcreteNifsRawProgram.FrameFits at fits
      unfold ConcreteNifsRawProgram.allocationWidth at fits
      dsimp only [actionPrefix] at sourceIndexLtPrefix
      omega
    have actionIndexLt :=
      action_index_lt application profile frame fits offset offsetLt
    have sourceOption :
        frame.temporaries.ids[sourceIndex]? =
          some (columnMap frame source) := by
      rw [List.getElem?_eq_getElem sourceIndexLt, ← sourceExact,
        columnMap_temporarySource frame sourceIndexLt]
    have actionOption :=
      denseColumn_getElem? application profile frame fits offset offsetLt
    have optionEqual :
        frame.temporaries.ids[sourceIndex]? =
          frame.temporaries.ids[
            actionPrefix application profile frame + offset]? := by
      rw [sourceOption, actionOption]
      exact congrArg some mappedEqual.symm
    have temporaryNodup :
        frame.temporaries.ids.Nodup :=
      (List.nodup_append.1 frame.allocationsNodup).2.1
    have indexEqual :=
      (List.getElem?_inj sourceIndexLt temporaryNodup).1 optionEqual
    omega

private theorem target_visible_column
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (target : Phi81RingAction.Frame FixedActive.arity.total)
    (targetMember :
      target ∈ ConcreteNifsPiRlcActionRows.frames application profile frame
        (ConcreteNifsRawProgram.actionBase application profile frame))
    (column : ColumnId)
    (member : column ∈ Phi81RingAction.visibleIds target) :
    column ∈ frame.visibleIds := by
  let visible :=
    visibleFrame_of_mem application profile frame target targetMember
  unfold Phi81RingAction.visibleIds at member
  rcases List.mem_cons.1 member with isOne | inFamilies
  · subst column
    rw [visible.one]
    simp [CallFrame.visibleIds]
  · rcases List.mem_append.1 inFamilies with inInputs | inOutput
    · rcases List.mem_append.1 inInputs with inChallenges | inValues
      · unfold Phi81RingAction.familyIds at inChallenges
        rcases List.mem_flatMap.1 inChallenges with
          ⟨source, _, sourceMember⟩
        unfold Phi81RingAction.carriedIds at sourceMember
        rcases List.mem_flatMap.1 sourceMember with
          ⟨lane, _, laneMember⟩
        unfold Phi81RingAction.combinationIds at laneMember
        rcases List.mem_map.1 laneMember with
          ⟨term, termMember, rfl⟩
        exact visible.challenges source lane term termMember
      · unfold Phi81RingAction.familyIds at inValues
        rcases List.mem_flatMap.1 inValues with
          ⟨source, _, sourceMember⟩
        unfold Phi81RingAction.carriedIds at sourceMember
        rcases List.mem_flatMap.1 sourceMember with
          ⟨lane, _, laneMember⟩
        unfold Phi81RingAction.combinationIds at laneMember
        rcases List.mem_map.1 laneMember with
          ⟨term, termMember, rfl⟩
        exact visible.values source lane term termMember
    · unfold Phi81RingAction.carriedIds at inOutput
      rcases List.mem_flatMap.1 inOutput with
        ⟨lane, _, laneMember⟩
      unfold Phi81RingAction.combinationIds at laneMember
      rcases List.mem_map.1 laneMember with
        ⟨term, termMember, rfl⟩
      exact visible.output lane term termMember

private theorem productEntry_mem
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (assignment : ColumnId → Field)
    (target : Phi81RingAction.Frame FixedActive.arity.total)
    (targetMember :
      target ∈ ConcreteNifsPiRlcActionRows.frames application profile frame
        (ConcreteNifsRawProgram.actionBase application profile frame))
    (entry : Phi81RingAction.ProductEntry)
    (entryMember :
      entry ∈ Phi81RingAction.productEntries target assignment) :
    entry ∈ productEntries application profile frame assignment := by
  unfold productEntries
  exact List.mem_flatMap.2 ⟨target, targetMember, entryMember⟩

private theorem honestAssignment_exact_entry
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (fits : ConcreteNifsRawProgram.FrameFits application profile frame)
    (assignment : ColumnId → Field)
    (target : Phi81RingAction.Frame FixedActive.arity.total)
    (targetMember :
      target ∈ ConcreteNifsPiRlcActionRows.frames application profile frame
        (ConcreteNifsRawProgram.actionBase application profile frame))
    (entry : Phi81RingAction.ProductEntry)
    (entryMember :
      entry ∈ Phi81RingAction.productEntries target assignment) :
    honestAssignment application profile frame assignment entry.column =
      entry.value := by
  unfold honestAssignment
  exact Phi81RingAction.writeEntries_exact assignment
    (productEntries application profile frame assignment)
    (productEntries_nodup application profile frame fits assignment)
    entry
    (productEntry_mem application profile frame assignment target
      targetMember entry entryMember)

private theorem localAssignment_exact_entry
    (assignment : ColumnId → Field)
    (target : Phi81RingAction.Frame FixedActive.arity.total)
    (wellFormed : Phi81RingAction.WellFormed target)
    (entry : Phi81RingAction.ProductEntry)
    (entryMember :
      entry ∈ Phi81RingAction.productEntries target assignment) :
    Phi81RingAction.honestAssignment target assignment entry.column =
      entry.value := by
  unfold Phi81RingAction.honestAssignment
  apply Phi81RingAction.writeEntries_exact assignment
    (Phi81RingAction.productEntries target assignment)
  · rw [Phi81RingAction.productEntries_columns]
    exact wellFormed.productsNodup
  · exact entryMember

private theorem eval_congr_on_terms
    (left right : ColumnId → Field)
    (combination : LinearCombination)
    (agree :
      ∀ term ∈ combination, left term.column = right term.column) :
    combination.eval left = combination.eval right := by
  induction combination with
  | nil =>
      rfl
  | cons head tail inductionHypothesis =>
      rw [LinearCombination.eval, LinearCombination.eval,
        agree head (by simp)]
      rw [inductionHypothesis
        (fun term member => agree term (by simp [member]))]

private theorem rowHolds_congr
    (left right : ColumnId → Field)
    (row : Row)
    (agree :
      ∀ column ∈ row.columnIds, left column = right column) :
    row.Holds left ↔ row.Holds right := by
  have aExact :
      row.a.eval left = row.a.eval right :=
    eval_congr_on_terms left right row.a (fun term member =>
      agree term.column (by
        unfold Row.columnIds
        apply List.mem_map.2
        exact ⟨term, by simp [member], rfl⟩))
  have bExact :
      row.b.eval left = row.b.eval right :=
    eval_congr_on_terms left right row.b (fun term member =>
      agree term.column (by
        unfold Row.columnIds
        apply List.mem_map.2
        exact ⟨term, by simp [member], rfl⟩))
  have cExact :
      row.c.eval left = row.c.eval right :=
    eval_congr_on_terms left right row.c (fun term member =>
      agree term.column (by
        unfold Row.columnIds
        apply List.mem_map.2
        exact ⟨term, by simp [member], rfl⟩))
  unfold Row.Holds
  rw [aExact, bExact, cExact]

private theorem assignments_agree_on_row
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (fits : ConcreteNifsRawProgram.FrameFits application profile frame)
    (assignment : ColumnId → Field)
    (target : Phi81RingAction.Frame FixedActive.arity.total)
    (targetMember :
      target ∈ ConcreteNifsPiRlcActionRows.frames application profile frame
        (ConcreteNifsRawProgram.actionBase application profile frame))
    (row : Row)
    (rowMember : row ∈ Phi81RingAction.rawRows target)
    (column : ColumnId)
    (columnMember : column ∈ row.columnIds) :
    honestAssignment application profile frame assignment column =
      Phi81RingAction.honestAssignment target assignment column := by
  let wellFormed :=
    frame_wellFormed application profile frame fits target targetMember
  have allowed :=
    Phi81RingAction.rawRows_supported target row rowMember
      column columnMember
  rcases List.mem_append.1 allowed with visible | product
  · have outerVisible :=
      target_visible_column application profile frame target targetMember
        column visible
    rw [honestAssignment_preserves_visible
      application profile frame fits assignment column outerVisible]
    rw [Phi81RingAction.honestAssignment_preserves_column
      target assignment column]
    exact visible_not_product application profile frame fits target
      targetMember column outerVisible
  · have mapped :
        column ∈
          (Phi81RingAction.productEntries target assignment).map
            Phi81RingAction.ProductEntry.column := by
      rw [Phi81RingAction.productEntries_columns]
      exact product
    rcases List.mem_map.1 mapped with
      ⟨entry, entryMember, entryColumn⟩
    rw [← entryColumn,
      honestAssignment_exact_entry application profile frame fits
        assignment target targetMember entry entryMember,
      localAssignment_exact_entry assignment target wellFormed
        entry entryMember]

private theorem rawSatisfies_member
    {source : List Row}
    {assignment : ColumnId → Field}
    (satisfied : RawSatisfies source assignment)
    {row : Row}
    (member : row ∈ source) :
    row.Holds assignment := by
  induction source with
  | nil =>
      simp at member
  | cons head tail inductionHypothesis =>
      rcases List.mem_cons.1 member with rfl | tailMember
      · exact satisfied.1
      · exact inductionHypothesis satisfied.2 tailMember

private theorem rawSatisfies_of_forall
    (source : List Row)
    (assignment : ColumnId → Field)
    (holds : ∀ row ∈ source, row.Holds assignment) :
    RawSatisfies source assignment := by
  induction source with
  | nil =>
      exact True.intro
  | cons head tail inductionHypothesis =>
      exact ⟨holds head (by simp),
        inductionHypothesis fun row member => holds row (by simp [member])⟩

theorem honestAssignment_agreesOn_visible
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (fits : ConcreteNifsRawProgram.FrameFits application profile frame)
    (assignment : ColumnId → Field) :
    AgreesOn frame.visibleIds assignment
      (honestAssignment application profile frame assignment) := by
  intro column visible
  exact honestAssignment_preserves_visible application profile frame fits
    assignment column visible

theorem honestAssignment_constantWire
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (fits : ConcreteNifsRawProgram.FrameFits application profile frame)
    (assignment : ColumnId → Field)
    (constantWire : assignment frame.one = 1) :
    honestAssignment application profile frame assignment frame.one = 1 := by
  rw [honestAssignment_preserves_visible application profile frame fits
    assignment frame.one (by simp [CallFrame.visibleIds])]
  exact constantWire

private theorem target_rows_honest
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (fits : ConcreteNifsRawProgram.FrameFits application profile frame)
    (assignment : ColumnId → Field)
    (target : Phi81RingAction.Frame FixedActive.arity.total)
    (targetMember :
      target ∈ ConcreteNifsPiRlcActionRows.frames application profile frame
        (ConcreteNifsRawProgram.actionBase application profile frame))
    (constantWire : assignment frame.one = 1)
    (semantic :
      Phi81RingAction.decoded assignment target.output =
        Phi81RingAction.combine
          (fun source =>
            Phi81RingAction.decoded assignment
              (target.challenges source))
          (fun source =>
            Phi81RingAction.decoded assignment (target.values source))) :
    RawSatisfies (Phi81RingAction.rawRows target)
      (honestAssignment application profile frame assignment) := by
  have wellFormed :=
    frame_wellFormed application profile frame fits target targetMember
  have targetOne :
      target.one = frame.one :=
    (visibleFrame_of_mem application profile frame target targetMember).one
  have localSatisfied :=
    Phi81RingAction.rawRows_honest target assignment wellFormed
      (by rw [targetOne]; exact constantWire)
      semantic
  apply rawSatisfies_of_forall
  intro row rowMember
  have localHolds :=
    rawSatisfies_member localSatisfied rowMember
  exact
    (rowHolds_congr
      (honestAssignment application profile frame assignment)
      (Phi81RingAction.honestAssignment target assignment)
      row
      (assignments_agree_on_row application profile frame fits assignment
        target targetMember row rowMember)).2
      localHolds

/-- **Headline selected-action honest completeness.** Exact semantic parent
equations extend the encoded call assignment only on the emitter's declared
product allocation and satisfy every selected Phi81 action row. -/
theorem rows_honest
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (fits : ConcreteNifsRawProgram.FrameFits application profile frame)
    (assignment : ColumnId → Field)
    (running :
      SelectedRunning shape publicRingColumns publicFits verifierRows)
    (fresh :
      SelectedFresh shape publicRingColumns publicFits verifierRows)
    (proof :
      SelectedProof shape TranscriptState publicRingColumns publicFits
        verifierRows)
    (output :
      SelectedRunning shape publicRingColumns publicFits verifierRows)
    (constantWire : assignment frame.one = 1)
    (decodedInputs :
      frame.operands.Decodes (FamilyFor application) assignment
        (.cons running (.cons fresh (.cons proof .nil))))
    (decodedOutput :
      frame.outputs.Decodes (FamilyFor application) assignment
        (.cons output .nil))
    (equations :
      ConcreteNifsPiRlcActionSemantics.Equations
        (keys := keys) running fresh proof output) :
    RawSatisfies
      (ConcreteNifsPiRlcActionRows.rows application profile frame
        (ConcreteNifsRawProgram.actionBase application profile frame))
      (honestAssignment application profile frame assignment) := by
  apply rawSatisfies_of_forall
  intro row rowMember
  rcases List.mem_flatMap.1 rowMember with
    ⟨target, targetMember, targetRowMember⟩
  have semantic :=
    ConcreteNifsPiRlcActionSemantics.frame_semantic_of_equations
      application profile frame assignment running fresh proof output
      (ConcreteNifsRawProgram.actionBase application profile frame)
      decodedInputs decodedOutput equations target targetMember
  have targetSatisfied :=
    target_rows_honest application profile frame fits assignment target
      targetMember constantWire semantic
  exact rawSatisfies_member targetSatisfied targetRowMember

end SelectedFrame

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsPiRlcActionHonest
