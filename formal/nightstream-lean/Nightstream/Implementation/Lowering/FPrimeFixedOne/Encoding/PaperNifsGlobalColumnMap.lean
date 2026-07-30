import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsCallColumnMap
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsCallFrame

/-!
Contract: one numeric namespace for every row gadget inside a typed
`nifsVerify` call.

The namespace is the exact ordered concatenation of the call's constant,
activation, three operand bundles, output bundle, and declared temporary
bundles.  Numeric gadgets receive this one map; no component may relocate
shared reads independently.

`Location` is a proof-carrying index into that sequence.  Constructing a
location from membership does not allocate or copy a column.

Emits constraints: none.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsGlobalColumnMap

open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsCallColumnMap
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsCallFrame

universe u

/-- Exact physical sequence addressed by the canonical numeric programs. -/
def orderedIds
    {parameters : Parameters}
    {family : Family (typeSystem parameters)}
    {context : Schema (typeSystem parameters)}
    {running :
      Ref (typeSystem parameters) context (.data .running)}
    {fresh :
      Ref (typeSystem parameters) context (.data .fresh)}
    {proof :
      Ref (typeSystem parameters) context (.data .nifsProof)}
    (frame :
      CallFrame (signature := signature parameters) family Call.nifsVerify
        (Refs.cons running (Refs.cons fresh (Refs.cons proof .nil)))) :
    List ColumnId :=
  [frame.one, frame.active] ++ frame.operands.ids ++
    frame.outputs.ids ++ frame.temporaries.ids

/-- Equal joined lists and equal segment boundaries determine the exact
middle segment. -/
theorem segment_eq_of_joined_eq
    {α : Type}
    (leftPrefix leftSegment leftSuffix :
      List α)
    (rightPrefix rightSegment rightSuffix :
      List α)
    (prefixLength :
      leftPrefix.length = rightPrefix.length)
    (segmentLength :
      leftSegment.length = rightSegment.length)
    (joined :
      leftPrefix ++ leftSegment ++ leftSuffix =
        rightPrefix ++ rightSegment ++ rightSuffix) :
    leftSegment = rightSegment := by
  have selected :=
    congrArg
      (fun values =>
        (values.drop leftPrefix.length).take leftSegment.length)
      joined
  simpa [prefixLength, segmentLength] using selected

/-- Equal complete namespaces with equal operand widths have the same exact
ordered operand namespace. No codec-coordinate reduction is required. -/
theorem operand_ids_eq_of_orderedIds_eq
    {leftParameters rightParameters : Parameters}
    {leftFamily : Family (typeSystem leftParameters)}
    {rightFamily : Family (typeSystem rightParameters)}
    {leftContext : Schema (typeSystem leftParameters)}
    {rightContext : Schema (typeSystem rightParameters)}
    {leftRunning :
      Ref (typeSystem leftParameters) leftContext (.data .running)}
    {rightRunning :
      Ref (typeSystem rightParameters) rightContext (.data .running)}
    {leftFresh :
      Ref (typeSystem leftParameters) leftContext (.data .fresh)}
    {rightFresh :
      Ref (typeSystem rightParameters) rightContext (.data .fresh)}
    {leftProof :
      Ref (typeSystem leftParameters) leftContext (.data .nifsProof)}
    {rightProof :
      Ref (typeSystem rightParameters) rightContext (.data .nifsProof)}
    (leftFrame :
      CallFrame (signature := signature leftParameters)
        leftFamily Call.nifsVerify
        (Refs.cons leftRunning
          (Refs.cons leftFresh (Refs.cons leftProof .nil))))
    (rightFrame :
      CallFrame (signature := signature rightParameters)
        rightFamily Call.nifsVerify
        (Refs.cons rightRunning
          (Refs.cons rightFresh (Refs.cons rightProof .nil))))
    (operandLengths :
      leftFrame.operands.ids.length =
        rightFrame.operands.ids.length)
    (orderedEqual :
      orderedIds leftFrame = orderedIds rightFrame) :
    leftFrame.operands.ids = rightFrame.operands.ids := by
  apply segment_eq_of_joined_eq
    [leftFrame.one, leftFrame.active] leftFrame.operands.ids
      (leftFrame.outputs.ids ++ leftFrame.temporaries.ids)
    [rightFrame.one, rightFrame.active] rightFrame.operands.ids
      (rightFrame.outputs.ids ++ rightFrame.temporaries.ids)
  · rfl
  · exact operandLengths
  · simpa only [orderedIds, List.append_assoc] using orderedEqual

/-- The authoritative visible prefix.  Every temporary follows this exact
sequence; component programs may choose offsets inside the temporary suffix
but may not introduce a second numeric namespace. -/
def visibleIds
    {parameters : Parameters}
    {family : Family (typeSystem parameters)}
    {context : Schema (typeSystem parameters)}
    {running :
      Ref (typeSystem parameters) context (.data .running)}
    {fresh :
      Ref (typeSystem parameters) context (.data .fresh)}
    {proof :
      Ref (typeSystem parameters) context (.data .nifsProof)}
    (frame :
      CallFrame (signature := signature parameters) family Call.nifsVerify
        (Refs.cons running (Refs.cons fresh (Refs.cons proof .nil)))) :
    List ColumnId :=
  [frame.one, frame.active] ++ frame.operands.ids ++ frame.outputs.ids

@[simp] theorem orderedIds_eq_visible_append_temporaries
    {parameters : Parameters}
    {family : Family (typeSystem parameters)}
    {context : Schema (typeSystem parameters)}
    {running :
      Ref (typeSystem parameters) context (.data .running)}
    {fresh :
      Ref (typeSystem parameters) context (.data .fresh)}
    {proof :
      Ref (typeSystem parameters) context (.data .nifsProof)}
    (frame :
      CallFrame (signature := signature parameters) family Call.nifsVerify
        (Refs.cons running (Refs.cons fresh (Refs.cons proof .nil)))) :
    orderedIds frame = visibleIds frame ++ frame.temporaries.ids := by
  simp [orderedIds, visibleIds, List.append_assoc]

/-- First numeric source owned by the call's temporary suffix. -/
def temporaryBase
    {parameters : Parameters}
    {family : Family (typeSystem parameters)}
    {context : Schema (typeSystem parameters)}
    {running :
      Ref (typeSystem parameters) context (.data .running)}
    {fresh :
      Ref (typeSystem parameters) context (.data .fresh)}
    {proof :
      Ref (typeSystem parameters) context (.data .nifsProof)}
    (frame :
      CallFrame (signature := signature parameters) family Call.nifsVerify
        (Refs.cons running (Refs.cons fresh (Refs.cons proof .nil)))) :
    Nat :=
  (visibleIds frame).length

/-- Numeric source of one declared temporary coordinate. -/
def temporarySource
    {parameters : Parameters}
    {family : Family (typeSystem parameters)}
    {context : Schema (typeSystem parameters)}
    {running :
      Ref (typeSystem parameters) context (.data .running)}
    {fresh :
      Ref (typeSystem parameters) context (.data .fresh)}
    {proof :
      Ref (typeSystem parameters) context (.data .nifsProof)}
    (frame :
      CallFrame (signature := signature parameters) family Call.nifsVerify
        (Refs.cons running (Refs.cons fresh (Refs.cons proof .nil))))
    (index : Nat) :
    Nat :=
  temporaryBase frame + index

theorem temporarySource_lt
    {parameters : Parameters}
    {family : Family (typeSystem parameters)}
    {context : Schema (typeSystem parameters)}
    {running :
      Ref (typeSystem parameters) context (.data .running)}
    {fresh :
      Ref (typeSystem parameters) context (.data .fresh)}
    {proof :
      Ref (typeSystem parameters) context (.data .nifsProof)}
    (frame :
      CallFrame (signature := signature parameters) family Call.nifsVerify
        (Refs.cons running (Refs.cons fresh (Refs.cons proof .nil))))
    {index : Nat}
    (indexLt : index < frame.temporaries.ids.length) :
    temporarySource frame index < (orderedIds frame).length := by
  rw [orderedIds_eq_visible_append_temporaries, List.length_append]
  exact Nat.add_lt_add_left indexLt (visibleIds frame).length

/-- The sole numeric-to-typed map for one call occurrence. -/
def columnMap
    {parameters : Parameters}
    {family : Family (typeSystem parameters)}
    {context : Schema (typeSystem parameters)}
    {running :
      Ref (typeSystem parameters) context (.data .running)}
    {fresh :
      Ref (typeSystem parameters) context (.data .fresh)}
    {proof :
      Ref (typeSystem parameters) context (.data .nifsProof)}
    (frame :
      CallFrame (signature := signature parameters) family Call.nifsVerify
        (Refs.cons running (Refs.cons fresh (Refs.cons proof .nil)))) :
    Nat → ColumnId :=
  fun source => (orderedIds frame).getD source frame.one

/-- Every numeric source maps into the call's actual visible or temporary
coordinates.  An out-of-range source maps to the already-visible constant-one
wire; semantic constructions separately prove that every allocated source is
in range. -/
theorem columnMap_supported
    {parameters : Parameters}
    {family : Family (typeSystem parameters)}
    {context : Schema (typeSystem parameters)}
    {running :
      Ref (typeSystem parameters) context (.data .running)}
    {fresh :
      Ref (typeSystem parameters) context (.data .fresh)}
    {proof :
      Ref (typeSystem parameters) context (.data .nifsProof)}
    (frame :
      CallFrame (signature := signature parameters) family Call.nifsVerify
        (Refs.cons running (Refs.cons fresh (Refs.cons proof .nil))))
    (source : Nat) :
    columnMap frame source ∈
      frame.visibleIds ++ frame.temporaries.ids := by
  unfold columnMap
  by_cases inRange : source < (orderedIds frame).length
  · rw [List.getD_eq_getElem?_getD,
      List.getElem?_eq_getElem inRange, Option.getD_some]
    let id := (orderedIds frame)[source]
    change id ∈ frame.visibleIds ++ frame.temporaries.ids
    have member : id ∈ orderedIds frame := List.getElem_mem inRange
    have supported :
        id ∈ visibleIds frame ++ frame.temporaries.ids := by
      simpa only [orderedIds_eq_visible_append_temporaries] using member
    rcases List.mem_append.1 supported with inVisible | inTemporary
    · apply List.mem_append.2
      left
      simp only [PaperNifsGlobalColumnMap.visibleIds] at inVisible
      simp only [CallFrame.visibleIds]
      rcases List.mem_append.1 inVisible with inPrefix | inOutput
      · rcases List.mem_append.1 inPrefix with inStructural | inOperand
        · exact List.mem_append.2
            (Or.inl (List.mem_append.2 (Or.inl inStructural)))
        · exact List.mem_append.2
            (Or.inl (List.mem_append.2
              (Or.inr
                (RefBundles.fromSchema_ids_subset
                  _ frame.contextBundles _ inOperand))))
      · exact List.mem_append.2 (Or.inr inOutput)
    · exact List.mem_append.2 (Or.inr inTemporary)
  · have pastEnd : (orderedIds frame).length ≤ source :=
      Nat.le_of_not_gt inRange
    rw [List.getD_eq_getElem?_getD,
      List.getElem?_eq_none pastEnd, Option.getD_none]
    apply List.mem_append.2
    left
    simp [CallFrame.visibleIds]

@[simp] theorem columnMap_temporarySource
    {parameters : Parameters}
    {family : Family (typeSystem parameters)}
    {context : Schema (typeSystem parameters)}
    {running :
      Ref (typeSystem parameters) context (.data .running)}
    {fresh :
      Ref (typeSystem parameters) context (.data .fresh)}
    {proof :
      Ref (typeSystem parameters) context (.data .nifsProof)}
    (frame :
      CallFrame (signature := signature parameters) family Call.nifsVerify
        (Refs.cons running (Refs.cons fresh (Refs.cons proof .nil))))
    {index : Nat}
    (indexLt : index < frame.temporaries.ids.length) :
    columnMap frame (temporarySource frame index) =
      frame.temporaries.ids[index] := by
  unfold columnMap temporarySource temporaryBase
  rw [orderedIds_eq_visible_append_temporaries,
    List.getD_eq_getElem?_getD,
    List.getElem?_append_right
      (Nat.le_add_right (visibleIds frame).length index)]
  simp [indexLt]

@[simp] theorem columnMap_zero
    {parameters : Parameters}
    {family : Family (typeSystem parameters)}
    {context : Schema (typeSystem parameters)}
    {running :
      Ref (typeSystem parameters) context (.data .running)}
    {fresh :
      Ref (typeSystem parameters) context (.data .fresh)}
    {proof :
      Ref (typeSystem parameters) context (.data .nifsProof)}
    (frame :
      CallFrame (signature := signature parameters) family Call.nifsVerify
        (Refs.cons running (Refs.cons fresh (Refs.cons proof .nil)))) :
    columnMap frame 0 = frame.one := by
  simp [columnMap, orderedIds]

@[simp] theorem columnMap_one
    {parameters : Parameters}
    {family : Family (typeSystem parameters)}
    {context : Schema (typeSystem parameters)}
    {running :
      Ref (typeSystem parameters) context (.data .running)}
    {fresh :
      Ref (typeSystem parameters) context (.data .fresh)}
    {proof :
      Ref (typeSystem parameters) context (.data .nifsProof)}
    (frame :
      CallFrame (signature := signature parameters) family Call.nifsVerify
        (Refs.cons running (Refs.cons fresh (Refs.cons proof .nil)))) :
    columnMap frame 1 = frame.active := by
  simp [columnMap, orderedIds]

/-- One exact occurrence of a physical identity in the shared sequence.
Repeated operand references may yield multiple valid locations; every such
location maps to the same physical identity. -/
structure Location
    {parameters : Parameters}
    {family : Family (typeSystem parameters)}
    {context : Schema (typeSystem parameters)}
    {running :
      Ref (typeSystem parameters) context (.data .running)}
    {fresh :
      Ref (typeSystem parameters) context (.data .fresh)}
    {proof :
      Ref (typeSystem parameters) context (.data .nifsProof)}
    (frame :
      CallFrame (signature := signature parameters) family Call.nifsVerify
        (Refs.cons running (Refs.cons fresh (Refs.cons proof .nil))))
    (id : ColumnId) where
  source : Nat
  sourceLt : source < (orderedIds frame).length
  sourceAt : (orderedIds frame)[source] = id

theorem Location.mapped
    {parameters : Parameters}
    {family : Family (typeSystem parameters)}
    {context : Schema (typeSystem parameters)}
    {running :
      Ref (typeSystem parameters) context (.data .running)}
    {fresh :
      Ref (typeSystem parameters) context (.data .fresh)}
    {proof :
      Ref (typeSystem parameters) context (.data .nifsProof)}
    {frame :
      CallFrame (signature := signature parameters) family Call.nifsVerify
        (Refs.cons running (Refs.cons fresh (Refs.cons proof .nil)))}
    {id : ColumnId}
    (location : Location frame id) :
    columnMap frame location.source = id := by
  unfold columnMap
  rw [List.getD_eq_getElem?_getD,
    List.getElem?_eq_getElem location.sourceLt, location.sourceAt]
  rfl

/-- Every identity in the numeric namespace's visible prefix is also visible
to the enclosing call recipe.  The latter may contain unrelated context
columns in addition to the three addressed operands. -/
theorem visibleIds_supported
    {parameters : Parameters}
    {family : Family (typeSystem parameters)}
    {context : Schema (typeSystem parameters)}
    {running :
      Ref (typeSystem parameters) context (.data .running)}
    {fresh :
      Ref (typeSystem parameters) context (.data .fresh)}
    {proof :
      Ref (typeSystem parameters) context (.data .nifsProof)}
    (frame :
      CallFrame (signature := signature parameters) family Call.nifsVerify
        (Refs.cons running (Refs.cons fresh (Refs.cons proof .nil))))
    {id : ColumnId}
    (member : id ∈ visibleIds frame) :
    id ∈ frame.visibleIds := by
  unfold visibleIds at member
  unfold CallFrame.visibleIds
  rcases List.mem_append.1 member with inPrefix | inOutput
  · rcases List.mem_append.1 inPrefix with inStructural | inOperand
    · exact List.mem_append_left _
        (List.mem_append_left _ inStructural)
    · exact List.mem_append_left _
        (List.mem_append_right _
          (RefBundles.fromSchema_ids_subset _ frame.contextBundles _
            inOperand))
  · exact List.mem_append_right _ inOutput

/-- A numeric source strictly before the temporary suffix maps to an
authoritative visible column.  This is the exact-prefix strengthening of
`columnMap_supported`; it is used when later programs must prove that they do
not read an activation-owned temporary suffix. -/
theorem columnMap_before_temporaryBase
    {parameters : Parameters}
    {family : Family (typeSystem parameters)}
    {context : Schema (typeSystem parameters)}
    {running :
      Ref (typeSystem parameters) context (.data .running)}
    {fresh :
      Ref (typeSystem parameters) context (.data .fresh)}
    {proof :
      Ref (typeSystem parameters) context (.data .nifsProof)}
    (frame :
      CallFrame (signature := signature parameters) family Call.nifsVerify
        (Refs.cons running (Refs.cons fresh (Refs.cons proof .nil))))
    (source : Nat)
    (before : source < temporaryBase frame) :
    columnMap frame source ∈ frame.visibleIds := by
  have sourceBound :
      source < (orderedIds frame).length := by
    rw [orderedIds_eq_visible_append_temporaries, List.length_append]
    simpa [temporaryBase] using
      Nat.lt_of_lt_of_le before
        (Nat.le_add_right (visibleIds frame).length
          frame.temporaries.ids.length)
  have sourceBoundAppend :
      source <
        (visibleIds frame ++ frame.temporaries.ids).length := by
    simpa only [orderedIds_eq_visible_append_temporaries] using sourceBound
  have visibleBound : source < (visibleIds frame).length := by
    simpa [temporaryBase] using before
  have selected :
      (orderedIds frame)[source]'sourceBound =
        (visibleIds frame)[source]'visibleBound := by
    change
      (visibleIds frame ++ frame.temporaries.ids)[
          source]'sourceBoundAppend =
        (visibleIds frame)[source]'visibleBound
    rw [List.getElem_append_left]
  have mapped :
      columnMap frame source =
        (orderedIds frame)[source]'sourceBound := by
    unfold columnMap
    rw [List.getD_eq_getElem?_getD,
      List.getElem?_eq_getElem sourceBound]
    rfl
  rw [mapped, selected]
  exact visibleIds_supported frame (List.getElem_mem visibleBound)

/-- A location of a visible identity must occur before the temporary suffix.
This rejects a physical alias between a later activation residual and an
authoritative input/output even when the numeric lookup itself is defined by
first occurrence. -/
theorem Location.source_lt_temporaryBase_of_visible
    {parameters : Parameters}
    {family : Family (typeSystem parameters)}
    {context : Schema (typeSystem parameters)}
    {running :
      Ref (typeSystem parameters) context (.data .running)}
    {fresh :
      Ref (typeSystem parameters) context (.data .fresh)}
    {proof :
      Ref (typeSystem parameters) context (.data .nifsProof)}
    {frame :
      CallFrame (signature := signature parameters) family Call.nifsVerify
        (Refs.cons running (Refs.cons fresh (Refs.cons proof .nil)))}
    {id : ColumnId}
    (location : Location frame id)
    (visible : id ∈ visibleIds frame) :
    location.source < temporaryBase frame := by
  by_cases before : location.source < temporaryBase frame
  · exact before
  have afterPrefix :
      (visibleIds frame).length ≤ location.source := by
    simpa [temporaryBase] using Nat.le_of_not_gt before
  have sourceBound :
      location.source <
        (visibleIds frame ++ frame.temporaries.ids).length := by
    simpa only [orderedIds_eq_visible_append_temporaries] using
      location.sourceLt
  have offsetBound :
      location.source - (visibleIds frame).length <
        frame.temporaries.ids.length := by
    rw [List.length_append] at sourceBound
    omega
  have selected :
      frame.temporaries.ids[
          location.source - (visibleIds frame).length] = id := by
    have sourceAt :
        (visibleIds frame ++ frame.temporaries.ids)[
          location.source]'sourceBound = id := by
      simpa only [orderedIds_eq_visible_append_temporaries] using
        location.sourceAt
    rw [List.getElem_append_right afterPrefix] at sourceAt
    exact sourceAt
  have inTemporary : id ∈ frame.temporaries.ids := by
    rw [← selected]
    exact List.getElem_mem offsetBound
  exact False.elim
    (frame.temporariesDisjointVisible id inTemporary
      (visibleIds_supported frame visible))

/-- Construct a numeric location from a proof that the physical identity is
part of the authoritative call sequence. -/
def locate
    {parameters : Parameters}
    {family : Family (typeSystem parameters)}
    {context : Schema (typeSystem parameters)}
    {running :
      Ref (typeSystem parameters) context (.data .running)}
    {fresh :
      Ref (typeSystem parameters) context (.data .fresh)}
    {proof :
      Ref (typeSystem parameters) context (.data .nifsProof)}
    (frame :
      CallFrame (signature := signature parameters) family Call.nifsVerify
        (Refs.cons running (Refs.cons fresh (Refs.cons proof .nil))))
    (id : ColumnId)
    (member : id ∈ orderedIds frame) :
    Location frame id := by
  match found : (orderedIds frame).idxOf? id with
  | none =>
      exact False.elim ((List.idxOf?_eq_none_iff.mp found) member)
  | some source =>
      have witness := List.idxOf?_eq_some_iff.mp found
      have bound : source < (orderedIds frame).length :=
        Exists.elim witness fun bound _ => bound
      have selected : (orderedIds frame)[source] = id :=
        Exists.elim witness fun _ selectedAndFirst => selectedAndFirst.1
      exact ⟨source, bound, selected⟩

/-- Forget the proof-carrying location and retain its first numeric source. -/
theorem locate_source_eq_idxOf_getD
    {parameters : Parameters}
    {family : Family (typeSystem parameters)}
    {context : Schema (typeSystem parameters)}
    {running :
      Ref (typeSystem parameters) context (.data .running)}
    {fresh :
      Ref (typeSystem parameters) context (.data .fresh)}
    {proof :
      Ref (typeSystem parameters) context (.data .nifsProof)}
    (frame :
      CallFrame (signature := signature parameters) family Call.nifsVerify
        (Refs.cons running (Refs.cons fresh (Refs.cons proof .nil))))
    (id : ColumnId)
    (member : id ∈ orderedIds frame) :
    (locate frame id member).source =
      ((orderedIds frame).idxOf? id).getD 0 := by
  unfold locate
  split
  next found =>
    exact False.elim ((List.idxOf?_eq_none_iff.mp found) member)
  next source found =>
    simpa only [Option.getD_some] using
      (congrArg (fun candidate => candidate.getD 0) found).symm

/-- Numeric location is determined only by the ordered physical namespace and
the selected physical identity. Dependent frame proofs do not change it. -/
theorem locate_source_congr
    {leftParameters rightParameters : Parameters}
    {leftFamily : Family (typeSystem leftParameters)}
    {rightFamily : Family (typeSystem rightParameters)}
    {leftContext : Schema (typeSystem leftParameters)}
    {rightContext : Schema (typeSystem rightParameters)}
    {leftRunning :
      Ref (typeSystem leftParameters) leftContext (.data .running)}
    {rightRunning :
      Ref (typeSystem rightParameters) rightContext (.data .running)}
    {leftFresh :
      Ref (typeSystem leftParameters) leftContext (.data .fresh)}
    {rightFresh :
      Ref (typeSystem rightParameters) rightContext (.data .fresh)}
    {leftProof :
      Ref (typeSystem leftParameters) leftContext (.data .nifsProof)}
    {rightProof :
      Ref (typeSystem rightParameters) rightContext (.data .nifsProof)}
    (leftFrame :
      CallFrame (signature := signature leftParameters)
        leftFamily Call.nifsVerify
        (Refs.cons leftRunning
          (Refs.cons leftFresh (Refs.cons leftProof .nil))))
    (rightFrame :
      CallFrame (signature := signature rightParameters)
        rightFamily Call.nifsVerify
        (Refs.cons rightRunning
          (Refs.cons rightFresh (Refs.cons rightProof .nil))))
    {leftId rightId : ColumnId}
    (leftMember : leftId ∈ orderedIds leftFrame)
    (rightMember : rightId ∈ orderedIds rightFrame)
    (orderedEqual : orderedIds leftFrame = orderedIds rightFrame)
    (idEqual : leftId = rightId) :
    (locate leftFrame leftId leftMember).source =
      (locate rightFrame rightId rightMember).source := by
  rw [locate_source_eq_idxOf_getD, locate_source_eq_idxOf_getD,
    orderedEqual, idEqual]

theorem operand_mem
    {parameters : Parameters}
    {family : Family (typeSystem parameters)}
    {context : Schema (typeSystem parameters)}
    {running :
      Ref (typeSystem parameters) context (.data .running)}
    {fresh :
      Ref (typeSystem parameters) context (.data .fresh)}
    {proof :
      Ref (typeSystem parameters) context (.data .nifsProof)}
    (frame :
      CallFrame (signature := signature parameters) family Call.nifsVerify
        (Refs.cons running (Refs.cons fresh (Refs.cons proof .nil))))
    {id : ColumnId}
    (member : id ∈ frame.operands.ids) :
    id ∈ orderedIds frame := by
  unfold orderedIds
  exact
    List.mem_append_left frame.temporaries.ids
      (List.mem_append_left frame.outputs.ids
        (List.mem_append_right [frame.one, frame.active] member))

theorem operand_mem_visible
    {parameters : Parameters}
    {family : Family (typeSystem parameters)}
    {context : Schema (typeSystem parameters)}
    {running :
      Ref (typeSystem parameters) context (.data .running)}
    {fresh :
      Ref (typeSystem parameters) context (.data .fresh)}
    {proof :
      Ref (typeSystem parameters) context (.data .nifsProof)}
    (frame :
      CallFrame (signature := signature parameters) family Call.nifsVerify
        (Refs.cons running (Refs.cons fresh (Refs.cons proof .nil))))
    {id : ColumnId}
    (member : id ∈ frame.operands.ids) :
    id ∈ visibleIds frame := by
  unfold visibleIds
  exact List.mem_append_left frame.outputs.ids
    (List.mem_append_right [frame.one, frame.active] member)

theorem runningOperand_mem
    {parameters : Parameters}
    {family : Family (typeSystem parameters)}
    {context : Schema (typeSystem parameters)}
    {running :
      Ref (typeSystem parameters) context (.data .running)}
    {fresh :
      Ref (typeSystem parameters) context (.data .fresh)}
    {proof :
      Ref (typeSystem parameters) context (.data .nifsProof)}
    (frame :
      CallFrame (signature := signature parameters) family Call.nifsVerify
        (Refs.cons running (Refs.cons fresh (Refs.cons proof .nil))))
    {id : ColumnId}
    (member : id ∈ (runningOperand frame.operands).ids) :
    id ∈ orderedIds frame := by
  apply operand_mem frame
  have inOperands :
      id ∈ (runningOperand frame.operands).ids ++
        ((freshOperand frame.operands).ids ++
          (proofOperand frame.operands).ids) :=
    List.mem_append_left _ member
  simpa only [operand_ids] using inOperands

theorem runningOperand_mem_visible
    {parameters : Parameters}
    {family : Family (typeSystem parameters)}
    {context : Schema (typeSystem parameters)}
    {running :
      Ref (typeSystem parameters) context (.data .running)}
    {fresh :
      Ref (typeSystem parameters) context (.data .fresh)}
    {proof :
      Ref (typeSystem parameters) context (.data .nifsProof)}
    (frame :
      CallFrame (signature := signature parameters) family Call.nifsVerify
        (Refs.cons running (Refs.cons fresh (Refs.cons proof .nil))))
    {id : ColumnId}
    (member : id ∈ (runningOperand frame.operands).ids) :
    id ∈ visibleIds frame := by
  apply operand_mem_visible frame
  have inOperands :
      id ∈ (runningOperand frame.operands).ids ++
        ((freshOperand frame.operands).ids ++
          (proofOperand frame.operands).ids) :=
    List.mem_append_left _ member
  simpa only [operand_ids] using inOperands

theorem freshOperand_mem
    {parameters : Parameters}
    {family : Family (typeSystem parameters)}
    {context : Schema (typeSystem parameters)}
    {running :
      Ref (typeSystem parameters) context (.data .running)}
    {fresh :
      Ref (typeSystem parameters) context (.data .fresh)}
    {proof :
      Ref (typeSystem parameters) context (.data .nifsProof)}
    (frame :
      CallFrame (signature := signature parameters) family Call.nifsVerify
        (Refs.cons running (Refs.cons fresh (Refs.cons proof .nil))))
    {id : ColumnId}
    (member : id ∈ (freshOperand frame.operands).ids) :
    id ∈ orderedIds frame := by
  apply operand_mem frame
  have inOperands :
      id ∈ (runningOperand frame.operands).ids ++
        ((freshOperand frame.operands).ids ++
          (proofOperand frame.operands).ids) :=
    List.mem_append_right _
      (List.mem_append_left _ member)
  simpa only [operand_ids] using inOperands

theorem freshOperand_mem_visible
    {parameters : Parameters}
    {family : Family (typeSystem parameters)}
    {context : Schema (typeSystem parameters)}
    {running :
      Ref (typeSystem parameters) context (.data .running)}
    {fresh :
      Ref (typeSystem parameters) context (.data .fresh)}
    {proof :
      Ref (typeSystem parameters) context (.data .nifsProof)}
    (frame :
      CallFrame (signature := signature parameters) family Call.nifsVerify
        (Refs.cons running (Refs.cons fresh (Refs.cons proof .nil))))
    {id : ColumnId}
    (member : id ∈ (freshOperand frame.operands).ids) :
    id ∈ visibleIds frame := by
  apply operand_mem_visible frame
  have inOperands :
      id ∈ (runningOperand frame.operands).ids ++
        ((freshOperand frame.operands).ids ++
          (proofOperand frame.operands).ids) :=
    List.mem_append_right _
      (List.mem_append_left _ member)
  simpa only [operand_ids] using inOperands

theorem proofOperand_mem
    {parameters : Parameters}
    {family : Family (typeSystem parameters)}
    {context : Schema (typeSystem parameters)}
    {running :
      Ref (typeSystem parameters) context (.data .running)}
    {fresh :
      Ref (typeSystem parameters) context (.data .fresh)}
    {proof :
      Ref (typeSystem parameters) context (.data .nifsProof)}
    (frame :
      CallFrame (signature := signature parameters) family Call.nifsVerify
        (Refs.cons running (Refs.cons fresh (Refs.cons proof .nil))))
    {id : ColumnId}
    (member : id ∈ (proofOperand frame.operands).ids) :
    id ∈ orderedIds frame := by
  apply operand_mem frame
  have inOperands :
      id ∈ (runningOperand frame.operands).ids ++
        ((freshOperand frame.operands).ids ++
          (proofOperand frame.operands).ids) :=
    List.mem_append_right _
      (List.mem_append_right _ member)
  simpa only [operand_ids] using inOperands

theorem proofOperand_mem_visible
    {parameters : Parameters}
    {family : Family (typeSystem parameters)}
    {context : Schema (typeSystem parameters)}
    {running :
      Ref (typeSystem parameters) context (.data .running)}
    {fresh :
      Ref (typeSystem parameters) context (.data .fresh)}
    {proof :
      Ref (typeSystem parameters) context (.data .nifsProof)}
    (frame :
      CallFrame (signature := signature parameters) family Call.nifsVerify
        (Refs.cons running (Refs.cons fresh (Refs.cons proof .nil))))
    {id : ColumnId}
    (member : id ∈ (proofOperand frame.operands).ids) :
    id ∈ visibleIds frame := by
  apply operand_mem_visible frame
  have inOperands :
      id ∈ (runningOperand frame.operands).ids ++
        ((freshOperand frame.operands).ids ++
          (proofOperand frame.operands).ids) :=
    List.mem_append_right _
      (List.mem_append_right _ member)
  simpa only [operand_ids] using inOperands

theorem output_mem
    {parameters : Parameters}
    {family : Family (typeSystem parameters)}
    {context : Schema (typeSystem parameters)}
    {running :
      Ref (typeSystem parameters) context (.data .running)}
    {fresh :
      Ref (typeSystem parameters) context (.data .fresh)}
    {proof :
      Ref (typeSystem parameters) context (.data .nifsProof)}
    (frame :
      CallFrame (signature := signature parameters) family Call.nifsVerify
        (Refs.cons running (Refs.cons fresh (Refs.cons proof .nil))))
    {id : ColumnId}
    (member : id ∈ frame.outputs.ids) :
    id ∈ orderedIds frame := by
  unfold orderedIds
  exact
    List.mem_append_left frame.temporaries.ids
      (List.mem_append_right
      ([frame.one, frame.active] ++ frame.operands.ids) member)

theorem output_mem_visible
    {parameters : Parameters}
    {family : Family (typeSystem parameters)}
    {context : Schema (typeSystem parameters)}
    {running :
      Ref (typeSystem parameters) context (.data .running)}
    {fresh :
      Ref (typeSystem parameters) context (.data .fresh)}
    {proof :
      Ref (typeSystem parameters) context (.data .nifsProof)}
    (frame :
      CallFrame (signature := signature parameters) family Call.nifsVerify
        (Refs.cons running (Refs.cons fresh (Refs.cons proof .nil))))
    {id : ColumnId}
    (member : id ∈ frame.outputs.ids) :
    id ∈ visibleIds frame := by
  unfold visibleIds
  exact
    List.mem_append_right
      ([frame.one, frame.active] ++ frame.operands.ids) member

theorem temporary_mem
    {parameters : Parameters}
    {family : Family (typeSystem parameters)}
    {context : Schema (typeSystem parameters)}
    {running :
      Ref (typeSystem parameters) context (.data .running)}
    {fresh :
      Ref (typeSystem parameters) context (.data .fresh)}
    {proof :
      Ref (typeSystem parameters) context (.data .nifsProof)}
    (frame :
      CallFrame (signature := signature parameters) family Call.nifsVerify
        (Refs.cons running (Refs.cons fresh (Refs.cons proof .nil))))
    {id : ColumnId}
    (member : id ∈ frame.temporaries.ids) :
    id ∈ orderedIds frame := by
  unfold orderedIds
  exact List.mem_append_right _ member

/-- Turn one visible physical base-field coordinate into the singleton
numeric expression consumed by canonical base-field gadgets. -/
def fLocation
    {parameters : Parameters}
    {family : Family (typeSystem parameters)}
    {context : Schema (typeSystem parameters)}
    {running :
      Ref (typeSystem parameters) context (.data .running)}
    {fresh :
      Ref (typeSystem parameters) context (.data .fresh)}
    {proof :
      Ref (typeSystem parameters) context (.data .nifsProof)}
    (frame :
      CallFrame (signature := signature parameters) family Call.nifsVerify
        (Refs.cons running (Refs.cons fresh (Refs.cons proof .nil))))
    (typed : PaperNifsCodecProjection.FColumnId)
    (member : typed.column ∈ orderedIds frame) :
    FLocation (columnMap frame) typed := by
  let located := locate frame typed.column member
  exact
    { numeric := located.source
      mapped := located.mapped }

/-- Turn two visible physical coordinates into the singleton numeric
expressions consumed by the canonical `K` gadgets. -/
def kLocation
    {parameters : Parameters}
    {family : Family (typeSystem parameters)}
    {context : Schema (typeSystem parameters)}
    {running :
      Ref (typeSystem parameters) context (.data .running)}
    {fresh :
      Ref (typeSystem parameters) context (.data .fresh)}
    {proof :
      Ref (typeSystem parameters) context (.data .nifsProof)}
    (frame :
      CallFrame (signature := signature parameters) family Call.nifsVerify
        (Refs.cons running (Refs.cons fresh (Refs.cons proof .nil))))
    (typed : PaperNifsCodecProjection.KColumnIds)
    (c0Member : typed.c0 ∈ orderedIds frame)
    (c1Member : typed.c1 ∈ orderedIds frame) :
    KLocation (columnMap frame) typed := by
  let c0 := locate frame typed.c0 c0Member
  let c1 := locate frame typed.c1 c1Member
  exact
    { numeric := { c0 := c0.source, c1 := c1.source }
      c0Mapped := c0.mapped
      c1Mapped := c1.mapped }

/-- A base-field location of an authoritative visible identity is strictly
before the temporary suffix. -/
theorem fLocation_numeric_lt_temporaryBase
    {parameters : Parameters}
    {family : Family (typeSystem parameters)}
    {context : Schema (typeSystem parameters)}
    {running :
      Ref (typeSystem parameters) context (.data .running)}
    {fresh :
      Ref (typeSystem parameters) context (.data .fresh)}
    {proof :
      Ref (typeSystem parameters) context (.data .nifsProof)}
    (frame :
      CallFrame (signature := signature parameters) family Call.nifsVerify
        (Refs.cons running (Refs.cons fresh (Refs.cons proof .nil))))
    (typed : PaperNifsCodecProjection.FColumnId)
    (member : typed.column ∈ orderedIds frame)
    (visible : typed.column ∈ visibleIds frame) :
    (fLocation frame typed member).numeric < temporaryBase frame := by
  unfold fLocation
  exact Location.source_lt_temporaryBase_of_visible
    (locate frame typed.column member) visible

/-- Both coordinates of an authoritative extension-field location are
strictly before the temporary suffix. -/
theorem kLocation_numeric_lt_temporaryBase
    {parameters : Parameters}
    {family : Family (typeSystem parameters)}
    {context : Schema (typeSystem parameters)}
    {running :
      Ref (typeSystem parameters) context (.data .running)}
    {fresh :
      Ref (typeSystem parameters) context (.data .fresh)}
    {proof :
      Ref (typeSystem parameters) context (.data .nifsProof)}
    (frame :
      CallFrame (signature := signature parameters) family Call.nifsVerify
        (Refs.cons running (Refs.cons fresh (Refs.cons proof .nil))))
    (typed : PaperNifsCodecProjection.KColumnIds)
    (c0Member : typed.c0 ∈ orderedIds frame)
    (c1Member : typed.c1 ∈ orderedIds frame)
    (c0Visible : typed.c0 ∈ visibleIds frame)
    (c1Visible : typed.c1 ∈ visibleIds frame) :
    (kLocation frame typed c0Member c1Member).numeric.c0 <
        temporaryBase frame
      ∧ (kLocation frame typed c0Member c1Member).numeric.c1 <
        temporaryBase frame := by
  unfold kLocation
  exact
    ⟨Location.source_lt_temporaryBase_of_visible
        (locate frame typed.c0 c0Member) c0Visible,
      Location.source_lt_temporaryBase_of_visible
        (locate frame typed.c1 c1Member) c1Visible⟩

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsGlobalColumnMap
