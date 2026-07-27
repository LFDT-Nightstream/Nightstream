import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.Poseidon23HashCallCommon

/-!
Contract: exact typed-call placement of the prior binding-hash occurrence.

Owns: operand/output ordering, the nine mandatory temporary bundles, and
derivation of the embedded core's allocation facts from the call receipt.

Does not own: semantic soundness, application deployment selection, Rust,
generated rows, or collision resistance.
-/

set_option autoImplicit false
set_option maxRecDepth 32768

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding

open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls
open Poseidon23HashCallCommon

namespace Poseidon23HashPriorFrame

private theorem footprint_exact
    (parameters : Parameters)
    (profile : Poseidon23ApplicationProfile parameters) :
    (signature parameters).callFootprint Call.hashPrior =
      Poseidon23Hash.footprint profile.alignmentWidth := by
  simpa [signature, callFootprint] using profile.hashFootprint

private theorem temporary_layouts_exact
    (parameters : Parameters)
    (profile : Poseidon23ApplicationProfile parameters) :
    ((signature parameters).callFootprint Call.hashPrior).temporaries =
      temporaryLayouts profile.alignmentWidth := by
  rw [footprint_exact parameters profile]
  rfl

private def frameTemporaries
    (parameters : Parameters)
    (profile : Poseidon23ApplicationProfile parameters)
    {context : Schema (typeSystem parameters)}
    {iteration : Ref (typeSystem parameters) context (.data .nat)}
    {z0 current : Ref (typeSystem parameters) context (.data .state)}
    {running : Ref (typeSystem parameters) context (.data .running)}
    (frame :
      CallFrame (signature := signature parameters)
        profile.family Call.hashPrior
        (Refs.cons iteration
          (Refs.cons z0 (Refs.cons current (Refs.cons running .nil))))) :
    LayoutBundles (temporaryLayouts profile.alignmentWidth) :=
  temporary_layouts_exact parameters profile ▸ frame.temporaries

private theorem frameTemporaries_ids
    (parameters : Parameters)
    (profile : Poseidon23ApplicationProfile parameters)
    {context : Schema (typeSystem parameters)}
    {iteration : Ref (typeSystem parameters) context (.data .nat)}
    {z0 current : Ref (typeSystem parameters) context (.data .state)}
    {running : Ref (typeSystem parameters) context (.data .running)}
    (frame :
      CallFrame (signature := signature parameters)
        profile.family Call.hashPrior
        (Refs.cons iteration
          (Refs.cons z0 (Refs.cons current (Refs.cons running .nil))))) :
    (frameTemporaries parameters profile frame).ids =
      frame.temporaries.ids :=
  layoutBundles_ids_cast
    (temporary_layouts_exact parameters profile) frame.temporaries

private theorem frameTemporaries_columns
    (parameters : Parameters)
    (profile : Poseidon23ApplicationProfile parameters)
    {context : Schema (typeSystem parameters)}
    {iteration : Ref (typeSystem parameters) context (.data .nat)}
    {z0 current : Ref (typeSystem parameters) context (.data .state)}
    {running : Ref (typeSystem parameters) context (.data .running)}
    (frame :
      CallFrame (signature := signature parameters)
        profile.family Call.hashPrior
        (Refs.cons iteration
          (Refs.cons z0 (Refs.cons current (Refs.cons running .nil))))) :
    (frameTemporaries parameters profile frame).columns =
      frame.temporaries.columns :=
  layoutBundles_columns_cast
    (temporary_layouts_exact parameters profile) frame.temporaries

private theorem iteration_width_one
    (parameters : Parameters)
    (profile : Poseidon23ApplicationProfile parameters)
    {context : Schema (typeSystem parameters)}
    {iteration : Ref (typeSystem parameters) context (.data .nat)}
    {z0 current : Ref (typeSystem parameters) context (.data .state)}
    {running : Ref (typeSystem parameters) context (.data .running)}
    (frame :
      CallFrame (signature := signature parameters)
        profile.family Call.hashPrior
        (Refs.cons iteration
          (Refs.cons z0 (Refs.cons current (Refs.cons running .nil))))) :
    iteration.port.layout.owners.length = 1 := by
  have width := frame.operandWidthsAgree.1
  simpa [Poseidon23ApplicationProfile.family,
    TerminalEqualityProfile.family, Profile.family, DataCodecs.family,
    Family.codecFor] using width.symm

private def iterationColumn
    (parameters : Parameters)
    (profile : Poseidon23ApplicationProfile parameters)
    {context : Schema (typeSystem parameters)}
    {iteration : Ref (typeSystem parameters) context (.data .nat)}
    {z0 current : Ref (typeSystem parameters) context (.data .state)}
    {running : Ref (typeSystem parameters) context (.data .running)}
    (frame :
      CallFrame (signature := signature parameters)
        profile.family Call.hashPrior
        (Refs.cons iteration
          (Refs.cons z0 (Refs.cons current (Refs.cons running .nil))))) :
    OwnedColumn :=
  bundleColumn (firstOperand frame.operands)
    ⟨0, Eq.mpr
      (congrArg (fun width => 0 < width)
        (iteration_width_one parameters profile frame))
      (by decide)⟩

private def sourceTail
    (parameters : Parameters)
    (profile : Poseidon23ApplicationProfile parameters)
    {context : Schema (typeSystem parameters)}
    {iteration : Ref (typeSystem parameters) context (.data .nat)}
    {z0 current : Ref (typeSystem parameters) context (.data .state)}
    {running : Ref (typeSystem parameters) context (.data .running)}
    (frame :
      CallFrame (signature := signature parameters)
        profile.family Call.hashPrior
        (Refs.cons iteration
          (Refs.cons z0 (Refs.cons current (Refs.cons running .nil))))) :
    List OwnedColumn :=
  (secondOperand frame.operands).columns ++
    ((thirdOperand frame.operands).columns ++
      (fourthOperand frame.operands).columns)

private theorem sourceTail_length
    (parameters : Parameters)
    (profile : Poseidon23ApplicationProfile parameters)
    {context : Schema (typeSystem parameters)}
    {iteration : Ref (typeSystem parameters) context (.data .nat)}
    {z0 current : Ref (typeSystem parameters) context (.data .state)}
    {running : Ref (typeSystem parameters) context (.data .running)}
    (frame :
      CallFrame (signature := signature parameters)
        profile.family Call.hashPrior
        (Refs.cons iteration
          (Refs.cons z0 (Refs.cons current (Refs.cons running .nil))))) :
    (sourceTail parameters profile frame).length + 1 =
      Poseidon23Hash.sourceWidth profile.codecs := by
  have z0Width := frame.operandWidthsAgree.2.1
  have currentWidth := frame.operandWidthsAgree.2.2.1
  have runningWidth := frame.operandWidthsAgree.2.2.2.1
  have z0Exact :
      z0.port.layout.owners.length = profile.codecs.state.width := by
    simpa [Poseidon23ApplicationProfile.family,
      TerminalEqualityProfile.family, Profile.family, DataCodecs.family,
      Family.codecFor] using z0Width.symm
  have currentExact :
      current.port.layout.owners.length = profile.codecs.state.width := by
    simpa [Poseidon23ApplicationProfile.family,
      TerminalEqualityProfile.family, Profile.family, DataCodecs.family,
      Family.codecFor] using currentWidth.symm
  have runningExact :
      running.port.layout.owners.length = profile.codecs.running.width := by
    simpa [Poseidon23ApplicationProfile.family,
      TerminalEqualityProfile.family, Profile.family, DataCodecs.family,
      Family.codecFor] using runningWidth.symm
  unfold sourceTail Poseidon23Hash.sourceWidth
  rw [List.length_append, List.length_append,
    ColumnBundle.length_eq, ColumnBundle.length_eq, ColumnBundle.length_eq]
  calc
    z0.port.layout.owners.length +
          (current.port.layout.owners.length +
            running.port.layout.owners.length) + 1 =
        current.port.layout.owners.length +
          (current.port.layout.owners.length +
            running.port.layout.owners.length) + 1 := by
      exact congrArg
        (fun width =>
          width +
            (current.port.layout.owners.length +
              running.port.layout.owners.length) + 1)
        (z0Exact.trans currentExact.symm)
    _ = 1 + current.port.layout.owners.length +
          current.port.layout.owners.length +
          running.port.layout.owners.length := by omega
    _ = 1 + profile.codecs.state.width +
          profile.codecs.state.width +
          profile.codecs.running.width := by
      simpa only [currentExact, runningExact]

private theorem output_length
    (parameters : Parameters)
    (profile : Poseidon23ApplicationProfile parameters)
    {context : Schema (typeSystem parameters)}
    {iteration : Ref (typeSystem parameters) context (.data .nat)}
    {z0 current : Ref (typeSystem parameters) context (.data .state)}
    {running : Ref (typeSystem parameters) context (.data .running)}
    (frame :
      CallFrame (signature := signature parameters)
        profile.family Call.hashPrior
        (Refs.cons iteration
          (Refs.cons z0 (Refs.cons current (Refs.cons running .nil))))) :
    (unaryOutput frame.outputs).columns.length = 5 := by
  have width :=
    frame.outputWidthsAgree (Ports.auxiliaryDigest parameters) (by
      change Ports.auxiliaryDigest parameters ∈
        callOutputs parameters Call.hashPrior
      exact List.mem_cons_self)
  have exact :
      (Ports.auxiliaryDigest parameters).layout.owners.length =
        profile.codecs.digest.width := by
    unfold PortWidthAgrees at width
    simpa [Poseidon23ApplicationProfile.family,
      TerminalEqualityProfile.family, Profile.family, DataCodecs.family,
      Family.codecFor] using width.symm
  rw [ColumnBundle.length_eq, exact, profile.digestWidth]

/-- Exact prior-hash physical occurrence derived from its typed call frame. -/
def occurrence
    (parameters : Parameters)
    (profile : Poseidon23ApplicationProfile parameters)
    {context : Schema (typeSystem parameters)}
    {iteration : Ref (typeSystem parameters) context (.data .nat)}
    {z0 current : Ref (typeSystem parameters) context (.data .state)}
    {running : Ref (typeSystem parameters) context (.data .running)}
    (frame :
      CallFrame (signature := signature parameters)
        profile.family Call.hashPrior
        (Refs.cons iteration
          (Refs.cons z0 (Refs.cons current (Refs.cons running .nil))))) :
    Poseidon23HashOccurrence.Frame
      (Poseidon23Hash.sourceWidth profile.codecs)
      profile.alignmentWidth :=
  Poseidon23HashCallCommon.occurrence frame.owner frame.one frame.active false
    (iterationColumn parameters profile frame)
    (sourceTail parameters profile frame)
    (sourceTail_length parameters profile frame)
    (unaryOutput frame.outputs).columns
    (output_length parameters profile frame)
    (splitTemporaries (frameTemporaries parameters profile frame))
    profile.hashPlan

theorem iteration_values_eq_singleton
    (parameters : Parameters)
    (profile : Poseidon23ApplicationProfile parameters)
    {context : Schema (typeSystem parameters)}
    {iteration : Ref (typeSystem parameters) context (.data .nat)}
    {z0 current : Ref (typeSystem parameters) context (.data .state)}
    {running : Ref (typeSystem parameters) context (.data .running)}
    (frame :
      CallFrame (signature := signature parameters)
        profile.family Call.hashPrior
        (Refs.cons iteration
          (Refs.cons z0 (Refs.cons current (Refs.cons running .nil)))))
    (assignment : ColumnId -> Field) :
    (firstOperand frame.operands).values assignment =
      [assignment (occurrence parameters profile frame).iteration.id] := by
  simpa [occurrence, Poseidon23HashCallCommon.occurrence,
    iterationColumn] using
    bundle_values_eq_singleton (firstOperand frame.operands) assignment
      (iteration_width_one parameters profile frame)

theorem source_tail_values
    (parameters : Parameters)
    (profile : Poseidon23ApplicationProfile parameters)
    {context : Schema (typeSystem parameters)}
    {iteration : Ref (typeSystem parameters) context (.data .nat)}
    {z0 current : Ref (typeSystem parameters) context (.data .state)}
    {running : Ref (typeSystem parameters) context (.data .running)}
    (frame :
      CallFrame (signature := signature parameters)
        profile.family Call.hashPrior
        (Refs.cons iteration
          (Refs.cons z0 (Refs.cons current (Refs.cons running .nil)))))
    (assignment : ColumnId -> Field) :
    (occurrence parameters profile frame).sourceTail.map
        (fun column => assignment column.id) =
      (secondOperand frame.operands).values assignment ++
        ((thirdOperand frame.operands).values assignment ++
          (fourthOperand frame.operands).values assignment) := by
  simp [occurrence, Poseidon23HashCallCommon.occurrence, sourceTail,
    ColumnBundle.values]

theorem temporary_ids_exact
    (parameters : Parameters)
    (profile : Poseidon23ApplicationProfile parameters)
    {context : Schema (typeSystem parameters)}
    {iteration : Ref (typeSystem parameters) context (.data .nat)}
    {z0 current : Ref (typeSystem parameters) context (.data .state)}
    {running : Ref (typeSystem parameters) context (.data .running)}
    (frame :
      CallFrame (signature := signature parameters)
        profile.family Call.hashPrior
        (Refs.cons iteration
          (Refs.cons z0 (Refs.cons current (Refs.cons running .nil))))) :
    (occurrence parameters profile frame).temporaryIds =
      frame.temporaries.ids := by
  rw [occurrence, occurrence_temporaryIds, splitTemporaries_ids,
    frameTemporaries_ids parameters profile frame]

theorem visible_subset
    (parameters : Parameters)
    (profile : Poseidon23ApplicationProfile parameters)
    {context : Schema (typeSystem parameters)}
    {iteration : Ref (typeSystem parameters) context (.data .nat)}
    {z0 current : Ref (typeSystem parameters) context (.data .state)}
    {running : Ref (typeSystem parameters) context (.data .running)}
    (frame :
      CallFrame (signature := signature parameters)
        profile.family Call.hashPrior
        (Refs.cons iteration
          (Refs.cons z0 (Refs.cons current (Refs.cons running .nil))))) :
    ∀ id, id ∈ (occurrence parameters profile frame).visibleIds ->
      id ∈ frame.visibleIds := by
  intro id member
  change id ∈
    [frame.one, frame.active,
      (iterationColumn parameters profile frame).id] ++
      (sourceTail parameters profile frame).map (fun column => column.id) ++
      (unaryOutput frame.outputs).columns.map (fun column => column.id)
    at member
  rcases List.mem_append.mp member with prefixMember | outputMember
  rcases List.mem_append.mp prefixMember with headerMember | sourceMember
  have headerCases :
      id = frame.one ∨ id = frame.active ∨
        id = (iterationColumn parameters profile frame).id := by
    simpa only [List.mem_cons, List.not_mem_nil, or_false] using headerMember
  rcases headerCases with one | active | iterationMember
  · subst id
    simp [CallFrame.visibleIds]
  · subst id
    simp [CallFrame.visibleIds]
  · have firstMember :
        id ∈ (firstOperand frame.operands).ids := by
      rw [iterationMember]
      exact bundleColumn_id_mem (firstOperand frame.operands)
        ⟨0, Eq.mpr
          (congrArg (fun width => 0 < width)
            (iteration_width_one parameters profile frame))
          (by decide)⟩
    have operandMember : id ∈ frame.operands.ids := by
      have joined :
          id ∈ (firstOperand frame.operands).ids ++
            ((secondOperand frame.operands).ids ++
              ((thirdOperand frame.operands).ids ++
                (fourthOperand frame.operands).ids)) :=
        List.mem_append_left _ firstMember
      simpa only [operand_ids] using joined
    have contextMember :=
      RefBundles.fromSchema_ids_subset
        (Refs.cons iteration
          (Refs.cons z0 (Refs.cons current (Refs.cons running .nil))))
        frame.contextBundles id operandMember
    change id ∈ [frame.one, frame.active] ++
      frame.contextBundles.ids ++ frame.outputs.ids
    exact List.mem_append_left frame.outputs.ids
      (List.mem_append_right [frame.one, frame.active] contextMember)
  · have operandMember : id ∈ frame.operands.ids := by
      have tailMember :
          id ∈ (secondOperand frame.operands).ids ++
            ((thirdOperand frame.operands).ids ++
              (fourthOperand frame.operands).ids) := by
        simpa [sourceTail, ColumnBundle.ids] using sourceMember
      have joined :
          id ∈ (firstOperand frame.operands).ids ++
            ((secondOperand frame.operands).ids ++
              ((thirdOperand frame.operands).ids ++
                (fourthOperand frame.operands).ids)) :=
        List.mem_append_right _ tailMember
      simpa only [operand_ids] using joined
    have contextMember :=
      RefBundles.fromSchema_ids_subset
        (Refs.cons iteration
          (Refs.cons z0 (Refs.cons current (Refs.cons running .nil))))
        frame.contextBundles id operandMember
    change id ∈ [frame.one, frame.active] ++
      frame.contextBundles.ids ++ frame.outputs.ids
    exact List.mem_append_left frame.outputs.ids
      (List.mem_append_right [frame.one, frame.active] contextMember)
  · have output : id ∈ frame.outputs.ids := by
      have unaryMember : id ∈ (unaryOutput frame.outputs).ids := by
        exact outputMember
      simpa only [unaryOutput_ids] using unaryMember
    change id ∈ [frame.one, frame.active] ++
      frame.contextBundles.ids ++ frame.outputs.ids
    exact List.mem_append_right
      ([frame.one, frame.active] ++ frame.contextBundles.ids) output

/-- Allocation facts for the embedded canonical sponge are consequences of
the complete typed call receipt. -/
def coreFacts
    (parameters : Parameters)
    (profile : Poseidon23ApplicationProfile parameters)
    {context : Schema (typeSystem parameters)}
    {iteration : Ref (typeSystem parameters) context (.data .nat)}
    {z0 current : Ref (typeSystem parameters) context (.data .state)}
    {running : Ref (typeSystem parameters) context (.data .running)}
    (frame :
      CallFrame (signature := signature parameters)
        profile.family Call.hashPrior
        (Refs.cons iteration
          (Refs.cons z0 (Refs.cons current (Refs.cons running .nil))))) :
    Poseidon23HashOccurrence.CoreAllocationFacts
      (occurrence parameters profile frame) := by
  let bundles := frameTemporaries parameters profile frame
  let parts := splitTemporaries bundles
  apply coreAllocationFacts frame.owner frame.one frame.active false
    (iterationColumn parameters profile frame)
    (sourceTail parameters profile frame)
    (sourceTail_length parameters profile frame)
    (unaryOutput frame.outputs).columns
    (output_length parameters profile frame)
    parts profile.hashPlan
  · have temporaryNodup : frame.temporaries.ids.Nodup :=
      (List.nodup_append.mp frame.allocationsNodup).2.1
    rw [splitTemporaries_ids, frameTemporaries_ids parameters profile frame]
    exact temporaryNodup
  · intro id temporaryMember visibleMember
    apply frame.temporariesDisjointVisible id
    · rw [← frameTemporaries_ids parameters profile frame,
        ← splitTemporaries_ids bundles]
      exact temporaryMember
    · exact visible_subset parameters profile frame id visibleMember
  · intro column member
    apply frame.allocationsOwned column
    apply List.mem_append_right frame.outputs.columns
    rw [← frameTemporaries_columns parameters profile frame,
      ← splitTemporaries_columns bundles]
    exact member

end Poseidon23HashPriorFrame

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding
