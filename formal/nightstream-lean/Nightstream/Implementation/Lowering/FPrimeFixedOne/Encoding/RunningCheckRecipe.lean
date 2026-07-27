import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.TerminalEqualityProfile
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls.Equality

/-!
Contract: canonical `runningCheck` recipe for a selected terminal-equality
application profile.

Owns: equality of the authoritative running and running-witness coordinate
strings, exact transport to the frozen executable running checker, active and
inactive behavior, ownership, receipt support, and honest completion.

Does not own: the fresh relation, NIFS, a final fold, a deployment profile,
Rust, or generated rows.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding

open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls

namespace RunningCheckRecipe

private theorem footprint_exact
    (parameters : Parameters)
    (profile : TerminalEqualityProfile parameters) :
    (signature parameters).callFootprint Call.runningCheck =
      equalityFootprint profile.codecs.running.width := by
  simpa [signature, callFootprint] using profile.runningFootprint

private theorem temporary_layouts_exact
    (parameters : Parameters)
    (profile : TerminalEqualityProfile parameters) :
    ((signature parameters).callFootprint Call.runningCheck).temporaries =
      [auxiliaryLayout profile.codecs.running.width,
        auxiliaryLayout profile.codecs.running.width,
        auxiliaryLayout profile.codecs.running.width.pred] := by
  rw [footprint_exact parameters profile]
  rfl

private def frameTemporaries
    (parameters : Parameters)
    (profile : TerminalEqualityProfile parameters)
    {context : Schema (typeSystem parameters)}
    {running :
      Ref (typeSystem parameters) context (.data .running)}
    {witness :
      Ref (typeSystem parameters) context (.data .runningWitness)}
    (frame :
      CallFrame (signature := signature parameters)
        profile.family Call.runningCheck
        (Refs.cons running (Refs.cons witness .nil))) :
    LayoutBundles
      [auxiliaryLayout profile.codecs.running.width,
        auxiliaryLayout profile.codecs.running.width,
        auxiliaryLayout profile.codecs.running.width.pred] :=
  temporary_layouts_exact parameters profile ▸ frame.temporaries

private theorem frameTemporaries_ids
    (parameters : Parameters)
    (profile : TerminalEqualityProfile parameters)
    {context : Schema (typeSystem parameters)}
    {running :
      Ref (typeSystem parameters) context (.data .running)}
    {witness :
      Ref (typeSystem parameters) context (.data .runningWitness)}
    (frame :
      CallFrame (signature := signature parameters)
        profile.family Call.runningCheck
        (Refs.cons running (Refs.cons witness .nil))) :
    (frameTemporaries parameters profile frame).ids =
      frame.temporaries.ids :=
  layoutBundles_ids_cast
    (temporary_layouts_exact parameters profile) frame.temporaries

private theorem output_width_one
    (parameters : Parameters)
    (profile : TerminalEqualityProfile parameters)
    {context : Schema (typeSystem parameters)}
    {running :
      Ref (typeSystem parameters) context (.data .running)}
    {witness :
      Ref (typeSystem parameters) context (.data .runningWitness)}
    (frame :
      CallFrame (signature := signature parameters)
        profile.family Call.runningCheck
        (Refs.cons running (Refs.cons witness .nil))) :
    (Ports.auxiliaryBit parameters).layout.owners.length = 1 := by
  have width :=
    frame.outputWidthsAgree (Ports.auxiliaryBit parameters) (by
      change Ports.auxiliaryBit parameters ∈
        callOutputs parameters Call.runningCheck
      exact List.mem_cons_self)
  unfold PortWidthAgrees at width
  simpa [TerminalEqualityProfile.family, Profile.family,
    DataCodecs.family, Family.codecFor] using width.symm

private theorem output_width_positive
    (parameters : Parameters)
    (profile : TerminalEqualityProfile parameters)
    {context : Schema (typeSystem parameters)}
    {running :
      Ref (typeSystem parameters) context (.data .running)}
    {witness :
      Ref (typeSystem parameters) context (.data .runningWitness)}
    (frame :
      CallFrame (signature := signature parameters)
        profile.family Call.runningCheck
        (Refs.cons running (Refs.cons witness .nil))) :
    0 < (Ports.auxiliaryBit parameters).layout.owners.length :=
  Eq.mpr
    (congrArg (fun width => 0 < width)
      (output_width_one parameters profile frame))
    (by decide)

private def occurrence
    (parameters : Parameters)
    (profile : TerminalEqualityProfile parameters)
    {context : Schema (typeSystem parameters)}
    {running :
      Ref (typeSystem parameters) context (.data .running)}
    {witness :
      Ref (typeSystem parameters) context (.data .runningWitness)}
    (frame :
      CallFrame (signature := signature parameters)
        profile.family Call.runningCheck
        (Refs.cons running (Refs.cons witness .nil))) :
    EqualityRecipe where
  owner := frame.owner
  one := frame.one
  active := frame.active
  left := (firstBinaryOperand frame.operands).columns
  right := (secondBinaryOperand frame.operands).columns
  output :=
    bundleColumn (unaryOutput frame.outputs)
      ⟨0, output_width_positive parameters profile frame⟩
  inverses :=
    (firstTemporary (frameTemporaries parameters profile frame)).columns
  equals :=
    (secondTemporary (frameTemporaries parameters profile frame)).columns
  products :=
    (thirdTemporary (frameTemporaries parameters profile frame)).columns
  rightLength := by
    rw [ColumnBundle.length_eq, ColumnBundle.length_eq]
    have leftWidth := frame.operandWidthsAgree.1
    have rightWidth := frame.operandWidthsAgree.2.1
    have leftExact :
        profile.codecs.running.width =
          running.port.layout.owners.length := by
      simpa [TerminalEqualityProfile.family, Profile.family,
        DataCodecs.family, Family.codecFor] using leftWidth
    have rightExact :
        profile.codecs.runningWitness.width =
          witness.port.layout.owners.length := by
      simpa [TerminalEqualityProfile.family, Profile.family,
        DataCodecs.family, Family.codecFor] using rightWidth
    calc
      witness.port.layout.owners.length =
          profile.codecs.runningWitness.width := rightExact.symm
      _ = profile.codecs.running.width := profile.runningWidthsEqual
      _ = running.port.layout.owners.length := leftExact
  inverseLength := by
    rw [ColumnBundle.length_eq, ColumnBundle.length_eq]
    simp [auxiliaryLayout, ownedLayout]
    have leftWidth := frame.operandWidthsAgree.1
    simpa [TerminalEqualityProfile.family, Profile.family,
      DataCodecs.family, Family.codecFor] using leftWidth
  equalLength := by
    rw [ColumnBundle.length_eq, ColumnBundle.length_eq]
    simp [auxiliaryLayout, ownedLayout]
    have leftWidth := frame.operandWidthsAgree.1
    simpa [TerminalEqualityProfile.family, Profile.family,
      DataCodecs.family, Family.codecFor] using leftWidth
  productLength := by
    rw [ColumnBundle.length_eq, ColumnBundle.length_eq]
    simp [auxiliaryLayout, ownedLayout]
    have leftWidth := frame.operandWidthsAgree.1
    simpa [TerminalEqualityProfile.family, Profile.family,
      DataCodecs.family, Family.codecFor] using
        congrArg Nat.pred leftWidth

private theorem temporary_ids_exact
    (parameters : Parameters)
    (profile : TerminalEqualityProfile parameters)
    {context : Schema (typeSystem parameters)}
    {running :
      Ref (typeSystem parameters) context (.data .running)}
    {witness :
      Ref (typeSystem parameters) context (.data .runningWitness)}
    (frame :
      CallFrame (signature := signature parameters)
        profile.family Call.runningCheck
        (Refs.cons running (Refs.cons witness .nil))) :
    frame.temporaries.ids =
      (occurrence parameters profile frame).inverses.map
          (fun column => column.id) ++
        ((occurrence parameters profile frame).equals.map
            (fun column => column.id) ++
          (occurrence parameters profile frame).products.map
            (fun column => column.id)) := by
  rw [← frameTemporaries_ids parameters profile frame,
    threeTemporary_ids]
  simp [occurrence, ColumnBundle.ids, List.append_assoc]

private theorem output_decodes
    (parameters : Parameters)
    (profile : TerminalEqualityProfile parameters)
    {context : Schema (typeSystem parameters)}
    {running :
      Ref (typeSystem parameters) context (.data .running)}
    {witness :
      Ref (typeSystem parameters) context (.data .runningWitness)}
    (frame :
      CallFrame (signature := signature parameters)
        profile.family Call.runningCheck
        (Refs.cons running (Refs.cons witness .nil)))
    (assignment : ColumnId -> Field)
    (value : Bool)
    (coordinate :
      assignment (occurrence parameters profile frame).output.id =
        if value then 1 else 0) :
    (unaryOutput frame.outputs).Decodes
      profile.family .bit assignment value := by
  unfold ColumnBundle.Decodes
  rw [bundle_values_eq_singleton
    (unaryOutput frame.outputs) assignment
    (output_width_one parameters profile frame)]
  change
    boolCodec.decode
        [assignment (occurrence parameters profile frame).output.id] =
      some value
  cases value with
  | false =>
      exact (boolCodec_decode_false_iff _).mpr (by simpa using coordinate)
  | true =>
      exact (boolCodec_decode_true_iff _).mpr (by simpa using coordinate)

private theorem output_coordinate_of_encodes
    (parameters : Parameters)
    (profile : TerminalEqualityProfile parameters)
    {context : Schema (typeSystem parameters)}
    {running :
      Ref (typeSystem parameters) context (.data .running)}
    {witness :
      Ref (typeSystem parameters) context (.data .runningWitness)}
    (frame :
      CallFrame (signature := signature parameters)
        profile.family Call.runningCheck
        (Refs.cons running (Refs.cons witness .nil)))
    (assignment : ColumnId -> Field)
    (value : Bool)
    (encoded :
      (unaryOutput frame.outputs).Encodes
        profile.family .bit assignment value) :
    assignment (occurrence parameters profile frame).output.id =
      if value then 1 else 0 := by
  have decoded :=
    (unaryOutput frame.outputs).decodes_of_encodes
      profile.family .bit assignment value encoded
  unfold ColumnBundle.Decodes at decoded
  rw [bundle_values_eq_singleton
    (unaryOutput frame.outputs) assignment
    (output_width_one parameters profile frame)] at decoded
  change
    boolCodec.decode
        [assignment (occurrence parameters profile frame).output.id] =
      some value at decoded
  cases value with
  | false =>
      exact (boolCodec_decode_false_iff _).mp decoded
  | true =>
      exact (boolCodec_decode_true_iff _).mp decoded

private theorem completion_agrees
    (parameters : Parameters)
    (profile : TerminalEqualityProfile parameters)
    {context : Schema (typeSystem parameters)}
    {running :
      Ref (typeSystem parameters) context (.data .running)}
    {witness :
      Ref (typeSystem parameters) context (.data .runningWitness)}
    (frame :
      CallFrame (signature := signature parameters)
        profile.family Call.runningCheck
        (Refs.cons running (Refs.cons witness .nil)))
    (assignment : ColumnId -> Field) :
    AgreesOn frame.visibleIds assignment
      ((occurrence parameters profile frame).completion
        profile.inverseLaw frame.temporaries.ids assignment) :=
  writeColumns_agreesOn assignment frame.temporaries.ids frame.visibleIds
    _ frame.temporariesDisjointVisible

private theorem operand_values_preserved
    (parameters : Parameters)
    (profile : TerminalEqualityProfile parameters)
    {context : Schema (typeSystem parameters)}
    {running :
      Ref (typeSystem parameters) context (.data .running)}
    {witness :
      Ref (typeSystem parameters) context (.data .runningWitness)}
    (frame :
      CallFrame (signature := signature parameters)
        profile.family Call.runningCheck
        (Refs.cons running (Refs.cons witness .nil)))
    (before after : ColumnId -> Field)
    (agrees : AgreesOn frame.visibleIds before after) :
    (occurrence parameters profile frame).left.map
          (fun column => after column.id) =
        (occurrence parameters profile frame).left.map
          (fun column => before column.id) ∧
      (occurrence parameters profile frame).right.map
          (fun column => after column.id) =
        (occurrence parameters profile frame).right.map
          (fun column => before column.id) := by
  constructor
  · apply List.map_congr_left
    intro column member
    apply agrees
    have operandMember : column.id ∈ frame.operands.ids := by
      have firstMember :
          column.id ∈ (firstBinaryOperand frame.operands).ids := by
        unfold ColumnBundle.ids
        exact List.mem_map.mpr ⟨column, by
          simpa [occurrence] using member, rfl⟩
      have joined :=
        List.mem_append_left
          (secondBinaryOperand frame.operands).ids firstMember
      simpa only [binaryOperand_ids] using joined
    have contextMember :=
      RefBundles.fromSchema_ids_subset
        (Refs.cons running (Refs.cons witness .nil))
        frame.contextBundles column.id operandMember
    change column.id ∈
      [frame.one, frame.active] ++
        frame.contextBundles.ids ++ frame.outputs.ids
    exact List.mem_append_left frame.outputs.ids
      (List.mem_append_right [frame.one, frame.active] contextMember)
  · apply List.map_congr_left
    intro column member
    apply agrees
    have operandMember : column.id ∈ frame.operands.ids := by
      have secondMember :
          column.id ∈ (secondBinaryOperand frame.operands).ids := by
        unfold ColumnBundle.ids
        exact List.mem_map.mpr ⟨column, by
          simpa [occurrence] using member, rfl⟩
      have joined :=
        List.mem_append_right
          (firstBinaryOperand frame.operands).ids secondMember
      simpa only [binaryOperand_ids] using joined
    have contextMember :=
      RefBundles.fromSchema_ids_subset
        (Refs.cons running (Refs.cons witness .nil))
        frame.contextBundles column.id operandMember
    change column.id ∈
      [frame.one, frame.active] ++
        frame.contextBundles.ids ++ frame.outputs.ids
    exact List.mem_append_left frame.outputs.ids
      (List.mem_append_right [frame.one, frame.active] contextMember)

private theorem coordinate_values_of_decodes
    (parameters : Parameters)
    (profile : TerminalEqualityProfile parameters)
    {context : Schema (typeSystem parameters)}
    {runningRef :
      Ref (typeSystem parameters) context (.data .running)}
    {witnessRef :
      Ref (typeSystem parameters) context (.data .runningWitness)}
    (frame :
      CallFrame (signature := signature parameters)
        profile.family Call.runningCheck
        (Refs.cons runningRef (Refs.cons witnessRef .nil)))
    (assignment : ColumnId -> Field)
    (running : parameters.Running)
    (witness : parameters.RunningWitness)
    (decoded :
      frame.operands.Decodes profile.family assignment
        (.cons running (.cons witness .nil))) :
    (occurrence parameters profile frame).left.map
          (fun column => assignment column.id) =
        profile.codecs.running.encode running ∧
      (occurrence parameters profile frame).right.map
          (fun column => assignment column.id) =
        profile.codecs.runningWitness.encode witness := by
  have pair :=
    (binaryOperand_decodes_iff profile.family assignment
      frame.operands running witness).mp decoded
  have leftEncoded :=
    (profile.codecs.running.encode_decode
      ((firstBinaryOperand frame.operands).values assignment)
      running (by
        simpa [TerminalEqualityProfile.family, Profile.family,
          DataCodecs.family, Family.codecFor] using pair.1)).2
  have rightEncoded :=
    (profile.codecs.runningWitness.encode_decode
      ((secondBinaryOperand frame.operands).values assignment)
      witness (by
        simpa [TerminalEqualityProfile.family, Profile.family,
          DataCodecs.family, Family.codecFor] using pair.2)).2
  exact ⟨by simpa [occurrence, ColumnBundle.values] using leftEncoded.symm,
    by simpa [occurrence, ColumnBundle.values] using rightEncoded.symm⟩

/-- Certified running relation recipe. -/
def recipe
    (parameters : Parameters)
    (profile : TerminalEqualityProfile parameters) :
    CallRecipe (signature parameters) profile.family Call.runningCheck := by
  refine
    { rows := ?_
      rowCount := ?_
      rowsOwned := ?_
      rowIdsNodup := ?_
      rowsSupported := ?_
      activeSoundness := ?_
      activeHonestCompleteness := ?_
      inactiveSatisfiable := ?_ }
  · intro context references frame
    cases references with
    | cons running tail =>
        cases tail with
        | cons witness tail =>
            cases tail
            exact (occurrence parameters profile frame).rows
  · intro context references frame
    cases references with
    | cons running tail =>
        cases tail with
        | cons witness tail =>
            cases tail
            rw [footprint_exact parameters profile]
            have leftWidth := frame.operandWidthsAgree.1
            have leftLength :
                (occurrence parameters profile frame).left.length =
                  profile.codecs.running.width := by
              rw [occurrence, ColumnBundle.length_eq]
              simpa [TerminalEqualityProfile.family, Profile.family,
                DataCodecs.family, Family.codecFor] using leftWidth.symm
            change
              (occurrence parameters profile frame).rows.length =
                (equalityFootprint profile.codecs.running.width).recurringRows
            rw [(occurrence parameters profile frame).row_count, leftLength]
            rfl
  · intro context references frame row member
    cases references with
    | cons running tail =>
        cases tail with
        | cons witness tail =>
            cases tail
            exact (occurrence parameters profile frame).rows_owned row member
  · intro context references frame
    cases references with
    | cons running tail =>
        cases tail with
        | cons witness tail =>
            cases tail
            exact (occurrence parameters profile frame).row_ids_nodup
  · intro context references frame row member column columnMember
    cases references with
    | cons running tail =>
        cases tail with
        | cons witness tail =>
            cases tail
            rcases
                (occurrence parameters profile frame).rows_supported
                  row member column columnMember with
              one | active | leftMember | rightMember | output |
                inverse | equal | product
            · subst column
              simp [occurrence, CallFrame.visibleIds]
            · subst column
              simp [occurrence, CallFrame.visibleIds]
            · have operandMember : column ∈ frame.operands.ids := by
                have firstMember :
                    column ∈ (firstBinaryOperand frame.operands).ids := by
                  simpa [occurrence, ColumnBundle.ids] using leftMember
                simpa only [binaryOperand_ids] using
                  List.mem_append_left
                    (secondBinaryOperand frame.operands).ids firstMember
              have contextMember :=
                RefBundles.fromSchema_ids_subset
                  (Refs.cons running (Refs.cons witness .nil))
                  frame.contextBundles column operandMember
              simp [CallFrame.visibleIds, contextMember]
            · have operandMember : column ∈ frame.operands.ids := by
                have secondMember :
                    column ∈ (secondBinaryOperand frame.operands).ids := by
                  simpa [occurrence, ColumnBundle.ids] using rightMember
                simpa only [binaryOperand_ids] using
                  List.mem_append_right
                    (firstBinaryOperand frame.operands).ids secondMember
              have contextMember :=
                RefBundles.fromSchema_ids_subset
                  (Refs.cons running (Refs.cons witness .nil))
                  frame.contextBundles column operandMember
              simp [CallFrame.visibleIds, contextMember]
            · subst column
              have outputMember :
                  (occurrence parameters profile frame).output.id ∈
                    frame.outputs.ids := by
                simpa [occurrence] using
                  bundleColumn_id_mem (unaryOutput frame.outputs)
                    ⟨0, output_width_positive parameters profile frame⟩
              simp [CallFrame.visibleIds, outputMember]
            · apply List.mem_append_right frame.visibleIds
              rw [temporary_ids_exact parameters profile frame]
              exact List.mem_append_left _ inverse
            · apply List.mem_append_right frame.visibleIds
              rw [temporary_ids_exact parameters profile frame]
              exact List.mem_append_right _ (List.mem_append_left _ equal)
            · apply List.mem_append_right frame.visibleIds
              rw [temporary_ids_exact parameters profile frame]
              exact List.mem_append_right _ (List.mem_append_right _ product)
  · intro context references frame assignment inputs
      constantOne activeOne decoded holds
    cases references with
    | cons runningRef tail =>
        cases tail with
        | cons witnessRef tail =>
            cases tail
            cases inputs with
            | cons running inputs =>
                cases inputs with
                | cons witness inputs =>
                    cases inputs
                    have coordinates :=
                      coordinate_values_of_decodes parameters profile frame
                        assignment running witness decoded
                    have raw :=
                      (occurrence parameters profile frame).active_sound
                        profile.fieldLaws assignment constantOne activeOne
                        holds
                    let checked :=
                      parameters.terminalChecks.runningCheck
                        Step.selected
                        (parameters.setup.verifierKeys Step.selected)
                        running witness
                    have checkExact :
                        checked =
                          decide
                            (profile.codecs.running.encode running =
                              profile.codecs.runningWitness.encode witness) :=
                      profile.runningCheck_exact _ _ _
                    have outputCoordinate :
                        assignment
                            (occurrence parameters profile frame).output.id =
                          if checked then 1 else 0 := by
                      rw [raw, coordinates.1, coordinates.2, checkExact]
                      by_cases equal :
                          profile.codecs.running.encode running =
                            profile.codecs.runningWitness.encode witness
                      · simp [equal]
                      · simp [equal]
                    refine ⟨.cons checked .nil, ?_, ?_⟩
                    · rfl
                    · apply
                        (unaryOutput_decodes_iff profile.family assignment
                          frame.outputs checked).mpr
                      exact output_decodes parameters profile frame assignment
                        checked outputCoordinate
  · intro context references frame assignment inputs outputs
      constantOne activeOne inputsEncoded outputsEncoded evaluated
    cases references with
    | cons runningRef tail =>
        cases tail with
        | cons witnessRef tail =>
            cases tail
            cases inputs with
            | cons running inputs =>
                cases inputs with
                | cons witness inputs =>
                    cases inputs
                    cases outputs with
                    | cons output outputs =>
                        cases outputs
                        let checked :=
                          parameters.terminalChecks.runningCheck
                            Step.selected
                            (parameters.setup.verifierKeys Step.selected)
                            running witness
                        have outputEqual : output = checked :=
                          congrArg HVec.head
                            (Option.some.inj evaluated.symm)
                        subst output
                        have encodedPair :=
                          (binaryOperand_encodes_iff profile.family assignment
                            frame.operands running witness).mp inputsEncoded
                        have coordinates :
                            (occurrence parameters profile frame).left.map
                                  (fun column => assignment column.id) =
                                profile.codecs.running.encode running ∧
                              (occurrence parameters profile frame).right.map
                                  (fun column => assignment column.id) =
                                profile.codecs.runningWitness.encode witness := by
                          constructor
                          · simpa [occurrence, ColumnBundle.values,
                              TerminalEqualityProfile.family, Profile.family,
                              DataCodecs.family, Family.codecFor] using
                              encodedPair.1.2
                          · simpa [occurrence, ColumnBundle.values,
                              TerminalEqualityProfile.family, Profile.family,
                              DataCodecs.family, Family.codecFor] using
                              encodedPair.2.2
                        have outputEncoded :=
                          (unaryOutput_encodes_iff profile.family assignment
                            frame.outputs checked).mp outputsEncoded
                        have outputCoordinate :=
                          output_coordinate_of_encodes parameters profile frame
                            assignment checked outputEncoded
                        let selected := occurrence parameters profile frame
                        let completed :=
                          selected.completion profile.inverseLaw
                            frame.temporaries.ids assignment
                        have agrees :
                            AgreesOn frame.visibleIds assignment completed :=
                          completion_agrees parameters profile frame assignment
                        have changes :
                            ChangesOnly frame.temporaries.ids assignment
                              completed :=
                          selected.completion_changesOnly
                            profile.inverseLaw frame.temporaries.ids assignment
                        have temporaryNodup : frame.temporaries.ids.Nodup :=
                          (List.nodup_append.mp frame.allocationsNodup).2.1
                        have witnessValues :=
                          selected.completion_values profile.inverseLaw
                            frame.temporaries.ids assignment
                            (temporary_ids_exact parameters profile frame)
                            temporaryNodup
                        have preserved :=
                          operand_values_preserved parameters profile frame
                            assignment completed agrees
                        have oneCompleted : completed frame.one = 1 := by
                          rw [agrees frame.one (by
                            simp [CallFrame.visibleIds]), constantOne]
                        have activeCompleted : completed frame.active = 1 := by
                          rw [agrees frame.active (by
                            simp [CallFrame.visibleIds]), activeOne]
                        have outputPreserved :
                            completed selected.output.id =
                              assignment selected.output.id := by
                          apply agrees
                          have outputMember :
                              selected.output.id ∈ frame.outputs.ids := by
                            simpa [selected, occurrence] using
                              bundleColumn_id_mem (unaryOutput frame.outputs)
                                ⟨0, output_width_positive parameters profile
                                  frame⟩
                          simp [CallFrame.visibleIds, outputMember]
                        refine ⟨completed, agrees, changes, ?_⟩
                        apply selected.active_complete profile.inverseLaw
                          completed oneCompleted activeCompleted
                        · rw [preserved.1, preserved.2]
                          simpa [completed] using witnessValues.1
                        · rw [preserved.1, preserved.2]
                          simpa [completed] using witnessValues.2.1
                        · have equalWitness :
                              selected.equals.map
                                  (fun column => completed column.id) =
                                coordinateEqualValues
                                  (selected.left.map
                                    (fun column => assignment column.id))
                                  (selected.right.map
                                    (fun column => assignment column.id)) := by
                            simpa [completed] using witnessValues.2.1
                          rw [equalWitness]
                          simpa [completed] using witnessValues.2.2
                        · rw [outputPreserved, outputCoordinate,
                            preserved.1, preserved.2, coordinates.1,
                            coordinates.2]
                          have checkExact :
                              checked =
                                decide
                                  (profile.codecs.running.encode running =
                                    profile.codecs.runningWitness.encode
                                      witness) :=
                            profile.runningCheck_exact _ _ _
                          change
                            (if checked then 1 else 0) =
                              if profile.codecs.running.encode running =
                                  profile.codecs.runningWitness.encode witness
                                then 1 else 0
                          rw [checkExact]
                          by_cases equal :
                              profile.codecs.running.encode running =
                                profile.codecs.runningWitness.encode witness
                          · simp [equal]
                          · simp [equal]
  · intro context references frame assignment constantOne activeZero
    cases references with
    | cons running tail =>
        cases tail with
        | cons witness tail =>
            cases tail
            let selected := occurrence parameters profile frame
            let completed :=
              selected.completion profile.inverseLaw
                frame.temporaries.ids assignment
            have agrees :
                AgreesOn frame.visibleIds assignment completed :=
              completion_agrees parameters profile frame assignment
            have changes :
                ChangesOnly frame.temporaries.ids assignment completed :=
              selected.completion_changesOnly profile.inverseLaw
                frame.temporaries.ids assignment
            have temporaryNodup : frame.temporaries.ids.Nodup :=
              (List.nodup_append.mp frame.allocationsNodup).2.1
            have witnessValues :=
              selected.completion_values profile.inverseLaw
                frame.temporaries.ids assignment
                (temporary_ids_exact parameters profile frame) temporaryNodup
            have preserved :=
              operand_values_preserved parameters profile frame
                assignment completed agrees
            have oneCompleted : completed frame.one = 1 := by
              rw [agrees frame.one (by
                simp [CallFrame.visibleIds]), constantOne]
            have activeCompleted : completed frame.active = 0 := by
              rw [agrees frame.active (by
                simp [CallFrame.visibleIds]), activeZero]
            refine ⟨completed, agrees, changes, ?_⟩
            apply selected.inactive_complete profile.inverseLaw completed
              oneCompleted activeCompleted
            · rw [preserved.1, preserved.2]
              simpa [completed] using witnessValues.1
            · rw [preserved.1, preserved.2]
              simpa [completed] using witnessValues.2.1
            · have equalWitness :
                  selected.equals.map (fun column => completed column.id) =
                    coordinateEqualValues
                      (selected.left.map
                        (fun column => assignment column.id))
                      (selected.right.map
                        (fun column => assignment column.id)) := by
                simpa [completed] using witnessValues.2.1
              rw [equalWitness]
              simpa [completed] using witnessValues.2.2

end RunningCheckRecipe

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding
