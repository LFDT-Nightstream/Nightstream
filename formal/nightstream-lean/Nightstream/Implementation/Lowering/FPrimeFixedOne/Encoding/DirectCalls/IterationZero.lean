import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls.Footprints

/-!
Contract: exact `iterationZero` call recipe.

The recipe tests the sole canonical bounded-natural coordinate, emits the
three selected zero-test rows, and completes exactly two one-coordinate
temporary bundles.
-/

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls

open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary

private theorem footprint_exact
    (parameters : Parameters)
    (profile : IterationZeroProfile parameters) :
    (signature parameters).callFootprint Call.iterationZero =
      zeroFootprint := by
  simpa [signature, callFootprint] using profile.iterationZeroFootprint

private theorem input_width_one
    (parameters : Parameters)
    (profile : IterationZeroProfile parameters)
    {context : Schema (typeSystem parameters)}
    {reference : Ref (typeSystem parameters) context (.data .nat)}
    (frame :
      CallFrame (signature := signature parameters)
        profile.family Call.iterationZero
        (Refs.cons reference .nil)) :
    reference.port.layout.owners.length = 1 := by
  simpa [IterationZeroProfile.family, Encoding.Profile.family,
    DataCodecs.family, Family.codecFor] using
      frame.operandWidthsAgree.1.symm

private theorem output_width_one
    (parameters : Parameters)
    (profile : IterationZeroProfile parameters)
    {context : Schema (typeSystem parameters)}
    {reference : Ref (typeSystem parameters) context (.data .nat)}
    (frame :
      CallFrame (signature := signature parameters)
        profile.family Call.iterationZero
        (Refs.cons reference .nil)) :
    (Ports.auxiliaryBit parameters).layout.owners.length = 1 := by
  have width :=
    frame.outputWidthsAgree
      (Ports.auxiliaryBit parameters) (by
        change Ports.auxiliaryBit parameters ∈
          callOutputs parameters Call.iterationZero
        exact List.mem_cons_self)
  unfold PortWidthAgrees at width
  simpa [IterationZeroProfile.family, Encoding.Profile.family,
    DataCodecs.family, Family.codecFor] using width.symm

private theorem input_width_positive
    (parameters : Parameters)
    (profile : IterationZeroProfile parameters)
    {context : Schema (typeSystem parameters)}
    {reference : Ref (typeSystem parameters) context (.data .nat)}
    (frame :
      CallFrame (signature := signature parameters)
        profile.family Call.iterationZero
        (Refs.cons reference .nil)) :
    0 < reference.port.layout.owners.length := by
  exact Eq.mpr
    (congrArg (fun width => 0 < width)
      (input_width_one parameters profile frame))
    (by decide)

private theorem output_width_positive
    (parameters : Parameters)
    (profile : IterationZeroProfile parameters)
    {context : Schema (typeSystem parameters)}
    {reference : Ref (typeSystem parameters) context (.data .nat)}
    (frame :
      CallFrame (signature := signature parameters)
        profile.family Call.iterationZero
        (Refs.cons reference .nil)) :
    0 < (Ports.auxiliaryBit parameters).layout.owners.length := by
  exact Eq.mpr
    (congrArg (fun width => 0 < width)
      (output_width_one parameters profile frame))
    (by decide)

private theorem temporary_layouts_exact
    (parameters : Parameters)
    (profile : IterationZeroProfile parameters) :
    ((signature parameters).callFootprint
        Call.iterationZero).temporaries =
      [auxiliaryLayout 1, auxiliaryLayout 1] := by
  rw [footprint_exact parameters profile]
  rfl

private def frameTemporaries
    (parameters : Parameters)
    (profile : IterationZeroProfile parameters)
    {context : Schema (typeSystem parameters)}
    {reference : Ref (typeSystem parameters) context (.data .nat)}
    (frame :
      CallFrame (signature := signature parameters)
        profile.family Call.iterationZero
        (Refs.cons reference .nil)) :
    LayoutBundles [auxiliaryLayout 1, auxiliaryLayout 1] :=
  temporary_layouts_exact parameters profile ▸
    frame.temporaries

private theorem frameTemporaries_ids
    (parameters : Parameters)
    (profile : IterationZeroProfile parameters)
    {context : Schema (typeSystem parameters)}
    {reference : Ref (typeSystem parameters) context (.data .nat)}
    (frame :
      CallFrame (signature := signature parameters)
        profile.family Call.iterationZero
        (Refs.cons reference .nil)) :
    (frameTemporaries parameters profile frame).ids =
      frame.temporaries.ids := by
  exact layoutBundles_ids_cast
    (temporary_layouts_exact parameters profile)
    frame.temporaries

private def frameRecipe
    (parameters : Parameters)
    (profile : IterationZeroProfile parameters)
    {context : Schema (typeSystem parameters)}
    {reference : Ref (typeSystem parameters) context (.data .nat)}
    (frame :
      CallFrame (signature := signature parameters)
        profile.family Call.iterationZero
        (Refs.cons reference .nil)) :
    ZeroRecipe where
  owner := frame.owner
  one := frame.one
  active := frame.active
  input :=
    bundleColumn (unaryOperand frame.operands)
      ⟨0, input_width_positive parameters profile frame⟩
  output :=
    bundleColumn (unaryOutput frame.outputs)
      ⟨0, output_width_positive parameters profile frame⟩
  inverse :=
    bundleColumn
      (firstTemporary (frameTemporaries parameters profile frame))
      ⟨0, by
        unfold auxiliaryLayout ownedLayout
        simp⟩
  equal :=
    bundleColumn
      (secondTemporary (frameTemporaries parameters profile frame))
      ⟨0, by
        unfold auxiliaryLayout ownedLayout
        simp⟩

private theorem temporary_ids_exact
    (parameters : Parameters)
    (profile : IterationZeroProfile parameters)
    {context : Schema (typeSystem parameters)}
    {reference : Ref (typeSystem parameters) context (.data .nat)}
    (frame :
      CallFrame (signature := signature parameters)
        profile.family Call.iterationZero
        (Refs.cons reference .nil)) :
    frame.temporaries.ids =
      [(frameRecipe parameters profile frame).inverse.id,
        (frameRecipe parameters profile frame).equal.id] := by
  rw [← frameTemporaries_ids parameters profile frame,
    twoTemporary_ids]
  rw [bundle_ids_eq_singleton (widthOne := by rfl),
    bundle_ids_eq_singleton (widthOne := by rfl)]
  rfl

private def completion
    (parameters : Parameters)
    (profile : IterationZeroProfile parameters)
    {context : Schema (typeSystem parameters)}
    {reference : Ref (typeSystem parameters) context (.data .nat)}
    (frame :
      CallFrame (signature := signature parameters)
        profile.family Call.iterationZero
        (Refs.cons reference .nil))
    (assignment : ColumnId -> Field) : ColumnId -> Field :=
  let occurrence := frameRecipe parameters profile frame
  writeColumns assignment frame.temporaries.ids
    [coordinateInverseValue profile.inverseLaw
        (assignment occurrence.input.id) 0,
      coordinateEqualValue (assignment occurrence.input.id) 0]

private theorem completion_spec
    (parameters : Parameters)
    (profile : IterationZeroProfile parameters)
    {context : Schema (typeSystem parameters)}
    {reference : Ref (typeSystem parameters) context (.data .nat)}
    (frame :
      CallFrame (signature := signature parameters)
        profile.family Call.iterationZero
        (Refs.cons reference .nil))
    (assignment : ColumnId -> Field) :
    AgreesOn frame.visibleIds assignment
        (completion parameters profile frame assignment) ∧
      ChangesOnly frame.temporaries.ids assignment
        (completion parameters profile frame assignment) ∧
      completion parameters profile frame assignment
          (frameRecipe parameters profile frame).inverse.id =
        coordinateInverseValue profile.inverseLaw
          (assignment
            (frameRecipe parameters profile frame).input.id) 0 ∧
      completion parameters profile frame assignment
          (frameRecipe parameters profile frame).equal.id =
        coordinateEqualValue
          (assignment
            (frameRecipe parameters profile frame).input.id) 0 := by
  let inverseValue :=
    coordinateInverseValue profile.inverseLaw
      (assignment (frameRecipe parameters profile frame).input.id) 0
  let equalValue :=
    coordinateEqualValue
      (assignment (frameRecipe parameters profile frame).input.id) 0
  have temporaryNodup : frame.temporaries.ids.Nodup := by
    exact (List.nodup_append.mp frame.allocationsNodup).2.1
  have recovered :=
    writeColumns_map_eq assignment frame.temporaries.ids
      [inverseValue, equalValue]
      (by
        rw [temporary_ids_exact parameters profile frame]
        rfl)
      temporaryNodup
  have pair :
      completion parameters profile frame assignment
            (frameRecipe parameters profile frame).inverse.id =
          inverseValue ∧
        completion parameters profile frame assignment
            (frameRecipe parameters profile frame).equal.id =
          equalValue := by
    simpa [completion, inverseValue, equalValue,
      temporary_ids_exact parameters profile frame] using recovered
  exact ⟨
    writeColumns_agreesOn assignment frame.temporaries.ids
      frame.visibleIds [inverseValue, equalValue]
      frame.temporariesDisjointVisible,
    writeColumns_changesOnly assignment frame.temporaries.ids
      [inverseValue, equalValue],
    pair.1,
    pair.2⟩

private theorem input_zero_iff
    (parameters : Parameters)
    (profile : IterationZeroProfile parameters)
    {context : Schema (typeSystem parameters)}
    {reference : Ref (typeSystem parameters) context (.data .nat)}
    (frame :
      CallFrame (signature := signature parameters)
        profile.family Call.iterationZero
        (Refs.cons reference .nil))
    (assignment : ColumnId -> Field)
    (iteration : Nat)
    (decoded :
      (unaryOperand frame.operands).Decodes
        profile.family (.data .nat) assignment iteration) :
    assignment (frameRecipe parameters profile frame).input.id = 0 ↔
      iteration = 0 := by
  have coordinates :=
    bundle_values_eq_singleton
      (unaryOperand frame.operands) assignment
      (input_width_one parameters profile frame)
  have decodedCoordinate :
      boundedNatCodec.decode
          [assignment (frameRecipe parameters profile frame).input.id] =
        some iteration := by
    unfold ColumnBundle.Decodes at decoded
    rw [coordinates] at decoded
    simpa [IterationZeroProfile.family, Encoding.Profile.family,
      DataCodecs.family, Family.codecFor, frameRecipe] using decoded
  have representative :=
    (boundedNatCodec_decode_singleton_iff
      (assignment (frameRecipe parameters profile frame).input.id)
      iteration).mp decodedCoordinate
  constructor
  · intro coordinateZero
    calc
      iteration =
          (assignment
            (frameRecipe parameters profile frame).input.id).val :=
        representative.symm
      _ = 0 := by rw [coordinateZero]; rfl
  · intro iterationZero
    apply Fin.ext
    simpa [iterationZero] using representative

private theorem output_decodes
    (parameters : Parameters)
    (profile : IterationZeroProfile parameters)
    {context : Schema (typeSystem parameters)}
    {reference : Ref (typeSystem parameters) context (.data .nat)}
    (frame :
      CallFrame (signature := signature parameters)
        profile.family Call.iterationZero
        (Refs.cons reference .nil))
    (assignment : ColumnId -> Field)
    (value : Bool)
    (coordinate :
      assignment (frameRecipe parameters profile frame).output.id =
        if value then 1 else 0) :
    (unaryOutput frame.outputs).Decodes
      profile.family .bit assignment value := by
  unfold ColumnBundle.Decodes
  rw [bundle_values_eq_singleton
    (unaryOutput frame.outputs) assignment
    (output_width_one parameters profile frame)]
  change
    boolCodec.decode
        [assignment (frameRecipe parameters profile frame).output.id] =
      some value
  cases value with
  | false =>
      exact (boolCodec_decode_false_iff _).mpr (by simpa using coordinate)
  | true =>
      exact (boolCodec_decode_true_iff _).mpr (by simpa using coordinate)

private theorem output_coordinate_of_encodes
    (parameters : Parameters)
    (profile : IterationZeroProfile parameters)
    {context : Schema (typeSystem parameters)}
    {reference : Ref (typeSystem parameters) context (.data .nat)}
    (frame :
      CallFrame (signature := signature parameters)
        profile.family Call.iterationZero
        (Refs.cons reference .nil))
    (assignment : ColumnId -> Field)
    (value : Bool)
    (encoded :
      (unaryOutput frame.outputs).Encodes
        profile.family .bit assignment value) :
    assignment (frameRecipe parameters profile frame).output.id =
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
        [assignment (frameRecipe parameters profile frame).output.id] =
      some value at decoded
  cases value with
  | false =>
      exact (boolCodec_decode_false_iff _).mp decoded
  | true =>
      exact (boolCodec_decode_true_iff _).mp decoded

/-- Certified physical recipe for the exact direct `iterationZero` call,
using only the profile fields that this call consumes. -/
def iterationZeroRecipeForProfile
    (parameters : Parameters)
    (profile : IterationZeroProfile parameters) :
    CallRecipe (signature parameters) profile.family
      Call.iterationZero := by
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
    | cons reference tail =>
        cases tail
        exact (frameRecipe parameters profile frame).rows
  · intro context references frame
    cases references with
    | cons reference tail =>
        cases tail
        rw [footprint_exact parameters profile]
        exact (frameRecipe parameters profile frame).row_count
  · intro context references frame row member
    cases references with
    | cons reference tail =>
        cases tail
        exact (frameRecipe parameters profile frame).rows_owned row member
  · intro context references frame
    cases references with
    | cons reference tail =>
        cases tail
        exact (frameRecipe parameters profile frame).row_ids_nodup
  · intro context references frame row member column columnMember
    cases references with
    | cons reference tail =>
        cases tail
        rcases
            (frameRecipe parameters profile frame).rows_supported
              row member column columnMember with
          one | active | input | output | inverse | equal
        · subst column
          simp [frameRecipe, CallFrame.visibleIds]
        · subst column
          simp [frameRecipe, CallFrame.visibleIds]
        · subst column
          have operandMember :
              (frameRecipe parameters profile frame).input.id ∈
                frame.operands.ids := by
            simpa [frameRecipe] using
              bundleColumn_id_mem (unaryOperand frame.operands)
                ⟨0, input_width_positive parameters profile frame⟩
          have contextMember :=
            RefBundles.fromSchema_ids_subset
              (Refs.cons reference .nil) frame.contextBundles
              _ operandMember
          simp [CallFrame.visibleIds, contextMember]
        · subst column
          have outputMember :
              (frameRecipe parameters profile frame).output.id ∈
                frame.outputs.ids := by
            simpa [frameRecipe] using
              bundleColumn_id_mem (unaryOutput frame.outputs)
                ⟨0, output_width_positive parameters profile frame⟩
          simp [CallFrame.visibleIds, outputMember]
        · subst column
          apply List.mem_append_right frame.visibleIds
          rw [temporary_ids_exact parameters profile frame]
          simp
        · subst column
          apply List.mem_append_right frame.visibleIds
          rw [temporary_ids_exact parameters profile frame]
          simp
  · intro context references frame assignment inputs
      constantOne activeOne decoded holds
    cases references with
    | cons reference tail =>
        cases tail
        cases inputs with
        | cons iteration inputs =>
            cases inputs
            change Nat at iteration
            have inputDecoded :=
              (unaryOperand_decodes_iff profile.family assignment
                frame.operands iteration).mp decoded
            have zeroIff :=
              input_zero_iff parameters profile frame assignment
                iteration inputDecoded
            have coordinate :=
              (frameRecipe parameters profile frame).active_sound
                profile.fieldLaws assignment constantOne activeOne holds
            have outputCoordinate :
                assignment
                    (frameRecipe parameters profile frame).output.id =
                  if decide (iteration = 0) then 1 else 0 := by
              rw [coordinate]
              by_cases iterationZero : iteration = 0
              · rw [if_pos (zeroIff.mpr iterationZero)]
                simp [iterationZero]
              · rw [if_neg (fun coordinateZero =>
                    iterationZero (zeroIff.mp coordinateZero))]
                simp [iterationZero]
            refine ⟨.cons (decide (iteration = 0)) .nil, rfl, ?_⟩
            apply (unaryOutput_decodes_iff profile.family assignment
              frame.outputs (decide (iteration = 0))).mpr
            exact output_decodes parameters profile frame assignment
              (decide (iteration = 0)) outputCoordinate
  · intro context references frame assignment inputs outputs
      constantOne activeOne inputsEncoded outputsEncoded evaluated
    cases references with
    | cons reference tail =>
        cases tail
        cases inputs with
        | cons iteration inputs =>
            cases inputs
            cases outputs with
            | cons output outputs =>
                cases outputs
                change Nat at iteration
                change Bool at output
                have outputEqual :
                    output = decide (iteration = 0) := by
                  exact congrArg HVec.head
                    (Option.some.inj evaluated.symm)
                subst output
                have inputEncoded :=
                  (unaryOperand_encodes_iff profile.family assignment
                    frame.operands iteration).mp inputsEncoded
                have inputDecoded :=
                  (unaryOperand frame.operands).decodes_of_encodes
                    profile.family (.data .nat) assignment iteration
                    inputEncoded
                have zeroIff :=
                  input_zero_iff parameters profile frame assignment
                    iteration inputDecoded
                have outputEncoded :=
                  (unaryOutput_encodes_iff profile.family assignment
                    frame.outputs (decide (iteration = 0))).mp
                    outputsEncoded
                have outputCoordinate :=
                  output_coordinate_of_encodes parameters profile frame
                    assignment (decide (iteration = 0)) outputEncoded
                rcases completion_spec parameters profile frame assignment
                    with
                  ⟨agrees, changes, inverseValue, equalValue⟩
                let completed :=
                  completion parameters profile frame assignment
                have oneCompleted : completed frame.one = 1 := by
                  change
                    completion parameters profile frame assignment
                        frame.one =
                      1
                  rw [agrees frame.one (by
                    simp [CallFrame.visibleIds]), constantOne]
                have activeCompleted : completed frame.active = 1 := by
                  change
                    completion parameters profile frame assignment
                        frame.active =
                      1
                  rw [agrees frame.active (by
                    simp [CallFrame.visibleIds]), activeOne]
                have inputPreserved :
                    completion parameters profile frame assignment
                        (frameRecipe parameters profile frame).input.id =
                      assignment
                        (frameRecipe parameters profile frame).input.id := by
                  apply agrees
                  have operandMember :
                      (frameRecipe parameters profile frame).input.id ∈
                        frame.operands.ids := by
                    simpa [frameRecipe] using
                      bundleColumn_id_mem (unaryOperand frame.operands)
                        ⟨0,
                          input_width_positive parameters profile frame⟩
                  have contextMember :=
                    RefBundles.fromSchema_ids_subset
                    (Refs.cons reference .nil) frame.contextBundles
                    _ operandMember
                  simp [CallFrame.visibleIds, contextMember]
                have outputPreserved :
                    completion parameters profile frame assignment
                        (frameRecipe parameters profile frame).output.id =
                      assignment
                        (frameRecipe parameters profile frame).output.id := by
                  apply agrees
                  have outputMember :
                      (frameRecipe parameters profile frame).output.id ∈
                        frame.outputs.ids := by
                    simpa [frameRecipe] using
                      bundleColumn_id_mem (unaryOutput frame.outputs)
                        ⟨0,
                          output_width_positive parameters profile frame⟩
                  simp [CallFrame.visibleIds, outputMember]
                refine ⟨completed, agrees, changes, ?_⟩
                apply (frameRecipe parameters profile frame).complete
                  profile.inverseLaw completed oneCompleted
                  (Or.inl activeCompleted)
                · change
                    completion parameters profile frame assignment
                          (frameRecipe parameters profile frame).inverse.id =
                      coordinateInverseValue profile.inverseLaw
                        (completion parameters profile frame assignment
                          (frameRecipe parameters profile frame).input.id) 0
                  rw [inverseValue, inputPreserved]
                · change
                    completion parameters profile frame assignment
                          (frameRecipe parameters profile frame).equal.id =
                      coordinateEqualValue
                        (completion parameters profile frame assignment
                          (frameRecipe parameters profile frame).input.id) 0
                  rw [equalValue, inputPreserved]
                · intro _
                  change
                    completion parameters profile frame assignment
                          (frameRecipe parameters profile frame).output.id =
                      if completion parameters profile frame assignment
                          (frameRecipe parameters profile frame).input.id = 0
                      then 1 else 0
                  rw [outputPreserved, inputPreserved, outputCoordinate]
                  by_cases iterationZero : iteration = 0
                  · rw [if_pos (zeroIff.mpr iterationZero)]
                    simp [iterationZero]
                  · rw [if_neg (fun coordinateZero =>
                        iterationZero (zeroIff.mp coordinateZero))]
                    simp [iterationZero]
  · intro context references frame assignment constantOne activeZero
    cases references with
    | cons reference tail =>
        cases tail
        rcases completion_spec parameters profile frame assignment with
          ⟨agrees, changes, inverseValue, equalValue⟩
        let completed := completion parameters profile frame assignment
        have oneCompleted : completed frame.one = 1 := by
          change
            completion parameters profile frame assignment frame.one = 1
          rw [agrees frame.one (by
            simp [CallFrame.visibleIds]), constantOne]
        have activeCompleted : completed frame.active = 0 := by
          change
            completion parameters profile frame assignment frame.active = 0
          rw [agrees frame.active (by
            simp [CallFrame.visibleIds]), activeZero]
        have inputPreserved :
            completion parameters profile frame assignment
                (frameRecipe parameters profile frame).input.id =
              assignment
                (frameRecipe parameters profile frame).input.id := by
          apply agrees
          have operandMember :
              (frameRecipe parameters profile frame).input.id ∈
                frame.operands.ids := by
            simpa [frameRecipe] using
              bundleColumn_id_mem (unaryOperand frame.operands)
                ⟨0, input_width_positive parameters profile frame⟩
          have contextMember :=
            RefBundles.fromSchema_ids_subset
            (Refs.cons reference .nil) frame.contextBundles _ operandMember
          simp [CallFrame.visibleIds, contextMember]
        refine ⟨completed, agrees, changes, ?_⟩
        apply (frameRecipe parameters profile frame).complete
          profile.inverseLaw completed oneCompleted
          (Or.inr activeCompleted)
        · change
            completion parameters profile frame assignment
                  (frameRecipe parameters profile frame).inverse.id =
              coordinateInverseValue profile.inverseLaw
                (completion parameters profile frame assignment
                  (frameRecipe parameters profile frame).input.id) 0
          rw [inverseValue, inputPreserved]
        · change
            completion parameters profile frame assignment
                  (frameRecipe parameters profile frame).equal.id =
              coordinateEqualValue
                (completion parameters profile frame assignment
                  (frameRecipe parameters profile frame).input.id) 0
          rw [equalValue, inputPreserved]
        · intro activeImpossible
          have activeOneCompleted : completed frame.active = 1 := by
            simpa [frameRecipe] using activeImpossible
          rw [activeCompleted] at activeOneCompleted
          exact False.elim (by
            have zeroNeOne : (0 : Field) ≠ 1 := by decide
            exact zeroNeOne activeOneCompleted)

/-- Compatibility wrapper for a complete direct-call profile. -/
def iterationZeroRecipe
    (parameters : Parameters)
    (profile : DirectProfile parameters) :
    CallRecipe (signature parameters) profile.family
      Call.iterationZero :=
  iterationZeroRecipeForProfile parameters
    (profile.iterationZeroProfile parameters)

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls
