import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls.Footprints

/-!
Contract: exact `encodedEqual` call recipe.

The recipe compares the two canonical encoded coordinate strings, emits the
selected equality rows, and completes exactly the declared inverse, equality,
and product-chain temporary bundles.
-/

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls

open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary

private theorem footprint_exact
    (parameters : Parameters)
    (profile : EncodedEqualProfile parameters) :
    (signature parameters).callFootprint Call.encodedEqual =
      equalityFootprint profile.codecs.encoded.width := by
  simpa [signature, callFootprint] using profile.encodedEqualFootprint

private theorem temporary_layouts_exact
    (parameters : Parameters)
    (profile : EncodedEqualProfile parameters) :
    ((signature parameters).callFootprint Call.encodedEqual).temporaries =
      [auxiliaryLayout profile.codecs.encoded.width,
        auxiliaryLayout profile.codecs.encoded.width,
        auxiliaryLayout profile.codecs.encoded.width.pred] := by
  rw [footprint_exact parameters profile]
  rfl

private def frameTemporaries
    (parameters : Parameters)
    (profile : EncodedEqualProfile parameters)
    {context : Schema (typeSystem parameters)}
    {left right :
      Ref (typeSystem parameters) context (.data .encoded)}
    (frame :
      CallFrame (signature := signature parameters)
        profile.family Call.encodedEqual
        (Refs.cons left (Refs.cons right .nil))) :
    LayoutBundles
      [auxiliaryLayout profile.codecs.encoded.width,
        auxiliaryLayout profile.codecs.encoded.width,
        auxiliaryLayout profile.codecs.encoded.width.pred] :=
  temporary_layouts_exact parameters profile ▸ frame.temporaries

private theorem frameTemporaries_ids
    (parameters : Parameters)
    (profile : EncodedEqualProfile parameters)
    {context : Schema (typeSystem parameters)}
    {left right :
      Ref (typeSystem parameters) context (.data .encoded)}
    (frame :
      CallFrame (signature := signature parameters)
        profile.family Call.encodedEqual
        (Refs.cons left (Refs.cons right .nil))) :
    (frameTemporaries parameters profile frame).ids =
      frame.temporaries.ids :=
  layoutBundles_ids_cast
    (temporary_layouts_exact parameters profile) frame.temporaries

private theorem output_width_one
    (parameters : Parameters)
    (profile : EncodedEqualProfile parameters)
    {context : Schema (typeSystem parameters)}
    {left right :
      Ref (typeSystem parameters) context (.data .encoded)}
    (frame :
      CallFrame (signature := signature parameters)
        profile.family Call.encodedEqual
        (Refs.cons left (Refs.cons right .nil))) :
    (Ports.auxiliaryBit parameters).layout.owners.length = 1 := by
  have width :=
    frame.outputWidthsAgree (Ports.auxiliaryBit parameters) (by
      change Ports.auxiliaryBit parameters ∈
        callOutputs parameters Call.encodedEqual
      exact List.mem_cons_self)
  unfold PortWidthAgrees at width
  simpa [EncodedEqualProfile.family, Encoding.Profile.family,
    DataCodecs.family, Family.codecFor] using width.symm

private theorem output_width_positive
    (parameters : Parameters)
    (profile : EncodedEqualProfile parameters)
    {context : Schema (typeSystem parameters)}
    {left right :
      Ref (typeSystem parameters) context (.data .encoded)}
    (frame :
      CallFrame (signature := signature parameters)
        profile.family Call.encodedEqual
        (Refs.cons left (Refs.cons right .nil))) :
    0 < (Ports.auxiliaryBit parameters).layout.owners.length :=
  Eq.mpr
    (congrArg (fun width => 0 < width)
      (output_width_one parameters profile frame))
    (by decide)

private def frameRecipe
    (parameters : Parameters)
    (profile : EncodedEqualProfile parameters)
    {context : Schema (typeSystem parameters)}
    {left right :
      Ref (typeSystem parameters) context (.data .encoded)}
    (frame :
      CallFrame (signature := signature parameters)
        profile.family Call.encodedEqual
        (Refs.cons left (Refs.cons right .nil))) :
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
        profile.codecs.encoded.width =
          left.port.layout.owners.length := by
      simpa [EncodedEqualProfile.family, Encoding.Profile.family,
        DataCodecs.family, Family.codecFor] using leftWidth
    have rightExact :
        profile.codecs.encoded.width =
          right.port.layout.owners.length := by
      simpa [EncodedEqualProfile.family, Encoding.Profile.family,
        DataCodecs.family, Family.codecFor] using rightWidth
    omega
  inverseLength := by
    rw [ColumnBundle.length_eq, ColumnBundle.length_eq]
    unfold auxiliaryLayout ownedLayout
    simp
    have leftWidth := frame.operandWidthsAgree.1
    simpa [EncodedEqualProfile.family, Encoding.Profile.family,
      DataCodecs.family, Family.codecFor] using leftWidth
  equalLength := by
    rw [ColumnBundle.length_eq, ColumnBundle.length_eq]
    unfold auxiliaryLayout ownedLayout
    simp
    have leftWidth := frame.operandWidthsAgree.1
    simpa [EncodedEqualProfile.family, Encoding.Profile.family,
      DataCodecs.family, Family.codecFor] using leftWidth
  productLength := by
    rw [ColumnBundle.length_eq, ColumnBundle.length_eq]
    unfold auxiliaryLayout ownedLayout
    simp
    have leftWidth := frame.operandWidthsAgree.1
    simpa [EncodedEqualProfile.family, Encoding.Profile.family,
      DataCodecs.family, Family.codecFor] using
        congrArg Nat.pred leftWidth

private theorem temporary_ids_exact
    (parameters : Parameters)
    (profile : EncodedEqualProfile parameters)
    {context : Schema (typeSystem parameters)}
    {left right :
      Ref (typeSystem parameters) context (.data .encoded)}
    (frame :
      CallFrame (signature := signature parameters)
        profile.family Call.encodedEqual
        (Refs.cons left (Refs.cons right .nil))) :
    frame.temporaries.ids =
      (frameRecipe parameters profile frame).inverses.map
          (fun column => column.id) ++
        ((frameRecipe parameters profile frame).equals.map
            (fun column => column.id) ++
          (frameRecipe parameters profile frame).products.map
            (fun column => column.id)) := by
  rw [← frameTemporaries_ids parameters profile frame,
    threeTemporary_ids]
  simp [frameRecipe, ColumnBundle.ids, List.append_assoc]

private theorem semantic_equality_iff
    (parameters : Parameters)
    (profile : EncodedEqualProfile parameters)
    {context : Schema (typeSystem parameters)}
    {leftRef rightRef :
      Ref (typeSystem parameters) context (.data .encoded)}
    (frame :
      CallFrame (signature := signature parameters)
        profile.family Call.encodedEqual
        (Refs.cons leftRef (Refs.cons rightRef .nil)))
    (assignment : ColumnId -> Field)
    (left right : parameters.Encoded)
    (leftDecoded :
      (firstBinaryOperand frame.operands).Decodes
        profile.family (.data .encoded) assignment left)
    (rightDecoded :
      (secondBinaryOperand frame.operands).Decodes
        profile.family (.data .encoded) assignment right) :
    (firstBinaryOperand frame.operands).values assignment =
        (secondBinaryOperand frame.operands).values assignment ↔
      left = right := by
  have leftExact :
      profile.codecs.encoded.decode
          ((firstBinaryOperand frame.operands).values assignment) =
        some left := by
    simpa [EncodedEqualProfile.family, Encoding.Profile.family,
      DataCodecs.family, Family.codecFor] using leftDecoded
  have rightExact :
      profile.codecs.encoded.decode
          ((secondBinaryOperand frame.operands).values assignment) =
        some right := by
    simpa [EncodedEqualProfile.family, Encoding.Profile.family,
      DataCodecs.family, Family.codecFor] using rightDecoded
  constructor
  · intro coordinatesEqual
    apply profile.codecs.encoded.decoded_value_unique leftExact
    rw [coordinatesEqual]
    exact rightExact
  · intro valuesEqual
    have leftEncoding :=
      (profile.codecs.encoded.encode_decode _ _ leftExact).2
    have rightEncoding :=
      (profile.codecs.encoded.encode_decode _ _ rightExact).2
    calc
      (firstBinaryOperand frame.operands).values assignment =
          profile.codecs.encoded.encode left :=
        leftEncoding.symm
      _ = profile.codecs.encoded.encode right := by rw [valuesEqual]
      _ = (secondBinaryOperand frame.operands).values assignment :=
        rightEncoding

private theorem recipe_equality_iff
    (parameters : Parameters)
    (profile : EncodedEqualProfile parameters)
    {context : Schema (typeSystem parameters)}
    {leftRef rightRef :
      Ref (typeSystem parameters) context (.data .encoded)}
    (frame :
      CallFrame (signature := signature parameters)
        profile.family Call.encodedEqual
        (Refs.cons leftRef (Refs.cons rightRef .nil)))
    (assignment : ColumnId -> Field)
    (left right : parameters.Encoded)
    (leftDecoded :
      (firstBinaryOperand frame.operands).Decodes
        profile.family (.data .encoded) assignment left)
    (rightDecoded :
      (secondBinaryOperand frame.operands).Decodes
        profile.family (.data .encoded) assignment right) :
    (frameRecipe parameters profile frame).left.map
          (fun column => assignment column.id) =
        (frameRecipe parameters profile frame).right.map
          (fun column => assignment column.id) ↔
      left = right := by
  simpa [frameRecipe, ColumnBundle.values] using
    semantic_equality_iff parameters profile frame assignment left right
      leftDecoded rightDecoded

private theorem output_decodes
    (parameters : Parameters)
    (profile : EncodedEqualProfile parameters)
    {context : Schema (typeSystem parameters)}
    {left right :
      Ref (typeSystem parameters) context (.data .encoded)}
    (frame :
      CallFrame (signature := signature parameters)
        profile.family Call.encodedEqual
        (Refs.cons left (Refs.cons right .nil)))
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
    (profile : EncodedEqualProfile parameters)
    {context : Schema (typeSystem parameters)}
    {left right :
      Ref (typeSystem parameters) context (.data .encoded)}
    (frame :
      CallFrame (signature := signature parameters)
        profile.family Call.encodedEqual
        (Refs.cons left (Refs.cons right .nil)))
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

private theorem completion_agrees
    (parameters : Parameters)
    (profile : EncodedEqualProfile parameters)
    {context : Schema (typeSystem parameters)}
    {left right :
      Ref (typeSystem parameters) context (.data .encoded)}
    (frame :
      CallFrame (signature := signature parameters)
        profile.family Call.encodedEqual
        (Refs.cons left (Refs.cons right .nil)))
    (assignment : ColumnId -> Field) :
    AgreesOn frame.visibleIds assignment
      ((frameRecipe parameters profile frame).completion
        profile.inverseLaw frame.temporaries.ids assignment) :=
  writeColumns_agreesOn assignment frame.temporaries.ids frame.visibleIds
    ((frameRecipe parameters profile frame).witnessValues
      profile.inverseLaw assignment)
    frame.temporariesDisjointVisible

private theorem recipe_values_preserved
    (parameters : Parameters)
    (profile : EncodedEqualProfile parameters)
    {context : Schema (typeSystem parameters)}
    {left right :
      Ref (typeSystem parameters) context (.data .encoded)}
    (frame :
      CallFrame (signature := signature parameters)
        profile.family Call.encodedEqual
        (Refs.cons left (Refs.cons right .nil)))
    (assignment completed : ColumnId -> Field)
    (agrees : AgreesOn frame.visibleIds assignment completed) :
    (frameRecipe parameters profile frame).left.map
          (fun column => completed column.id) =
        (frameRecipe parameters profile frame).left.map
          (fun column => assignment column.id) ∧
      (frameRecipe parameters profile frame).right.map
          (fun column => completed column.id) =
        (frameRecipe parameters profile frame).right.map
          (fun column => assignment column.id) := by
  constructor
  · apply List.map_congr_left
    intro column member
    apply agrees
    have operandMember : column.id ∈ frame.operands.ids := by
      have firstMember :
          column.id ∈ (firstBinaryOperand frame.operands).ids := by
        unfold ColumnBundle.ids
        exact List.mem_map.mpr ⟨column, by
          simpa [frameRecipe] using member, rfl⟩
      have joined :=
        List.mem_append_left
          (secondBinaryOperand frame.operands).ids firstMember
      simpa only [binaryOperand_ids] using joined
    have contextMember :=
      RefBundles.fromSchema_ids_subset
        (Refs.cons left (Refs.cons right .nil)) frame.contextBundles
        column.id operandMember
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
          simpa [frameRecipe] using member, rfl⟩
      have joined :=
        List.mem_append_right
          (firstBinaryOperand frame.operands).ids secondMember
      simpa only [binaryOperand_ids] using joined
    have contextMember :=
      RefBundles.fromSchema_ids_subset
        (Refs.cons left (Refs.cons right .nil)) frame.contextBundles
        column.id operandMember
    change column.id ∈
      [frame.one, frame.active] ++
        frame.contextBundles.ids ++ frame.outputs.ids
    exact List.mem_append_left frame.outputs.ids
      (List.mem_append_right [frame.one, frame.active] contextMember)

/-- Certified physical recipe for the exact direct `encodedEqual` call,
using only the profile fields consumed by encoded equality. -/
def encodedEqualRecipeForProfile
    (parameters : Parameters)
    (profile : EncodedEqualProfile parameters) :
    CallRecipe (signature parameters) profile.family Call.encodedEqual := by
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
    | cons left tail =>
        cases tail with
        | cons right tail =>
            cases tail
            exact (frameRecipe parameters profile frame).rows
  · intro context references frame
    cases references with
    | cons left tail =>
        cases tail with
        | cons right tail =>
            cases tail
            rw [footprint_exact parameters profile]
            have leftWidth := frame.operandWidthsAgree.1
            have leftLength :
                (frameRecipe parameters profile frame).left.length =
                  profile.codecs.encoded.width := by
              rw [frameRecipe, ColumnBundle.length_eq]
              simpa [EncodedEqualProfile.family, Encoding.Profile.family,
                DataCodecs.family, Family.codecFor] using leftWidth.symm
            change
              (frameRecipe parameters profile frame).rows.length =
                (equalityFootprint profile.codecs.encoded.width).recurringRows
            rw [(frameRecipe parameters profile frame).row_count, leftLength]
            rfl
  · intro context references frame row member
    cases references with
    | cons left tail =>
        cases tail with
        | cons right tail =>
            cases tail
            exact (frameRecipe parameters profile frame).rows_owned row member
  · intro context references frame
    cases references with
    | cons left tail =>
        cases tail with
        | cons right tail =>
            cases tail
            exact (frameRecipe parameters profile frame).row_ids_nodup
  · intro context references frame row member column columnMember
    cases references with
    | cons left tail =>
        cases tail with
        | cons right tail =>
            cases tail
            rcases
                (frameRecipe parameters profile frame).rows_supported
                  row member column columnMember with
              one | active | leftMember | rightMember | output |
                inverse | equal | product
            · subst column
              simp [frameRecipe, CallFrame.visibleIds]
            · subst column
              simp [frameRecipe, CallFrame.visibleIds]
            · have operandMember : column ∈ frame.operands.ids := by
                have firstMember :
                    column ∈ (firstBinaryOperand frame.operands).ids := by
                  simpa [frameRecipe, ColumnBundle.ids] using leftMember
                have joined :=
                  List.mem_append_left
                    (secondBinaryOperand frame.operands).ids firstMember
                simpa only [binaryOperand_ids] using joined
              have contextMember :=
                RefBundles.fromSchema_ids_subset
                  (Refs.cons left (Refs.cons right .nil))
                  frame.contextBundles column operandMember
              change column ∈
                ([frame.one, frame.active] ++
                  frame.contextBundles.ids ++ frame.outputs.ids) ++
                    frame.temporaries.ids
              exact List.mem_append_left frame.temporaries.ids
                (List.mem_append_left frame.outputs.ids
                  (List.mem_append_right
                    [frame.one, frame.active] contextMember))
            · have operandMember : column ∈ frame.operands.ids := by
                have secondMember :
                    column ∈ (secondBinaryOperand frame.operands).ids := by
                  simpa [frameRecipe, ColumnBundle.ids] using rightMember
                have joined :=
                  List.mem_append_right
                    (firstBinaryOperand frame.operands).ids secondMember
                simpa only [binaryOperand_ids] using joined
              have contextMember :=
                RefBundles.fromSchema_ids_subset
                  (Refs.cons left (Refs.cons right .nil))
                  frame.contextBundles column operandMember
              change column ∈
                ([frame.one, frame.active] ++
                  frame.contextBundles.ids ++ frame.outputs.ids) ++
                    frame.temporaries.ids
              exact List.mem_append_left frame.temporaries.ids
                (List.mem_append_left frame.outputs.ids
                  (List.mem_append_right
                    [frame.one, frame.active] contextMember))
            · subst column
              have outputMember :
                  (frameRecipe parameters profile frame).output.id ∈
                    frame.outputs.ids := by
                simpa [frameRecipe] using
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
    | cons leftRef tail =>
        cases tail with
        | cons rightRef tail =>
            cases tail
            cases inputs with
            | cons left inputs =>
                cases inputs with
                | cons right inputs =>
                    cases inputs
                    change parameters.Encoded at left right
                    have decodedPair :=
                      (binaryOperand_decodes_iff profile.family assignment
                        frame.operands left right).mp decoded
                    have equalityIff :=
                      recipe_equality_iff parameters profile frame
                        assignment left right decodedPair.1 decodedPair.2
                    have raw :=
                      (frameRecipe parameters profile frame).active_sound
                        profile.fieldLaws assignment constantOne activeOne
                        holds
                    have outputCoordinate :
                        assignment
                            (frameRecipe parameters profile frame).output.id =
                          if encodedEqual parameters left right then 1 else 0 := by
                      rw [raw]
                      by_cases same : left = right
                      · rw [if_pos (equalityIff.mpr same)]
                        simp [encodedEqual, same]
                      · rw [if_neg (fun equal => same (equalityIff.mp equal))]
                        simp [encodedEqual, same]
                    refine
                      ⟨.cons (encodedEqual parameters left right) .nil,
                        rfl, ?_⟩
                    apply (unaryOutput_decodes_iff profile.family assignment
                      frame.outputs (encodedEqual parameters left right)).mpr
                    exact output_decodes parameters profile frame assignment
                      (encodedEqual parameters left right) outputCoordinate
  · intro context references frame assignment inputs outputs
      constantOne activeOne inputsEncoded outputsEncoded evaluated
    cases references with
    | cons leftRef tail =>
        cases tail with
        | cons rightRef tail =>
            cases tail
            cases inputs with
            | cons left inputs =>
                cases inputs with
                | cons right inputs =>
                    cases inputs
                    cases outputs with
                    | cons output outputs =>
                        cases outputs
                        change parameters.Encoded at left right
                        change Bool at output
                        have outputEqual :
                            output = encodedEqual parameters left right := by
                          exact congrArg HVec.head
                            (Option.some.inj evaluated.symm)
                        subst output
                        have encodedPair :=
                          (binaryOperand_encodes_iff profile.family assignment
                            frame.operands left right).mp inputsEncoded
                        have decodedPair := And.intro
                          ((firstBinaryOperand frame.operands
                            ).decodes_of_encodes profile.family
                              (.data .encoded) assignment left encodedPair.1)
                          ((secondBinaryOperand frame.operands
                            ).decodes_of_encodes profile.family
                              (.data .encoded) assignment right encodedPair.2)
                        have equalityIff :=
                          recipe_equality_iff parameters profile frame
                            assignment left right decodedPair.1 decodedPair.2
                        have outputEncoded :=
                          (unaryOutput_encodes_iff profile.family assignment
                            frame.outputs
                            (encodedEqual parameters left right)).mp
                            outputsEncoded
                        have outputCoordinate :=
                          output_coordinate_of_encodes parameters profile frame
                            assignment (encodedEqual parameters left right)
                            outputEncoded
                        let occurrence := frameRecipe parameters profile frame
                        let completed :=
                          occurrence.completion profile.inverseLaw
                            frame.temporaries.ids assignment
                        have agrees :
                            AgreesOn frame.visibleIds assignment completed := by
                          exact completion_agrees parameters profile frame
                            assignment
                        have changes :
                            ChangesOnly frame.temporaries.ids assignment
                              completed :=
                          occurrence.completion_changesOnly
                            profile.inverseLaw frame.temporaries.ids assignment
                        have temporaryNodup : frame.temporaries.ids.Nodup :=
                          (List.nodup_append.mp frame.allocationsNodup).2.1
                        have witnessValues :=
                          occurrence.completion_values profile.inverseLaw
                            frame.temporaries.ids assignment
                            (temporary_ids_exact parameters profile frame)
                            temporaryNodup
                        have preserved :=
                          recipe_values_preserved parameters profile frame
                            assignment completed agrees
                        have leftPreserved :
                            occurrence.left.map
                                  (fun column => completed column.id) =
                              occurrence.left.map
                                  (fun column => assignment column.id) := by
                          simpa [occurrence] using preserved.1
                        have rightPreserved :
                            occurrence.right.map
                                  (fun column => completed column.id) =
                              occurrence.right.map
                                  (fun column => assignment column.id) := by
                          simpa [occurrence] using preserved.2
                        have inverseWitness :
                            occurrence.inverses.map
                                  (fun column => completed column.id) =
                              coordinateInverseValues profile.inverseLaw
                                (occurrence.left.map
                                  (fun column => assignment column.id))
                                (occurrence.right.map
                                  (fun column => assignment column.id)) := by
                          simpa [completed] using witnessValues.1
                        have equalWitness :
                            occurrence.equals.map
                                  (fun column => completed column.id) =
                              coordinateEqualValues
                                (occurrence.left.map
                                  (fun column => assignment column.id))
                                (occurrence.right.map
                                  (fun column => assignment column.id)) := by
                          simpa [completed] using witnessValues.2.1
                        have productWitness :
                            occurrence.products.map
                                  (fun column => completed column.id) =
                              productValues
                                (coordinateEqualValues
                                  (occurrence.left.map
                                    (fun column => assignment column.id))
                                  (occurrence.right.map
                                    (fun column => assignment column.id))) := by
                          simpa [completed] using witnessValues.2.2
                        have oneCompleted : completed frame.one = 1 := by
                          rw [agrees frame.one (by
                            simp [CallFrame.visibleIds]), constantOne]
                        have activeCompleted : completed frame.active = 1 := by
                          rw [agrees frame.active (by
                            simp [CallFrame.visibleIds]), activeOne]
                        have outputPreserved :
                            completed occurrence.output.id =
                              assignment occurrence.output.id := by
                          apply agrees
                          have outputMember :
                              occurrence.output.id ∈ frame.outputs.ids := by
                            simpa [occurrence, frameRecipe] using
                              bundleColumn_id_mem (unaryOutput frame.outputs)
                                ⟨0, output_width_positive parameters profile
                                  frame⟩
                          simp [CallFrame.visibleIds, outputMember]
                        refine ⟨completed, agrees, changes, ?_⟩
                        apply occurrence.active_complete profile.inverseLaw
                          completed oneCompleted activeCompleted
                        · rw [leftPreserved, rightPreserved]
                          exact inverseWitness
                        · rw [leftPreserved, rightPreserved]
                          exact equalWitness
                        · rw [equalWitness]
                          exact productWitness
                        · rw [outputPreserved, outputCoordinate,
                            leftPreserved, rightPreserved]
                          by_cases same : left = right
                          · rw [if_pos (equalityIff.mpr same)]
                            simp [encodedEqual, same]
                          · rw [if_neg (fun equal =>
                              same (equalityIff.mp equal))]
                            simp [encodedEqual, same]
  · intro context references frame assignment constantOne activeZero
    cases references with
    | cons left tail =>
        cases tail with
        | cons right tail =>
            cases tail
            let occurrence := frameRecipe parameters profile frame
            let completed :=
              occurrence.completion profile.inverseLaw
                frame.temporaries.ids assignment
            have agrees :
                AgreesOn frame.visibleIds assignment completed := by
              exact completion_agrees parameters profile frame assignment
            have changes :
                ChangesOnly frame.temporaries.ids assignment completed :=
              occurrence.completion_changesOnly profile.inverseLaw
                frame.temporaries.ids assignment
            have temporaryNodup : frame.temporaries.ids.Nodup :=
              (List.nodup_append.mp frame.allocationsNodup).2.1
            have witnessValues :=
              occurrence.completion_values profile.inverseLaw
                frame.temporaries.ids assignment
                (temporary_ids_exact parameters profile frame) temporaryNodup
            have preserved :=
              recipe_values_preserved parameters profile frame
                assignment completed agrees
            have leftPreserved :
                occurrence.left.map (fun column => completed column.id) =
                  occurrence.left.map
                    (fun column => assignment column.id) := by
              simpa [occurrence] using preserved.1
            have rightPreserved :
                occurrence.right.map (fun column => completed column.id) =
                  occurrence.right.map
                    (fun column => assignment column.id) := by
              simpa [occurrence] using preserved.2
            have inverseWitness :
                occurrence.inverses.map
                      (fun column => completed column.id) =
                    coordinateInverseValues profile.inverseLaw
                      (occurrence.left.map
                        (fun column => assignment column.id))
                      (occurrence.right.map
                        (fun column => assignment column.id)) := by
              simpa [completed] using witnessValues.1
            have equalWitness :
                occurrence.equals.map
                      (fun column => completed column.id) =
                    coordinateEqualValues
                      (occurrence.left.map
                        (fun column => assignment column.id))
                      (occurrence.right.map
                        (fun column => assignment column.id)) := by
              simpa [completed] using witnessValues.2.1
            have productWitness :
                occurrence.products.map
                      (fun column => completed column.id) =
                    productValues
                      (coordinateEqualValues
                        (occurrence.left.map
                          (fun column => assignment column.id))
                        (occurrence.right.map
                          (fun column => assignment column.id))) := by
              simpa [completed] using witnessValues.2.2
            have oneCompleted : completed frame.one = 1 := by
              rw [agrees frame.one (by
                simp [CallFrame.visibleIds]), constantOne]
            have activeCompleted : completed frame.active = 0 := by
              rw [agrees frame.active (by
                simp [CallFrame.visibleIds]), activeZero]
            refine ⟨completed, agrees, changes, ?_⟩
            apply occurrence.inactive_complete profile.inverseLaw completed
              oneCompleted activeCompleted
            · rw [leftPreserved, rightPreserved]
              exact inverseWitness
            · rw [leftPreserved, rightPreserved]
              exact equalWitness
            · rw [equalWitness]
              exact productWitness

/-- Compatibility wrapper for a complete direct-call profile. -/
def encodedEqualRecipe
    (parameters : Parameters)
    (profile : DirectProfile parameters) :
    CallRecipe (signature parameters) profile.family Call.encodedEqual :=
  encodedEqualRecipeForProfile parameters
    (profile.encodedEqualProfile parameters)

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls
