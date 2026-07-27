import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.Poseidon23HashPriorFrame

/-!
Contract: total canonical recipe for the prior Construction-2 binding hash.

Owns: exact typed-call semantics, active soundness, active honest
completeness, inactive satisfiability, row ownership, support, and receipt
cost for the application-selected fixed-23 profile.

Does not own: deployment selection, collision resistance, Fiat--Shamir,
Rust, generated rows, `step`, or `nifsVerify`.
-/

set_option autoImplicit false
set_option maxRecDepth 32768

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding

open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls
open Poseidon23HashCallCommon

namespace Poseidon23HashPriorRecipe

private theorem footprint_exact
    (parameters : Parameters)
    (profile : Poseidon23ApplicationProfile parameters) :
    (signature parameters).callFootprint Call.hashPrior =
      Poseidon23Hash.footprint profile.alignmentWidth := by
  simpa [signature, callFootprint] using profile.hashFootprint

private theorem operand_coordinates_of_decodes
    (parameters : Parameters)
    (profile : Poseidon23ApplicationProfile parameters)
    {context : Schema (typeSystem parameters)}
    {iterationRef : Ref (typeSystem parameters) context (.data .nat)}
    {z0Ref currentRef : Ref (typeSystem parameters) context (.data .state)}
    {runningRef : Ref (typeSystem parameters) context (.data .running)}
    (frame :
      CallFrame (signature := signature parameters)
        profile.family Call.hashPrior
        (Refs.cons iterationRef
          (Refs.cons z0Ref
            (Refs.cons currentRef (Refs.cons runningRef .nil)))))
    (assignment : ColumnId -> Field)
    (iteration : Nat)
    (z0 current : parameters.State)
    (running : parameters.Running)
    (decoded :
      frame.operands.Decodes profile.family assignment
        (.cons iteration
          (.cons z0 (.cons current (.cons running .nil))))) :
    (firstOperand frame.operands).values assignment =
        boundedNatCodec.encode iteration ∧
      (secondOperand frame.operands).values assignment =
        profile.codecs.state.encode z0 ∧
      (thirdOperand frame.operands).values assignment =
        profile.codecs.state.encode current ∧
      (fourthOperand frame.operands).values assignment =
        profile.codecs.running.encode running := by
  have parts :=
    (fourOperand_decodes_iff profile.family assignment frame.operands
      iteration z0 current running).mp decoded
  exact
    ⟨(boundedNatCodec.encode_decode _ iteration parts.1).2.symm,
      (profile.codecs.state.encode_decode _ z0 parts.2.1).2.symm,
      (profile.codecs.state.encode_decode _ current parts.2.2.1).2.symm,
      (profile.codecs.running.encode_decode _ running parts.2.2.2).2.symm⟩

private theorem operand_coordinates_of_encodes
    (parameters : Parameters)
    (profile : Poseidon23ApplicationProfile parameters)
    {context : Schema (typeSystem parameters)}
    {iterationRef : Ref (typeSystem parameters) context (.data .nat)}
    {z0Ref currentRef : Ref (typeSystem parameters) context (.data .state)}
    {runningRef : Ref (typeSystem parameters) context (.data .running)}
    (frame :
      CallFrame (signature := signature parameters)
        profile.family Call.hashPrior
        (Refs.cons iterationRef
          (Refs.cons z0Ref
            (Refs.cons currentRef (Refs.cons runningRef .nil)))))
    (assignment : ColumnId -> Field)
    (iteration : Nat)
    (z0 current : parameters.State)
    (running : parameters.Running)
    (encoded :
      frame.operands.Encodes profile.family assignment
        (.cons iteration
          (.cons z0 (.cons current (.cons running .nil))))) :
    (firstOperand frame.operands).values assignment =
        boundedNatCodec.encode iteration ∧
      (secondOperand frame.operands).values assignment =
        profile.codecs.state.encode z0 ∧
      (thirdOperand frame.operands).values assignment =
        profile.codecs.state.encode current ∧
      (fourthOperand frame.operands).values assignment =
        profile.codecs.running.encode running := by
  have parts :=
    (fourOperand_encodes_iff profile.family assignment frame.operands
      iteration z0 current running).mp encoded
  exact ⟨parts.1.2, parts.2.1.2, parts.2.2.1.2, parts.2.2.2.2⟩

private theorem honest_source_values
    (parameters : Parameters)
    (profile : Poseidon23ApplicationProfile parameters)
    {context : Schema (typeSystem parameters)}
    {iterationRef : Ref (typeSystem parameters) context (.data .nat)}
    {z0Ref currentRef : Ref (typeSystem parameters) context (.data .state)}
    {runningRef : Ref (typeSystem parameters) context (.data .running)}
    (frame :
      CallFrame (signature := signature parameters)
        profile.family Call.hashPrior
        (Refs.cons iterationRef
          (Refs.cons z0Ref
            (Refs.cons currentRef (Refs.cons runningRef .nil)))))
    (assignment : ColumnId -> Field)
    (iteration : Nat)
    (z0 current : parameters.State)
    (running : parameters.Running)
    (coordinates :
      (firstOperand frame.operands).values assignment =
          boundedNatCodec.encode iteration ∧
        (secondOperand frame.operands).values assignment =
          profile.codecs.state.encode z0 ∧
        (thirdOperand frame.operands).values assignment =
          profile.codecs.state.encode current ∧
        (fourthOperand frame.operands).values assignment =
          profile.codecs.running.encode running) :
    Poseidon23HashOccurrence.Honest.sourceValues
        (Poseidon23HashPriorFrame.occurrence parameters profile frame)
        assignment =
      Poseidon23Hash.sourceCoordinates profile.codecs false
        iteration z0 current running := by
  unfold Poseidon23HashOccurrence.Honest.sourceValues
    Poseidon23HashOccurrence.Honest.normalizedValue
    Poseidon23Hash.sourceCoordinates Poseidon23Hash.normalizedIteration
  have iterationValues := coordinates.1
  rw [Poseidon23HashPriorFrame.iteration_values_eq_singleton
    parameters profile frame assignment] at iterationValues
  have iterationHead :
      assignment
          ((Poseidon23HashPriorFrame.occurrence parameters profile frame).iteration.id) =
        (boundedNatCodec.encode iteration).getD 0 0 := by
    rw [← iterationValues]
    rfl
  rw [iterationHead]
  rw [Poseidon23HashPriorFrame.source_tail_values
    parameters profile frame assignment]
  rw [coordinates.2.1, coordinates.2.2.1, coordinates.2.2.2]
  have nextFalse :
      (Poseidon23HashPriorFrame.occurrence parameters profile frame).next =
        false := rfl
  simp [nextFalse]

private theorem active_source_values
    (parameters : Parameters)
    (profile : Poseidon23ApplicationProfile parameters)
    {context : Schema (typeSystem parameters)}
    {iterationRef : Ref (typeSystem parameters) context (.data .nat)}
    {z0Ref currentRef : Ref (typeSystem parameters) context (.data .state)}
    {runningRef : Ref (typeSystem parameters) context (.data .running)}
    (frame :
      CallFrame (signature := signature parameters)
        profile.family Call.hashPrior
        (Refs.cons iterationRef
          (Refs.cons z0Ref
            (Refs.cons currentRef (Refs.cons runningRef .nil)))))
    (assignment : ColumnId -> Field)
    (normalized :
      assignment ((Poseidon23HashPriorFrame.occurrence
          parameters profile frame).normalizedColumn.id) =
        assignment ((Poseidon23HashPriorFrame.occurrence
          parameters profile frame).iteration.id)) :
    Poseidon23HashOccurrence.sourceValues
        (Poseidon23HashPriorFrame.occurrence parameters profile frame)
        assignment =
      Poseidon23HashOccurrence.Honest.sourceValues
        (Poseidon23HashPriorFrame.occurrence parameters profile frame)
        assignment := by
  unfold Poseidon23HashOccurrence.sourceValues
    Poseidon23HashOccurrence.Honest.sourceValues
    Poseidon23HashOccurrence.Honest.normalizedValue
    Poseidon23HashOccurrence.Frame.source
  simp only [List.map_cons]
  rw [normalized]
  have nextFalse :
      (Poseidon23HashPriorFrame.occurrence parameters profile frame).next =
        false := rfl
  simp [nextFalse]

private theorem temporary_disjoint_visible
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
    IdsDisjoint
      (Poseidon23HashPriorFrame.occurrence
        parameters profile frame).temporaryIds
      (Poseidon23HashPriorFrame.occurrence
        parameters profile frame).visibleIds := by
  intro id temporary visible
  apply frame.temporariesDisjointVisible id
  · rw [← Poseidon23HashPriorFrame.temporary_ids_exact
      parameters profile frame]
    exact temporary
  · exact Poseidon23HashPriorFrame.visible_subset
      parameters profile frame id visible

private theorem completed_agrees
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
    (assignment completed : ColumnId -> Field)
    (changes : ChangesOnly frame.temporaries.ids assignment completed) :
    AgreesOn frame.visibleIds assignment completed := by
  intro id visible
  apply changes id
  intro temporary
  exact frame.temporariesDisjointVisible id temporary visible

/-- Total prior-hash recipe for the selected application profile. -/
def recipe
    (parameters : Parameters)
    (profile : Poseidon23ApplicationProfile parameters) :
    CallRecipe (signature parameters) profile.family Call.hashPrior := by
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
    | cons iteration tail =>
        cases tail with
        | cons z0 tail =>
            cases tail with
            | cons current tail =>
                cases tail with
                | cons running tail =>
                    cases tail
                    exact Poseidon23HashOccurrence.rows
                      (Poseidon23HashPriorFrame.occurrence
                        parameters profile frame)
                      (Poseidon23HashPriorFrame.coreFacts
                        parameters profile frame)
  · intro context references frame
    cases references with
    | cons iteration tail =>
        cases tail with
        | cons z0 tail =>
            cases tail with
            | cons current tail =>
                cases tail with
                | cons running tail =>
                    cases tail
                    rw [footprint_exact parameters profile]
                    exact
                      Poseidon23HashOccurrence.rows_length
                        (Poseidon23HashPriorFrame.occurrence
                          parameters profile frame)
                        (Poseidon23HashPriorFrame.coreFacts
                          parameters profile frame)
  · intro context references frame row member
    cases references with
    | cons iteration tail =>
        cases tail with
        | cons z0 tail =>
            cases tail with
            | cons current tail =>
                cases tail with
                | cons running tail =>
                    cases tail
                    exact Poseidon23HashOccurrence.rows_owned
                      (Poseidon23HashPriorFrame.occurrence
                        parameters profile frame)
                      (Poseidon23HashPriorFrame.coreFacts
                        parameters profile frame) row member
  · intro context references frame
    cases references with
    | cons iteration tail =>
        cases tail with
        | cons z0 tail =>
            cases tail with
            | cons current tail =>
                cases tail with
                | cons running tail =>
                    cases tail
                    exact Poseidon23HashOccurrence.row_ids_nodup
                      (Poseidon23HashPriorFrame.occurrence
                        parameters profile frame)
                      (Poseidon23HashPriorFrame.coreFacts
                        parameters profile frame)
  · intro context references frame row member column columnMember
    cases references with
    | cons iteration tail =>
        cases tail with
        | cons z0 tail =>
            cases tail with
            | cons current tail =>
                cases tail with
                | cons running tail =>
                    cases tail
                    have supported :=
                      Poseidon23HashOccurrence.rows_supported
                        (Poseidon23HashPriorFrame.occurrence
                          parameters profile frame)
                        (Poseidon23HashPriorFrame.coreFacts
                          parameters profile frame)
                        row member column columnMember
                    rcases List.mem_append.mp supported with
                      visible | temporary
                    · exact List.mem_append_left _
                        (Poseidon23HashPriorFrame.visible_subset
                          parameters profile frame column visible)
                    · apply List.mem_append_right frame.visibleIds
                      rw [← Poseidon23HashPriorFrame.temporary_ids_exact
                        parameters profile frame]
                      exact temporary
  · intro context references frame assignment inputs
      constantOne activeOne decoded holds
    cases references with
    | cons iterationRef tail =>
        cases tail with
        | cons z0Ref tail =>
            cases tail with
            | cons currentRef tail =>
                cases tail with
                | cons runningRef tail =>
                    cases tail
                    cases inputs with
                    | cons iteration inputs =>
                        cases inputs with
                        | cons z0 inputs =>
                            cases inputs with
                            | cons current inputs =>
                                cases inputs with
                                | cons running inputs =>
                                    cases inputs
                                    let selected :=
                                      Poseidon23HashPriorFrame.occurrence
                                        parameters profile frame
                                    let facts :=
                                      Poseidon23HashPriorFrame.coreFacts
                                        parameters profile frame
                                    have coordinates :=
                                      operand_coordinates_of_decodes
                                        parameters profile frame assignment
                                        iteration z0 current running decoded
                                    have raw :=
                                      Poseidon23HashOccurrence.active_sound
                                        profile.fieldLaws selected facts
                                        assignment constantOne activeOne holds
                                    have semanticSource :
                                        Poseidon23HashOccurrence.sourceValues
                                            selected assignment =
                                          Poseidon23Hash.sourceCoordinates
                                            profile.codecs false iteration z0
                                            current running := by
                                      rw [active_source_values parameters
                                        profile frame assignment (by
                                          simpa [selected] using raw.1)]
                                      exact honest_source_values parameters
                                        profile frame assignment iteration z0
                                        current running coordinates
                                    let result :=
                                      parameters.machine.hash {
                                        verifierKeys :=
                                          parameters.setup.verifierKeys
                                        iteration := iteration
                                        z0 := z0
                                        current := current
                                        running := fun _ => running
                                        pc := 1
                                      }
                                    have outputCoordinates :
                                        Poseidon23HashOccurrence.outputValues
                                            selected assignment =
                                          profile.codecs.digest.encode result := by
                                      rw [raw.2, semanticSource]
                                      exact (profile.hashPrior_exact
                                        iteration z0 current running).symm
                                    refine ⟨.cons result .nil, rfl, ?_⟩
                                    apply
                                      (unaryOutput_decodes_iff profile.family
                                        assignment frame.outputs result).mpr
                                    unfold ColumnBundle.Decodes
                                    have outputValues :
                                        (unaryOutput frame.outputs).values
                                            assignment =
                                          profile.codecs.digest.encode result := by
                                      simpa [selected,
                                        Poseidon23HashPriorFrame.occurrence,
                                        Poseidon23HashOccurrence.outputValues,
                                        ColumnBundle.values] using
                                        outputCoordinates
                                    change
                                      profile.codecs.digest.decode
                                          ((unaryOutput frame.outputs).values
                                            assignment) =
                                        some result
                                    rw [outputValues]
                                    exact profile.codecs.digest.decode_encode
                                      result (profile.digestAdmissible result)
  · intro context references frame assignment inputs outputs
      constantOne activeOne inputsEncoded outputsEncoded evaluated
    cases references with
    | cons iterationRef tail =>
        cases tail with
        | cons z0Ref tail =>
            cases tail with
            | cons currentRef tail =>
                cases tail with
                | cons runningRef tail =>
                    cases tail
                    cases inputs with
                    | cons iteration inputs =>
                        cases inputs with
                        | cons z0 inputs =>
                            cases inputs with
                            | cons current inputs =>
                                cases inputs with
                                | cons running inputs =>
                                    cases inputs
                                    cases outputs with
                                    | cons output outputs =>
                                        cases outputs
                                        let result :=
                                          parameters.machine.hash {
                                            verifierKeys :=
                                              parameters.setup.verifierKeys
                                            iteration := iteration
                                            z0 := z0
                                            current := current
                                            running := fun _ => running
                                            pc := 1
                                          }
                                        have outputEqual : output = result :=
                                          congrArg HVec.head
                                            (Option.some.inj evaluated.symm)
                                        subst output
                                        have coordinates :=
                                          operand_coordinates_of_encodes
                                            parameters profile frame assignment
                                            iteration z0 current running
                                            inputsEncoded
                                        have sourceExact :=
                                          honest_source_values parameters
                                            profile frame assignment iteration
                                            z0 current running coordinates
                                        have outputEncoded :=
                                          (unaryOutput_encodes_iff
                                            profile.family assignment
                                            frame.outputs result).mp
                                            outputsEncoded
                                        let selected :=
                                          Poseidon23HashPriorFrame.occurrence
                                            parameters profile frame
                                        let facts :=
                                          Poseidon23HashPriorFrame.coreFacts
                                            parameters profile frame
                                        have outputCorrect :
                                            Poseidon23HashOccurrence.outputValues
                                                selected assignment =
                                              Poseidon23Hash.resultCoordinates
                                                profile.hashPlan
                                                (Poseidon23HashOccurrence.Honest.sourceValues
                                                  selected assignment) := by
                                          rw [sourceExact,
                                            ← profile.hashPrior_exact
                                              iteration z0 current running]
                                          simpa [selected,
                                            Poseidon23HashPriorFrame.occurrence,
                                            Poseidon23HashOccurrence.outputValues,
                                            ColumnBundle.values] using
                                            outputEncoded.2
                                        let completed :=
                                          Poseidon23HashOccurrence.Honest.complete
                                            profile.inverseLaw selected facts
                                              assignment
                                        have temporaryNodup :
                                            selected.temporaryIds.Nodup := by
                                          rw [Poseidon23HashPriorFrame.temporary_ids_exact
                                            parameters profile frame]
                                          exact
                                            (List.nodup_append.mp
                                              frame.allocationsNodup).2.1
                                        have disjoint :=
                                          temporary_disjoint_visible
                                            parameters profile frame
                                        have changesSelected :=
                                          Poseidon23HashOccurrence.Honest.complete_changesOnly
                                            profile.inverseLaw selected facts
                                              assignment
                                        have changes :
                                            ChangesOnly frame.temporaries.ids
                                              assignment completed := by
                                          rw [← Poseidon23HashPriorFrame.temporary_ids_exact
                                            parameters profile frame]
                                          exact changesSelected
                                        have agrees :=
                                          completed_agrees parameters profile
                                            frame assignment completed changes
                                        refine ⟨completed, agrees, changes, ?_⟩
                                        exact
                                          Poseidon23HashOccurrence.Honest.active_complete
                                            profile.inverseLaw selected facts assignment
                                          constantOne activeOne outputCorrect
                                          temporaryNodup disjoint
  · intro context references frame assignment constantOne activeZero
    cases references with
    | cons iteration tail =>
        cases tail with
        | cons z0 tail =>
            cases tail with
            | cons current tail =>
                cases tail with
                | cons running tail =>
                    cases tail
                    let selected :=
                      Poseidon23HashPriorFrame.occurrence
                        parameters profile frame
                    let facts :=
                      Poseidon23HashPriorFrame.coreFacts
                        parameters profile frame
                    let completed :=
                      Poseidon23HashOccurrence.Honest.complete
                        profile.inverseLaw selected facts assignment
                    have temporaryNodup :
                        selected.temporaryIds.Nodup := by
                      rw [Poseidon23HashPriorFrame.temporary_ids_exact
                        parameters profile frame]
                      exact
                        (List.nodup_append.mp frame.allocationsNodup).2.1
                    have disjoint :=
                      temporary_disjoint_visible parameters profile frame
                    have changesSelected :=
                      Poseidon23HashOccurrence.Honest.complete_changesOnly
                        profile.inverseLaw selected facts assignment
                    have changes :
                        ChangesOnly frame.temporaries.ids assignment
                          completed := by
                      rw [← Poseidon23HashPriorFrame.temporary_ids_exact
                        parameters profile frame]
                      exact changesSelected
                    have agrees :=
                      completed_agrees parameters profile frame assignment
                        completed changes
                    refine ⟨completed, agrees, changes, ?_⟩
                    exact Poseidon23HashOccurrence.Honest.inactive_complete
                      profile.inverseLaw selected facts assignment constantOne
                      activeZero temporaryNodup disjoint

end Poseidon23HashPriorRecipe

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding
