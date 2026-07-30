import Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.Codecs
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls.Common
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls.Footprints

/-!
Contract: physical Step-row core for the 42-times-6 WASM integration fixture.

Assurance tier: model-level.

Owns: one exact eleven-row R1CS program for the seven-coordinate fixture
transition, four temporary columns, stable row ownership, active soundness,
active honest completeness, inactive satisfiability, support, and the
completion witness used by the public `CallRecipe`.

Does not own: a general WASM compiler, the F-prime application profile,
NIFS, relation setup, a recursive fixed point, Rust, or artifacts.

The rows encode the same three-instruction batch boundary as `Semantics.step`.
They do not import rows, columns, or an acceptance bit from the Rust
benchmark.

Emits constraints: exactly eleven rows and four auxiliary columns.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6

open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsCodecProjection
open Nightstream.SuperNeo.Concrete

/-- The application step computes four multiplication witnesses and gates
seven output-coordinate equations. -/
def stepFootprint : CallFootprint where
  recurringRows := 11
  temporaries := [auxiliaryLayout 4]

/-- Explicit two-sided representation bridge from the selected application
state type to the benchmark state. -/
structure StateEquivalence (parameters : Parameters) where
  toBenchmark : parameters.State -> State
  fromBenchmark : State -> parameters.State
  leftInverse : Function.LeftInverse fromBenchmark toBenchmark
  rightInverse : Function.RightInverse fromBenchmark toBenchmark

/-- Exact representation and semantic boundary needed by this Step recipe. -/
structure StepProfile (parameters : Parameters)
    extends Encoding.Profile parameters where
  stateEquiv : StateEquivalence parameters
  stateEncodeExact :
    ∀ state,
      codecs.state.encode state =
        List.ofFn (stateEquiv.toBenchmark state)
  stateAdmissible : ∀ state, codecs.state.Admissible state
  stateRecoverable : codecs.state.ExactWidthRecoverable
  stepFootprintExact : parameters.footprints.step = stepFootprint
  stepExact :
    ∀ state witness,
      stateEquiv.toBenchmark
          (parameters.machine.step
            Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary.Step.selected
            state witness) =
        WasmBenchmark42x6.step
          (stateEquiv.toBenchmark state) WasmBenchmark42x6.noWitness

namespace StepProfile

def family
    (parameters : Parameters)
    (profile : StepProfile parameters) :
    Family (typeSystem parameters) :=
  profile.toProfile.family parameters

end StepProfile

private theorem getD_ofFn
    {Item : Type}
    {count : Nat}
    (items : Fin count -> Item)
    (index : Fin count)
    (default : Item) :
    (List.ofFn items).getD index.val default = items index := by
  rw [List.getD_eq_getElem?_getD,
    List.getElem?_eq_getElem (by simp), List.getElem_ofFn]
  simp

private theorem selectedStateCodec_width
    {parameters : Parameters}
    (profile : StepProfile parameters) :
    profile.codecs.state.width = 7 := by
  have encodedLength :=
    profile.codecs.state.encode_length
      (profile.stateEquiv.fromBenchmark initial)
  rw [profile.stateEncodeExact] at encodedLength
  simpa using encodedLength.symm

private theorem footprint_exact
    (parameters : Parameters)
    (profile : StepProfile parameters) :
    (signature parameters).callFootprint Call.step = stepFootprint := by
  simpa [signature, callFootprint] using profile.stepFootprintExact

private theorem temporary_layouts_exact
    (parameters : Parameters)
    (profile : StepProfile parameters) :
    ((signature parameters).callFootprint Call.step).temporaries =
      [auxiliaryLayout 4] := by
  rw [footprint_exact parameters profile]
  rfl

private noncomputable def stateViewFor
    {parameters : Parameters}
    (profile : StepProfile parameters)
    (coordinate : StateCoordinate) :
    FView profile.codecs.state
      (fun state =>
        profile.stateEquiv.toBenchmark state coordinate.index) where
  index := ⟨coordinate.index.val, by
    rw [selectedStateCodec_width profile]
    exact coordinate.index.isLt⟩
  encodeValue := by
    intro state
    rw [profile.stateEncodeExact]
    exact getD_ofFn
      (profile.stateEquiv.toBenchmark state) coordinate.index 0

private theorem inputState_width
    (parameters : Parameters)
    (profile : StepProfile parameters)
    {context : Schema (typeSystem parameters)}
    {stateReference :
      Ref (typeSystem parameters) context (.data .state)}
    {witnessReference :
      Ref (typeSystem parameters) context (.data .witness)}
    (frame :
      CallFrame (signature := signature parameters)
        profile.family Call.step
        (.cons stateReference (.cons witnessReference .nil))) :
    profile.codecs.state.width =
      stateReference.port.layout.owners.length := by
  simpa [StepProfile.family, Encoding.Profile.family,
    DataCodecs.family, Family.codecFor] using
      frame.operandWidthsAgree.1

private theorem outputState_width
    (parameters : Parameters)
    (profile : StepProfile parameters)
    {context : Schema (typeSystem parameters)}
    {stateReference :
      Ref (typeSystem parameters) context (.data .state)}
    {witnessReference :
      Ref (typeSystem parameters) context (.data .witness)}
    (frame :
      CallFrame (signature := signature parameters)
        profile.family Call.step
        (.cons stateReference (.cons witnessReference .nil))) :
    profile.codecs.state.width =
      (Ports.committedState parameters).layout.owners.length := by
  have width :=
    frame.outputWidthsAgree
      (Ports.committedState parameters) (by
        change Ports.committedState parameters ∈
          callOutputs parameters Call.step
        exact List.mem_cons_self)
  unfold PortWidthAgrees at width
  simpa [StepProfile.family, Encoding.Profile.family,
    DataCodecs.family, Family.codecFor] using width

private def frameTemporaries
    (parameters : Parameters)
    (profile : StepProfile parameters)
    {context : Schema (typeSystem parameters)}
    {stateReference :
      Ref (typeSystem parameters) context (.data .state)}
    {witnessReference :
      Ref (typeSystem parameters) context (.data .witness)}
    (frame :
      CallFrame (signature := signature parameters)
        profile.family Call.step
        (.cons stateReference (.cons witnessReference .nil))) :
    LayoutBundles [auxiliaryLayout 4] :=
  temporary_layouts_exact parameters profile ▸ frame.temporaries

private theorem frameTemporaries_ids
    (parameters : Parameters)
    (profile : StepProfile parameters)
    {context : Schema (typeSystem parameters)}
    {stateReference :
      Ref (typeSystem parameters) context (.data .state)}
    {witnessReference :
      Ref (typeSystem parameters) context (.data .witness)}
    (frame :
      CallFrame (signature := signature parameters)
        profile.family Call.step
        (.cons stateReference (.cons witnessReference .nil))) :
    (frameTemporaries parameters profile frame).ids =
      frame.temporaries.ids :=
  layoutBundles_ids_cast
    (temporary_layouts_exact parameters profile) frame.temporaries

private noncomputable def inputColumn
    (parameters : Parameters)
    (profile : StepProfile parameters)
    {context : Schema (typeSystem parameters)}
    {stateReference :
      Ref (typeSystem parameters) context (.data .state)}
    {witnessReference :
      Ref (typeSystem parameters) context (.data .witness)}
    (frame :
      CallFrame (signature := signature parameters)
        profile.family Call.step
        (.cons stateReference (.cons witnessReference .nil)))
    (coordinate : StateCoordinate) : ColumnId :=
  ((stateViewFor profile coordinate).column
    (firstBinaryOperand frame.operands)
    (inputState_width parameters profile frame)).column

private noncomputable def outputColumn
    (parameters : Parameters)
    (profile : StepProfile parameters)
    {context : Schema (typeSystem parameters)}
    {stateReference :
      Ref (typeSystem parameters) context (.data .state)}
    {witnessReference :
      Ref (typeSystem parameters) context (.data .witness)}
    (frame :
      CallFrame (signature := signature parameters)
        profile.family Call.step
        (.cons stateReference (.cons witnessReference .nil)))
    (coordinate : StateCoordinate) : ColumnId :=
  ((stateViewFor profile coordinate).column
    (unaryOutput frame.outputs)
    (outputState_width parameters profile frame)).column

private def temporaryColumn
    (parameters : Parameters)
    (profile : StepProfile parameters)
    {context : Schema (typeSystem parameters)}
    {stateReference :
      Ref (typeSystem parameters) context (.data .state)}
    {witnessReference :
      Ref (typeSystem parameters) context (.data .witness)}
    (frame :
      CallFrame (signature := signature parameters)
        profile.family Call.step
        (.cons stateReference (.cons witnessReference .nil)))
    (coordinate : Fin 4) : ColumnId :=
  (bundleColumn
    (firstTemporary (frameTemporaries parameters profile frame))
    ⟨coordinate.val, by
      change coordinate.val <
        (auxiliaryLayout 4).owners.length
      simpa [auxiliaryLayout, ownedLayout] using coordinate.isLt⟩).id

private def affine3
    (first : ColumnId) (firstCoefficient : Field)
    (second : ColumnId) (secondCoefficient : Field)
    (third : ColumnId) (thirdCoefficient : Field) :
    LinearCombination :=
  [ { column := first, coefficient := firstCoefficient }
  , { column := second, coefficient := secondCoefficient }
  , { column := third, coefficient := thirdCoefficient }
  ]

private theorem inputColumn_mem_visible
    (parameters : Parameters)
    (profile : StepProfile parameters)
    {context : Schema (typeSystem parameters)}
    {stateReference :
      Ref (typeSystem parameters) context (.data .state)}
    {witnessReference :
      Ref (typeSystem parameters) context (.data .witness)}
    (frame :
      CallFrame (signature := signature parameters)
        profile.family Call.step
        (.cons stateReference (.cons witnessReference .nil)))
    (coordinate : StateCoordinate) :
    inputColumn parameters profile frame coordinate ∈ frame.visibleIds := by
  have operandMember :
      inputColumn parameters profile frame coordinate ∈
        frame.operands.ids := by
    have firstMember :
        inputColumn parameters profile frame coordinate ∈
          (firstBinaryOperand frame.operands).ids := by
      simpa [inputColumn] using
      (stateViewFor profile coordinate).column_mem
        (firstBinaryOperand frame.operands)
        (inputState_width parameters profile frame)
    have combined :
        inputColumn parameters profile frame coordinate ∈
          (firstBinaryOperand frame.operands).ids ++
            (secondBinaryOperand frame.operands).ids :=
      List.mem_append_left _ firstMember
    simpa only [binaryOperand_ids] using combined
  have contextMember :=
    RefBundles.fromSchema_ids_subset
      (.cons stateReference (.cons witnessReference .nil))
      frame.contextBundles _ operandMember
  exact List.mem_append_left frame.outputs.ids
    (List.mem_append_right [frame.one, frame.active] contextMember)

private theorem outputColumn_mem_visible
    (parameters : Parameters)
    (profile : StepProfile parameters)
    {context : Schema (typeSystem parameters)}
    {stateReference :
      Ref (typeSystem parameters) context (.data .state)}
    {witnessReference :
      Ref (typeSystem parameters) context (.data .witness)}
    (frame :
      CallFrame (signature := signature parameters)
        profile.family Call.step
        (.cons stateReference (.cons witnessReference .nil)))
    (coordinate : StateCoordinate) :
    outputColumn parameters profile frame coordinate ∈ frame.visibleIds := by
  have outputMember :
      outputColumn parameters profile frame coordinate ∈
        frame.outputs.ids := by
    simpa [outputColumn] using
      (stateViewFor profile coordinate).column_mem
        (unaryOutput frame.outputs)
        (outputState_width parameters profile frame)
  exact List.mem_append_right
    ([frame.one, frame.active] ++ frame.contextBundles.ids) outputMember

private theorem firstTemporary_ids_subset
    {first : Layout}
    {rest : List Layout}
    (bundles : LayoutBundles (first :: rest)) :
    ∀ column, column ∈ (firstTemporary bundles).ids ->
      column ∈ bundles.ids := by
  cases bundles with
  | cons bundle tail =>
      intro column member
      simp only [firstTemporary, LayoutBundles.ids,
        LayoutBundles.columns, LayoutBundles.bundleColumns,
        List.flatten_cons, List.map_append, List.mem_append]
      exact Or.inl member

private theorem temporaryColumn_mem
    (parameters : Parameters)
    (profile : StepProfile parameters)
    {context : Schema (typeSystem parameters)}
    {stateReference :
      Ref (typeSystem parameters) context (.data .state)}
    {witnessReference :
      Ref (typeSystem parameters) context (.data .witness)}
    (frame :
      CallFrame (signature := signature parameters)
        profile.family Call.step
        (.cons stateReference (.cons witnessReference .nil)))
    (coordinate : Fin 4) :
    temporaryColumn parameters profile frame coordinate ∈
      frame.temporaries.ids := by
  have firstMember :
      temporaryColumn parameters profile frame coordinate ∈
        (firstTemporary
          (frameTemporaries parameters profile frame)).ids := by
    exact bundleColumn_id_mem _ _
  have allMember :=
    firstTemporary_ids_subset
      (frameTemporaries parameters profile frame)
      _ firstMember
  rw [frameTemporaries_ids parameters profile frame] at allMember
  exact allMember

private theorem bundle_ids_eq_ofFn
    {layout : Layout}
    (bundle : ColumnBundle layout) :
    bundle.ids =
      List.ofFn (fun coordinate : Fin layout.owners.length =>
        (bundleColumn bundle coordinate).id) := by
  apply List.ext_get
  · simp [ColumnBundle.ids, bundle.length_eq]
  · intro index leftLt rightLt
    simp [ColumnBundle.ids, bundleColumn]

private theorem oneTemporary_ids
    {layout : Layout}
    (bundles : LayoutBundles [layout]) :
    bundles.ids = (firstTemporary bundles).ids := by
  cases bundles with
  | cons bundle tail =>
      cases tail
      simp [LayoutBundles.ids, LayoutBundles.columns,
        LayoutBundles.bundleColumns, firstTemporary,
        ColumnBundle.ids]

private theorem temporary_ids_exact
    (parameters : Parameters)
    (profile : StepProfile parameters)
    {context : Schema (typeSystem parameters)}
    {stateReference :
      Ref (typeSystem parameters) context (.data .state)}
    {witnessReference :
      Ref (typeSystem parameters) context (.data .witness)}
    (frame :
      CallFrame (signature := signature parameters)
        profile.family Call.step
        (.cons stateReference (.cons witnessReference .nil))) :
    frame.temporaries.ids =
      List.ofFn (fun coordinate : Fin 4 =>
        temporaryColumn parameters profile frame coordinate) := by
  rw [← frameTemporaries_ids parameters profile frame]
  rw [oneTemporary_ids
    (frameTemporaries parameters profile frame)]
  rw [bundle_ids_eq_ofFn
    (firstTemporary (frameTemporaries parameters profile frame))]
  rfl

private noncomputable def rawRows
    (parameters : Parameters)
    (profile : StepProfile parameters)
    {context : Schema (typeSystem parameters)}
    {stateReference :
      Ref (typeSystem parameters) context (.data .state)}
    {witnessReference :
      Ref (typeSystem parameters) context (.data .witness)}
    (frame :
      CallFrame (signature := signature parameters)
        profile.family Call.step
        (.cons stateReference (.cons witnessReference .nil))) :
    List Row :=
  let phase := inputColumn parameters profile frame .phase
  let memory := inputColumn parameters profile frame .memoryWord
  let left := inputColumn parameters profile frame .leftOperand
  let right := inputColumn parameters profile frame .rightOperand
  let output := inputColumn parameters profile frame .output
  let trapped := inputColumn parameters profile frame .trapped
  let deltaLeft := temporaryColumn parameters profile frame ⟨0, by decide⟩
  let deltaRight := temporaryColumn parameters profile frame ⟨1, by decide⟩
  let product := temporaryColumn parameters profile frame ⟨2, by decide⟩
  let deltaOutput := temporaryColumn parameters profile frame ⟨3, by decide⟩
  [
    { a := singleton phase 1
      b := difference left memory
      c := singleton deltaLeft 1 },
    { a := singleton phase 1
      b := [ { column := right, coefficient := 1 }
             , { column := frame.one, coefficient := -6 } ]
      c := singleton deltaRight 1 },
    { a := singleton left 1
      b := singleton right 1
      c := singleton product 1 },
    { a := singleton phase 1
      b := difference product output
      c := singleton deltaOutput 1 },
    { a := singleton frame.active 1
      b := difference frame.one
        (outputColumn parameters profile frame .phase)
      c := [] },
    { a := singleton frame.active 1
      b := difference memory
        (outputColumn parameters profile frame .memoryWord)
      c := [] },
    { a := singleton frame.active 1
      b := affine3 memory 1 deltaLeft 1
        (outputColumn parameters profile frame .leftOperand) (-1)
      c := [] },
    { a := singleton frame.active 1
      b := affine3 frame.one 6 deltaRight 1
        (outputColumn parameters profile frame .rightOperand) (-1)
      c := [] },
    { a := singleton frame.active 1
      b := affine3 output 1 deltaOutput 1
        (outputColumn parameters profile frame .output) (-1)
      c := [] },
    { a := singleton frame.active 1
      b := difference phase
        (outputColumn parameters profile frame .halted)
      c := [] },
    { a := singleton frame.active 1
      b := difference trapped
        (outputColumn parameters profile frame .trapped)
      c := [] }
  ]

@[simp] private theorem rawRows_length
    (parameters : Parameters)
    (profile : StepProfile parameters)
    {context : Schema (typeSystem parameters)}
    {stateReference :
      Ref (typeSystem parameters) context (.data .state)}
    {witnessReference :
      Ref (typeSystem parameters) context (.data .witness)}
    (frame :
      CallFrame (signature := signature parameters)
        profile.family Call.step
        (.cons stateReference (.cons witnessReference .nil))) :
    (rawRows parameters profile frame).length = 11 := by
  rfl

private theorem rawRows_supported
    (parameters : Parameters)
    (profile : StepProfile parameters)
    {context : Schema (typeSystem parameters)}
    {stateReference :
      Ref (typeSystem parameters) context (.data .state)}
    {witnessReference :
      Ref (typeSystem parameters) context (.data .witness)}
    (frame :
      CallFrame (signature := signature parameters)
        profile.family Call.step
        (.cons stateReference (.cons witnessReference .nil))) :
    RawRowsSupportedBy
      (frame.visibleIds ++ frame.temporaries.ids)
      (rawRows parameters profile frame) := by
  have one :
      frame.one ∈ frame.visibleIds ++ frame.temporaries.ids := by
    apply List.mem_append_left
    simp [CallFrame.visibleIds]
  have active :
      frame.active ∈ frame.visibleIds ++ frame.temporaries.ids := by
    apply List.mem_append_left
    simp [CallFrame.visibleIds]
  have input (coordinate : StateCoordinate) :
      inputColumn parameters profile frame coordinate ∈
        frame.visibleIds ++ frame.temporaries.ids :=
    List.mem_append_left _
      (inputColumn_mem_visible parameters profile frame coordinate)
  have output (coordinate : StateCoordinate) :
      outputColumn parameters profile frame coordinate ∈
        frame.visibleIds ++ frame.temporaries.ids :=
    List.mem_append_left _
      (outputColumn_mem_visible parameters profile frame coordinate)
  have temporary (coordinate : Fin 4) :
      temporaryColumn parameters profile frame coordinate ∈
        frame.visibleIds ++ frame.temporaries.ids :=
    List.mem_append_right _
      (temporaryColumn_mem parameters profile frame coordinate)
  intro row rowMember column columnMember
  simp [rawRows] at rowMember
  rcases rowMember with
    rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
  all_goals
    simp [Row.columnIds,
      Nightstream.Implementation.Lowering.Goldilocks.singleton,
      difference, affine3] at columnMember
    rcases columnMember with
      rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
    all_goals first
      | exact one
      | exact active
      | exact input _
      | exact output _
      | exact temporary _

private theorem rawRows_equations
    (parameters : Parameters)
    (profile : StepProfile parameters)
    {context : Schema (typeSystem parameters)}
    {stateReference :
      Ref (typeSystem parameters) context (.data .state)}
    {witnessReference :
      Ref (typeSystem parameters) context (.data .witness)}
    (frame :
      CallFrame (signature := signature parameters)
        profile.family Call.step
        (.cons stateReference (.cons witnessReference .nil)))
    (assignment : ColumnId -> Field)
    (holds :
      RawSatisfies (rawRows parameters profile frame) assignment) :
    assignment (inputColumn parameters profile frame .phase) *
          (assignment (inputColumn parameters profile frame .leftOperand) -
            assignment (inputColumn parameters profile frame .memoryWord)) =
        assignment (temporaryColumn parameters profile frame ⟨0, by decide⟩) ∧
    assignment (inputColumn parameters profile frame .phase) *
          (assignment (inputColumn parameters profile frame .rightOperand) -
            6 * assignment frame.one) =
        assignment (temporaryColumn parameters profile frame ⟨1, by decide⟩) ∧
    assignment (inputColumn parameters profile frame .leftOperand) *
          assignment (inputColumn parameters profile frame .rightOperand) =
        assignment (temporaryColumn parameters profile frame ⟨2, by decide⟩) ∧
    assignment (inputColumn parameters profile frame .phase) *
          (assignment (temporaryColumn parameters profile frame ⟨2, by decide⟩) -
            assignment (inputColumn parameters profile frame .output)) =
        assignment (temporaryColumn parameters profile frame ⟨3, by decide⟩) ∧
    assignment frame.active *
          (assignment frame.one -
            assignment (outputColumn parameters profile frame .phase)) = 0 ∧
    assignment frame.active *
          (assignment (inputColumn parameters profile frame .memoryWord) -
            assignment (outputColumn parameters profile frame .memoryWord)) = 0 ∧
    assignment frame.active *
          (assignment (inputColumn parameters profile frame .memoryWord) +
            assignment (temporaryColumn parameters profile frame ⟨0, by decide⟩) -
            assignment (outputColumn parameters profile frame .leftOperand)) = 0 ∧
    assignment frame.active *
          (6 * assignment frame.one +
            assignment (temporaryColumn parameters profile frame ⟨1, by decide⟩) -
            assignment (outputColumn parameters profile frame .rightOperand)) = 0 ∧
    assignment frame.active *
          (assignment (inputColumn parameters profile frame .output) +
            assignment (temporaryColumn parameters profile frame ⟨3, by decide⟩) -
            assignment (outputColumn parameters profile frame .output)) = 0 ∧
    assignment frame.active *
          (assignment (inputColumn parameters profile frame .phase) -
            assignment (outputColumn parameters profile frame .halted)) = 0 ∧
    assignment frame.active *
          (assignment (inputColumn parameters profile frame .trapped) -
            assignment (outputColumn parameters profile frame .trapped)) = 0 := by
  simpa [rawRows, Row.Holds, LinearCombination.eval,
    Nightstream.Implementation.Lowering.Goldilocks.singleton,
    difference, affine3, Fin.one_mul, Fin.mul_one, Fin.add_zero,
    Lean.Grind.Fin.neg_mul, Fin.sub_eq_add_neg,
    Lean.Grind.Fin.add_assoc] using holds

private theorem inputCoordinate_value
    (parameters : Parameters)
    (profile : StepProfile parameters)
    {context : Schema (typeSystem parameters)}
    {stateReference :
      Ref (typeSystem parameters) context (.data .state)}
    {witnessReference :
      Ref (typeSystem parameters) context (.data .witness)}
    (frame :
      CallFrame (signature := signature parameters)
        profile.family Call.step
        (.cons stateReference (.cons witnessReference .nil)))
    (assignment : ColumnId -> Field)
    (state : parameters.State)
    (decoded :
      (firstBinaryOperand frame.operands).Decodes
        profile.family (.data .state) assignment state)
    (coordinate : StateCoordinate) :
    assignment (inputColumn parameters profile frame coordinate) =
      profile.stateEquiv.toBenchmark state coordinate.index := by
  simpa [inputColumn, FColumnId.value] using
    (stateViewFor profile coordinate).value_eq_of_bundle_decodes
      profile.family (.data .state)
      (firstBinaryOperand frame.operands)
      (inputState_width parameters profile frame)
      assignment state decoded

private theorem outputCoordinate_value
    (parameters : Parameters)
    (profile : StepProfile parameters)
    {context : Schema (typeSystem parameters)}
    {stateReference :
      Ref (typeSystem parameters) context (.data .state)}
    {witnessReference :
      Ref (typeSystem parameters) context (.data .witness)}
    (frame :
      CallFrame (signature := signature parameters)
        profile.family Call.step
        (.cons stateReference (.cons witnessReference .nil)))
    (assignment : ColumnId -> Field)
    (state : parameters.State)
    (decoded :
      (unaryOutput frame.outputs).Decodes
        profile.family (.data .state) assignment state)
    (coordinate : StateCoordinate) :
    assignment (outputColumn parameters profile frame coordinate) =
      profile.stateEquiv.toBenchmark state coordinate.index := by
  simpa [outputColumn, FColumnId.value] using
    (stateViewFor profile coordinate).value_eq_of_bundle_decodes
      profile.family (.data .state)
      (unaryOutput frame.outputs)
      (outputState_width parameters profile frame)
      assignment state decoded

private theorem output_decode_exists
    (parameters : Parameters)
    (profile : StepProfile parameters)
    {context : Schema (typeSystem parameters)}
    {stateReference :
      Ref (typeSystem parameters) context (.data .state)}
    {witnessReference :
      Ref (typeSystem parameters) context (.data .witness)}
    (frame :
      CallFrame (signature := signature parameters)
        profile.family Call.step
        (.cons stateReference (.cons witnessReference .nil)))
    (assignment : ColumnId -> Field) :
    ∃ state : parameters.State,
      (unaryOutput frame.outputs).Decodes
        profile.family (.data .state) assignment state := by
  have lengthExact :
      ((unaryOutput frame.outputs).values assignment).length =
        profile.codecs.state.width := by
    rw [ColumnBundle.values_length,
      ← outputState_width parameters profile frame]
  rcases
      Codec.decode_exists_of_exactWidthRecoverable
        profile.stateRecoverable
        ((unaryOutput frame.outputs).values assignment)
        lengthExact with
    ⟨state, decoded⟩
  refine ⟨state, ?_⟩
  simpa [ColumnBundle.Decodes, StepProfile.family,
    Encoding.Profile.family, DataCodecs.family,
    Family.codecFor] using decoded

private theorem decodedOutput_eq_step
    (parameters : Parameters)
    (profile : StepProfile parameters)
    {context : Schema (typeSystem parameters)}
    {stateReference :
      Ref (typeSystem parameters) context (.data .state)}
    {witnessReference :
      Ref (typeSystem parameters) context (.data .witness)}
    (frame :
      CallFrame (signature := signature parameters)
        profile.family Call.step
        (.cons stateReference (.cons witnessReference .nil)))
    (assignment : ColumnId -> Field)
    (state outputState : parameters.State)
    (witness : parameters.Witness)
    (constantOne : assignment frame.one = 1)
    (activeOne : assignment frame.active = 1)
    (inputDecoded :
      (firstBinaryOperand frame.operands).Decodes
        profile.family (.data .state) assignment state)
    (outputDecoded :
      (unaryOutput frame.outputs).Decodes
        profile.family (.data .state) assignment outputState)
    (holds :
      Satisfies
        (ownRows frame.owner (rawRows parameters profile frame))
        assignment) :
    outputState =
      parameters.machine.step
        Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary.Step.selected
        state witness := by
  have rawHolds :
      RawSatisfies (rawRows parameters profile frame) assignment := by
    exact
      (satisfies_ownRows_iff frame.owner
        (rawRows parameters profile frame) assignment).mp holds
  rcases rawRows_equations parameters profile frame assignment rawHolds with
    ⟨deltaLeft, deltaRight, product, deltaOutput,
      phaseRow, memoryRow, leftRow, rightRow, outputRow,
      haltedRow, trappedRow⟩
  have inputValue (coordinate : StateCoordinate) :=
    inputCoordinate_value parameters profile frame assignment
      state inputDecoded coordinate
  have outputValue (coordinate : StateCoordinate) :=
    outputCoordinate_value parameters profile frame assignment
      outputState outputDecoded coordinate
  have outputPhase :
      profile.stateEquiv.toBenchmark outputState
          StateCoordinate.phase.index =
        1 := by
    rw [activeOne, constantOne, outputValue .phase,
      Fin.one_mul] at phaseRow
    exact
      (Lean.Grind.AddCommGroup.sub_eq_zero_iff.mp phaseRow).symm
  have outputMemory :
      profile.stateEquiv.toBenchmark outputState
          StateCoordinate.memoryWord.index =
        profile.stateEquiv.toBenchmark state
          StateCoordinate.memoryWord.index := by
    rw [activeOne, inputValue .memoryWord,
      outputValue .memoryWord, Fin.one_mul] at memoryRow
    exact
      (Lean.Grind.AddCommGroup.sub_eq_zero_iff.mp memoryRow).symm
  have outputLeft :
      profile.stateEquiv.toBenchmark outputState
          StateCoordinate.leftOperand.index =
        profile.stateEquiv.toBenchmark state
            StateCoordinate.memoryWord.index +
          profile.stateEquiv.toBenchmark state StateCoordinate.phase.index *
            (profile.stateEquiv.toBenchmark state
                StateCoordinate.leftOperand.index -
              profile.stateEquiv.toBenchmark state
                StateCoordinate.memoryWord.index) := by
    rw [activeOne, inputValue .memoryWord,
      outputValue .leftOperand, Fin.one_mul] at leftRow
    have leftExact :=
      Lean.Grind.AddCommGroup.sub_eq_zero_iff.mp leftRow
    rw [inputValue .phase, inputValue .leftOperand,
      inputValue .memoryWord] at deltaLeft
    rw [← deltaLeft] at leftExact
    exact leftExact.symm
  have outputRight :
      profile.stateEquiv.toBenchmark outputState
          StateCoordinate.rightOperand.index =
        6 +
          profile.stateEquiv.toBenchmark state StateCoordinate.phase.index *
            (profile.stateEquiv.toBenchmark state
                StateCoordinate.rightOperand.index - 6) := by
    rw [activeOne, constantOne,
      outputValue .rightOperand, Fin.one_mul, Fin.mul_one] at rightRow
    have rightExact :=
      Lean.Grind.AddCommGroup.sub_eq_zero_iff.mp rightRow
    rw [constantOne, inputValue .phase, inputValue .rightOperand,
      Fin.mul_one] at deltaRight
    rw [← deltaRight] at rightExact
    exact rightExact.symm
  have outputProduct :
      assignment
          (temporaryColumn parameters profile frame ⟨2, by decide⟩) =
        profile.stateEquiv.toBenchmark state
            StateCoordinate.leftOperand.index *
          profile.stateEquiv.toBenchmark state
            StateCoordinate.rightOperand.index := by
    rw [inputValue .leftOperand, inputValue .rightOperand] at product
    exact product.symm
  have outputOutput :
      profile.stateEquiv.toBenchmark outputState
          StateCoordinate.output.index =
        profile.stateEquiv.toBenchmark state StateCoordinate.output.index +
          profile.stateEquiv.toBenchmark state StateCoordinate.phase.index *
            (profile.stateEquiv.toBenchmark state
                StateCoordinate.leftOperand.index *
              profile.stateEquiv.toBenchmark state
                StateCoordinate.rightOperand.index -
              profile.stateEquiv.toBenchmark state
                StateCoordinate.output.index) := by
    rw [activeOne, inputValue .output,
      outputValue .output, Fin.one_mul] at outputRow
    have outputExact :=
      Lean.Grind.AddCommGroup.sub_eq_zero_iff.mp outputRow
    rw [inputValue .phase, inputValue .output,
      outputProduct] at deltaOutput
    rw [← deltaOutput] at outputExact
    exact outputExact.symm
  have outputHalted :
      profile.stateEquiv.toBenchmark outputState
          StateCoordinate.halted.index =
        profile.stateEquiv.toBenchmark state StateCoordinate.phase.index := by
    rw [activeOne, inputValue .phase,
      outputValue .halted, Fin.one_mul] at haltedRow
    exact
      (Lean.Grind.AddCommGroup.sub_eq_zero_iff.mp haltedRow).symm
  have outputTrapped :
      profile.stateEquiv.toBenchmark outputState
          StateCoordinate.trapped.index =
        profile.stateEquiv.toBenchmark state StateCoordinate.trapped.index := by
    rw [activeOne, inputValue .trapped,
      outputValue .trapped, Fin.one_mul] at trappedRow
    exact
      (Lean.Grind.AddCommGroup.sub_eq_zero_iff.mp trappedRow).symm
  apply profile.stateEquiv.leftInverse.injective
  rw [profile.stepExact]
  apply State.ext_coordinates
  intro coordinate
  cases coordinate with
  | phase =>
      simpa [WasmBenchmark42x6.step, read, writeCoordinates] using
        outputPhase
  | memoryWord =>
      simpa [WasmBenchmark42x6.step, read, writeCoordinates] using
        outputMemory
  | leftOperand =>
      simpa [WasmBenchmark42x6.step, read, writeCoordinates] using
        outputLeft
  | rightOperand =>
      simpa [WasmBenchmark42x6.step, read, writeCoordinates] using
        outputRight
  | output =>
      simpa [WasmBenchmark42x6.step, read, writeCoordinates] using
        outputOutput
  | halted =>
      simpa [WasmBenchmark42x6.step, read, writeCoordinates] using
        outputHalted
  | trapped =>
      simpa [WasmBenchmark42x6.step, read, writeCoordinates] using
        outputTrapped

private noncomputable def temporaryValues
    (parameters : Parameters)
    (profile : StepProfile parameters)
    {context : Schema (typeSystem parameters)}
    {stateReference :
      Ref (typeSystem parameters) context (.data .state)}
    {witnessReference :
      Ref (typeSystem parameters) context (.data .witness)}
    (frame :
      CallFrame (signature := signature parameters)
        profile.family Call.step
        (.cons stateReference (.cons witnessReference .nil)))
    (assignment : ColumnId -> Field) : List Field :=
  let phase := assignment (inputColumn parameters profile frame .phase)
  let memory := assignment (inputColumn parameters profile frame .memoryWord)
  let left := assignment (inputColumn parameters profile frame .leftOperand)
  let right := assignment (inputColumn parameters profile frame .rightOperand)
  let output := assignment (inputColumn parameters profile frame .output)
  [ phase * (left - memory)
  , phase * (right - 6 * assignment frame.one)
  , left * right
  , phase * (left * right - output)
  ]

private noncomputable def completion
    (parameters : Parameters)
    (profile : StepProfile parameters)
    {context : Schema (typeSystem parameters)}
    {stateReference :
      Ref (typeSystem parameters) context (.data .state)}
    {witnessReference :
      Ref (typeSystem parameters) context (.data .witness)}
    (frame :
      CallFrame (signature := signature parameters)
        profile.family Call.step
        (.cons stateReference (.cons witnessReference .nil)))
    (assignment : ColumnId -> Field) : ColumnId -> Field :=
  writeColumns assignment frame.temporaries.ids
    (temporaryValues parameters profile frame assignment)

private theorem completion_spec
    (parameters : Parameters)
    (profile : StepProfile parameters)
    {context : Schema (typeSystem parameters)}
    {stateReference :
      Ref (typeSystem parameters) context (.data .state)}
    {witnessReference :
      Ref (typeSystem parameters) context (.data .witness)}
    (frame :
      CallFrame (signature := signature parameters)
        profile.family Call.step
        (.cons stateReference (.cons witnessReference .nil)))
    (assignment : ColumnId -> Field) :
    AgreesOn frame.visibleIds assignment
        (completion parameters profile frame assignment) ∧
      ChangesOnly frame.temporaries.ids assignment
        (completion parameters profile frame assignment) ∧
      completion parameters profile frame assignment
          (temporaryColumn parameters profile frame ⟨0, by decide⟩) =
        (temporaryValues parameters profile frame assignment).getD 0 0 ∧
      completion parameters profile frame assignment
          (temporaryColumn parameters profile frame ⟨1, by decide⟩) =
        (temporaryValues parameters profile frame assignment).getD 1 0 ∧
      completion parameters profile frame assignment
          (temporaryColumn parameters profile frame ⟨2, by decide⟩) =
        (temporaryValues parameters profile frame assignment).getD 2 0 ∧
      completion parameters profile frame assignment
          (temporaryColumn parameters profile frame ⟨3, by decide⟩) =
        (temporaryValues parameters profile frame assignment).getD 3 0 := by
  have temporaryNodup : frame.temporaries.ids.Nodup :=
    (List.nodup_append.mp frame.allocationsNodup).2.1
  have lengthEqual :
      frame.temporaries.ids.length =
        (temporaryValues parameters profile frame assignment).length := by
    rw [temporary_ids_exact parameters profile frame]
    rfl
  have recovered :=
    writeColumns_map_eq assignment frame.temporaries.ids
      (temporaryValues parameters profile frame assignment)
      lengthEqual temporaryNodup
  have valuesExact :
      [ completion parameters profile frame assignment
          (temporaryColumn parameters profile frame ⟨0, by decide⟩)
      , completion parameters profile frame assignment
          (temporaryColumn parameters profile frame ⟨1, by decide⟩)
      , completion parameters profile frame assignment
          (temporaryColumn parameters profile frame ⟨2, by decide⟩)
      , completion parameters profile frame assignment
          (temporaryColumn parameters profile frame ⟨3, by decide⟩)
      ] =
        temporaryValues parameters profile frame assignment := by
    simpa [completion,
      temporary_ids_exact parameters profile frame] using recovered
  have separated :
      completion parameters profile frame assignment
            (temporaryColumn parameters profile frame ⟨0, by decide⟩) =
          (temporaryValues parameters profile frame assignment).getD 0 0 ∧
        completion parameters profile frame assignment
            (temporaryColumn parameters profile frame ⟨1, by decide⟩) =
          (temporaryValues parameters profile frame assignment).getD 1 0 ∧
        completion parameters profile frame assignment
            (temporaryColumn parameters profile frame ⟨2, by decide⟩) =
          (temporaryValues parameters profile frame assignment).getD 2 0 ∧
        completion parameters profile frame assignment
            (temporaryColumn parameters profile frame ⟨3, by decide⟩) =
          (temporaryValues parameters profile frame assignment).getD 3 0 := by
    simpa [temporaryValues] using congrArg
      (fun values =>
        (values.getD 0 0, values.getD 1 0,
          values.getD 2 0, values.getD 3 0))
      valuesExact
  exact ⟨
    writeColumns_agreesOn assignment frame.temporaries.ids
      frame.visibleIds
      (temporaryValues parameters profile frame assignment)
      frame.temporariesDisjointVisible,
    writeColumns_changesOnly assignment frame.temporaries.ids
      (temporaryValues parameters profile frame assignment),
    separated.1, separated.2.1, separated.2.2.1,
    separated.2.2.2⟩

/-- Exact stable row occurrences for one Step call. -/
noncomputable def rows
    (parameters : Parameters)
    (profile : StepProfile parameters)
    {context : Schema (typeSystem parameters)}
    {stateReference :
      Ref (typeSystem parameters) context (.data .state)}
    {witnessReference :
      Ref (typeSystem parameters) context (.data .witness)}
    (frame :
      CallFrame (signature := signature parameters)
        profile.family Call.step
        (.cons stateReference (.cons witnessReference .nil))) :
    List OwnedRow :=
  ownRows frame.owner (rawRows parameters profile frame)

@[simp] theorem rows_length
    (parameters : Parameters)
    (profile : StepProfile parameters)
    {context : Schema (typeSystem parameters)}
    {stateReference :
      Ref (typeSystem parameters) context (.data .state)}
    {witnessReference :
      Ref (typeSystem parameters) context (.data .witness)}
    (frame :
      CallFrame (signature := signature parameters)
        profile.family Call.step
        (.cons stateReference (.cons witnessReference .nil))) :
    (rows parameters profile frame).length = 11 := by
  simp [rows]

theorem rows_owner
    (parameters : Parameters)
    (profile : StepProfile parameters)
    {context : Schema (typeSystem parameters)}
    {stateReference :
      Ref (typeSystem parameters) context (.data .state)}
    {witnessReference :
      Ref (typeSystem parameters) context (.data .witness)}
    (frame :
      CallFrame (signature := signature parameters)
        profile.family Call.step
        (.cons stateReference (.cons witnessReference .nil)))
    (row : OwnedRow)
    (member : row ∈ rows parameters profile frame) :
    row.id.owner = frame.owner :=
  ownRows_owner frame.owner (rawRows parameters profile frame) row member

theorem rowIds_nodup
    (parameters : Parameters)
    (profile : StepProfile parameters)
    {context : Schema (typeSystem parameters)}
    {stateReference :
      Ref (typeSystem parameters) context (.data .state)}
    {witnessReference :
      Ref (typeSystem parameters) context (.data .witness)}
    (frame :
      CallFrame (signature := signature parameters)
        profile.family Call.step
        (.cons stateReference (.cons witnessReference .nil))) :
    ((rows parameters profile frame).map fun row => row.id).Nodup :=
  ownRows_ids_nodup frame.owner (rawRows parameters profile frame)

theorem rows_supported
    (parameters : Parameters)
    (profile : StepProfile parameters)
    {context : Schema (typeSystem parameters)}
    {stateReference :
      Ref (typeSystem parameters) context (.data .state)}
    {witnessReference :
      Ref (typeSystem parameters) context (.data .witness)}
    (frame :
      CallFrame (signature := signature parameters)
        profile.family Call.step
        (.cons stateReference (.cons witnessReference .nil)))
    (row : OwnedRow)
    (member : row ∈ rows parameters profile frame)
    (column : ColumnId)
    (columnMember : column ∈ row.columnIds) :
    column ∈ frame.visibleIds ++ frame.temporaries.ids := by
  apply ownRows_supported frame.owner
    (rawRows parameters profile frame)
    (frame.visibleIds ++ frame.temporaries.ids)
    (rawRows_supported parameters profile frame)
    row
  · simpa [rows] using member
  · exact columnMember

theorem active_soundness
    (parameters : Parameters)
    (profile : StepProfile parameters)
    {context : Schema (typeSystem parameters)}
    {stateReference :
      Ref (typeSystem parameters) context (.data .state)}
    {witnessReference :
      Ref (typeSystem parameters) context (.data .witness)}
    (frame :
      CallFrame (signature := signature parameters)
        profile.family Call.step
        (.cons stateReference (.cons witnessReference .nil)))
    (assignment : ColumnId -> Field)
    (inputs :
      HVec (typeSystem parameters).Value
        ((signature parameters).callInputs Call.step))
    (constantOne : assignment frame.one = 1)
    (activeOne : assignment frame.active = 1)
    (decoded : frame.operands.Decodes profile.family assignment inputs)
    (holds : Satisfies (rows parameters profile frame) assignment) :
    ∃ outputs :
        Schema.Values (typeSystem parameters)
          ((signature parameters).callOutputs Call.step),
      (signature parameters).callEval Call.step inputs = some outputs ∧
        frame.outputs.Decodes profile.family assignment outputs := by
  cases inputs with
  | cons state tail =>
      cases tail with
      | cons witness tail =>
          cases tail
          have inputDecoded :=
            (binaryOperand_decodes_iff profile.family assignment
              frame.operands state witness).mp decoded
          rcases output_decode_exists parameters profile frame assignment with
            ⟨outputState, outputDecoded⟩
          have outputExact :=
            decodedOutput_eq_step parameters profile frame assignment
              state outputState witness constantOne activeOne
              inputDecoded.1 outputDecoded
              (by simpa [rows] using holds)
          subst outputState
          refine
            ⟨.cons
                (parameters.machine.step
                  Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary.Step.selected
                  state witness)
                .nil,
              rfl, ?_⟩
          apply (unaryOutput_decodes_iff profile.family assignment
            frame.outputs _).mpr
          exact outputDecoded

theorem active_honest_completeness
    (parameters : Parameters)
    (profile : StepProfile parameters)
    {context : Schema (typeSystem parameters)}
    {stateReference :
      Ref (typeSystem parameters) context (.data .state)}
    {witnessReference :
      Ref (typeSystem parameters) context (.data .witness)}
    (frame :
      CallFrame (signature := signature parameters)
        profile.family Call.step
        (.cons stateReference (.cons witnessReference .nil)))
    (assignment : ColumnId -> Field)
    (inputs :
      HVec (typeSystem parameters).Value
        ((signature parameters).callInputs Call.step))
    (outputs :
      Schema.Values (typeSystem parameters)
        ((signature parameters).callOutputs Call.step))
    (constantOne : assignment frame.one = 1)
    (activeOne : assignment frame.active = 1)
    (inputsEncoded :
      frame.operands.Encodes profile.family assignment inputs)
    (outputsEncoded :
      frame.outputs.Encodes profile.family assignment outputs)
    (evaluated :
      (signature parameters).callEval Call.step inputs = some outputs) :
    ∃ completed : ColumnId -> Field,
      AgreesOn frame.visibleIds assignment completed ∧
        ChangesOnly frame.temporaries.ids assignment completed ∧
        Satisfies (rows parameters profile frame) completed := by
  cases inputs with
  | cons state inputTail =>
      cases inputTail with
      | cons witness inputTail =>
          cases inputTail
          cases outputs with
          | cons outputState outputTail =>
              cases outputTail
              have outputExact :
                  outputState =
                    parameters.machine.step
                      Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary.Step.selected
                      state witness :=
                congrArg HVec.head (Option.some.inj evaluated.symm)
              have inputEncoded :=
                (binaryOperand_encodes_iff profile.family assignment
                  frame.operands state witness).mp inputsEncoded
              have inputDecoded :=
                (firstBinaryOperand frame.operands).decodes_of_encodes
                  profile.family (.data .state) assignment state
                  inputEncoded.1
              have outputEncoded :=
                (unaryOutput_encodes_iff profile.family assignment
                  frame.outputs outputState).mp outputsEncoded
              have outputDecoded :=
                (unaryOutput frame.outputs).decodes_of_encodes
                  profile.family (.data .state) assignment outputState
                  outputEncoded
              rcases completion_spec parameters profile frame assignment with
                ⟨agrees, changes, temp0, temp1, temp2, temp3⟩
              simp [temporaryValues] at temp0 temp1 temp2 temp3
              let completed :=
                completion parameters profile frame assignment
              change completed
                  (temporaryColumn parameters profile frame ⟨0, by decide⟩) =
                _ at temp0
              change completed
                  (temporaryColumn parameters profile frame ⟨1, by decide⟩) =
                _ at temp1
              change completed
                  (temporaryColumn parameters profile frame ⟨2, by decide⟩) =
                _ at temp2
              change completed
                  (temporaryColumn parameters profile frame ⟨3, by decide⟩) =
                _ at temp3
              have inputPreserved (coordinate : StateCoordinate) :
                  completed (inputColumn parameters profile frame coordinate) =
                    assignment
                      (inputColumn parameters profile frame coordinate) :=
                agrees _ (inputColumn_mem_visible
                  parameters profile frame coordinate)
              have outputPreserved (coordinate : StateCoordinate) :
                  completed
                      (outputColumn parameters profile frame coordinate) =
                    assignment
                      (outputColumn parameters profile frame coordinate) :=
                agrees _ (outputColumn_mem_visible
                  parameters profile frame coordinate)
              have onePreserved :
                  completed frame.one = assignment frame.one :=
                agrees frame.one (by simp [CallFrame.visibleIds])
              have activePreserved :
                  completed frame.active = assignment frame.active :=
                agrees frame.active (by simp [CallFrame.visibleIds])
              have inputValue (coordinate : StateCoordinate) :=
                inputCoordinate_value parameters profile frame assignment
                  state inputDecoded coordinate
              have outputValue (coordinate : StateCoordinate) :=
                outputCoordinate_value parameters profile frame assignment
                  outputState outputDecoded coordinate
              have semantic :
                  profile.stateEquiv.toBenchmark outputState =
                    WasmBenchmark42x6.step
                      (profile.stateEquiv.toBenchmark state)
                      WasmBenchmark42x6.noWitness := by
                rw [outputExact, profile.stepExact]
              have outputSemantic (coordinate : StateCoordinate) :
                  assignment
                      (outputColumn parameters profile frame coordinate) =
                    WasmBenchmark42x6.step
                        (profile.stateEquiv.toBenchmark state)
                        WasmBenchmark42x6.noWitness coordinate.index := by
                rw [outputValue]
                exact congrFun semantic coordinate.index
              have equations :
                  completed
                        (inputColumn parameters profile frame .phase) *
                      (completed
                          (inputColumn parameters profile frame .leftOperand) -
                        completed
                          (inputColumn parameters profile frame .memoryWord)) =
                    completed
                      (temporaryColumn parameters profile frame ⟨0, by decide⟩) ∧
                  completed
                        (inputColumn parameters profile frame .phase) *
                      (completed
                          (inputColumn parameters profile frame .rightOperand) -
                        6 * completed frame.one) =
                    completed
                      (temporaryColumn parameters profile frame ⟨1, by decide⟩) ∧
                  completed
                        (inputColumn parameters profile frame .leftOperand) *
                      completed
                        (inputColumn parameters profile frame .rightOperand) =
                    completed
                      (temporaryColumn parameters profile frame ⟨2, by decide⟩) ∧
                  completed
                        (inputColumn parameters profile frame .phase) *
                      (completed
                          (temporaryColumn parameters profile frame ⟨2, by decide⟩) -
                        completed
                          (inputColumn parameters profile frame .output)) =
                    completed
                      (temporaryColumn parameters profile frame ⟨3, by decide⟩) ∧
                  completed frame.active *
                      (completed frame.one -
                        completed (outputColumn parameters profile frame .phase)) = 0 ∧
                  completed frame.active *
                      (completed (inputColumn parameters profile frame .memoryWord) -
                        completed (outputColumn parameters profile frame .memoryWord)) = 0 ∧
                  completed frame.active *
                      (completed (inputColumn parameters profile frame .memoryWord) +
                        completed
                          (temporaryColumn parameters profile frame ⟨0, by decide⟩) -
                        completed (outputColumn parameters profile frame .leftOperand)) = 0 ∧
                  completed frame.active *
                      (6 * completed frame.one +
                        completed
                          (temporaryColumn parameters profile frame ⟨1, by decide⟩) -
                        completed (outputColumn parameters profile frame .rightOperand)) = 0 ∧
                  completed frame.active *
                      (completed (inputColumn parameters profile frame .output) +
                        completed
                          (temporaryColumn parameters profile frame ⟨3, by decide⟩) -
                        completed (outputColumn parameters profile frame .output)) = 0 ∧
                  completed frame.active *
                      (completed (inputColumn parameters profile frame .phase) -
                        completed (outputColumn parameters profile frame .halted)) = 0 ∧
                  completed frame.active *
                      (completed (inputColumn parameters profile frame .trapped) -
                        completed (outputColumn parameters profile frame .trapped)) = 0 := by
                simp only [inputPreserved, outputPreserved, onePreserved,
                  activePreserved, constantOne, activeOne]
                rw [temp0, temp1, temp2, temp3]
                simp [constantOne, inputValue, outputSemantic,
                  WasmBenchmark42x6.step, read, writeCoordinates,
                  StateCoordinate.index, Fin.one_mul, Fin.mul_one,
                  Fin.sub_self, Fin.mul_zero]
              refine ⟨completed, agrees, changes, ?_⟩
              apply (satisfies_ownRows_iff frame.owner
                (rawRows parameters profile frame) completed).mpr
              simpa [rawRows, Row.Holds, LinearCombination.eval,
                Nightstream.Implementation.Lowering.Goldilocks.singleton,
                difference, affine3, Fin.one_mul, Fin.mul_one, Fin.add_zero,
                Lean.Grind.Fin.neg_mul, Fin.sub_eq_add_neg,
                Lean.Grind.Fin.add_assoc] using equations

theorem inactive_satisfiable
    (parameters : Parameters)
    (profile : StepProfile parameters)
    {context : Schema (typeSystem parameters)}
    {stateReference :
      Ref (typeSystem parameters) context (.data .state)}
    {witnessReference :
      Ref (typeSystem parameters) context (.data .witness)}
    (frame :
      CallFrame (signature := signature parameters)
        profile.family Call.step
        (.cons stateReference (.cons witnessReference .nil)))
    (assignment : ColumnId -> Field)
    (constantOne : assignment frame.one = 1)
    (activeZero : assignment frame.active = 0) :
    ∃ completed : ColumnId -> Field,
      AgreesOn frame.visibleIds assignment completed ∧
        ChangesOnly frame.temporaries.ids assignment completed ∧
        Satisfies (rows parameters profile frame) completed := by
  rcases completion_spec parameters profile frame assignment with
    ⟨agrees, changes, temp0, temp1, temp2, temp3⟩
  simp [temporaryValues] at temp0 temp1 temp2 temp3
  let completed := completion parameters profile frame assignment
  change completed
      (temporaryColumn parameters profile frame ⟨0, by decide⟩) = _ at temp0
  change completed
      (temporaryColumn parameters profile frame ⟨1, by decide⟩) = _ at temp1
  change completed
      (temporaryColumn parameters profile frame ⟨2, by decide⟩) = _ at temp2
  change completed
      (temporaryColumn parameters profile frame ⟨3, by decide⟩) = _ at temp3
  have inputPreserved (coordinate : StateCoordinate) :
      completed (inputColumn parameters profile frame coordinate) =
        assignment (inputColumn parameters profile frame coordinate) :=
    agrees _ (inputColumn_mem_visible parameters profile frame coordinate)
  have onePreserved : completed frame.one = assignment frame.one :=
    agrees frame.one (by simp [CallFrame.visibleIds])
  have activePreserved : completed frame.active = assignment frame.active :=
    agrees frame.active (by simp [CallFrame.visibleIds])
  have firstRow :
      completed (inputColumn parameters profile frame .phase) *
          (completed (inputColumn parameters profile frame .leftOperand) -
            completed (inputColumn parameters profile frame .memoryWord)) =
        completed
          (temporaryColumn parameters profile frame ⟨0, by decide⟩) := by
    rw [inputPreserved, inputPreserved, inputPreserved, temp0]
  have secondRow :
      completed (inputColumn parameters profile frame .phase) *
          (completed (inputColumn parameters profile frame .rightOperand) -
            6 * completed frame.one) =
        completed
          (temporaryColumn parameters profile frame ⟨1, by decide⟩) := by
    rw [inputPreserved, inputPreserved, onePreserved, temp1]
  have thirdRow :
      completed (inputColumn parameters profile frame .leftOperand) *
          completed (inputColumn parameters profile frame .rightOperand) =
        completed
          (temporaryColumn parameters profile frame ⟨2, by decide⟩) := by
    rw [inputPreserved, inputPreserved, temp2]
  have fourthRow :
      completed (inputColumn parameters profile frame .phase) *
          (completed
              (temporaryColumn parameters profile frame ⟨2, by decide⟩) -
            completed (inputColumn parameters profile frame .output)) =
        completed
          (temporaryColumn parameters profile frame ⟨3, by decide⟩) := by
    rw [inputPreserved, inputPreserved, temp2, temp3]
  refine ⟨completed, agrees, changes, ?_⟩
  apply (satisfies_ownRows_iff frame.owner
    (rawRows parameters profile frame) completed).mpr
  simp [rawRows, Row.Holds, LinearCombination.eval,
    Nightstream.Implementation.Lowering.Goldilocks.singleton,
    difference, affine3, activePreserved, activeZero,
    Fin.one_mul, Fin.mul_one, Fin.add_zero, Fin.zero_mul,
    Lean.Grind.Fin.neg_mul, Fin.sub_eq_add_neg,
    Lean.Grind.Fin.add_assoc]
  simpa only [Fin.sub_eq_add_neg] using
    And.intro firstRow
      (And.intro secondRow (And.intro thirdRow fourthRow))

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6
