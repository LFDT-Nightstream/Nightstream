import Nightstream.Implementation.NebulaV2.Production.FPrime.Terminal.InvocationRowsSoundFor
import Nightstream.Implementation.R1CS.Core.LinearOutputs

/-!
Contract: exact terminal public-result link rows for one production candidate.

The rows link the terminally consumed F-prime state and its closed memory
carry to verifier-owned public statement columns. They check the invocation
count, immutable initial application state, final application state, real-row
count, final segment, final timestamp, and all four final-memory-root lanes.

The soundness theorem derives `PublicChecks`; it does not take that structure
as a premise. `StatementPlaced` is only the typed parser-to-column boundary.

Assurance tier: exponent-indexed implementation-to-protocol bridge.

Does not own public-byte parsing, bit-to-field recomposition rows, the WASM
application compiler, generated-artifact containment, Rust refinement, or a
compact terminal backend.

Emits constraints: 178 rows.
-/

set_option autoImplicit false
set_option maxRecDepth 10000

namespace Nightstream.Implementation.NebulaV2.ProductionPaperTerminalPublicRowsFor

open Nightstream.Implementation.NebulaV2
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.LinearOutputs
open Nightstream.Implementation.R1CS.Program
open Nightstream.Protocol.NebulaV2
open Nightstream.Protocol.NebulaV2.FPrime
open Nightstream.Protocol.NebulaV2.ProductionProfileCandidates
open Nightstream.Protocol.NebulaV2.WasmPublicStatementEncoding
open Nightstream.Protocol.NebulaV2.WasmStatement
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

abbrev FullShape
    (rowVariables logicalWidth : Nat)
    (publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth) :=
  ProductPaperAlgebraFor.FullShape rowVariables logicalWidth publicFits

/-- Recombined public values. A later codec row family must derive these
columns from the exact 7,868 public statement bits. -/
structure StatementLayout where
  segmentCountColumn : Nat
  finalTimestampColumn : Nat
  initialApplicationColumn : Fin 85 -> Nat
  realApplicationRowsColumn : Nat
  finalApplicationColumn : Fin 85 -> Nat
  finalMemoryRootColumn : Fin 4 -> Nat

/-- Exact typed placement of the verifier-owned statement at the recombined
public columns. This structure contains no terminal equality conclusion. -/
structure StatementPlaced
    {Program : Type} (layout : StatementLayout) (assignment : Nat -> Nat)
    (statement : ProductionStatement Program) : Prop where
  segmentCount : assignment layout.segmentCountColumn =
    statement.base.segmentCount
  finalTimestamp : assignment layout.finalTimestampColumn =
    statement.base.finalGlobalTimestamp
  initialApplication : forall index : Fin 85,
    assignment (layout.initialApplicationColumn index) =
      (ProductionWasmStateFields.encode
        (WasmStateEncoding.encode statement.base.initialApplicationState)).get
        (Fin.cast
          (ProductionWasmStateFields.encode_length
            (WasmStateEncoding.encode
              statement.base.initialApplicationState)).symm index)
  realApplicationRows : assignment layout.realApplicationRowsColumn =
    statement.resultImage.realApplicationRowCount
  finalApplication : forall index : Fin 85,
    assignment (layout.finalApplicationColumn index) =
      (ProductionWasmStateFields.encode
        statement.resultImage.finalApplicationState).get
        (Fin.cast
          (ProductionWasmStateFields.encode_length
            statement.resultImage.finalApplicationState).symm index)
  finalMemoryRoot : forall lane : Fin 4,
    assignment (layout.finalMemoryRootColumn lane) =
      (statement.resultImage.finalMemoryRoot.lanes lane).val

def link (output input coefficient : Nat) : Check where
  output := output
  terms := [(input, coefficient)]
  orientation := .forward

def invocationCheck
    (candidate : Id) {rowVariables : Nat}
    (prior : ProductionPaperPriorStateAuthorityRowsFor.Layout candidate
      rowVariables) (statement : StatementLayout) : Check :=
  link prior.state.invocationColumn statement.segmentCountColumn
    (claimsPerSegment candidate)

def initialApplicationChecks
    {candidate : Id} {rowVariables : Nat}
    (prior : ProductionPaperPriorStateAuthorityRowsFor.Layout candidate
      rowVariables) (statement : StatementLayout) : List Check :=
  List.ofFn fun index : Fin 85 =>
    link (prior.state.initialApplicationColumn index)
      (statement.initialApplicationColumn index) 1

def finalApplicationChecks
    {candidate : Id} {rowVariables : Nat}
    (prior : ProductionPaperPriorStateAuthorityRowsFor.Layout candidate
      rowVariables) (statement : StatementLayout) : List Check :=
  List.ofFn fun index : Fin 85 =>
    link (prior.state.applicationColumn index)
      (statement.finalApplicationColumn index) 1

def finalCarry
    (candidate : Id) (rowVariables : Nat)
    (prior : ProductionPaperPriorStateAuthorityRowsFor.Layout candidate
      rowVariables) : MemoryCarryPublicRows.Layout :=
  (prior.ccs.core.batch.frame.memory.boundaries
    (Fin.last (ProductionMemoryCheckedBatchRows.StepCount candidate))).reference

def scalarChecks
    (candidate : Id) {rowVariables : Nat}
    (prior : ProductionPaperPriorStateAuthorityRowsFor.Layout candidate
      rowVariables) (statement : StatementLayout) : List Check :=
  [ link prior.state.realRowsColumn statement.realApplicationRowsColumn 1
  , link ((finalCarry candidate rowVariables prior).carry.fieldColumn
      .segmentIndex)
      statement.segmentCountColumn 1
  , link ((finalCarry candidate rowVariables prior).carry.fieldColumn
      .globalTimestamp)
      statement.finalTimestampColumn 1
  ]

def finalMemoryRootChecks
    (candidate : Id) {rowVariables : Nat}
    (prior : ProductionPaperPriorStateAuthorityRowsFor.Layout candidate
      rowVariables) (statement : StatementLayout) : List Check :=
  List.ofFn fun lane : Fin 4 =>
    link
      ((finalCarry candidate rowVariables prior).carry.fieldColumn
        (.root .memory lane))
      (statement.finalMemoryRootColumn lane) 1

def checks
    (candidate : Id) {rowVariables : Nat}
    (prior : ProductionPaperPriorStateAuthorityRowsFor.Layout candidate
      rowVariables) (statement : StatementLayout) : List Check :=
  [invocationCheck candidate prior statement] ++
    (initialApplicationChecks prior statement ++
      (finalApplicationChecks prior statement ++
        (scalarChecks candidate prior statement ++
          finalMemoryRootChecks candidate prior statement)))

def rows
    (candidate : Id) {rowVariables : Nat}
    (prior : ProductionPaperPriorStateAuthorityRowsFor.Layout candidate
      rowVariables) (statement : StatementLayout) : List Row :=
  LinearOutputs.rows (checks candidate prior statement)

theorem rows_length_exact
    (candidate : Id) {rowVariables : Nat}
    (prior : ProductionPaperPriorStateAuthorityRowsFor.Layout candidate
      rowVariables) (statement : StatementLayout) :
    (rows candidate prior statement).length = 178 := by
  simp [rows, LinearOutputs.rows, checks, initialApplicationChecks,
    finalApplicationChecks, scalarChecks, finalMemoryRootChecks]

private theorem link_canonical
    {output input coefficient : Nat}
    (positive : 0 < coefficient) (bounded : coefficient < goldilocksP) :
    (link output input coefficient).Canonical := by
  intro term member
  have exact : term = (input, coefficient) := by
    simpa [link] using member
  subst term
  exact And.intro positive bounded

private theorem claimsPerSegment_positive (candidate : Id) :
    0 < claimsPerSegment candidate := by
  cases candidate <;> decide

private theorem claimsPerSegment_below_field (candidate : Id) :
    claimsPerSegment candidate < goldilocksP := by
  cases candidate <;> norm_num [claimsPerSegment, stepsPerSegment,
    checkedStepsPerFreshClaim, goldilocksP]

theorem checks_canonical
    (candidate : Id) {rowVariables : Nat}
    (prior : ProductionPaperPriorStateAuthorityRowsFor.Layout candidate
      rowVariables) (statement : StatementLayout) :
    LinearOutputs.Canonical (checks candidate prior statement) := by
  intro check member
  rw [checks] at member
  rcases List.mem_append.mp member with invocation | rest
  · have equal : check = invocationCheck candidate prior statement := by
      simpa using invocation
    subst check
    exact link_canonical (claimsPerSegment_positive candidate)
      (claimsPerSegment_below_field candidate)
  rcases List.mem_append.mp rest with initial | rest
  · rcases List.mem_ofFn.mp initial with ⟨index, rfl⟩
    exact link_canonical (by decide) (by norm_num [goldilocksP])
  rcases List.mem_append.mp rest with final | rest
  · rcases List.mem_ofFn.mp final with ⟨index, rfl⟩
    exact link_canonical (by decide) (by norm_num [goldilocksP])
  rcases List.mem_append.mp rest with scalar | root
  · have cases :
        check = link prior.state.realRowsColumn
            statement.realApplicationRowsColumn 1 ∨
        check = link
            ((finalCarry candidate rowVariables prior).carry.fieldColumn
              .segmentIndex)
            statement.segmentCountColumn 1 ∨
        check = link
            ((finalCarry candidate rowVariables prior).carry.fieldColumn
              .globalTimestamp)
            statement.finalTimestampColumn 1 := by
      simpa [scalarChecks] using scalar
    rcases cases with rfl | rfl | rfl
    all_goals exact link_canonical (by decide) (by norm_num [goldilocksP])
  · rcases List.mem_ofFn.mp root with ⟨lane, rfl⟩
    exact link_canonical (by decide) (by norm_num [goldilocksP])

private theorem invocationCheck_mem
    (candidate : Id) {rowVariables : Nat}
    (prior : ProductionPaperPriorStateAuthorityRowsFor.Layout candidate
      rowVariables) (statement : StatementLayout) :
    invocationCheck candidate prior statement ∈
      checks candidate prior statement := by
  rw [checks]
  exact List.mem_append_left _ (by simp)

private theorem initialApplicationCheck_mem
    {candidate : Id} {rowVariables : Nat}
    (prior : ProductionPaperPriorStateAuthorityRowsFor.Layout candidate
      rowVariables) (statement : StatementLayout) (index : Fin 85) :
    link (prior.state.initialApplicationColumn index)
        (statement.initialApplicationColumn index) 1 ∈
      checks candidate prior statement := by
  rw [checks]
  apply List.mem_append_right
  apply List.mem_append_left
  exact List.mem_ofFn.mpr ⟨index, rfl⟩

private theorem finalApplicationCheck_mem
    {candidate : Id} {rowVariables : Nat}
    (prior : ProductionPaperPriorStateAuthorityRowsFor.Layout candidate
      rowVariables) (statement : StatementLayout) (index : Fin 85) :
    link (prior.state.applicationColumn index)
        (statement.finalApplicationColumn index) 1 ∈
      checks candidate prior statement := by
  rw [checks]
  apply List.mem_append_right
  apply List.mem_append_right
  apply List.mem_append_left
  exact List.mem_ofFn.mpr ⟨index, rfl⟩

private theorem scalarCheck_mem
    {candidate : Id} {rowVariables : Nat}
    (prior : ProductionPaperPriorStateAuthorityRowsFor.Layout candidate
      rowVariables) (statement : StatementLayout)
    {check : Check} (member : check ∈ scalarChecks candidate prior statement) :
    check ∈ checks candidate prior statement := by
  rw [checks]
  apply List.mem_append_right
  apply List.mem_append_right
  apply List.mem_append_right
  exact List.mem_append_left _ member

private theorem finalMemoryRootCheck_mem
    (candidate : Id) {rowVariables : Nat}
    (prior : ProductionPaperPriorStateAuthorityRowsFor.Layout candidate
      rowVariables) (statement : StatementLayout) (lane : Fin 4) :
    link
        ((finalCarry candidate rowVariables prior).carry.fieldColumn
          (.root .memory lane))
        (statement.finalMemoryRootColumn lane) 1 ∈
      checks candidate prior statement := by
  rw [checks]
  apply List.mem_append_right
  apply List.mem_append_right
  apply List.mem_append_right
  apply List.mem_append_right
  exact List.mem_ofFn.mpr ⟨lane, rfl⟩

private theorem link_value
    {assignment : Nat -> Nat} {output input coefficient : Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    {allChecks : List Check}
    (allCanonical : LinearOutputs.Canonical allChecks)
    (holds : Satisfies (LinearOutputs.rows allChecks) assignment)
    (member : link output input coefficient ∈ allChecks) :
    assignment output =
      (coefficient * assignment input) % goldilocksP := by
  have exact := LinearOutputs.rows_sound canonical one allCanonical holds
    (link output input coefficient) member
  simpa [link, Check.expected, lcEval, rawLcEval] using exact

private theorem link_equal
    {assignment : Nat -> Nat} {output input : Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    {allChecks : List Check}
    (allCanonical : LinearOutputs.Canonical allChecks)
    (holds : Satisfies (LinearOutputs.rows allChecks) assignment)
    (member : link output input 1 ∈ allChecks) :
    assignment output = assignment input := by
  have exact := link_value canonical one allCanonical holds member
  simpa [Nat.mod_eq_of_lt (canonical input)] using exact

private theorem segment_product_below_field
    {Program : Type} {candidate : Id}
    {statement : ProductionStatement Program}
    (bound : statement.base.segmentCount <=
      Nightstream.Protocol.NebulaV2.Lifecycle.maximumSegments) :
    claimsPerSegment candidate * statement.base.segmentCount < goldilocksP := by
  cases candidate <;>
    norm_num [claimsPerSegment, stepsPerSegment,
      checkedStepsPerFreshClaim,
      Nightstream.Protocol.NebulaV2.Lifecycle.maximumSegments,
      goldilocksP] at bound ⊢ <;> omega

private theorem initial_application_fields_equal
    {Program : Type} {candidate : Id} {rowVariables : Nat}
    {fullShape : Phi81Relation.Shape}
    {contract : ProductNifsCodec.FullShapeContractFor rowVariables fullShape}
    {prior : ProductionPaperPriorStateAuthorityRowsFor.Layout candidate
      rowVariables}
    {publicLayout : StatementLayout} {assignment : Nat -> Nat}
    {statement : ProductionStatement Program}
    {priorState : ProductionSuccessorStateBinding.Value candidate fullShape}
    (priorPlaced : ProductionSuccessorStateBindingRowsFor.Placed contract
      prior.state assignment priorState)
    (statementPlaced : StatementPlaced publicLayout assignment statement)
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (holds : Satisfies (rows candidate prior publicLayout) assignment) :
    ProductionWasmStateFields.encode priorState.initialApplicationState =
      ProductionWasmStateFields.encode
        (WasmStateEncoding.encode statement.base.initialApplicationState) := by
  apply List.ext_get
  · rw [ProductionWasmStateFields.encode_length,
      ProductionWasmStateFields.encode_length]
  · intro index leftBound rightBound
    have coordinateBound : index < 85 := by
      simpa [ProductionWasmStateFields.encode_length] using leftBound
    let coordinate : Fin 85 := ⟨index, coordinateBound⟩
    have linked := link_equal canonical one
      (checks_canonical candidate prior publicLayout) holds
      (output := prior.state.initialApplicationColumn coordinate)
      (input := publicLayout.initialApplicationColumn coordinate)
      (initialApplicationCheck_mem prior publicLayout coordinate)
    calc
      (ProductionWasmStateFields.encode
          priorState.initialApplicationState).get ⟨index, leftBound⟩ =
          assignment (prior.state.initialApplicationColumn coordinate) := by
        exact (by simpa [coordinate] using
          (priorPlaced.initialApplication coordinate).symm)
      _ = assignment (publicLayout.initialApplicationColumn coordinate) :=
        linked
      _ = (ProductionWasmStateFields.encode
          (WasmStateEncoding.encode statement.base.initialApplicationState)).get
            ⟨index, rightBound⟩ := by
        exact (by simpa [coordinate] using
          statementPlaced.initialApplication coordinate)

private theorem final_application_fields_equal
    {Program : Type} {candidate : Id} {rowVariables : Nat}
    {fullShape : Phi81Relation.Shape}
    {contract : ProductNifsCodec.FullShapeContractFor rowVariables fullShape}
    {prior : ProductionPaperPriorStateAuthorityRowsFor.Layout candidate
      rowVariables}
    {publicLayout : StatementLayout} {assignment : Nat -> Nat}
    {statement : ProductionStatement Program}
    {priorState : ProductionSuccessorStateBinding.Value candidate fullShape}
    (priorPlaced : ProductionSuccessorStateBindingRowsFor.Placed contract
      prior.state assignment priorState)
    (statementPlaced : StatementPlaced publicLayout assignment statement)
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (holds : Satisfies (rows candidate prior publicLayout) assignment) :
    ProductionWasmStateFields.encode priorState.applicationState =
      ProductionWasmStateFields.encode
        statement.resultImage.finalApplicationState := by
  apply List.ext_get
  · rw [ProductionWasmStateFields.encode_length,
      ProductionWasmStateFields.encode_length]
  · intro index leftBound rightBound
    have coordinateBound : index < 85 := by
      simpa [ProductionWasmStateFields.encode_length] using leftBound
    let coordinate : Fin 85 := ⟨index, coordinateBound⟩
    have linked := link_equal canonical one
      (checks_canonical candidate prior publicLayout) holds
      (output := prior.state.applicationColumn coordinate)
      (input := publicLayout.finalApplicationColumn coordinate)
      (finalApplicationCheck_mem prior publicLayout coordinate)
    calc
      (ProductionWasmStateFields.encode priorState.applicationState).get
          ⟨index, leftBound⟩ =
          assignment (prior.state.applicationColumn coordinate) := by
        exact (by simpa [coordinate] using
          (priorPlaced.application coordinate).symm)
      _ = assignment (publicLayout.finalApplicationColumn coordinate) :=
        linked
      _ = (ProductionWasmStateFields.encode
          statement.resultImage.finalApplicationState).get
            ⟨index, rightBound⟩ := by
        exact (by simpa [coordinate] using
          statementPlaced.finalApplication coordinate)

/-- The terminal public checks are consequences of the link rows, the exact
typed prior-state placement, the exact parsed final carry, and parser-owned
public statement placement. -/
theorem rows_imply_publicChecks
    {Program : Type} (candidate : Id)
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    {statementId : ProductConcreteNifsFor.StatementId}
    {config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape}
    {artifact : ProductConcreteNifsFor.RelationArtifact rowVariables
      logicalWidth publicFits}
    {prior : ProductionPaperPriorStateAuthorityRowsFor.Layout candidate
      rowVariables}
    {assignment : Nat -> Nat} {headers : ChainHeaders Digest.Value}
    {priorPrefix : ProductionPaperPriorStateAuthorityRowsFor.Prefix candidate
      (FullShape rowVariables logicalWidth publicFits)}
    {value : ProductionFieldNativeFullClaim.Value candidate
      (FullShape rowVariables logicalWidth publicFits)}
    {proof : ProductionProductPiCcsTypedBridgeFor.ExactProof rowVariables}
    (recursive : ProductionPaperRecursiveRelationRowsSoundFor.Result candidate
      statementId config artifact prior assignment headers priorPrefix value
      proof)
    (statement : ProductionStatement Program)
    {publicImage : PublicImage}
    (statementCanonical :
      publicImage.DecodesFor (identity candidate) statement)
    (publicLayout : StatementLayout)
    (statementPlaced : StatementPlaced publicLayout assignment statement)
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (holds : Satisfies (rows candidate prior publicLayout) assignment) :
    ProductionPaperTerminalInvocationRowsSoundFor.PublicChecks candidate
      statement recursive.priorState
      (ProductionPaperTerminalInvocationRowsSoundFor.finalClosed
        recursive.memoryResult) := by
  let allChecks := checks candidate prior publicLayout
  have allCanonical := checks_canonical candidate prior publicLayout
  have priorPlaced := recursive.priorAuthorityResult.priorPlaced
  have finalParsed := recursive.memoryResult.boundaryParsed
    (Fin.last (ProductionMemoryCheckedBatchRows.StepCount candidate))
  have invocationLinked := link_value canonical one allCanonical holds
    (output := prior.state.invocationColumn)
    (input := publicLayout.segmentCountColumn)
    (coefficient := claimsPerSegment candidate)
    (invocationCheck_mem candidate prior publicLayout)
  have realRowsLinked := link_equal canonical one allCanonical holds
    (output := prior.state.realRowsColumn)
    (input := publicLayout.realApplicationRowsColumn)
    (scalarCheck_mem prior publicLayout (by simp [scalarChecks]))
  have segmentLinked := link_equal canonical one allCanonical holds
    (output := (finalCarry candidate rowVariables prior).carry.fieldColumn
      .segmentIndex)
    (input := publicLayout.segmentCountColumn)
    (scalarCheck_mem prior publicLayout (by simp [scalarChecks]))
  have timestampLinked := link_equal canonical one allCanonical holds
    (output :=
      (finalCarry candidate rowVariables prior).carry.fieldColumn
        .globalTimestamp)
    (input := publicLayout.finalTimestampColumn)
    (scalarCheck_mem prior publicLayout (by simp [scalarChecks]))
  have initialFields := initial_application_fields_equal priorPlaced
    statementPlaced canonical one holds
  have finalFields := final_application_fields_equal priorPlaced
    statementPlaced canonical one holds
  refine
    { invocationIndex := ?_
      initialApplication := ?_
      finalApplication := ?_
      realApplicationRows := ?_
      finalSegment := ?_
      finalTimestamp := ?_
      finalMemoryRoot := ?_ }
  · have segmentBound :=
      statementCanonical.segmentCountBound
    have productBound := segment_product_below_field
      (candidate := candidate) segmentBound
    rw [priorPlaced.invocation, statementPlaced.segmentCount] at invocationLinked
    rw [Nat.mod_eq_of_lt productBound] at invocationLinked
    simpa [Nat.mul_comm] using invocationLinked
  · exact ProductionWasmStateFields.encode_injective initialFields
  · exact ProductionWasmStateFields.encode_injective finalFields
  · calc
      recursive.priorState.realApplicationRowCount =
          assignment prior.state.realRowsColumn := priorPlaced.realRows.symm
      _ = assignment publicLayout.realApplicationRowsColumn := realRowsLinked
      _ = statement.resultImage.realApplicationRowCount :=
        statementPlaced.realApplicationRows
  · have placed := finalParsed.placed MemoryCarryCodec.FieldTag.segmentIndex
    calc
      (ProductionPaperTerminalInvocationRowsSoundFor.finalClosed
          recursive.memoryResult).segmentIndex =
          assignment
            ((finalCarry candidate rowVariables prior).carry.fieldColumn
              .segmentIndex) := by
        simpa [ProductionPaperTerminalInvocationRowsSoundFor.finalClosed,
          ProductionPaperTerminalInvocationRowsSoundFor.finalWire,
          MemoryOpenSegmentSound.closedOfWire,
          finalCarry] using placed.symm
      _ = assignment publicLayout.segmentCountColumn := segmentLinked
      _ = statement.base.segmentCount := statementPlaced.segmentCount
  · have placed := finalParsed.placed MemoryCarryCodec.FieldTag.globalTimestamp
    calc
      (ProductionPaperTerminalInvocationRowsSoundFor.finalClosed
          recursive.memoryResult).globalTimestamp =
          assignment
            ((finalCarry candidate rowVariables prior).carry.fieldColumn
              .globalTimestamp) := by
        simpa [ProductionPaperTerminalInvocationRowsSoundFor.finalClosed,
          ProductionPaperTerminalInvocationRowsSoundFor.finalWire,
          MemoryOpenSegmentSound.closedOfWire,
          finalCarry] using placed.symm
      _ = assignment publicLayout.finalTimestampColumn := timestampLinked
      _ = statement.base.finalGlobalTimestamp := statementPlaced.finalTimestamp
  · apply Digest.Value.ext
    funext lane
    apply Subtype.ext
    have linked := link_equal canonical one allCanonical holds
      (output :=
        (finalCarry candidate rowVariables prior).carry.fieldColumn
          (.root .memory lane))
      (input := publicLayout.finalMemoryRootColumn lane)
      (finalMemoryRootCheck_mem candidate prior publicLayout lane)
    have placed := finalParsed.placed
      (MemoryCarryCodec.FieldTag.root .memory lane)
    calc
      ((ProductionPaperTerminalInvocationRowsSoundFor.finalClosed
          recursive.memoryResult).memoryRoot.lanes lane).val =
          assignment
            ((finalCarry candidate rowVariables prior).carry.fieldColumn
              (.root .memory lane)) := by
        simpa [ProductionPaperTerminalInvocationRowsSoundFor.finalClosed,
          ProductionPaperTerminalInvocationRowsSoundFor.finalWire,
          MemoryOpenSegmentSound.closedOfWire, finalCarry,
          MemoryCarryCodec.Value.fieldValue,
          MemoryCarryCodec.rootSourceValue] using placed.symm
      _ = assignment (publicLayout.finalMemoryRootColumn lane) := linked
      _ = (statement.resultImage.finalMemoryRoot.lanes lane).val :=
        statementPlaced.finalMemoryRoot lane

end Nightstream.Implementation.NebulaV2.ProductionPaperTerminalPublicRowsFor
