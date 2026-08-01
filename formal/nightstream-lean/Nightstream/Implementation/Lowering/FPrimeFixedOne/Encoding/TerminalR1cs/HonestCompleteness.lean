import Mathlib.Data.List.FinRange
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.TerminalR1cs.Honest

/-!
Contract: honest completeness of the selected SuperNeo terminal R1CS.

Assurance tier: model-level.

Owns: completion of only the fresh native-CCS residual columns, preservation
of authoritative terminal inputs, and satisfaction of the proof-free terminal
manifest from exact running CE and fresh CCS witnesses.

Does not own: selection of a benchmark statement, terminal semantic
soundness, Spartan, WHIR, Rust, or Ajtai binding security.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.TerminalR1cs.HonestCompleteness

open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsParameters
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.Typed
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc

private abbrev RelationShape
    (program : NativeCcsProgram.Program)
    (domain : NativeCcsCompiler.RowDomain program)
    (publicRingColumns : Nat)
    (publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length) :=
  NativeCcsPhi81.shape program domain publicRingColumns publicFits

private theorem getD_ofFn
    {Item : Type}
    {count : Nat}
    (items : Fin count → Item)
    (index : Fin count)
    (default : Item) :
    (List.ofFn items).getD index.val default = items index := by
  rw [List.getD_eq_getElem?_getD,
    List.getElem?_eq_getElem (by simp), List.getElem_ofFn]
  simp

/-- Exact ordered residual identities of the fresh native-CCS lowering. -/
def residualColumns
    {program : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain program}
    {publicRingColumns : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length}
    {verifierRows : Nat}
    (key :
      Commitment.Key
        (RelationShape program domain publicRingColumns publicFits)
        verifierRows) : List ColumnId :=
  List.ofFn (Layout.freshFrame key).residual

/-- Each residual is computed from the authoritative base assignment. -/
def residualValues
    {program : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain program}
    {publicRingColumns : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length}
    {verifierRows : Nat}
    {valid : NativeCcsCompiler.Valid program}
    {key :
      Commitment.Key
        (RelationShape program domain publicRingColumns publicFits)
        verifierRows}
    {statements :
      Terminal.RunningStatements program domain publicRingColumns publicFits
        verifierRows}
    {freshPayload :
      FixedActive.Canonical.FreshPayload
        (Fresh.semanticShape program domain)
        publicRingColumns publicFits verifierRows}
    (input : Honest.Input valid key statements freshPayload) : List F :=
  List.ofFn fun source =>
    ActivatedRawProgram.residualValue
      (FreshCcs.mappedRow valid (Fresh.ccsFrame (Layout.freshFrame key))
        (program.rows.get source)).source.row
      (Honest.baseAssignment input)

/-- Honest terminal assignment. Only native-CCS residuals are overwritten. -/
def assignment
    {program : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain program}
    {publicRingColumns : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length}
    {verifierRows : Nat}
    {valid : NativeCcsCompiler.Valid program}
    {key :
      Commitment.Key
        (RelationShape program domain publicRingColumns publicFits)
        verifierRows}
    {statements :
      Terminal.RunningStatements program domain publicRingColumns publicFits
        verifierRows}
    {freshPayload :
      FixedActive.Canonical.FreshPayload
        (Fresh.semanticShape program domain)
        publicRingColumns publicFits verifierRows}
    (input : Honest.Input valid key statements freshPayload) :
    ColumnId → F :=
  writeColumns (Honest.baseAssignment input)
    (residualColumns key) (residualValues input)

@[simp] theorem residualColumns_length
    {program : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain program}
    {publicRingColumns : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length}
    {verifierRows : Nat}
    (key :
      Commitment.Key
        (RelationShape program domain publicRingColumns publicFits)
        verifierRows) :
    (residualColumns key).length = program.rows.length := by
  simp [residualColumns]

@[simp] theorem residualValues_length
    {program : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain program}
    {publicRingColumns : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length}
    {verifierRows : Nat}
    {valid : NativeCcsCompiler.Valid program}
    {key :
      Commitment.Key
        (RelationShape program domain publicRingColumns publicFits)
        verifierRows}
    {statements :
      Terminal.RunningStatements program domain publicRingColumns publicFits
        verifierRows}
    {freshPayload :
      FixedActive.Canonical.FreshPayload
        (Fresh.semanticShape program domain)
        publicRingColumns publicFits verifierRows}
    (input : Honest.Input valid key statements freshPayload) :
    (residualValues input).length = program.rows.length := by
  simp [residualValues]

theorem residualColumns_nodup
    {program : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain program}
    {publicRingColumns : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length}
    {verifierRows : Nat}
    (key :
      Commitment.Key
        (RelationShape program domain publicRingColumns publicFits)
        verifierRows) :
    (residualColumns key).Nodup := by
  rw [residualColumns, List.nodup_ofFn]
  intro first second equal
  apply Fin.ext
  have coordinateEqual :=
    congrArg (fun column : ColumnId => column.coordinateIndex) equal
  simp only [Layout.freshFrame, Layout.localColumn] at coordinateEqual
  omega

theorem assignment_residual
    {program : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain program}
    {publicRingColumns : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length}
    {verifierRows : Nat}
    {valid : NativeCcsCompiler.Valid program}
    {key :
      Commitment.Key
        (RelationShape program domain publicRingColumns publicFits)
        verifierRows}
    {statements :
      Terminal.RunningStatements program domain publicRingColumns publicFits
        verifierRows}
    {freshPayload :
      FixedActive.Canonical.FreshPayload
        (Fresh.semanticShape program domain)
        publicRingColumns publicFits verifierRows}
    (input : Honest.Input valid key statements freshPayload)
    (source : Fin program.rows.length) :
    assignment input ((Layout.freshFrame key).residual source) =
      ActivatedRawProgram.residualValue
        (FreshCcs.mappedRow valid (Fresh.ccsFrame (Layout.freshFrame key))
          (program.rows.get source)).source.row
        (Honest.baseAssignment input) := by
  have recovered :=
    writeColumns_map_eq (Honest.baseAssignment input)
      (residualColumns key) (residualValues input)
      (by simp) (residualColumns_nodup key)
  have atSource :=
    congrArg (fun values => values.getD source.val 0) recovered
  simpa [assignment, residualColumns, residualValues, List.map_ofFn,
    getD_ofFn] using atSource

theorem runningOwner_ne_freshOwner
    (child : Fin productionGlobalParams.k) :
    Layout.runningOwner child ≠ Layout.freshOwner := by
  intro equal
  have pathEqual :
      Layout.claimPath child.val =
        Layout.claimPath productionGlobalParams.k := by
    simpa [Layout.runningOwner, Layout.freshOwner] using equal
  have indexEqual := congrArg Honest.claimIndex? pathEqual
  simp only [Honest.claimIndex_claimPath] at indexEqual
  exact (Nat.ne_of_lt child.isLt) (Option.some.inj indexEqual)

@[simp] theorem assignment_runningLocal
    {program : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain program}
    {publicRingColumns : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length}
    {verifierRows : Nat}
    {valid : NativeCcsCompiler.Valid program}
    {key :
      Commitment.Key
        (RelationShape program domain publicRingColumns publicFits)
        verifierRows}
    {statements :
      Terminal.RunningStatements program domain publicRingColumns publicFits
        verifierRows}
    {freshPayload :
      FixedActive.Canonical.FreshPayload
        (Fresh.semanticShape program domain)
        publicRingColumns publicFits verifierRows}
    (input : Honest.Input valid key statements freshPayload)
    (child : Fin productionGlobalParams.k)
    (coordinate : Nat) :
    assignment input
        (Layout.localColumn (Layout.runningOwner child) coordinate) =
      Honest.baseAssignment input
        (Layout.localColumn (Layout.runningOwner child) coordinate) := by
  apply writeColumns_of_not_mem
  intro member
  rcases List.mem_ofFn.mp member with ⟨source, equal⟩
  have ownerEqual :=
    congrArg (fun column : ColumnId => column.owner) equal
  exact runningOwner_ne_freshOwner child (by
    simpa [residualColumns, Layout.freshFrame, Layout.localColumn]
      using ownerEqual.symm)

@[simp] theorem assignment_freshBeforeResidual
    {program : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain program}
    {publicRingColumns : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length}
    {verifierRows : Nat}
    {valid : NativeCcsCompiler.Valid program}
    {key :
      Commitment.Key
        (RelationShape program domain publicRingColumns publicFits)
        verifierRows}
    {statements :
      Terminal.RunningStatements program domain publicRingColumns publicFits
        verifierRows}
    {freshPayload :
      FixedActive.Canonical.FreshPayload
        (Fresh.semanticShape program domain)
        publicRingColumns publicFits verifierRows}
    (input : Honest.Input valid key statements freshPayload)
    (coordinate : Nat)
    (before :
      coordinate <
        Layout.freshInputWidth
            (RelationShape program domain publicRingColumns publicFits)
            verifierRows +
          (RelationShape program domain publicRingColumns publicFits).carrierWidth) :
    assignment input (Layout.localColumn Layout.freshOwner coordinate) =
      Honest.baseAssignment input
        (Layout.localColumn Layout.freshOwner coordinate) := by
  apply writeColumns_of_not_mem
  intro member
  rcases List.mem_ofFn.mp member with ⟨source, equal⟩
  have coordinateEqual :=
    congrArg (fun column : ColumnId => column.coordinateIndex) equal
  simp only [residualColumns, Layout.freshFrame, Layout.localColumn]
    at coordinateEqual
  change
    Layout.freshInputWidth
          (RelationShape program domain publicRingColumns publicFits)
          verifierRows +
        (RelationShape program domain publicRingColumns publicFits).carrierWidth +
      source.val = coordinate at coordinateEqual
  omega

@[simp] theorem assignment_one
    {program : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain program}
    {publicRingColumns : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length}
    {verifierRows : Nat}
    {valid : NativeCcsCompiler.Valid program}
    {key :
      Commitment.Key
        (RelationShape program domain publicRingColumns publicFits)
        verifierRows}
    {statements :
      Terminal.RunningStatements program domain publicRingColumns publicFits
        verifierRows}
    {freshPayload :
      FixedActive.Canonical.FreshPayload
        (Fresh.semanticShape program domain)
        publicRingColumns publicFits verifierRows}
    (input : Honest.Input valid key statements freshPayload) :
    assignment input oneColumn = 1 := by
  rw [assignment, writeColumns_of_not_mem]
  · exact Honest.baseAssignment_one input
  · intro member
    rcases List.mem_ofFn.mp member with ⟨source, equal⟩
    have ownerEqual :=
      congrArg (fun column : ColumnId => column.owner) equal
    exact PhysicalOwner.noConfusion ownerEqual

@[simp] theorem assignment_freshWitness
    {program : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain program}
    {publicRingColumns : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length}
    {verifierRows : Nat}
    {valid : NativeCcsCompiler.Valid program}
    {key :
      Commitment.Key
        (RelationShape program domain publicRingColumns publicFits)
        verifierRows}
    {statements :
      Terminal.RunningStatements program domain publicRingColumns publicFits
        verifierRows}
    {freshPayload :
      FixedActive.Canonical.FreshPayload
        (Fresh.semanticShape program domain)
        publicRingColumns publicFits verifierRows}
    (input : Honest.Input valid key statements freshPayload)
    (coordinate :
      Fin (RelationShape program domain publicRingColumns publicFits).carrierWidth) :
    assignment input ((Layout.freshFrame key).witness coordinate) =
      input.freshWitness coordinate := by
  rw [show (Layout.freshFrame key).witness coordinate =
      Layout.localColumn Layout.freshOwner coordinate.val by rfl]
  rw [assignment_freshBeforeResidual input coordinate.val (by
    simp [Layout.freshInputWidth]
    omega)]
  exact Honest.baseAssignment_freshWitness input coordinate

private theorem combination_eval_congr
    (left right : ColumnId → F)
    {combination : LinearCombination}
    (equal :
      ∀ term ∈ combination, left term.column = right term.column) :
    combination.eval left = combination.eval right := by
  induction combination with
  | nil =>
      rfl
  | cons term tail inductionHypothesis =>
      simp only [LinearCombination.eval]
      rw [equal term List.mem_cons_self,
        inductionHypothesis (fun item member =>
          equal item (List.mem_cons_of_mem term member))]

private theorem mappedRow_eval_eq
    {program : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain program}
    {publicRingColumns : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length}
    {verifierRows : Nat}
    {valid : NativeCcsCompiler.Valid program}
    {key :
      Commitment.Key
        (RelationShape program domain publicRingColumns publicFits)
        verifierRows}
    {statements :
      Terminal.RunningStatements program domain publicRingColumns publicFits
        verifierRows}
    {freshPayload :
      FixedActive.Canonical.FreshPayload
        (Fresh.semanticShape program domain)
        publicRingColumns publicFits verifierRows}
    (input : Honest.Input valid key statements freshPayload)
    (source : Fin program.rows.length)
    (combination : LinearCombination)
    (part :
      (FreshCcs.mappedRow valid
        (Fresh.ccsFrame (Layout.freshFrame key))
        (program.rows.get source)).source.row.a =
          combination ∨
        (FreshCcs.mappedRow valid
          (Fresh.ccsFrame (Layout.freshFrame key))
          (program.rows.get source)).source.row.b =
            combination ∨
        (FreshCcs.mappedRow valid
          (Fresh.ccsFrame (Layout.freshFrame key))
          (program.rows.get source)).source.row.c =
            combination) :
    combination.eval (assignment input) =
      combination.eval (Honest.baseAssignment input) := by
  apply combination_eval_congr
  intro term member
  have columnMember :
      term.column ∈
        (FreshCcs.mappedRow valid
          (Fresh.ccsFrame (Layout.freshFrame key))
          (program.rows.get source)).columnIds := by
    rw [NativeCcsSelector.SelectedRow.columnIds]
    apply List.mem_cons_of_mem
    rw [OwnedRow.columnIds, Row.columnIds]
    apply List.mem_map.mpr
    refine ⟨term, ?_, rfl⟩
    rcases part with a | b | c
    · rw [a]
      exact List.mem_append_left _
        (List.mem_append_left _ member)
    · rw [b]
      exact List.mem_append_left _
        (List.mem_append_right _ member)
    · rw [c]
      exact List.mem_append_right _ member
  rcases
      FreshCcs.mappedRow_supported valid
        (Fresh.ccsFrame (Layout.freshFrame key))
        (program.rows.get source) term.column columnMember with
    ⟨coordinate, equal⟩
  rw [equal]
  change
    assignment input ((Layout.freshFrame key).witness coordinate) =
      Honest.baseAssignment input
        ((Layout.freshFrame key).witness coordinate)
  rw [assignment_freshWitness input coordinate,
    Honest.baseAssignment_freshWitness input coordinate]

theorem mappedRow_residualValue_eq
    {program : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain program}
    {publicRingColumns : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length}
    {verifierRows : Nat}
    {valid : NativeCcsCompiler.Valid program}
    {key :
      Commitment.Key
        (RelationShape program domain publicRingColumns publicFits)
        verifierRows}
    {statements :
      Terminal.RunningStatements program domain publicRingColumns publicFits
        verifierRows}
    {freshPayload :
      FixedActive.Canonical.FreshPayload
        (Fresh.semanticShape program domain)
        publicRingColumns publicFits verifierRows}
    (input : Honest.Input valid key statements freshPayload)
    (source : Fin program.rows.length) :
    ActivatedRawProgram.residualValue
        (FreshCcs.mappedRow valid (Fresh.ccsFrame (Layout.freshFrame key))
          (program.rows.get source)).source.row
        (assignment input) =
      ActivatedRawProgram.residualValue
        (FreshCcs.mappedRow valid (Fresh.ccsFrame (Layout.freshFrame key))
          (program.rows.get source)).source.row
        (Honest.baseAssignment input) := by
  unfold ActivatedRawProgram.residualValue
  rw [mappedRow_eval_eq input source
    (combination :=
      (FreshCcs.mappedRow valid
        (Fresh.ccsFrame (Layout.freshFrame key))
        (program.rows.get source)).source.row.a)
    (by exact Or.inl rfl)]
  rw [mappedRow_eval_eq input source
    (combination :=
      (FreshCcs.mappedRow valid
        (Fresh.ccsFrame (Layout.freshFrame key))
        (program.rows.get source)).source.row.b)
    (by exact Or.inr (Or.inl rfl))]
  rw [mappedRow_eval_eq input source
    (combination :=
      (FreshCcs.mappedRow valid
        (Fresh.ccsFrame (Layout.freshFrame key))
        (program.rows.get source)).source.row.c)
    (by exact Or.inr (Or.inr rfl))]

theorem runningDistinct
    {program : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain program}
    {publicRingColumns : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length}
    {verifierRows : Nat}
    (key :
      Commitment.Key
        (RelationShape program domain publicRingColumns publicFits)
        verifierRows)
    (child : Fin productionGlobalParams.k) :
    Norm.Distinct (Running.normFrame (Layout.runningFrame key child)) := by
  constructor
  · intro first second equal
    apply Fin.ext
    have coordinateEqual :=
      congrArg (fun column : ColumnId => column.coordinateIndex) equal
    simp only [Running.normFrame, Layout.runningFrame,
      Layout.localColumn, Layout.runningInputWidth,
      Layout.runningStatementWidth] at coordinateEqual
    omega
  · intro witnessCoordinate squareCoordinate equal
    have coordinateEqual :=
      congrArg (fun column : ColumnId => column.coordinateIndex) equal
    simp only [Running.normFrame, Layout.runningFrame,
      Layout.localColumn, Layout.runningInputWidth,
      Layout.runningStatementWidth] at coordinateEqual
    have witnessLt := witnessCoordinate.isLt
    omega

theorem freshDistinct
    {program : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain program}
    {publicRingColumns : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length}
    {verifierRows : Nat}
    (key :
      Commitment.Key
        (RelationShape program domain publicRingColumns publicFits)
        verifierRows) :
    Norm.Distinct (Fresh.normFrame (Layout.freshFrame key)) := by
  constructor
  · intro first second equal
    apply Fin.ext
    have coordinateEqual :=
      congrArg (fun column : ColumnId => column.coordinateIndex) equal
    simp only [Fresh.normFrame, Layout.freshFrame,
      Layout.localColumn, Layout.freshInputWidth,
      Layout.freshStatementWidth] at coordinateEqual
    omega
  · intro witnessCoordinate squareCoordinate equal
    have coordinateEqual :=
      congrArg (fun column : ColumnId => column.coordinateIndex) equal
    simp only [Fresh.normFrame, Layout.freshFrame,
      Layout.localColumn, Layout.freshInputWidth,
      Layout.freshStatementWidth] at coordinateEqual
    change witnessCoordinate.val =
      (Phi81CarrierLayout.carrierWidth program.columnIds.length +
        (verifierRows * ringDegree +
          (RelationShape program domain publicRingColumns publicFits).publicWidth)) +
        squareCoordinate.val at coordinateEqual
    have witnessLt :
        witnessCoordinate.val <
          Phi81CarrierLayout.carrierWidth program.columnIds.length :=
      witnessCoordinate.isLt
    omega

@[simp] theorem assignment_runningWitness
    {program : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain program}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length}
    {valid : NativeCcsCompiler.Valid program}
    {key : Commitment.Key
      (RelationShape program domain publicRingColumns publicFits) verifierRows}
    {statements : Terminal.RunningStatements program domain
      publicRingColumns publicFits verifierRows}
    {freshPayload : FixedActive.Canonical.FreshPayload
      (Fresh.semanticShape program domain) publicRingColumns publicFits
      verifierRows}
    (input : Honest.Input valid key statements freshPayload)
    (child : Fin productionGlobalParams.k)
    (coordinate :
      Fin (RelationShape program domain publicRingColumns publicFits).carrierWidth) :
    assignment input ((Layout.runningFrame key child).witness coordinate) =
      input.runningWitness child coordinate := by
  rw [show (Layout.runningFrame key child).witness coordinate =
      Layout.localColumn (Layout.runningOwner child) coordinate.val by rfl]
  rw [assignment_runningLocal input child coordinate.val]
  exact Honest.baseAssignment_runningWitness input child coordinate

@[simp] theorem assignment_runningCommitment
    {program : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain program}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length}
    {valid : NativeCcsCompiler.Valid program}
    {key : Commitment.Key
      (RelationShape program domain publicRingColumns publicFits) verifierRows}
    {statements : Terminal.RunningStatements program domain
      publicRingColumns publicFits verifierRows}
    {freshPayload : FixedActive.Canonical.FreshPayload
      (Fresh.semanticShape program domain) publicRingColumns publicFits
      verifierRows}
    (input : Honest.Input valid key statements freshPayload)
    (child : Fin productionGlobalParams.k)
    (verifierRow : Fin verifierRows)
    (output : Fin ringDegree) :
    assignment input
        ((Layout.runningFrame key child).commitment verifierRow output) =
      Commitment.commit key (input.runningWitness child)
        verifierRow output := by
  rw [show (Layout.runningFrame key child).commitment verifierRow output =
      Layout.localColumn (Layout.runningOwner child)
        ((RelationShape program domain publicRingColumns publicFits).carrierWidth +
          verifierRow.val * ringDegree + output.val) by rfl]
  rw [assignment_runningLocal]
  exact Honest.baseAssignment_runningCommitment input child verifierRow output

@[simp] theorem assignment_runningPublic
    {program : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain program}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length}
    {valid : NativeCcsCompiler.Valid program}
    {key : Commitment.Key
      (RelationShape program domain publicRingColumns publicFits) verifierRows}
    {statements : Terminal.RunningStatements program domain
      publicRingColumns publicFits verifierRows}
    {freshPayload : FixedActive.Canonical.FreshPayload
      (Fresh.semanticShape program domain) publicRingColumns publicFits
      verifierRows}
    (input : Honest.Input valid key statements freshPayload)
    (child : Fin productionGlobalParams.k)
    (coordinate :
      Fin (RelationShape program domain publicRingColumns publicFits).publicWidth) :
    assignment input
        ((Layout.runningFrame key child).publicColumn coordinate) =
      Phi81Relation.projectPublicInput
        (input.runningWitness child) coordinate := by
  rw [show (Layout.runningFrame key child).publicColumn coordinate =
      Layout.localColumn (Layout.runningOwner child)
        ((RelationShape program domain publicRingColumns publicFits).carrierWidth +
          verifierRows * ringDegree + coordinate.val) by rfl]
  rw [assignment_runningLocal]
  exact Honest.baseAssignment_runningPublic input child coordinate

@[simp] theorem assignment_runningEvaluationLow
    {program : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain program}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length}
    {valid : NativeCcsCompiler.Valid program}
    {key : Commitment.Key
      (RelationShape program domain publicRingColumns publicFits) verifierRows}
    {statements : Terminal.RunningStatements program domain
      publicRingColumns publicFits verifierRows}
    {freshPayload : FixedActive.Canonical.FreshPayload
      (Fresh.semanticShape program domain) publicRingColumns publicFits
      verifierRows}
    (input : Honest.Input valid key statements freshPayload)
    (child : Fin productionGlobalParams.k)
    (matrix :
      Fin (RelationShape program domain publicRingColumns publicFits).matrixCount)
    (lane : Fin ringDegree) :
    assignment input
        ((Layout.runningFrame key child).evaluationLow matrix lane) =
      (Phi81Relation.matrixEvaluation
        (statements child).constraintSystem
        (input.runningWitness child) (statements child).point matrix lane).c0 := by
  rw [show (Layout.runningFrame key child).evaluationLow matrix lane =
      Layout.localColumn (Layout.runningOwner child)
        ((RelationShape program domain publicRingColumns publicFits).carrierWidth +
          verifierRows * ringDegree +
          (RelationShape program domain publicRingColumns publicFits).publicWidth +
          matrix.val * ringDegree + lane.val) by rfl]
  rw [assignment_runningLocal]
  exact Honest.baseAssignment_runningEvaluationLow input child matrix lane

@[simp] theorem assignment_runningEvaluationHigh
    {program : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain program}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length}
    {valid : NativeCcsCompiler.Valid program}
    {key : Commitment.Key
      (RelationShape program domain publicRingColumns publicFits) verifierRows}
    {statements : Terminal.RunningStatements program domain
      publicRingColumns publicFits verifierRows}
    {freshPayload : FixedActive.Canonical.FreshPayload
      (Fresh.semanticShape program domain) publicRingColumns publicFits
      verifierRows}
    (input : Honest.Input valid key statements freshPayload)
    (child : Fin productionGlobalParams.k)
    (matrix :
      Fin (RelationShape program domain publicRingColumns publicFits).matrixCount)
    (lane : Fin ringDegree) :
    assignment input
        ((Layout.runningFrame key child).evaluationHigh matrix lane) =
      (Phi81Relation.matrixEvaluation
        (statements child).constraintSystem
        (input.runningWitness child) (statements child).point matrix lane).c1 := by
  rw [show (Layout.runningFrame key child).evaluationHigh matrix lane =
      Layout.localColumn (Layout.runningOwner child)
        ((RelationShape program domain publicRingColumns publicFits).carrierWidth +
          verifierRows * ringDegree +
          (RelationShape program domain publicRingColumns publicFits).publicWidth +
          (RelationShape program domain publicRingColumns publicFits).matrixCount *
            ringDegree + matrix.val * ringDegree + lane.val) by rfl]
  rw [assignment_runningLocal]
  exact Honest.baseAssignment_runningEvaluationHigh input child matrix lane

@[simp] theorem assignment_runningSquare
    {program : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain program}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length}
    {valid : NativeCcsCompiler.Valid program}
    {key : Commitment.Key
      (RelationShape program domain publicRingColumns publicFits) verifierRows}
    {statements : Terminal.RunningStatements program domain
      publicRingColumns publicFits verifierRows}
    {freshPayload : FixedActive.Canonical.FreshPayload
      (Fresh.semanticShape program domain) publicRingColumns publicFits
      verifierRows}
    (input : Honest.Input valid key statements freshPayload)
    (child : Fin productionGlobalParams.k)
    (coordinate :
      Fin (RelationShape program domain publicRingColumns publicFits).carrierWidth) :
    assignment input ((Layout.runningFrame key child).square coordinate) =
      input.runningWitness child coordinate *
        input.runningWitness child coordinate := by
  rw [show (Layout.runningFrame key child).square coordinate =
      Layout.localColumn (Layout.runningOwner child)
        (Layout.runningInputWidth
          (RelationShape program domain publicRingColumns publicFits)
          verifierRows + coordinate.val) by rfl]
  rw [assignment_runningLocal]
  exact Honest.baseAssignment_runningSquare input child coordinate

@[simp] theorem assignment_freshCommitment
    {program : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain program}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length}
    {valid : NativeCcsCompiler.Valid program}
    {key : Commitment.Key
      (RelationShape program domain publicRingColumns publicFits) verifierRows}
    {statements : Terminal.RunningStatements program domain
      publicRingColumns publicFits verifierRows}
    {freshPayload : FixedActive.Canonical.FreshPayload
      (Fresh.semanticShape program domain) publicRingColumns publicFits
      verifierRows}
    (input : Honest.Input valid key statements freshPayload)
    (verifierRow : Fin verifierRows)
    (output : Fin ringDegree) :
    assignment input ((Layout.freshFrame key).commitment verifierRow output) =
      Commitment.commit key input.freshWitness verifierRow output := by
  rw [show (Layout.freshFrame key).commitment verifierRow output =
      Layout.localColumn Layout.freshOwner
        ((RelationShape program domain publicRingColumns publicFits).carrierWidth +
          verifierRow.val * ringDegree + output.val) by rfl]
  rw [assignment_freshBeforeResidual input _ (by
    have pairLt :
        verifierRow.val * ringDegree + output.val <
          verifierRows * ringDegree := by
      simpa only [Ajtai.pairIndex] using
        (Ajtai.pairIndex verifierRow output).isLt
    simp only [Layout.freshInputWidth, Layout.freshStatementWidth]
    omega)]
  exact Honest.baseAssignment_freshCommitment input verifierRow output

@[simp] theorem assignment_freshPublic
    {program : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain program}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length}
    {valid : NativeCcsCompiler.Valid program}
    {key : Commitment.Key
      (RelationShape program domain publicRingColumns publicFits) verifierRows}
    {statements : Terminal.RunningStatements program domain
      publicRingColumns publicFits verifierRows}
    {freshPayload : FixedActive.Canonical.FreshPayload
      (Fresh.semanticShape program domain) publicRingColumns publicFits
      verifierRows}
    (input : Honest.Input valid key statements freshPayload)
    (coordinate :
      Fin (RelationShape program domain publicRingColumns publicFits).publicWidth) :
    assignment input ((Layout.freshFrame key).publicColumn coordinate) =
      Phi81Relation.projectPublicInput input.freshWitness coordinate := by
  rw [show (Layout.freshFrame key).publicColumn coordinate =
      Layout.localColumn Layout.freshOwner
        ((RelationShape program domain publicRingColumns publicFits).carrierWidth +
          verifierRows * ringDegree + coordinate.val) by rfl]
  rw [assignment_freshBeforeResidual input _ (by
    simp [Layout.freshInputWidth, Layout.freshStatementWidth]
    omega)]
  exact Honest.baseAssignment_freshPublic input coordinate

@[simp] theorem assignment_freshSquare
    {program : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain program}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length}
    {valid : NativeCcsCompiler.Valid program}
    {key : Commitment.Key
      (RelationShape program domain publicRingColumns publicFits) verifierRows}
    {statements : Terminal.RunningStatements program domain
      publicRingColumns publicFits verifierRows}
    {freshPayload : FixedActive.Canonical.FreshPayload
      (Fresh.semanticShape program domain) publicRingColumns publicFits
      verifierRows}
    (input : Honest.Input valid key statements freshPayload)
    (coordinate :
      Fin (RelationShape program domain publicRingColumns publicFits).carrierWidth) :
    assignment input ((Layout.freshFrame key).square coordinate) =
      input.freshWitness coordinate * input.freshWitness coordinate := by
  rw [show (Layout.freshFrame key).square coordinate =
      Layout.localColumn Layout.freshOwner
        (Layout.freshInputWidth
          (RelationShape program domain publicRingColumns publicFits)
          verifierRows + coordinate.val) by rfl]
  rw [assignment_freshBeforeResidual input _ (by
    simp)]
  exact Honest.baseAssignment_freshSquare input coordinate

/-- The completed assignment stores each fresh native-CCS residual as the
residual of the completed assignment itself. -/
theorem assignment_freshResidual
    {program : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain program}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length}
    {valid : NativeCcsCompiler.Valid program}
    {key : Commitment.Key
      (RelationShape program domain publicRingColumns publicFits) verifierRows}
    {statements : Terminal.RunningStatements program domain
      publicRingColumns publicFits verifierRows}
    {freshPayload : FixedActive.Canonical.FreshPayload
      (Fresh.semanticShape program domain) publicRingColumns publicFits
      verifierRows}
    (input : Honest.Input valid key statements freshPayload)
    (source : Fin program.rows.length) :
    assignment input ((Layout.freshFrame key).residual source) =
      ActivatedRawProgram.residualValue
        (FreshCcs.mappedRow valid (Fresh.ccsFrame (Layout.freshFrame key))
          (program.rows.get source)).source.row
        (assignment input) := by
  exact (assignment_residual input source).trans
    (mappedRow_residualValue_eq input source).symm

theorem runningWitness_eq
    {program : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain program}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length}
    {valid : NativeCcsCompiler.Valid program}
    {key : Commitment.Key
      (RelationShape program domain publicRingColumns publicFits) verifierRows}
    {statements : Terminal.RunningStatements program domain
      publicRingColumns publicFits verifierRows}
    {freshPayload : FixedActive.Canonical.FreshPayload
      (Fresh.semanticShape program domain) publicRingColumns publicFits
      verifierRows}
    (input : Honest.Input valid key statements freshPayload)
    (child : Fin productionGlobalParams.k) :
    (fun coordinate =>
      assignment input ((Layout.runningFrame key child).witness coordinate)) =
        input.runningWitness child := by
  funext coordinate
  exact assignment_runningWitness input child coordinate

theorem freshWitness_eq
    {program : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain program}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length}
    {valid : NativeCcsCompiler.Valid program}
    {key : Commitment.Key
      (RelationShape program domain publicRingColumns publicFits) verifierRows}
    {statements : Terminal.RunningStatements program domain
      publicRingColumns publicFits verifierRows}
    {freshPayload : FixedActive.Canonical.FreshPayload
      (Fresh.semanticShape program domain) publicRingColumns publicFits
      verifierRows}
    (input : Honest.Input valid key statements freshPayload) :
    (fun coordinate =>
      assignment input ((Layout.freshFrame key).witness coordinate)) =
        input.freshWitness := by
  funext coordinate
  exact assignment_freshWitness input coordinate

/-- One exact running CE witness satisfies its statement-specialized terminal
rows in the shared physical assignment. -/
theorem runningRows_honest
    {program : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain program}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length}
    {valid : NativeCcsCompiler.Valid program}
    {key : Commitment.Key
      (RelationShape program domain publicRingColumns publicFits) verifierRows}
    {statements : Terminal.RunningStatements program domain
      publicRingColumns publicFits verifierRows}
    {freshPayload : FixedActive.Canonical.FreshPayload
      (Fresh.semanticShape program domain) publicRingColumns publicFits
      verifierRows}
    (input : Honest.Input valid key statements freshPayload)
    (child : Fin productionGlobalParams.k) :
    Satisfies
      (Running.rows (Layout.runningFrame key child) (statements child))
      (assignment input) := by
  have facts :=
    (Phi81Relation.ceMembership_iff_evaluationsBound
      (Commitment.commit key) productionGlobalParams
      (statements child) (input.runningWitness child)).mp
      (input.runningHolds child)
  apply Running.rows_honest
      (Layout.runningFrame key child) (statements child) (assignment input)
      (assignment_one input) (runningDistinct key child)
  · rw [runningWitness_eq input child]
    funext verifierRow output
    exact assignment_runningCommitment input child verifierRow output
  · rw [runningWitness_eq input child]
    funext coordinate
    exact assignment_runningPublic input child coordinate
  · rw [runningWitness_eq input child]
    simpa [input.runningFreshStage child] using facts.2.2.1
  · intro coordinate
    simp
  · intro matrix lane
    rw [assignment_runningEvaluationLow input child matrix lane,
      assignment_runningEvaluationHigh input child matrix lane,
      runningWitness_eq input child]

/-- The exact fresh CCS witness satisfies the native-CCS terminal lowering,
including the residual columns completed above. -/
theorem freshRows_honest
    {program : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain program}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length}
    {valid : NativeCcsCompiler.Valid program}
    {key : Commitment.Key
      (RelationShape program domain publicRingColumns publicFits) verifierRows}
    {statements : Terminal.RunningStatements program domain
      publicRingColumns publicFits verifierRows}
    {freshPayload : FixedActive.Canonical.FreshPayload
      (Fresh.semanticShape program domain) publicRingColumns publicFits
      verifierRows}
    (input : Honest.Input valid key statements freshPayload) :
    Satisfies (Fresh.rows valid (Layout.freshFrame key))
      (assignment input) := by
  have facts :=
    (Phi81Relation.ccsMembership_iff
      (Commitment.commit key) productionGlobalParams
      (freshPayload.materialize
        (NativeCcsPhi81.relation program valid domain
          publicRingColumns publicFits))
      input.freshWitness).mp input.freshHolds
  have sourceSatisfied :=
    (NativeCcsPhi81.ccsSatisfied_arbitrary_iff
      program valid domain publicRingColumns publicFits
      input.freshWitness).mp facts.2.2.2
  apply Fresh.rows_honest valid (Layout.freshFrame key) (assignment input)
      (assignment_one input) (freshDistinct key)
  · rw [freshWitness_eq input]
    funext verifierRow output
    exact assignment_freshCommitment input verifierRow output
  · rw [freshWitness_eq input]
    funext coordinate
    exact assignment_freshPublic input coordinate
  · rw [freshWitness_eq input]
    simpa [FixedActive.Canonical.FreshPayload.materialize,
      NormStage.bound, productionGlobalParams] using facts.2.2.1
  · intro coordinate
    simp
  · change NativeCcsSelector.Satisfies program.rows (fun column =>
      assignment input ((Layout.freshFrame key).witness
        (Phi81CarrierLayout.embedLogical
          (NativeCcsCompiler.ColumnIndex.index program valid column))))
    change NativeCcsSelector.Satisfies program.rows (fun column =>
      input.freshWitness
        (Phi81CarrierLayout.embedLogical
          (NativeCcsCompiler.ColumnIndex.index program valid column)))
      at sourceSatisfied
    simpa only [assignment_freshWitness] using sourceSatisfied
  · exact assignment_freshResidual input

/-- All fourteen running relations and the independent fresh relation share
one honest physical terminal assignment. -/
theorem terminalRows_honest
    {program : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain program}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length}
    {valid : NativeCcsCompiler.Valid program}
    {key : Commitment.Key
      (RelationShape program domain publicRingColumns publicFits) verifierRows}
    {statements : Terminal.RunningStatements program domain
      publicRingColumns publicFits verifierRows}
    {freshPayload : FixedActive.Canonical.FreshPayload
      (Fresh.semanticShape program domain) publicRingColumns publicFits
      verifierRows}
    (input : Honest.Input valid key statements freshPayload) :
    Satisfies
      (Terminal.rows valid (Layout.frame key) statements)
      (assignment input) := by
  apply Terminal.rows_honest valid (Layout.frame key) statements
      (assignment input)
  · exact fun child => runningRows_honest input child
  · exact freshRows_honest input

/-- Honest paper terminal witnesses satisfy the exact decoded proof-free
manifest that Rust must consume. -/
theorem decodedProgram_honest
    {program : NativeCcsProgram.Program}
    {domain : NativeCcsCompiler.RowDomain program}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth program.columnIds.length}
    {valid : NativeCcsCompiler.Valid program}
    {key : Commitment.Key
      (RelationShape program domain publicRingColumns publicFits) verifierRows}
    {statements : Terminal.RunningStatements program domain
      publicRingColumns publicFits verifierRows}
    {freshPayload : FixedActive.Canonical.FreshPayload
      (Fresh.semanticShape program domain) publicRingColumns publicFits
      verifierRows}
    (input : Honest.Input valid key statements freshPayload) :
    Satisfies (Layout.program valid key statements).decode.rows
      (assignment input) :=
  (Program.decoded_satisfies_iff valid key statements
    (assignment input)).mpr (terminalRows_honest input)

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.TerminalR1cs.HonestCompleteness
