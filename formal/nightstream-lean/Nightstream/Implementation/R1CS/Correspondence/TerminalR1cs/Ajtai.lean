import Nightstream.Implementation.R1CS.Correspondence.TerminalR1cs.Atoms
import Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra.Commitment

/-!
Contract: Lean-owned structural R1CS compiler for one typed Ajtai opening.

Assurance tier: model-level.

Owns: the exact verifier-key coefficient expansion, one linear row per
commitment coefficient, structural ownership, cost, soundness, and honest
completeness.

Does not own: key generation, key serialization, Ajtai binding or MSIS
security, witness/public column allocation, norm checks, matrix evaluation,
terminal assembly, or Rust.

Emits constraints: `verifierRows * 54` linear rows and no auxiliary columns.
The rows are symbolic; this module does not force closed materialization of a
large deployment key.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.TerminalR1cs.Ajtai

open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.Typed
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism
open Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.MatrixCoefficientSource

/-- Physical locations and verifier-owned key for one opening. -/
structure Frame (shape : Phi81Relation.Shape) (verifierRows : Nat) where
  owner : PhysicalOwner
  firstOrdinal : Nat
  one : ColumnId
  key : Commitment.Key shape verifierRows
  witness : Fin shape.carrierWidth → ColumnId
  commitment : Fin verifierRows → Fin ringDegree → ColumnId

/-- One base-field term of a verifier-fixed ring product. The inactive branch
is unreachable for indices produced by `List.range ringDegree`. -/
def laneTerm {shape : Phi81Relation.Shape} {verifierRows : Nat}
    (frame : Frame shape verifierRows)
    (verifierRow : Fin verifierRows)
    (output : Fin ringDegree)
    (block :
      Fin (Phi81ColumnLayout.blockCount shape.carrierWidth))
    (index : Nat) : Term :=
  if indexLt : index < ringDegree then
    { column :=
        frame.witness (CarrierAction.carrierColumn block ⟨index, indexLt⟩)
      coefficient :=
        CarrierAction.rightCoefficient
          (frame.key verifierRow block) output ⟨index, indexLt⟩ }
  else
    { column := frame.one, coefficient := 0 }

/-- Exact 54-lane linearization of one key-block product. -/
def blockTerms {shape : Phi81Relation.Shape} {verifierRows : Nat}
    (frame : Frame shape verifierRows)
    (verifierRow : Fin verifierRows)
    (output : Fin ringDegree)
    (block :
      Fin (Phi81ColumnLayout.blockCount shape.carrierWidth)) :
    LinearCombination :=
  (List.range ringDegree).map
    (laneTerm frame verifierRow output block)

/-- Exact ordered linear combination for one commitment coefficient. -/
def terms {shape : Phi81Relation.Shape} {verifierRows : Nat}
    (frame : Frame shape verifierRows)
    (verifierRow : Fin verifierRows)
    (output : Fin ringDegree) :
    LinearCombination :=
  (List.ofFn fun block :
      Fin (Phi81ColumnLayout.blockCount shape.carrierWidth) =>
        blockTerms frame verifierRow output block).flatten

private theorem eval_append
    (assignment : ColumnId → F)
    (left right : LinearCombination) :
    LinearCombination.eval assignment (left ++ right) =
      LinearCombination.eval assignment left +
        LinearCombination.eval assignment right := by
  induction left with
  | nil =>
      simp
  | cons head tail inductionHypothesis =>
      simp only [List.cons_append, LinearCombination.eval_cons,
        inductionHypothesis]
      exact (Lean.Grind.Fin.add_assoc _ _ _).symm

private theorem eval_range_map
    (assignment : ColumnId → F)
    (count : Nat)
    (term : Nat → Term) :
    LinearCombination.eval assignment
        ((List.range count).map term) =
      sumRange ConcreteCarrier.baseOps count fun index =>
        (term index).coefficient * assignment (term index).column := by
  induction count with
  | zero => rfl
  | succ count inductionHypothesis =>
      rw [List.range_succ, List.map_append, eval_append,
        inductionHypothesis]
      simp [LinearCombination.eval, sumRange, ConcreteCarrier.baseOps]

/-- One block's sparse linear combination is exactly the corresponding
coefficient of the fixed ring product. -/
theorem blockTerms_eval {shape : Phi81Relation.Shape} {verifierRows : Nat}
    (frame : Frame shape verifierRows)
    (verifierRow : Fin verifierRows)
    (output : Fin ringDegree)
    (block :
      Fin (Phi81ColumnLayout.blockCount shape.carrierWidth))
    (assignment : ColumnId → F) :
    LinearCombination.eval assignment
        (blockTerms frame verifierRow output block) =
      ringFMul (frame.key verifierRow block)
        (CarrierAction.assignmentBlock
          (fun coordinate => assignment (frame.witness coordinate))
          block) output := by
  rw [blockTerms, eval_range_map]
  rw [CarrierAction.ringFMul_apply_eq_rightLinear]
  apply sumRange_congr
  intro index indexLt
  simp only [laneTerm, dif_pos indexLt]
  rfl

private theorem eval_flatten_ofFn_eq_ringFSum :
    ∀ {count : Nat}
      (combinations : Fin count → LinearCombination)
      (values : Fin count → RingF)
      (assignment : ColumnId → F)
      (output : Fin ringDegree),
      (∀ index,
        LinearCombination.eval assignment (combinations index) =
          values index output) →
      LinearCombination.eval assignment
          (List.ofFn combinations).flatten =
        Commitment.ringFSum values output
  | 0, combinations, values, assignment, output, exactValue => by
      rfl
  | _ + 1, combinations, values, assignment, output, exactValue => by
      rw [List.ofFn_succ, List.flatten_cons, eval_append,
        Commitment.ringFSum, exactValue 0]
      rw [eval_flatten_ofFn_eq_ringFSum
        (fun index => combinations index.succ)
        (fun index => values index.succ)
        assignment output
        (fun index => exactValue index.succ)]
      rfl

/-- The emitted sparse combination computes the exact semantic Ajtai row. -/
theorem terms_eval_eq_commit {shape : Phi81Relation.Shape}
    {verifierRows : Nat}
    (frame : Frame shape verifierRows)
    (verifierRow : Fin verifierRows)
    (output : Fin ringDegree)
    (assignment : ColumnId → F) :
    LinearCombination.eval assignment
        (terms frame verifierRow output) =
      Commitment.commit frame.key
        (fun coordinate => assignment (frame.witness coordinate))
        verifierRow output := by
  unfold terms Commitment.commit Commitment.ajtaiRow Commitment.blockSum
  apply eval_flatten_ofFn_eq_ringFSum
  intro block
  exact blockTerms_eval frame verifierRow output block assignment

/-! ## Physical row family -/

private theorem verifierRowIndex_lt
    {verifierRows : Nat}
    (coordinate : Fin (verifierRows * ringDegree)) :
    coordinate.val / ringDegree < verifierRows := by
  have coordinateLt := coordinate.isLt
  simp only [ringDegree] at coordinateLt ⊢
  omega

/-- Decode the commitment row from a flat physical occurrence index. -/
def verifierRowAt {verifierRows : Nat}
    (coordinate : Fin (verifierRows * ringDegree)) :
    Fin verifierRows :=
  ⟨coordinate.val / ringDegree, verifierRowIndex_lt coordinate⟩

/-- Decode the output lane from a flat physical occurrence index. -/
def outputAt {verifierRows : Nat}
    (coordinate : Fin (verifierRows * ringDegree)) :
    Fin ringDegree :=
  ⟨coordinate.val % ringDegree,
    Nat.mod_lt _ (by decide : 0 < ringDegree)⟩

/-- Canonical flat occurrence index of one row/lane pair. -/
def pairIndex {verifierRows : Nat}
    (verifierRow : Fin verifierRows)
    (output : Fin ringDegree) :
    Fin (verifierRows * ringDegree) :=
  ⟨verifierRow.val * ringDegree + output.val, by
    have rowLt := verifierRow.isLt
    have outputLt := output.isLt
    simp only [ringDegree] at rowLt outputLt ⊢
    omega⟩

@[simp] theorem verifierRowAt_pairIndex {verifierRows : Nat}
    (verifierRow : Fin verifierRows)
    (output : Fin ringDegree) :
    verifierRowAt (pairIndex verifierRow output) = verifierRow := by
  apply Fin.ext
  have outputLt := output.isLt
  simp only [verifierRowAt, pairIndex, ringDegree] at outputLt ⊢
  omega

@[simp] theorem outputAt_pairIndex {verifierRows : Nat}
    (verifierRow : Fin verifierRows)
    (output : Fin ringDegree) :
    outputAt (pairIndex verifierRow output) = output := by
  apply Fin.ext
  simp [outputAt, pairIndex, ringDegree]

/-- One exact commitment coefficient row. -/
def row {shape : Phi81Relation.Shape} {verifierRows : Nat}
    (frame : Frame shape verifierRows)
    (coordinate : Fin (verifierRows * ringDegree)) : OwnedRow :=
  Atoms.linearCheckOwnedRow frame.owner
    (frame.firstOrdinal + coordinate.val) frame.one
    (terms frame (verifierRowAt coordinate) (outputAt coordinate))
    (Nightstream.Implementation.Lowering.Goldilocks.singleton
      (frame.commitment (verifierRowAt coordinate) (outputAt coordinate)) 1)

/-- Complete ordered Ajtai opening program. -/
def rows {shape : Phi81Relation.Shape} {verifierRows : Nat}
    (frame : Frame shape verifierRows) : List OwnedRow :=
  List.ofFn (row frame)

@[simp] theorem rows_length {shape : Phi81Relation.Shape}
    {verifierRows : Nat}
    (frame : Frame shape verifierRows) :
    (rows frame).length = verifierRows * ringDegree := by
  simp [rows]

/-- This slice reads a committed assignment and a public commitment but
allocates neither one. -/
def columns {shape : Phi81Relation.Shape} {verifierRows : Nat}
    (_frame : Frame shape verifierRows) : List OwnedColumn :=
  []

@[simp] theorem columns_length {shape : Phi81Relation.Shape}
    {verifierRows : Nat}
    (frame : Frame shape verifierRows) :
    (columns frame).length = 0 :=
  rfl

theorem columnIds_nodup {shape : Phi81Relation.Shape}
    {verifierRows : Nat}
    (frame : Frame shape verifierRows) :
    ((columns frame).map fun column => column.id).Nodup := by
  simp [columns]

private theorem nodup_ofFn_of_injective
    {alpha : Type} :
    ∀ {count : Nat}
      (function : Fin count → alpha),
      Function.Injective function →
      (List.ofFn function).Nodup
  | 0, function, injective => by
      simp
  | _ + 1, function, injective => by
      rw [List.ofFn_succ, List.nodup_cons]
      constructor
      · intro member
        rcases List.mem_ofFn.mp member with ⟨index, equal⟩
        exact Fin.succ_ne_zero index (injective equal)
      · exact nodup_ofFn_of_injective
          (fun index => function index.succ)
          (fun first second equal =>
            Fin.succ_inj.mp (injective equal))

theorem rowIds_nodup {shape : Phi81Relation.Shape}
    {verifierRows : Nat}
    (frame : Frame shape verifierRows) :
    ((rows frame).map fun owned => owned.id).Nodup := by
  rw [rows, List.map_ofFn]
  apply nodup_ofFn_of_injective
  intro first second equal
  apply Fin.ext
  exact Nat.add_left_cancel
    (congrArg (fun id : RowId => id.ordinal) equal)

theorem rows_owned {shape : Phi81Relation.Shape} {verifierRows : Nat}
    (frame : Frame shape verifierRows)
    (owned : OwnedRow)
    (member : owned ∈ rows frame) :
    owned.id.owner = frame.owner := by
  rcases List.mem_ofFn.mp member with ⟨coordinate, rfl⟩
  rfl

private theorem terms_supported
    {shape : Phi81Relation.Shape}
    {verifierRows : Nat}
    (frame : Frame shape verifierRows)
    (verifierRow : Fin verifierRows)
    (output : Fin ringDegree)
    (column : ColumnId)
    (mentioned :
      column ∈
        (terms frame verifierRow output).map fun term => term.column) :
    ∃ coordinate, column = frame.witness coordinate := by
  rcases List.mem_map.mp mentioned with
    ⟨term, termMember, rfl⟩
  rcases List.mem_flatten.mp termMember with
    ⟨blockTerms, blockTermsMember, termMember⟩
  rcases List.mem_ofFn.mp blockTermsMember with ⟨block, rfl⟩
  rcases List.mem_map.mp termMember with
    ⟨index, indexMember, rfl⟩
  have indexLt : index < ringDegree := List.mem_range.mp indexMember
  refine ⟨CarrierAction.carrierColumn block ⟨index, indexLt⟩, ?_⟩
  simp [laneTerm, indexLt]

/-- An Ajtai row mentions only the constant wire, the complete assignment,
or the corresponding public commitment coefficient. -/
theorem row_supported
    {shape : Phi81Relation.Shape}
    {verifierRows : Nat}
    (frame : Frame shape verifierRows)
    (coordinate : Fin (verifierRows * ringDegree))
    (column : ColumnId)
    (mentioned : column ∈ (row frame coordinate).columnIds) :
    column = frame.one ∨
      (∃ witnessCoordinate,
        column = frame.witness witnessCoordinate) ∨
    column =
        frame.commitment
          (verifierRowAt coordinate) (outputAt coordinate) := by
  change
    column ∈
      ((terms frame (verifierRowAt coordinate) (outputAt coordinate) ++
        Nightstream.Implementation.Lowering.Goldilocks.singleton frame.one 1 ++
        Nightstream.Implementation.Lowering.Goldilocks.singleton
          (frame.commitment
            (verifierRowAt coordinate) (outputAt coordinate)) 1).map
        fun term => term.column) at mentioned
  rw [List.map_append, List.map_append] at mentioned
  rcases List.mem_append.mp mentioned with beforeCommitment | commitment
  · rcases List.mem_append.mp beforeCommitment with termMember | one
    · exact Or.inr (Or.inl
        (terms_supported frame _ _ column termMember))
    · exact Or.inl (by
        simpa [Nightstream.Implementation.Lowering.Goldilocks.singleton]
          using one)
  · exact Or.inr (Or.inr
      (by
        simpa [Nightstream.Implementation.Lowering.Goldilocks.singleton]
          using commitment))

private theorem satisfies_ofFn_iff
    {count : Nat}
    (function : Fin count → OwnedRow)
    (assignment : ColumnId → F) :
    Nightstream.Implementation.Lowering.Goldilocks.Satisfies
      (List.ofFn function) assignment ↔
      ∀ coordinate, (function coordinate).row.Holds assignment := by
  induction count with
  | zero =>
      simp
  | succ count inductionHypothesis =>
      rw [List.ofFn_succ, satisfies_cons,
        inductionHypothesis (fun index => function index.succ)]
      constructor
      · rintro ⟨head, tail⟩ coordinate
        exact Fin.cases head tail coordinate
      · intro every
        exact ⟨every 0, fun index => every index.succ⟩

/-- Physical satisfaction binds every public commitment coefficient to the
semantic Ajtai commitment of the exact private witness columns. -/
theorem rows_sound {shape : Phi81Relation.Shape} {verifierRows : Nat}
    (frame : Frame shape verifierRows)
    (assignment : ColumnId → F)
    (constantOne : assignment frame.one = 1)
    (satisfied : Nightstream.Implementation.Lowering.Goldilocks.Satisfies
      (rows frame) assignment) :
    (fun verifierRow output =>
      assignment (frame.commitment verifierRow output)) =
        Commitment.commit frame.key
          (fun coordinate => assignment (frame.witness coordinate)) := by
  funext verifierRow output
  have holds :=
    (satisfies_ofFn_iff (row frame) assignment).mp satisfied
      (pairIndex verifierRow output)
  have equality :=
    (Atoms.linearCheckRow_iff assignment frame.one
      (terms frame verifierRow output)
      (Nightstream.Implementation.Lowering.Goldilocks.singleton
        (frame.commitment verifierRow output) 1)
      constantOne).mp (by simpa [row] using holds)
  rw [terms_eval_eq_commit] at equality
  simpa [Nightstream.Implementation.Lowering.Goldilocks.singleton,
    LinearCombination.eval,
    Fin.one_mul, Fin.add_zero] using equality.symm

/-- An honest commitment has a satisfying assignment without any new witness
column. -/
theorem rows_honest {shape : Phi81Relation.Shape} {verifierRows : Nat}
    (frame : Frame shape verifierRows)
    (assignment : ColumnId → F)
    (constantOne : assignment frame.one = 1)
    (commitmentMatches :
      (fun verifierRow output =>
        assignment (frame.commitment verifierRow output)) =
          Commitment.commit frame.key
            (fun coordinate => assignment (frame.witness coordinate))) :
    Nightstream.Implementation.Lowering.Goldilocks.Satisfies
      (rows frame) assignment := by
  apply (satisfies_ofFn_iff (row frame) assignment).mpr
  intro coordinate
  apply
    (Atoms.linearCheckRow_iff assignment frame.one
      (terms frame (verifierRowAt coordinate) (outputAt coordinate))
      (Nightstream.Implementation.Lowering.Goldilocks.singleton
        (frame.commitment
          (verifierRowAt coordinate) (outputAt coordinate)) 1)
      constantOne).mpr
  rw [terms_eval_eq_commit]
  have pointwise :=
    congrFun
      (congrFun commitmentMatches (verifierRowAt coordinate))
      (outputAt coordinate)
  simpa [Nightstream.Implementation.Lowering.Goldilocks.singleton,
    LinearCombination.eval,
    Fin.one_mul, Fin.add_zero] using pointwise.symm

/-- Exact local resource receipt. Dense coefficients affect row density and
nonzero count, but not the number of R1CS rows or allocated columns. -/
def cost (verifierRows : Nat) : Cost :=
  ⟨verifierRows * ringDegree, 0, 0, 0⟩

@[simp] theorem cost_rows {shape : Phi81Relation.Shape}
    {verifierRows : Nat}
    (frame : Frame shape verifierRows) :
    (rows frame).length = (cost verifierRows).recurringRows :=
  rows_length frame

@[simp] theorem cost_auxiliary (verifierRows : Nat) :
    (cost verifierRows).auxiliaryColumns = 0 :=
  rfl

end Nightstream.Implementation.R1CS.TerminalR1cs.Ajtai
