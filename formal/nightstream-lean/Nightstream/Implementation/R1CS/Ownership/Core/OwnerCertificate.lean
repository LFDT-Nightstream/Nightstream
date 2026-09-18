import Nightstream.Implementation.R1CS.Correspondence.CanonicalU64.CanonicalU64Complete
import Nightstream.Implementation.R1CS.Core.CheckedProgram
import Nightstream.Implementation.R1CS.Core.Poseidon2Call
import Nightstream.Implementation.R1CS.Core.Relabel
import Nightstream.Implementation.R1CS.Core.SeededPhi81
import Nightstream.Implementation.R1CS.Ownership.ShiftedTernary.ShiftedTernary
import Nightstream.Implementation.R1CS.Correspondence.ShiftedTernary.ShiftedTernaryComplete
import Nightstream.Implementation.R1CS.Core.TranscriptCertificate

/-!
Contract: compact, ordered row ownership for large generated R1CS families.

An owner is partitioned into literal sparse-row runs, exact renamed Poseidon2
calls, and exact compact seeded-Phi81 blocks.  The generated schedule records
global row spans in emission order.  Hashes remain drift metadata; semantic
authority comes from the rows reconstructed by each piece.
-/

namespace Nightstream.Implementation.R1CS.OwnerCertificate

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program

inductive Payload where
  | ordinary (rows : List Row)
  | poseidon (call : Poseidon2Call.Call)
  | seededPhi81 (block : SeededPhi81.Block)
  | shiftedTernary (fieldColumn digitStart : Nat)
  | canonicalU64 (fieldColumn bitStart : Nat)
deriving DecidableEq, Repr, Inhabited

def shiftedTernaryColumnMap (fieldColumn digitStart : Nat) : List Nat :=
  [0, fieldColumn] ++ List.replicate 56 0 ++
    (List.range 122).map (fun index => digitStart + index)

def canonicalU64ColumnMap (fieldColumn bitStart : Nat) : List Nat :=
  [0, fieldColumn] ++ (List.range 66).map (fun index => bitStart + index)

def Payload.rows : Payload → List Row
  | .ordinary rows => rows
  | .poseidon call => call.rows
  | .seededPhi81 block => block.rows
  | .shiftedTernary fieldColumn digitStart =>
      ShiftedTernaryCompiler.canonicalRows.map
        (Relabel.row (shiftedTernaryColumnMap fieldColumn digitStart))
  | .canonicalU64 fieldColumn bitStart =>
      CanonicalU64.rows.map
        (Relabel.row (canonicalU64ColumnMap fieldColumn bitStart))

def Payload.rowCount : Payload → Nat
  | .ordinary rows => rows.length
  | .poseidon _ => Poseidon2Permutation.rows.length
  | .seededPhi81 block => block.kappa * SeededPhi81.dimension
  | .shiftedTernary _ _ => ShiftedTernaryCompiler.canonicalRows.length
  | .canonicalU64 _ _ => CanonicalU64.rows.length

theorem Payload.rows_length (payload : Payload) :
    payload.rows.length = payload.rowCount := by
  cases payload with
  | ordinary rows => rfl
  | poseidon call => simp [Payload.rows, Payload.rowCount, Poseidon2Call.Call.rows]
  | seededPhi81 block =>
      simp [Payload.rows, Payload.rowCount, SeededPhi81.Block.rows_length]
  | shiftedTernary fieldColumn digitStart =>
      simp [Payload.rows, Payload.rowCount]
  | canonicalU64 fieldColumn bitStart =>
      simp [Payload.rows, Payload.rowCount]

structure Piece where
  rowStart : Nat
  rowEnd : Nat
  payload : Payload
deriving DecidableEq, Repr, Inhabited

def Piece.Valid (piece : Piece) : Prop :=
  piece.rowStart ≤ piece.rowEnd ∧
  piece.rowEnd - piece.rowStart = piece.payload.rowCount

instance (piece : Piece) : Decidable piece.Valid := by
  unfold Piece.Valid
  infer_instance

def Piece.rows (piece : Piece) : List Row := piece.payload.rows

theorem Piece.rows_length (piece : Piece) (valid : piece.Valid) :
    piece.rows.length = piece.rowEnd - piece.rowStart := by
  rw [Piece.rows, Payload.rows_length, ← valid.2]

def contiguousFrom : Nat → List Piece → Bool
  | _, [] => true
  | cursor, piece :: rest =>
      decide (piece.rowStart = cursor) && contiguousFrom piece.rowEnd rest

structure Owner where
  rowStart : Nat
  rowEnd : Nat
  pieces : List Piece
deriving DecidableEq, Repr, Inhabited

def Owner.rows (owner : Owner) : List Row :=
  (owner.pieces.map Piece.rows).flatten

def Owner.Valid (owner : Owner) : Prop :=
  owner.rowStart ≤ owner.rowEnd ∧
  owner.pieces.all (fun piece => decide piece.Valid) = true ∧
  contiguousFrom owner.rowStart owner.pieces = true ∧
  owner.pieces.getLast?.map Piece.rowEnd =
    (if owner.pieces.isEmpty then none else some owner.rowEnd) ∧
  (owner.pieces.map (fun piece => piece.rowEnd - piece.rowStart)).sum =
    owner.rowEnd - owner.rowStart

instance (owner : Owner) : Decidable owner.Valid := by
  unfold Owner.Valid
  infer_instance

private theorem piece_lengths
    {pieces : List Piece}
    (valid : pieces.all (fun piece => decide piece.Valid) = true) :
    (pieces.map Piece.rows).map List.length =
      pieces.map (fun piece => piece.rowEnd - piece.rowStart) := by
  rw [List.map_map]
  apply List.map_inj_left.mpr
  intro piece member
  exact Piece.rows_length piece
    (of_decide_eq_true ((List.all_eq_true.mp valid) piece member))

/-- A valid ordered schedule covers exactly its declared half-open interval. -/
theorem Owner.rows_length {owner : Owner} (valid : owner.Valid) :
    owner.rows.length = owner.rowEnd - owner.rowStart := by
  unfold Owner.rows
  rw [List.length_flatten, piece_lengths valid.2.1]
  exact valid.2.2.2.2

/-! ## Independent executable semantics -/

/-- A certifying execution of an ordinary row segment.

The segment is reconstructed as a checked SSA program.  `rowIdentity` is a
compiler-classification certificate, while `checksHold` is evaluated on the
source state before `output` identifies the interpreter result with the shared
artifact assignment.  In particular, this witness contains neither
`Satisfies rows assignment` nor an owner-level acceptance conclusion. -/
structure OrdinaryExecution (rows : List Row) (assignment : Nat → Nat) where
  inputColumns : List Nat
  instructions : List CheckedProgram.Instruction
  source : Nat → Nat
  rowIdentity : CheckedProgram.rows instructions = rows
  wellFormed : Program.WellFormed inputColumns
    (CheckedProgram.definitions instructions)
  canonicalDefinitions : ∀ definition ∈
    CheckedProgram.definitions instructions, definition.Canonical
  sourceCanonical : ∀ column, source column < goldilocksP
  constantOneOwned : 0 ∈ inputColumns
  sourceOne : source 0 = 1
  executed : CheckedProgram.execute? source instructions = some assignment

/-- A checked SSA execution constructs every exact ordinary row. -/
theorem OrdinaryExecution.compiles
    {rows : List Row} {assignment : Nat → Nat}
    (execution : OrdinaryExecution rows assignment) :
    Satisfies rows assignment := by
  have compiled := CheckedProgram.complete_of_execute execution.wellFormed
    execution.canonicalDefinitions execution.sourceCanonical
    execution.constantOneOwned execution.sourceOne execution.executed
  rw [execution.rowIdentity] at compiled
  exact compiled

/-- One verifier equation, stated independently of `Satisfies`.  Ordinary
owner residuals have no stronger protocol-level interpretation, so this is
their explicit semantic endpoint. -/
def EquationHolds (assignment : Nat → Nat) (row : Row) : Prop :=
  lcEval assignment row.a * lcEval assignment row.b % goldilocksP =
    lcEval assignment row.c

instance (assignment : Nat → Nat) (row : Row) :
    Decidable (EquationHolds assignment row) := by
  unfold EquationHolds
  infer_instance

def EquationsAccepted (rows : List Row) (assignment : Nat → Nat) : Prop :=
  ∀ row ∈ rows, EquationHolds assignment row

def equationsCheck (rows : List Row) (assignment : Nat → Nat) : Bool :=
  rows.all fun row => decide (EquationHolds assignment row)

theorem equationsCheck_eq_true_iff (rows : List Row)
    (assignment : Nat → Nat) :
    equationsCheck rows assignment = true ↔
      EquationsAccepted rows assignment := by
  simp [equationsCheck, EquationsAccepted, List.all_eq_true,
    decide_eq_true_eq]

/-- Semantic endpoint selected by each compact payload.  Poseidon calls replay
the fixed SSA permutation, seeded-Phi81 blocks check their named linear forms,
and residual/compiler rows expose their exact verifier equations. -/
def Payload.Accepted (payload : Payload) (assignment : Nat → Nat) : Prop :=
  match payload with
  | .ordinary rows => EquationsAccepted rows assignment
  | .poseidon call => TranscriptCertificate.CallAccepted call assignment
  | .seededPhi81 block => block.Holds assignment
  | .shiftedTernary fieldColumn digitStart =>
      EquationsAccepted
        ((Payload.shiftedTernary fieldColumn digitStart).rows) assignment
  | .canonicalU64 fieldColumn bitStart =>
      EquationsAccepted
        ((Payload.canonicalU64 fieldColumn bitStart).rows) assignment

def Payload.check (payload : Payload) (assignment : Nat → Nat) : Bool :=
  match payload with
  | .ordinary rows => equationsCheck rows assignment
  | .poseidon call => TranscriptCertificate.callCheck call assignment
  | .seededPhi81 block => block.check assignment
  | .shiftedTernary fieldColumn digitStart =>
      equationsCheck
        ((Payload.shiftedTernary fieldColumn digitStart).rows) assignment
  | .canonicalU64 fieldColumn bitStart =>
      equationsCheck
        ((Payload.canonicalU64 fieldColumn bitStart).rows) assignment

theorem Payload.check_eq_true_iff (payload : Payload)
    (assignment : Nat → Nat) :
    payload.check assignment = true ↔ payload.Accepted assignment := by
  cases payload with
  | ordinary rows => exact equationsCheck_eq_true_iff rows assignment
  | poseidon call =>
      exact TranscriptCertificate.callCheck_eq_true_iff call assignment
  | seededPhi81 block => exact block.check_eq_true_iff assignment
  | shiftedTernary fieldColumn digitStart =>
      exact equationsCheck_eq_true_iff _ assignment
  | canonicalU64 fieldColumn bitStart =>
      exact equationsCheck_eq_true_iff _ assignment

/-- Exact payload rows reconstruct their independent semantic endpoint. -/
theorem Payload.sound {payload : Payload} {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies payload.rows assignment) :
    payload.Accepted assignment := by
  cases payload with
  | ordinary rows => exact satisfies
  | poseidon call =>
      exact Poseidon2PermutationSound.poseidon2Permutation_renamed_sound
        call.columnMap call.columnMap_zero canonical one satisfies
  | seededPhi81 block => exact SeededPhi81.sound canonical one satisfies
  | shiftedTernary fieldColumn digitStart => exact satisfies
  | canonicalU64 fieldColumn bitStart => exact satisfies

/-- Independent payload acceptance compiles back to every exact payload row. -/
theorem Payload.complete {payload : Payload} {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (accepted : payload.Accepted assignment) :
    Satisfies payload.rows assignment := by
  cases payload with
  | ordinary rows => exact accepted
  | poseidon call =>
      exact TranscriptCertificate.call_complete call canonical one accepted
  | seededPhi81 block => exact SeededPhi81.complete canonical one accepted
  | shiftedTernary fieldColumn digitStart => exact accepted
  | canonicalU64 fieldColumn bitStart => exact accepted

/-! ## Compiler executions

Unlike `Accepted`, the witness below is used by CIR-COMPLETE and never falls
back to raw row equations.  Ordinary segments must execute a checked SSA
program; compact payloads use their dedicated native interpreters. -/

inductive Payload.ExecutionWitness
    (field : CanonicalU64Complete.FieldInverse)
    (assignment : Nat → Nat) : Payload → Type where
  | ordinary {rows} : OrdinaryExecution rows assignment →
      ExecutionWitness field assignment (.ordinary rows)
  | poseidon {call} : TranscriptCertificate.CallAccepted call assignment →
      ExecutionWitness field assignment (.poseidon call)
  | seededPhi81 {block} : block.Holds assignment →
      ExecutionWitness field assignment (.seededPhi81 block)
  | shiftedTernary {fieldColumn digitStart} :
      ShiftedTernaryComplete.CanonicalWitness
        (Relabel.assignment
          (shiftedTernaryColumnMap fieldColumn digitStart) assignment) →
      ExecutionWitness field assignment
        (.shiftedTernary fieldColumn digitStart)
  | canonicalU64 {fieldColumn bitStart} :
      CanonicalU64Complete.ExecutionWitness field
        (Relabel.assignment
          (canonicalU64ColumnMap fieldColumn bitStart) assignment) →
      ExecutionWitness field assignment (.canonicalU64 fieldColumn bitStart)

theorem Payload.execution_complete
    {field : CanonicalU64Complete.FieldInverse}
    {payload : Payload} {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (witness : Payload.ExecutionWitness field assignment payload) :
    Satisfies payload.rows assignment := by
  cases witness with
  | ordinary execution => exact execution.compiles
  | poseidon accepted =>
      exact TranscriptCertificate.call_complete _ canonical one accepted
  | seededPhi81 holds => exact SeededPhi81.complete canonical one holds
  | shiftedTernary witness =>
      apply (Relabel.satisfies_mapped_iff ShiftedTernaryCompiler.canonicalRows
        (shiftedTernaryColumnMap _ _) assignment).mpr
      exact ShiftedTernaryComplete.canonicalRows_complete witness
  | canonicalU64 witness =>
      exact CanonicalU64Complete.mapped_complete
        (canonicalU64ColumnMap _ _) witness

def Piece.Accepted (piece : Piece) (assignment : Nat → Nat) : Prop :=
  piece.payload.Accepted assignment

def Piece.check (piece : Piece) (assignment : Nat → Nat) : Bool :=
  piece.payload.check assignment

theorem Piece.check_eq_true_iff (piece : Piece) (assignment : Nat → Nat) :
    piece.check assignment = true ↔ piece.Accepted assignment :=
  piece.payload.check_eq_true_iff assignment

/-- Every ordered piece is independently accepted.  This predicate carries
no owner-level row-satisfaction witness or prover-supplied acceptance bit. -/
def Owner.Accepted (owner : Owner) (assignment : Nat → Nat) : Prop :=
  ∀ piece ∈ owner.pieces, piece.Accepted assignment

def Owner.check (owner : Owner) (assignment : Nat → Nat) : Bool :=
  owner.pieces.all fun piece => piece.check assignment

theorem Owner.check_eq_true_iff (owner : Owner) (assignment : Nat → Nat) :
    owner.check assignment = true ↔ owner.Accepted assignment := by
  constructor
  · intro checked piece member
    exact (piece.check_eq_true_iff assignment).mp
      ((List.all_eq_true.mp checked) piece member)
  · intro accepted
    apply List.all_eq_true.mpr
    intro piece member
    exact (piece.check_eq_true_iff assignment).mpr (accepted piece member)

private theorem pieceSatisfies
    {owner : Owner} {assignment : Nat → Nat}
    (satisfies : Satisfies owner.rows assignment)
    {piece : Piece} (member : piece ∈ owner.pieces) :
    Satisfies piece.rows assignment := by
  rw [Owner.rows] at satisfies
  have allPieces := (satisfies_flatten_iff
    (owner.pieces.map Piece.rows) assignment).mp satisfies
  exact allPieces piece.rows (List.mem_map.mpr ⟨piece, member, rfl⟩)

/-- Exact ordered owner rows reconstruct every independent piece endpoint. -/
theorem Owner.sound {owner : Owner} {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies owner.rows assignment) :
    owner.Accepted assignment := by
  intro piece member
  exact Payload.sound canonical one (pieceSatisfies satisfies member)

/-- Independent piece acceptance satisfies the owner's exact ordered rows. -/
theorem Owner.complete {owner : Owner} {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (accepted : owner.Accepted assignment) :
    Satisfies owner.rows assignment := by
  rw [Owner.rows]
  apply (satisfies_flatten_iff
    (owner.pieces.map Piece.rows) assignment).mpr
  intro rows member
  rcases List.mem_map.mp member with ⟨piece, pieceMember, rfl⟩
  exact Payload.complete canonical one (accepted piece pieceMember)

def Piece.ExecutionWitness
    (field : CanonicalU64Complete.FieldInverse)
    (piece : Piece) (assignment : Nat → Nat) : Type :=
  Payload.ExecutionWitness field assignment piece.payload

/-- Successful compiler execution for every piece in one generated owner. -/
def Owner.ExecutionWitness
    (field : CanonicalU64Complete.FieldInverse)
    (owner : Owner) (assignment : Nat → Nat) : Type :=
  ∀ piece ∈ owner.pieces, piece.ExecutionWitness field assignment

/-- Piecewise native/compiler execution reconstructs every exact owner row. -/
theorem Owner.execution_complete
    {field : CanonicalU64Complete.FieldInverse}
    {owner : Owner} {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (witness : owner.ExecutionWitness field assignment) :
    Satisfies owner.rows assignment := by
  rw [Owner.rows]
  apply (satisfies_flatten_iff
    (owner.pieces.map Piece.rows) assignment).mpr
  intro rows rowsMember
  rcases List.mem_map.mp rowsMember with ⟨piece, pieceMember, rfl⟩
  exact Payload.execution_complete canonical one
    (witness piece pieceMember)

end Nightstream.Implementation.R1CS.OwnerCertificate
