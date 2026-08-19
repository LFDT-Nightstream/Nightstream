import Nightstream.Implementation.R1CS.Core.ChaCha8
import Nightstream.Implementation.R1CS.Core.ChaCha8Fast
import Nightstream.Implementation.R1CS.Core.Program

/-!
Contract: compact exact semantics for seeded Phi81 `A` blocks.

Rust stores these rows as `SeededPhi81LinearBlock` metadata instead of
materializing the dense `A` coefficients.  This module follows the same
shape.  Chunk seeds are expanded by the executable ChaCha8 source, each
message bit rotates one coefficient vector in `F[X] / (Phi_81)`, and the
result is compiled to the exact linear equations `A_row * 1 = output`.

The quantified soundness and completeness theorems operate on this compact
compiler; no expanded production matrix or row digest is a premise.  The
remaining cross-language boundary is the source translation itself.  It is
explicitly isolated in `ChaCha8.lean` and must be pinned to Rust with
generated stream/coefficient vectors.
-/

namespace Nightstream.Implementation.R1CS.SeededPhi81

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program

def dimension : Nat := 54

structure SeedSchedule where
  chunkSize : Nat
  seedsByOutput : List (List (List Nat))
  rejectionFuel : Nat
deriving DecidableEq, Repr

private def nextAccepted (seed : List Nat) : Nat → Nat → Option (Nat × Nat)
  | _, 0 => none
  | wordPosition, fuel + 1 =>
      let candidate := (ChaCha8Fast.u64s seed wordPosition 1).getD 0 0
      if candidate < goldilocksP then some (candidate, wordPosition + 2)
      else nextAccepted seed (wordPosition + 2) fuel

private def repairRejected (seed : List Nat) (fuel : Nat) :
    List Nat → Nat → Option (List Nat × Nat)
  | [], wordPosition => some ([], wordPosition)
  | candidate :: tail, wordPosition =>
      let accepted :=
        if candidate < goldilocksP then some (candidate, wordPosition)
        else nextAccepted seed wordPosition fuel
      match accepted with
      | none => none
      | some (value, nextPosition) =>
          match repairRejected seed fuel tail nextPosition with
          | none => none
          | some (values, finalPosition) =>
              some (value :: values, finalPosition)

private def sampleVector (seed : List Nat) (fuel wordPosition : Nat) :
    Option (List Nat × Nat) :=
  let raw := ChaCha8Fast.u64s seed wordPosition dimension
  repairRejected seed fuel raw (wordPosition + 2 * dimension)

private def sampleVectors.go (seed : List Nat) (fuel : Nat) :
    Nat → Nat → List (List Nat) → Option (List (List Nat))
  | 0, _, reversed => some reversed.reverse
  | count + 1, wordPosition, reversed =>
      match sampleVector seed fuel wordPosition with
      | none => none
      | some (vector, nextPosition) =>
          sampleVectors.go seed fuel count nextPosition (vector :: reversed)

/-- Tail-recursive stream walk. Production commitment blocks contain several
thousand message columns, so the structurally equivalent cons-after-recursion
definition exhausts the native evaluator stack even though ChaCha itself is
fast. -/
private def sampleVectors (seed : List Nat) (fuel : Nat)
    (count wordPosition : Nat) : Option (List (List Nat)) :=
  sampleVectors.go seed fuel count wordPosition []

private def chunkMessageCount
    (messageCols chunkSize chunkIndex : Nat) : Nat :=
  let start := chunkIndex * chunkSize
  if start < messageCols then Nat.min chunkSize (messageCols - start) else 0

private def sampleOutput (messageCols chunkSize fuel : Nat) :
    Nat → List (List Nat) → Option (List (List Nat))
  | _, [] => some []
  | chunkIndex, seed :: tail =>
      match sampleVectors seed fuel
          (chunkMessageCount messageCols chunkSize chunkIndex) 0 with
      | none => none
      | some vectors =>
          match sampleOutput messageCols chunkSize fuel (chunkIndex + 1) tail with
          | none => none
          | some rest => some (vectors ++ rest)

/-- Expand only the `kappa * messageCols` base rotations.  The full dense
matrix is another factor of `dimension` larger and is never stored. -/
def SeedSchedule.baseRotations (schedule : SeedSchedule)
    (messageCols : Nat) : Option (List (List (List Nat))) :=
  let rec outputs : List (List (List Nat)) → Option (List (List (List Nat)))
    | [] => some []
    | seeds :: tail =>
        match sampleOutput messageCols schedule.chunkSize
            schedule.rejectionFuel 0 seeds with
        | none => none
        | some rotations =>
            match outputs tail with
            | none => none
            | some rest => some (rotations :: rest)
  outputs schedule.seedsByOutput

structure Block where
  rowStart : Nat
  wordStarts : List Nat
  wordWidth : Nat
  kappa : Nat
  messageCols : Nat
  outputColumns : List Nat
  superneoTransformedColumns : Bool
  schedule : SeedSchedule
deriving DecidableEq, Repr

private def allSeedsValid (schedule : SeedSchedule) : Bool :=
  schedule.seedsByOutput.all fun outputSeeds =>
    outputSeeds.all fun seed =>
      decide (seed.length = 32) && seed.all fun byte => decide (byte < 256)

private def allRotationsCanonical
    (rotations : List (List (List Nat))) : Bool :=
  rotations.all fun output =>
    output.all fun vector =>
      decide (vector.length = dimension) &&
        vector.all fun coefficient => decide (coefficient < goldilocksP)

private def seedChunkCountsValid (block : Block) : Bool :=
  block.schedule.seedsByOutput.all fun outputSeeds =>
    decide (outputSeeds.length =
      (block.messageCols + block.schedule.chunkSize - 1) /
        block.schedule.chunkSize)

private def samplerValid (block : Block) : Bool :=
  match block.schedule.baseRotations block.messageCols with
  | none => false
  | some rotations =>
      decide (rotations.length = block.kappa) &&
      rotations.all (fun output => decide (output.length = block.messageCols)) &&
      allRotationsCanonical rotations

/-- Executable geometry/source certificate matching
`SeededPhi81LinearBlock::new_with_word_width`. -/
def Block.Valid (block : Block) : Prop :=
  0 < block.wordWidth ∧
  0 < block.kappa ∧
  0 < block.schedule.chunkSize ∧
  block.superneoTransformedColumns = false ∧
  block.messageCols =
    (block.wordStarts.length * block.wordWidth + dimension - 1) / dimension ∧
  block.outputColumns.length = dimension * block.kappa ∧
  block.schedule.seedsByOutput.length = block.kappa ∧
  seedChunkCountsValid block = true ∧
  allSeedsValid block.schedule = true ∧
  samplerValid block = true

instance (block : Block) : Decidable block.Valid := by
  unfold Block.Valid seedChunkCountsValid allSeedsValid samplerValid
    allRotationsCanonical
  infer_instance

private def fieldNeg (value : Nat) : Nat :=
  let value := value % goldilocksP
  if value = 0 then 0 else goldilocksP - value

private def fieldSub (left right : Nat) : Nat :=
  (left % goldilocksP + fieldNeg right) % goldilocksP

/-- Multiplication by `X` modulo `Phi_81 = X^54 + X^27 + 1`. -/
def rotatePhi81 (current : List Nat) : List Nat :=
  let last := current.getD (dimension - 1) 0
  (List.range dimension).map fun coordinate =>
    if coordinate = 0 then fieldNeg last
    else if coordinate = 27 then
      fieldSub (current.getD 26 0) last
    else current.getD (coordinate - 1) 0

private def rotatePow : Nat → List Nat → List Nat
  | 0, current => current
  | count + 1, current => rotatePow count (rotatePhi81 current)

def Block.baseRotations (block : Block) : List (List (List Nat)) :=
  (block.schedule.baseRotations block.messageCols).getD []

def Block.coefficient (block : Block)
    (output messageCol messageRow coordinate : Nat) : Nat :=
  let base := ((block.baseRotations.getD output []).getD messageCol [])
  (rotatePow messageRow base).getD coordinate 0

def Block.bitColumn (block : Block) (bitIndex : Nat) : Option Nat :=
  if block.wordWidth = 0 then none
  else if bitIndex < block.wordStarts.length * block.wordWidth then
    some (block.wordStarts.getD (bitIndex / block.wordWidth) 0 +
      bitIndex % block.wordWidth)
  else none

def Block.terms (block : Block) (output coordinate : Nat) :
    List (Nat × Nat) :=
  (List.range block.messageCols).flatMap fun messageCol =>
    (List.range dimension).filterMap fun messageRow =>
      match block.bitColumn (messageRow * block.messageCols + messageCol) with
      | none => none
      | some column =>
          let coefficient :=
            block.coefficient output messageCol messageRow coordinate
          if coefficient = 0 then none else some (column, coefficient)

def Block.definition (block : Block) (output coordinate : Nat) : Definition :=
  ⟨block.outputColumns.getD (output * dimension + coordinate) 0,
    .linear (block.terms output coordinate)⟩

def Block.definitions (block : Block) : List Definition :=
  (List.range block.kappa).flatMap fun output =>
    (List.range dimension).map fun coordinate =>
      block.definition output coordinate

def Block.rows (block : Block) : List Row :=
  block.definitions.map Definition.row

theorem Block.definitions_length (block : Block) :
    block.definitions.length = block.kappa * dimension := by
  simp [Block.definitions, List.map_const']

theorem Block.rows_length (block : Block) :
    block.rows.length = block.kappa * dimension := by
  simp [Block.rows, Block.definitions_length]

/-- Semantic conclusion of the compact compiler: every output coordinate is
the seeded Phi81 linear form of the named input-word columns. -/
def Block.Holds (block : Block) (assignment : Nat → Nat) : Prop :=
  ∀ definition ∈ block.definitions, Definition.Holds assignment definition

instance (block : Block) (assignment : Nat → Nat) :
    Decidable (block.Holds assignment) := by
  unfold Block.Holds
  infer_instance

/-- Executable independent semantics of one compact seeded linear map. -/
def Block.check (block : Block) (assignment : Nat → Nat) : Bool :=
  block.definitions.all fun definition =>
    decide (Definition.Holds assignment definition)

theorem Block.check_eq_true_iff (block : Block) (assignment : Nat → Nat) :
    block.check assignment = true ↔ block.Holds assignment := by
  simp [Block.check, Block.Holds, List.all_eq_true, decide_eq_true_eq]

/-- Exact compact-row soundness.  No commitment claim, digest, or
prover-supplied validity bit appears among the premises. -/
theorem sound {block : Block} {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies block.rows assignment) :
    block.Holds assignment := by
  exact definitions_sound canonical one satisfies

private theorem definition_complete
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (definition : Definition)
    (holds : Definition.Holds assignment definition) :
    RowHolds assignment definition.row := by
  cases definition with
  | mk output rhs =>
      cases rhs with
      | linear terms =>
          have outputLt := canonical output
          simpa [Definition.Holds, Definition.row, Rhs.eval, RowHolds,
            lcEval, one, Nat.mod_eq_of_lt outputLt] using holds.symm
      | product left right =>
          have outputLt := canonical output
          simpa [Definition.Holds, Definition.row, Rhs.eval, RowHolds,
            lcEval, Nat.mod_eq_of_lt outputLt] using holds.symm

/-- Exact compact-row completeness: the linear equations themselves are
sufficient to satisfy every emitted seeded block row. -/
theorem complete {block : Block} {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (holds : block.Holds assignment) :
    Satisfies block.rows assignment := by
  intro row member
  rcases List.mem_map.mp member with ⟨definition, definitionMember, rfl⟩
  exact definition_complete canonical one definition
    (holds definition definitionMember)

end Nightstream.Implementation.R1CS.SeededPhi81
