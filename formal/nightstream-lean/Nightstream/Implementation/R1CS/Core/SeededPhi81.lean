import Nightstream.Implementation.R1CS.Core.ChaCha8
import Nightstream.Implementation.R1CS.Core.ChaCha8Fast
import Nightstream.Implementation.R1CS.Core.Program
import Nightstream.Implementation.R1CS.Core.SeededPhi81Sampler

/-!
Contract: compact exact semantics for seeded Phi81 `A` blocks.

Rust stores these rows as `SeededPhi81LinearBlock` metadata instead of
materializing the dense `A` coefficients.  This module follows the same
shape.  Chunk seeds are expanded by the executable ChaCha8 source, each
message bit rotates one coefficient vector in `F[X] / (Phi_81)`, and the
result is compiled to the exact linear equations `A_row * 1 = output`.

The quantified soundness and completeness theorems operate on this compact
compiler; no expanded production matrix or row digest is a premise. Sampling
is owned by `SeededPhi81Sampler` and instantiated here with the optimized
stream. `ChaCha8Refinement` proves that stream equal to the pure model; Rust
`rand_chacha` conformance remains a separate boundary.
-/

namespace Nightstream.Implementation.R1CS.SeededPhi81

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program

def dimension : Nat := SeededPhi81Sampler.dimension

abbrev SeedSchedule := SeededPhi81Sampler.Schedule

/-- Expand only the `kappa * messageCols` base rotations.  The full dense
matrix is another factor of `dimension` larger and is never stored. -/
def SeedSchedule.baseRotations (schedule : SeedSchedule)
    (messageCols : Nat) : Option (List (List (List Nat))) :=
  SeededPhi81Sampler.Schedule.baseRotations schedule
    ChaCha8Fast.u64s messageCols

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

/-- A valid compact block contains an actual successful sampler execution;
the fallback value of `Block.baseRotations` is therefore unreachable. -/
theorem Block.Valid.baseRotations_success {block : Block}
    (valid : block.Valid) :
    exists rotations,
      block.schedule.baseRotations block.messageCols = some rotations := by
  rcases valid with ⟨_, _, _, _, _, _, _, _, _, sampler⟩
  unfold samplerValid at sampler
  cases execution : block.schedule.baseRotations block.messageCols with
  | none => simp [execution] at sampler
  | some rotations => exact ⟨rotations, rfl⟩

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
