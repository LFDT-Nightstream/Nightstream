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

def fieldNeg (value : Nat) : Nat :=
  let value := value % goldilocksP
  if value = 0 then 0 else goldilocksP - value

def fieldSub (left right : Nat) : Nat :=
  (left % goldilocksP + fieldNeg right) % goldilocksP

/-- Multiplication by `X` modulo `Phi_81 = X^54 + X^27 + 1`. -/
def rotatePhi81 (current : List Nat) : List Nat :=
  let last := current.getD (dimension - 1) 0
  (List.range dimension).map fun coordinate =>
    if coordinate = 0 then fieldNeg last
    else if coordinate = 27 then
      fieldSub (current.getD 26 0) last
    else current.getD (coordinate - 1) 0

def rotatePow : Nat → List Nat → List Nat
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

/-- One optional sparse term before list assembly. -/
def Block.term (block : Block) (output coordinate messageCol messageRow : Nat) :
    Option (Nat × Nat) :=
  match block.bitColumn (messageRow * block.messageCols + messageCol) with
  | none => none
  | some column =>
      let coefficient :=
        block.coefficient output messageCol messageRow coordinate
      if coefficient = 0 then none else some (column, coefficient)

def Block.terms (block : Block) (output coordinate : Nat) :
    List (Nat × Nat) :=
  (List.range block.messageCols).flatMap fun messageCol =>
    (List.range dimension).filterMap fun messageRow =>
      block.term output coordinate messageCol messageRow

/-- Assignment value selected by one dense matrix coordinate. Missing tail
coordinates have the authoritative value zero. -/
def Block.inputValue
    (block : Block) (assignment : Nat → Nat) (messageCol messageRow : Nat) :
    Nat :=
  match block.bitColumn (messageRow * block.messageCols + messageCol) with
  | none => 0
  | some column => assignment column

theorem Block.inputValue_eq_of_bitColumn_some
    {block : Block} {assignment : Nat → Nat}
    {messageCol messageRow column : Nat}
    (selected : block.bitColumn
      (messageRow * block.messageCols + messageCol) = some column) :
    block.inputValue assignment messageCol messageRow = assignment column := by
  simp [Block.inputValue, selected]

theorem Block.inputValue_eq_zero_of_bitColumn_none
    {block : Block} {assignment : Nat → Nat}
    {messageCol messageRow : Nat}
    (absent : block.bitColumn
      (messageRow * block.messageCols + messageCol) = none) :
    block.inputValue assignment messageCol messageRow = 0 := by
  simp [Block.inputValue, absent]

/-- Unreduced contribution of one dense seeded-matrix coordinate. -/
def Block.termValue
    (block : Block) (assignment : Nat → Nat)
    (output coordinate messageCol messageRow : Nat) : Nat :=
  block.coefficient output messageCol messageRow coordinate *
    block.inputValue assignment messageCol messageRow

/-- Exact dense meaning of one compact output row. -/
def Block.linearValue
    (block : Block) (assignment : Nat → Nat)
    (output coordinate : Nat) : Nat :=
  ((List.range block.messageCols).foldl (fun outer messageCol =>
    (List.range dimension).foldl (fun inner messageRow =>
      inner + block.termValue assignment output coordinate
        messageCol messageRow) outer) 0) % goldilocksP

theorem Block.linearValue_lt
    (block : Block) (assignment : Nat → Nat)
    (output coordinate : Nat) :
    block.linearValue assignment output coordinate < goldilocksP := by
  unfold Block.linearValue
  exact Nat.mod_lt _ (by decide)

private theorem foldl_term_exact
    (block : Block) (assignment : Nat → Nat)
    (output coordinate messageCol : Nat) (messageRows : List Nat) :
    ∀ initial,
      (messageRows.filterMap fun messageRow =>
        block.term output coordinate messageCol messageRow).foldl
          (fun accumulated term =>
            accumulated + term.2 * assignment term.1) initial =
        messageRows.foldl (fun accumulated messageRow =>
          accumulated + block.termValue assignment output coordinate
            messageCol messageRow) initial := by
  intro initial
  induction messageRows generalizing initial with
  | nil => rfl
  | cons messageRow messageRows inductionHypothesis =>
      cases source : block.bitColumn
          (messageRow * block.messageCols + messageCol) with
      | none =>
          have termNone :
              block.term output coordinate messageCol messageRow = none := by
            simp [Block.term, source]
          have valueZero :
              block.termValue assignment output coordinate
                messageCol messageRow = 0 := by
            simp [Block.termValue, Block.inputValue, source]
          simp only [List.filterMap_cons, termNone, List.foldl_cons,
            valueZero, Nat.add_zero]
          exact inductionHypothesis initial
      | some column =>
          by_cases zero :
              block.coefficient output messageCol messageRow coordinate = 0
          · have termNone :
                block.term output coordinate messageCol messageRow = none := by
              simp [Block.term, source, zero]
            have valueZero :
                block.termValue assignment output coordinate
                  messageCol messageRow = 0 := by
              simp [Block.termValue, Block.inputValue, source, zero]
            simp only [List.filterMap_cons, termNone, List.foldl_cons,
              valueZero, Nat.add_zero]
            exact inductionHypothesis initial
          · have termSome :
                block.term output coordinate messageCol messageRow =
                  some (column,
                    block.coefficient output messageCol messageRow
                      coordinate) := by
              simp [Block.term, source, zero]
            have valueSome :
                block.termValue assignment output coordinate
                    messageCol messageRow =
                  block.coefficient output messageCol messageRow coordinate *
                    assignment column := by
              simp [Block.termValue, Block.inputValue, source]
            simp only [List.filterMap_cons, termSome, List.foldl_cons,
              valueSome]
            exact inductionHypothesis _

private theorem foldl_terms_exact
    (block : Block) (assignment : Nat → Nat)
    (output coordinate : Nat) (messageCols : List Nat) :
    ∀ initial,
      (messageCols.flatMap fun messageCol =>
        (List.range dimension).filterMap fun messageRow =>
          block.term output coordinate messageCol messageRow).foldl
          (fun accumulated term =>
            accumulated + term.2 * assignment term.1) initial =
        messageCols.foldl (fun outer messageCol =>
          (List.range dimension).foldl (fun inner messageRow =>
            inner + block.termValue assignment output coordinate
              messageCol messageRow) outer) initial := by
  intro initial
  induction messageCols generalizing initial with
  | nil => rfl
  | cons messageCol messageCols inductionHypothesis =>
      simp only [List.flatMap_cons, List.foldl_append, List.foldl_cons]
      rw [foldl_term_exact, inductionHypothesis]

/-- Sparse zero elision and the absent 28-coordinate tail do not change the
dense linear value of a compact row. -/
theorem Block.lcEval_terms_eq_linearValue
    (block : Block) (assignment : Nat → Nat)
    (output coordinate : Nat) :
    lcEval assignment (block.terms output coordinate) =
      block.linearValue assignment output coordinate := by
  unfold lcEval Block.terms Block.linearValue
  exact congrArg (fun value => value % goldilocksP)
    (foldl_terms_exact block assignment output coordinate
      (List.range block.messageCols) 0)

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

/-- Every accepted compact output definition equals the dense seeded linear
value derived from the same input columns. -/
theorem Block.output_eq_linearValue
    {block : Block} {assignment : Nat → Nat}
    (holds : block.Holds assignment)
    (output : Fin block.kappa) (coordinate : Fin dimension) :
    assignment
        (block.outputColumns.getD
          (output.val * dimension + coordinate.val) 0) =
      block.linearValue assignment output.val coordinate.val := by
  have member : block.definition output.val coordinate.val ∈
      block.definitions := by
    unfold Block.definitions
    apply List.mem_flatMap.mpr
    refine ⟨output.val, List.mem_range.mpr output.isLt, ?_⟩
    exact List.mem_map.mpr
      ⟨coordinate.val, List.mem_range.mpr coordinate.isLt, rfl⟩
  have definitionHolds := holds _ member
  change assignment
      (block.outputColumns.getD
        (output.val * dimension + coordinate.val) 0) =
    lcEval assignment (block.terms output.val coordinate.val) at definitionHolds
  rw [definitionHolds]
  exact block.lcEval_terms_eq_linearValue assignment output.val coordinate.val

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
