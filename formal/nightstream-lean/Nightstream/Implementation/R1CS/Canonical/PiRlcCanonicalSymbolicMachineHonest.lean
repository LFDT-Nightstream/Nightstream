import Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalSymbolicMachine
import Nightstream.Implementation.R1CS.Canonical.SymbolicDuplexPhysical

/-!
Contract: structural placement and honest completeness for the exact
fixed-active `Pi_RLC` symbolic transcript schedule.

Owns: proof that every raw-pair word and gate marker stays in the authoritative
prefix, closure of `WellPlaced` through the eight-block/fifteen-coordinate
recurrence, and the explicit satisfying assignment for its 135 emitted
permutations.

Does not own: canonical-u64 decomposition, rejection candidates, or selection;
those are the downstream sampler suffix.

Assurance tier: model-level canonical encoding.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalSymbolicMachineHonest

open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Core
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Schedule
open Nightstream.Implementation.R1CS.Canonical.SymbolicDuplexHonest
open Nightstream.Implementation.R1CS.Canonical.SymbolicDuplexPlacement
open Nightstream.Implementation.R1CS.Canonical.SymbolicDuplexPhysical

theorem fieldWord_before (base value : Nat) (positive : 0 < base) :
    ValueBefore base (PiRlcCanonicalSymbolicMachine.fieldWord value) := by
  intro column mentioned
  have same : column = 0 := by
    simpa [PiRlcCanonicalSymbolicMachine.fieldWord, Mentions] using mentioned
  rw [same]
  simp only [outputBase, callBase]
  omega

theorem rawPairFields_before
    (base first second : Nat) (positive : 0 < base) :
    ∀ value ∈ PiRlcCanonicalSymbolicMachine.rawPairFields first second,
      ValueBefore base value := by
  intro value member
  simp only [PiRlcCanonicalSymbolicMachine.rawPairFields, List.mem_cons,
    List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl | rfl
  · exact fieldWord_before base 2 positive
  · exact fieldWord_before base first positive
  · exact fieldWord_before base second positive

theorem fieldWord_inPrefix (base value : Nat) (positive : 0 < base) :
    ValueInPrefix base (PiRlcCanonicalSymbolicMachine.fieldWord value) := by
  intro column mentioned
  have same : column = 0 := by
    simpa [PiRlcCanonicalSymbolicMachine.fieldWord, Mentions] using mentioned
  omega

theorem rawPairFields_inPrefix
    (base first second : Nat) (positive : 0 < base) :
    ∀ value ∈ PiRlcCanonicalSymbolicMachine.rawPairFields first second,
      ValueInPrefix base value := by
  intro value member
  simp only [PiRlcCanonicalSymbolicMachine.rawPairFields, List.mem_cons,
    List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl | rfl
  · exact fieldWord_inPrefix base 2 positive
  · exact fieldWord_inPrefix base first positive
  · exact fieldWord_inPrefix base second positive

theorem appendRawPair_wellPlaced
    (base first second : Nat) (builder : SymbolicDuplex.Builder)
    (positive : 0 < base) (placed : WellPlaced base builder) :
    WellPlaced base
      (PiRlcCanonicalSymbolicMachine.appendRawPair
        base first second builder) := by
  unfold PiRlcCanonicalSymbolicMachine.appendRawPair
  exact absorbMany_wellPlaced base
    (PiRlcCanonicalSymbolicMachine.rawPairFields first second) builder placed
    (rawPairFields_before base first second positive)

theorem enterScalar_wellPlaced
    (base : Nat) (builder : SymbolicDuplex.Builder)
    (coordinate : Nat) (positive : 0 < base)
    (placed : WellPlaced base builder) :
    WellPlaced base
      (PiRlcCanonicalSymbolicMachine.enterScalar
        base builder coordinate) :=
  appendRawPair_wellPlaced base 0 coordinate builder positive placed

theorem digestBlock_wellPlaced
    (base : Nat) (builder : SymbolicDuplex.Builder)
    (counter : Nat) (positive : 0 < base)
    (placed : WellPlaced base builder) :
    WellPlaced base
      (PiRlcCanonicalSymbolicMachine.digestBlock base builder counter) := by
  apply gate_wellPlaced base _ positive
  exact appendRawPair_wellPlaced base 1 counter builder positive placed

theorem stateBeforeBlock_wellPlaced
    (base : Nat) (entered : SymbolicDuplex.Builder)
    (seed : Nat) (positive : 0 < base)
    (placed : WellPlaced base entered) :
    ∀ round,
      WellPlaced base
        (PiRlcCanonicalSymbolicMachine.stateBeforeBlock
          base entered seed round)
  | 0 => placed
  | round + 1 =>
      digestBlock_wellPlaced base
        (PiRlcCanonicalSymbolicMachine.stateBeforeBlock
          base entered seed round)
        (seed + round) positive
        (stateBeforeBlock_wellPlaced base entered seed positive placed round)

theorem scalarBuilder_wellPlaced
    (base : Nat) (builder : SymbolicDuplex.Builder)
    (coordinate : Nat) (positive : 0 < base)
    (placed : WellPlaced base builder) :
    WellPlaced base
      (PiRlcCanonicalSymbolicMachine.scalarBuilder
        base builder coordinate) := by
  unfold PiRlcCanonicalSymbolicMachine.scalarBuilder
  apply stateBeforeBlock_wellPlaced base _ _ positive
  exact enterScalar_wellPlaced base builder coordinate positive placed

theorem stateAt_wellPlaced
    (base : Nat) (initial : SymbolicDuplex.Builder)
    (positive : 0 < base) (placed : WellPlaced base initial) :
    ∀ coordinate,
      WellPlaced base
        (PiRlcCanonicalSymbolicMachine.stateAt
          base initial coordinate)
  | 0 => placed
  | coordinate + 1 =>
      scalarBuilder_wellPlaced base
        (PiRlcCanonicalSymbolicMachine.stateAt base initial coordinate)
        coordinate positive
        (stateAt_wellPlaced base initial positive placed coordinate)

theorem appendRawPair_wellOwned
    (base first second : Nat) (builder : SymbolicDuplex.Builder)
    (positive : 0 < base) (owned : WellOwned base builder) :
    WellOwned base
      (PiRlcCanonicalSymbolicMachine.appendRawPair
        base first second builder) := by
  unfold PiRlcCanonicalSymbolicMachine.appendRawPair
  exact absorbMany_wellOwned base
    (PiRlcCanonicalSymbolicMachine.rawPairFields first second) builder owned
    (rawPairFields_inPrefix base first second positive)

theorem enterScalar_wellOwned
    (base : Nat) (builder : SymbolicDuplex.Builder)
    (coordinate : Nat) (positive : 0 < base)
    (owned : WellOwned base builder) :
    WellOwned base
      (PiRlcCanonicalSymbolicMachine.enterScalar
        base builder coordinate) :=
  appendRawPair_wellOwned base 0 coordinate builder positive owned

theorem digestBlock_wellOwned
    (base : Nat) (builder : SymbolicDuplex.Builder)
    (counter : Nat) (positive : 0 < base)
    (owned : WellOwned base builder) :
    WellOwned base
      (PiRlcCanonicalSymbolicMachine.digestBlock base builder counter) := by
  apply gate_wellOwned base _ positive
  exact appendRawPair_wellOwned base 1 counter builder positive owned

theorem stateBeforeBlock_wellOwned
    (base : Nat) (entered : SymbolicDuplex.Builder)
    (seed : Nat) (positive : 0 < base)
    (owned : WellOwned base entered) :
    ∀ round,
      WellOwned base
        (PiRlcCanonicalSymbolicMachine.stateBeforeBlock
          base entered seed round)
  | 0 => owned
  | round + 1 =>
      digestBlock_wellOwned base
        (PiRlcCanonicalSymbolicMachine.stateBeforeBlock
          base entered seed round)
        (seed + round) positive
        (stateBeforeBlock_wellOwned base entered seed positive owned round)

theorem scalarBuilder_wellOwned
    (base : Nat) (builder : SymbolicDuplex.Builder)
    (coordinate : Nat) (positive : 0 < base)
    (owned : WellOwned base builder) :
    WellOwned base
      (PiRlcCanonicalSymbolicMachine.scalarBuilder
        base builder coordinate) := by
  unfold PiRlcCanonicalSymbolicMachine.scalarBuilder
  apply stateBeforeBlock_wellOwned base _ _ positive
  exact enterScalar_wellOwned base builder coordinate positive owned

theorem stateAt_wellOwned
    (base : Nat) (initial : SymbolicDuplex.Builder)
    (positive : 0 < base) (owned : WellOwned base initial) :
    ∀ coordinate,
      WellOwned base
        (PiRlcCanonicalSymbolicMachine.stateAt
          base initial coordinate)
  | 0 => owned
  | coordinate + 1 =>
      scalarBuilder_wellOwned base
        (PiRlcCanonicalSymbolicMachine.stateAt base initial coordinate)
        coordinate positive
        (stateAt_wellOwned base initial positive owned coordinate)

/-- The selected recipe boundary: the complete thirteen-matrix PiCCS replay
hands off its post-output lanes at cursor one, with no PiRLC-local permutation
entries yet. -/
def initialBuilder (lanes : State) : SymbolicDuplex.Builder :=
  SymbolicDuplex.start lanes 1

@[simp] theorem initialBuilder_absorbed (lanes : State) :
    (initialBuilder lanes).absorbed = 1 := rfl

@[simp] theorem initialBuilder_entries_length (lanes : State) :
    (initialBuilder lanes).entries.length = 0 := rfl

theorem initialBuilder_wellPlaced
    (base : Nat) (lanes : State)
    (lanesBefore :
      ∀ lane : Fin width, ∀ column,
        Mentions (lanes lane) column → column < outputBase base 0) :
    WellPlaced base (initialBuilder lanes) :=
  start_wellPlaced base lanes 1 lanesBefore

theorem initialBuilder_wellOwned
    (base : Nat) (lanes : State)
    (lanesInPrefix :
      ∀ lane : Fin width, ValueInPrefix base (lanes lane)) :
    WellOwned base (initialBuilder lanes) :=
  start_wellOwned base lanes 1 lanesInPrefix

/-- Exact fixed-active builder after all fifteen coefficient coordinates. -/
def fixedBuilder (base : Nat) (lanes : State) : SymbolicDuplex.Builder :=
  PiRlcCanonicalSymbolicMachine.stateAt base (initialBuilder lanes) 15

theorem fixedBuilder_wellPlaced
    (base : Nat) (lanes : State)
    (positive : 0 < base)
    (lanesBefore :
      ∀ lane : Fin width, ∀ column,
        Mentions (lanes lane) column → column < outputBase base 0) :
    WellPlaced base (fixedBuilder base lanes) :=
  stateAt_wellPlaced base (initialBuilder lanes) positive
    (initialBuilder_wellPlaced base lanes lanesBefore) 15

theorem fixedBuilder_wellOwned
    (base : Nat) (lanes : State) (positive : 0 < base)
    (lanesInPrefix :
      ∀ lane : Fin width, ValueInPrefix base (lanes lane)) :
    WellOwned base (fixedBuilder base lanes) :=
  stateAt_wellOwned base (initialBuilder lanes) positive
    (initialBuilder_wellOwned base lanes lanesInPrefix) 15

theorem fixedBuilder_entries_length (base : Nat) (lanes : State) :
    (fixedBuilder base lanes).entries.length = 135 := by
  unfold fixedBuilder initialBuilder
  simpa using
    PiRlcCanonicalSymbolicMachine.fixedActive_entries_length_of_one
      base (SymbolicDuplex.start lanes 1) rfl

theorem fixedRows_length
    (base : Nat) (constants : Constants) (lanes : State) :
    (SymbolicDuplex.rows base constants (fixedBuilder base lanes)).length =
      47520 := by
  rw [SymbolicDuplex.rows_length, fixedBuilder_entries_length]

def fixedAllocation (base : Nat) : List Nat :=
  temporaryColumns base 135

theorem fixedAllocation_length (base : Nat) :
    (fixedAllocation base).length = 47520 := by
  unfold fixedAllocation
  rw [temporaryColumns_length]
  rfl

theorem fixedAllocation_nodup (base : Nat) :
    (fixedAllocation base).Nodup :=
  temporaryColumns_nodup base 135

theorem fixedRows_ownership
    (base : Nat) (constants : Constants) (lanes : State) :
    (SymbolicDuplex.rows base constants (fixedBuilder base lanes)).length =
        (SymbolicDuplexPhysical.owners
          (fixedBuilder base lanes).entries).length
      ∧
        (SymbolicDuplexPhysical.owners
          (fixedBuilder base lanes).entries).Nodup
      ∧
        SymbolicDuplex.rows base constants (fixedBuilder base lanes) =
          (SymbolicDuplexPhysical.owners
            (fixedBuilder base lanes).entries).map
            (SymbolicDuplexPhysical.ownedRow base constants) :=
  ownership_is_positional base constants (fixedBuilder base lanes)

theorem fixedRows_conservation
    (base : Nat) (constants : Constants) (lanes : State)
    (positive : 0 < base)
    (lanesInPrefix :
      ∀ lane : Fin width, ValueInPrefix base (lanes lane))
    (row : Row)
    (member :
      row ∈ SymbolicDuplex.rows base constants (fixedBuilder base lanes))
    (column : Nat)
    (mentioned :
      Mentions row.a column ∨ Mentions row.b column ∨
        Mentions row.c column) :
    column < base ∨ column ∈ fixedAllocation base := by
  have placed : WellPlaced base (fixedBuilder base lanes) := by
    apply fixedBuilder_wellPlaced
    · exact positive
    intro lane column inLane
    simpa only [outputBase, callBase] using
      lanesInPrefix lane column inLane
  have owned : WellOwned base (fixedBuilder base lanes) :=
    fixedBuilder_wellOwned base lanes positive lanesInPrefix
  have conserved :=
    SymbolicDuplexPhysical.rows_conservation base constants
      (fixedBuilder base lanes) positive placed owned row member column
      mentioned
  rw [fixedBuilder_entries_length] at conserved
  exact conserved

/-- The explicit final assignment for the exact 135-call transcript. -/
def fixedWitness
    (base : Nat) (constants : Constants) (lanes : State)
    (initial : Nat → Nat) : Nat → Nat :=
  SymbolicDuplexHonest.witnesses base constants
    (fixedBuilder base lanes).entries initial

theorem fixedWitness_residues
    (base : Nat) (constants : Constants) (lanes : State)
    (initial : Nat → Nat)
    (initialResidues : ∀ column, initial column < goldilocksP) :
    ∀ column,
      fixedWitness base constants lanes initial column < goldilocksP :=
  SymbolicDuplexHonest.witnesses_residues base constants
    (fixedBuilder base lanes).entries initial initialResidues

theorem fixedWitness_constantWire
    (base : Nat) (constants : Constants) (lanes : State)
    (initial : Nat → Nat) (basePositive : 0 < base) :
    fixedWitness base constants lanes initial 0 = initial 0 :=
  SymbolicDuplexHonest.witnesses_constantWire base constants basePositive
    (fixedBuilder base lanes).entries initial

/-- Honest completeness of the exact 47,520-row symbolic transcript.  No
already-satisfied transcript or caller-supplied permutation result appears as
a premise. -/
theorem fixedRows_honest
    (base : Nat) (constants : Constants) (lanes : State)
    (initial : Nat → Nat)
    (lanesBefore :
      ∀ lane : Fin width, ∀ column,
        Mentions (lanes lane) column → column < outputBase base 0)
    (basePositive : 0 < base)
    (initialResidues : ∀ column, initial column < goldilocksP)
    (constantWire : initial 0 = 1) :
    Satisfies
      (SymbolicDuplex.rows base constants (fixedBuilder base lanes))
      (fixedWitness base constants lanes initial) :=
  SymbolicDuplexHonest.rows_honest base constants
    (fixedBuilder base lanes) initial
    (fixedBuilder_wellPlaced base lanes basePositive lanesBefore)
    basePositive initialResidues constantWire

end Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalSymbolicMachineHonest
