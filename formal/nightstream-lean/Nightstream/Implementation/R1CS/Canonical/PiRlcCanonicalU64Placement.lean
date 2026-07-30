import Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalU64Honest
import Nightstream.Implementation.R1CS.Canonical.SymbolicDuplexPhysical

/-!
Contract: discharge the canonical-u64 sampler's caller-column placement from
the actual symbolic transcript layout.

The symbolic transcript owns one 361-column Poseidon2 space per emitted
permutation.  The selected post-PiCCS cursor-one handoff emits five
permutations per sampler coordinate.  This module proves that placing the u64
allocation after the exact end of the `count`-coordinate transcript constructs
`InputsBelow`; no caller supplies a per-lane ownership conclusion.

Assurance tier: model-level canonical encoding.
-/

set_option autoImplicit false
set_option maxRecDepth 100000

namespace Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalU64Placement

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalU64Honest
open Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler.ProductionAlphabet

/-- First column not owned by the complete symbolic sampler transcript. -/
def transcriptEnd
    (duplexBase count : Nat)
    (initialBuilder : SymbolicDuplex.Builder) : Nat :=
  duplexBase +
    (initialBuilder.entries.length + count * 5) * SymbolicDuplex.stride

private theorem guarded_entries_length_mono
    (base : Nat) (builder : SymbolicDuplex.Builder) :
    builder.entries.length ≤
      (SymbolicDuplex.guarded base builder).entries.length := by
  unfold SymbolicDuplex.guarded
  split
  · simp only [SymbolicDuplex.permute_entries_length]
    omega
  · exact Nat.le_refl _

private theorem absorb_entries_length_mono
    (base : Nat) (value : LinComb)
    (builder : SymbolicDuplex.Builder) :
    builder.entries.length ≤
      (SymbolicDuplex.absorb base value builder).entries.length := by
  unfold SymbolicDuplex.absorb
  exact guarded_entries_length_mono base builder

private theorem absorbMany_entries_length_mono
    (base : Nat) :
    ∀ (values : List LinComb) (builder : SymbolicDuplex.Builder),
      builder.entries.length ≤
        (SymbolicDuplex.absorbMany base values builder).entries.length
  | [], _ => Nat.le_refl _
  | value :: rest, builder =>
      Nat.le_trans (absorb_entries_length_mono base value builder)
        (absorbMany_entries_length_mono base rest
          (SymbolicDuplex.absorb base value builder))

private theorem digestBlock_entries_length_mono
    (base : Nat) (builder : SymbolicDuplex.Builder) (counter : Nat) :
    builder.entries.length ≤
      (PiRlcCanonicalSymbolicMachine.digestBlock
        base builder counter).entries.length := by
  unfold PiRlcCanonicalSymbolicMachine.digestBlock
    SymbolicDuplex.gate
  have appendMono :=
    absorbMany_entries_length_mono base
      (PiRlcCanonicalSymbolicMachine.rawPairFields 1 counter) builder
  have oneMono :=
    absorb_entries_length_mono base SymbolicDuplex.one
      (PiRlcCanonicalSymbolicMachine.appendRawPair
        base 1 counter builder)
  rw [SymbolicDuplex.permute_entries_length]
  exact Nat.le_trans appendMono
    (Nat.le_trans oneMono (Nat.le_add_right _ _))

private theorem stateBeforeBlock_entries_length_mono
    (base : Nat) (entered : SymbolicDuplex.Builder) (seed : Nat) :
    ∀ start finish,
      start ≤ finish →
      (PiRlcCanonicalSymbolicMachine.stateBeforeBlock
          base entered seed start).entries.length ≤
        (PiRlcCanonicalSymbolicMachine.stateBeforeBlock
          base entered seed finish).entries.length
  | start, 0, bounded => by
      have : start = 0 := by omega
      subst start
      exact Nat.le_refl _
  | start, finish + 1, bounded => by
      by_cases atEnd : start = finish + 1
      · subst start
        exact Nat.le_refl _
      · have beforeEnd : start ≤ finish := by omega
        exact Nat.le_trans
          (stateBeforeBlock_entries_length_mono
            base entered seed start finish beforeEnd)
          (digestBlock_entries_length_mono base
            (PiRlcCanonicalSymbolicMachine.stateBeforeBlock
              base entered seed finish)
            (seed + finish))

private theorem digestLanes_member_lt
    (base : Nat) (builder : SymbolicDuplex.Builder) (counter : Nat)
    (lane : Fin 4) (column coefficient : Nat)
    (member :
      (column, coefficient) ∈
        PiRlcCanonicalSymbolicMachine.digestLanes
          base builder counter lane) :
    column <
      base +
        (PiRlcCanonicalSymbolicMachine.digestBlock
          base builder counter).entries.length *
          SymbolicDuplex.stride := by
  simp only [PiRlcCanonicalSymbolicMachine.digestLanes,
    PiRlcCanonicalSymbolicMachine.digestBlock, SymbolicDuplex.gate,
    SymbolicDuplex.permute, SymbolicDuplex.outputState,
    SymbolicDuplex.layoutAt, Poseidon2Layout.shiftedLayout,
    List.mem_singleton] at member ⊢
  have laneLt := lane.isLt
  rcases Prod.mk.inj member with ⟨rfl, _⟩
  rw [SymbolicDuplex.stride_eq]
  simp only [List.length_append, List.length_singleton]
  omega

private theorem digestLanes_member_temporaryColumns
    (base totalCalls : Nat) (builder : SymbolicDuplex.Builder)
    (counter : Nat) (lane : Fin 4) (column coefficient : Nat)
    (member :
      (column, coefficient) ∈
        PiRlcCanonicalSymbolicMachine.digestLanes
          base builder counter lane)
    (bounded :
      (PiRlcCanonicalSymbolicMachine.digestBlock
        base builder counter).entries.length ≤ totalCalls) :
    column ∈
      SymbolicDuplexPhysical.temporaryColumns base totalCalls := by
  let ready :=
    SymbolicDuplex.absorb base SymbolicDuplex.one
      (PiRlcCanonicalSymbolicMachine.appendRawPair
        base 1 counter builder)
  have finalLength :
      (PiRlcCanonicalSymbolicMachine.digestBlock
        base builder counter).entries.length =
        ready.entries.length + 1 := by
    simp only [PiRlcCanonicalSymbolicMachine.digestBlock,
      SymbolicDuplex.gate, SymbolicDuplex.permute_entries_length, ready]
  have callLt : ready.entries.length < totalCalls := by
    rw [finalLength] at bounded
    omega
  have outputMember :=
    SymbolicDuplexPhysical.outputColumn_mem
      base totalCalls ready.entries.length callLt
      ⟨lane.val, by
        have laneLt := lane.isLt
        simp only [Poseidon2Core.width] at *
        omega⟩
  simp only [PiRlcCanonicalSymbolicMachine.digestLanes,
    PiRlcCanonicalSymbolicMachine.digestBlock, SymbolicDuplex.gate,
    SymbolicDuplex.permute, SymbolicDuplex.outputState,
    List.mem_singleton] at member
  rcases Prod.mk.inj member with ⟨rfl, _⟩
  exact outputMember

private theorem laneInput_member_lt_transcriptEnd
    (duplexBase count : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    (cursorOne : initialBuilder.absorbed = 1)
    (coordinate : Fin count)
    (position : Fin PiRlcCanonicalU64.lanesPerScalar)
    (column coefficient : Nat)
    (member :
      (column, coefficient) ∈
        PiRlcCanonicalU64.laneInput duplexBase
          (PiRlcCanonicalSymbolicMachine.stateAt
            duplexBase initialBuilder coordinate.val)
          coordinate.val position) :
    column < transcriptEnd duplexBase count initialBuilder := by
  let state :=
    PiRlcCanonicalSymbolicMachine.stateAt
      duplexBase initialBuilder coordinate.val
  let entered :=
    PiRlcCanonicalSymbolicMachine.enterScalar
      duplexBase state coordinate.val
  let round := (PiRlcCanonicalU64.blockOf position).val
  have laneBound :=
    digestLanes_member_lt duplexBase
      (PiRlcCanonicalU64.beforeBlock
        duplexBase state coordinate.val position)
      (coordinate.val + round)
      (PiRlcCanonicalU64.laneOf position) column coefficient member
  have roundLt : round < digestRounds := by
    have := (PiRlcCanonicalU64.blockOf position).isLt
    simpa only [digestRounds] using this
  have blockToScalar :
      (PiRlcCanonicalSymbolicMachine.digestBlock duplexBase
          (PiRlcCanonicalU64.beforeBlock
            duplexBase state coordinate.val position)
          (coordinate.val + round)).entries.length ≤
        (PiRlcCanonicalSymbolicMachine.scalarBuilder
          duplexBase state coordinate.val).entries.length := by
    change
      (PiRlcCanonicalSymbolicMachine.stateBeforeBlock
          duplexBase entered coordinate.val (round + 1)).entries.length ≤
        (PiRlcCanonicalSymbolicMachine.stateBeforeBlock
          duplexBase entered coordinate.val
            digestRounds).entries.length
    exact stateBeforeBlock_entries_length_mono
      duplexBase entered coordinate.val (round + 1)
      digestRounds (by omega)
  have stateCursor :
      state.absorbed = if coordinate.val = 0 then 1 else 0 := by
    dsimp only [state]
    cases coordinate.val with
    | zero => simpa using cursorOne
    | succ previous =>
        simp only [PiRlcCanonicalSymbolicMachine.stateAt,
          PiRlcCanonicalSymbolicMachine.scalarBuilder_absorbed]
        simp
  have scalarLength :
      (PiRlcCanonicalSymbolicMachine.scalarBuilder
          duplexBase state coordinate.val).entries.length =
        initialBuilder.entries.length + (coordinate.val + 1) * 5 := by
    by_cases first : coordinate.val = 0
    · have stateOne : state.absorbed = 1 := by
        simpa [first] using stateCursor
      rw [PiRlcCanonicalSymbolicMachine.scalarBuilder_entries_length_of_one
        duplexBase state coordinate.val stateOne,
        PiRlcCanonicalSymbolicMachine.stateAt_entries_length_of_one
          duplexBase initialBuilder cursorOne coordinate.val]
      omega
    · have stateZero : state.absorbed = 0 := by
        simpa [first] using stateCursor
      rw [PiRlcCanonicalSymbolicMachine.scalarBuilder_entries_length_of_zero
        duplexBase state coordinate.val stateZero,
        PiRlcCanonicalSymbolicMachine.stateAt_entries_length_of_one
          duplexBase initialBuilder cursorOne coordinate.val]
      omega
  unfold transcriptEnd
  rw [scalarLength] at blockToScalar
  have coordinateBound : coordinate.val + 1 ≤ count := by omega
  have scalarToBatch :
      initialBuilder.entries.length + (coordinate.val + 1) * 5 ≤
        initialBuilder.entries.length + count * 5 := by
    omega
  have blockToBatch :
      (PiRlcCanonicalSymbolicMachine.digestBlock duplexBase
          (PiRlcCanonicalU64.beforeBlock
            duplexBase state coordinate.val position)
          (coordinate.val + round)).entries.length ≤
        initialBuilder.entries.length + count * 5 :=
    Nat.le_trans blockToScalar scalarToBatch
  exact Nat.lt_of_lt_of_le laneBound
    (Nat.add_le_add_left
      (Nat.mul_le_mul_right SymbolicDuplex.stride blockToBatch)
      duplexBase)

/-- Every canonical-u64 source read is one of the exact carried output
columns allocated by the symbolic transcript. -/
theorem laneInput_member_temporaryColumns
    (duplexBase count : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    (cursorOne : initialBuilder.absorbed = 1)
    (coordinate : Fin count)
    (position : Fin PiRlcCanonicalU64.lanesPerScalar)
    (column coefficient : Nat)
    (member :
      (column, coefficient) ∈
        PiRlcCanonicalU64.laneInput duplexBase
          (PiRlcCanonicalSymbolicMachine.stateAt
            duplexBase initialBuilder coordinate.val)
          coordinate.val position) :
    column ∈
      SymbolicDuplexPhysical.temporaryColumns duplexBase
        (initialBuilder.entries.length + count * 5) := by
  let state :=
    PiRlcCanonicalSymbolicMachine.stateAt
      duplexBase initialBuilder coordinate.val
  let entered :=
    PiRlcCanonicalSymbolicMachine.enterScalar
      duplexBase state coordinate.val
  let round := (PiRlcCanonicalU64.blockOf position).val
  have roundLt : round < digestRounds := by
    have bounded := (PiRlcCanonicalU64.blockOf position).isLt
    simpa only [digestRounds] using bounded
  have blockToScalar :
      (PiRlcCanonicalSymbolicMachine.digestBlock duplexBase
          (PiRlcCanonicalU64.beforeBlock
            duplexBase state coordinate.val position)
          (coordinate.val + round)).entries.length ≤
        (PiRlcCanonicalSymbolicMachine.scalarBuilder
          duplexBase state coordinate.val).entries.length := by
    change
      (PiRlcCanonicalSymbolicMachine.stateBeforeBlock
          duplexBase entered coordinate.val (round + 1)).entries.length ≤
        (PiRlcCanonicalSymbolicMachine.stateBeforeBlock
          duplexBase entered coordinate.val
            digestRounds).entries.length
    exact stateBeforeBlock_entries_length_mono
      duplexBase entered coordinate.val (round + 1)
      digestRounds (by omega)
  have stateCursor :
      state.absorbed = if coordinate.val = 0 then 1 else 0 := by
    dsimp only [state]
    cases coordinate.val with
    | zero => simpa using cursorOne
    | succ previous =>
        simp only [PiRlcCanonicalSymbolicMachine.stateAt,
          PiRlcCanonicalSymbolicMachine.scalarBuilder_absorbed]
        simp
  have scalarLength :
      (PiRlcCanonicalSymbolicMachine.scalarBuilder
          duplexBase state coordinate.val).entries.length =
        initialBuilder.entries.length + (coordinate.val + 1) * 5 := by
    by_cases first : coordinate.val = 0
    · have stateOne : state.absorbed = 1 := by
        simpa [first] using stateCursor
      rw [PiRlcCanonicalSymbolicMachine.scalarBuilder_entries_length_of_one
        duplexBase state coordinate.val stateOne,
        PiRlcCanonicalSymbolicMachine.stateAt_entries_length_of_one
          duplexBase initialBuilder cursorOne coordinate.val]
      omega
    · have stateZero : state.absorbed = 0 := by
        simpa [first] using stateCursor
      rw [PiRlcCanonicalSymbolicMachine.scalarBuilder_entries_length_of_zero
        duplexBase state coordinate.val stateZero,
        PiRlcCanonicalSymbolicMachine.stateAt_entries_length_of_one
          duplexBase initialBuilder cursorOne coordinate.val]
      omega
  rw [scalarLength] at blockToScalar
  have coordinateBound : coordinate.val + 1 ≤ count := by omega
  have blockToBatch :
      (PiRlcCanonicalSymbolicMachine.digestBlock duplexBase
          (PiRlcCanonicalU64.beforeBlock
            duplexBase state coordinate.val position)
          (coordinate.val + round)).entries.length ≤
        initialBuilder.entries.length + count * 5 := by
    exact Nat.le_trans blockToScalar (by omega)
  exact digestLanes_member_temporaryColumns
    duplexBase (initialBuilder.entries.length + count * 5)
    (PiRlcCanonicalU64.beforeBlock
      duplexBase state coordinate.val position)
    (coordinate.val + round) (PiRlcCanonicalU64.laneOf position)
    column coefficient member blockToBatch

/-- Exact physical separation constructs the u64 input-ownership premise.
The premise now contains no per-lane assertion supplied by a caller. -/
theorem inputsBelow_of_transcriptEnd
    (duplexBase u64Base count : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    (cursorOne : initialBuilder.absorbed = 1)
    (separated : transcriptEnd duplexBase count initialBuilder ≤ u64Base) :
    InputsBelow duplexBase u64Base count initialBuilder := by
  constructor
  intro coordinate position column coefficient member
  exact Nat.lt_of_lt_of_le
    (laneInput_member_lt_transcriptEnd
      duplexBase count initialBuilder cursorOne coordinate position
      column coefficient member)
    separated

end Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalU64Placement
