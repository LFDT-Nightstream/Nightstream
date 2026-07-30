import Nightstream.Implementation.R1CS.Canonical.SymbolicDuplexHonest

/-!
Contract: structural placement closure for the symbolic duplex builder.

Owns: constructors and preservation lemmas for
`SymbolicDuplexHonest.WellPlaced` across `start`, `permute`, guarded overwrite,
list absorption, and the pre-squeeze gate.

Does not own: any protocol serialization.  A caller proves its absorbed
expressions lie in the authoritative prefix and then composes these lemmas.

Assurance tier: model-level canonical encoding.
-/

set_option autoImplicit false
set_option maxRecDepth 100000

namespace Nightstream.Implementation.R1CS.Canonical.SymbolicDuplexPlacement

open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Core
open Nightstream.Implementation.R1CS.Canonical.SymbolicDuplexHonest

/-- A row-free absorbed expression reads only the caller-owned prefix before
the first call's output space. -/
def ValueBefore (base : Nat) (value : LinCombNormal.LinComb) : Prop :=
  ∀ column, Mentions value column → column < outputBase base 0

theorem start_wellPlaced
    (base : Nat) (lanes : State) (absorbed : Nat)
    (lanesBefore :
      ∀ lane : Fin width, ∀ column,
        Mentions (lanes lane) column → column < outputBase base 0) :
    WellPlaced base (SymbolicDuplex.start lanes absorbed) where
  calls := rfl
  entrySources := by
    intro entry member
    cases member
  lanesBefore := lanesBefore

theorem empty_wellPlaced (base : Nat) :
    WellPlaced base SymbolicDuplex.empty :=
  start_wellPlaced base (fun _ => []) 0 (by
    intro lane column mentioned
    cases mentioned)

theorem permute_wellPlaced
    (base : Nat) (builder : SymbolicDuplex.Builder)
    (placed : WellPlaced base builder) :
    WellPlaced base (SymbolicDuplex.permute base builder) where
  calls := by
    simp only [SymbolicDuplex.permute, List.map_append, List.map_cons,
      List.map_nil, List.length_append, List.length_cons, List.length_nil,
      placed.calls]
    rw [List.range_succ]
  entrySources := by
    intro entry member
    simp only [SymbolicDuplex.permute, List.mem_append, List.mem_cons,
      List.not_mem_nil, or_false] at member
    rcases member with old | rfl
    · exact placed.entrySources entry old
    · exact placed.lanesBefore
  lanesBefore := by
    intro lane column mentioned
    simp only [SymbolicDuplex.permute, SymbolicDuplex.outputState,
      Mentions, List.map_cons, List.map_nil, List.mem_singleton] at mentioned
    rw [mentioned]
    simp only [SymbolicDuplex.permute, List.length_append, List.length_cons,
      List.length_nil]
    have laneLt := lane.isLt
    simp only [width] at laneLt
    simp only [SymbolicDuplex.layoutAt, outputBase, callBase,
      SymbolicDuplex.stride, width]
    omega

theorem guarded_wellPlaced
    (base : Nat) (builder : SymbolicDuplex.Builder)
    (placed : WellPlaced base builder) :
    WellPlaced base (SymbolicDuplex.guarded base builder) := by
  unfold SymbolicDuplex.guarded
  split
  · exact permute_wellPlaced base builder placed
  · exact placed

theorem absorb_wellPlaced
    (base : Nat) (value : LinCombNormal.LinComb)
    (builder : SymbolicDuplex.Builder)
    (placed : WellPlaced base builder)
    (valueBefore : ValueBefore base value) :
    WellPlaced base (SymbolicDuplex.absorb base value builder) := by
  let ready := SymbolicDuplex.guarded base builder
  have readyPlaced : WellPlaced base ready :=
    guarded_wellPlaced base builder placed
  refine
    { calls := readyPlaced.calls
      entrySources := readyPlaced.entrySources
      lanesBefore := ?_ }
  intro lane column mentioned
  change
    Mentions
        ((if lane.val = ready.absorbed then value else ready.lanes lane))
        column at mentioned
  split at mentioned
  · exact Nat.lt_of_lt_of_le
      (valueBefore column mentioned)
      (by
        simp only [outputBase, callBase, SymbolicDuplex.stride,
          Poseidon2Layout.canonicalColumnTotal, width, sboxCount,
          externalRounds, partialRounds, Poseidon2Program.columnsPerSbox]
        omega)
  · exact readyPlaced.lanesBefore lane column mentioned

theorem absorbMany_wellPlaced
    (base : Nat) :
    ∀ (values : List LinCombNormal.LinComb)
      (builder : SymbolicDuplex.Builder),
      WellPlaced base builder →
      (∀ value ∈ values, ValueBefore base value) →
      WellPlaced base (SymbolicDuplex.absorbMany base values builder)
  | [], _, placed, _ => placed
  | value :: rest, builder, placed, valuesBefore =>
      absorbMany_wellPlaced base rest
        (SymbolicDuplex.absorb base value builder)
        (absorb_wellPlaced base value builder placed
          (valuesBefore value (by simp)))
        (fun other member => valuesBefore other (by simp [member]))

theorem one_before (base : Nat) (positive : 0 < base) :
    ValueBefore base SymbolicDuplex.one := by
  intro column mentioned
  have same : column = 0 := by
    simpa [SymbolicDuplex.one, Mentions] using mentioned
  rw [same]
  simp only [outputBase, callBase]
  omega

theorem gate_wellPlaced
    (base : Nat) (builder : SymbolicDuplex.Builder)
    (positive : 0 < base)
    (placed : WellPlaced base builder) :
    WellPlaced base (SymbolicDuplex.gate base builder) := by
  exact permute_wellPlaced base
    (SymbolicDuplex.absorb base SymbolicDuplex.one builder)
    (absorb_wellPlaced base SymbolicDuplex.one builder placed
      (one_before base positive))

theorem squeezeK_wellPlaced
    (base : Nat) (builder : SymbolicDuplex.Builder)
    (positive : 0 < base)
    (placed : WellPlaced base builder) :
    WellPlaced base (SymbolicDuplex.squeezeK base builder).2 :=
  gate_wellPlaced base builder positive placed

/-! ## Exact source ownership

`WellPlaced` is sufficient for witness sequencing, but deliberately says only
that a source precedes the call that reads it.  Physical conservation needs
the stronger fact below: a source is either in the caller-owned prefix or is
an output of a strictly earlier local call. -/

def SourceOwned (base call column : Nat) : Prop :=
  column < base ∨
    ∃ previous : Nat, ∃ lane : Fin width,
      previous < call ∧
        column = (SymbolicDuplex.layoutAt base previous).outputPort lane

def ValueInPrefix (base : Nat) (value : LinCombNormal.LinComb) : Prop :=
  ∀ column, Mentions value column → column < base

structure WellOwned (base : Nat) (builder : SymbolicDuplex.Builder) : Prop where
  entrySources :
    ∀ entry ∈ builder.entries, ∀ lane : Fin width, ∀ column,
      Mentions (entry.state lane) column →
        SourceOwned base entry.call column
  lanes :
    ∀ lane : Fin width, ∀ column,
      Mentions (builder.lanes lane) column →
        SourceOwned base builder.entries.length column

theorem start_wellOwned
    (base : Nat) (lanes : State) (absorbed : Nat)
    (lanesInPrefix :
      ∀ lane : Fin width, ValueInPrefix base (lanes lane)) :
    WellOwned base (SymbolicDuplex.start lanes absorbed) where
  entrySources := by
    intro entry member
    cases member
  lanes := by
    intro lane column mentioned
    exact Or.inl (lanesInPrefix lane column mentioned)

theorem empty_wellOwned (base : Nat) :
    WellOwned base SymbolicDuplex.empty :=
  start_wellOwned base (fun _ => []) 0 (by
    intro lane column mentioned
    cases mentioned)

theorem permute_wellOwned
    (base : Nat) (builder : SymbolicDuplex.Builder)
    (owned : WellOwned base builder) :
    WellOwned base (SymbolicDuplex.permute base builder) where
  entrySources := by
    intro entry member
    simp only [SymbolicDuplex.permute, List.mem_append, List.mem_cons,
      List.not_mem_nil, or_false] at member
    rcases member with old | rfl
    · exact owned.entrySources entry old
    · exact owned.lanes
  lanes := by
    intro lane column mentioned
    simp only [SymbolicDuplex.permute, SymbolicDuplex.outputState,
      Mentions, List.map_cons, List.map_nil, List.mem_singleton] at mentioned
    exact Or.inr
      ⟨builder.entries.length, lane, by
        simp only [SymbolicDuplex.permute, List.length_append,
          List.length_cons, List.length_nil]
        omega,
        mentioned⟩

theorem guarded_wellOwned
    (base : Nat) (builder : SymbolicDuplex.Builder)
    (owned : WellOwned base builder) :
    WellOwned base (SymbolicDuplex.guarded base builder) := by
  unfold SymbolicDuplex.guarded
  split
  · exact permute_wellOwned base builder owned
  · exact owned

theorem absorb_wellOwned
    (base : Nat) (value : LinCombNormal.LinComb)
    (builder : SymbolicDuplex.Builder)
    (owned : WellOwned base builder)
    (valueInPrefix : ValueInPrefix base value) :
    WellOwned base (SymbolicDuplex.absorb base value builder) := by
  let ready := SymbolicDuplex.guarded base builder
  have readyOwned : WellOwned base ready :=
    guarded_wellOwned base builder owned
  refine
    { entrySources := readyOwned.entrySources
      lanes := ?_ }
  intro lane column mentioned
  change
    Mentions
        (if lane.val = ready.absorbed then value else ready.lanes lane)
        column at mentioned
  split at mentioned
  · exact Or.inl (valueInPrefix column mentioned)
  · exact readyOwned.lanes lane column mentioned

theorem absorbMany_wellOwned
    (base : Nat) :
    ∀ (values : List LinCombNormal.LinComb)
      (builder : SymbolicDuplex.Builder),
      WellOwned base builder →
      (∀ value ∈ values, ValueInPrefix base value) →
      WellOwned base (SymbolicDuplex.absorbMany base values builder)
  | [], _, owned, _ => owned
  | value :: rest, builder, owned, valuesInPrefix =>
      absorbMany_wellOwned base rest
        (SymbolicDuplex.absorb base value builder)
        (absorb_wellOwned base value builder owned
          (valuesInPrefix value (by simp)))
        (fun other member => valuesInPrefix other (by simp [member]))

theorem one_inPrefix (base : Nat) (positive : 0 < base) :
    ValueInPrefix base SymbolicDuplex.one := by
  intro column mentioned
  have same : column = 0 := by
    simpa [SymbolicDuplex.one, Mentions] using mentioned
  omega

theorem gate_wellOwned
    (base : Nat) (builder : SymbolicDuplex.Builder)
    (positive : 0 < base)
    (owned : WellOwned base builder) :
    WellOwned base (SymbolicDuplex.gate base builder) :=
  permute_wellOwned base
    (SymbolicDuplex.absorb base SymbolicDuplex.one builder)
    (absorb_wellOwned base SymbolicDuplex.one builder owned
      (one_inPrefix base positive))

theorem squeezeK_wellOwned
    (base : Nat) (builder : SymbolicDuplex.Builder)
    (positive : 0 < base)
    (owned : WellOwned base builder) :
    WellOwned base (SymbolicDuplex.squeezeK base builder).2 :=
  gate_wellOwned base builder positive owned

end Nightstream.Implementation.R1CS.Canonical.SymbolicDuplexPlacement
