import Nightstream.Implementation.R1CS.Canonical.Poseidon2Schedule

/-!
Contract: layout well-formedness for the canonical Poseidon2 permutation.

Owns: the predicate distinguishing a usable column layout from an incoherent
one, a concrete layout satisfying it, and the disjointness facts that
`Poseidon2Program`'s ownership theorems could not state.

Does not own: the schedule, the row program, or any assignment.

## The gap this closes

`Poseidon2Program.everyPermutationColumn_has_exact_owner` proves the 344
auxiliary columns are mutually distinct.  It says nothing about `inputPort` or
`outputPort`, because `Layout` is unconstrained — nothing forbade
`inputPort 0 = auxBase + 5`.

That is not a cosmetic gap.  Two things depend on it:

  * the cost claim.  `⟨352, 0, 0, 344⟩` counts 344 *auxiliary* columns; if a
    port aliased one, the same column would be counted as both a port and a
    temporary and the auxiliary total would be wrong.
  * honest completeness.  An honest witness assigns input values to input
    ports, chain values to auxiliaries and reference values to output ports.
    If those ranges overlap, no such function exists for arbitrary inputs, so
    the completeness statement cannot even be formed.

Well-formedness is therefore a precondition of completeness, not an independent
tidiness property.  `WellFormed` is a *predicate on an existing structure*
rather than a new premise threaded through the encoding: `canonicalLayout`
constructs it, so nothing here moves an obligation to a hypothesis no consumer
discharges.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Canonical.Poseidon2Layout

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Core
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Program

/-! ## Well-formedness -/

/-- A layout is well formed when its declared ports are genuinely separate
columns: never the constant wire, never inside the auxiliary block, never
aliasing each other. -/
structure WellFormed (layout : Layout) : Prop where
  /-- Column 0 is the shared constant wire; no port may own it. -/
  inputNotConstantWire : ∀ lane : Fin width, layout.inputPort lane ≠ 0
  outputNotConstantWire : ∀ lane : Fin width, layout.outputPort lane ≠ 0
  /-- Ports sit below the auxiliary block, so they cannot alias a temporary. -/
  inputBelowAux : ∀ lane : Fin width, layout.inputPort lane < layout.auxBase
  outputBelowAux : ∀ lane : Fin width, layout.outputPort lane < layout.auxBase
  /-- Distinct lanes get distinct columns, or one column would have to carry
  two different values. -/
  inputInjective : ∀ a b : Fin width, layout.inputPort a = layout.inputPort b → a = b
  outputInjective : ∀ a b : Fin width, layout.outputPort a = layout.outputPort b → a = b
  /-- An input port is never an output port. -/
  portsDisjoint : ∀ a b : Fin width, layout.inputPort a ≠ layout.outputPort b

/-! ## Consequences for the auxiliary block -/

/-- Every auxiliary column lies at or above the block base. -/
theorem auxiliaryColumns_ge (layout : Layout) (column : Nat)
    (member : column ∈ auxiliaryColumns layout) :
    layout.auxBase ≤ column := by
  rcases List.mem_flatMap.1 member with ⟨index, _, slotMember⟩
  rcases List.mem_map.1 slotMember with ⟨slot, _, rfl⟩
  simp only [sboxColumn]
  omega

/-- **No input port is a temporary.** -/
theorem inputPort_not_auxiliary
    (layout : Layout) (wellFormed : WellFormed layout) (lane : Fin width) :
    layout.inputPort lane ∉ auxiliaryColumns layout := by
  intro member
  have atLeast := auxiliaryColumns_ge layout _ member
  have below := wellFormed.inputBelowAux lane
  omega

/-- **No output port is a temporary.** -/
theorem outputPort_not_auxiliary
    (layout : Layout) (wellFormed : WellFormed layout) (lane : Fin width) :
    layout.outputPort lane ∉ auxiliaryColumns layout := by
  intro member
  have atLeast := auxiliaryColumns_ge layout _ member
  have below := wellFormed.outputBelowAux lane
  omega

/-- **The constant wire is not a temporary**, provided the block starts above
it.  Together with the two lemmas above this is what makes `⟨352, 0, 0, 344⟩`
an auxiliary count rather than a count of columns that happen to be mentioned. -/
theorem constantWire_not_auxiliary
    (layout : Layout) (basePositive : 0 < layout.auxBase) :
    0 ∉ auxiliaryColumns layout := by
  intro member
  have atLeast := auxiliaryColumns_ge layout _ member
  omega

/-! ## A concrete well-formed layout

Columns are laid out in the obvious order: the constant wire, then the eight
inputs, then the eight outputs, then the auxiliary block.  Concrete rather than
merely asserted to exist, so the completeness witness can be defined by
arithmetic on the column index and needs no choice principle. -/

def canonicalLayout : Layout where
  auxBase := 17
  inputPort := fun lane => 1 + lane.val
  outputPort := fun lane => 9 + lane.val

theorem canonicalLayout_wellFormed : WellFormed canonicalLayout where
  inputNotConstantWire := by decide
  outputNotConstantWire := by decide
  inputBelowAux := by decide
  outputBelowAux := by decide
  inputInjective := by decide
  outputInjective := by decide
  portsDisjoint := by decide

/-- Total columns the canonical layout occupies: the constant wire, eight
inputs, eight outputs, and the 344-column auxiliary block. -/
def canonicalColumnTotal : Nat :=
  1 + width + width + sboxCount * columnsPerSbox

theorem canonicalColumnTotal_eq : canonicalColumnTotal = 361 := by decide

/-- The auxiliary block of the canonical layout starts exactly where the ports
end, so the column space has no hole. -/
theorem canonicalLayout_contiguous :
    canonicalLayout.auxBase = 1 + width + width := by decide


/-! ## Row ownership uniqueness

`Poseidon2Program.everyPermutationRow_has_owner` proves coverage only: its
conclusion is a disjunction of existentials with nothing forbidding two
receipts from emitting the same row value.  These close that gap
(`POSEIDON2-ROW-OWNERSHIP-UNIQUENESS`).

The argument runs through the `c` operand.  Every emitted row writes to exactly
one column, that column identifies the receipt, and well-formedness keeps the
three families of write targets apart. -/

open Nightstream.Implementation.R1CS.Canonical.Poseidon2Schedule in
/-- Every S-box row writes to one of its own four frame columns. -/
theorem sboxRows_target
    (layout : Layout) (index : Fin sboxCount)
    (comb : Poseidon2Core.LinComb) (row : Row)
    (member : row ∈ sboxRows (frameAt layout index comb)) :
    ∃ slot : Fin columnsPerSbox, row.c = [(sboxColumn layout index slot, 1)] := by
  simp only [sboxRows, List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl | rfl | rfl
  · exact ⟨⟨0, by decide⟩, rfl⟩
  · exact ⟨⟨1, by decide⟩, rfl⟩
  · exact ⟨⟨2, by decide⟩, rfl⟩
  · exact ⟨⟨3, by decide⟩, rfl⟩

open Nightstream.Implementation.R1CS.Canonical.Poseidon2Schedule in
/-- **Distinct S-boxes emit disjoint rows.** -/
theorem sboxRows_disjoint
    (layout : Layout) (first second : Fin sboxCount)
    (firstComb secondComb : Poseidon2Core.LinComb) (row : Row)
    (distinct : first ≠ second)
    (inFirst : row ∈ sboxRows (frameAt layout first firstComb))
    (inSecond : row ∈ sboxRows (frameAt layout second secondComb)) : False := by
  rcases sboxRows_target layout first firstComb row inFirst with ⟨slotA, targetA⟩
  rcases sboxRows_target layout second secondComb row inSecond with ⟨slotB, targetB⟩
  rw [targetA] at targetB
  have columns : sboxColumn layout first slotA = sboxColumn layout second slotB := by
    simpa using targetB
  exact distinct (sboxColumn_injective layout columns).1

open Nightstream.Implementation.R1CS.Canonical.Poseidon2Schedule in
/-- **A terminal binding row is never an S-box row.**  Their write targets lie
on opposite sides of the auxiliary boundary. -/
theorem bindRow_not_sboxRow
    (layout : Layout) (wellFormed : WellFormed layout)
    (index : Fin sboxCount) (comb : Poseidon2Core.LinComb)
    (state : State) (lane : Fin width) :
    bindRow (state lane) (layout.outputPort lane)
      ∉ sboxRows (frameAt layout index comb) := by
  intro member
  rcases sboxRows_target layout index comb _ member with ⟨slot, target⟩
  have columns : layout.outputPort lane = sboxColumn layout index slot := by
    simpa [bindRow] using target
  have atLeast : layout.auxBase ≤ layout.outputPort lane := by
    rw [columns]; simp only [sboxColumn]; omega
  have below := wellFormed.outputBelowAux lane
  omega

/-- **Distinct lanes emit distinct binding rows.** -/
theorem bindRow_injective
    (layout : Layout) (wellFormed : WellFormed layout) (state : State)
    (first second : Fin width)
    (equal : bindRow (state first) (layout.outputPort first)
      = bindRow (state second) (layout.outputPort second)) :
    first = second := by
  have columns : layout.outputPort first = layout.outputPort second := by
    have := congrArg Row.c equal
    simpa [bindRow] using this
  exact wellFormed.outputInjective first second columns


/-! ## Shifted layouts

A sponge runs several permutation calls, each needing its own column space.
`shiftedLayout` is `canonicalLayout` translated by a base offset, so distinct
bases give disjoint allocations.  This is what a `SpongeLayout` well-formedness
predicate is built from: per-call well-formedness plus a stride large enough
that the ranges cannot overlap. -/

def shiftedLayout (base : Nat) : Layout where
  auxBase := base + 17
  inputPort := fun lane => base + 1 + lane.val
  outputPort := fun lane => base + 9 + lane.val

theorem shiftedLayout_wellFormed (base : Nat) : WellFormed (shiftedLayout base) where
  inputNotConstantWire := by
    intro lane; show base + 1 + lane.val ≠ 0; omega
  outputNotConstantWire := by
    intro lane; show base + 9 + lane.val ≠ 0; omega
  inputBelowAux := by
    intro lane
    have := lane.isLt; simp only [width] at this
    show base + 1 + lane.val < base + 17; omega
  outputBelowAux := by
    intro lane
    have := lane.isLt; simp only [width] at this
    show base + 9 + lane.val < base + 17; omega
  inputInjective := by
    intro a b equal
    exact Fin.ext (by
      have : base + 1 + a.val = base + 1 + b.val := equal
      omega)
  outputInjective := by
    intro a b equal
    exact Fin.ext (by
      have : base + 9 + a.val = base + 9 + b.val := equal
      omega)
  portsDisjoint := by
    intro a b
    have := a.isLt; have := b.isLt
    simp only [width] at *
    show base + 1 + a.val ≠ base + 9 + b.val
    omega

/-- `canonicalLayout` is the shifted layout at base zero, so every fact proved
about one transfers to the other. -/
theorem canonicalLayout_eq_shifted : canonicalLayout = shiftedLayout 0 := by
  unfold canonicalLayout shiftedLayout
  congr 1 <;> funext lane <;> omega

/-- **Distinct bases give disjoint auxiliary blocks**, provided the stride
clears the column total.  This is the disjointness a sponge needs: without it
two calls could allocate the same column and the per-call costs would not
add. -/
theorem shiftedLayout_aux_disjoint
    (first second : Nat) (stride : Nat)
    (strideClears : canonicalColumnTotal ≤ stride)
    (distinct : first ≠ second)
    (index : Fin sboxCount) (slot : Fin columnsPerSbox)
    (other : Fin sboxCount) (otherSlot : Fin columnsPerSbox) :
    sboxColumn (shiftedLayout (first * stride)) index slot
      ≠ sboxColumn (shiftedLayout (second * stride)) other otherSlot := by
  have indexLt := index.isLt
  have slotLt := slot.isLt
  have otherLt := other.isLt
  have otherSlotLt := otherSlot.isLt
  simp only [sboxCount, externalRounds, width, partialRounds,
    columnsPerSbox] at *
  simp only [canonicalColumnTotal, width, sboxCount, externalRounds,
    partialRounds, columnsPerSbox] at strideClears
  show (first * stride) + 17 + 4 * index.val + slot.val
    ≠ (second * stride) + 17 + 4 * other.val + otherSlot.val
  rcases Nat.lt_or_ge first second with below | above
  · have : first * stride + stride ≤ second * stride := by
      calc first * stride + stride = (first + 1) * stride := (Nat.succ_mul first stride).symm
        _ ≤ second * stride := Nat.mul_le_mul_right stride below
    omega
  · have : second ≠ first := fun equal => distinct equal.symm
    have secondBelow : second < first := by omega
    have : second * stride + stride ≤ first * stride := by
      calc second * stride + stride = (second + 1) * stride := (Nat.succ_mul second stride).symm
        _ ≤ first * stride := Nat.mul_le_mul_right stride secondBelow
    omega

end Nightstream.Implementation.R1CS.Canonical.Poseidon2Layout
