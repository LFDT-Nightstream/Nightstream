import Nightstream.Implementation.Nebula.FPrime.Claim.Envelope
import Nightstream.Implementation.R1CS.Core.EqualityPins

/-!
Contract: exact row link from one V2 full claim to one NIFS input block.

Assurance tier: implementation model.

Owns two separate full-envelope column windows, one equality row per bit,
typed placement, row soundness, exact section recovery, local completeness,
and an artifact-facing row-inclusion certificate.

Does not own absolute generated columns, final compiler widths, NIFS verifier
rows, proof parsing, or recursive-size closure.

Emits constraints: yes, through `EqualityPins.rows`.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Nebula.FullClaimEnvelopeRows

open Nightstream.Implementation.Nebula.FullClaimEnvelope
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.EqualityPins
open Nightstream.Implementation.R1CS.Program

/-- The source claim and NIFS input occupy two non-overlapping positive column
windows. This prevents a manifest from hiding a missing link through aliases. -/
structure Layout (widths : CompilerWidths) where
  claimBitStart : Nat
  nifsInputBitStart : Nat
  claimStartPositive : 0 < claimBitStart
  nifsInputStartPositive : 0 < nifsInputBitStart
  windowsSeparated :
    claimBitStart + widths.totalBits ≤ nifsInputBitStart ∨
      nifsInputBitStart + widths.totalBits ≤ claimBitStart
deriving Repr

def Layout.pairAt {widths : CompilerWidths}
    (layout : Layout widths) (index : Nat) : Nat × Nat :=
  (layout.nifsInputBitStart + index, layout.claimBitStart + index)

def Layout.pairs {widths : CompilerWidths}
    (layout : Layout widths) : List (Nat × Nat) :=
  (List.range widths.totalBits).map layout.pairAt

def rows {widths : CompilerWidths} (layout : Layout widths) : List Row :=
  EqualityPins.rows layout.pairs

theorem Layout.pairs_length {widths : CompilerWidths}
    (layout : Layout widths) : layout.pairs.length = widths.totalBits := by
  simp [Layout.pairs]

theorem rows_length {widths : CompilerWidths} (layout : Layout widths) :
    (rows layout).length = widths.totalBits := by
  simp [rows, EqualityPins.rows, layout.pairs_length]

theorem Layout.windows_do_not_share_index {widths : CompilerWidths}
    (layout : Layout widths) (left right : Fin widths.totalBits) :
    layout.claimBitStart + left.val ≠ layout.nifsInputBitStart + right.val := by
  rcases layout.windowsSeparated with before | after
  · have leftUpper : layout.claimBitStart + left.val <
        layout.claimBitStart + widths.totalBits := by omega
    have rightLower : layout.nifsInputBitStart ≤
        layout.nifsInputBitStart + right.val := Nat.le_add_right _ _
    omega
  · have rightUpper : layout.nifsInputBitStart + right.val <
        layout.nifsInputBitStart + widths.totalBits := by omega
    have leftLower : layout.claimBitStart ≤
        layout.claimBitStart + left.val := Nat.le_add_right _ _
    omega

def envelopeBit {widths : CompilerWidths} (value : Value widths)
    (index : Fin widths.totalBits) : Nat :=
  value.encode.get ⟨index.val, by simpa [value.encode_length] using index.isLt⟩

def inputBit {widths : CompilerWidths}
    (input : FixedBits.Word widths.totalBits)
    (index : Fin widths.totalBits) : Nat :=
  input.val.get ⟨index.val, by simpa [input.property.1] using index.isLt⟩

def Placed {widths : CompilerWidths} (layout : Layout widths)
    (assignment : Nat → Nat) (value : Value widths)
    (input : FixedBits.Word widths.totalBits) : Prop :=
  ∀ index : Fin widths.totalBits,
    assignment (layout.claimBitStart + index.val) = envelopeBit value index ∧
      assignment (layout.nifsInputBitStart + index.val) = inputBit input index

def RowsHold {widths : CompilerWidths} (layout : Layout widths)
    (assignment : Nat → Nat) : Prop :=
  Satisfies (rows layout) assignment

private theorem pairAt_mem {widths : CompilerWidths}
    (layout : Layout widths) (index : Fin widths.totalBits) :
    layout.pairAt index.val ∈ layout.pairs := by
  exact List.mem_map.mpr
    ⟨index.val, List.mem_range.mpr index.isLt, rfl⟩

theorem bits_equal_of_rows
    {widths : CompilerWidths} {layout : Layout widths}
    {assignment : Nat → Nat} {value : Value widths}
    {input : FixedBits.Word widths.totalBits}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (placed : Placed layout assignment value input)
    (holds : RowsHold layout assignment) :
    ∀ index, inputBit input index = envelopeBit value index := by
  intro index
  have columnEqual := EqualityPins.rows_sound canonical one holds
    (layout.pairAt index.val) (pairAt_mem layout index)
  simp only [Layout.pairAt] at columnEqual
  rw [(placed index).2, (placed index).1] at columnEqual
  exact columnEqual

/-- Satisfying all link rows makes the selected NIFS input equal the complete
typed claim block. -/
theorem input_eq_block
    {widths : CompilerWidths} {layout : Layout widths}
    {assignment : Nat → Nat} {value : Value widths}
    {input : FixedBits.Word widths.totalBits}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (placed : Placed layout assignment value input)
    (holds : RowsHold layout assignment) : input = value.block := by
  apply Subtype.ext
  apply List.ext_get
  · rw [input.property.1, value.block.property.1]
  · intro index inputBound encodeBound
    let bounded : Fin widths.totalBits :=
      ⟨index, by simpa [input.property.1] using inputBound⟩
    simpa [inputBit, envelopeBit, Value.block, bounded] using
      bits_equal_of_rows canonical one placed holds bounded

/-- Every selected-verifier section is therefore the exact typed section of
the same claim. -/
theorem input_section_exact
    {widths : CompilerWidths} {layout : Layout widths}
    {assignment : Nat → Nat} {value : Value widths}
    {input : FixedBits.Word widths.totalBits}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (placed : Placed layout assignment value input)
    (holds : RowsHold layout assignment) (part : Section) :
    (input.val.drop (part.bitOffset widths)).take (part.width widths) =
      value.sectionBits part := by
  have inputExact := input_eq_block canonical one placed holds
  rw [inputExact]
  exact value.encode_slice part

/-- Honest equal placement satisfies the complete link block. -/
theorem rows_complete
    {widths : CompilerWidths} {layout : Layout widths}
    {assignment : Nat → Nat} {value : Value widths}
    {input : FixedBits.Word widths.totalBits}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (placed : Placed layout assignment value input)
    (equal : input = value.block) : RowsHold layout assignment := by
  apply EqualityPins.rows_complete canonical one
  intro pair member
  rcases List.mem_map.mp member with ⟨index, indexMember, pairEqual⟩
  subst pair
  let bounded : Fin widths.totalBits :=
    ⟨index, List.mem_range.mp indexMember⟩
  calc
    assignment (layout.nifsInputBitStart + index) = inputBit input bounded :=
      (placed bounded).2
    _ = envelopeBit value bounded := by
      rw [equal]
      simp [inputBit, envelopeBit, Value.block]
    _ = assignment (layout.claimBitStart + index) := (placed bounded).1.symm

/-- Artifact-facing certificate. An actual generated V2 relation must supply
the concrete layout and prove inclusion in its emitted rows. -/
structure CallSite {widths : CompilerWidths}
    (programRows : List Row) (assignment : Nat → Nat)
    (value : Value widths) (input : FixedBits.Word widths.totalBits) where
  layout : Layout widths
  rowsIncluded :
    rowsIncluded (rows layout) programRows = true
  canonicalAssignment : ∀ column, assignment column < goldilocksP
  one : assignment 0 = 1
  placed : Placed layout assignment value input

theorem CallSite.sound
    {widths : CompilerWidths} {programRows : List Row}
    {assignment : Nat → Nat} {value : Value widths}
    {input : FixedBits.Word widths.totalBits}
    (site : CallSite programRows assignment value input)
    (satisfies : Satisfies programRows assignment) : input = value.block := by
  apply Subtype.ext
  apply List.ext_get
  · rw [input.property.1, value.block.property.1]
  · intro index inputBound encodeBound
    let bounded : Fin widths.totalBits :=
      ⟨index, by simpa [input.property.1] using inputBound⟩
    have columnEqual := EqualityPins.sound site.rowsIncluded
      site.canonicalAssignment site.one satisfies
      (site.layout.pairAt bounded.val) (pairAt_mem site.layout bounded)
    simp only [Layout.pairAt] at columnEqual
    rw [(site.placed bounded).2, (site.placed bounded).1] at columnEqual
    simpa [inputBit, envelopeBit, Value.block, bounded] using columnEqual

end Nightstream.Implementation.Nebula.FullClaimEnvelopeRows
