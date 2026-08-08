import Nightstream.Implementation.R1CS.Canonical.CanonicalU64Recipe
import Nightstream.Implementation.Lowering.Typed.Cost
import Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler.ProductionAlphabet

/-!
Contract: Lean-owned physical classification of one 16-bit `Pi_RLC`
rejection-sampling candidate.

The raw source bits and prior accepted count are caller-owned reads.  Their
bitwise complement is the candidate.  This recipe allocates and constrains:

* the exact accept bit for the unique rejected value `65535`;
* the quotient/remainder decomposition `chunk = 5 * quotient + residue`;
* the five-element residue range;
* fourteen Boolean quotient bits; and
* the successor accepted-prefix count.

The residue itself is the verifier's symbol in `Fin 5`; a redundant centered
symbol column is intentionally not allocated.  Centering belongs to the later
ring-assembly boundary.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalCandidate

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler.ProductionAlphabet

def sourceBitCount : Nat := 16
def quotientBitCount : Nat := 14
def auxiliaryCount : Nat := 22

structure Layout where
  base : Nat
  sourceBit : Fin sourceBitCount → Nat
  prior : LinComb

def acceptColumn (layout : Layout) : Nat := layout.base
def inverseColumn (layout : Layout) : Nat := layout.base + 1
def residueColumn (layout : Layout) : Nat := layout.base + 2
def quotientColumn (layout : Layout) : Nat := layout.base + 3
def productColumn (layout : Layout) (stage : Nat) : Nat :=
  layout.base + 4 + stage
def quotientBitColumn (layout : Layout) (offset : Nat) : Nat :=
  layout.base + 7 + offset
def cumulativeColumn (layout : Layout) : Nat := layout.base + 21

def allocation (layout : Layout) : List Nat :=
  (List.range auxiliaryCount).map (fun offset => layout.base + offset)

theorem allocation_length (layout : Layout) :
    (allocation layout).length = auxiliaryCount := by
  simp [allocation]

theorem allocation_nodup (layout : Layout) :
    (allocation layout).Nodup := by
  unfold allocation
  exact nodup_map _ _ (fun _ _ equal => by omega) List.nodup_range

theorem allocation_mem_iff (layout : Layout) (column : Nat) :
    column ∈ allocation layout ↔
      layout.base ≤ column ∧ column < layout.base + auxiliaryCount := by
  unfold allocation
  constructor
  · intro member
    rcases List.mem_map.mp member with ⟨offset, inRange, rfl⟩
    have bounded := List.mem_range.mp inRange
    omega
  · intro ⟨lower, upper⟩
    exact List.mem_map.mpr
      ⟨column - layout.base, List.mem_range.mpr (by omega), by omega⟩

theorem allocation_nonzero
    (layout : Layout) (positive : 0 < layout.base)
    (column : Nat) (member : column ∈ allocation layout) :
    column ≠ 0 := by
  have window := (allocation_mem_iff layout column).mp member
  omega

def chunkTerms (layout : Layout) : LinComb :=
  [(0, rejectionBucket)] ++
    (List.finRange sourceBitCount).map fun offset =>
      (layout.sourceBit offset, goldilocksP - 2 ^ offset.val)

def quotientTerms (layout : Layout) : LinComb :=
  (List.range quotientBitCount).map fun offset =>
    (quotientBitColumn layout offset, 2 ^ offset)

def differenceTerms (layout : Layout) : LinComb :=
  chunkTerms layout ++ [(0, goldilocksP - rejectionBucket)]

def oneMinusAccept (layout : Layout) : LinComb :=
  [(acceptColumn layout, goldilocksP - 1), (0, 1)]

def acceptanceRows (layout : Layout) : List Row :=
  [ bitRow (acceptColumn layout),
    ⟨oneMinusAccept layout, differenceTerms layout, []⟩,
    ⟨differenceTerms layout, [(inverseColumn layout, 1)],
      [(acceptColumn layout, 1)]⟩,
    ⟨oneMinusAccept layout, [(inverseColumn layout, 1)], []⟩ ]

def residueRangeRows (layout : Layout) : List Row :=
  [ ⟨[(residueColumn layout, 1)],
      [(residueColumn layout, 1), (0, goldilocksP - 1)],
      [(productColumn layout 0, 1)]⟩,
    ⟨[(productColumn layout 0, 1)],
      [(residueColumn layout, 1), (0, goldilocksP - 2)],
      [(productColumn layout 1, 1)]⟩,
    ⟨[(productColumn layout 1, 1)],
      [(residueColumn layout, 1), (0, goldilocksP - 3)],
      [(productColumn layout 2, 1)]⟩,
    ⟨[(productColumn layout 2, 1)],
      [(residueColumn layout, 1), (0, goldilocksP - 4)], []⟩ ]

def quotientBitRows (layout : Layout) : List Row :=
  (List.range quotientBitCount).map fun offset =>
    bitRow (quotientBitColumn layout offset)

/-- Positive equality `quotient = Σ 2^i bit_i`. -/
def quotientRecompositionRow (layout : Layout) : Row :=
  ⟨[(quotientColumn layout, 1)], [(0, 1)], quotientTerms layout⟩

/-- Positive equality `chunk = 5 * quotient + residue`. -/
def decompositionRow (layout : Layout) : Row :=
  ⟨chunkTerms layout, [(0, 1)],
    [(quotientColumn layout, 5), (residueColumn layout, 1)]⟩

/-- Positive equality `next = prior + accept`. -/
def cumulativeRow (layout : Layout) : Row :=
  ⟨[(cumulativeColumn layout, 1)], [(0, 1)],
    layout.prior ++ [(acceptColumn layout, 1)]⟩

def rows (layout : Layout) : List Row :=
  acceptanceRows layout ++ residueRangeRows layout ++
    quotientBitRows layout ++
    [quotientRecompositionRow layout, decompositionRow layout,
      cumulativeRow layout]

theorem rows_length (layout : Layout) :
    (rows layout).length = 25 := by
  simp [rows, acceptanceRows, residueRangeRows, quotientBitRows,
    quotientBitCount]

def cost : Nightstream.Implementation.Lowering.Typed.Cost where
  recurringRows := 25
  committedColumns := 0
  publicColumns := 0
  auxiliaryColumns := 22

theorem cost_rows (layout : Layout) :
    (rows layout).length = cost.recurringRows := by
  exact rows_length layout

theorem cost_columns (layout : Layout) :
    (allocation layout).length = cost.auxiliaryColumns := by
  exact allocation_length layout

/-- The caller-owned source bits and prior-count expression precede the
candidate allocation. -/
structure InputsBelowBase (layout : Layout) : Prop where
  source : ∀ index, layout.sourceBit index < layout.base
  prior : ∀ column coefficient, (column, coefficient) ∈ layout.prior →
    column < layout.base

end Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalCandidate
