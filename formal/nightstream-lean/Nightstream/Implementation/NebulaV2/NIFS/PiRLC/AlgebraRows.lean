import Nightstream.Implementation.NebulaV2.NIFS.PiRLC.RingCombinationRows
import Nightstream.Implementation.NebulaV2.NIFS.PiDEC.Rows

/-!
Contract: exact placement of all V2 PiRLC algebra ring families.

The parent contains 110 base-ring families: 72 commitment rings, ten public
input rings, and 28 evaluation-limb rings. Each family uses the same 15
transcript-derived challenge rings and one exact ring-combination occurrence.

This file owns the family enumeration, disjoint auxiliary windows, aggregate
row and column counts, and restriction of aggregate row satisfaction to each
family. It does not own the typed paper-parent bridge or transcript sampling.

Emits constraints: 4,817,340 R1CS rows and owns 4,811,400 auxiliary columns.
-/

set_option autoImplicit false
set_option maxRecDepth 100000

namespace Nightstream.Implementation.NebulaV2.ProductPiRlcAlgebraRows

open Nightstream.Implementation.R1CS
open Nightstream.Protocol.NebulaV2.CommitmentBundle
open Nightstream.SuperNeo.Concrete

abbrev Source := ProductPiRlcRingCombinationRows.Source
abbrev Lane := ProductPiRlcRingCombinationRows.Lane
abbrev CommitmentRow := ProductPiDecRows.CommitmentRow
abbrev MatrixIndex := ProductPiDecRows.MatrixIndex
abbrev ExtensionLimb := ProductPiDecRows.ExtensionLimb
abbrev PublicBlock := Fin 10

/-- One base-ring-valued coordinate family in the complete PiRLC parent. -/
inductive Family where
  | commitment (component : Component) (row : CommitmentRow)
  | publicInput (block : PublicBlock)
  | evaluation (matrix : MatrixIndex) (limb : ExtensionLimb)
  deriving DecidableEq

def componentIndex : Component -> Fin 4
  | .full => 0
  | .operations => 1
  | .initialSnapshot => 2
  | .finalSnapshot => 3

theorem componentIndex_injective : Function.Injective componentIndex := by
  intro left right equal
  cases left <;> cases right <;> simp [componentIndex] at equal ⊢

/-- Verifier-key-bound family order: bundle, public input, evaluation. -/
def familyOrdinal : Family -> Nat
  | .commitment component row => componentIndex component * 18 + row.val
  | .publicInput block => 72 + block.val
  | .evaluation matrix limb => 82 + matrix.val * 2 + limb.val

theorem familyOrdinal_lt (family : Family) : familyOrdinal family < 110 := by
  cases family with
  | commitment component row =>
      have componentLt := (componentIndex component).isLt
      have rowLt := row.isLt
      change row.val < 18 at rowLt
      simp only [familyOrdinal]
      omega
  | publicInput block =>
      have blockLt := block.isLt
      simp only [familyOrdinal]
      omega
  | evaluation matrix limb =>
      have matrixLt := matrix.isLt
      have limbLt := limb.isLt
      change matrix.val < 14 at matrixLt
      simp only [familyOrdinal]
      omega

theorem familyOrdinal_injective : Function.Injective familyOrdinal := by
  intro left right equal
  cases left with
  | commitment leftComponent leftRow =>
      cases right with
      | commitment rightComponent rightRow =>
          have leftComponentLt := (componentIndex leftComponent).isLt
          have rightComponentLt := (componentIndex rightComponent).isLt
          have leftRowLt := leftRow.isLt
          have rightRowLt := rightRow.isLt
          change leftRow.val < 18 at leftRowLt
          change rightRow.val < 18 at rightRowLt
          simp only [familyOrdinal] at equal
          have componentEqual : componentIndex leftComponent =
              componentIndex rightComponent := by
            apply Fin.ext
            omega
          have rowEqual : leftRow = rightRow := by
            apply Fin.ext
            omega
          rw [componentIndex_injective componentEqual, rowEqual]
      | publicInput rightBlock =>
          have leftComponentLt := (componentIndex leftComponent).isLt
          have leftRowLt := leftRow.isLt
          have rightBlockLt := rightBlock.isLt
          change leftRow.val < 18 at leftRowLt
          simp only [familyOrdinal] at equal
          omega
      | evaluation rightMatrix rightLimb =>
          have leftComponentLt := (componentIndex leftComponent).isLt
          have leftRowLt := leftRow.isLt
          have rightMatrixLt := rightMatrix.isLt
          have rightLimbLt := rightLimb.isLt
          change leftRow.val < 18 at leftRowLt
          change rightMatrix.val < 14 at rightMatrixLt
          simp only [familyOrdinal] at equal
          omega
  | publicInput leftBlock =>
      cases right with
      | commitment rightComponent rightRow =>
          have leftBlockLt := leftBlock.isLt
          have rightComponentLt := (componentIndex rightComponent).isLt
          have rightRowLt := rightRow.isLt
          change rightRow.val < 18 at rightRowLt
          simp only [familyOrdinal] at equal
          omega
      | publicInput rightBlock =>
          congr 1
          apply Fin.ext
          simpa [familyOrdinal] using equal
      | evaluation rightMatrix rightLimb =>
          have leftBlockLt := leftBlock.isLt
          have rightMatrixLt := rightMatrix.isLt
          have rightLimbLt := rightLimb.isLt
          change rightMatrix.val < 14 at rightMatrixLt
          simp only [familyOrdinal] at equal
          omega
  | evaluation leftMatrix leftLimb =>
      cases right with
      | commitment rightComponent rightRow =>
          have leftMatrixLt := leftMatrix.isLt
          have leftLimbLt := leftLimb.isLt
          have rightComponentLt := (componentIndex rightComponent).isLt
          have rightRowLt := rightRow.isLt
          change leftMatrix.val < 14 at leftMatrixLt
          change rightRow.val < 18 at rightRowLt
          simp only [familyOrdinal] at equal
          omega
      | publicInput rightBlock =>
          have leftMatrixLt := leftMatrix.isLt
          have leftLimbLt := leftLimb.isLt
          have rightBlockLt := rightBlock.isLt
          change leftMatrix.val < 14 at leftMatrixLt
          simp only [familyOrdinal] at equal
          omega
      | evaluation rightMatrix rightLimb =>
          have leftMatrixLt := leftMatrix.isLt
          have rightMatrixLt := rightMatrix.isLt
          have leftLimbLt := leftLimb.isLt
          have rightLimbLt := rightLimb.isLt
          change leftMatrix.val < 14 at leftMatrixLt
          change rightMatrix.val < 14 at rightMatrixLt
          simp only [familyOrdinal] at equal
          have matrixEqual : leftMatrix = rightMatrix := by
            apply Fin.ext
            omega
          have limbEqual : leftLimb = rightLimb := by
            apply Fin.ext
            omega
          rw [matrixEqual, limbEqual]

def commitmentFamilies : List Family :=
  ProductPiDecRows.components.flatMap fun component =>
    (ProductPiDecRows.indices ProductCommitmentAlgebra.Rank).map fun row =>
      .commitment component row

def publicFamilies : List Family :=
  (ProductPiDecRows.indices 10).map Family.publicInput

def evaluationFamilies : List Family :=
  (ProductPiDecRows.indices ProductNifsCodec.shape.matrixCount).flatMap fun matrix =>
    (ProductPiDecRows.indices 2).map fun limb => .evaluation matrix limb

def families : List Family :=
  commitmentFamilies ++ publicFamilies ++ evaluationFamilies

theorem family_mem (family : Family) : family ∈ families := by
  cases family with
  | commitment component row =>
      apply List.mem_append_left
      apply List.mem_append_left
      apply List.mem_flatMap.mpr
      exact ⟨component, ProductPiDecRows.component_mem component,
        List.mem_map.mpr ⟨row, ProductPiDecRows.index_mem row, rfl⟩⟩
  | publicInput block =>
      apply List.mem_append_left
      apply List.mem_append_right
      exact List.mem_map.mpr ⟨block, ProductPiDecRows.index_mem block, rfl⟩
  | evaluation matrix limb =>
      apply List.mem_append_right
      apply List.mem_flatMap.mpr
      exact ⟨matrix, ProductPiDecRows.index_mem matrix,
        List.mem_map.mpr ⟨limb, ProductPiDecRows.index_mem limb, rfl⟩⟩

private theorem length_flatMap_uniform
    {Alpha Beta : Type} (items : List Alpha) (values : Alpha -> List Beta)
    (count : Nat) (uniform : forall item, (values item).length = count) :
    (items.flatMap values).length = items.length * count := by
  induction items with
  | nil => simp
  | cons head tail inductionHypothesis =>
      simp [uniform, inductionHypothesis, Nat.add_mul, Nat.add_comm]

theorem commitmentFamilies_length : commitmentFamilies.length = 72 := by
  unfold commitmentFamilies
  rw [length_flatMap_uniform _ _ 18]
  · decide
  · intro component
    change (ProductPiDecRows.indices 18).length = 18
    exact ProductPiDecRows.indices_length 18

theorem publicFamilies_length : publicFamilies.length = 10 := by
  simp [publicFamilies, ProductPiDecRows.indices]

theorem evaluationFamilies_length : evaluationFamilies.length = 28 := by
  unfold evaluationFamilies
  rw [length_flatMap_uniform _ _ 2]
  · decide
  · intro matrix
    simp [ProductPiDecRows.indices]

theorem families_length : families.length = 110 := by
  simp [families, commitmentFamilies_length, publicFamilies_length,
    evaluationFamilies_length]

/-- Caller-owned coordinates and this occurrence's disjoint auxiliary base. -/
structure Layout where
  base : Nat
  challengeSymbol : Source -> Lane -> Nat
  inputBundle : Source -> Component -> CommitmentRow -> Lane -> Nat
  outputBundle : Component -> CommitmentRow -> Lane -> Nat
  inputPublic : Source -> PublicBlock -> Lane -> Nat
  outputPublic : PublicBlock -> Lane -> Nat
  inputEvaluation : Source -> MatrixIndex -> ExtensionLimb -> Lane -> Nat
  outputEvaluation : MatrixIndex -> ExtensionLimb -> Lane -> Nat

def familyInput (layout : Layout) (family : Family) : Source -> Lane -> Nat
  | source, lane =>
      match family with
      | .commitment component row =>
          layout.inputBundle source component row lane
      | .publicInput block => layout.inputPublic source block lane
      | .evaluation matrix limb =>
          layout.inputEvaluation source matrix limb lane

def familyOutput (layout : Layout) (family : Family) : Lane -> Nat
  | lane =>
      match family with
      | .commitment component row => layout.outputBundle component row lane
      | .publicInput block => layout.outputPublic block lane
      | .evaluation matrix limb => layout.outputEvaluation matrix limb lane

def familyBase (layout : Layout) (family : Family) : Nat :=
  layout.base + familyOrdinal family *
    ProductPiRlcRingCombinationRows.auxiliaryCount

def familyLayout (layout : Layout) (family : Family) :
    ProductPiRlcRingCombinationRows.Layout where
  base := familyBase layout family
  challengeSymbol := layout.challengeSymbol
  input := familyInput layout family
  output := familyOutput layout family

def rows (layout : Layout) : List Row :=
  families.flatMap fun family =>
    ProductPiRlcRingCombinationRows.rows (familyLayout layout family)

theorem rows_length (layout : Layout) : (rows layout).length = 4817340 := by
  unfold rows
  rw [length_flatMap_uniform _ _ 43794]
  · rw [families_length]
  · intro family
    exact ProductPiRlcRingCombinationRows.rows_length
      (familyLayout layout family)

def allocation (layout : Layout) : List Nat :=
  families.flatMap fun family =>
    ProductPiRlcRingCombinationRows.allocation (familyLayout layout family)

theorem allocation_length (layout : Layout) :
    (allocation layout).length = 4811400 := by
  unfold allocation
  rw [length_flatMap_uniform _ _
    ProductPiRlcRingCombinationRows.auxiliaryCount]
  · rw [families_length]
    decide
  · intro family
    exact ProductPiRlcRingCombinationRows.allocation_length
      (familyLayout layout family)

/-- Different families own disjoint half-open auxiliary windows. -/
theorem family_windows_disjoint
    (layout : Layout) {left right : Family} (different : left ≠ right) :
    familyBase layout left + ProductPiRlcRingCombinationRows.auxiliaryCount <=
        familyBase layout right \/
      familyBase layout right +
          ProductPiRlcRingCombinationRows.auxiliaryCount <=
        familyBase layout left := by
  have ordinalDifferent : familyOrdinal left ≠ familyOrdinal right := by
    intro equal
    exact different (familyOrdinal_injective equal)
  rcases Nat.lt_or_gt_of_ne ordinalDifferent with before | after
  · left
    have auxiliaryExact :
        ProductPiRlcRingCombinationRows.auxiliaryCount = 43740 := by
      decide
    simp only [familyBase, auxiliaryExact]
    omega
  · right
    have auxiliaryExact :
        ProductPiRlcRingCombinationRows.auxiliaryCount = 43740 := by
      decide
    simp only [familyBase, auxiliaryExact]
    omega

/-- Aggregate satisfaction implies satisfaction of each exact ring family. -/
theorem family_satisfies
    {layout : Layout} {assignment : Nat -> Nat}
    (satisfied : Satisfies (rows layout) assignment) (family : Family) :
    Satisfies
      (ProductPiRlcRingCombinationRows.rows (familyLayout layout family))
      assignment := by
  intro row member
  apply satisfied row
  unfold rows
  exact List.mem_flatMap.mpr ⟨family, family_mem family, member⟩

end Nightstream.Implementation.NebulaV2.ProductPiRlcAlgebraRows
