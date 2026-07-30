import Nightstream.Implementation.R1CS.Canonical.KPiCcsInitialHonest
import Nightstream.Implementation.R1CS.Canonical.KHornerOwnership

/-!
Receipts, conservation, and exact cost for the joint `Pi_CCS` initial claim.

The program owns only its Horner frames.  Gamma, carried coefficients, and the
chain initial are shared reads; the final equality adds two rows and no
columns.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Canonical.KPiCcsInitialOwnership

open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.Implementation.R1CS.Canonical.KMul
open Nightstream.Implementation.R1CS.Canonical.KHorner
open Nightstream.Implementation.R1CS.Canonical.KHornerSupport
open Nightstream.Implementation.R1CS.Canonical.KPiCcsInitial

/-- One positional owner for every emitted row. -/
inductive RowOwner where
  | horner (receipt : KHornerOwnership.Receipt)
  | binding (owner : KEquality.RowOwner)
deriving DecidableEq, Repr

def allOwners
    {shape : Shape} (input : Input shape) : List RowOwner :=
  (KHornerOwnership.receipts ((coefficients input).length - 1)).map
      RowOwner.horner ++
    KEquality.allOwners.map RowOwner.binding

def ownedRow
    {shape : Shape} (input : Input shape) : RowOwner → Row
  | .horner receipt =>
      KHornerOwnership.receiptRow input.gamma
        (KFrames.frameAt input.frameBase) (coefficients input) 0 receipt
  | .binding owner =>
      KEquality.ownedRow (evaluated input) input.initial owner

theorem allOwners_nodup
    {shape : Shape} (input : Input shape) :
    (allOwners input).Nodup := by
  unfold allOwners
  rw [List.nodup_append]
  refine ⟨
    LinCombNormal.nodup_map _ RowOwner.horner
      (fun left right equal => by
        cases equal
        rfl)
      (KHornerOwnership.receipts_nodup _),
    LinCombNormal.nodup_map _ RowOwner.binding
      (fun left right equal => by
        cases equal
        rfl)
      KEquality.allOwners_nodup,
    ?_⟩
  intro left leftMember right rightMember equal
  rcases List.mem_map.1 leftMember with ⟨receipt, _, rfl⟩
  rcases List.mem_map.1 rightMember with ⟨owner, _, rfl⟩
  cases equal

/-- The physical program is exactly the owner list's image. -/
theorem rows_eq_map_owners
    {shape : Shape} (input : Input shape) :
    KPiCcsInitial.rows input =
      (allOwners input).map (ownedRow input) := by
  unfold KPiCcsInitial.rows allOwners
  rw [List.map_append, List.map_map, List.map_map,
    KHornerOwnership.hornerRows_eq_map_receipts,
    KEquality.rows_eq_map_owners]
  rfl

/-- Columns read from the authoritative source layer. -/
def SourceColumn
    {shape : Shape} (input : Input shape) (column : Nat) : Prop :=
  Mentions input.gamma.low column ∨ Mentions input.gamma.high column
    ∨ (∃ coefficient ∈ coefficients input,
        Mentions coefficient.low column ∨ Mentions coefficient.high column)
    ∨ Mentions input.initial.low column ∨ Mentions input.initial.high column

/-- Columns allocated by this initial-claim occurrence. -/
def Allocated
    {shape : Shape} (input : Input shape) (column : Nat) : Prop :=
  input.frameBase ≤ column ∧
    column < input.frameBase + 3 * ((coefficients input).length - 1)

private theorem frameOfRun_allocated
    {shape : Shape} (input : Input shape) (column : Nat)
    (frame :
      FrameOfRun (KFrames.frameAt input.frameBase)
        (coefficients input) 0 column) :
    Allocated input column := by
  rcases frame with ⟨later, _, bounded, inFrame⟩
  rcases inFrame with rfl | rfl | rfl <;>
    simp only [Allocated, KFrames.frameAt, KFrames.frameColumn,
      KFrames.columnsPerFrame] <;> omega

/-- No emitted row reaches outside the constant wire, authoritative sources,
and the exactly declared Horner allocation. -/
theorem rows_conservation
    {shape : Shape} (input : Input shape)
    (row : Row) (member : row ∈ KPiCcsInitial.rows input)
    (column : Nat)
    (mentioned : Mentions row.a column ∨ Mentions row.b column
      ∨ Mentions row.c column) :
    column = 0 ∨ SourceColumn input column ∨ Allocated input column := by
  unfold KPiCcsInitial.rows at member
  rcases List.mem_append.1 member with inHorner | inBinding
  · rcases hornerRows_mentions input.gamma
        (KFrames.frameAt input.frameBase) (coefficients input) 0 row
        inHorner column mentioned with
      inGamma | inCoefficient | inFrame
    · refine Or.inr (Or.inl ?_)
      rcases inGamma with inLow | inHigh
      · exact Or.inl inLow
      · exact Or.inr (Or.inl inHigh)
    · exact Or.inr (Or.inl
        (Or.inr (Or.inr (Or.inl inCoefficient))))
    · exact Or.inr (Or.inr (frameOfRun_allocated input column inFrame))
  · rcases KEquality.rows_conservation (evaluated input) input.initial row
        inBinding column mentioned with
      wire | inEvaluatedLow | inEvaluatedHigh | inInitialLow | inInitialHigh
    · exact Or.inl wire
    · rcases hornerCarried_mentions input.gamma
          (KFrames.frameAt input.frameBase) (coefficients input) 0 column
          (Or.inl inEvaluatedLow) with inCoefficient | inFrame
      · exact Or.inr (Or.inl
          (Or.inr (Or.inr (Or.inl inCoefficient))))
      · exact Or.inr (Or.inr (frameOfRun_allocated input column inFrame))
    · rcases hornerCarried_mentions input.gamma
          (KFrames.frameAt input.frameBase) (coefficients input) 0 column
          (Or.inr inEvaluatedHigh) with inCoefficient | inFrame
      · exact Or.inr (Or.inl
          (Or.inr (Or.inr (Or.inl inCoefficient))))
      · exact Or.inr (Or.inr (frameOfRun_allocated input column inFrame))
    · exact Or.inr (Or.inl
        (Or.inr (Or.inr (Or.inr (Or.inl inInitialLow)))))
    · exact Or.inr (Or.inl
        (Or.inr (Or.inr (Or.inr (Or.inr inInitialHigh)))))

/-- Exact cost of the initial-claim computation.  All source expressions are
shared reads; only the Horner frames are allocated here. -/
def cost (shape : Shape) :
    Nightstream.Implementation.Lowering.Typed.Cost where
  recurringRows := 3 * (shape.jointCoefficientCount - 1) + 2
  committedColumns := 0
  publicColumns := 0
  auxiliaryColumns := 3 * (shape.jointCoefficientCount - 1)

theorem cost_rows
    {shape : Shape} (input : Input shape) :
    (rows input).length = (cost shape).recurringRows :=
  rows_length input

theorem cost_columns
    {shape : Shape} (input : Input shape) :
    (columns input).length = (cost shape).auxiliaryColumns :=
  columns_length input

end Nightstream.Implementation.R1CS.Canonical.KPiCcsInitialOwnership
