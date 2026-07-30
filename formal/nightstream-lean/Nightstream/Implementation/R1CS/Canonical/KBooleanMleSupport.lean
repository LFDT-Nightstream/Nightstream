import Nightstream.Implementation.R1CS.Canonical.KBooleanMle
import Nightstream.Implementation.R1CS.Canonical.KMulOwnership

/-!
Contract: placement support for the canonical Boolean-MLE program.

The only placement premise is that authoritative leaves and point coordinates
precede the auxiliary base.  From it this module proves exact upper bounds for
every carried value and every row in a subtree.  Honest completeness consumes
those bounds to extend an assignment in postorder without changing an earlier
row or a source value.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Canonical.KBooleanMleSupport

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.Implementation.R1CS.Canonical.KMul
open Nightstream.Implementation.R1CS.Canonical.KMulChain
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

/-- Both coordinates of a carried value mention only columns below `limit`. -/
def CarriedBelow (value : Carried) (limit : Nat) : Prop :=
  ∀ column,
    (Mentions value.low column ∨ Mentions value.high column) →
      column < limit

/-- Every authoritative leaf precedes the auxiliary base. -/
def TableBelowBase :
    {variables : Nat} → BooleanTable Carried variables → Nat → Prop
  | 0, .leaf value, base => CarriedBelow value base
  | _ + 1, .branch low high, base =>
      TableBelowBase low base ∧ TableBelowBase high base

/-- Every coordinate consumed by the MLE precedes the auxiliary base. -/
def CoordinatesBelowBase (coordinates : List Carried) (base : Nat) : Prop :=
  ∀ coordinate ∈ coordinates, CarriedBelow coordinate base

theorem carriedBelow_mono
    {value : Carried} {lower upper : Nat}
    (below : CarriedBelow value lower) (ordered : lower ≤ upper) :
    CarriedBelow value upper :=
  fun column mentioned => Nat.lt_of_lt_of_le (below column mentioned) ordered

theorem tail_below
    {coordinate : Carried} {coordinates : List Carried} {base : Nat}
    (below : CoordinatesBelowBase (coordinate :: coordinates) base) :
    CoordinatesBelowBase coordinates base :=
  fun value member => below value (List.mem_cons_of_mem coordinate member)

theorem head_below
    {coordinate : Carried} {coordinates : List Carried} {base : Nat}
    (below : CoordinatesBelowBase (coordinate :: coordinates) base) :
    CarriedBelow coordinate base :=
  below coordinate List.mem_cons_self

theorem headCoordinate_below
    (coordinates : List Carried) (base : Nat)
    (below : CoordinatesBelowBase coordinates base) :
    CarriedBelow (KBooleanMle.headCoordinate coordinates) base := by
  cases coordinates with
  | nil =>
      intro column mentioned
      simp [KBooleanMle.headCoordinate, KLinear.zeroCarried, Mentions] at mentioned
  | cons coordinate coordinates =>
      exact head_below below

theorem tailCoordinates_below
    (coordinates : List Carried) (base : Nat)
    (below : CoordinatesBelowBase coordinates base) :
    CoordinatesBelowBase (KBooleanMle.tailCoordinates coordinates) base := by
  cases coordinates with
  | nil => exact below
  | cons coordinate coordinates => exact tail_below below

/-- Linear subtraction introduces no new source column. -/
theorem subCarried_mentions
    (left right : Carried) (column : Nat)
    (mentioned :
      Mentions (KLinear.subCarried left right).low column ∨
        Mentions (KLinear.subCarried left right).high column) :
    (Mentions left.low column ∨ Mentions left.high column) ∨
      (Mentions right.low column ∨ Mentions right.high column) := by
  rcases mentioned with low | high
  · simp only [KLinear.subCarried, KLinear.addCarried] at low
    rw [mentions_append] at low
    rcases low with inLeft | inRight
    · exact Or.inl (Or.inl inLeft)
    · exact Or.inr (Or.inl
        ((mentions_map_scale (goldilocksP - 1) right.low column).mp inRight))
  · simp only [KLinear.subCarried, KLinear.addCarried] at high
    rw [mentions_append] at high
    rcases high with inLeft | inRight
    · exact Or.inl (Or.inr inLeft)
    · exact Or.inr (Or.inr
        ((mentions_map_scale (goldilocksP - 1) right.high column).mp inRight))

/-- Subtraction preserves any common column upper bound. -/
theorem subCarried_below
    (left right : Carried) (limit : Nat)
    (leftBelow : CarriedBelow left limit)
    (rightBelow : CarriedBelow right limit) :
    CarriedBelow (KLinear.subCarried left right) limit := by
  intro column mentioned
  rcases subCarried_mentions left right column mentioned with
    inLeft | inRight
  · exact leftBelow column inLeft
  · exact rightBelow column inRight

/-- A frame output is bounded by the end of its own three-column block. -/
theorem frameOutput_below (base step : Nat) :
    CarriedBelow (frameOutput (KFrames.frameAt base step))
      (base + 3 * (step + 1)) := by
  intro column mentioned
  rcases mentioned with low | high
  · simp only [frameOutput, outLow, Mentions, List.map_cons, List.map_nil,
      List.mem_cons, List.not_mem_nil, or_false] at low
    rcases low with rfl | rfl <;>
      simp only [KFrames.frameAt, KFrames.frameColumn,
        KFrames.columnsPerFrame] <;> omega
  · simp only [frameOutput, outHigh, Mentions, List.map_cons, List.map_nil,
      List.mem_cons, List.not_mem_nil, or_false] at high
    rcases high with rfl | rfl | rfl <;>
      simp only [KFrames.frameAt, KFrames.frameColumn,
        KFrames.columnsPerFrame] <;> omega

/-- The symbolic result of a subtree reaches only sources or frames in that
subtree's exact interval. -/
theorem carried_below
    (base : Nat) :
    ∀ {variables : Nat}
      (table : BooleanTable Carried variables)
      (coordinates : List Carried) (step : Nat),
      TableBelowBase table base →
      CoordinatesBelowBase coordinates base →
      CarriedBelow
        (KBooleanMle.carried (KFrames.frameAt base)
          table coordinates step)
        (base + 3 * (step + KBooleanMle.frameCount variables))
  | 0, .leaf value, _, step, tableBelow, _ => by
      intro column mentioned
      have source := tableBelow column mentioned
      omega
  | tailVariables + 1, .branch low high, coordinates, step,
      tableBelow, coordinatesBelow => by
      intro column mentioned
      simp only [KBooleanMle.carried, KLinear.addCarried, Mentions,
        List.map_append, List.mem_append] at mentioned
      rcases mentioned with (inLow | inFrame) | (inLow | inFrame)
      · have bounded :=
          carried_below base low (KBooleanMle.tailCoordinates coordinates)
            step tableBelow.1
            (tailCoordinates_below coordinates base coordinatesBelow)
            column (Or.inl inLow)
        simp only [KBooleanMle.frameCount]
        omega
      · have bounded :=
          frameOutput_below base
            (step + 2 * KBooleanMle.frameCount tailVariables)
            column (Or.inl inFrame)
        simp only [KBooleanMle.frameCount]
        omega
      · have bounded :=
          carried_below base low (KBooleanMle.tailCoordinates coordinates)
            step tableBelow.1
            (tailCoordinates_below coordinates base coordinatesBelow)
            column (Or.inr inLow)
        simp only [KBooleanMle.frameCount]
        omega
      · have bounded :=
          frameOutput_below base
            (step + 2 * KBooleanMle.frameCount tailVariables)
            column (Or.inr inFrame)
        simp only [KBooleanMle.frameCount]
        omega

/-- Every column mentioned by a subtree row lies below the end of the subtree's
exact frame interval. -/
theorem rows_below
    (base : Nat) :
    ∀ {variables : Nat}
      (table : BooleanTable Carried variables)
      (coordinates : List Carried) (step : Nat),
      TableBelowBase table base →
      CoordinatesBelowBase coordinates base →
      ∀ row ∈ KBooleanMle.rows (KFrames.frameAt base)
          table coordinates step,
        ∀ column,
          (Mentions row.a column ∨ Mentions row.b column ∨
            Mentions row.c column) →
          column < base + 3 * (step + KBooleanMle.frameCount variables)
  | 0, .leaf _, _, _, _, _, row, member, _, _ => by
      simp [KBooleanMle.rows] at member
  | tailVariables + 1, .branch low high, coordinates, step,
      tableBelow, coordinatesBelow, row, member, column, mentioned => by
      simp only [KBooleanMle.rows, List.mem_append] at member
      rcases member with inSubtrees | inRoot
      · rcases inSubtrees with inLow | inHigh
        · have bounded :=
            rows_below base low (KBooleanMle.tailCoordinates coordinates)
              step tableBelow.1
              (tailCoordinates_below coordinates base coordinatesBelow)
              row inLow column mentioned
          simp only [KBooleanMle.frameCount]
          omega
        · have bounded :=
            rows_below base high (KBooleanMle.tailCoordinates coordinates)
              (step + KBooleanMle.frameCount tailVariables) tableBelow.2
              (tailCoordinates_below coordinates base coordinatesBelow)
              row inHigh column mentioned
          simp only [KBooleanMle.frameCount]
          omega
      · rcases KMulOwnership.rows_conservation
          (KBooleanMle.headCoordinate coordinates)
          (KLinear.subCarried
            (KBooleanMle.carried (KFrames.frameAt base) high
              (KBooleanMle.tailCoordinates coordinates)
              (step + KBooleanMle.frameCount tailVariables))
            (KBooleanMle.carried (KFrames.frameAt base) low
              (KBooleanMle.tailCoordinates coordinates) step))
          (KBooleanMle.rootFrame (KFrames.frameAt base) step tailVariables)
          row inRoot column mentioned with operand | inFrame
        · rcases operand with inCoordinateLow | inCoordinateHigh |
              inDifferenceLow | inDifferenceHigh
          · have source :=
              headCoordinate_below coordinates base coordinatesBelow
                column (Or.inl inCoordinateLow)
            simp only [KBooleanMle.frameCount]
            omega
          · have source :=
              headCoordinate_below coordinates base coordinatesBelow
                column (Or.inr inCoordinateHigh)
            simp only [KBooleanMle.frameCount]
            omega
          · rcases subCarried_mentions _ _ column (Or.inl inDifferenceLow) with
              inHigh | inLow
            · have bounded :=
                carried_below base high
                  (KBooleanMle.tailCoordinates coordinates)
                  (step + KBooleanMle.frameCount tailVariables)
                  tableBelow.2
                  (tailCoordinates_below coordinates base coordinatesBelow)
                  column inHigh
              simp only [KBooleanMle.frameCount]
              omega
            · have bounded :=
                carried_below base low
                  (KBooleanMle.tailCoordinates coordinates) step
                  tableBelow.1
                  (tailCoordinates_below coordinates base coordinatesBelow)
                  column inLow
              simp only [KBooleanMle.frameCount]
              omega
          · rcases subCarried_mentions _ _ column (Or.inr inDifferenceHigh) with
              inHigh | inLow
            · have bounded :=
                carried_below base high
                  (KBooleanMle.tailCoordinates coordinates)
                  (step + KBooleanMle.frameCount tailVariables)
                  tableBelow.2
                  (tailCoordinates_below coordinates base coordinatesBelow)
                  column inHigh
              simp only [KBooleanMle.frameCount]
              omega
            · have bounded :=
                carried_below base low
                  (KBooleanMle.tailCoordinates coordinates) step
                  tableBelow.1
                  (tailCoordinates_below coordinates base coordinatesBelow)
                  column inLow
              simp only [KBooleanMle.frameCount]
              omega
        · rcases inFrame with rfl | rfl | rfl <;>
            simp only [KBooleanMle.rootFrame, KFrames.frameAt,
              KFrames.frameColumn, KFrames.columnsPerFrame,
              KBooleanMle.frameCount] <;> omega

end Nightstream.Implementation.R1CS.Canonical.KBooleanMleSupport
