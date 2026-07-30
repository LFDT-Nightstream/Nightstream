import Nightstream.Implementation.R1CS.Canonical.KBooleanMleSupport
import Nightstream.Implementation.R1CS.Canonical.KMulChainHonest

/-!
Contract: honest completeness for the canonical Boolean-MLE row program.

The witness follows the emitter's postorder exactly: low subtree, high subtree,
then the root interpolation multiplication.  The source-placement hypotheses
are syntactic and are discharged by a caller's column layout; no evaluation
value or acceptance conclusion is supplied.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Canonical.KBooleanMleHonest

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.Implementation.R1CS.Canonical.KMul
open Nightstream.Implementation.R1CS.Canonical.KMulChain
open Nightstream.Implementation.R1CS.Canonical.KHornerSupport
open Nightstream.Implementation.R1CS.Canonical.KBooleanMleSupport
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

/-- Extend an assignment in the same postorder as `KBooleanMle.rows`. -/
def witness (assignment : Nat → Nat) (base : Nat) :
    {variables : Nat} →
      BooleanTable Carried variables → List Carried → Nat → (Nat → Nat)
  | 0, .leaf _, _, _ => assignment
  | tailVariables + 1, .branch low high, coordinates, step =>
      let tail := KBooleanMle.tailCoordinates coordinates
      let lowWitness := witness assignment base low tail step
      let highWitness :=
        witness lowWitness base high tail
          (step + KBooleanMle.frameCount tailVariables)
      let lowValue :=
        KBooleanMle.carried (KFrames.frameAt base) low tail step
      let highValue :=
        KBooleanMle.carried (KFrames.frameAt base) high tail
          (step + KBooleanMle.frameCount tailVariables)
      KMulHonest.witness highWitness
        (KBooleanMle.headCoordinate coordinates)
        (KLinear.subCarried highValue lowValue)
        (KBooleanMle.rootFrame (KFrames.frameAt base) step tailVariables)

/-- A subtree witness writes no column preceding its first frame. -/
theorem witness_off_before
    (assignment : Nat → Nat) (base : Nat) :
    ∀ {variables : Nat}
      (table : BooleanTable Carried variables)
      (coordinates : List Carried) (step column : Nat),
      column < base + 3 * step →
      witness assignment base table coordinates step column =
        assignment column
  | 0, .leaf _, _, _, _, _ => rfl
  | tailVariables + 1, .branch low high, coordinates, step, column, below => by
      rw [witness,
        KMulHonest.witness_off_frame _ _ _ _ column]
      · rw [witness_off_before
            (witness assignment base low
              (KBooleanMle.tailCoordinates coordinates) step)
            base high (KBooleanMle.tailCoordinates coordinates)
            (step + KBooleanMle.frameCount tailVariables) column (by omega),
          witness_off_before assignment base low
            (KBooleanMle.tailCoordinates coordinates) step column below]
      all_goals
        simp only [KBooleanMle.rootFrame, KFrames.frameAt,
          KFrames.frameColumn, KFrames.columnsPerFrame]
        omega

private theorem witness_root_preserves_below
    (assignment : Nat → Nat)
    (left right : Carried)
    (base rootStep limit : Nat)
    (ordered : limit ≤ base + 3 * rootStep)
    (column : Nat) (below : column < limit) :
    KMulHonest.witness assignment left right
        (KFrames.frameAt base rootStep) column =
      assignment column := by
  rw [KMulHonest.witness_off_frame]
  all_goals
    simp only [KFrames.frameAt, KFrames.frameColumn,
      KFrames.columnsPerFrame]
    omega

/-- The honest postorder extension satisfies every emitted row. -/
theorem witness_satisfies
    (assignment : Nat → Nat) (base : Nat) :
    ∀ {variables : Nat}
      (table : BooleanTable Carried variables)
      (coordinates : List Carried) (step : Nat),
      TableBelowBase table base →
      CoordinatesBelowBase coordinates base →
      Satisfies
        (KBooleanMle.rows (KFrames.frameAt base)
          table coordinates step)
        (witness assignment base table coordinates step)
  | 0, .leaf _, _, _, _, _ => by
      intro row member
      simp [KBooleanMle.rows] at member
  | tailVariables + 1, .branch low high, coordinates, step,
      tableBelow, coordinatesBelow => by
      let tail := KBooleanMle.tailCoordinates coordinates
      let lowStep := step
      let highStep := step + KBooleanMle.frameCount tailVariables
      let rootStep := step + 2 * KBooleanMle.frameCount tailVariables
      let lowValue :=
        KBooleanMle.carried (KFrames.frameAt base) low tail lowStep
      let highValue :=
        KBooleanMle.carried (KFrames.frameAt base) high tail highStep
      let difference := KLinear.subCarried highValue lowValue
      let lowWitness := witness assignment base low tail lowStep
      let highWitness := witness lowWitness base high tail highStep
      let finalWitness :=
        KMulHonest.witness highWitness
          (KBooleanMle.headCoordinate coordinates) difference
          (KFrames.frameAt base rootStep)
      have tailBelow := tailCoordinates_below coordinates base coordinatesBelow
      have lowSatisfied :
          Satisfies
            (KBooleanMle.rows (KFrames.frameAt base) low tail lowStep)
            lowWitness :=
        witness_satisfies assignment base low tail lowStep tableBelow.1 tailBelow
      have highSatisfied :
          Satisfies
            (KBooleanMle.rows (KFrames.frameAt base) high tail highStep)
            highWitness :=
        witness_satisfies lowWitness base high tail highStep tableBelow.2 tailBelow
      have lowSatisfiedHigh :
          Satisfies
            (KBooleanMle.rows (KFrames.frameAt base) low tail lowStep)
            highWitness := by
        apply satisfies_extend _ lowWitness highWitness
        · intro row member column mentioned
          symm
          apply witness_off_before
          exact rows_below base low tail lowStep tableBelow.1 tailBelow
            row member column mentioned
        · exact lowSatisfied
      have lowSatisfiedFinal :
          Satisfies
            (KBooleanMle.rows (KFrames.frameAt base) low tail lowStep)
            finalWitness := by
        apply satisfies_extend _ highWitness finalWitness
        · intro row member column mentioned
          symm
          apply witness_root_preserves_below
            highWitness (KBooleanMle.headCoordinate coordinates)
            difference base rootStep
            (base + 3 * (lowStep + KBooleanMle.frameCount tailVariables))
          · simp only [lowStep, rootStep]
            omega
          · exact rows_below base low tail lowStep tableBelow.1 tailBelow
              row member column mentioned
        · exact lowSatisfiedHigh
      have highSatisfiedFinal :
          Satisfies
            (KBooleanMle.rows (KFrames.frameAt base) high tail highStep)
            finalWitness := by
        apply satisfies_extend _ highWitness finalWitness
        · intro row member column mentioned
          symm
          apply witness_root_preserves_below
            highWitness (KBooleanMle.headCoordinate coordinates)
            difference base rootStep
            (base + 3 * (highStep + KBooleanMle.frameCount tailVariables))
          · simp only [highStep, rootStep]
            omega
          · exact rows_below base high tail highStep tableBelow.2 tailBelow
              row member column mentioned
        · exact highSatisfied
      have lowValueBelow :
          CarriedBelow lowValue (base + 3 * rootStep) := by
        apply carriedBelow_mono
          (carried_below base low tail lowStep tableBelow.1 tailBelow)
        simp only [lowStep, rootStep]
        omega
      have highValueBelow :
          CarriedBelow highValue (base + 3 * rootStep) := by
        apply carriedBelow_mono
          (carried_below base high tail
            (step + KBooleanMle.frameCount tailVariables)
            tableBelow.2 tailBelow)
        simp only [highValue, highStep, rootStep]
        omega
      have differenceBelow :
          CarriedBelow difference (base + 3 * rootStep) :=
        subCarried_below highValue lowValue _ highValueBelow lowValueBelow
      have coordinateBelow :=
        headCoordinate_below coordinates base coordinatesBelow
      have rootSatisfied :
          Satisfies
            (KMul.rows (KBooleanMle.headCoordinate coordinates) difference
              (KFrames.frameAt base rootStep))
            finalWitness := by
        exact KMulHonest.witness_satisfies highWitness
          (KBooleanMle.headCoordinate coordinates) difference
          (KFrames.frameAt base rootStep)
          (KMulHonest.canonical_distinct base rootStep)
          (KMulChainHonest.fresh_of_belowStep _
            base rootStep
            (KMulChainHonest.belowBase_to_belowStep _ base rootStep
              (fun column mentioned =>
                coordinateBelow column (Or.inl mentioned))))
          (KMulChainHonest.fresh_of_belowStep _
            base rootStep
            (KMulChainHonest.belowBase_to_belowStep _ base rootStep
              (fun column mentioned =>
                coordinateBelow column (Or.inr mentioned))))
          (KMulChainHonest.fresh_of_belowStep _
            base rootStep
            (fun column mentioned =>
              differenceBelow column (Or.inl mentioned)))
          (KMulChainHonest.fresh_of_belowStep _
            base rootStep
            (fun column mentioned =>
              differenceBelow column (Or.inr mentioned)))
      intro row member
      simp only [KBooleanMle.rows, List.mem_append] at member
      change RowHolds finalWitness row
      rcases member with (inLow | inHigh) | inRoot
      · exact lowSatisfiedFinal row inLow
      · exact highSatisfiedFinal row inHigh
      · exact rootSatisfied row inRoot

/-- Common entry point: a complete MLE placed at frame zero. -/
theorem witness_satisfies_from_base
    (assignment : Nat → Nat) (base : Nat)
    {variables : Nat}
    (table : BooleanTable Carried variables)
    (coordinates : List Carried)
    (tableBelow : TableBelowBase table base)
    (coordinatesBelow : CoordinatesBelowBase coordinates base) :
    Satisfies
      (KBooleanMle.rows (KFrames.frameAt base) table coordinates 0)
      (witness assignment base table coordinates 0) :=
  witness_satisfies assignment base table coordinates 0
    tableBelow coordinatesBelow

end Nightstream.Implementation.R1CS.Canonical.KBooleanMleHonest
