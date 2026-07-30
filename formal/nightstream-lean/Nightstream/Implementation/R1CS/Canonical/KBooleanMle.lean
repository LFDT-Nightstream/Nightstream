import Nightstream.Implementation.R1CS.Canonical.KLinear
import Nightstream.Implementation.R1CS.Canonical.KMulChain
import Nightstream.Implementation.Lowering.Typed.Cost
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.BooleanEvaluation

/-!
Contract: canonical multilinear evaluation of an explicit Boolean table over
the Goldilocks quadratic extension.

The table is evaluated in its authoritative low/high order:

`low + coordinate * (high - low)`.

The program traverses the table in postorder.  Every internal node owns one
three-row `KMul` frame; additions and subtraction remain symbolic linear
combinations.  Thus a table in `variables` variables owns exactly
`2^variables - 1` frames.  No Boolean-table value, challenge coordinate, or
evaluation result is accepted as an equation from the caller.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Canonical.KBooleanMle

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.Implementation.R1CS.Canonical.KMul
open Nightstream.Implementation.R1CS.Canonical.KHorner
open Nightstream.Implementation.R1CS.Canonical.KMulChain
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

/-- Number of nonlinear interpolation nodes in a full Boolean table. -/
def frameCount : Nat → Nat
  | 0 => 0
  | variables + 1 => 2 * frameCount variables + 1

theorem frameCount_eq (variables : Nat) :
    frameCount variables = 2 ^ variables - 1 := by
  induction variables with
  | zero => rfl
  | succ variables inductionHypothesis =>
      rw [frameCount, inductionHypothesis, Nat.pow_succ]
      have positive : 0 < 2 ^ variables := Nat.two_pow_pos variables
      omega

/-- Missing coordinates are totalized as zero.  Dimension-checked consumers
never take this branch. -/
def headCoordinate : List Carried → Carried
  | [] => KLinear.zeroCarried
  | coordinate :: _ => coordinate

def tailCoordinates : List Carried → List Carried
  | [] => []
  | _ :: coordinates => coordinates

/-- The frame owned by the root of a table with `tailVariables` variables in
each child.  Low and high subtrees occupy the preceding two frame blocks. -/
def rootFrame (frames : Nat → Frame) (step tailVariables : Nat) : Frame :=
  frames (step + 2 * frameCount tailVariables)

/-- Symbolic output carried by the complete table evaluation. -/
def carried (frames : Nat → Frame) :
    {variables : Nat} →
      BooleanTable Carried variables → List Carried → Nat → Carried
  | 0, .leaf value, _, _ => value
  | tailVariables + 1, .branch low high, coordinates, step =>
      let tail := tailCoordinates coordinates
      let lowValue := carried frames low tail step
      let frame := rootFrame frames step tailVariables
      KLinear.addCarried lowValue (frameOutput frame)

/-- One multiplication block per internal node, in postorder. -/
def rows (frames : Nat → Frame) :
    {variables : Nat} →
      BooleanTable Carried variables → List Carried → Nat → List Row
  | 0, .leaf _, _, _ => []
  | tailVariables + 1, .branch low high, coordinates, step =>
      let tail := tailCoordinates coordinates
      let lowValue := carried frames low tail step
      let highValue :=
        carried frames high tail (step + frameCount tailVariables)
      let frame := rootFrame frames step tailVariables
      rows frames low tail step ++
        rows frames high tail (step + frameCount tailVariables) ++
          KMul.rows (headCoordinate coordinates)
            (KLinear.subCarried highValue lowValue) frame

/-- Exact row count, derived from the emitted program. -/
theorem rows_length
    (frames : Nat → Frame) :
    ∀ {variables : Nat}
      (table : BooleanTable Carried variables)
      (coordinates : List Carried) (step : Nat),
      (rows frames table coordinates step).length =
        3 * frameCount variables
  | 0, .leaf _, _, _ => rfl
  | tailVariables + 1, .branch low high, coordinates, step => by
      simp only [rows, List.length_append,
        rows_length frames low (tailCoordinates coordinates) step,
        rows_length frames high (tailCoordinates coordinates)
          (step + frameCount tailVariables),
        KMul.rows_length, frameCount]
      omega

/-- Exact auxiliary allocation for a table placed at `base`. -/
def columns (base variables : Nat) : List Nat :=
  KFrames.frameColumns base (frameCount variables)

theorem columns_length (base variables : Nat) :
    (columns base variables).length = 3 * frameCount variables :=
  KFrames.frameColumns_length _ _

theorem columns_nodup (base variables : Nat) :
    (columns base variables).Nodup :=
  KFrames.frameColumns_nodup _ _

/-- The decoded reference recursion.  It is defined directly from the
assignment, so no caller can provide a table value or challenge value. -/
def decodedValue (assignment : Nat → Nat) :
    {variables : Nat} →
      BooleanTable Carried variables → List Carried → Pair
  | 0, .leaf value, _ => carriedValue assignment value
  | _ + 1, .branch low high, coordinates =>
      let tail := tailCoordinates coordinates
      let lowValue := decodedValue assignment low tail
      let highValue := decodedValue assignment high tail
      addPair lowValue
        (mulPair (carriedValue assignment (headCoordinate coordinates))
          (KPairLaws.subPair highValue lowValue))

private theorem satisfies_append_left
    {left right : List Row} {assignment : Nat → Nat}
    (satisfied : Satisfies (left ++ right) assignment) :
    Satisfies left assignment :=
  fun row member => satisfied row (List.mem_append_left _ member)

private theorem satisfies_append_right
    {left right : List Row} {assignment : Nat → Nat}
    (satisfied : Satisfies (left ++ right) assignment) :
    Satisfies right assignment :=
  fun row member => satisfied row (List.mem_append_right _ member)

/-- Satisfying rows compute the exact recursive Boolean-table MLE. -/
theorem rows_sound
    (assignment : Nat → Nat) (frames : Nat → Frame) :
    ∀ {variables : Nat}
      (table : BooleanTable Carried variables)
      (coordinates : List Carried) (step : Nat),
      Satisfies (rows frames table coordinates step) assignment →
      carriedValue assignment (carried frames table coordinates step) =
        decodedValue assignment table coordinates
  | 0, .leaf _, _, _, _ => rfl
  | tailVariables + 1, .branch low high, coordinates, step, satisfied => by
      have subtreesSatisfied :
          Satisfies
            (rows frames low (tailCoordinates coordinates) step ++
              rows frames high (tailCoordinates coordinates)
                (step + frameCount tailVariables))
            assignment :=
        satisfies_append_left satisfied
      have lowSatisfied :
          Satisfies (rows frames low (tailCoordinates coordinates) step)
            assignment :=
        satisfies_append_left subtreesSatisfied
      have highSatisfied :
          Satisfies
            (rows frames high (tailCoordinates coordinates)
              (step + frameCount tailVariables))
            assignment :=
        satisfies_append_right subtreesSatisfied
      have rootSatisfied :
          Satisfies
            (KMul.rows (headCoordinate coordinates)
              (KLinear.subCarried
                (carried frames high (tailCoordinates coordinates)
                  (step + frameCount tailVariables))
                (carried frames low (tailCoordinates coordinates) step))
              (rootFrame frames step tailVariables))
            assignment :=
        satisfies_append_right satisfied
      have lowSound :=
        rows_sound assignment frames low (tailCoordinates coordinates) step
          lowSatisfied
      have highSound :=
        rows_sound assignment frames high (tailCoordinates coordinates)
          (step + frameCount tailVariables) highSatisfied
      have rootSound :=
        frameOutput_sound assignment
          (headCoordinate coordinates)
          (KLinear.subCarried
            (carried frames high (tailCoordinates coordinates)
              (step + frameCount tailVariables))
            (carried frames low (tailCoordinates coordinates) step))
          (rootFrame frames step tailVariables) rootSatisfied
      simp only [carried, decodedValue]
      rw [KLinear.carriedValue_add, rootSound,
        KLinear.carriedValue_sub, lowSound, highSound]

/-- Intrinsic cost of one Boolean-table evaluation. -/
def cost (variables : Nat) :
    Nightstream.Implementation.Lowering.Typed.Cost where
  recurringRows := 3 * frameCount variables
  committedColumns := 0
  publicColumns := 0
  auxiliaryColumns := 3 * frameCount variables

theorem cost_rows
    (frames : Nat → Frame)
    {variables : Nat}
    (table : BooleanTable Carried variables)
    (coordinates : List Carried) (step : Nat) :
    (rows frames table coordinates step).length =
      (cost variables).recurringRows :=
  rows_length frames table coordinates step

theorem cost_columns (base variables : Nat) :
    (columns base variables).length = (cost variables).auxiliaryColumns :=
  columns_length base variables

end Nightstream.Implementation.R1CS.Canonical.KBooleanMle
