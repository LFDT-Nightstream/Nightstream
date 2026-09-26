import NightstreamFPrime.Gadgets.Sampling.WideReduction
import NightstreamFPrime.Circuit.StraightLine

/-! Existing-hint witness program for wide reduction. Temporary values are
allocated before the unchanged checked gadget. The program materializes each
small accumulator once, so its 37 bit hints do not repeat a long expression.
This file defines execution; its correctness and read exclusion are separate. -/

namespace NightstreamFPrime.Gadgets.Sampling.WideReduction.HintProgram

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcWideSampler

def sourceBitCount : Nat := 4 * 64
def limbCount : Nat := 16
def limbBits : Nat := 16
def accumulatorBits : Nat := 37
def limbStride : Nat := 1 + accumulatorBits
def divisionLimbs : Nat := 5
def divisionBits : Nat := 60
def divisionStride : Nat := 2 * divisionLimbs
def helperCount : Nat := sourceBitCount + limbCount * limbStride + digitCount * divisionStride

def bitTerms (start count : Nat) : List (Nat × Expr) :=
  (List.range count).map fun bit => (2 ^ bit, Expr.var (start + bit))

def sourceBitHints (interface : Interface) (offset : Nat) : List Hint :=
  (List.finRange fieldCount).flatMap fun lane =>
    (List.range 64).map fun bit => .bit (interface.source lane offset) bit

def limbStart (offset limb : Nat) : Nat := offset + sourceBitCount + limb * limbStride

def coefficient (lane limb : Nat) : Nat :=
  goldilocksModulus ^ lane / (2 ^ limbBits) ^ limb % 2 ^ limbBits

def limbTerms (offset limb : Nat) : List (Nat × Expr) :=
  let carry := if limb = 0 then [] else
    bitTerms (limbStart offset (limb - 1) + 1 + limbBits) (accumulatorBits - limbBits)
  carry ++ (List.range fieldCount).flatMap fun lane =>
    (List.range 4).flatMap fun word =>
      if word ≤ limb then
        (List.range limbBits).map fun bit =>
          (coefficient lane (limb - word) * 2 ^ bit,
            Expr.var (offset + lane * 64 + word * limbBits + bit))
      else []

def limbHints (offset limb : Nat) : List Hint :=
  -- The proved accumulator bound also gives 5 * accumulator < p.
  .quotientFive (Expr.const (fieldOfNat 5) * linearExpr (limbTerms offset limb)) ::
    (List.range accumulatorBits).map fun bit => .bit (Expr.var (limbStart offset limb)) bit

def integerBit (offset bit : Nat) : Expr :=
  Expr.var (limbStart offset (bit / limbBits) + 1 + bit % limbBits)

def initialDivisionLimb (offset limb : Nat) : Expr :=
  linearExpr ((List.range divisionBits).filterMap fun bit =>
    let index := limb * divisionBits + bit
    if index < limbCount * limbBits then some (2 ^ bit, integerBit offset index) else none)

def divisionStart (offset : Nat) : Nat := offset + sourceBitCount + limbCount * limbStride

/-- Division visits limb 4 first and limb 0 last. Each pair is q then r. -/
def divisionColumn (offset round position : Nat) : Nat :=
  divisionStart offset + round * divisionStride + 2 * position

def divisionInput (offset round position : Nat) : Expr :=
  let carry := if position = 0 then Expr.const 0
    else Expr.var (divisionColumn offset round (position - 1) + 1)
  let limb := if round = 0 then initialDivisionLimb offset (divisionLimbs - 1 - position)
    else Expr.var (divisionColumn offset (round - 1) position)
  Expr.const (fieldOfNat (2 ^ divisionBits)) * carry + limb

def divisionHints (offset round : Nat) : List Hint :=
  (List.range divisionLimbs).flatMap fun position =>
    let source := divisionInput offset round position
    [.quotientFive source, .remainderFive source]

def helpers (interface : Interface) (offset : Nat) : List Hint :=
  sourceBitHints interface offset ++
    (List.range limbCount).flatMap (limbHints offset) ++
    (List.range digitCount).flatMap (divisionHints offset)

def quotientHint (helperOffset bit : Nat) : Hint :=
  .bit (Expr.var (divisionColumn helperOffset (digitCount - 1)
    (divisionLimbs - 1 - bit / divisionBits))) (bit % divisionBits)

def digitHint (helperOffset digit bit : Nat) : Hint :=
  .bit (Expr.var (divisionColumn helperOffset digit (divisionLimbs - 1) + 1)) bit

def checkQuotient (gadgetOffset : Nat) (check : Fin checkCount) : Expr :=
  (linearExpr (reduceTerms (modulus check) (drawTerms gadgetOffset)) +
      Expr.const (fieldOfNat (modulus check * checkBias)) -
    linearExpr (reduceTerms (modulus check) (resultTerms gadgetOffset))) *
      Expr.const (Hint.inverse (fieldOfNat (modulus check)))

def resultHints (helperOffset gadgetOffset : Nat) : List Hint :=
  (List.range quotientBitCount).map (quotientHint helperOffset) ++
    (List.range digitCount).flatMap (fun digit =>
      (List.range digitBitCount).map (digitHint helperOffset digit)) ++
    (List.finRange checkCount).flatMap (fun check =>
      (List.range checkBitCount).map fun bit => .bit (checkQuotient gadgetOffset check) bit)

theorem helperCount_eq : helperCount = 1404 := rfl

theorem sourceBitHints_length (interface : Interface) (offset : Nat) :
    (sourceBitHints interface offset).length = sourceBitCount := by
  simp only [sourceBitHints, List.length_flatMap, List.length_map, List.length_range,
    List.map_const', List.length_finRange, List.sum_replicate, smul_eq_mul]
  rfl

theorem limbHints_length (offset limb : Nat) :
    (limbHints offset limb).length = limbStride := by
  simp only [limbHints, List.length_cons, List.length_map, List.length_range, limbStride]
  omega

theorem divisionHints_length (offset round : Nat) :
    (divisionHints offset round).length = divisionStride := by
  simp only [divisionHints, List.length_flatMap, List.length_cons, List.length_nil,
    List.map_const', List.length_range, List.sum_replicate, smul_eq_mul]
  rfl

theorem helpers_length (interface : Interface) (offset : Nat) :
    (helpers interface offset).length = helperCount := by
  simp only [helpers, List.length_append, sourceBitHints_length, List.length_flatMap,
    limbHints_length, divisionHints_length, List.map_const', List.length_range,
    List.sum_replicate, smul_eq_mul]
  unfold helperCount
  omega

theorem resultHints_length (helperOffset gadgetOffset : Nat) :
    (resultHints helperOffset gadgetOffset).length = newBitCount := by
  simp [resultHints, newBitCount, List.length_flatMap, List.map_const',
    List.sum_replicate, smul_eq_mul, Nat.mul_comm, Nat.add_assoc]

def helperEnv (interface : Interface) (env : Env) (offset : Nat) : Env :=
  executeHints env offset (helpers interface offset)

theorem helperEnv_agreesOutside (interface : Interface) (env : Env) (offset : Nat) :
    AgreesOutside env (helperEnv interface env offset) offset helperCount := by
  unfold helperEnv
  rw [← helpers_length interface offset]
  exact executeHints_agreesOutside _ _ _

theorem helper_rows_empty (interface : Interface) (offset : Nat) :
    flatConstraints [Op.witness (WitnessBatch.hinted offset (helpers interface offset))] = [] := rfl

end NightstreamFPrime.Gadgets.Sampling.WideReduction.HintProgram
