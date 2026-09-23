import Mathlib.Data.Fin.VecNotation
import Mathlib.Algebra.BigOperators.Fin
import NightstreamFPrime.Circuit.Sequence
import NightstreamFPrime.Gadgets.Range.CanonicalU64
import NightstreamFPrime.Gadgets.Sampling.WideReduction.Linear
import NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcWideSampler.Definition

/-!
Owns the constraint system that decodes one whole-vector PiRLC scalar.

Inputs: four caller-owned field expressions below the gadget offset.
Private layout: four canonical-u64 children, then 131 quotient bits `Q`,
three bits for each of the 54 base-five digits `D_j`, and ten bits for each
of the six check quotients `K_r`.
Rows: the children, Booleanity of every new bit, `D_j ≤ 4`, and for each
modulus `m_r` one check row
`Σ (c mod m) x + m B = Σ (e mod m) y + m K_r`,
where `Σ c x = Σ p^i h_i` and `Σ e y = Q N + Σ 5^j D_j`.

Soundness: every accepted assignment has `D = sample (h₀, h₁, h₂, h₃)`. The
witness program of the new bits is a parameter; soundness holds for every
witness value. This module does not own the witness program or completeness.
-/

namespace NightstreamFPrime.Gadgets.Sampling.WideReduction

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Gadgets.Range
open NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcWideSampler
open NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcSampler
open ProductionAlphabet ProductionStrongSet

/-! ### Parameters and layout -/

abbrev fieldCount : Nat := 4
def childWidth : Nat := CanonicalU64.auxiliaryCount
def quotientBitCount : Nat := 131
def digitCount : Nat := 54
def digitBitCount : Nat := 3
abbrev checkCount : Nat := 6
def checkBitCount : Nat := 10
/-- Number of result-side terms; it keeps honest check quotients nonnegative. -/
def checkBias : Nat := quotientBitCount + digitCount * digitBitCount
def newBitCount : Nat :=
  quotientBitCount + digitCount * digitBitCount + checkCount * checkBitCount
def privateCount : Nat := childWidth * fieldCount + newBitCount

/-- Pairwise coprime check moduli below `2^50`. -/
def modulus : Fin checkCount → Nat :=
  ![1125899906842623, 1125899906842621, 1125899906842619,
    1125899906842615, 1125899906842613, 1125899906842609]

theorem modulus_lt (index : Fin checkCount) : modulus index < 2 ^ 50 := by
  fin_cases index <;> decide

theorem modulus_pos (index : Fin checkCount) : 0 < modulus index := by
  fin_cases index <;> decide

def childOffset (offset index : Nat) : Nat := offset + childWidth * index
def quotientStart (offset : Nat) : Nat := offset + childWidth * fieldCount
def digitStart (offset : Nat) : Nat := quotientStart offset + quotientBitCount
def checkStart (offset : Nat) : Nat := digitStart offset + digitCount * digitBitCount

def fieldBit (offset index bit : Nat) : Expr := CanonicalU64.bitExpr (childOffset offset index) bit
def quotientBit (offset bit : Nat) : Expr := Expr.var (quotientStart offset + bit)
def digitBit (offset digit bit : Nat) : Expr :=
  Expr.var (digitStart offset + digitBitCount * digit + bit)
def checkBit (offset check bit : Nat) : Expr :=
  Expr.var (checkStart offset + checkBitCount * check + bit)

structure Interface where
  source : Fin fieldCount → Nat → Expr

def childInterface (interface : Interface) (parentOffset : Nat) (index : Fin fieldCount) :
    CanonicalU64.Interface where
  source := fun _ => interface.source index parentOffset

/-! ### Linear forms -/

def fieldTerms (offset index : Nat) : List (Nat × Expr) :=
  (List.range CanonicalU64.bitCount).map fun bit =>
    (2 ^ bit * goldilocksModulus ^ index, fieldBit offset index bit)

def drawTerms (offset : Nat) : List (Nat × Expr) :=
  fieldTerms offset 0 ++ fieldTerms offset 1 ++ fieldTerms offset 2 ++ fieldTerms offset 3

def quotientTerms (offset : Nat) : List (Nat × Expr) :=
  (List.range quotientBitCount).map fun bit => (2 ^ bit * scalarCount, quotientBit offset bit)

def digitTerms (offset digit : Nat) : List (Nat × Expr) :=
  (List.range digitBitCount).map fun bit => (2 ^ bit * 5 ^ digit, digitBit offset digit bit)

def resultTerms (offset : Nat) : List (Nat × Expr) :=
  quotientTerms offset ++ (List.range digitCount).flatMap (digitTerms offset)

def checkTerms (offset : Nat) (check : Fin checkCount) : List (Nat × Expr) :=
  (List.range checkBitCount).map fun bit => (modulus check * 2 ^ bit, checkBit offset check bit)

/-! ### Rows and operations -/

def booleanRow (atom : Expr) : Expr := atom * (atom - 1)

def digitRangeRow (offset digit : Nat) : Expr :=
  digitBit offset digit 2 * (digitBit offset digit 0 + digitBit offset digit 1)

def checkRow (offset : Nat) (check : Fin checkCount) : Expr :=
  (linearExpr (reduceTerms (modulus check) (drawTerms offset)) +
      Expr.const (fieldOfNat (modulus check * checkBias))) -
    (linearExpr (reduceTerms (modulus check) (resultTerms offset)) +
      linearExpr (checkTerms offset check))

/-- Every new bit, in allocation order. -/
def newBits (offset : Nat) : List Expr :=
  (List.range quotientBitCount).map (quotientBit offset) ++
    (List.range digitCount).flatMap (fun digit => (List.range digitBitCount).map (digitBit offset digit)) ++
    (List.finRange checkCount).flatMap
      (fun check => (List.range checkBitCount).map (checkBit offset check.val))

def childName (index : Fin fieldCount) : String :=
  "pirlc.wide_sampler.canonical_u64." ++ toString index.val

def childOps (interface : Interface) (offset : Nat) : List Op :=
  (List.finRange fieldCount).map fun index =>
    Sequence.childOp (childName index) (CanonicalU64.circuit (childInterface interface offset index))
      (childOffset offset index)

def rowOps (offset : Nat) : List Op :=
  (newBits offset).map (fun atom => Op.assertZero (booleanRow atom)) ++
    (List.range digitCount).map (fun digit => Op.assertZero (digitRangeRow offset digit)) ++
    (List.finRange checkCount).map (fun check => Op.assertZero (checkRow offset check))

/-- The witness program of the new bits is supplied by the caller. -/
def operations (interface : Interface) (hints : Nat → List Hint) (offset : Nat) : List Op :=
  childOps interface offset ++
    [Op.witness (WitnessBatch.hinted (quotientStart offset) (hints offset))] ++ rowOps offset

/-! ### Specification -/

def drawOf (interface : Interface) (env : Env) (offset : Nat) : Draw :=
  fun index => (interface.source index offset).eval env

/-- Value `b₀ + 2 b₁ + 4 b₂` of digit `j`. -/
def digitValue (env : Env) (offset digit : Nat) : Nat :=
  ((digitBit offset digit 0).eval env).val + 2 * ((digitBit offset digit 1).eval env).val +
    4 * ((digitBit offset digit 2).eval env).val

def Assumptions (interface : Interface) (offset : Nat) : Prop :=
  ∀ index, (interface.source index offset).VarsBelow offset

/-- Every digit is the sampled coefficient. -/
def SpecHolds (interface : Interface) (offset : Nat) (env : Env) : Prop :=
  ∀ digit : Fin coefficientCount,
    digitValue env offset digit.val = (sample (drawOf interface env offset) digit).val

end NightstreamFPrime.Gadgets.Sampling.WideReduction
