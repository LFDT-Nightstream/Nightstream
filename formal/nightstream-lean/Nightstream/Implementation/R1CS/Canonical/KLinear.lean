import Nightstream.Implementation.R1CS.Canonical.KPairLaws
import Nightstream.Implementation.R1CS.Canonical.KConcreteBridge
import Nightstream.Implementation.R1CS.Core.LinearSubstitution

/-!
Contract: row-free linear arithmetic on carried quadratic-extension values.

Owns: zero, one, addition, public scaling, subtraction, and their exact
evaluation laws. None of these operations emits a row or allocates a column;
they only rewrite linear-combination coefficients.

The constant-one wire is consumed only by `oneCarried`. Subtraction uses the
coefficient `p - 1`, with an explicit proof that it is modular negation.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Canonical.KLinear

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.Implementation.R1CS.Canonical.KMul
open Nightstream.Implementation.R1CS.Canonical.KHorner
open Nightstream.SuperNeo.Concrete

def zeroCarried : Carried := ⟨[], []⟩

def oneCarried : Carried := ⟨[(0, 1)], []⟩

/-- A verifier-owned concrete extension constant, compiled as coefficients on
the constant-one wire. -/
def constantCarried (value : K) : Carried :=
  ⟨[(0, value.c0.val)], [(0, value.c1.val)]⟩

def addCarried (left right : Carried) : Carried :=
  ⟨left.low ++ right.low, left.high ++ right.high⟩

def scaleCarried (scalar : Nat) (value : Carried) : Carried :=
  ⟨LinearSubstitution.scaleTerms scalar value.low,
    LinearSubstitution.scaleTerms scalar value.high⟩

def subCarried (left right : Carried) : Carried :=
  addCarried left (scaleCarried (goldilocksP - 1) right)

def oneMinus (value : Carried) : Carried :=
  subCarried oneCarried value

theorem scaled_term_mod (scalar coefficient value : Nat) :
    (scalar * coefficient % goldilocksP) * value % goldilocksP =
      scalar * (coefficient * value) % goldilocksP := by
  rw [Nat.mul_mod, Nat.mod_mod, ← Nat.mul_mod, Nat.mul_assoc]

theorem rawSum_scaleTerms (assignment : Nat → Nat) (scalar : Nat)
    (combination : LinComb) :
    rawSum assignment (LinearSubstitution.scaleTerms scalar combination) %
        goldilocksP =
      scalar * rawSum assignment combination % goldilocksP := by
  induction combination with
  | nil => simp [LinearSubstitution.scaleTerms, rawSum]
  | cons term rest inductionHypothesis =>
      have unfoldCons :
          LinearSubstitution.scaleTerms scalar (term :: rest) =
            (term.1, scalar * term.2 % goldilocksP) ::
              LinearSubstitution.scaleTerms scalar rest := rfl
      rw [unfoldCons, rawSum_cons, rawSum_cons, Nat.add_mod,
        inductionHypothesis, scaled_term_mod, Nat.mul_add, Nat.add_mod,
        Nat.mod_mod]
      simp only [Nat.mod_mod, ← Nat.add_mod]

theorem lcEval_scaleTerms (assignment : Nat → Nat) (scalar : Nat)
    (combination : LinComb) :
    lcEval assignment (LinearSubstitution.scaleTerms scalar combination) =
      scalar * lcEval assignment combination % goldilocksP := by
  rw [lcEval_eq_rawSum, rawSum_scaleTerms, lcEval_eq_rawSum,
    KMul.mul_mod_right_reduce]

theorem minusOne_mul_mod (value : Nat) :
    (goldilocksP - 1) * value % goldilocksP =
      (goldilocksP - value % goldilocksP) % goldilocksP := by
  calc
    (goldilocksP - 1) * value % goldilocksP =
        (goldilocksP - 1) * (value % goldilocksP) % goldilocksP :=
      (KMul.mul_mod_right_reduce _ _).symm
    _ = (goldilocksP - value % goldilocksP) % goldilocksP := by
      generalize residueEq : value % goldilocksP = residue
      have residueLt : residue < goldilocksP := by
        rw [← residueEq]
        exact Nat.mod_lt _ (by decide)
      cases residue with
      | zero => simp
      | succ residue =>
          have expand :
              (goldilocksP - 1) * (residue + 1) =
                (goldilocksP - (residue + 1)) +
                  goldilocksP * residue := by
            rw [Nat.sub_one_mul, Nat.mul_succ]
            omega
          rw [expand, Nat.add_mul_mod_self_left,
            Nat.mod_eq_of_lt (by omega)]

theorem carriedValue_zero (assignment : Nat → Nat) :
    carriedValue assignment zeroCarried = ⟨0, 0⟩ := rfl

theorem carriedValue_one (assignment : Nat → Nat)
    (constantWire : assignment 0 = 1) :
    carriedValue assignment oneCarried = ⟨1, 0⟩ := by
  unfold carriedValue oneCarried
  simp only [Pair.mk.injEq]
  constructor
  · rw [KMul.lcEval_singleton_col, constantWire]
    decide
  · rfl

theorem carriedValue_constant (assignment : Nat → Nat) (value : K)
    (constantWire : assignment 0 = 1) :
    carriedValue assignment (constantCarried value) =
      KConcreteBridge.ofConcrete value := by
  unfold carriedValue constantCarried KConcreteBridge.ofConcrete
  simp only [Pair.mk.injEq]
  constructor
  · have coordinateLt : value.c0.val < goldilocksP := by
      rw [← KConcreteBridge.moduli_eq]
      exact value.c0.isLt
    simp [lcEval, constantWire, Nat.mod_eq_of_lt coordinateLt]
  · have coordinateLt : value.c1.val < goldilocksP := by
      rw [← KConcreteBridge.moduli_eq]
      exact value.c1.isLt
    simp [lcEval, constantWire, Nat.mod_eq_of_lt coordinateLt]

theorem carriedValue_add (assignment : Nat → Nat) (left right : Carried) :
    carriedValue assignment (addCarried left right) =
      addPair (carriedValue assignment left) (carriedValue assignment right) := by
  unfold carriedValue addCarried addPair
  simp only [Pair.mk.injEq]
  exact ⟨KHorner.lcEval_append _ _ _, KHorner.lcEval_append _ _ _⟩

theorem carriedValue_scale (assignment : Nat → Nat) (scalar : Nat)
    (value : Carried) :
    carriedValue assignment (scaleCarried scalar value) =
      ⟨scalar * (carriedValue assignment value).low % goldilocksP,
        scalar * (carriedValue assignment value).high % goldilocksP⟩ := by
  unfold carriedValue scaleCarried
  simp only [Pair.mk.injEq]
  exact ⟨lcEval_scaleTerms _ _ _, lcEval_scaleTerms _ _ _⟩

theorem lcEval_sub (assignment : Nat → Nat) (left right : LinComb) :
    lcEval assignment
        (left ++ LinearSubstitution.scaleTerms (goldilocksP - 1) right) =
      (lcEval assignment left +
        (goldilocksP - lcEval assignment right % goldilocksP)) %
        goldilocksP := by
  rw [KHorner.lcEval_append, lcEval_scaleTerms, minusOne_mul_mod]
  rw [Nat.add_mod, Nat.mod_mod, ← Nat.add_mod]

theorem carriedValue_sub (assignment : Nat → Nat) (left right : Carried) :
    carriedValue assignment (subCarried left right) =
      KPairLaws.subPair
        (carriedValue assignment left) (carriedValue assignment right) := by
  unfold carriedValue subCarried addCarried scaleCarried KPairLaws.subPair
  simp only [Pair.mk.injEq]
  exact ⟨lcEval_sub _ _ _, lcEval_sub _ _ _⟩

theorem carriedValue_oneMinus (assignment : Nat → Nat) (value : Carried)
    (constantWire : assignment 0 = 1) :
    carriedValue assignment (oneMinus value) =
      KPairLaws.subPair ⟨1, 0⟩ (carriedValue assignment value) := by
  rw [oneMinus, carriedValue_sub, carriedValue_one assignment constantWire]

end Nightstream.Implementation.R1CS.Canonical.KLinear
