import Nightstream.Implementation.Lowering.Goldilocks.NativeCcsSelector
import Nightstream.Implementation.Lowering.Nebula.StepPolynomial

/-!
Combined sparse CCS polynomial for the modular 42-times-6 deployment.

Assurance tier: model-level.

This file embeds the existing four-role F-prime selector polynomial and the
existing fifteen-role Nebula memory polynomial into one disjoint nineteen-role
CCS relation. The term lists are lifted from their owners. They are not copied.

It owns the combined matrix order, sparse syntax, degree bound, and exact
evaluation split. It does not own matrix entries, column placement, the
recursive fixed point, a Rust manifest, Fiat--Shamir challenges, or a security
reduction.
-/

set_option autoImplicit false
set_option maxRecDepth 20000

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.NebulaCombinedPolynomial

open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.Nebula
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.CCSResidualTable
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier

/-- Four F-prime roles followed by fifteen Nebula roles. -/
def matrixCount : Nat :=
  NativeCcsSelector.matrixCount + StepPolynomial.matrixCount

/-- Embed one F-prime matrix position in the combined prefix. -/
def nativeIndex
    (index : Fin NativeCcsSelector.matrixCount) : Fin matrixCount :=
  ⟨index.val, by
    have bound := index.isLt
    simp only [matrixCount] at bound ⊢
    omega⟩

/-- Embed one Nebula matrix position after the four F-prime positions. -/
def nebulaIndex
    (index : Fin StepPolynomial.matrixCount) : Fin matrixCount :=
  ⟨NativeCcsSelector.matrixCount + index.val, by
    have bound := index.isLt
    simp only [matrixCount] at bound ⊢
    omega⟩

/-- Read the F-prime prefix of one combined matrix-image point. -/
def nativePoint (point : Fin matrixCount -> F) :
    Fin NativeCcsSelector.matrixCount -> F :=
  fun index => point (nativeIndex index)

/-- Read the Nebula suffix of one combined matrix-image point. -/
def nebulaPoint (point : Fin matrixCount -> F) :
    Fin StepPolynomial.matrixCount -> F :=
  fun index => point (nebulaIndex index)

def liftNativeExponents
    (exponents : Fin NativeCcsSelector.matrixCount -> Nat) :
    Fin matrixCount -> Nat
  | ⟨0, _⟩ => exponents ⟨0, by decide⟩
  | ⟨1, _⟩ => exponents ⟨1, by decide⟩
  | ⟨2, _⟩ => exponents ⟨2, by decide⟩
  | ⟨3, _⟩ => exponents ⟨3, by decide⟩
  | ⟨4, _⟩ => 0
  | ⟨5, _⟩ => 0
  | ⟨6, _⟩ => 0
  | ⟨7, _⟩ => 0
  | ⟨8, _⟩ => 0
  | ⟨9, _⟩ => 0
  | ⟨10, _⟩ => 0
  | ⟨11, _⟩ => 0
  | ⟨12, _⟩ => 0
  | ⟨13, _⟩ => 0
  | ⟨14, _⟩ => 0
  | ⟨15, _⟩ => 0
  | ⟨16, _⟩ => 0
  | ⟨17, _⟩ => 0
  | ⟨18, _⟩ => 0
  | _ => 0

def liftNebulaExponents
    (exponents : Fin StepPolynomial.matrixCount -> Nat) :
    Fin matrixCount -> Nat
  | ⟨0, _⟩ => 0
  | ⟨1, _⟩ => 0
  | ⟨2, _⟩ => 0
  | ⟨3, _⟩ => 0
  | ⟨4, _⟩ => exponents ⟨0, by decide⟩
  | ⟨5, _⟩ => exponents ⟨1, by decide⟩
  | ⟨6, _⟩ => exponents ⟨2, by decide⟩
  | ⟨7, _⟩ => exponents ⟨3, by decide⟩
  | ⟨8, _⟩ => exponents ⟨4, by decide⟩
  | ⟨9, _⟩ => exponents ⟨5, by decide⟩
  | ⟨10, _⟩ => exponents ⟨6, by decide⟩
  | ⟨11, _⟩ => exponents ⟨7, by decide⟩
  | ⟨12, _⟩ => exponents ⟨8, by decide⟩
  | ⟨13, _⟩ => exponents ⟨9, by decide⟩
  | ⟨14, _⟩ => exponents ⟨10, by decide⟩
  | ⟨15, _⟩ => exponents ⟨11, by decide⟩
  | ⟨16, _⟩ => exponents ⟨12, by decide⟩
  | ⟨17, _⟩ => exponents ⟨13, by decide⟩
  | ⟨18, _⟩ => exponents ⟨14, by decide⟩
  | _ => 0

def liftNativeMonomial
    (term : Monomial F NativeCcsSelector.matrixCount) :
    Monomial F matrixCount where
  coefficient := term.coefficient
  exponents := liftNativeExponents term.exponents

def liftNebulaMonomial
    (term : Monomial F StepPolynomial.matrixCount) :
    Monomial F matrixCount where
  coefficient := term.coefficient
  exponents := liftNebulaExponents term.exponents

/-- Exact sparse syntax: two existing F-prime terms, then eleven existing
Nebula terms. -/
def terms : List (Monomial F matrixCount) :=
  NativeCcsSelector.constraintPolynomial.terms.map liftNativeMonomial ++
    StepPolynomial.polynomial.terms.map liftNebulaMonomial

theorem terms_exact :
    terms =
      NativeCcsSelector.constraintPolynomial.terms.map liftNativeMonomial ++
        StepPolynomial.polynomial.terms.map liftNebulaMonomial :=
  rfl

theorem term_count_exact : terms.length = 13 := by
  rfl

private theorem every_term_degree_checked :
    terms.all (fun term => decide (term.totalDegree < 5)) = true := by
  decide

/-- The combined relation still has degree four and therefore uses strict
degree bound five. -/
def polynomial : ConstraintPolynomial F matrixCount where
  degreeBound := 5
  terms := terms
  termsBelowDegree := by
    intro term member
    exact of_decide_eq_true
      ((List.all_eq_true.mp every_term_degree_checked) term member)

theorem matrixCount_exact : matrixCount = 19 := by
  rfl

theorem canonicalEqualityGatedDegreeBound_exact :
    polynomial.canonicalEqualityGatedDegreeBound = 5 := by
  decide

/-- Direct evaluation of the combined sparse syntax. -/
def evaluate (point : Fin matrixCount -> F) : F :=
  evaluatePolynomial baseOps polynomial point

private theorem evaluateMonomial_liftNative
    (term : Monomial F NativeCcsSelector.matrixCount)
    (point : Fin matrixCount -> F) :
    evaluateMonomial baseOps (liftNativeMonomial term) point =
      evaluateMonomial baseOps term (nativePoint point) := by
  simp [evaluateMonomial, canonicalFinIndices, liftNativeMonomial,
    liftNativeExponents, nativePoint, nativeIndex, matrixCount,
    NativeCcsSelector.matrixCount, StepPolynomial.matrixCount, pow, baseOps,
    Fin.mul_one] <;> congr 1

private theorem evaluateMonomial_liftNebula
    (term : Monomial F StepPolynomial.matrixCount)
    (point : Fin matrixCount -> F) :
    evaluateMonomial baseOps (liftNebulaMonomial term) point =
      evaluateMonomial baseOps term (nebulaPoint point) := by
  simp [evaluateMonomial, canonicalFinIndices, liftNebulaMonomial,
    liftNebulaExponents, nebulaPoint, nebulaIndex, matrixCount,
    NativeCcsSelector.matrixCount, StepPolynomial.matrixCount, pow, baseOps,
    Fin.mul_one] <;> congr 1

/-- Evaluation splits into the two source relations. No cross-family term is
introduced by the composition. -/
theorem evaluate_eq_components (point : Fin matrixCount -> F) :
    evaluate point =
      NativeCcsSelector.evaluate (nativePoint point) +
        StepPolynomial.evaluate (nebulaPoint point) := by
  simp only [evaluate, polynomial, terms, evaluatePolynomial,
    List.foldl_append, List.foldl_map]
  simp only [evaluateMonomial_liftNative, evaluateMonomial_liftNebula]
  unfold NativeCcsSelector.evaluate StepPolynomial.evaluate
  simp only [evaluatePolynomial, NativeCcsSelector.constraintPolynomial,
    StepPolynomial.polynomial, StepPolynomial.terms, List.foldl_cons,
    List.foldl_nil, baseOps]
  simp only [Fin.zero_add, Lean.Grind.Fin.add_assoc]

/-- A point whose Nebula suffix is zero satisfies the combined relation
exactly when its F-prime prefix has zero residual. -/
theorem evaluate_native_only
    (point : Fin matrixCount -> F)
    (nebulaZero : forall index, nebulaPoint point index = 0) :
    evaluate point = NativeCcsSelector.evaluate (nativePoint point) := by
  rw [evaluate_eq_components]
  have zeroPoint : nebulaPoint point = fun _ => 0 := by
    funext index
    exact nebulaZero index
  rw [zeroPoint, StepPolynomial.evaluate_eq_residual]
  simp [StepPolynomial.residual, Fin.mul_zero, Fin.zero_mul, Fin.add_zero,
    Fin.zero_add, Lean.Grind.AddCommGroup.neg_zero]

/-- A point whose F-prime prefix is zero satisfies the combined relation
exactly when its Nebula suffix has zero residual. -/
theorem evaluate_nebula_only
    (point : Fin matrixCount -> F)
    (nativeZero : forall index, nativePoint point index = 0) :
    evaluate point = StepPolynomial.evaluate (nebulaPoint point) := by
  rw [evaluate_eq_components]
  have zeroPoint : nativePoint point = fun _ => 0 := by
    funext index
    exact nativeZero index
  rw [zeroPoint, NativeCcsSelector.evaluate_exact]
  simp [NativeCcsSelector.polynomial, Fin.zero_add, Fin.zero_mul,
    Fin.mul_zero]

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.NebulaCombinedPolynomial
