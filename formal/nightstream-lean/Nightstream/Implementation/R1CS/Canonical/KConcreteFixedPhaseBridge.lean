import Nightstream.Implementation.R1CS.Canonical.KConcreteBridge
import Nightstream.Implementation.R1CS.Canonical.KFixedPhaseSumCheckSupport
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier.Algebra

/-!
Contract: representation refinement from the paper's concrete quadratic
extension carrier to the canonical row layer's extension carrier, specialized
to fixed-phase SumCheck.

Owns:
- the coordinate-preserving map from `SuperNeo.Concrete.K` to
  `ProjectionProgram.K`;
- preservation of zero, one, addition, and multiplication;
- fixed-polynomial evaluation transport; and
- transport of the exact frozen `FixedPhase.Chain`.

This module changes representation only. It does not choose a protocol shape,
derive transcript challenges, or emit rows.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Canonical.KConcreteFixedPhaseBridge

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.SumCheck.Finite
open Nightstream.Implementation.R1CS.Canonical.KFixedPhaseSumCheck

/-- The paper and row carriers have the same two canonical Goldilocks
coordinates. -/
def toProjection
    (value : Nightstream.SuperNeo.Concrete.K) :
    Nightstream.Implementation.R1CS.ProjectionProgram.K where
  c0 := ⟨value.c0.val, by exact value.c0.isLt⟩
  c1 := ⟨value.c1.val, by exact value.c1.isLt⟩

/-- The inverse coordinate map.  It is used only to state authoritative
paper-carrier values decoded from canonical row columns. -/
def ofProjection
    (value : Nightstream.Implementation.R1CS.ProjectionProgram.K) :
    Nightstream.SuperNeo.Concrete.K where
  c0 := ⟨value.c0.val, by exact value.c0.isLt⟩
  c1 := ⟨value.c1.val, by exact value.c1.isLt⟩

@[simp] theorem toProjection_ofProjection
    (value : Nightstream.Implementation.R1CS.ProjectionProgram.K) :
    toProjection (ofProjection value) = value := by
  cases value
  rfl

@[simp] theorem ofProjection_toProjection
    (value : Nightstream.SuperNeo.Concrete.K) :
    ofProjection (toProjection value) = value := by
  cases value
  rfl

@[simp] theorem toPair_toProjection
    (value : Nightstream.SuperNeo.Concrete.K) :
    KBridge.toPair (toProjection value) =
      KConcreteBridge.ofConcrete value := by
  rfl

theorem toProjection_injective
    {left right : Nightstream.SuperNeo.Concrete.K}
    (equal : toProjection left = toProjection right) :
    left = right := by
  have coordinates := congrArg KBridge.toPair equal
  simp only [toPair_toProjection, KConcreteBridge.ofConcrete,
    KHorner.Pair.mk.injEq] at coordinates
  cases left
  cases right
  simp only [Nightstream.SuperNeo.Concrete.K.mk.injEq]
  exact ⟨Fin.ext coordinates.1, Fin.ext coordinates.2⟩

theorem toProjection_zero :
    toProjection Nightstream.SuperNeo.Concrete.K.zero =
      Nightstream.Implementation.R1CS.ProjectionProgram.K.zero := by
  apply KBridge.toPair_injective
  rw [toPair_toProjection, KConcreteBridge.ofConcrete_zero,
    KBridge.toPair_zero]

theorem toProjection_one :
    toProjection Nightstream.SuperNeo.Concrete.K.one =
      Nightstream.Implementation.R1CS.ProjectionProgram.K.one := by
  rfl

theorem toProjection_add
    (left right : Nightstream.SuperNeo.Concrete.K) :
    toProjection (Nightstream.SuperNeo.Concrete.K.add left right) =
      Nightstream.Implementation.R1CS.ProjectionProgram.K.add
        (toProjection left) (toProjection right) := by
  apply KBridge.toPair_injective
  rw [toPair_toProjection, KConcreteBridge.ofConcrete_add,
    KBridge.toPair_add, toPair_toProjection, toPair_toProjection]

theorem toProjection_mul
    (left right : Nightstream.SuperNeo.Concrete.K) :
    toProjection (Nightstream.SuperNeo.Concrete.K.mul left right) =
      Nightstream.Implementation.R1CS.ProjectionProgram.K.mul
        (toProjection left) (toProjection right) := by
  apply KBridge.toPair_injective
  rw [toPair_toProjection, KConcreteBridge.ofConcrete_mul,
    KBridge.toPair_mul, toPair_toProjection, toPair_toProjection]

/-- Map a fixed-width paper polynomial without changing its width. -/
def mapPolynomial {degree : Nat}
    (polynomial :
      FixedPolynomial Nightstream.SuperNeo.Concrete.K degree) :
    FixedPolynomial
      Nightstream.Implementation.R1CS.ProjectionProgram.K degree where
  coefficients := polynomial.coefficients.map toProjection
  coefficients_length := by
    rw [List.length_map, polynomial.coefficients_length]

theorem evaluateCoefficients_map
    (point : Nightstream.SuperNeo.Concrete.K) :
    ∀ coefficients : List Nightstream.SuperNeo.Concrete.K,
      toProjection
          (Message.evaluateCoefficients
            ConcreteCarrier.extensionOps.toOps point coefficients) =
        Message.evaluateCoefficients sumCheckOps (toProjection point)
          (coefficients.map toProjection)
  | [] => toProjection_zero
  | coefficient :: rest => by
      change
        toProjection
            (Nightstream.SuperNeo.Concrete.K.add coefficient
              (Nightstream.SuperNeo.Concrete.K.mul point
                (Message.evaluateCoefficients
                  ConcreteCarrier.extensionOps.toOps point rest))) =
          Nightstream.Implementation.R1CS.ProjectionProgram.K.add
            (toProjection coefficient)
            (Nightstream.Implementation.R1CS.ProjectionProgram.K.mul
              (toProjection point)
              (Message.evaluateCoefficients sumCheckOps
                (toProjection point) (rest.map toProjection)))
      rw [toProjection_add, toProjection_mul,
        evaluateCoefficients_map point rest]

/-- Fixed-polynomial evaluation commutes with the representation map. -/
theorem evaluate_mapPolynomial
    {degree : Nat}
    (polynomial :
      FixedPolynomial Nightstream.SuperNeo.Concrete.K degree)
    (point : Nightstream.SuperNeo.Concrete.K) :
    FixedPolynomial.evaluate sumCheckOps (mapPolynomial polynomial)
        (toProjection point) =
      toProjection
        (FixedPolynomial.evaluate ConcreteCarrier.extensionOps.toOps
          polynomial point) := by
  unfold FixedPolynomial.evaluate FixedPolynomial.toMessage Message.evaluate
  exact (evaluateCoefficients_map point polynomial.coefficients).symm

/-- **The row carrier checks exactly the paper carrier's fixed-phase chain.**

This is a one-way representation transport, not a soundness assumption. Every
operation used by `Chain` was proved preserved above. -/
theorem chain_toProjection
    {degree : Nat}
    (current terminal : Nightstream.SuperNeo.Concrete.K)
    (rounds :
      List (FixedPolynomial Nightstream.SuperNeo.Concrete.K degree))
    (challenges : List Nightstream.SuperNeo.Concrete.K)
    (chain :
      FixedPhase.Chain ConcreteCarrier.extensionOps.toOps
        current rounds challenges terminal) :
    FixedPhase.Chain sumCheckOps
      (toProjection current)
      (rounds.map mapPolynomial)
      (challenges.map toProjection)
      (toProjection terminal) := by
  induction rounds generalizing current challenges with
  | nil =>
      cases challenges with
      | nil =>
          change toProjection current = toProjection terminal
          exact congrArg toProjection chain
      | cons _ _ => simp [FixedPhase.Chain] at chain
  | cons polynomial polynomials inductionHypothesis =>
      cases challenges with
      | nil => simp [FixedPhase.Chain] at chain
      | cons challenge challenges =>
          simp only [List.map_cons, FixedPhase.Chain] at chain ⊢
          constructor
          · have zeroEvaluation :=
              evaluate_mapPolynomial polynomial
                Nightstream.SuperNeo.Concrete.K.zero
            have oneEvaluation :=
              evaluate_mapPolynomial polynomial
                Nightstream.SuperNeo.Concrete.K.one
            rw [toProjection_zero] at zeroEvaluation
            rw [toProjection_one] at oneEvaluation
            change
              toProjection current =
                Nightstream.Implementation.R1CS.ProjectionProgram.K.add
                  (FixedPolynomial.evaluate sumCheckOps
                    (mapPolynomial polynomial)
                    Nightstream.Implementation.R1CS.ProjectionProgram.K.zero)
                  (FixedPolynomial.evaluate sumCheckOps
                    (mapPolynomial polynomial)
                    Nightstream.Implementation.R1CS.ProjectionProgram.K.one)
            rw [zeroEvaluation, oneEvaluation, ← toProjection_add]
            exact congrArg toProjection chain.1
          · rw [evaluate_mapPolynomial]
            exact inductionHypothesis _ challenges chain.2

/-- Reflection of a row-carrier chain back into the paper carrier.  The
representation map is injective, so the physical checker cannot accept a
different fixed-phase chain merely because both carriers share coordinates. -/
theorem chain_of_toProjection
    {degree : Nat}
    (current terminal : Nightstream.SuperNeo.Concrete.K)
    (rounds :
      List (FixedPolynomial Nightstream.SuperNeo.Concrete.K degree))
    (challenges : List Nightstream.SuperNeo.Concrete.K)
    (chain :
      FixedPhase.Chain sumCheckOps
        (toProjection current)
        (rounds.map mapPolynomial)
        (challenges.map toProjection)
        (toProjection terminal)) :
    FixedPhase.Chain ConcreteCarrier.extensionOps.toOps
      current rounds challenges terminal := by
  induction rounds generalizing current challenges with
  | nil =>
      cases challenges with
      | nil =>
          exact toProjection_injective chain
      | cons _ _ => simp [FixedPhase.Chain] at chain
  | cons polynomial polynomials inductionHypothesis =>
      cases challenges with
      | nil => simp [FixedPhase.Chain] at chain
      | cons challenge challenges =>
          simp only [List.map_cons, FixedPhase.Chain] at chain ⊢
          constructor
          · apply toProjection_injective
            change
              toProjection current =
                toProjection
                  (Nightstream.SuperNeo.Concrete.K.add
                    (FixedPolynomial.evaluate
                      ConcreteCarrier.extensionOps.toOps polynomial
                      Nightstream.SuperNeo.Concrete.K.zero)
                    (FixedPolynomial.evaluate
                      ConcreteCarrier.extensionOps.toOps polynomial
                      Nightstream.SuperNeo.Concrete.K.one))
            rw [toProjection_add, ← evaluate_mapPolynomial,
              ← evaluate_mapPolynomial, toProjection_zero, toProjection_one]
            exact chain.1
          · rw [evaluate_mapPolynomial] at chain
            exact inductionHypothesis _ challenges chain.2

theorem chain_iff_toProjection
    {degree : Nat}
    (current terminal : Nightstream.SuperNeo.Concrete.K)
    (rounds :
      List (FixedPolynomial Nightstream.SuperNeo.Concrete.K degree))
    (challenges : List Nightstream.SuperNeo.Concrete.K) :
    FixedPhase.Chain ConcreteCarrier.extensionOps.toOps
        current rounds challenges terminal ↔
      FixedPhase.Chain sumCheckOps
        (toProjection current)
        (rounds.map mapPolynomial)
        (challenges.map toProjection)
        (toProjection terminal) :=
  ⟨chain_toProjection current terminal rounds challenges,
    chain_of_toProjection current terminal rounds challenges⟩

end Nightstream.Implementation.R1CS.Canonical.KConcreteFixedPhaseBridge
