import NightstreamFPrime.Gadgets.Polynomial.Horner
import NightstreamFPrime.Lifecycle.PiCCS.v1_2.ChallengeDerivation
import NightstreamFPrime.Spec.Folding.PiCCS.FinalIdentity

/-!
Paper authority: SuperNeo v1.2, Section 7.3, Step 2, claimed sum `T`.
Obligation: Enforce the exact initial SumCheck claim
`T_K + γ^(k*d) · T_A`.

Inputs:
- verifier-derived `γ`;
- separate public `Eval_K` and `Eval_A` coefficient families.

Outputs:
- one child-owned initial SumCheck claim consumed by the fixed chain.

Constraint groups:
- C1: canonical Horner evaluation of `Eval_K ++ Eval_A`;

Parent coverage:
- `ProtocolPolynomial.VerifierInput.initial` inside production `piCcsCheck`.
-/

namespace NightstreamFPrime.Lifecycle.PiCCS.v1_2.InitialClaim

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Gadgets.Polynomial
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier

/-- The leaf's inputs for a Π_CCS shape: the production shape, or the bounded-depth scheme's light shape with
one running source. -/
structure ShapeInterface (shape : Shape) where
  gamma : Nat → KExpr
  eval_K : Nat → PadCoordinate shape → KExpr
  eval_A : Nat → MatrixCoordinate shape → KExpr

abbrev Interface := ShapeInterface productionShape

section Shape

variable {shape : Shape}

def coefficientExprs (interface : ShapeInterface shape) (offset : Nat) : List KExpr :=
  (canonicalPadCoordinates shape).map (interface.eval_K offset) ++
    (canonicalMatrixCoordinates shape).map (interface.eval_A offset)

def ownedInterface (interface : ShapeInterface shape) : Horner.Owned.Interface where
  point := interface.gamma
  coefficients := coefficientExprs interface

def program (interface : ShapeInterface shape) (offset : Nat) : Horner.Program :=
  Horner.Owned.program (ownedInterface interface) offset

def output (interface : ShapeInterface shape) (offset : Nat) : KExpr :=
  Horner.Owned.output (ownedInterface interface) offset

abbrev Assumptions (interface : ShapeInterface shape) (offset : Nat) (env : Env) : Prop :=
  Horner.Owned.Assumptions (ownedInterface interface) offset env

abbrev SpecHolds (interface : ShapeInterface shape) (offset : Nat) (env : Env) : Prop :=
  Horner.Owned.SpecHolds (ownedInterface interface) offset env

def circuit (interface : ShapeInterface shape) : FormalCircuit :=
  Horner.Owned.circuit (ownedInterface interface)

theorem soundness (interface : ShapeInterface shape) (env : Env) (offset : Nat)
    (assumptions : Assumptions interface offset env)
    (rows : holds env (Circuit.ops (circuit interface).main offset)) :
    SpecHolds interface offset env :=
  Horner.Owned.soundness (ownedInterface interface) env offset assumptions rows

theorem build (interface : ShapeInterface shape) (env : Env) (offset : Nat)
    (assumptions : Assumptions interface offset env) :
    ∃ completed,
      AgreesOutside env completed offset
        (localLength (Circuit.ops (circuit interface).main offset)) ∧
      holdsFlat completed (Circuit.ops (circuit interface).main offset) :=
  Horner.Owned.build (ownedInterface interface) env offset assumptions

theorem completeness (interface : ShapeInterface shape) (env : Env) (offset : Nat)
    (assumptions : Assumptions interface offset env)
    (_specification : SpecHolds interface offset env) :
    ∃ completed,
      AgreesOutside env completed offset
        (localLength (Circuit.ops (circuit interface).main offset)) ∧
      holdsFlat completed (Circuit.ops (circuit interface).main offset) :=
  build interface env offset assumptions

theorem flatConstraints_varsBelow (interface : ShapeInterface shape) (offset : Nat)
    (env : Env) (assumptions : Assumptions interface offset env) :
    ∀ expression ∈ flatConstraints
      (Circuit.ops (circuit interface).main offset),
      expression.VarsBelow
        (offset + localLength (Circuit.ops (circuit interface).main offset)) := by
  exact Horner.Owned.flatConstraints_varsBelow
    (ownedInterface interface) offset env assumptions

/-- The leaf's output is the claimed sum `T` of SuperNeo v1.2 Section 7.3, Step 2, for any
verifier input whose claimed coefficients and `γ` the leaf's inputs carry. -/
theorem spec_implies_initial (interface : ShapeInterface shape) (offset : Nat) (env : Env)
    (input : ProtocolPolynomial.VerifierInput K shape) (gamma : K)
    (gamma_eq : (interface.gamma offset).eval env = gamma)
    (eval_K_eq : ∀ coordinate,
      (interface.eval_K offset coordinate).eval env = input.claimedPadCoefficient coordinate)
    (eval_A_eq : ∀ coordinate,
      (interface.eval_A offset coordinate).eval env = input.claimedMatrixCoefficient coordinate)
    (specification : SpecHolds interface offset env) :
    (output interface offset).eval env = input.initial extensionOps gamma := by
  have coefficients_eq :
      (coefficientExprs interface offset).map (KExpr.eval env) =
        NightstreamFPrime.Spec.Folding.PiCCS.FinalIdentity.targetCoefficientList input := by
    unfold coefficientExprs
      NightstreamFPrime.Spec.Folding.PiCCS.FinalIdentity.targetCoefficientList
    rw [List.map_append, List.map_map, List.map_map]
    apply congrArg₂ List.append
    · apply List.map_congr_left
      intro coordinate _
      exact eval_K_eq coordinate
    · apply List.map_congr_left
      intro coordinate _
      exact eval_A_eq coordinate
  change (output interface offset).eval env =
    SumCheck.Finite.Message.evaluateCoefficients extensionOps.toOps
      ((interface.gamma offset).eval env)
      ((coefficientExprs interface offset).map (KExpr.eval env)) at specification
  rw [gamma_eq, coefficients_eq] at specification
  exact specification.trans
    (NightstreamFPrime.Spec.Folding.PiCCS.FinalIdentity.evaluateTargetCoefficients_eq_initial
      extensionOps extensionLaws input gamma)

end Shape

theorem coefficientExprs_length (interface : Interface) (offset : Nat) :
    (coefficientExprs interface offset).length = 5120 := by
  simp only [coefficientExprs, List.length_append, List.length_map]
  simp [canonicalPadCoordinates_length,
    canonicalMatrixCoordinates_length, productionShape, productionProfile,
    RingMatrixSource.ringShape, Shape.padEvaluationCount,
    Shape.matrixEvaluationCount, cubeVariables, ringDegree]

def privateCount : Nat := 15357

theorem localLength_eq (interface : Interface) (offset : Nat) :
    localLength (Circuit.ops (circuit interface).main offset) = 15357 := by
  change localLength (Circuit.ops
    (Horner.Owned.circuit (ownedInterface interface)).main offset) = 15357
  rw [Horner.Owned.localLength_eq]
  change 3 * ((coefficientExprs interface offset).length - 1) = 15357
  rw [coefficientExprs_length]

theorem operations_length (interface : Interface) (offset : Nat) :
    (Circuit.ops (circuit interface).main offset).length = 1 :=
  Horner.Owned.operations_length (ownedInterface interface) offset

theorem flatConstraints_length (interface : Interface) (offset : Nat) :
    (flatConstraints (Circuit.ops (circuit interface).main offset)).length =
      15357 := by
  change (flatConstraints (Circuit.ops
    (Horner.Owned.circuit (ownedInterface interface)).main offset)).length = 15357
  rw [Horner.Owned.flatConstraints_length]
  change 3 * ((coefficientExprs interface offset).length - 1) = 15357
  rw [coefficientExprs_length]

theorem spec_implies_keyInitial
    {logicalWidth : Nat}
    {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
      RingCarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : PaperAlgebra.AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (running : Running
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (fresh : Fresh
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (proof : Proof (ProductionKey.degreeBound relation))
    (interface : Interface) (offset : Nat) (env : Env)
    (gamma_eq : (interface.gamma offset).eval env =
      ((ProductionKey.key relation ajtai).piCcsExecution
        running fresh proof).coins.gamma)
    (eval_K_eq : ∀ coordinate,
      (interface.eval_K offset coordinate).eval env =
        (ChallengeDerivation.productionContext
          relation ajtai running fresh).input.claimedPadCoefficient coordinate)
    (eval_A_eq : ∀ coordinate,
      (interface.eval_A offset coordinate).eval env =
        (ChallengeDerivation.productionContext
          relation ajtai running fresh).input.claimedMatrixCoefficient coordinate)
    (specification : SpecHolds interface offset env) :
    (output interface offset).eval env =
      (ChallengeDerivation.productionContext
        relation ajtai running fresh).input.initial extensionOps
          ((ProductionKey.key relation ajtai).piCcsExecution
            running fresh proof).coins.gamma := by
  let input := (ChallengeDerivation.productionContext
    relation ajtai running fresh).input
  let gamma := ((ProductionKey.key relation ajtai).piCcsExecution
    running fresh proof).coins.gamma
  have coefficients_eq :
      (coefficientExprs interface offset).map (KExpr.eval env) =
        NightstreamFPrime.Spec.Folding.PiCCS.FinalIdentity.targetCoefficientList
          input := by
    unfold coefficientExprs
      NightstreamFPrime.Spec.Folding.PiCCS.FinalIdentity.targetCoefficientList
    rw [List.map_append, List.map_map, List.map_map]
    apply congrArg₂ List.append
    · apply List.map_congr_left
      intro coordinate _
      exact eval_K_eq coordinate
    · apply List.map_congr_left
      intro coordinate _
      exact eval_A_eq coordinate
  change (output interface offset).eval env =
    SumCheck.Finite.Message.evaluateCoefficients extensionOps.toOps
      ((interface.gamma offset).eval env)
      ((coefficientExprs interface offset).map (KExpr.eval env)) at specification
  rw [gamma_eq, coefficients_eq] at specification
  exact specification.trans
    (NightstreamFPrime.Spec.Folding.PiCCS.FinalIdentity.evaluateTargetCoefficients_eq_initial
      extensionOps extensionLaws input gamma)

end NightstreamFPrime.Lifecycle.PiCCS.v1_2.InitialClaim
