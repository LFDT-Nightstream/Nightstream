import NightstreamFPrime.Gadgets.Polynomial.Horner
import NightstreamFPrime.Lifecycle.PiCCS.v1_1.ChallengeDerivation
import NightstreamFPrime.Spec.Folding.PiCCS.FinalIdentity

/-!
Paper authority: SuperNeo v1.1, Section 7.3, Step 2, claimed sum `T`.
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

This leaf reuses the generic Horner compiler and owns its output expression.
The parent wires that expression directly into the SumCheck chain. A file
boundary adds no expected-output copy rows.
-/

namespace NightstreamFPrime.Lifecycle.PiCCS.v1_1.InitialClaim

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Gadgets.Polynomial
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier

structure Interface where
  gamma : Nat → KExpr
  eval_K : Nat → PadCoordinate productionShape → KExpr
  eval_A : Nat → MatrixCoordinate productionShape → KExpr

/-- Exact constant-first expression order: all Pad coefficients, then all
genuine CCS-matrix coefficients. -/
def coefficientExprs (interface : Interface) (offset : Nat) : List KExpr :=
  (canonicalPadCoordinates productionShape).map (interface.eval_K offset) ++
    (canonicalMatrixCoordinates productionShape).map (interface.eval_A offset)

def ownedInterface (interface : Interface) : Horner.Owned.Interface where
  point := interface.gamma
  coefficients := coefficientExprs interface

def program (interface : Interface) (offset : Nat) : Horner.Program :=
  Horner.Owned.program (ownedInterface interface) offset

/-- The child-owned symbolic initial claim. -/
def output (interface : Interface) (offset : Nat) : KExpr :=
  Horner.Owned.output (ownedInterface interface) offset

abbrev Assumptions (interface : Interface) (offset : Nat) (env : Env) : Prop :=
  Horner.Owned.Assumptions (ownedInterface interface) offset env

/-- Named semantic predicate: the owned output is the canonical Horner
evaluation of the separate v1.1 target families. -/
abbrev SpecHolds (interface : Interface) (offset : Nat) (env : Env) : Prop :=
  Horner.Owned.SpecHolds (ownedInterface interface) offset env

/-- The sole logical circuit for this leaf. -/
def circuit (interface : Interface) : FormalCircuit :=
  Horner.Owned.circuit (ownedInterface interface)

theorem soundness (interface : Interface) (env : Env) (offset : Nat)
    (assumptions : Assumptions interface offset env)
    (rows : holds env (Circuit.ops (circuit interface).main offset)) :
    SpecHolds interface offset env :=
  Horner.Owned.soundness (ownedInterface interface) env offset assumptions rows

/-- Honest execution constructs this leaf without a caller-supplied result. -/
theorem build (interface : Interface) (env : Env) (offset : Nat)
    (assumptions : Assumptions interface offset env) :
    ∃ completed,
      AgreesOutside env completed offset
        (localLength (Circuit.ops (circuit interface).main offset)) ∧
      holdsFlat completed (Circuit.ops (circuit interface).main offset) :=
  Horner.Owned.build (ownedInterface interface) env offset assumptions

theorem completeness (interface : Interface) (env : Env) (offset : Nat)
    (assumptions : Assumptions interface offset env)
    (_specification : SpecHolds interface offset env) :
    ∃ completed,
      AgreesOutside env completed offset
        (localLength (Circuit.ops (circuit interface).main offset)) ∧
      holdsFlat completed (Circuit.ops (circuit interface).main offset) :=
  build interface env offset assumptions

theorem flatConstraints_varsBelow (interface : Interface) (offset : Nat)
    (env : Env) (assumptions : Assumptions interface offset env) :
    ∀ expression ∈ flatConstraints
      (Circuit.ops (circuit interface).main offset),
      expression.VarsBelow
        (offset + localLength (Circuit.ops (circuit interface).main offset)) := by
  exact Horner.Owned.flatConstraints_varsBelow
    (ownedInterface interface) offset env assumptions

theorem coefficientExprs_length (interface : Interface) (offset : Nat) :
    (coefficientExprs interface offset).length = 12960 := by
  simp [coefficientExprs, canonicalPadCoordinates_length,
    canonicalMatrixCoordinates_length, productionShape, productionProfile,
    Phi81MatrixSource.phi81Shape, Shape.padEvaluationCount,
    Shape.matrixEvaluationCount, cubeVariables, ringDegree]

/-- Private symbolic variables owned by the fixed production leaf. -/
def privateCount : Nat := 25918

/-- Exact private symbolic footprint of the optimized Horner child. -/
theorem localLength_eq (interface : Interface) (offset : Nat) :
    localLength (Circuit.ops (circuit interface).main offset) = 25918 := by
  change localLength (Circuit.ops
    (Horner.Owned.circuit (ownedInterface interface)).main offset) = 25918
  rw [Horner.Owned.localLength_eq]
  change 2 * ((coefficientExprs interface offset).length - 1) = 25918
  rw [coefficientExprs_length]

theorem operations_length (interface : Interface) (offset : Nat) :
    (Circuit.ops (circuit interface).main offset).length = 1 :=
  Horner.Owned.operations_length (ownedInterface interface) offset

theorem flatConstraints_length (interface : Interface) (offset : Nat) :
    (flatConstraints (Circuit.ops (circuit interface).main offset)).length =
      25918 := by
  change (flatConstraints (Circuit.ops
    (Horner.Owned.circuit (ownedInterface interface)).main offset)).length = 25918
  rw [Horner.Owned.flatConstraints_length]
  change 2 * ((coefficientExprs interface offset).length - 1) = 25918
  rw [coefficientExprs_length]

/-- Concrete parent coverage: shared target wires and the challenge leaf make
this initial claim exactly the value consumed by production `piCcsCheck`. -/
theorem spec_implies_keyInitial
    {logicalWidth : Nat}
    {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
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

end NightstreamFPrime.Lifecycle.PiCCS.v1_1.InitialClaim
