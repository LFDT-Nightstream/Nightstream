import NightstreamFPrime.Gadgets.Polynomial.Horner
import NightstreamFPrime.Lifecycle.PiCCS.v1_1.ChallengeDerivation
import NightstreamFPrime.Lifecycle.ProductionKey
import NightstreamFPrime.Spec.Folding.PiCCS.FinalIdentity

/-!
Paper authority: SuperNeo v1.1, Section 7.3, Step 4, `N`.
Obligation: Enforce
`N = sum_(i=1)^(K+k) gamma^(i-1) (x_i + 1) x_i (x_i - 1)`.

Inputs:
- the verifier-derived challenge `gamma`;
- all 17 output source-assignment values in exact `K + k` order.

Outputs:
- the child-owned exact strict-`b = 2` norm residual term.

Constraint groups:
- C1: one explicit cubic residual per source;
- C2: the opaque reusable Horner circuit for the indexed gamma sum.

Parent coverage:
- `ProtocolPolynomial.normAtMessage` inside `PiCCS.v1_1.Coverage.chain`.
-/

namespace NightstreamFPrime.Lifecycle.PiCCS.v1_1.NormTerminal

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Gadgets.Polynomial
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec.Folding.Nifs.PaperNonInteractive
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier

structure Interface where
  gamma : Nat → KExpr
  sourceAssignment : Nat → Fin productionShape.sourceCount → KExpr

def residualExpr (value : KExpr) : KExpr :=
  KExpr.mul (KExpr.mul (KExpr.add value KExpr.one) value)
    (KExpr.sub value KExpr.one)

theorem eval_residualExpr (env : Env) (value : KExpr) :
    (residualExpr value).eval env =
      ProtocolPolynomial.strictNormResidual extensionOps (value.eval env) := by
  rfl

def coefficientExprs (interface : Interface) (offset : Nat) : List KExpr :=
  (canonicalFinIndices productionShape.sourceCount).map fun source =>
    residualExpr (interface.sourceAssignment offset source)

def ownedInterface (interface : Interface) : Horner.Owned.Interface where
  point := interface.gamma
  coefficients := coefficientExprs interface

/-- The child-owned symbolic norm term. -/
def output (interface : Interface) (offset : Nat) : KExpr :=
  Horner.Owned.output (ownedInterface interface) offset

abbrev Assumptions (interface : Interface) (offset : Nat) (env : Env) : Prop :=
  Horner.Owned.Assumptions (ownedInterface interface) offset env

abbrev SpecHolds (interface : Interface) (offset : Nat) (env : Env) : Prop :=
  Horner.Owned.SpecHolds (ownedInterface interface) offset env

def circuit (interface : Interface) : FormalCircuit :=
  Horner.Owned.circuit (ownedInterface interface)

theorem soundness (interface : Interface) (env : Env) (offset : Nat)
    (assumptions : Assumptions interface offset env)
    (rows : holds env (Circuit.ops (circuit interface).main offset)) :
    SpecHolds interface offset env :=
  Horner.Owned.soundness (ownedInterface interface) env offset assumptions rows

/-- Honest execution constructs the norm term without a caller-supplied
result. -/
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

theorem coefficientExprs_length (interface : Interface) (offset : Nat) :
    (coefficientExprs interface offset).length = 17 := by
  simp [coefficientExprs, canonicalFinIndices_length, productionShape,
    productionProfile, Phi81MatrixSource.phi81Shape, Shape.sourceCount]

theorem localLength_eq (interface : Interface) (offset : Nat) :
    localLength (Circuit.ops (circuit interface).main offset) = 32 := by
  change localLength
    (Circuit.ops (Horner.Owned.circuit (ownedInterface interface)).main offset) = _
  rw [Horner.Owned.localLength_eq]
  change 2 * ((coefficientExprs interface offset).length - 1) = 32
  rw [coefficientExprs_length]

/-- Private symbolic variables owned by the fixed production leaf. -/
def privateCount : Nat := 32

theorem operations_length (interface : Interface) (offset : Nat) :
    (Circuit.ops (circuit interface).main offset).length = 1 :=
  Horner.Owned.operations_length (ownedInterface interface) offset

theorem flatConstraints_length (interface : Interface) (offset : Nat) :
    (flatConstraints (Circuit.ops (circuit interface).main offset)).length =
      32 := by
  change (flatConstraints
    (Circuit.ops (Horner.Owned.circuit
      (ownedInterface interface)).main offset)).length = _
  rw [Horner.Owned.flatConstraints_length]
  change 2 * ((coefficientExprs interface offset).length - 1) = 32
  rw [coefficientExprs_length]

theorem flatConstraints_varsBelow (interface : Interface) (offset : Nat)
    (env : Env) (assumptions : Assumptions interface offset env) :
    ∀ expression ∈ flatConstraints
      (Circuit.ops (circuit interface).main offset),
      expression.VarsBelow
        (offset + localLength (Circuit.ops (circuit interface).main offset)) := by
  have scope := Horner.Owned.flatConstraints_varsBelow (ownedInterface interface)
    offset env assumptions
  exact scope

/-- Concrete parent coverage: the indexed cubic residual Horner value is
exactly production `ProtocolPolynomial.normAtMessage`. -/
theorem spec_implies_keyNormAtMessage
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (running : Running
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (fresh : Fresh
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (proof : Proof (ProductionKey.degreeBound relation))
    (interface : Interface) (offset : Nat) (env : Env)
    (gammaEq : (interface.gamma offset).eval env =
      ((ProductionKey.key relation ajtai).piCcsExecution
        running fresh proof).coins.gamma)
    (sourceAssignmentEq : ∀ source,
      (interface.sourceAssignment offset source).eval env =
        ((ProductionKey.key relation ajtai).piCcsCertificate
          running fresh proof).output.sourceAssignment source)
    (specification : SpecHolds interface offset env) :
    (output interface offset).eval env =
      ProtocolPolynomial.normAtMessage extensionOps
        ((ProductionKey.key relation ajtai).piCcsExecution
          running fresh proof).coins.gamma
        ((ProductionKey.key relation ajtai).piCcsCertificate
          running fresh proof).output := by
  let gamma := ((ProductionKey.key relation ajtai).piCcsExecution
    running fresh proof).coins.gamma
  let message := ((ProductionKey.key relation ajtai).piCcsCertificate
    running fresh proof).output
  have coefficientsEq :
      (coefficientExprs interface offset).map (KExpr.eval env) =
        (canonicalFinIndices productionShape.sourceCount).map fun source =>
          ProtocolPolynomial.strictNormResidual extensionOps
            (message.sourceAssignment source) := by
    unfold coefficientExprs
    rw [List.map_map]
    apply List.map_congr_left
    intro source _
    rw [Function.comp_apply, eval_residualExpr, sourceAssignmentEq]
  unfold SpecHolds Horner.Owned.SpecHolds ownedInterface at specification
  rw [gammaEq, coefficientsEq] at specification
  calc
    (output interface offset).eval env =
        SumCheck.Finite.Message.evaluateCoefficients extensionOps.toOps gamma
          ((canonicalFinIndices productionShape.sourceCount).map fun source =>
            ProtocolPolynomial.strictNormResidual extensionOps
              (message.sourceAssignment source)) := by
      simpa [gamma, message] using specification
    _ = FiniteSumAlgebra.sumMap extensionOps
        (canonicalFinIndices productionShape.sourceCount) fun source =>
          SignedJointIdentity.gammaTerm extensionOps gamma source.val
            (ProtocolPolynomial.strictNormResidual extensionOps
              (message.sourceAssignment source)) :=
      SignedCoefficientPolynomial.evaluate_canonicalFinMap_eq_gammaSum
        extensionOps extensionLaws gamma productionShape.sourceCount
          (fun source => ProtocolPolynomial.strictNormResidual extensionOps
            (message.sourceAssignment source))
    _ = ProtocolPolynomial.normAtMessage extensionOps gamma message := by
      rfl

end NightstreamFPrime.Lifecycle.PiCCS.v1_1.NormTerminal
