import NightstreamFPrime.Gadgets.Polynomial.Sparse
import NightstreamFPrime.Lifecycle.PiCCS.v1_1.ChallengeDerivation
import NightstreamFPrime.Lifecycle.ProductionKey
import NightstreamFPrime.Spec.Folding.PiCCS.FinalIdentity

/-!
Paper authority: SuperNeo v1.1, Section 7.3, Step 4, `F`.
Obligation: Enforce
`F = sum_(i=1)^K gamma^(i-1) f(ct(y'_(i,1)), ..., ct(y'_(i,t)))`.

Inputs:
- all 14 fresh CCS-matrix evaluations for the sole production fresh source;
- the relation-owned selective sparse constraint polynomial.

Outputs:
- the exact symbolic fresh CCS residual term.

Constraint groups:
- C1: the opaque reusable `Sparse.Owned` polynomial evaluator.

Parent coverage:
- `ProtocolPolynomial.ccsAtMessage` inside `PiCCS.v1_1.Coverage.chain`.

The fixed production profile has one fresh source, so its source weight is
`gamma^0 = 1`. The two checked output wires flow directly to the
final-identity leaf.
-/

namespace NightstreamFPrime.Lifecycle.PiCCS.v1_1.CcsTerminal

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Gadgets.Polynomial
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec.Folding.Nifs.PaperNonInteractive
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.CCSResidualTable

structure Interface where
  freshMatrix : Nat → Fin productionShape.matrixCount → KExpr

/-- Static verifier-key polynomial. No proof value changes circuit syntax. -/
def polynomial
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    ConstraintPolynomial K productionShape.matrixCount :=
  ConstraintPolynomialLift.liftConstraintPolynomial K.embed
    (PaperAlgebra.matrixSource relation.system).constraintPolynomial

def sparseInterface (interface : Interface) :
    Sparse.Owned.Interface productionShape.matrixCount where
  point := interface.freshMatrix

/-- Exact symbolic production residual. -/
def output
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface) (offset : Nat) : KExpr :=
  Sparse.Owned.output (polynomial relation) (sparseInterface interface) offset

abbrev Assumptions
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface) (offset : Nat) (env : Env) : Prop :=
  Sparse.Owned.Assumptions (polynomial relation) (sparseInterface interface)
    offset env

/-- Named semantic predicate for the exact selective polynomial evaluation. -/
abbrev SpecHolds
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface) (offset : Nat) (env : Env) : Prop :=
  Sparse.Owned.SpecHolds (polynomial relation) (sparseInterface interface)
    offset env

/-- Sole logical circuit for the CCS-terminal leaf. -/
def circuit
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface) : FormalCircuit :=
  Sparse.Owned.circuit (polynomial relation) (sparseInterface interface)

theorem soundness
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface) (env : Env) (offset : Nat)
    (assumptions : Assumptions relation interface offset env)
    (rows : holds env
      (Circuit.ops (circuit relation interface).main offset)) :
    SpecHolds relation interface offset env :=
  Sparse.Owned.soundness (polynomial relation) (sparseInterface interface)
    env offset assumptions rows

theorem build
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface) (env : Env) (offset : Nat)
    (assumptions : Assumptions relation interface offset env) :
    ∃ completed,
      AgreesOutside env completed offset
        (localLength
          (Circuit.ops (circuit relation interface).main offset)) ∧
      holdsFlat completed
        (Circuit.ops (circuit relation interface).main offset) :=
  Sparse.Owned.build (polynomial relation) (sparseInterface interface)
    env offset assumptions

theorem completeness
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface) (env : Env) (offset : Nat)
    (assumptions : Assumptions relation interface offset env)
    (_specification : SpecHolds relation interface offset env) :
    ∃ completed,
      AgreesOutside env completed offset
        (localLength
          (Circuit.ops (circuit relation interface).main offset)) ∧
      holdsFlat completed
        (Circuit.ops (circuit relation interface).main offset) :=
  build relation interface env offset assumptions

def privateCount : Nat := 2
def rowCount : Nat := 2

theorem localLength_eq
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface) (offset : Nat) :
    localLength (Circuit.ops (circuit relation interface).main offset) =
      privateCount := by
  exact Sparse.Owned.localLength_eq (polynomial relation)
    (sparseInterface interface) offset

theorem operations_length
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface) (offset : Nat) :
    (Circuit.ops (circuit relation interface).main offset).length = 1 :=
  Sparse.Owned.operations_length (polynomial relation)
    (sparseInterface interface) offset

theorem flatConstraints_length
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface) (offset : Nat) :
    (flatConstraints
      (Circuit.ops (circuit relation interface).main offset)).length =
      rowCount := by
  exact Sparse.Owned.flatConstraints_length (polynomial relation)
    (sparseInterface interface) offset

theorem flatConstraints_varsBelow
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface) (offset : Nat)
    (_assumptions : Assumptions relation interface offset (fun _ => 0)) :
    ∀ constraint ∈ flatConstraints
      (Circuit.ops (circuit relation interface).main offset),
      constraint.VarsBelow (offset + privateCount) := by
  simpa [privateCount] using
    Sparse.Owned.flatConstraints_varsBelow (polynomial relation)
      (sparseInterface interface) offset _assumptions

private def freshIndex : Fin productionShape.freshCount :=
  ⟨0, by
    norm_num [productionShape, productionProfile,
      Phi81MatrixSource.phi81Shape]⟩

theorem ccsAtMessage_eq_singleFresh
    (input : ProtocolPolynomial.VerifierInput K productionShape)
    (gamma : K)
    (message : ProtocolPolynomial.OutputMessage K productionShape) :
    ProtocolPolynomial.ccsAtMessage extensionOps input gamma message =
      CCSResidualTable.evaluatePolynomial extensionOps
        input.constraintPolynomial (message.freshMatrixImage freshIndex) := by
  simp [ProtocolPolynomial.ccsAtMessage, SignedJointIdentity.sumMap,
    SignedJointIdentity.gammaTerm, TargetPolynomial.power,
    BooleanTable.finiteSum, canonicalFinIndices, productionShape,
    productionProfile, Phi81MatrixSource.phi81Shape, freshIndex,
    extensionLaws.one_mul, extensionLaws.add_zero]

/-- The selective evaluation is exactly production `ccsAtMessage`. -/
theorem spec_implies_keyCcsAtMessage
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
    (freshMatrixEq : ∀ matrix,
      (interface.freshMatrix offset matrix).eval env =
        ((ProductionKey.key relation ajtai).piCcsCertificate
          running fresh proof).output.freshMatrixImage freshIndex matrix)
    (specification : SpecHolds relation interface offset env) :
    (output relation interface offset).eval env =
      ProtocolPolynomial.ccsAtMessage extensionOps
        (ChallengeDerivation.productionContext
          relation ajtai running fresh).input
        ((ProductionKey.key relation ajtai).piCcsExecution
          running fresh proof).coins.gamma
        ((ProductionKey.key relation ajtai).piCcsCertificate
          running fresh proof).output := by
  let input := (ChallengeDerivation.productionContext
    relation ajtai running fresh).input
  let gamma := ((ProductionKey.key relation ajtai).piCcsExecution
    running fresh proof).coins.gamma
  let message := ((ProductionKey.key relation ajtai).piCcsCertificate
    running fresh proof).output
  have pointEq :
      (fun matrix => (interface.freshMatrix offset matrix).eval env) =
        message.freshMatrixImage freshIndex := by
    funext matrix
    exact freshMatrixEq matrix
  have polynomialEq : polynomial relation = input.constraintPolynomial := by
    rfl
  change (output relation interface offset).eval env =
    CCSResidualTable.evaluatePolynomial extensionOps (polynomial relation)
      (fun matrix => (interface.freshMatrix offset matrix).eval env)
    at specification
  rw [polynomialEq, pointEq] at specification
  exact specification.trans
    (ccsAtMessage_eq_singleFresh input gamma message).symm

end NightstreamFPrime.Lifecycle.PiCCS.v1_1.CcsTerminal
