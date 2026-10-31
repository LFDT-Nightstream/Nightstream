import NightstreamFPrime.Lifecycle.OptionB.Keys
import NightstreamFPrime.Lifecycle.PiCCS.v1_2.InitialClaim
import NightstreamFPrime.Lifecycle.PiCCS.v1_2.OutputBinding
import NightstreamFPrime.Lifecycle.PiCCS.v1_2.EvalKTerminal
import NightstreamFPrime.Lifecycle.PiCCS.v1_2.EvalATerminal
import NightstreamFPrime.Lifecycle.PiCCS.v1_2.NormTerminal
import NightstreamFPrime.Lifecycle.PiCCS.v1_2.CcsTerminal

/-! The light Π_CCS leaves (CW-2.2), one per production leaf made generic in the shape. Each
bridge gives a value of the production light key's Π_CCS statement from the leaf's rows. -/

namespace NightstreamFPrime.Lifecycle.OptionB.LightPiCCS

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Gadgets.Multilinear
open NightstreamFPrime.Lifecycle.PiCCS.v1_2

variable {logicalWidth : Nat}
  {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}

/-- The light Π_CCS claimed sum `T` (SuperNeo v1.2 Section 7.3, Step 2) over the single running
claim: 54 + 216 = 270 claimed coefficients, against 4,320 today
(`InitialClaim.coefficientExprs_length`). -/
theorem initial_eq (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (running : Spec.Folding.Nifs.PaperNonInteractive.Running K PaperAlgebra.Commitment
      (PaperAlgebra.PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits))
      LightShape)
    (fresh : Spec.Folding.Nifs.PaperNonInteractive.Fresh PaperAlgebra.Commitment
      (PaperAlgebra.PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits))
      LightShape)
    (interface : InitialClaim.ShapeInterface LightShape) (offset : Nat) (env : Env)
    (gamma : K) (gamma_eq : (interface.gamma offset).eval env = gamma)
    (eval_K_eq : ∀ coordinate, (interface.eval_K offset coordinate).eval env =
      (((keys relation ajtai).light.statement running fresh).verifierInput
        (keys relation ajtai).light.lift).claimedPadCoefficient coordinate)
    (eval_A_eq : ∀ coordinate, (interface.eval_A offset coordinate).eval env =
      (((keys relation ajtai).light.statement running fresh).verifierInput
        (keys relation ajtai).light.lift).claimedMatrixCoefficient coordinate)
    (specification : InitialClaim.SpecHolds interface offset env) :
    (InitialClaim.output interface offset).eval env =
      (((keys relation ajtai).light.statement running fresh).verifierInput
        (keys relation ajtai).light.lift).initial extensionOps gamma :=
  InitialClaim.spec_implies_initial interface offset env _ gamma gamma_eq eval_K_eq eval_A_eq
    specification

theorem light_coefficientCount (interface : InitialClaim.ShapeInterface LightShape)
    (offset : Nat) : (InitialClaim.coefficientExprs interface offset).length = 270 := by
  simp only [InitialClaim.coefficientExprs, List.length_append, List.length_map,
    canonicalPadCoordinates_length, canonicalMatrixCoordinates_length]
  rfl

/-- The light Π_CCS output binding (SuperNeo v1.2 Section 7.3, Step 5): the rows absorb the
outputs of the two sources, which is the light key's `absorbPiCcsOutput`: 2 × 540 = 1,080 words,
against 17 × 540 = 9,180 today. -/
theorem outgoing_eq (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (interface : OutputBinding.ShapeInterface LightShape) (offset : Nat) (env : Env)
    (state : Transcript.State) (output : FullOutputCoordinates.FullOutput K LightShape)
    (initialEq : List.ofFn (Layer.evalState env (interface.initialState offset)) = state)
    (padEq : ∀ source coefficient,
      ((interface.output offset).padCoordinate source coefficient).eval env =
        output.padCoordinate source coefficient)
    (matrixEq : ∀ source matrix coefficient,
      ((interface.output offset).matrixCoordinate source matrix coefficient).eval env =
        output.matrixCoordinate source matrix coefficient)
    (specification : OutputBinding.SpecHolds interface offset env) :
    List.ofFn (Layer.evalState env (OutputBinding.finalState interface offset)) =
      (keys relation ajtai).light.absorbPiCcsOutput state output :=
  OutputBinding.spec_implies_absorb interface offset env state output initialEq padEq matrixEq
    specification

section Terminals

variable (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
  (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
  (running : Spec.Folding.Nifs.PaperNonInteractive.Running K PaperAlgebra.Commitment
    (PaperAlgebra.PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits))
    LightShape)
  (fresh : Spec.Folding.Nifs.PaperNonInteractive.Fresh PaperAlgebra.Commitment
    (PaperAlgebra.PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits))
    LightShape)

/-- The light statement's verifier input (SuperNeo v1.2 Section 7.3). -/
noncomputable abbrev lightInput :=
  ((keys relation ajtai).light.statement running fresh).verifierInput (keys relation ajtai).light.lift

/-- `E_K` of the light Π_CCS from the generic `Eval_K` terminal: 54 output coefficients. -/
theorem padAtMessage_eq (interface : EvalKTerminal.ShapeInterface LightShape) (offset : Nat)
    (env : Env) (gamma : K) (roundPoint : CubePoint K LightShape.cubeVariables)
    (message : ProtocolPolynomial.OutputMessage K LightShape)
    (roundPointEq : PointEquality.Owned.evalLeftPoint
      (EvalKTerminal.pointInterfaceAt interface offset) offset env = roundPoint)
    (priorPointEq : PointEquality.Owned.evalRightPoint
      (EvalKTerminal.pointInterfaceAt interface offset) offset env =
        (lightInput relation ajtai running fresh).priorPoint)
    (gammaEq : (interface.gamma offset).eval env = gamma)
    (outputEvalEq : ∀ coordinate,
      (interface.outputEval_K offset coordinate).eval env = message.padImage coordinate)
    (specification : EvalKTerminal.SpecHolds interface offset env) :
    (EvalKTerminal.output interface offset).eval env =
      ProtocolPolynomial.padAtMessage extensionOps (lightInput relation ajtai running fresh)
        gamma roundPoint message :=
  EvalKTerminal.spec_implies_padAtMessage interface offset env _ gamma roundPoint message
    roundPointEq priorPointEq gammaEq outputEvalEq specification

/-- `E_A` of the light Π_CCS from the generic `Eval_A` terminal: 216 output coefficients. -/
theorem matrixAtMessage_eq (interface : EvalATerminal.ShapeInterface LightShape) (offset : Nat)
    (env : Env) (gamma : K) (roundPoint : CubePoint K LightShape.cubeVariables)
    (message : ProtocolPolynomial.OutputMessage K LightShape)
    (roundPointEq : PointEquality.Owned.evalLeftPoint
      (EvalATerminal.pointInterfaceAt interface offset) offset env = roundPoint)
    (priorPointEq : PointEquality.Owned.evalRightPoint
      (EvalATerminal.pointInterfaceAt interface offset) offset env =
        (lightInput relation ajtai running fresh).priorPoint)
    (gammaEq : (interface.gamma offset).eval env = gamma)
    (outputEvalEq : ∀ coordinate,
      (interface.outputEval_A offset coordinate).eval env = message.matrixImage coordinate)
    (specification : EvalATerminal.SpecHolds interface offset env) :
    (EvalATerminal.output interface offset).eval env =
      ProtocolPolynomial.matrixAtMessage extensionOps (lightInput relation ajtai running fresh)
        gamma roundPoint message :=
  EvalATerminal.spec_implies_matrixAtMessage interface offset env _ gamma roundPoint message
    roundPointEq priorPointEq gammaEq outputEvalEq specification

end Terminals

/-- The norm terms of the light Π_CCS over its two sources. The light verifier checks the same
terms as production; the running source's zero norm term is a prover-side fact
(`Spec/Folding/Light/PiCCS.lean`). -/
theorem normAtMessage_eq (interface : NormTerminal.ShapeInterface LightShape) (offset : Nat)
    (env : Env) (gamma : K) (message : ProtocolPolynomial.OutputMessage K LightShape)
    (gammaEq : (interface.gamma offset).eval env = gamma)
    (sourceAssignmentEq : ∀ source,
      (interface.sourceAssignment offset source).eval env = message.sourceAssignment source)
    (specification : NormTerminal.SpecHolds interface offset env) :
    (NormTerminal.output interface offset).eval env =
      ProtocolPolynomial.normAtMessage extensionOps gamma message :=
  NormTerminal.spec_implies_normAtMessage interface offset env gamma message gammaEq
    sourceAssignmentEq specification

private def lightFreshIndex : Fin LightShape.freshCount := ⟨0, by decide⟩

/-- The light Π_CCS has one fresh source, so its CCS term is the constraint polynomial at that
source's matrix evaluations. -/
theorem ccsAtMessage_eq_singleFresh (input : ProtocolPolynomial.VerifierInput K LightShape)
    (gamma : K) (message : ProtocolPolynomial.OutputMessage K LightShape) :
    ProtocolPolynomial.ccsAtMessage extensionOps input gamma message =
      CCSResidualTable.evaluatePolynomial extensionOps
        input.constraintPolynomial (message.freshMatrixImage lightFreshIndex) := by
  simp [ProtocolPolynomial.ccsAtMessage, SignedJointIdentity.sumMap,
    SignedJointIdentity.gammaTerm, TargetPolynomial.power,
    BooleanTable.finiteSum, canonicalFinIndices, LightShape,
    HyperNova.Construction2.OptionBFolds.lightShape, productionShape,
    productionProfile, Phi81MatrixSource.phi81Shape, lightFreshIndex,
    extensionLaws.one_mul, extensionLaws.add_zero]

end NightstreamFPrime.Lifecycle.OptionB.LightPiCCS
