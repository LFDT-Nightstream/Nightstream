import NightstreamFPrime.Lifecycle.OptionB.Keys
import NightstreamFPrime.Lifecycle.PiCCS.v1_2.InitialClaim
import NightstreamFPrime.Lifecycle.PiCCS.v1_2.OutputBinding
import NightstreamFPrime.Lifecycle.PiCCS.v1_2.EvalKTerminal
import NightstreamFPrime.Lifecycle.PiCCS.v1_2.EvalATerminal
import NightstreamFPrime.Lifecycle.PiCCS.v1_2.NormTerminal
import NightstreamFPrime.Lifecycle.PiCCS.v1_2.CcsTerminal
import NightstreamFPrime.Lifecycle.PiCCS.v1_2.FinalIdentity
import NightstreamFPrime.Lifecycle.PiCCS.v1_2.StatementAbsorption

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
  ((keys relation ajtai).light.statement running fresh).verifierInput
    (keys relation ajtai).light.lift

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

/-- The light final identity's `γ` exponents: 54 for `E_A` and 270 = 54 × 5 for the constraint
terms, against 864 and 4,320 today (`FinalIdentity.matrixExponent_eq`,
`FinalIdentity.constraintExponent_eq`). -/
theorem light_exponents :
    FinalIdentity.matrixExponentOf LightShape = 54 ∧
      FinalIdentity.constraintExponentOf LightShape = 270 :=
  ⟨rfl, rfl⟩

/-- The light Π_CCS terminal `v = Q(r')` (SuperNeo v1.2 Section 7.3, Step 4) from the generic
final identity. -/
theorem terminal_eq (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (running : Spec.Folding.Nifs.PaperNonInteractive.Running K PaperAlgebra.Commitment
      (PaperAlgebra.PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits))
      LightShape)
    (fresh : Spec.Folding.Nifs.PaperNonInteractive.Fresh PaperAlgebra.Commitment
      (PaperAlgebra.PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits))
      LightShape)
    (interface : FinalIdentity.ShapeInterface LightShape) (offset : Nat) (env : Env)
    (alpha roundPoint : CubePoint K LightShape.cubeVariables) (gamma : K)
    (message : ProtocolPolynomial.OutputMessage K LightShape)
    (roundPointEq : PointEquality.Owned.evalLeftPoint
      (FinalIdentity.pointInterfaceAt interface offset) offset env = roundPoint)
    (alphaEq : PointEquality.Owned.evalRightPoint
      (FinalIdentity.pointInterfaceAt interface offset) offset env = alpha)
    (gammaEq : (interface.gamma offset).eval env = gamma)
    (evalKEq : (interface.eval_K offset).eval env =
      ProtocolPolynomial.padAtMessage extensionOps (lightInput relation ajtai running fresh)
        gamma roundPoint message)
    (evalAEq : (interface.eval_A offset).eval env =
      ProtocolPolynomial.matrixAtMessage extensionOps (lightInput relation ajtai running fresh)
        gamma roundPoint message)
    (ccsEq : (interface.ccs offset).eval env =
      ProtocolPolynomial.ccsAtMessage extensionOps (lightInput relation ajtai running fresh)
        gamma message)
    (normEq : (interface.norm offset).eval env =
      ProtocolPolynomial.normAtMessage extensionOps gamma message)
    (specification : FinalIdentity.SpecHolds interface offset env) :
    (interface.terminal offset).eval env =
      ProtocolPolynomial.terminalFromMessage extensionOps
        (lightInput relation ajtai running fresh) alpha gamma roundPoint message :=
  FinalIdentity.spec_implies_terminal interface offset env rfl _ alpha roundPoint gamma message
    roundPointEq alphaEq gammaEq evalKEq evalAEq ccsEq normEq specification

/-- The light Π_CCS statement absorption (SuperNeo v1.2 Section 7.3): the generic absorption from
the light fold domain ends in the light key's prior transcript state, the state its Π_CCS
challenges start from. The absorbed statement is today's: the prior digest and the fresh blocks,
1,462 words. -/
theorem statementState_eq (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (running : Spec.Folding.Nifs.PaperNonInteractive.Running K PaperAlgebra.Commitment
      (PaperAlgebra.PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits))
      LightShape)
    (fresh : Spec.Folding.Nifs.PaperNonInteractive.Fresh PaperAlgebra.Commitment
      (PaperAlgebra.PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits))
      LightShape)
    (interface : StatementAbsorption.Interface logicalWidth publicFits) (offset : Nat)
    (env : Env) (final : Transcript.State)
    (freshWords : StatementAbsorption.evalFresh (interface.fresh offset) env =
      ⟨fresh.commitments, fresh.publicInputs⟩)
    (trace : Duplex.Formal.TraceHolds (StatementAbsorption.evalState env Hash.zeroE)
      ((StatementAbsorption.domainActions lightFoldDomainChunk interface offset).map
        (Duplex.Formal.Action.eval env)) final) :
    final = (keys relation ajtai).light.absorbPublicInput
      (keys relation ajtai).light.initialTranscriptState running fresh := by
  rw [StatementAbsorption.domainTrace_eq lightFoldDomainChunk interface offset env final trace,
    freshWords]
  rfl

end NightstreamFPrime.Lifecycle.OptionB.LightPiCCS
