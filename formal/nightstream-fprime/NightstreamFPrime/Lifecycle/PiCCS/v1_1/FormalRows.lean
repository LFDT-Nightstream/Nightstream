import NightstreamFPrime.Lifecycle.PiCCS.v1_1.Formal

/-!
Owns relation-free executable projections of the fixed production PiCCS
assembler. The projections do not define another verifier. Each equality
theorem ties one projection to the corresponding child of `Formal.circuit`.
-/

namespace NightstreamFPrime.Lifecycle.PiCCS.v1_1.Formal

open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.UnifiedSources

/-- Verifier-owned coefficient selected by every production relation. -/
def rowConstantCoefficient : Fin productionShape.coefficientCount :=
  Phi81CoefficientKernel.constant

theorem constantCoefficient_eq_rowConstantCoefficient
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    constantCoefficient relation = rowConstantCoefficient := by
  rfl

/-- Relation-free executable projection of the fixed production CCS wires. -/
def ccsRowInterface
    {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth degreeBound publicFits) :
    CcsTerminal.Interface where
  freshMatrix := fun offset matrix =>
    (interface.output offset).matrixCoordinate
      (freshSourceIndex freshIndex) matrix rowConstantCoefficient

theorem ccsInterface_eq_ccsRowInterface
    {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface logicalWidth degreeBound publicFits) :
    ccsInterface relation interface = ccsRowInterface interface := by
  rfl

/-- Relation-free projection of the one fixed lifted selective polynomial. -/
def ccsRowPolynomial :
    NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.CCSResidualTable.ConstraintPolynomial
      K productionShape.matrixCount :=
  ConstraintPolynomialLift.liftConstraintPolynomial K.embed
    ProductionRelation.polynomial

theorem ccsPolynomial_eq_ccsRowPolynomial
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    CcsTerminal.polynomial relation = ccsRowPolynomial := by
  rfl

/-- Relation-free child-owned CCS residual wires. -/
def ccsRowOutput
    {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth degreeBound publicFits)
    (_offset : Nat) : KExpr :=
  NightstreamFPrime.Gadgets.Polynomial.Sparse.Owned.output ccsRowPolynomial
    (CcsTerminal.sparseInterface (ccsRowInterface interface))
    (ccsStart interface)

/-- Relation-free executable projection of the fixed production norm wires. -/
def normRowInterface
    {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth degreeBound publicFits) :
    NormTerminal.Interface where
  gamma := challengeGamma interface
  sourceAssignment := fun offset source =>
    (interface.output offset).padCoordinate source rowConstantCoefficient

theorem normInterface_eq_normRowInterface
    {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface logicalWidth degreeBound publicFits) :
    normInterface relation interface = normRowInterface interface := by
  rfl

/-- Relation-free child-owned strict-base-2 norm wires. -/
def normRowOutput
    {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth degreeBound publicFits)
    (_offset : Nat) : KExpr :=
  NormTerminal.output (normRowInterface interface) (normStart interface)

/-- Relation-free executable projection of the fixed v1_1 final identity. -/
def finalIdentityRowInterface {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth degreeBound publicFits) :
    FinalIdentity.Interface where
  roundPoint := roundPoint interface
  alpha := challengeAlpha interface
  gamma := challengeGamma interface
  eval_K := evalKOutput interface
  eval_A := evalAOutput interface
  ccs := ccsRowOutput interface
  norm := normRowOutput interface
  terminal := sumcheckOutput interface

theorem finalIdentityInterface_eq_finalIdentityRowInterface
    {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface logicalWidth degreeBound publicFits) :
    finalIdentityInterface relation interface =
      finalIdentityRowInterface interface := by
  rfl

/-- Relation-free executable rows of the canonical CCS terminal child. -/
def ccsRowMain
    {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth degreeBound publicFits) : Circuit Unit :=
  (NightstreamFPrime.Gadgets.Polynomial.Sparse.Owned.circuit ccsRowPolynomial
    (CcsTerminal.sparseInterface (ccsRowInterface interface))).main

theorem ccsCircuit_main_eq_rowMain
    {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface logicalWidth degreeBound publicFits) :
    (ccsCircuit relation interface).main = ccsRowMain interface := by
  rfl

/-- Relation-free executable rows of the canonical norm terminal child. -/
def normRowMain
    {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth degreeBound publicFits) : Circuit Unit :=
  (NormTerminal.circuit (normRowInterface interface)).main

theorem normCircuit_main_eq_rowMain
    {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface logicalWidth degreeBound publicFits) :
    (normCircuit relation interface).main = normRowMain interface := by
  rfl

/-- Relation-free executable rows of the canonical v1_1 terminal identity. -/
def finalIdentityRowMain
    {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth degreeBound publicFits) : Circuit Unit :=
  (FinalIdentity.circuit (finalIdentityRowInterface interface)).main

theorem finalIdentityCircuit_main_eq_rowMain
    {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface logicalWidth degreeBound publicFits) :
    (finalIdentityCircuit relation interface).main =
      finalIdentityRowMain interface := by
  rfl

/-- Relation-free executable start of the initial-claim child. -/
def initialClaimRowOffset (degreeBound offset : Nat) : Nat :=
  offset + 224368 + 51504 +
    productionShape.cubeVariables *
      RoundTranscript.perRoundRecipeCount degreeBound

theorem initialClaimOffset_eq_initialClaimRowOffset
    {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth degreeBound publicFits)
    (offset : Nat) :
    initialClaimOffset interface offset =
      initialClaimRowOffset degreeBound offset := by
  unfold initialClaimOffset initialClaimRowOffset nextOffset childLength
    roundTranscriptCircuit
  rw [FormalCircuit.withConstantFootprint_main,
    RoundTranscript.localLength_eq, roundTranscriptOffset_eq,
    challengeOffset_eq]

/-- Relation-free executable start of the SumCheck child. -/
def sumcheckRowOffset (degreeBound offset : Nat) : Nat :=
  initialClaimRowOffset degreeBound offset + InitialClaim.privateCount

theorem sumcheckOffset_eq_sumcheckRowOffset
    {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth degreeBound publicFits)
    (offset : Nat) :
    sumcheckOffset interface offset = sumcheckRowOffset degreeBound offset := by
  unfold sumcheckOffset sumcheckRowOffset nextOffset childLength
    initialClaimCircuit
  rw [FormalCircuit.withConstantFootprint_main, InitialClaim.localLength_eq,
    initialClaimOffset_eq_initialClaimRowOffset]
  rfl

/-- Relation-free executable start of the `Eval_K` child. -/
def evalKRowOffset (degreeBound offset : Nat) : Nat :=
  sumcheckRowOffset degreeBound offset

theorem evalKOffset_eq_evalKRowOffset
    {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth degreeBound publicFits)
    (offset : Nat) :
    evalKOffset interface offset = evalKRowOffset degreeBound offset := by
  unfold evalKOffset evalKRowOffset nextOffset childLength sumcheckCircuit
  rw [FormalCircuit.withConstantFootprint_main, SumcheckChain.localLength_eq,
    sumcheckOffset_eq_sumcheckRowOffset]
  omega

/-- Relation-free executable start of the `Eval_A` child. -/
def evalARowOffset (degreeBound offset : Nat) : Nat :=
  evalKRowOffset degreeBound offset + EvalKTerminal.privateCount

theorem evalAOffset_eq_evalARowOffset
    {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth degreeBound publicFits)
    (offset : Nat) :
    evalAOffset interface offset = evalARowOffset degreeBound offset := by
  unfold evalAOffset evalARowOffset nextOffset childLength evalKCircuit
  rw [FormalCircuit.withConstantFootprint_main, EvalKTerminal.localLength_eq,
    evalKOffset_eq_evalKRowOffset]
  rfl

/-- Relation-free executable start of the CCS terminal child. -/
def ccsRowOffset (degreeBound offset : Nat) : Nat :=
  evalARowOffset degreeBound offset + EvalATerminal.privateCount

theorem ccsOffset_eq_ccsRowOffset
    {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth degreeBound publicFits)
    (offset : Nat) :
    ccsOffset interface offset = ccsRowOffset degreeBound offset := by
  unfold ccsOffset ccsRowOffset nextOffset childLength evalACircuit
  rw [FormalCircuit.withConstantFootprint_main, EvalATerminal.localLength_eq,
    evalAOffset_eq_evalARowOffset]
  rfl

/-- Relation-free executable start of the norm child. -/
def normRowOffset
    {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth degreeBound publicFits)
    (offset : Nat) : Nat :=
  ccsRowOffset degreeBound offset + CcsTerminal.privateCount

theorem normOffset_eq_normRowOffset
    {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface logicalWidth degreeBound publicFits)
    (offset : Nat) :
    normOffset relation interface offset = normRowOffset interface offset := by
  unfold normOffset normRowOffset nextOffset childLength ccsCircuit
  rw [FormalCircuit.withConstantFootprint_main,
    CcsTerminal.localLength_eq, ccsOffset_eq_ccsRowOffset]

/-- Relation-free executable start of the final-identity child. -/
def finalIdentityRowOffset
    {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth degreeBound publicFits)
    (offset : Nat) : Nat :=
  normRowOffset interface offset + NormTerminal.privateCount

theorem finalIdentityOffset_eq_finalIdentityRowOffset
    {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface logicalWidth degreeBound publicFits)
    (offset : Nat) :
    finalIdentityOffset relation interface offset =
      finalIdentityRowOffset interface offset := by
  unfold finalIdentityOffset finalIdentityRowOffset nextOffset childLength
    normCircuit
  rw [normOffset_eq_normRowOffset, FormalCircuit.withConstantFootprint_main,
    NormTerminal.localLength_eq]
  rfl

/-- Relation-free executable start of the output-binding child. -/
def outputBindingRowOffset
    {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth degreeBound publicFits)
    (offset : Nat) : Nat :=
  finalIdentityRowOffset interface offset + FinalIdentity.privateCount

theorem outputBindingOffset_eq_outputBindingRowOffset
    {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface logicalWidth degreeBound publicFits)
    (offset : Nat) :
    outputBindingOffset relation interface offset =
      outputBindingRowOffset interface offset := by
  unfold outputBindingOffset outputBindingRowOffset nextOffset childLength
    finalIdentityCircuit
  rw [finalIdentityOffset_eq_finalIdentityRowOffset,
    FormalCircuit.withConstantFootprint_main, FinalIdentity.localLength_eq]
  rfl

/-- Relation-free executable endpoint of the complete PiCCS assembler. -/
def finalRowOffset
    {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth degreeBound publicFits)
    (offset : Nat) : Nat :=
  outputBindingRowOffset interface offset + 4076512

theorem finalOffset_eq_finalRowOffset
    {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface logicalWidth degreeBound publicFits)
    (offset : Nat) :
    finalOffset relation interface offset = finalRowOffset interface offset := by
  unfold finalOffset finalRowOffset nextOffset childLength outputBindingCircuit
  rw [outputBindingOffset_eq_outputBindingRowOffset,
    FormalCircuit.withConstantFootprint_main, OutputBinding.localLength_eq]

/-- The closed-form endpoint is the phase start plus the aggregate symbolic
footprint. This proof never inspects a child operation list. -/
theorem finalRowOffset_eq_add
    {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth degreeBound publicFits)
    (offset : Nat) :
    finalRowOffset interface offset =
      offset + (4432230 + productionShape.cubeVariables *
        RoundTranscript.perRoundRecipeCount degreeBound) := by
  unfold finalRowOffset outputBindingRowOffset finalIdentityRowOffset
    normRowOffset ccsRowOffset evalARowOffset evalKRowOffset
    sumcheckRowOffset initialClaimRowOffset
  norm_num [FinalIdentity.privateCount, NormTerminal.privateCount,
    CcsTerminal.privateCount, EvalATerminal.privateCount,
    EvalKTerminal.privateCount, InitialClaim.privateCount]
  omega

/-- The production degree-nine PiCCS endpoint advances by exactly 4,581,414
private variables. -/
theorem finalRowOffset_eq_add_of_degreeBound_eq_nine
    {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth degreeBound publicFits)
    (offset : Nat) (degreeEq : degreeBound = 9) :
    finalRowOffset interface offset = offset + 4581414 := by
  rw [finalRowOffset_eq_add, degreeEq]
  norm_num [RoundTranscript.perRoundRecipeCount, productionShape,
    Phi81MatrixSource.phi81Shape, cubeVariables]

end NightstreamFPrime.Lifecycle.PiCCS.v1_1.Formal
