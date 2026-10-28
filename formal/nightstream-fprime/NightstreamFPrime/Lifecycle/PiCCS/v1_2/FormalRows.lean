import NightstreamFPrime.Lifecycle.PiCCS.v1_2.Formal

/-!
Owns relation-free executable projections of the fixed production PiCCS
assembler. The projections do not define another verifier. Each equality
theorem ties one projection to the corresponding child of `Formal.circuit`.
-/

namespace NightstreamFPrime.Lifecycle.PiCCS.v1_2.Formal

open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.UnifiedSources

def rowConstantCoefficient : Fin productionShape.coefficientCount :=
  Phi81CoefficientKernel.constant

theorem constantCoefficient_eq_rowConstantCoefficient
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    constantCoefficient relation = rowConstantCoefficient := by
  rfl

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

def ccsRowOutput
    {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth degreeBound publicFits)
    (_offset : Nat) : KExpr :=
  NightstreamFPrime.Gadgets.Polynomial.Sparse.Owned.output ccsRowPolynomial
    (CcsTerminal.sparseInterface (ccsRowInterface interface))
    (ccsStart interface)

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

def normRowOutput
    {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth degreeBound publicFits)
    (_offset : Nat) : KExpr :=
  NormTerminal.output (normRowInterface interface) (normStart interface)

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

def initialClaimRowOffset (degreeBound offset : Nat) : Nat :=
  offset + 135078 + 4384 +
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

def evalKRowOffset (degreeBound offset : Nat) : Nat :=
  sumcheckRowOffset degreeBound offset + SumcheckChain.privateCount degreeBound

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

def normRowOffset
    {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (_interface : Interface logicalWidth degreeBound publicFits)
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

def finalRowOffset
    {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth degreeBound publicFits)
    (offset : Nat) : Nat :=
  outputBindingRowOffset interface offset + 838440

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

theorem finalRowOffset_eq_add
    {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth degreeBound publicFits)
    (offset : Nat) :
    finalRowOffset interface offset =
      offset + (1006821 + productionShape.cubeVariables *
        RoundTranscript.perRoundRecipeCount degreeBound +
        SumcheckChain.privateCount degreeBound) := by
  unfold finalRowOffset outputBindingRowOffset finalIdentityRowOffset
    normRowOffset ccsRowOffset evalARowOffset evalKRowOffset
    sumcheckRowOffset initialClaimRowOffset
  norm_num [FinalIdentity.privateCount, NormTerminal.privateCount,
    CcsTerminal.privateCount, EvalATerminal.privateCount,
    EvalKTerminal.privateCount, InitialClaim.privateCount]
  omega

theorem finalRowOffset_eq_add_of_degreeBound_eq_eight
    {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth degreeBound publicFits)
    (offset : Nat) (degreeEq : degreeBound = 8) :
    finalRowOffset interface offset = offset + 1068869 := by
  rw [finalRowOffset_eq_add, degreeEq]
  norm_num [RoundTranscript.perRoundRecipeCount, SumcheckChain.privateCount,
    NightstreamFPrime.Gadgets.SumCheck.FixedChain.Owned.privateCount,
    productionShape, Phi81MatrixSource.phi81Shape, cubeVariables]

end NightstreamFPrime.Lifecycle.PiCCS.v1_2.Formal
