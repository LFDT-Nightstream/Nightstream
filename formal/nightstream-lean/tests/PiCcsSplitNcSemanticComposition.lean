import Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.ProductionRefinement.SemanticComposition
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.BlockLane.SemanticComposition

/-!
Focused structural regressions for the concrete PiCCS arithmetization and
fixed-active composition join.
-/

set_option autoImplicit false

namespace NightstreamTests.PiCcsSplitNcSemanticComposition

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.Authority
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open Nightstream.SuperNeo.SumCheck.Finite
open Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc
open Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.ProductionRefinement

universe uState

private abbrev ops := ConcreteCarrier.extensionOps

variable
  {shape : SemanticShape}
  {State : Type uState}
  {publicRingColumns verifierRows : Nat}
  {publicFits : ringDegree * publicRingColumns <= shape.carrierWidth}

/-! The semantic NC view follows the transcript-owned base/delayed tag. -/

example
    (input : AuthoritativeInput (shape := shape) (State := State)
      (publicRingColumns := publicRingColumns) (verifierRows := verifierRows)
      (publicFits := publicFits))
    (certificate : Certificate input)
    (pending : input.full.pending = none) :
    SemanticAttempt.ncInstance input certificate =
      FixedPhase.symbolicInstance ops.toOps
        (Polynomial.Nc.BlockLane.InitialSum.sumcheckPolynomial
          input.full.covers input.data input.full.ncCoins)
        Polynomial.Nc.Degree.ncSumcheckDegreeBound
        input.full.challengeSetSize
        Polynomial.Nc.BlockLane.InitialSum.claimedInitial
        (ProductionPiCcs.ncPoint input.full
          certificate.materialize).coordinates
        certificate.nc.toSumCheck := by
  simp [SemanticAttempt.ncInstance, ProductionPiCcs.rawPolynomial,
    ProductionPiCcs.rawInitial, pending]

example
    (input : AuthoritativeInput (shape := shape) (State := State)
      (publicRingColumns := publicRingColumns) (verifierRows := verifierRows)
      (publicFits := publicFits))
    (certificate : Certificate input)
    (pending : ProductionDelayedBlockLane)
    (pendingEq : input.full.pending = some pending) :
    SemanticAttempt.ncInstance input certificate =
      FixedPhase.symbolicInstance ops.toOps
        (Polynomial.Nc.BlockLane.DelayedCombinedNc.sumcheckPolynomial
          input.full.covers input.data input.full.ncCoins
          (ProductionProjection.productionWeights input.full)
          input.full.producerBeta input.full.batchWeight pending.oldBlock)
        Polynomial.Nc.Degree.ncSumcheckDegreeBound
        input.full.challengeSetSize
        (K.mul input.full.batchWeight
          (DelayedPackedProjection.projectedValue pending.parentYZcol
            input.full.producerBeta))
        (ProductionPiCcs.ncPoint input.full
          certificate.materialize).coordinates
        certificate.nc.toSumCheck := by
  simp [SemanticAttempt.ncInstance, ProductionPiCcs.rawPolynomial,
    ProductionPiCcs.rawInitial, pendingEq]

/-! Every concrete production failure remains a literal final branch. -/

example
    (input : AuthoritativeInput (shape := shape) (State := State)
      (publicRingColumns := publicRingColumns) (verifierRows := verifierRows)
      (publicFits := publicFits))
    (certificate : Certificate input)
    (bindingOps : PiRLC.RelaxedBindingOps
      (SourceAssignment shape) (CommitmentValue verifierRows) RingF)
    (sampling : PiRLC.SamplingBoundary FixedActive.arity.total)
    (failure : FeFailure input certificate) :
    SemanticComposition.BadEvent input certificate bindingOps sampling :=
  Or.inr (Or.inl failure)

example
    (input : AuthoritativeInput (shape := shape) (State := State)
      (publicRingColumns := publicRingColumns) (verifierRows := verifierRows)
      (publicFits := publicFits))
    (certificate : Certificate input)
    (bindingOps : PiRLC.RelaxedBindingOps
      (SourceAssignment shape) (CommitmentValue verifierRows) RingF)
    (sampling : PiRLC.SamplingBoundary FixedActive.arity.total)
    (failure : NcFailure input certificate) :
    SemanticComposition.BadEvent input certificate bindingOps sampling :=
  Or.inr (Or.inr (Or.inl failure))

example
    (input : AuthoritativeInput (shape := shape) (State := State)
      (publicRingColumns := publicRingColumns) (verifierRows := verifierRows)
      (publicFits := publicFits))
    (certificate : Certificate input)
    (bindingOps : PiRLC.RelaxedBindingOps
      (SourceAssignment shape) (CommitmentValue verifierRows) RingF)
    (sampling : PiRLC.SamplingBoundary FixedActive.arity.total)
    (failure : RegisteredDeviationObligation input certificate) :
    SemanticComposition.BadEvent input certificate bindingOps sampling :=
  Or.inr (Or.inr (Or.inr failure))

end NightstreamTests.PiCcsSplitNcSemanticComposition
