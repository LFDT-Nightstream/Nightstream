import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.ProductionTerminalBridge.DelayedKernels

/-!
Concrete final-claim bridge for the fixed production combined-NC artifact.

Owns: composition of the materialized terminal trace with the claims-level
production terminal.  The final SumCheck claim is obtained only from
`TerminalProgram.Computed.final`; the terminal RHS is obtained only from
`TerminalProgram.Computed.fullRhs` and the boundary-column bindings.

Does not own: assignment construction, transcript replay, parent/raw-child
authority, commitment binding, `y_ring`, Poseidon2, Ajtai, costs, or row
removal.  Neither the desired terminal RHS nor the final claim is a premise.
-/

/-!
Emits constraints: none; this module derives terminal semantics from generated rows.

| Stable stage path | Obligation | Authority class |
|---|---|---|
| `f_prime.pi_ccs_nc.delayed.combined.production_terminal_bridge` | Compose carrier algebra and delayed kernels into the production terminal formula. | derived |

-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.ProductionTerminalBridge

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.PiCCS.OutputClaims
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized
open Nightstream.Implementation.R1CS.PiCcsNc
open Nightstream.Implementation.R1CS.PiCcsNc.Authority
open Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.CombinedNc

private abbrev laws := ConcreteCarrier.extensionLaws

universe uState

private theorem mappedOrdinaryExpression
    (assignment : Nat -> Nat) :
    ProductionMessageAcceptance.toConcreteK
        (TerminalProgram.ordinaryExpression assignment) =
      K.mul
        (K.mul
          (sourceValue assignment TerminalProgram.blockEquality.output)
          (sourceValue assignment TerminalProgram.laneEquality.output))
        (sourceValue assignment TerminalProgram.ordinarySum.output) := by
  unfold TerminalProgram.ordinaryExpression sourceValue
  rw [ProductionMessageAcceptance.toConcreteK_mul,
    ProductionMessageAcceptance.toConcreteK_mul]

private theorem mappedDelayedExpression
    (assignment : Nat -> Nat) :
    ProductionMessageAcceptance.toConcreteK
        (TerminalProgram.delayedExpression assignment) =
      K.mul
        (K.mul
          (K.mul
            (sourceValue assignment TerminalProgram.batchWeightColumns)
            (sourceValue assignment TerminalProgram.oldBlockEquality.output))
          (sourceValue assignment TerminalProgram.selectorOutput))
        (sourceValue assignment TerminalProgram.runningSum.output) := by
  unfold TerminalProgram.delayedExpression sourceValue
  rw [ProductionMessageAcceptance.toConcreteK_mul,
    ProductionMessageAcceptance.toConcreteK_mul,
    ProductionMessageAcceptance.toConcreteK_mul]

/-- Exact fixed-profile terminal refinement.  In particular, the theorem has
no premise mentioning `terminalRhsColumns`, `finalSumColumns`,
`ProductionPiCcs.messageTerminal`, or any acceptance predicate. -/
theorem computed_finalSum_eq_messageTerminal
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits : ringDegree * publicRingColumns <= shape.carrierWidth}
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (certificate : FixedActive.Certificate context)
    (pending : ProductionDelayedBlockLane)
    (assignment : Nat -> Nat)
    (constantOne : assignment 0 = 1)
    (computed : TerminalProgram.Computed assignment)
    (bindings : TerminalColumnBindings context certificate pending assignment) :
    sourceValue assignment TerminalProgram.finalSumColumns =
      ProductionPiCcs.messageTerminal context certificate := by
  have block := blockEqualityValue_eq context certificate pending assignment
    constantOne computed bindings
  have lane := laneEqualityValue_eq context certificate pending assignment
    constantOne computed bindings
  have ordinary := ordinarySumValue_eq_mixedRangeAt context certificate pending
    assignment constantOne computed bindings
  have oldBlock := oldBlockEqualityValue_eq context certificate pending
    assignment constantOne computed bindings
  have selector := selectorOutputValue_eq_betaPowerSelector context certificate
    pending assignment constantOne computed bindings
  have running := runningSumValue_eq_runningValueFromMessage context certificate
    pending assignment constantOne computed bindings
  have ordinaryExpression :
      ProductionMessageAcceptance.toConcreteK
          (TerminalProgram.ordinaryExpression assignment) =
        Terminal.terminalFromMessage certificate.piCcs.output context.ncCoins
          (ProductionPiCcs.ncPoint context certificate) := by
    rw [mappedOrdinaryExpression, block, lane, ordinary]
    rfl
  have delayedExpression :
      ProductionMessageAcceptance.toConcreteK
          (TerminalProgram.delayedExpression assignment) =
        MessageTerminal.delayedFromMessage certificate.piCcs.output
          (Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.CombinedNc.ProductionProjection.productionWeights
            context)
          context.producerBeta context.batchWeight pending.oldBlock
          (ProductionPiCcs.ncPoint context certificate) := by
    rw [mappedDelayedExpression, bindings.batchWeight, oldBlock, selector,
      running]
    unfold MessageTerminal.delayedFromMessage
    exact
      (laws.mul_assoc
        (K.mul context.batchWeight
          (SumCheckTruthPath.pointEquality ConcreteCarrier.extensionOps
            (ProductionPiCcs.ncPoint context certificate).block
            pending.oldBlock))
        (betaPowerSelector context.producerBeta
          (ProductionPiCcs.ncPoint context certificate).lane)
        (MessageTerminal.runningValueFromMessage certificate.piCcs.output
          (Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.CombinedNc.ProductionProjection.productionWeights
            context)
          (ProductionPiCcs.ncPoint context certificate).lane)).trans
      (laws.mul_assoc context.batchWeight
        (SumCheckTruthPath.pointEquality ConcreteCarrier.extensionOps
          (ProductionPiCcs.ncPoint context certificate).block
          pending.oldBlock)
        (K.mul
          (betaPowerSelector context.producerBeta
            (ProductionPiCcs.ncPoint context certificate).lane)
          (MessageTerminal.runningValueFromMessage certificate.piCcs.output
            (Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.CombinedNc.ProductionProjection.productionWeights
              context)
            (ProductionPiCcs.ncPoint context certificate).lane)))
  have messageTerminalDefinition :
      ProductionPiCcs.messageTerminal context certificate =
        MessageTerminal.verifierTerminal certificate.piCcs.output
          context.ncCoins
          (Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.CombinedNc.ProductionProjection.productionWeights
            context)
          context.producerBeta context.batchWeight pending.oldBlock
          (ProductionPiCcs.ncPoint context certificate) := by
    simpa only [ProductionPiCcs.messageTerminal, bindings.pendingEq]
  calc
    sourceValue assignment TerminalProgram.finalSumColumns =
        ProductionMessageAcceptance.toConcreteK
          (TerminalProgram.fullTerminalExpression assignment) := by
      unfold sourceValue
      rw [computed.final, computed.fullRhs]
    _ = K.add
        (ProductionMessageAcceptance.toConcreteK
          (TerminalProgram.ordinaryExpression assignment))
        (ProductionMessageAcceptance.toConcreteK
          (TerminalProgram.delayedExpression assignment)) := by
      unfold TerminalProgram.fullTerminalExpression
      rw [ProductionMessageAcceptance.toConcreteK_add]
    _ = K.add
        (Terminal.terminalFromMessage certificate.piCcs.output context.ncCoins
          (ProductionPiCcs.ncPoint context certificate))
        (MessageTerminal.delayedFromMessage certificate.piCcs.output
          (Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.CombinedNc.ProductionProjection.productionWeights
            context)
          context.producerBeta context.batchWeight pending.oldBlock
          (ProductionPiCcs.ncPoint context certificate)) := by
      rw [ordinaryExpression, delayedExpression]
    _ = MessageTerminal.verifierTerminal certificate.piCcs.output
        context.ncCoins
        (Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.CombinedNc.ProductionProjection.productionWeights
          context)
        context.producerBeta context.batchWeight pending.oldBlock
        (ProductionPiCcs.ncPoint context certificate) := rfl
    _ = ProductionPiCcs.messageTerminal context certificate :=
      messageTerminalDefinition.symm

end ProductionTerminalBridge
