import NightstreamFPrime.Export.Stage1.PiCCSPhaseTransportSupport
import NightstreamFPrime.Lifecycle.PiCCS.v1_1.Completeness.Core

/-!
Owns proof-only transport of deterministic PiCCS phase semantics between two
environments that agree through the exact phase endpoint. It adds no row,
verifier input, or alternate protocol predicate.
-/

namespace NightstreamFPrime.Export.Stage1.PiCCSPhaseTransport

open NightstreamFPrime.Circuit
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.PiCCS.v1_1
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

/-- PiCCS phase semantics depend only on columns below the exact phase end. -/
theorem phaseHolds_of_agree_below_final
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (interface : Formal.Interface logicalWidth
      (ProductionKey.degreeBound relation) publicFits)
    (offset : Nat) (left right : Env)
    (template : Proof (ProductionKey.degreeBound relation))
    (assumptions : Formal.Assumptions relation interface offset left)
    (agrees : ∀ index, index < Formal.finalOffset relation interface offset →
      left index = right index)
    (phase : Formal.PhaseHolds relation ajtai interface offset left template) :
    Formal.PhaseHolds relation ajtai interface offset right template := by
  have offsetLeFinal := PiCCSPhaseTransportSupport.offset_le_finalOffset
    relation interface offset
  have agreesAtOffset : ∀ index, index < offset → left index = right index := by
    intro index bound
    exact agrees index (lt_of_lt_of_le bound offsetLeFinal)
  have stateBinding := StateBinding.specHolds_of_agree_below
    (Formal.statementBindingInterface (Formal.atOffset interface offset)).state
    offset left right assumptions.statementBinding
    (fun index bound => (agreesAtOffset index bound).symm) phase.stateBinding
  have accepted := Formal.CompletenessSupport.accepted_of_agree_below relation
    ajtai interface offset left right template assumptions.external
    agreesAtOffset phase.accepted
  have runningEq :=
    Formal.CompletenessSupport.evalRunning_eq_of_agree_below interface offset
      left right assumptions.external agreesAtOffset
  have freshEq :=
    Formal.CompletenessSupport.evalFresh_eq_of_agree_below interface offset
      left right assumptions.external agreesAtOffset
  have proofEq :=
    Formal.CompletenessSupport.evalProof_eq_of_agree_below relation interface
      offset left right template assumptions.external agreesAtOffset
  have stateEq : StatementAbsorption.evalState right
      (Formal.outputBindingFinalState relation interface offset) =
    StatementAbsorption.evalState left
      (Formal.outputBindingFinalState relation interface offset) := by
    unfold StatementAbsorption.evalState
    apply congrArg List.ofFn
    funext lane
    apply Expr.eval_eq_of_agree_below
      (Formal.outputBindingFinalState relation interface offset lane)
      (Formal.finalOffset relation interface offset) right left
      (PiCCSPhaseTransportSupport.outputBindingFinalState_varsBelow relation
        interface offset left assumptions lane)
    intro index bound
    exact (agrees index bound).symm
  refine {
    stateBinding := stateBinding
    accepted := accepted
    outgoingState := ?_
  }
  rw [stateEq, ← runningEq, ← freshEq, ← proofEq]
  exact phase.outgoingState

end NightstreamFPrime.Export.Stage1.PiCCSPhaseTransport
