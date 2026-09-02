import NightstreamFPrime.Lifecycle.PiRLC.v1_1.Semantics

/-!
Owns deterministic equality for two accepted PiRLC phases. It derives the
same challenges from the same sampler seed and then the same combined output
from the accepted PiRLC equations. It adds no transcript or circuit rule.
-/

namespace NightstreamFPrime.Lifecycle.PiRLC.v1_1.Semantics.PhaseHolds

open NightstreamFPrime.Circuit
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

theorem challenges_eq_of_initialState_eq
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    {relation : ProductionKey.LogicalRelation logicalWidth publicFits}
    {ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits)}
    {interface : Formal.Interface logicalWidth publicFits}
    {offset : Nat} {left right : Env}
    (leftPhase : Semantics.PhaseHolds relation ajtai interface offset left)
    (rightPhase : Semantics.PhaseHolds relation ajtai interface offset right)
    (initialStateEq :
      SamplerChain.evalInitialState
          (Formal.samplerInterface (Formal.atOffset interface offset))
          (Formal.samplerOffset offset) left =
        SamplerChain.evalInitialState
          (Formal.samplerInterface (Formal.atOffset interface offset))
          (Formal.samplerOffset offset) right) :
    Semantics.evalChallenges interface offset left =
      Semantics.evalChallenges interface offset right := by
  have leftResponse := leftPhase.response
  have rightResponse := rightPhase.response
  rw [initialStateEq] at leftResponse
  exact Option.some.inj (leftResponse.symm.trans rightResponse)

theorem evalOutput_eq_of_shared
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (interface : Formal.Interface logicalWidth publicFits)
    (offset : Nat) (left right : Env)
    (leftPhase : Semantics.PhaseHolds relation ajtai interface offset left)
    (rightPhase : Semantics.PhaseHolds relation ajtai interface offset right)
    (inputsEq : Semantics.evalInputs relation interface offset left =
      Semantics.evalInputs relation interface offset right)
    (initialStateEq :
      SamplerChain.evalInitialState
          (Formal.samplerInterface (Formal.atOffset interface offset))
          (Formal.samplerOffset offset) left =
        SamplerChain.evalInitialState
          (Formal.samplerInterface (Formal.atOffset interface offset))
          (Formal.samplerOffset offset) right)
    (pointEq : (Semantics.evalOutput relation interface offset left).point =
      (Semantics.evalOutput relation interface offset right).point) :
    Semantics.evalOutput relation interface offset left =
      Semantics.evalOutput relation interface offset right := by
  have challengesEq := challenges_eq_of_initialState_eq leftPhase rightPhase
    initialStateEq
  have constraintSystemEq :
      (Semantics.evalOutput relation interface offset left).constraintSystem =
        (Semantics.evalOutput relation interface offset right).constraintSystem :=
    rfl
  calc
    Semantics.evalOutput relation interface offset left =
        Spec.Folding.PiRLC.combinedOutput (piRlcAlgebra ajtai)
          (Semantics.evalOutput relation interface offset left).constraintSystem
          (Semantics.evalOutput relation interface offset left).point
          (Semantics.evalInputs relation interface offset left)
          (Semantics.evalChallenges interface offset left) :=
      Semantics.output_eq_combinedOutput relation ajtai interface offset left
        leftPhase
    _ = Spec.Folding.PiRLC.combinedOutput (piRlcAlgebra ajtai)
          (Semantics.evalOutput relation interface offset right).constraintSystem
          (Semantics.evalOutput relation interface offset right).point
          (Semantics.evalInputs relation interface offset right)
          (Semantics.evalChallenges interface offset right) := by
      rw [constraintSystemEq, inputsEq, challengesEq, pointEq]
    _ = Semantics.evalOutput relation interface offset right :=
      (Semantics.output_eq_combinedOutput relation ajtai interface offset right
        rightPhase).symm

end NightstreamFPrime.Lifecycle.PiRLC.v1_1.Semantics.PhaseHolds
