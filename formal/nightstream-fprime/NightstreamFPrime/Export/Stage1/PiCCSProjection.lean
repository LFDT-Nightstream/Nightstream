import NightstreamFPrime.Export.Stage1.PiCCSInvocations

/-!
Owns recipe-free symbolic projections of the PiCCS transcript states used by
the compact permutation plan.

The full Duplex compiler and the existing compact traces remain the proof
meaning. The equalities below permit package emission to skip construction of
Poseidon2 recipes that the compact plan does not serialize.
-/

namespace NightstreamFPrime.Export.Stage1.PiCCSProjection

open NightstreamFPrime.Spec
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Gadgets.Poseidon2.Duplex
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

abbrev EState := Layer.EState

def fastOutput (witnessStart : Nat) (delayed : Unit → EState)
    (actions : List Formal.Action) : EState :=
  (Formal.compileWiringLazy witnessStart delayed actions).output

theorem fastOutput_eq_compile (witnessStart : Nat)
    (delayed : Unit → EState) (state : EState)
    (actions : List Formal.Action) (stateEq : delayed () = state) :
    fastOutput witnessStart delayed actions =
      (Formal.compile witnessStart state actions).output := by
  unfold fastOutput
  rw [Formal.compileWiringLazy_eq witnessStart delayed state actions stateEq]
  exact (Formal.compileWiring_matches witnessStart state actions).2

def fastStatementState (logicalWidth : Nat) (publicFits :
    ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) : EState :=
  fastOutput PiCCSInvocations.statementWitnessStart (fun _ => Hash.zeroE)
    (PiCCSInvocations.statementActions logicalWidth publicFits)

theorem fastStatementState_eq (logicalWidth : Nat) (publicFits :
    ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    fastStatementState logicalWidth publicFits =
      (PiCCSInvocations.statementTrace logicalWidth publicFits).state := by
  calc
    fastStatementState logicalWidth publicFits =
        (Formal.compile PiCCSInvocations.statementWitnessStart Hash.zeroE
          (PiCCSInvocations.statementActions logicalWidth publicFits)).output :=
      fastOutput_eq_compile _ _ _ _ rfl
    _ = _ := by
      simpa [PiCCSInvocations.statementTrace] using
        (Invocations.compileActions_state_eq
          PiCCSInvocations.statementPhase PiCCSInvocations.statementRowStart
          PiCCSInvocations.statementWitnessStart Hash.zeroE
          (PiCCSInvocations.statementActions logicalWidth publicFits)).symm

def fastChallengeState (logicalWidth : Nat) (publicFits :
    ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) : EState :=
  fastOutput PiCCSInvocations.challengeWitnessStart
    (fun _ => fastStatementState logicalWidth publicFits)
    (PiCCSInvocations.challengeActions logicalWidth publicFits)

theorem fastChallengeState_eq (logicalWidth : Nat) (publicFits :
    ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    fastChallengeState logicalWidth publicFits =
      (PiCCSInvocations.challengeTrace logicalWidth publicFits).state := by
  calc
    fastChallengeState logicalWidth publicFits =
        (Formal.compile PiCCSInvocations.challengeWitnessStart
          (PiCCSInvocations.statementTrace logicalWidth publicFits).state
          (PiCCSInvocations.challengeActions logicalWidth publicFits)).output :=
      fastOutput_eq_compile _ _ _ _
        (fastStatementState_eq logicalWidth publicFits)
    _ = _ := by
      simpa [PiCCSInvocations.challengeTrace] using
        (Invocations.compileActions_state_eq
          PiCCSInvocations.challengePhase PiCCSInvocations.challengeRowStart
          PiCCSInvocations.challengeWitnessStart
          (PiCCSInvocations.statementTrace logicalWidth publicFits).state
          (PiCCSInvocations.challengeActions logicalWidth publicFits)).symm

def fastRoundState (logicalWidth : Nat) (publicFits :
    ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) : EState :=
  fastOutput PiCCSInvocations.roundWitnessStart
    (fun _ => fastChallengeState logicalWidth publicFits)
    (PiCCSInvocations.roundActions logicalWidth publicFits)

theorem fastRoundState_eq (logicalWidth : Nat) (publicFits :
    ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    fastRoundState logicalWidth publicFits =
      (PiCCSInvocations.roundTrace logicalWidth publicFits).state := by
  calc
    fastRoundState logicalWidth publicFits =
        (Formal.compile PiCCSInvocations.roundWitnessStart
          (PiCCSInvocations.challengeTrace logicalWidth publicFits).state
          (PiCCSInvocations.roundActions logicalWidth publicFits)).output :=
      fastOutput_eq_compile _ _ _ _
        (fastChallengeState_eq logicalWidth publicFits)
    _ = _ := by
      simpa [PiCCSInvocations.roundTrace] using
        (Invocations.compileActions_state_eq
          PiCCSInvocations.roundPhase PiCCSInvocations.roundRowStart
          PiCCSInvocations.roundWitnessStart
          (PiCCSInvocations.challengeTrace logicalWidth publicFits).state
          (PiCCSInvocations.roundActions logicalWidth publicFits)).symm

def fastOutputState (logicalWidth : Nat) (publicFits :
    ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) : EState :=
  fastOutput PiCCSInvocations.outputWitnessStart
    (fun _ => fastRoundState logicalWidth publicFits)
    (PiCCSInvocations.outputActions logicalWidth publicFits)

theorem fastOutputState_eq (logicalWidth : Nat) (publicFits :
    ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    fastOutputState logicalWidth publicFits =
      (PiCCSInvocations.outputTrace logicalWidth publicFits).state := by
  calc
    fastOutputState logicalWidth publicFits =
        (Formal.compile PiCCSInvocations.outputWitnessStart
          (PiCCSInvocations.roundTrace logicalWidth publicFits).state
          (PiCCSInvocations.outputActions logicalWidth publicFits)).output :=
      fastOutput_eq_compile _ _ _ _
        (fastRoundState_eq logicalWidth publicFits)
    _ = _ := by
      simpa [PiCCSInvocations.outputTrace] using
        (Invocations.compileActions_state_eq
          PiCCSInvocations.outputPhase PiCCSInvocations.outputRowStart
          PiCCSInvocations.outputWitnessStart
          (PiCCSInvocations.roundTrace logicalWidth publicFits).state
          (PiCCSInvocations.outputActions logicalWidth publicFits)).symm

end NightstreamFPrime.Export.Stage1.PiCCSProjection
