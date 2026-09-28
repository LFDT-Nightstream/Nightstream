import NightstreamFPrime.Export.Stage1.PermutationActionCompleteness
import NightstreamFPrime.Export.Stage1.PiCCSInvocations
import NightstreamFPrime.Layout.PiCCS.v1_1.Preservation

/-!
Owns the projection of actual PiCCS physical rows to the existing compact
transcript invocation packets. The completed assignment and each source
compiler use the same Spartan pullback and the same child coordinates.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiCCSPhysicalPackets

open NightstreamFPrime.Circuit
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.PiCCS.v1_1
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Gadgets.Poseidon2.Duplex
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Export.Package
open PiCCSInvocations
open Invocations

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth}

private theorem child_constraints
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (target : Env)
    (physical : NightstreamFPrime.Layout.PiCCS.v1_1.PhysicalHolds relation
      (parentInterface logicalWidth publicFits) PiCCSInputs.phaseOffset (Spartan.pullback target))
    (child : FormalCircuit) (start : Nat)
    (member : NightstreamFPrime.Layout.PiCCS.v1_1.childConstraints child start ∈
      NightstreamFPrime.Layout.PiCCS.v1_1.childConstraintLists relation
        (parentInterface logicalWidth publicFits) PiCCSInputs.phaseOffset) :
    ConstraintsHold (Spartan.pullback target)
      (NightstreamFPrime.Layout.PiCCS.v1_1.childConstraints child start) := by
  have logical := NightstreamFPrime.Layout.PiCCS.v1_1.physical_implies_holdsFlat
    relation (parentInterface logicalWidth publicFits) PiCCSInputs.phaseOffset
    (Spartan.pullback target) physical
  change ConstraintsHold (Spartan.pullback target)
    (NightstreamFPrime.Layout.PiCCS.v1_1.logicalConstraints relation
      (parentInterface logicalWidth publicFits) PiCCSInputs.phaseOffset) at logical
  rw [NightstreamFPrime.Layout.PiCCS.v1_1.logicalConstraints_eq_flatten] at logical
  intro expression expressionMember
  exact logical expression (List.mem_flatten.mpr ⟨_, member, expressionMember⟩)

private theorem statement_rows
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (target : Env)
    (physical : NightstreamFPrime.Layout.PiCCS.v1_1.PhysicalHolds relation
      (parentInterface logicalWidth publicFits) PiCCSInputs.phaseOffset (Spartan.pullback target)) :
    ConstraintsHold (Spartan.pullback target)
      (recipeConstraints statementWitnessStart
        (StatementAbsorption.program (statementInterface logicalWidth publicFits)
          statementWitnessStart).recipes) := by
  have rows := child_constraints relation target physical
    (NightstreamFPrime.Lifecycle.PiCCS.v1_1.Formal.statementAbsorptionCircuit
      (sharedInterface logicalWidth publicFits))
    (NightstreamFPrime.Lifecycle.PiCCS.v1_1.Formal.statementAbsorptionOffset
      (parentInterface logicalWidth publicFits) PiCCSInputs.phaseOffset) (by
        apply List.mem_cons_of_mem
        exact List.mem_cons_self)
  rw [← statementWitnessStart_matches logicalWidth publicFits] at rows
  change ConstraintsHold (Spartan.pullback target)
    (flatConstraints (Circuit.ops
      (StatementAbsorption.circuit (statementInterface logicalWidth publicFits)).main
      statementWitnessStart)) at rows
  rw [StatementAbsorption.circuit_ops, StatementAbsorption.flatConstraints_opsAt] at rows
  exact rows

private theorem challenge_rows
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (target : Env)
    (physical : NightstreamFPrime.Layout.PiCCS.v1_1.PhysicalHolds relation
      (parentInterface logicalWidth publicFits) PiCCSInputs.phaseOffset (Spartan.pullback target)) :
    ConstraintsHold (Spartan.pullback target)
      (recipeConstraints challengeWitnessStart
        (ChallengeDerivation.program (challengeInterface logicalWidth publicFits)
          challengeWitnessStart).recipes) := by
  have rows := child_constraints relation target physical
    (NightstreamFPrime.Lifecycle.PiCCS.v1_1.Formal.challengeCircuit
      (parentInterface logicalWidth publicFits) PiCCSInputs.phaseOffset)
    (NightstreamFPrime.Lifecycle.PiCCS.v1_1.Formal.challengeOffset
      (parentInterface logicalWidth publicFits) PiCCSInputs.phaseOffset) (by
        apply List.mem_cons_of_mem
        apply List.mem_cons_of_mem
        exact List.mem_cons_self)
  rw [← challengeWitnessStart_matches logicalWidth publicFits] at rows
  change ConstraintsHold (Spartan.pullback target)
    (flatConstraints (Circuit.ops
      (ChallengeDerivation.circuit (challengeInterface logicalWidth publicFits)).main
      challengeWitnessStart)) at rows
  rw [ChallengeDerivation.circuit_ops, ChallengeDerivation.flatConstraints_opsAt] at rows
  exact rows

private theorem round_rows
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (target : Env)
    (physical : NightstreamFPrime.Layout.PiCCS.v1_1.PhysicalHolds relation
      (parentInterface logicalWidth publicFits) PiCCSInputs.phaseOffset (Spartan.pullback target)) :
    ConstraintsHold (Spartan.pullback target)
      (recipeConstraints roundWitnessStart
        (RoundTranscript.program (roundInterface logicalWidth publicFits)
          roundWitnessStart).recipes) := by
  have rows := child_constraints relation target physical
    (NightstreamFPrime.Lifecycle.PiCCS.v1_1.Formal.roundTranscriptCircuit
      (sharedInterface logicalWidth publicFits))
    (NightstreamFPrime.Lifecycle.PiCCS.v1_1.Formal.roundTranscriptOffset
      (parentInterface logicalWidth publicFits) PiCCSInputs.phaseOffset) (by
        apply List.mem_cons_of_mem
        apply List.mem_cons_of_mem
        apply List.mem_cons_of_mem
        exact List.mem_cons_self)
  rw [← roundWitnessStart_matches logicalWidth publicFits] at rows
  change ConstraintsHold (Spartan.pullback target)
    (flatConstraints (Circuit.ops (RoundTranscript.main
      (roundInterface logicalWidth publicFits)) roundWitnessStart)) at rows
  rw [RoundTranscript.main_ops, RoundTranscript.flatConstraints_opsAt] at rows
  exact rows

private theorem output_rows
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (target : Env)
    (physical : NightstreamFPrime.Layout.PiCCS.v1_1.PhysicalHolds relation
      (parentInterface logicalWidth publicFits) PiCCSInputs.phaseOffset (Spartan.pullback target)) :
    ConstraintsHold (Spartan.pullback target)
      (recipeConstraints outputWitnessStart
        (Formal.compile outputWitnessStart
          ((outputInterface logicalWidth publicFits).initialState outputWitnessStart)
          (outputActions logicalWidth publicFits)).recipes) := by
  have rows := child_constraints relation target physical
    (NightstreamFPrime.Lifecycle.PiCCS.v1_1.Formal.outputBindingCircuit
      (sharedInterface logicalWidth publicFits))
    (NightstreamFPrime.Lifecycle.PiCCS.v1_1.Formal.outputBindingOffset relation
      (parentInterface logicalWidth publicFits) PiCCSInputs.phaseOffset) (by
        apply List.mem_cons_of_mem
        apply List.mem_cons_of_mem
        apply List.mem_cons_of_mem
        apply List.mem_cons_of_mem
        apply List.mem_cons_of_mem
        apply List.mem_cons_of_mem
        apply List.mem_cons_of_mem
        apply List.mem_cons_of_mem
        apply List.mem_cons_of_mem
        apply List.mem_cons_of_mem
        apply List.mem_cons_of_mem
        exact List.mem_cons_self)
  rw [← outputWitnessStart_matches logicalWidth publicFits relation] at rows
  change ConstraintsHold (Spartan.pullback target)
    (flatConstraints (Circuit.ops (Formal.Owned.main
      (OutputBinding.duplexInterface (outputInterface logicalWidth publicFits)))
      outputWitnessStart)) at rows
  rw [Formal.Owned.main_ops, Formal.Owned.flatConstraints_opsAt,
    OutputBinding.noAssertions, List.append_nil] at rows
  simpa only [Formal.Owned.program, OutputBinding.duplexInterface, outputActions] using rows

/-- The four existing compact PiCCS transcript packets follow from the actual
physical C rows in the completed Spartan assignment. All source recipe rows
and affine input conditions are derived from their existing child owners. -/
theorem permutations_of_physical
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (target : Env)
    (physical : NightstreamFPrime.Layout.PiCCS.v1_1.PhysicalHolds relation
      (parentInterface logicalWidth publicFits) PiCCSInputs.phaseOffset (Spartan.pullback target)) :
    ∀ current ∈ PiCCSInvocations.invocations logicalWidth publicFits,
      PermutationInvocationHolds (PilotData.circuitPackage ()) current target := by
  have statement : ∀ current ∈ (statementTrace logicalWidth publicFits).invocations,
      PermutationInvocationHolds (PilotData.circuitPackage ()) current target := by
    apply PermutationCompilerTransport.compileActions_complete_of_sourceConstraints
      statementPhase statementRowStart statementWitnessStart Hash.zeroE
      (statementActions logicalWidth publicFits) target
    · change Spartan.piCcsPhaseOffset ≤ PiCCSStarts.statementWitnessStart
      rw [PiCCSStarts.statementWitnessStart_eq]
      norm_num [Spartan.piCcsPhaseOffset]
    · exact Poseidon2.zeroE_affine
    · apply actionsInvocationInputsAffine_of_actionsAffine
      exact NightstreamFPrime.Layout.PiCCS.v1_1.Leaves.StatementAbsorption.actions_affine
        (statementInterface logicalWidth publicFits) statementWitnessStart
        ((inputShapes logicalWidth publicFits relation).statementAbsorption statementWitnessStart)
    · exact statement_rows relation target physical
  have challenge : ∀ current ∈ (challengeTrace logicalWidth publicFits).invocations,
      PermutationInvocationHolds (PilotData.circuitPackage ()) current target := by
    rw [challengeTrace_eq_semantic]
    apply PermutationCompilerTransport.compileActions_complete_of_sourceConstraints
      challengePhase challengeRowStart challengeWitnessStart
      ((challengeInterface logicalWidth publicFits).initialState challengeWitnessStart)
      (ChallengeDerivation.actions (challengeInterface logicalWidth publicFits) challengeWitnessStart) target
    · change Spartan.piCcsPhaseOffset ≤ PiCCSStarts.challengeWitnessStart
      rw [PiCCSStarts.challengeWitnessStart_eq]
      norm_num [Spartan.piCcsPhaseOffset]
    · exact ((inputShapes logicalWidth publicFits relation).challengeDerivation challengeWitnessStart).initialState
    · apply actionsInvocationInputsAffine_of_actionsAffine
      exact NightstreamFPrime.Layout.PiCCS.v1_1.Leaves.ChallengeDerivation.actions_affine
        (challengeInterface logicalWidth publicFits) challengeWitnessStart
        ((inputShapes logicalWidth publicFits relation).challengeDerivation challengeWitnessStart)
    · exact challenge_rows relation target physical
  have rounds : ∀ current ∈ (roundTrace logicalWidth publicFits).invocations,
      PermutationInvocationHolds (PilotData.circuitPackage ()) current target := by
    rw [roundTrace_eq_semantic]
    apply PermutationCompilerTransport.compileActions_complete_of_sourceConstraints
      roundPhase roundRowStart roundWitnessStart
      ((roundInterface logicalWidth publicFits).initialState roundWitnessStart)
      (RoundTranscript.actions (roundInterface logicalWidth publicFits) roundWitnessStart) target
    · change Spartan.piCcsPhaseOffset ≤ PiCCSStarts.roundTranscriptWitnessStart
      rw [PiCCSStarts.roundTranscriptWitnessStart_eq]
      norm_num [Spartan.piCcsPhaseOffset]
    · exact ((inputShapes logicalWidth publicFits relation).roundTranscript roundWitnessStart).initialState
    · apply actionsInvocationInputsAffine_of_actionsAffine
      exact NightstreamFPrime.Layout.PiCCS.v1_1.Leaves.RoundTranscript.actions_affine
        (roundInterface logicalWidth publicFits) roundWitnessStart
        ((inputShapes logicalWidth publicFits relation).roundTranscript roundWitnessStart)
    · exact round_rows relation target physical
  have output : ∀ current ∈ (outputTrace logicalWidth publicFits).invocations,
      PermutationInvocationHolds (PilotData.circuitPackage ()) current target := by
    rw [outputTrace_eq_semantic]
    apply PermutationCompilerTransport.compileActions_complete_of_sourceConstraints
      outputPhase outputRowStart outputWitnessStart
      ((outputInterface logicalWidth publicFits).initialState outputWitnessStart)
      (outputActions logicalWidth publicFits) target
    · change Spartan.piCcsPhaseOffset ≤ PiCCSStarts.outputBindingWitnessStart
      rw [PiCCSStarts.outputBindingWitnessStart_eq]
      norm_num [Spartan.piCcsPhaseOffset]
    · exact ((inputShapes logicalWidth publicFits relation).outputBinding outputWitnessStart).initialState
    · apply actionsInvocationInputsAffine_of_actionsAffine
      exact NightstreamFPrime.Layout.PiCCS.v1_1.Leaves.OutputBinding.actions_affine
        (outputInterface logicalWidth publicFits) outputWitnessStart
        ((inputShapes logicalWidth publicFits relation).outputBinding outputWitnessStart)
    · exact output_rows relation target physical
  intro current member
  rcases List.mem_append.mp member with beforeOutput | outputMember
  · rcases List.mem_append.mp beforeOutput with beforeRound | roundMember
    · rcases List.mem_append.mp beforeRound with statementMember | challengeMember
      · exact statement current statementMember
      · exact challenge current challengeMember
    · exact rounds current roundMember
  · exact output current outputMember

end NightstreamFPrime.Export.Stage1.PiCCSPhysicalPackets
