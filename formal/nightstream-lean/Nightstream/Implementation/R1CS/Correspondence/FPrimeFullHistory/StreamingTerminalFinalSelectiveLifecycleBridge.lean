import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalXOutFinalPublicBinding

/-!
Contract: derive the terminal lifecycle outer-hash binding from exact source
decoder rows and exact final selective rows.

Owns the 32-field source-frame order and the composition into the existing
typed terminal XOut refinement. The verifier-owned public input is read from
the projected final assignment. No supplied digest equality is authoritative.

Does not own exact complete-matrix row identity, finalizer semantics,
recursive-size closure, or Poseidon2 collision resistance.

Assurance tier: security-reduced for the Nightstream b2/k16 terminal profile,
conditional on the explicit artifact-row premises.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFinalSelectiveLifecycleBridge

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.Nebula.StateOutputPoseidonBinding
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Interpreter
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleRelation
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalSourceBinding.Artifact
open Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalXOutFinalPublicBinding
open Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalXOutLifecycleBridge
open Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalXOutPoseidonCallBridge
open Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalXOutPoseidonCallSequence
open Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalXOutPoseidonOutputCopyBridge
open Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalXOutPublicBinding
open Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalXOutSourceFinalBridge
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalXOutPublicHash
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalXOutPublicHash

universe uParams uStructure uRunning uFresh uNifsProof uNebulaOpen
  uRunningWitness uFreshWitness

private abbrev phaseArtifact :=
  Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalFullPhaseSemantic.rawArtifact

private abbrev nebulaArtifact :=
  Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalFullNebulaStateDigest.rawArtifact

private abbrev sourceArtifact :=
  Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalSourceBinding.rawArtifact

/-- The full terminal context artifact and the source decoder use the same
ordered 32 decoded XOut columns. -/
theorem assignmentFrame_eq_decodedXOutValues (source : Nat → Nat) :
    assignmentFrame source = decodedXOutValues source := by
  unfold assignmentFrame decodedXOutValues
  rw [Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalFullXOutContext.xOutColumns_exact,
    List.map_ofFn]
  rfl

/-- Exact source and final rows replace the former outer-hash equality premise
in the typed terminal XOut lifecycle refinement. -/
theorem final_rows_bind_terminal_frame_or_collision
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {NifsProof : Type uNifsProof}
    {Nebula : Type}
    {NebulaOpen : Type uNebulaOpen}
    {RunningWitness : Type uRunningWitness}
    {FreshWitness : Type uFreshWitness}
    {armCount : Nat}
    {configuration : Configuration Params StructureDigest Running Fresh
      NifsProof Nebula NebulaOpen armCount}
    {authority :
      TerminalAuthority Running Fresh RunningWitness FreshWitness Nebula}
    (terminal : Terminal configuration authority)
    (phaseEncoding : PhasePayloadEncoding Fresh)
    (nebulaEncoding : NebulaEncoding Nebula)
    (outerCompatible :
      Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPreludeXOutLifecycleBridge.Poseidon2Compatible
        configuration)
    (phaseCompatible : PhasePoseidon2Compatible configuration phaseEncoding)
    (nebulaCompatible : NebulaPoseidon2Compatible configuration nebulaEncoding)
    (source : Nat → Nat)
    (canonical : ∀ column, source column < goldilocksP)
    (sourceOne : source 0 = 1)
    (sourceSatisfied : sourceArtifact.Satisfied source)
    (contextSatisfied :
      Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalFullXOutContext.Satisfied
        source)
    (phaseSatisfied : phaseArtifact.Satisfied source)
    (nebulaSatisfied : nebulaArtifact.Satisfied source)
    {rows : Nat}
    {relation : InterpretedRelation rows callPlacement8.finalColumns}
    (exact0 : FinalRowSliceExact callPlacement0 callPlacement0_valid relation
      (projectedFinalAssignment source canonical))
    (exact1 : FinalRowSliceExact callPlacement1 callPlacement1_valid relation
      (projectedFinalAssignment source canonical))
    (exact2 : FinalRowSliceExact callPlacement2 callPlacement2_valid relation
      (projectedFinalAssignment source canonical))
    (exact3 : FinalRowSliceExact callPlacement3 callPlacement3_valid relation
      (projectedFinalAssignment source canonical))
    (exact4 : FinalRowSliceExact callPlacement4 callPlacement4_valid relation
      (projectedFinalAssignment source canonical))
    (exact5 : FinalRowSliceExact callPlacement5 callPlacement5_valid relation
      (projectedFinalAssignment source canonical))
    (exact6 : FinalRowSliceExact callPlacement6 callPlacement6_valid relation
      (projectedFinalAssignment source canonical))
    (exact7 : FinalRowSliceExact callPlacement7 callPlacement7_valid relation
      (projectedFinalAssignment source canonical))
    (exact8 : FinalRowSliceExact callPlacement8 callPlacement8_valid relation
      (projectedFinalAssignment source canonical))
    (copyExact : ∀ lane : Fin 4,
      OutputCopyRowExact (outputCopyAt lane) relation
        (projectedFinalAssignment source canonical))
    (linkExact : ∀ lane : Fin 4, ∀ bit : Fin 64,
      PublicLinkRowExact lane bit relation
        (projectedFinalAssignment source canonical))
    (finalSatisfied : AllRowsSatisfied relation
      (projectedFinalAssignment source canonical))
    (finalOne :
      absoluteValue (projectedFinalAssignment source canonical) 0 = 1)
    (finalSelectorOne :
      absoluteValue (projectedFinalAssignment source canonical)
        callPlacement8.selectorColumn = 1)
    (publicBinding : PublicAssignmentBinding (projectedFinalValues source)
      terminal.publicXOut) :
    ((assignmentFrame source =
          frame configuration terminal.state terminal.nebula ∧
        phaseInput source =
          phasePreimage phaseEncoding terminal.phaseState terminal.latest ∧
        assignmentLaneBranch source =
          nebulaEncoding.branch terminal.nebula ∧
        laneInput source = nebulaEncoding.preimage terminal.nebula) ∨
      Failure) := by
  have terminalValues := source_rows_imply_terminal_x_out_values source
    canonical sourceOne sourceSatisfied
  have finalHash := final_rows_imply_outer_hash source canonical exact0 exact1
    exact2 exact3 exact4 exact5 exact6 exact7 exact8 copyExact linkExact
    finalSatisfied finalOne finalSelectorOne terminal.publicXOut publicBinding
  have outerHashBound :
      outerHash (assignmentFrame source) = digestValues terminal.publicXOut := by
    rw [assignmentFrame_eq_decodedXOutValues, terminalValues]
    exact finalHash
  exact (rows_bind_terminal_frame_or_collision terminal phaseEncoding
    nebulaEncoding outerCompatible phaseCompatible nebulaCompatible source
    canonical sourceOne contextSatisfied phaseSatisfied nebulaSatisfied
    outerHashBound terminal.frame_canonical).2

end Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFinalSelectiveLifecycleBridge
