import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPiRLCFamilyPhasedLifecycle

/-!
Contract: terminal lifecycle inclusion for the complete production PiRLC family run.

Owns the adapter from the full final family state to the global phase-terminal
authority, with exact phase-envelope preimage recovery.

Does not own the family rows, the global terminal relation, other phase
families, or Poseidon2 collision resistance.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyTerminalLifecycle

open Nightstream.Implementation.Nebula.ProductionStreamingFPrimeProgram
open Nightstream.Implementation.Nebula.ProductionStreamingPiRlcAuthority
open Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilySequence
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleRelation
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyContinuity
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPhasedLifecycle
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyXOutContinuity
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyXOutSequence.AcceptedFullStateRun

universe uParams uStructure uRunning uFresh uNifsProof uNebulaOpen

/-- A complete phase authority includes the typed PiRLC family finish. The
full family state remains the authority; its digest is recomputed here. -/
def TerminalIncluded
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Running Fresh : Type}
    {NifsProof : Type uNifsProof}
    {Nebula : Type}
    {NebulaOpen : Type uNebulaOpen}
    (configuration : Configuration Params StructureDigest Running Fresh
      NifsProof Nebula NebulaOpen (program productionConfig).length) : Prop :=
  ∀ family latest, FamilyFinishRelation family →
    configuration.phaseAuthority.terminal (familyStateDigest family) latest

/-- The accepted final family state discharges the global phase-terminal
relation. The only alternate result is an exact phase-envelope collision. -/
theorem acceptedRun_terminal_or_collision
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Running Fresh : Type}
    {NifsProof : Type uNifsProof}
    {Nebula : Type}
    {NebulaOpen : Type uNebulaOpen}
    {configuration : Configuration Params StructureDigest Running Fresh
      NifsProof Nebula NebulaOpen (program productionConfig).length}
    (compatible : PhaseEnvelopeCompatibility configuration)
    (included : TerminalIncluded configuration)
    {setup : InputBindingSetup}
    (run :
      FPrimeFullHistoryStreamingPiRLCFamilyXOutSequence.AcceptedFullStateRun
        Running Fresh configuration.hashSemantics configuration.context setup)
    (finish : FamilyFinishRelation
      (run.boundaryState exactFamilyCount))
    (phaseState : Digest) (latest : List Fresh)
    (outer : OuterState Running Fresh Nebula)
    (outerExact :
      (run.arm lastOrdinal).afterOuter = outer)
    (semanticEnvelopeExact : outer.semanticState =
      configuration.phaseEnvelopeDigest phaseState latest) :
    configuration.phaseAuthority.terminal phaseState latest ∨
      Poseidon2PhaseEnvelopeCollision := by
  have finalBoundary :
      run.boundaryState exactFamilyCount =
        (run.arm lastOrdinal).physical.afterState := by
    simp [FPrimeFullHistoryStreamingPiRLCFamilyXOutSequence.AcceptedFullStateRun.boundaryState]
  have finalFinish : FamilyFinishRelation
      (run.arm lastOrdinal).physical.afterState := by
    rw [← finalBoundary]
    exact finish
  have physicalExact :=
    FPrimeFullHistoryStreamingPiRLCFamilyXOutSequence.accepted_semantic_eq_phase_envelope
      (run.arm lastOrdinal) .after
  have lifecycleExact :
      (run.arm lastOrdinal).afterOuter.semanticState =
        configuration.phaseEnvelopeDigest phaseState latest := by
    rw [outerExact]
    exact semanticEnvelopeExact
  rcases phasePreimage_exact_or_collision compatible
      (run.arm lastOrdinal).physical.afterState
      (FPrimeFullHistoryStreamingPiRLCFamilyXOutSequence.acceptedPhasePayload
        (run.arm lastOrdinal).physical)
      phaseState latest (run.arm lastOrdinal).afterOuter.semanticState
      (by simpa using physicalExact) lifecycleExact with exact | collision
  · rw [← exact.1]
    exact Or.inl (included _ _ finalFinish)
  · exact Or.inr collision

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyTerminalLifecycle
