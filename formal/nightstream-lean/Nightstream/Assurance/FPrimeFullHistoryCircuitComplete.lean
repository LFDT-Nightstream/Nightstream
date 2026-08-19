import Nightstream.Assurance.FPrimeFullHistoryCircuit
import Nightstream.Assurance.FPrimeFullHistoryNifsReassembly
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.FPrimeFullHistoryCounterSound
import Nightstream.Assurance.FPrimeFullHistory.TerminalShell

/-!
Contract: honest compiler completeness for the exact supported full-history
F-prime artifact.

The public witness contains successful source/interpreter executions for row
families with intermediate columns and direct semantic inputs for row families
without auxiliaries.  It deliberately contains no `Satisfies`, aggregate
accepted-conclusion, or prover-supplied verifier-result field.  The theorem
reassembles those independent executions into the exact 4,076,614 sparse rows
in production manifest order.
-/

namespace Nightstream.Assurance.FPrimeFullHistoryCircuit

open Nightstream.Implementation.R1CS

set_option maxRecDepth 1048576
set_option maxHeartbeats 8000000

/-- Independent successful compiler executions for the exact supported
plain/stateless `[1,1]` full-history profile. -/
structure CompilerWitness
    (field : CanonicalU64Complete.FieldInverse)
    (assignment : Nat → Nat) where
  base : CheckedProgram.ExecutionWitness
    FPrimeFullHistoryBase.instructions assignment
  recursivePrelude : OwnerCertificate.Owner.ExecutionWitness field
    FPrimeFullHistoryRecursivePrelude.owner assignment
  recursiveNifs : FPrimeConcreteNifs.RecursiveExecutionWitness field assignment
  priorLink : CheckedProgram.ExecutionWitness
    FPrimeFullHistoryPriorLink.instructions assignment
  counter : FPrimeFullHistoryCounterSound.Compiler.ExecutionWitness field
    assignment
  recursiveOutput : CheckedProgram.ExecutionWitness
    FPrimeFullHistoryRecursiveOutput.instructions assignment
  stateLink : ∀ pair ∈ FPrimeFullHistoryStateLink.pairs,
    assignment pair.1 = assignment pair.2
  terminal : FPrimeFullHistoryTerminalShellSound.CompilerWitness field
    assignment

private theorem base_complete
    {assignment : Nat → Nat}
    (execution : CheckedProgram.ExecutionWitness
      FPrimeFullHistoryBase.instructions assignment) :
    Satisfies FPrimeFullHistoryBase.rows assignment := by
  exact CheckedProgram.ExecutionWitness.compiles execution
    FPrimeFullHistoryBase.definitions_wellFormed
    FPrimeFullHistoryBase.definitions_canonical (by native_decide)

private theorem priorLink_complete
    {assignment : Nat → Nat}
    (execution : CheckedProgram.ExecutionWitness
      FPrimeFullHistoryPriorLink.instructions assignment) :
    Satisfies FPrimeFullHistoryPriorLink.rows assignment := by
  exact CheckedProgram.ExecutionWitness.compiles execution
    FPrimeFullHistoryPriorLink.definitions_wellFormed
    FPrimeFullHistoryPriorLink.definitions_canonical (by native_decide)

private theorem recursiveOutput_complete
    {assignment : Nat → Nat}
    (execution : CheckedProgram.ExecutionWitness
      FPrimeFullHistoryRecursiveOutput.instructions assignment) :
    Satisfies FPrimeFullHistoryRecursiveOutput.rows assignment := by
  exact CheckedProgram.ExecutionWitness.compiles execution
    FPrimeFullHistoryRecursiveOutput.definitions_wellFormed
    FPrimeFullHistoryRecursiveOutput.definitions_canonical (by native_decide)

/-- Honest compiler execution satisfies every row of the exact generated
full-history artifact.  This is CIR-COMPLETE for the supported fixed profile. -/
theorem fPrimeCircuit_complete
    (prime : EuclidPrime goldilocksP)
    {field : CanonicalU64Complete.FieldInverse}
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (witness : CompilerWitness field assignment) :
    Satisfies FPrimeFullHistoryRows.fullRows assignment := by
  have baseRows : Satisfies FPrimeFullHistoryBase.rows assignment :=
    base_complete witness.base
  have preludeRows :
      Satisfies FPrimeFullHistoryRecursivePrelude.rows assignment :=
    OwnerCertificate.Owner.execution_complete canonical one
      witness.recursivePrelude
  have recursiveCertificate : FPrimeConcreteNifs.RecursiveRows assignment :=
    FPrimeConcreteNifs.recursive_rows_complete canonical one
      witness.recursiveNifs
  have recursiveNifsRows :
      Satisfies FPrimeFullHistoryRows.recursiveNifsRows assignment :=
    FPrimeFullHistoryNifsReassembly.recursiveNifs_satisfies
      recursiveCertificate
  have priorRows : Satisfies FPrimeFullHistoryPriorLink.rows assignment :=
    priorLink_complete witness.priorLink
  have counterRows :
      Satisfies FPrimeFullHistoryCounterSound.globalRows assignment :=
    FPrimeFullHistoryCounterSound.Compiler.complete prime witness.counter
  have counterTransitionRows :
      Satisfies FPrimeFullHistoryRows.counterTransitionRows assignment := by
    intro row member
    apply counterRows row
    rw [← FPrimeFullHistoryRows.counterRows_partition]
    exact List.mem_append_right _ member
  have outputRows :
      Satisfies FPrimeFullHistoryRecursiveOutput.rows assignment :=
    recursiveOutput_complete witness.recursiveOutput
  have recursiveRows :
      Satisfies FPrimeFullHistoryRows.recursiveRows assignment := by
    apply (FPrimeFullHistoryRows.recursive_satisfies_iff assignment).mpr
    intro rows member
    simp only [FPrimeFullHistoryRows.recursivePieces, List.mem_cons,
      List.not_mem_nil, or_false] at member
    rcases member with rfl | rfl | rfl | rfl | rfl | rfl | rfl
    · exact preludeRows
    · exact recursiveCertificate.transcript
    · exact recursiveNifsRows
    · exact priorRows
    · exact recursiveCertificate.accumulator
    · exact counterTransitionRows
    · exact outputRows
  have stateRows : Satisfies FPrimeFullHistoryStateLink.rows assignment :=
    FPrimeFullHistoryStateLinkSound.complete canonical one witness.stateLink
  have terminalCertificate :
      FPrimeFullHistoryTerminalShellSound.TerminalRows assignment :=
    FPrimeFullHistoryTerminalShellSound.complete canonical one witness.terminal
  have terminalNifsRows :
      Satisfies FPrimeFullHistoryRows.terminalNifsRows assignment :=
    FPrimeFullHistoryNifsReassembly.terminalNifs_satisfies
      terminalCertificate.nifs
  have terminalRows :
      Satisfies FPrimeFullHistoryRows.terminalRows assignment := by
    apply (FPrimeFullHistoryRows.terminal_satisfies_iff assignment).mpr
    intro rows member
    simp only [FPrimeFullHistoryRows.terminalPieces, List.mem_cons,
      List.not_mem_nil, or_false] at member
    rcases member with rfl | rfl | rfl | rfl | rfl
    · exact terminalNifsRows
    · exact terminalCertificate.runningLink
    · exact terminalCertificate.parentLink
    · exact terminalCertificate.latestLink
    · exact terminalCertificate.accumulator
  apply (FPrimeFullHistoryRows.full_satisfies_iff assignment).mpr
  intro rows member
  simp only [FPrimeFullHistoryRows.topLevelPieces, List.mem_cons,
    List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl | rfl | rfl | rfl | rfl | rfl
  · exact baseRows
  · exact recursiveRows
  · exact stateRows
  · exact terminalRows
  · exact terminalCertificate.continuity
  · exact terminalCertificate.publicPins
  · exact terminalCertificate.terminalCe

/-- Closing the two M4 directions: a successful compiler execution produces
the exact rows, and exact-row soundness yields the closed two-edge execution or
one of the named recursive/terminal projection-root events. -/
theorem fPrimeCircuit_execution_sound_or_bad
    (prime : EuclidPrime goldilocksP)
    {field : CanonicalU64Complete.FieldInverse}
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (witness : CompilerWitness field assignment) :
    Nightstream.Assurance.ValidExecution Edge
        (TerminalValid assignment canonical)
        initialState (finalState assignment canonical) 2 ∨
      BadEvent assignment :=
  fPrimeCircuit_sound_or_bad prime canonical one
    (fPrimeCircuit_complete prime canonical one witness)

end Nightstream.Assurance.FPrimeFullHistoryCircuit
