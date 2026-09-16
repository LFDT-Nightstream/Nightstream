import tests.AxiomAudit
import tests.AxiomsStoredWitnessExecution
import tests.AxiomsCachedAssignmentPlan
import tests.AxiomsCachedAssignmentProducts
import tests.AxiomsFreshCommitmentBlock
import tests.EvidenceMetadata

/-! Exact procedure and value refinements for the fresh-witness executable.
This target does not claim whole-plan row satisfaction or a compact-row bridge. -/

namespace LeanGraph.Targets

open NightstreamFPrime
open Spec Circuit Export.Stage1
open Spec.Phi81Relation.EvaluationHomomorphism
open Spec.Phi81Relation.EvaluationHomomorphism.StoredRingArithmetic (StoredRing)

def FreshWitnessKernels : Prop :=
  (∀ (values : Array F) (start : Nat) (recipes : List Expr),
    start + recipes.length ≤ values.size →
      StoredWitnessExecution.asEnv (StoredWitnessExecution.executeRecipes values start recipes) =
        Circuit.executeRecipes (StoredWitnessExecution.asEnv values) start recipes) ∧
  (∀ (values : Array F) (start : Nat) (hints : List Hint),
    start + hints.length ≤ values.size →
      StoredWitnessExecution.asEnv (StoredWitnessExecution.executeHints values start hints) =
        Circuit.executeHints (StoredWitnessExecution.asEnv values) start hints) ∧
  (∀ (invocation : Export.Package.PermutationInvocation) (values : Array F),
    invocation.witnessStart + 592 ≤ values.size →
      StoredWitnessExecution.asEnv (StoredPermutationExecution.execute invocation values) =
        Export.Pilot.completePermutationInvocationEnv invocation (StoredWitnessExecution.asEnv values)) ∧
  (∀ (program : Lifecycle.Stage1.Application.Program)
      (prepared : CachedAssignmentProducts.Prepared program)
      (base : CachedAssignmentProducts.BaseValues program),
    CachedAssignmentProducts.rawValues prepared base =
      PerApplicationAssignmentTransportExecution.canonicalRawValues program base) ∧
  (∀ (program : Lifecycle.Stage1.Application.Program)
      (widths : CachedAssignmentPlan.Widths program)
      (raw : PerApplicationCanonicalAssignment.RawValues program),
    CachedAssignmentPlan.expand widths raw = PerApplicationAssignmentPlan.expand raw) ∧
  (∀ (rows columns : Nat) (setup : AjtaiSetupV1.Setup rows columns)
      (row : Fin rows) (block : Fin columns) (digit : StoredRing)
      (initial : PiDECNativeProduct.Accumulator),
    (FreshCommitmentBlock.accumulatePrepared setup row block
      (PiDECNativeProduct.prepareDigit digit) initial).finish.get =
      ringFAdd initial.finish.get (ringFMul (setup.verifierKey row block) digit.get))

theorem freshWitnessKernels : FreshWitnessKernels := by
  refine ⟨StoredWitnessExecution.executeRecipes_eq,
    StoredWitnessExecution.executeHints_eq, StoredPermutationExecution.execute_eq, ?_, ?_, ?_⟩
  · intro program prepared base
    exact CachedAssignmentProducts.rawValues_eq prepared base
  · intro program widths raw
    exact CachedAssignmentPlan.expand_eq widths raw
  · intro rows columns setup row block digit initial
    exact FreshCommitmentBlock.accumulatePrepared_value setup row block digit initial

#audit_axioms freshWitnessKernels

end LeanGraph.Targets
