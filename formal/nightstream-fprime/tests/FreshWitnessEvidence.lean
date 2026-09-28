import tests.AxiomsStoredPhysicalRowCheck
import tests.AxiomsCompactRowExecution
import tests.AxiomAudit
import tests.AxiomsStoredWitnessExecution
import tests.AxiomsCachedAssignmentPlan
import tests.AxiomsCachedAssignmentProducts
import tests.AxiomsFreshCommitmentBlock
import tests.EvidenceMetadata

/-! Exact procedure and value refinements for the fresh-witness executable.
Compact completion retains explicit physical geometry and output-scope premises.
Final checks of the concrete prepared plan imply all selected physical rows.
Caller parsing and the separate PiCCS source-polynomial bridge remain outside this target. -/

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
  (∀ (instruction : Export.Package.WitnessInstruction) (values : Array F),
    instruction.target < values.size →
      StoredWitnessExecution.asEnv (StoredInstructionExecution.execute instruction values) =
        instruction.execute (StoredWitnessExecution.asEnv values)) ∧
  (∀ (inputCount outputInput localStart : Nat) (inputColumn : Nat → Nat)
      (recipe : Expr) (values : Array F),
    inputColumn outputInput < values.size →
    localStart + Layout.R1CS.mulCount (Expr.var outputInput - recipe) ≤ values.size →
    inputCount ≤ localStart →
    (∀ input, input < inputCount →
      inputColumn input < localStart ∨
        localStart + Layout.R1CS.mulCount (Expr.var outputInput - recipe) ≤ inputColumn input) →
    outputInput < inputCount →
    recipe.VarsBelow outputInput →
    (∀ input, input < outputInput → inputColumn input ≠ inputColumn outputInput) →
    (StoredCompactRowExecution.execute inputColumn localStart
      (CompactRows.compactTemplate inputCount outputInput recipe) values).map
        (CompactRowRelocation.pullback inputCount localStart inputColumn ∘
          StoredWitnessExecution.asEnv) =
      some (Layout.R1CS.executeExpression
        (CompactRowRelocation.pullback inputCount localStart inputColumn
          (Env.set (StoredWitnessExecution.asEnv values) (inputColumn outputInput)
            (recipe.eval (fun input => StoredWitnessExecution.asEnv values (inputColumn input)))))
        (Expr.var outputInput - recipe) inputCount)) ∧
  (∀ (inputCount outputInput localStart : Nat) (inputColumn : Nat → Nat)
      (recipe : Expr) (values : Array F),
    inputColumn outputInput < values.size →
    localStart + Layout.R1CS.constraintFreshCount (Expr.var outputInput - recipe) ≤ values.size →
    inputCount ≤ localStart →
    (∀ input, input < inputCount →
      inputColumn input < localStart ∨
        localStart + Layout.R1CS.constraintFreshCount (Expr.var outputInput - recipe) ≤
          inputColumn input) →
    outputInput < inputCount →
    recipe.VarsBelow outputInput →
    (∀ input, input < outputInput → inputColumn input ≠ inputColumn outputInput) →
    (StoredCompactRowExecution.execute inputColumn localStart
      (CompactRows.compactConstraintTemplate inputCount outputInput recipe) values).map
        (CompactRowRelocation.pullback inputCount localStart inputColumn ∘
          StoredWitnessExecution.asEnv) =
      some (Layout.R1CS.executeConstraint
        (CompactRowRelocation.pullback inputCount localStart inputColumn
          (Env.set (StoredWitnessExecution.asEnv values) (inputColumn outputInput)
            (recipe.eval (fun input => StoredWitnessExecution.asEnv values (inputColumn input)))))
        (Expr.var outputInput - recipe) inputCount)) ∧
  (∀ (sources : PreparedPhysicalInputs.Inputs) (env : Env) (workers : Nat),
    0 < workers →
    StridedArrayAll.all (StoredPhysicalPlan.ofSources sources).rowEvents
      (fun event => event.check
        { (StoredPhysicalPlan.ofSources sources).pilot.val with
          compactRowTemplates := (StoredPhysicalPlan.ofSources sources).templates.toList }
        env) workers = true →
    ((StoredPhysicalPlan.ofSources sources).assertions.all
      (fun row => StoredPhysicalRowCheck.sparseRow row env)) = true →
    (Poseidon2HashChainV1Package.package ()).RowsHold env) ∧
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
    StoredWitnessExecution.executeHints_eq, StoredPermutationExecution.execute_eq,
    StoredInstructionExecution.execute_eq,
    StoredCompactCompletion.execute_compactTemplate,
    StoredCompactCompletion.execute_compactConstraintTemplate, ?_, ?_, ?_, ?_⟩
  · intro sources env workers positive eventChecks assertionChecks
    rw [StridedArrayAll.all_eq _ _ workers positive,
      Array.all_eq_true'] at eventChecks
    rw [Array.all_eq_true'] at assertionChecks
    exact StoredPhysicalPlan.ofSources_rowsHold sources env eventChecks assertionChecks
  · intro program prepared base
    exact CachedAssignmentProducts.rawValues_eq prepared base
  · intro program widths raw
    exact CachedAssignmentPlan.expand_eq widths raw
  · intro rows columns setup row block digit initial
    exact FreshCommitmentBlock.accumulatePrepared_value setup row block digit initial

#audit_axioms freshWitnessKernels

end LeanGraph.Targets
