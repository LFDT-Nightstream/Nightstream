import NightstreamFPrime.Export.RowSemantics

/-!
Final-state checks for the existing physical package components. Call each
check with StoredWitnessExecution.asEnv values for an array-backed witness.
Template lists are traversed directly; no per-invocation row list is expanded.
These checks do not establish canonical component coverage or input origin.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.StoredPhysicalRowCheck

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout
open NightstreamFPrime.Export.Package

private def templateRow (row : TemplateRow) (value : ColumnRef → F) : Bool :=
  decide (row.a.eval value * row.b.eval value = row.c.eval value)

/-- Check every physical permutation row for one hash-chain ordinal. -/
def hashInvocation (package : CircuitPackage) (chain : HashChain)
    (ordinal : Nat) (env : Env) : Bool :=
  package.permutation.rows.all fun row =>
    templateRow row fun column =>
      (instantiateColumn package chain ordinal column).eval env

/-- Check the same physical permutation template with the explicit invocation's
existing sparse input combinations and local columns. -/
def permutationInvocation (package : CircuitPackage)
    (invocation : PermutationInvocation) (env : Env) : Bool :=
  package.permutation.rows.all fun row =>
    templateRow row fun column =>
      (instantiateInvocationColumn invocation column).eval env

private def compactRow (row : CompactTemplateRow) (value : ColumnRef → F) : Bool :=
  decide (row.a.eval value * row.b.eval value = row.c.eval value)

/-- A missing compact template rejects. A present template checks every row
in the final environment, including all witness rows and assertion rows. -/
def compactInvocation (package : CircuitPackage)
    (invocation : CompactRowInvocation) (env : Env) : Bool :=
  match package.compactRowTemplates[invocation.templateIndex]? with
  | none => false
  | some template =>
      template.rows.all fun row =>
        compactRow row fun column => env (instantiateCompactColumn invocation column)

/-- Check the authoritative final equation, not the instruction's write hint. -/
def instruction (item : WitnessInstruction) (env : Env) : Bool :=
  decide (item.a.eval env * item.b.eval env = env item.target)

def sparseRow (row : SparseRow) (env : Env) : Bool :=
  decide (row.a.eval env * row.b.eval env = row.c.eval env)

theorem hashInvocation_iff (package : CircuitPackage) (chain : HashChain)
    (ordinal : Nat) (env : Env) :
    hashInvocation package chain ordinal env = true ↔
      TemplateInvocationHolds package chain ordinal env := by
  simp only [hashInvocation, templateRow, List.all_eq_true, decide_eq_true_eq,
    TemplateInvocationHolds, instantiateRow_holds, TemplateRow.Holds]

theorem permutationInvocation_iff (package : CircuitPackage)
    (invocation : PermutationInvocation) (env : Env) :
    permutationInvocation package invocation env = true ↔
      PermutationInvocationHolds package invocation env := by
  simp only [permutationInvocation, templateRow, List.all_eq_true, decide_eq_true_eq,
    PermutationInvocationHolds, instantiateInvocationRow_holds, TemplateRow.Holds]

private theorem compactCombination_eval (invocation : CompactRowInvocation)
    (combination : TemplateCombination) (env : Env) :
    (instantiateCompactCombination invocation combination).eval env =
      combination.eval (fun column => env (instantiateCompactColumn invocation column)) := by
  simp [instantiateCompactCombination, R1CS.LinearCombination.eval,
    TemplateCombination.eval, ColumnRef.eval, List.map_map, Function.comp_def]

private theorem compactRow_iff (invocation : CompactRowInvocation)
    (row : CompactTemplateRow) (env : Env) :
    compactRow row (fun column => env (instantiateCompactColumn invocation column)) = true ↔
      (instantiateCompactRow invocation row).Holds env := by
  simp only [compactRow, decide_eq_true_eq, instantiateCompactRow, R1CS.Row.Holds,
    compactCombination_eval]

theorem compactInvocation_iff (package : CircuitPackage)
    (invocation : CompactRowInvocation) (env : Env) :
    compactInvocation package invocation env = true ↔
      CompactRowInvocationHolds package invocation env := by
  unfold compactInvocation CompactRowInvocationHolds
  cases selected : package.compactRowTemplates[invocation.templateIndex]? with
  | none => simp
  | some template =>
      simp only [List.all_eq_true]
      constructor
      · intro checked row member
        rcases List.mem_map.mp member with ⟨source, sourceMember, rfl⟩
        exact (compactRow_iff invocation source env).mp (checked source sourceMember)
      · intro holds row member
        exact (compactRow_iff invocation row env).mpr
          (holds (instantiateCompactRow invocation row)
            (List.mem_map.mpr ⟨row, member, rfl⟩))

theorem instruction_iff (item : WitnessInstruction) (env : Env) :
    instruction item env = true ↔ item.Holds env := by
  simp only [instruction, decide_eq_true_eq, WitnessInstruction.Holds]

theorem sparseRow_iff (row : SparseRow) (env : Env) :
    sparseRow row env = true ↔ row.Holds env := by
  simp only [sparseRow, decide_eq_true_eq, SparseRow.Holds]

end NightstreamFPrime.Export.Stage1.StoredPhysicalRowCheck
