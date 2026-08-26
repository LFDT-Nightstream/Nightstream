import NightstreamFPrime.Export.Package
import NightstreamFPrime.Layout.R1CS
import NightstreamFPrime.Circuit.StraightLine

/-!
Owns the Lean semantics of the compact circuit-package row program. Template
rows instantiate to ordinary `Layout.R1CS.Row` values. Hash-chain substitution
uses only package fields. Rust may execute this program, but it does not define
its meaning.
-/

namespace NightstreamFPrime.Export.Package

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout

def fieldValue (value : Nat) : F := Poseidon2.ofNat value

def ColumnRef.eval (value : ColumnRef → F) (column : ColumnRef) : F :=
  value column

def TemplateCombination.eval (combination : TemplateCombination)
    (value : ColumnRef → F) : F :=
  fieldValue combination.constant +
    (combination.terms.map fun term =>
      fieldValue term.coefficient * term.column.eval value).sum

def TemplateRow.Holds (row : TemplateRow) (value : ColumnRef → F) : Prop :=
  row.a.eval value * row.b.eval value = row.c.eval value

private def sumCombinations : List R1CS.LinearCombination →
    R1CS.LinearCombination
  | [] => R1CS.LinearCombination.zero
  | combination :: rest =>
      R1CS.LinearCombination.add combination (sumCombinations rest)

private theorem sumCombinations_eval (env : Env)
    (combinations : List R1CS.LinearCombination) :
    (sumCombinations combinations).eval env =
      (combinations.map fun combination => combination.eval env).sum := by
  induction combinations with
  | nil => rfl
  | cons combination rest ih =>
      simp [sumCombinations, ih]

/-- The affine input to one permutation invocation. Invocation zero starts
from zero. Later invocations use the previous permutation output. Absorb
invocations add one rate-width input block; the final invocation adds the
lane-zero `+1` padding marker. -/
def invocationInput (package : CircuitPackage) (chain : HashChain)
    (invocation lane : Nat) : R1CS.LinearCombination :=
  let previous :=
    if invocation = 0 then
      R1CS.LinearCombination.zero
    else
      R1CS.LinearCombination.ofVar
        (chain.witnessStart +
          (invocation - 1) * package.permutation.localColumnCount +
          package.permutation.outputLocalStart + lane)
  if invocation < chain.absorbCount then
    let inputOffset := invocation * package.poseidon.rate + lane
    if lane < package.poseidon.rate ∧ inputOffset < chain.inputLength then
      R1CS.LinearCombination.add previous
        (R1CS.LinearCombination.ofVar (chain.inputStart + inputOffset))
    else
      previous
  else if lane = 0 then
    R1CS.LinearCombination.add previous R1CS.LinearCombination.one
  else
    previous

def invocationLocalStart (package : CircuitPackage) (chain : HashChain)
    (invocation : Nat) : Nat :=
  chain.witnessStart + invocation * package.permutation.localColumnCount

def instantiateColumn (package : CircuitPackage) (chain : HashChain)
    (invocation : Nat) : ColumnRef → R1CS.LinearCombination
  | .input lane => invocationInput package chain invocation lane
  | .local index =>
      R1CS.LinearCombination.ofVar
        (invocationLocalStart package chain invocation + index)

def instantiateCombination (package : CircuitPackage) (chain : HashChain)
    (invocation : Nat) (combination : TemplateCombination) :
    R1CS.LinearCombination :=
  R1CS.LinearCombination.add
    (R1CS.LinearCombination.const (fieldValue combination.constant))
    (sumCombinations (combination.terms.map fun term =>
      R1CS.LinearCombination.scale (fieldValue term.coefficient)
        (instantiateColumn package chain invocation term.column)))

def instantiateRow (package : CircuitPackage) (chain : HashChain)
    (invocation : Nat) (row : TemplateRow) : R1CS.Row :=
  ⟨instantiateCombination package chain invocation row.a,
    instantiateCombination package chain invocation row.b,
    instantiateCombination package chain invocation row.c⟩

/-- Instantiation has exactly the template-row semantics under the package's
chain substitution. -/
theorem instantiateRow_holds (package : CircuitPackage) (chain : HashChain)
    (invocation : Nat) (row : TemplateRow) (env : Env) :
    (instantiateRow package chain invocation row).Holds env ↔
      row.Holds (fun column =>
        (instantiateColumn package chain invocation column).eval env) := by
  have combinationEval (combination : TemplateCombination) :
      (instantiateCombination package chain invocation combination).eval env =
        combination.eval (fun column =>
          (instantiateColumn package chain invocation column).eval env) := by
    unfold instantiateCombination TemplateCombination.eval
    simp [sumCombinations_eval, List.map_map, Function.comp_def,
      ColumnRef.eval]
  simp [R1CS.Row.Holds, instantiateRow, TemplateRow.Holds,
    combinationEval]

def TemplateInvocationHolds (package : CircuitPackage) (chain : HashChain)
    (invocation : Nat) (env : Env) : Prop :=
  ∀ row ∈ package.permutation.rows,
    (instantiateRow package chain invocation row).Holds env

def HashChainHolds (package : CircuitPackage) (chain : HashChain)
    (env : Env) : Prop :=
  ∀ invocation, invocation ≤ chain.absorbCount →
    TemplateInvocationHolds package chain invocation env

def SparseCombination.eval (combination : SparseCombination) (env : Env) : F :=
  fieldValue combination.constant +
    (combination.terms.map fun term =>
      fieldValue term.coefficient * env term.column).sum

def SparseRow.Holds (row : SparseRow) (env : Env) : Prop :=
  row.a.eval env * row.b.eval env = row.c.eval env

def SparseCombination.toR1CS (combination : SparseCombination) :
    R1CS.LinearCombination :=
  ⟨fieldValue combination.constant,
    combination.terms.map fun term =>
      (term.column, fieldValue term.coefficient)⟩

def SparseRow.toR1CS (row : SparseRow) : R1CS.Row :=
  ⟨row.a.toR1CS, row.b.toR1CS, row.c.toR1CS⟩

def zeroSparseCombination : SparseCombination := ⟨0, []⟩

def invocationInputCombination (invocation : PermutationInvocation)
    (lane : Nat) : SparseCombination :=
  invocation.inputs.getD lane zeroSparseCombination

def instantiateInvocationColumn (invocation : PermutationInvocation) :
    ColumnRef → R1CS.LinearCombination
  | .input lane => (invocationInputCombination invocation lane).toR1CS
  | .local index =>
      R1CS.LinearCombination.ofVar (invocation.witnessStart + index)

def instantiateInvocationCombination (invocation : PermutationInvocation)
    (combination : TemplateCombination) : R1CS.LinearCombination :=
  R1CS.LinearCombination.add
    (R1CS.LinearCombination.const (fieldValue combination.constant))
    (sumCombinations (combination.terms.map fun term =>
      R1CS.LinearCombination.scale (fieldValue term.coefficient)
        (instantiateInvocationColumn invocation term.column)))

def instantiateInvocationRow (invocation : PermutationInvocation)
    (row : TemplateRow) : R1CS.Row :=
  ⟨instantiateInvocationCombination invocation row.a,
    instantiateInvocationCombination invocation row.b,
    instantiateInvocationCombination invocation row.c⟩

theorem instantiateInvocationRow_holds
    (invocation : PermutationInvocation) (row : TemplateRow) (env : Env) :
    (instantiateInvocationRow invocation row).Holds env ↔
      row.Holds (fun column =>
        (instantiateInvocationColumn invocation column).eval env) := by
  have combinationEval (combination : TemplateCombination) :
      (instantiateInvocationCombination invocation combination).eval env =
        combination.eval (fun column =>
          (instantiateInvocationColumn invocation column).eval env) := by
    unfold instantiateInvocationCombination TemplateCombination.eval
    simp [sumCombinations_eval, List.map_map, Function.comp_def,
      ColumnRef.eval]
  simp [R1CS.Row.Holds, instantiateInvocationRow, TemplateRow.Holds,
    combinationEval]

def PermutationInvocationHolds (package : CircuitPackage)
    (invocation : PermutationInvocation) (env : Env) : Prop :=
  ∀ row ∈ package.permutation.rows,
    (instantiateInvocationRow invocation row).Holds env

/-- The final package column bound to one compact-template input. The strict
loader proves that the ranges form a total, ordered partition of the declared
input interval. -/
def compactInputColumn (ranges : List CompactInputRange)
    (input : Nat) : Nat :=
  match ranges.find? fun range =>
      range.inputStart ≤ input ∧ input < range.inputStart + range.inputCount with
  | some range =>
      range.columnStart + (input - range.inputStart) * range.columnStride
  | none => 0

def instantiateCompactColumn (invocation : CompactRowInvocation) :
    ColumnRef → Nat
  | .input input => compactInputColumn invocation.inputRanges input
  | .local localIndex => invocation.localStart + localIndex

def instantiateCompactCombination (invocation : CompactRowInvocation)
    (combination : TemplateCombination) : R1CS.LinearCombination :=
  ⟨fieldValue combination.constant,
    combination.terms.map fun term =>
      (instantiateCompactColumn invocation term.column,
        fieldValue term.coefficient)⟩

def instantiateCompactRow (invocation : CompactRowInvocation)
    (row : CompactTemplateRow) : R1CS.Row :=
  ⟨instantiateCompactCombination invocation row.a,
    instantiateCompactCombination invocation row.b,
    instantiateCompactCombination invocation row.c⟩

/-- One compact invocation is fail-closed on its selected template and then
checks every exact instantiated `A * B = C` row. -/
def CompactRowInvocationHolds (package : CircuitPackage)
    (invocation : CompactRowInvocation) (env : Env) : Prop :=
  match package.compactRowTemplates[invocation.templateIndex]? with
  | none => False
  | some template =>
      R1CS.RowsHold env
        (template.rows.map (instantiateCompactRow invocation))

def compactInvocationRowCountFor (templates : List CompactRowTemplate)
    (invocation : CompactRowInvocation) : Nat :=
  match templates[invocation.templateIndex]? with
  | none => 0
  | some template => template.rows.length

def compactRowCountFor (templates : List CompactRowTemplate)
    (invocations : List CompactRowInvocation) : Nat :=
  (invocations.map (compactInvocationRowCountFor templates)).sum

theorem compactRowCountFor_append (templates : List CompactRowTemplate)
    (first second : List CompactRowInvocation) :
    compactRowCountFor templates (first ++ second) =
      compactRowCountFor templates first + compactRowCountFor templates second := by
  simp [compactRowCountFor, List.map_append]

def compactInvocationRowCount (package : CircuitPackage)
    (invocation : CompactRowInvocation) : Nat :=
  compactInvocationRowCountFor package.compactRowTemplates invocation

def CircuitPackage.compactRowCount (package : CircuitPackage) : Nat :=
  compactRowCountFor package.compactRowTemplates
    package.compactRowInvocations

/-- The authoritative R1CS row checked for one generic witness instruction. -/
def WitnessInstruction.toR1CS (instruction : WitnessInstruction) : R1CS.Row :=
  ⟨instruction.a.toR1CS, instruction.b.toR1CS,
    R1CS.LinearCombination.ofVar instruction.target⟩

def WitnessInstruction.Holds (instruction : WitnessInstruction)
    (env : Env) : Prop :=
  instruction.a.eval env * instruction.b.eval env = env instruction.target

theorem witnessInstruction_toR1CS_holds
    (instruction : WitnessInstruction) (env : Env) :
    instruction.toR1CS.Holds env ↔ instruction.Holds env := by
  have combinationEval (combination : SparseCombination) :
      combination.toR1CS.eval env = combination.eval env := by
    simp [SparseCombination.toR1CS, SparseCombination.eval,
      R1CS.LinearCombination.eval, List.map_map, Function.comp_def]
  simp [R1CS.Row.Holds, WitnessInstruction.toR1CS,
    WitnessInstruction.Holds, combinationEval]

/-- An instruction input does not read the value that it is about to write. -/
def SparseCombination.Avoids (combination : SparseCombination)
    (target : Nat) : Prop :=
  ∀ term ∈ combination.terms, term.column ≠ target

theorem SparseCombination.eval_set_of_avoids
    (combination : SparseCombination) (env : Env) (target : Nat) (value : F)
    (avoids : combination.Avoids target) :
    combination.eval (Env.set env target value) = combination.eval env := by
  have termsEqual :
      (combination.terms.map fun term =>
        fieldValue term.coefficient * (Env.set env target value) term.column) =
      (combination.terms.map fun term =>
        fieldValue term.coefficient * env term.column) := by
    apply List.map_congr_left
    intro term member
    rw [Env.set_of_ne env target term.column value (avoids term member)]
  unfold SparseCombination.eval
  rw [termsEqual]

/-- Execute the non-authoritative hint carried by one instruction. -/
def WitnessInstruction.execute (instruction : WitnessInstruction)
    (env : Env) : Env :=
  Env.set env instruction.target
    (instruction.a.eval env * instruction.b.eval env)

theorem WitnessInstruction.execute_holds
    (instruction : WitnessInstruction) (env : Env)
    (aAvoids : instruction.a.Avoids instruction.target)
    (bAvoids : instruction.b.Avoids instruction.target) :
    instruction.Holds (instruction.execute env) := by
  unfold WitnessInstruction.Holds WitnessInstruction.execute
  rw [instruction.a.eval_set_of_avoids env instruction.target _ aAvoids,
    instruction.b.eval_set_of_avoids env instruction.target _ bAvoids]
  simp

theorem sparseRow_holds (row : SparseRow) (env : Env) :
    row.toR1CS.Holds env ↔ row.Holds env := by
  have combinationEval (combination : SparseCombination) :
      combination.toR1CS.eval env = combination.eval env := by
    simp [SparseCombination.toR1CS, SparseCombination.eval,
      R1CS.LinearCombination.eval, List.map_map, Function.comp_def]
  simp [R1CS.Row.Holds, SparseRow.toR1CS, SparseRow.Holds,
    combinationEval]

def AssertionsHold (package : CircuitPackage) (env : Env) : Prop :=
  ∀ row ∈ package.assertionRows, row.Holds env

/-- Authoritative row semantics of a loaded circuit package. -/
def CircuitPackage.RowsHold (package : CircuitPackage) (env : Env) : Prop :=
  (∀ chain ∈ package.hashChains, HashChainHolds package chain env) ∧
    (∀ invocation ∈ package.permutationInvocations,
      PermutationInvocationHolds package invocation env) ∧
    (∀ invocation ∈ package.compactRowInvocations,
      CompactRowInvocationHolds package invocation env) ∧
    (∀ instruction ∈ package.witnessInstructions,
      instruction.Holds env) ∧
    AssertionsHold package env

end NightstreamFPrime.Export.Package
