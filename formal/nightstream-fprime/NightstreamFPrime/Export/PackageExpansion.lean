import NightstreamFPrime.Export.RowSemantics

/-!
Owns one generic R1CS expansion semantics for a circuit package. The package
may group rows as hash chains, explicit permutation invocations, compact
template invocations, witness instructions, and assertions. This module
places those exact rows in one list and proves that list satisfaction is
equivalent to the grouped package predicate.

The expansion is generic in the package value. No production artifact is
evaluated in the kernel, and row order inside a protocol phase remains a
separate layout-refinement obligation.
-/

namespace NightstreamFPrime.Export.Package

open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout

def hashChainRows (package : CircuitPackage) (chain : HashChain) :
    List R1CS.Row :=
  (List.range (chain.absorbCount + 1)).flatMap fun invocation =>
    package.permutation.rows.map
      (instantiateRow package chain invocation)

def CircuitPackage.hashRows (package : CircuitPackage) : List R1CS.Row :=
  package.hashChains.flatMap (hashChainRows package)

def permutationInvocationRows (package : CircuitPackage)
    (invocation : PermutationInvocation) : List R1CS.Row :=
  package.permutation.rows.map (instantiateInvocationRow invocation)

def CircuitPackage.permutationRows (package : CircuitPackage) :
    List R1CS.Row :=
  package.permutationInvocations.flatMap
    (permutationInvocationRows package)

def compactInvocationRows (package : CircuitPackage)
    (invocation : CompactRowInvocation) : List R1CS.Row :=
  match package.compactRowTemplates[invocation.templateIndex]? with
  | none => []
  | some template => template.rows.map (instantiateCompactRow invocation)

def CircuitPackage.compactRows (package : CircuitPackage) : List R1CS.Row :=
  package.compactRowInvocations.flatMap (compactInvocationRows package)

def CircuitPackage.CompactTemplatesValid (package : CircuitPackage) : Prop :=
  ∀ invocation ∈ package.compactRowInvocations,
    ∃ template,
      package.compactRowTemplates[invocation.templateIndex]? = some template

def CircuitPackage.instructionRows (package : CircuitPackage) :
    List R1CS.Row :=
  package.witnessInstructions.map WitnessInstruction.toR1CS

def CircuitPackage.assertionR1CSRows (package : CircuitPackage) :
    List R1CS.Row :=
  package.assertionRows.map SparseRow.toR1CS

/-- One exact row list for the grouped package semantics. Category order is
canonical for this expansion; protocol row-index order is proved separately. -/
def CircuitPackage.expandedRows (package : CircuitPackage) : List R1CS.Row :=
  package.hashRows ++ package.permutationRows ++ package.compactRows ++
    package.instructionRows ++ package.assertionR1CSRows

private theorem hashChainRows_hold_iff (package : CircuitPackage)
    (chain : HashChain) (env : Env) :
    R1CS.RowsHold env (hashChainRows package chain) ↔
      HashChainHolds package chain env := by
  constructor
  · intro rows invocation bound row member
    apply rows (instantiateRow package chain invocation row)
    apply List.mem_flatMap.mpr
    exact ⟨invocation, List.mem_range.mpr (by omega),
      List.mem_map.mpr ⟨row, member, rfl⟩⟩
  · intro holds row member
    rcases List.mem_flatMap.mp member with
      ⟨invocation, invocationMember, rowMember⟩
    rcases List.mem_map.mp rowMember with ⟨source, sourceMember, rfl⟩
    exact holds invocation (by
      have below := List.mem_range.mp invocationMember
      omega) source sourceMember

theorem hashRows_hold_iff (package : CircuitPackage) (env : Env) :
    R1CS.RowsHold env package.hashRows ↔
      ∀ chain ∈ package.hashChains, HashChainHolds package chain env := by
  constructor
  · intro rows chain chainMember
    apply (hashChainRows_hold_iff package chain env).mp
    intro row member
    exact rows row (List.mem_flatMap.mpr
      ⟨chain, chainMember, member⟩)
  · intro holds row member
    rcases List.mem_flatMap.mp member with ⟨chain, chainMember, rowMember⟩
    exact (hashChainRows_hold_iff package chain env).mpr
      (holds chain chainMember) row rowMember

theorem permutationRows_hold_iff (package : CircuitPackage) (env : Env) :
    R1CS.RowsHold env package.permutationRows ↔
      ∀ invocation ∈ package.permutationInvocations,
        PermutationInvocationHolds package invocation env := by
  constructor
  · intro rows invocation invocationMember row rowMember
    apply rows (instantiateInvocationRow invocation row)
    apply List.mem_flatMap.mpr
    exact ⟨invocation, invocationMember,
      List.mem_map.mpr ⟨row, rowMember, rfl⟩⟩
  · intro holds row member
    rcases List.mem_flatMap.mp member with
      ⟨invocation, invocationMember, rowMember⟩
    rcases List.mem_map.mp rowMember with ⟨source, sourceMember, rfl⟩
    exact holds invocation invocationMember source sourceMember

theorem compactRows_hold_iff (package : CircuitPackage) (env : Env) :
    (∀ invocation ∈ package.compactRowInvocations,
        CompactRowInvocationHolds package invocation env) ↔
      package.CompactTemplatesValid ∧
        R1CS.RowsHold env package.compactRows := by
  constructor
  · intro holds
    constructor
    · intro invocation invocationMember
      cases templateEq :
          package.compactRowTemplates[invocation.templateIndex]? with
      | none =>
          have impossible := holds invocation invocationMember
          simp [CompactRowInvocationHolds, templateEq] at impossible
      | some template => exact ⟨template, rfl⟩
    · intro row member
      rcases List.mem_flatMap.mp member with
        ⟨invocation, invocationMember, rowMember⟩
      cases templateEq :
          package.compactRowTemplates[invocation.templateIndex]? with
      | none => simp [compactInvocationRows, templateEq] at rowMember
      | some template =>
          have invocationHolds := holds invocation invocationMember
          unfold CompactRowInvocationHolds at invocationHolds
          rw [templateEq] at invocationHolds
          exact invocationHolds row (by
            simpa [compactInvocationRows, templateEq] using rowMember)
  · rintro ⟨valid, rows⟩ invocation invocationMember
    obtain ⟨template, templateEq⟩ := valid invocation invocationMember
    unfold CompactRowInvocationHolds
    rw [templateEq]
    intro row member
    apply rows row
    apply List.mem_flatMap.mpr
    exact ⟨invocation, invocationMember, by
      simpa [compactInvocationRows, templateEq] using member⟩

private theorem instructionRows_hold_iff (package : CircuitPackage)
    (env : Env) :
    R1CS.RowsHold env package.instructionRows ↔
      ∀ instruction ∈ package.witnessInstructions,
        instruction.Holds env := by
  constructor
  · intro rows instruction member
    exact (witnessInstruction_toR1CS_holds instruction env).mp
      (rows instruction.toR1CS (List.mem_map_of_mem member))
  · intro holds row member
    rcases List.mem_map.mp member with ⟨instruction, sourceMember, rfl⟩
    exact (witnessInstruction_toR1CS_holds instruction env).mpr
      (holds instruction sourceMember)

private theorem assertionRows_hold_iff (package : CircuitPackage)
    (env : Env) :
    R1CS.RowsHold env package.assertionR1CSRows ↔
      AssertionsHold package env := by
  constructor
  · intro rows assertion member
    exact (sparseRow_holds assertion env).mp
      (rows assertion.toR1CS (List.mem_map_of_mem member))
  · intro holds row member
    rcases List.mem_map.mp member with ⟨assertion, sourceMember, rfl⟩
    exact (sparseRow_holds assertion env).mpr
      (holds assertion sourceMember)

/-- The grouped package predicate is exactly one expanded R1CS row predicate,
plus fail-closed validity of every compact template reference. -/
theorem rowsHold_iff_expandedRows (package : CircuitPackage) (env : Env) :
    package.RowsHold env ↔
      package.CompactTemplatesValid ∧
        R1CS.RowsHold env package.expandedRows := by
  rw [show package.RowsHold env =
      ((∀ chain ∈ package.hashChains, HashChainHolds package chain env) ∧
        (∀ invocation ∈ package.permutationInvocations,
          PermutationInvocationHolds package invocation env) ∧
        (∀ invocation ∈ package.compactRowInvocations,
          CompactRowInvocationHolds package invocation env) ∧
        (∀ instruction ∈ package.witnessInstructions,
          instruction.Holds env) ∧ AssertionsHold package env) by rfl]
  rw [← hashRows_hold_iff package env,
    ← permutationRows_hold_iff package env,
    compactRows_hold_iff package env,
    ← instructionRows_hold_iff package env,
    ← assertionRows_hold_iff package env]
  simp only [CircuitPackage.expandedRows, R1CS.rowsHold_append]
  tauto

end NightstreamFPrime.Export.Package
