import NightstreamFPrime.Layout.R1CS.Completeness
import NightstreamFPrime.Layout.R1CS.Support
import NightstreamFPrime.Layout.Stage1.Spartan

namespace NightstreamFPrime.Layout.Stage1.Spartan

open NightstreamFPrime.Layout

theorem remapCombination_varsBelow (combination : R1CS.LinearCombination)
    (scope : combination.VarsBelow SourceColumnCount) :
    (remapCombination combination).VarsBelow spartanColumnCount := by
  intro term member
  simp only [remapCombination, List.mem_map] at member
  rcases member with ⟨source, sourceMember, rfl⟩
  exact sourceToSpartan_lt source.1 (scope source sourceMember)

theorem remapRow_varsBelow (row : R1CS.Row)
    (scope : row.VarsBelow SourceColumnCount) :
    (remapRow row).VarsBelow spartanColumnCount := by
  exact ⟨remapCombination_varsBelow row.a scope.1,
    remapCombination_varsBelow row.b scope.2.1,
    remapCombination_varsBelow row.c scope.2.2⟩

theorem remapRows_varsBelow (rows : List R1CS.Row)
    (scope : ∀ row ∈ rows, row.VarsBelow SourceColumnCount) :
    ∀ row ∈ remapRows rows, row.VarsBelow spartanColumnCount := by
  intro row member
  simp only [remapRows, List.mem_map] at member
  rcases member with ⟨source, sourceMember, rfl⟩
  exact remapRow_varsBelow source (scope source sourceMember)

theorem remapCombination_varsSatisfy (sourceTarget : Nat → Prop)
    (target : Nat → Prop) (combination : R1CS.LinearCombination)
    (scope : combination.VarsSatisfy sourceTarget)
    (transport : ∀ column, sourceTarget column →
      target (sourceToSpartan column)) :
    (remapCombination combination).VarsSatisfy target := by
  intro term member
  simp only [remapCombination, List.mem_map] at member
  rcases member with ⟨source, sourceMember, rfl⟩
  exact transport source.1 (scope source sourceMember)

theorem remapRow_varsSatisfy (sourceTarget target : Nat → Prop)
    (row : R1CS.Row) (scope : row.VarsSatisfy sourceTarget)
    (transport : ∀ column, sourceTarget column →
      target (sourceToSpartan column)) :
    (remapRow row).VarsSatisfy target := by
  exact ⟨remapCombination_varsSatisfy sourceTarget target row.a scope.1
      transport,
    remapCombination_varsSatisfy sourceTarget target row.b scope.2.1
      transport,
    remapCombination_varsSatisfy sourceTarget target row.c scope.2.2
      transport⟩

theorem remapRows_varsSatisfy (sourceTarget target : Nat → Prop)
    (rows : List R1CS.Row)
    (scope : ∀ row ∈ rows, row.VarsSatisfy sourceTarget)
    (transport : ∀ column, sourceTarget column →
      target (sourceToSpartan column)) :
    ∀ row ∈ remapRows rows, row.VarsSatisfy target := by
  intro row member
  simp only [remapRows, List.mem_map] at member
  rcases member with ⟨source, sourceMember, rfl⟩
  exact remapRow_varsSatisfy sourceTarget target source
    (scope source sourceMember) transport

end NightstreamFPrime.Layout.Stage1.Spartan
