import Nightstream.Implementation.R1CS.Correspondence.FieldEncoding.InactiveNoninterference

/-!
Executable theorem regressions and premise-necessity witnesses for inactive
field noninterference.

| Witness | Dropped premise | Observed failure |
|---|---|---|
| `changeConfinement_is_necessary` | assignments agree outside inactive support | selected acceptance changes through an undeclared column |
| `selectorDisjointness_is_necessary` | inactive/selector disjointness | the chosen branch changes |
| `selectedDisjointness_is_necessary` | inactive/selected-equation disjointness | selected acceptance changes |
| `authorityDisjointness_is_necessary` | inactive/authority-output disjointness | semantic output changes |
| `rightAlwaysOn_is_necessary_for_soundness` | new assignment satisfies always-on gates/norm | accepted old assignment maps to rejected new assignment |
| `leftAlwaysOn_is_necessary_for_completeness` | old assignment satisfies always-on gates/norm | accepted new assignment maps to rejected old assignment |
| final three witnesses | exact read-support proof | an observer changes despite agreement on its falsely declared support |
-/

namespace NightstreamTests.InactiveNoninterference

open Nightstream.Implementation.R1CS.InactiveFieldNoninterference

private def allFalse : Assignment Bool := fun _ => false

private def trueAtZero : Assignment Bool
  | 0 => true
  | _ => false

private def trueAtTwo : Assignment Bool
  | 2 => true
  | _ => false

private def safeSupports : SupportManifest Bool where
  selector := Support.ofList [0]
  inactive
    | false => Support.ofList [2]
    | true => Support.ofList [3]
  selectedEquations
    | false => Support.ofList [1]
    | true => Support.ofList [4]
  authorityOutput
    | false => Support.ofList [5]
    | true => Support.ofList [6]

private def safeBoundary : Boundary Bool Bool Bool where
  supports := safeSupports
  selector assignment := assignment 0
  alwaysOn assignment := assignment 7 = false
  selectedEquations branch assignment :=
    if branch then assignment 4 = false else assignment 1 = false
  authorityOutput branch assignment :=
    if branch then assignment 6 else assignment 5
  selectorReads := by
    intro left right agrees
    exact agrees 0 (by simp [safeSupports, Support.ofList])
  selectedEquationsRead := by
    intro branch left right agrees
    cases branch <;> simp only [Bool.false_eq_true, ↓reduceIte]
    · rw [agrees 1 (by simp [safeSupports, Support.ofList])]
    · rw [agrees 4 (by simp [safeSupports, Support.ofList])]
  authorityOutputReads := by
    intro branch left right agrees
    cases branch <;> simp only [Bool.false_eq_true, ↓reduceIte]
    · exact agrees 5 (by simp [safeSupports, Support.ofList])
    · exact agrees 6 (by simp [safeSupports, Support.ofList])

/-- Non-vacuous positive regression: changing the inactive false-branch column
preserves the selected equation, always-on gate, and authority output. -/
example :
    safeBoundary.selector allFalse = safeBoundary.selector trueAtTwo ∧
      (safeBoundary.Accepts allFalse ↔ safeBoundary.Accepts trueAtTwo) ∧
      safeBoundary.Output allFalse = safeBoundary.Output trueAtTwo := by
  apply inactiveNoninterference safeBoundary
  · intro column outside
    simp [safeBoundary, safeSupports, Support.ofList, allFalse] at outside
    cases column with
    | zero => rfl
    | succ column =>
        cases column with
        | zero => rfl
        | succ column =>
            cases column with
            | zero => exact (outside rfl).elim
            | succ => rfl
  · simp [SupportsDisjoint, safeBoundary, safeSupports, Support.ofList, allFalse]
  · simp [SupportsDisjoint, safeBoundary, safeSupports, Support.ofList, allFalse]
  · simp [SupportsDisjoint, safeBoundary, safeSupports, Support.ofList, allFalse]
  · rfl
  · rfl

/-- Without confinement, even supports disjoint from the declared inactive set
do not prevent an undeclared selected column from changing. -/
theorem changeConfinement_is_necessary :
    let inactive : Support := Support.ofList []
    let selectedSupport : Support := Support.ofList [0]
    let selected : Assignment Bool → Prop := fun assignment =>
      assignment 0 = false
    SupportsDisjoint inactive selectedSupport ∧
      PredicateReadsOnly selectedSupport selected ∧
      selected allFalse ∧ ¬ selected trueAtZero := by
  dsimp
  refine ⟨by simp [SupportsDisjoint, Support.ofList], ?_, by decide, by decide⟩
  intro left right agrees
  change left 0 = false ↔ right 0 = false
  rw [agrees 0 (by simp [Support.ofList])]

/-- If the inactive set contains the selector support, the selected branch can
change even though the mutation is otherwise perfectly confined. -/
theorem selectorDisjointness_is_necessary :
    ConfinedTo (Support.ofList [0]) allFalse trueAtZero ∧
      ValueReadsOnly (Support.ofList [0])
        (fun assignment : Assignment Bool => assignment 0) ∧
      allFalse 0 ≠ trueAtZero 0 := by
  refine ⟨?_, ?_, by decide⟩
  · intro column outside
    cases column with
    | zero => simp [Support.ofList] at outside
    | succ => rfl
  · intro left right agrees
    exact agrees 0 (by simp [Support.ofList])

/-- If a selected equation reads inactive support, selected acceptance can
change while the selector remains fixed. -/
theorem selectedDisjointness_is_necessary :
    let selected : Assignment Bool → Prop := fun assignment =>
      assignment 0 = false
    ConfinedTo (Support.ofList [0]) allFalse trueAtZero ∧
      PredicateReadsOnly (Support.ofList [0]) selected ∧
      selected allFalse ∧ ¬ selected trueAtZero := by
  dsimp
  refine ⟨?_, ?_, by decide, by decide⟩
  · intro column outside
    cases column with
    | zero => simp [Support.ofList] at outside
    | succ => rfl
  · intro left right agrees
    change left 0 = false ↔ right 0 = false
    rw [agrees 0 (by simp [Support.ofList])]

/-- Authority-visible outputs require their own support disjointness; selected
equation invariance alone says nothing about them. -/
theorem authorityDisjointness_is_necessary :
    let output : Assignment Bool → Bool := fun assignment => assignment 0
    ConfinedTo (Support.ofList [0]) allFalse trueAtZero ∧
      ValueReadsOnly (Support.ofList [0]) output ∧
      output allFalse ≠ output trueAtZero := by
  dsimp
  refine ⟨?_, ?_, by decide⟩
  · intro column outside
    cases column with
    | zero => simp [Support.ofList] at outside
    | succ => rfl
  · intro left right agrees
    exact agrees 0 (by simp [Support.ofList])

/-- Soundness needs the changed assignment to retain every always-on
encoding/norm obligation. -/
theorem rightAlwaysOn_is_necessary_for_soundness :
    let alwaysOn : Assignment Bool → Prop := fun assignment =>
      assignment 0 = false
    let selected : Assignment Bool → Prop := fun assignment =>
      assignment 1 = false
    SupportsDisjoint (Support.ofList [0]) (Support.ofList [1]) ∧
      ConfinedTo (Support.ofList [0]) allFalse trueAtZero ∧
      PredicateReadsOnly (Support.ofList [1]) selected ∧
      (alwaysOn allFalse ∧ selected allFalse) ∧
      ¬ (alwaysOn trueAtZero ∧ selected trueAtZero) := by
  dsimp
  refine ⟨by simp [SupportsDisjoint, Support.ofList], ?_, ?_, by decide, by decide⟩
  · intro column outside
    cases column with
    | zero => simp [Support.ofList] at outside
    | succ => rfl
  · intro left right agrees
    change left 1 = false ↔ right 1 = false
    rw [agrees 1 (by simp [Support.ofList])]

/-- Completeness symmetrically needs the original assignment to retain every
always-on encoding/norm obligation. -/
theorem leftAlwaysOn_is_necessary_for_completeness :
    let alwaysOn : Assignment Bool → Prop := fun assignment =>
      assignment 0 = false
    let selected : Assignment Bool → Prop := fun assignment =>
      assignment 1 = false
    SupportsDisjoint (Support.ofList [0]) (Support.ofList [1]) ∧
      ConfinedTo (Support.ofList [0]) trueAtZero allFalse ∧
      PredicateReadsOnly (Support.ofList [1]) selected ∧
      (alwaysOn allFalse ∧ selected allFalse) ∧
      ¬ (alwaysOn trueAtZero ∧ selected trueAtZero) := by
  dsimp
  refine ⟨by simp [SupportsDisjoint, Support.ofList], ?_, ?_, by decide, by decide⟩
  · intro column outside
    cases column with
    | zero => simp [Support.ofList] at outside
    | succ => rfl
  · intro left right agrees
    change left 1 = false ↔ right 1 = false
    rw [agrees 1 (by simp [Support.ofList])]

/-- A false selector support declaration cannot replace the read-support
proof required by `Boundary`. -/
theorem selectorReadProof_is_necessary :
    ¬ ValueReadsOnly (Support.ofList [])
      (fun assignment : Assignment Bool => assignment 0) := by
  intro reads
  have changed := reads allFalse trueAtZero (by simp [AgreesOn, Support.ofList])
  contradiction

/-- A false selected-equation support declaration is rejected by the
predicate read-support contract. -/
theorem selectedReadProof_is_necessary :
    ¬ PredicateReadsOnly (Support.ofList [])
      (fun assignment : Assignment Bool => assignment 0 = false) := by
  intro reads
  have changed := reads allFalse trueAtZero (by simp [AgreesOn, Support.ofList])
  simp [allFalse, trueAtZero] at changed

/-- A false authority-output support declaration is likewise rejected. -/
theorem authorityReadProof_is_necessary :
    ¬ ValueReadsOnly (Support.ofList [])
      (fun assignment : Assignment Bool => assignment 0) :=
  selectorReadProof_is_necessary

end NightstreamTests.InactiveNoninterference
