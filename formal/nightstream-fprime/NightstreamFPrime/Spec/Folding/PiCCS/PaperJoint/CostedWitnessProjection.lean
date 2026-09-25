import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.WitnessProjection

/-!
The B.2 projection executes a field accessor which returns its value and work
from the same call. Its clock includes source lookup, coordinate lookup, and
representation work. The direct-index driver charges list construction,
index movement, the final reverse, and returns.
The bound on this accessor is an explicit implementation premise; arbitrary
assignment functions are not assigned a unit cost.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.CostedWitnessProjection

open NightstreamFPrime.Spec
open StrongReduction UnifiedSources WitnessProjection
open _root_.NightstreamFPrime.Spec.Folding.PiRLC.PaperForkExtractionWork (Result)

abbrev Accessor (shape : Shape) (carrier : Phi81Relation.Shape) :=
  OutputWitness shape carrier.carrierWidth → Fin shape.sourceCount →
    Fin carrier.carrierWidth → Result F

/-- The accessor reads the same full witness returned by the adversary. -/
def Correct {shape : Shape} {carrier : Phi81Relation.Shape}
    (access : Accessor shape carrier) : Prop :=
  ∀ witness source column, (access witness source column).value = witness.assignments source column

/-- The work includes every operation of the selected accessor implementation. -/
def Bounded {shape : Shape} {carrier : Phi81Relation.Shape}
    (access : Accessor shape carrier) (bound : Nat) : Prop :=
  ∀ witness source column, (access witness source column).work ≤ bound

private def collectAction {Value : Type} {count : Nat}
    (read : Fin count → Result Value) (index : Fin count) : StateM Nat Value := fun work =>
  let result := read index
  (result.value, work + result.work + 3)

private theorem collect_state_value {Value : Type} {count : Nat}
    (read : Fin count → Result Value) (initial : Nat) :
    ((List.ofFnM (collectAction read)).run initial).1 =
      List.ofFn (fun index => (read index).value) := by
  induction count generalizing initial with
  | zero => rw [List.ofFnM_zero, List.ofFn_zero]; rfl
  | succ count ih =>
      rw [List.ofFnM_succ_last, List.ofFn_succ_last]
      simp only [StateT.run_bind, StateT.run_pure]
      change (((List.ofFnM (collectAction (fun index => read index.castSucc))).run initial).1) ++
        [(read (Fin.last count)).value] = _
      rw [ih]

private theorem collect_state_work {Value : Type} {count : Nat}
    (read : Fin count → Result Value) (bound : Nat)
    (bounded : ∀ index, (read index).work ≤ bound) (initial : Nat) :
    ((List.ofFnM (collectAction read)).run initial).2 ≤ initial + count * (bound + 3) := by
  induction count generalizing initial with
  | zero =>
      rw [List.ofFnM_zero]
      change initial ≤ initial + 0 * (bound + 3)
      omega
  | succ count ih =>
      rw [List.ofFnM_succ_last]
      simp only [StateT.run_bind, StateT.run_pure]
      change ((List.ofFnM (collectAction (fun index => read index.castSucc))).run initial).2 +
        (read (Fin.last count)).work + 3 ≤ _
      have previous := ih (fun index => read index.castSucc)
        (fun index => bounded index.castSucc) initial
      have last := bounded (Fin.last count)
      rw [Nat.add_mul]
      omega

/-- The library loop uses each original index directly, accumulates a list,
then reverses it once. Count branch/index/cons for each forward step and
match/tail/cons for each reverse step, plus initialization and returns. -/
private def collect {Value : Type} {count : Nat}
    (read : Fin count → Result Value) : Result (List.Vector Value count) :=
  let result := (List.ofFnM (collectAction read)).run 1
  ⟨⟨result.1, by rw [collect_state_value]; exact List.length_ofFn⟩,
    result.2 + 3 * count + 2⟩

private theorem collect_get {Value : Type} {count : Nat}
    (read : Fin count → Result Value) (index : Fin count) :
    (collect read).value.get index = (read index).value := by
  simp [collect, List.Vector.get, collect_state_value, List.get_eq_getElem]

private theorem collect_work_le {Value : Type} (bound : Nat) {count : Nat}
    (read : Fin count → Result Value) (bounded : ∀ index, (read index).work ≤ bound) :
    (collect read).work ≤ count * (bound + 6) + 3 := by
  have boundState := collect_state_work read bound bounded 1
  change ((List.ofFnM (collectAction read)).run 1).2 + 3 * count + 2 ≤ _
  calc
    _ ≤ (1 + count * (bound + 3)) + 3 * count + 2 := by omega
    _ = _ := by ring

/-- Copy one concrete read program in source order. The read program can
retain its actual array representation instead of an erased function. -/
def projectReads {shape : Shape} {carrier : Phi81Relation.Shape}
    (read : Fin shape.sourceCount → Fin carrier.carrierWidth → Result F) :
    Result (SourceWitness shape carrier) :=
  let fresh := collect (fun source : Fin shape.freshCount =>
    collect (fun column => read (freshSourceIndex source) (privateColumn carrier column)))
  let running := collect (fun source : Fin shape.runningCount =>
    collect (read (runningSourceIndex source)))
  ⟨⟨fresh.value, running.value⟩, fresh.work + running.work + 1⟩

/-- Execute the supplied semantic-witness accessor through the same copier. -/
def project {shape : Shape} {carrier : Phi81Relation.Shape}
    (access : Accessor shape carrier) (witness : OutputWitness shape carrier.carrierWidth) :
    Result (SourceWitness shape carrier) := projectReads (access witness)

theorem project_fresh {shape : Shape} {carrier : Phi81Relation.Shape}
    (access : Accessor shape carrier) (correct : Correct access)
    (witness : OutputWitness shape carrier.carrierWidth)
    (source : Fin shape.freshCount) (column : Fin (privateWidth carrier)) :
    ((project access witness).value.fresh.get source).get column =
      witness.assignments (freshSourceIndex source) (privateColumn carrier column) := by
  simp only [project, projectReads, collect_get]
  exact correct witness (freshSourceIndex source) (privateColumn carrier column)

theorem project_running {shape : Shape} {carrier : Phi81Relation.Shape}
    (access : Accessor shape carrier) (correct : Correct access)
    (witness : OutputWitness shape carrier.carrierWidth)
    (source : Fin shape.runningCount) (column : Fin carrier.carrierWidth) :
    ((project access witness).value.running.get source).get column =
      witness.assignments (runningSourceIndex source) column := by
  simp only [project, projectReads, collect_get]
  exact correct witness (runningSourceIndex source) column

private theorem sourceWitness_ext {shape : Shape} {carrier : Phi81Relation.Shape}
    {left right : SourceWitness shape carrier}
    (fresh : left.fresh = right.fresh) (running : left.running = right.running) : left = right := by
  cases left
  cases right
  cases fresh
  cases running
  rfl

/-- Cost erasure returns exactly the canonical tails and full running vectors. -/
theorem projectReads_value {shape : Shape} {carrier : Phi81Relation.Shape}
    (read : Fin shape.sourceCount → Fin carrier.carrierWidth → Result F)
    (witness : OutputWitness shape carrier.carrierWidth)
    (correct : ∀ source column, (read source column).value = witness.assignments source column) :
    (projectReads read).value = (WitnessProjection.project carrier witness).value := by
  apply sourceWitness_ext
  · apply List.Vector.ext
    intro source
    apply List.Vector.ext
    intro column
    simp only [projectReads, collect_get, WitnessProjection.project_fresh, correct]
  · apply List.Vector.ext
    intro source
    apply List.Vector.ext
    intro column
    simp only [projectReads, collect_get, WitnessProjection.project_running, correct]

theorem project_value {shape : Shape} {carrier : Phi81Relation.Shape}
    (access : Accessor shape carrier) (correct : Correct access)
    (witness : OutputWitness shape carrier.carrierWidth) :
    (project access witness).value = (WitnessProjection.project carrier witness).value :=
  projectReads_value (access witness) witness (correct witness)

/-- Reads, constructors, and result return for the actual access bound. -/
def workBound (shape : Shape) (carrier : Phi81Relation.Shape) (accessBound : Nat) : Nat :=
  shape.freshCount * (privateWidth carrier * (accessBound + 6) + 9) +
    shape.runningCount * (carrier.carrierWidth * (accessBound + 6) + 9) + 7

theorem projectReads_work_le {shape : Shape} {carrier : Phi81Relation.Shape}
    (read : Fin shape.sourceCount → Fin carrier.carrierWidth → Result F)
    (accessBound : Nat) (bounded : ∀ source column, (read source column).work ≤ accessBound) :
    (projectReads read).work ≤ workBound shape carrier accessBound := by
  have freshWork := collect_work_le (privateWidth carrier * (accessBound + 6) + 3)
    (fun source : Fin shape.freshCount => collect (fun column =>
      read (freshSourceIndex source) (privateColumn carrier column)))
    (fun source => collect_work_le accessBound _
      (fun column => bounded (freshSourceIndex source) (privateColumn carrier column)))
  have runningWork := collect_work_le (carrier.carrierWidth * (accessBound + 6) + 3)
    (fun source : Fin shape.runningCount => collect (read (runningSourceIndex source)))
    (fun source => collect_work_le accessBound _
      (fun column => bounded (runningSourceIndex source) column))
  dsimp only [projectReads]
  unfold workBound
  simp only [Nat.add_assoc, Nat.reduceAdd] at freshWork runningWork
  omega

theorem project_work_le {shape : Shape} {carrier : Phi81Relation.Shape}
    (access : Accessor shape carrier) (accessBound : Nat) (bounded : Bounded access accessBound)
    (witness : OutputWitness shape carrier.carrierWidth) :
    (project access witness).work ≤ workBound shape carrier accessBound :=
  projectReads_work_le (access witness) accessBound (bounded witness)

end NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.CostedWitnessProjection
