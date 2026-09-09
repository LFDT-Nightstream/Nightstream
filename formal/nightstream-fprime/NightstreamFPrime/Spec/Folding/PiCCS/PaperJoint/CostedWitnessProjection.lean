import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.WitnessProjection

/-!
The B.2 projection executes a field accessor which returns its value and work
from the same call. Its clock includes source lookup, coordinate lookup, and
representation work. The driver charges each vector constructor and return.
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

private def collect {Value : Type*} : {count : Nat} →
    (Fin count → Result Value) → Result (List.Vector Value count)
  | 0, _ => ⟨List.Vector.nil, 1⟩
  | _ + 1, read =>
      let head := read 0
      let tail := collect (fun index => read index.succ)
      ⟨List.Vector.cons head.value tail.value, head.work + tail.work + 1⟩

private theorem collect_get {Value : Type*} : ∀ {count : Nat}
    (read : Fin count → Result Value) (index : Fin count),
    (collect read).value.get index = (read index).value
  | 0, _, index => Fin.elim0 index
  | _ + 1, read, index => by
      refine Fin.cases ?_ (fun prior => ?_) index
      · simp only [collect, List.Vector.get_cons_zero]
      · simpa only [collect, List.Vector.get_cons_succ] using
          collect_get (fun index => read index.succ) prior

private theorem collect_work_le {Value : Type*} (bound : Nat) : ∀ {count : Nat}
    (read : Fin count → Result Value),
    (∀ index, (read index).work ≤ bound) →
    (collect read).work ≤ count * (bound + 1) + 1
  | 0, _, _ => by simp only [collect, Nat.zero_mul, Nat.zero_add, Nat.le_refl]
  | count + 1, read, bounded => by
      have head := bounded 0
      have tail := collect_work_le bound (fun index => read index.succ)
        (fun index => bounded index.succ)
      change (read 0).work + (collect (fun index => read index.succ)).work + 1 ≤ _
      rw [Nat.add_mul, Nat.one_mul]
      omega

/-- Execute the accessor once at each returned coordinate, in source order. -/
def project {shape : Shape} {carrier : Phi81Relation.Shape}
    (access : Accessor shape carrier) (witness : OutputWitness shape carrier.carrierWidth) :
    Result (SourceWitness shape carrier) :=
  let fresh := collect (fun source : Fin shape.freshCount =>
    collect (fun column => access witness (freshSourceIndex source) (privateColumn carrier column)))
  let running := collect (fun source : Fin shape.runningCount =>
    collect (access witness (runningSourceIndex source)))
  ⟨⟨fresh.value, running.value⟩, fresh.work + running.work + 1⟩

theorem project_fresh {shape : Shape} {carrier : Phi81Relation.Shape}
    (access : Accessor shape carrier) (correct : Correct access)
    (witness : OutputWitness shape carrier.carrierWidth)
    (source : Fin shape.freshCount) (column : Fin (privateWidth carrier)) :
    ((project access witness).value.fresh.get source).get column =
      witness.assignments (freshSourceIndex source) (privateColumn carrier column) := by
  simp only [project, collect_get]
  exact correct witness (freshSourceIndex source) (privateColumn carrier column)

theorem project_running {shape : Shape} {carrier : Phi81Relation.Shape}
    (access : Accessor shape carrier) (correct : Correct access)
    (witness : OutputWitness shape carrier.carrierWidth)
    (source : Fin shape.runningCount) (column : Fin carrier.carrierWidth) :
    ((project access witness).value.running.get source).get column =
      witness.assignments (runningSourceIndex source) column := by
  simp only [project, collect_get]
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
theorem project_value {shape : Shape} {carrier : Phi81Relation.Shape}
    (access : Accessor shape carrier) (correct : Correct access)
    (witness : OutputWitness shape carrier.carrierWidth) :
    (project access witness).value = (WitnessProjection.project carrier witness).value := by
  apply sourceWitness_ext
  · apply List.Vector.ext
    intro source
    apply List.Vector.ext
    intro column
    rw [project_fresh access correct, WitnessProjection.project_fresh]
  · apply List.Vector.ext
    intro source
    apply List.Vector.ext
    intro column
    rw [project_running access correct, WitnessProjection.project_running]

/-- Reads, constructors, and result return for the actual access bound. -/
def workBound (shape : Shape) (carrier : Phi81Relation.Shape) (accessBound : Nat) : Nat :=
  shape.freshCount * (privateWidth carrier * (accessBound + 1) + 2) +
    shape.runningCount * (carrier.carrierWidth * (accessBound + 1) + 2) + 3

theorem project_work_le {shape : Shape} {carrier : Phi81Relation.Shape}
    (access : Accessor shape carrier) (accessBound : Nat) (bounded : Bounded access accessBound)
    (witness : OutputWitness shape carrier.carrierWidth) :
    (project access witness).work ≤ workBound shape carrier accessBound := by
  have freshWork := collect_work_le (privateWidth carrier * (accessBound + 1) + 1)
    (fun source : Fin shape.freshCount => collect (fun column =>
      access witness (freshSourceIndex source) (privateColumn carrier column)))
    (fun source => collect_work_le accessBound _
      (fun column => bounded witness (freshSourceIndex source) (privateColumn carrier column)))
  have runningWork := collect_work_le (carrier.carrierWidth * (accessBound + 1) + 1)
    (fun source : Fin shape.runningCount => collect (access witness (runningSourceIndex source)))
    (fun source => collect_work_le accessBound _
      (fun column => bounded witness (runningSourceIndex source) column))
  dsimp only [project]
  unfold workBound
  simp only [Nat.add_assoc, Nat.reduceAdd] at freshWork runningWork
  omega

end NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.CostedWitnessProjection
