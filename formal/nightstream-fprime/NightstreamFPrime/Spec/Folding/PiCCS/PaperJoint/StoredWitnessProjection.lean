import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.CostedWitnessProjection

/-!
Array-backed full witnesses and the existing B.2 source projection. The
read clock counts two array lookups and result construction. Array creation
and validation belong to the producing call; no arbitrary function receives
a constant lookup cost. The semantic view only erases storage.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.StoredWitnessProjection

open NightstreamFPrime.Spec
open StrongReduction WitnessProjection
open _root_.NightstreamFPrime.Spec.Folding.PiRLC.PaperForkExtractionWork (Result)

abbrev StoredWitness (shape : Shape) (carrier : Phi81Relation.Shape) :=
  Vector (Vector F carrier.carrierWidth) shape.sourceCount

def view {shape : Shape} {carrier : Phi81Relation.Shape}
    (stored : StoredWitness shape carrier) : OutputWitness shape carrier.carrierWidth where
  assignments := fun source column => (stored.get source).get column

/-- One source-array lookup, one coefficient lookup, and the returned result. -/
def read {shape : Shape} {carrier : Phi81Relation.Shape}
    (stored : StoredWitness shape carrier) (source : Fin shape.sourceCount)
    (column : Fin carrier.carrierWidth) : Result F :=
  let row : Result (Vector F carrier.carrierWidth) := ⟨stored.get source, 1⟩
  let field : Result F := ⟨row.value.get column, 1⟩
  ⟨field.value, row.work + field.work + 1⟩

def project {shape : Shape} {carrier : Phi81Relation.Shape}
    (stored : StoredWitness shape carrier) : Result (SourceWitness shape carrier) :=
  CostedWitnessProjection.projectReads (read stored)

theorem project_value {shape : Shape} {carrier : Phi81Relation.Shape}
    (stored : StoredWitness shape carrier) :
    (project stored).value = (WitnessProjection.project carrier (view stored)).value :=
  CostedWitnessProjection.projectReads_value (read stored) (view stored) (fun _ _ => rfl)

/-- The access count is derived from the two reads and return above. The
remaining factors count the exact copied private tails and running vectors. -/
theorem project_work_le {shape : Shape} {carrier : Phi81Relation.Shape}
    (stored : StoredWitness shape carrier) :
    (project stored).work ≤ CostedWitnessProjection.workBound shape carrier (1 + 1 + 1) :=
  CostedWitnessProjection.projectReads_work_le (read stored) (1 + 1 + 1) (fun _ _ => Nat.le_refl _)

theorem reconstruct_project {shape : Shape} {carrier : Phi81Relation.Shape}
    (publicInputs : Fin shape.sourceCount → Phi81Relation.PublicInput carrier)
    (stored : StoredWitness shape carrier)
    (prefixMatches : ∀ source : Fin shape.freshCount,
      Phi81Relation.projectPublicInput ((view stored).assignments (UnifiedSources.freshSourceIndex source)) =
        publicInputs (UnifiedSources.freshSourceIndex source)) :
    reconstruct publicInputs (project stored).value = view stored := by
  rw [project_value]
  exact WitnessProjection.reconstruct_project carrier publicInputs (view stored) prefixMatches

end NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.StoredWitnessProjection
