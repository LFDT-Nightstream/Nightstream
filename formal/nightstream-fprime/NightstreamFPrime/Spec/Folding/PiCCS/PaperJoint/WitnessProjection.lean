import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.SourceMembership
import NightstreamFPrime.Spec.Phi81Relation.Types
import NightstreamFPrime.Spec.Folding.PiRLC.PaperForkExtractionWork
import Mathlib.Data.Vector.Basic
import Mathlib.Tactic.Ring

/-!
SuperNeo B.2's source-witness projection on the actual Phi81 public prefix.
Fresh witnesses contain only coordinates after `publicWidth`; running
witnesses retain every coordinate. The program materializes the returned
vectors and charges each field read and list construction.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.WitnessProjection

open NightstreamFPrime.Spec
open StrongReduction UnifiedSources
open _root_.NightstreamFPrime.Spec.Folding.PiRLC.PaperForkExtractionWork (Result)

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

private theorem collect_work {Value : Type*} (cost : Nat) : ∀ {count : Nat}
    (read : Fin count → Result Value),
    (∀ index, (read index).work = cost) →
    (collect read).work = count * (cost + 1) + 1
  | 0, _, _ => by simp only [collect, Nat.zero_mul, Nat.zero_add]
  | count + 1, read, exactCost => by
      change (read 0).work + (collect (fun index => read index.succ)).work + 1 = _
      rw [exactCost 0, collect_work cost (fun index => read index.succ)
        (fun index => exactCost index.succ), Nat.add_mul, Nat.one_mul]
      omega

private def copyFields {width : Nat} (read : Fin width → F) : Result (List.Vector F width) :=
  collect (fun index => ⟨read index, 1⟩)

private theorem copyFields_get {width : Nat} (read : Fin width → F) (index : Fin width) :
    (copyFields read).value.get index = read index :=
  collect_get _ index

private theorem copyFields_work {width : Nat} (read : Fin width → F) :
    (copyFields read).work = width * 2 + 1 :=
  collect_work 1 _ (fun _ => rfl)

def privateWidth (carrier : Phi81Relation.Shape) : Nat :=
  carrier.carrierWidth - carrier.publicWidth

def privateColumn (carrier : Phi81Relation.Shape) (index : Fin (privateWidth carrier)) :
    Fin carrier.carrierWidth :=
  ⟨carrier.publicWidth + index.val, by
    have fits : carrier.publicWidth ≤ carrier.carrierWidth := carrier.publicFits
    have bound := index.isLt
    unfold privateWidth at bound
    omega⟩

/-- Materialized fresh tails and full running witnesses in exact source order. -/
structure SourceWitness (shape : Shape) (carrier : Phi81Relation.Shape) where
  fresh : List.Vector (List.Vector F (privateWidth carrier)) shape.freshCount
  running : List.Vector (List.Vector F carrier.carrierWidth) shape.runningCount

/-- Copy the actual returned witness values. No source-validity proof or
honest-encoder witness is used to compute the result. -/
def project {shape : Shape} (carrier : Phi81Relation.Shape)
    (witness : OutputWitness shape carrier.carrierWidth) : Result (SourceWitness shape carrier) :=
  let fresh := collect (fun source : Fin shape.freshCount =>
    copyFields (fun column => witness.assignments (freshSourceIndex source) (privateColumn carrier column)))
  let running := collect (fun source : Fin shape.runningCount =>
    copyFields (witness.assignments (runningSourceIndex source)))
  ⟨⟨fresh.value, running.value⟩, fresh.work + running.work + 1⟩

/-- The exact number of field reads, vector constructors, and result return. -/
def projectionWork (shape : Shape) (carrier : Phi81Relation.Shape) : Nat :=
  shape.freshCount * (privateWidth carrier * 2 + 2) +
    shape.runningCount * (carrier.carrierWidth * 2 + 2) + 3

theorem project_work {shape : Shape} (carrier : Phi81Relation.Shape)
    (witness : OutputWitness shape carrier.carrierWidth) :
    (project carrier witness).work = projectionWork shape carrier := by
  dsimp only [project]
  unfold projectionWork
  rw [collect_work (privateWidth carrier * 2 + 1) _ (fun _ => copyFields_work _),
    collect_work (carrier.carrierWidth * 2 + 1) _ (fun _ => copyFields_work _)]
  ring

theorem project_fresh {shape : Shape} (carrier : Phi81Relation.Shape)
    (witness : OutputWitness shape carrier.carrierWidth)
    (source : Fin shape.freshCount) (column : Fin (privateWidth carrier)) :
    ((project carrier witness).value.fresh.get source).get column =
      witness.assignments (freshSourceIndex source) (privateColumn carrier column) := by
  simp only [project, collect_get, copyFields_get]

theorem project_running {shape : Shape} (carrier : Phi81Relation.Shape)
    (witness : OutputWitness shape carrier.carrierWidth)
    (source : Fin shape.runningCount) (column : Fin carrier.carrierWidth) :
    ((project carrier witness).value.running.get source).get column =
      witness.assignments (runningSourceIndex source) column := by
  simp only [project, collect_get, copyFields_get]

/-- Reattach only the verifier-owned public prefix to a fresh private tail. -/
def joinFresh {carrier : Phi81Relation.Shape} (publicInput : Phi81Relation.PublicInput carrier)
    (tail : List.Vector F (privateWidth carrier)) : Phi81Relation.Assignment carrier :=
  fun column => if inside : column.val < carrier.publicWidth then publicInput ⟨column.val, inside⟩ else
    tail.get ⟨column.val - carrier.publicWidth, by
      have fits : carrier.publicWidth ≤ carrier.carrierWidth := carrier.publicFits
      have bound := column.isLt
      unfold privateWidth
      omega⟩

theorem joinFresh_project {shape : Shape} (carrier : Phi81Relation.Shape)
    (witness : OutputWitness shape carrier.carrierWidth) (source : Fin shape.freshCount)
    (publicInput : Phi81Relation.PublicInput carrier)
    (prefixMatches : Phi81Relation.projectPublicInput (witness.assignments (freshSourceIndex source)) = publicInput) :
    joinFresh publicInput ((project carrier witness).value.fresh.get source) =
      witness.assignments (freshSourceIndex source) := by
  funext column
  unfold joinFresh
  split
  next inside =>
    have atColumn := congrFun prefixMatches ⟨column.val, inside⟩
    exact atColumn.symm
  next outside =>
    rw [project_fresh]
    apply congrArg (witness.assignments (freshSourceIndex source))
    apply Fin.ext
    simp only [privateColumn]
    omega

/-- Interpret the returned source witness through the existing complete
assignment representation. Fresh public values come only from the statement. -/
def reconstruct {shape : Shape} {carrier : Phi81Relation.Shape}
    (publicInputs : Fin shape.sourceCount → Phi81Relation.PublicInput carrier)
    (witness : SourceWitness shape carrier) : OutputWitness shape carrier.carrierWidth where
  assignments := Fin.addCases
    (fun fresh => joinFresh (publicInputs (freshSourceIndex fresh)) (witness.fresh.get fresh))
    (fun running => (witness.running.get running).get)

theorem reconstruct_fresh {shape : Shape} {carrier : Phi81Relation.Shape}
    (publicInputs : Fin shape.sourceCount → Phi81Relation.PublicInput carrier)
    (witness : SourceWitness shape carrier) (source : Fin shape.freshCount) :
    (reconstruct publicInputs witness).assignments (freshSourceIndex source) =
      joinFresh (publicInputs (freshSourceIndex source)) (witness.fresh.get source) := by
  dsimp only [reconstruct]
  exact Fin.addCases_left (m := shape.freshCount) (n := shape.runningCount)
    (motive := fun _ => Phi81Relation.Assignment carrier)
    (left := fun fresh : Fin shape.freshCount =>
      joinFresh (publicInputs (freshSourceIndex fresh)) (witness.fresh.get fresh))
    (right := fun running : Fin shape.runningCount => (witness.running.get running).get)
    source

theorem reconstruct_running {shape : Shape} {carrier : Phi81Relation.Shape}
    (publicInputs : Fin shape.sourceCount → Phi81Relation.PublicInput carrier)
    (witness : SourceWitness shape carrier) (source : Fin shape.runningCount) :
    (reconstruct publicInputs witness).assignments (runningSourceIndex source) =
      (witness.running.get source).get := by
  dsimp only [reconstruct]
  exact Fin.addCases_right (m := shape.freshCount) (n := shape.runningCount)
    (motive := fun _ => Phi81Relation.Assignment carrier)
    (left := fun fresh : Fin shape.freshCount =>
      joinFresh (publicInputs (freshSourceIndex fresh)) (witness.fresh.get fresh))
    (right := fun running : Fin shape.runningCount => (witness.running.get running).get)
    source

/-- Prefix binding makes the returned tails reconstruct the same full witness. -/
theorem reconstruct_project {shape : Shape} (carrier : Phi81Relation.Shape)
    (publicInputs : Fin shape.sourceCount → Phi81Relation.PublicInput carrier)
    (witness : OutputWitness shape carrier.carrierWidth)
    (prefixMatches : ∀ source : Fin shape.freshCount,
      Phi81Relation.projectPublicInput (witness.assignments (freshSourceIndex source)) =
        publicInputs (freshSourceIndex source)) :
    reconstruct publicInputs (project carrier witness).value = witness := by
  cases witness with
  | mk assignments =>
      let full : OutputWitness shape carrier.carrierWidth := ⟨assignments⟩
      apply congrArg OutputWitness.mk
      funext source column
      rcases source_eq_fresh_or_running source with ⟨fresh, rfl⟩ | ⟨running, rfl⟩
      · exact (congrFun (reconstruct_fresh publicInputs (project carrier full).value fresh) column).trans
          (congrFun (joinFresh_project carrier full fresh _ (prefixMatches fresh)) column)
      · exact (congrFun (reconstruct_running publicInputs (project carrier full).value running) column).trans
          (project_running carrier full running column)

end NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.WitnessProjection
