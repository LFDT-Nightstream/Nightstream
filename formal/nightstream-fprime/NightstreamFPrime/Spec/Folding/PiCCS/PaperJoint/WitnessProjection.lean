import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.SourceMembership
import NightstreamFPrime.Spec.Phi81Relation.Types
import NightstreamFPrime.Spec.Folding.PiRLC.PaperForkExtractionWork
import Mathlib.Data.Vector.Basic
import Mathlib.Tactic.Ring

/-!
SuperNeo B.2's source-witness projection on the actual Phi81 public prefix.
Fresh witnesses contain only coordinates after `publicWidth`; running
witnesses retain every coordinate. `reconstruct` restores the full
assignments from the verifier-owned public prefix.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.WitnessProjection

open NightstreamFPrime.Spec
open StrongReduction UnifiedSources
open _root_.NightstreamFPrime.Spec.Folding.PiRLC.PaperForkExtractionWork (Result)

def privateWidth (carrier : Phi81Relation.Shape) : Nat :=
  carrier.carrierWidth - carrier.publicWidth

/-- Materialized fresh tails and full running witnesses in exact source order. -/
structure SourceWitness (shape : Shape) (carrier : Phi81Relation.Shape) where
  fresh : List.Vector (List.Vector F (privateWidth carrier)) shape.freshCount
  running : List.Vector (List.Vector F carrier.carrierWidth) shape.runningCount

/-- The exact number of field reads, vector constructors, and result return. -/
def projectionWork (shape : Shape) (carrier : Phi81Relation.Shape) : Nat :=
  shape.freshCount * (privateWidth carrier * 2 + 2) +
    shape.runningCount * (carrier.carrierWidth * 2 + 2) + 3

/-- Reattach only the verifier-owned public prefix to a fresh private tail. -/
def joinFresh {carrier : Phi81Relation.Shape} (publicInput : Phi81Relation.PublicInput carrier)
    (tail : List.Vector F (privateWidth carrier)) : Phi81Relation.Assignment carrier :=
  fun column => if inside : column.val < carrier.publicWidth then publicInput ⟨column.val, inside⟩ else
    tail.get ⟨column.val - carrier.publicWidth, by
      have fits : carrier.publicWidth ≤ carrier.carrierWidth := carrier.publicFits
      have bound := column.isLt
      unfold privateWidth
      omega⟩

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

end NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.WitnessProjection
