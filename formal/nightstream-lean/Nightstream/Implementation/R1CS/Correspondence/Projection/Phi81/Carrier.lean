import Nightstream.Implementation.R1CS.Core.ProjectionProgram
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier

/-!
Profile-neutral coefficient-list carrier for Phi81 projection identities.

Assurance tier: model-level. This module owns only explicit field/list
computations; it emits no rows and imports no generated or ownership profile.

| Stage family | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `projection.inputs` | columns decode in order to Goldilocks residues | computed | `values`, `values_getD_of_length` |
| `projection.product_sum` | list folding is the typed Phi81 product sum | derived | `ringOfList_phi81Combine` |

Owns: residue decoding, the 54-coefficient carrier, and canonical finite sums.
Does not own: column authority, transcript challenges, trace exactness, costs,
or Rust/R1CS conformance.
-/

namespace Nightstream.Implementation.R1CS.ProjectionPhi81

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

abbrev Scalar := Concrete.F
abbrev Ring := List Scalar

def residue (value : Nat) : Scalar :=
  ⟨value % goldilocksP, Nat.mod_lt _ (by decide)⟩

def semanticAssignment (assignment : Nat -> Nat) : Nat -> Scalar :=
  fun column => residue (assignment column)

@[simp] theorem semanticAssignment_apply
    (assignment : Nat -> Nat) (column : Nat) :
    semanticAssignment assignment column = residue (assignment column) := rfl

/-- The semantic assignment is exactly the base-field assignment used by the
generic projection interpreter. -/
theorem semanticAssignment_eq_baseAt (assignment : Nat -> Nat) :
    semanticAssignment assignment =
      ProjectionProgram.baseAt assignment := by
  funext column
  rfl

def values (assignment : Nat -> Nat) (columns : List Nat) : Ring :=
  columns.map fun column => residue (assignment column)

theorem values_eq_map_semanticAssignment
    (assignment : Nat -> Nat) (columns : List Nat) :
    values assignment columns = columns.map (semanticAssignment assignment) :=
  rfl

/-- A typed lane read from an exact-width column list decodes the same column
as the corresponding list read. -/
theorem values_getD_of_length
    (assignment : Nat -> Nat) (columns : List Nat)
    (width : columns.length = Concrete.ringDegree)
    (lane : Fin Concrete.ringDegree) :
    (values assignment columns).getD lane.val 0 =
      residue (assignment (columns.getD lane.val 0)) := by
  have laneLt : lane.val < columns.length := by
    rw [width]
    exact lane.isLt
  unfold values
  rw [List.getD_eq_getElem?_getD, List.getElem?_map,
    List.getElem?_eq_getElem laneLt]
  simp only [Option.map_some, Option.getD_some]
  apply congrArg (fun column => residue (assignment column))
  rw [List.getElem_eq_getD]

/-- Interpret a coefficient list as a Phi81 base-ring element. Exact-width
premises at callers exclude the default-value branch. -/
def ringOfList (coefficients : Ring) : Concrete.RingF :=
  fun coefficient => coefficients.getD coefficient.val 0

/-- Canonical finite sum in coefficient-list form. -/
def phi81Combine {count : Nat} (challenges inputs : Fin count -> Ring) : Ring :=
  List.ofFn fun coefficient : Fin Concrete.ringDegree =>
    (List.ofFn fun index : Fin count =>
      Concrete.ringFMul (ringOfList (challenges index))
        (ringOfList (inputs index)) coefficient).foldl
      (fun sum item => sum + item) 0

/-- Canonical head-first scalar sum. -/
def scalarSum : List Scalar -> Scalar
  | [] => 0
  | value :: rest => value + scalarSum rest

private theorem foldl_eq_add_scalarSum
    (items : List Scalar) (initial : Scalar) :
    items.foldl (fun sum item => sum + item) initial =
      initial + scalarSum items := by
  induction items generalizing initial with
  | nil => exact (ConcreteCarrier.baseLaws.add_zero initial).symm
  | cons value rest inductionHypothesis =>
      rw [List.foldl_cons, inductionHypothesis]
      exact ConcreteCarrier.baseLaws.add_assoc _ _ _

private theorem foldl_zero_eq_scalarSum (items : List Scalar) :
    items.foldl (fun sum item => sum + item) 0 = scalarSum items := by
  rw [foldl_eq_add_scalarSum]
  exact ConcreteCarrier.baseLaws.zero_add _

theorem phi81Combine_eq_scalarSum
    {count : Nat} (challenges inputs : Fin count -> Ring) :
    phi81Combine challenges inputs =
      List.ofFn fun coefficient : Fin Concrete.ringDegree =>
        scalarSum (List.ofFn fun index : Fin count =>
          Concrete.ringFMul (ringOfList (challenges index))
            (ringOfList (inputs index)) coefficient) := by
  unfold phi81Combine
  apply congrArg List.ofFn
  funext coefficient
  exact foldl_zero_eq_scalarSum _

theorem phi81Combine_coefficient
    {count : Nat} (challenges inputs : Fin count -> Ring)
    (coefficient : Fin Concrete.ringDegree) :
    ringOfList (phi81Combine challenges inputs) coefficient =
      scalarSum (List.ofFn fun index : Fin count =>
        Concrete.ringFMul (ringOfList (challenges index))
          (ringOfList (inputs index)) coefficient) := by
  unfold ringOfList phi81Combine
  rw [List.getD_eq_getElem?_getD,
    List.getElem?_eq_getElem (by simp), List.getElem_ofFn]
  simp only [Option.getD_some]
  exact foldl_zero_eq_scalarSum _

/-- Canonical typed head-first challenge-times-input sum. -/
def productSum : {count : Nat} ->
    (Fin count -> Concrete.RingF) ->
    (Fin count -> Concrete.RingF) -> Concrete.RingF
  | 0, _, _ => Concrete.ringFZero
  | _ + 1, challenges, inputs =>
      Concrete.ringFAdd
        (Concrete.ringFMul (challenges 0) (inputs 0))
        (productSum
          (fun index => challenges index.succ)
          (fun index => inputs index.succ))

private theorem productSum_coefficient
    {count : Nat} (challenges inputs : Fin count -> Concrete.RingF)
    (coefficient : Fin Concrete.ringDegree) :
    productSum challenges inputs coefficient =
      scalarSum (List.ofFn fun index : Fin count =>
        Concrete.ringFMul (challenges index) (inputs index) coefficient) := by
  induction count with
  | zero => rfl
  | succ count inductionHypothesis =>
      simp only [productSum, List.ofFn_succ, scalarSum, Concrete.ringFAdd]
      rw [inductionHypothesis
        (fun index => challenges index.succ)
        (fun index => inputs index.succ)]

/-- Decoding the complete list operation gives the typed Phi81 product sum. -/
theorem ringOfList_phi81Combine
    {count : Nat} (challenges inputs : Fin count -> Ring) :
    ringOfList (phi81Combine challenges inputs) =
      productSum
        (fun index => ringOfList (challenges index))
        (fun index => ringOfList (inputs index)) := by
  funext coefficient
  rw [phi81Combine_coefficient]
  exact (productSum_coefficient
    (fun index => ringOfList (challenges index))
    (fun index => ringOfList (inputs index)) coefficient).symm

end Nightstream.Implementation.R1CS.ProjectionPhi81
