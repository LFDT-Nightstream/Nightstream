import NightstreamFPrime.Gadgets.Poseidon2.Formal
import NightstreamFPrime.Lifecycle.Relation

/-!
Owns the logical builder for HyperNova Construction-2's prior-state public
input. It calls the opaque Poseidon2 child and enforces the concrete
`encHash = [1, digest, 0…]` layout of the one public ring column.
-/

namespace NightstreamFPrime.Lifecycle.PriorStateHash

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.HyperNova.Construction2.Paper
open NightstreamFPrime.Lifecycle.PaperAlgebra

/-- The one public ring column has exactly 54 field cells. -/
def publicWidth : Nat := ringDegree * publicRingColumns

theorem publicWidth_eq : publicWidth = 54 := by
  rfl

def markerIndex : Fin publicWidth := ⟨0, by decide⟩

def digestIndex (lane : Fin 4) : Fin publicWidth :=
  ⟨lane.val + 1, by
    rw [publicWidth_eq]
    omega⟩

def tailIndex (lane : Fin 49) : Fin publicWidth :=
  ⟨lane.val + 5, by
    rw [publicWidth_eq]
    omega⟩

/-- Field-level public layout, independent of expression allocation. -/
def encodedHash (digest : List F) : Fin publicWidth → F :=
  fun column => (1 :: digest).getD column.val 0

@[simp] theorem encodedHash_marker (digest : List F) :
    encodedHash digest markerIndex = 1 := by
  rfl

@[simp] theorem encodedHash_digest (digest : Fin 4 → F) (lane : Fin 4) :
    encodedHash (List.ofFn digest) (digestIndex lane) = digest lane := by
  fin_cases lane <;>
    simp [encodedHash, digestIndex, List.ofFn_succ]

@[simp] theorem encodedHash_tail (digest : Fin 4 → F) (lane : Fin 49) :
    encodedHash (List.ofFn digest) (tailIndex lane) = 0 := by
  unfold encodedHash
  apply List.getD_eq_default
  simp [tailIndex]

theorem ofFn_getD {Alpha : Type} {count : Nat}
    (values : Fin count → Alpha) (lane : Fin count) (fallback : Alpha) :
    (List.ofFn values).getD lane.val fallback = values lane := by
  rw [List.getD_eq_get (List.ofFn values) fallback
    ⟨lane.val, by simp⟩]
  simp

/-- External expressions owned by the lifecycle parent. -/
structure Interface where
  preimage : Nat → List Expr
  publicInput : Nat → Fin publicWidth → Expr

def hashInterface (interface : Interface) : Formal.Interface where
  input := interface.preimage
  expected := fun offset lane => interface.publicInput offset (digestIndex lane)

@[simp] theorem hashInterface_input (interface : Interface) (offset : Nat) :
    (hashInterface interface).input offset = interface.preimage offset := by
  rfl

def Assumptions (interface : Interface) (offset : Nat) (_env : Env) : Prop :=
  (∀ expression ∈ interface.preimage offset, expression.VarsBelow offset) ∧
    ∀ column, (interface.publicInput offset column).VarsBelow offset

def SpecHolds (interface : Interface) (offset : Nat) (env : Env) : Prop :=
  (fun column => (interface.publicInput offset column).eval env) =
    encodedHash (Poseidon2.hash
      (Hash.evalList env (interface.preimage offset)))

def childCircuit (interface : Interface) : FormalCircuit :=
  Formal.circuit (hashInterface interface)

def childAt (interface : Interface) (offset : Nat) : Subcircuit :=
  (childCircuit interface).asSubcircuit "poseidon2.hash" offset

def bindingAssertions (interface : Interface) (offset : Nat) : List Op :=
  Op.assertZero (interface.publicInput offset markerIndex - 1) ::
    List.ofFn (fun lane : Fin 49 =>
      Op.assertZero (interface.publicInput offset (tailIndex lane)))

def opsAt (interface : Interface) (offset : Nat) : List Op :=
  Op.subcircuit (childAt interface offset) :: bindingAssertions interface offset

def main (interface : Interface) : Circuit Unit := fun offset =>
  let child := childAt interface offset
  ((), offset + child.localLength, opsAt interface offset)

@[simp] theorem main_ops (interface : Interface) (offset : Nat) :
    Circuit.ops (main interface) offset = opsAt interface offset := by
  rfl

theorem bindingAssertions_localLength (interface : Interface) (offset : Nat) :
    localLength (bindingAssertions interface offset) = 0 := by
  change (0 :: List.ofFn (fun _ : Fin 49 => 0)).sum = 0
  simp

theorem opsAt_localLength (interface : Interface) (offset : Nat) :
    localLength (opsAt interface offset) = (childAt interface offset).localLength := by
  unfold opsAt
  change (childAt interface offset).localLength +
    localLength (bindingAssertions interface offset) = _
  rw [bindingAssertions_localLength]
  omega

theorem main_localLength (interface : Interface) (offset : Nat) :
    localLength (Circuit.ops (main interface) offset) =
      (Hash.compile offset (interface.preimage offset)).recipes.length := by
  rw [main_ops, opsAt_localLength]
  rfl

theorem markerAssertion_mem (interface : Interface) (offset : Nat) :
    Op.assertZero (interface.publicInput offset markerIndex - 1) ∈
      bindingAssertions interface offset := by
  simp [bindingAssertions]

theorem tailAssertion_mem (interface : Interface) (offset : Nat)
    (lane : Fin 49) :
    Op.assertZero (interface.publicInput offset (tailIndex lane)) ∈
      bindingAssertions interface offset := by
  simp only [bindingAssertions, List.mem_cons]
  apply Or.inr
  rw [List.mem_ofFn']
  exact Set.mem_range_self lane

theorem childAssumptions (interface : Interface) (offset : Nat) (env : Env)
    (assumptions : Assumptions interface offset env) :
    Formal.Assumptions (hashInterface interface) offset env := by
  exact ⟨assumptions.1, fun lane => assumptions.2 (digestIndex lane)⟩

theorem childSpec_of_spec (interface : Interface) (offset : Nat) (env : Env)
    (specification : SpecHolds interface offset env) :
    Formal.SpecHolds (hashInterface interface) offset env := by
  unfold Formal.SpecHolds hashInterface
  have canonical := specification
  unfold SpecHolds at canonical
  rw [← Hash.hashF_eq_reference] at canonical
  have cells : (fun lane =>
      (interface.publicInput offset (digestIndex lane)).eval env) =
      Hash.hashF
        (Hash.evalList env (interface.preimage offset)) := by
    funext lane
    have selected := congrFun canonical (digestIndex lane)
    calc
      (interface.publicInput offset (digestIndex lane)).eval env =
          encodedHash (List.ofFn (Hash.hashF
            (Hash.evalList env (interface.preimage offset))))
            (digestIndex lane) := selected
      _ = Hash.hashF
          (Hash.evalList env (interface.preimage offset)) lane :=
        encodedHash_digest _ lane
  calc
    List.ofFn (fun lane =>
        (interface.publicInput offset (digestIndex lane)).eval env) =
        List.ofFn (Hash.hashF
          (Hash.evalList env (interface.preimage offset))) :=
      congrArg List.ofFn cells
    _ = Poseidon2.hash
          (Hash.evalList env (interface.preimage offset)) :=
      Hash.hashF_eq_reference _

theorem spec_of_parts (interface : Interface) (offset : Nat) (env : Env)
    (hashSpec : Formal.SpecHolds (hashInterface interface) offset env)
    (marker : (interface.publicInput offset markerIndex).eval env = 1)
    (tail : ∀ lane,
      (interface.publicInput offset (tailIndex lane)).eval env = 0) :
    SpecHolds interface offset env := by
  have hashSpec' := hashSpec
  unfold Formal.SpecHolds hashInterface at hashSpec'
  rw [← Hash.hashF_eq_reference] at hashSpec'
  have digestCell (lane : Fin 4) :
      (interface.publicInput offset (digestIndex lane)).eval env =
        Hash.hashF
          (Hash.evalList env (interface.preimage offset)) lane := by
    have selected := congrArg (fun values : List F => values.getD lane.val 0)
      hashSpec'
    calc
      (interface.publicInput offset (digestIndex lane)).eval env =
          (List.ofFn (fun current =>
            (interface.publicInput offset (digestIndex current)).eval env)).getD
            lane.val 0 := (ofFn_getD (fun current : Fin 4 =>
              (interface.publicInput offset (digestIndex current)).eval env)
              lane 0).symm
      _ = (List.ofFn (Hash.hashF
            (Hash.evalList env (interface.preimage offset)))).getD lane.val 0 :=
        selected
      _ = Hash.hashF
          (Hash.evalList env (interface.preimage offset)) lane :=
        ofFn_getD (Hash.hashF
          (Hash.evalList env (interface.preimage offset))) lane 0
  unfold SpecHolds
  rw [← Hash.hashF_eq_reference]
  funext column
  by_cases isMarker : column.val = 0
  · have columnEq : column = markerIndex := Fin.ext isMarker
    subst column
    simpa using marker
  · by_cases isDigest : column.val < 5
    · let lane : Fin 4 := ⟨column.val - 1, by omega⟩
      have columnEq : digestIndex lane = column := by
        apply Fin.ext
        simp [digestIndex, lane]
        omega
      rw [← columnEq, digestCell lane, encodedHash_digest]
    · let lane : Fin 49 := ⟨column.val - 5, by
        have columnBound := column.isLt
        change column.val < 54 at columnBound
        omega⟩
      have columnEq : tailIndex lane = column := by
        apply Fin.ext
        simp [tailIndex, lane]
        omega
      rw [← columnEq, tail lane, encodedHash_tail]

theorem soundness (interface : Interface) (env : Env) (offset : Nat)
    (assumptions : Assumptions interface offset env)
    (hholds : holds env (Circuit.ops (main interface) offset)) :
    SpecHolds interface offset env := by
  have childLogical := hholds (Op.subcircuit (childAt interface offset)) (by
    simp [main_ops, opsAt])
  have hashSpec : Formal.SpecHolds (hashInterface interface) offset env :=
    childLogical (childAssumptions interface offset env assumptions)
  have markerHolds := hholds
    (Op.assertZero (interface.publicInput offset markerIndex - 1)) (by
      simp only [main_ops, opsAt, List.mem_cons]
      exact Or.inr (markerAssertion_mem interface offset))
  have marker : (interface.publicInput offset markerIndex).eval env = 1 := by
    change (interface.publicInput offset markerIndex - 1).eval env = 0 at markerHolds
    exact sub_eq_zero.mp (by simpa only [Expr.eval_sub] using markerHolds)
  have tail (lane : Fin 49) :
      (interface.publicInput offset (tailIndex lane)).eval env = 0 := by
    have tailHolds := hholds
      (Op.assertZero (interface.publicInput offset (tailIndex lane))) (by
        simp only [main_ops, opsAt, List.mem_cons]
        exact Or.inr (tailAssertion_mem interface offset lane))
    exact tailHolds
  exact spec_of_parts interface offset env hashSpec marker tail

theorem completeness (interface : Interface) (env : Env) (offset : Nat)
    (assumptions : Assumptions interface offset env)
    (specification : SpecHolds interface offset env) :
    ∃ completed,
      AgreesOutside env completed offset
        (localLength (Circuit.ops (main interface) offset)) ∧
      holdsFlat completed (Circuit.ops (main interface) offset) := by
  have hashAssumptions := childAssumptions interface offset env assumptions
  have hashSpec := childSpec_of_spec interface offset env specification
  rcases Formal.completeness (hashInterface interface) env offset
      hashAssumptions hashSpec with
    ⟨completed, outside, childRows⟩
  have agreesBelow : ∀ index, index < offset → completed index = env index := by
    intro index indexLt
    exact outside index (Or.inl indexLt)
  have markerAtEnv := congrFun specification markerIndex
  have markerAtCompleted :
      (interface.publicInput offset markerIndex).eval completed = 1 := by
    calc
      (interface.publicInput offset markerIndex).eval completed =
          (interface.publicInput offset markerIndex).eval env :=
        (interface.publicInput offset markerIndex).eval_eq_of_agree_below
          offset completed env (assumptions.2 markerIndex) agreesBelow
      _ = 1 := by simpa using markerAtEnv
  have tailAtCompleted (lane : Fin 49) :
      (interface.publicInput offset (tailIndex lane)).eval completed = 0 := by
    calc
      (interface.publicInput offset (tailIndex lane)).eval completed =
          (interface.publicInput offset (tailIndex lane)).eval env :=
        (interface.publicInput offset (tailIndex lane)).eval_eq_of_agree_below
          offset completed env (assumptions.2 (tailIndex lane)) agreesBelow
      _ = 0 := by
        have atEnv := congrFun specification (tailIndex lane)
        rw [← Hash.hashF_eq_reference] at atEnv
        simpa using atEnv
  refine ⟨completed, ?_, ?_⟩
  · simpa [main_ops, opsAt_localLength] using outside
  · change ConstraintsHold completed
      (flatConstraints (opsAt interface offset))
    intro expression member
    simp only [flatConstraints, List.mem_flatMap] at member
    rcases member with ⟨operation, operationMember, constraintMember⟩
    simp only [opsAt, List.mem_cons] at operationMember
    rcases operationMember with rfl | operationMember
    · exact childRows expression constraintMember
    · simp only [bindingAssertions, List.mem_cons] at operationMember
      rcases operationMember with rfl | operationMember
      · simp only [Op.flatConstraints, List.mem_singleton] at constraintMember
        subst expression
        change (interface.publicInput offset markerIndex - 1).eval completed = 0
        simp only [Expr.eval_sub]
        exact sub_eq_zero.mpr markerAtCompleted
      · rw [List.mem_ofFn'] at operationMember
        rcases operationMember with ⟨lane, rfl⟩
        simp only [Op.flatConstraints, List.mem_singleton] at constraintMember
        subst expression
        exact tailAtCompleted lane

/-- The production logical builder for the `priorStateHash` phase. -/
def circuit (interface : Interface) : FormalCircuit where
  main := main interface
  assumptions := Assumptions interface
  spec := SpecHolds interface
  soundness := soundness interface
  completeness := completeness interface

theorem circuit_localLength (interface : Interface) (offset : Nat) :
    localLength (Circuit.ops (circuit interface).main offset) =
      (Hash.compile offset (interface.preimage offset)).recipes.length :=
  main_localLength interface offset

section Relation

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns <=
    Phi81CarrierLayout.carrierWidth logicalWidth}

def RepresentsPreimage (interface : Interface) (offset : Nat) (env : Env)
    (preimage : HashPreimage (logicalWidth := logicalWidth)
      (publicFits := publicFits)) : Prop :=
  Hash.evalList env (interface.preimage offset) =
    serializePreimage (publicFits := publicFits) preimage

def RepresentsPublicInput (interface : Interface) (offset : Nat) (env : Env)
    (publicInput : PublicInput (logicalWidth := logicalWidth)
      (publicFits := publicFits)) : Prop :=
  ∀ column, (interface.publicInput offset column).eval env = publicInput column

/-- The logical builder's specification implies the exact recursive relation
slot, with the same authoritative preimage and public input. -/
theorem builder_implies_priorPublicInput
    (interface : Interface) (offset : Nat) (env : Env)
    (preimage : HashPreimage (logicalWidth := logicalWidth)
      (publicFits := publicFits))
    (publicInput : PublicInput (logicalWidth := logicalWidth)
      (publicFits := publicFits))
    (specification : SpecHolds interface offset env)
    (preimageRepresents : RepresentsPreimage interface offset env preimage)
    (publicRepresents : RepresentsPublicInput interface offset env publicInput) :
    publicInput = encHash (publicFits := publicFits)
      (stateHash (publicFits := publicFits) preimage) := by
  funext column
  calc
    publicInput column = (interface.publicInput offset column).eval env :=
      (publicRepresents column).symm
    _ = encodedHash (Poseidon2.hash
        (Hash.evalList env (interface.preimage offset))) column :=
      congrFun specification column
    _ = encodedHash (Poseidon2.hash
        (serializePreimage (publicFits := publicFits) preimage)) column := by
      rw [preimageRepresents]
    _ = encHash (publicFits := publicFits)
        (stateHash (publicFits := publicFits) preimage) column := by
      rfl

/-- Production specialization: the builder proves the exact
`RecursiveHolds.priorPublicInput` equation of `StepHolds`. -/
theorem builder_implies_recursive_slot
    (interface : Interface) (offset : Nat) (env : Env)
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (vk : KeyDigest) (F : AppState → AppWitness → AppState)
    (input : Input KeyDigest AppState AppWitness
      (Running (logicalWidth := logicalWidth) (publicFits := publicFits))
      (Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
      (Proof (ProductionKey.degreeBound relation)) slotCount)
    (specification : SpecHolds interface offset env)
    (preimageRepresents : RepresentsPreimage interface offset env
      (priorHashPreimage (setup relation ajtai vk) input))
    (publicRepresents : RepresentsPublicInput interface offset env
      ((machine publicFits F).freshPublic input.fresh)) :
    (machine publicFits F).freshPublic input.fresh =
      (machine publicFits F).encodeInstance
        ((machine publicFits F).hash
          (priorHashPreimage (setup relation ajtai vk) input)) := by
  exact builder_implies_priorPublicInput interface offset env
    (priorHashPreimage (setup relation ajtai vk) input)
    ((machine publicFits F).freshPublic input.fresh)
    specification preimageRepresents publicRepresents

end Relation

end NightstreamFPrime.Lifecycle.PriorStateHash
