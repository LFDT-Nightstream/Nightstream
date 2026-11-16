import NightstreamFPrime.Gadgets.Poseidon2.Clean

/-! Children placed one after another (SPEC.md section 4.4). A composite gadget's operations are
its children, each a subcircuit at the end of the one before it. The composite is placed when each
child is placed (`Placed.child_cons`), and it is clean when each child is clean
(`Clean.child_cons`). Each child inherits `OwnLive` from the composite because the children after
it are placed after it (`OwnLive.child_cons`). -/

set_option autoImplicit false

namespace NightstreamFPrime.Gadgets.Poseidon2

open NightstreamFPrime.Circuit

variable {live : Nat → Prop}

theorem localLength_child_cons (circuit : FormalCircuit) (name : String) (offset : Nat)
    (rest : List Op) :
    localLength (Op.subcircuit (circuit.asSubcircuit name offset) :: rest) =
      localLength (Circuit.ops circuit.main offset) + localLength rest := by
  rw [← localLength_subcircuit circuit name offset]
  rfl

theorem Placed.nil (offset : Nat) : Placed offset (localLength []) [] :=
  .of_marks_nil rfl

theorem Clean.nil : Clean live [] :=
  .of_free rfl fun _ member => by simp [flatConstraints] at member

/-- A child at `offset` with `count` locals, then the operations after it. -/
theorem Placed.child_cons {circuit : FormalCircuit} {name : String} {offset count : Nat}
    {rest : List Op} (length : localLength (Circuit.ops circuit.main offset) = count)
    (child : Placed offset (localLength (Circuit.ops circuit.main offset))
      (Circuit.ops circuit.main offset))
    (tail : Placed (offset + count) (localLength rest) rest) :
    Placed offset (localLength (Op.subcircuit (circuit.asSubcircuit name offset) :: rest))
      (Op.subcircuit (circuit.asSubcircuit name offset) :: rest) := by
  rw [localLength_child_cons]
  subst length
  exact (Placed.subcircuit child).append tail

theorem Clean.child_cons {circuit : FormalCircuit} {name : String} {offset : Nat}
    {rest : List Op} (child : Clean live (Circuit.ops circuit.main offset))
    (tail : Clean live rest) :
    Clean live (Op.subcircuit (circuit.asSubcircuit name offset) :: rest) :=
  (Clean.subcircuit child).append tail

/-- A child's `OwnLive` and the `OwnLive` of the operations after it. -/
theorem OwnLive.child_cons {circuit : FormalCircuit} {name : String} {offset count : Nat}
    {rest : List Op}
    (own : OwnLive live offset
      (localLength (Op.subcircuit (circuit.asSubcircuit name offset) :: rest))
      (Op.subcircuit (circuit.asSubcircuit name offset) :: rest))
    (length : localLength (Circuit.ops circuit.main offset) = count)
    (child : Placed offset (localLength (Circuit.ops circuit.main offset))
      (Circuit.ops circuit.main offset))
    (tail : Placed (offset + count) (localLength rest) rest) :
    OwnLive live offset (localLength (Circuit.ops circuit.main offset))
        (Circuit.ops circuit.main offset) ∧
      OwnLive live (offset + count) (localLength rest) rest := by
  rw [localLength_child_cons] at own
  subst length
  exact ⟨(OwnLive.left (first := [Op.subcircuit (circuit.asSubcircuit name offset)])
      (second := rest) own tail).subcircuit,
    OwnLive.right (first := [Op.subcircuit (circuit.asSubcircuit name offset)]) (second := rest)
      own (Placed.subcircuit child)⟩

end NightstreamFPrime.Gadgets.Poseidon2
