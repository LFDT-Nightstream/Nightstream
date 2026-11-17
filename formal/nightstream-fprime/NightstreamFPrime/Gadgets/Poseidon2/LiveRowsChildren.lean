import NightstreamFPrime.Gadgets.Poseidon2.LiveRows

/-! Children placed one after another (SPEC.md section 4.4). A composite gadget's operations are
its children, each a subcircuit at the end of the one before it. The composite is placed when each
child is placed (`Placed.child_cons`), and it has live rows when each child has
(`LiveRows.child_cons`). Each child inherits `OwnLive` from the composite because the children after
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

theorem LiveRows.nil : LiveRows live [] :=
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

theorem LiveRows.child_cons {circuit : FormalCircuit} {name : String} {offset : Nat}
    {rest : List Op} (child : LiveRows live (Circuit.ops circuit.main offset))
    (tail : LiveRows live rest) :
    LiveRows live (Op.subcircuit (circuit.asSubcircuit name offset) :: rest) :=
  (LiveRows.subcircuit child).append tail

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

/-! ### Blocks that end at the next offset

A parent can place a block of operations inline: another layout's operations, which end where
the next child starts. -/

theorem localLength_append_split (first rest : List Op) :
    localLength (first ++ rest) = localLength first + localLength rest := by
  unfold localLength
  rw [List.map_append, List.sum_append]

/-- A block that ends at `next`, then the operations after it. -/
theorem Placed.append_ends {offset next : Nat} {first rest : List Op}
    (ends : offset + localLength first = next)
    (firstPlaced : Placed offset (localLength first) first)
    (tail : Placed next (localLength rest) rest) :
    Placed offset (localLength (first ++ rest)) (first ++ rest) := by
  rw [localLength_append_split]
  subst ends
  exact firstPlaced.append tail

/-- A block's `OwnLive` and the `OwnLive` of the operations after it. -/
theorem OwnLive.append_ends {offset next : Nat} {first rest : List Op}
    (own : OwnLive live offset (localLength (first ++ rest)) (first ++ rest))
    (ends : offset + localLength first = next)
    (firstPlaced : Placed offset (localLength first) first)
    (tail : Placed next (localLength rest) rest) :
    OwnLive live offset (localLength first) first ∧
      OwnLive live next (localLength rest) rest := by
  rw [localLength_append_split] at own
  subst ends
  exact ⟨own.left tail, own.right firstPlaced⟩

/-- Children placed one after another end at `last` when the first child and the rest do. -/
theorem localLength_ends_cons {circuit : FormalCircuit} {name : String} {offset last : Nat}
    {rest : List Op}
    (ends : offset + localLength (Circuit.ops circuit.main offset) + localLength rest = last) :
    offset + localLength (Op.subcircuit (circuit.asSubcircuit name offset) :: rest) = last := by
  rw [localLength_child_cons, ← Nat.add_assoc]
  exact ends

theorem localLength_ends_nil (offset : Nat) : offset + localLength [] = offset :=
  rfl

/-- `OwnLive` of the columns before a bound at or after the operations' end. -/
theorem OwnLive.before {offset count bound : Nat} {ops : List Op}
    (own : OwnLive live offset count ops) (le : offset + count ≤ bound) :
    OwnLive (fun column => column < bound ∧ live column) offset count ops :=
  fun column lower upper outside =>
    ⟨Nat.lt_of_lt_of_le upper le, own column lower upper outside⟩

end NightstreamFPrime.Gadgets.Poseidon2
