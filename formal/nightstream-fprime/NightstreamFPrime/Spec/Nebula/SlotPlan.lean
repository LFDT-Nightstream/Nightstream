import NightstreamFPrime.Spec.Nebula.EventBinding

/-! Owns the meaning of the slot plan rules of SPEC.md section 7.6 in one row
witness. A slot plan lists, for each operation slot, the ranks of the ports
that share it. The rank of a port is its position in catalog order: memories,
then each memory's ports. When the plan names each port once (rule 5), at most
one port of each slot is active (rule 1), and of two dependent active ports in
distinct slots the port of the earlier slot has no higher rank (rule 2), the
events of the active ports in slot order are a `Reorders` of the events of the
row. The rules hold for any catalog, so also for a batched catalog (`Batch`).
It does not own the Rust rows that enforce the rules, or the address and
value rows (rules 3 and 4). -/

namespace NightstreamFPrime.Spec.Nebula.MemoryTrace

/-- The ports of a catalog in rank order, each with its memory id. -/
def rankedPorts (catalog : List MemorySpec) : List (ℕ × PortSpec) :=
  catalog.flatMap fun m => m.ports.map fun port => (m.memoryId, port)

/-- The event of the port of rank `r` in a row: `none` when the port is
inactive or when no port has rank `r`. -/
def rankEvent (catalog : List MemorySpec) (w : Witness) (r : ℕ) : Option Event :=
  (rankedPorts catalog)[r]?.bind fun port => portEvent port.1 w port.2

/-- The events of the active ports of a slot plan, in slot order. -/
def slotEvents (catalog : List MemorySpec) (w : Witness) (plan : List (List ℕ)) : List Event :=
  plan.flatMap fun slot => slot.filterMap (rankEvent catalog w)

/-- A slot plan that obeys rules 5, 1 and 2 of SPEC.md section 7.6 in a row. -/
structure SlotOrder (catalog : List MemorySpec) (w : Witness) (plan : List (List ℕ)) : Prop where
  /-- Rule 5: the plan names each port once. -/
  names : plan.flatten.Perm (List.range (rankedPorts catalog).length)
  /-- Rule 1: at most one port of each slot is active. -/
  single : ∀ slot ∈ plan, (slot.filter fun r => (rankEvent catalog w r).isSome).length ≤ 1
  /-- Rule 2: of two dependent active ports in distinct slots, the port of the
  earlier slot has no higher rank. -/
  order : plan.Pairwise fun earlier later => ∀ r ∈ earlier, ∀ r' ∈ later,
    ∀ e ∈ rankEvent catalog w r, ∀ e' ∈ rankEvent catalog w r', e.Dependent e' → r ≤ r'

/-- The events of a row: the events of each memory's active ports, memory by
memory. -/
theorem eventsForRow_eq (catalog : List MemorySpec) :
    eventsForRow catalog = fun w =>
      catalog.flatMap fun m => m.ports.filterMap (portEvent m.memoryId w) := by
  funext w
  unfold eventsForRow
  refine List.flatMap_congr fun m _ => ?_
  rw [List.filterMap_eq_flatMap_toList]
  refine List.flatMap_congr fun port _ => ?_
  cases portEvent m.memoryId w port <;> rfl

private theorem filterMap_range {α β : Type} (l : List α) (f : α → Option β) :
    l.filterMap f = (List.range l.length).filterMap fun r => l[r]?.bind f := by
  induction l with
  | nil => rfl
  | cons a l ih =>
    rw [List.length_cons, List.range_succ_eq_map, List.filterMap_cons, List.filterMap_cons,
      List.filterMap_map]
    simp only [Function.comp_def, Nat.succ_eq_add_one, List.getElem?_cons_succ,
      List.getElem?_cons_zero, ← ih]
    rfl

private theorem pairwise_of_length_filter {α : Type} {p : α → Bool} {l : List α}
    (single : (l.filter p).length ≤ 1) : l.Pairwise fun a b => ¬ (p a = true ∧ p b = true) :=
  List.pairwise_iff_forall_sublist.mpr fun {a b} sub both => by
    have := (sub.filter p).length_le
    simp [both.1, both.2] at this
    omega

/-- A slot plan that obeys rules 5, 1 and 2 lists the events of a row in an
order that keeps each dependent pair. -/
theorem reorders_of_slotOrder {catalog : List MemorySpec} {w : Witness} {plan : List (List ℕ)}
    (slots : SlotOrder catalog w plan) :
    Reorders (eventsForRow catalog w) (slotEvents catalog w plan) := by
  have events : eventsForRow catalog w =
      (List.range (rankedPorts catalog).length).filterMap (rankEvent catalog w) := by
    have ranked : eventsForRow catalog w =
        (rankedPorts catalog).filterMap fun port => portEvent port.1 w port.2 := by
      simp only [eventsForRow_eq, rankedPorts, List.filterMap_flatMap, List.filterMap_map,
        Function.comp_def]
    rw [ranked, filterMap_range]
    rfl
  have order : slotEvents catalog w plan = plan.flatten.filterMap (rankEvent catalog w) := by
    rw [slotEvents, List.filterMap_flatten, List.flatMap_def]
  rw [events, order]
  refine reorders_filterMap _ List.pairwise_lt_range slots.names ?_
  have nodup : plan.flatten.Nodup := slots.names.nodup_iff.mpr List.nodup_range
  have lower : plan.flatten.Pairwise fun r r' => ∀ e ∈ rankEvent catalog w r,
      ∀ e' ∈ rankEvent catalog w r', e.Dependent e' → r ≤ r' := by
    refine List.pairwise_flatten.mpr ⟨fun slot member => ?_, slots.order⟩
    refine (pairwise_of_length_filter (slots.single slot member)).imp
      fun {r r'} apart e shown e' shown' _ => absurd ⟨?_, ?_⟩ apart
    · rw [Option.mem_def.mp shown]; rfl
    · rw [Option.mem_def.mp shown']; rfl
  exact (nodup.and lower).imp fun ⟨apart, below⟩ e shown e' shown' dependent =>
    Nat.lt_of_le_of_ne (below e shown e' shown' dependent) apart

end NightstreamFPrime.Spec.Nebula.MemoryTrace
