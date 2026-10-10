import Mathlib.Algebra.Order.Group.Multiset
import Mathlib.Data.List.Induction
import Mathlib.Data.Multiset.Range
import NightstreamFPrime.Spec.Nebula.Reference

/-! Owns security note Lemma 5 and its converse. Lemma 5: if the write stamps
are consecutive, every operation has `rt < wt` and is valid, and
`IS ∪ WS = RS ∪ FS`, then the operations replay on the reference machine from
the IS memory and end at the FS memory. The converse: an honest replay, with
`rt` read from the current stamp, is balanced. No IS or FS timestamp range
check is needed. -/

namespace NightstreamFPrime.Spec.Nebula

private theorem snapshot_eq (p : Plan) (mem : Memory) :
    snapshot p mem = (Multiset.range p.cells).map fun g => ((mem g).stamp, g, (mem g).value) :=
  rfl

private theorem cell_ext {a b : Cell} (value : a.value = b.value) (stamp : a.stamp = b.stamp) :
    a = b := by
  cases a
  cases b
  simp_all

/-- Updating one cell below `R + M` swaps one tuple of the snapshot. -/
theorem snapshot_update (p : Plan) (mem : Memory) {g : ℕ} (hg : g < p.cells) (c : Cell) :
    snapshot p (Function.update mem g c) + {((mem g).stamp, g, (mem g).value)} =
      snapshot p mem + {(c.stamp, g, c.value)} := by
  have split : Multiset.range p.cells = g ::ₘ (Multiset.range p.cells).erase g :=
    (Multiset.cons_erase (Multiset.mem_range.mpr hg)).symm
  have notMem : g ∉ (Multiset.range p.cells).erase g := (Multiset.nodup_range _).notMem_erase
  have rest : ((Multiset.range p.cells).erase g).map
      (fun x => ((Function.update mem g c x).stamp, x, (Function.update mem g c x).value)) =
      ((Multiset.range p.cells).erase g).map fun x => ((mem x).stamp, x, (mem x).value) :=
    Multiset.map_congr rfl fun x hx => by
      rw [Function.update_of_ne (ne_of_mem_of_not_mem hx notMem)]
  rw [snapshot_eq, snapshot_eq, split, Multiset.map_cons, Multiset.map_cons, rest]
  simp only [Function.update_self, ← Multiset.singleton_add]
  simp only [add_comm, add_left_comm]

/-- Equal snapshots agree on every cell below `R + M`. -/
theorem eqOn_of_snapshot_eq {p : Plan} {mem₁ mem₂ : Memory}
    (same : snapshot p mem₁ = snapshot p mem₂) :
    Set.EqOn mem₁ mem₂ (Set.Iio p.cells) := by
  intro g hg
  have hmem : ((mem₁ g).stamp, g, (mem₁ g).value) ∈ snapshot p mem₂ := by
    rw [← same, snapshot_eq]
    exact Multiset.mem_map_of_mem _ (Multiset.mem_range.mpr (Set.mem_Iio.mp hg))
  rw [snapshot_eq, Multiset.mem_map] at hmem
  obtain ⟨x, -, hx⟩ := hmem
  simp only [Prod.mk.injEq] at hx
  obtain ⟨stamp, rfl, value⟩ := hx
  exact cell_ext value.symm stamp.symm

private theorem coe_map_append (f : MemOp → Tuple) (ops : List MemOp) (o : MemOp) :
    (((ops ++ [o]).map f : List Tuple) : Multiset Tuple) =
      ((ops.map f : List Tuple) : Multiset Tuple) + {f o} := by
  rw [List.map_append, ← Multiset.coe_add]
  rfl

/-- Security note Lemma 5. -/
theorem segment_sequential (p : Plan) {start : Machine} {final : Memory} {ops : List MemOp}
    (stamps : ∀ k (hk : k < ops.length), (ops.get ⟨k, hk⟩).wt = start.ts + k + 1)
    (fresh : ∀ o ∈ ops, o.rt < o.wt) (valid : ∀ o ∈ ops, o.access.Valid p)
    (balanced : snapshot p start.memory + ((ops.map (MemOp.write p) : List Tuple) : Multiset Tuple) =
      ((ops.map (MemOp.read p) : List Tuple) : Multiset Tuple) + snapshot p final) :
    ∃ m, accessAll p start (ops.map MemOp.access) = some ⟨m, start.ts + ops.length⟩ ∧
      Set.EqOn m final (Set.Iio p.cells) := by
  induction ops using List.reverseRecOn generalizing final with
  | nil =>
    refine ⟨start.memory, rfl, eqOn_of_snapshot_eq ?_⟩
    simpa using balanced
  | append_singleton ops o ih =>
    have lastStamp : o.wt = start.ts + ops.length + 1 := by
      simpa using stamps ops.length (by simp)
    have prefixStamps : ∀ k (hk : k < ops.length), (ops.get ⟨k, hk⟩).wt = start.ts + k + 1 := by
      intro k hk
      simpa [List.getElem_append_left hk] using stamps k (by simp; omega)
    have below : ∀ o' ∈ ops ++ [o], o'.rt < o.wt := by
      intro o' ho'
      have fresh' := fresh o' ho'
      rcases List.mem_append.mp ho' with h | h
      · obtain ⟨⟨k, hk⟩, rfl⟩ := List.get_of_mem h
        have := prefixStamps k hk
        omega
      · rw [List.mem_singleton] at h
        exact h ▸ fresh'
    have wFinal : MemOp.write p o ∈ snapshot p final := by
      have hw : MemOp.write p o ∈ snapshot p start.memory +
          (((ops ++ [o]).map (MemOp.write p) : List Tuple) : Multiset Tuple) := by
        rw [coe_map_append]
        simp
      rw [balanced, Multiset.mem_add] at hw
      refine hw.resolve_left fun hr => ?_
      rw [Multiset.mem_coe, List.mem_map] at hr
      obtain ⟨o', ho', same⟩ := hr
      have := below o' ho'
      have := congrArg Prod.fst same
      simp only [MemOp.read, MemOp.write] at this
      omega
    obtain ⟨finalStamp, finalValue, hg⟩ : (final (o.access.globalIndex p)).stamp = o.wt ∧
        (final (o.access.globalIndex p)).value = o.access.vw ∧
        o.access.globalIndex p < p.cells := by
      rw [snapshot_eq, Multiset.mem_map] at wFinal
      obtain ⟨x, hx, same⟩ := wFinal
      simp only [MemOp.write, Prod.mk.injEq] at same
      obtain ⟨stamp, rfl, value⟩ := same
      exact ⟨stamp, value, Multiset.mem_range.mp hx⟩
    set final' := Function.update final (o.access.globalIndex p) ⟨o.access.vr, o.rt⟩ with hfinal'
    have swap : snapshot p final' + {MemOp.write p o} = snapshot p final + {MemOp.read p o} := by
      have := snapshot_update p final hg ⟨o.access.vr, o.rt⟩
      rwa [finalStamp, finalValue] at this
    have balanced' : snapshot p start.memory +
        ((ops.map (MemOp.write p) : List Tuple) : Multiset Tuple) =
        ((ops.map (MemOp.read p) : List Tuple) : Multiset Tuple) + snapshot p final' := by
      rw [coe_map_append, coe_map_append] at balanced
      apply add_right_cancel (b := ({MemOp.write p o} : Multiset Tuple))
      rw [add_assoc, balanced, add_assoc _ (snapshot p final'), swap]
      simp only [add_comm, add_left_comm]
    obtain ⟨m', run, agree⟩ := ih (final := final') prefixStamps
      (fun o' h => fresh o' (List.mem_append_left _ h))
      (fun o' h => valid o' (List.mem_append_left _ h)) balanced'
    have readOk : (m' (o.access.globalIndex p)).value = o.access.vr := by
      rw [agree (Set.mem_Iio.mpr hg), hfinal', Function.update_self]
    refine ⟨Function.update m' (o.access.globalIndex p) ⟨o.access.vw, start.ts + ops.length + 1⟩,
      ?_, fun x hx => ?_⟩
    · rw [List.map_append, accessAll_append, run]
      simp [accessAll, access, valid o (by simp), readOk, Machine.apply, Nat.add_assoc]
    · by_cases h : x = o.access.globalIndex p
      · subst h
        rw [Function.update_self]
        exact cell_ext finalValue.symm (by rw [finalStamp, lastStamp])
      · rw [Function.update_of_ne h, agree hx, hfinal', Function.update_of_ne h]

/-- The honest operations of a replay: `rt` is the stamp of the cell when the
access runs, and `wt` is the new global timestamp. -/
def honestOps (p : Plan) : Machine → List PortAccess → List MemOp
  | _, [] => []
  | m, a :: rest => ⟨a, (m.memory (a.globalIndex p)).stamp, m.ts + 1⟩ :: honestOps p (m.apply p a) rest

/-- Converse of Lemma 5: an honest replay from a machine whose stamps do not
exceed its timestamp is balanced, fresh, and has consecutive write stamps. -/
theorem honest_balanced (p : Plan) {start m : Machine} {as : List PortAccess}
    (below : ∀ g, (start.memory g).stamp ≤ start.ts)
    (run : accessAll p start as = some m) :
    (honestOps p start as).map MemOp.access = as ∧
      (∀ o ∈ honestOps p start as, o.rt < o.wt) ∧
      (∀ k (hk : k < (honestOps p start as).length),
        ((honestOps p start as).get ⟨k, hk⟩).wt = start.ts + k + 1) ∧
      snapshot p start.memory +
          (((honestOps p start as).map (MemOp.write p) : List Tuple) : Multiset Tuple) =
        (((honestOps p start as).map (MemOp.read p) : List Tuple) : Multiset Tuple) +
          snapshot p m.memory ∧
      m.ts = start.ts + as.length ∧ (∀ g, (m.memory g).stamp ≤ m.ts) := by
  induction as generalizing start with
  | nil =>
    simp only [accessAll_nil, Option.some.injEq] at run
    subst run
    simp [honestOps, below]
  | cons a rest ih =>
    obtain ⟨m₁, h₁, run₁⟩ := Option.bind_eq_some_iff.mp
      (show (access p start a).bind (fun m' => accessAll p m' rest) = some m from run)
    unfold access at h₁
    split_ifs at h₁ with hcond
    obtain rfl := Option.some.inj h₁
    obtain ⟨hv, hr⟩ := hcond
    have hg : a.globalIndex p < p.cells := hv.globalIndex_lt
    have below₁ : ∀ g, ((start.apply p a).memory g).stamp ≤ (start.apply p a).ts := by
      intro g
      simp only [Machine.apply]
      by_cases h : g = a.globalIndex p
      · subst h
        simp
      · rw [Function.update_of_ne h]
        exact Nat.le_succ_of_le (below g)
    obtain ⟨accesses, fresh, stamps, balanced, length, bound⟩ :=
      ih (start := start.apply p a) below₁ run₁
    have swap : snapshot p (start.apply p a).memory +
        {MemOp.read p ⟨a, (start.memory (a.globalIndex p)).stamp, start.ts + 1⟩} =
        snapshot p start.memory +
          {MemOp.write p ⟨a, (start.memory (a.globalIndex p)).stamp, start.ts + 1⟩} := by
      have := snapshot_update p start.memory hg ⟨a.vw, start.ts + 1⟩
      rw [hr] at this
      exact this
    refine ⟨by simp [honestOps, accesses], ?_, ?_, ?_, ?_, bound⟩
    · intro o ho
      simp only [honestOps, List.mem_cons] at ho
      rcases ho with rfl | ho
      · exact Nat.lt_succ_of_le (below _)
      · exact fresh o ho
    · intro k hk
      cases k with
      | zero => simp [honestOps]
      | succ k =>
        have := stamps k (by simpa [honestOps] using hk)
        simp only [List.get_eq_getElem] at this ⊢
        simp only [honestOps, List.getElem_cons_succ]
        rw [this]
        simp only [Machine.apply]
        omega
    · simp only [honestOps, List.map_cons, ← Multiset.cons_coe, ← Multiset.singleton_add]
      rw [← add_assoc, ← swap, add_right_comm, balanced]
      simp only [add_comm, add_assoc]
    · simp only [List.length_cons, length, Machine.apply]
      omega

end NightstreamFPrime.Spec.Nebula
