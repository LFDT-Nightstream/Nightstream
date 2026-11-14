import NightstreamFPrime.Spec.Compression.Rust.Layout

/-! The placement of `layoutNew`: each slice of `placeAll` is aligned, after the start cursor and
before the end cursor, and has the variables of its request (`placeAll_spec`), and `lookup` finds
the slice of each requested claim (`lookup_slice`). So the witness slice of each claim has the
variables of its blocks, is aligned and is inside the witness slices, and inside the span when the
claim has a table (`cellOf_witness`). The count slice of each table holds the table and is inside
the cells of `W` (`cellOf_table`). -/

namespace NightstreamFPrime.Spec.Compression.Rust

open NightstreamFPrime.Spec.Compression.Verifier (Slice)

theorem nextMultiple_spec (cursor size : ℕ) (positive : 0 < size) :
    size ∣ nextMultiple cursor size ∧ cursor ≤ nextMultiple cursor size := by
  unfold nextMultiple
  refine ⟨Dvd.intro_left _ rfl, ?_⟩
  have split := Nat.div_add_mod (cursor + size - 1) size
  have small := Nat.mod_lt (cursor + size - 1) positive
  rw [Nat.mul_comm] at split
  generalize (cursor + size - 1) / size * size = multiple at split ⊢
  generalize (cursor + size - 1) % size = rest at split small
  omega

/-- Each slice of `placeAll` is aligned, after the start and before the end, and has the variables
of its request; the end is after the start; the claims keep their order. -/
theorem placeAll_spec : ∀ (requests : List (ℕ × ℕ)) (cursor : ℕ),
    cursor ≤ (placeAll cursor requests).2 ∧
    (placeAll cursor requests).1.map Prod.fst = requests.map Prod.fst ∧
    ∀ claim s, (claim, s) ∈ (placeAll cursor requests).1 →
      (claim, s.vars) ∈ requests ∧ 2 ^ s.vars ∣ s.offset ∧ cursor ≤ s.offset ∧
        s.offset + 2 ^ s.vars ≤ (placeAll cursor requests).2
  | [], cursor => by simp [placeAll]
  | (claim, vars) :: rest, cursor => by
      obtain ⟨aligned, after⟩ := nextMultiple_spec cursor (2 ^ vars) (pow_pos two_pos _)
      obtain ⟨grows, keys, each⟩ := placeAll_spec rest (nextMultiple cursor (2 ^ vars) + 2 ^ vars)
      refine ⟨by simp only [placeAll, place]; omega,
        by simp only [placeAll, place, List.map_cons, keys], ?_⟩
      intro claim' s member
      simp only [placeAll, place, List.mem_cons, Prod.mk.injEq] at member
      rcases member with ⟨rfl, rfl⟩ | member
      · exact ⟨List.mem_cons_self .., aligned, after, by simp only [placeAll, place]; omega⟩
      · obtain ⟨request, divides, start, inside⟩ := each claim' s member
        exact ⟨List.mem_cons_of_mem _ request, divides, by omega,
          by simp only [placeAll, place]; omega⟩

theorem lookup_mem {l : List (ℕ × Slice)} {a : ℕ} {b : Slice} (found : l.lookup a = some b) :
    (a, b) ∈ l := by
  induction l with
  | nil => simp [List.lookup] at found
  | cons head tail ih =>
      obtain ⟨k, v⟩ := head
      by_cases same : a = k
      · subst same
        simp only [List.lookup, beq_self_eq_true, Option.some.injEq] at found
        subst found
        exact List.mem_cons_self ..
      · have differ : (a == k) = false := by simp [same]
        simp only [List.lookup, differ] at found
        exact List.mem_cons_of_mem _ (ih found)

theorem lookup_isSome {l : List (ℕ × Slice)} {a : ℕ} (key : a ∈ l.map Prod.fst) :
    (l.lookup a).isSome := by
  induction l with
  | nil => simp at key
  | cons head tail ih =>
      obtain ⟨k, v⟩ := head
      by_cases same : a = k
      · subst same
        simp [List.lookup]
      · have differ : (a == k) = false := by simp [same]
        simp only [List.map_cons, List.mem_cons, same, false_or] at key
        simp only [List.lookup, differ]
        exact ih key

/-- The slice that `lookup` finds for a claim of the requests. -/
theorem lookup_slice {requests : List (ℕ × ℕ)} {cursor claim vars : ℕ}
    (request : (claim, vars) ∈ requests) :
    ∃ s, ((placeAll cursor requests).1.lookup claim).getD ⟨0, 0⟩ = s ∧
      (claim, s) ∈ (placeAll cursor requests).1 := by
  have key : claim ∈ (placeAll cursor requests).1.map Prod.fst := by
    rw [(placeAll_spec requests cursor).2.1]
    exact List.mem_map.mpr ⟨_, request, rfl⟩
  obtain ⟨s, found⟩ := Option.isSome_iff_exists.mp (lookup_isSome key)
  exact ⟨s, by rw [found]; rfl, lookup_mem found⟩

theorem mem_groupOrder {claims : List Claim} {table? : Bool} {i : ℕ} :
    i ∈ groupOrder claims table? ↔
      i < claims.length ∧ (claims.getD i default).table.isSome = table? := by
  simp [groupOrder, List.mem_mergeSort, List.mem_filter, List.mem_range]

theorem mem_countRequests {claims : List Claim} {i vars : ℕ} :
    (i, vars) ∈ countRequests claims ↔ i < claims.length ∧
      ∃ bound, (claims.getD i default).table = some bound ∧ vars = varsFor (2 * bound + 1) := by
  simp only [countRequests, List.mem_filterMap, List.mem_range, Option.map_eq_some_iff,
    Prod.mk.injEq]
  constructor
  · rintro ⟨j, below, bound, table, rfl, rfl⟩
    exact ⟨below, bound, table, rfl⟩
  · rintro ⟨below, bound, table, rfl⟩
    exact ⟨i, below, bound, table, rfl, rfl⟩

theorem tables_le_others (claims : List Claim) :
    (tableSlices claims).2 ≤ (otherSlices claims).2 :=
  (placeAll_spec _ _).1

theorem others_le_counts (claims : List Claim) :
    (otherSlices claims).2 ≤ (countSlices claims).2 :=
  (placeAll_spec _ _).1

private theorem request_vars {claims : List Claim} {table? : Bool} {i vars : ℕ}
    (request : (i, vars) ∈ (groupOrder claims table?).map fun j => (j, witnessVarsOf claims j)) :
    vars = witnessVarsOf claims i ∧ (claims.getD i default).table.isSome = table? := by
  obtain ⟨j, member, same⟩ := List.mem_map.mp request
  simp only [Prod.mk.injEq] at same
  obtain ⟨rfl, rfl⟩ := same
  exact ⟨rfl, (mem_groupOrder.mp member).2⟩

/-- The witness slice of claim `i`: the variables of its blocks, aligned, inside the witness
slices, and inside the span when the claim has a table. -/
theorem cellOf_witness {claims : List Claim} {i : ℕ} (below : i < claims.length) :
    (cellOf claims i).witness.vars = witnessVarsOf claims i ∧
      2 ^ (cellOf claims i).witness.vars ∣ (cellOf claims i).witness.offset ∧
      (cellOf claims i).witness.offset + 2 ^ (cellOf claims i).witness.vars ≤
        (otherSlices claims).2 ∧
      ((claims.getD i default).table.isSome = true →
        (cellOf claims i).witness.offset + 2 ^ (cellOf claims i).witness.vars ≤
          (tableSlices claims).2) := by
  have key : i ∈ ((tableSlices claims).1 ++ (otherSlices claims).1).map Prod.fst := by
    rw [List.map_append, List.mem_append, tableSlices, otherSlices,
      (placeAll_spec _ _).2.1, (placeAll_spec _ _).2.1]
    cases group : (claims.getD i default).table.isSome
    · exact Or.inr (List.mem_map.mpr ⟨(i, witnessVarsOf claims i),
        List.mem_map.mpr ⟨i, mem_groupOrder.mpr ⟨below, group⟩, rfl⟩, rfl⟩)
    · exact Or.inl (List.mem_map.mpr ⟨(i, witnessVarsOf claims i),
        List.mem_map.mpr ⟨i, mem_groupOrder.mpr ⟨below, group⟩, rfl⟩, rfl⟩)
  obtain ⟨s, found⟩ := Option.isSome_iff_exists.mp (lookup_isSome key)
  have witness : (cellOf claims i).witness = s := by simp [cellOf, found]
  rw [witness]
  rcases List.mem_append.mp (lookup_mem found) with inTables | inOthers
  · obtain ⟨request, aligned, _, inside⟩ := (placeAll_spec _ 0).2.2 i s inTables
    have grows := tables_le_others claims
    exact ⟨(request_vars request).1, aligned, inside.trans grows, fun _ => inside⟩
  · obtain ⟨request, aligned, _, inside⟩ := (placeAll_spec _ _).2.2 i s inOthers
    obtain ⟨vars, group⟩ := request_vars request
    exact ⟨vars, aligned, inside, fun table => by rw [group] at table; cases table⟩

/-- The count slice of a claim with the table `[−bound, bound]`: it holds the table, is aligned
and is inside the cells of `W`. -/
theorem cellOf_table {claims : List Claim} {i bound : ℕ} (below : i < claims.length)
    (table : (claims.getD i default).table = some bound) :
    ∃ s, (cellOf claims i).table = some (bound, s) ∧ s.vars = varsFor (2 * bound + 1) ∧
      2 ^ s.vars ∣ s.offset ∧ s.offset + 2 ^ s.vars ≤ (countSlices claims).2 := by
  obtain ⟨s, found, member⟩ := lookup_slice (cursor := (otherSlices claims).2)
    (mem_countRequests.mpr ⟨below, bound, table, rfl⟩)
  obtain ⟨request, aligned, _, inside⟩ := (placeAll_spec _ _).2.2 i s member
  obtain ⟨_, bound', table', vars⟩ := mem_countRequests.mp request
  rw [table, Option.some.injEq] at table'
  subst table'
  refine ⟨s, ?_, vars, aligned, inside⟩
  simp only [cellOf, table, Option.map_some, countSlices]
  rw [found]

end NightstreamFPrime.Spec.Compression.Rust
