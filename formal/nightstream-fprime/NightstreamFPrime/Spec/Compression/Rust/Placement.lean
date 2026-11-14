import NightstreamFPrime.Spec.Compression.Rust.Layout

/-! The placement of `layoutNew`: each slice of `placeAll` is aligned, after the start cursor and
before the end cursor, and has the variables of its request (`placeAll_spec`), and `lookup` finds
the slice of each requested claim (`lookup_slice`). -/

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

end NightstreamFPrime.Spec.Compression.Rust
