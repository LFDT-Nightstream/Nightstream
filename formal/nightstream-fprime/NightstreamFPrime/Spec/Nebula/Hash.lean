import Mathlib.Data.List.Basic

/-! Owns the record chains of spec §9.2 over an abstract hash, the challenge
transcript input of §9.3, and security note Lemma 2 (chain binding). A hash is
any function. A collision is always a pair of inputs from explicit, finite
input lists that a run determines, so a reduction can find it by search; a
collision "somewhere" would say nothing for a hash with finite output. The
concrete Poseidon2 framing (obligation Ob3) is in `Lifecycle/Nebula/Framing`. -/

namespace NightstreamFPrime.Spec.Nebula

/-- The two chain families: the ops lane, and the IS and FS lanes (shared). -/
inductive Lane
  | ops
  | mem
deriving DecidableEq

/-- Every hash input of the memory phase. The constructor and the lane are the
domain tag of spec §9.2. -/
inductive HashInput (Digest : Type) where
  | header (lane : Lane) (planDigest : Digest)
  | chain (lane : Lane) (index : ℕ) (previous : Digest) (packed : List ℕ)

/-- The §9.3 transcript input, in absorb order. -/
structure EtaInput (Digest : Type) where
  planDigest : Digest
  ts : ℕ
  opsRoot : Digest
  memRoot : Digest
  finalRoot : Digest

/-- A collision between an input of `xs` and an input of `ys`. -/
def CollisionIn {Digest : Type} (H : HashInput Digest → Digest)
    (xs ys : List (HashInput Digest)) : Prop :=
  ∃ a ∈ xs, ∃ b ∈ ys, a ≠ b ∧ H a = H b

theorem CollisionIn.mono {Digest : Type} {H : HashInput Digest → Digest}
    {xs ys xs' ys' : List (HashInput Digest)} (hx : xs ⊆ xs') (hy : ys ⊆ ys')
    (collision : CollisionIn H xs ys) : CollisionIn H xs' ys' := by
  obtain ⟨a, ha, b, hb, ne, eq⟩ := collision
  exact ⟨a, hx ha, b, hy hb, ne, eq⟩

/-- Spec §9.2: `D[0] = header`, `D[j+1] = chain(j, D[j], P[j])`. The pair
carries the current digest and the next index. -/
def chainState {Digest : Type} (H : HashInput Digest → Digest) (lane : Lane) (pd : Digest)
    (packed : List (List ℕ)) : Digest × ℕ :=
  packed.foldl (fun acc P => (H (.chain lane acc.2 acc.1 P), acc.2 + 1))
    (H (.header lane pd), 0)

/-- The root of a lane chain. -/
def chainRoot {Digest : Type} (H : HashInput Digest → Digest) (lane : Lane) (pd : Digest)
    (packed : List (List ℕ)) : Digest :=
  (chainState H lane pd packed).1

/-- The hash inputs of a chain computation: the header, then each link. -/
def chainInputs {Digest : Type} (H : HashInput Digest → Digest) (lane : Lane) (pd : Digest)
    (packed : List (List ℕ)) : List (HashInput Digest) :=
  .header lane pd ::
    (List.range packed.length).zipWith
      (fun j P => .chain lane j (chainRoot H lane pd (packed.take j)) P) packed

/-- The root of an empty chain is the header. -/
theorem chainRoot_nil {Digest : Type} (H : HashInput Digest → Digest) (lane : Lane)
    (pd : Digest) : chainRoot H lane pd [] = H (.header lane pd) := rfl

private theorem chainState_append {Digest : Type} (H : HashInput Digest → Digest) (lane : Lane)
    (pd : Digest) (ps : List (List ℕ)) (P : List ℕ) :
    chainState H lane pd (ps ++ [P]) =
      (H (.chain lane (chainState H lane pd ps).2 (chainState H lane pd ps).1 P),
        (chainState H lane pd ps).2 + 1) := by
  unfold chainState
  rw [List.foldl_append]
  rfl

private theorem chainState_index {Digest : Type} (H : HashInput Digest → Digest) (lane : Lane)
    (pd : Digest) (ps : List (List ℕ)) : (chainState H lane pd ps).2 = ps.length := by
  induction size : ps.length generalizing ps with
  | zero => rw [List.eq_nil_of_length_eq_zero size]; rfl
  | succ n ih =>
    rcases List.eq_nil_or_concat' ps with rfl | ⟨ps, P, rfl⟩
    · simp at size
    rw [chainState_append, ih ps (by simpa using size)]

/-- One more link. -/
theorem chainRoot_append {Digest : Type} (H : HashInput Digest → Digest) (lane : Lane)
    (pd : Digest) (ps : List (List ℕ)) (P : List ℕ) :
    chainRoot H lane pd (ps ++ [P]) =
      H (.chain lane ps.length (chainRoot H lane pd ps) P) := by
  unfold chainRoot
  rw [chainState_append, chainState_index]

private theorem zipWith_range_congr {α β : Type} {n : ℕ} {l : List α} {f g : ℕ → α → β}
    (agree : ∀ j < n, ∀ a, f j a = g j a) :
    (List.range n).zipWith f l = (List.range n).zipWith g l := by
  apply List.ext_getElem (by simp)
  intro i hf _
  simp only [List.getElem_zipWith, List.getElem_range]
  exact agree i (by simp at hf; omega) _

private theorem chainInputs_append {Digest : Type} (H : HashInput Digest → Digest) (lane : Lane)
    (pd : Digest) (ps : List (List ℕ)) (P : List ℕ) :
    chainInputs H lane pd (ps ++ [P]) =
      chainInputs H lane pd ps ++ [.chain lane ps.length (chainRoot H lane pd ps) P] := by
  have prefixLinks : (List.range ps.length).zipWith
      (fun j Q => HashInput.chain lane j (chainRoot H lane pd ((ps ++ [P]).take j)) Q) ps =
      (List.range ps.length).zipWith
        (fun j Q => HashInput.chain lane j (chainRoot H lane pd (ps.take j)) Q) ps :=
    zipWith_range_congr fun j lt _ => by rw [List.take_append_of_le_length (Nat.le_of_lt lt)]
  unfold chainInputs
  rw [List.length_append, List.length_singleton, List.range_succ, List.zipWith_append (by simp),
    prefixLinks]
  simp [List.take_left']

private theorem header_mem {Digest : Type} {H : HashInput Digest → Digest} {lane : Lane}
    {pd : Digest} {ps : List (List ℕ)} :
    HashInput.header lane pd ∈ chainInputs H lane pd ps :=
  List.mem_cons_self

private theorem last_mem {Digest : Type} {H : HashInput Digest → Digest} {lane : Lane}
    {pd : Digest} {ps : List (List ℕ)} {P : List ℕ} :
    HashInput.chain lane ps.length (chainRoot H lane pd ps) P ∈
      chainInputs H lane pd (ps ++ [P]) := by
  rw [chainInputs_append]
  exact List.mem_append_right _ (List.mem_singleton_self _)

private theorem inputs_subset {Digest : Type} {H : HashInput Digest → Digest} {lane : Lane}
    {pd : Digest} {ps : List (List ℕ)} {P : List ℕ} :
    chainInputs H lane pd ps ⊆ chainInputs H lane pd (ps ++ [P]) := by
  rw [chainInputs_append]
  exact List.subset_append_left _ _

private theorem eq_or_collision_of_length {Digest : Type} (H : HashInput Digest → Digest)
    (lane : Lane) (pd : Digest) :
    ∀ (n : ℕ) {ps qs : List (List ℕ)}, ps.length = n →
      chainRoot H lane pd ps = chainRoot H lane pd qs →
      ps = qs ∨ CollisionIn H (chainInputs H lane pd ps) (chainInputs H lane pd qs)
  | 0, ps, qs, size, same => by
    obtain rfl := List.eq_nil_of_length_eq_zero size
    rcases List.eq_nil_or_concat' qs with rfl | ⟨qs, Q, rfl⟩
    · exact Or.inl rfl
    · rw [chainRoot_nil, chainRoot_append] at same
      exact Or.inr ⟨_, header_mem, _, last_mem, nofun, same⟩
  | n + 1, ps, qs, size, same => by
    rcases List.eq_nil_or_concat' ps with rfl | ⟨ps, P, rfl⟩
    · simp at size
    rcases List.eq_nil_or_concat' qs with rfl | ⟨qs, Q, rfl⟩
    · rw [chainRoot_nil, chainRoot_append] at same
      exact Or.inr ⟨_, last_mem, _, header_mem, nofun, same⟩
    rw [chainRoot_append, chainRoot_append] at same
    by_cases links : HashInput.chain lane ps.length (chainRoot H lane pd ps) P =
        HashInput.chain lane qs.length (chainRoot H lane pd qs) Q
    · obtain ⟨-, -, roots, rfl⟩ := HashInput.chain.inj links
      rcases eq_or_collision_of_length H lane pd n (by simpa using size) roots with rfl | collision
      · exact Or.inl rfl
      · exact Or.inr (collision.mono inputs_subset inputs_subset)
    · exact Or.inr ⟨_, last_mem, _, last_mem, links, same⟩

/-- Security note Lemma 2: equal roots give equal packed lanes, or a collision
between the two chains' own inputs. -/
theorem chainRoot_eq_or_collision {Digest : Type} (H : HashInput Digest → Digest)
    (lane : Lane) (pd : Digest) {ps qs : List (List ℕ)}
    (same : chainRoot H lane pd ps = chainRoot H lane pd qs) :
    ps = qs ∨ CollisionIn H (chainInputs H lane pd ps) (chainInputs H lane pd qs) :=
  eq_or_collision_of_length H lane pd ps.length rfl same

/-- A chain input whose words are canonical: its index is below `indexBound`,
its packed lane has the plan's length for its lane, and every packed element is
below `2 ^ 63` (Lemma 1). A header input is always canonical. -/
def HashInput.Canonical {Digest : Type} (laneLength : Lane → ℕ) (indexBound : ℕ) :
    HashInput Digest → Prop
  | .header _ _ => True
  | .chain lane index _ packed =>
    index < indexBound ∧ packed.length = laneLength lane ∧ ∀ x ∈ packed, x < 2 ^ 63

/-- Every input of a chain over at most `indexBound` lanes of canonical shape
is canonical. -/
theorem chainInputs_canonical {Digest : Type} (H : HashInput Digest → Digest) (lane : Lane)
    (pd : Digest) {laneLength : Lane → ℕ} {indexBound : ℕ} {ps : List (List ℕ)}
    (count : ps.length ≤ indexBound)
    (shape : ∀ P ∈ ps, P.length = laneLength lane ∧ ∀ x ∈ P, x < 2 ^ 63) :
    ∀ x ∈ chainInputs H lane pd ps, x.Canonical laneLength indexBound := by
  intro x member
  rcases List.mem_cons.1 member with rfl | member
  · trivial
  obtain ⟨i, hi, rfl⟩ := List.mem_iff_getElem.1 member
  simp only [List.length_zipWith, List.length_range, Nat.min_self] at hi
  simp only [List.getElem_zipWith, List.getElem_range]
  exact ⟨by omega, shape _ (List.getElem_mem hi)⟩

end NightstreamFPrime.Spec.Nebula
