namespace NightstreamFPrime.Spec

theorem flatMap_eq_of_lengths {α β : Type _} (indices : List α) (left right : α → List β)
    (lengths : ∀ index ∈ indices, (left index).length = (right index).length)
    (same : indices.flatMap left = indices.flatMap right) :
    ∀ index ∈ indices, left index = right index := by
  induction indices with
  | nil => simp
  | cons head tail inductionHypothesis =>
      simp only [List.flatMap_cons] at same
      obtain ⟨headEqual, tailEqual⟩ := List.append_inj same (lengths head (by simp))
      intro index member
      rcases List.mem_cons.mp member with rfl | member
      · exact headEqual
      · exact inductionHypothesis
          (fun index member => lengths index (List.mem_cons_of_mem _ member))
          tailEqual index member

theorem flatMap_length_eq {α β : Type _} (indices : List α) (left right : α → List β)
    (lengths : ∀ index ∈ indices, (left index).length = (right index).length) :
    (indices.flatMap left).length = (indices.flatMap right).length := by
  rw [List.length_flatMap, List.length_flatMap]
  congr 1
  exact List.map_congr_left lengths

theorem flatMap_length_constant {α β : Type _} (indices : List α) (values : α → List β)
    (count : Nat) (each : ∀ index, (values index).length = count) :
    (indices.flatMap values).length = indices.length * count := by
  induction indices with
  | nil => simp
  | cons head tail inductionHypothesis =>
      rw [List.flatMap_cons, List.length_append, each, inductionHypothesis]
      simp [Nat.succ_mul, Nat.add_comm]

end NightstreamFPrime.Spec
