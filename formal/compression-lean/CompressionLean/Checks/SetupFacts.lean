import CompressionLean.Final.RustSetup
import CompressionLean.Checks.KeyFacts
import Std.Data.HashSet.Lemmas
import CompressionLean.Lib.List
import CompressionLean.Spec.Checks

/-!
Owns the proof of the check of the layer-1 setup that Rust builds for the
production key (`setupFailure`, `Spec/Checks.lean`). On the key and the runs
that `PackageRows::visit_rows` pushes, the check decides that `build` returns `poseidon2Setup` at the carrier of the
Poseidon2 hash chain, that this setup fits the key's shape (`Setup.Fits`) and
that the key's structure describes it (`Structure.Describes`).
`setupFailure_sound` proves that a pass gives the three facts. The runs are
read at run time, so the `setupCheck` executable runs the check on the runs
that it reads. `columnsFailure_sound` proves that a pass of the column check
makes `poseidon2Setup`'s run oracles the columns that Rust commits.

The check reads the runs a fixed number of times. It computes `classesOf` and
`keysOf` with a hash set of the distinct values and a sort of them
(`sortedValues`), which is `sortedSet` (`sortedValues_toList`).
-/

namespace CompressionLean.Checks

open CompressionLean.Spec
open CompressionLean.Final CompressionLean.Setup
open CompressionLean.Lib
open NightstreamFPrime.Spec NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open CompressionLean.Circuit (Layer1Key)

/-! ## Sorted sets in one pass -/

/-- A sorted list without repeats is the `sortedSet` of every list with its
members. -/
theorem sortedSet_eq_of {l r : List ℕ} (sorted : r.Pairwise (· ≤ ·)) (nodup : r.Nodup)
    (mem : ∀ x, x ∈ r ↔ x ∈ l) : sortedSet l = r := by
  refine List.Perm.eq_of_pairwise (fun _ _ _ _ => Nat.le_antisymm) (List.pairwise_insertionSort _ _) sorted ?_
  rw [sortedSet, List.perm_ext_iff_of_nodup ((List.perm_insertionSort _ _).nodup_iff.mpr (List.nodup_dedup l)) nodup]
  intro x
  rw [List.mem_insertionSort, List.mem_dedup, mem]

theorem sortedSet_congr {l l' : List ℕ} (mem : ∀ x, x ∈ l ↔ x ∈ l') : sortedSet l = sortedSet l' :=
  sortedSet_eq_of (List.pairwise_insertionSort _ _)
    ((List.perm_insertionSort _ _).nodup_iff.mpr (List.nodup_dedup l')) fun x => by
      rw [mem_sortedSet, mem]

theorem mem_foldl_insert (f : Push → ℕ) (x : ℕ) : ∀ (pushes : List Push) (set : Std.HashSet ℕ),
    x ∈ pushes.foldl (fun set p => set.insert (f p)) set ↔ x ∈ set ∨ x ∈ pushes.map f
  | [], set => by simp
  | p :: pushes, set => by
      rw [List.foldl_cons, mem_foldl_insert f x pushes, Std.HashSet.mem_insert, beq_iff_eq, List.map_cons,
        List.mem_cons, eq_comm (a := x)]
      tauto

/-- `sortedValues` is the `sortedSet` of the values. -/
theorem sortedValues_toList (f : Push → ℕ) (pushes : List Push) :
    (sortedValues f pushes).toList = sortedSet (pushes.map f) := by
  rw [sortedValues, List.toList_toArray]
  symm
  refine sortedSet_eq_of (List.pairwise_mergeSort' (· ≤ ·) _) ((List.mergeSort_perm _ _).nodup_iff.mpr ?_)
    fun x => ?_
  · exact Std.HashSet.distinct_toList.imp fun h => by simpa using h
  · rw [List.mem_mergeSort, Std.HashSet.mem_toList, mem_foldl_insert]
    simp

/-! ## The run list -/

theorem runList_length (pushes : List Push) (m : ℕ) :
    (runList m pushes).length = pushes.countP fun p => p.matrix < m := by
  rw [(runList_perm pushes m).length_eq, List.countP_eq_length_filter]

theorem runList_succ (pushes : List Push) (m : ℕ) :
    runList (m + 1) pushes = runList m pushes ++ pushes.filter fun p => p.matrix = m := by
  simp [runList, List.range_succ, List.flatMap_append]

/-- The runs of the run list from the count of the runs of the matrices below
`j` to the count of those up to `j` are the runs of matrix `j`. -/
theorem runList_matrix (pushes : List Push) {m j k : ℕ} (lt : j < m)
    (lo : (pushes.countP fun p => p.matrix < j) ≤ k) (hi : k < pushes.countP fun p => p.matrix < j + 1) :
    ((runList m pushes).getD k Push.pad).matrix = j := by
  induction m with
  | zero => omega
  | succ m ih =>
      rw [runList_succ]
      rcases Nat.lt_or_ge j m with below | top
      · have inside : k < (runList m pushes).length := by
          rw [runList_length]
          exact lt_of_lt_of_le hi (List.countP_mono_left fun p _ h => by simp at h ⊢; omega)
        rw [List.getD_append _ _ _ _ inside]
        exact ih below
      · obtain rfl : j = m := by omega
        have short := runList_length pushes j
        have long := runList_length pushes (j + 1)
        rw [runList_succ, List.length_append] at long
        have inside : k - (runList j pushes).length < (pushes.filter fun p => p.matrix = j).length := by omega
        rw [List.getD_append_right _ _ _ _ (by omega), List.getD_eq_getElem _ _ inside]
        simpa using (List.mem_filter.mp (List.getElem_mem inside)).2

/-- With every run in a matrix below `m`, the run list holds the runs. -/
theorem runList_perm_of {pushes : List Push} {m : ℕ} (inside : ∀ p ∈ pushes, p.matrix < m) :
    (runList m pushes).Perm pushes := by
  have := runList_perm pushes m
  rwa [List.filter_eq_self.mpr fun p member => by simpa using inside p member] at this

/-- With every run in a matrix below `m`, the run list has the classes and the
slot keys that the checks sort from the runs. -/
theorem runList_sorted {pushes : List Push} {m : ℕ} (inside : ∀ p ∈ pushes, p.matrix < m) :
    classesOf (runList m pushes) = (sortedValues Push.classCode pushes).toList ∧
      keysOf (runList m pushes) =
        (sortedValues (slotKey (sortedValues Push.classCode pushes).toList) pushes).toList := by
  have perm := runList_perm_of inside
  have members (f : Push → ℕ) (x : ℕ) : x ∈ (runList m pushes).map f ↔ x ∈ pushes.map f := by
    simp only [List.mem_map, perm.mem_iff]
  have hc : classesOf (runList m pushes) = (sortedValues Push.classCode pushes).toList := by
    rw [sortedValues_toList]
    exact sortedSet_congr (members _)
  refine ⟨hc, ?_⟩
  rw [sortedValues_toList, keysOf, hc]
  exact sortedSet_congr (members _)

theorem getD_mem_or_pad (l : List Push) (k : ℕ) : l.getD k Push.pad ∈ l ∨ l.getD k Push.pad = Push.pad := by
  rcases Nat.lt_or_ge k l.length with inside | past
  · exact Or.inl (List.getD_eq_getElem _ _ inside ▸ List.getElem_mem inside)
  · exact Or.inr (List.getD_eq_default _ _ past)

/-! ## The check -/

theorem poseidon2Columns_eq : poseidon2Columns = Phi81CarrierLayout.carrierWidth 43963750 := by
  rw [poseidon2Columns, Production.poseidon2Width_eq]

/-- **The setup check.** When the key and the runs fail no fact of the check,
`build` returns `poseidon2Setup` at the carrier of the Poseidon2 hash chain,
the setup fits the key's shape, and the key's structure describes it. -/
theorem setupFailure_sound {key : Layer1Key} {pushes : List Push} (seed : List ℕ)
    (pass : setupFailure key pushes = none) :
    build seed poseidon2Columns Production.matrices pushes = some (poseidon2Setup seed pushes) ∧
      (poseidon2Setup seed pushes).Fits (shapeOf key.relation) ∧
      (structureOf key.relation.key.layout).Describes (shapeOf key.relation) (poseidon2Setup seed pushes) := by
  obtain ⟨runs, matrices, bounds, count, blocks, slots, cover, sizes, shapes, -⟩ := all_of_firstFailure pass
  dsimp only at runs matrices bounds count blocks slots cover sizes shapes
  unfold poseidon2Setup
  obtain ⟨matrices, matrixVars⟩ := matrices
  obtain ⟨slots, slotBits⟩ := slots
  set m := Production.matrices
  set classes := (sortedValues Push.classCode pushes).toList with classes_def
  set keys := sortedValues (slotKey classes) pushes with keys_def
  have perm : (runList m pushes).Perm pushes := by
    have := runList_perm pushes m
    rwa [List.filter_eq_self.mpr fun p member => by simpa using (runs p member).1] at this
  have members (f : Push → ℕ) (x : ℕ) : x ∈ (runList m pushes).map f ↔ x ∈ pushes.map f := by
    simp only [List.mem_map, perm.mem_iff]
  have hc : classesOf (runList m pushes) = classes := by
    rw [classes_def, sortedValues_toList]
    exact sortedSet_congr (members _)
  have hk : keysOf (runList m pushes) = keys.toList := by
    rw [keys_def, sortedValues_toList, keysOf, hc]
    exact sortedSet_congr (members _)
  have runMem (k : ℕ) (inside : k < (runList m pushes).length) : (runList m pushes).getD k Push.pad ∈ pushes :=
    perm.mem_iff.mp (List.getD_eq_getElem _ _ inside ▸ List.getElem_mem inside)
  have run (k : ℕ) :
      (runList m pushes).getD k Push.pad ∈ pushes ∨ (runList m pushes).getD k Push.pad = Push.pad :=
    (getD_mem_or_pad _ k).imp_left perm.mem_iff.mp
  have bound (j : ℕ) (le : j ≤ m) : key.relation.key.layout.runBounds[j]! = pushes.countP fun p => p.matrix < j :=
    bounds j (by omega)
  have keyMem (k : ℕ) (inside : k < (runList m pushes).length) :
      slotKey classes ((runList m pushes).getD k Push.pad) ∈ keys.toList := by
    rw [← hk, ← hc]
    exact mem_sortedSet.mpr (List.mem_map_of_mem (List.getD_eq_getElem _ _ inside ▸ List.getElem_mem inside))
  have word (σ : ℕ) : keys.toList.getD σ 0 = keys.getD σ 0 := by
    rw [List.getD_eq_getElem?_getD, Array.getElem?_toList, Array.getD_eq_getD_getElem?]
  have size (σ : ℕ) := sizes _ (getD_mem_cons keys.toList σ)
  have accepts : Accepts poseidon2Columns m pushes :=
    ⟨by rw [poseidon2Columns_eq]; decide, by decide, by rw [poseidon2Columns_eq]; decide,
      by rw [poseidon2Columns_eq]; decide, fun p member =>
      ⟨(runs p member).1, (runs p member).2.1, (runs p member).2.2.1⟩⟩
  refine ⟨by rw [build, ite_eq_left accepts], ?_, ?_⟩ <;> simp only [setupOfRuns, hc, hk]
  · refine ⟨?_, fun k past => ?_, fun k => ?_, fun k => ?_, fun k => ?_, ?_, fun k inside => ?_, fun k inside => ?_,
      fun σ => Nat.mod_lt _ (by decide), fun σ => (size σ).1, fun σ => (size σ).2, fun i b l lane pad => ?_⟩ <;>
      dsimp only [shapeOf, Replay.Relation.matrices, Replay.laneVars] at *
    · rw [runList_length, ← bound m le_rfl]
      exact count
    · rw [List.getD_eq_default (runList m pushes) Push.pad past, ite_eq_right (not_lt.mpr past)]
      exact ⟨rfl, rfl, rfl, rfl⟩
    · rcases run k with member | pad
      · exact (runs _ member).2.2.2
      · rw [pad]
        exact Nat.two_pow_pos _
    · split_ifs with inside
      · exact lt_of_lt_of_le (List.idxOf_lt_length_of_mem (keyMem k inside)) (by simpa using slotBits)
      · exact Nat.two_pow_pos _
    · rcases run k with member | pad
      · exact lt_of_lt_of_le (runs _ member).1 matrixVars
      · rw [pad]
        exact Nat.two_pow_pos _
    · simpa using slotBits
    · rw [ite_eq_left inside]
      exact List.idxOf_lt_length_of_mem (keyMem k inside)
    · rw [matrices]
      exact (runs _ (runMem k inside)).1
    · have low : (64 * b + l) % 64 = l := by omega
      have high : (64 * b + l) / 64 = b := by omega
      have same : Replay.D = ringDegree := rfl
      rw [seedKey, show (2 : ℕ) ^ 6 = 64 from rfl, low, high, ite_eq_right]
      omega
  · refine ⟨?_, fun j k lt lo hi => ?_, slots, cover, fun i σ lt lo hi => ?_⟩ <;>
      dsimp only [shapeOf, Replay.Relation.matrices] at *
    · rw [runList_length, matrices]
      exact (bound m le_rfl).symm
    · have lt : j < m := by rw [← matrices]; exact lt
      replace lo : (pushes.countP fun p => p.matrix < j) ≤ k := by rw [← bound j lt.le]; exact lo
      replace hi : k < pushes.countP fun p => p.matrix < j + 1 := by rw [← bound (j + 1) lt]; exact hi
      exact runList_matrix pushes lt lo hi
    · have shape := shapes i lt (σ - (structureOf key.relation.key.layout).segmentStart i) (by omega)
      rw [Nat.add_sub_cancel' lo] at shape
      rw [word]
      exact shape

/-- **The column check.** When the setup check and the column check pass, the
run oracles of `poseidon2Setup` (`setupOracle` past the key rows) are the
columns that Rust commits, padded with zeros. -/
theorem columnsFailure_sound {key : Layer1Key} {pushes : List Push} {cols : SetupColumns} (seed : List ℕ)
    (setup : setupFailure key pushes = none) (pass : columnsFailure pushes cols = none) (x : ℕ) :
    ((poseidon2Setup seed pushes).runRow x : Gl) = cols.row.getD x 0 ∧
      ((poseidon2Setup seed pushes).runSlot x : Gl) = cols.slot.getD x 0 ∧
      (poseidon2Setup seed pushes).runCoef x = cols.coef.getD x 0 ∧
      (if x < (poseidon2Setup seed pushes).slots then ((poseidon2Setup seed pushes).slotBlock x : Gl) else 0) =
        cols.block.getD x 0 := by
  obtain ⟨runs, -⟩ := all_of_firstFailure setup
  obtain ⟨sizes, rows, slots, coefs, blocks, -⟩ := all_of_firstFailure pass
  dsimp only at runs sizes rows slots coefs blocks
  obtain ⟨hc, hk⟩ := runList_sorted (m := Production.matrices) fun p member => (runs p member).1
  obtain ⟨rowSize, slotSize, coefSize, blockSize⟩ := sizes
  simp only [List.size_toArray] at rowSize slotSize coefSize rows slots coefs
  have get (k : ℕ) : (runList Production.matrices pushes).toArray.getD k Push.pad =
      (runList Production.matrices pushes).getD k Push.pad := by
    simp [List.getD_eq_getElem?_getD, Array.getD_eq_getD_getElem?]
  have word (a : Array ℕ) (σ : ℕ) : a.toList.getD σ 0 = a.getD σ 0 := by
    rw [List.getD_eq_getElem?_getD, Array.getElem?_toList, Array.getD_eq_getD_getElem?]
  have beyond (a : Array Gl) (k : ℕ) (past : a.size ≤ k) : a.getD k 0 = 0 := by
    rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_none past, Option.getD_none]
  have nodup : (sortedValues (slotKey (sortedValues Push.classCode pushes).toList) pushes).toList.Nodup := by
    rw [sortedValues_toList]
    exact (List.perm_insertionSort _ _).nodup_iff.mpr (List.nodup_dedup _)
  unfold poseidon2Setup setupOfRuns
  dsimp only
  rw [hc, hk]
  rcases Nat.lt_or_ge x (runList Production.matrices pushes).length with inside | past
  · refine ⟨by rw [rows x inside, get], ?_, by rw [coefs x inside, get], ?_⟩
    · obtain ⟨small, same⟩ := slots x inside
      rw [ite_eq_left inside, get] at *
      rw [← same, ← word, List.getD_eq_getElem _ _ (by simpa using small), nodup.idxOf_getElem,
        ZMod.natCast_zmod_val]
    · split_ifs with slot
      · rw [blocks x (by simpa using slot), word]
      · exact (beyond _ _ (by rw [blockSize]; simpa using slot)).symm
  · have pad := List.getD_eq_default (runList Production.matrices pushes) Push.pad past
    refine ⟨?_, ?_, ?_, ?_⟩
    · rw [pad, beyond _ _ (by omega)]
      rfl
    · rw [ite_eq_right (not_lt.mpr past), beyond _ _ (by omega)]
      rfl
    · rw [pad, beyond _ _ (by omega)]
      rfl
    · split_ifs with slot
      · rw [blocks x (by simpa using slot), word]
      · exact (beyond _ _ (by rw [blockSize]; simpa using slot)).symm

end CompressionLean.Checks
