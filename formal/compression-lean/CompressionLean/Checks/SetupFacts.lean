import CompressionLean.Final.RustSetup
import CompressionLean.Checks.KeyFacts
import Std.Data.HashSet.Lemmas
import CompressionLean.Lib.List
import CompressionLean.Lib.Nat

/-!
Owns the check of the layer-1 setup that Rust builds for the production key
(`setupFailure`). On the key and the runs that `PackageRows::visit_rows`
pushes, it decides that `build` returns `poseidon2Setup` at the carrier of the
Poseidon2 hash chain, that this setup fits the key's shape (`Setup.Fits`) and
that the key's structure describes it (`Structure.Describes`).
`setupFacts_of_check` proves that a pass gives the three facts. The runs are
read at run time, so the `setupCheck` executable runs the check on the runs
that it reads.

The check reads the runs a fixed number of times. It computes `classesOf` and
`keysOf` with a hash set of the distinct values and a sort of them
(`sortedValues`), which is `sortedSet` (`sortedValues_toList`).
-/

namespace CompressionLean.Checks

open CompressionLean.Final
open CompressionLean.Lib
open NightstreamFPrime.Spec NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open CompressionLean.Circuit (Layer1Input)
open CompressionLean.Bridge (shapeOf structureOf)

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

/-- The distinct values of `f` on the runs in increasing order, as
`Tables::from_rows` collects the classes (a `BTreeSet`) and the slot keys (a
sort and a `dedup`): a hash set of the values in one pass, then a sort of the
distinct ones. -/
def sortedValues (f : Push → ℕ) (pushes : List Push) : Array ℕ :=
  ((pushes.foldl (fun set p => set.insert (f p)) (∅ : Std.HashSet ℕ)).toList.mergeSort).toArray

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

theorem getD_mem_or_pad (l : List Push) (k : ℕ) : l.getD k Push.pad ∈ l ∨ l.getD k Push.pad = Push.pad := by
  rcases Nat.lt_or_ge k l.length with inside | past
  · exact Or.inl (List.getD_eq_getElem _ _ inside ▸ List.getElem_mem inside)
  · exact Or.inr (List.getD_eq_default _ _ past)

/-! ## The check -/

/-- The columns of the Poseidon2 hash chain's carrier, `carrierWidth` at its
logical width: the `MatrixShape::columns` of the production circuit. -/
def poseidon2Columns : ℕ := Phi81CarrierLayout.carrierWidth poseidon2Width

/-- The setup that `build` returns for `seed` and the runs at the Poseidon2
hash chain's carrier and F′'s four matrices (`Setup::build`). -/
def poseidon2Setup (seed : List ℕ) (pushes : List Push) : Layer1.Setup Gl :=
  setupOfRuns seed (poseidon2Columns / ringDegree) (runList productionProfile.ccsMatrices pushes)

/-- The start column of the slot with key `w` (`start` in `Tables::from_rows`),
as `slotStart` reads it. -/
def keyStart (w : ℕ) : ℕ := w % 2 ^ 32 * ringDegree + w / 2 ^ 32 % 2 ^ 8

/-- Each fact of the check by name, decided on the key and the runs. -/
def setupChecks (key : Layer1Input) (pushes : List Push) : List (String × Bool) :=
  let rel := key.relation
  let layout := rel.key.layout
  let st := structureOf layout
  let matrices := productionProfile.ccsMatrices
  let rows := 2 ^ layout.rowVariables
  let classes := (sortedValues Push.classCode pushes).toList
  let keys := sortedValues (slotKey classes) pushes
  [("runs", pushes.all fun p => decide (p.matrix < matrices ∧ p.len ≤ ringDegree ∧
      p.start + p.len ≤ poseidon2Columns ∧ p.row < rows)),
    ("matrices", decide (layout.matrices = matrices ∧ matrices ≤ 2 ^ rel.matrixVars)),
    ("bounds", Nat.all (matrices + 1) fun j _ =>
      decide (layout.runBounds[j]! = pushes.countP fun p => decide (p.matrix < j))),
    ("run count", decide (layout.runBounds[matrices]! ≤ 2 ^ layout.runVariables)),
    ("blocks", decide (poseidon2Columns / ringDegree ≤ rel.key.blocks)),
    ("slots", decide (keys.size = layout.slots ∧ keys.size ≤ 2 ^ layout.slotVariables)),
    ("cover", decide (layout.segments.size = 0 → layout.slots = 0)),
    ("slot sizes", (0 :: keys.toList).all fun w =>
      decide (classes.getD (w / 2 ^ 40) 0 / 2 ^ 64 ≤ ringDegree ∧
        ringDegree * (keyStart w / ringDegree) + keyStart w % ringDegree + classes.getD (w / 2 ^ 40) 0 / 2 ^ 64 ≤
          ringDegree * rel.key.blocks)),
    ("slot shapes", Nat.all st.segments fun i _ =>
      Nat.all (st.segmentStart (i + 1) - st.segmentStart i) fun t _ =>
        let w := keys.getD (st.segmentStart i + t) 0
        decide (classes.getD (w / 2 ^ 40) 0 / 2 ^ 64 = st.classLength (st.segmentClass i) ∧
          ((classes.getD (w / 2 ^ 40) 0 % 2 ^ 64 : ℕ) : Gl) = st.classRatio (st.segmentClass i) ∧
          keyStart w % ringDegree = st.segmentLane i))]

/-- The name of the first fact of the check that the key and the runs fail. -/
def setupFailure (key : Layer1Input) (pushes : List Push) : Option String :=
  ((setupChecks key pushes).find? fun check => !check.2).map (·.1)

/-- **The setup check.** When the key and the runs fail no fact of the check,
`build` returns `poseidon2Setup` at the carrier of the Poseidon2 hash chain,
the setup fits the key's shape, and the key's structure describes it. -/
theorem setupFacts_of_check {key : Layer1Input} {pushes : List Push} (seed : List ℕ)
    (pass : setupFailure key pushes = none) :
    build seed poseidon2Columns productionProfile.ccsMatrices pushes = some (poseidon2Setup seed pushes) ∧
      (poseidon2Setup seed pushes).Fits (shapeOf key.relation) ∧
      (structureOf key.relation.key.layout).Describes (shapeOf key.relation) (poseidon2Setup seed pushes) := by
  have all : ∀ check ∈ setupChecks key pushes, check.2 = true := by
    simpa [setupFailure, List.find?_eq_none] using pass
  simp only [setupChecks, List.forall_mem_cons, List.not_mem_nil, IsEmpty.forall_iff, implies_true, and_true] at all
  obtain ⟨runs, matrices, bounds, count, blocks, slots, cover, sizes, shapes⟩ := all
  simp only [List.all_eq_true, decide_eq_true_eq] at runs sizes
  simp only [decide_eq_true_eq] at matrices count blocks slots cover
  simp only [nat_all_iff, decide_eq_true_eq] at bounds shapes
  obtain ⟨matrices, matrixVars⟩ := matrices
  obtain ⟨slots, slotBits⟩ := slots
  set m := productionProfile.ccsMatrices
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
  have keysLength : (keysOf (runList m pushes)).length = keys.size := by rw [hk, Array.length_toList]
  have keyMem (k : ℕ) (inside : k < (runList m pushes).length) :
      slotKey (classesOf (runList m pushes)) ((runList m pushes).getD k Push.pad) ∈ keysOf (runList m pushes) :=
    mem_sortedSet.mpr (List.mem_map_of_mem (List.getD_eq_getElem _ _ inside ▸ List.getElem_mem inside))
  have word (σ : ℕ) : (keysOf (runList m pushes)).getD σ 0 = keys.getD σ 0 := by
    rw [hk, List.getD_eq_getElem?_getD, Array.getElem?_toList, Array.getD_eq_getD_getElem?]
  have size (σ : ℕ) := sizes _ (hk ▸ getD_mem_cons (keysOf (runList m pushes)) σ)
  have accepts : Accepts poseidon2Columns m pushes :=
    ⟨by decide, by decide, by decide, by decide, fun p member =>
      ⟨(runs p member).1, (runs p member).2.1, (runs p member).2.2.1⟩⟩
  refine ⟨by rw [build, if_pos accepts]; rfl, ?_, ?_⟩
  · refine ⟨?_, fun k past => ?_, fun k => ?_, fun k => ?_, fun k => ?_, ?_, fun k inside => ?_, fun k inside => ?_,
      fun σ => ?_, fun σ => ?_, fun σ => ?_, fun i b l lane pad => ?_⟩
    · show (runList m pushes).length ≤ 2 ^ key.relation.key.layout.runVariables
      rw [runList_length, ← bound m le_rfl]
      exact count
    · replace past : (runList m pushes).length ≤ k := past
      show ((runList m pushes).getD k Push.pad).initial = 0 ∧ ((runList m pushes).getD k Push.pad).row = 0 ∧
        (if k < (runList m pushes).length then _ else 0) = 0 ∧ ((runList m pushes).getD k Push.pad).matrix = 0
      rw [List.getD_eq_default (runList m pushes) Push.pad past, if_neg (not_lt.mpr past)]
      exact ⟨rfl, rfl, rfl, rfl⟩
    · show ((runList m pushes).getD k Push.pad).row < 2 ^ key.relation.key.layout.rowVariables
      rcases run k with member | pad
      · exact (runs _ member).2.2.2
      · rw [pad]
        exact Nat.two_pow_pos _
    · show (if k < (runList m pushes).length then _ else 0) < 2 ^ key.relation.key.layout.slotVariables
      split_ifs with inside
      · exact lt_of_lt_of_le (List.idxOf_lt_length_of_mem (keyMem k inside)) (keysLength ▸ slotBits)
      · exact Nat.two_pow_pos _
    · show ((runList m pushes).getD k Push.pad).matrix < 2 ^ key.relation.matrixVars
      rcases run k with member | pad
      · exact lt_of_lt_of_le (runs _ member).1 matrixVars
      · rw [pad]
        exact Nat.two_pow_pos _
    · show (keysOf (runList m pushes)).length ≤ 2 ^ key.relation.key.layout.slotVariables
      rw [keysLength]
      exact slotBits
    · replace inside : k < (runList m pushes).length := inside
      show (if k < (runList m pushes).length then _ else 0) < (keysOf (runList m pushes)).length
      rw [if_pos inside]
      exact List.idxOf_lt_length_of_mem (keyMem k inside)
    · replace inside : k < (runList m pushes).length := inside
      show ((runList m pushes).getD k Push.pad).matrix < key.relation.key.layout.matrices
      rw [matrices]
      exact (runs _ (runMem k inside)).1
    · exact Nat.mod_lt _ (by decide)
    · show (classesOf (runList m pushes)).getD ((keysOf (runList m pushes)).getD σ 0 / 2 ^ 40) 0 / 2 ^ 64 ≤
        ringDegree
      rw [hc, hk]
      exact (size σ).1
    · show ringDegree * (keyStart ((keysOf (runList m pushes)).getD σ 0) / ringDegree) +
          keyStart ((keysOf (runList m pushes)).getD σ 0) % ringDegree +
          (classesOf (runList m pushes)).getD ((keysOf (runList m pushes)).getD σ 0 / 2 ^ 40) 0 / 2 ^ 64 ≤
        ringDegree * key.relation.key.blocks
      rw [hc, hk]
      exact (size σ).2
    · replace lane : l < 64 := lane
      replace pad : key.relation.key.blocks ≤ b ∨ ringDegree ≤ l := pad
      show seedKey seed (poseidon2Columns / ringDegree) i (64 * b + l) = 0
      have low : (64 * b + l) % 64 = l := by omega
      have high : (64 * b + l) / 64 = b := by omega
      rw [seedKey, low, high, if_neg]
      omega
  · refine ⟨?_, fun j k lt lo hi => ?_, keysLength.trans slots, cover, fun i σ lt lo hi => ?_⟩
    · show (runList m pushes).length = key.relation.key.layout.runBounds[key.relation.key.layout.matrices]!
      rw [runList_length, matrices, bound m le_rfl]
    · show ((runList m pushes).getD k Push.pad).matrix = j
      have lt : j < m := by rw [← matrices]; exact lt
      replace lo : (pushes.countP fun p => p.matrix < j) ≤ k := by rw [← bound j lt.le]; exact lo
      replace hi : k < pushes.countP fun p => p.matrix < j + 1 := by rw [← bound (j + 1) lt]; exact hi
      exact runList_matrix pushes lt lo hi
    · have shape := shapes i lt (σ - (structureOf key.relation.key.layout).segmentStart i) (by omega)
      rw [Nat.add_sub_cancel' lo] at shape
      show (classesOf (runList m pushes)).getD ((keysOf (runList m pushes)).getD σ 0 / 2 ^ 40) 0 / 2 ^ 64 = _ ∧
        (((classesOf (runList m pushes)).getD ((keysOf (runList m pushes)).getD σ 0 / 2 ^ 40) 0 % 2 ^ 64 : ℕ) : Gl) =
          _ ∧ keyStart ((keysOf (runList m pushes)).getD σ 0) % ringDegree = _
      rw [hc, word]
      exact shape

end CompressionLean.Checks
