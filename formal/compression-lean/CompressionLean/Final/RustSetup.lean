import CompressionLean.Ports.Setup
import CompressionLean.Final.Parent
import CompressionLean.Production
import CompressionLean.Spec.Parent

/-! Rust test `derived_structure_steps_read_back_the_runs` checks `ReadsBack` of `from_rows`. -/

namespace CompressionLean.Final

open CompressionLean.Spec
open NightstreamFPrime NightstreamFPrime.Spec NightstreamFPrime.Lifecycle
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open CompressionLean.Setup

variable {lw : ℕ} {pf : ringDegree * PaperAlgebra.publicRingColumns ≤ Phi81CarrierLayout.carrierWidth lw}

structure Lists (S : Layer1.Setup Gl) (runs : List Push) : Prop where
  count : S.runs = runs.length
  run : ∀ k, k < runs.length →
    S.runMatrix k = (runs.getD k Push.pad).matrix ∧ S.runRow k = (runs.getD k Push.pad).row ∧
      S.runCoef k = (runs.getD k Push.pad).initial ∧
      ringDegree * S.slotBlock (S.runSlot k) + S.slotLane (S.runSlot k) = (runs.getD k Push.pad).start ∧
      S.slotLength (S.runSlot k) = (runs.getD k Push.pad).len ∧
      ((runs.getD k Push.pad).len = 1 ∨ S.slotRatio (S.runSlot k) = (runs.getD k Push.pad).ratio)

def ReadsBack (S : Layer1.Setup Gl) (pushes : List Push) : Prop :=
  ∃ runs, runs.Perm pushes ∧ Lists S runs

theorem sum_range_getD {M : Type} [AddCommMonoid M] (g : Push → M) :
    ∀ l : List Push, ∑ k ∈ Finset.range l.length, g (l.getD k Push.pad) = (l.map g).sum
  | [] => by simp
  | a :: l => by
      rw [List.length_cons, Finset.sum_range_succ', List.map_cons, List.sum_cons, add_comm]
      simp only [List.getD_cons_succ, List.getD_cons_zero, sum_range_getD g l]

theorem runEntry_eq {S : Layer1.Setup Gl} {pushes : List Push} (back : ReadsBack S pushes) (j i c : ℕ) :
    runEntry S ringDegree j i c = pushedEntry pushes j i c := by
  obtain ⟨runs, perm, lists⟩ := back
  unfold runEntry pushedEntry
  rw [Finset.sum_filter, Finset.sum_filter, lists.count,
    ← (perm.map fun p => if p.matrix = j ∧ p.row = i then p.value c else 0).sum_eq,
    ← sum_range_getD (fun p => if p.matrix = j ∧ p.row = i then p.value c else 0)]
  apply Finset.sum_congr rfl
  intro k member
  obtain ⟨matrix, row, coef, start, length, ratio⟩ := lists.run k (Finset.mem_range.mp member)
  rw [matrix, row, coef, start, length]
  generalize runs.getD k Push.pad = p at ratio ⊢
  unfold Push.value
  by_cases hin : p.start ≤ c ∧ c < p.start + p.len
  · have power : S.slotRatio (S.runSlot k) ^ (c - p.start) = p.ratio ^ (c - p.start) := by
      rcases ratio with one | same
      · rw [show c - p.start = 0 by omega, pow_zero, pow_zero]
      · rw [same]
    by_cases hm : p.matrix = j <;> by_cases hr : p.row = i <;> simp [hm, hr, hin, power]
  · by_cases hm : p.matrix = j <;> by_cases hr : p.row = i <;> simp [hm, hr, hin]

theorem runRow_mem {S : Layer1.Setup Gl} {pushes : List Push} (back : ReadsBack S pushes) {k : ℕ}
    (inside : k < S.runs) : ∃ p ∈ pushes, S.runRow k = p.row := by
  obtain ⟨runs, perm, lists⟩ := back
  rw [lists.count] at inside
  refine ⟨runs.getD k Push.pad, perm.mem_iff.mp ?_, (lists.run k inside).2.1⟩
  rw [List.getD_eq_getElem _ _ inside]
  exact List.getElem_mem inside

theorem mem_sortedSet {l : List ℕ} {x : ℕ} : x ∈ sortedSet l ↔ x ∈ l := by
  rw [sortedSet, List.mem_insertionSort, List.mem_dedup]

theorem runList_perm (pushes : List Push) :
    ∀ m, (runList m pushes).Perm (pushes.filter fun p => p.matrix < m)
  | 0 => by simp [runList]
  | m + 1 => by
      have split : runList (m + 1) pushes = runList m pushes ++ pushes.filter fun p => p.matrix = m := by
        simp [runList, List.range_succ, List.flatMap_append]
      rw [split]
      refine ((runList_perm pushes m).append_right _).trans ?_
      have parts := List.filter_append_perm (fun p : Push => decide (p.matrix < m))
        (pushes.filter fun p => p.matrix < m + 1)
      rw [List.filter_filter, List.filter_filter] at parts
      convert parts using 2
      · exact List.filter_congr fun p _ => by by_cases h : p.matrix < m <;> simp [h]; omega
      · exact List.filter_congr fun p _ => by by_cases h : p.matrix < m <;> simp [h] <;> omega

theorem slotKey_eq {classes : List ℕ} {p : Push} (block : p.start / ringDegree < 2 ^ 32) :
    keyStart (slotKey classes p) = p.start ∧
      keyCode classes (slotKey classes p) = classes.getD (classes.idxOf p.classCode) 0 := by
  have lane : p.start % ringDegree < 2 ^ 8 := (Nat.mod_lt _ (by decide)).trans (by decide)
  have shape : slotKey classes p =
      p.start / ringDegree + (classes.idxOf p.classCode * 2 ^ 8 + p.start % ringDegree) * 2 ^ 32 := by
    unfold slotKey
    ring
  have low : slotKey classes p % 2 ^ 32 = p.start / ringDegree := by
    rw [shape, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt block]
  have high : slotKey classes p / 2 ^ 32 = classes.idxOf p.classCode * 2 ^ 8 + p.start % ringDegree := by
    rw [shape, Nat.add_mul_div_right _ _ (by positivity), Nat.div_eq_of_lt block, zero_add]
  have index : slotKey classes p / 2 ^ 40 = classes.idxOf p.classCode := by
    rw [show (2 : ℕ) ^ 40 = 2 ^ 32 * 2 ^ 8 by norm_num, ← Nat.div_div_eq_div_mul, high]
    simp only [Nat.reducePow] at lane ⊢
    omega
  refine ⟨?_, by rw [keyCode, index]⟩
  rw [keyStart, low, high, Nat.mul_add_mod_of_lt lane]
  exact Nat.div_add_mod' p.start ringDegree

theorem classCode_decode (p : Push) :
    codeLength p.classCode = p.len ∧ (p.len = 1 ∨ codeRatio p.classCode = p.ratio) := by
  have word : p.ratioWord < 2 ^ 64 := by
    unfold Push.ratioWord
    split_ifs
    · norm_num
    · exact (ZMod.val_lt p.ratio).trans (by norm_num [goldilocksModulus])
  refine ⟨by rw [codeLength, Push.classCode]; omega, ?_⟩
  by_cases one : p.len = 1
  · exact Or.inl one
  · right
    rw [codeRatio, Push.classCode, show (p.len * 2 ^ 64 + p.ratioWord) % 2 ^ 64 = p.ratioWord by omega,
      Push.ratioWord, ite_eq_right one, ZMod.natCast_zmod_val]

theorem getD_idxOf {l : List ℕ} {x : ℕ} (member : x ∈ l) : l.getD (l.idxOf x) 0 = x := by
  rw [List.getD_eq_getElem _ _ (List.idxOf_lt_length_of_mem member), List.getElem_idxOf]

theorem lists_setupOfRuns (seed : List ℕ) (blocks : ℕ) (runs : List Push)
    (block : ∀ p ∈ runs, p.start / ringDegree < 2 ^ 32) : Lists (setupOfRuns seed blocks runs) runs := by
  refine ⟨rfl, fun k inside => ?_⟩
  have member : runs.getD k Push.pad ∈ runs := by
    rw [List.getD_eq_getElem _ _ inside]
    exact List.getElem_mem inside
  set p := runs.getD k Push.pad
  have key : (keysOf runs).getD ((keysOf runs).idxOf (slotKey (classesOf runs) p)) 0 = slotKey (classesOf runs) p :=
    getD_idxOf (mem_sortedSet.mpr (List.mem_map.mpr ⟨p, member, rfl⟩))
  have code : (classesOf runs).getD ((classesOf runs).idxOf p.classCode) 0 = p.classCode :=
    getD_idxOf (mem_sortedSet.mpr (List.mem_map.mpr ⟨p, member, rfl⟩))
  obtain ⟨start, index⟩ := slotKey_eq (classes := classesOf runs) (block p member)
  obtain ⟨length, ratio⟩ := classCode_decode p
  simp only [setupOfRuns, ite_eq_left inside]
  refine ⟨rfl, rfl, rfl, ?_, ?_, ?_⟩
  · rw [key, start]
    exact Nat.div_add_mod p.start ringDegree
  · rwa [key, index, code]
  · rwa [key, index, code]

theorem readsBack_of_build {seed : List ℕ} {columns matrices : ℕ} {pushes : List Push} {S : Layer1.Setup Gl}
    (built : build seed columns matrices pushes = some S) :
    ReadsBack S pushes ∧ S.key = seedKey seed (columns / ringDegree) := by
  unfold build at built
  split_ifs at built with accepts
  obtain rfl := Option.some.inj built.symm
  obtain ⟨-, -, -, few, runs⟩ := accepts
  have perm : (runList matrices pushes).Perm pushes := by
    have := runList_perm pushes matrices
    rwa [List.filter_eq_self.mpr fun p member => by simpa using (runs p member).1] at this
  refine ⟨⟨_, perm, lists_setupOfRuns seed _ _ fun p member => ?_⟩, rfl⟩
  have inside := (runs p (perm.mem_iff.mp member)).2.2
  have : p.start / ringDegree ≤ columns / ringDegree := Nat.div_le_div_right (by omega)
  simp only [Nat.reducePow]
  omega

theorem gl_natCast (n : ℕ) (small : n < goldilocksModulus) : ((n : ℕ) : Gl) = gl ⟨n, small⟩ := by
  apply ZMod.val_injective
  rw [ZMod.val_natCast, Nat.mod_eq_of_lt small]
  rfl

theorem build_honest (s : Layer1.Shape) (D : s.D = ringDegree) (lanes : s.laneVars = 6)
    (setup : AjtaiSetupV1.Setup productionProfile.commitmentWidth
      (Phi81ColumnLayout.blockCount (Phi81CarrierLayout.carrierWidth lw)))
    (relation : ProductionKey.LogicalRelation lw pf) (pushes : List Push) {S : Layer1.Setup Gl}
    (built : build setup.seed.bytes (Phi81CarrierLayout.carrierWidth lw) productionProfile.ccsMatrices pushes = some S)
    (gives : RunsGive pushes relation) :
    HonestSetup s S relation setup.verifierKey := by
  obtain ⟨back, key⟩ := readsBack_of_build built
  refine ⟨fun i b l => ?_, fun k inside => ?_, fun j i c inside => ?_⟩
  · have blocks : Phi81CarrierLayout.carrierWidth lw / ringDegree =
        Phi81ColumnLayout.blockCount (Phi81CarrierLayout.carrierWidth lw) := by
      rw [Phi81CarrierLayout.blockCount_carrierWidth, Phi81CarrierLayout.carrierWidth,
        Nat.mul_div_cancel _ (by decide)]
    have lane := l.isLt
    have low : (64 * b.val + l.val) % 64 = l.val := by simp only [ringDegree] at lane; omega
    have high : (64 * b.val + l.val) / 64 = b.val := by simp only [ringDegree] at lane; omega
    rw [key, lanes, seedKey, show (2 : ℕ) ^ 6 = 64 from rfl, low, high, blocks, ite_eq_left ⟨lane, b.isLt⟩]
    exact gl_natCast _ _
  · obtain ⟨p, member, row⟩ := runRow_mem back inside
    rw [row]
    exact gives.row_lt p member
  · rw [D, runEntry_eq back]
    exact gives.entry j i c inside

/-- Rust tests `relation_shape_has_the_lean_constants` and `parent_relation_arguments_are_the_lean_constants` pin these numbers. -/
theorem parentShape_of (s : Layer1.Shape) (lw : ℕ) (D : s.D = ringDegree) (lanes : s.laneVars = 6)
    (kappa : s.kappa = Production.kappa) (matrices : s.matrices = Production.matrices)
    (publicBlocks : s.publicBlocks = Production.publicBlocks) (blocks : s.blocks = Phi81ColumnLayout.blockCount lw)
    (norm : s.normBound = Production.normBound) : ParentShape s lw where
  D := D
  laneVars := lanes
  kappa := kappa
  matrices := matrices
  publicBlocks := publicBlocks
  blocks := by rw [blocks, Phi81CarrierLayout.blockCount_carrierWidth]
  norm := by rw [norm]; decide

end CompressionLean.Final
