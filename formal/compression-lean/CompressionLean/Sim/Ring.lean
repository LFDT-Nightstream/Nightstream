import CompressionLean.Sim.Sumcheck
import CompressionLean.Ports.Circuit.Ring
import CompressionLean.Ports.Circuit.Mle
import CompressionLean.Fields

namespace CompressionLean.Sim

open CompressionLean.Circuit
open CompressionLean.Lib
open CompressionLean.Replay (Word Ext Kx Relation D)
open NightstreamFPrime.Spec (goldilocksModulus)
open Polynomial

theorem ext_add_zero (a : Ext) : a + 0 = a :=
  Ext.eq_of_toK (K := Cubic) (by rw [Ext.toK_add, Ext.toK_zero, add_zero])

theorem ext_add_assoc (a b d : Ext) : a + b + d = a + (b + d) :=
  Ext.eq_of_toK (K := Cubic) (by rw [Ext.toK_add, Ext.toK_add, Ext.toK_add, Ext.toK_add, add_assoc])

theorem ext_scale_zero (a : Ext) : a.scale 0 = 0 :=
  Ext.eq_of_toK (K := Cubic) (by rw [Ext.toK_scale, Ext.toK_zero, Replay.Word.toF_zero, map_zero, mul_zero])

private theorem ext_isZero {a : Ext} (h : a ≠ 0) : a.isZero = false := by
  cases e : a.isZero
  · rfl
  · exact absurd (Ext.eq_of_beq e) h

theorem sumRange_zero (f : ℕ → Ext) : Replay.sumRange 0 f = 0 := rfl

theorem map_range_succ' {α : Type} (f : ℕ → α) (k : ℕ) :
    (Array.range (k + 1)).map f = #[f 0] ++ (Array.range k).map fun i => f (i + 1) := by
  apply Array.ext'
  simp [List.range_succ_eq_map]

theorem forall₂_range {α : Type} [Inhabited α] {R : ℕ → α → Prop} {k : ℕ} {xs : Array α}
    (h : List.Forall₂ R (Array.range k).toList xs.toList) : xs.size = k ∧ ∀ i < k, R i xs[i]! := by
  have size := h.length_eq
  simp only [Array.length_toList, Array.size_range] at size
  refine ⟨size.symm, fun i hi => ?_⟩
  have := (List.forall₂_iff_get.mp h).2 i (by simpa using hi) (by simpa [← size] using hi)
  rw [getElem!_pos xs i (by omega)]
  simpa using this

variable {F : Type} [CommRing F]

def IsK (c : Ctx F) (n : ℕ) (a : KVar) (g : Word × Word) : Prop := IsWord c n a.1 g.1 ∧ IsWord c n a.2 g.2

theorem IsK.mono {c : Ctx F} {n m : ℕ} {a : KVar} {g : Word × Word} (h : IsK c n a g) (le : n ≤ m := by omega) :
    IsK c m a g :=
  ⟨h.1.mono le, h.2.mono le⟩

def IsKs (c : Ctx F) (n : ℕ) (xs : Array KVar) (gs : Array (Word × Word)) : Prop :=
  xs.size = gs.size ∧ ∀ i < xs.size, IsK c n xs[i]! gs[i]!

namespace IsKs

variable {c : Ctx F} {n : ℕ} {xs : Array KVar} {gs : Array (Word × Word)}

theorem mono (h : IsKs c n xs gs) {m : ℕ} (le : n ≤ m := by omega) : IsKs c m xs gs :=
  ⟨h.1, fun i hi => (h.2 i hi).mono le⟩

theorem get (h : IsKs c n xs gs) {i : ℕ} (hi : i < gs.size) : IsK c n xs[i]! gs[i]! :=
  h.2 i (h.1 ▸ hi)

theorem extract (h : IsKs c n xs gs) (a : ℕ) : IsKs c n (xs.extract a) (gs.extract a) := by
  refine ⟨by simp [h.1], fun i hi => ?_⟩
  rw [Array.size_extract] at hi
  rw [getElem!_pos _ _ (by simp; omega), getElem!_pos _ _ (by simp [← h.1]; omega), Array.getElem_extract,
    Array.getElem_extract, ← getElem!_pos, ← getElem!_pos]
  exact h.2 _ (by omega)

end IsKs

structure IsClaim (c : Ctx F) (n : ℕ) (v : ClaimView) (st : Replay.Statement) : Prop where
  commitment : v.commitment.size = st.commitment.size ∧
    ∀ i < v.commitment.size, IsWords c n v.commitment[i]! st.commitment[i]!
  publicInput : v.publicInput.size = st.publicInput.size ∧
    ∀ i < v.publicInput.size, IsWords c n v.publicInput[i]! st.publicInput[i]!
  point : IsKs c n v.point st.point
  evalK : IsKs c n v.evalK st.evalK
  evalA : v.evalA.size = st.evalA.size ∧ ∀ j < v.evalA.size, IsKs c n v.evalA[j]! st.evalA[j]!

theorem IsClaim.mono {c : Ctx F} {n m : ℕ} {v : ClaimView} {st : Replay.Statement} (h : IsClaim c n v st)
    (le : n ≤ m := by omega) : IsClaim c m v st :=
  ⟨⟨h.commitment.1, fun i hi => (h.commitment.2 i hi).mono le⟩,
    ⟨h.publicInput.1, fun i hi => (h.publicInput.2 i hi).mono le⟩, h.point.mono le, h.evalK.mono le,
    ⟨h.evalA.1, fun j hj => (h.evalA.2 j hj).mono le⟩⟩

def IsMixing (c : Ctx F) (n : ℕ) (v : MixingVar) (m : Replay.Mixing) : Prop :=
  v.kappa = m.kappa ∧ v.matrices = m.matrices ∧ ∀ i < v.powers.size, IsExt c n v.powers[i]! (m.lam.pow i)

namespace IsMixing

variable {c : Ctx F} {n : ℕ} {v : MixingVar} {m : Replay.Mixing}

theorem mono (h : IsMixing c n v m) {n' : ℕ} (le : n ≤ n' := by omega) : IsMixing c n' v m :=
  ⟨h.1, h.2.1, fun i hi => (h.2.2 i hi).mono le⟩

theorem commitment (h : IsMixing c n v m) {row : ℕ} (hr : row < v.powers.size) :
    IsExt c n (v.commitment row) (m.commitment row) :=
  h.2.2 row hr

theorem evalK (h : IsMixing c n v m) (hr : m.kappa + 1 < v.powers.size) :
    IsExt c n v.evalK.1 m.evalK.1 ∧ IsExt c n v.evalK.2 m.evalK.2 := by
  simp only [MixingVar.evalK, Replay.Mixing.evalK, h.1]
  exact ⟨h.2.2 _ (by omega), h.2.2 _ hr⟩

theorem evalA (h : IsMixing c n v m) {j : ℕ} (hr : m.kappa + 3 + 2 * j < v.powers.size) :
    IsExt c n (v.evalA j).1 (m.evalA j).1 ∧ IsExt c n (v.evalA j).2 (m.evalA j).2 := by
  simp only [MixingVar.evalA, Replay.Mixing.evalA, h.1]
  exact ⟨h.2.2 _ (by omega), h.2.2 _ hr⟩

theorem «public» (h : IsMixing c n v m) {block : ℕ} (hr : m.kappa + 2 + 2 * m.matrices + block < v.powers.size) :
    IsExt c n (v.public block) (m.public block) := by
  simp only [MixingVar.«public», Replay.Mixing.«public», h.1, h.2.1]
  exact h.2.2 _ hr

end IsMixing

namespace IsExts

variable {c : Ctx F} {n : ℕ} {xs : Array ExtVar} {gs : Array Ext}

theorem map_range_iff {k : ℕ} {g : ℕ → Ext} :
    IsExts c n xs ((Array.range k).map g) ↔ xs.size = k ∧ ∀ i < k, IsExt c n xs[i]! (g i) := by
  refine ⟨fun h => ⟨by simpa using h.1, fun i hi => ?_⟩, fun h => ⟨by simpa using h.1, fun i hi => ?_⟩⟩
  · have size : xs.size = k := by simpa using h.1
    have := h.2 i (size ▸ hi)
    rwa [getElem!_map_range _ _ hi] at this
  · rw [getElem!_map_range _ _ (h.1 ▸ hi)]
    exact h.2 i (h.1 ▸ hi)

end IsExts

instance MixingVar.new_appends (lambda : ExtVar) (rel : Relation) : Appends (MixingVar.new lambda rel) := by
  unfold MixingVar.new; appends
instance Ring.project_appends (value : KVar) (weights : ExtVar × ExtVar) : Appends (Ring.project value weights) := by
  unfold Ring.project; appends
instance Ring.targets_appends (statement : ClaimView) (m : MixingVar) : Appends (Ring.targets statement m) := by
  unfold Ring.targets; appends
instance Ring.liftedTarget_appends (target quotient : Array ExtVar) (zeta : ExtVar) :
    Appends (Ring.liftedTarget target quotient zeta) := by
  unfold Ring.liftedTarget; appends
instance Ring.tau_appends (zeta : ExtVar) : Appends (Ring.tau zeta) := by unfold Ring.tau; appends
instance Mle.scaledEqPoint_appends (cv : ExtVar) (bits : ℕ) : Appends (Mle.scaledEqPoint cv bits) := by
  unfold Mle.scaledEqPoint; appends

section Rows

variable [CharP F goldilocksModulus] {c : Ctx F} {n : ℕ}

theorem MixingVar.new_ensures {lambda : ExtVar} {lam : Ext} (rel : Relation) (hl : IsExt c n lambda lam) :
    Ensures c n (MixingVar.new lambda rel)
      fun v m => IsMixing c m v ⟨lam, rel.kappa, rel.matrices⟩ ∧ v.powers.size = rel.rowCount := by
  unfold MixingVar.new
  refine extOne_ensures.bind fun one _ _ hone => ?_
  dsimp only
  refine (Ensures.loopRange (fun k acc m => IsExts c m acc ((Array.range (k + 1)).map lam.pow)) ?_
    fun k _ acc m _ hacc => ?_).bind fun powers _ _ hpowers => Ensures.pure ?_
  · refine ⟨by simp, fun i hi => ?_⟩
    obtain rfl : i = 0 := by simpa using hi
    simpa [ext_pow_zero] using hone
  · have back : IsExt c m acc.back! (lam.pow k) := by
      obtain ⟨size, get⟩ := IsExts.map_range_iff.mp hacc
      rw [Array.back!, size, Nat.add_sub_cancel]
      exact get k (by omega)
    refine (extMul_ensures back hl.mono).bind fun x _ _ hx => Ensures.pure ⟨rfl, ?_⟩
    rw [map_range_succ, ext_pow_succ]
    exact hacc.mono.push hx
  · have rows : 2 ≤ rel.rowCount := by simp only [Relation.rowCount]; omega
    obtain ⟨size, get⟩ := IsExts.map_range_iff.mp hpowers
    rw [show rel.rowCount - 1 + 1 = rel.rowCount by omega] at size get
    exact ⟨⟨rfl, rfl, fun i hi => get i (size ▸ hi)⟩, size⟩

theorem Ring.project_ensures {value : KVar} {gv : Word × Word} {weights : ExtVar × ExtVar} {gw : Ext × Ext}
    (hv : IsK c n value gv) (hw : IsExt c n weights.1 gw.1 ∧ IsExt c n weights.2 gw.2) :
    Ensures c n (Ring.project value weights) fun r m => IsExt c m r (Replay.project gv gw) :=
  (extScale_ensures hw.1 hv.1).bind fun _ _ _ hre => (extScale_ensures hw.2.mono hv.2.mono).bind
    fun _ _ _ him => extAdd_ensures hre.mono him

theorem Ring.targets_ensures {v : ClaimView} {st : Replay.Statement} {mv : MixingVar} {lam : Ext} {rel : Relation}
    (fits : st.fits rel = true) (hv : IsClaim c n v st) (hm : IsMixing c n mv ⟨lam, rel.kappa, rel.matrices⟩)
    (hsize : mv.powers.size = rel.rowCount) :
    Ensures c n (Ring.targets v mv) fun r m => IsExts c m r (Replay.targets st ⟨lam, rel.kappa, rel.matrices⟩) := by
  have sizes := fits
  simp only [Replay.Statement.fits, Bool.and_eq_true, beq_iff_eq, Array.all_eq_true] at sizes
  obtain ⟨⟨⟨⟨⟨⟨⟨comSize, comRows⟩, pubSize⟩, pubRows⟩, -⟩, evalKSize⟩, evalASize⟩, evalARows⟩ :=
    sizes
  have rows : rel.rowCount = rel.kappa + 2 + 2 * rel.matrices + rel.publicBlocks := rfl
  set mix : Replay.Mixing := ⟨lam, rel.kappa, rel.matrices⟩ with hmix
  have mk : mix.kappa = rel.kappa := rfl
  have mm : mix.matrices = rel.matrices := rfl
  obtain ⟨T, hT⟩ : ∃ T : ℕ → Ext, T = fun lane =>
      Replay.sumRange st.commitment.size (fun row => (mix.commitment row).scale (st.commitment[row]!)[lane]!) +
        Replay.project st.evalK[lane]! mix.evalK +
        Replay.sumRange st.evalA.size (fun j => Replay.project (st.evalA[j]!)[lane]! (mix.evalA j)) +
        Replay.sumRange st.publicInput.size (fun block => (mix.public block).scale (st.publicInput[block]!)[lane]!) :=
    ⟨_, rfl⟩
  have target : Replay.targets st mix = (Array.range D).map T := by rw [hT]; rfl
  rw [target]
  unfold Ring.targets
  refine (Ensures.mapM (fun lane a m => IsExt c m a (T lane)) (fun _ _ _ _ h le => h.mono le)
    fun lane hlane m _ => ?_).post fun res m h => IsExts.map_range_iff.mpr (forall₂_range h)
  have hl : lane < D := by simpa using hlane
  rw [hT]
  dsimp only
  refine extZero_ensures.bind fun total _ _ hzero => ?_
  refine (Ensures.loopArray (fun k acc m' => IsExt c m' acc
      (Replay.sumRange k fun row => (mix.commitment row).scale (st.commitment[row]!)[lane]!)) hzero
    fun k hk acc m' _ hacc => ?_).bind fun total₁ _ _ htotal₁ => ?_
  · have hk' : k < v.commitment.size := by simpa using hk
    have hc := hv.commitment.1
    rw [show v.commitment.zipIdx[k] = (v.commitment[k]!, k) by simp [getElem!_pos, hk']]
    dsimp only
    have row := comRows k (by omega)
    rw [← getElem!_pos st.commitment k (by omega)] at row
    refine (extScale_ensures (ga := mix.commitment k) (gs := (st.commitment[k]!)[lane]!)
      (hm.commitment (by rw [hsize, rows]; omega)).mono
      ((hv.commitment.2 k hk').get (by rw [row]; exact hl)).mono).bind fun term _ _ hterm => ?_
    exact (extAdd_ensures hacc.mono hterm).bind fun total' _ _ htotal =>
      Ensures.pure ⟨rfl, by rw [sumRange_succ]; exact htotal⟩
  rw [Array.size_zipIdx, hv.commitment.1] at htotal₁
  refine (Ring.project_ensures (gv := st.evalK[lane]!) (gw := mix.evalK)
    (hv.evalK.2 lane (by rw [hv.evalK.1, evalKSize]; exact hl)).mono
    ((hm.evalK (by rw [hsize, rows]; omega)).imp (·.mono) (·.mono))).bind fun term _ _ hterm => ?_
  refine (extAdd_ensures htotal₁.mono hterm).bind fun total₂ _ _ htotal₂ => ?_
  refine (Ensures.loopArray (fun k acc m' => IsExt c m' acc
      (Replay.sumRange st.commitment.size (fun row => (mix.commitment row).scale (st.commitment[row]!)[lane]!) +
        Replay.project st.evalK[lane]! mix.evalK +
        Replay.sumRange k fun j => Replay.project (st.evalA[j]!)[lane]! (mix.evalA j)))
    (by rw [sumRange_zero, ext_add_zero]; exact htotal₂) fun k hk acc m' _ hacc => ?_).bind
    fun total₃ _ _ htotal₃ => ?_
  · have hk' : k < v.evalA.size := by simpa using hk
    have ha := hv.evalA.1
    rw [show v.evalA.zipIdx[k] = (v.evalA[k]!, k) by simp [getElem!_pos, hk']]
    dsimp only
    have row := evalARows k (by omega)
    rw [← getElem!_pos st.evalA k (by omega)] at row
    refine (Ring.project_ensures (gv := (st.evalA[k]!)[lane]!) (gw := mix.evalA k)
      ((hv.evalA.2 k hk').2 lane (by rw [(hv.evalA.2 k hk').1, row]; exact hl)).mono
      ((hm.evalA (by rw [hsize, rows]; omega)).imp (·.mono) (·.mono))).bind fun term _ _ hterm => ?_
    exact (extAdd_ensures hacc.mono hterm).bind fun total' _ _ htotal =>
      Ensures.pure ⟨rfl, by rw [sumRange_succ, ← ext_add_assoc]; exact htotal⟩
  rw [Array.size_zipIdx, hv.evalA.1] at htotal₃
  refine (Ensures.loopArray (fun k acc m' => IsExt c m' acc
      (Replay.sumRange st.commitment.size (fun row => (mix.commitment row).scale (st.commitment[row]!)[lane]!) +
        Replay.project st.evalK[lane]! mix.evalK +
        Replay.sumRange st.evalA.size (fun j => Replay.project (st.evalA[j]!)[lane]! (mix.evalA j)) +
        Replay.sumRange k fun block => (mix.public block).scale (st.publicInput[block]!)[lane]!))
    (by rw [sumRange_zero, ext_add_zero]; exact htotal₃) fun k hk acc m' _ hacc => ?_).bind
    fun total₄ _ _ htotal₄ => Ensures.pure ?_
  · have hk' : k < v.publicInput.size := by simpa using hk
    have hp := hv.publicInput.1
    rw [show v.publicInput.zipIdx[k] = (v.publicInput[k]!, k) by simp [getElem!_pos, hk']]
    dsimp only
    have row := pubRows k (by omega)
    rw [← getElem!_pos st.publicInput k (by omega)] at row
    refine (extScale_ensures (ga := mix.public k) (gs := (st.publicInput[k]!)[lane]!)
      (hm.public (by rw [hsize, rows]; omega)).mono
      ((hv.publicInput.2 k hk').get (by rw [row]; exact hl)).mono).bind fun term _ _ hterm => ?_
    exact (extAdd_ensures hacc.mono hterm).bind fun total' _ _ htotal =>
      Ensures.pure ⟨rfl, by rw [sumRange_succ, ← ext_add_assoc]; exact htotal⟩
  rwa [Array.size_zipIdx, hv.publicInput.1] at htotal₄

theorem Ring.liftedTarget_ensures {target quotient : Array ExtVar} {zeta : ExtVar} {gt gq : Array Ext} {gz : Ext}
    (ht : IsExts c n target gt) (hq : IsExts c n quotient gq) (hz : IsExt c n zeta gz) :
    Ensures c n (Ring.liftedTarget target quotient zeta) fun r m => IsExt c m r (Replay.liftedTarget gt gq gz) := by
  unfold Ring.liftedTarget Replay.liftedTarget
  dsimp only
  refine (Ensures.loopRange (fun k acc m => IsExt c m acc (gz.pow (k + 1))) ?_ fun k _ acc m _ hacc => ?_).bind
    fun middle _ _ hmiddle => ?_
  · rw [ext_pow_succ, ext_pow_zero, ext_one_mul]
    exact hz
  · exact (extMul_ensures hacc hz.mono).bind fun x _ _ hx =>
      Ensures.pure ⟨rfl, by rw [ext_pow_succ _ (k + 1)]; exact hx⟩
  refine (extMul_ensures hmiddle hmiddle).bind fun top _ _ htop => ?_
  refine extOne_ensures.bind fun one _ _ hone => ?_
  refine (extAdd_ensures htop.mono hmiddle.mono).bind fun sum _ _ hsum => ?_
  refine (extAdd_ensures hsum hone.mono).bind fun phi _ _ hphi => ?_
  refine (horner_ensures ht.mono hz.mono).bind fun target' _ _ htarget => ?_
  refine (horner_ensures hq.mono hz.mono).bind fun quotient' _ _ hquotient => ?_
  refine (extMul_ensures hquotient hphi.mono).bind fun lifted _ _ hlifted => ?_
  refine (extAdd_ensures htarget.mono hlifted).post fun _ _ h => ?_
  rwa [show (27 - 1 + 1 : ℕ) = 27 from rfl] at h

theorem Ring.tau_ensures {zeta : ExtVar} {gz : Ext} (hz : IsExt c n zeta gz) :
    Ensures c n (Ring.tau zeta) fun r m => IsExts c m r (Replay.tau gz) := by
  obtain ⟨S, hS⟩ : ∃ S : ℕ → ℕ → Ext,
      S = fun p l => Replay.sumRange p fun q => (gz.pow q).scale (Replay.bar q l) := ⟨_, rfl⟩
  have tau : Replay.tau gz = (Array.range D).map (S D) := by rw [hS]; rfl
  rw [tau]
  unfold Ring.tau
  refine extZero_ensures.bind fun zero _ _ hzero => ?_
  dsimp only
  refine extOne_ensures.bind fun one _ _ hone => ?_
  refine (Ensures.loopRange (fun p acc m => IsExts c m acc.1 ((Array.range D).map (S p)) ∧
      IsExt c m acc.2 (gz.pow p)) ⟨?_, by rw [ext_pow_zero]; exact hone⟩ fun p hp acc m _ hacc => ?_).bind
    fun res _ _ hres => Ensures.pure hres.1
  · refine IsExts.map_range_iff.mpr ⟨by simp, fun i hi => ?_⟩
    rw [getElem!_pos _ i (by simpa using hi), Array.getElem_replicate, hS]
    exact hzero.mono
  obtain ⟨htau, hpower⟩ := hacc
  rw [Nat.zero_add]
  refine (Ensures.loopRange (fun col acc' m' => IsExts c m' acc'
      ((Array.range D).map fun l => if l < col then S (p + 1) l else S p l)) ?_
    fun col hcol acc' m' _ hacc' => ?_).bind fun tau' m₄ _ htau' => ?_
  · exact IsExts.map_range_iff.mpr ⟨(IsExts.map_range_iff.mp htau).1, fun i hi => by
      rw [ite_eq_right (Nat.not_lt_zero _)]
      exact (IsExts.map_range_iff.mp htau).2 i hi⟩
  · rw [Nat.zero_add]
    have hc : col < D := by simpa using hcol
    have succ : S (p + 1) col = S p col + (gz.pow p).scale (Replay.bar p col) := by
      rw [hS]
      exact sumRange_succ _ _
    split
    · rename_i zero
      refine Ensures.pure ⟨rfl, ?_⟩
      obtain ⟨size, get⟩ := IsExts.map_range_iff.mp hacc'
      refine IsExts.map_range_iff.mpr ⟨size, fun i hi => ?_⟩
      have e := get i hi
      by_cases ic : i = col
      · subst ic
        rw [ite_eq_right (Nat.lt_irrefl _)] at e
        rw [ite_eq_left (Nat.lt_succ_self _), succ, Replay.Word.eq_of_beq zero, ext_scale_zero, ext_add_zero]
        exact e
      · by_cases lt : i < col
        · rw [ite_eq_left lt, ite_eq_left (by omega)] at *; assumption
        · rw [ite_eq_right lt, ite_eq_right (by omega)] at *; assumption
    · have entry : IsExt c m' acc'[col]! (S p col) := by
        have e := (IsExts.map_range_iff.mp hacc').2 col hc
        rwa [ite_eq_right (Nat.lt_irrefl _)] at e
      refine (extScaleConstant_ensures (Replay.bar p col) hpower.mono).bind fun term _ _ hterm => ?_
      refine (extAdd_ensures entry.mono hterm).bind fun x m₆ _ hx => Ensures.pure ⟨rfl, ?_⟩
      obtain ⟨size, get⟩ := IsExts.map_range_iff.mp (hacc'.mono (m := m₆))
      rw [← succ] at hx
      show IsExts c m₆ (acc'.set! col x) _
      refine IsExts.map_range_iff.mpr ⟨by simp [size], fun i hi => ?_⟩
      rw [getElem!_set! _ _ _ _ (by simp [size]; exact hi)]
      by_cases ic : col = i
      · subst ic
        rw [ite_eq_left rfl, ite_eq_left (Nat.lt_succ_self _)]
        exact hx
      · rw [ite_eq_right ic]
        have e := get i hi
        by_cases lt : i < col
        · rw [ite_eq_left lt, ite_eq_left (by omega)] at *; assumption
        · rw [ite_eq_right lt, ite_eq_right (by omega)] at *; assumption
  · refine (extMul_ensures hpower.mono hz.mono).bind fun power m₅ _ hpower' =>
      Ensures.pure ⟨rfl, ?_, by rw [ext_pow_succ]; exact hpower'⟩
    obtain ⟨size, get⟩ := IsExts.map_range_iff.mp (htau'.mono (m := m₅))
    refine IsExts.map_range_iff.mpr ⟨size, fun i hi => ?_⟩
    have e := get i hi
    rwa [ite_eq_left (show i < D - 0 from hi)] at e

end Rows

private theorem scaledEqPoint_eq (bits : ℕ) : ∀ sq : ℕ → Ext, (∀ t, sq (t + 1) = sq t * sq t) →
    (∀ t < bits, 1 + sq t ≠ 0) →
      Replay.scaledEqPoint (sq 0) bits =
        .ok (Replay.prodRange bits fun t => 1 + sq t, (Array.range bits).map fun t => sq t * (1 + sq t).inv) := by
  induction bits with
  | zero =>
      intro sq _ _
      show Except.ok (1, #[]) = Except.ok (1, _)
      congr 2
      apply Array.ext'
      simp
  | succ bits ih =>
      intro sq hsq poles
      have p0 : (1 + sq 0).isZero = false := ext_isZero (poles 0 (Nat.succ_pos _))
      have rest := ih (fun t => sq (t + 1)) (fun t => hsq (t + 1)) fun t ht => poles (t + 1) (by omega)
      rw [Nat.zero_add, hsq 0] at rest
      rw [Replay.scaledEqPoint]
      simp only [p0, Bool.false_eq_true, ite_false, rest, bind_pure_comp, Except.map_ok, prodRange_succ',
        map_range_succ']

section ScaledEqPoint

variable [CharP F goldilocksModulus] {c : Ctx F} {n : ℕ}

theorem Mle.scaledEqPoint_ensures {cv : ExtVar} {gc : Ext} (bits : ℕ) (hc : IsExt c n cv gc) :
    Ensures c n (Mle.scaledEqPoint cv bits)
      fun res m => ∃ kappa point, Replay.scaledEqPoint gc bits = .ok (kappa, point) ∧
        IsExt c m res.1 kappa ∧ IsExts c m res.2 point := by
  obtain ⟨sq, hsq0, hsq⟩ : ∃ sq : ℕ → Ext, sq 0 = gc ∧ ∀ t, sq (t + 1) = sq t * sq t :=
    ⟨fun t => (fun x : Ext => x * x)^[t] gc, rfl, fun t => Function.iterate_succ_apply' _ _ _⟩
  unfold Mle.scaledEqPoint
  refine extOne_ensures.bind fun one _ _ hone => ?_
  dsimp only
  refine (Ensures.loopRange (fun k acc m => IsExt c m acc.1 (Replay.prodRange k fun t => 1 + sq t) ∧
      IsExts c m acc.2.1 ((Array.range k).map fun t => sq t * (1 + sq t).inv) ∧ IsExt c m acc.2.2 (sq k) ∧
      ∀ t < k, 1 + sq t ≠ 0)
    ⟨by rw [prodRange_zero]; exact hone, IsExts.map_range_iff.mpr ⟨rfl, fun i hi => absurd hi (by omega)⟩,
      hsq0 ▸ hc.mono, fun t ht => absurd ht (by omega)⟩ fun k _ acc m _ hacc => ?_).bind
    fun res _ _ hres => Ensures.pure ?_
  · obtain ⟨hkappa, hpoint, hpower, poles⟩ := hacc
    refine (extAdd_ensures hone.mono hpower).bind fun d _ _ hd => ?_
    refine (extMul_ensures hkappa.mono hd).bind fun kappa _ _ hkappa' => ?_
    refine (extInv_ensures hd.mono).bind fun inv _ _ hinv => ?_
    refine (extMul_ensures hpower.mono hinv.2).bind fun x _ _ hx => ?_
    refine (extMul_ensures hpower.mono hpower.mono).bind fun power _ _ hpower' =>
      Ensures.pure ⟨rfl, ?_, ?_, ?_, ?_⟩
    · rw [prodRange_succ]
      exact hkappa'.mono
    · rw [map_range_succ]
      exact hpoint.mono.push hx.mono
    · rw [hsq]
      exact hpower'
    · intro t ht
      rcases Nat.lt_succ_iff_lt_or_eq.mp ht with lt | rfl
      · exact poles t lt
      · exact hinv.1
  · rw [Nat.sub_zero] at hres
    exact ⟨_, _, by rw [← hsq0]; exact scaledEqPoint_eq bits sq hsq hres.2.2.2, hres.1, hres.2.1⟩

end ScaledEqPoint

end CompressionLean.Sim
