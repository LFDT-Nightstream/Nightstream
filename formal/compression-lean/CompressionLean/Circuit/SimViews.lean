import CompressionLean.Circuit.SimMerkle
import CompressionLean.Circuit.Verifier
import CompressionLean.Replay.Stages

/-!
Owns the decoding of a layer-1 proof view (`ProofView`) to the replay's proof
(`Replay.Proof`), and the simulation of the proof reads of `Views.lean` and
`Verifier.lean` (`ProofView.read`): the handles hold the words of the decoded
proof, and the replay's shape check (`Replay.readProof`) accepts it.

- `Decodes c w`: the value of each handle is the word that `w` gives it.
- `IsGkr`, `IsLeaf`, `IsOpening`, `IsProof`: the handles of a view are the
  words of a replay value, field by field. `Replay.Opening` has no WHIR query
  rows, so `IsOpening` gives each row the words of `w` (`IsRows`).
- The shapes come from the reads. The only premise is that each GKR tree has
  a depth and at most two factors, which `readGkr` checks. For
  `ProofView.read` it follows from three facts about the relation.
-/

namespace CompressionLean.Circuit

open CompressionLean.Replay (Word Ext Relation TreeShape WhirShape foldVars layerCount layerFactors layerDegree
  activeTrees partFactors layerShapeOk)

variable {F : Type} [CommRing F]

/-- The value of each handle is the word that `w` gives it. -/
def Decodes (c : Ctx F) (w : Var → Word) : Prop := ∀ h, c.val h = word (w h)

/-- The extension value of three handles. -/
def decodeExt (w : Var → Word) (e : ExtVar) : Ext := ⟨w (e 0), w (e 1), w (e 2)⟩

/-- The four words of a digest. -/
def decodeDigest (w : Var → Word) (d : DigestVar) : Array Word := #[w (d 0), w (d 1), w (d 2), w (d 3)]

/-- A pair of extension values. -/
def decodePair (w : Var → Word) (p : ExtVar × ExtVar) : Ext × Ext := (decodeExt w p.1, decodeExt w p.2)

/-- One GKR layer. -/
def GkrLayerView.decode (w : Var → Word) (l : GkrLayerView) : Replay.GkrLayer :=
  { rounds := l.rounds.map (·.map (decodeExt w)), children := l.children.map (·.map (decodeExt w)) }

/-- A GKR proof. -/
def GkrView.decode (w : Var → Word) (v : GkrView) : Replay.GkrProof :=
  { roots := v.roots.map (decodePair w), layers := v.layers.map (·.decode w) }

/-- A queried setup leaf. -/
def LeafView.decode (w : Var → Word) (v : LeafView) : Replay.Leaf :=
  { values := v.values.map w, path := v.path.map (decodeDigest w) }

/-- One WHIR round, without its query rows. -/
def RoundView.decode (w : Var → Word) (r : RoundView) : Replay.WhirRound :=
  { root := decodeDigest w r.root, ood := r.ood.map (decodeExt w), sumcheck := r.sumcheck.map (decodePair w) }

/-- One WHIR opening, without its query rows. -/
def OpeningView.decode (w : Var → Word) (v : OpeningView) : Replay.Opening :=
  { initialOod := v.initialOod.map (decodeExt w), evals := v.evals.map (·.map (decodeExt w)),
    initialSumcheck := v.initialSumcheck.map (decodePair w), rounds := v.rounds.map (·.decode w),
    finalPoly := v.finalPoly.map (decodeExt w), finalSumcheck := v.finalSumcheck.map (decodePair w) }

/-- A layer-1 proof. The run values are `e_re`, `e_im`, `row`, the order of
`observe_early` and of the replay's `earlyStage`. -/
def ProofView.decode (w : Var → Word) (v : ProofView) : Replay.Proof :=
  { p0 := decodeDigest w v.p0, histogram := v.histogram.map w, early := v.early.decode w,
    earlyValues :=
      { eRe := decodeExt w v.earlyValues.runs[0]!, eIm := decodeExt w v.earlyValues.runs[1]!,
        row := decodeExt w v.earlyValues.runs[2]!, pairs := v.earlyValues.pairs.map (decodePair w),
        block := decodeExt w v.earlyValues.block },
    quotient := v.quotient.map (decodeExt w), linear := v.linear.map (·.map (decodeExt w)),
    finals := v.finals.map (decodeExt w), partials := v.partials.map (·.map (decodeExt w)),
    p1 := decodeDigest w v.p1, leaves := v.leaves.map (·.decode w), late := v.late.decode w,
    lateValues := v.lateValues.map (decodePair w), p1Opening := v.p1Opening.decode w,
    p0Opening := v.p0Opening.decode w }

/-- The round messages and the children of one GKR layer. -/
def IsGkrLayer (c : Ctx F) (n : ℕ) (v : GkrLayerView) (g : Replay.GkrLayer) : Prop :=
  List.Forall₂ (IsExts c n) v.rounds.toList g.rounds.toList ∧
    List.Forall₂ (IsExts c n) v.children.toList g.children.toList

theorem IsGkrLayer.mono {c : Ctx F} {n m : ℕ} {v : GkrLayerView} {g : Replay.GkrLayer} (h : IsGkrLayer c n v g)
    (le : n ≤ m) : IsGkrLayer c m v g :=
  ⟨h.1.imp fun _ _ h => h.mono le, h.2.imp fun _ _ h => h.mono le⟩

/-- The roots and the layers of a GKR proof. -/
def IsGkr (c : Ctx F) (n : ℕ) (v : GkrView) (g : Replay.GkrProof) : Prop :=
  List.Forall₂ (IsPair c n) v.roots.toList g.roots.toList ∧
    List.Forall₂ (IsGkrLayer c n) v.layers.toList g.layers.toList

theorem IsGkr.mono {c : Ctx F} {n m : ℕ} {v : GkrView} {g : Replay.GkrProof} (h : IsGkr c n v g) (le : n ≤ m) :
    IsGkr c m v g :=
  ⟨h.1.imp fun _ _ h => h.mono le, h.2.imp fun _ _ h => h.mono le⟩

/-- The values and the path of a setup leaf. -/
def IsLeaf (c : Ctx F) (n : ℕ) (v : LeafView) (g : Replay.Leaf) : Prop :=
  IsWords c n v.values g.values ∧ List.Forall₂ (IsDigest c n) v.path.toList g.path.toList

theorem IsLeaf.mono {c : Ctx F} {n m : ℕ} {v : LeafView} {g : Replay.Leaf} (h : IsLeaf c n v g) (le : n ≤ m) :
    IsLeaf c m v g :=
  ⟨h.1.mono le, h.2.imp fun _ _ h => h.mono le⟩

/-- Each query row holds the words that `w` gives it. -/
def IsRows (c : Ctx F) (w : Var → Word) (n : ℕ) (rows : Array (Array Var)) : Prop :=
  ∀ row ∈ rows, IsWords c n row (row.map w)

theorem IsRows.mono {c : Ctx F} {w : Var → Word} {n m : ℕ} {rows : Array (Array Var)} (h : IsRows c w n rows)
    (le : n ≤ m) : IsRows c w m rows :=
  fun row hr => (h row hr).mono le

/-- The root, the answers, the sum-check and the query rows of one WHIR round. -/
def IsRound (c : Ctx F) (w : Var → Word) (n : ℕ) (v : RoundView) (g : Replay.WhirRound) : Prop :=
  IsDigest c n v.root g.root ∧ IsExts c n v.ood g.ood ∧
    List.Forall₂ (IsPair c n) v.sumcheck.toList g.sumcheck.toList ∧ IsRows c w n v.queries

theorem IsRound.mono {c : Ctx F} {w : Var → Word} {n m : ℕ} {v : RoundView} {g : Replay.WhirRound}
    (h : IsRound c w n v g) (le : n ≤ m) : IsRound c w m v g :=
  ⟨h.1.mono le, h.2.1.mono le, h.2.2.1.imp fun _ _ h => h.mono le, h.2.2.2.mono le⟩

/-- Each field of one WHIR opening, and its query rows. -/
def IsOpening (c : Ctx F) (w : Var → Word) (n : ℕ) (v : OpeningView) (g : Replay.Opening) : Prop :=
  IsExts c n v.initialOod g.initialOod ∧ List.Forall₂ (IsExts c n) v.evals.toList g.evals.toList ∧
    List.Forall₂ (IsPair c n) v.initialSumcheck.toList g.initialSumcheck.toList ∧
    List.Forall₂ (IsRound c w n) v.rounds.toList g.rounds.toList ∧ IsExts c n v.finalPoly g.finalPoly ∧
    IsRows c w n v.finalQueries ∧ List.Forall₂ (IsPair c n) v.finalSumcheck.toList g.finalSumcheck.toList

theorem IsOpening.mono {c : Ctx F} {w : Var → Word} {n m : ℕ} {v : OpeningView} {g : Replay.Opening}
    (h : IsOpening c w n v g) (le : n ≤ m) : IsOpening c w m v g :=
  ⟨h.1.mono le, h.2.1.imp fun _ _ h => h.mono le, h.2.2.1.imp fun _ _ h => h.mono le,
    h.2.2.2.1.imp fun _ _ h => h.mono le, h.2.2.2.2.1.mono le, h.2.2.2.2.2.1.mono le,
    h.2.2.2.2.2.2.imp fun _ _ h => h.mono le⟩

/-- Each field of a layer-1 proof. -/
def IsProof (c : Ctx F) (w : Var → Word) (n : ℕ) (v : ProofView) (g : Replay.Proof) : Prop :=
  IsDigest c n v.p0 g.p0 ∧ IsWords c n v.histogram g.histogram ∧ IsGkr c n v.early g.early ∧
    IsExts c n v.earlyValues.runs #[g.earlyValues.eRe, g.earlyValues.eIm, g.earlyValues.row] ∧
    List.Forall₂ (IsPair c n) v.earlyValues.pairs.toList g.earlyValues.pairs.toList ∧
    IsExt c n v.earlyValues.block g.earlyValues.block ∧ IsExts c n v.quotient g.quotient ∧
    List.Forall₂ (IsExts c n) v.linear.toList g.linear.toList ∧ IsExts c n v.finals g.finals ∧
    List.Forall₂ (IsExts c n) v.partials.toList g.partials.toList ∧ IsDigest c n v.p1 g.p1 ∧
    List.Forall₂ (IsLeaf c n) v.leaves.toList g.leaves.toList ∧ IsGkr c n v.late g.late ∧
    List.Forall₂ (IsPair c n) v.lateValues.toList g.lateValues.toList ∧
    IsOpening c w n v.p1Opening g.p1Opening ∧ IsOpening c w n v.p0Opening g.p0Opening

theorem IsProof.mono {c : Ctx F} {w : Var → Word} {n m : ℕ} {v : ProofView} {g : Replay.Proof}
    (h : IsProof c w n v g) (le : n ≤ m) : IsProof c w m v g := by
  obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16⟩ := h
  exact ⟨h1.mono le, h2.mono le, h3.mono le, h4.mono le, h5.imp fun _ _ h => h.mono le, h6.mono le, h7.mono le,
    h8.imp fun _ _ h => h.mono le, h9.mono le, h10.imp fun _ _ h => h.mono le, h11.mono le,
    h12.imp fun _ _ h => h.mono le, h13.mono le, h14.imp fun _ _ h => h.mono le, h15.mono le, h16.mono le⟩

section Decoded

variable {c : Ctx F} {w : Var → Word} {m : ℕ}

/-- Three handles below `m` are their decoded value. -/
theorem isExt_decode (hw : Decodes c w) {e : ExtVar} (h : ∀ k, e k < m) : IsExt c m e (decodeExt w e) := by
  intro k
  refine ⟨h k, ?_⟩
  rw [hw]
  fin_cases k <;> rfl

/-- Handles below `m` are their decoded words. -/
theorem isWords_map (hw : Decodes c w) {hs : Array Var} (h : ∀ i (hi : i < hs.size), hs[i] < m) :
    IsWords c m hs (hs.map w) := by
  refine ⟨(Array.size_map ..).symm, fun i hi => ?_⟩
  rw [getElem!_pos hs i hi, getElem!_pos _ i (by simpa using hi), Array.getElem_map]
  exact ⟨h i hi, hw _⟩

/-- Extension handles below `m` are their decoded values. -/
theorem isExts_map (hw : Decodes c w) {xs : Array ExtVar} (h : ∀ i (hi : i < xs.size), ∀ k, xs[i] k < m) :
    IsExts c m xs (xs.map (decodeExt w)) := by
  refine ⟨(Array.size_map ..).symm, fun i hi => ?_⟩
  rw [getElem!_pos xs i hi, getElem!_pos _ i (by simpa using hi), Array.getElem_map]
  exact isExt_decode hw (h i hi)

/-- An array and its image under `f` meet `Forall₂ R` when each entry and its image meet `R`. -/
theorem forall₂_map {α β : Type} {R : α → β → Prop} {f : α → β} {xs : Array α}
    (h : ∀ i (hi : i < xs.size), R xs[i] (f xs[i])) : List.Forall₂ R xs.toList (xs.map f).toList := by
  rw [Array.toList_map, List.forall₂_map_right_iff, List.forall₂_same]
  intro x hx
  obtain ⟨i, hi, rfl⟩ := List.mem_iff_getElem.mp hx
  simpa using h i (by simpa using hi)

/-- When `R` relates `l` to `xs` and gives `S`, each entry of `xs` meets `S`. -/
theorem forall₂_right {α β : Type} {R : α → β → Prop} {S : β → Prop} {l : List α} {xs : Array β}
    (h : List.Forall₂ R l xs.toList) (hs : ∀ a b, R a b → S b) : ∀ i (hi : i < xs.size), S xs[i] := by
  intro i hi
  have len := h.length_eq
  simp only [Array.length_toList] at len
  have := hs _ _ (h.get (i := i) (by omega) (by simpa using hi))
  rwa [List.get_eq_getElem, Array.getElem_toList] at this

/-- A loop of checks passes when each step passes. -/
theorem forIn_ok {α ε : Type} {l : List α} {body : α → PUnit → Except ε (ForInStep PUnit)}
    (h : ∀ x ∈ l, body x ⟨⟩ = .ok (.yield ⟨⟩)) : forIn l ⟨⟩ body = .ok ⟨⟩ := by
  induction l with
  | nil => rfl
  | cons x l ih =>
      rw [List.forIn_cons, h x List.mem_cons_self]
      exact ih fun y hy => h y (List.mem_cons_of_mem _ hy)

end Decoded

namespace Sim

variable {c : Ctx F} {n : ℕ} {P : Prop} {w : Var → Word}

/-- `(0..count).map(…)`: entry `i` meets `R i`. -/
theorem rangeMapM {α : Type} {f : ℕ → Build α} (count : ℕ) (R : ℕ → α → ℕ → Prop)
    (mono : ∀ i a m m', R i a m → m ≤ m' → R i a m') (h : ∀ i < count, ∀ m, n ≤ m → Sim c m P (f i) (R i) True) :
    Sim c n P ((Array.range count).mapM f) (fun as m => as.size = count ∧ ∀ i (hi : i < as.size), R i as[i] m)
      True :=
  (Sim.mapM R mono fun i hi m le => h i (Array.mem_range.mp hi) m le).post (fun _ as m hf => by
    rw [Array.toList_range, List.forall₂_iff_get] at hf
    have size : as.size = count := by simpa using hf.1.symm
    refine ⟨size, fun i hi => ?_⟩
    simpa using hf.2 i (by simp; omega) (by simpa using hi)) fun _ => Iff.rfl

/-- `bind₀` that passes `Decodes` on to the rest of the program. -/
theorem bindDecodes {α β : Type} {b : Build α} {f : α → Build β} {Q : α → ℕ → Prop} {R : β → ℕ → Prop} {C : Prop}
    (hw : P → Decodes c w) (hb : (P → Decodes c w) → Sim c n P b Q True)
    (hf : ∀ a m, n ≤ m → (P ∧ Q a m → Decodes c w) → Sim c m (P ∧ Q a m) (f a) R C) : Sim c n P (b >>= f) R C :=
  (hb hw).bind₀ fun a m le => hf a m le fun h => hw h.1

/-- `count` proof words, one `private` each (`algebra.rs:16`). -/
theorem privateWords (count : ℕ) (hw : P → Decodes c w) :
    Sim c n P ((Array.range count).mapM fun _ => Build.priv) (fun hs m => IsWords c m hs (hs.map w) ∧ hs.size = count)
      True :=
  (rangeMapM count (fun _ h m => h < m) (fun _ _ _ _ h le => lt_of_lt_of_le h le) fun _ _ _ _ =>
    Sim.priv.post (fun _ _ _ h => h.1) fun _ => Iff.rfl).post
    (fun hP _ _ h => ⟨isWords_map (hw hP) h.2, h.1⟩) fun _ => Iff.rfl

/-- `exts` (`verifier.rs:58-62`): `count` extension values. -/
theorem privateExts (count : ℕ) (hw : P → Decodes c w) :
    Sim c n P (Circuit.privateExts count)
      (fun xs m => IsExts c m xs (xs.map (decodeExt w)) ∧ xs.size = count) True :=
  (rangeMapM count (fun _ a m => ∀ k, a k < m) (fun _ _ _ _ h le k => lt_of_lt_of_le (h k) le) fun _ _ _ _ =>
    Sim.privateExt.post (fun _ _ _ h k => (h k).2.1) fun _ => Iff.rfl).post
    (fun hP _ _ h => ⟨isExts_map (hw hP) h.2, h.1⟩) fun _ => Iff.rfl

/-- `pairs` (`verifier.rs:63-67`): `count` pairs of extension values. -/
theorem privatePairs (count : ℕ) (hw : P → Decodes c w) :
    Sim c n P (Circuit.privatePairs count)
      (fun xs m => List.Forall₂ (IsPair c m) xs.toList (xs.map (decodePair w)).toList ∧ xs.size = count) True :=
  (rangeMapM count (fun _ (p : ExtVar × ExtVar) m => (∀ k, p.1 k < m) ∧ ∀ k, p.2 k < m)
    (fun _ _ _ _ h le => ⟨fun k => lt_of_lt_of_le (h.1 k) le, fun k => lt_of_lt_of_le (h.2 k) le⟩)
    fun _ _ _ _ => Sim.privateExt.bind₀ fun _ _ l₁ => Sim.privateExt.bind₀ fun _ _ _ => Sim.pure fun h =>
      ⟨fun k => lt_of_lt_of_le (h.1.2 k).2.1 (by omega), fun k => (h.2 k).2.1⟩).post
    (fun hP _ _ h => ⟨forall₂_map fun i hi =>
      ⟨isExt_decode (hw hP) (h.2 i hi).1, isExt_decode (hw hP) (h.2 i hi).2⟩, h.1⟩) fun _ => Iff.rfl

/-- `root` (`verifier.rs:68-75`): a digest. -/
theorem privateDigest (hw : P → Decodes c w) :
    Sim c n P Circuit.privateDigest (fun d m => IsDigest c m d (decodeDigest w d)) True :=
  (privateWords 4 hw).bind₀ fun _ _ _ => Sim.pure fun h => ⟨rfl, fun lane => by
    refine ⟨(h.2.1.2 lane (by have := h.2.2; omega)).1, ?_⟩
    rw [hw h.1]
    fin_cases lane <;> rfl⟩

/-- `rows` (`whir/mod.rs:76-113`): `draws` rows of the row length. -/
theorem queryRows (draws width : ℕ) (base : Bool) (hw : P → Decodes c w) :
    Sim c n P (Circuit.queryRows draws width base)
      (fun rows m => IsRows c w m rows ∧ rows.size = draws ∧
        ∀ row ∈ rows, row.size = if base then width else 3 * width) True :=
  (rangeMapM draws (fun _ row m => IsWords c m row (row.map w) ∧ row.size = if base then width else 3 * width)
    (fun _ _ _ _ h le => ⟨h.1.mono le, h.2⟩) fun _ _ _ _ => privateWords _ hw).post
    (fun _ _ _ h => by
      refine ⟨fun row hr => ?_, h.1, fun row hr => ?_⟩ <;> obtain ⟨i, hi, rfl⟩ := Array.mem_iff_getElem.mp hr
      exacts [(h.2 i hi).1, (h.2 i hi).2]) fun _ => Iff.rfl

/-- `LeafView::read` (`setup.rs:222-236`), with the sizes that `readProof` checks. -/
theorem LeafView.read (oracles setupVars : ℕ) (hw : P → Decodes c w) :
    Sim c n P (Circuit.LeafView.read oracles setupVars)
      (fun v m => IsLeaf c m v (v.decode w) ∧ (v.decode w).values.size = oracles * 2 ^ foldVars ∧
        (v.decode w).path.size = setupVars + 1 - foldVars) True :=
  bindDecodes hw (privateWords _) fun _ _ _ hw =>
    (rangeMapM _ (fun _ d m => IsDigest c m d (decodeDigest w d)) (fun _ _ _ _ h le => h.mono le)
      fun _ _ _ _ => privateDigest hw).bind₀ fun _ _ l₂ => Sim.pure fun h =>
        ⟨⟨h.1.2.1.mono l₂, forall₂_map h.2.2⟩, by simp [LeafView.decode, h.1.2.2], by simp [LeafView.decode, h.2.1]⟩

/-- `GkrView::read` (`gkr.rs:608-656`): the replay's `readGkr` accepts the decoded proof. -/
theorem GkrView.read (shapes : Array TreeShape) (hs : ∀ s ∈ shapes, s.depth ≠ 0 ∧ s.factors ≤ 2)
    (hw : P → Decodes c w) :
    Sim c n P (Circuit.GkrView.read shapes)
      (fun v m => IsGkr c m v (v.decode w) ∧ Replay.readGkr (v.decode w) shapes = .ok ()) True := by
  refine bindDecodes hw (privatePairs _) fun roots m₁ l₁ hw => ?_
  refine (rangeMapM (layerCount shapes)
    (fun i l m => IsGkrLayer c m l (l.decode w) ∧ layerShapeOk shapes (i + 1) (l.decode w) = true)
    (fun _ _ _ _ h le => ⟨h.1.mono le, h.2⟩) fun i _ m _ => ?layer).bind₀ fun layers m₂ l₂ => Sim.pure fun h => ?final
  case layer =>
    dsimp only
    refine bindDecodes hw (fun hw => rangeMapM _ (fun _ a m => IsExts c m a (a.map (decodeExt w)) ∧
        a.size = layerDegree (layerFactors shapes (i + 1)))
      (fun _ _ _ _ h le => ⟨h.1.mono le, h.2⟩) fun _ _ _ _ => privateExts _ hw) fun rounds m₃ l₃ hw => ?_
    refine (Sim.mapM (fun f a m => IsExts c m a (a.map (decodeExt w)) ∧ a.size = 2 * (f + 1))
      (fun _ _ _ _ h le => ⟨h.1.mono le, h.2⟩) fun _ _ _ _ => privateExts _ hw).bind₀
      fun children m₄ l₄ => Sim.pure fun h => ?_
    obtain ⟨⟨-, hrs, hr⟩, hc⟩ := h
    have hf : List.Forall₂ (fun t (a : Array Ext) => a.size = 2 * (partFactors shapes (i + 1) t + 1))
        (activeTrees shapes (i + 1)).toList (children.map (·.map (decodeExt w))).toList := by
      rw [Array.toList_map, List.forall₂_map_right_iff]
      have := hc.imp fun _ _ h => h.2
      simpa [layerFactors] using this
    refine ⟨⟨forall₂_map fun j hj => (hr j hj).1.mono l₄, forall₂_map (forall₂_right hc fun _ _ h => h.1)⟩, ?_⟩
    simp only [layerShapeOk, GkrLayerView.decode, Bool.and_eq_true, beq_iff_eq, Array.all_eq_true']
    refine ⟨⟨⟨by simpa using hrs, fun a ha => ?_⟩, by simpa [layerFactors] using hc.length_eq.symm⟩,
      fun ⟨t, a⟩ hp => ?_⟩
    · obtain ⟨j, hj, rfl⟩ := Array.mem_iff_getElem.mp ha
      simpa using (hr j (by simpa using hj)).2
    · rw [← Array.mem_toList_iff, Array.toList_zip] at hp
      simpa using List.forall₂_zip hf hp
  case final =>
    obtain ⟨⟨-, hroots, hrs⟩, hls, hl⟩ := h
    refine ⟨⟨hroots.imp fun _ _ h => h.mono l₂, forall₂_map fun j hj => (hl j hj).1⟩, ?_⟩
    unfold Replay.readGkr
    have hany : (shapes.any fun s => s.depth == 0 || decide (s.factors > 2)) = false := by
      rw [Bool.eq_false_iff, ne_eq, Array.any_eq_true']
      rintro ⟨s, hm, hb⟩
      have := hs s hm
      simp at hb
      omega
    have hall : ((List.range (layerCount shapes)).all fun i =>
        layerShapeOk shapes (i + 1) (layers.map (GkrLayerView.decode w))[i]!) = true := by
      rw [List.all_eq_true]
      intro i hi
      rw [List.mem_range] at hi
      rw [getElem!_pos _ i (by simp; omega), Array.getElem_map]
      exact (hl i (by omega)).2
    simp only [GkrView.decode, Array.size_map, hrs, hls, hany]
    simp [Replay.ensure, hall]
    rfl

/-- `OpeningView::read` (`whir/mod.rs:125-223`): the replay's `readOpening` accepts the decoded
opening. -/
theorem OpeningView.read (shape : WhirShape) (rows : OpeningRows) (hw : P → Decodes c w) :
    Sim c n P (Circuit.OpeningView.read shape rows)
      (fun v m => IsOpening c w m v (v.decode w) ∧ Replay.readOpening shape (v.decode w) = .ok ()) True := by
  refine bindDecodes hw (privateExts _) fun initialOod m₁ l₁ hw => ?_
  refine bindDecodes hw (fun hw => Sim.mapM (fun (b : Replay.WhirBatch) a m => IsExts c m a (a.map (decodeExt w)) ∧
    a.size = b.size) (fun _ _ _ _ h le => ⟨h.1.mono le, h.2⟩) fun _ _ _ _ => privateExts _ hw) fun evals m₂ l₂ hw => ?_
  refine bindDecodes hw (privatePairs _) fun initialSumcheck m₃ l₃ hw => ?_
  refine bindDecodes hw (fun hw => Sim.mapM (fun (x : Replay.WhirRoundShape × ℕ) r m => IsRound c w m r (r.decode w) ∧
      r.ood.size = x.1.oodSamples ∧ r.sumcheck.size = x.1.sumcheck.rounds)
    (fun _ _ _ _ h le => ⟨h.1.mono le, h.2⟩) fun x _ m _ => ?round) fun rounds m₄ l₄ hw => ?_
  case round =>
    obtain ⟨round, r⟩ := x
    dsimp only
    refine bindDecodes hw privateDigest fun root m₅ l₅ hw => ?_
    refine bindDecodes hw (privateExts _) fun ood m₆ l₆ hw => ?_
    refine bindDecodes hw (queryRows _ _ _) fun queries m₇ l₇ hw => ?_
    refine (privatePairs _ hw).bind₀ fun sumcheck m₈ l₈ => Sim.pure fun h => ?_
    obtain ⟨⟨⟨⟨-, hroot⟩, hood⟩, hq⟩, hsum⟩ := h
    exact ⟨⟨hroot.mono (by omega), hood.1.mono (by omega), hsum.1, hq.1.mono l₈⟩, hood.2, hsum.2⟩
  refine bindDecodes hw (privateExts _) fun finalPoly m₅ l₅ hw => ?_
  refine bindDecodes hw (queryRows _ _ _) fun finalQueries m₆ l₆ hw => ?_
  refine (privatePairs _ hw).bind₀ fun finalSumcheck m₇ l₇ => Sim.pure fun h => ?final
  case final =>
    obtain ⟨⟨⟨⟨⟨⟨⟨-, h1⟩, h2⟩, h3⟩, h4⟩, h5⟩, h6⟩, h7⟩ := h
    refine ⟨⟨h1.1.mono (by omega),
      forall₂_map fun i hi => (forall₂_right h2 (fun _ _ h => h.1) i hi).mono (by omega),
      h3.1.imp fun _ _ h => h.mono (by omega),
      forall₂_map fun i hi => (forall₂_right h4 (fun _ _ h => h.1) i hi).mono (by omega),
      h5.1.mono (by omega), h6.1.mono l₇, h7.1⟩, ?_⟩
    have hrs : rounds.size = shape.rounds.size := by simpa using h4.length_eq.symm
    have hes : evals.size = shape.batches.size := by simpa using h2.length_eq.symm
    have hb : List.Forall₂ (fun (b : Replay.WhirBatch) (a : Array Ext) => a.size = b.size) shape.batches.toList
        (evals.map (·.map (decodeExt w))).toList := by
      rw [Array.toList_map, List.forall₂_map_right_iff]
      exact h2.imp fun _ _ h => by simpa using h.2
    have hr : List.Forall₂ (fun (x : Replay.WhirRoundShape) (a : Replay.WhirRound) =>
        a.ood.size = x.oodSamples ∧ a.sumcheck.size = x.sumcheck.rounds) shape.rounds.toList
        (rounds.map (RoundView.decode w)).toList := by
      rw [Array.toList_map, List.forall₂_map_right_iff, ← List.zipIdx_map_fst 0 shape.rounds.toList,
        List.forall₂_map_left_iff]
      rw [Array.toList_zipIdx] at h4
      exact h4.imp fun _ _ h => by simpa [RoundView.decode] using h.2
    have hball : ((shape.batches.zip (evals.map (·.map (decodeExt w)))).all fun pair =>
        pair.2.size == pair.1.size) = true := by
      rw [Array.all_eq_true']
      rintro ⟨b, a⟩ hp
      rw [← Array.mem_toList_iff, Array.toList_zip] at hp
      simpa using List.forall₂_zip hb hp
    unfold Replay.readOpening
    simp only [OpeningView.decode]
    rw [← Array.forIn_toList, forIn_ok]
    · simp only [hball]
      simp [Replay.ensure, hrs, hes, h1.2, h3.2, h5.2, h7.2]
    · rintro ⟨x, a⟩ hp
      rw [Array.toList_zip] at hp
      have := List.forall₂_zip hr hp
      simp [Replay.ensure, this.1, this.2]
      rfl

/-- `ProofView::read` (`verifier.rs:43-122`): the replay's `readProof` accepts the decoded
proof. -/
theorem ProofView.read (rel : Relation) (p1Rows p0Rows : OpeningRows)
    (hrun : rel.key.layout.runVariables ≠ 0) (hrow : rel.key.layout.rowVariables ≠ 0)
    (hslot : rel.key.layout.slotVariables + rel.matrixVars ≠ 0) (hw : P → Decodes c w) :
    Sim c n P (Circuit.ProofView.read rel p1Rows p0Rows)
      (fun v m => IsProof c w m v (v.decode w) ∧ Replay.readProof rel (v.decode w) = .ok ()) True := by
  have early : ∀ s ∈ rel.earlyShapes, s.depth ≠ 0 ∧ s.factors ≤ 2 := by
    simp [Relation.earlyShapes, Relation.cubeVars, Replay.laneVars, hrun, hrow]
    omega
  have late : ∀ s ∈ rel.lateShapes, s.depth ≠ 0 ∧ s.factors ≤ 2 := by
    simp [Relation.lateShapes, Relation.blockVars, Replay.bitsAtLeastOne]
  refine bindDecodes hw privateDigest fun p0 m₁ l₁ hw => ?_
  refine bindDecodes hw (privateWords _) fun histogram m₂ l₂ hw => ?_
  refine bindDecodes hw (GkrView.read _ early) fun earlyGkr m₃ l₃ hw => ?_
  refine bindDecodes hw (privateExts 3) fun runs m₄ l₄ hw => ?_
  refine bindDecodes hw (privatePairs _) fun pairs m₅ l₅ hw => ?_
  refine bindDecodes hw (fun hw => Sim.privateExt.post (Q' := fun a m => IsExt c m a (decodeExt w a))
    (fun h _ _ hr => isExt_decode (hw h) fun k => (hr k).2.1) fun _ => Iff.rfl) fun block m₆ l₆ hw => ?_
  refine bindDecodes hw (privateExts _) fun quotient m₇ l₇ hw => ?_
  refine bindDecodes hw (fun hw => rangeMapM _ (fun _ a m => IsExts c m a (a.map (decodeExt w)) ∧ a.size = 2)
    (fun _ _ _ _ h le => ⟨h.1.mono le, h.2⟩) fun _ _ _ _ => privateExts _ hw) fun linear m₈ l₈ hw => ?_
  refine bindDecodes hw (privateExts 3) fun finals m₉ l₉ hw => ?_
  refine bindDecodes hw (fun hw => rangeMapM _ (fun _ a m => IsExts c m a (a.map (decodeExt w)) ∧
    a.size = 2 ^ foldVars) (fun _ _ _ _ h le => ⟨h.1.mono le, h.2⟩) fun _ _ _ _ => privateExts _ hw)
    fun partials m₁₀ l₁₀ hw => ?_
  refine bindDecodes hw privateDigest fun p1 m₁₁ l₁₁ hw => ?_
  refine bindDecodes hw (fun hw => rangeMapM _ (fun _ v m => IsLeaf c m v (v.decode w) ∧
      (v.decode w).values.size = (rel.kappa + Replay.runOracles) * 2 ^ foldVars ∧
      (v.decode w).path.size = rel.setupVars + 1 - foldVars)
    (fun _ _ _ _ h le => ⟨h.1.mono le, h.2⟩) fun _ _ _ _ => LeafView.read _ _ hw) fun leaves m₁₂ l₁₂ hw => ?_
  refine bindDecodes hw (GkrView.read _ late) fun lateGkr m₁₃ l₁₃ hw => ?_
  refine bindDecodes hw (privatePairs _) fun lateValues m₁₄ l₁₄ hw => ?_
  refine bindDecodes hw (OpeningView.read _ _) fun p1Opening m₁₅ l₁₅ hw => ?_
  refine bindDecodes hw (OpeningView.read _ _) fun p0Opening m₁₆ l₁₆ hw => Sim.pure fun h => ?_
  have hd := hw h
  obtain ⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨-, h1⟩, h2⟩, h3⟩, h4⟩, h5⟩, h6⟩, h7⟩, h8⟩, h9⟩, h10⟩, h11⟩, h12⟩, h13⟩, h14⟩, h15⟩,
    h16⟩ := h
  refine ⟨⟨h1.mono (by omega), h2.1.mono (by omega), h3.1.mono (by omega), ?runs,
    h5.1.imp fun _ _ h => h.mono (by omega), h6.mono (by omega), h7.1.mono (by omega),
    forall₂_map fun i hi => (h8.2 i hi).1.mono (by omega), h9.1.mono (by omega),
    forall₂_map fun i hi => (h10.2 i hi).1.mono (by omega), h11.mono (by omega),
    forall₂_map fun i hi => (h12.2 i hi).1.mono (by omega), h13.1.mono (by omega),
    h14.1.imp fun _ _ h => h.mono (by omega), h15.1.mono l₁₆, h16.1⟩, ?shape⟩
  case runs =>
    show IsExts c m₁₆ runs #[decodeExt w runs[0]!, decodeExt w runs[1]!, decodeExt w runs[2]!]
    refine ⟨by simp [h4.2], fun i hi => ?_⟩
    have lt : ∀ k, runs[i]! k < m₁₆ := fun k => lt_of_lt_of_le (h4.1.2 i hi k).1 (show m₄ ≤ m₁₆ by omega)
    rw [h4.2] at hi
    interval_cases i <;> exact isExt_decode hd lt
  case shape =>
    have hlin : (linear.map (·.map (decodeExt w))).all (·.size == 2) = true := by
      rw [Array.all_eq_true']
      intro a ha
      obtain ⟨i, hi, rfl⟩ := Array.mem_iff_getElem.mp ha
      simpa using (h8.2 i (by simpa using hi)).2
    have hpart : (partials.map (·.map (decodeExt w))).all (·.size == 2 ^ foldVars) = true := by
      rw [Array.all_eq_true']
      intro a ha
      obtain ⟨i, hi, rfl⟩ := Array.mem_iff_getElem.mp ha
      simpa using (h10.2 i (by simpa using hi)).2
    have hleaf : ((leaves.map (·.decode w)).all fun leaf =>
        leaf.values.size == (rel.kappa + Replay.runOracles) * 2 ^ foldVars &&
          leaf.path.size == rel.setupVars + 1 - foldVars) = true := by
      rw [Array.all_eq_true']
      intro a ha
      obtain ⟨i, hi, rfl⟩ := Array.mem_iff_getElem.mp ha
      have := (h12.2 i (by simpa using hi)).2
      simp only [Array.getElem_map, this.1, this.2, beq_self_eq_true, Bool.and_self]
    unfold Replay.readProof
    simp only [ProofView.decode, hlin, hpart, hleaf]
    simp [Replay.ensure, decodeDigest, h2.2, h5.2, h7.2, h8.1, h9.2, h10.1, h12.1, h14.2, h3.2, h13.2, h15.2,
      h16.2]
    rfl

end Sim

end CompressionLean.Circuit
