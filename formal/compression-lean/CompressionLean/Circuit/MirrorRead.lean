import CompressionLean.Circuit.Mirror

/-!
Owns what `verifyWith` reads before it verifies: the layer-1 seed of the
handoff and the proof view (`ProofView.read`). On a run whose checks hold,
their handles are values of the run (`*.Below`), and the views have the
sizes that `readProof` checks (`*.Shaped`).
-/

namespace CompressionLean.Circuit

open CompressionLean.Replay (Gl Ext)

/-! ## Handles below a count -/

def ExtVar.Below (m : ℕ) (a : ExtVar) : Prop := ∀ k, a k < m

def Exts.Below (m : ℕ) (as : Array ExtVar) : Prop := ∀ a ∈ as, ExtVar.Below m a

def Pairs.Below (m : ℕ) (ps : Array (ExtVar × ExtVar)) : Prop :=
  ∀ p ∈ ps, ExtVar.Below m p.1 ∧ ExtVar.Below m p.2

def DigestVar.Below (m : ℕ) (d : DigestVar) : Prop := ∀ lane, d lane < m

theorem ExtVar.Below.mono {m m' : ℕ} {a : ExtVar} (h : ExtVar.Below m a) (l : m ≤ m') : ExtVar.Below m' a :=
  fun k => lt_of_lt_of_le (h k) l

theorem Exts.Below.mono {m m' : ℕ} {as : Array ExtVar} (h : Exts.Below m as) (l : m ≤ m') : Exts.Below m' as :=
  fun a ha => (h a ha).mono l

theorem Pairs.Below.mono {m m' : ℕ} {ps : Array (ExtVar × ExtVar)} (h : Pairs.Below m ps) (l : m ≤ m') :
    Pairs.Below m' ps :=
  fun p hp => ⟨(h p hp).1.mono l, (h p hp).2.mono l⟩

theorem DigestVar.Below.mono {m m' : ℕ} {d : DigestVar} (h : DigestVar.Below m d) (l : m ≤ m') :
    DigestVar.Below m' d :=
  fun lane => lt_of_lt_of_le (h lane) l

variable {F : Type} [CommRing F] {c : Ctx F} {n : ℕ} {P : Prop}

theorem getElem!_mem {α : Type} [Inhabited α] {xs : Array α} {i : ℕ} (h : i < xs.size) : xs[i]! ∈ xs := by
  rw [getElem!_pos xs i h]
  exact Array.getElem_mem h

namespace Accepts

theorem priv : Accepts c n P Build.priv fun h m => h < m :=
  (emit .priv).post fun _ _ _ e => by rw [e.1, e.2]; simp [Instr.outputs]

theorem privateExt : Accepts c n P Algebra.privateExt fun a m => ExtVar.Below m a :=
  priv.bind fun x m₁ _ => priv.bind fun y m₂ l₂ => priv.bind fun z m₃ l₃ => pure fun h k => by
    fin_cases k
    · exact lt_of_lt_of_le h.1.1.2 (l₂.trans l₃)
    · exact lt_of_lt_of_le h.1.2 l₃
    · exact h.2

theorem privateExts (count : ℕ) :
    Accepts c n P (privateExts count) fun as m => as.size = count ∧ Exts.Below m as := by
  refine (arrayMapM (R := fun _ a m => ExtVar.Below m a) (fun _ _ _ _ h l k => lt_of_lt_of_le (h k) l) _
    fun _ _ _ _ => privateExt).post fun _ as m h => ⟨by simpa using h.1, fun a ha => ?_⟩
  obtain ⟨i, hi, rfl⟩ := Array.mem_iff_getElem.mp ha
  exact h.2 i (by simpa [h.1] using hi) hi

/-- `mapM` with a per-entry fact, read as a fact of every entry. -/
theorem arrayMapM_mem {δ γ : Type} {f : δ → Build γ} (S : γ → ℕ → Prop)
    (mono : ∀ y m m', S y m → m ≤ m' → S y m')
    (xs : Array δ) (each : ∀ x ∈ xs, ∀ m, n ≤ m → Accepts c m P (f x) S) :
    Accepts c n P (xs.mapM f) fun ys m => ys.size = xs.size ∧ ∀ y ∈ ys, S y m := by
  refine (arrayMapM (R := fun _ y m => S y m) (fun _ y m m' h l => mono y m m' h l) xs each).post
    fun _ ys m h => ⟨h.1, fun y hy => ?_⟩
  obtain ⟨i, hi, rfl⟩ := Array.mem_iff_getElem.mp hy
  exact h.2 i (by rw [← h.1]; exact hi) hi

theorem privatePairs (count : ℕ) :
    Accepts c n P (privatePairs count) fun ps m => ps.size = count ∧ Pairs.Below m ps := by
  refine (arrayMapM_mem (fun (p : ExtVar × ExtVar) m => ExtVar.Below m p.1 ∧ ExtVar.Below m p.2)
    (fun _ _ _ h l => ⟨fun k => lt_of_lt_of_le (h.1 k) l, fun k => lt_of_lt_of_le (h.2 k) l⟩) _
    fun _ _ _ _ => privateExt.bind fun a m₁ _ => privateExt.bind fun b m₂ l₂ => pure fun h =>
      ⟨fun k => lt_of_lt_of_le (h.1.2 k) l₂, h.2⟩).post fun _ ps m h => ⟨by simpa using h.1, h.2⟩

theorem privateDigest : Accepts c n P Circuit.privateDigest fun d m => DigestVar.Below m d := by
  refine (arrayMapM_mem (fun h m => h < m) (fun _ _ _ h l => lt_of_lt_of_le h l) _ fun _ _ _ _ => priv).bind
    fun words m _ => pure fun h lane => h.2.2 _ (getElem!_mem (by rw [h.2.1]; simp))

end Accepts

open CompressionLean.Replay (TreeShape layerCount layerFactors layerDegree)

/-- A GKR view's handles below `m`. -/
def GkrView.Below (m : ℕ) (g : GkrView) : Prop :=
  Pairs.Below m g.roots ∧
    ∀ layer ∈ g.layers, (∀ r ∈ layer.rounds, Exts.Below m r) ∧ ∀ ch ∈ layer.children, Exts.Below m ch

/-- A GKR view has the sizes that `GkrView.read` gives for `shapes`. -/
def GkrView.Shaped (shapes : Array TreeShape) (g : GkrView) : Prop :=
  g.roots.size = shapes.size ∧ g.layers.size = layerCount shapes ∧
    ∀ i (hi : i < g.layers.size), g.layers[i].rounds.size = i ∧
      (∀ r ∈ g.layers[i].rounds, r.size = layerDegree (layerFactors shapes (i + 1))) ∧
      g.layers[i].children.size = (layerFactors shapes (i + 1)).size ∧
      ∀ j (hj : j < g.layers[i].children.size), g.layers[i].children[j].size =
        2 * ((layerFactors shapes (i + 1))[j]! + 1)

namespace Accepts

theorem gkrRead (shapes : Array TreeShape) :
    Accepts c n P (GkrView.read shapes) fun g m => g.Below m ∧ g.Shaped shapes := by
  unfold GkrView.read
  refine (privatePairs _).bind fun roots m₁ l₁ => ?_
  refine (arrayMapM (R := fun index (layer : GkrLayerView) m =>
      ((∀ r ∈ layer.rounds, Exts.Below m r) ∧ ∀ ch ∈ layer.children, Exts.Below m ch) ∧
      layer.rounds.size = index ∧ (∀ r ∈ layer.rounds, r.size = layerDegree (layerFactors shapes (index + 1))) ∧
      layer.children.size = (layerFactors shapes (index + 1)).size ∧
      ∀ j (hj : j < layer.children.size), layer.children[j].size = 2 * ((layerFactors shapes (index + 1))[j]! + 1))
    (fun _ _ _ _ h l => ⟨⟨fun r hr a ha k => lt_of_lt_of_le (h.1.1 r hr a ha k) l,
      fun ch hch a ha k => lt_of_lt_of_le (h.1.2 ch hch a ha k) l⟩, h.2⟩) _ fun index _ m _ => ?_).bind
    fun layers m₂ l₂ => pure fun h => ?_
  · refine (arrayMapM_mem (fun (r : Array ExtVar) m => r.size = layerDegree (layerFactors shapes (index + 1)) ∧
      Exts.Below m r) (fun _ _ _ h l => ⟨h.1, fun a ha k => lt_of_lt_of_le (h.2 a ha k) l⟩) _
      fun _ _ _ _ => privateExts _).bind fun rounds m₃ l₃ => ?_
    refine (arrayMapM (R := fun f (ch : Array ExtVar) m => ch.size = 2 * (f + 1) ∧ Exts.Below m ch)
      (fun _ _ _ _ h l => ⟨h.1, fun a ha k => lt_of_lt_of_le (h.2 a ha k) l⟩) _
      fun _ _ _ _ => privateExts _).bind fun children m₄ l₄ => pure fun h => ?_
    refine ⟨⟨fun r hr => ?_, fun ch hch => ?_⟩, by simpa using h.1.2.1, fun r hr => (h.1.2.2 r hr).1, h.2.1, ?_⟩
    · exact fun a ha k => lt_of_lt_of_le ((h.1.2.2 r hr).2 a ha k) l₄
    · obtain ⟨j, hj, rfl⟩ := Array.mem_iff_getElem.mp hch
      exact (h.2.2 j (by rw [← h.2.1]; exact hj) hj).2
    · intro j hj
      rw [(h.2.2 j (by rw [← h.2.1]; exact hj) hj).1, getElem!_pos _ j (by rw [← h.2.1]; exact hj)]
  · refine ⟨⟨fun p hp => ⟨fun k => lt_of_lt_of_le ((h.1.2.2 p hp).1 k) l₂,
      fun k => lt_of_lt_of_le ((h.1.2.2 p hp).2 k) l₂⟩, fun layer hl => ?_⟩, by rw [h.1.2.1],
      by simpa using h.2.1,
      fun i hi => ?_⟩
    · obtain ⟨i, hi, rfl⟩ := Array.mem_iff_getElem.mp hl
      exact (h.2.2 i (by simpa [h.2.1] using hi) hi).1
    · have := (h.2.2 i (by simpa [h.2.1] using hi) hi).2
      simpa using this

theorem leafRead (oracles setupVars : ℕ) :
    Accepts c n P (LeafView.read oracles setupVars) fun leaf m =>
      ((∀ h ∈ leaf.values, h < m) ∧ ∀ d ∈ leaf.path, DigestVar.Below m d) ∧
        leaf.values.size = oracles * 2 ^ Replay.foldVars ∧ leaf.path.size = setupVars + 1 - Replay.foldVars := by
  unfold LeafView.read
  refine (arrayMapM_mem (fun h m => h < m) (fun _ _ _ h l => lt_of_lt_of_le h l) _ fun _ _ _ _ => priv).bind
    fun values m₁ l₁ => (arrayMapM_mem (fun d m => DigestVar.Below m d)
      (fun _ _ _ h l lane => lt_of_lt_of_le (h lane) l)
      _ fun _ _ _ _ => privateDigest).bind fun path m₂ l₂ => pure fun h =>
        ⟨⟨fun x hx => lt_of_lt_of_le (h.1.2.2 x hx) l₂, h.2.2⟩, by simpa using h.1.2.1, by simpa using h.2.1⟩

end Accepts

/-- A WHIR opening view's decoded handles below `m`. -/
def OpeningView.Below (m : ℕ) (o : OpeningView) : Prop :=
  Exts.Below m o.initialOod ∧ (∀ e ∈ o.evals, Exts.Below m e) ∧ Pairs.Below m o.initialSumcheck ∧
    (∀ r ∈ o.rounds, DigestVar.Below m r.root ∧ Exts.Below m r.ood ∧ Pairs.Below m r.sumcheck) ∧
    Exts.Below m o.finalPoly ∧ Pairs.Below m o.finalSumcheck

/-- A WHIR opening view has the counts of `shape` (`readOpening`). -/
def OpeningView.Shaped (shape : Replay.WhirShape) (o : OpeningView) : Prop :=
  o.rounds.size = shape.rounds.size ∧ o.evals.size = shape.batches.size ∧ o.initialOod.size = shape.oodSamples ∧
    (∀ i (h : i < o.evals.size) (h' : i < shape.batches.size), o.evals[i].size = shape.batches[i].size) ∧
    o.initialSumcheck.size = shape.initialSumcheck.rounds ∧
    (∀ i (h : i < o.rounds.size) (h' : i < shape.rounds.size), o.rounds[i].ood.size = shape.rounds[i].oodSamples ∧
      o.rounds[i].sumcheck.size = shape.rounds[i].sumcheck.rounds) ∧
    o.finalPoly.size = shape.finalPolyLen ∧ o.finalSumcheck.size = shape.finalSumcheck.rounds

theorem queryRows.grows (draws width : ℕ) (base : Bool) : Grows (queryRows draws width base) := by grows

namespace Accepts

theorem openingRead (shape : Replay.WhirShape) (rows : OpeningRows) :
    Accepts c n P (OpeningView.read shape rows) fun o m => o.Below m ∧ o.Shaped shape := by
  unfold OpeningView.read
  refine (privateExts _).bind fun initialOod m₁ l₁ => ?_
  refine (arrayMapM (R := fun (b : Replay.WhirBatch) (e : Array ExtVar) m => e.size = b.size ∧ Exts.Below m e)
    (fun _ _ _ _ h l => ⟨h.1, fun a ha k => lt_of_lt_of_le (h.2 a ha k) l⟩) _
    fun _ _ _ _ => privateExts _).bind fun evals m₂ l₂ => ?_
  refine (privatePairs _).bind fun initialSumcheck m₃ l₃ => ?_
  refine (arrayMapM (R := fun (x : Replay.WhirRoundShape × ℕ) (r : RoundView) m =>
      (DigestVar.Below m r.root ∧ Exts.Below m r.ood ∧ Pairs.Below m r.sumcheck) ∧
      r.ood.size = x.1.oodSamples ∧ r.sumcheck.size = x.1.sumcheck.rounds)
    (fun _ _ _ _ h l => ⟨⟨fun lane => lt_of_lt_of_le (h.1.1 lane) l,
      fun a ha k => lt_of_lt_of_le (h.1.2.1 a ha k) l,
      fun p hp => ⟨fun k => lt_of_lt_of_le ((h.1.2.2 p hp).1 k) l,
        fun k => lt_of_lt_of_le ((h.1.2.2 p hp).2 k) l⟩⟩,
      h.2⟩) _ fun x _ m _ => ?_).bind fun rounds m₄ l₄ => ?_
  · refine privateDigest.bind fun root m₅ l₅ => (privateExts _).bind fun ood m₆ l₆ =>
      (Accepts.of_grows (queryRows.grows _ _ _)).bind fun _ m₇ l₇ => (privatePairs _).bind fun sumcheck m₈ l₈ =>
        pure fun h => ⟨⟨fun lane => ?_, fun a ha k => ?_, h.2.2⟩, h.1.1.2.1, h.2.1⟩
    · exact lt_of_lt_of_le (h.1.1.1.2 lane) (l₆.trans (l₇.trans l₈))
    · exact lt_of_lt_of_le (h.1.1.2.2 a ha k) (l₇.trans l₈)
  refine (privateExts _).bind fun finalPoly m₅ l₅ => (Accepts.of_grows (queryRows.grows _ _ _)).bind
    fun _ m₆ l₆ => (privatePairs _).bind fun finalSumcheck m₇ l₇ => pure fun h => ?_
  obtain ⟨⟨⟨⟨⟨⟨⟨_, oodSize, oodBelow⟩, evalsSize, evalsAll⟩, isSize, isBelow⟩, roundsSize, roundsAll⟩,
    fpSize,
    fpBelow⟩, -⟩, fsSize, fsBelow⟩ := h
  have l₂₇ : m₂ ≤ m₇ := l₃.trans (l₄.trans (l₅.trans (l₆.trans l₇)))
  refine ⟨⟨oodBelow.mono (l₂.trans l₂₇), fun e he => ?_,
    isBelow.mono (l₄.trans (l₅.trans (l₆.trans l₇))),
    fun r hr => ?_, fpBelow.mono (l₆.trans l₇), fsBelow⟩,
    by simpa using roundsSize, by simpa using evalsSize, oodSize, fun i h h' => ?_, isSize, fun i h h' => ?_,
    fpSize, fsSize⟩
  · obtain ⟨i, hi, rfl⟩ := Array.mem_iff_getElem.mp he
    exact (evalsAll i (by rw [← evalsSize]; exact hi) hi).2.mono l₂₇
  · obtain ⟨i, hi, rfl⟩ := Array.mem_iff_getElem.mp hr
    obtain ⟨⟨a, b, d⟩, -⟩ := roundsAll i (by simpa [← roundsSize] using hi) hi
    have l : m₅ ≤ m₇ := l₆.trans l₇
    exact ⟨a.mono (l₅.trans l), b.mono (l₅.trans l), d.mono (l₅.trans l)⟩
  · exact (evalsAll i h' h).1
  · have := (roundsAll i (by simpa using h') h).2
    simpa using this

end Accepts

theorem GkrView.Below.mono {m m' : ℕ} {g : GkrView} (h : g.Below m) (l : m ≤ m') : g.Below m' :=
  ⟨h.1.mono l, fun layer hl =>
    ⟨fun r hr => (h.2 layer hl).1 r hr |>.mono l, fun ch hc => (h.2 layer hl).2 ch hc |>.mono l⟩⟩

theorem OpeningView.Below.mono {m m' : ℕ} {o : OpeningView} (h : o.Below m) (l : m ≤ m') : o.Below m' :=
  ⟨h.1.mono l, fun e he => (h.2.1 e he).mono l, h.2.2.1.mono l, fun r hr =>
    ⟨(h.2.2.2.1 r hr).1.mono l, (h.2.2.2.1 r hr).2.1.mono l, (h.2.2.2.1 r hr).2.2.mono l⟩, h.2.2.2.2.1.mono l,
    h.2.2.2.2.2.mono l⟩

/-- A proof view's decoded handles below `m`. -/
def ProofView.Below (m : ℕ) (p : ProofView) : Prop :=
  DigestVar.Below m p.p0 ∧ (∀ h ∈ p.histogram, h < m) ∧ p.early.Below m ∧ Exts.Below m p.earlyValues.runs ∧
    Pairs.Below m p.earlyValues.pairs ∧ ExtVar.Below m p.earlyValues.block ∧ Exts.Below m p.quotient ∧
    (∀ r ∈ p.linear, Exts.Below m r) ∧ Exts.Below m p.finals ∧ (∀ r ∈ p.partials, Exts.Below m r) ∧
    DigestVar.Below m p.p1 ∧
    (∀ leaf ∈ p.leaves, (∀ h ∈ leaf.values, h < m) ∧ ∀ d ∈ leaf.path, DigestVar.Below m d) ∧
    p.late.Below m ∧ Pairs.Below m p.lateValues ∧ p.p1Opening.Below m ∧ p.p0Opening.Below m

/-- A proof view has the sizes that `ProofView.read` gives for `rel`. -/
def ProofView.Shaped (rel : Replay.Relation) (p : ProofView) : Prop :=
  p.histogram.size = 2 * rel.normBound + 1 ∧ p.early.Shaped rel.earlyShapes ∧ p.earlyValues.runs.size = 3 ∧
    p.earlyValues.pairs.size = rel.matrices ∧ p.quotient.size = Replay.D - 1 ∧ p.linear.size = rel.cubeVars ∧
    (∀ r ∈ p.linear, r.size = 2) ∧ p.finals.size = 3 ∧ p.partials.size = 2 ∧
    (∀ r ∈ p.partials, r.size = 2 ^ Replay.foldVars) ∧ p.leaves.size = rel.queries ∧
    (∀ leaf ∈ p.leaves, leaf.values.size = (rel.kappa + Replay.runOracles) * 2 ^ Replay.foldVars ∧
      leaf.path.size = rel.setupVars + 1 - Replay.foldVars) ∧
    p.late.Shaped rel.lateShapes ∧ p.lateValues.size = rel.matrices ∧ p.p1Opening.Shaped rel.p1Whir ∧
    p.p0Opening.Shaped rel.p0Whir

namespace Accepts

theorem proofRead (rel : Replay.Relation) (p1Rows p0Rows : OpeningRows) :
    Accepts c n P (ProofView.read rel p1Rows p0Rows) fun p m => p.Below m ∧ p.Shaped rel := by
  unfold ProofView.read
  refine privateDigest.bind fun p0 m₁ l₁ => ?_
  refine (arrayMapM_mem (fun h m => h < m) (fun _ _ _ h l => lt_of_lt_of_le h l) _ fun _ _ _ _ => priv).bind
    fun histogram m₂ l₂ => ?_
  refine (gkrRead _).bind fun early m₃ l₃ => (privateExts 3).bind fun runs m₄ l₄ =>
    (privatePairs _).bind fun pairs m₅ l₅ => privateExt.bind fun block m₆ l₆ =>
    (privateExts _).bind fun quotient m₇ l₇ => ?_
  refine (arrayMapM_mem (fun (r : Array ExtVar) m => r.size = 2 ∧ Exts.Below m r)
    (fun _ _ _ h l => ⟨h.1, h.2.mono l⟩) _ fun _ _ _ _ => privateExts 2).bind fun linear m₈ l₈ => ?_
  refine (privateExts 3).bind fun finals m₉ l₉ => ?_
  refine (arrayMapM_mem (fun (r : Array ExtVar) m => r.size = 2 ^ Replay.foldVars ∧ Exts.Below m r)
    (fun _ _ _ h l => ⟨h.1, h.2.mono l⟩) _ fun _ _ _ _ => privateExts _).bind fun partials m₁₀ l₁₀ => ?_
  refine privateDigest.bind fun p1 m₁₁ l₁₁ => ?_
  refine (arrayMapM_mem (fun (leaf : LeafView) m => ((∀ h ∈ leaf.values, h < m) ∧
      ∀ d ∈ leaf.path, DigestVar.Below m d) ∧
      leaf.values.size = (rel.kappa + Replay.runOracles) * 2 ^ Replay.foldVars ∧
      leaf.path.size = rel.setupVars + 1 - Replay.foldVars)
    (fun _ _ _ h l => ⟨⟨fun x hx => lt_of_lt_of_le (h.1.1 x hx) l, fun d hd => (h.1.2 d hd).mono l⟩, h.2⟩) _
    fun _ _ _ _ => leafRead _ _).bind fun leaves m₁₂ l₁₂ => ?_
  refine (gkrRead _).bind fun late m₁₃ l₁₃ => (privatePairs _).bind fun lateValues m₁₄ l₁₄ =>
    (openingRead _ _).bind fun p1Opening m₁₅ l₁₅ => (openingRead _ _).bind fun p0Opening m₁₆ l₁₆ =>
    pure fun h => ?_
  obtain ⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨_, hp0⟩, hhs, hh⟩, he⟩, hrs, hr⟩, hps, hp⟩, hb⟩, hqs, hq⟩, hls, hl⟩, hfs, hf⟩,
    hpas, hpa⟩,
    hp1⟩, hlvs, hlv⟩, hla⟩, hlts, hlt⟩, ho1⟩, ho0⟩ := h
  have u₂ : m₂ ≤ m₁₆ := by omega
  have u₁₂ : m₁₂ ≤ m₁₆ := by omega
  refine ⟨⟨hp0.mono (by omega), fun x hx => lt_of_lt_of_le (hh x hx) u₂, he.1.mono (by omega),
    hr.mono (by omega), hp.mono (by omega), hb.mono (by omega), hq.mono (by omega),
    fun r hr' => (hl r hr').2.mono (by omega), hf.mono (by omega), fun r hr' => (hpa r hr').2.mono (by omega),
    hp1.mono (by omega), fun leaf hleaf => ⟨fun x hx => lt_of_lt_of_le ((hlv leaf hleaf).1.1 x hx) u₁₂,
      fun d hd => ((hlv leaf hleaf).1.2 d hd).mono (by omega)⟩, hla.1.mono (by omega), hlt.mono (by omega),
    ho1.1.mono (by omega), ho0.1⟩,
    by simpa using hhs, he.2, hrs, hps, hqs, by simpa using hls, fun r hr' => (hl r hr').1, hfs,
    by simpa using hpas, fun r hr' => (hpa r hr').1, by simpa using hlvs, fun leaf hleaf => (hlv leaf hleaf).2,
    hla.2, hlts, ho1.2, ho0.2⟩

end Accepts

/-! ## The handoff -/

/-- Sixteen lanes, each below `m`. -/
def LanesBelow (m : ℕ) (state : Array Var) : Prop := state.size = 16 ∧ ∀ i < 16, state[i]! < m

theorem LanesBelow.mono {m m' : ℕ} {s : Array Var} (h : LanesBelow m s) (l : m ≤ m') : LanesBelow m' s :=
  ⟨h.1, fun i hi => lt_of_lt_of_le (h.2 i hi) l⟩

theorem LanesBelow.set {m : ℕ} {s : Array Var} (h : LanesBelow m s) (j : ℕ) {x : Var} (hx : x < m) :
    LanesBelow m (s.set! j x) := by
  refine ⟨by simp [h.1], fun i hi => ?_⟩
  rw [getElem!_set! _ _ _ _ (by rw [h.1]; exact hi)]
  split
  · exact hx
  · exact h.2 i hi

/-- A permutation's outputs are sixteen new handles. -/
theorem permute_lanes (state : Fin 16 → Var) :
    Accepts c n P (Build.permute state) fun out m => LanesBelow m (Array.ofFn out) :=
  (Accepts.emit (.permute state)).map fun _ m e => ⟨by simp, fun i hi => by
    rw [getElem!_pos _ i (by simpa using hi), Array.getElem_ofFn, e.2]
    simp [Build.handles, Instr.outputs, e.1]
    omega⟩

theorem absorb_below {t : FoldTranscript} {words : Array Var} (ht : P → LanesBelow n t.state) :
    Accepts c n P (t.absorb words) fun r m => LanesBelow m r.state := by
  unfold FoldTranscript.absorb
  dsimp only
  rw [← Array.forIn_toList]
  refine ((Accepts.loop (I := fun _ acc m => LanesBelow m acc) fun k hk acc m le => ?_).pre fun h => ⟨h, ht h⟩).bind
    fun r m _ => Accepts.pure fun h => h.2
  rw [← Array.forIn_toList]
  refine (((Accepts.loop (I := fun _ acc m' => LanesBelow m' acc) fun j hj acc' m' le' => ?_).pre
    fun h => ⟨h, h.2⟩).bind fun r m' l' => ?_)
  · exact (Accepts.emit _).bind fun sum m'' l'' => Accepts.pure fun h => ⟨rfl, (h.1.2.mono l'').set _ (by
      rw [h.2.1, h.2.2]; simp [Instr.outputs])⟩
  · exact (permute_lanes _).bind fun out m'' l'' => Accepts.pure fun h => ⟨rfl, h.2⟩

theorem handoff_below (state : Fin 16 → Var) (hs : P → ∀ lane, state lane < n) :
    Accepts c n P (FoldTranscript.fromState state).handoff fun seed m => ∀ lane, seed lane < m := by
  unfold FoldTranscript.handoff FoldTranscript.absorbConstants FoldTranscript.squeezeDigest
  have lanes : P → LanesBelow n (FoldTranscript.fromState state).state := fun h => by
    refine ⟨by simp [FoldTranscript.fromState], fun i hi => ?_⟩
    rw [getElem!_pos _ i (by simpa [FoldTranscript.fromState] using hi)]
    simpa [FoldTranscript.fromState] using hs h ⟨i, hi⟩
  refine ((Accepts.of_grows (c := c) (P := P) (Grows.mapM_array fun _ => Grows.emit _)).bind fun words m₁ l₁ =>
    absorb_below fun h => (lanes h.1).mono l₁).bind fun t m₂ l₂ => (permute_lanes _).bind fun _ m₃ l₃ =>
      Accepts.pure fun h lane => ?_
  show t.state[lane.val]! < m₃
  exact lt_of_lt_of_le (h.1.2.2 lane (by omega)) l₃

end CompressionLean.Circuit
