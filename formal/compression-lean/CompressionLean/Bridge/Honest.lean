import CompressionLean.Bridge.Late

/-!
Owns the bridge of the honest-fold stage (`Replay.honestStage`) to the
model's `honestGame`. The replay's matrix step is the extension of the
setup's matrix column (`toK_matrixStep`), so its run value is the model's.
A successful honest-fold replay is a path of `honestGame` that draws the
replay's `μ`, `α` and query words, receives the partials and the P1
candidates, and ends in the model's `HonestOut` of the replay's values,
with both partial checks (`honest_reaches`). The setup queries' leaf checks
feed the P1 openings, not this path (`Openings.lean`).
-/

namespace CompressionLean.Bridge

open Polynomial
open CompressionLean.Replay CompressionLean.Layer1

variable {K : Type} [Field K] [Algebra Gl K] [Fintype K] [ChallengeField K]

local notation "ι" => algebraMap Gl K

theorem toK_matrixStep (rel : Replay.Relation) (S : Setup Gl) (fits : S.Fits (shapeOf rel))
    (checked : (structureOf rel.key.layout).Checked (shapeOf rel))
    (describes : (structureOf rel.key.layout).Describes (shapeOf rel) S)
    (bounds : rel.matrices ≤ rel.key.layout.runBounds.size) (p : Array Ext) :
    (matrixStep rel.key.layout p).toK (K := K) = mle (pointK p) (matrixColumn S) := by
  have same : rel.matrices = rel.key.layout.matrices := rfl
  rw [matrixStep, toK_step, ← matrixStep_eq (shapeOf rel) S (structureOf rel.key.layout) fits checked describes]
  congr 1
  apply List.ext_getElem
  · simp only [List.length_map, List.length_zip, Array.length_toList, Array.size_extract, List.length_range,
      shapeOf]
    omega
  intro j h1 h2
  have hj : j < rel.matrices := by simpa [shapeOf] using h2
  simp only [List.getElem_map, List.getElem_zip, Array.getElem_toList, Array.getElem_extract, List.getElem_range]
  rw [Ext.toK_ofGl, Word.toF_ofNat, map_natCast]
  simp only [structureOf]
  rw [getElem!_pos rel.key.layout.runBounds j (by omega)]
  simp

theorem pointK_padded (n : ℕ) (p : Array Ext) : pointK (K := K) (Replay.padded n p) = Layer1.padded n (pointK p) := by
  simp [Replay.padded, Layer1.padded, pointK]

/-- The model's partials of the proof's. -/
def partialsOf (proof : Replay.Proof) (g : Fin 2) : ℕ → K := tableK (proof.partials[(g : ℕ)]!)

/-- The model's messages of the replay's honest fold: the partials and the
P1 candidates. -/
def honestMsgs (proof : Replay.Proof) {L1 : ℕ} (p1 : {l : List (P1Tables K) // l.length ≤ L1}) :
    List (Σ M : Type, M) :=
  [⟨Fin 2 → ℕ → K, partialsOf proof⟩, ⟨{l : List (P1Tables K) // l.length ≤ L1}, p1⟩]

/-- The model's query words of the replay's. -/
def wordsOf (queries : ℕ) (ws : Array Word) : Fin queries → Fin goldilocks :=
  fun j => ⟨(ws[(j : ℕ)]!).word.toNat, word_lt _⟩

/-- The relation's sizes the honest fold reads. -/
structure HonestSizes (rel : Replay.Relation) : Prop where
  bounds : rel.matrices ≤ rel.key.layout.runBounds.size
  setup : 3 ≤ rel.setupVars

/-- What the openings read of a successful honest-fold replay. -/
structure HonestRun (rel : Replay.Relation) (proof : Replay.Proof) (early : Replay.EarlyOut) (lin : Replay.LinearOut)
    (honest : Replay.HonestOut) : Prop where
  alpha : honest.alpha = #[honest.alpha[0]!, honest.alpha[1]!, honest.alpha[2]!]
  words : honest.words.size = rel.queries
  queries : honest.queries = honest.words.map fun word => word.toNat % 2 ^ (rel.setupVars + 1 - foldVars)
  queried : ((honest.queries.zip proof.leaves).toList.foldlM (fun queried pair => do
    pure (queried ++ (← setupCheck rel.key.root rel.setupVars pair.1 pair.2
      #[(Array.range rel.kappa).map lin.mixing.commitment, (Array.range runOracles).map honest.mu.pow]
      honest.alpha))) #[]) = .ok honest.queried

theorem toK_embedded (p : Array Ext) (n : ℕ) (block : Ext) :
    ((p.extract n p.size).foldl (fun total x => total * oneMinus x) block).toK (K := K) =
      block.toK * (((pointK p).drop n).map fun x => 1 - x).prod := by
  rw [← Array.foldl_toList, ← pointK_extract_drop, pointK]
  generalize (p.extract n p.size).toList = l
  induction l generalizing block with
  | nil => simp
  | cons x rest ih =>
      rw [List.foldl_cons, ih, Ext.toK_mul, toK_oneMinus]
      simp only [List.map_cons, List.prod_cons]
      ring

/-- The replay's run value is the model's, at the model's early output. -/
theorem toK_runValue (rel : Replay.Relation) (proof : Replay.Proof) (S : Setup Gl) (fits : S.Fits (shapeOf rel))
    (checked : (structureOf rel.key.layout).Checked (shapeOf rel))
    (describes : (structureOf rel.key.layout).Describes (shapeOf rel) S) (sizes : HonestSizes rel)
    (cands : List (P0Tables Gl)) {early : Replay.EarlyOut} {mc : Layer1.GkrClaims K}
    (agree : ClaimsAgree rel.earlyShapes.size early.gkr mc) (values : early.values = proof.earlyValues)
    (openings : early.openings = earlyOpeningsOf rel.key.layout early.challenges early.gkr proof.earlyValues) (mu : Ext) :
    (#[early.openings.coefClaim, early.openings.slotClaim, early.openings.rowClaim].foldl
        (fun total value => value + mu * total)
        ((early.openings.runPoint.extract rel.key.layout.slotVariables early.openings.runPoint.size).foldl
          (fun total x => total * oneMinus x) early.values.block)).toK (K := K) =
      runValue (shapeOf rel) S
        (⟨cands, histOf proof, challengesOf early.challenges, mc, valuesOf early.values⟩ : Layer1.EarlyOut Gl K)
        mu.toK := by
  have small : ∀ u, u < 5 → u < rel.earlyShapes.size := fun u lt => by rw [earlyShapes_size]; exact lt
  rw [List.foldl_toArray' _ _ _ (by simp)]
  simp only [List.foldl_cons, List.foldl_nil, Ext.toK_add, Ext.toK_mul]
  rw [toK_embedded]
  rw [openings]
  simp only [earlyOpeningsOf, runValue, Layer1.EarlyOut.rowClaim, Layer1.EarlyOut.slotClaim,
    Layer1.EarlyOut.coefClaim, Layer1.EarlyOut.embeddedBlock, Layer1.EarlyOut.runPoint, valuesOf, challengesOf,
    pairK, Ext.toK_sub, Ext.toK_scale, Word.toF_ofNat, Nat.cast_pow, Nat.cast_ofNat, map_pow, map_ofNat,
    toK_matrixStep rel S fits checked describes sizes.bounds, agree.leafPoint 3 (small 3 (by norm_num)),
    agree.leafValues 3 (small 3 (by norm_num)), values, shapeOf]
  ring

variable [DecidableEq K]

/-- A successful honest-fold replay is a path of `honestGame` that receives
the partials and the P1 candidates, draws the replay's `μ`, `α` and query
words and ends in the model's `HonestOut` of the replay's values; along it
both partial checks hold. -/
theorem honest_reaches (rel : Replay.Relation) (proof : Replay.Proof) (shape : ProofShape rel proof) (S : Setup Gl)
    (fits : S.Fits (shapeOf rel)) (checked : (structureOf rel.key.layout).Checked (shapeOf rel))
    (describes : (structureOf rel.key.layout).Describes (shapeOf rel) S) (sizes : HonestSizes rel)
    (cands : List (P0Tables Gl)) {early : Replay.EarlyOut} {mc : Layer1.GkrClaims K}
    (agree : ClaimsAgree rel.earlyShapes.size early.gkr mc)
    (lengths : ∀ u, u < rel.earlyShapes.size → (early.gkr.leafPoint[u]!).size = (rel.earlyShapes[u]!).depth)
    (values : early.values = proof.earlyValues)
    (openings : early.openings = earlyOpeningsOf rel.key.layout early.challenges early.gkr proof.earlyValues)
    {lin : Replay.LinearOut} (linRun : LinearRun (K := K) rel proof lin)
    (L1 : ℕ) (p1 : List (P1Tables K)) (short : p1.length ≤ L1) (q : ℕ → K)
    {t t' : Replay.Transcript} {honest : Replay.HonestOut} (run : Runs (honestStage rel proof early lin) t honest t') :
    ∃ ds : List Draw, t'.draws.toList = t.draws.toList ++ ds ∧ HonestRun rel proof early lin honest ∧
      Game.Reaches (honestGame (shapeOf rel) S L1 (2 ^ (rel.setupVars - 2)) goldilocks q
          (⟨cands, histOf proof, challengesOf early.challenges, mc, valuesOf early.values⟩ : Layer1.EarlyOut Gl K)
          (linearOutOf proof lin))
        (honestMsgs proof ⟨p1, short⟩) (ds.map (drawn (K := K) rel.queries))
        (some ⟨honest.mu.toK, partialsOf proof, pointK honest.alpha, p1,
          lowBits (2 ^ (rel.setupVars - 2)) (wordsOf rel.queries honest.words)⟩) := by
  have small : ∀ u, u < 5 → u < rel.earlyShapes.size := fun u lt => by rw [earlyShapes_size]; exact lt
  have runSize : (early.gkr.leafPoint[3]!).size = rel.key.layout.runVariables := lengths 3 (small 3 (by norm_num))
  have setupRun : rel.key.layout.runVariables ≤ rel.setupVars := le_max_right _ _
  have setupCube : rel.blockVars + laneVars ≤ rel.setupVars := le_max_left _ _
  have laneSize : lin.lanePoint.size = laneVars := by
    have := congrArg List.length linRun.lanePoint
    rwa [pointK_length, lanePoint_length] at this
  have keySize : (Replay.padded rel.setupVars (lin.lanePoint ++ lin.blockPoint)).size = rel.setupVars := by
    simp only [Replay.padded, Array.size_append, Array.size_replicate, laneSize, linRun.blockSize]
    omega
  have runPointSize : early.openings.runPoint.size = rel.key.layout.runVariables := by
    rw [openings]
    exact runSize
  have runPaddedSize : (Replay.padded rel.setupVars early.openings.runPoint).size = rel.setupVars := by
    simp only [Replay.padded, Array.size_append, Array.size_replicate, runPointSize]
    omega
  have three : min foldVars rel.setupVars - 0 = foldVars := by
    have := sizes.setup
    simp only [foldVars] at this ⊢
    omega
  have partialSize : ∀ g, g < 2 → (proof.partials[g]!).size = 2 ^ foldVars := fun g lt =>
    shape.partialSizes _ (by
      rw [getElem!_pos proof.partials g (by rw [shape.partials]; exact lt)]
      exact Array.getElem_mem_toList _)
  unfold honestStage at run
  obtain ⟨mu, t1, h1, run⟩ := run_bind run
  obtain ⟨_, t2, h2, run⟩ := run_bind run
  obtain ⟨a0, t3, h3, run⟩ := run_bind run
  obtain ⟨a1, t4, h4, run⟩ := run_bind run
  obtain ⟨a2, t5, h5, run⟩ := run_bind run
  obtain ⟨_, t6, h6, run⟩ := run_bind run
  obtain ⟨_, t7, h7, run⟩ := run_bind run
  obtain ⟨_, t8, h8, run⟩ := run_bind run
  obtain ⟨_, t9, h9, run⟩ := run_bind run
  obtain ⟨words, t10, h10, run⟩ := run_bind run
  obtain ⟨queried, t11, h11, run⟩ := run_bind run
  obtain ⟨rfl, rfl⟩ := run_pure run
  obtain ⟨keyCheck, rfl⟩ := run_ensure h6
  obtain ⟨runCheck, rfl⟩ := run_ensure h7
  obtain ⟨queriedOk, rfl⟩ := run_liftExcept h11
  obtain ⟨wordDraws, wordSize⟩ := run_sampleWords h10
  have checks : mle ((keyPoint (shapeOf rel) (linearOutOf proof lin)).take 3) (partialsOf proof 0) =
        (linearOutOf (K := K) proof lin).keyValue ∧
      mle ((Layer1.padded (shapeOf rel).setupVars
          (Layer1.EarlyOut.runPoint (⟨cands, histOf proof, challengesOf early.challenges, mc, valuesOf early.values⟩ :
            Layer1.EarlyOut Gl K))).take 3) (partialsOf proof 1) =
        runValue (shapeOf rel) S
          (⟨cands, histOf proof, challengesOf early.challenges, mc, valuesOf early.values⟩ : Layer1.EarlyOut Gl K)
          mu.toK := by
    constructor
    · have key := Ext.toK_of_beq (K := K) keyCheck
      rw [toK_combine _ _ (by rw [partialSize 0 (by norm_num), Array.size_extract, keySize, three]),
        pointK_extract_take, pointK_padded, pointK_append, linRun.lanePoint] at key
      exact key
    · have value := Ext.toK_of_beq (K := K) runCheck
      rw [toK_combine _ _ (by rw [partialSize 1 (by norm_num), Array.size_extract, runPaddedSize, three]),
        pointK_extract_take, pointK_padded,
        toK_runValue rel proof S fits checked describes sizes cands agree values openings mu] at value
      rw [Layer1.EarlyOut.runPoint, agree.leafPoint 3 (small 3 (by norm_num)), ← value]
      rw [openings]
      rfl
  refine ⟨[.ext mu, .ext a0, .ext a1, .ext a2, .words words], ?_, ⟨rfl, wordSize, rfl, queriedOk⟩, ?_⟩
  · rw [wordDraws, run_observeAll h9, run_observeAll h8, run_sampleExt h5, run_sampleExt h4, run_sampleExt h3,
      run_observeExts h2, run_sampleExt h1]
    simp
  · unfold honestGame honestMsgs
    simp only [List.map_cons, List.map_nil, drawn_ext]
    apply Game.Reaches.draw mu.toK
    apply Game.Reaches.recv (partialsOf proof)
    refine Game.Reaches.bind (ms₁ := []) (ds₁ := [⟨K, a0.toK⟩, ⟨K, a1.toK⟩, ⟨K, a2.toK⟩])
      (a := pointK #[a0, a1, a2]) ?_ ?_
    · apply Game.Reaches.draw a0.toK
      apply Game.Reaches.draw a1.toK
      apply Game.Reaches.draw a2.toK
      exact Game.Reaches.pure _
    · rw [if_pos checks]
      apply Game.Reaches.recv (⟨p1, short⟩ : {l : List (P1Tables K) // l.length ≤ L1})
      exact Game.Reaches.draw (wordsOf rel.queries words) (Game.Reaches.pure _)

end CompressionLean.Bridge
