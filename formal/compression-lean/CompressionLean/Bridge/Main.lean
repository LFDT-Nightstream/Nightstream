import CompressionLean.Bridge.Openings

/-!
Owns the bridge of the whole replay (`Replay.verify`) to the model's
`layer1Game`. A run that the replay accepts is a path of `layer1Game`: the
replay's draws before WHIR's are the path's challenges, and the path ends in
the model's transcript of the replay's values (`replay_accepts_implies_path`).
This holds under WHIR at the points that the replay returns (`WhirOpens`),
Merkle binding (`MerkleBinds`) and the relation facts that the replay does
not check (`RelationFacts`).

With `layer1Game_avoids`, a path that avoids every bad set ends in a
statement that a P0 candidate satisfies (`replay_accepts_sound`). The other
paths are the event that `layer1_sound` bounds. The bridge's game meets two
more premises of `layer1_sound`: distinct query points (`queryAt_distinct`)
and a monic `Φ81` of degree `D` (`phi_monic`, `phi_natDegree`).
-/

namespace CompressionLean.Bridge

open Polynomial
open CompressionLean.Replay CompressionLean.Layer1

/-! ## The game's parameters -/

theorem phi_natDegree : phi.natDegree = 54 := by
  unfold phi
  compute_degree!

theorem phi_monic : phi.Monic := by
  unfold phi
  monicity!

theorem phi_ne_one : phi ≠ 1 := by
  intro one
  have degree := phi_natDegree
  rw [one, natDegree_one] at degree
  omega

/-- Plonky3's generator of order `2^m` has order `2^m`. -/
theorem twoAdic_order (m : ℕ) (high : m ≤ 32) : orderOf (twoAdic m) = 2 ^ m := by
  cases m with
  | zero => rw [twoAdic, Nat.sub_zero, generator_pow_32, orderOf_one, pow_zero]
  | succ n =>
      apply orderOf_eq_prime_pow
      · rw [twoAdic, ← pow_mul, ← pow_add, show 32 - (n + 1) + n = 31 by omega, generator_pow_31]
        exact neg_one_ne_one
      · rw [twoAdic, ← pow_mul, ← pow_add, show 32 - (n + 1) + (n + 1) = 32 by omega, generator_pow_32]

variable {K : Type} [Field K] [Algebra Fp K] [Fintype K] [ChallengeField K]

omit [Fintype K] [ChallengeField K] in
/-- Distinct leaves have distinct query points. -/
theorem queryAt_distinct (rel : Replay.Relation) (high : rel.setupVars + 1 ≤ 32) :
    ∀ i j, i < 2 ^ (rel.setupVars - 2) → j < 2 ^ (rel.setupVars - 2) →
      queryAt (K := K) rel i = queryAt rel j → i = j := by
  intro i j hi hj same
  have base := (algebraMap Fp K).injective same
  unfold queryPoint at base
  have powers := mul_left_cancel₀ (pow_ne_zero 8 seven_ne_zero) base
  have order := twoAdic_order (rel.setupVars - 2) (by omega)
  exact pow_injOn_Iio_orderOf (show i < orderOf (twoAdic (rel.setupVars - 2)) by rw [order]; exact hi)
    (show j < orderOf (twoAdic (rel.setupVars - 2)) by rw [order]; exact hj) powers

variable [DecidableEq K]

/-- The relation facts that the replay does not check: sizes, and the key's
structure for the setup `S`. -/
structure RelationFacts (rel : Replay.Relation) (S : Setup Fp) : Prop where
  rowFits : rel.key.layout.rowVariables ≤ rel.pointVariables
  matricesFit : rel.matrices ≤ 2 ^ rel.matrixVars
  bits : EvalKBits rel
  publicFits : rel.publicBlocks ≤ 2 ^ rel.blockVars
  sizes : HonestSizes rel
  high : rel.setupVars + 1 ≤ 32
  fits : S.Fits (shapeOf rel)
  checked : (structureOf rel.key.layout).Checked (shapeOf rel)
  describes : (structureOf rel.key.layout).Describes (shapeOf rel) S

/-- The model's messages of the replay's layer-1 proof: the statement, then
the early, linear, honest-fold and late messages. -/
def layer1Msgs (rel : Replay.Relation) (st : Replay.Statement) (proof : Replay.Proof) {L0 L1 : ℕ}
    (cands : {l : List (P0Tables Fp) // l.length ≤ L0}) (p1 : {l : List (P1Tables K) // l.length ≤ L1}) :
    List (Σ M : Type, M) :=
  ⟨Layer1.Statement Fp, statementOf st⟩ :: (earlyMsgs (K := K) rel proof cands ++
    (linearMsgs (K := K) rel proof ++ (honestMsgs proof p1 ++ lateMsgs (K := K) rel proof)))

/-- **The bridge.** Take a run that the replay accepts, with WHIR's guarantee
at the points that it returns. Some path of `layer1Game` receives the
replay's statement, proof and candidate lists, draws the replay's
challenges before WHIR's own and ends in a model transcript of the replay's
statement and the prover's P0 candidates. -/
theorem replay_accepts_implies_path (rel : Replay.Relation) (S : Setup Fp) (facts : RelationFacts rel S)
    (seed : Array Gl) (st : Replay.Statement) (proof : Replay.Proof) (merkle : MerkleBinds rel S proof.leaves)
    (L0 L1 : ℕ) (cands : List (P0Tables Fp)) (short0 : cands.length ≤ L0) (p1 : List (P1Tables K))
    (short1 : p1.length ≤ L1) {t t' : Replay.Transcript} {points : OpeningPoints}
    (run : Runs (verify rel seed st proof) t points t') (whir : WhirOpens (shapeOf rel) proof points cands p1) :
    ∃ (tr : Transcript Fp K (shapeOf rel).queries (2 ^ (rel.setupVars - 2))) (ds rest : List Draw),
      t'.draws.toList = t.draws.toList ++ ds ++ rest ∧ tr.statement = statementOf st ∧
      tr.early.candidates = cands ∧
      Game.Reaches (layer1Game (shapeOf rel) phi barOf S L0 L1 (2 ^ (rel.setupVars - 2)) goldilocks (queryAt rel))
        (layer1Msgs rel st proof ⟨cands, short0⟩ ⟨p1, short1⟩) (ds.map (drawn (K := K) rel.queries)) (some tr) := by
  unfold verify at run
  obtain ⟨_, t1, h1, r1⟩ := run_bind run
  obtain ⟨_, t2, h2, r2⟩ := run_bind r1
  obtain ⟨_, t3, h3, r3⟩ := run_bind r2
  obtain ⟨_, t4, h4, r4⟩ := run_bind r3
  obtain ⟨_, t5, h5, r5⟩ := run_bind r4
  obtain ⟨early, t6, h6, r6⟩ := run_bind r5
  obtain ⟨lin, t7, h7, r7⟩ := run_bind r6
  obtain ⟨honest, t8, h8, r8⟩ := run_bind r7
  obtain ⟨late, t9, h9, r9⟩ := run_bind r8
  obtain ⟨_, t10, h10, r10⟩ := run_bind r9
  obtain ⟨_, t11, h11, r11⟩ := run_bind r10
  obtain ⟨_, t12, h12, r12⟩ := run_bind r11
  obtain ⟨_, t13, h13, h14⟩ := run_bind r12
  clear run r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12
  obtain ⟨rfl, rfl⟩ := run_pure h14
  obtain ⟨fits, rfl⟩ := run_ensure h1
  obtain ⟨readOk, rfl⟩ := run_liftExcept h2
  obtain ⟨p1ok, rfl⟩ := run_liftExcept h11
  obtain ⟨p0ok, rfl⟩ := run_liftExcept h13
  have shape := readProof_ok readOk
  obtain ⟨dsE, drawsE, mc, agree, lengths, values, zLeaf, openings, reachE⟩ :=
    early_reaches (K := K) rel st proof shape fits facts.rowFits facts.matricesFit S L0 cands short0 rel.queries h6
  obtain ⟨dsL, drawsL, linRun, reachL⟩ :=
    linear_reaches (K := K) rel st proof shape fits facts.bits facts.publicFits S cands rel.queries h7
  obtain ⟨dsH, drawsH, honestRun, reachH⟩ := honest_reaches (K := K) rel proof shape S facts.fits facts.checked
    facts.describes facts.sizes cands agree lengths values openings linRun L1 p1 short1 (queryAt rel) h8
  obtain ⟨dsT, drawsT, mcL, agreeL, -, latePairs, lateOpenings, reachT⟩ := late_reaches (K := K) rel proof shape S
    facts.fits facts.checked facts.describes linRun cands p1 rel.queries h9
  obtain ⟨dsW1, drawsW1⟩ := whirTranscript_grows _ _ _ _ _ h10
  obtain ⟨dsW0, drawsW0⟩ := whirTranscript_grows _ _ _ _ _ h12
  have opens := opens_of_replay (K := K) rel st proof shape S merkle facts.sizes.setup facts.high cands p1 agree
    values zLeaf openings linRun honestRun agreeL latePairs lateOpenings p1ok p0ok whir
  refine ⟨transcriptOf rel st proof cands p1 early mc lin honest late mcL, dsE ++ dsL ++ dsH ++ dsT, dsW1 ++ dsW0,
    ?_, rfl, rfl, ?_⟩
  · rw [drawsW0, drawsW1, drawsT, drawsH, drawsL, drawsE, run_observeAll h5, run_observeAll h4, run_observeAll h3]
    simp only [List.append_assoc]
  · unfold layer1Game layer1Msgs
    apply Game.Reaches.recv (statementOf st)
    simp only [List.map_append, List.append_assoc]
    refine Game.Reaches.bind reachE ?_
    refine Game.Reaches.bind reachL ?_
    refine Game.Reaches.bind reachH ?_
    refine Game.Reaches.congr (Game.Reaches.bind reachT ?_) (List.append_nil _) (List.append_nil _)
    dsimp only
    split_ifs with holds
    · exact Game.Reaches.pure _
    · exact absurd opens holds

/-- **The bridge, composed with the model.** A run that the replay accepts,
with WHIR's guarantee at the points that it returns, is a path of
`layer1Game` that draws the replay's challenges and accepts. If the path
avoids every bad set, a P0 candidate is a CE(B) witness for the replay's
statement. The paths that do not avoid them are the event that
`layer1_sound` bounds. -/
theorem replay_accepts_sound (rel : Replay.Relation) (S : Setup Fp) (facts : RelationFacts rel S) (N : ℕ)
    (sizes : (shapeOf rel).Sizes N) (cast : ∀ a b : ℕ, a < N → b < N → (a : K) = b → a = b) (seed : Array Gl)
    (st : Replay.Statement) (proof : Replay.Proof) (merkle : MerkleBinds rel S proof.leaves) (L0 L1 : ℕ)
    (cands : List (P0Tables Fp)) (short0 : cands.length ≤ L0) (p1 : List (P1Tables K)) (short1 : p1.length ≤ L1)
    {t t' : Replay.Transcript} {points : OpeningPoints} (run : Runs (verify rel seed st proof) t points t')
    (whir : WhirOpens (shapeOf rel) proof points cands p1) :
    ∃ ds rest : List Draw, t'.draws.toList = t.draws.toList ++ ds ++ rest ∧
      ∃ p : Game.Pos (layer1Game (shapeOf rel) phi barOf S L0 L1 (2 ^ (rel.setupVars - 2)) goldilocks
          (queryAt (K := K) rel)),
        Game.Pos.draws _ p = ds.map (drawn (K := K) rel.queries) ∧
        (∃ tr, Game.Pos.node _ p = .pure (some tr)) ∧
        (Game.Pos.Avoids _ p → ∃ w ∈ cands, Relation (shapeOf rel) phi barOf S (statementOf st) w.z) := by
  obtain ⟨tr, ds, rest, draws, statement, candidates, p, node, pathDraws, -⟩ :=
    replay_accepts_implies_path rel S facts seed st proof merkle L0 L1 cands short0 p1 short1 run whir
  refine ⟨ds, rest, draws, p, pathDraws, ⟨tr, node⟩, fun clean => ?_⟩
  have witness := Game.Pos.sound (layer1Game_avoids (shapeOf rel) phi barOf S L0 L1 (2 ^ (rel.setupVars - 2))
    goldilocks (queryAt rel) N sizes cast facts.fits phi_monic phi_ne_one) p (some tr) clean node tr rfl
  rw [candidates, statement] at witness
  exact witness

end CompressionLean.Bridge
