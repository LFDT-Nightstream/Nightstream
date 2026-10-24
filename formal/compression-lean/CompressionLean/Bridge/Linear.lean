import CompressionLean.Bridge.EvalK

/-!
Owns the bridge of the linear stage (`Replay.linearStage`) to the model's
`linearGame`. A successful linear replay is a path of `linearGame`: it draws
the replay's `λ`, `ζ` and sum-check challenges, receives the quotient and
the finals, and ends in the model's `LinearOut` of the replay's values,
whose `LinearCheck` the replay's checks give (`linear_reaches`).
-/

namespace CompressionLean.Bridge

open Polynomial
open CompressionLean.Replay CompressionLean.Layer1

variable {K : Type} [Field K] [Algebra Gl K] [Fintype K] [ChallengeField K]

/-- The model's linear output of the replay's. -/
def linearOutOf (proof : Replay.Proof) (lin : Replay.LinearOut) : Layer1.LinearOut K :=
  ⟨lin.lam.toK, tableK proof.quotient, lin.zeta.toK, pointK lin.blockPoint, lin.middle.toK, pointK lin.lanePart,
    lin.last.toK, lin.keyValue.toK, lin.omegaValue.toK, lin.zValue.toK⟩

/-- The model's messages of the replay's linear stage: the quotient, the
block and lane sum-check rounds, and the finals. -/
def linearMsgs (rel : Replay.Relation) (proof : Replay.Proof) : List (Σ M : Type, M) :=
  ⟨ℕ → K, tableK proof.quotient⟩ ::
    (sumcheckMsgs (K := K) 1 (proof.linear.extract 0 rel.blockVars).toList ++
      (sumcheckMsgs (K := K) 1 (proof.linear.extract rel.blockVars proof.linear.size).toList ++
        [⟨K × K × K, (proof.finals[0]!.toK, proof.finals[1]!.toK, proof.finals[2]!.toK)⟩]))

/-- What the later stages read of a successful linear replay. -/
structure LinearRun (rel : Replay.Relation) (proof : Replay.Proof) (lin : Replay.LinearOut) : Prop where
  mixing : lin.mixing = ⟨lin.lam, rel.kappa, rel.matrices⟩
  tau : lin.tau = Replay.tau lin.zeta
  lanePoint : pointK (K := K) lin.lanePoint = Layer1.lanePoint lin.zeta.toK laneVars
  blockSize : lin.blockPoint.size = rel.blockVars
  laneSize : lin.lanePart.size = laneVars
  keyValue : lin.keyValue = proof.finals[0]!
  omegaValue : lin.omegaValue = proof.finals[1]!
  zValue : lin.zValue = proof.finals[2]!

theorem mem_extract {a : Array (Array Ext)} {i j : ℕ} {m : Array Ext} (member : m ∈ (a.extract i j).toList) :
    m ∈ a.toList := by
  rw [Array.toList_extract, List.extract_eq_take_drop] at member
  exact List.mem_of_mem_drop (List.mem_of_mem_take member)

variable [DecidableEq K]

/-- A successful linear replay is a path of `linearGame` that receives the
proof's messages, draws the replay's challenges and ends in the model's
`LinearOut` of the replay's values; along it `LinearCheck` holds. -/
theorem linear_reaches (rel : Replay.Relation) (st : Replay.Statement) (proof : Replay.Proof)
    (shape : ProofShape rel proof) (fits : st.fits rel = true) (bits : EvalKBits rel)
    (publicFits : rel.publicBlocks ≤ 2 ^ rel.blockVars) (S : Setup Gl) (cands : List (P0Tables Gl)) (q : ℕ)
    {t t' : Replay.Transcript} {lin : Replay.LinearOut} (run : Runs (linearStage rel st proof) t lin t') :
    ∃ ds : List Draw, t'.draws.toList = t.draws.toList ++ ds ∧ LinearRun (K := K) rel proof lin ∧
      Game.Reaches (linearGame (K := K) (shapeOf rel) phi barOf S (statementOf st) cands)
        (linearMsgs (K := K) rel proof) (ds.map (drawn (K := K) q)) (some (linearOutOf proof lin)) := by
  have sizes := fits
  simp only [Replay.Statement.fits, Bool.and_eq_true, beq_iff_eq] at sizes
  have pointSize : st.point.size = rel.pointVariables := sizes.1.1.1.2
  unfold linearStage at run
  obtain ⟨lam, t1, h1, run⟩ := run_bind run
  obtain ⟨_, t2, h2, run⟩ := run_bind run
  obtain ⟨zeta, t3, h3, run⟩ := run_bind run
  obtain ⟨⟨blockPoint, middle⟩, t4, h4, run⟩ := run_bind run
  obtain ⟨⟨lanePart, last⟩, t5, h5, run⟩ := run_bind run
  obtain ⟨_, t6, h6, run⟩ := run_bind run
  obtain ⟨⟨scale, lanePoint⟩, t7, h7, run⟩ := run_bind run
  obtain ⟨_, t8, h8, run⟩ := run_bind run
  obtain ⟨rfl, rfl⟩ := run_pure run
  obtain ⟨scaleOk, rfl⟩ := run_liftExcept h7
  obtain ⟨check, rfl⟩ := run_ensure h8
  obtain ⟨scaleValue, lanePointValue, poles⟩ := scaledEqPoint_ok (K := K) laneVars zeta scaleOk
  have blockLength : (proof.linear.extract 0 rel.blockVars).toList.length = rel.blockVars := by
    simp [shape.linear, Replay.Relation.cubeVars]
  have laneLength : (proof.linear.extract rel.blockVars proof.linear.size).toList.length = laneVars := by
    simp [shape.linear, Replay.Relation.cubeVars]
  have blockSize : blockPoint.size = rel.blockVars := by
    rw [sumcheckRounds_size _ _ _ _ _ _ h4, blockLength]
    simp
  have laneSize : lanePart.size = laneVars := by
    rw [sumcheckRounds_size _ _ _ _ _ _ h5, laneLength]
    simp
  obtain ⟨dsB, drawsB, reachB⟩ := sumcheck_reaches (K := K) (Finset.univ : Finset (Fin cands.length)) 1
    (fun w => combRound (shapeOf rel).blockVars
      (blockTables (shapeOf rel) barOf S (statementOf st) lam.toK zeta.toK (cands.get w).z) productComb)
    (fun _ => ∅) q _ _ _ t3 t4 _ (fun m member => shape.rounds m (mem_extract member)) h4
  obtain ⟨dsL, drawsL, reachL⟩ := sumcheck_reaches (K := K) (Finset.univ : Finset (Fin cands.length)) 1
    (fun w => combRound (shapeOf rel).laneVars
      (laneTables (shapeOf rel) barOf S (statementOf st) lam.toK zeta.toK (cands.get w).z (pointK blockPoint))
      productComb)
    (fun _ => ∅) q _ _ _ t4 t5 _ (fun m member => shape.rounds m (mem_extract member)) h5
  rw [blockLength, show pointK (K := K) #[] = [] from rfl,
    toK_liftedTarget rel st fits lam zeta proof.quotient shape.quotient S] at reachB
  rw [laneLength, show pointK (K := K) #[] = [] from rfl] at reachL
  refine ⟨.ext lam :: .ext zeta :: (dsB ++ dsL), ?_, ⟨rfl, rfl, lanePointValue, blockSize, laneSize, rfl, rfl, rfl⟩, ?_⟩
  · rw [run_observeExts h6, drawsL, drawsB, run_sampleExt h3, run_observeExts h2, run_sampleExt h1]
    simp
  · unfold linearGame linearMsgs
    simp only [List.map_cons, List.map_append, drawn_ext]
    apply Game.Reaches.draw lam.toK
    apply Game.Reaches.recv (tableK proof.quotient)
    apply Game.Reaches.draw zeta.toK
    refine Game.Reaches.bind reachB ?_
    refine Game.Reaches.congr (Game.Reaches.bind reachL ?_) rfl (List.append_nil _)
    apply Game.Reaches.recv (proof.finals[0]!.toK, proof.finals[1]!.toK, proof.finals[2]!.toK)
    rw [if_pos]
    · exact Game.Reaches.pure _
    · refine ⟨poles, ?_⟩
      dsimp only
      rw [Ext.toK_of_beq check, Ext.toK_mul, Ext.toK_mul,
        toK_omegaComputed rel st pointSize bits publicFits lam zeta scale _ _ blockPoint blockSize scaleValue,
        toK_laneValue (shapeOf rel) rfl zeta lanePart laneSize]

end CompressionLean.Bridge
