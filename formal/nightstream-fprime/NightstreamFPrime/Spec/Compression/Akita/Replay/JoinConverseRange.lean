import NightstreamFPrime.Spec.Compression.Akita.Replay.JoinConverseRuns

/-! Theorem B of the join, direction "←": a first stop of the replay at a check before the fused
sum-check of the level 5 of `key20.aks`, against the draws of the game of this level (Akita
revision `d50f700`, `verify_fold` of `fold/mod.rs`; ePrint 2026/1983, revision of 2026-11-23,
(189): the draws of one fold).

- `rounds_capped`: after a run without a stop, the rounds of a sum-check are not more than the
  rounds of each of its `.ended` operations.
- `levelCap`, `levelCap20`, `ended_le`: each sum-check of the level 5 of the skeleton of
  `layout20` has 17 rounds or less. `rangeFrees20`: no operation before the index 1397 ends the
  fused sum-check of the level 5 or the tensor reduction of its claim.
- `countDraws_foldPathSq_le`, `countDraws_tensorPath_le`: the draws of the paths of the code are
  not more than the draws of the recorded rounds.
- `range20_ok`: when a path of the level 5 on the states before its `fused.end` and before the
  terminal has the `foldDrawsSq` draws of its game, the step of each check at an index up to 1397
  has no stop when the run before it has none. After a stop, these states are the state of the
  stop, which has no round of the fused sum-check. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

open NightstreamFPrime.Spec.Compression.Lib NightstreamFPrime.Spec.Compression.Game
open NightstreamFPrime.Spec.Compression (Game)
open NightstreamFPrime.Spec.Compression.Akita
open NightstreamFPrime.Spec.Compression.Akita.Extract (countDraws countDraws_append)

open Replayer

namespace Replayer

section Cap

variable (pub : Public) (proof : ByteArray) (messages : Array Message)

theorem endRoundsIn_length_le (s : Replayer) (key : Key) (n : ℕ) :
    (endRoundsIn s key n).length ≤ n := by
  unfold endRoundsIn
  rw [List.length_zip, List.length_map, List.length_range]
  exact Nat.min_le_left _ _

/-- After a run without a stop of operations whose `.ended` of `key` have `n` rounds or less,
from a state with `n` rounds of `key` or less, `key` has `n` rounds or less. -/
theorem rounds_capped (key : Key) (n : ℕ) : ∀ (l : List (Op × ℕ)) (s : Replayer),
    (∀ y ∈ l, ∀ m before, y.1 = .ended key m before → m ≤ n) → (roundsIn s key).length ≤ n →
      (runFrom pub proof messages l s).1 = .ok () →
        (roundsIn (runFrom pub proof messages l s).2 key).length ≤ n
  | [], _, _, small, _ => small
  | x :: l, s, caps, small, ran => by
      obtain ⟨stepped, tail⟩ := (runFrom_cons_fst pub proof messages _ _ _).mp ran
      rw [runFrom_cons_snd pub proof messages _ _ _ stepped]
      refine rounds_capped key n l _ (fun y inside => caps y (List.mem_cons_of_mem _ inside)) ?_
        tail
      by_cases ends : key ∈ x.1.ends
      · obtain ⟨op, i⟩ := x
        cases op with
        | ended k m before =>
            obtain rfl : k = key := (List.mem_singleton.mp ends).symm
            have capped := caps _ List.mem_cons_self m before rfl
            have long := endRoundsIn_length_le s k m
            unfold roundsIn at small ⊢
            rw [step_ended_ok pub proof messages i k m before s stepped, Array.find?_push]
            cases found : s.derived.rounds.find? (·.1 == k) with
            | none =>
                simp only [Option.none_or, beq_self_eq_true, ↓reduceIte, Option.map_some,
                  Option.getD_some]
                omega
            | some e =>
                rw [found] at small
                exact small
        | _ => exact absurd ends List.not_mem_nil
      · rw [roundsIn_tail (step_tail pub proof messages x.2 x.1 s) ends]
        exact small

end Cap

end Replayer

section Skeleton

/-- An `.ended` of the level `level` has `n` rounds or less. -/
def opCap (level n : ℕ) : Op → Bool
  | .ended key rounds _ => key.level != level || decide (rounds ≤ n)
  | _ => true

/-- Each `.ended` of the level `level` in `l` has `n` rounds or less. -/
def levelCap (level n : ℕ) (l : List (Op × ℕ)) : Bool := l.all fun x => opCap level n x.1

theorem levelCap20 : levelCap 5 17 sk20 = true := by decide +kernel

theorem rangeFrees20 : frees (sk20.take 1397) []
    [⟨5, "fused", none⟩, ⟨6, "tensor.sumcheck", none⟩] = true := by
  decide +kernel

variable {L : Layout} {ops : Array Op}

/-- An `.ended` of the level `level` in operations whose erased forms are a skeleton with
`levelCap` has `n` rounds or less. -/
theorem ended_le (erased : ops.toList.map Op.erase = skeleton L) {level n : ℕ}
    (cap : levelCap level n (skeleton L).zipIdx = true) {y : Op × ℕ}
    (inside : y ∈ ops.zipIdx.toList) {key : Key} {m : ℕ} {before : Before}
    (same : y.1 = .ended key m before) (at_ : key.level = level) : m ≤ n := by
  have member : eraseAt y ∈ (skeleton L).zipIdx := by
    rw [← zipIdx_erase erased]
    exact List.mem_map_of_mem inside
  have holds : opCap level n y.1.erase = true := List.all_eq_true.mp cap _ member
  rw [same] at holds
  have bound : (key.level != level || decide (m ≤ n)) = true := holds
  rw [at_] at bound
  simpa using bound

end Skeleton

section Counts

variable {K : Type} [Field K] (φ : Ext → K)

theorem countDraws_plainPath (d : ℕ) : ∀ (rounds : List (List Ext × Ext)) (claim : Ext),
    countDraws (plainPath φ d claim rounds) = rounds.length
  | [], _ => rfl
  | (message, r) :: rounds, claim => by
      show countDraws (plainPath φ d (plainStep claim message r) rounds) + 1 = _
      rw [countDraws_plainPath d rounds]
      rfl

theorem countDraws_eqCodePath (D : ℕ) : ∀ rounds : List (List Ext × Ext),
    countDraws (eqCodePath φ D rounds) = rounds.length
  | [] => rfl
  | (message, r) :: rounds => by
      show countDraws (eqCodePath φ D rounds) + 1 = _
      rw [countDraws_eqCodePath D rounds]
      rfl

/-- The draws of the path of the range tree: for each product stage its rounds and one batch
challenge or less, and the draws of the last stage. -/
theorem countDraws_treePath_le {last : ℕ → (ℕ → Ext) → List Ext → Ext → List Move} {bound μ : ℕ}
    (lastLe : ∀ nodes weights point claim, countDraws (last nodes weights point claim) ≤ bound) :
    ∀ (stages : List Product) (nodes : ℕ) (weights : ℕ → Ext) (point : List Ext) (claim : Ext),
      (∀ S ∈ stages, S.rounds.length ≤ μ) →
        countDraws (treePath φ last stages nodes weights point claim) ≤
          stages.length * (μ + 1) + bound
  | [], nodes, weights, point, claim, _ => by
      rw [List.length_nil, Nat.zero_mul, Nat.zero_add]
      exact lastLe nodes weights point claim
  | S :: rest, nodes, weights, point, claim, short => by
      have head := short S List.mem_cons_self
      have tail := fun n w p c => countDraws_treePath_le lastLe rest n w p c fun Q inside =>
        short Q (List.mem_cons_of_mem _ inside)
      rw [treePath, countDraws_append, countDraws_eqCodePath, List.length_cons, Nat.succ_mul]
      show S.rounds.length + countDraws (match eqRun claim Ext.one point S.rounds with
        | none => []
        | some ended =>
            if ended.claim.same (stageEnd nodes S.arity weights S.children) then
              Move.draw (φ S.gamma) :: treePath φ last rest (nodes * S.arity) (powExt S.gamma)
                ended.point (batched (nodes * S.arity) S.children S.gamma)
            else []) ≤ _
      split
      · show S.rounds.length + 0 ≤ _
        omega
      · split
        · rename_i ended _ _
          have next := tail (nodes * S.arity) (powExt S.gamma) ended.point
            (batched (nodes * S.arity) S.children S.gamma)
          show S.rounds.length + (countDraws (treePath φ last rest (nodes * S.arity)
            (powExt S.gamma) ended.point (batched (nodes * S.arity) S.children S.gamma)) + 1) ≤ _
          omega
        · show S.rounds.length + 0 ≤ _
          omega

theorem countDraws_normPath_le (leaf limit energy : ℕ) (merge : Ext)
    (rounds : List (List Ext × Ext)) (virtual image claim : Ext) :
    countDraws (normPath φ leaf limit energy merge rounds virtual image claim) ≤
      1 + rounds.length := by
  unfold normPath
  show countDraws (if energy ≤ limit then _ else _) ≤ _
  split
  · show countDraws (plainPath φ leaf _ rounds ++ [Move.recv (φ virtual, φ image)]) + 1 ≤ _
    rw [countDraws_append, countDraws_plainPath]
    show rounds.length + 0 + 1 ≤ _
    omega
  · show 0 ≤ _
    omega

end Counts

section Paths

variable {E : Type} [Field E] [Algebra Zq E] [Degree2 E]

omit [Field E] [Algebra Zq E] [Degree2 E] in
theorem countDraws_binPath (bin : Bool) (ζ : E) : countDraws (binPath bin ζ) = binDraws bin := by
  cases bin <;> rfl

/-- The draws of the path of `foldCheckSq` of the code: `α`, `τ_0`, `τ_1`, the draws of the range
tree with `ζ_norm`, and after an accepted range proof `η_norm`, `ζ_bin`, `γ` and the recorded
rounds of the fused sum-check. -/
theorem countDraws_foldPathSq_le (vars batch low leaf limit : ℕ) (bin : Bool) (W : FoldWords)
    (N : NormWords) (claim : Ext) (finish : List Ext → FusedEnd) {μ : ℕ}
    (stages : ∀ S ∈ W.stages, S.rounds.length ≤ μ) (norm : N.rounds.length ≤ μ) :
    countDraws (foldPathSq E vars batch low leaf limit bin W N claim finish) ≤
      3 + (W.stages.length * (μ + 1) + (1 + μ)) + (1 + binDraws bin + 1 + W.rounds.length) := by
  have tree := countDraws_treePath_le (Ext.into (E := E))
    (last := fun _ _ _ claim => normPath (Ext.into (E := E)) leaf limit N.energy N.merge N.rounds
      N.virtual W.image claim) (bound := 1 + μ)
    (fun _ _ _ claim => (countDraws_normPath_le _ leaf limit N.energy N.merge N.rounds N.virtual
      W.image claim).trans (by omega)) W.stages 1 (fun _ => Ext.one) (W.point low) Ext.zero stages
  unfold foldPathSq
  show countDraws (_ ++ _) + 1 + 1 + 1 ≤ _
  rw [countDraws_append]
  split
  · show _ + 0 + 1 + 1 + 1 ≤ _
    omega
  · show _ + (countDraws (binPath bin (W.binary.into : E) ++
      Move.draw (W.batch.into : E) :: fusedPath E claim W.rounds finish) + 1) + 1 + 1 + 1 ≤ _
    rw [countDraws_append, countDraws_binPath]
    show _ + (binDraws bin + (countDraws (fusedPath E claim W.rounds finish) + 1) + 1) + 1 + 1 +
      1 ≤ _
    unfold fusedPath
    rw [countDraws_append, countDraws_plainPath]
    show _ + (binDraws bin + (W.rounds.length + 0 + 1) + 1) + 1 + 1 + 1 ≤ _
    omega

/-- The draws of the path of the tensor reduction of the code: `η` and the recorded rounds. -/
theorem countDraws_tensorPath_le (ν : ℕ) (point : List Ext) (v p0 p1 η : Ext)
    (rounds : List (List Ext × Ext)) (final : Ext) :
    countDraws (tensorPath E ν point v p0 p1 η rounds final) ≤ 1 + rounds.length := by
  cases point with
  | nil => exact Nat.zero_le _
  | cons head tail =>
      show countDraws (if tail.length = ν then _ else _) ≤ _
      split
      · show countDraws (if v.same ((Ext.one - head) * p0 + head * p1) then _ else _) ≤ _
        split
        · show countDraws (plainPath (Ext.into (E := E)) 1 (tensorInput p0 p1 η) rounds ++
            [Move.recv (final.into : E)]) + 1 ≤ _
          rw [countDraws_append, countDraws_plainPath]
          show rounds.length + 0 + 1 ≤ _
          omega
        · exact Nat.zero_le _
      · exact Nat.zero_le _

end Paths

section Stop

variable {E : Type} [Field E] [Algebra Zq E] [Degree2 E]
  (pub : Public) (proof : ByteArray) (messages : Array Message)

/-- A first stop at a check before the fused sum-check of the level 5 against the draws of its
game: when the path of the level 5 on the state before its `fused.end` and the state before the
terminal has `foldDrawsSq` draws or more, the step of a check at an index up to 1397 has no stop
when the run before it has none. -/
theorem range20_ok {zMax rice : ℕ} (layout : pub.layout = layout20 zMax rice)
    (scripted : script pub.layout pub.plan = .ok pub.ops) {c : ℕ} {key : Key} {check : Check}
    (at_ : pub.ops[c]? = some (.check key check)) (le : c ≤ 1397)
    (before : (runFrom pub proof messages (pub.ops.zipIdx.toList.take c) (channelStart pub)).1 =
      .ok ())
    (need : Extract.foldDrawsSq (fold20 5).vars Key20.level₅.shape.certifiedBase
        (Key20.level₅.opening.compressed Key20.level₅.fold) ≤
      countDraws (foldPathIn E pub (fused20 pub proof messages 5) 5 (fold20 5) (setup20 pub 5)
          (fold20 5).vars (fold20 5).batch ++
        tensoredPathIn E Key20.level₅.tail 5
          (stateBefore pub proof messages pub.ops (pub.ops.size - 1)))) :
    (((step pub proof messages c (.check key check)).run).run
      (stateBefore pub proof messages pub.ops c)).1 = .ok () := by
  by_contra stopped
  have erased : pub.ops.toList.map Op.erase = skeleton (layout20 0 0) := by
    rw [script_erase (layout ▸ scripted), skeleton20]
  obtain ⟨-, -, -, -, -, f5, size⟩ := indices20 layout scripted
  have kept : ∀ k, roundsIn (((step pub proof messages c (.check key check)).run).run
      (stateBefore pub proof messages pub.ops c)).2 k =
        roundsIn (stateBefore pub proof messages pub.ops c) k := fun k =>
    check_keeps pub proof messages (fun s => roundsIn s k) (fun _ _ _ => rfl) (fun _ _ => rfl) c
      key check _
  have start : ∀ k, roundsIn (channelStart pub) k = [] := fun k => by
    unfold roundsIn
    rw [channelStart_empty]
    rfl
  have capped : ∀ k : Key, k.level = 5 → (roundsIn (((step pub proof messages c
      (.check key check)).run).run (stateBefore pub proof messages pub.ops c)).2 k).length ≤
        17 := fun k level => by
    rw [kept]
    unfold stateBefore
    refine rounds_capped pub proof messages k 17 _ _ (fun y inside m b same => ?_)
      (by rw [start]; exact Nat.zero_le _) before
    exact ended_le erased levelCap20 (List.mem_of_mem_take inside) same level
  have cut : pub.ops.zipIdx.toList.take c = (pub.ops.zipIdx.toList.take 1397).take c := by
    rw [List.take_take, Nat.min_eq_left le]
  have freed := joined_frees_take erased 1397 [] _ rangeFrees20
  have free : ∀ k ∈ [(⟨5, "fused", none⟩ : Key), ⟨6, "tensor.sumcheck", none⟩],
      roundsIn (((step pub proof messages c (.check key check)).run).run
        (stateBefore pub proof messages pub.ops c)).2 k = [] := fun k member => by
    rw [kept]
    unfold stateBefore
    rw [roundsIn_free pub proof messages _ _ k fun y inside => frees_ends freed member y (by
      rw [cut] at inside
      exact List.mem_of_mem_take inside)]
    exact start k
  have fusedFree := free ⟨5, "fused", none⟩ List.mem_cons_self
  have tensorFree := free ⟨6, "tensor.sumcheck", none⟩ (List.mem_cons_of_mem _ List.mem_cons_self)
  have euclidean : (fold20 5).euclidean = true := by decide +kernel
  have stagesLength : (wordsIn (((step pub proof messages c (.check key check)).run).run
      (stateBefore pub proof messages pub.ops c)).2 5 (fold20 5)).stages.length = 2 := by
    show (productsIn _ 5 _).length = 2
    unfold productsIn
    rw [List.length_map, List.length_zipIdx]
    decide +kernel
  have binZero : binDraws (fold20 5).compressed = 0 := by decide +kernel
  have foldLe := countDraws_foldPathSq_le (E := E) (fold20 5).vars (fold20 5).batch (fold20 5).low
    (leafDegree (fold20 5).logBasis) (fold20 5).limit (fold20 5).compressed
    (wordsIn (((step pub proof messages c (.check key check)).run).run
      (stateBefore pub proof messages pub.ops c)).2 5 (fold20 5))
    (normsIn (((step pub proof messages c (.check key check)).run).run
      (stateBefore pub proof messages pub.ops c)).2 5)
    (stage2In pub (((step pub proof messages c (.check key check)).run).run
      (stateBefore pub proof messages pub.ops c)).2 5 (fold20 5)).fused
    (finishIn (setup20 pub 5) (((step pub proof messages c (.check key check)).run).run
      (stateBefore pub proof messages pub.ops c)).2 5 (fold20 5)) (μ := 17)
    (fun Q member => by
      have member' : Q ∈ productsIn (((step pub proof messages c (.check key check)).run).run
        (stateBefore pub proof messages pub.ops c)).2 5
          ((productStages (fold20 5).logBasis).map (·.1)) := member
      unfold productsIn at member'
      obtain ⟨⟨arity, k⟩, -, rfl⟩ := List.mem_map.mp member'
      exact capped ⟨5, _, none⟩ rfl)
    (capped ⟨5, "norm.sumcheck", none⟩ rfl)
  have roundsLength : (wordsIn (((step pub proof messages c (.check key check)).run).run
      (stateBefore pub proof messages pub.ops c)).2 5 (fold20 5)).rounds.length = 0 := by
    show (roundsIn _ ⟨5, "fused", none⟩).length = 0
    rw [fusedFree]
    rfl
  rw [stagesLength, roundsLength, binZero] at foldLe
  have tensorLe := countDraws_tensorPath_le (E := E) Key20.level₅.tail
    (recordedIn (((step pub proof messages c (.check key check)).run).run
      (stateBefore pub proof messages pub.ops c)).2 ⟨5, "fused.point", none⟩).toList
    (firstIn (((step pub proof messages c (.check key check)).run).run
      (stateBefore pub proof messages pub.ops c)).2 ⟨5, "witness_evaluation", none⟩)
    ((recordedIn (((step pub proof messages c (.check key check)).run).run
      (stateBefore pub proof messages pub.ops c)).2 ⟨5 + 1, "tensor.partials", none⟩).getD 0
        Ext.zero)
    ((recordedIn (((step pub proof messages c (.check key check)).run).run
      (stateBefore pub proof messages pub.ops c)).2 ⟨5 + 1, "tensor.partials", none⟩).getD 1
        Ext.zero)
    (firstIn (((step pub proof messages c (.check key check)).run).run
      (stateBefore pub proof messages pub.ops c)).2 ⟨5 + 1, "tensor.eta", none⟩)
    (roundsIn (((step pub proof messages c (.check key check)).run).run
      (stateBefore pub proof messages pub.ops c)).2 ⟨6, "tensor.sumcheck", none⟩)
    (firstIn (((step pub proof messages c (.check key check)).run).run
      (stateBefore pub proof messages pub.ops c)).2 ⟨5 + 1, "tensor.final_claims", none⟩)
  have value : Extract.foldDrawsSq (fold20 5).vars Key20.level₅.shape.certifiedBase
      (Key20.level₅.opening.compressed Key20.level₅.fold) = 76 := by decide +kernel
  rw [value, countDraws_append, fused20,
    stateBefore_stopped pub proof messages at_ (show c < fusedIndex pub.layout 5 by
      rw [f5]; exact Nat.lt_of_le_of_lt le (by decide)) before stopped,
    stateBefore_stopped pub proof messages at_ (show c < pub.ops.size - 1 by
      rw [size]; exact Nat.lt_of_le_of_lt le (by decide)) before stopped] at need
  have foldLe' : countDraws (foldPathIn E pub (((step pub proof messages c
      (.check key check)).run).run (stateBefore pub proof messages pub.ops c)).2 5 (fold20 5)
        (setup20 pub 5) (fold20 5).vars (fold20 5).batch) ≤ 59 := by
    unfold foldPathIn
    rw [ite_eq_left euclidean]
    exact foldLe
  have tensorLe' : countDraws (tensoredPathIn E Key20.level₅.tail 5 (((step pub proof messages c
      (.check key check)).run).run (stateBefore pub proof messages pub.ops c)).2) ≤
        1 + (roundsIn (((step pub proof messages c (.check key check)).run).run
          (stateBefore pub proof messages pub.ops c)).2 ⟨6, "tensor.sumcheck", none⟩).length :=
    tensorLe
  rw [tensorFree] at tensorLe'
  exact absurd (need.trans (Nat.add_le_add foldLe' tensorLe')) (by decide)

end Stop

end NightstreamFPrime.Spec.Compression.Akita.Replay
