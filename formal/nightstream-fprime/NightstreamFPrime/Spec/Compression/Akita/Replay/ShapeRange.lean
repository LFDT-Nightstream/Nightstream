import NightstreamFPrime.Spec.Compression.Akita.Replay.ShapeErase

/-! The checks `range.k.end` of the product stages of the range proof against the stage function
of the last check of the range proof (finding 16; Akita revision `d50f700`, `stages/stage1.rs`;
`Build.range` of `Replay/Script.lean`, `treeRun` of `Replay/Range.lean`).

- `treeInputs` are the inputs of the product stage `k` of `treeRun` after the stages before it,
  and `stageHolds` is the condition of one stage: its rounds end, and the end claim is
  `stageEnd` of its children. `treeRun_stage`: when `treeRun` accepts, each stage holds on its
  inputs. `treeRun_none_of_stage`: when a stage fails on its inputs, `treeRun` is `none`.
- `step_absorb_zero`, `step_absorb_batched`: the absorbs of the input claims of the stages.
  `stateBefore_le`, `recorded_between`, `rounds_between`, `first_between`, `first_of_writer`,
  `rounds_of_end`, `output_of_end`: the values of a state before an operation from the operations
  before it.
- `stage0_ident`, `stage1_ident`: on the state before an operation after the two product stages,
  the recorded output of each stage is the end of `eqRun` on the inputs of `treeRun`.
  `stage0_forward`, `stage1_forward`: the check `range.k.end` holds when the stage function of
  the last range check accepts; `stage0_backward`, `stage1_backward`: when it fails, `treeRun`
  fails for each later stage and each leaf, so the level function is `none`
  (`foldRunIn_none_of_tree`).
- `stage0At`, `stage1At`, `stagesShape` are Bools on the skeleton with the indices and the frames
  that these theorems read; `erased_stages` gives them on operations whose erased forms are the
  skeleton. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

/-- The condition of the product stage `S` with the inputs `x = (nodes, weights, point, claim)`:
its equality-factored rounds end, and the end claim is `stageEnd` of its children. -/
def stageHolds (S : Product) (x : ℕ × (ℕ → Ext) × List Ext × Ext) : Bool :=
  match eqRun x.2.2.2 Ext.one x.2.2.1 S.rounds with
  | none => false
  | some ended => ended.claim.same (stageEnd x.1 S.arity x.2.1 S.children)

/-- The inputs of the product stage `k` of `treeRun`, when the stages before it hold. -/
def treeInputs : List Product → ℕ → (ℕ → Ext) → List Ext → Ext → ℕ →
    Option (ℕ × (ℕ → Ext) × List Ext × Ext)
  | _, nodes, weights, point, claim, 0 => some (nodes, weights, point, claim)
  | [], _, _, _, _, _ + 1 => none
  | S :: rest, nodes, weights, point, claim, k + 1 =>
      match eqRun claim Ext.one point S.rounds with
      | none => none
      | some ended =>
          if ended.claim.same (stageEnd nodes S.arity weights S.children) then
            treeInputs rest (nodes * S.arity) (powExt S.gamma) ended.point
              (batched (nodes * S.arity) S.children S.gamma) k
          else none

/-- When `treeRun` accepts, each product stage holds on its inputs. -/
theorem treeRun_stage {β : Type} (last : ℕ → (ℕ → Ext) → List Ext → Ext → Option β) :
    ∀ (k : ℕ) (stages : List Product) (nodes : ℕ) (weights : ℕ → Ext) (point : List Ext)
      (claim : Ext) {S : Product}, stages[k]? = some S →
      treeRun last stages nodes weights point claim ≠ none →
      ∃ x, treeInputs stages nodes weights point claim k = some x ∧ stageHolds S x = true
  | k, [], _, _, _, _, S, found, _ => by simp at found
  | 0, T :: rest, nodes, weights, point, claim, S, found, accepts => by
      simp only [List.getElem?_cons_zero, Option.some.injEq] at found
      subst found
      refine ⟨_, rfl, ?_⟩
      unfold stageHolds
      simp only [treeRun] at accepts
      split at accepts
      · exact absurd rfl accepts
      · rename_i ended found
        rw [found]
        split at accepts
        · assumption
        · exact absurd rfl accepts
  | k + 1, T :: rest, nodes, weights, point, claim, S, found, accepts => by
      simp only [List.getElem?_cons_succ] at found
      simp only [treeRun] at accepts
      unfold treeInputs
      split at accepts
      · exact absurd rfl accepts
      · rename_i ended ran
        rw [ran]
        simp only
        split at accepts
        · rename_i held
          simp only [held, ↓reduceIte]
          exact treeRun_stage last k rest _ _ _ _ found accepts
        · exact absurd rfl accepts

/-- When a product stage fails on its inputs, `treeRun` is `none`. -/
theorem treeRun_none_of_stage {β : Type} (last : ℕ → (ℕ → Ext) → List Ext → Ext → Option β) :
    ∀ (k : ℕ) (stages : List Product) (nodes : ℕ) (weights : ℕ → Ext) (point : List Ext)
      (claim : Ext) {S : Product} {x : ℕ × (ℕ → Ext) × List Ext × Ext}, stages[k]? = some S →
      treeInputs stages nodes weights point claim k = some x → stageHolds S x = false →
      treeRun last stages nodes weights point claim = none
  | k, stages, nodes, weights, point, claim, S, x, found, inputs, fails => by
      by_contra accepts
      obtain ⟨y, inputs', holds⟩ := treeRun_stage last k stages nodes weights point claim found accepts
      rw [inputs] at inputs'
      cases inputs'
      rw [holds] at fails
      cases fails

namespace Replayer

variable (pub : Public) (proof : ByteArray) (messages : Array Message)

/-- The absorb of the input claim zero of the first product stage records zero. -/
theorem step_absorb_zero (i : ℕ) (key : Key) (s : Replayer) :
    (((step pub proof messages i (.absorb key .zero)).run).run s).1 = .ok () ∧
      recordedIn (((step pub proof messages i (.absorb key .zero)).run).run s).2 key =
        recordedIn s key ++ #[Ext.zero] := by
  simp only [step, value]
  simp only [run_bind, run_pure, run_absorb, decodeExts_extBytes_array, run_record, run_modify]
  exact ⟨by trivial, recordedIn_of_env rfl⟩

/-- The absorb of the batched claim of the children of a product stage records
`Σ_i γ^i·c_i` of the recorded children `c` and the recorded batch challenge `γ`. -/
theorem step_absorb_batched (i : ℕ) (key children gamma : Key) (s : Replayer) :
    (((step pub proof messages i (.absorb key (.batched children gamma))).run).run s).1 = .ok () ∧
      recordedIn (((step pub proof messages i (.absorb key (.batched children gamma))).run).run
          s).2 key =
        recordedIn s key ++ #[batched (recordedIn s children).size (recordedIn s children).toList
          (firstIn s gamma)] := by
  simp only [step, value]
  simp only [run_bind, run_pure, run_recorded, run_first, run_absorb, decodeExts_extBytes_array,
    run_record, run_modify]
  exact ⟨by trivial, recordedIn_of_env rfl⟩

section States

variable (ops : Array Op)

/-- The run up to `j'` is the run up to `j ≤ j'` and then the operations between them. -/
theorem stateBefore_le {j j' : ℕ} (le : j ≤ j')
    (ok : (runFrom pub proof messages (ops.zipIdx.toList.take j') (channelStart pub)).1 = .ok ()) :
    (runFrom pub proof messages (ops.zipIdx.toList.take j) (channelStart pub)).1 = .ok () ∧
      (runFrom pub proof messages ((ops.zipIdx.toList.take j').drop j)
        (stateBefore pub proof messages ops j)).1 = .ok () ∧
      stateBefore pub proof messages ops j' = (runFrom pub proof messages
        ((ops.zipIdx.toList.take j').drop j) (stateBefore pub proof messages ops j)).2 := by
  have cut : ops.zipIdx.toList.take j' =
      ops.zipIdx.toList.take j ++ (ops.zipIdx.toList.take j').drop j := by
    conv_lhs => rw [← List.take_append_drop j (ops.zipIdx.toList.take j')]
    rw [List.take_take, Nat.min_eq_left le]
  rw [cut, runFrom_append_fst] at ok
  refine ⟨ok.1, ok.2, ?_⟩
  unfold stateBefore
  conv_lhs => rw [cut]
  rw [runFrom_append_snd pub proof messages _ _ _ ok.1]

/-- A key that no operation between `j ≤ j'` writes keeps its recorded values. -/
theorem recorded_between {j j' : ℕ} (le : j ≤ j')
    (ok : (runFrom pub proof messages (ops.zipIdx.toList.take j') (channelStart pub)).1 = .ok ())
    {key : Key} (free : ∀ y ∈ (ops.zipIdx.toList.take j').drop j, key ∉ y.1.writes) :
    recordedIn (stateBefore pub proof messages ops j') key =
      recordedIn (stateBefore pub proof messages ops j) key := by
  rw [(stateBefore_le pub proof messages ops le ok).2.2]
  exact recorded_free pub proof messages _ _ key free

/-- The rounds of a key that no operation between `j ≤ j'` ends do not change. -/
theorem rounds_between {j j' : ℕ} (le : j ≤ j')
    (ok : (runFrom pub proof messages (ops.zipIdx.toList.take j') (channelStart pub)).1 = .ok ())
    {key : Key} (free : ∀ y ∈ (ops.zipIdx.toList.take j').drop j, key ∉ y.1.ends) :
    roundsIn (stateBefore pub proof messages ops j') key =
      roundsIn (stateBefore pub proof messages ops j) key := by
  rw [(stateBefore_le pub proof messages ops le ok).2.2]
  exact roundsIn_free pub proof messages _ _ key free

/-- The first recorded value of a key does not change once the key has a value. -/
theorem first_between {j j' : ℕ} (le : j ≤ j')
    (ok : (runFrom pub proof messages (ops.zipIdx.toList.take j') (channelStart pub)).1 = .ok ())
    {key : Key} (some_ : (recordedIn (stateBefore pub proof messages ops j) key).size ≠ 0) :
    firstIn (stateBefore pub proof messages ops j') key =
      firstIn (stateBefore pub proof messages ops j) key := by
  rw [(stateBefore_le pub proof messages ops le ok).2.2]
  obtain ⟨tail, grown⟩ := (runFrom_grows pub proof messages ((ops.zipIdx.toList.take j').drop j)
    (stateBefore pub proof messages ops j)).prefix_ key
  unfold firstIn
  rw [← Array.getElem?_toList, ← grown, List.getElem?_append_left (by simpa [Nat.pos_iff_ne_zero]
    using some_), Array.getElem?_toList]

/-- The state after the operation `j`: its step on the state before it, without a stop. -/
theorem stateBefore_succ {j : ℕ} {op : Op} (at_ : ops[j]? = some op)
    (ok : (runFrom pub proof messages (ops.zipIdx.toList.take (j + 1)) (channelStart pub)).1 =
      .ok ()) :
    (((step pub proof messages j op).run).run (stateBefore pub proof messages ops j)).1 = .ok () ∧
      stateBefore pub proof messages ops (j + 1) =
        (((step pub proof messages j op).run).run (stateBefore pub proof messages ops j)).2 := by
  rw [take_succ_zipIdx ops at_, runFrom_append_fst, runFrom_cons_fst] at ok
  refine ⟨ok.2.1, ?_⟩
  unfold stateBefore
  rw [take_succ_zipIdx ops at_, runFrom_append_snd pub proof messages _ _ _ ok.1,
    runFrom_cons_snd pub proof messages _ _ _ ok.2.1]
  rfl

/-- A key that no operation before `j` writes has no recorded value before `j`. -/
theorem recorded_fresh {j : ℕ} {key : Key}
    (free : ∀ y ∈ ops.zipIdx.toList.take j, key ∉ y.1.writes) :
    recordedIn (stateBefore pub proof messages ops j) key = #[] := by
  unfold stateBefore
  rw [recorded_free pub proof messages _ _ key free, channelStart_fresh]

/-- The first value of a key whose first writer is the operation `j`: the first value that it
appends. -/
theorem first_of_writer {j j' : ℕ} (lt : j < j') {op : Op} (at_ : ops[j]? = some op)
    (ok : (runFrom pub proof messages (ops.zipIdx.toList.take j') (channelStart pub)).1 = .ok ())
    {key : Key} (free : ∀ y ∈ ops.zipIdx.toList.take j, key ∉ y.1.writes) {v : Ext}
    {vs : Array Ext} (writes : recordedIn (((step pub proof messages j op).run).run
      (stateBefore pub proof messages ops j)).2 key =
        recordedIn (stateBefore pub proof messages ops j) key ++ (#[v] ++ vs)) :
    firstIn (stateBefore pub proof messages ops j') key = v := by
  have okSucc := (stateBefore_le pub proof messages ops lt ok).1
  obtain ⟨-, after⟩ := stateBefore_succ pub proof messages ops at_ okSucc
  have here : recordedIn (stateBefore pub proof messages ops (j + 1)) key = #[v] ++ vs := by
    rw [after, writes, recorded_fresh pub proof messages ops free, Array.empty_append]
  rw [first_between pub proof messages ops lt ok (by rw [here]; simp)]
  unfold firstIn
  rw [here]
  simp

/-- The rounds of a sum-check after its first end: the rounds of `endRoundsIn` on the state before
the end. -/
theorem rounds_of_end {e j : ℕ} (lt : e < j) {key : Key} {n : ℕ} {before : Before}
    (at_ : ops[e]? = some (.ended key n before))
    (ok : (runFrom pub proof messages (ops.zipIdx.toList.take j) (channelStart pub)).1 = .ok ())
    (free : ∀ y ∈ ops.zipIdx.toList.take e, key ∉ y.1.ends) :
    roundsIn (stateBefore pub proof messages ops j) key =
      endRoundsIn (stateBefore pub proof messages ops e) key n := by
  have okSucc := (stateBefore_le pub proof messages ops lt ok).1
  obtain ⟨stepped, after⟩ := stateBefore_succ pub proof messages ops at_ okSucc
  have missing : (stateBefore pub proof messages ops e).derived.rounds.find? (·.1 == key) =
      none := by
    unfold stateBefore
    rw [find_tail (runFrom_tail pub proof messages _ _) fun inside => by
      obtain ⟨y, member, found⟩ := List.mem_flatMap.mp inside
      exact free y member found, channelStart_empty]
    rfl
  have found : (stateBefore pub proof messages ops (e + 1)).derived.rounds.find? (·.1 == key) =
      some (key, endRoundsIn (stateBefore pub proof messages ops e) key n) := by
    rw [after, step_ended_ok pub proof messages _ _ _ _ _ stepped, Array.find?_push, missing]
    simp
  rw [(stateBefore_le pub proof messages ops lt ok).2.2]
  unfold roundsIn
  rw [find_grows (runFrom_grows pub proof messages _ _).rounds found]
  rfl

end States

/-- The weights of a product stage on the recorded values. -/
def weightsIn (s : Replayer) : Weights → ℕ → Ext
  | .ones => fun _ => Ext.one
  | .powers gamma => powExt (firstIn s gamma)

theorem run_holds_stage (nodes arity : ℕ) (weights : Weights) (output children : Key)
    (s : Replayer) :
    ((holds pub (.stage nodes arity weights output children)).run).run s =
      (.ok ((firstIn s output).same (stageEnd nodes arity (weightsIn s weights)
        (recordedIn s children).toList)), s) := by
  cases weights <;> rfl

section Stages

variable (ops : Array Op)

/-- The output of a sum-check after its end, whose end is the first writer of the output: the end
claim of `sumcheckRun` on the state before the end. -/
theorem output_of_end {e j : ℕ} (lt : e < j) {key : Key} {n : ℕ} {before : Before}
    (at_ : ops[e]? = some (.ended key n before))
    (ok : (runFrom pub proof messages (ops.zipIdx.toList.take j) (channelStart pub)).1 = .ok ())
    (free : ∀ y ∈ ops.zipIdx.toList.take e,
      (⟨key.level, key.site ++ ".output", none⟩ : Key) ∉ y.1.writes) :
    ∃ ended, sumcheckRun (formIn (stateBefore pub proof messages ops e) before)
        (firstIn (stateBefore pub proof messages ops e) ⟨key.level, key.site ++ ".claim", none⟩)
        (endRoundsIn (stateBefore pub proof messages ops e) key n) = some ended ∧
      firstIn (stateBefore pub proof messages ops j) ⟨key.level, key.site ++ ".output", none⟩ =
        ended.claim := by
  have okSucc := (stateBefore_le pub proof messages ops lt ok).1
  obtain ⟨stepped, -⟩ := stateBefore_succ pub proof messages ops at_ okSucc
  rw [step_ended] at stepped
  cases found : sumcheckRun (formIn (stateBefore pub proof messages ops e) before)
      (firstIn (stateBefore pub proof messages ops e) ⟨key.level, key.site ++ ".claim", none⟩)
      (endRoundsIn (stateBefore pub proof messages ops e) key n) with
  | none =>
      rw [found] at stepped
      cases stepped
  | some ended =>
      refine ⟨ended, rfl, first_of_writer pub proof messages ops lt at_ ok free (vs := #[]) ?_⟩
      rw [step_ended, found, Array.append_empty]
      exact recordedIn_of_env rfl

/-- The first product stage on the state before `j`, after its claim at `a` and its end at `e`:
the end of its rounds from the claim zero and the turned point `τ_0` is the recorded output. -/
theorem stage0_ident {ℓ a e j vars low : ℕ} (ae : a < e) (ej : e < j)
    (atA : ops[a]? = some (.absorb ⟨ℓ, s!"range.{0}" ++ ".claim", none⟩ .zero))
    (atE : ops[e]? = some (.ended ⟨ℓ, s!"range.{0}", none⟩ vars (.turned ⟨ℓ, "tau0", none⟩ low)))
    (ok : (runFrom pub proof messages (ops.zipIdx.toList.take j) (channelStart pub)).1 = .ok ())
    (claimFree : ∀ y ∈ ops.zipIdx.toList.take a,
      (⟨ℓ, s!"range.{0}" ++ ".claim", none⟩ : Key) ∉ y.1.writes)
    (outputFree : ∀ y ∈ ops.zipIdx.toList.take e,
      (⟨ℓ, s!"range.{0}" ++ ".output", none⟩ : Key) ∉ y.1.writes)
    (endFree : ∀ y ∈ ops.zipIdx.toList.take e, (⟨ℓ, s!"range.{0}", none⟩ : Key) ∉ y.1.ends)
    (tauFree : ∀ y ∈ (ops.zipIdx.toList.take j).drop e, (⟨ℓ, "tau0", none⟩ : Key) ∉ y.1.writes) :
    ∃ ended, eqRun Ext.zero Ext.one
        (pointIn (stateBefore pub proof messages ops j) (.turned ⟨ℓ, "tau0", none⟩ low))
        (roundsIn (stateBefore pub proof messages ops j) ⟨ℓ, s!"range.{0}", none⟩) = some ended ∧
      firstIn (stateBefore pub proof messages ops j) ⟨ℓ, s!"range.{0}" ++ ".output", none⟩ =
        ended.claim := by
  have okE := (stateBefore_le pub proof messages ops ej.le ok).1
  have claim : firstIn (stateBefore pub proof messages ops e) ⟨ℓ, s!"range.{0}" ++ ".claim", none⟩ =
      Ext.zero :=
    first_of_writer pub proof messages ops ae atA okE claimFree (vs := #[])
      (by rw [(step_absorb_zero pub proof messages a _ _).2]; rfl)
  obtain ⟨ended, ran, output⟩ := output_of_end pub proof messages ops ej atE ok outputFree
  refine ⟨ended, ?_, output⟩
  have point : pointIn (stateBefore pub proof messages ops j) (.turned ⟨ℓ, "tau0", none⟩ low) =
      pointIn (stateBefore pub proof messages ops e) (.turned ⟨ℓ, "tau0", none⟩ low) := by
    simp only [pointIn, recorded_between pub proof messages ops ej.le ok tauFree]
  rw [point, rounds_of_end pub proof messages ops ej atE ok endFree]
  dsimp only at ran
  rw [claim] at ran
  exact sumcheckRun_equality_eqRun ran

theorem output_key (ℓ k : ℕ) :
    (⟨ℓ, s!"range.{k}" ++ ".output", none⟩ : Key) = ⟨ℓ, s!"range.{k}.output", none⟩ := rfl

/-- At `range.0.end`, when the stage function of the last check of the range proof at `R` accepts:
the check holds on the state before it. -/
theorem stage0_forward {ℓ a e c R vars low a0 : ℕ} (ae : a < e) (ec : e < c)
    (cR : c < R)
    (atA : ops[a]? = some (.absorb ⟨ℓ, s!"range.{0}" ++ ".claim", none⟩ .zero))
    (atE : ops[e]? = some (.ended ⟨ℓ, s!"range.{0}", none⟩ vars (.turned ⟨ℓ, "tau0", none⟩ low)))
    (okR : (runFrom pub proof messages (ops.zipIdx.toList.take R) (channelStart pub)).1 = .ok ())
    (claimFree : ∀ y ∈ ops.zipIdx.toList.take a,
      (⟨ℓ, s!"range.{0}" ++ ".claim", none⟩ : Key) ∉ y.1.writes)
    (outputFree : ∀ y ∈ ops.zipIdx.toList.take e,
      (⟨ℓ, s!"range.{0}" ++ ".output", none⟩ : Key) ∉ y.1.writes)
    (endFree : ∀ y ∈ ops.zipIdx.toList.take e, (⟨ℓ, s!"range.{0}", none⟩ : Key) ∉ y.1.ends)
    (tauFree : ∀ y ∈ (ops.zipIdx.toList.take R).drop e, (⟨ℓ, "tau0", none⟩ : Key) ∉ y.1.writes)
    (outputAfter : ∀ y ∈ (ops.zipIdx.toList.take R).drop c,
      (⟨ℓ, s!"range.{0}.output", none⟩ : Key) ∉ y.1.writes)
    (childAfter : ∀ y ∈ (ops.zipIdx.toList.take R).drop c,
      (⟨ℓ, s!"range.{0}.children", none⟩ : Key) ∉ y.1.writes)
    {β : Type} (last : ℕ → (ℕ → Ext) → List Ext → Ext → Option β) (rest : List Product) (γ : Ext)
    (accepts : treeRun last (⟨a0, roundsIn (stateBefore pub proof messages ops R)
        ⟨ℓ, s!"range.{0}", none⟩, (recordedIn (stateBefore pub proof messages ops R)
          ⟨ℓ, s!"range.{0}.children", none⟩).toList, γ⟩ :: rest) 1 (fun _ => Ext.one)
        (pointIn (stateBefore pub proof messages ops R) (.turned ⟨ℓ, "tau0", none⟩ low)) Ext.zero ≠
      none) :
    (((holds pub (.stage 1 a0 .ones ⟨ℓ, s!"range.{0}.output", none⟩
      ⟨ℓ, s!"range.{0}.children", none⟩)).run).run (stateBefore pub proof messages ops c)).1 =
        .ok true := by
  obtain ⟨x, inputs, held⟩ := treeRun_stage last 0 _ 1 _ _ _ rfl accepts
  cases inputs
  obtain ⟨ended, ran, output⟩ := stage0_ident pub proof messages ops (j := R) ae (ec.trans cR) atA
    atE okR claimFree outputFree endFree tauFree
  unfold stageHolds at held
  dsimp only at held
  rw [ran] at held
  rw [run_holds_stage]
  simp only [weightsIn]
  have outEq : firstIn (stateBefore pub proof messages ops c) ⟨ℓ, s!"range.{0}.output", none⟩ =
      firstIn (stateBefore pub proof messages ops R) ⟨ℓ, s!"range.{0}.output", none⟩ := by
    unfold firstIn
    rw [recorded_between pub proof messages ops cR.le okR outputAfter]
  rw [outEq, ← output_key, output, ← recorded_between pub proof messages ops cR.le okR childAfter]
  exact congrArg _ held

/-- A stop at `range.0.end`: `treeRun` fails at its first product stage on the state before the
check, for each later stage and each leaf. -/
theorem stage0_backward {ℓ a e c vars low a0 : ℕ} (ae : a < e) (ec : e < c)
    (atA : ops[a]? = some (.absorb ⟨ℓ, s!"range.{0}" ++ ".claim", none⟩ .zero))
    (atE : ops[e]? = some (.ended ⟨ℓ, s!"range.{0}", none⟩ vars (.turned ⟨ℓ, "tau0", none⟩ low)))
    (okC : (runFrom pub proof messages (ops.zipIdx.toList.take c) (channelStart pub)).1 = .ok ())
    (claimFree : ∀ y ∈ ops.zipIdx.toList.take a,
      (⟨ℓ, s!"range.{0}" ++ ".claim", none⟩ : Key) ∉ y.1.writes)
    (outputFree : ∀ y ∈ ops.zipIdx.toList.take e,
      (⟨ℓ, s!"range.{0}" ++ ".output", none⟩ : Key) ∉ y.1.writes)
    (endFree : ∀ y ∈ ops.zipIdx.toList.take e, (⟨ℓ, s!"range.{0}", none⟩ : Key) ∉ y.1.ends)
    (tauFree : ∀ y ∈ (ops.zipIdx.toList.take c).drop e, (⟨ℓ, "tau0", none⟩ : Key) ∉ y.1.writes)
    (fails : (((holds pub (.stage 1 a0 .ones ⟨ℓ, s!"range.{0}.output", none⟩
      ⟨ℓ, s!"range.{0}.children", none⟩)).run).run (stateBefore pub proof messages ops c)).1 =
        .ok false)
    {β : Type} (last : ℕ → (ℕ → Ext) → List Ext → Ext → Option β) (rest : List Product)
    (γ : Ext) :
    treeRun last (⟨a0, roundsIn (stateBefore pub proof messages ops c)
        ⟨ℓ, s!"range.{0}", none⟩, (recordedIn (stateBefore pub proof messages ops c)
          ⟨ℓ, s!"range.{0}.children", none⟩).toList, γ⟩ :: rest) 1 (fun _ => Ext.one)
        (pointIn (stateBefore pub proof messages ops c) (.turned ⟨ℓ, "tau0", none⟩ low))
        Ext.zero = none := by
  obtain ⟨ended, ran, output⟩ := stage0_ident pub proof messages ops (j := c) ae ec atA atE okC
    claimFree outputFree endFree tauFree
  refine treeRun_none_of_stage last 0 _ 1 _ _ _ rfl rfl ?_
  unfold stageHolds
  dsimp only
  rw [ran]
  rw [run_holds_stage] at fails
  simp only [weightsIn, ← output_key, output, Except.ok.injEq] at fails
  exact fails

theorem mem_drop_of_le {l : List (Op × ℕ)} {m n : ℕ} (le : m ≤ n) {y : Op × ℕ}
    (member : y ∈ l.drop n) : y ∈ l.drop m := by
  rw [show n = m + (n - m) by omega, ← List.drop_drop] at member
  exact List.mem_of_mem_drop member

/-- The second product stage on the state before `j`: the first stage held, and the end of the
rounds of the second stage from the batched claim of the children of the first stage and the
point of the first stage is the recorded output. -/
theorem stage1_ident {ℓ a0 e0 c0 a1 e1 j vars low arity : ℕ} {key0 : Key} (ae0 : a0 < e0)
    (ec0 : e0 < c0) (ca1 : c0 < a1) (ae1 : a1 < e1) (ej : e1 < j)
    (atA0 : ops[a0]? = some (.absorb ⟨ℓ, s!"range.{0}" ++ ".claim", none⟩ .zero))
    (atE0 : ops[e0]? = some (.ended ⟨ℓ, s!"range.{0}", none⟩ vars (.turned ⟨ℓ, "tau0", none⟩ low)))
    (atC0 : ops[c0]? = some (.check key0 (.stage 1 arity .ones ⟨ℓ, s!"range.{0}.output", none⟩
      ⟨ℓ, s!"range.{0}.children", none⟩)))
    (atA1 : ops[a1]? = some (.absorb ⟨ℓ, s!"range.{1}" ++ ".claim", none⟩
      (.batched ⟨ℓ, s!"range.{0}.children", none⟩ ⟨ℓ, s!"range.{0}.batch", none⟩)))
    (atE1 : ops[e1]? = some (.ended ⟨ℓ, s!"range.{1}", none⟩ vars
      (.point ⟨ℓ, s!"range.{0}" ++ ".point", none⟩)))
    (ok : (runFrom pub proof messages (ops.zipIdx.toList.take j) (channelStart pub)).1 = .ok ())
    (claimFree0 : ∀ y ∈ ops.zipIdx.toList.take a0,
      (⟨ℓ, s!"range.{0}" ++ ".claim", none⟩ : Key) ∉ y.1.writes)
    (outputFree0 : ∀ y ∈ ops.zipIdx.toList.take e0,
      (⟨ℓ, s!"range.{0}" ++ ".output", none⟩ : Key) ∉ y.1.writes)
    (endFree0 : ∀ y ∈ ops.zipIdx.toList.take e0, (⟨ℓ, s!"range.{0}", none⟩ : Key) ∉ y.1.ends)
    (tauFree : ∀ y ∈ (ops.zipIdx.toList.take j).drop e0, (⟨ℓ, "tau0", none⟩ : Key) ∉ y.1.writes)
    (outputAfter0 : ∀ y ∈ (ops.zipIdx.toList.take j).drop c0,
      (⟨ℓ, s!"range.{0}.output", none⟩ : Key) ∉ y.1.writes)
    (childAfter0 : ∀ y ∈ (ops.zipIdx.toList.take j).drop c0,
      (⟨ℓ, s!"range.{0}.children", none⟩ : Key) ∉ y.1.writes)
    (batchAfter0 : ∀ y ∈ (ops.zipIdx.toList.take j).drop a1,
      (⟨ℓ, s!"range.{0}.batch", none⟩ : Key) ∉ y.1.writes)
    (pointAfter0 : ∀ y ∈ (ops.zipIdx.toList.take j).drop e0,
      (⟨ℓ, s!"range.{0}" ++ ".point", none⟩ : Key) ∉ y.1.writes)
    (pointCount : (recordedIn (stateBefore pub proof messages ops e0)
      ⟨ℓ, s!"range.{0}" ++ ".point", none⟩).size = vars)
    (childCount : (recordedIn (stateBefore pub proof messages ops j)
      ⟨ℓ, s!"range.{0}.children", none⟩).size = 1 * arity)
    (claimFree1 : ∀ y ∈ ops.zipIdx.toList.take a1,
      (⟨ℓ, s!"range.{1}" ++ ".claim", none⟩ : Key) ∉ y.1.writes)
    (outputFree1 : ∀ y ∈ ops.zipIdx.toList.take e1,
      (⟨ℓ, s!"range.{1}" ++ ".output", none⟩ : Key) ∉ y.1.writes)
    (endFree1 : ∀ y ∈ ops.zipIdx.toList.take e1, (⟨ℓ, s!"range.{1}", none⟩ : Key) ∉ y.1.ends) :
    ∃ ended0 ended1, eqRun Ext.zero Ext.one
        (pointIn (stateBefore pub proof messages ops j) (.turned ⟨ℓ, "tau0", none⟩ low))
        (roundsIn (stateBefore pub proof messages ops j) ⟨ℓ, s!"range.{0}", none⟩) = some ended0 ∧
      ended0.claim.same (stageEnd 1 arity (fun _ => Ext.one)
        (recordedIn (stateBefore pub proof messages ops j) ⟨ℓ, s!"range.{0}.children", none⟩).toList)
          = true ∧
      eqRun (batched (1 * arity) (recordedIn (stateBefore pub proof messages ops j)
          ⟨ℓ, s!"range.{0}.children", none⟩).toList
          (firstIn (stateBefore pub proof messages ops j) ⟨ℓ, s!"range.{0}.batch", none⟩)) Ext.one
        ended0.point (roundsIn (stateBefore pub proof messages ops j) ⟨ℓ, s!"range.{1}", none⟩) =
          some ended1 ∧
      firstIn (stateBefore pub proof messages ops j) ⟨ℓ, s!"range.{1}" ++ ".output", none⟩ =
        ended1.claim := by
  have ec0j : c0 < j := ca1.trans (ae1.trans ej)
  have e0j : e0 < j := ec0.trans ec0j
  obtain ⟨ended0, ran0, output0⟩ := stage0_ident pub proof messages ops (j := j) ae0 e0j atA0 atE0
    ok claimFree0 outputFree0 endFree0 tauFree
  have okC0succ := (stateBefore_le pub proof messages ops (Nat.succ_le_of_lt ec0j) ok).1
  obtain ⟨stepped, -⟩ := stateBefore_succ pub proof messages ops atC0 okC0succ
  have held := (step_check_iff pub proof messages c0 key0 _ _).mp stepped
  rw [run_holds_stage] at held
  simp only [weightsIn, Except.ok.injEq] at held
  have outEq : firstIn (stateBefore pub proof messages ops c0) ⟨ℓ, s!"range.{0}.output", none⟩ =
      firstIn (stateBefore pub proof messages ops j) ⟨ℓ, s!"range.{0}.output", none⟩ := by
    unfold firstIn
    rw [recorded_between pub proof messages ops ec0j.le ok outputAfter0]
  rw [outEq, ← recorded_between pub proof messages ops ec0j.le ok childAfter0, ← output_key,
    output0] at held
  have okE1 := (stateBefore_le pub proof messages ops ej.le ok).1
  have a1j : a1 ≤ j := (ae1.trans ej).le
  have childA1 := recorded_between pub proof messages ops a1j ok
    (fun y member => childAfter0 y (mem_drop_of_le ca1.le member))
  have batchA1 := recorded_between pub proof messages ops a1j ok batchAfter0
  have claim1 : firstIn (stateBefore pub proof messages ops e1)
      ⟨ℓ, s!"range.{1}" ++ ".claim", none⟩ =
      batched (1 * arity) (recordedIn (stateBefore pub proof messages ops j)
        ⟨ℓ, s!"range.{0}.children", none⟩).toList
        (firstIn (stateBefore pub proof messages ops j) ⟨ℓ, s!"range.{0}.batch", none⟩) := by
    rw [first_of_writer pub proof messages ops ae1 atA1 okE1 claimFree1 (vs := #[])
      (by rw [(step_absorb_batched pub proof messages a1 _ _ _ _).2]; rfl)]
    rw [childA1] at childCount ⊢
    rw [childCount]
    unfold firstIn
    rw [batchA1]
  have pointE1 := recorded_between pub proof messages ops ej.le ok
    (fun y member => pointAfter0 y (mem_drop_of_le (ec0.trans (ca1.trans ae1)).le member))
  have pointE0 := recorded_between pub proof messages ops e0j.le ok pointAfter0
  have point1 : (recordedIn (stateBefore pub proof messages ops e1)
      ⟨ℓ, s!"range.{0}" ++ ".point", none⟩).toList = ended0.point := by
    rw [eqRun_point _ _ _ _ ran0, rounds_of_end pub proof messages ops e0j atE0 ok endFree0, ←
      pointE1, pointE0]
    unfold endRoundsIn
    dsimp only
    refine (List.map_snd_zip ?_).symm
    rw [List.length_map, List.length_range, Array.length_toList, pointCount]
  obtain ⟨ended1, ran1, output1⟩ := output_of_end pub proof messages ops ej atE1 ok outputFree1
  refine ⟨ended0, ended1, ran0, held, ?_, output1⟩
  dsimp only at ran1
  rw [claim1, ← rounds_of_end pub proof messages ops ej atE1 ok endFree1] at ran1
  change sumcheckRun (.equality (recordedIn (stateBefore pub proof messages ops e1)
    ⟨ℓ, s!"range.{0}" ++ ".point", none⟩).toList) _ _ = _ at ran1
  rw [point1] at ran1
  exact sumcheckRun_equality_eqRun ran1

/-- The inputs of the second product stage of `treeRun` on the values of `stage1_ident`. -/
theorem treeInputs_one {ended0 : Ended} {arity0 arity1 : ℕ} {rounds0 rounds1 : List (List Ext × Ext)}
    {children0 children1 : List Ext} {γ0 γ1 : Ext} {point : List Ext} (rest : List Product)
    (ran0 : eqRun Ext.zero Ext.one point rounds0 = some ended0)
    (held : ended0.claim.same (stageEnd 1 arity0 (fun _ => Ext.one) children0) = true) :
    treeInputs (⟨arity0, rounds0, children0, γ0⟩ :: ⟨arity1, rounds1, children1, γ1⟩ :: rest) 1
        (fun _ => Ext.one) point Ext.zero 1 =
      some (1 * arity0, powExt γ0, ended0.point, batched (1 * arity0) children0 γ0) := by
  simp only [treeInputs, ran0, held, ↓reduceIte]

/-- At `range.1.end`, when the stage function of the last check of the range proof at `R`
accepts: the check holds on the state before it. -/
theorem stage1_forward {ℓ a0 e0 c0 a1 e1 c1 R vars low arity arity1 nodes1 : ℕ} {key0 : Key}
    (ae0 : a0 < e0)
    (ec0 : e0 < c0) (ca1 : c0 < a1) (ae1 : a1 < e1) (ej : e1 < c1) (jR : c1 < R)
    (atA0 : ops[a0]? = some (.absorb ⟨ℓ, s!"range.{0}" ++ ".claim", none⟩ .zero))
    (atE0 : ops[e0]? = some (.ended ⟨ℓ, s!"range.{0}", none⟩ vars (.turned ⟨ℓ, "tau0", none⟩ low)))
    (atC0 : ops[c0]? = some (.check key0 (.stage 1 arity .ones ⟨ℓ, s!"range.{0}.output", none⟩
      ⟨ℓ, s!"range.{0}.children", none⟩)))
    (atA1 : ops[a1]? = some (.absorb ⟨ℓ, s!"range.{1}" ++ ".claim", none⟩
      (.batched ⟨ℓ, s!"range.{0}.children", none⟩ ⟨ℓ, s!"range.{0}.batch", none⟩)))
    (atE1 : ops[e1]? = some (.ended ⟨ℓ, s!"range.{1}", none⟩ vars
      (.point ⟨ℓ, s!"range.{0}" ++ ".point", none⟩)))
    (ok : (runFrom pub proof messages (ops.zipIdx.toList.take R) (channelStart pub)).1 = .ok ())
    (claimFree0 : ∀ y ∈ ops.zipIdx.toList.take a0,
      (⟨ℓ, s!"range.{0}" ++ ".claim", none⟩ : Key) ∉ y.1.writes)
    (outputFree0 : ∀ y ∈ ops.zipIdx.toList.take e0,
      (⟨ℓ, s!"range.{0}" ++ ".output", none⟩ : Key) ∉ y.1.writes)
    (endFree0 : ∀ y ∈ ops.zipIdx.toList.take e0, (⟨ℓ, s!"range.{0}", none⟩ : Key) ∉ y.1.ends)
    (tauFree : ∀ y ∈ (ops.zipIdx.toList.take R).drop e0, (⟨ℓ, "tau0", none⟩ : Key) ∉ y.1.writes)
    (outputAfter0 : ∀ y ∈ (ops.zipIdx.toList.take R).drop c0,
      (⟨ℓ, s!"range.{0}.output", none⟩ : Key) ∉ y.1.writes)
    (childAfter0 : ∀ y ∈ (ops.zipIdx.toList.take R).drop c0,
      (⟨ℓ, s!"range.{0}.children", none⟩ : Key) ∉ y.1.writes)
    (batchAfter0 : ∀ y ∈ (ops.zipIdx.toList.take R).drop a1,
      (⟨ℓ, s!"range.{0}.batch", none⟩ : Key) ∉ y.1.writes)
    (pointAfter0 : ∀ y ∈ (ops.zipIdx.toList.take R).drop e0,
      (⟨ℓ, s!"range.{0}" ++ ".point", none⟩ : Key) ∉ y.1.writes)
    (pointCount : (recordedIn (stateBefore pub proof messages ops e0)
      ⟨ℓ, s!"range.{0}" ++ ".point", none⟩).size = vars)
    (childCount : (recordedIn (stateBefore pub proof messages ops R)
      ⟨ℓ, s!"range.{0}.children", none⟩).size = 1 * arity)
    (claimFree1 : ∀ y ∈ ops.zipIdx.toList.take a1,
      (⟨ℓ, s!"range.{1}" ++ ".claim", none⟩ : Key) ∉ y.1.writes)
    (outputFree1 : ∀ y ∈ ops.zipIdx.toList.take e1,
      (⟨ℓ, s!"range.{1}" ++ ".output", none⟩ : Key) ∉ y.1.writes)
    (endFree1 : ∀ y ∈ ops.zipIdx.toList.take e1, (⟨ℓ, s!"range.{1}", none⟩ : Key) ∉ y.1.ends)
    (nodes : nodes1 = 1 * arity)
    (outputAfter1 : ∀ y ∈ (ops.zipIdx.toList.take R).drop c1,
      (⟨ℓ, s!"range.{1}.output", none⟩ : Key) ∉ y.1.writes)
    (childAfter1 : ∀ y ∈ (ops.zipIdx.toList.take R).drop c1,
      (⟨ℓ, s!"range.{1}.children", none⟩ : Key) ∉ y.1.writes)
    {β : Type} (last : ℕ → (ℕ → Ext) → List Ext → Ext → Option β) (rest : List Product) (γ1 : Ext)
    (accepts : treeRun last
        (⟨arity, roundsIn (stateBefore pub proof messages ops R) ⟨ℓ, s!"range.{0}", none⟩,
          (recordedIn (stateBefore pub proof messages ops R) ⟨ℓ, s!"range.{0}.children", none⟩).toList,
          firstIn (stateBefore pub proof messages ops R) ⟨ℓ, s!"range.{0}.batch", none⟩⟩ ::
        ⟨arity1, roundsIn (stateBefore pub proof messages ops R) ⟨ℓ, s!"range.{1}", none⟩,
          (recordedIn (stateBefore pub proof messages ops R) ⟨ℓ, s!"range.{1}.children", none⟩).toList,
          γ1⟩ :: rest) 1 (fun _ => Ext.one)
        (pointIn (stateBefore pub proof messages ops R) (.turned ⟨ℓ, "tau0", none⟩ low)) Ext.zero ≠
      none) :
    (((holds pub (.stage nodes1 arity1 (.powers ⟨ℓ, s!"range.{0}.batch", none⟩)
      ⟨ℓ, s!"range.{1}.output", none⟩ ⟨ℓ, s!"range.{1}.children", none⟩)).run).run
        (stateBefore pub proof messages ops c1)).1 = .ok true := by
  obtain ⟨ended0, ended1, ran0, held0, ran1, output1⟩ := stage1_ident pub proof messages ops
    (j := R) ae0 ec0 ca1 ae1 (ej.trans jR) atA0 atE0 atC0 atA1 atE1 ok claimFree0 outputFree0
    endFree0 tauFree outputAfter0 childAfter0 batchAfter0 pointAfter0 pointCount childCount
    claimFree1 outputFree1 endFree1
  obtain ⟨x, inputs, held⟩ := treeRun_stage last 1 _ 1 _ _ _ rfl accepts
  rw [treeInputs_one rest ran0 held0] at inputs
  cases inputs
  unfold stageHolds at held
  dsimp only at held
  rw [ran1] at held
  rw [run_holds_stage]
  simp only [weightsIn]
  have outEq : firstIn (stateBefore pub proof messages ops c1) ⟨ℓ, s!"range.{1}.output", none⟩ =
      firstIn (stateBefore pub proof messages ops R) ⟨ℓ, s!"range.{1}.output", none⟩ := by
    unfold firstIn
    rw [recorded_between pub proof messages ops jR.le ok outputAfter1]
  have batchEq : firstIn (stateBefore pub proof messages ops c1) ⟨ℓ, s!"range.{0}.batch", none⟩ =
      firstIn (stateBefore pub proof messages ops R) ⟨ℓ, s!"range.{0}.batch", none⟩ := by
    unfold firstIn
    rw [recorded_between pub proof messages ops jR.le ok
      (fun y member => batchAfter0 y (mem_drop_of_le (ae1.trans ej).le member))]
  rw [outEq, batchEq, ← output_key, output1, nodes,
    ← recorded_between pub proof messages ops jR.le ok childAfter1]
  exact congrArg _ held

/-- A stop at `range.1.end`: `treeRun` fails at its second product stage on the state before the
check, for each later stage and each leaf. -/
theorem stage1_backward {ℓ a0 e0 c0 a1 e1 c1 vars low arity arity1 nodes1 : ℕ} {key0 : Key}
    (ae0 : a0 < e0)
    (ec0 : e0 < c0) (ca1 : c0 < a1) (ae1 : a1 < e1) (ej : e1 < c1)
    (atA0 : ops[a0]? = some (.absorb ⟨ℓ, s!"range.{0}" ++ ".claim", none⟩ .zero))
    (atE0 : ops[e0]? = some (.ended ⟨ℓ, s!"range.{0}", none⟩ vars (.turned ⟨ℓ, "tau0", none⟩ low)))
    (atC0 : ops[c0]? = some (.check key0 (.stage 1 arity .ones ⟨ℓ, s!"range.{0}.output", none⟩
      ⟨ℓ, s!"range.{0}.children", none⟩)))
    (atA1 : ops[a1]? = some (.absorb ⟨ℓ, s!"range.{1}" ++ ".claim", none⟩
      (.batched ⟨ℓ, s!"range.{0}.children", none⟩ ⟨ℓ, s!"range.{0}.batch", none⟩)))
    (atE1 : ops[e1]? = some (.ended ⟨ℓ, s!"range.{1}", none⟩ vars
      (.point ⟨ℓ, s!"range.{0}" ++ ".point", none⟩)))
    (ok : (runFrom pub proof messages (ops.zipIdx.toList.take c1) (channelStart pub)).1 = .ok ())
    (claimFree0 : ∀ y ∈ ops.zipIdx.toList.take a0,
      (⟨ℓ, s!"range.{0}" ++ ".claim", none⟩ : Key) ∉ y.1.writes)
    (outputFree0 : ∀ y ∈ ops.zipIdx.toList.take e0,
      (⟨ℓ, s!"range.{0}" ++ ".output", none⟩ : Key) ∉ y.1.writes)
    (endFree0 : ∀ y ∈ ops.zipIdx.toList.take e0, (⟨ℓ, s!"range.{0}", none⟩ : Key) ∉ y.1.ends)
    (tauFree : ∀ y ∈ (ops.zipIdx.toList.take c1).drop e0, (⟨ℓ, "tau0", none⟩ : Key) ∉ y.1.writes)
    (outputAfter0 : ∀ y ∈ (ops.zipIdx.toList.take c1).drop c0,
      (⟨ℓ, s!"range.{0}.output", none⟩ : Key) ∉ y.1.writes)
    (childAfter0 : ∀ y ∈ (ops.zipIdx.toList.take c1).drop c0,
      (⟨ℓ, s!"range.{0}.children", none⟩ : Key) ∉ y.1.writes)
    (batchAfter0 : ∀ y ∈ (ops.zipIdx.toList.take c1).drop a1,
      (⟨ℓ, s!"range.{0}.batch", none⟩ : Key) ∉ y.1.writes)
    (pointAfter0 : ∀ y ∈ (ops.zipIdx.toList.take c1).drop e0,
      (⟨ℓ, s!"range.{0}" ++ ".point", none⟩ : Key) ∉ y.1.writes)
    (pointCount : (recordedIn (stateBefore pub proof messages ops e0)
      ⟨ℓ, s!"range.{0}" ++ ".point", none⟩).size = vars)
    (childCount : (recordedIn (stateBefore pub proof messages ops c1)
      ⟨ℓ, s!"range.{0}.children", none⟩).size = 1 * arity)
    (claimFree1 : ∀ y ∈ ops.zipIdx.toList.take a1,
      (⟨ℓ, s!"range.{1}" ++ ".claim", none⟩ : Key) ∉ y.1.writes)
    (outputFree1 : ∀ y ∈ ops.zipIdx.toList.take e1,
      (⟨ℓ, s!"range.{1}" ++ ".output", none⟩ : Key) ∉ y.1.writes)
    (endFree1 : ∀ y ∈ ops.zipIdx.toList.take e1, (⟨ℓ, s!"range.{1}", none⟩ : Key) ∉ y.1.ends)
    (nodes : nodes1 = 1 * arity)
    (fails : (((holds pub (.stage nodes1 arity1 (.powers ⟨ℓ, s!"range.{0}.batch", none⟩)
      ⟨ℓ, s!"range.{1}.output", none⟩ ⟨ℓ, s!"range.{1}.children", none⟩)).run).run
        (stateBefore pub proof messages ops c1)).1 = .ok false)
    {β : Type} (last : ℕ → (ℕ → Ext) → List Ext → Ext → Option β) (rest : List Product)
    (γ1 : Ext) :
    treeRun last
        (⟨arity, roundsIn (stateBefore pub proof messages ops c1) ⟨ℓ, s!"range.{0}", none⟩,
          (recordedIn (stateBefore pub proof messages ops c1) ⟨ℓ, s!"range.{0}.children", none⟩).toList,
          firstIn (stateBefore pub proof messages ops c1) ⟨ℓ, s!"range.{0}.batch", none⟩⟩ ::
        ⟨arity1, roundsIn (stateBefore pub proof messages ops c1) ⟨ℓ, s!"range.{1}", none⟩,
          (recordedIn (stateBefore pub proof messages ops c1) ⟨ℓ, s!"range.{1}.children", none⟩).toList,
          γ1⟩ :: rest) 1 (fun _ => Ext.one)
        (pointIn (stateBefore pub proof messages ops c1) (.turned ⟨ℓ, "tau0", none⟩ low)) Ext.zero =
      none := by
  obtain ⟨ended0, ended1, ran0, held0, ran1, output1⟩ := stage1_ident pub proof messages ops
    (j := c1) ae0 ec0 ca1 ae1 ej atA0 atE0 atC0 atA1 atE1 ok claimFree0 outputFree0 endFree0
    tauFree outputAfter0 childAfter0 batchAfter0 pointAfter0 pointCount childCount claimFree1
    outputFree1 endFree1
  refine treeRun_none_of_stage last 1 _ 1 _ _ _ rfl (treeInputs_one rest ran0 held0) ?_
  unfold stageHolds
  dsimp only
  rw [ran1]
  rw [run_holds_stage] at fails
  simp only [weightsIn, ← output_key, output1, nodes, Except.ok.injEq] at fails
  exact fails

end Stages

/-- The level function is `none` when `treeRun` of the product stages of the range proof fails on
the recorded values for each leaf. -/
theorem foldRunIn_none_of_tree (s : Replayer) (ℓ : ℕ) (F : Fold)
    (tree : ∀ {β : Type} (last : ℕ → (ℕ → Ext) → List Ext → Ext → Option β),
      treeRun last (productsIn s ℓ ((productStages F.logBasis).map (·.1))) 1 (fun _ => Ext.one)
        (pointIn s (.turned ⟨ℓ, "tau0", none⟩ F.low)) Ext.zero = none) :
    foldRunIn pub s ℓ F = none := by
  unfold foldRunIn
  cases pub.setupFor F.setupCount with
  | none => rfl
  | some setup =>
      simp only [Option.bind_some]
      split
      · rw [foldRunSq_if]
        have empty : treeRun (normRun (leafDegree F.logBasis) F.limit (normsIn s ℓ).energy
            (normsIn s ℓ).merge (normsIn s ℓ).rounds (normsIn s ℓ).virtual (wordsIn s ℓ F).image)
            (wordsIn s ℓ F).stages 1 (fun _ => Ext.one) ((wordsIn s ℓ F).point F.low) Ext.zero =
              none := tree _
        rw [empty]
        rfl
      · rw [foldRun_if]
        have empty : rangeRun (2 ^ F.logBasis) (wordsIn s ℓ F).stages
            (roundsIn s ⟨ℓ, "range.leaf", none⟩) (wordsIn s ℓ F).image ((wordsIn s ℓ F).point F.low) =
              none := tree _
        rw [empty]
        rfl

end Replayer

section Shape

open Replayer

/-- The absorb of the claim zero of the first product stage of `level`. -/
def isClaim0 (level : ℕ) : Op → Bool
  | .absorb key .zero => key == ⟨level, s!"range.{0}" ++ ".claim", none⟩
  | _ => false

/-- The absorb of the batched claim of the second product stage of `level`. -/
def isClaim1 (level : ℕ) : Op → Bool
  | .absorb key (.batched children gamma) => key == ⟨level, s!"range.{1}" ++ ".claim", none⟩ &&
      children == ⟨level, s!"range.{0}.children", none⟩ && gamma == ⟨level, s!"range.{0}.batch", none⟩
  | _ => false

/-- The end of the rounds of the first product stage of `level` from the turned `τ_0`. -/
def isEnd0 (level vars low : ℕ) : Op → Bool
  | .ended key n (.turned t l) => key == ⟨level, s!"range.{0}", none⟩ && n == vars &&
      t == ⟨level, "tau0", none⟩ && l == low
  | _ => false

/-- The end of the rounds of the second product stage of `level` from the point of the first. -/
def isEnd1 (level vars : ℕ) : Op → Bool
  | .ended key n (.point p) => key == ⟨level, s!"range.{1}", none⟩ && n == vars &&
      p == ⟨level, s!"range.{0}" ++ ".point", none⟩
  | _ => false

/-- The check `range.0.end` of `level`. -/
def isCheck0 (level arity : ℕ) : Op → Bool
  | .check _ (.stage n a .ones out ch) => n == 1 && a == arity &&
      out == ⟨level, s!"range.{0}.output", none⟩ && ch == ⟨level, s!"range.{0}.children", none⟩
  | _ => false

/-- The check `range.1.end` of `level`. -/
def isCheck1 (level nodes arity : ℕ) : Op → Bool
  | .check _ (.stage n a (.powers g) out ch) => n == nodes && a == arity &&
      g == ⟨level, s!"range.{0}.batch", none⟩ && out == ⟨level, s!"range.{1}.output", none⟩ &&
      ch == ⟨level, s!"range.{1}.children", none⟩
  | _ => false

theorem isClaim0_sound {level : ℕ} {op : Op} (h : isClaim0 level op = true) :
    op = .absorb ⟨level, s!"range.{0}" ++ ".claim", none⟩ .zero := by
  unfold isClaim0 at h
  split at h
  · simp only [beq_iff_eq] at h
    rw [h]
  · cases h

theorem isClaim1_sound {level : ℕ} {op : Op} (h : isClaim1 level op = true) :
    op = .absorb ⟨level, s!"range.{1}" ++ ".claim", none⟩
      (.batched ⟨level, s!"range.{0}.children", none⟩ ⟨level, s!"range.{0}.batch", none⟩) := by
  unfold isClaim1 at h
  split at h
  · simp only [Bool.and_eq_true, beq_iff_eq] at h
    obtain ⟨⟨rfl, rfl⟩, rfl⟩ := h
    rfl
  · cases h

theorem isEnd0_sound {level vars low : ℕ} {op : Op} (h : isEnd0 level vars low op = true) :
    op = .ended ⟨level, s!"range.{0}", none⟩ vars (.turned ⟨level, "tau0", none⟩ low) := by
  unfold isEnd0 at h
  split at h
  · simp only [Bool.and_eq_true, beq_iff_eq] at h
    obtain ⟨⟨⟨rfl, rfl⟩, rfl⟩, rfl⟩ := h
    rfl
  · cases h

theorem isEnd1_sound {level vars : ℕ} {op : Op} (h : isEnd1 level vars op = true) :
    op = .ended ⟨level, s!"range.{1}", none⟩ vars (.point ⟨level, s!"range.{0}" ++ ".point", none⟩) := by
  unfold isEnd1 at h
  split at h
  · simp only [Bool.and_eq_true, beq_iff_eq] at h
    obtain ⟨⟨rfl, rfl⟩, rfl⟩ := h
    rfl
  · cases h

theorem isCheck0_sound {level arity : ℕ} {op : Op} (h : isCheck0 level arity op = true) :
    ∃ key, op = .check key (.stage 1 arity .ones ⟨level, s!"range.{0}.output", none⟩
      ⟨level, s!"range.{0}.children", none⟩) := by
  unfold isCheck0 at h
  split at h
  · rename_i key _ _ _ _
    simp only [Bool.and_eq_true, beq_iff_eq] at h
    obtain ⟨⟨⟨rfl, rfl⟩, rfl⟩, rfl⟩ := h
    exact ⟨key, rfl⟩
  · cases h

theorem isCheck1_sound {level nodes arity : ℕ} {op : Op} (h : isCheck1 level nodes arity op = true) :
    ∃ key, op = .check key (.stage nodes arity (.powers ⟨level, s!"range.{0}.batch", none⟩)
      ⟨level, s!"range.{1}.output", none⟩ ⟨level, s!"range.{1}.children", none⟩) := by
  unfold isCheck1 at h
  split at h
  · rename_i key _ _ _ _ _
    simp only [Bool.and_eq_true, beq_iff_eq] at h
    obtain ⟨⟨⟨⟨rfl, rfl⟩, rfl⟩, rfl⟩, rfl⟩ := h
    exact ⟨key, rfl⟩
  · cases h

/-- The shape of the first product stage of `level` with the claim at `a`, the end at `e` and the
check `range.0.end` at `c`, before the last check of the range proof at `R`. -/
def stage0At (l : List (Op × ℕ)) (level : ℕ) (F : Fold) (R arity a e c : ℕ) : Bool :=
  (l[a]?.any fun x => isClaim0 level x.1) && (l[e]?.any fun x => isEnd0 level F.vars F.low x.1) &&
    (l[c]?.any fun x => isCheck0 level arity x.1) && decide (a < e) && decide (e < c) &&
    decide (c < R) && frees (l.take a) [⟨level, s!"range.{0}" ++ ".claim", none⟩] [] &&
    frees (l.take e) [⟨level, s!"range.{0}" ++ ".output", none⟩] [⟨level, s!"range.{0}", none⟩] &&
    frees ((l.take R).drop e) [⟨level, "tau0", none⟩] [] &&
    frees ((l.take R).drop c)
      [⟨level, s!"range.{0}.output", none⟩, ⟨level, s!"range.{0}.children", none⟩] []

/-- The shape of the second product stage of `level` with the claim at `a1`, the end at `e1` and
the check `range.1.end` at `c1`, after the first stage at `(a0, e0, c0)`. -/
def stage1At (l : List (Op × ℕ)) (level : ℕ) (F : Fold) (R arity nodes arity1 e0 c0 a1 e1 c1 : ℕ) :
    Bool :=
  (l[a1]?.any fun x => isClaim1 level x.1) && (l[e1]?.any fun x => isEnd1 level F.vars x.1) &&
    (l[c1]?.any fun x => isCheck1 level nodes arity1 x.1) && decide (c0 < a1) && decide (a1 < e1) &&
    decide (e1 < c1) && decide (c1 < R) && nodes == 1 * arity &&
    frees ((l.take R).drop a1) [⟨level, s!"range.{0}.batch", none⟩] [] &&
    frees ((l.take R).drop e0) [⟨level, s!"range.{0}" ++ ".point", none⟩] [] &&
    drawnCount ⟨level, s!"range.{0}" ++ ".point", none⟩ (l.take e0) == some F.vars &&
    extsCount ⟨level, s!"range.{0}.children", none⟩ (l.take c1) == some (1 * arity) &&
    extsCount ⟨level, s!"range.{0}.children", none⟩ (l.take R) == some (1 * arity) &&
    frees (l.take a1) [⟨level, s!"range.{1}" ++ ".claim", none⟩] [] &&
    frees (l.take e1) [⟨level, s!"range.{1}" ++ ".output", none⟩] [⟨level, s!"range.{1}", none⟩] &&
    frees ((l.take R).drop c1)
      [⟨level, s!"range.{1}.output", none⟩, ⟨level, s!"range.{1}.children", none⟩] []

/-- The first index of `l` whose operation has `p`. -/
def indexOf (l : List (Op × ℕ)) (p : Op → Bool) : ℕ := (l.findIdx? fun x => p x.1).getD 0

/-- The shapes of the product stages of the range proof of the fold `F` of `level` in `l`: at most
two stages, each with `stage0At` or `stage1At` at the first indices of its operations. -/
def stagesShape (l : List (Op × ℕ)) (level : ℕ) (F : Fold) : Bool :=
  (l[indexOf l (isRangeCheck level F)]?.any fun x => isRangeCheck level F x.1) &&
    match productStages F.logBasis with
    | [] => true
    | [(arity, _)] => stage0At l level F (indexOf l (isRangeCheck level F)) arity
        (indexOf l (isClaim0 level)) (indexOf l (isEnd0 level F.vars F.low))
        (indexOf l (isCheck0 level arity))
    | [(arity, nodes), (arity1, _)] =>
        stage0At l level F (indexOf l (isRangeCheck level F)) arity (indexOf l (isClaim0 level))
          (indexOf l (isEnd0 level F.vars F.low)) (indexOf l (isCheck0 level arity)) &&
        stage1At l level F (indexOf l (isRangeCheck level F)) arity nodes arity1
          (indexOf l (isEnd0 level F.vars F.low)) (indexOf l (isCheck0 level arity))
          (indexOf l (isClaim1 level)) (indexOf l (isEnd1 level F.vars))
          (indexOf l (isCheck1 level nodes arity1))
    | _ => false

theorem mem_drop_take_of_le {l : List (Op × ℕ)} {m n n' : ℕ} (le : n ≤ n') {y : Op × ℕ}
    (member : y ∈ (l.take n).drop m) : y ∈ (l.take n').drop m := by
  rw [List.drop_take] at member ⊢
  have cut : (l.drop m).take (n - m) = ((l.drop m).take (n' - m)).take (n - m) := by
    rw [List.take_take, Nat.min_eq_left (by omega)]
  rw [cut] at member
  exact List.mem_of_mem_take member

theorem Op.erase_absorb {op : Op} {key : Key} {v : Value} (h : op.erase = .absorb key v) :
    op = .absorb key v := by
  cases op <;> first | exact h | cases h

theorem Op.erase_ended {op : Op} {key : Key} {n : ℕ} {before : Before}
    (h : op.erase = .ended key n before) : op = .ended key n before := by
  cases op <;> first | exact h | cases h

variable {L : Layout} {ops : Array Op}

theorem erased_at (erased : ops.toList.map Op.erase = skeleton L) {c : ℕ} {p : Op → Bool}
    (shape : (skeleton L).zipIdx[c]?.any (fun x => p x.1) = true) :
    ∃ op op', p op = true ∧ ops[c]? = some op' ∧ op'.erase = op := by
  rw [List.getElem?_zipIdx] at shape
  rcases found : (skeleton L)[c]? with _ | op
  · rw [found] at shape
    cases shape
  · rw [found] at shape
    rw [← erased, List.getElem?_map] at found
    obtain ⟨op', found', same⟩ := Option.map_eq_some_iff.mp found
    exact ⟨op, op', shape, by rw [← Array.getElem?_toList]; exact found', same⟩

theorem erased_frees_take (erased : ops.toList.map Op.erase = skeleton L) {a : ℕ} {K R : List Key}
    (shape : frees ((skeleton L).zipIdx.take a) K R = true) :
    frees (ops.zipIdx.toList.take a) K R = true := by
  rw [← take_erase erased, frees_erase] at shape
  exact shape

theorem erased_frees_drop (erased : ops.toList.map Op.erase = skeleton L) {R e : ℕ}
    {K K' : List Key} (shape : frees (((skeleton L).zipIdx.take R).drop e) K K' = true) :
    frees ((ops.zipIdx.toList.take R).drop e) K K' = true := by
  rw [← take_erase erased, ← List.map_drop, frees_erase] at shape
  exact shape

theorem erased_drawnCount (erased : ops.toList.map Op.erase = skeleton L) (key : Key) (e : ℕ) :
    drawnCount key (ops.zipIdx.toList.take e) = drawnCount key ((skeleton L).zipIdx.take e) := by
  rw [← take_erase erased, drawnCount_erase]

theorem erased_extsCount (erased : ops.toList.map Op.erase = skeleton L) (key : Key) (e : ℕ) :
    extsCount key (ops.zipIdx.toList.take e) = extsCount key ((skeleton L).zipIdx.take e) := by
  rw [← take_erase erased, extsCount_erase]

/-- The stage function of the last check of the range proof accepts only when `treeRun` of the
product stages accepts for some leaf. -/
theorem tree_of_accepts (pub : Public) (s : Replayer) (ℓ : ℕ) (F : Fold)
    (accepts : ((stageValue pub s (rangeCheckOf ℓ F)).map StageOut.accepts).getD false = true) :
    ∃ (β : Type) (last : ℕ → (ℕ → Ext) → List Ext → Ext → Option β),
      treeRun last (productsIn s ℓ ((productStages F.logBasis).map (·.1))) 1 (fun _ => Ext.one)
        (pointIn s (.turned ⟨ℓ, "tau0", none⟩ F.low)) Ext.zero ≠ none := by
  cases euclidean : F.euclidean
  · rw [rangeCheckOf_plain ℓ F euclidean] at accepts
    simp only [stageValue, Option.map_some, Option.getD_some, StageOut.accepts] at accepts
    exact ⟨_, _, Option.isSome_iff_ne_none.mp accepts⟩
  · rw [rangeCheckOf_euclidean ℓ F euclidean] at accepts
    simp only [stageValue, Option.map_some, Option.getD_some, StageOut.accepts] at accepts
    exact ⟨_, _, Option.isSome_iff_ne_none.mp accepts⟩

/-- The first product stage from its shape on the skeleton. -/
theorem stage0_of_shape (pub : Public) (proof : ByteArray) (messages : Array Message)
    (erased : ops.toList.map Op.erase = skeleton L) {ℓ : ℕ} {F : Fold} {arity : ℕ}
    {arities : List ℕ} (hp : (productStages F.logBasis).map (·.1) = arity :: arities)
    {R a e c : ℕ} (shape : stage0At (skeleton L).zipIdx ℓ F R arity a e c = true) :
    c < R ∧ (∃ key, ops[c]? = some (.check key (.stage 1 arity .ones
        ⟨ℓ, s!"range.{0}.output", none⟩ ⟨ℓ, s!"range.{0}.children", none⟩))) ∧
      ((runFrom pub proof messages (ops.zipIdx.toList.take R) (channelStart pub)).1 = .ok () →
        ((stageValue pub (stateBefore pub proof messages ops R) (rangeCheckOf ℓ F)).map
          StageOut.accepts).getD false = true →
        (((holds pub (.stage 1 arity .ones ⟨ℓ, s!"range.{0}.output", none⟩
          ⟨ℓ, s!"range.{0}.children", none⟩)).run).run
            (stateBefore pub proof messages ops c)).1 = .ok true) ∧
      ((runFrom pub proof messages (ops.zipIdx.toList.take c) (channelStart pub)).1 = .ok () →
        (((holds pub (.stage 1 arity .ones ⟨ℓ, s!"range.{0}.output", none⟩
          ⟨ℓ, s!"range.{0}.children", none⟩)).run).run
            (stateBefore pub proof messages ops c)).1 = .ok false →
        foldRunIn pub (stateBefore pub proof messages ops c) ℓ F = none) := by
  unfold stage0At at shape
  simp only [Bool.and_eq_true, decide_eq_true_eq] at shape
  obtain ⟨⟨⟨⟨⟨⟨⟨⟨⟨claimAt, endAt⟩, checkAt⟩, ae⟩, ec⟩, cR⟩, claimFree⟩, outFree⟩, tauFree⟩,
    afterFree⟩ := shape
  obtain ⟨opA, opA', isA, atA, sameA⟩ := erased_at erased claimAt
  rw [isClaim0_sound isA] at sameA
  rw [Op.erase_absorb sameA] at atA
  obtain ⟨opE, opE', isE, atE, sameE⟩ := erased_at erased endAt
  rw [isEnd0_sound isE] at sameE
  rw [Op.erase_ended sameE] at atE
  obtain ⟨opC, opC', isC, atC, sameC⟩ := erased_at erased checkAt
  obtain ⟨keyC, hC⟩ := isCheck0_sound isC
  rw [hC] at sameC
  rw [Op.erase_check sameC] at atC
  have claimFree' := erased_frees_take erased claimFree
  have outFree' := erased_frees_take erased outFree
  have tauFree' := erased_frees_drop erased tauFree
  have afterFree' := erased_frees_drop erased afterFree
  refine ⟨cR, ⟨keyC, atC⟩, fun okR accepts => ?_, fun okC fails => ?_⟩
  · obtain ⟨β, last, tree⟩ := tree_of_accepts pub _ ℓ F accepts
    rw [hp] at tree
    exact stage0_forward pub proof messages ops ae ec cR atA atE okR
      (frees_writes claimFree' (List.mem_singleton_self _))
      (frees_writes outFree' (List.mem_singleton_self _))
      (frees_ends outFree' (List.mem_singleton_self _))
      (frees_writes tauFree' (List.mem_singleton_self _))
      (frees_writes afterFree' (by simp)) (frees_writes afterFree' (by simp)) last _ _ tree
  · refine foldRunIn_none_of_tree pub _ ℓ F fun last => ?_
    rw [hp]
    exact stage0_backward pub proof messages ops ae ec atA atE okC
      (frees_writes claimFree' (List.mem_singleton_self _))
      (frees_writes outFree' (List.mem_singleton_self _))
      (frees_ends outFree' (List.mem_singleton_self _))
      (fun y member => frees_writes tauFree' (List.mem_singleton_self _) y
        (mem_drop_take_of_le cR.le member)) fails last _ _

/-- The second product stage from the shapes of the two stages on the skeleton. -/
theorem stage1_of_shape (pub : Public) (proof : ByteArray) (messages : Array Message)
    (erased : ops.toList.map Op.erase = skeleton L)
    (aligned : ReadsAligned messages ops.zipIdx.toList 0) {ℓ : ℕ} {F : Fold}
    {arity arity1 nodes : ℕ} {arities : List ℕ}
    (hp : (productStages F.logBasis).map (·.1) = arity :: arity1 :: arities)
    {R a0 e0 c0 a1 e1 c1 : ℕ} (shape0 : stage0At (skeleton L).zipIdx ℓ F R arity a0 e0 c0 = true)
    (shape1 : stage1At (skeleton L).zipIdx ℓ F R arity nodes arity1 e0 c0 a1 e1 c1 = true) :
    c1 < R ∧ (∃ key, ops[c1]? = some (.check key (.stage nodes arity1
        (.powers ⟨ℓ, s!"range.{0}.batch", none⟩) ⟨ℓ, s!"range.{1}.output", none⟩
        ⟨ℓ, s!"range.{1}.children", none⟩))) ∧
      ((runFrom pub proof messages (ops.zipIdx.toList.take R) (channelStart pub)).1 = .ok () →
        ((stageValue pub (stateBefore pub proof messages ops R) (rangeCheckOf ℓ F)).map
          StageOut.accepts).getD false = true →
        (((holds pub (.stage nodes arity1 (.powers ⟨ℓ, s!"range.{0}.batch", none⟩)
          ⟨ℓ, s!"range.{1}.output", none⟩ ⟨ℓ, s!"range.{1}.children", none⟩)).run).run
            (stateBefore pub proof messages ops c1)).1 = .ok true) ∧
      ((runFrom pub proof messages (ops.zipIdx.toList.take c1) (channelStart pub)).1 = .ok () →
        (((holds pub (.stage nodes arity1 (.powers ⟨ℓ, s!"range.{0}.batch", none⟩)
          ⟨ℓ, s!"range.{1}.output", none⟩ ⟨ℓ, s!"range.{1}.children", none⟩)).run).run
            (stateBefore pub proof messages ops c1)).1 = .ok false →
        foldRunIn pub (stateBefore pub proof messages ops c1) ℓ F = none) := by
  unfold stage0At at shape0
  simp only [Bool.and_eq_true, decide_eq_true_eq] at shape0
  obtain ⟨⟨⟨⟨⟨⟨⟨⟨⟨claimAt, endAt⟩, checkAt⟩, ae0⟩, ec0⟩, c0R⟩, claimFree⟩, outFree⟩, tauFree⟩,
    afterFree⟩ := shape0
  unfold stage1At at shape1
  simp only [Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at shape1
  obtain ⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨claim1At, end1At⟩, check1At⟩, ca1⟩, ae1⟩, ec1⟩, c1R⟩, nodesEq⟩,
    batchFree⟩, pointFree⟩, pointCnt⟩, childCnt1⟩, childCntR⟩, claimFree1⟩, outFree1⟩,
    afterFree1⟩ := shape1
  obtain ⟨opA, opA', isA, atA0, sameA⟩ := erased_at erased claimAt
  rw [isClaim0_sound isA] at sameA
  rw [Op.erase_absorb sameA] at atA0
  obtain ⟨opE, opE', isE, atE0, sameE⟩ := erased_at erased endAt
  rw [isEnd0_sound isE] at sameE
  rw [Op.erase_ended sameE] at atE0
  obtain ⟨opC, opC', isC, atC0, sameC⟩ := erased_at erased checkAt
  obtain ⟨keyC, hC⟩ := isCheck0_sound isC
  rw [hC] at sameC
  rw [Op.erase_check sameC] at atC0
  obtain ⟨opA1, opA1', isA1, atA1, sameA1⟩ := erased_at erased claim1At
  rw [isClaim1_sound isA1] at sameA1
  rw [Op.erase_absorb sameA1] at atA1
  obtain ⟨opE1, opE1', isE1, atE1, sameE1⟩ := erased_at erased end1At
  rw [isEnd1_sound isE1] at sameE1
  rw [Op.erase_ended sameE1] at atE1
  obtain ⟨opC1, opC1', isC1, atC1, sameC1⟩ := erased_at erased check1At
  obtain ⟨keyC1, hC1⟩ := isCheck1_sound isC1
  rw [hC1] at sameC1
  rw [Op.erase_check sameC1] at atC1
  have claimFree' := erased_frees_take erased claimFree
  have outFree' := erased_frees_take erased outFree
  have tauFree' := erased_frees_drop erased tauFree
  have afterFree' := erased_frees_drop erased afterFree
  have batchFree' := erased_frees_drop erased batchFree
  have pointFree' := erased_frees_drop erased pointFree
  have claimFree1' := erased_frees_take erased claimFree1
  have outFree1' := erased_frees_take erased outFree1
  have afterFree1' := erased_frees_drop erased afterFree1
  have pointCount : ∀ {j : ℕ}, e0 ≤ j →
      (runFrom pub proof messages (ops.zipIdx.toList.take j) (channelStart pub)).1 = .ok () →
      (recordedIn (stateBefore pub proof messages ops e0)
        ⟨ℓ, s!"range.{0}" ++ ".point", none⟩).size = F.vars := fun le ok => by
    have okE0 := (stateBefore_le pub proof messages ops le ok).1
    unfold stateBefore
    rw [size_drawnCount pub proof messages _ _ _ (by rw [erased_drawnCount erased]; exact pointCnt)
      okE0, channelStart_fresh]
    simp
  have childCount : ∀ {j : ℕ}, extsCount ⟨ℓ, s!"range.{0}.children", none⟩
      ((skeleton L).zipIdx.take j) = some (1 * arity) →
      (runFrom pub proof messages (ops.zipIdx.toList.take j) (channelStart pub)).1 = .ok () →
      (recordedIn (stateBefore pub proof messages ops j)
        ⟨ℓ, s!"range.{0}.children", none⟩).size = 1 * arity := fun count ok => by
    unfold stateBefore
    rw [size_extsCount pub proof messages _ _ _ (by rw [erased_extsCount erased]; exact count)
      (by rw [channelStart_next]; exact aligned_take messages _ _ 0 aligned) ok, channelStart_fresh]
    simp
  have e0c1 : e0 ≤ c1 := (ec0.trans (ca1.trans (ae1.trans ec1))).le
  refine ⟨c1R, ⟨keyC1, atC1⟩, fun okR accepts => ?_, fun okC fails => ?_⟩
  · obtain ⟨β, last, tree⟩ := tree_of_accepts pub _ ℓ F accepts
    rw [hp] at tree
    exact stage1_forward pub proof messages ops ae0 ec0 ca1 ae1 ec1 c1R atA0 atE0 atC0 atA1 atE1
      okR (frees_writes claimFree' (List.mem_singleton_self _))
      (frees_writes outFree' (List.mem_singleton_self _))
      (frees_ends outFree' (List.mem_singleton_self _))
      (frees_writes tauFree' (List.mem_singleton_self _)) (frees_writes afterFree' (by simp))
      (frees_writes afterFree' (by simp)) (frees_writes batchFree' (List.mem_singleton_self _))
      (frees_writes pointFree' (List.mem_singleton_self _))
      (pointCount (e0c1.trans c1R.le) okR) (childCount childCntR okR)
      (frees_writes claimFree1' (List.mem_singleton_self _))
      (frees_writes outFree1' (List.mem_singleton_self _))
      (frees_ends outFree1' (List.mem_singleton_self _)) nodesEq
      (frees_writes afterFree1' (by simp)) (frees_writes afterFree1' (by simp)) last _ _ tree
  · refine foldRunIn_none_of_tree pub _ ℓ F fun last => ?_
    rw [hp]
    have inner : ∀ {k : Key} {d : ℕ}, (∀ y ∈ (ops.zipIdx.toList.take R).drop d, k ∉ y.1.writes) →
        ∀ y ∈ (ops.zipIdx.toList.take c1).drop d, k ∉ y.1.writes :=
      fun free y member => free y (mem_drop_take_of_le c1R.le member)
    exact stage1_backward pub proof messages ops ae0 ec0 ca1 ae1 ec1 atA0 atE0 atC0 atA1 atE1 okC
      (frees_writes claimFree' (List.mem_singleton_self _))
      (frees_writes outFree' (List.mem_singleton_self _))
      (frees_ends outFree' (List.mem_singleton_self _))
      (inner (frees_writes tauFree' (List.mem_singleton_self _)))
      (inner (frees_writes afterFree' (by simp))) (inner (frees_writes afterFree' (by simp)))
      (inner (frees_writes batchFree' (List.mem_singleton_self _)))
      (inner (frees_writes pointFree' (List.mem_singleton_self _)))
      (pointCount e0c1 okC) (childCount childCnt1 okC)
      (frees_writes claimFree1' (List.mem_singleton_self _))
      (frees_writes outFree1' (List.mem_singleton_self _))
      (frees_ends outFree1' (List.mem_singleton_self _)) nodesEq fails last _ _

/-- Finding 16 for `range.k.end`: on operations whose erased forms are the skeleton of a layout
with `stagesShape` at the fold `F` of `ℓ`, each product stage `k` has its check `range.k.end` at
some `c` before the last check of the range proof at `R`. The check holds on the state before it
when the run reaches `R` and the stage function there accepts; and when the check does not hold on
the state before it, the level function is `none` there. -/
theorem erased_stages (pub : Public) (proof : ByteArray) (messages : Array Message)
    (erased : ops.toList.map Op.erase = skeleton L)
    (aligned : ReadsAligned messages ops.zipIdx.toList 0) {ℓ : ℕ} {F : Fold}
    (shaped : stagesShape (skeleton L).zipIdx ℓ F = true) :
    ∃ R, (∃ key, ops[R]? = some (.check key (rangeCheckOf ℓ F))) ∧
      ∀ k < (productStages F.logBasis).length, ∃ c nodes arity weights, c < R ∧
        (∃ key, ops[c]? = some (.check key (.stage nodes arity weights
          ⟨ℓ, s!"range.{k}.output", none⟩ ⟨ℓ, s!"range.{k}.children", none⟩))) ∧
        ((runFrom pub proof messages (ops.zipIdx.toList.take R) (channelStart pub)).1 = .ok () →
          ((stageValue pub (stateBefore pub proof messages ops R) (rangeCheckOf ℓ F)).map
            StageOut.accepts).getD false = true →
          (((holds pub (.stage nodes arity weights ⟨ℓ, s!"range.{k}.output", none⟩
            ⟨ℓ, s!"range.{k}.children", none⟩)).run).run
              (stateBefore pub proof messages ops c)).1 = .ok true) ∧
        ((runFrom pub proof messages (ops.zipIdx.toList.take c) (channelStart pub)).1 = .ok () →
          (((holds pub (.stage nodes arity weights ⟨ℓ, s!"range.{k}.output", none⟩
            ⟨ℓ, s!"range.{k}.children", none⟩)).run).run
              (stateBefore pub proof messages ops c)).1 = .ok false →
          foldRunIn pub (stateBefore pub proof messages ops c) ℓ F = none) := by
  unfold stagesShape at shaped
  simp only [Bool.and_eq_true] at shaped
  obtain ⟨rangeAt, stagesAt⟩ := shaped
  obtain ⟨opR, opR', isR, atR, sameR⟩ := erased_at erased rangeAt
  obtain ⟨keyR, hR⟩ := isRangeCheck_sound isR
  rw [hR] at sameR
  refine ⟨_, ⟨keyR, by rw [atR, Op.erase_check sameR]⟩, fun k below => ?_⟩
  revert stagesAt below
  rcases hp : productStages F.logBasis with _ | ⟨⟨arity, nodes⟩, _ | ⟨⟨arity1, nodes1⟩, _ | ⟨x, rest⟩⟩⟩ <;>
    intro stagesAt below
  · simp at below
  · obtain rfl : k = 0 := by simpa using below
    obtain ⟨cR, check, fwd, bwd⟩ := stage0_of_shape pub proof messages erased
      (by rw [hp]; rfl) stagesAt
    exact ⟨_, _, _, _, cR, check, fwd, bwd⟩
  · simp only [Bool.and_eq_true] at stagesAt
    obtain ⟨first, second⟩ := stagesAt
    have small : k = 0 ∨ k = 1 := by
      simp only [List.length_cons, List.length_nil] at below
      omega
    rcases small with rfl | rfl
    · obtain ⟨cR, check, fwd, bwd⟩ := stage0_of_shape pub proof messages erased
        (by rw [hp]; rfl) first
      exact ⟨_, _, _, _, cR, check, fwd, bwd⟩
    · obtain ⟨cR, check, fwd, bwd⟩ := stage1_of_shape pub proof messages erased aligned
        (by rw [hp]; rfl) first second
      exact ⟨_, _, _, _, cR, check, fwd, bwd⟩
  · cases stagesAt

end Shape

end NightstreamFPrime.Spec.Compression.Akita.Replay
