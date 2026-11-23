import NightstreamFPrime.Spec.Compression.Akita.Replay.Transcript
import NightstreamFPrime.Spec.Compression.Akita.Replay.WeightTraceSpec

/-! The end check of the fused sum-check of each level and the end check of Stage 3 in the replay
(`Replay/Transcript.lean`, `stageValue`; Akita revision `d50f700`: `validate_stage2_replay` of
`stages/stage2.rs` after `verify_stage3` in `fold/mod.rs`, `verify_setup_sumcheck` of
`stages/stage3.rs`).

- `step_fused`: at `fused.end` the replay records the end check and rejects exactly when it does
  not hold.
- `step_fused_runs`: at `fused.end` of a level with evaluation trace with the shapes `T` and the
  setup prefix `setup`, the replay records `fusedIn` and rejects exactly when it does not hold,
  and it holds exactly when the final check of `fusedCheck` of the specification holds for
  `traceReduction` of `traceStatementOf` (`traceFused_holds_iff_of`), for recorded values of the
  shapes of the script.
- `step_packing_runs`: at `fused.end` of a level with coefficient packing with the shapes `P`,
  the replay records `foldFused` of the recorded values and rejects exactly when it does not
  hold.
- `step_stage3_runs`: at `stage3.end` the replay records `stage3In`, and `setupCheck` of the
  specification ends in its result on the path of the recorded values (`stage3Run_runs`), for a
  weight table `w` whose multilinear extension at the point of Stage 3 is the weight of the
  code. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

open NightstreamFPrime.Spec.Compression.Lib NightstreamFPrime.Spec.Compression.Game
open NightstreamFPrime.Spec.Compression.Akita

namespace Replayer

/-- The step of `fused.end` records the end check `f` of the stage function and rejects exactly
when it does not hold. -/
theorem step_fused (pub : Public) (proof : ByteArray) (messages : Array Message) (i : ℕ)
    (key : Key) (level : ℕ) (F : Fold) (s : Replayer) {f : FusedEnd}
    (value : stageValue pub s (.fused level F) = some (.fused f.passing)) :
    ((step pub proof messages i (.check key (.fused level F))).run).run s =
      if f.holds then (.ok (), checked (staged s key (.fused (some f))) key)
      else (.error (.reject i key.site), staged s key (.fused none)) := by
  rw [step_stage pub proof messages i key _ s value]
  unfold FusedEnd.passing
  cases f.holds <;> rfl

/-- `fused.end` of a level with coefficient packing: the replay records `foldFused` of the recorded
values (`packingIn`) and rejects exactly when it does not hold. -/
theorem step_packing_runs (pub : Public) (proof : ByteArray) (messages : Array Message) (i : ℕ)
    (key : Key) (level : ℕ) (F : Fold) (s : Replayer) {P : PackShape}
    (shaped : F.fused = .packing P) {setup : Array Fq}
    (expanded : pub.setupFor (P.setupCount F.stage3.isSome) = some setup) :
    let f := foldFused (packInputsIn P setup s level F) (packGroupsIn P s level F)
      (packingValuesIn s level F) (setupClaimIn s level F)
    ((step pub proof messages i (.check key (.fused level F))).run).run s =
      if f.holds then (.ok (), checked (staged s key (.fused (some f))) key)
      else (.error (.reject i key.site), staged s key (.fused none)) := by
  intro f
  refine step_fused pub proof messages i key level F s ?_
  simp only [stageValue, Fold.setupCount, fusedEndIn, shaped, expanded, Option.map_some]
  rfl

variable {E : Type} [Field E] [Fintype E] [DecidableEq E] [Algebra Zq E] [Degree2 E]

omit [DecidableEq E] in
/-- `fused.end` of a trace level: the replay records `fusedIn` of the recorded values and rejects
exactly when it fails, and it holds exactly when the final check of `fusedCheck` (with the weight
of `foldCheckSq` at a Euclidean level) holds for `traceReduction` of the statement of the setup
prefix, with the fold challenges that the replay drew. -/
theorem step_fused_runs (pub : Public) (proof : ByteArray) (messages : Array Message) (i : ℕ)
    (key : Key) (level : ℕ) (F : Fold) (s : Replayer) {T : TraceShape}
    (shaped : F.fused = .trace T) {setup : Array Fq}
    (expanded : pub.setupFor (traceSetupCount T.opening) = some setup)
    (quarter : 4 * (T.shape.innerDegree / 4) = T.shape.innerDegree)
    (two : ((2 * (T.shape.innerDegree / 4) : ℕ) : Zq) ≠ 0) (payload : T.shape.Payload)
    (openPayload : T.opening.Payload) (value : E) {μ : ℕ} (fits : T.opening.cells T.fold ≤ 2 ^ μ)
    {batch : ℕ} (rowsFit : T.opening.traceRowCount + 1 ≤ 2 ^ batch)
    (digits : 0 < T.fold.responseDigits) (length : (traceValuesIn s level F).r2.length = μ)
    (virtualLength : (traceValuesIn s level F).virtual.length = μ)
    (tau : (traceValuesIn s level F).tau1.length = batch)
    (compressed : F.compressed = T.opening.compressed T.fold) :
    let V := traceValuesIn s level F
    let challenge : Fin T.shape.blocks → Rq T.shape.innerDegree := fun j =>
      Poly.toRq T.shape.innerDegree (((drawnIn s level 0).map
        (Poly.ofSparse T.shape.innerDegree)).getD j (Poly.zero T.shape.innerDegree))
    ((step pub proof messages i (.check key (.fused level F))).run).run s =
        (if (fusedIn T setup s level F).holds then
          (.ok (), checked (staged s key (.fused (some (fusedIn T setup s level F)))) key)
        else (.error (.reject i key.site), staged s key (.fused none))) ∧
      ((fusedIn T setup s level F).holds = true ↔
        (V.output.into : E) = V.value.into * mle (V.r2.map Ext.into) (fun x =>
        ((traceReduction (traceStatementOf T.opening quarter setup V payload openPayload value)
          challenge μ fits batch rowsFit).rows V.alpha.into).batchWeight (V.tau1.map Ext.into) x +
          (if V.euclidean then (V.norm.into : E) else 0) *
            normWeight μ (T.shape.normKernel T.fold) (V.virtual.map Ext.into) x) +
        mle (V.r2.map Ext.into) (virtWeight (T.opening.binarySupport T.fold)
          (V.virtual.map Ext.into) V.batch.into (if V.compressed then (V.binary.into : E) else 0)) *
          (V.value.into * (V.value.into + 1))) := by
  intro V challenge
  have unfolded : fusedIn T setup s level F = traceFused T.opening T.fold setup
      ((drawnIn s level 0).map (Poly.ofSparse T.shape.innerDegree)) V := rfl
  refine ⟨step_fused pub proof messages i key level F s ?_, ?_⟩
  · simp only [stageValue, Fold.setupCount, fusedEndIn, shaped, expanded, Option.map_some]
  · rw [unfolded]
    exact traceFused_holds_iff_of quarter two setup _ V payload openPayload value challenge
      (fun _ => rfl) fits rowsFit digits length virtualLength tau compressed

/-- `stage3.end`: the replay records `stage3In` of the recorded values and rejects when it is
`none`, and `setupCheck` of the specification ends in its result on the path of the recorded
values, for a weight table `w` whose multilinear extension at the point of the rounds is the
weight `stage3Weight` of the plan. -/
theorem step_stage3_runs (pub : Public) (proof : ByteArray) (messages : Array Message) (i : ℕ)
    (key : Key) (level : ℕ) (plan : SetupPlan) (s : Replayer) (ν : ℕ) (flat : Table)
    (w : ℕ → E) (count : (roundsIn s ⟨level, "stage3", none⟩).length = ν)
    (lengths : ∀ x ∈ roundsIn s ⟨level, "stage3", none⟩, x.1.length = 2)
    (weighs : ((stage3Weight plan (firstIn s ⟨level, "alpha", none⟩)
        (recordedIn s ⟨level, "tau1", none⟩).toList (recordedIn s ⟨level, "fused.point", none⟩).toList
        (plainRun (firstIn s ⟨level, "stage3.claim", none⟩)
          (roundsIn s ⟨level, "stage3", none⟩)).point).into : E) =
      mle ((plainRun (firstIn s ⟨level, "stage3.claim", none⟩)
        (roundsIn s ⟨level, "stage3", none⟩)).point.map Ext.into) w) :
    let out := stage3In s level plan
    ((step pub proof messages i (.check key (.stage3 level plan))).run).run s =
        (if out.isSome then (.ok (), checked (staged s key (.stage3 out)) key)
        else (.error (.reject i key.site), staged s key (.stage3 out))) ∧
      (setupCheck ν flat w (firstIn s ⟨level, "stage3.claim", none⟩).into).Runs
        (stage3Path E (firstIn s ⟨level, "stage3.claim", none⟩) (roundsIn s ⟨level, "stage3", none⟩)
          (firstIn s ⟨level, "stage3.prefix_evaluation", none⟩))
        (out.map fun o => (o.1.map Ext.into, o.2.into)) := by
  intro out
  exact ⟨step_stage (out := .stage3 out) pub proof messages i key _ s rfl,
    stage3Run_runs ν flat w _ _ _ _ count lengths weighs⟩

end Replayer

end NightstreamFPrime.Spec.Compression.Akita.Replay
