import NightstreamFPrime.Spec.Compression.Akita.Replay.Stage3Weight
import NightstreamFPrime.Spec.Compression.Akita.Extract.Offload

/-! Stage 3 of the pinned Akita verifier on words (Akita revision `d50f700`,
`akita-verifier/src/stages/stage3.rs`, `verify_setup_sumcheck`; ePrint 2026/1983, Section 9 with
(134) and (135)), and its relation to `setupCheck` of the specification.

- `stage3Run` is the plain sum-check from the claim `σ` (`stage3_claim`), and then the end check
  `stage3.end`: the end claim is `value·weight(ρ)` for the prefix evaluation `value` of the proof
  (`stage3_prefix_eval`) and the weight at the point `ρ` of the rounds. `none` is a rejection.
- `stage3Run_runs`: `setupCheck` on the path of the messages and the challenges of the code ends
  in the result of `stage3Run`, for a weight of the code that is the multilinear extension at
  `ρ` of the weight table of the specification. A result `none` is a rejection of the
  specification on the same path. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

open NightstreamFPrime.Spec.Compression.Lib NightstreamFPrime.Spec.Compression.Game
open NightstreamFPrime.Spec.Compression (Game)
open NightstreamFPrime.Spec.Compression.Akita

def stage3Run (claim : Ext) (rounds : List (List Ext × Ext)) (value : Ext)
    (weight : List Ext → Ext) : Option (List Ext × Ext) :=
  if (plainRun claim rounds).claim.same (value * weight (plainRun claim rounds).point) then
    some ((plainRun claim rounds).point, value)
  else none

section Spec

variable {E : Type} [Field E] [Fintype E] [DecidableEq E] [Algebra Zq E] [Degree2 E]

/-- The path of `setupCheck`: the rounds and then the prefix evaluation. -/
noncomputable def stage3Path (E : Type) [Field E] [Algebra Zq E] [Degree2 E] (claim : Ext)
    (rounds : List (List Ext × Ext)) (value : Ext) : List Move :=
  plainPath (Ext.into (E := E)) 1 claim rounds ++ [Move.recv (value.into : E)]

/-- `setupCheck` of the specification, on the path of the messages and the challenges of the
code, ends in the result of `stage3Run`: `none` when the end check fails, and otherwise the point
`ρ` and the prefix evaluation. `weighs`: the weight of the code at the point of the rounds is the
multilinear extension of the weight table `w` there. -/
theorem stage3Run_runs (ν : ℕ) (flat : Table) (w : ℕ → E) (claim : Ext)
    (rounds : List (List Ext × Ext)) (value : Ext) (weight : List Ext → Ext)
    (count : rounds.length = ν) (lengths : ∀ x ∈ rounds, x.1.length = 2)
    (weighs : ((weight (plainRun claim rounds).point).into : E) =
      mle ((plainRun claim rounds).point.map Ext.into) w) :
    (setupCheck ν flat w claim.into).Runs (stage3Path E claim rounds value)
      ((stage3Run claim rounds value weight).map fun out => (out.1.map Ext.into, out.2.into)) := by
  have rounds_run := plainRun_runs (degree2_extHom (E := E)) 1
    (combRound ν ![lifted E flat, w] productComb) rounds [] claim lengths
  rw [count] at rounds_run
  have key : (plainRun claim rounds).claim.same
      (value * weight (plainRun claim rounds).point) = true ↔
        ((plainRun claim rounds).claim.into : E) =
          value.into * mle ((plainRun claim rounds).point.map Ext.into) w := by
    rw [same_degree2 (E := E)]
    show ((plainRun claim rounds).claim.into : E) =
        (Ext.into (value * weight (plainRun claim rounds).point) : E) ↔ _
    rw [Ext.into_mul, weighs]
  refine runs_bind rounds_run (runs_recv ?_)
  show (if ((plainRun claim rounds).claim.into : E) =
        value.into * mle ([] ++ (plainRun claim rounds).point.map Ext.into) w then
      Game.pure (some ([] ++ (plainRun claim rounds).point.map Ext.into, value.into))
    else Game.pure none).Runs [] _
  simp only [List.nil_append]
  unfold stage3Run
  by_cases ends : (plainRun claim rounds).claim.same
      (value * weight (plainRun claim rounds).point) = true
  · rw [ite_eq_left (key.mp ends), ite_eq_left ends]
    rfl
  · rw [ite_eq_right (fun equal => ends (key.mpr equal)), ite_eq_right ends]
    rfl

end Spec

end NightstreamFPrime.Spec.Compression.Akita.Replay
