import NightstreamFPrime.Spec.Compression.Akita.Replay.ShapeRange
import NightstreamFPrime.Spec.Compression.Akita.Replay.ShapeLayout20
import NightstreamFPrime.Spec.Compression.Akita.Replay.ShapeLayout30

/-! Finding 16 for `range.k.end` on the scripts of `layout20` and `layout30` (Akita revision
`d50f700`; `Replay/ShapeRange.lean`).

- `skeleton20_stages0` to `skeleton30_w26_stages6`: `stagesShape` on the skeletons, closed
  computations of the kernel.
- `script20_stages` and `script30_stages`: each product stage `k` of each level `ℓ` of a script of
  the layout has its check `range.k.end` before the last check of the range proof; the check holds
  on the state before it when the stage function of the last check accepts, and when it does not
  hold, the level function is `none` on the state before it.
- `verifyProof_reject_stage20`: a reject of `runProof` at the check `range.k.end` of a level of
  `layout20` gives the level function `none` on the state before the stop. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

open NightstreamFPrime.Spec.Compression.Akita

open Replayer

theorem skeleton20_stages0 : stagesShape (skeleton (layout20 0 0)).zipIdx 0 (fold20 0) = true := by
  decide +kernel

theorem skeleton20_stages1 : stagesShape (skeleton (layout20 0 0)).zipIdx 1 (fold20 1) = true := by
  decide +kernel

theorem skeleton20_stages2 : stagesShape (skeleton (layout20 0 0)).zipIdx 2 (fold20 2) = true := by
  decide +kernel

theorem skeleton20_stages3 : stagesShape (skeleton (layout20 0 0)).zipIdx 3 (fold20 3) = true := by
  decide +kernel

theorem skeleton20_stages4 : stagesShape (skeleton (layout20 0 0)).zipIdx 4 (fold20 4) = true := by
  decide +kernel

theorem skeleton20_stages5 : stagesShape (skeleton (layout20 0 0)).zipIdx 5 (fold20 5) = true := by
  decide +kernel

theorem skeleton30_w27_stages0 :
    stagesShape (skeleton (layout30 Key30.w27 0 0)).zipIdx 0 (fold30 Key30.w27 0) = true := by
  decide +kernel

theorem skeleton30_w27_stages1 :
    stagesShape (skeleton (layout30 Key30.w27 0 0)).zipIdx 1 (fold30 Key30.w27 1) = true := by
  decide +kernel

theorem skeleton30_w27_stages2 :
    stagesShape (skeleton (layout30 Key30.w27 0 0)).zipIdx 2 (fold30 Key30.w27 2) = true := by
  decide +kernel

theorem skeleton30_w27_stages3 :
    stagesShape (skeleton (layout30 Key30.w27 0 0)).zipIdx 3 (fold30 Key30.w27 3) = true := by
  decide +kernel

theorem skeleton30_w27_stages4 :
    stagesShape (skeleton (layout30 Key30.w27 0 0)).zipIdx 4 (fold30 Key30.w27 4) = true := by
  decide +kernel

theorem skeleton30_w27_stages5 :
    stagesShape (skeleton (layout30 Key30.w27 0 0)).zipIdx 5 (fold30 Key30.w27 5) = true := by
  decide +kernel

theorem skeleton30_w27_stages6 :
    stagesShape (skeleton (layout30 Key30.w27 0 0)).zipIdx 6 (fold30 Key30.w27 6) = true := by
  decide +kernel

theorem skeleton30_w26_stages0 :
    stagesShape (skeleton (layout30 Key30.w26 0 0)).zipIdx 0 (fold30 Key30.w26 0) = true := by
  decide +kernel

theorem skeleton30_w26_stages1 :
    stagesShape (skeleton (layout30 Key30.w26 0 0)).zipIdx 1 (fold30 Key30.w26 1) = true := by
  decide +kernel

theorem skeleton30_w26_stages2 :
    stagesShape (skeleton (layout30 Key30.w26 0 0)).zipIdx 2 (fold30 Key30.w26 2) = true := by
  decide +kernel

theorem skeleton30_w26_stages3 :
    stagesShape (skeleton (layout30 Key30.w26 0 0)).zipIdx 3 (fold30 Key30.w26 3) = true := by
  decide +kernel

theorem skeleton30_w26_stages4 :
    stagesShape (skeleton (layout30 Key30.w26 0 0)).zipIdx 4 (fold30 Key30.w26 4) = true := by
  decide +kernel

theorem skeleton30_w26_stages5 :
    stagesShape (skeleton (layout30 Key30.w26 0 0)).zipIdx 5 (fold30 Key30.w26 5) = true := by
  decide +kernel

theorem skeleton30_w26_stages6 :
    stagesShape (skeleton (layout30 Key30.w26 0 0)).zipIdx 6 (fold30 Key30.w26 6) = true := by
  decide +kernel

theorem skeleton20_stages (ℓ : Fin 6) :
    stagesShape (skeleton (layout20 0 0)).zipIdx ℓ (fold20 ℓ) = true := by
  match ℓ with
  | 0 => exact skeleton20_stages0
  | 1 => exact skeleton20_stages1
  | 2 => exact skeleton20_stages2
  | 3 => exact skeleton20_stages3
  | 4 => exact skeleton20_stages4
  | 5 => exact skeleton20_stages5

theorem skeleton30_stages {r : Key30.Row} (member : r ∈ Key30.rows) (ℓ : Fin 7) :
    stagesShape (skeleton (layout30 r 0 0)).zipIdx ℓ (fold30 r ℓ) = true := by
  simp only [Key30.rows, List.mem_cons, List.mem_nil_iff, or_false] at member
  rcases member with rfl | rfl
  · match ℓ with
    | 0 => exact skeleton30_w27_stages0
    | 1 => exact skeleton30_w27_stages1
    | 2 => exact skeleton30_w27_stages2
    | 3 => exact skeleton30_w27_stages3
    | 4 => exact skeleton30_w27_stages4
    | 5 => exact skeleton30_w27_stages5
    | 6 => exact skeleton30_w27_stages6
  · match ℓ with
    | 0 => exact skeleton30_w26_stages0
    | 1 => exact skeleton30_w26_stages1
    | 2 => exact skeleton30_w26_stages2
    | 3 => exact skeleton30_w26_stages3
    | 4 => exact skeleton30_w26_stages4
    | 5 => exact skeleton30_w26_stages5
    | 6 => exact skeleton30_w26_stages6

variable (pub : Public) (proof : ByteArray) (messages : Array Message)

/-- Finding 16 for `range.k.end` on each script of `layout20` with the messages of `decode`. -/
theorem script20_stages {zMax rice : ℕ} {plan : Array Run} {ops : Array Op}
    (scripted : script (layout20 zMax rice) plan = .ok ops) {rest : ℕ}
    (decoded : decode ops proof = .ok (messages, rest)) (ℓ : Fin 6) :
    ∃ R, (∃ key, ops[R]? = some (.check key (rangeCheckOf ℓ (fold20 ℓ)))) ∧
      ∀ k < (productStages (fold20 ℓ).logBasis).length, ∃ c nodes arity weights, c < R ∧
        (∃ key, ops[c]? = some (.check key (.stage nodes arity weights
          ⟨ℓ, s!"range.{k}.output", none⟩ ⟨ℓ, s!"range.{k}.children", none⟩))) ∧
        ((runFrom pub proof messages (ops.zipIdx.toList.take R) (channelStart pub)).1 = .ok () →
          ((stageValue pub (stateBefore pub proof messages ops R) (rangeCheckOf ℓ (fold20 ℓ))).map
            StageOut.accepts).getD false = true →
          (((holds pub (.stage nodes arity weights ⟨ℓ, s!"range.{k}.output", none⟩
            ⟨ℓ, s!"range.{k}.children", none⟩)).run).run
              (stateBefore pub proof messages ops c)).1 = .ok true) ∧
        ((runFrom pub proof messages (ops.zipIdx.toList.take c) (channelStart pub)).1 = .ok () →
          (((holds pub (.stage nodes arity weights ⟨ℓ, s!"range.{k}.output", none⟩
            ⟨ℓ, s!"range.{k}.children", none⟩)).run).run
              (stateBefore pub proof messages ops c)).1 = .ok false →
          foldRunIn pub (stateBefore pub proof messages ops c) ℓ (fold20 ℓ) = none) := by
  have erased : ops.toList.map Op.erase = skeleton (layout20 0 0) := by
    rw [script_erase scripted, skeleton20]
  exact erased_stages pub proof messages erased (decode_aligned decoded) (skeleton20_stages ℓ)

/-- Finding 16 for `range.k.end` on each script of `layout30` with the messages of `decode`. -/
theorem script30_stages {r : Key30.Row} (member : r ∈ Key30.rows) {zMax rice : ℕ}
    {plan : Array Run} {ops : Array Op} (scripted : script (layout30 r zMax rice) plan = .ok ops)
    {rest : ℕ} (decoded : decode ops proof = .ok (messages, rest)) (ℓ : Fin 7) :
    ∃ R, (∃ key, ops[R]? = some (.check key (rangeCheckOf ℓ (fold30 r ℓ)))) ∧
      ∀ k < (productStages (fold30 r ℓ).logBasis).length, ∃ c nodes arity weights, c < R ∧
        (∃ key, ops[c]? = some (.check key (.stage nodes arity weights
          ⟨ℓ, s!"range.{k}.output", none⟩ ⟨ℓ, s!"range.{k}.children", none⟩))) ∧
        ((runFrom pub proof messages (ops.zipIdx.toList.take R) (channelStart pub)).1 = .ok () →
          ((stageValue pub (stateBefore pub proof messages ops R)
            (rangeCheckOf ℓ (fold30 r ℓ))).map StageOut.accepts).getD false = true →
          (((holds pub (.stage nodes arity weights ⟨ℓ, s!"range.{k}.output", none⟩
            ⟨ℓ, s!"range.{k}.children", none⟩)).run).run
              (stateBefore pub proof messages ops c)).1 = .ok true) ∧
        ((runFrom pub proof messages (ops.zipIdx.toList.take c) (channelStart pub)).1 = .ok () →
          (((holds pub (.stage nodes arity weights ⟨ℓ, s!"range.{k}.output", none⟩
            ⟨ℓ, s!"range.{k}.children", none⟩)).run).run
              (stateBefore pub proof messages ops c)).1 = .ok false →
          foldRunIn pub (stateBefore pub proof messages ops c) ℓ (fold30 r ℓ) = none) := by
  have erased : ops.toList.map Op.erase = skeleton (layout30 r 0 0) := by
    rw [script_erase scripted, skeleton30]
  exact erased_stages pub proof messages erased (decode_aligned decoded)
    (skeleton30_stages member ℓ)

/-- A reject of `runProof` at the check `range.k.end` of the level `ℓ` of `layout20`: the level
function is `none` on the state before the stop. -/
theorem verifyProof_reject_stage20 {zMax rice : ℕ} (layout : pub.layout = layout20 zMax rice)
    {ops : Array Op} (scripted : script pub.layout pub.plan = .ok ops) {rest : ℕ}
    (decoded : decode ops proof = .ok (messages, rest)) (ℓ : Fin 6) {k : ℕ}
    (below : k < (productStages (fold20 ℓ).logBasis).length) :
    ∃ c nodes arity weights, (∃ key, ops[c]? = some (.check key (.stage nodes arity weights
        ⟨ℓ, s!"range.{k}.output", none⟩ ⟨ℓ, s!"range.{k}.children", none⟩))) ∧
      ∀ site, runProof pub proof = .verdict (.reject c site) →
        foldRunIn pub (stateBefore pub proof messages ops c) ℓ (fold20 ℓ) = none := by
  obtain ⟨R, -, stages⟩ := script20_stages pub proof messages (layout ▸ scripted) decoded ℓ
  obtain ⟨c, nodes, arity, weights, -, ⟨key, atC⟩, -, stop⟩ := stages k below
  refine ⟨c, nodes, arity, weights, ⟨key, atC⟩, fun site rejected => ?_⟩
  obtain ⟨m, rest', decoded', -, before, held⟩ := verifyProof_reject pub proof scripted rejected
  rw [decoded] at decoded'
  cases decoded'
  rcases held with ⟨key', check, at', -, fails⟩ | ⟨level, at', -⟩
  · rw [atC] at at'
    cases at'
    exact stop before fails
  · rw [atC] at at'
    cases at'

end NightstreamFPrime.Spec.Compression.Akita.Replay
