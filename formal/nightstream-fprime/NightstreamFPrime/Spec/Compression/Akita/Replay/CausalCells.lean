import NightstreamFPrime.Spec.Compression.Akita.Replay.JoinGiven
import NightstreamFPrime.Spec.Compression.Akita.Replay.CausalTrace
import NightstreamFPrime.Spec.Compression.Akita.Replay.CausalSegments

/-! The cells of the wire message `k` of the model from the state of the replay before the work of
the stage `k`, for `key20.aks` (Akita revision `d50f700`; `docs/akita-work/w20/design.md`, module
W and decision D1; `Replay/JoinGiven.lean`, `Replay/Causal.lean`, `Replay/CausalTrace.lean`).

- `roundsFrom s key n` are the rounds of the sum-check `key` from the recorded values of a state:
  the message of each round `r < n` and the challenge `r` of its point (zero when the state does
  not have it). `productFrom` is one product stage with these rounds.
- `pathAt20 E pub s ℓ` is the path of the level `ℓ` from the recorded values of the one state
  `s`, with no condition on a check (`treeFull`, `normFull`, `closingFull`, `tensorFull`) and with
  no read of `derived.rounds`.
- `recvs20 E pub s k` are the messages of the checks of the wire message `k`:
  `Extract.segments` of `pathAt20` for the level and the draw of `stage20 k`. `recvs20_core`: it
  reads the state only through its core.
- `cellsAt20 E pub proof messages k` are the cells of `recvs20` on the state before the work of
  the stage `k`. `cellsAt20_causal`: they are a function of the events of the sponge before the
  squeeze of this work.
- `Need` is one value that the path reads, with the draw from which the path reads it:
  `Need.Holds j s t` says that two states have the same value when the draw is at most `j`.
  `needs20 ℓ` are the needs of `pathAt20`, and `pathAt20_same`: two states with the needs for `j`
  have the same path before its draw `j + 1`.
- `Need.check k₀ L` is the check of a need on the skeleton of `layout20` (no operation from the
  work of its stage to `L` writes the key; the challenge is drawn before the work of its stage),
  and `Need.holds` its use on a run without a stop. `needs20_checked`: each need of each level
  has the check. `pathAt20_stage`: the state before the work of the stage `k` and the state
  before the operation `last20 ℓ` have the same path before the draw after the stage.
- `cellsAt20_eq_of_final`: `cellsAt20 … k = cells20 … k` for a run without a stop of a script of
  `layout20`, when `pathAt20` on the state before `last20 ℓ` is `path20 … ℓ` for each level.
- `foldPathIn_final`: the path `foldPathIn` of a fold that passes is `pathAt` on a state with the
  same values and with `roundsFrom` equal to its rounds. `roundsCheck` and `roundsIn_of_check`:
  the rounds of a sum-check are `roundsFrom` on a later state. `tensor_passes`: the tensor
  reduction of an accepted proof passes. `final20_checked₀` to `final20_checked₅`: the checks
  on the skeleton.
- `pathAt20_final`: for an accepted proof of `layout20`, `pathAt20` on the state before
  `last20 ℓ` is `path20 … ℓ`. `cellsAt20_eq`: for an accepted proof, `cellsAt20 … k` is
  `cells20 … k` for each `k`. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

open NightstreamFPrime.Spec.Compression.Lib NightstreamFPrime.Spec.Compression.Game
open NightstreamFPrime.Spec.Compression (Game)
open NightstreamFPrime.Spec.Compression.Akita
open NightstreamFPrime.Spec.Compression.Akita.Extract (Same countDraws leading)

namespace Replayer

/-- The rounds of the sum-check `key` with `n` rounds from the recorded values: the message of
each round and the challenge of the round in the recorded point, or zero. -/
def roundsFrom (s : Replayer) (key : Key) (n : ℕ) : List (List Ext × Ext) :=
  (List.range n).map fun r =>
    ((recordedIn s ⟨key.level, key.site ++ ".round", some r⟩).toList,
      (recordedIn s ⟨key.level, key.site ++ ".point", none⟩).toList.getD r Ext.zero)

theorem length_roundsFrom (s : Replayer) (key : Key) (n : ℕ) : (roundsFrom s key n).length = n := by
  unfold roundsFrom
  rw [List.length_map, List.length_range]

/-- The product stage `x.2` of the arity `x.1` with the rounds `roundsFrom`. -/
def productFrom (s : Replayer) (level n : ℕ) (x : ℕ × ℕ) : Product :=
  { arity := x.1, rounds := roundsFrom s ⟨level, s!"range.{x.2}", none⟩ n
    children := (recordedIn s ⟨level, s!"range.{x.2}.children", none⟩).toList
    gamma := firstIn s ⟨level, s!"range.{x.2}.batch", none⟩ }

end Replayer

open Replayer

section Paths

variable (E : Type) [Field E] [Algebra Zq E] [Degree2 E] (pub : Public)

/-- The path after the last stage of the range proof from the recorded values of `s`. -/
noncomputable def closingAt (s : Replayer) (level : ℕ) (F : Fold) (bin : Bool) (n : ℕ)
    (after : List Move) : List Move :=
  closingFull E bin (firstIn s ⟨level, "binary_batch", none⟩)
    (firstIn s ⟨level, "stage2_batch", none⟩) (stage2In pub s level F).fused
    (roundsFrom s ⟨level, "fused", none⟩ n) (firstIn s ⟨level, "witness_evaluation", none⟩) after

/-- The path of the last stage of the range proof (the norm stage on the Euclidean route `sq`,
and the leaf stage on the coefficient route) and the path after it, from the recorded values of
`s`. -/
noncomputable def lastAt (s : Replayer) (level : ℕ) (F : Fold) (sq bin : Bool) (n : ℕ)
    (after : List Move) (claim : Ext) : List Move :=
  match sq with
  | true =>
      normFull (Ext.into (E := E)) (leafDegree F.logBasis)
        ((s.integers.get? ⟨level, "norm.response_l2_sq", none⟩).getD 0)
        (firstIn s ⟨level, "norm.merge", none⟩) (roundsFrom s ⟨level, "norm.sumcheck", none⟩ n)
        (firstIn s ⟨level, "norm.virtual_evaluations", none⟩)
        (firstIn s ⟨level, "range.image", none⟩)
        (Move.draw ((firstIn s ⟨level, "norm.virtual_batch", none⟩).into : E) ::
          closingAt E pub s level F bin n after) claim
  | false =>
      leafPath (Ext.into (E := E)) (rangeLeaf (2 ^ F.logBasis))
          (roundsFrom s ⟨level, "range.leaf", none⟩ n) (firstIn s ⟨level, "range.image", none⟩) ++
        closingAt E pub s level F bin n after

/-- The path of one fold with `n` variables and the product stages `stages` from the recorded
values of `s`, and the path `after`. -/
noncomputable def pathAt (s : Replayer) (level : ℕ) (F : Fold) (sq bin : Bool) (n batch : ℕ)
    (stages : List (ℕ × ℕ)) (after : List Move) : List Move :=
  Move.draw ((firstIn s ⟨level, "alpha", none⟩).into : E) ::
    Move.draw (listFn n ((recordedIn s ⟨level, "tau0", none⟩).toList.map Ext.into) : Fin n → E) ::
      Move.draw (listFn batch ((recordedIn s ⟨level, "tau1", none⟩).toList.map Ext.into) :
          Fin batch → E) ::
        treeFull (Ext.into (E := E)) (lastAt E pub s level F sq bin n after)
          (stages.map (productFrom s level n)) 1 Ext.zero

/-- The path of the tensor reduction of the claim of the level `level` from the recorded values
of `s`. -/
noncomputable def tensorAt (s : Replayer) (level ν : ℕ) : List Move :=
  tensorFull E ((recordedIn s ⟨level + 1, "tensor.partials", none⟩).getD 0 Ext.zero)
    ((recordedIn s ⟨level + 1, "tensor.partials", none⟩).getD 1 Ext.zero)
    (firstIn s ⟨level + 1, "tensor.eta", none⟩)
    (roundsFrom s ⟨level + 1, "tensor.sumcheck", none⟩ ν)
    (firstIn s ⟨level + 1, "tensor.final_claims", none⟩)

/-- The path after the fold of the level `ℓ`: for `ℓ ≥ 1` the tensor reduction of its claim. -/
noncomputable def afterAt (s : Replayer) (ℓ : Fin 6) : List Move :=
  if ℓ.val = 0 then [] else tensorAt E s ℓ (tail20 ℓ)

/-- The path of the level `ℓ` of `key20.aks` from the recorded values of the state `s`: the fold,
and for `ℓ ≥ 1` the tensor reduction of its claim. -/
noncomputable def pathAt20 (s : Replayer) (ℓ : Fin 6) : List Move :=
  pathAt E pub s ℓ (fold20 ℓ) (fold20 ℓ).euclidean (fold20 ℓ).compressed (fold20 ℓ).vars
    (fold20 ℓ).batch ((productStages (fold20 ℓ).logBasis).map (·.1)).zipIdx (afterAt E s ℓ)

/-- The messages of the checks of the wire message `k` of the model from the recorded values of
the state `s`: the messages of `pathAt20` after the draw of `stage20 k` and before the next draw.
A wire message without a level has no message. -/
noncomputable def recvs20 (s : Replayer) (k : ℕ) : List Move :=
  match stage20 k with
  | some (ℓ, j) => Extract.segments (pathAt20 E pub s ℓ) j
  | none => []

end Paths

section Core

variable {s s' : Replayer}

theorem core_recorded (same : core s = core s') (key : Key) :
    recordedIn s key = recordedIn s' key :=
  congrFun (congrArg Core.recorded same) key

theorem core_fields (same : core s = core s') (key : Key) :
    s.fields.get? key = s'.fields.get? key :=
  congrFun (congrArg Core.fields same) key

theorem core_integers (same : core s = core s') (key : Key) :
    s.integers.get? key = s'.integers.get? key :=
  congrFun (congrArg Core.integers same) key

variable (pub : Public)

theorem stage2In_core (same : core s = core s') (level : ℕ) (F : Fold) :
    stage2In pub s level F = stage2In pub s' level F := by
  have data : dataIn pub s level = dataIn pub s' level := by
    funext run
    unfold dataIn
    simp only [core_fields same]
  unfold stage2In
  rw [data]
  simp only [rowsIn, claimsIn, firstIn, core_recorded same]

variable (E : Type) [Field E] [Algebra Zq E] [Degree2 E]

/-- `pathAt20` reads a state only through its core. -/
theorem pathAt20_core (same : core s = core s') (ℓ : Fin 6) :
    pathAt20 E pub s ℓ = pathAt20 E pub s' ℓ := by
  have recorded := core_recorded same
  have integers := core_integers same
  have rounds : ∀ key n, roundsFrom s key n = roundsFrom s' key n := fun key n => by
    simp only [roundsFrom, recorded]
  have first : ∀ key, firstIn s key = firstIn s' key := fun key => by
    simp only [firstIn, recorded]
  have product : ∀ level n, productFrom s level n = productFrom s' level n := fun level n => by
    funext x
    simp only [productFrom, rounds, first, recorded]
  have closing : ∀ level F bin n after, closingAt E pub s level F bin n after =
      closingAt E pub s' level F bin n after := fun level F bin n after => by
    simp only [closingAt, rounds, first, stage2In_core pub same]
  have last : ∀ level F sq bin n after, lastAt E pub s level F sq bin n after =
      lastAt E pub s' level F sq bin n after := fun level F sq bin n after => by
    funext claim
    simp only [lastAt, closing, rounds, first, integers]
  simp only [pathAt20, pathAt, afterAt, tensorAt, product, last, first, recorded, rounds]

/-- `recvs20` reads a state only through its core. -/
theorem recvs20_core (same : core s = core s') (k : ℕ) :
    recvs20 E pub s k = recvs20 E pub s' k := by
  unfold recvs20
  cases stage20 k with
  | none => rfl
  | some at_ => simp only [pathAt20_core pub E same]

end Core

section Cells

variable (E : Type) [Field E] [Algebra Zq E] [Degree2 E]
  (pub : Public) (proof proof' : ByteArray) (messages messages' : Array Message)

/-- The cells of the wire message `k` of the model from the state of the replay before the work
of the stage `k`: the cell `i` has the message `i` of `recvs20`. -/
noncomputable def cellsAt20 (k : ℕ) : Fin 4 → Extract.Cell E 8 3029804899 :=
  Extract.cellsFrom (recvs20 E pub (stateBefore pub proof messages pub.ops (stageIndex20 pub k)) k) 0

variable {pub proof proof' messages messages'}

/-- The cells of the wire message `k` of the model are a function of the public input and the
events of the sponge before the squeeze of the work of the stage `k`: two proofs with their
decoded messages, whose runs have no stop before this work and have the same events before its
squeeze, have the same `cellsAt20 … k`. -/
theorem cellsAt20_causal {k : ℕ} {rest rest' : ℕ}
    (decoded : decode pub.ops proof = .ok (messages, rest))
    (decoded' : decode pub.ops proof' = .ok (messages', rest'))
    (ran : (runFrom pub proof messages (pub.ops.zipIdx.toList.take (stageIndex20 pub k))
      (channelStart pub)).1 = .ok ())
    (ran' : (runFrom pub proof' messages' (pub.ops.zipIdx.toList.take (stageIndex20 pub k))
      (channelStart pub)).1 = .ok ())
    (events : eventsAt20 pub proof messages k = eventsAt20 pub proof' messages' k) :
    cellsAt20 E pub proof messages k = cellsAt20 E pub proof' messages' k := by
  have core := core_causal pub proof proof' messages messages' _ decoded decoded' ran ran'
    (eventsAt20_parts events).2
  unfold cellsAt20
  rw [recvs20_core pub E core]

end Cells

section Needs

variable {j : ℕ} {s t : Replayer}

/-- The rounds `roundsFrom` of two states with the needs of the rounds, at the draw `base + r`:
close rounds before the end of the rounds, and equal rounds from the end. -/
theorem roundsFrom_agree {base n : ℕ} {key : Key}
    (needs : ∀ x ∈ roundNeeds base key n, x.Holds j s t) (r : ℕ) (at_ : j = base + r) :
    (r < n → Close r (roundsFrom s key n) (roundsFrom t key n)) ∧
      (n ≤ r → roundsFrom s key n = roundsFrom t key n) := by
  have message : ∀ i < n, base + i ≤ j →
      recordedIn s ⟨key.level, key.site ++ ".round", some i⟩ =
        recordedIn t ⟨key.level, key.site ++ ".round", some i⟩ := fun i below from_ =>
    ((needs (.values (base + i) ⟨key.level, key.site ++ ".round", some i⟩)
      (List.mem_flatMap.mpr ⟨i, List.mem_range.mpr below, List.mem_cons_self⟩)).of_values from_).1
  have drawn : ∀ i < n, base + i + 1 ≤ j →
      (recordedIn s ⟨key.level, key.site ++ ".point", none⟩).toList.getD i Ext.zero =
        (recordedIn t ⟨key.level, key.site ++ ".point", none⟩).toList.getD i Ext.zero :=
    fun i below from_ =>
      (needs (.value (base + i + 1) ⟨key.level, key.site ++ ".point", none⟩ i)
        (List.mem_flatMap.mpr ⟨i, List.mem_range.mpr below,
          List.mem_cons_of_mem _ List.mem_cons_self⟩)).of_value from_
  have entry : ∀ i < n, base + i + 1 ≤ j →
      ((recordedIn s ⟨key.level, key.site ++ ".round", some i⟩).toList,
        (recordedIn s ⟨key.level, key.site ++ ".point", none⟩).toList.getD i Ext.zero) =
      ((recordedIn t ⟨key.level, key.site ++ ".round", some i⟩).toList,
        (recordedIn t ⟨key.level, key.site ++ ".point", none⟩).toList.getD i Ext.zero) :=
    fun i below from_ => by rw [message i below (by omega), drawn i below from_]
  constructor
  · intro lt
    refine ⟨by rw [length_roundsFrom]; exact lt, ?_, ?_⟩
    · unfold roundsFrom
      rw [← List.map_take, ← List.map_take]
      refine List.map_congr_left fun i member => ?_
      rw [List.take_range] at member
      have below := List.mem_range.mp member
      exact entry i (by omega) (by omega)
    · unfold roundsFrom
      rw [List.getElem?_map, List.getElem?_map, List.getElem?_range lt]
      show some (recordedIn s _).toList = some (recordedIn t _).toList
      rw [message r lt (by omega)]
  · intro ge
    unfold roundsFrom
    refine List.map_congr_left fun i member => ?_
    have below := List.mem_range.mp member
    exact entry i below (by omega)

variable (E : Type) [Field E] [Algebra Zq E] [Degree2 E] (pub : Public)

/-- The tensor reduction of two states with its needs. -/
theorem tensorAt_same {level ν base : ℕ} (le : base ≤ j)
    (needs : ∀ x ∈ tensorNeeds level ν base, x.Holds j s t) :
    Same (j - base) (tensorAt E s level ν) (tensorAt E t level ν) := by
  unfold tensorNeeds at needs
  obtain ⟨partials, needs⟩ := List.forall_mem_cons.mp needs
  obtain ⟨eta, needs⟩ := List.forall_mem_cons.mp needs
  obtain ⟨rounds, needs⟩ := List.forall_mem_append.mp needs
  obtain ⟨final, -⟩ := List.forall_mem_cons.mp needs
  unfold tensorAt tensorFull
  rw [(partials.of_values le).1]
  refine Same.recv _ (Same.draw_of fun one => ?_)
  rw [firstIn_congr (eta.of_values (by omega)).1]
  refine ⟨rfl, ?_⟩
  obtain ⟨close, full⟩ := roundsFrom_agree rounds (j - base - 1) (by omega)
  refine plainPath_agree _ 1 _ (by rw [length_roundsFrom]; exact close) fun ge => ?_
  rw [length_roundsFrom] at ge
  rw [firstIn_congr (final.of_values (by omega)).1]
  exact ⟨full ge, Same.refl _ _⟩

variable {after after' : List Move} {afterNeeds : ℕ → List Need}

/-- The fused sum-check of two states with its needs. -/
theorem fusedAt_same (level : ℕ) (F : Fold) (n : ℕ)
    (ends : ∀ base, base ≤ j → (∀ x ∈ afterNeeds base, x.Holds j s t) →
      Same (j - base) after after') {base : ℕ} (le : base ≤ j)
    (needs : ∀ x ∈ fusedNeeds level n afterNeeds base, x.Holds j s t) :
    Same (j - base) (closingAt E pub s level F false n after)
      (closingAt E pub t level F false n after') := by
  unfold fusedNeeds at needs
  obtain ⟨batch, needs⟩ := List.forall_mem_cons.mp needs
  obtain ⟨stage2, needs⟩ := List.forall_mem_append.mp needs
  obtain ⟨rounds, needs⟩ := List.forall_mem_append.mp needs
  obtain ⟨value, later⟩ := List.forall_mem_cons.mp needs
  show Same (j - base) (Move.draw ((firstIn s ⟨level, "stage2_batch", none⟩).into : E) ::
      (plainPath (Ext.into (E := E)) 2 (stage2In pub s level F).fused
        (roundsFrom s ⟨level, "fused", none⟩ n) ++
          Move.recv ((firstIn s ⟨level, "witness_evaluation", none⟩).into : E) :: after))
    (Move.draw ((firstIn t ⟨level, "stage2_batch", none⟩).into : E) ::
      (plainPath (Ext.into (E := E)) 2 (stage2In pub t level F).fused
        (roundsFrom t ⟨level, "fused", none⟩ n) ++
          Move.recv ((firstIn t ⟨level, "witness_evaluation", none⟩).into : E) :: after'))
  refine Same.draw_of fun one => ?_
  have claim : stage2In pub t level F = stage2In pub s level F :=
    stage2In_frame pub level F
      (fun k inside => ((stage2 _ (List.mem_map_of_mem inside)).of_values (by omega)).1.symm)
      (fun k inside => ((stage2 _ (List.mem_map_of_mem inside)).of_values (by omega)).2.1.symm)
  rw [firstIn_congr (batch.of_values (by omega)).1, claim]
  refine ⟨rfl, ?_⟩
  obtain ⟨close, full⟩ := roundsFrom_agree rounds (j - base - 1) (by omega)
  refine plainPath_agree _ 2 _ (by rw [length_roundsFrom]; exact close) fun ge => ?_
  rw [length_roundsFrom] at ge ⊢
  rw [firstIn_congr (value.of_values (by omega)).1]
  refine ⟨full ge, Same.recv _ ?_⟩
  have tail := ends (base + 1 + n) (by omega) later
  rw [show j - base - 1 - n = j - (base + 1 + n) by omega]
  exact tail

/-- The path after the last stage of the range proof of two states with its needs. -/
theorem closingAt_same (level : ℕ) (F : Fold) (bin : Bool) (n : ℕ)
    (ends : ∀ base, base ≤ j → (∀ x ∈ afterNeeds base, x.Holds j s t) →
      Same (j - base) after after') {base : ℕ} (le : base ≤ j)
    (needs : ∀ x ∈ closingNeeds level bin n afterNeeds base, x.Holds j s t) :
    Same (j - base) (closingAt E pub s level F bin n after)
      (closingAt E pub t level F bin n after') := by
  cases bin with
  | false => exact fusedAt_same E pub level F n ends le needs
  | true =>
      have needs : ∀ x ∈ Need.values (base + 1) ⟨level, "binary_batch", none⟩ ::
          fusedNeeds level n afterNeeds (base + 1), x.Holds j s t := needs
      obtain ⟨binary, needs⟩ := List.forall_mem_cons.mp needs
      show Same (j - base) (Move.draw ((firstIn s ⟨level, "binary_batch", none⟩).into : E) ::
          closingAt E pub s level F false n after)
        (Move.draw ((firstIn t ⟨level, "binary_batch", none⟩).into : E) ::
          closingAt E pub t level F false n after')
      refine Same.draw_of fun one => ?_
      rw [firstIn_congr (binary.of_values (by omega)).1]
      refine ⟨rfl, ?_⟩
      have tail := fusedAt_same E pub level F n ends (base := base + 1) (by omega) needs
      rw [Nat.sub_sub]
      exact tail

/-- The last stage of the range proof of two states with its needs. -/
theorem lastAt_same (level : ℕ) (F : Fold) (sq bin : Bool) (n : ℕ)
    (ends : ∀ base, base ≤ j → (∀ x ∈ afterNeeds base, x.Holds j s t) →
      Same (j - base) after after') (base : ℕ) (le : base ≤ j)
    (needs : ∀ x ∈ lastNeeds level sq bin n afterNeeds base, x.Holds j s t) (claim : Ext) :
    Same (j - base) (lastAt E pub s level F sq bin n after claim)
      (lastAt E pub t level F sq bin n after' claim) := by
  cases sq with
  | true =>
      simp only [lastNeeds] at needs
      obtain ⟨energy, needs⟩ := List.forall_mem_cons.mp needs
      obtain ⟨merge, needs⟩ := List.forall_mem_cons.mp needs
      obtain ⟨rounds, needs⟩ := List.forall_mem_append.mp needs
      obtain ⟨virtual, needs⟩ := List.forall_mem_cons.mp needs
      obtain ⟨image, needs⟩ := List.forall_mem_cons.mp needs
      obtain ⟨eta, needs⟩ := List.forall_mem_cons.mp needs
      simp only [lastAt, normFull]
      rw [(energy.of_values le).2.2]
      refine Same.recv _ (Same.draw_of fun one => ?_)
      rw [firstIn_congr (merge.of_values (by omega)).1]
      refine ⟨rfl, ?_⟩
      obtain ⟨close, full⟩ := roundsFrom_agree rounds (j - base - 1) (by omega)
      refine plainPath_agree _ _ _ (by rw [length_roundsFrom]; exact close) fun ge => ?_
      rw [length_roundsFrom] at ge ⊢
      rw [firstIn_congr (virtual.of_values (by omega)).1, firstIn_congr (image.of_values (by omega)).1]
      refine ⟨full ge, Same.recv _ (Same.draw_of fun two => ?_)⟩
      rw [firstIn_congr (eta.of_values (by omega)).1]
      refine ⟨rfl, ?_⟩
      have tail := closingAt_same E pub level F bin n ends (base := base + n + 2) (by omega) needs
      rw [show j - base - 1 - n - 1 = j - (base + n + 2) by omega]
      exact tail
  | false =>
      simp only [lastNeeds] at needs
      obtain ⟨rounds, needs⟩ := List.forall_mem_append.mp needs
      obtain ⟨image, needs⟩ := List.forall_mem_cons.mp needs
      simp only [lastAt, leafPath]
      rw [List.append_assoc, List.append_assoc, List.singleton_append, List.singleton_append]
      obtain ⟨close, full⟩ := roundsFrom_agree rounds (j - base) (by omega)
      refine eqCodePath_agree _ _ (by rw [length_roundsFrom]; exact close) fun ge => ?_
      rw [length_roundsFrom] at ge ⊢
      rw [firstIn_congr (image.of_values (by omega)).1]
      refine ⟨full ge, Same.recv _ ?_⟩
      have tail := closingAt_same E pub level F bin n ends (base := base + n) (by omega) needs
      rw [Nat.sub_sub]
      exact tail

/-- The product stages of two states with their needs, and then two paths that are equal with
the needs `rest`. -/
theorem stages_same (level n : ℕ) {last last' : Ext → List Move} {rest : ℕ → List Need}
    (ends : ∀ base, base ≤ j → (∀ x ∈ rest base, x.Holds j s t) →
      ∀ claim, Same (j - base) (last claim) (last' claim)) :
    ∀ (stages : List (ℕ × ℕ)) (base nodes : ℕ) (claim : Ext), base ≤ j →
      (∀ x ∈ stageNeeds level n rest stages base, x.Holds j s t) →
      Same (j - base)
        (treeFull (Ext.into (E := E)) last (stages.map (productFrom s level n)) nodes claim)
        (treeFull (Ext.into (E := E)) last' (stages.map (productFrom t level n)) nodes claim)
  | [], base, _, claim, le, needs => ends base le needs claim
  | x :: more, base, nodes, claim, le, needs => by
      have needs : ∀ y ∈ roundNeeds base ⟨level, s!"range.{x.2}", none⟩ n ++
          (Need.values (base + n) ⟨level, s!"range.{x.2}.children", none⟩ ::
            Need.values (base + n + 1) ⟨level, s!"range.{x.2}.batch", none⟩ ::
              stageNeeds level n rest more (base + n + 1)), y.Holds j s t := needs
      obtain ⟨rounds, needs⟩ := List.forall_mem_append.mp needs
      obtain ⟨children, needs⟩ := List.forall_mem_cons.mp needs
      obtain ⟨batch, needs⟩ := List.forall_mem_cons.mp needs
      show Same (j - base)
        (eqCodePath (Ext.into (E := E)) x.1 (roundsFrom s ⟨level, s!"range.{x.2}", none⟩ n) ++
          Move.recv (listFn (nodes * x.1)
            ((recordedIn s ⟨level, s!"range.{x.2}.children", none⟩).toList.map
              (Ext.into (E := E)))) ::
            Move.draw ((firstIn s ⟨level, s!"range.{x.2}.batch", none⟩).into : E) ::
              treeFull (Ext.into (E := E)) last (more.map (productFrom s level n)) (nodes * x.1)
                (batched (nodes * x.1) (recordedIn s ⟨level, s!"range.{x.2}.children", none⟩).toList
                  (firstIn s ⟨level, s!"range.{x.2}.batch", none⟩)))
        (eqCodePath (Ext.into (E := E)) x.1 (roundsFrom t ⟨level, s!"range.{x.2}", none⟩ n) ++
          Move.recv (listFn (nodes * x.1)
            ((recordedIn t ⟨level, s!"range.{x.2}.children", none⟩).toList.map
              (Ext.into (E := E)))) ::
            Move.draw ((firstIn t ⟨level, s!"range.{x.2}.batch", none⟩).into : E) ::
              treeFull (Ext.into (E := E)) last' (more.map (productFrom t level n)) (nodes * x.1)
                (batched (nodes * x.1) (recordedIn t ⟨level, s!"range.{x.2}.children", none⟩).toList
                  (firstIn t ⟨level, s!"range.{x.2}.batch", none⟩)))
      obtain ⟨close, full⟩ := roundsFrom_agree rounds (j - base) (by omega)
      refine eqCodePath_agree _ _ (by rw [length_roundsFrom]; exact close) fun ge => ?_
      rw [length_roundsFrom] at ge ⊢
      rw [(children.of_values (by omega)).1]
      refine ⟨full ge, Same.recv _ (Same.draw_of fun one => ?_)⟩
      rw [firstIn_congr (batch.of_values (by omega)).1]
      refine ⟨rfl, ?_⟩
      have tail := stages_same level n ends more (base + n + 1) (nodes * x.1)
        (batched (nodes * x.1) (recordedIn t ⟨level, s!"range.{x.2}.children", none⟩).toList
          (firstIn t ⟨level, s!"range.{x.2}.batch", none⟩)) (by omega) needs
      rw [show j - base - n - 1 = j - (base + n + 1) by omega]
      exact tail

/-- Two states with the needs of `pathAt` for the draw `j` have the same path before its draw
`j + 1`. -/
theorem pathAt_same (level : ℕ) (F : Fold) (sq bin : Bool) (n batch : ℕ) (stages : List (ℕ × ℕ))
    (ends : ∀ base, base ≤ j → (∀ x ∈ afterNeeds base, x.Holds j s t) →
      Same (j - base) after after')
    (needs : ∀ x ∈ pathNeeds level sq bin n stages afterNeeds, x.Holds j s t) :
    Same j (pathAt E pub s level F sq bin n batch stages after)
      (pathAt E pub t level F sq bin n batch stages after') := by
  unfold pathNeeds at needs
  obtain ⟨alpha, needs⟩ := List.forall_mem_cons.mp needs
  obtain ⟨tau0, needs⟩ := List.forall_mem_cons.mp needs
  obtain ⟨tau1, needs⟩ := List.forall_mem_cons.mp needs
  unfold pathAt
  refine Same.draw_of fun one => ?_
  rw [firstIn_congr (alpha.of_values one).1]
  refine ⟨rfl, Same.draw_of fun two => ?_⟩
  rw [(tau0.of_values (by omega)).1]
  refine ⟨rfl, Same.draw_of fun three => ?_⟩
  rw [(tau1.of_values (by omega)).1]
  refine ⟨rfl, ?_⟩
  have tail := stages_same E level n
    (fun base le rest claim => lastAt_same E pub level F sq bin n ends base le rest claim)
    stages 3 1 Ext.zero (by omega) needs
  rw [show j - 1 - 1 - 1 = j - 3 by omega]
  exact tail

/-- Two states with the needs `needs20 ℓ` for the draw `j` have the same `pathAt20 … ℓ` before
its draw `j + 1`. -/
theorem pathAt20_same (ℓ : Fin 6) (needs : ∀ x ∈ needs20 ℓ, x.Holds j s t) :
    Same j (pathAt20 E pub s ℓ) (pathAt20 E pub t ℓ) := by
  refine pathAt_same E pub ℓ (fold20 ℓ) _ _ _ _ _ (fun base le rest => ?_) needs
  unfold afterAt
  unfold afterNeeds20 at rest
  by_cases zero : ℓ.val = 0
  · rw [ite_eq_left zero, ite_eq_left zero]
    exact Same.refl _ _
  · rw [ite_eq_right zero, ite_eq_right zero]
    rw [ite_eq_right zero] at rest
    exact tensorAt_same E le rest

end Needs

section Checks

variable (pub : Public) (proof : ByteArray) (messages : Array Message)

/-- A key that no operation between `c ≤ c'` writes keeps its recorded values, its field message
and its integer. -/
theorem same_between (ops : Array Op) {c c' : ℕ} (le : c ≤ c')
    (ok : (runFrom pub proof messages (ops.zipIdx.toList.take c') (channelStart pub)).1 = .ok ())
    {key : Key} (free : ∀ y ∈ (ops.zipIdx.toList.take c').drop c, key ∉ y.1.writes) :
    recordedIn (stateBefore pub proof messages ops c') key =
        recordedIn (stateBefore pub proof messages ops c) key ∧
      (stateBefore pub proof messages ops c').fields.get? key =
        (stateBefore pub proof messages ops c).fields.get? key ∧
      (stateBefore pub proof messages ops c').integers.get? key =
        (stateBefore pub proof messages ops c).integers.get? key := by
  rw [(stateBefore_le pub proof messages ops le ok).2.2]
  have same := (runFrom_grows pub proof messages ((ops.zipIdx.toList.take c').drop c)
    (stateBefore pub proof messages ops c)).same key fun inside => by
      obtain ⟨y, member, found⟩ := List.mem_flatMap.mp inside
      exact free y member found
  exact ⟨same.1, same.2.1, same.2.2.1⟩

/-- The recorded values of a key before `c` are a prefix of those before `c' ≥ c`. -/
theorem prefix_between (ops : Array Op) {c c' : ℕ} (le : c ≤ c')
    (ok : (runFrom pub proof messages (ops.zipIdx.toList.take c') (channelStart pub)).1 = .ok ())
    (key : Key) :
    (recordedIn (stateBefore pub proof messages ops c) key).toList <+:
      (recordedIn (stateBefore pub proof messages ops c') key).toList := by
  rw [(stateBefore_le pub proof messages ops le ok).2.2]
  exact (runFrom_grows pub proof messages _ _).prefix_ key

variable {pub proof messages}

/-- A need with the check holds between the state before the work of each stage `k₀ + j` and
the state before `L`, on a run of a script of `layout20` without a stop before `L`. -/
theorem Need.holds {ops : Array Op} (erased : ops.toList.map Op.erase = skeleton (layout20 0 0))
    {k₀ L : ℕ}
    (ok : (runFrom pub proof messages (ops.zipIdx.toList.take L) (channelStart pub)).1 = .ok ())
    {x : Need} (checked : x.check k₀ L = true) {j : ℕ} (inside : k₀ + j < 473)
    (le : workIndices20.getD (k₀ + j) 0 ≤ L) :
    x.Holds j (stateBefore pub proof messages ops (workIndices20.getD (k₀ + j) 0))
      (stateBefore pub proof messages ops L) := by
  cases x with
  | values from_ key =>
      show from_ ≤ j → _ ∧ _ ∧ _
      intro before
      have mono := workIndices20_mono (Nat.add_le_add_left before k₀) inside
      have held : held20 (workIndices20.getD (k₀ + from_) 0) L [key] = true := checked
      unfold held20 at held
      simp only [Bool.and_eq_true, decide_eq_true_eq] at held
      have free := joined_frees erased _ _ _ _ held.2
      have same := same_between pub proof messages ops le ok fun y member =>
        frees_writes free List.mem_cons_self y (mem_drop_of_le mono member)
      exact ⟨same.1.symm, same.2.1.symm, same.2.2.symm⟩
  | value from_ key index =>
      show from_ ≤ j → _
      intro before
      have mono := workIndices20_mono (Nat.add_le_add_left before k₀) inside
      have counted : (drawnCount key (sk20.take (workIndices20.getD (k₀ + from_) 0))).any
          (fun count => decide (index + 1 ≤ count)) = true := checked
      cases found : drawnCount key (sk20.take (workIndices20.getD (k₀ + from_) 0)) with
      | none =>
          rw [found] at counted
          cases counted
      | some count =>
          rw [found] at counted
          have enough : index + 1 ≤ count := of_decide_eq_true counted
          have okFrom := (stateBefore_le pub proof messages ops (Nat.le_trans mono le) ok).1
          have size := size_drawnCount pub proof messages key _ (channelStart pub)
            (joined_drawnCount erased _ key found) okFrom
          have sized : (recordedIn (stateBefore pub proof messages ops
              (workIndices20.getD (k₀ + from_) 0)) key).size = count := by
            simpa [stateBefore, channelStart_fresh] using size
          have okStage := (stateBefore_le pub proof messages ops le ok).1
          obtain ⟨front, grown⟩ := prefix_between pub proof messages ops mono okStage key
          obtain ⟨tail, later⟩ := prefix_between pub proof messages ops le ok key
          have long : index < (recordedIn (stateBefore pub proof messages ops
              (workIndices20.getD (k₀ + j) 0)) key).toList.length := by
            rw [← grown, List.length_append, Array.length_toList, sized]
            omega
          have front_ : ∀ a b : List Ext, index < a.length →
              a.getD index Ext.zero = (a ++ b).getD index Ext.zero := fun a b lt => by
            rw [List.getD_eq_getElem?_getD, List.getD_eq_getElem?_getD,
              List.getElem?_append_left lt]
          rw [← later]
          exact front_ _ _ long

end Checks

section Stage

/-- The first stage after the fold stage of the level `ℓ`: the stage of its draw `0`. -/
def first20 (ℓ : Fin 6) : ℕ := [2, 52, 121, 204, 285, 379].getD ℓ 0

/-- The last stage of the level `ℓ`. -/
def bound20 (ℓ : Fin 6) : ℕ := [51, 120, 203, 284, 378, 472].getD ℓ 0

/-- The index of the last state that the path of the level `ℓ` reads: the `fused.end` of the next
level, or the terminal. -/
def last20 (ℓ : Fin 6) : ℕ := [316, 586, 848, 1153, 1456, 1521].getD ℓ 0

theorem stage20_some {k : ℕ} {ℓ : Fin 6} {j : ℕ} (at_ : stage20 k = some (ℓ, j)) :
    k = first20 ℓ + j ∧ k ≤ bound20 ℓ := by
  unfold stage20 at at_
  split_ifs at at_ <;> cases at_
  · exact ⟨show k = 2 + (k - 2) by omega, by assumption⟩
  · exact ⟨show k = 52 + (k - 52) by omega, by assumption⟩
  · exact ⟨show k = 121 + (k - 121) by omega, by assumption⟩
  · exact ⟨show k = 204 + (k - 204) by omega, by assumption⟩
  · exact ⟨show k = 285 + (k - 285) by omega, by assumption⟩
  · exact ⟨show k = 379 + (k - 379) by omega, by assumption⟩

theorem needs20_checked (ℓ : Fin 6) :
    (needs20 ℓ).all (Need.check (first20 ℓ) (last20 ℓ)) = true := by
  match ℓ with
  | 0 => exact needs20_checked₀
  | 1 => exact needs20_checked₁
  | 2 => exact needs20_checked₂
  | 3 => exact needs20_checked₃
  | 4 => exact needs20_checked₄
  | 5 => exact needs20_checked₅

theorem bound20_last : ∀ ℓ : Fin 6, bound20 ℓ < 473 ∧
    workIndices20.getD (bound20 ℓ) 0 ≤ last20 ℓ := by
  decide +kernel

variable (E : Type) [Field E] [Algebra Zq E] [Degree2 E]
  {pub : Public} {proof : ByteArray} {messages : Array Message}

/-- On a run of a script of `layout20` without a stop, the state before the work of the stage `k`
of the level `ℓ` and the state before the operation `last20 ℓ` have the same `pathAt20 … ℓ` before
the draw after the stage `k`. -/
theorem pathAt20_stage (erased : pub.ops.toList.map Op.erase = skeleton (layout20 0 0))
    (ran : ∀ c, (runFrom pub proof messages (pub.ops.zipIdx.toList.take c) (channelStart pub)).1 =
      .ok ()) {k : ℕ} {ℓ : Fin 6} {j : ℕ} (at_ : stage20 k = some (ℓ, j)) :
    Same j (pathAt20 E pub (stateBefore pub proof messages pub.ops (stageIndex20 pub k)) ℓ)
      (pathAt20 E pub (stateBefore pub proof messages pub.ops (last20 ℓ)) ℓ) := by
  obtain ⟨rfl, bounded⟩ := stage20_some at_
  obtain ⟨below, last⟩ := bound20_last ℓ
  have inside : first20 ℓ + j < 473 := Nat.lt_of_le_of_lt bounded below
  rw [stageIndex20_lt pub inside]
  exact pathAt20_same E pub ℓ fun x member =>
    Need.holds erased (ran _) (List.all_eq_true.mp (needs20_checked ℓ) x member) inside
      (Nat.le_trans (workIndices20_mono bounded below) last)

/-- `cellsAt20 … k` is `cells20 … k` on a run of a script of `layout20` without a stop, when
`pathAt20` on the state before `last20 ℓ` is the path `path20 … ℓ` of the code for each level. -/
theorem cellsAt20_eq_of_final (erased : pub.ops.toList.map Op.erase = skeleton (layout20 0 0))
    (ran : ∀ c, (runFrom pub proof messages (pub.ops.zipIdx.toList.take c) (channelStart pub)).1 =
      .ok ())
    (final : ∀ ℓ : Fin 6, pathAt20 E pub (stateBefore pub proof messages pub.ops (last20 ℓ)) ℓ =
      path20 pub proof messages E ℓ) (k : ℕ) :
    cellsAt20 E pub proof messages k = cells20 E pub proof messages k := by
  unfold cellsAt20 cells20 recvs20
  cases at_ : stage20 k with
  | none => rfl
  | some x =>
      obtain ⟨ℓ, j⟩ := x
      show Extract.cellsFrom (Extract.segments (pathAt20 E pub _ ℓ) j) 0 =
        Extract.cellsOf (path20 pub proof messages E ℓ) j
      unfold Extract.cellsOf
      rw [(pathAt20_stage E erased ran at_).segments (Nat.le_refl j), final]

end Stage

section Final

/-- The keys other than round keys that the fold of `pathAt` reads. -/
def foldKeys (level : ℕ) (F : Fold) : List Key :=
  [⟨level, "alpha", none⟩, ⟨level, "tau0", none⟩, ⟨level, "tau1", none⟩,
    ⟨level, "range.image", none⟩, ⟨level, "binary_batch", none⟩, ⟨level, "stage2_batch", none⟩,
    ⟨level, "witness_evaluation", none⟩, ⟨level, "norm.response_l2_sq", none⟩,
    ⟨level, "norm.merge", none⟩, ⟨level, "norm.virtual_evaluations", none⟩,
    ⟨level, "norm.virtual_batch", none⟩] ++
    (stage2Keys level ++ productKeys level (productStages F.logBasis).length)

/-- The sum-checks whose rounds the fold of `pathAt` reads. -/
def foldRounds (level : ℕ) (F : Fold) : List Key :=
  ⟨level, "fused", none⟩ ::
    (if F.euclidean then ⟨level, "norm.sumcheck", none⟩ else ⟨level, "range.leaf", none⟩) ::
      (List.range (productStages F.logBasis).length).map fun k => ⟨level, s!"range.{k}", none⟩

/-- The index of the end of the sum-check `key` with `n` rounds in the skeleton of `layout20`. -/
def endIndex (key : Key) (n : ℕ) : ℕ := sk20.findIdx fun x => isEndOf key n x.1

/-- The check on the skeleton of `layout20` that the rounds of the sum-check `key` before the
operation `c` are `roundsFrom` on the state before `L`: its end is before `c`, no earlier
operation ends it, its point has `n` challenges at the end, and no operation from the end to `L`
writes its point or a round message. -/
def roundsCheck (key : Key) (n c L : ℕ) : Bool :=
  (sk20[endIndex key n]?).any (fun x => isEndOf key n x.1) && decide (endIndex key n < c) &&
    decide (c ≤ L) && frees (sk20.take (endIndex key n)) [] [key] &&
    held20 (endIndex key n) L (⟨key.level, key.site ++ ".point", none⟩ ::
      (List.range n).map fun r => ⟨key.level, key.site ++ ".round", some r⟩) &&
    (drawnCount ⟨key.level, key.site ++ ".point", none⟩ (sk20.take (endIndex key n)) == some n)

theorem zip_range_getD {α β : Type} (f : ℕ → α) (l : List β) (d : β) {n : ℕ}
    (long : l.length = n) :
    ((List.range n).map f).zip l = (List.range n).map fun r => (f r, l.getD r d) := by
  subst long
  refine List.ext_getElem (by simp) fun i h1 h2 => ?_
  have lt : i < l.length := by simpa using h1
  simp [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem lt]

/-- The message `v_W` of the end check `finishIn` is the recorded `witness_evaluation`. -/
theorem finishIn_value (setup : Array Fq) (s : Replayer) (level : ℕ) (F : Fold) (p : List Ext) :
    (finishIn setup s level F p).value = firstIn s ⟨level, "witness_evaluation", none⟩ := by
  unfold finishIn
  cases F.fused <;> rfl

/-- A fold on the coefficient route that passes has a range proof that passes. -/
theorem rangeRun_of_fold {pub : Public} {s : Replayer} {level : ℕ} {F : Fold}
    (plain : F.euclidean = false) (ran : (foldRunIn pub s level F).isSome = true) :
    (rangeRun (2 ^ F.logBasis) (wordsIn s level F).stages
      (roundsIn s ⟨level, "range.leaf", none⟩) (wordsIn s level F).image
      ((wordsIn s level F).point F.low)).isSome = true := by
  unfold foldRunIn at ran
  cases found : pub.setupFor F.setupCount with
  | none =>
      rw [found] at ran
      cases ran
  | some setup =>
      rw [found] at ran
      simp only [Option.bind_some, plain, Bool.false_eq_true, ↓reduceIte] at ran
      rw [foldRun_if] at ran
      by_contra fails
      rw [ite_eq_right fails] at ran
      cases ran

/-- A fold on the Euclidean route that passes has product stages and a norm stage that pass. -/
theorem treeRun_of_fold {pub : Public} {s : Replayer} {level : ℕ} {F : Fold}
    (euclidean : F.euclidean = true) (ran : (foldRunIn pub s level F).isSome = true) :
    (treeRun (normRun (leafDegree F.logBasis) F.limit (normsIn s level).energy
        (normsIn s level).merge (normsIn s level).rounds (normsIn s level).virtual
        (wordsIn s level F).image) (wordsIn s level F).stages 1 (fun _ => Ext.one)
      ((wordsIn s level F).point F.low) Ext.zero).isSome = true := by
  unfold foldRunIn at ran
  cases found : pub.setupFor F.setupCount with
  | none =>
      rw [found] at ran
      cases ran
  | some setup =>
      rw [found] at ran
      simp only [Option.bind_some, euclidean, ↓reduceIte] at ran
      rw [foldRunSq_if] at ran
      by_contra fails
      rw [ite_eq_right fails] at ran
      cases ran

variable (E : Type) [Field E] [Algebra Zq E] [Degree2 E] (pub : Public)

/-- The path `foldPathIn` of a fold that passes on the state `s`, with the path `after` after it,
is `pathAt` on a state `t` with the same values of `foldKeys` and whose `roundsFrom` are the
rounds of `s`. -/
theorem foldPathIn_final {s t : Replayer} (level : ℕ) (F : Fold) (setup : Array Fq)
    (vars batch : ℕ) (after : List Move) (ran : (foldRunIn pub s level F).isSome = true)
    (keys : ∀ k ∈ foldKeys level F, recordedIn t k = recordedIn s k ∧
      t.fields.get? k = s.fields.get? k ∧ t.integers.get? k = s.integers.get? k)
    (rounds : ∀ k ∈ foldRounds level F, roundsIn s k = roundsFrom t k vars) :
    foldPathIn E pub s level F setup vars batch ++ after =
      pathAt E pub t level F F.euclidean F.compressed vars batch
        ((productStages F.logBasis).map (·.1)).zipIdx after := by
  have first : ∀ k ∈ foldKeys level F, firstIn t k = firstIn s k := fun k inside =>
    firstIn_congr (keys k inside).1
  have claim : stage2In pub t level F = stage2In pub s level F :=
    stage2In_frame pub level F (fun k inside => (keys k (by simp [foldKeys, inside])).1)
      (fun k inside => (keys k (by simp [foldKeys, inside])).2.1)
  have stages : productsIn s level ((productStages F.logBasis).map (·.1)) =
      ((productStages F.logBasis).map (·.1)).zipIdx.map (productFrom t level vars) := by
    unfold productsIn
    refine List.map_congr_left fun x member => ?_
    obtain ⟨arity, k⟩ := x
    have below : k < (productStages F.logBasis).length := by
      have inside := (List.mem_zipIdx' member).1
      rwa [List.length_map] at inside
    have products : ∀ key ∈ [(⟨level, s!"range.{k}.children", none⟩ : Key),
        ⟨level, s!"range.{k}.batch", none⟩], key ∈ foldKeys level F := fun key inside =>
      List.mem_append_right _ (List.mem_append_right _
        (List.mem_flatMap.mpr ⟨k, List.mem_range.mpr below, inside⟩))
    show ({ arity := arity, rounds := roundsIn s ⟨level, s!"range.{k}", none⟩
            children := (recordedIn s ⟨level, s!"range.{k}.children", none⟩).toList
            gamma := firstIn s ⟨level, s!"range.{k}.batch", none⟩ } : Product) =
      { arity := arity, rounds := roundsFrom t ⟨level, s!"range.{k}", none⟩ vars
        children := (recordedIn t ⟨level, s!"range.{k}.children", none⟩).toList
        gamma := firstIn t ⟨level, s!"range.{k}.batch", none⟩ }
    rw [rounds _ (List.mem_cons_of_mem _ (List.mem_cons_of_mem _
        (List.mem_map.mpr ⟨k, List.mem_range.mpr below, rfl⟩))),
      (keys _ (products _ List.mem_cons_self)).1,
      first _ (products _ (List.mem_cons_of_mem _ List.mem_cons_self))]
  have fused := rounds ⟨level, "fused", none⟩ List.mem_cons_self
  have alpha := first ⟨level, "alpha", none⟩ (by simp [foldKeys])
  have tau0 := (keys ⟨level, "tau0", none⟩ (by simp [foldKeys])).1
  have tau1 := (keys ⟨level, "tau1", none⟩ (by simp [foldKeys])).1
  have image := first ⟨level, "range.image", none⟩ (by simp [foldKeys])
  have binary := first ⟨level, "binary_batch", none⟩ (by simp [foldKeys])
  have batched_ := first ⟨level, "stage2_batch", none⟩ (by simp [foldKeys])
  have value := first ⟨level, "witness_evaluation", none⟩ (by simp [foldKeys])
  unfold pathAt
  cases sq : F.euclidean with
  | false =>
      have leaf := rounds ⟨level, "range.leaf", none⟩ (by simp [foldRounds, sq])
      have last : lastAt E pub t level F false F.compressed vars after = fun _ =>
          leafPath (Ext.into (E := E)) (rangeLeaf (2 ^ F.logBasis))
            (roundsIn s ⟨level, "range.leaf", none⟩) (wordsIn s level F).image ++
            closingFull E F.compressed (wordsIn s level F).binary (wordsIn s level F).batch
              (stage2In pub s level F).fused (wordsIn s level F).rounds
              (finishIn setup s level F ((wordsIn s level F).rounds.map (·.2))).value after := by
        funext tree
        simp only [lastAt, closingAt, wordsIn, finishIn_value]
        rw [← leaf, ← fused, claim, image, binary, batched_, value]
      simp only [foldPathIn, sq, Bool.false_eq_true, ↓reduceIte]
      rw [foldPath_full E _ _ _ _ _ _ _ _ _ after (rangeRun_of_fold sq ran), last, ← stages,
        alpha, tau0, tau1]
      rfl
  | true =>
      have norm := rounds ⟨level, "norm.sumcheck", none⟩ (by simp [foldRounds, sq])
      have energy := (keys ⟨level, "norm.response_l2_sq", none⟩ (by simp [foldKeys])).2.2
      have merge := first ⟨level, "norm.merge", none⟩ (by simp [foldKeys])
      have virtual := first ⟨level, "norm.virtual_evaluations", none⟩ (by simp [foldKeys])
      have eta := first ⟨level, "norm.virtual_batch", none⟩ (by simp [foldKeys])
      have last : lastAt E pub t level F true F.compressed vars after =
          normFull (Ext.into (E := E)) (leafDegree F.logBasis) (normsIn s level).energy
            (normsIn s level).merge (normsIn s level).rounds (normsIn s level).virtual
            (wordsIn s level F).image
            (Move.draw ((normsIn s level).eta.into : E) ::
              closingFull E F.compressed (wordsIn s level F).binary (wordsIn s level F).batch
                (stage2In pub s level F).fused (wordsIn s level F).rounds
                (finishIn setup s level F ((wordsIn s level F).rounds.map (·.2))).value
                after) := by
        funext tree
        simp only [lastAt, closingAt, wordsIn, normsIn, finishIn_value]
        rw [← norm, ← fused, claim, energy, merge, virtual, image, eta, binary, batched_, value]
      simp only [foldPathIn, sq, ↓reduceIte]
      rw [foldPathSq_full E _ _ _ _ _ _ _ _ _ _ after (treeRun_of_fold sq ran), last, ← stages,
        alpha, tau0, tau1]
      rfl

variable {pub} {proof : ByteArray} {messages : Array Message}

/-- A sum-check with the check `roundsCheck`: its rounds before `c` are `roundsFrom` on the state
before `L`, on a run of a script of `layout20` without a stop before `L`. -/
theorem roundsIn_of_check {ops : Array Op}
    (erased : ops.toList.map Op.erase = skeleton (layout20 0 0)) {key : Key} {n c L : ℕ}
    (checked : roundsCheck key n c L = true)
    (ok : (runFrom pub proof messages (ops.zipIdx.toList.take L) (channelStart pub)).1 = .ok ()) :
    roundsIn (stateBefore pub proof messages ops c) key =
      roundsFrom (stateBefore pub proof messages ops L) key n := by
  unfold roundsCheck at checked
  simp only [Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at checked
  obtain ⟨⟨⟨⟨⟨found, lt⟩, le⟩, noEnd⟩, held⟩, count⟩ := checked
  obtain ⟨op, op', isEnd, at_, same⟩ := erased_at erased found
  obtain ⟨before, rfl⟩ := isEndOf_sound isEnd
  rw [Op.erase_ended same] at at_
  have okc := (stateBefore_le pub proof messages ops le ok).1
  have okE := (stateBefore_le pub proof messages ops (Nat.le_trans (Nat.le_of_lt lt) le) ok).1
  rw [rounds_of_end pub proof messages ops lt at_ okc
    (frees_ends (joined_frees_take erased _ _ _ noEnd) List.mem_cons_self)]
  have size := size_drawnCount pub proof messages _ _ (channelStart pub)
    (joined_drawnCount erased _ _ count) okE
  have sized : (recordedIn (stateBefore pub proof messages ops (endIndex key n))
      ⟨key.level, key.site ++ ".point", none⟩).size = n := by
    simpa [stateBefore, channelStart_fresh] using size
  unfold held20 at held
  simp only [Bool.and_eq_true, decide_eq_true_eq] at held
  have free := joined_frees erased _ _ _ _ held.2
  have kept : ∀ k ∈ (⟨key.level, key.site ++ ".point", none⟩ : Key) ::
      (List.range n).map (fun r => (⟨key.level, key.site ++ ".round", some r⟩ : Key)),
      recordedIn (stateBefore pub proof messages ops L) k =
        recordedIn (stateBefore pub proof messages ops (endIndex key n)) k := fun k inside =>
    (same_between pub proof messages ops held.1 ok (frees_writes free inside)).1
  unfold endRoundsIn roundsFrom
  rw [kept _ List.mem_cons_self,
    zip_range_getD _ _ Ext.zero (by rw [Array.length_toList]; exact sized)]
  refine List.map_congr_left fun r member => ?_
  rw [kept _ (List.mem_cons_of_mem _ (List.mem_map_of_mem member))]

/-- `tensorRun` of the tensor reduction of the claim of the level `level` on the recorded values
of `s`. -/
def tensorOn (s : Replayer) (level ν : ℕ) : Option Ended :=
  tensorRun ν (recordedIn s ⟨level, "fused.point", none⟩).toList
    (firstIn s ⟨level, "witness_evaluation", none⟩)
    ((recordedIn s ⟨level + 1, "tensor.partials", none⟩).getD 0 Ext.zero)
    ((recordedIn s ⟨level + 1, "tensor.partials", none⟩).getD 1 Ext.zero)
    (firstIn s ⟨level + 1, "tensor.eta", none⟩) (roundsIn s ⟨level + 1, "tensor.sumcheck", none⟩)
    (firstIn s ⟨level + 1, "tensor.final_claims", none⟩)

/-- The tensor reduction of the claim of the level `level` passes on the state before `c'`, when
the step of its check has no stop. -/
theorem tensor_passes {level ν c t c' : ℕ} (frame : TensorFrame pub.ops level ν c t c')
    (stepped : StepOk pub proof messages t) :
    (tensorOn (stateBefore pub proof messages pub.ops c') level ν).isSome = true := by
  have accepts : (StageOut.tensor (tensorOn (stateBefore pub proof messages pub.ops t) level
      ν)).accepts = true :=
    step_stage_ok pub proof messages t _ _ _ rfl (stepped _ frame.at_)
  obtain ⟨frames, rounds⟩ := stateBefore_frees pub proof messages frame.after frame.kept
  have point := (frames ⟨level, "fused.point", none⟩ (by simp [tensorKeys])).1
  have value := firstIn_congr (frames ⟨level, "witness_evaluation", none⟩ (by simp [tensorKeys])).1
  have partials := (frames ⟨level + 1, "tensor.partials", none⟩ (by simp [tensorKeys])).1
  have eta := firstIn_congr (frames ⟨level + 1, "tensor.eta", none⟩ (by simp [tensorKeys])).1
  have final :=
    firstIn_congr (frames ⟨level + 1, "tensor.final_claims", none⟩ (by simp [tensorKeys])).1
  have same := rounds ⟨level + 1, "tensor.sumcheck", none⟩ List.mem_cons_self
  unfold tensorOn
  rw [point, value, partials, eta, same, final]
  exact accepts

end Final

section Accepted

/-- The index of the `fused.end` of the level `ℓ` in the skeleton of `layout20`. -/
def fusedAt20 (ℓ : Fin 6) : ℕ := [165, 316, 586, 848, 1153, 1456].getD ℓ 0

/-- The checks on the skeleton of `layout20` for the last state of the level `ℓ`: no operation
from its `fused.end` to `last20 ℓ` writes a key of `foldKeys`, each sum-check of the fold has
`roundsCheck` at its `fused.end`, and for `ℓ ≥ 1` the sum-check of the tensor reduction has
`roundsCheck` at `last20 ℓ`. -/
def finalCheck (ℓ : Fin 6) : Bool :=
  held20 (fusedAt20 ℓ) (last20 ℓ) (foldKeys ℓ (fold20 ℓ)) &&
    (foldRounds ℓ (fold20 ℓ)).all (fun k => roundsCheck k (fold20 ℓ).vars (fusedAt20 ℓ) (last20 ℓ)) &&
    (decide (ℓ.val = 0) ||
      roundsCheck ⟨ℓ + 1, "tensor.sumcheck", none⟩ (tail20 ℓ) (last20 ℓ) (last20 ℓ))

theorem final20_checked₀ : finalCheck 0 = true := by decide +kernel

theorem final20_checked₁ : finalCheck 1 = true := by decide +kernel

theorem final20_checked₂ : finalCheck 2 = true := by decide +kernel

theorem final20_checked₃ : finalCheck 3 = true := by decide +kernel

theorem final20_checked₄ : finalCheck 4 = true := by decide +kernel

theorem final20_checked₅ : finalCheck 5 = true := by decide +kernel

theorem final20_checked (ℓ : Fin 6) : finalCheck ℓ = true := by
  match ℓ with
  | 0 => exact final20_checked₀
  | 1 => exact final20_checked₁
  | 2 => exact final20_checked₂
  | 3 => exact final20_checked₃
  | 4 => exact final20_checked₄
  | 5 => exact final20_checked₅

variable (E : Type) [Field E] [Algebra Zq E] [Degree2 E]
  {pub : Public} {proof : ByteArray} {messages : Array Message}

/-- The indices of `path20` on a script of `layout20`. -/
theorem index20 {zMax rice : ℕ} (layout : pub.layout = layout20 zMax rice)
    (erased : pub.ops.toList.map Op.erase = skeleton (layout20 0 0)) (ℓ : Fin 6) :
    fusedIndex pub.layout ℓ = fusedAt20 ℓ ∧ receiver20 pub ℓ = last20 ℓ := by
  obtain ⟨f0, f1, f2, f3, f4, f5⟩ := fusedIndex20_values
  have size : pub.ops.size - 1 = 1521 := by rw [size_skeleton erased, skeleton20_length]
  unfold receiver20
  rw [layout]
  match ℓ with
  | 0 => exact ⟨f0, f1⟩
  | 1 => exact ⟨f1, f2⟩
  | 2 => exact ⟨f2, f3⟩
  | 3 => exact ⟨f3, f4⟩
  | 4 => exact ⟨f4, f5⟩
  | 5 => exact ⟨f5, size⟩

/-- The tensor reduction of the claim of each level `ℓ ≥ 1` of an accepted proof of `layout20`
passes on the state before `receiver20 pub ℓ`. -/
theorem tensor20_passes {zMax rice : ℕ} (layout : pub.layout = layout20 zMax rice)
    (accepted : runProof pub proof = .verdict .accept) {full : Array Fq}
    (terminalSetup : pub.setupFor pub.layout.terminal.setupCount = some full) {rest : ℕ}
    (decoded : decode pub.ops proof = .ok (messages, rest)) (ℓ : Fin 6) (positive : ℓ.val ≠ 0) :
    (tensorOn (stateBefore pub proof messages pub.ops (receiver20 pub ℓ)) ℓ (tail20 ℓ)).isSome =
      true := by
  obtain ⟨scripted, ends, -⟩ := verifyProof_script20 layout accepted
  have ok := stepOk_of_accept scripted ends terminalSetup accepted decoded
  obtain ⟨-, -, f1, f2, f3, f4, f5, -⟩ := verifyProof_frames20 layout accepted
  match ℓ with
  | 0 => exact absurd rfl positive
  | 1 => exact tensor_passes f1 (ok _)
  | 2 => exact tensor_passes f2 (ok _)
  | 3 => exact tensor_passes f3 (ok _)
  | 4 => exact tensor_passes f4 (ok _)
  | 5 => exact tensor_passes f5 (ok _)

/-- For an accepted proof of `layout20`, `pathAt20` on the state before `last20 ℓ` is the path
`path20 … ℓ` of the code. -/
theorem pathAt20_final {zMax rice : ℕ} (layout : pub.layout = layout20 zMax rice)
    (accepted : runProof pub proof = .verdict .accept) {full : Array Fq}
    (terminalSetup : pub.setupFor pub.layout.terminal.setupCount = some full) {rest : ℕ}
    (decoded : decode pub.ops proof = .ok (messages, rest)) (ℓ : Fin 6) :
    pathAt20 E pub (stateBefore pub proof messages pub.ops (last20 ℓ)) ℓ =
      path20 pub proof messages E ℓ := by
  obtain ⟨scripted, ends, levels⟩ := verifyProof_script20 layout accepted
  have none_ := (replay_of_decoded pub proof scripted ends terminalSetup accepted decoded).2
  have erased : pub.ops.toList.map Op.erase = skeleton (layout20 0 0) := by
    have scripted' := scripted
    rw [layout] at scripted'
    rw [script_erase scripted', skeleton20]
  have ran : ∀ c, (runFrom pub proof messages (pub.ops.zipIdx.toList.take c)
      (channelStart pub)).1 = .ok () := fun c =>
    runFrom_take_ok pub proof messages _ c _ ((replay_none_iff pub pub.ops proof messages).mp none_)
  obtain ⟨fusedAt, received⟩ := index20 layout erased ℓ
  obtain ⟨r, a, e, shape⟩ := (levels ℓ).shape
  have folded := foldRunIn_of_run pub proof messages (channelStart pub) (channelStart_fresh pub)
    (channelStart_empty pub) _ _ _ ℓ (fold20 ℓ) shape
    (replay_ran pub proof messages none_ (levels ℓ).at_)
  have passes : (foldRunIn pub (stateBefore pub proof messages pub.ops (fusedAt20 ℓ)) ℓ
      (fold20 ℓ)).isSome = true := by
    rw [← fusedAt]
    exact Option.isSome_iff_exists.mpr ⟨_, folded⟩
  have checked := final20_checked ℓ
  unfold finalCheck at checked
  simp only [Bool.and_eq_true, Bool.or_eq_true, decide_eq_true_eq] at checked
  obtain ⟨⟨held, rounds⟩, tensor⟩ := checked
  unfold held20 at held
  simp only [Bool.and_eq_true, decide_eq_true_eq] at held
  have free := joined_frees erased _ _ _ _ held.2
  have fold := foldPathIn_final E pub
    (s := stateBefore pub proof messages pub.ops (fusedAt20 ℓ))
    (t := stateBefore pub proof messages pub.ops (last20 ℓ)) ℓ (fold20 ℓ) (setup20 pub ℓ)
    (fold20 ℓ).vars (fold20 ℓ).batch
    (afterAt E (stateBefore pub proof messages pub.ops (last20 ℓ)) ℓ) passes
    (fun k inside => same_between pub proof messages pub.ops held.1 (ran _)
      (frees_writes free inside))
    (fun k inside => roundsIn_of_check erased (List.all_eq_true.mp rounds k inside) (ran _))
  unfold pathAt20 path20
  rw [← fold]
  simp only [fused20]
  rw [fusedAt, received]
  refine congrArg _ ?_
  unfold afterAt
  by_cases zero : ℓ.val = 0
  · rw [ite_eq_left zero, ite_eq_left zero]
  · rw [ite_eq_right zero, ite_eq_right zero]
    have passed := tensor20_passes layout accepted terminalSetup decoded ℓ zero
    rw [received] at passed
    have sums := roundsIn_of_check (pub := pub) (proof := proof) (messages := messages) erased
      (tensor.resolve_left zero) (ran _)
    show _ = tensorPath E (tail20 ℓ) _ _ _ _ _ _ _
    rw [tensorPath_full E _ _ _ _ _ _ _ _ passed]
    unfold tensorAt
    rw [sums]

/-- The cells from the state before the work of each stage are the cells of the page of the
proof: for an accepted proof of `layout20` with its decoded messages, `cellsAt20 … k` is
`cells20 … k` for each `k`. -/
theorem cellsAt20_eq {zMax rice : ℕ} (layout : pub.layout = layout20 zMax rice)
    (accepted : runProof pub proof = .verdict .accept) {full : Array Fq}
    (terminalSetup : pub.setupFor pub.layout.terminal.setupCount = some full) {rest : ℕ}
    (decoded : decode pub.ops proof = .ok (messages, rest)) (k : ℕ) :
    cellsAt20 E pub proof messages k = cells20 E pub proof messages k := by
  obtain ⟨scripted, ends, -⟩ := verifyProof_script20 layout accepted
  have none_ := (replay_of_decoded pub proof scripted ends terminalSetup accepted decoded).2
  have erased : pub.ops.toList.map Op.erase = skeleton (layout20 0 0) := by
    have scripted' := scripted
    rw [layout] at scripted'
    rw [script_erase scripted', skeleton20]
  exact cellsAt20_eq_of_final E erased
    (fun c => runFrom_take_ok pub proof messages _ c _
      ((replay_none_iff pub pub.ops proof messages).mp none_))
    (fun ℓ => pathAt20_final E layout accepted terminalSetup decoded ℓ) k

end Accepted

end NightstreamFPrime.Spec.Compression.Akita.Replay
