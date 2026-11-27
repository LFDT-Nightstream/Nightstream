import NightstreamFPrime.Spec.Compression.Akita.Replay.CausalCells
import NightstreamFPrime.Spec.Compression.Akita.Replay.StageReadings

/-! The draws of the path of each level of the opening of `key20.aks` as readings of the pinned
Akita code on the answers of an ideal sponge (Akita revision `d50f700`; ePrint 2026/1983,
revision of 2026-11-23, Section 8.2 `[KECCAK]` and Section 10.3; `docs/akita-work/w21/design.md`,
track Q; `Replay/CausalCells.lean`, `Replay/StageReadings.lean`).

- `Extract.draws path` are the draws of a path. `Extract.samples_of_draws`: a path on which the
  game runs has `Extract.Samples` on answers that give each draw of the path.
- `Draw` is one draw of the path of a level: the key of its recorded values, and the value
  `index` (`one`) or the first `count` values (`many`). `Draw.value E s` is the challenge on the
  recorded values of a state, and `Draw.read E values` the challenge on an output of `decodePool`.
- `draws20 ℓ` are the draws of `pathAt20 … ℓ` (`draws_pathAt20`), for each state.
- `Draw.check k L` is the check of a draw on the skeleton of `layout20`: the stage `k` has the
  field draw of the key and the count of the draw at the position of the draw, before the
  operation `L`. `draws20_checked`: the draw `j` of each level `ℓ` has the check at the stage
  `first20 ℓ + j`, a stage before the fold stage of the next level.
- `draw20_value`: on a run with no stop of a script of `layout20` with an ideal sponge, the draw
  `j` of the level `ℓ` on the state before `last20 ℓ` is `Draw.read` of the values that
  `decodePool` gives on the answer for the squeeze of the work of the stage `first20 ℓ + j` and
  `eventsAt20`.
- `path20_draws` and `path20_draw`: for an accepted proof, the draws of `path20 … ℓ` are
  `draws20 ℓ` on the state before `last20 ℓ`, and its draw `j` is `Draw.read` of these values. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Extract

open NightstreamFPrime.Spec.Compression (Game)
open NightstreamFPrime.Spec.Compression.Game (Move)

/-- The draws of a path. -/
def draws : List Move → List Move
  | [] => []
  | Move.draw challenge :: path => Move.draw challenge :: draws path
  | Move.recv _ :: path => draws path

theorem draws_nil : draws [] = [] := rfl

theorem draws_draw {C : Type} (challenge : C) (path : List Move) :
    draws (Move.draw challenge :: path) = Move.draw challenge :: draws path := rfl

theorem draws_recv {M : Type} (message : M) (path : List Move) :
    draws (Move.recv message :: path) = draws path := rfl

theorem draws_append (first second : List Move) :
    draws (first ++ second) = draws first ++ draws second := by
  induction first with
  | nil => rfl
  | cons move first induction =>
      cases move with
      | draw c =>
          show Move.draw c :: draws (first ++ second) = Move.draw c :: draws first ++ draws second
          rw [induction, List.cons_append]
      | recv m => exact induction

/-- A path on which the game runs has `Samples` on the answers `views`, when the answer `j` gives
the draw `j` of the path for each draw. -/
theorem samples_of_draws {Reply : Type} [Fintype Reply] {α : Type} (game : Game α)
    (path : List Move) (views : List Reply) {result : α} (runs : game.Runs path result)
    (reads : ∀ (j : ℕ) (C : Type) (c : C), (draws path)[j]? = some (Move.draw c) →
      ∃ reply, views[j]? = some reply ∧ ∀ (finite : Fintype C) (nonempty : Nonempty C),
        @sample Reply _ C finite nonempty reply = c) :
    Samples Reply game path views := by
  induction game generalizing path views with
  | pure a =>
      cases path with
      | nil => trivial
      | cons move path => cases move <;> exact runs.elim
  | draw C bad next induction =>
      cases path with
      | nil => exact runs.elim
      | cons move path =>
          cases move with
          | recv m => exact runs.elim
          | draw c =>
              obtain ⟨same, runs⟩ := runs
              subst same
              obtain ⟨reply, found, sampled⟩ := reads 0 _ c rfl
              cases views with
              | nil => cases found
              | cons head views =>
                  obtain rfl : head = reply := Option.some.inj found
                  exact ⟨rfl, sampled _ _, induction c path views runs fun j C' c' at_ =>
                    reads (j + 1) C' c' at_⟩
  | recv M next induction =>
      cases path with
      | nil => exact runs.elim
      | cons move path =>
          cases move with
          | draw c => exact runs.elim
          | recv m =>
              obtain ⟨same, runs⟩ := runs
              subst same
              exact ⟨rfl, induction m path views runs reads⟩

end NightstreamFPrime.Spec.Compression.Akita.Extract

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

open NightstreamFPrime.Spec.Compression.Game
open NightstreamFPrime.Spec.Compression.Akita
open NightstreamFPrime.Spec.Compression.Akita.Extract (draws draws_nil draws_draw draws_recv
  draws_append)

open Replayer

/-- One draw of the path of a level. `one`: the value `index` of the recorded values of the key.
`many`: the first `count` recorded values of the key. -/
inductive Draw where
  | one (key : Key) (index : ℕ)
  | many (key : Key) (count : ℕ)

namespace Draw

/-- The key of the recorded values of the draw. -/
def key : Draw → Key
  | one key _ => key
  | many key _ => key

/-- The count of the field challenges of the draw. -/
def count : Draw → ℕ
  | one .. => 1
  | many _ count => count

/-- The position of the first value of the draw in the recorded values of its key. -/
def index : Draw → ℕ
  | one _ index => index
  | many .. => 0

/-- The challenge space of the draw. -/
def Space (E : Type) : Draw → Type
  | one .. => E
  | many _ count => Fin count → E

variable (E : Type) [Field E] [Algebra Zq E] [Degree2 E]

/-- The challenge of the draw on an output of `decodePool`. -/
def read : (d : Draw) → Array Ext → d.Space E
  | one .., values => ((values[0]!).into : E)
  | many _ count, values => (listFn count (values.toList.map Ext.into) : Fin count → E)

/-- The challenge of the draw on the recorded values of a state. -/
def value (s : Replayer) : (d : Draw) → d.Space E
  | one key index => (((recordedIn s key).toList.getD index Ext.zero).into : E)
  | many key count =>
      (listFn count ((recordedIn s key).toList.map Ext.into) : Fin count → E)

/-- The move of the draw on the recorded values of a state. -/
def move (s : Replayer) (d : Draw) : Move := Move.draw (d.value E s)

theorem move_first (s : Replayer) (key : Key) :
    move E s (one key 0) = Move.draw ((firstIn s key).into : E) := by
  show Move.draw (((recordedIn s key).toList.getD 0 Ext.zero).into : E) = _
  rw [List.getD_eq_getElem?_getD, Array.getElem?_toList]
  rfl

/-- The challenge on the recorded values is the challenge on the values `values`, when the
recorded values of the key have `values` from the position of the draw. -/
theorem value_eq_read (s : Replayer) (d : Draw) {values : Array Ext}
    (size : values.size = d.count)
    (slice : ((recordedIn s d.key).toList.drop d.index).take d.count = values.toList) :
    d.value E s = d.read E values := by
  cases d with
  | one key index =>
      have inside : 0 < values.size := by rw [size]; exact Nat.one_pos
      have slice' : ((recordedIn s key).toList.drop index).take 1 = values.toList := slice
      have entry : (recordedIn s key).toList[index]? = some values[0] := by
        have first : ((recordedIn s key).toList.drop index)[0]? = values.toList[0]? := by
          rw [← slice', List.getElem?_take_of_lt Nat.one_pos]
        rw [List.getElem?_drop, Nat.add_zero] at first
        rw [first, Array.getElem?_toList, Array.getElem?_eq_getElem inside]
      show (((recordedIn s key).toList.getD index Ext.zero).into : E) = ((values[0]!).into : E)
      rw [getElem!_pos values 0 inside, List.getD_eq_getElem?_getD, entry]
      rfl
  | many key count =>
      show (listFn count ((recordedIn s key).toList.map Ext.into) : Fin count → E) =
        listFn count (values.toList.map Ext.into)
      funext i
      show ((recordedIn s key).toList.map (Ext.into (E := E))).getD i 0 =
        (values.toList.map (Ext.into (E := E))).getD i 0
      have slice' : (recordedIn s key).toList.take count = values.toList := slice
      rw [List.getD_eq_getElem?_getD, List.getD_eq_getElem?_getD, List.getElem?_map,
        List.getElem?_map, ← slice', List.getElem?_take_of_lt i.isLt]

end Draw

section Lists

/-- The draws of the rounds of a sum-check: the challenge `r` of its point for each round. -/
def roundDraws (key : Key) (n : ℕ) : List Draw :=
  (List.range n).map fun r => .one ⟨key.level, key.site ++ ".point", none⟩ r

/-- The draws of the tensor reduction of the claim of the level `level`. -/
def tensorDraws (level ν : ℕ) : List Draw :=
  .one ⟨level + 1, "tensor.eta", none⟩ 0 :: roundDraws ⟨level + 1, "tensor.sumcheck", none⟩ ν

/-- The draws of the path after the last stage of the range proof. -/
def closingDraws (level : ℕ) (bin : Bool) (n : ℕ) (after : List Draw) : List Draw :=
  (if bin then [Draw.one ⟨level, "binary_batch", none⟩ 0] else []) ++
    .one ⟨level, "stage2_batch", none⟩ 0 :: (roundDraws ⟨level, "fused", none⟩ n ++ after)

/-- The draws of the last stage of the range proof and of the path after it. -/
def lastDraws (level : ℕ) (sq bin : Bool) (n : ℕ) (after : List Draw) : List Draw :=
  match sq with
  | true =>
      .one ⟨level, "norm.merge", none⟩ 0 ::
        (roundDraws ⟨level, "norm.sumcheck", none⟩ n ++
          .one ⟨level, "norm.virtual_batch", none⟩ 0 :: closingDraws level bin n after)
  | false => roundDraws ⟨level, "range.leaf", none⟩ n ++ closingDraws level bin n after

/-- The draws of the product stages `stages`, and then the draws `last`. -/
def stageDraws (level n : ℕ) (last : List Draw) : List (ℕ × ℕ) → List Draw
  | [] => last
  | x :: more =>
      roundDraws ⟨level, s!"range.{x.2}", none⟩ n ++
        .one ⟨level, s!"range.{x.2}.batch", none⟩ 0 :: stageDraws level n last more

/-- The draws of `pathAt`. -/
def pathDraws (level : ℕ) (sq bin : Bool) (n batch : ℕ) (stages : List (ℕ × ℕ))
    (after : List Draw) : List Draw :=
  .one ⟨level, "alpha", none⟩ 0 :: .many ⟨level, "tau0", none⟩ n ::
    .many ⟨level, "tau1", none⟩ batch :: stageDraws level n (lastDraws level sq bin n after) stages

/-- The draws of `pathAt20`. -/
def draws20 (ℓ : Fin 6) : List Draw :=
  pathDraws ℓ (fold20 ℓ).euclidean (fold20 ℓ).compressed (fold20 ℓ).vars (fold20 ℓ).batch
    ((productStages (fold20 ℓ).logBasis).map (·.1)).zipIdx
    (if ℓ.val = 0 then [] else tensorDraws ℓ (tail20 ℓ))

end Lists

section Paths

variable {K : Type} [Field K] (φ : Ext → K)

theorem draws_plainPath (d : ℕ) : ∀ (rounds : List (List Ext × Ext)) (claim : Ext),
    draws (plainPath φ d claim rounds) = rounds.map fun x => Move.draw (φ x.2)
  | [], _ => rfl
  | (message, r) :: rounds, claim => by
      show Move.draw (φ r) :: draws (plainPath φ d (plainStep claim message r) rounds) = _
      rw [draws_plainPath d rounds, List.map_cons]

theorem draws_eqCodePath (D : ℕ) : ∀ rounds : List (List Ext × Ext),
    draws (eqCodePath φ D rounds) = rounds.map fun x => Move.draw (φ x.2)
  | [] => rfl
  | (message, r) :: rounds => by
      show Move.draw (φ r) :: draws (eqCodePath φ D rounds) = _
      rw [draws_eqCodePath D rounds, List.map_cons]

variable (E : Type) [Field E] [Algebra Zq E] [Degree2 E] (pub : Public) (s : Replayer)

theorem roundsFrom_moves (key : Key) (n : ℕ) :
    ((roundsFrom s key n).map fun x => Move.draw ((x.2.into : E))) =
      (roundDraws key n).map (Draw.move E s) := by
  unfold roundsFrom roundDraws
  rw [List.map_map, List.map_map]
  rfl

theorem draws_tensorAt (level ν : ℕ) :
    draws (tensorAt E s level ν) = (tensorDraws level ν).map (Draw.move E s) := by
  unfold tensorAt tensorFull tensorDraws
  rw [draws_recv, draws_draw, draws_append, draws_plainPath, draws_recv, draws_nil,
    List.append_nil, List.map_cons, Draw.move_first, roundsFrom_moves]

theorem draws_closingAt (level : ℕ) (F : Fold) (bin : Bool) (n : ℕ) {after : List Move}
    {afterD : List Draw} (ends : draws after = afterD.map (Draw.move E s)) :
    draws (closingAt E pub s level F bin n after) =
      (closingDraws level bin n afterD).map (Draw.move E s) := by
  unfold closingAt closingFull closingDraws binPath
  rw [draws_append, draws_draw, draws_append, draws_plainPath, draws_recv, ends, List.map_append,
    List.map_cons, List.map_append, Draw.move_first, roundsFrom_moves]
  cases bin
  · rfl
  · show [Move.draw _] ++ _ = [Draw.move E s _] ++ _
    rw [Draw.move_first]

theorem draws_lastAt (level : ℕ) (F : Fold) (sq bin : Bool) (n : ℕ) {after : List Move}
    {afterD : List Draw} (ends : draws after = afterD.map (Draw.move E s)) (claim : Ext) :
    draws (lastAt E pub s level F sq bin n after claim) =
      (lastDraws level sq bin n afterD).map (Draw.move E s) := by
  cases sq
  · show draws (leafPath _ _ _ _ ++ closingAt E pub s level F bin n after) =
      (roundDraws ⟨level, "range.leaf", none⟩ n ++ closingDraws level bin n afterD).map _
    unfold leafPath
    rw [draws_append, draws_append, draws_eqCodePath, draws_recv, draws_nil, List.append_nil,
      draws_closingAt E pub s level F bin n ends, List.map_append, roundsFrom_moves]
  · show draws (normFull _ _ _ _ _ _ _ (Move.draw _ :: closingAt E pub s level F bin n after)
        claim) =
      (Draw.one ⟨level, "norm.merge", none⟩ 0 ::
        (roundDraws ⟨level, "norm.sumcheck", none⟩ n ++
          .one ⟨level, "norm.virtual_batch", none⟩ 0 :: closingDraws level bin n afterD)).map _
    unfold normFull
    rw [draws_recv, draws_draw, draws_append, draws_plainPath, draws_recv, draws_draw,
      draws_closingAt E pub s level F bin n ends, List.map_cons, List.map_append, List.map_cons,
      Draw.move_first, Draw.move_first, roundsFrom_moves]

theorem draws_treeFull (level n : ℕ) {last : Ext → List Move} {lastD : List Draw}
    (ends : ∀ claim, draws (last claim) = lastD.map (Draw.move E s)) :
    ∀ (stages : List (ℕ × ℕ)) (nodes : ℕ) (claim : Ext),
      draws (treeFull (Ext.into (E := E)) last (stages.map (productFrom s level n)) nodes
        claim) = (stageDraws level n lastD stages).map (Draw.move E s)
  | [], _, claim => ends claim
  | x :: more, nodes, _ => by
      show draws (eqCodePath (Ext.into (E := E)) x.1
          (roundsFrom s ⟨level, s!"range.{x.2}", none⟩ n) ++
        Move.recv _ :: Move.draw ((firstIn s ⟨level, s!"range.{x.2}.batch", none⟩).into : E) ::
          treeFull (Ext.into (E := E)) last (more.map (productFrom s level n)) _ _) =
        (roundDraws ⟨level, s!"range.{x.2}", none⟩ n ++
          .one ⟨level, s!"range.{x.2}.batch", none⟩ 0 :: stageDraws level n lastD more).map _
      rw [draws_append, draws_eqCodePath, draws_recv, draws_draw,
        draws_treeFull level n ends more, List.map_append, List.map_cons, Draw.move_first,
        roundsFrom_moves]

theorem draws_pathAt (level : ℕ) (F : Fold) (sq bin : Bool) (n batch : ℕ)
    (stages : List (ℕ × ℕ)) {after : List Move} {afterD : List Draw}
    (ends : draws after = afterD.map (Draw.move E s)) :
    draws (pathAt E pub s level F sq bin n batch stages after) =
      (pathDraws level sq bin n batch stages afterD).map (Draw.move E s) := by
  unfold pathAt pathDraws
  rw [draws_draw, draws_draw, draws_draw,
    draws_treeFull E s level n (draws_lastAt E pub s level F sq bin n ends) stages,
    List.map_cons, List.map_cons, List.map_cons, Draw.move_first]
  rfl

/-- The draws of `pathAt20 … ℓ` on a state are the draws `draws20 ℓ` on its recorded values. -/
theorem draws_pathAt20 (ℓ : Fin 6) :
    draws (pathAt20 E pub s ℓ) = (draws20 ℓ).map (Draw.move E s) := by
  unfold pathAt20 draws20
  refine draws_pathAt E pub s _ _ _ _ _ _ _ ?_
  unfold afterAt
  by_cases zero : ℓ.val = 0
  · rw [ite_eq_left zero, ite_eq_left zero]
    rfl
  · rw [ite_eq_right zero, ite_eq_right zero]
    exact draws_tensorAt E s ℓ (tail20 ℓ)

end Paths

section Checks

/-- The check of a draw on the skeleton of `layout20`, for the stage `k` and the last state
before the operation `L`: the stage has the field draw of the key and the count of the draw,
after `index` recorded values of the key, and the draw is before `L`. -/
def Draw.check (k L : ℕ) (d : Draw) : Bool :=
  decide (fieldDraw20 k = some (d.key, d.count)) && decide (offset20 k = d.index) &&
    decide (workIndices20.getD k 0 + 1 + 1 ≤ L)

/-- The check of each draw of the level `ℓ`: the draw `j` has the check at the stage
`first20 ℓ + j`, and this stage is before the fold stage of the next level. -/
def drawsChecked20 (ℓ : Fin 6) : Bool :=
  (draws20 ℓ).zipIdx.all fun x =>
    x.1.check (first20 ℓ + x.2) (last20 ℓ) && decide (first20 ℓ + x.2 < bound20 ℓ)

theorem draws20_checked₀ : drawsChecked20 0 = true := by decide +kernel

theorem draws20_checked₁ : drawsChecked20 1 = true := by decide +kernel

theorem draws20_checked₂ : drawsChecked20 2 = true := by decide +kernel

theorem draws20_checked₃ : drawsChecked20 3 = true := by decide +kernel

theorem draws20_checked₄ : drawsChecked20 4 = true := by decide +kernel

theorem draws20_checked₅ : drawsChecked20 5 = true := by decide +kernel

/-- The draw `j` of the level `ℓ` is the field draw of the stage `first20 ℓ + j`, a stage before
the fold stage of the next level, at the position of the draw and before `last20 ℓ`. -/
theorem draws20_checked (ℓ : Fin 6) {j : ℕ} {d : Draw} (at_ : (draws20 ℓ)[j]? = some d) :
    fieldDraw20 (first20 ℓ + j) = some (d.key, d.count) ∧ offset20 (first20 ℓ + j) = d.index ∧
      workIndices20.getD (first20 ℓ + j) 0 + 1 + 1 ≤ last20 ℓ ∧ first20 ℓ + j < bound20 ℓ := by
  have checked : drawsChecked20 ℓ = true := by
    match ℓ with
    | 0 => exact draws20_checked₀
    | 1 => exact draws20_checked₁
    | 2 => exact draws20_checked₂
    | 3 => exact draws20_checked₃
    | 4 => exact draws20_checked₄
    | 5 => exact draws20_checked₅
  have member : (d, j) ∈ (draws20 ℓ).zipIdx := by
    refine List.mem_iff_getElem?.mpr ⟨j, ?_⟩
    rw [List.getElem?_zipIdx, at_, Nat.zero_add]
    rfl
  have fit := List.all_eq_true.mp checked (d, j) member
  unfold Draw.check at fit
  simp only [Bool.and_eq_true, decide_eq_true_eq] at fit
  exact ⟨fit.1.1.1, fit.1.1.2, fit.1.2, fit.2⟩

end Checks

section Readings

variable (E : Type) [Field E] [Algebra Zq E] [Degree2 E]
  {pub : Public} {proof : ByteArray} {messages : Array Message}

/-- The draw `j` of the level `ℓ` on the state before `last20 ℓ`, for a script of `layout20` with
the work bits of the model, a run with no stop before `last20 ℓ` and an ideal sponge:
`Draw.read` of the values that `decodePool` gives on the answer for the squeeze of the work of
the stage `first20 ℓ + j` and `eventsAt20`. -/
theorem draw20_value (erased : pub.ops.toList.map Op.erase = skeleton (layout20 0 0))
    (bits : Bits20 pub) (ℓ : Fin 6) {j : ℕ} {d : Draw} (at_ : (draws20 ℓ)[j]? = some d)
    (ran : (runFrom pub proof messages (pub.ops.zipIdx.toList.take (last20 ℓ))
      (channelStart pub)).1 = .ok ())
    {answer : List Duplex.Event → ℕ → ByteArray} (ideal : pub.sponge.ideal = some answer) :
    ∃ values,
      decodePool (answer (.squeeze 8 :: eventsAt20 pub proof messages (first20 ℓ + j))
        (poolBytes (2 * d.count))) d.count = some values ∧
      d.value E (stateBefore pub proof messages pub.ops (last20 ℓ)) = d.read E values := by
  obtain ⟨draw, offset, later, bounded⟩ := draws20_checked ℓ at_
  have inside : first20 ℓ + j < 473 := Nat.lt_trans bounded (bound20_last ℓ).1
  rw [← stageIndex20_lt pub inside] at later
  obtain ⟨values, decoded, slice⟩ := challenge20_recorded erased bits inside draw later ran ideal
  rw [offset] at slice
  exact ⟨values, decoded, Draw.value_eq_read E _ d (decodePool_size decoded) slice⟩

variable {E}

/-- An accepted proof of `layout20` has a script of `layout20` and a run with no stop. -/
theorem ran_of_accept {zMax rice : ℕ} (layout : pub.layout = layout20 zMax rice)
    (accepted : runProof pub proof = .verdict .accept) {full : Array Fq}
    (terminalSetup : pub.setupFor pub.layout.terminal.setupCount = some full) {rest : ℕ}
    (decoded : decode pub.ops proof = .ok (messages, rest)) :
    pub.ops.toList.map Op.erase = skeleton (layout20 0 0) ∧
      ∀ c, (runFrom pub proof messages (pub.ops.zipIdx.toList.take c) (channelStart pub)).1 =
        .ok () := by
  obtain ⟨scripted, ends, -⟩ := verifyProof_script20 layout accepted
  have none_ := (replay_of_decoded pub proof scripted ends terminalSetup accepted decoded).2
  have erased : pub.ops.toList.map Op.erase = skeleton (layout20 0 0) := by
    have scripted' := scripted
    rw [layout] at scripted'
    rw [script_erase scripted', skeleton20]
  exact ⟨erased, fun c => runFrom_take_ok pub proof messages _ c _
    ((replay_none_iff pub pub.ops proof messages).mp none_)⟩

variable (E)

/-- The draws of the path `path20 … ℓ` of the code of an accepted proof of `layout20` are the
draws `draws20 ℓ` on the state before `last20 ℓ`. -/
theorem path20_draws {zMax rice : ℕ} (layout : pub.layout = layout20 zMax rice)
    (accepted : runProof pub proof = .verdict .accept) {full : Array Fq}
    (terminalSetup : pub.setupFor pub.layout.terminal.setupCount = some full) {rest : ℕ}
    (decoded : decode pub.ops proof = .ok (messages, rest)) (ℓ : Fin 6) :
    draws (path20 pub proof messages E ℓ) =
      (draws20 ℓ).map (Draw.move E (stateBefore pub proof messages pub.ops (last20 ℓ))) := by
  rw [← pathAt20_final E layout accepted terminalSetup decoded ℓ]
  exact draws_pathAt20 E pub _ ℓ

/-- The draw `j` of the path `path20 … ℓ` of the code, for an accepted proof of `layout20` with
the work bits of the model and an ideal sponge: the stage `first20 ℓ + j` is before the fold
stage of the next level, and the draw is `Draw.read` of the values that `decodePool` gives on the
answer for the squeeze of the work of this stage and `eventsAt20`. -/
theorem path20_draw {zMax rice : ℕ} (layout : pub.layout = layout20 zMax rice)
    (accepted : runProof pub proof = .verdict .accept) {full : Array Fq}
    (terminalSetup : pub.setupFor pub.layout.terminal.setupCount = some full) {rest : ℕ}
    (decoded : decode pub.ops proof = .ok (messages, rest)) (bits : Bits20 pub)
    {answer : List Duplex.Event → ℕ → ByteArray} (ideal : pub.sponge.ideal = some answer)
    (ℓ : Fin 6) {j : ℕ} {d : Draw} (at_ : (draws20 ℓ)[j]? = some d) :
    first20 ℓ + j < bound20 ℓ ∧ ∃ values,
      decodePool (answer (.squeeze 8 :: eventsAt20 pub proof messages (first20 ℓ + j))
        (poolBytes (2 * d.count))) d.count = some values ∧
      (draws (path20 pub proof messages E ℓ))[j]? = some (Move.draw (d.read E values)) := by
  obtain ⟨erased, ran⟩ := ran_of_accept layout accepted terminalSetup decoded
  obtain ⟨values, pooled, same⟩ := draw20_value E erased bits ℓ at_ (ran _) ideal
  refine ⟨(draws20_checked ℓ at_).2.2.2, values, pooled, ?_⟩
  rw [path20_draws E layout accepted terminalSetup decoded ℓ, List.getElem?_map, at_, ← same]
  rfl

end Readings

end NightstreamFPrime.Spec.Compression.Akita.Replay
