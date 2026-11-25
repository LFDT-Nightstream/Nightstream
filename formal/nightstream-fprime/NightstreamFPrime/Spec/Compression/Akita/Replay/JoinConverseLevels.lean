import NightstreamFPrime.Spec.Compression.Akita.Extract.StagedDraws
import NightstreamFPrime.Spec.Compression.Akita.Replay.JoinConverseRuns

/-! Theorem B of the join, direction "←", at the levels 0 to 5 of `key20.aks` (Akita revision
`d50f700`; ePrint 2026/1983, revision of 2026-11-23, Section 5.4 and Figure 4, steps 0, 3 and 5;
`Tier.Linked`, `Tier.TraceLinked` and `Terminal.Linked` of the model).

- `model20_level0` to `model20_level5`: on a transcript that presents the bytes of a proof, when
  the link of the level of `Key20.Tower.accepts` holds and the run before the `fused.end` of the
  level has no stop, the game of the fold of the level runs on the path of the code to the result
  of `foldRunIn`, and with the tensor reduction of its claim (the levels 1 to 5) it runs to
  `some` on the path of the fold and the path of the tensor reduction of the code. The statement
  of the game is the statement of the code by the links `key20_rootLink_before`,
  `key20_packLink_before` and `key20_traceLink_before`.
- `model20_draws5`: on such a transcript, when the link of the level 5 holds, the path of the
  code of the level 5 has the `foldDrawsSq` draws of its game or more. It has no hypothesis on
  the run of the replay.
- `level_ok_of_model`: from such runs, the steps of `fused.end`, `tensor.opening` and
  `tensor.end` of a level have no stop when the run before them has none.
- `level20_ok₀` to `level20_ok₅`: these steps for the levels of `key20.aks`, as `StepOk`. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

open NightstreamFPrime.Spec.Compression.Lib NightstreamFPrime.Spec.Compression.Game
open NightstreamFPrime.Spec.Compression (Game)
open NightstreamFPrime.Spec.Compression.Akita
open NightstreamFPrime.Spec.Compression.Akita.Extract (Tier Presents view Prefix)

open Replayer

variable {E : Type} [Field E] [Fintype E] [DecidableEq E] [Algebra Zq E] [Degree2 E]

section Combine

variable (pub : Public) (proof : ByteArray) (messages : Array Message) {ops : Array Op}

/-- The steps of `fused.end` at `c`, of `tensor.opening` at `o` and of `tensor.end` at `t` of a
level have no stop when the run before them has none, from the runs `model` of the game of the
level: after a run without a stop before `c`, the game of the fold runs on `path` to the result of
`foldRunIn`, and with the tensor reduction it runs to `some` on `path` and the path of the tensor
reduction on the state before `c'`. -/
theorem level_ok_of_model (aligned : ReadsAligned messages ops.zipIdx.toList 0)
    {level c o t c' ν : ℕ} {F : Fold} {r a e : ℕ}
    (fusedAt : ops[c]? = some (.check ⟨level, "fused.end", none⟩ (.fused level F)))
    (shape : foldShape (ops.zipIdx.toList.take c) level F r a e = true)
    (frame : TensorFrame ops level ν c t c')
    (openingAt : ops[o]? = some (.check ⟨level + 1, "tensor.opening", none⟩
      (.tensorOpening ⟨level, "fused.point", none⟩ ⟨level, "witness_evaluation", none⟩
        ⟨level + 1, "tensor.partials", none⟩)))
    (held : frees ((ops.zipIdx.toList.take o).drop c)
      [⟨level, "fused.point", none⟩, ⟨level, "witness_evaluation", none⟩] [] = true)
    (co : c < o) (ot : o < t) (tc : t < c') {path : List Move}
    (model : (runFrom pub proof messages (ops.zipIdx.toList.take c) (channelStart pub)).1 =
        .ok () →
      ∃ (g : Game (Option (List E × E))) (x : List E × E × E),
        g.Runs path ((foldRunIn pub (stateBefore pub proof messages ops c) level F).map fun o =>
          (o.1.map Ext.into, o.2.into)) ∧
        (g.bind fun
          | none => .pure none
          | some (r2, v) => tensorCheck ν 0 r2 v).Runs
          (path ++ tensoredPathIn E ν level (stateBefore pub proof messages ops c')) (some x)) :
    ((runFrom pub proof messages (ops.zipIdx.toList.take c) (channelStart pub)).1 = .ok () →
        (((step pub proof messages c (.check ⟨level, "fused.end", none⟩ (.fused level F))).run).run
          (stateBefore pub proof messages ops c)).1 = .ok ()) ∧
      ((runFrom pub proof messages (ops.zipIdx.toList.take o) (channelStart pub)).1 = .ok () →
        (((step pub proof messages o (.check ⟨level + 1, "tensor.opening", none⟩
          (.tensorOpening ⟨level, "fused.point", none⟩ ⟨level, "witness_evaluation", none⟩
            ⟨level + 1, "tensor.partials", none⟩))).run).run
              (stateBefore pub proof messages ops o)).1 = .ok ()) ∧
      ((runFrom pub proof messages (ops.zipIdx.toList.take t) (channelStart pub)).1 = .ok () →
        (((step pub proof messages t (.check ⟨level + 1, "tensor.end", none⟩
          (.tensor (level + 1) ν ⟨level, "fused.point", none⟩
            ⟨level, "witness_evaluation", none⟩))).run).run
              (stateBefore pub proof messages ops t)).1 = .ok ()) := by
  refine ⟨fun before => ?_, fun before => ?_, fun before => ?_⟩
  · obtain ⟨g, x, fold, runs⟩ := model before
    exact fused_ok_of_runs pub proof messages shape before fold runs (by rfl)
  · obtain ⟨g, x, fold, runs⟩ := model (before_le pub proof messages (Nat.le_of_lt co) before)
    exact opening_ok_of_runs pub proof messages fusedAt shape co openingAt held (lt_trans ot tc)
      before fold runs
  · obtain ⟨g, x, fold, runs⟩ := model (before_le pub proof messages (by omega) before)
    exact tensor_ok_of_runs pub proof messages aligned frame tc fusedAt shape (lt_trans co ot)
      before fold runs

end Combine

variable {Coord : Type} [Fintype Coord] [DecidableEq Coord] {total : ℕ}
  (o : Key20.Tower E (Key20.Wire E) Coord total)

local notation "code₁" =>
  (CodeSlot.open Key20.open64 Key20.open64_fits :
    CodeSlot E → Option (Rq Key20.level₁.opening.carrierDegree))

local notation "code₂" =>
  (CodeSlot.open Key20.open128 Key20.open128_fits :
    CodeSlot E → Option (Rq Key20.level₂.shape.innerDegree))

local notation "zeroth" =>
  Extract.liftBatch o.β o.key o.openKey o.claim Key20.rootVars (And.left Key20.root_sized) 8
    Key20.quotientDigits Key20.root_room Key20.rootBatch (And.left (And.right Key20.root_sized))
    o.block (Key20.rootDecode E) o.read Key20.rootMass (0 + 1)

local notation "first" =>
  Tier.tensored (Extract.liftFold o.β₁ o.key₁ o.openKey₁ o.claim₁ Key20.level₁.vars
    (And.left Key20.level₁_sized) Key20.quotientDigits (And.left (And.right Key20.level₁_sized))
    Key20.level₁.batch (And.left (And.right (And.right Key20.level₁_sized))) o.block₁ code₁ o.read
    51 (Tier.finish zeroth)) Key20.level₁.tail

local notation "second" =>
  Tier.tensored (Extract.committedTrace (fold := Key20.level₂.fold) o.key₂ o.openKey₂ o.claim₂
    Key20.level₂.vars (And.left Key20.sized₂) Key20.level₂.batch (And.left (And.right Key20.sized₂))
    o.block₂ code₂ o.read 31 (Tier.finish first)) Key20.level₂.tail

local notation "code₃" =>
  (CodeSlot.kept128 : CodeSlot E → Option (Rq Key20.level₃.shape.innerDegree))

local notation "code₄" =>
  (CodeSlot.kept64 : CodeSlot E → Option (Rq Key20.level₄.shape.innerDegree))

local notation "code₅" =>
  (CodeSlot.kept64 : CodeSlot E → Option (Rq Key20.level₅.shape.innerDegree))

local notation "third" =>
  Tier.tensored (Extract.committedTraceSq (fold := Key20.level₃.fold) o.key₃ o.openKey₃ o.claim₃
    Key20.level₃.vars (And.left Key20.sized₃) Key20.level₃.batch
    (And.left (And.right Key20.sized₃)) o.block₃ code₃ o.read 31 Key20.norm₃.operator
    Key20.norm₃.limit (Tier.finish second)) Key20.level₃.tail

local notation "fourth" =>
  Tier.tensored (Extract.committedTraceSq (fold := Key20.level₄.fold) o.key₄ o.openKey₄ o.claim₄
    Key20.level₄.vars (And.left Key20.sized₄) Key20.level₄.batch
    (And.left (And.right Key20.sized₄)) o.block₄ code₄ o.read 53 Key20.norm₄.operator
    Key20.norm₄.limit (Tier.finish third)) Key20.level₄.tail

local notation "fifth" =>
  Tier.tensored (Extract.committedTraceSq (fold := Key20.level₅.fold) o.key₅ o.openKey₅ o.claim₅
    Key20.level₅.vars (And.left Key20.sized₅) Key20.level₅.batch
    (And.left (And.right Key20.sized₅)) o.block₅ code₅ o.read 53 Key20.norm₅.operator
    Key20.norm₅.limit (Tier.finish fourth)) Key20.level₅.tail

variable (pub : Public) (proof : ByteArray) (messages : Array Message)
  (tLast : ℕ → Rq Key20.terminal.innerDegree) (eLast : Rq Key20.terminal.innerDegree)
  {block : Key20.root.Block → Coord} {block₁ : Fin Key20.level₁.shape.blocks → Coord}
  {block₂ : Fin Key20.level₂.shape.blocks → Coord} {block₃ : Fin Key20.level₃.shape.blocks → Coord}
  {block₄ : Fin Key20.level₄.shape.blocks → Coord} {block₅ : Fin Key20.level₅.shape.blocks → Coord}
  {blockL : Fin Key20.terminal.blocks → Coord}

/-- Theorem B "←" at the root fold: on a transcript that presents the bytes of a proof, when the
link `Tier.Linked` of the root fold holds and the run before its `fused.end` has no stop, the
game of the root fold runs on the path of the code to the result of `foldRunIn` and to `some`. -/
theorem model20_level0 {zMax rice : ℕ} (layout : pub.layout = layout20 zMax rice)
    (scripted : script pub.layout pub.plan = .ok pub.ops)
    (aligned : ReadsAligned messages pub.ops.zipIdx.toList 0)
    (points : ∀ g : Fin 3, ((pub.groups[g.val]?.map (·.point)).getD #[]).size = ![20, 16, 16] g)
    (values : (pub.groups.flatMap (·.value)).size = 3)
    (expanded : (pub.setupFor (fold20 0).setupCount).isSome)
    (tower : o = tower20 pub block block₁ block₂ block₃ block₄ block₅ blockL total 472)
    {point : Prefix (Key20.Wire E) (Option Coord → CodeSlot E) total}
    (presents : PresentsAt20 o pub proof messages tLast eLast point)
    (linked : Tier.Linked (zeroth) (first) o.commit₁ point)
    (before : (runFrom pub proof messages (pub.ops.zipIdx.toList.take (fusedIndex pub.layout 0))
      (channelStart pub)).1 = .ok ()) :
    ∃ (g : Game (Option (List E × E))) (x : List E × E),
      g.Runs (foldPathIn E pub (fused20 pub proof messages 0) 0 (fold20 0) (setup20 pub 0)
          (fold20 0).vars (fold20 0).batch)
        ((foldRunIn pub (fused20 pub proof messages 0) 0 (fold20 0)).map fun o =>
          (o.1.map Ext.into, o.2.into)) ∧
      g.Runs (foldPathIn E pub (fused20 pub proof messages 0) 0 (fold20 0) (setup20 pub 0)
          (fold20 0).vars (fold20 0).batch) (some x) := by
  obtain ⟨⟨reply, answered, decodedAt⟩, presented⟩ := presents.level₀
  have challenged := Extract.challengeAt_of_reply _ _ _ answered decodedAt
  subst tower
  obtain ⟨-, -, _, _, _, _, _, gameEq, -⟩ := linked
  obtain ⟨rootFrame, -⟩ := frames20 layout scripted
  have found : pub.setupFor (fold20 0).setupCount = some (setup20 pub 0) := by
    obtain ⟨setup, found⟩ := Option.isSome_iff_exists.mp expanded
    simp [setup20, found]
  have rooted := (key20_rootLink_before (E := E) pub proof before (layout20_fold zMax rice 0)
    rootFrame.weights (fun g => 3 + g.val) rootFrame.pointAt rootFrame.pointFresh
    rootFrame.pointAfter rootFrame.pointKept rootFrame.openingsAt rootFrame.openingsFresh
    rootFrame.openingsAfter rootFrame.openingsKept points values (setup20 pub 0)).2
  obtain ⟨fits, room, rowsFit, runs⟩ := key20_root_before_of (E := E) pub proof layout scripted
    aligned before rfl found (fun _ => basis20 E) (fun _ t => basis20_apply t)
    (fun g i => challenge20₀ pub proof messages ⟨g, i⟩) (fun k i => by
      simp only [challenge20₀, Equiv.symm_apply_apply]) 0
  have rooted' : batchStatementFor (E := E) Key20.root (setup20 pub 0)
      (claimPointIn (fused20 pub proof messages 0) 0) (dataIn pub (fused20 pub proof messages 0) 0)
      (fun k => .root (groupOrder 3 k).val) (rowsIn (fused20 pub proof messages 0) 0 (fold20 0))
      (claimsIn (fused20 pub proof messages 0) 0 (fold20 0)) =
      ((point.wire 1).data.claim).statement
        (fun g => Key.ofSetup (flatOf (setup20 pub 0)) (Key20.root.shape g))
        ⟨Setup.shared (flatOf (setup20 pub 0)) Key20.root.openDegree Key20.root.openRank _
          Key20.root.openSetupCode, PathKey.ofSetup (flatOf (setup20 pub 0)) Key20.root.chain⟩ := by
    rw [presents.reads.claim]
    exact rooted
  rw [rooted'] at runs
  rw [path20_zero] at presented
  have ran := (Extract.Tier.game_eq_pure_iff _ 0 point presented _).mp gameEq
  unfold Tier.challenge at ran
  rw [challenged] at ran
  exact ⟨_, _, runs, ran⟩

/-- Theorem B "←" at the level 1: on a transcript that presents the bytes of a proof, when the
link `Tier.TraceLinked` of the level 1 holds and the run before its `fused.end` has no stop, the
game of its fold runs on the path of the code to the result of `foldRunIn`, and with the tensor
reduction of its claim it runs to `some`. -/
theorem model20_level1 {zMax rice : ℕ} (layout : pub.layout = layout20 zMax rice)
    (scripted : script pub.layout pub.plan = .ok pub.ops)
    (aligned : ReadsAligned messages pub.ops.zipIdx.toList 0)
    (expanded : (pub.setupFor (fold20 1).setupCount).isSome)
    (tower : o = tower20 pub block block₁ block₂ block₃ block₄ block₅ blockL total 472)
    {point : Prefix (Key20.Wire E) (Option Coord → CodeSlot E) total}
    (presents : PresentsAt20 o pub proof messages tLast eLast point)
    (linked : Tier.TraceLinked (first) (second) (And.left Key20.quarters) o.commit₂ point)
    (before : (runFrom pub proof messages (pub.ops.zipIdx.toList.take (fusedIndex pub.layout 1))
      (channelStart pub)).1 = .ok ()) :
    ∃ (g : Game (Option (List E × E))) (x : List E × E × E),
      g.Runs (foldPathIn E pub (fused20 pub proof messages 1) 1 (fold20 1) (setup20 pub 1)
          (fold20 1).vars (fold20 1).batch)
        ((foldRunIn pub (fused20 pub proof messages 1) 1 (fold20 1)).map fun o =>
          (o.1.map Ext.into, o.2.into)) ∧
      (g.bind fun
        | none => .pure none
        | some (r2, v) => tensorCheck Key20.level₁.tail 0 r2 v).Runs
        (foldPathIn E pub (fused20 pub proof messages 1) 1 (fold20 1) (setup20 pub 1)
            (fold20 1).vars (fold20 1).batch ++
          tensoredPathIn E Key20.level₁.tail 1 (stateBefore pub proof messages pub.ops (fusedIndex pub.layout 2)))
        (some x) := by
  obtain ⟨⟨reply, answered, decodedAt⟩, presented⟩ := presents.level₁
  have challenged := Extract.challengeAt_of_reply _ _ _ answered decodedAt
  subst tower
  obtain ⟨-, -, _, _, _, _, _, _, gameEq, -⟩ := linked
  have levels := levels20 layout scripted
  obtain ⟨-, pack, -⟩ := frames20 layout scripted
  have found : pub.setupFor (fold20 1).setupCount = some (setup20 pub 1) := by
    obtain ⟨setup, found⟩ := Option.isSome_iff_exists.mp expanded
    simp [setup20, found]
  have game₀ : gameShape (pub.ops.zipIdx.toList.take (fusedIndex pub.layout 0)) 0 (fold20 0) =
      true := (levels 0).game
  have packed := (key20_packLink_before (E := E) pub proof aligned before
    (layout20_fold zMax rice 0) game₀ pack.pointAt pack.openingsAt pack.pointBefore
    pack.openingsBefore pack.pointAfter pack.openingsAfter pack.pointHeld pack.openingsHeld
    pack.pointFresh pack.openingsFresh pack.pointKept pack.openingsKept
    (layout20_fold zMax rice 1) (setup20 pub 1)).2.2.2.2.2
  obtain ⟨fits, room, rowsFit, runs⟩ := key20_level1_before_of (E := E) pub proof layout scripted
    aligned before rfl found (basis20 E) basis20_apply (challenge20₁ pub proof messages)
    (fun _ => rfl) 0
  have packed' : packStatementFor (E := E) Key20.level₁.opening (setup20 pub 1)
      (claimPointIn (fused20 pub proof messages 1) 1 0) (dataIn pub (fused20 pub proof messages 1) 1)
      .witness (rowsIn (fused20 pub proof messages 1) 1 (fold20 1))
      (claimsIn (fused20 pub proof messages 1) 1 (fold20 1)) =
      ((point.wire 51).data.claim₁).statement (Key.ofSetup (flatOf (setup20 pub 1)) Key20.level₁.shape)
        (OpeningKey.ofSetup (flatOf (setup20 pub 1)) Key20.level₁.opening) := by
    rw [presents.reads.claim₁]
    exact packed
  rw [packed'] at runs
  rw [path20_trace pub proof messages 1 (by decide), receiver20_lt pub (by decide)] at presented
  have ran := (Extract.Tier.game_eq_pure_iff _ 0 point presented _).mp gameEq
  unfold Tier.challenge at ran
  rw [challenged] at ran
  exact ⟨_, _, runs, by rw [tensoredPathIn_eq]; exact ran⟩

/-- Theorem B "←" at the level 2: on a transcript that presents the bytes of a proof, when the
link `Tier.TraceLinked` of the level 2 holds and the run before its `fused.end` has no stop, the
game of its fold runs on the path of the code to the result of `foldRunIn`, and with the tensor
reduction of its claim it runs to `some`. -/
theorem model20_level2 {zMax rice : ℕ} (layout : pub.layout = layout20 zMax rice)
    (scripted : script pub.layout pub.plan = .ok pub.ops)
    (aligned : ReadsAligned messages pub.ops.zipIdx.toList 0)
    (expanded : (pub.setupFor (fold20 2).setupCount).isSome)
    (tower : o = tower20 pub block block₁ block₂ block₃ block₄ block₅ blockL total 472)
    {point : Prefix (Key20.Wire E) (Option Coord → CodeSlot E) total}
    (presents : PresentsAt20 o pub proof messages tLast eLast point)
    (linked : Tier.TraceLinked (second) (third) (And.left (And.right Key20.quarters)) o.commit₃ point)
    (before : (runFrom pub proof messages (pub.ops.zipIdx.toList.take (fusedIndex pub.layout 2))
      (channelStart pub)).1 = .ok ()) :
    ∃ (g : Game (Option (List E × E))) (x : List E × E × E),
      g.Runs (foldPathIn E pub (fused20 pub proof messages 2) 2 (fold20 2) (setup20 pub 2)
          (fold20 2).vars (fold20 2).batch)
        ((foldRunIn pub (fused20 pub proof messages 2) 2 (fold20 2)).map fun o =>
          (o.1.map Ext.into, o.2.into)) ∧
      (g.bind fun
        | none => .pure none
        | some (r2, v) => tensorCheck Key20.level₂.tail 0 r2 v).Runs
        (foldPathIn E pub (fused20 pub proof messages 2) 2 (fold20 2) (setup20 pub 2)
            (fold20 2).vars (fold20 2).batch ++
          tensoredPathIn E Key20.level₂.tail 2 (stateBefore pub proof messages pub.ops (fusedIndex pub.layout 3)))
        (some x) := by
  obtain ⟨⟨reply, answered, decodedAt⟩, presented⟩ := presents.level₂
  have challenged := Extract.challengeAt_of_reply _ _ _ answered decodedAt
  subst tower
  obtain ⟨-, -, _, _, _, _, _, _, gameEq, -⟩ := linked
  obtain ⟨-, -, frames⟩ := frames20 layout scripted
  have tensorBefore := frames.1
  obtain ⟨-, -, f2, f3, f4, f5, -⟩ := indices20 layout scripted
  have found : pub.setupFor (fold20 2).setupCount = some (setup20 pub 2) := by
    obtain ⟨setup, found⟩ := Option.isSome_iff_exists.mp expanded
    simp [setup20, found]
  obtain ⟨ranT, steppedT⟩ := before_step pub proof messages tensorBefore.at_
    (by rw [f2]; decide) before
  have link := (key20_traceLink_before (E := E) pub proof aligned List.mem_cons_self
    tensorBefore.rounds ranT steppedT tensorBefore.after tensorBefore.kept
    (layout20_fold zMax rice 2) (by decide) (setup20 pub 2)).2.2.2.2
  obtain ⟨quarter, fits, rowsFit, runs⟩ := key20_trace_before_of (E := E) pub proof layout
    scripted aligned before rfl found (challengeTrace20 pub proof messages 2 Key20.level₂)
    (fun _ => rfl) 0
  have before' : traceStatementFor (E := E) Key20.level₂.opening quarter (setup20 pub 2)
      (traceValuesIn (stateBefore pub proof messages pub.ops (fusedIndex pub.layout 2)) 2 (fold20 2)) (dataIn pub (stateBefore pub proof messages pub.ops (fusedIndex pub.layout 2)) 2) .witness
      (rowsIn (stateBefore pub proof messages pub.ops (fusedIndex pub.layout 2)) 2 (fold20 2)) (claimsIn (stateBefore pub proof messages pub.ops (fusedIndex pub.layout 2)) 2 (fold20 2)) =
      ((point.wire 120).data.claim₂).statement
        (Key.ofSetup (flatOf (setup20 pub 2)) Key20.level₂.shape)
        (OpeningKey.ofSetup (flatOf (setup20 pub 2)) Key20.level₂.opening) := by
    rw [presents.reads.claim₂]
    exact link
  rw [before'] at runs
  rw [path20_trace pub proof messages 2 (by decide), receiver20_lt pub (by decide)] at presented
  have ran := (Extract.Tier.game_eq_pure_iff _ 0 point presented _).mp gameEq
  unfold Tier.challenge at ran
  rw [challenged] at ran
  exact ⟨_, _, runs, by rw [tensoredPathIn_eq]; exact ran⟩

/-- Theorem B "←" at the level 3: on a transcript that presents the bytes of a proof, when the
link `Tier.TraceLinked` of the level 3 holds and the run before its `fused.end` has no stop, the
game of its fold runs on the path of the code to the result of `foldRunIn`, and with the tensor
reduction of its claim it runs to `some`. -/
theorem model20_level3 {zMax rice : ℕ} (layout : pub.layout = layout20 zMax rice)
    (scripted : script pub.layout pub.plan = .ok pub.ops)
    (aligned : ReadsAligned messages pub.ops.zipIdx.toList 0)
    (expanded : (pub.setupFor (fold20 3).setupCount).isSome)
    (tower : o = tower20 pub block block₁ block₂ block₃ block₄ block₅ blockL total 472)
    {point : Prefix (Key20.Wire E) (Option Coord → CodeSlot E) total}
    (presents : PresentsAt20 o pub proof messages tLast eLast point)
    (linked : Tier.TraceLinked (third) (fourth) (And.left (And.right (And.right Key20.quarters))) o.commit₄ point)
    (before : (runFrom pub proof messages (pub.ops.zipIdx.toList.take (fusedIndex pub.layout 3))
      (channelStart pub)).1 = .ok ()) :
    ∃ (g : Game (Option (List E × E))) (x : List E × E × E),
      g.Runs (foldPathIn E pub (fused20 pub proof messages 3) 3 (fold20 3) (setup20 pub 3)
          (fold20 3).vars (fold20 3).batch)
        ((foldRunIn pub (fused20 pub proof messages 3) 3 (fold20 3)).map fun o =>
          (o.1.map Ext.into, o.2.into)) ∧
      (g.bind fun
        | none => .pure none
        | some (r2, v) => tensorCheck Key20.level₃.tail 0 r2 v).Runs
        (foldPathIn E pub (fused20 pub proof messages 3) 3 (fold20 3) (setup20 pub 3)
            (fold20 3).vars (fold20 3).batch ++
          tensoredPathIn E Key20.level₃.tail 3 (stateBefore pub proof messages pub.ops (fusedIndex pub.layout 4)))
        (some x) := by
  obtain ⟨⟨reply, answered, decodedAt⟩, presented⟩ := presents.level₃
  have challenged := Extract.challengeAt_of_reply _ _ _ answered decodedAt
  subst tower
  obtain ⟨-, -, _, _, _, _, _, _, gameEq, -⟩ := linked
  obtain ⟨-, -, frames⟩ := frames20 layout scripted
  have tensorBefore := frames.2.1
  obtain ⟨-, -, f2, f3, f4, f5, -⟩ := indices20 layout scripted
  have found : pub.setupFor (fold20 3).setupCount = some (setup20 pub 3) := by
    obtain ⟨setup, found⟩ := Option.isSome_iff_exists.mp expanded
    simp [setup20, found]
  obtain ⟨ranT, steppedT⟩ := before_step pub proof messages tensorBefore.at_
    (by rw [f3]; decide) before
  have link := (key20_traceLink_before (E := E) pub proof aligned (List.mem_cons_of_mem _ List.mem_cons_self)
    tensorBefore.rounds ranT steppedT tensorBefore.after tensorBefore.kept
    (layout20_fold zMax rice 3) (by decide) (setup20 pub 3)).2.2.2.2
  obtain ⟨quarter, fits, rowsFit, runs⟩ := key20_traceSq_before_of (E := E) 3 Key20.level₃
    List.mem_cons_self pub proof layout scripted aligned before rfl found
    (challengeTrace20 pub proof messages 3 Key20.level₃) (fun _ => rfl) 0
  have before' : traceStatementFor (E := E) Key20.level₃.opening quarter (setup20 pub 3)
      (traceValuesIn (stateBefore pub proof messages pub.ops (fusedIndex pub.layout ((3 : Fin 6) : ℕ))) ((3 : Fin 6) : ℕ) (fold20 3)) (dataIn pub (stateBefore pub proof messages pub.ops (fusedIndex pub.layout ((3 : Fin 6) : ℕ))) ((3 : Fin 6) : ℕ)) .witness
      (rowsIn (stateBefore pub proof messages pub.ops (fusedIndex pub.layout ((3 : Fin 6) : ℕ))) ((3 : Fin 6) : ℕ) (fold20 3)) (claimsIn (stateBefore pub proof messages pub.ops (fusedIndex pub.layout ((3 : Fin 6) : ℕ))) ((3 : Fin 6) : ℕ) (fold20 3)) =
      ((point.wire 203).data.claim₃).statement
        (Key.ofSetup (flatOf (setup20 pub 3)) Key20.level₃.shape)
        (OpeningKey.ofSetup (flatOf (setup20 pub 3)) Key20.level₃.opening) := by
    rw [presents.reads.claim₃]
    exact link
  rw [before'] at runs
  rw [path20_trace pub proof messages 3 (by decide), receiver20_lt pub (by decide)] at presented
  have ran := (Extract.Tier.game_eq_pure_iff _ 0 point presented _).mp gameEq
  unfold Tier.challenge at ran
  rw [challenged] at ran
  exact ⟨_, _, runs, by rw [tensoredPathIn_eq]; exact ran⟩

/-- Theorem B "←" at the level 4: on a transcript that presents the bytes of a proof, when the
link `Tier.TraceLinked` of the level 4 holds and the run before its `fused.end` has no stop, the
game of its fold runs on the path of the code to the result of `foldRunIn`, and with the tensor
reduction of its claim it runs to `some`. -/
theorem model20_level4 {zMax rice : ℕ} (layout : pub.layout = layout20 zMax rice)
    (scripted : script pub.layout pub.plan = .ok pub.ops)
    (aligned : ReadsAligned messages pub.ops.zipIdx.toList 0)
    (expanded : (pub.setupFor (fold20 4).setupCount).isSome)
    (tower : o = tower20 pub block block₁ block₂ block₃ block₄ block₅ blockL total 472)
    {point : Prefix (Key20.Wire E) (Option Coord → CodeSlot E) total}
    (presents : PresentsAt20 o pub proof messages tLast eLast point)
    (linked : Tier.TraceLinked (fourth) (fifth) (And.left (And.right (And.right (And.right Key20.quarters)))) o.commit₅ point)
    (before : (runFrom pub proof messages (pub.ops.zipIdx.toList.take (fusedIndex pub.layout 4))
      (channelStart pub)).1 = .ok ()) :
    ∃ (g : Game (Option (List E × E))) (x : List E × E × E),
      g.Runs (foldPathIn E pub (fused20 pub proof messages 4) 4 (fold20 4) (setup20 pub 4)
          (fold20 4).vars (fold20 4).batch)
        ((foldRunIn pub (fused20 pub proof messages 4) 4 (fold20 4)).map fun o =>
          (o.1.map Ext.into, o.2.into)) ∧
      (g.bind fun
        | none => .pure none
        | some (r2, v) => tensorCheck Key20.level₄.tail 0 r2 v).Runs
        (foldPathIn E pub (fused20 pub proof messages 4) 4 (fold20 4) (setup20 pub 4)
            (fold20 4).vars (fold20 4).batch ++
          tensoredPathIn E Key20.level₄.tail 4 (stateBefore pub proof messages pub.ops (fusedIndex pub.layout 5)))
        (some x) := by
  obtain ⟨started, -⟩ := fourth20_numbers o
  obtain ⟨⟨reply, answered, decodedAt⟩, presented⟩ := presents.level₄
  have challenged := Extract.challengeAt_of_reply _ _ _ answered decodedAt
  subst tower
  obtain ⟨-, -, _, _, _, _, _, _, gameEq, -⟩ := linked
  obtain ⟨-, -, frames⟩ := frames20 layout scripted
  have tensorBefore := frames.2.2.1
  obtain ⟨-, -, f2, f3, f4, f5, -⟩ := indices20 layout scripted
  have found : pub.setupFor (fold20 4).setupCount = some (setup20 pub 4) := by
    obtain ⟨setup, found⟩ := Option.isSome_iff_exists.mp expanded
    simp [setup20, found]
  obtain ⟨ranT, steppedT⟩ := before_step pub proof messages tensorBefore.at_
    (by rw [f4]; decide) before
  have link := (key20_traceLink_before (E := E) pub proof aligned (List.mem_cons_of_mem _ (List.mem_cons_of_mem _ List.mem_cons_self))
    tensorBefore.rounds ranT steppedT tensorBefore.after tensorBefore.kept
    (layout20_fold zMax rice 4) (by decide) (setup20 pub 4)).2.2.2.2
  obtain ⟨quarter, fits, rowsFit, runs⟩ := key20_traceSq_before_of (E := E) 4 Key20.level₄
    (List.mem_cons_of_mem _ List.mem_cons_self) pub proof layout scripted aligned before rfl found
    (challengeTrace20 pub proof messages 4 Key20.level₄) (fun _ => rfl) 0
  have before' : traceStatementFor (E := E) Key20.level₄.opening quarter (setup20 pub 4)
      (traceValuesIn (stateBefore pub proof messages pub.ops (fusedIndex pub.layout ((4 : Fin 6) : ℕ))) ((4 : Fin 6) : ℕ) (fold20 4)) (dataIn pub (stateBefore pub proof messages pub.ops (fusedIndex pub.layout ((4 : Fin 6) : ℕ))) ((4 : Fin 6) : ℕ)) .witness
      (rowsIn (stateBefore pub proof messages pub.ops (fusedIndex pub.layout ((4 : Fin 6) : ℕ))) ((4 : Fin 6) : ℕ) (fold20 4)) (claimsIn (stateBefore pub proof messages pub.ops (fusedIndex pub.layout ((4 : Fin 6) : ℕ))) ((4 : Fin 6) : ℕ) (fold20 4)) =
      ((point.wire 284).data.claim₄).statement
        (Key.ofSetup (flatOf (setup20 pub 4)) Key20.level₄.shape)
        (OpeningKey.ofSetup (flatOf (setup20 pub 4)) Key20.level₄.opening) := by
    rw [presents.reads.claim₄]
    exact link
  rw [before'] at runs
  rw [path20_trace pub proof messages 4 (by decide), receiver20_lt pub (by decide)] at presented
  have ran := (Extract.Tier.game_eq_pure_iff _ 0 point presented _).mp gameEq
  unfold Tier.challenge at ran
  rw [challenged, started] at ran
  exact ⟨_, _, runs, by rw [tensoredPathIn_eq]; exact ran⟩

/-- Theorem B "←" at the level 5: on a transcript that presents the bytes of a proof, when the
link `Terminal.Linked` of the level 5 holds and the run before its `fused.end` has no stop, the
game of its fold runs on the path of the code to the result of `foldRunIn`, and with the tensor
reduction of its claim it runs to `some`. -/
theorem model20_level5 {zMax rice : ℕ} (layout : pub.layout = layout20 zMax rice)
    (scripted : script pub.layout pub.plan = .ok pub.ops)
    (aligned : ReadsAligned messages pub.ops.zipIdx.toList 0)
    (expanded : (pub.setupFor (fold20 5).setupCount).isSome)
    (tower : o = tower20 pub block block₁ block₂ block₃ block₄ block₅ blockL total 472)
    {point : Prefix (Key20.Wire E) (Option Coord → CodeSlot E) total}
    (presents : PresentsAt20 o pub proof messages tLast eLast point)
    (linked : Extract.Terminal.Linked (fifth) o.terminal
      (And.right (And.right (And.right (And.right Key20.quarters)))) o.stateOf point)
    (before : (runFrom pub proof messages (pub.ops.zipIdx.toList.take (fusedIndex pub.layout 5))
      (channelStart pub)).1 = .ok ()) :
    ∃ (g : Game (Option (List E × E))) (x : List E × E × E),
      g.Runs (foldPathIn E pub (fused20 pub proof messages 5) 5 (fold20 5) (setup20 pub 5)
          (fold20 5).vars (fold20 5).batch)
        ((foldRunIn pub (fused20 pub proof messages 5) 5 (fold20 5)).map fun o =>
          (o.1.map Ext.into, o.2.into)) ∧
      (g.bind fun
        | none => .pure none
        | some (r2, v) => tensorCheck Key20.level₅.tail 0 r2 v).Runs
        (foldPathIn E pub (fused20 pub proof messages 5) 5 (fold20 5) (setup20 pub 5)
            (fold20 5).vars (fold20 5).batch ++
          tensoredPathIn E Key20.level₅.tail 5 (stateBefore pub proof messages pub.ops (pub.ops.size - 1)))
        (some x) := by
  obtain ⟨started, -⟩ := fifth20_numbers o
  obtain ⟨⟨reply, answered, decodedAt⟩, presented⟩ := presents.level₅
  have challenged := Extract.challengeAt_of_reply _ _ _ answered decodedAt
  subst tower
  obtain ⟨-, -, _, _, _, _, _, gameEq, -⟩ := linked
  obtain ⟨-, -, frames⟩ := frames20 layout scripted
  have tensorBefore := frames.2.2.2.1
  obtain ⟨-, -, f2, f3, f4, f5, -⟩ := indices20 layout scripted
  have found : pub.setupFor (fold20 5).setupCount = some (setup20 pub 5) := by
    obtain ⟨setup, found⟩ := Option.isSome_iff_exists.mp expanded
    simp [setup20, found]
  obtain ⟨ranT, steppedT⟩ := before_step pub proof messages tensorBefore.at_
    (by rw [f5]; decide) before
  have link := (key20_traceLink_before (E := E) pub proof aligned (List.mem_cons_of_mem _ (List.mem_cons_of_mem _ (List.mem_cons_of_mem _ List.mem_cons_self)))
    tensorBefore.rounds ranT steppedT tensorBefore.after tensorBefore.kept
    (layout20_fold zMax rice 5) (by decide) (setup20 pub 5)).2.2.2.2
  obtain ⟨quarter, fits, rowsFit, runs⟩ := key20_traceSq_before_of (E := E) 5 Key20.level₅
    (List.mem_cons_of_mem _ (List.mem_cons_of_mem _ List.mem_cons_self)) pub proof layout scripted aligned before rfl found
    (challengeTrace20 pub proof messages 5 Key20.level₅) (fun _ => rfl) 0
  have before' : traceStatementFor (E := E) Key20.level₅.opening quarter (setup20 pub 5)
      (traceValuesIn (stateBefore pub proof messages pub.ops (fusedIndex pub.layout ((5 : Fin 6) : ℕ))) ((5 : Fin 6) : ℕ) (fold20 5)) (dataIn pub (stateBefore pub proof messages pub.ops (fusedIndex pub.layout ((5 : Fin 6) : ℕ))) ((5 : Fin 6) : ℕ)) .witness
      (rowsIn (stateBefore pub proof messages pub.ops (fusedIndex pub.layout ((5 : Fin 6) : ℕ))) ((5 : Fin 6) : ℕ) (fold20 5)) (claimsIn (stateBefore pub proof messages pub.ops (fusedIndex pub.layout ((5 : Fin 6) : ℕ))) ((5 : Fin 6) : ℕ) (fold20 5)) =
      ((point.wire 378).data.claim₅).statement
        (Key.ofSetup (flatOf (setup20 pub 5)) Key20.level₅.shape)
        (OpeningKey.ofSetup (flatOf (setup20 pub 5)) Key20.level₅.opening) := by
    rw [presents.reads.claim₅]
    exact link
  rw [before'] at runs
  rw [path20_trace pub proof messages 5 (by decide)] at presented
  have receiver : receiver20 pub ((5 : Fin 6) : ℕ) = pub.ops.size - 1 := rfl
  rw [receiver] at presented
  have ran := (Extract.Tier.game_eq_pure_iff _ 0 point presented _).mp gameEq
  unfold Tier.challenge at ran
  rw [challenged, started] at ran
  exact ⟨_, _, runs, by rw [tensoredPathIn_eq]; exact ran⟩

/-- The draws of the path of the code of the level 5: on a transcript that presents the bytes of a
proof, when the link `Terminal.Linked` of the level 5 holds, the path of its fold and of the
tensor reduction of its claim has the `foldDrawsSq` draws of `foldCheckSq` or more, because the
game of the level runs on it to `some`. -/
theorem model20_draws5
    (tower : o = tower20 pub block block₁ block₂ block₃ block₄ block₅ blockL total 472)
    {point : Prefix (Key20.Wire E) (Option Coord → CodeSlot E) total}
    (presents : PresentsAt20 o pub proof messages tLast eLast point)
    (linked : Extract.Terminal.Linked (fifth) o.terminal
      (And.right (And.right (And.right (And.right Key20.quarters)))) o.stateOf point) :
    Extract.foldDrawsSq (fold20 5).vars Key20.level₅.shape.certifiedBase
        (Key20.level₅.opening.compressed Key20.level₅.fold) ≤
      Extract.countDraws (foldPathIn E pub (fused20 pub proof messages 5) 5 (fold20 5)
          (setup20 pub 5) (fold20 5).vars (fold20 5).batch ++
        tensoredPathIn E Key20.level₅.tail 5
          (stateBefore pub proof messages pub.ops (pub.ops.size - 1))) := by
  obtain ⟨started, -⟩ := fifth20_numbers o
  obtain ⟨⟨reply, answered, decodedAt⟩, presented⟩ := presents.level₅
  have challenged := Extract.challengeAt_of_reply _ _ _ answered decodedAt
  subst tower
  obtain ⟨-, -, _, _, _, _, _, gameEq, -⟩ := linked
  rw [path20_trace pub proof messages 5 (by decide)] at presented
  have receiver : receiver20 pub ((5 : Fin 6) : ℕ) = pub.ops.size - 1 := rfl
  rw [receiver] at presented
  have ran := (Extract.Tier.game_eq_pure_iff _ 0 point presented _).mp gameEq
  unfold Tier.challenge at ran
  rw [challenged, started] at ran
  rw [tensoredPathIn_eq]
  have drawn := Extract.foldCheckSq_drawn ran rfl
  exact drawn

/-- Theorem B "←" at the root fold: on a transcript that presents the bytes of a proof, when the
link `Tier.Linked` of the root fold holds and the run before its `fused.end` has no stop, the
step of this `fused.end` has no stop. -/
theorem level20_ok₀ {zMax rice : ℕ} (layout : pub.layout = layout20 zMax rice)
    (scripted : script pub.layout pub.plan = .ok pub.ops)
    (aligned : ReadsAligned messages pub.ops.zipIdx.toList 0)
    (points : ∀ g : Fin 3, ((pub.groups[g.val]?.map (·.point)).getD #[]).size = ![20, 16, 16] g)
    (values : (pub.groups.flatMap (·.value)).size = 3)
    (expanded : (pub.setupFor (fold20 0).setupCount).isSome)
    (tower : o = tower20 pub block block₁ block₂ block₃ block₄ block₅ blockL total 472)
    {point : Prefix (Key20.Wire E) (Option Coord → CodeSlot E) total}
    (presents : PresentsAt20 o pub proof messages tLast eLast point)
    (linked : Tier.Linked (zeroth) (first) o.commit₁ point) :
    (runFrom pub proof messages (pub.ops.zipIdx.toList.take (fusedIndex pub.layout 0)) (channelStart pub)).1 =
        .ok () →
      StepOk pub proof messages (fusedIndex pub.layout 0) := by
  intro before op found
  obtain ⟨at', ⟨r, a, e, shape'⟩, -, -⟩ := levels20 layout scripted 0
  have at_ : pub.ops[fusedIndex pub.layout 0]? =
      some (.check ⟨0, "fused.end", none⟩ (.fused 0 (fold20 0))) := at'
  have shape : foldShape (pub.ops.zipIdx.toList.take (fusedIndex pub.layout 0)) 0 (fold20 0) r a
      e = true := shape'
  obtain ⟨g, x, fold, runs⟩ := model20_level0 o pub proof messages tLast eLast layout scripted
    aligned points values expanded tower presents linked before
  rw [at_] at found
  cases found
  exact fused_ok_of_run pub proof messages shape before fold runs

/-- Theorem B "←" at the level 1: on a transcript that presents the bytes of a proof, when the
link of the level 1 of `Key20.Tower.accepts` holds, the steps of its `fused.end`, of the
`tensor.opening` at 321 and of the `tensor.end` at 385 of its claim have no stop when the run
before them has none. -/
theorem level20_ok₁ {zMax rice : ℕ} (layout : pub.layout = layout20 zMax rice)
    (scripted : script pub.layout pub.plan = .ok pub.ops)
    (aligned : ReadsAligned messages pub.ops.zipIdx.toList 0)
    (expanded : (pub.setupFor (fold20 1).setupCount).isSome)
    (tower : o = tower20 pub block block₁ block₂ block₃ block₄ block₅ blockL total 472)
    {point : Prefix (Key20.Wire E) (Option Coord → CodeSlot E) total}
    (presents : PresentsAt20 o pub proof messages tLast eLast point)
    (linked : Tier.TraceLinked (first) (second) (And.left Key20.quarters) o.commit₂ point) :
    ((runFrom pub proof messages (pub.ops.zipIdx.toList.take (fusedIndex pub.layout 1)) (channelStart pub)).1 =
        .ok () →
      StepOk pub proof messages (fusedIndex pub.layout 1)) ∧
    ((runFrom pub proof messages (pub.ops.zipIdx.toList.take (321)) (channelStart pub)).1 =
        .ok () →
      StepOk pub proof messages 321) ∧
    ((runFrom pub proof messages (pub.ops.zipIdx.toList.take (385)) (channelStart pub)).1 =
        .ok () →
      StepOk pub proof messages 385) := by
  have erased : pub.ops.toList.map Op.erase = skeleton (layout20 0 0) := by
    rw [script_erase (layout ▸ scripted), skeleton20]
  obtain ⟨-, f1, f2, f3, f4, f5, size⟩ := indices20 layout scripted
  obtain ⟨-, -, frames⟩ := frames20 layout scripted
  have frame := frames.1
  obtain ⟨openingAt, held⟩ := opening_of erased skeleton20_openings.1
  rw [← f1] at held
  obtain ⟨at', ⟨r, a, e, shape'⟩, -, -⟩ := levels20 layout scripted 1
  have at_ : pub.ops[fusedIndex pub.layout 1]? =
      some (.check ⟨1, "fused.end", none⟩ (.fused 1 (fold20 1))) := at'
  have shape : foldShape (pub.ops.zipIdx.toList.take (fusedIndex pub.layout 1)) 1 (fold20 1) r a
      e = true := shape'
  obtain ⟨fusedOk, openingOk, tensorOk⟩ := level_ok_of_model pub proof messages aligned at_ shape
    frame openingAt held (by rw [f1]; decide) (by decide) (by rw [f2]; decide)
    (model20_level1 o pub proof messages tLast eLast layout scripted aligned expanded tower presents
      linked)
  refine ⟨fun before op found => ?_, fun before op found => ?_, fun before op found => ?_⟩
  · rw [at_] at found
    cases found
    exact fusedOk before
  · rw [openingAt] at found
    cases found
    exact openingOk before
  · rw [frame.at_] at found
    cases found
    exact tensorOk before

/-- Theorem B "←" at the level 2: on a transcript that presents the bytes of a proof, when the
link of the level 2 of `Key20.Tower.accepts` holds, the steps of its `fused.end`, of the
`tensor.opening` at 591 and of the `tensor.end` at 649 of its claim have no stop when the run
before them has none. -/
theorem level20_ok₂ {zMax rice : ℕ} (layout : pub.layout = layout20 zMax rice)
    (scripted : script pub.layout pub.plan = .ok pub.ops)
    (aligned : ReadsAligned messages pub.ops.zipIdx.toList 0)
    (expanded : (pub.setupFor (fold20 2).setupCount).isSome)
    (tower : o = tower20 pub block block₁ block₂ block₃ block₄ block₅ blockL total 472)
    {point : Prefix (Key20.Wire E) (Option Coord → CodeSlot E) total}
    (presents : PresentsAt20 o pub proof messages tLast eLast point)
    (linked : Tier.TraceLinked (second) (third) (And.left (And.right Key20.quarters)) o.commit₃ point) :
    ((runFrom pub proof messages (pub.ops.zipIdx.toList.take (fusedIndex pub.layout 2)) (channelStart pub)).1 =
        .ok () →
      StepOk pub proof messages (fusedIndex pub.layout 2)) ∧
    ((runFrom pub proof messages (pub.ops.zipIdx.toList.take (591)) (channelStart pub)).1 =
        .ok () →
      StepOk pub proof messages 591) ∧
    ((runFrom pub proof messages (pub.ops.zipIdx.toList.take (649)) (channelStart pub)).1 =
        .ok () →
      StepOk pub proof messages 649) := by
  have erased : pub.ops.toList.map Op.erase = skeleton (layout20 0 0) := by
    rw [script_erase (layout ▸ scripted), skeleton20]
  obtain ⟨-, f1, f2, f3, f4, f5, size⟩ := indices20 layout scripted
  obtain ⟨-, -, frames⟩ := frames20 layout scripted
  have frame := frames.2.1
  obtain ⟨openingAt, held⟩ := opening_of erased skeleton20_openings.2.1
  rw [← f2] at held
  obtain ⟨at', ⟨r, a, e, shape'⟩, -, -⟩ := levels20 layout scripted 2
  have at_ : pub.ops[fusedIndex pub.layout 2]? =
      some (.check ⟨2, "fused.end", none⟩ (.fused 2 (fold20 2))) := at'
  have shape : foldShape (pub.ops.zipIdx.toList.take (fusedIndex pub.layout 2)) 2 (fold20 2) r a
      e = true := shape'
  obtain ⟨fusedOk, openingOk, tensorOk⟩ := level_ok_of_model pub proof messages aligned at_ shape
    frame openingAt held (by rw [f2]; decide) (by decide) (by rw [f3]; decide)
    (model20_level2 o pub proof messages tLast eLast layout scripted aligned expanded tower presents
      linked)
  refine ⟨fun before op found => ?_, fun before op found => ?_, fun before op found => ?_⟩
  · rw [at_] at found
    cases found
    exact fusedOk before
  · rw [openingAt] at found
    cases found
    exact openingOk before
  · rw [frame.at_] at found
    cases found
    exact tensorOk before

/-- Theorem B "←" at the level 3: on a transcript that presents the bytes of a proof, when the
link of the level 3 of `Key20.Tower.accepts` holds, the steps of its `fused.end`, of the
`tensor.opening` at 853 and of the `tensor.end` at 908 of its claim have no stop when the run
before them has none. -/
theorem level20_ok₃ {zMax rice : ℕ} (layout : pub.layout = layout20 zMax rice)
    (scripted : script pub.layout pub.plan = .ok pub.ops)
    (aligned : ReadsAligned messages pub.ops.zipIdx.toList 0)
    (expanded : (pub.setupFor (fold20 3).setupCount).isSome)
    (tower : o = tower20 pub block block₁ block₂ block₃ block₄ block₅ blockL total 472)
    {point : Prefix (Key20.Wire E) (Option Coord → CodeSlot E) total}
    (presents : PresentsAt20 o pub proof messages tLast eLast point)
    (linked : Tier.TraceLinked (third) (fourth) (And.left (And.right (And.right Key20.quarters)))
      o.commit₄ point) :
    ((runFrom pub proof messages (pub.ops.zipIdx.toList.take (fusedIndex pub.layout 3)) (channelStart pub)).1 =
        .ok () →
      StepOk pub proof messages (fusedIndex pub.layout 3)) ∧
    ((runFrom pub proof messages (pub.ops.zipIdx.toList.take (853)) (channelStart pub)).1 =
        .ok () →
      StepOk pub proof messages 853) ∧
    ((runFrom pub proof messages (pub.ops.zipIdx.toList.take (908)) (channelStart pub)).1 =
        .ok () →
      StepOk pub proof messages 908) := by
  have erased : pub.ops.toList.map Op.erase = skeleton (layout20 0 0) := by
    rw [script_erase (layout ▸ scripted), skeleton20]
  obtain ⟨-, f1, f2, f3, f4, f5, size⟩ := indices20 layout scripted
  obtain ⟨-, -, frames⟩ := frames20 layout scripted
  have frame := frames.2.2.1
  obtain ⟨openingAt, held⟩ := opening_of erased skeleton20_openings.2.2.1
  rw [← f3] at held
  obtain ⟨at', ⟨r, a, e, shape'⟩, -, -⟩ := levels20 layout scripted 3
  have at_ : pub.ops[fusedIndex pub.layout 3]? =
      some (.check ⟨3, "fused.end", none⟩ (.fused 3 (fold20 3))) := at'
  have shape : foldShape (pub.ops.zipIdx.toList.take (fusedIndex pub.layout 3)) 3 (fold20 3) r a
      e = true := shape'
  obtain ⟨fusedOk, openingOk, tensorOk⟩ := level_ok_of_model pub proof messages aligned at_ shape
    frame openingAt held (by rw [f3]; decide) (by decide) (by rw [f4]; decide)
    (model20_level3 o pub proof messages tLast eLast layout scripted aligned expanded tower presents
      linked)
  refine ⟨fun before op found => ?_, fun before op found => ?_, fun before op found => ?_⟩
  · rw [at_] at found
    cases found
    exact fusedOk before
  · rw [openingAt] at found
    cases found
    exact openingOk before
  · rw [frame.at_] at found
    cases found
    exact tensorOk before

/-- Theorem B "←" at the level 4: on a transcript that presents the bytes of a proof, when the
link of the level 4 of `Key20.Tower.accepts` holds, the steps of its `fused.end`, of the
`tensor.opening` at 1158 and of the `tensor.end` at 1210 of its claim have no stop when the run
before them has none. -/
theorem level20_ok₄ {zMax rice : ℕ} (layout : pub.layout = layout20 zMax rice)
    (scripted : script pub.layout pub.plan = .ok pub.ops)
    (aligned : ReadsAligned messages pub.ops.zipIdx.toList 0)
    (expanded : (pub.setupFor (fold20 4).setupCount).isSome)
    (tower : o = tower20 pub block block₁ block₂ block₃ block₄ block₅ blockL total 472)
    {point : Prefix (Key20.Wire E) (Option Coord → CodeSlot E) total}
    (presents : PresentsAt20 o pub proof messages tLast eLast point)
    (linked : Tier.TraceLinked (fourth) (fifth)
      (And.left (And.right (And.right (And.right Key20.quarters)))) o.commit₅ point) :
    ((runFrom pub proof messages (pub.ops.zipIdx.toList.take (fusedIndex pub.layout 4)) (channelStart pub)).1 =
        .ok () →
      StepOk pub proof messages (fusedIndex pub.layout 4)) ∧
    ((runFrom pub proof messages (pub.ops.zipIdx.toList.take (1158)) (channelStart pub)).1 =
        .ok () →
      StepOk pub proof messages 1158) ∧
    ((runFrom pub proof messages (pub.ops.zipIdx.toList.take (1210)) (channelStart pub)).1 =
        .ok () →
      StepOk pub proof messages 1210) := by
  have erased : pub.ops.toList.map Op.erase = skeleton (layout20 0 0) := by
    rw [script_erase (layout ▸ scripted), skeleton20]
  obtain ⟨-, f1, f2, f3, f4, f5, size⟩ := indices20 layout scripted
  obtain ⟨-, -, frames⟩ := frames20 layout scripted
  have frame := frames.2.2.2.1
  obtain ⟨openingAt, held⟩ := opening_of erased skeleton20_openings.2.2.2.1
  rw [← f4] at held
  obtain ⟨at', ⟨r, a, e, shape'⟩, -, -⟩ := levels20 layout scripted 4
  have at_ : pub.ops[fusedIndex pub.layout 4]? =
      some (.check ⟨4, "fused.end", none⟩ (.fused 4 (fold20 4))) := at'
  have shape : foldShape (pub.ops.zipIdx.toList.take (fusedIndex pub.layout 4)) 4 (fold20 4) r a
      e = true := shape'
  obtain ⟨fusedOk, openingOk, tensorOk⟩ := level_ok_of_model pub proof messages aligned at_ shape
    frame openingAt held (by rw [f4]; decide) (by decide) (by rw [f5]; decide)
    (model20_level4 o pub proof messages tLast eLast layout scripted aligned expanded tower presents
      linked)
  refine ⟨fun before op found => ?_, fun before op found => ?_, fun before op found => ?_⟩
  · rw [at_] at found
    cases found
    exact fusedOk before
  · rw [openingAt] at found
    cases found
    exact openingOk before
  · rw [frame.at_] at found
    cases found
    exact tensorOk before

/-- Theorem B "←" at the level 5: on a transcript that presents the bytes of a proof, when the
link of the level 5 of `Key20.Tower.accepts` holds, the steps of its `fused.end`, of the
`tensor.opening` at 1461 and of the `tensor.end` at 1513 of its claim have no stop when the run
before them has none. -/
theorem level20_ok₅ {zMax rice : ℕ} (layout : pub.layout = layout20 zMax rice)
    (scripted : script pub.layout pub.plan = .ok pub.ops)
    (aligned : ReadsAligned messages pub.ops.zipIdx.toList 0)
    (expanded : (pub.setupFor (fold20 5).setupCount).isSome)
    (tower : o = tower20 pub block block₁ block₂ block₃ block₄ block₅ blockL total 472)
    {point : Prefix (Key20.Wire E) (Option Coord → CodeSlot E) total}
    (presents : PresentsAt20 o pub proof messages tLast eLast point)
    (linked : Extract.Terminal.Linked (fifth) o.terminal
      (And.right (And.right (And.right (And.right Key20.quarters)))) o.stateOf point) :
    ((runFrom pub proof messages (pub.ops.zipIdx.toList.take (fusedIndex pub.layout 5)) (channelStart pub)).1 =
        .ok () →
      StepOk pub proof messages (fusedIndex pub.layout 5)) ∧
    ((runFrom pub proof messages (pub.ops.zipIdx.toList.take (1461)) (channelStart pub)).1 =
        .ok () →
      StepOk pub proof messages 1461) ∧
    ((runFrom pub proof messages (pub.ops.zipIdx.toList.take (1513)) (channelStart pub)).1 =
        .ok () →
      StepOk pub proof messages 1513) := by
  have erased : pub.ops.toList.map Op.erase = skeleton (layout20 0 0) := by
    rw [script_erase (layout ▸ scripted), skeleton20]
  obtain ⟨-, f1, f2, f3, f4, f5, size⟩ := indices20 layout scripted
  obtain ⟨-, -, frames⟩ := frames20 layout scripted
  have frame := frames.2.2.2.2.1
  obtain ⟨openingAt, held⟩ := opening_of erased skeleton20_openings.2.2.2.2
  rw [← f5] at held
  obtain ⟨at', ⟨r, a, e, shape'⟩, -, -⟩ := levels20 layout scripted 5
  have at_ : pub.ops[fusedIndex pub.layout 5]? =
      some (.check ⟨5, "fused.end", none⟩ (.fused 5 (fold20 5))) := at'
  have shape : foldShape (pub.ops.zipIdx.toList.take (fusedIndex pub.layout 5)) 5 (fold20 5) r a
      e = true := shape'
  obtain ⟨fusedOk, openingOk, tensorOk⟩ := level_ok_of_model pub proof messages aligned at_ shape
    frame openingAt held (by rw [f5]; decide) (by decide) (by rw [size]; decide)
    (model20_level5 o pub proof messages tLast eLast layout scripted aligned expanded tower presents
      linked)
  refine ⟨fun before op found => ?_, fun before op found => ?_, fun before op found => ?_⟩
  · rw [at_] at found
    cases found
    exact fusedOk before
  · rw [openingAt] at found
    cases found
    exact openingOk before
  · rw [frame.at_] at found
    cases found
    exact tensorOk before

end NightstreamFPrime.Spec.Compression.Akita.Replay
