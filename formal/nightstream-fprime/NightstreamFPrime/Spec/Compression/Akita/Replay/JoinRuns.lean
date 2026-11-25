import NightstreamFPrime.Spec.Compression.Akita.Replay.JoinCount
import NightstreamFPrime.Spec.Compression.Akita.Extract.PresentChecks

/-! The runs of the checks of the levels of `key20.aks` on the paths of the code from each wire
message with the data `note20`, for theorem A of the join (Akita revision `d50f700`; ePrint
2026/1983, revision of 2026-11-23, Figure 4, Figure 6 and Section 8.2;
`docs/akita-work/w16/design.md` section 2).

- `energy20`: on the Euclidean route the energy of an accepted proof is at most `S_max` of the
  level (`foldRunIn_of_run`, `treeRun_normRun_none`).
- `fits_path20`: the path `path20 ℓ` of the code fits the cells of `Key20.Wire` from the message 0.
- `Tier.exists_block20`: `Extract.Tier.exists_block` on `Key20.Wire` for the data `note`, the
  nonce 0 and the work slots `work`: a block with these fields whose transcripts show the
  challenges and the path of the level.
- `runs20_zeroth` .. `runs20_fifth`: for each wire message with the data `note20`, the checks of the
  level of `tower20` for the fold challenges of the code run on `path20 ℓ` (the games of
  `verifyProof_key20_*_of` and `verifyProof_tensored`, as in `presents20_level*`).
- `checks20_zeroth` .. `checks20_fifth`: the checks of each level are drawable from the answers
  `Coord → CodeSlot E` for `22 ≤ |Coord|` and draw at most once for each bound of `Tier.costs`. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

open NightstreamFPrime.Spec.Compression.Lib NightstreamFPrime.Spec.Compression.Game
open NightstreamFPrime.Spec.Compression (Game)
open NightstreamFPrime.Spec.Compression.Akita
open NightstreamFPrime.Spec.Compression.Akita.Extract (Tier Presents view Prefix Packs)

open Replayer

section Code

variable {pub : Public} {proof : ByteArray} {messages : Array Message}

/-- On the Euclidean route the energy `E_resp` of an accepted proof of `layout20` is at most the
cap `S_max` of the level. -/
theorem energy20 {zMax rice : ℕ} (layout : pub.layout = layout20 zMax rice)
    (accepted : runProof pub proof = .verdict .accept) {full : Array Fq}
    (terminalSetup : pub.setupFor pub.layout.terminal.setupCount = some full) {rest : ℕ}
    (decoded : decode pub.ops proof = .ok (messages, rest)) (ℓ : Fin 6)
    (euclidean : (fold20 ℓ).euclidean = true) :
    (normsIn (fused20 pub proof messages ℓ) ℓ).energy ≤ (fold20 ℓ).limit := by
  obtain ⟨scripted, ends, levels⟩ := verifyProof_script20 layout accepted
  obtain ⟨r, a, e, shape⟩ := (levels ℓ).shape
  have ran := foldRunIn_of_run pub proof messages (channelStart pub) (channelStart_fresh pub)
    (channelStart_empty pub) _ _ _ ℓ (fold20 ℓ) shape (replay_ran pub proof messages
      (replay_of_decoded pub proof scripted ends terminalSetup accepted decoded).2 (levels ℓ).at_)
  cases fits : energyFits (normsIn (fused20 pub proof messages ℓ) ℓ).energy (fold20 ℓ).limit
  · have none : foldRunIn pub (fused20 pub proof messages ℓ) ℓ (fold20 ℓ) = none := by
      unfold foldRunIn
      cases pub.setupFor (fold20 ℓ).setupCount with
      | none => rfl
      | some setup =>
          simp only [Option.bind_some, euclidean, ↓reduceIte]
          rw [foldRunSq_if, treeRun_normRun_none fits]
          rfl
    exact absurd (none.symm.trans ran) (by simp)
  · simp only [energyFits, Bool.and_eq_true, decide_eq_true_eq] at fits
    exact fits.1

theorem narrow20 : ∀ ℓ : Fin 6, narrow 8 ((productStages (fold20 ℓ).logBasis).map (·.1)) 1 = true := by
  decide +kernel

theorem leaves20 : ∀ ℓ : Fin 6, rangeLeaf (2 ^ (fold20 ℓ).logBasis) ≤ 8 ∧
    leafDegree (fold20 ℓ).logBasis + 1 + 1 ≤ 8 ∧ (fold20 ℓ).limit ≤ 3029804899 := by
  decide +kernel

variable {E : Type} [Field E] [Fintype E] [Algebra Zq E] [Degree2 E]

/-- The path of the code of each level of an accepted proof of `layout20` fits the cells of
`Key20.Wire` from the message 0. -/
theorem fits_path20 {zMax rice : ℕ} (layout : pub.layout = layout20 zMax rice)
    (accepted : runProof pub proof = .verdict .accept) {full : Array Fq}
    (terminalSetup : pub.setupFor pub.layout.terminal.setupCount = some full) {rest : ℕ}
    (decoded : decode pub.ops proof = .ok (messages, rest)) (ℓ : Fin 6) :
    Fits E 8 3029804899 4 (path20 pub proof messages E ℓ) 0 2 := by
  unfold path20
  refine (fits_foldPathIn (by decide) (by decide) (by decide) pub _ ℓ (fold20 ℓ) _ _ _ (narrow20 ℓ)
    (leaves20 ℓ).1 (leaves20 ℓ).2.1 fun euclidean =>
      (energy20 layout accepted terminalSetup decoded ℓ euclidean).trans (leaves20 ℓ).2.2).append ?_
  exact fits_ite (fits_nil (by decide)) (fits_tensoredPath (by decide) (by decide) _ _ _ _ _ _ _
    (by decide))

end Code

section Block

variable {E : Type} [Field E] [Fintype E] [DecidableEq E] [Algebra Zq E] [Degree2 E]
  {Coord : Type} [Fintype Coord] [DecidableEq Coord]
  {Block : Type} {Ring : Block → Type} [∀ b, Zero (Ring b)] {Stmt Brk Wit Weak Result : Type}

omit [DecidableEq E] [Degree2 E] in
/-- The block of the stages of one level on `Key20.Wire` (`Extract.Tier.exists_block`): for a
path that fits the cells, checks that run on it from each wire message with the data `note`, are
drawable and draw at most once for each bound, and a fold answer `fold` that decodes to the
challenges `c`, a block of the stages of the level exists whose messages have the data `note` and
the nonce 0, whose answer `i` has the work slot `work (T.start + i)`, and such that each
transcript with `wire₀` before the stage `T.start` and the block from there shows `c` and the
path. -/
theorem Tier.exists_block20
    (T : Tier Block Ring Stmt Brk Wit Weak (Key20.Wire E) Coord (CodeSlot E) Result Table)
    (reads : T.read = Extract.Page.read) (note : Key20.Note E) (work : ℕ → CodeSlot E)
    {c : (b : Block) → Ring b} {path : List Move}
    (runs : ∀ wire : Key20.Wire E, wire.data = note → ∃ result, (T.check wire c 0).Runs path result)
    (drawable : ∀ wire : Key20.Wire E, Extract.Drawable (Coord → CodeSlot E) (T.check wire c 0))
    (draws : ∀ wire : Key20.Wire E, DrawsLe (T.check wire c 0) T.costs.length)
    (packs : Packs E 8 3029804899 4 path 0) (fold : Option Coord → CodeSlot E)
    (worked : fold none = work T.start) (decoded : ∀ b, T.decode b (fold (T.block b)) = some (c b))
    (wire₀ : Key20.Wire E) (data₀ : wire₀.data = note) :
    ∃ block : List ((Option Coord → CodeSlot E) × Key20.Wire E),
      block.length = T.costs.length + 1 ∧ (∀ x ∈ block, x.2.data = note ∧ x.2.nonce = 0) ∧
      (∀ i (inside : i < block.length), block[i].1 none = work (T.start + i)) ∧
      ∀ {total : ℕ} (point : Prefix (Key20.Wire E) (Option Coord → CodeSlot E) total),
        point.wire T.start = wire₀ → (point.rest.drop T.start).take (T.costs.length + 1) = block →
        Tier.Shows T c path 0 point := by
  obtain ⟨result, ran⟩ := runs wire₀ data₀
  obtain ⟨block, length, fields, head, later, presents⟩ := Extract.Tier.exists_block T reads wire₀
    fold decoded 0 ran (drawable wire₀) packs ((draws wire₀).runs ran)
    (fun i => work (T.start + 1 + i)) (fun _ => note) (fun _ => 0)
  have first : ∀ inside : 0 < block.length, block[0].1 = fold := by
    intro inside
    simpa [List.head?_eq_getElem?, List.getElem?_eq_getElem inside] using head
  refine ⟨block, length, fun x member => ?_, fun i inside => ?_, fun point wired taken => ?_⟩
  · obtain ⟨i, inside, rfl⟩ := List.getElem_of_mem member
    exact fields i inside
  · cases i with
    | zero => rw [first inside, worked, Nat.add_zero]
    | succ i =>
        rw [later i inside]
        exact congrArg work (by omega)
  · obtain ⟨-, -, -, presented, -⟩ := presents point wired taken
    have nonempty : block ≠ [] := List.ne_nil_of_length_pos (by omega)
    have reads := (Extract.Prefix.reads_of_take (point := point)
      (block := block) (by rw [length]; exact taken) nonempty).2 0 (by omega)
    rw [Nat.add_zero, first (by omega)] at reads
    exact ⟨⟨fold, reads.1, decoded⟩, presented⟩

end Block

section Levels

variable {E : Type} [Field E] [Fintype E] [DecidableEq E] [Algebra Zq E] [Degree2 E]
  {Coord : Type} [Fintype Coord] [DecidableEq Coord] {total : ℕ}
  (o : Key20.Tower E (Key20.Wire E) Coord total)

local notation "code₁" =>
  (CodeSlot.open Key20.open64 Key20.open64_fits :
    CodeSlot E → Option (Rq Key20.level₁.opening.carrierDegree))

local notation "code₂" =>
  (CodeSlot.open Key20.open128 Key20.open128_fits :
    CodeSlot E → Option (Rq Key20.level₂.shape.innerDegree))

local notation "code₃" =>
  (CodeSlot.kept128 : CodeSlot E → Option (Rq Key20.level₃.shape.innerDegree))

local notation "code₄" =>
  (CodeSlot.kept64 : CodeSlot E → Option (Rq Key20.level₄.shape.innerDegree))

local notation "code₅" =>
  (CodeSlot.kept64 : CodeSlot E → Option (Rq Key20.level₅.shape.innerDegree))

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

omit [Fintype Coord] [DecidableEq Coord] in
/-- A level with the tensor reduction of its final claim is presentable when the level is. -/
theorem Tier.tensored_presentable [Nonempty Coord] {S : Shape} {O : OpeningShape S}
    {fold : Descriptor} {Ring Stmt Brk Wire : Type}
    (T : Tier (Fin S.blocks) (fun _ => Ring) Stmt Brk (Witness O fold) (Weak O) Wire Coord
      (CodeSlot E) (Option (List E × E)) Table) (ν : ℕ) {wire : Wire} {c : Fin S.blocks → Ring}
    {source : Table} [Fintype Coord] [DecidableEq Coord]
    (checks : Extract.Presentable (Coord → CodeSlot E) (List E) (T.check wire c source)) :
    Extract.Presentable (Coord → CodeSlot E) (List E) ((T.tensored ν).check wire c source) := by
  refine Extract.Presentable.bind checks fun result => ?_
  rcases result with _ | ⟨r2, v⟩
  · exact trivial
  · exact Extract.presentable_tensorCheck ν source r2 v

omit [Degree2 E] in
/-- The checks of the root fold are drawable for `22 ≤ |Coord|` and draw at most once for each
bound. -/
theorem checks20_zeroth (slots : 22 ≤ Fintype.card Coord) (wire : Key20.Wire E)
    (c : (b : Key20.root.Block) → Rq (Key20.root.carrierDegree b.1)) :
    Extract.Drawable (Coord → CodeSlot E) ((zeroth).check wire c 0) ∧
      DrawsLe ((zeroth).check wire c 0) (zeroth).costs.length := by
  have : Nonempty Coord := Fintype.card_pos_iff.mp (by omega)
  refine ⟨Extract.Presentable.drawable (Chunk := List E) (Extract.presentable_foldCheck _ _ _ _ _
    (le_trans (by decide : Key20.rootVars ≤ 22) slots)
      (le_trans (by decide : Key20.rootBatch ≤ 22) slots)), ?_⟩
  simp only [Extract.liftBatch, Extract.foldCosts_length]
  exact foldCheck_draws _ _ _ _ _

/-- The checks of the level 1 are drawable for `22 ≤ |Coord|` and draw at most once for each
bound. -/
theorem checks20_first (slots : 22 ≤ Fintype.card Coord) (wire : Key20.Wire E)
    (c : Fin Key20.level₁.shape.blocks → Rq Key20.level₁.opening.carrierDegree) :
    Extract.Drawable (Coord → CodeSlot E) ((first).check wire c 0) ∧
      DrawsLe ((first).check wire c 0) (first).costs.length := by
  have : Nonempty Coord := Fintype.card_pos_iff.mp (by omega)
  refine ⟨Extract.Presentable.drawable (Chunk := List E) (Tier.tensored_presentable _ _
    (Extract.presentable_foldCheck _ _ _ _ _ (le_trans (by decide : Key20.level₁.vars ≤ 22) slots)
      (le_trans (by decide : Key20.level₁.batch ≤ 22) slots))), Tier.tensored_draws _ _ ?_⟩
  simp only [Extract.liftFold, Extract.foldCosts_length]
  exact foldCheck_draws _ _ _ _ _

/-- The checks of the level 2 are drawable for `22 ≤ |Coord|` and draw at most once for each
bound. -/
theorem checks20_second (slots : 22 ≤ Fintype.card Coord) (wire : Key20.Wire E)
    (c : Fin Key20.level₂.shape.blocks → Rq Key20.level₂.shape.innerDegree) :
    Extract.Drawable (Coord → CodeSlot E) ((second).check wire c 0) ∧
      DrawsLe ((second).check wire c 0) (second).costs.length := by
  have : Nonempty Coord := Fintype.card_pos_iff.mp (by omega)
  refine ⟨Extract.Presentable.drawable (Chunk := List E) (Tier.tensored_presentable _ _
    (Extract.presentable_foldCheck _ _ _ _ _ (le_trans (by decide : Key20.level₂.vars ≤ 22) slots)
      (le_trans (by decide : Key20.level₂.batch ≤ 22) slots))), Tier.tensored_draws _ _ ?_⟩
  simp only [Extract.committedTrace, Extract.foldCosts_length]
  exact foldCheck_draws _ _ _ _ _

/-- The checks of the level 3 are drawable for `22 ≤ |Coord|` and draw at most once for each
bound. -/
theorem checks20_third (slots : 22 ≤ Fintype.card Coord) (wire : Key20.Wire E)
    (c : Fin Key20.level₃.shape.blocks → Rq Key20.level₃.shape.innerDegree) :
    Extract.Drawable (Coord → CodeSlot E) ((third).check wire c 0) ∧
      DrawsLe ((third).check wire c 0) (third).costs.length := by
  have : Nonempty Coord := Fintype.card_pos_iff.mp (by omega)
  refine ⟨Extract.Presentable.drawable (Chunk := List E) (Tier.tensored_presentable _ _
    (Extract.presentable_foldCheckSq _ _ _ _ _ _ _ (le_trans (by decide : Key20.level₃.vars ≤ 22) slots)
      (le_trans (by decide : Key20.level₃.batch ≤ 22) slots))), Tier.tensored_draws _ _ ?_⟩
  simp only [Extract.committedTraceSq, Extract.foldCostsSq_length]
  exact foldCheckSq_draws _ _ _ _ _ _ _

/-- The checks of the level 4 are drawable for `22 ≤ |Coord|` and draw at most once for each
bound. -/
theorem checks20_fourth (slots : 22 ≤ Fintype.card Coord) (wire : Key20.Wire E)
    (c : Fin Key20.level₄.shape.blocks → Rq Key20.level₄.shape.innerDegree) :
    Extract.Drawable (Coord → CodeSlot E) ((fourth).check wire c 0) ∧
      DrawsLe ((fourth).check wire c 0) (fourth).costs.length := by
  have : Nonempty Coord := Fintype.card_pos_iff.mp (by omega)
  refine ⟨Extract.Presentable.drawable (Chunk := List E) (Tier.tensored_presentable _ _
    (Extract.presentable_foldCheckSq _ _ _ _ _ _ _ (le_trans (by decide : Key20.level₄.vars ≤ 22) slots)
      (le_trans (by decide : Key20.level₄.batch ≤ 22) slots))), Tier.tensored_draws _ _ ?_⟩
  simp only [Extract.committedTraceSq, Extract.foldCostsSq_length]
  exact foldCheckSq_draws _ _ _ _ _ _ _

/-- The checks of the level 5 are drawable for `22 ≤ |Coord|` and draw at most once for each
bound. -/
theorem checks20_fifth (slots : 22 ≤ Fintype.card Coord) (wire : Key20.Wire E)
    (c : Fin Key20.level₅.shape.blocks → Rq Key20.level₅.shape.innerDegree) :
    Extract.Drawable (Coord → CodeSlot E) ((fifth).check wire c 0) ∧
      DrawsLe ((fifth).check wire c 0) (fifth).costs.length := by
  have : Nonempty Coord := Fintype.card_pos_iff.mp (by omega)
  refine ⟨Extract.Presentable.drawable (Chunk := List E) (Tier.tensored_presentable _ _
    (Extract.presentable_foldCheckSq _ _ _ _ _ _ _ (le_trans (by decide : Key20.level₅.vars ≤ 22) slots)
      (le_trans (by decide : Key20.level₅.batch ≤ 22) slots))), Tier.tensored_draws _ _ ?_⟩
  simp only [Extract.committedTraceSq, Extract.foldCostsSq_length]
  exact foldCheckSq_draws _ _ _ _ _ _ _

variable (pub : Public) (proof : ByteArray) (messages : Array Message)
  (tLast : ℕ → Rq Key20.terminal.innerDegree) (eLast : Rq Key20.terminal.innerDegree)
  {block : Key20.root.Block → Coord} {block₁ : Fin Key20.level₁.shape.blocks → Coord}
  {block₂ : Fin Key20.level₂.shape.blocks → Coord} {block₃ : Fin Key20.level₃.shape.blocks → Coord}
  {block₄ : Fin Key20.level₄.shape.blocks → Coord} {block₅ : Fin Key20.level₅.shape.blocks → Coord}
  {blockL : Fin Key20.terminal.blocks → Coord}

omit [Fintype Coord] [DecidableEq Coord] in
/-- The checks of the root fold of `tower20` from each wire message with the data `note20` run on
the path of the code (the game of `verifyProof_key20_root_of`). -/
theorem runs20_zeroth {zMax rice : ℕ} (layout : pub.layout = layout20 zMax rice)
    (accepted : runProof pub proof = .verdict .accept) {full : Array Fq}
    (terminalSetup : pub.setupFor pub.layout.terminal.setupCount = some full) {rest : ℕ}
    (decoded : decode pub.ops proof = .ok (messages, rest))
    (points : ∀ g : Fin 3, ((pub.groups[g.val]?.map (·.point)).getD #[]).size = ![20, 16, 16] g)
    (values : (pub.groups.flatMap (·.value)).size = 3)
    (expanded : (pub.setupFor (fold20 0).setupCount).isSome)
    (tower : o = tower20 pub block block₁ block₂ block₃ block₄ block₅ blockL total 472)
    (wire : Key20.Wire E) (data : wire.data = note20 pub proof messages tLast eLast) :
    ∃ result, ((zeroth).check wire (challenge20₀ pub proof messages) 0).Runs
      (path20 pub proof messages E 0) result := by
  subst tower
  obtain ⟨scripted, ends, levels⟩ := verifyProof_script20 layout accepted
  obtain ⟨rootFrame, -⟩ := verifyProof_frames20 layout accepted
  have at_ : pub.ops[fusedIndex pub.layout 0]? =
      some (.check ⟨0, "fused.end", none⟩ (.fused 0 (fold20 0))) := (levels 0).at_
  have found : pub.setupFor (fold20 0).setupCount = some (setup20 pub 0) := by
    obtain ⟨setup, found⟩ := Option.isSome_iff_exists.mp expanded
    simp [setup20, found]
  have rooted := (verifyProof_key20_rootLink (E := E) pub proof scripted ends terminalSetup
    accepted decoded at_ (layout20_fold zMax rice 0) rootFrame.weights (fun g => 3 + g.val)
    rootFrame.pointAt rootFrame.pointFresh rootFrame.pointAfter rootFrame.pointKept
    rootFrame.openingsAt rootFrame.openingsFresh rootFrame.openingsAfter rootFrame.openingsKept
    points values (setup20 pub 0)).2
  obtain ⟨fits, room, rowsFit, runs⟩ := verifyProof_key20_root_of (E := E) pub proof layout
    accepted terminalSetup decoded rfl found (fun _ => basis20 E) (fun _ t => basis20_apply t)
    (fun g i => challenge20₀ pub proof messages ⟨g, i⟩) (fun k i => by
      simp only [challenge20₀, Equiv.symm_apply_apply]) 0
  have rooted' : batchStatementFor (E := E) Key20.root (setup20 pub 0)
      (claimPointIn (fused20 pub proof messages 0) 0) (dataIn pub (fused20 pub proof messages 0) 0)
      (fun k => .root (groupOrder 3 k).val) (rowsIn (fused20 pub proof messages 0) 0 (fold20 0))
      (claimsIn (fused20 pub proof messages 0) 0 (fold20 0)) =
      (wire.data.claim).statement
        (fun g => Key.ofSetup (flatOf (setup20 pub 0)) (Key20.root.shape g))
        ⟨Setup.shared (flatOf (setup20 pub 0)) Key20.root.openDegree Key20.root.openRank _
          Key20.root.openSetupCode, PathKey.ofSetup (flatOf (setup20 pub 0)) Key20.root.chain⟩ := by
    rw [data]
    exact rooted
  rw [rooted'] at runs
  rw [path20_zero]
  exact ⟨_, runs⟩

omit [Fintype Coord] [DecidableEq Coord] in
/-- The checks of the level 1 of `tower20` with the tensor reduction of the level 2, from each
wire message with the data `note20`, run on the path of the code. -/
theorem runs20_first {zMax rice : ℕ} (layout : pub.layout = layout20 zMax rice)
    (accepted : runProof pub proof = .verdict .accept) {full : Array Fq}
    (terminalSetup : pub.setupFor pub.layout.terminal.setupCount = some full) {rest : ℕ}
    (decoded : decode pub.ops proof = .ok (messages, rest))
    (expanded : (pub.setupFor (fold20 1).setupCount).isSome)
    (tower : o = tower20 pub block block₁ block₂ block₃ block₄ block₅ blockL total 472)
    (wire : Key20.Wire E) (data : wire.data = note20 pub proof messages tLast eLast) :
    ∃ result, ((first).check wire (challenge20₁ pub proof messages) 0).Runs
      (path20 pub proof messages E 1) result := by
  subst tower
  obtain ⟨scripted, ends, levels⟩ := verifyProof_script20 layout accepted
  obtain ⟨-, pack, tensor, -⟩ := verifyProof_frames20 layout accepted
  have at₀ : pub.ops[fusedIndex pub.layout 0]? =
      some (.check ⟨0, "fused.end", none⟩ (.fused 0 (fold20 0))) := (levels 0).at_
  have game₀ : gameShape (pub.ops.zipIdx.toList.take (fusedIndex pub.layout 0)) 0 (fold20 0) =
      true := (levels 0).game
  have found : pub.setupFor (fold20 1).setupCount = some (setup20 pub 1) := by
    obtain ⟨setup, found⟩ := Option.isSome_iff_exists.mp expanded
    simp [setup20, found]
  have packed := (verifyProof_key20_packLink (E := E) pub proof scripted ends terminalSetup accepted
    decoded at₀ (layout20_fold zMax rice 0) game₀ pack.pointAt pack.openingsAt pack.pointBefore
    pack.openingsBefore pack.pointAfter pack.openingsAfter pack.pointHeld pack.openingsHeld
    pack.pointFresh pack.openingsFresh pack.pointKept pack.openingsKept
    (layout20_fold zMax rice 1) (setup20 pub 1)).2.2.2.2.2
  obtain ⟨fits, room, rowsFit, runs⟩ := verifyProof_key20_level1_of (E := E) pub proof layout
    accepted terminalSetup decoded rfl found (basis20 E) basis20_apply
    (challenge20₁ pub proof messages) (fun _ => rfl) 0
  have tensored := verifyProof_tensored pub proof scripted ends terminalSetup accepted decoded
    tensor.at_ tensor.rounds tensor.before tensor.after tensor.held tensor.kept 0 runs
  dsimp only at tensored
  obtain ⟨-, -, runs'⟩ := tensored
  have packed' : packStatementFor (E := E) Key20.level₁.opening (setup20 pub 1)
      (claimPointIn (fused20 pub proof messages 1) 1 0) (dataIn pub (fused20 pub proof messages 1) 1)
      .witness (rowsIn (fused20 pub proof messages 1) 1 (fold20 1))
      (claimsIn (fused20 pub proof messages 1) 1 (fold20 1)) =
      (wire.data.claim₁).statement (Key.ofSetup (flatOf (setup20 pub 1)) Key20.level₁.shape)
        (OpeningKey.ofSetup (flatOf (setup20 pub 1)) Key20.level₁.opening) := by
    rw [data]
    exact packed
  rw [packed'] at runs'
  rw [path20_trace pub proof messages 1 (by decide), receiver20_lt pub (by decide)]
  exact ⟨_, runs'⟩

omit [Fintype Coord] [DecidableEq Coord] in
/-- The checks of the level 2 of `tower20` with the tensor reduction of the level 3, from each
wire message with the data `note20`, run on the path of the code. -/
theorem runs20_second {zMax rice : ℕ} (layout : pub.layout = layout20 zMax rice)
    (accepted : runProof pub proof = .verdict .accept) {full : Array Fq}
    (terminalSetup : pub.setupFor pub.layout.terminal.setupCount = some full) {rest : ℕ}
    (decoded : decode pub.ops proof = .ok (messages, rest))
    (expanded : (pub.setupFor (fold20 2).setupCount).isSome)
    (tower : o = tower20 pub block block₁ block₂ block₃ block₄ block₅ blockL total 472)
    (wire : Key20.Wire E) (data : wire.data = note20 pub proof messages tLast eLast) :
    ∃ result, ((second).check wire (challengeTrace20 pub proof messages 2 Key20.level₂) 0).Runs
      (path20 pub proof messages E 2) result := by
  subst tower
  obtain ⟨scripted, ends, -⟩ := verifyProof_script20 layout accepted
  obtain ⟨-, -, frames⟩ := verifyProof_frames20 layout accepted
  have tensorBefore := frames.1
  have tensor := frames.2.1
  have found : pub.setupFor (fold20 2).setupCount = some (setup20 pub 2) := by
    obtain ⟨setup, found⟩ := Option.isSome_iff_exists.mp expanded
    simp [setup20, found]
  have before := (verifyProof_key20_traceLink (E := E) pub proof scripted ends terminalSetup
    accepted decoded List.mem_cons_self tensorBefore.at_ tensorBefore.rounds tensorBefore.after
    tensorBefore.kept (layout20_fold zMax rice 2) (by decide) (setup20 pub 2)).2.2.2.2
  obtain ⟨quarter, fits, rowsFit, runs⟩ := verifyProof_key20_trace_of (E := E) pub proof layout
    accepted terminalSetup decoded rfl found (challengeTrace20 pub proof messages 2 Key20.level₂)
    (fun _ => rfl) 0
  have tensored := verifyProof_tensored pub proof scripted ends terminalSetup accepted decoded
    tensor.at_ tensor.rounds tensor.before tensor.after tensor.held tensor.kept 0 runs
  dsimp only at tensored before
  obtain ⟨-, -, runs'⟩ := tensored
  have before' : traceStatementFor (E := E) Key20.level₂.opening quarter (setup20 pub 2)
      (traceValuesIn (stateBefore pub proof messages pub.ops (fusedIndex pub.layout 2)) 2 (fold20 2))
      (dataIn pub (stateBefore pub proof messages pub.ops (fusedIndex pub.layout 2)) 2) .witness
      (rowsIn (stateBefore pub proof messages pub.ops (fusedIndex pub.layout 2)) 2 (fold20 2))
      (claimsIn (stateBefore pub proof messages pub.ops (fusedIndex pub.layout 2)) 2 (fold20 2)) =
      (wire.data.claim₂).statement (Key.ofSetup (flatOf (setup20 pub 2)) Key20.level₂.shape)
        (OpeningKey.ofSetup (flatOf (setup20 pub 2)) Key20.level₂.opening) := by
    rw [data]
    exact before
  rw [before'] at runs'
  rw [path20_trace pub proof messages 2 (by decide), receiver20_lt pub (by decide)]
  exact ⟨_, runs'⟩

omit [Fintype Coord] [DecidableEq Coord] in
/-- The checks of the level 3 of `tower20` with the tensor reduction of the level 4, from each
wire message with the data `note20`, run on the path of the code. -/
theorem runs20_third {zMax rice : ℕ} (layout : pub.layout = layout20 zMax rice)
    (accepted : runProof pub proof = .verdict .accept) {full : Array Fq}
    (terminalSetup : pub.setupFor pub.layout.terminal.setupCount = some full) {rest : ℕ}
    (decoded : decode pub.ops proof = .ok (messages, rest))
    (expanded : (pub.setupFor (fold20 3).setupCount).isSome)
    (tower : o = tower20 pub block block₁ block₂ block₃ block₄ block₅ blockL total 472)
    (wire : Key20.Wire E) (data : wire.data = note20 pub proof messages tLast eLast) :
    ∃ result, ((third).check wire (challengeTrace20 pub proof messages 3 Key20.level₃) 0).Runs
      (path20 pub proof messages E 3) result := by
  subst tower
  obtain ⟨scripted, ends, -⟩ := verifyProof_script20 layout accepted
  obtain ⟨-, -, frames⟩ := verifyProof_frames20 layout accepted
  have tensorBefore := frames.2.1
  have tensor := frames.2.2.1
  have found : pub.setupFor (fold20 3).setupCount = some (setup20 pub 3) := by
    obtain ⟨setup, found⟩ := Option.isSome_iff_exists.mp expanded
    simp [setup20, found]
  have before := (verifyProof_key20_traceLink (E := E) pub proof scripted ends terminalSetup
    accepted decoded (List.mem_cons_of_mem _ List.mem_cons_self) tensorBefore.at_
    tensorBefore.rounds tensorBefore.after tensorBefore.kept (layout20_fold zMax rice 3) (by decide)
    (setup20 pub 3)).2.2.2.2
  obtain ⟨quarter, fits, rowsFit, runs⟩ := verifyProof_key20_traceSq_of (E := E) 3 Key20.level₃
    List.mem_cons_self pub proof layout accepted terminalSetup decoded rfl found
    (challengeTrace20 pub proof messages 3 Key20.level₃) (fun _ => rfl) 0
  have tensored := verifyProof_tensored pub proof scripted ends terminalSetup accepted decoded
    tensor.at_ tensor.rounds tensor.before tensor.after tensor.held tensor.kept 0 runs
  dsimp only at tensored before
  obtain ⟨-, -, runs'⟩ := tensored
  have before' : traceStatementFor (E := E) Key20.level₃.opening quarter (setup20 pub 3)
      (traceValuesIn (stateBefore pub proof messages pub.ops (fusedIndex pub.layout ((3 : Fin 6) : ℕ)))
        ((3 : Fin 6) : ℕ) (fold20 3))
      (dataIn pub (stateBefore pub proof messages pub.ops (fusedIndex pub.layout ((3 : Fin 6) : ℕ)))
        ((3 : Fin 6) : ℕ)) .witness
      (rowsIn (stateBefore pub proof messages pub.ops (fusedIndex pub.layout ((3 : Fin 6) : ℕ)))
        ((3 : Fin 6) : ℕ) (fold20 3))
      (claimsIn (stateBefore pub proof messages pub.ops (fusedIndex pub.layout ((3 : Fin 6) : ℕ)))
        ((3 : Fin 6) : ℕ) (fold20 3)) =
      (wire.data.claim₃).statement (Key.ofSetup (flatOf (setup20 pub 3)) Key20.level₃.shape)
        (OpeningKey.ofSetup (flatOf (setup20 pub 3)) Key20.level₃.opening) := by
    rw [data]
    exact before
  rw [before'] at runs'
  rw [path20_trace pub proof messages 3 (by decide), receiver20_lt pub (by decide)]
  exact ⟨_, runs'⟩

omit [Fintype Coord] [DecidableEq Coord] in
/-- The checks of the level 4 of `tower20` with the tensor reduction of the level 5, from each
wire message with the data `note20`, run on the path of the code. -/
theorem runs20_fourth {zMax rice : ℕ} (layout : pub.layout = layout20 zMax rice)
    (accepted : runProof pub proof = .verdict .accept) {full : Array Fq}
    (terminalSetup : pub.setupFor pub.layout.terminal.setupCount = some full) {rest : ℕ}
    (decoded : decode pub.ops proof = .ok (messages, rest))
    (expanded : (pub.setupFor (fold20 4).setupCount).isSome)
    (tower : o = tower20 pub block block₁ block₂ block₃ block₄ block₅ blockL total 472)
    (wire : Key20.Wire E) (data : wire.data = note20 pub proof messages tLast eLast) :
    ∃ result, ((fourth).check wire (challengeTrace20 pub proof messages 4 Key20.level₄) 0).Runs
      (path20 pub proof messages E 4) result := by
  subst tower
  obtain ⟨scripted, ends, -⟩ := verifyProof_script20 layout accepted
  obtain ⟨-, -, frames⟩ := verifyProof_frames20 layout accepted
  have tensorBefore := frames.2.2.1
  have tensor := frames.2.2.2.1
  have found : pub.setupFor (fold20 4).setupCount = some (setup20 pub 4) := by
    obtain ⟨setup, found⟩ := Option.isSome_iff_exists.mp expanded
    simp [setup20, found]
  have before := (verifyProof_key20_traceLink (E := E) pub proof scripted ends terminalSetup
    accepted decoded (List.mem_cons_of_mem _ (List.mem_cons_of_mem _ List.mem_cons_self))
    tensorBefore.at_ tensorBefore.rounds tensorBefore.after tensorBefore.kept
    (layout20_fold zMax rice 4) (by decide) (setup20 pub 4)).2.2.2.2
  obtain ⟨quarter, fits, rowsFit, runs⟩ := verifyProof_key20_traceSq_of (E := E) 4 Key20.level₄
    (List.mem_cons_of_mem _ List.mem_cons_self) pub proof layout accepted terminalSetup decoded rfl
    found (challengeTrace20 pub proof messages 4 Key20.level₄) (fun _ => rfl) 0
  have tensored := verifyProof_tensored pub proof scripted ends terminalSetup accepted decoded
    tensor.at_ tensor.rounds tensor.before tensor.after tensor.held tensor.kept 0 runs
  dsimp only at tensored before
  obtain ⟨-, -, runs'⟩ := tensored
  have before' : traceStatementFor (E := E) Key20.level₄.opening quarter (setup20 pub 4)
      (traceValuesIn (stateBefore pub proof messages pub.ops (fusedIndex pub.layout ((4 : Fin 6) : ℕ)))
        ((4 : Fin 6) : ℕ) (fold20 4))
      (dataIn pub (stateBefore pub proof messages pub.ops (fusedIndex pub.layout ((4 : Fin 6) : ℕ)))
        ((4 : Fin 6) : ℕ)) .witness
      (rowsIn (stateBefore pub proof messages pub.ops (fusedIndex pub.layout ((4 : Fin 6) : ℕ)))
        ((4 : Fin 6) : ℕ) (fold20 4))
      (claimsIn (stateBefore pub proof messages pub.ops (fusedIndex pub.layout ((4 : Fin 6) : ℕ)))
        ((4 : Fin 6) : ℕ) (fold20 4)) =
      (wire.data.claim₄).statement (Key.ofSetup (flatOf (setup20 pub 4)) Key20.level₄.shape)
        (OpeningKey.ofSetup (flatOf (setup20 pub 4)) Key20.level₄.opening) := by
    rw [data]
    exact before
  rw [before'] at runs'
  rw [path20_trace pub proof messages 4 (by decide), receiver20_lt pub (by decide)]
  exact ⟨_, runs'⟩

omit [Fintype Coord] [DecidableEq Coord] in
/-- The checks of the level 5 of `tower20` with the tensor reduction of the terminal, from each
wire message with the data `note20`, run on the path of the code. -/
theorem runs20_fifth {zMax rice : ℕ} (layout : pub.layout = layout20 zMax rice)
    (accepted : runProof pub proof = .verdict .accept) {full : Array Fq}
    (terminalSetup : pub.setupFor pub.layout.terminal.setupCount = some full) {rest : ℕ}
    (decoded : decode pub.ops proof = .ok (messages, rest))
    (expanded : (pub.setupFor (fold20 5).setupCount).isSome)
    (tower : o = tower20 pub block block₁ block₂ block₃ block₄ block₅ blockL total 472)
    (wire : Key20.Wire E) (data : wire.data = note20 pub proof messages tLast eLast) :
    ∃ result, ((fifth).check wire (challengeTrace20 pub proof messages 5 Key20.level₅) 0).Runs
      (path20 pub proof messages E 5) result := by
  subst tower
  obtain ⟨scripted, ends, -⟩ := verifyProof_script20 layout accepted
  obtain ⟨-, -, frames⟩ := verifyProof_frames20 layout accepted
  have tensorBefore := frames.2.2.2.1
  have tensor := frames.2.2.2.2.1
  have found : pub.setupFor (fold20 5).setupCount = some (setup20 pub 5) := by
    obtain ⟨setup, found⟩ := Option.isSome_iff_exists.mp expanded
    simp [setup20, found]
  have before := (verifyProof_key20_traceLink (E := E) pub proof scripted ends terminalSetup
    accepted decoded (List.mem_cons_of_mem _ (List.mem_cons_of_mem _ (List.mem_cons_of_mem _
      List.mem_cons_self))) tensorBefore.at_ tensorBefore.rounds tensorBefore.after
    tensorBefore.kept (layout20_fold zMax rice 5) (by decide) (setup20 pub 5)).2.2.2.2
  obtain ⟨quarter, fits, rowsFit, runs⟩ := verifyProof_key20_traceSq_of (E := E) 5 Key20.level₅
    (List.mem_cons_of_mem _ (List.mem_cons_of_mem _ List.mem_cons_self)) pub proof layout accepted
    terminalSetup decoded rfl found (challengeTrace20 pub proof messages 5 Key20.level₅)
    (fun _ => rfl) 0
  have tensored := verifyProof_tensored pub proof scripted ends terminalSetup accepted decoded
    tensor.at_ tensor.rounds tensor.before tensor.after tensor.held tensor.kept 0 runs
  dsimp only at tensored before
  obtain ⟨-, -, runs'⟩ := tensored
  have before' : traceStatementFor (E := E) Key20.level₅.opening quarter (setup20 pub 5)
      (traceValuesIn (stateBefore pub proof messages pub.ops (fusedIndex pub.layout ((5 : Fin 6) : ℕ)))
        ((5 : Fin 6) : ℕ) (fold20 5))
      (dataIn pub (stateBefore pub proof messages pub.ops (fusedIndex pub.layout ((5 : Fin 6) : ℕ)))
        ((5 : Fin 6) : ℕ)) .witness
      (rowsIn (stateBefore pub proof messages pub.ops (fusedIndex pub.layout ((5 : Fin 6) : ℕ)))
        ((5 : Fin 6) : ℕ) (fold20 5))
      (claimsIn (stateBefore pub proof messages pub.ops (fusedIndex pub.layout ((5 : Fin 6) : ℕ)))
        ((5 : Fin 6) : ℕ) (fold20 5)) =
      (wire.data.claim₅).statement (Key.ofSetup (flatOf (setup20 pub 5)) Key20.level₅.shape)
        (OpeningKey.ofSetup (flatOf (setup20 pub 5)) Key20.level₅.opening) := by
    rw [data]
    exact before
  rw [before'] at runs'
  rw [path20_trace pub proof messages 5 (by decide)]
  rw [show receiver20 pub ((5 : Fin 6) : ℕ) = pub.ops.size - 1 from rfl]
  exact ⟨_, runs'⟩

end Levels

end NightstreamFPrime.Spec.Compression.Akita.Replay
