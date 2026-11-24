import NightstreamFPrime.Spec.Compression.Akita.Replay.RunProof
import NightstreamFPrime.Spec.Compression.Akita.Replay.FinalDrawn
import NightstreamFPrime.Spec.Compression.Akita.Replay.WeightPackingJoin
import NightstreamFPrime.Spec.Compression.Akita.Replay.Stage3Match

/-! The games of the packing levels of the specification when `verifyProof` accepts, with the
weights of the end check of `fused.end` and of Stage 3 as theorems (Akita revision `d50f700`;
`docs/akita-work/match-design.md` section 5.1; the weight theorems of `Replay/WeightPackingJoin.lean`
and `Replay/Stage3Match.lean`).

- `packGroupsIn_one`: the group of `packGroupsIn` at a packing level with one group, with the
  claim coefficient 1. `packGroup_fits`: the numbers `PackGroup.Fits` do not depend on the fold
  challenges, the point and the claim coefficient.
- `verifyProof_drawnIn`: when `verifyProof` accepts, the drawn challenges of a group on the state
  before an operation are the challenges of the sampler moved by `ambient` of the family of its
  first squeeze (`drawnIn_before`). `verifyProof_setupFor`: the setup prefix of each `fused.end`
  exists (`fused_accepts`).
- `verifyProof_pack_game` (a packing level with one group), `verifyProof_batch_game` (a fold with
  opening groups) and `verifyProof_offload_game` (the offloaded root of `key30.aks`, with
  `Batch.codeScan` over the setup prefix `flat`): when `verifyProof` accepts, the game of the
  specification level runs on the path of the recorded values to acceptance with the recorded
  `(fused.point, witness_evaluation)` (and the result of Stage 3). The hypotheses are facts of
  the layout, the shapes of the script, `endsFit` and the setup prefix of the terminal. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

open NightstreamFPrime.Spec.Compression.Lib NightstreamFPrime.Spec.Compression.Game
open NightstreamFPrime.Spec.Compression (Game)
open NightstreamFPrime.Spec.Compression.Akita

open Replayer

/-- The group of `packGroupsIn` at a packing level with one group: the drawn challenges of the
group 0 in `R_{q,d_car}` and in `R_{q,d_A}`, its point and the claim coefficient 1. -/
theorem packGroupsIn_one (P : PackShape) (s : Replayer) (level : ℕ) (F : Fold) {S : Shape}
    {O : OpeningShape S} {fold : Descriptor} (slots : P.slots = [⟨0, S, O, fold⟩])
    (one : F.groups.length = 1) :
    packGroupsIn P s level F = [⟨S, O, fold, (drawnIn s level 0).map fun c =>
      Poly.ofSparse O.carrierDegree (c.map fun x => (x.1 / (S.innerDegree / O.carrierDegree), x.2)),
      (drawnIn s level 0).map (Poly.ofSparse S.innerDegree), claimPointIn s level 0, Ext.one⟩] := by
  simp only [packGroupsIn, slots, one, lt_self_iff_false, ↓reduceIte, List.map_cons, List.map_nil]
  rfl

/-- The numbers `PackGroup.Fits` of a group do not depend on its fold challenges, its point and its
claim coefficient. -/
theorem packGroup_fits {S : Shape} {O : OpeningShape S} {fold : Descriptor}
    {cs amb : List (Array Fq)} {point : List Ext} {claim : Ext} (cs' amb' : List (Array Fq))
    (point' : List Ext) (claim' : Ext) {μ : ℕ}
    (numbers : PackGroup.Fits ⟨S, O, fold, cs, amb, point, claim⟩ μ) :
    PackGroup.Fits ⟨S, O, fold, cs', amb', point', claim'⟩ μ := by
  obtain ⟨a, b, c, d, e, f, g, h, i, j, l, m, o, p, q, r, t, u, v⟩ := numbers
  exact ⟨a, b, c, d, e, f, g, h, i, j, l, m, o, p, q, r, t, u, v⟩

section Games

variable (pub : Public) (proof : ByteArray) {ops : Array Op}

/-- When `verifyProof` accepts, the drawn challenges of the group `g` of `level` on the state before
the operation `c` are the challenges of the sampler moved by `ambient G`, for the family `G` of the
first squeeze of the group before `c`. -/
theorem verifyProof_drawnIn (scripted : script pub.layout pub.plan = .ok ops)
    (ends : endsFit ops.zipIdx.toList = true) {full : Array Fq}
    (terminalSetup : pub.setupFor pub.layout.terminal.setupCount = some full)
    (accepted : verifyProof pub proof = .verdict .accept) {messages : Array Message} {rest : ℕ}
    (decoded : decode ops proof = .ok (messages, rest)) {c : ℕ} {op : Op}
    (at_ : ops[c]? = some op) {level g : ℕ} {G : FoldGroup}
    (first : squeezeOf level g (ops.zipIdx.toList.take c) = some G) :
    ∃ cs : List Sparse, drawnIn (stateBefore pub proof messages ops c) level g =
      cs.map (ambient G) :=
  drawnIn_before pub proof messages ops (replay_step_ok pub proof messages
    (replay_of_decoded pub proof scripted ends terminalSetup accepted decoded).2 at_).1 first

/-- When `verifyProof` accepts, the setup prefix of each `fused.end` exists (the hypothesis
`expanded` of the games). -/
theorem verifyProof_setupFor (scripted : script pub.layout pub.plan = .ok ops)
    (ends : endsFit ops.zipIdx.toList = true) {full : Array Fq}
    (terminalSetup : pub.setupFor pub.layout.terminal.setupCount = some full)
    (accepted : verifyProof pub proof = .verdict .accept) {messages : Array Message} {rest : ℕ}
    (decoded : decode ops proof = .ok (messages, rest)) {c : ℕ} {key : Key} {level : ℕ}
    {F : Fold} (at_ : ops[c]? = some (.check key (.fused level F))) :
    ∃ setup, pub.setupFor F.setupCount = some setup :=
  let ⟨setup, expanded, _⟩ := (fused_accepts pub proof messages key c level F _).mp
    (replay_step_ok pub proof messages
      (replay_of_decoded pub proof scripted ends terminalSetup accepted decoded).2 at_).2
  ⟨setup, expanded⟩

variable {E : Type} [Field E] [Fintype E] [DecidableEq E] [Algebra Zq E] [Degree2 E]

/-- A packing level with one group without Stage 3: when `verifyProof` accepts, `foldCheck` of
`liftedReduction` runs on the path of the code to acceptance with the recorded
`(fused.point, witness_evaluation)` on the state before its `fused.end`. The weights of the end
check are theorems (`packWeight_one`, `packSupport_one`, `packAmbiented`). -/
theorem verifyProof_pack_game (scripted : script pub.layout pub.plan = .ok ops)
    (ends : endsFit ops.zipIdx.toList = true) {full : Array Fq}
    (terminalSetup : pub.setupFor pub.layout.terminal.setupCount = some full)
    (accepted : verifyProof pub proof = .verdict .accept) {messages : Array Message} {rest : ℕ}
    (decoded : decode ops proof = .ok (messages, rest)) {c : ℕ} {key : Key} {level : ℕ}
    {F : Fold} (at_ : ops[c]? = some (.check key (.fused level F))) {r a e : ℕ}
    (shape : foldShape (ops.zipIdx.toList.take c) level F r a e = true)
    (game : gameShape (ops.zipIdx.toList.take c) level F = true)
    (drawShape : drawnShape (ops.zipIdx.toList.take c) level F.groups = true) {s : Replayer}
    (state : stateBefore pub proof messages ops c = s)
    {P : PackShape} (shaped : F.fused = .packing P) (direct : F.stage3 = none)
    {setup : Array Fq} (expanded : pub.setupFor F.setupCount = some setup) {S : Shape}
    {O : OpeningShape S} {fold : Descriptor}
    (rows : (F.rhs, F.openingRow) = relation (fun _ : Fin 1 => S) O.openRank O.openDegree O.chain
      fun _ => .witness)
    (based : 2 ^ F.logBasis = S.certifiedBase) (low : F.low = O.low)
    (β : Module.Basis (Fin O.planes) Zq E) (basis : ∀ t, β t = (extBasis.getD t Ext.zero).into)
    (challenge : Fin S.blocks → Rq O.carrierDegree)
    (challenged : ∀ i, challenge i = Poly.toRq _ (((drawnIn s level 0).map fun c =>
      Poly.ofSparse O.carrierDegree (c.map fun x => (x.1 / (S.innerDegree / O.carrierDegree),
        x.2))).getD i (Poly.zero O.carrierDegree)))
    (fits : O.cells fold ≤ 2 ^ F.vars) {digits : ℕ}
    (room : O.quotientRoom fold digits ≤ 2 ^ F.vars) (rowsFit : O.rowCount + 1 ≤ 2 ^ F.batch)
    (w : Table) (plain : F.euclidean = false) (compressed : F.compressed = O.compressed fold)
    (arities : (productStages F.logBasis).map (·.1) = rangeArities S.certifiedBase)
    (leaf : leafDegree F.logBasis = rangeLeaf S.certifiedBase)
    (slots : P.slots = [⟨0, S, O, fold⟩]) (opening : P.path = ⟨O.openDegree, O.openRank, O.chain⟩)
    (digited : P.digits = digits) (numbers : PackGroup.Fits ⟨S, O, fold, [], [], [], Ext.one⟩ F.vars)
    (G : FoldGroup) (grouped : F.groups = [G]) (ringed : G.ring = S.innerDegree)
    (degreed : G.degree = O.carrierDegree) :
    (foldCheck (liftedReduction β (packStatementFor O setup (claimPointIn s level 0)
        (dataIn pub s level) .witness (rowsIn s level F) (claimsIn s level F)) challenge F.vars fits
        digits room F.batch rowsFit) S.certifiedBase (O.gaps fold digits).support
        (O.compressed fold) w).Runs
      (foldPathIn E pub s level F setup F.vars F.batch)
      (some ((recordedIn s ⟨level, "fused.point", none⟩).toList.map Ext.into,
        (firstIn s ⟨level, "witness_evaluation", none⟩).into)) := by
  have none_ := (replay_of_decoded pub proof scripted ends terminalSetup accepted decoded).2
  have before := (replay_step_ok pub proof messages none_ at_).1
  have st : LevelState (stateBefore pub proof messages ops c) level F :=
    level_state pub proof messages _ (channelStart_fresh pub) (channelStart_empty pub) _ level F
      game (aligned_take messages _ c _ (decode_aligned decoded)) before
  rw [state] at st
  obtain ⟨virtual, count, -⟩ := st.leaf plain
  obtain ⟨cs, drawn⟩ := drawnIn_before pub proof messages ops before
    (squeezeOf_of_shape drawShape (g := 0) (by rw [grouped]; rfl))
  rw [state] at drawn
  have one := packGroupsIn_one P s level F slots (by rw [grouped]; rfl)
  have weighs := packWeight_one β basis (packInputsIn P setup s level F) (packingValuesIn s level F)
    (dataIn pub s level) .witness (rowsIn s level F) (claimsIn s level F)
    ((drawnIn s level 0).map fun c => Poly.ofSparse O.carrierDegree
      (c.map fun x => (x.1 / (S.innerDegree / O.carrierDegree), x.2)))
    ((drawnIn s level 0).map (Poly.ofSparse S.innerDegree)) (claimPointIn s level 0) challenge
    (fun i => by rw [challenged i])
    (packAmbiented s level O G ringed degreed ⟨cs, drawn⟩ challenge challenged) fits room
    rowsFit (packGroup_fits _ _ _ _ numbers) rfl digited based opening rfl rfl
    (by rw [show (packingValuesIn s level F).tau1 = (recordedIn s ⟨level, "tau1", none⟩).toList
      from rfl, Array.length_toList]; exact st.tau)
  have supported := packSupport_one (E := E) (packInputsIn P setup s level F)
    (packingValuesIn s level F) ((drawnIn s level 0).map fun c => Poly.ofSparse O.carrierDegree
      (c.map fun x => (x.1 / (S.innerDegree / O.carrierDegree), x.2)))
    ((drawnIn s level 0).map (Poly.ofSparse S.innerDegree)) (claimPointIn s level 0) room digited
    opening (by rw [show (packingValuesIn s level F).virtual =
      (roundsIn s ⟨level, "range.leaf", none⟩).map (·.2) from virtual, List.length_map, count])
  exact foldRunIn_pack_game pub proof messages _ (channelStart_fresh pub) (channelStart_empty pub) _
    key c level F shape game (aligned_take messages _ c _ (decode_aligned decoded))
    (replay_ran pub proof messages none_ at_) state shaped direct expanded rows based low β challenge
    fits room rowsFit w plain compressed arities leaf (by rw [one]; exact weighs)
    (by rw [one]; exact supported)

/-- A fold with opening groups without Stage 3: when `verifyProof` accepts, `foldCheck` of
`Batch.liftedReduction` runs on the path of the code to acceptance with the recorded
`(fused.point, witness_evaluation)` on the state before its `fused.end`. The weights of the end
check are theorems (`batchWeight_groups`, `batchSupport_in`, `batchAmbiented`). -/
theorem verifyProof_batch_game (scripted : script pub.layout pub.plan = .ok ops)
    (ends : endsFit ops.zipIdx.toList = true) {full : Array Fq}
    (terminalSetup : pub.setupFor pub.layout.terminal.setupCount = some full)
    (accepted : verifyProof pub proof = .verdict .accept) {messages : Array Message} {rest : ℕ}
    (decoded : decode ops proof = .ok (messages, rest)) {c : ℕ} {key : Key} {level : ℕ}
    {F : Fold} (at_ : ops[c]? = some (.check key (.fused level F))) {r a e : ℕ}
    (shape : foldShape (ops.zipIdx.toList.take c) level F r a e = true)
    (game : gameShape (ops.zipIdx.toList.take c) level F = true)
    (drawShape : drawnShape (ops.zipIdx.toList.take c) level F.groups = true) {s : Replayer}
    (state : stateBefore pub proof messages ops c = s)
    {P : PackShape} (shaped : F.fused = .packing P) (direct : F.stage3 = none)
    {setup : Array Fq} (expanded : pub.setupFor F.setupCount = some setup) {n : ℕ}
    {B : Batch (Fin n)} (source : Fin n → RhsSource)
    (rows : (F.rhs, F.openingRow) =
      relation (fun k => B.shape (groupOrder n k)) B.openRank B.openDegree B.chain source)
    (base : ℕ) (based : 2 ^ F.logBasis = base) (low : F.low = B.low)
    (β : ∀ g, Module.Basis (Fin (B.planes g)) Zq E)
    (basis : ∀ g t, β g t = (extBasis.getD t Ext.zero).into) (challenge : B.Challenge)
    (challenged : ∀ k i, challenge (groupOrder n k) i = Poly.toRq _
      ((drawnCarrier B s level k).getD i (Poly.zero (B.carrierDegree (groupOrder n k)))))
    (fits : B.cells ≤ 2 ^ F.vars) {digits : ℕ} (room : B.quotientRoom digits ≤ 2 ^ F.vars)
    (rowsFit : B.rowCount + 1 ≤ 2 ^ F.batch) (w : Table) (plain : F.euclidean = false)
    (compressed : F.compressed = B.compressed)
    (arities : (productStages F.logBasis).map (·.1) = rangeArities base)
    (leaf : leafDegree F.logBasis = rangeLeaf base)
    (slots : P.slots = (List.finRange n).map fun k => ⟨(groupOrder n k).val,
      B.shape (groupOrder n k), B.opening (groupOrder n k), B.fold (groupOrder n k)⟩)
    (opening : P.path = ⟨B.openDegree, B.openRank, B.chain⟩) (digited : P.digits = digits)
    (numbers : ∀ k, (batchGroup B (fun _ => []) (fun _ => []) (fun _ => []) (fun _ => Ext.one)
      k).Fits F.vars)
    (G : Fin n → FoldGroup) (grouped : F.groups = (List.finRange n).map G)
    (ringed : ∀ g, (G g).ring = (B.shape g).innerDegree)
    (degreed : ∀ g, (G g).degree = B.carrierDegree g) (k₀ : Fin n) :
    (foldCheck (Batch.liftedReduction β (batchStatementFor B setup (claimPointIn s level)
        (dataIn pub s level) source (rowsIn s level F) (claimsIn s level F)) challenge F.vars fits
        base digits room F.batch rowsFit) base (B.gaps digits).support B.compressed w).Runs
      (foldPathIn E pub s level F setup F.vars F.batch)
      (some ((recordedIn s ⟨level, "fused.point", none⟩).toList.map Ext.into,
        (firstIn s ⟨level, "witness_evaluation", none⟩).into)) := by
  have none_ := (replay_of_decoded pub proof scripted ends terminalSetup accepted decoded).2
  have before := (replay_step_ok pub proof messages none_ at_).1
  have st : LevelState (stateBefore pub proof messages ops c) level F :=
    level_state pub proof messages _ (channelStart_fresh pub) (channelStart_empty pub) _ level F
      game (aligned_take messages _ c _ (decode_aligned decoded)) before
  rw [state] at st
  obtain ⟨virtual, count, -⟩ := st.leaf plain
  have drawnAt : ∀ k : Fin n, ∃ cs : List Sparse,
      drawnIn s level (groupOrder n k).val = cs.map (ambient (G (groupOrder n k))) := fun k => by
    obtain ⟨cs, drawn⟩ := drawnIn_before pub proof messages ops before
      (squeezeOf_of_shape (g := (groupOrder n k).val) (G := G (groupOrder n k)) drawShape
        (by rw [grouped]; exact getElem?_finRange_map G _))
    exact ⟨cs, state ▸ drawn⟩
  have numbersIn : ∀ g ∈ packGroupsIn P s level F, g.Fits F.vars := by
    rw [packGroupsIn_batch B s level P F slots]
    intro g member
    obtain ⟨k, -, rfl⟩ := List.mem_map.mp member
    exact batchGroup_fits B _ _ _ _ (numbers k)
  exact foldRunIn_batch_game pub proof messages _ (channelStart_fresh pub) (channelStart_empty pub)
    _ key c level F shape game (aligned_take messages _ c _ (decode_aligned decoded))
    (replay_ran pub proof messages none_ at_) state shaped direct expanded source rows base based
    low β challenge fits room rowsFit w plain compressed arities leaf
    (batchWeight_groups s level pub F slots opening digited setup source based β basis challenge
      challenged (batchAmbiented s level B (fun k => G (groupOrder n k)) (fun _ => ringed _)
        (fun _ => degreed _) drawnAt challenge challenged) fits room rowsFit numbersIn k₀ st.tau)
    (batchSupport_in s level F slots opening digited room
      (by rw [← Array.length_toList, virtual, List.length_map, count]))

/-- The offloaded root of `key30.aks`: when `verifyProof` accepts, Stage 3 gives `some o` and
`offloadCheck` of `Batch.liftedReduction` over the setup prefix `flat`, with the split
`Batch.codeScan` of the claims (`Key30.Tower.scan`), runs on the path of the code to acceptance
with the recorded `(fused.point, witness_evaluation)` and `o`. The weights of the end check and of
Stage 3 are theorems (`batchOffload_groups`, `batchSupport_in`, `batchAmbiented`,
`stage3Weight_codeScan`). -/
theorem verifyProof_offload_game (scripted : script pub.layout pub.plan = .ok ops)
    (ends : endsFit ops.zipIdx.toList = true) {full : Array Fq}
    (terminalSetup : pub.setupFor pub.layout.terminal.setupCount = some full)
    (accepted : verifyProof pub proof = .verdict .accept) {messages : Array Message} {rest : ℕ}
    (decoded : decode ops proof = .ok (messages, rest)) {c : ℕ} {key : Key} {level : ℕ}
    {F : Fold} (at_ : ops[c]? = some (.check key (.fused level F))) (plan : SetupPlan)
    {r a e b t : ℕ} (shape : offloadShape (ops.zipIdx.toList.take c) level F plan r a e b t = true)
    (game : gameShape (ops.zipIdx.toList.take c) level F = true) {ν : ℕ}
    (setupShape : roundsAt (ops.zipIdx.toList.take c) ⟨level, "stage3", none⟩ ν 2 = true)
    (drawShape : drawnShape (ops.zipIdx.toList.take c) level F.groups = true) {s : Replayer}
    (state : stateBefore pub proof messages ops c = s)
    {P : PackShape} (shaped : F.fused = .packing P) (offloaded : F.stage3 = some ν)
    {setup : Array Fq} (expanded : pub.setupFor F.setupCount = some setup) {n : ℕ}
    {B : Batch (Fin n)} (source : Fin n → RhsSource)
    (rows : (F.rhs, F.openingRow) =
      relation (fun k => B.shape (groupOrder n k)) B.openRank B.openDegree B.chain source)
    (base : ℕ) (based : 2 ^ F.logBasis = base) (low : F.low = B.low)
    (β : ∀ g, Module.Basis (Fin (B.planes g)) Zq E)
    (basis : ∀ g t, β g t = (extBasis.getD t Ext.zero).into) (challenge : B.Challenge)
    (challenged : ∀ k i, challenge (groupOrder n k) i = Poly.toRq _
      ((drawnCarrier B s level k).getD i (Poly.zero (B.carrierDegree (groupOrder n k)))))
    (fits : B.cells ≤ 2 ^ F.vars) {digits : ℕ} (room : B.quotientRoom digits ≤ 2 ^ F.vars)
    (rowsFit : B.rowCount + 1 ≤ 2 ^ F.batch) (w : Table) (flat : Table)
    (plain : F.euclidean = false) (compressed : F.compressed = B.compressed)
    (arities : (productStages F.logBasis).map (·.1) = rangeArities base)
    (leaf : leafDegree F.logBasis = rangeLeaf base)
    (slots : P.slots = (List.finRange n).map fun k => ⟨(groupOrder n k).val,
      B.shape (groupOrder n k), B.opening (groupOrder n k), B.fold (groupOrder n k)⟩)
    (opening : P.path = ⟨B.openDegree, B.openRank, B.chain⟩) (digited : P.digits = digits)
    (numbers : ∀ k, (batchGroup B (fun _ => []) (fun _ => []) (fun _ => []) (fun _ => Ext.one)
      k).Fits F.vars)
    (G : Fin n → FoldGroup) (grouped : F.groups = (List.finRange n).map G)
    (ringed : ∀ g, (G g).ring = (B.shape g).innerDegree)
    (degreed : ∀ g, (G g).degree = B.carrierDegree g) (k₀ : Fin n)
    (lowFits : 2 ^ F.low ≤ B.openDegree) (lowShort : F.low ≤ F.vars) (blocked : B.Blocked F.low)
    (planned : plan = SetupPlan.ofBatch B (groupOrder n) F.low F.logBasis)
    (gadget : ∀ g, (B.fold g).responseBase = 2 ^ F.logBasis)
    (innerPow : ∀ g, 2 ^ Nat.log 2 (B.shape g).innerDegree = (B.shape g).innerDegree)
    (outerPow : ∀ g, 2 ^ Nat.log 2 (B.shape g).outerDegree = (B.shape g).outerDegree)
    (openPow : 2 ^ Nat.log 2 B.openDegree = B.openDegree)
    (innerFit : ∀ g, ((B.shape g).innerInstance false 0).size ≤ 2 ^ ν)
    (outerFit : ∀ g, (B.shape g).outerInstance.size ≤ 2 ^ ν)
    (openFit : B.openInstance.size ≤ 2 ^ ν) :
    ∃ o, stage3In s level plan = some o ∧
      (offloadCheck (Batch.liftedReduction β ((batchStatementFor B setup (claimPointIn s level)
          (dataIn pub s level) source (rowsIn s level F) (claimsIn s level F)).over
            fun p : Fin (2 ^ ν) => flat p) challenge F.vars fits base digits room F.batch rowsFit)
          base (B.gaps digits).support B.compressed w ν flat
          (Batch.codeScan β (batchStatementFor B setup (claimPointIn s level) (dataIn pub s level)
            source (rowsIn s level F) (claimsIn s level F)) challenge F.vars fits base digits room
            F.batch rowsFit ν flat F.low blocked lowShort)).Runs
        (offloadPathIn E pub s level F setup F.vars F.batch)
        (some (((recordedIn s ⟨level, "fused.point", none⟩).toList.map Ext.into,
            (firstIn s ⟨level, "witness_evaluation", none⟩).into),
          (o.1.map Ext.into, o.2.into))) := by
  subst planned
  have none_ := (replay_of_decoded pub proof scripted ends terminalSetup accepted decoded).2
  have before := (replay_step_ok pub proof messages none_ at_).1
  have st : LevelState (stateBefore pub proof messages ops c) level F :=
    level_state pub proof messages _ (channelStart_fresh pub) (channelStart_empty pub) _ level F
      game (aligned_take messages _ c _ (decode_aligned decoded)) before
  rw [state] at st
  obtain ⟨virtual, count, -⟩ := st.leaf plain
  have drawnAt : ∀ k : Fin n, ∃ cs : List Sparse,
      drawnIn s level (groupOrder n k).val = cs.map (ambient (G (groupOrder n k))) := fun k => by
    obtain ⟨cs, drawn⟩ := drawnIn_before pub proof messages ops before
      (squeezeOf_of_shape (g := (groupOrder n k).val) (G := G (groupOrder n k)) drawShape
        (by rw [grouped]; exact getElem?_finRange_map G _))
    exact ⟨cs, state ▸ drawn⟩
  have numbersIn : ∀ g ∈ packGroupsIn P s level F, g.Fits F.vars := by
    rw [packGroupsIn_batch B s level P F slots]
    intro g member
    obtain ⟨k, -, rfl⟩ := List.mem_map.mp member
    exact batchGroup_fits B _ _ _ _ (numbers k)
  have weighs := batchOffload_groups s level pub F slots opening digited setup source based β basis
    challenge challenged (batchAmbiented s level B (fun k => G (groupOrder n k)) (fun _ => ringed _)
      (fun _ => degreed _) drawnAt challenge challenged) fits room rowsFit numbersIn k₀ st.tau
    lowFits lowShort (Batch.codeScan β (batchStatementFor B setup (claimPointIn s level)
      (dataIn pub s level) source (rowsIn s level F) (claimsIn s level F)) challenge F.vars fits
      base digits room F.batch rowsFit ν flat F.low blocked lowShort)
    (fun _ _ _ => rfl) (fun _ _ _ => rfl)
  exact offloadRunIn_game pub proof messages _ (channelStart_fresh pub) (channelStart_empty pub) _
    key c level F _ shape game setupShape (aligned_take messages _ c _ (decode_aligned decoded))
    (replay_ran pub proof messages none_ at_) state shaped offloaded expanded source rows base based
    low β challenge fits room rowsFit w flat _ plain compressed arities leaf weighs
    (batchSupport_in s level F slots opening digited room
      (by rw [← Array.length_toList, virtual, List.length_map, count]))
    fun r2 ρ r2Length ρLength => stage3Weight_codeScan β _ challenge F.vars fits base digits room
      F.batch rowsFit ν flat F.low blocked lowShort F.logBasis gadget innerPow outerPow openPow
      innerFit outerFit openFit (firstIn s ⟨level, "alpha", none⟩)
      (recordedIn s ⟨level, "tau1", none⟩).toList r2 ρ
      (by rw [Array.length_toList]; exact st.tau) r2Length ρLength

end Games

end NightstreamFPrime.Spec.Compression.Akita.Replay
