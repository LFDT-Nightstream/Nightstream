import NightstreamFPrime.Spec.Compression.Akita.Replay.LinkTrace

/-! The links between the levels of `key20.aks` on the replay side (Akita revision `d50f700`; the
folds of `layout20`; `Key20.Tower.accepts` of `Key20Opening.lean`, `towerLinks` of
`Extract/Tower.lean`).

- `key20_segments`: the numbers of `verifyProof_traceLink` for each receiver with evaluation
  trace (the levels 2 to 5 and the terminal) and the rounds `ν = μ − 1` of the level before it.
- `layout20_receiver`: the folds 2 to 5 of `layout20` have one group and evaluation trace with the
  shapes of `Key20.level₂` to `Key20.level₅`.
- `verifyProof_key20_traceLink`: for each pair `(ℓ, ℓ + 1)` of `(1, 2)`, `(2, 3)`, `(3, 4)`,
  `(4, 5)`, the statement of the replay at the level `ℓ + 1` is the claim that
  `Tier.TraceLinked` takes from the result `(ρ, θ, ṽ)` of the level `ℓ` with its tensor
  reduction. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

open NightstreamFPrime.Spec.Compression.Lib NightstreamFPrime.Spec.Compression.Game
open NightstreamFPrime.Spec.Compression (Game)
open NightstreamFPrime.Spec.Compression.Akita

open Replayer

/-- The receivers with evaluation trace of `key20.aks` with the rounds `ν = μ − 1` of the tensor
reduction of the level before them: `4·(d/4) = d`, `2·(d/4) = 2^{log_2(d/2)}`, `M` is a power of
two, `ν` has the coordinates of `ρ_pack` and `ρ_pos`, and `B ≤ 2^{|ρ_blk|}` (Akita Section 4.5,
(40) to (42)). -/
theorem key20_segments :
    ∀ x ∈ [(Key20.level₂.shape, Key20.level₁.tail), (Key20.level₃.shape, Key20.level₂.tail),
        (Key20.level₄.shape, Key20.level₃.tail), (Key20.level₅.shape, Key20.level₄.tail),
        (Key20.terminal, Key20.level₅.tail)],
      4 * (x.1.innerDegree / 4) = x.1.innerDegree ∧
      2 * (x.1.innerDegree / 4) = 2 ^ Nat.log 2 (x.1.innerDegree / 2) ∧
      x.1.blockLength = 2 ^ Nat.log 2 x.1.blockLength ∧
      Nat.log 2 (x.1.innerDegree / 2) + Nat.log 2 x.1.blockLength ≤ x.2 ∧
      x.1.blocks ≤ 2 ^ (x.2 - (Nat.log 2 (x.1.innerDegree / 2) + Nat.log 2 x.1.blockLength)) := by
  decide

/-- The folds 2 to 5 of `layout20` (the receivers `L'` after the levels `L`): one group,
evaluation trace with the rounds `L.tail` and the degree of `L'`, and the shapes of `L'`. -/
theorem layout20_receiver {zMax rice level : ℕ} {L L' : Key20.Level} {F : Fold}
    (pair : (level, L, L') ∈ [(1, Key20.level₁, Key20.level₂), (2, Key20.level₂, Key20.level₃),
      (3, Key20.level₃, Key20.level₄), (4, Key20.level₄, Key20.level₅)])
    (fold : (layout20 zMax rice).folds[level + 1]? = some F) :
    F.trace = some (L.tail, L'.shape.innerDegree) ∧ F.groups.length = 1 ∧
      F.fused = .trace ⟨L'.shape, L'.opening, L'.fold⟩ := by
  rw [show (layout20 zMax rice).folds = (layout20 0 0).folds from rfl] at fold
  simp only [List.mem_cons, Prod.mk.injEq, List.not_mem_nil, or_false] at pair
  rcases pair with ⟨rfl, rfl, rfl⟩ | ⟨rfl, rfl, rfl⟩ | ⟨rfl, rfl, rfl⟩ | ⟨rfl, rfl, rfl⟩
  all_goals exact ⟨fold_field fold Fold.trace rfl, fold_field fold (fun F => F.groups.length) rfl,
    fold_field fold Fold.fused rfl⟩

theorem cast_ne_zero {n : ℕ} (positive : 0 < n) (small : n < fieldModulus) : ((n : ℕ) : Zq) ≠ 0 :=
  fun zero => absurd (Nat.le_of_dvd positive ((ZMod.natCast_eq_zero_iff _ _).mp zero))
    (not_le.mpr small)

/-- The folds 2 to 5 of `layout20` (the levels `L` with evaluation trace): the numbers of the game
theorems `verifyProof_trace_game` and `verifyProof_traceSq_game`. The level 2 has the coefficient
route and the levels 3 to 5 the Euclidean route. -/
theorem layout20_trace {zMax rice level : ℕ} {L : Key20.Level} {F : Fold}
    (member : (level, L) ∈ [(2, Key20.level₂), (3, Key20.level₃), (4, Key20.level₄),
      (5, Key20.level₅)])
    (fold : (layout20 zMax rice).folds[level]? = some F) :
    F.fused = .trace ⟨L.shape, L.opening, L.fold⟩ ∧
      (F.rhs, F.openingRow) = relation (fun _ : Fin 1 => L.shape) L.opening.openRank
        L.opening.openDegree L.opening.chain (fun _ => .witness) ∧
      2 ^ F.logBasis = L.shape.certifiedBase ∧ F.low = L.opening.low ∧ F.vars = L.vars ∧
      F.batch = L.batch ∧ F.compressed = L.opening.compressed L.fold ∧
      (productStages F.logBasis).map (·.1) = rangeArities L.shape.certifiedBase ∧
      leafDegree F.logBasis = rangeLeaf L.shape.certifiedBase ∧
      F.euclidean = decide (2 < level) ∧ F.limit < fieldModulus ∧
      ((2 * (L.shape.innerDegree / 4) : ℕ) : Zq) ≠ 0 ∧
      L.opening.cells L.fold ≤ 2 ^ L.vars ∧ L.opening.traceRowCount + 1 ≤ 2 ^ L.batch ∧
      1 ≤ rangeLeaf L.shape.certifiedBase ∧ 0 < L.fold.responseDigits := by
  rw [show (layout20 zMax rice).folds = (layout20 0 0).folds from rfl] at fold
  simp only [List.mem_cons, Prod.mk.injEq, List.not_mem_nil, or_false] at member
  rcases member with ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩
  all_goals exact ⟨fold_field fold Fold.fused rfl,
    fold_field fold (fun F => (F.rhs, F.openingRow)) rfl,
    fold_field fold (fun F => 2 ^ F.logBasis) (by decide), fold_field fold Fold.low rfl,
    fold_field fold Fold.vars rfl, fold_field fold Fold.batch rfl,
    fold_field fold Fold.compressed (by decide),
    fold_field fold (fun F => (productStages F.logBasis).map (·.1)) (by decide),
    fold_field fold (fun F => leafDegree F.logBasis) (by decide),
    fold_field fold Fold.euclidean (by decide),
    of_decide_eq_true (fold_field fold (fun F => decide (F.limit < fieldModulus)) (by decide)),
    cast_ne_zero (by decide) (by decide), by decide, by decide, by decide, by decide⟩

section Proof

variable (pub : Public) (proof : ByteArray) {ops : Array Op}
variable {E : Type} [Field E] [Fintype E] [DecidableEq E] [Algebra Zq E] [Degree2 E]

/-- Level 1 of `key20.aks` with the tensor reduction of the level 2 on its claim
(`Tier.tensored (liftFold …) level₁.tail`, the notation `first` of `Key20Opening.lean`): when
`verifyProof` accepts, `foldCheck` of `liftedReduction` of the level and then `tensorCheck` run on
the path of the fold on the state before `c` and of the tensor reduction on the state before
`c'` to `(ρ, θ, ṽ)` of the recorded values on the state before `c'`. -/
theorem verifyProof_key20_level1_tensored {zMax rice : ℕ}
    (scripted : script pub.layout pub.plan = .ok ops)
    (ends : endsFit ops.zipIdx.toList = true) {full : Array Fq}
    (terminalSetup : pub.setupFor pub.layout.terminal.setupCount = some full)
    (accepted : verifyProof pub proof = .verdict .accept) {messages : Array Message} {rest : ℕ}
    (decoded : decode ops proof = .ok (messages, rest)) {c : ℕ} {key : Key} {level : ℕ}
    {F : Fold} (at_ : ops[c]? = some (.check key (.fused level F)))
    (fold : (layout20 zMax rice).folds[1]? = some F) {i a e : ℕ}
    (foldShaped : foldShape (ops.zipIdx.toList.take c) level F i a e = true)
    (game : gameShape (ops.zipIdx.toList.take c) level F = true)
    (drawShape : drawnShape (ops.zipIdx.toList.take c) level F.groups = true)
    {setup : Array Fq} (expanded : pub.setupFor F.setupCount = some setup)
    (β : Module.Basis (Fin Key20.level₁.opening.planes) Zq E)
    (basis : ∀ t, β t = (extBasis.getD t Ext.zero).into)
    (challenge : Fin Key20.level₁.shape.blocks → Rq Key20.level₁.opening.carrierDegree)
    (challenged : ∀ i, challenge i = Poly.toRq _
      (((drawnIn (stateBefore pub proof messages ops c) level 0).map fun c =>
        Poly.ofSparse Key20.level₁.opening.carrierDegree (c.map fun x =>
          (x.1 / (Key20.level₁.shape.innerDegree / Key20.level₁.opening.carrierDegree), x.2))).getD
        i (Poly.zero Key20.level₁.opening.carrierDegree)))
    {t c' : ℕ} {key' : Key}
    (tensorAt : ops[t]? = some (.check key' (.tensor (level + 1) Key20.level₁.tail
      ⟨level, "fused.point", none⟩ ⟨level, "witness_evaluation", none⟩)))
    (rounds : roundsAt (ops.zipIdx.toList.take t) ⟨level + 1, "tensor.sumcheck", none⟩
      Key20.level₁.tail 2 = true)
    (before : c ≤ t) (after : t ≤ c')
    (held : frees ((ops.zipIdx.toList.take t).drop c)
      [⟨level, "fused.point", none⟩, ⟨level, "witness_evaluation", none⟩] [] = true)
    (kept : frees ((ops.zipIdx.toList.take c').drop t) (tensorKeys level)
      [⟨level + 1, "tensor.sumcheck", none⟩] = true)
    (w : Table) :
    ∃ (fits : Key20.level₁.opening.cells Key20.level₁.fold ≤ 2 ^ F.vars)
      (room : Key20.level₁.opening.quotientRoom Key20.level₁.fold Key20.quotientDigits ≤
        2 ^ F.vars)
      (rowsFit : Key20.level₁.opening.rowCount + 1 ≤ 2 ^ F.batch),
      let s := stateBefore pub proof messages ops c
      let s' := stateBefore pub proof messages ops c'
      let r2 := (recordedIn s' ⟨level, "fused.point", none⟩).toList
      let ρ := (recordedIn s' ⟨level + 1, "tensor.sumcheck.point", none⟩).toList
      let η := firstIn s' ⟨level + 1, "tensor.eta", none⟩
      let final := firstIn s' ⟨level + 1, "tensor.final_claims", none⟩
      ρ.length = Key20.level₁.tail ∧ r2.tail.length = Key20.level₁.tail ∧
        ((foldCheck (liftedReduction β (packStatementFor Key20.level₁.opening setup
            (claimPointIn s level 0) (dataIn pub s level) .witness (rowsIn s level F)
            (claimsIn s level F)) challenge F.vars fits Key20.quotientDigits room F.batch rowsFit)
            Key20.level₁.shape.certifiedBase
            (Key20.level₁.opening.gaps Key20.level₁.fold Key20.quotientDigits).support
            (Key20.level₁.opening.compressed Key20.level₁.fold) w).bind fun
          | none => .pure none
          | some (r2, v) => tensorCheck Key20.level₁.tail w r2 v).Runs
          (foldPathIn E pub s level F setup F.vars F.batch ++
            tensoredPath E Key20.level₁.tail
              (some (r2, firstIn s' ⟨level, "witness_evaluation", none⟩))
              ((recordedIn s' ⟨level + 1, "tensor.partials", none⟩).getD 0 Ext.zero)
              ((recordedIn s' ⟨level + 1, "tensor.partials", none⟩).getD 1 Ext.zero) η
              (roundsIn s' ⟨level + 1, "tensor.sumcheck", none⟩) final)
          (some (ρ.map Ext.into,
            mle (ρ.map Ext.into) (tensorWeight (r2.tail.map Ext.into) η.into), final.into)) := by
  obtain ⟨fits, room, rowsFit, runs⟩ := verifyProof_key20_level1 pub proof scripted ends
    terminalSetup accepted decoded at_ fold foldShaped game drawShape rfl expanded β basis challenge
    challenged w
  exact ⟨fits, room, rowsFit, verifyProof_tensored pub proof scripted ends terminalSetup accepted
    decoded tensorAt rounds before after held kept w runs⟩

/-- Level 2 of `key20.aks` with the tensor reduction of the level 3 on its claim
(`Tier.tensored (committedTrace …) level₂.tail`, the notation `second`): when `verifyProof`
accepts, `foldCheck` of `traceReduction` of the level and then `tensorCheck` run to `(ρ, θ, ṽ)` of
the recorded values on the state before `c'`. The numbers of the layout are theorems
(`layout20_trace`). -/
theorem verifyProof_key20_trace_tensored {zMax rice : ℕ}
    (scripted : script pub.layout pub.plan = .ok ops)
    (ends : endsFit ops.zipIdx.toList = true) {full : Array Fq}
    (terminalSetup : pub.setupFor pub.layout.terminal.setupCount = some full)
    (accepted : verifyProof pub proof = .verdict .accept) {messages : Array Message} {rest : ℕ}
    (decoded : decode ops proof = .ok (messages, rest)) {c : ℕ} {key : Key} {F : Fold}
    (at_ : ops[c]? = some (.check key (.fused 2 F)))
    (fold : (layout20 zMax rice).folds[2]? = some F) {r a e : ℕ}
    (foldShaped : foldShape (ops.zipIdx.toList.take c) 2 F r a e = true)
    (game : gameShape (ops.zipIdx.toList.take c) 2 F = true)
    {setup : Array Fq} (expanded : pub.setupFor F.setupCount = some setup)
    (quarter : 4 * (Key20.level₂.shape.innerDegree / 4) = Key20.level₂.shape.innerDegree)
    (challenge : Fin Key20.level₂.shape.blocks → Rq Key20.level₂.shape.innerDegree)
    (challenged : ∀ i, challenge i = Poly.toRq Key20.level₂.shape.innerDegree
      (((drawnIn (stateBefore pub proof messages ops c) 2 0).map
        (Poly.ofSparse Key20.level₂.shape.innerDegree)).getD i
          (Poly.zero Key20.level₂.shape.innerDegree)))
    {t c' : ℕ} {key' : Key}
    (tensorAt : ops[t]? = some (.check key' (.tensor 3 Key20.level₂.tail
      ⟨2, "fused.point", none⟩ ⟨2, "witness_evaluation", none⟩)))
    (rounds : roundsAt (ops.zipIdx.toList.take t) ⟨3, "tensor.sumcheck", none⟩
      Key20.level₂.tail 2 = true)
    (before : c ≤ t) (after : t ≤ c')
    (held : frees ((ops.zipIdx.toList.take t).drop c)
      [⟨2, "fused.point", none⟩, ⟨2, "witness_evaluation", none⟩] [] = true)
    (kept : frees ((ops.zipIdx.toList.take c').drop t) (tensorKeys 2)
      [⟨3, "tensor.sumcheck", none⟩] = true)
    (w : Table) :
    ∃ (fits : Key20.level₂.opening.cells Key20.level₂.fold ≤ 2 ^ F.vars)
      (rowsFit : Key20.level₂.opening.traceRowCount + 1 ≤ 2 ^ F.batch),
      let s := stateBefore pub proof messages ops c
      let s' := stateBefore pub proof messages ops c'
      let r2 := (recordedIn s' ⟨2, "fused.point", none⟩).toList
      let ρ := (recordedIn s' ⟨3, "tensor.sumcheck.point", none⟩).toList
      let η := firstIn s' ⟨3, "tensor.eta", none⟩
      let final := firstIn s' ⟨3, "tensor.final_claims", none⟩
      ρ.length = Key20.level₂.tail ∧ r2.tail.length = Key20.level₂.tail ∧
        ((foldCheck (traceReduction (traceStatementFor (E := E) Key20.level₂.opening quarter setup
            (traceValuesIn s 2 F) (dataIn pub s 2) .witness (rowsIn s 2 F) (claimsIn s 2 F))
            challenge F.vars fits F.batch rowsFit) Key20.level₂.shape.certifiedBase
            (Key20.level₂.opening.binarySupport Key20.level₂.fold)
            (Key20.level₂.opening.compressed Key20.level₂.fold) w).bind fun
          | none => .pure none
          | some (r2, v) => tensorCheck Key20.level₂.tail w r2 v).Runs
          (foldPathIn E pub s 2 F setup F.vars F.batch ++
            tensoredPath E Key20.level₂.tail (some (r2, firstIn s' ⟨2, "witness_evaluation", none⟩))
              ((recordedIn s' ⟨3, "tensor.partials", none⟩).getD 0 Ext.zero)
              ((recordedIn s' ⟨3, "tensor.partials", none⟩).getD 1 Ext.zero) η
              (roundsIn s' ⟨3, "tensor.sumcheck", none⟩) final)
          (some (ρ.map Ext.into,
            mle (ρ.map Ext.into) (tensorWeight (r2.tail.map Ext.into) η.into), final.into)) := by
  obtain ⟨shaped, rows, based, low, vars, batch, compressed, arities, leaf, route, -, two, fits,
    rowsFit, -, -⟩ := layout20_trace (L := Key20.level₂) List.mem_cons_self fold
  refine ⟨vars ▸ fits, batch ▸ rowsFit, ?_⟩
  exact verifyProof_tensored pub proof scripted ends terminalSetup accepted decoded tensorAt rounds
    before after held kept w (verifyProof_trace_game pub proof scripted ends terminalSetup accepted
      decoded at_ foldShaped game rfl shaped expanded rows based low quarter two challenge
      challenged (vars ▸ fits) (batch ▸ rowsFit) w route compressed arities leaf)

/-- The levels 3 to 5 of `key20.aks` (the Euclidean route) with the tensor reduction of the next
level on their claim (`Tier.tensored (committedTraceSq …) L.tail`, the notations `third` to
`fifth`; the tensor reduction of the level 5 is the one of the terminal): when `verifyProof`
accepts, `foldCheckSq` of `traceReduction` of the level and then `tensorCheck` run to `(ρ, θ, ṽ)`
of the recorded values on the state before `c'`. The numbers of the layout are theorems
(`layout20_trace`). -/
theorem verifyProof_key20_traceSq_tensored {zMax rice : ℕ}
    (scripted : script pub.layout pub.plan = .ok ops)
    (ends : endsFit ops.zipIdx.toList = true) {full : Array Fq}
    (terminalSetup : pub.setupFor pub.layout.terminal.setupCount = some full)
    (accepted : verifyProof pub proof = .verdict .accept) {messages : Array Message} {rest : ℕ}
    (decoded : decode ops proof = .ok (messages, rest)) {level : ℕ} {L : Key20.Level}
    (sender : (level, L) ∈ [(3, Key20.level₃), (4, Key20.level₄), (5, Key20.level₅)])
    {c : ℕ} {key : Key} {F : Fold} (at_ : ops[c]? = some (.check key (.fused level F)))
    (fold : (layout20 zMax rice).folds[level]? = some F) {r a e : ℕ}
    (foldShaped : foldShape (ops.zipIdx.toList.take c) level F r a e = true)
    (game : gameShape (ops.zipIdx.toList.take c) level F = true)
    {setup : Array Fq} (expanded : pub.setupFor F.setupCount = some setup)
    (quarter : 4 * (L.shape.innerDegree / 4) = L.shape.innerDegree)
    (challenge : Fin L.shape.blocks → Rq L.shape.innerDegree)
    (challenged : ∀ i, challenge i = Poly.toRq L.shape.innerDegree
      (((drawnIn (stateBefore pub proof messages ops c) level 0).map
        (Poly.ofSparse L.shape.innerDegree)).getD i (Poly.zero L.shape.innerDegree)))
    {t c' : ℕ} {key' : Key}
    (tensorAt : ops[t]? = some (.check key' (.tensor (level + 1) L.tail
      ⟨level, "fused.point", none⟩ ⟨level, "witness_evaluation", none⟩)))
    (rounds : roundsAt (ops.zipIdx.toList.take t) ⟨level + 1, "tensor.sumcheck", none⟩ L.tail 2 =
      true)
    (before : c ≤ t) (after : t ≤ c')
    (held : frees ((ops.zipIdx.toList.take t).drop c)
      [⟨level, "fused.point", none⟩, ⟨level, "witness_evaluation", none⟩] [] = true)
    (kept : frees ((ops.zipIdx.toList.take c').drop t) (tensorKeys level)
      [⟨level + 1, "tensor.sumcheck", none⟩] = true)
    (w : Table) :
    ∃ (fits : L.opening.cells L.fold ≤ 2 ^ F.vars)
      (rowsFit : L.opening.traceRowCount + 1 ≤ 2 ^ F.batch),
      let s := stateBefore pub proof messages ops c
      let s' := stateBefore pub proof messages ops c'
      let r2 := (recordedIn s' ⟨level, "fused.point", none⟩).toList
      let ρ := (recordedIn s' ⟨level + 1, "tensor.sumcheck.point", none⟩).toList
      let η := firstIn s' ⟨level + 1, "tensor.eta", none⟩
      let final := firstIn s' ⟨level + 1, "tensor.final_claims", none⟩
      ρ.length = L.tail ∧ r2.tail.length = L.tail ∧
        ((foldCheckSq (traceReduction (traceStatementFor (E := E) L.opening quarter setup
            (traceValuesIn s level F) (dataIn pub s level) .witness (rowsIn s level F)
            (claimsIn s level F)) challenge F.vars fits F.batch rowsFit) L.shape.certifiedBase
            (L.opening.binarySupport L.fold) (L.opening.compressed L.fold) F.limit
            (L.shape.normKernel L.fold) w).bind fun
          | none => .pure none
          | some (r2, v) => tensorCheck L.tail w r2 v).Runs
          (foldPathIn E pub s level F setup F.vars F.batch ++
            tensoredPath E L.tail (some (r2, firstIn s' ⟨level, "witness_evaluation", none⟩))
              ((recordedIn s' ⟨level + 1, "tensor.partials", none⟩).getD 0 Ext.zero)
              ((recordedIn s' ⟨level + 1, "tensor.partials", none⟩).getD 1 Ext.zero) η
              (roundsIn s' ⟨level + 1, "tensor.sumcheck", none⟩) final)
          (some (ρ.map Ext.into,
            mle (ρ.map Ext.into) (tensorWeight (r2.tail.map Ext.into) η.into), final.into)) := by
  have member : (level, L) ∈ [(2, Key20.level₂), (3, Key20.level₃), (4, Key20.level₄),
      (5, Key20.level₅)] := List.mem_cons_of_mem _ sender
  have deep : 2 < level := by
    simp only [List.mem_cons, Prod.mk.injEq, List.not_mem_nil, or_false] at sender
    omega
  obtain ⟨shaped, rows, -, low, vars, batch, compressed, arities, leaf, route, small, two, fits,
    rowsFit, positive, digits⟩ := layout20_trace member fold
  refine ⟨vars ▸ fits, batch ▸ rowsFit, ?_⟩
  exact verifyProof_tensored pub proof scripted ends terminalSetup accepted decoded tensorAt rounds
    before after held kept w (verifyProof_traceSq_game pub proof scripted ends terminalSetup
      accepted decoded at_ foldShaped game rfl shaped expanded rows low leaf quarter two challenge
      challenged (vars ▸ fits) (batch ▸ rowsFit) w positive small digits
      (route.trans (decide_eq_true deep)) compressed arities)

/-- `Tier.TraceLinked` from the level `level` (`L`) to the level `level + 1` (`L'`) of
`key20.aks` on the replay (Akita Figure 4, steps 0, 3 and 5): when `verifyProof` accepts, the
tensor reduction of the level `level + 1` at `t` has the rounds `L.tail`, and the fold of the
level `level + 1` is the fold of `layout20`, on the state before `c'` the statement
`traceStatementFor` of the level `level + 1` is the folding statement of `TraceClaim.ofPoint` for
the segments `ρ = ρ_pack ++ (ρ_pos ++ ρ_blk)` of the tensor point, the factor `θ`, the final claim
`ṽ` and the payload of the words `next_witness` of the level `level`. -/
theorem verifyProof_key20_traceLink {zMax rice : ℕ}
    (scripted : script pub.layout pub.plan = .ok ops)
    (ends : endsFit ops.zipIdx.toList = true) {full : Array Fq}
    (terminalSetup : pub.setupFor pub.layout.terminal.setupCount = some full)
    (accepted : verifyProof pub proof = .verdict .accept) {messages : Array Message} {rest : ℕ}
    (decoded : decode ops proof = .ok (messages, rest)) {level t c' : ℕ} {key : Key}
    {L L' : Key20.Level}
    (pair : (level, L, L') ∈ [(1, Key20.level₁, Key20.level₂), (2, Key20.level₂, Key20.level₃),
      (3, Key20.level₃, Key20.level₄), (4, Key20.level₄, Key20.level₅)])
    (at_ : ops[t]? = some (.check key (.tensor (level + 1) L.tail ⟨level, "fused.point", none⟩
      ⟨level, "witness_evaluation", none⟩)))
    (shape : roundsAt (ops.zipIdx.toList.take t) ⟨level + 1, "tensor.sumcheck", none⟩ L.tail 2 =
      true)
    (after : t ≤ c')
    (kept : frees ((ops.zipIdx.toList.take c').drop t) (tensorKeys level)
      [⟨level + 1, "tensor.sumcheck", none⟩] = true)
    {F : Fold} (fold : (layout20 zMax rice).folds[level + 1]? = some F)
    (quarter : 4 * (L'.shape.innerDegree / 4) = L'.shape.innerDegree) (setup : Array Fq) :
    let s := stateBefore pub proof messages ops c'
    let r2 := (recordedIn s ⟨level, "fused.point", none⟩).toList
    let ρ := (recordedIn s ⟨level + 1, "tensor.sumcheck.point", none⟩).toList
    let η := firstIn s ⟨level + 1, "tensor.eta", none⟩
    let pack := (packPoint L'.shape ρ).map (Ext.into (E := E))
    let position := (positionPoint L'.shape ρ).map (Ext.into (E := E))
    let block := (blockPoint L'.shape ρ).map (Ext.into (E := E))
    pack ++ (position ++ block) = ρ.map Ext.into ∧ L'.shape.blocks ≤ 2 ^ block.length ∧
      L'.shape.blockLength = 2 ^ position.length ∧
      2 * (L'.shape.innerDegree / 4) = 2 ^ pack.length ∧
      traceStatementFor (E := E) L'.opening quarter setup (traceValuesIn s (level + 1) F)
          (dataIn pub s (level + 1)) .witness (rowsIn s (level + 1) F)
          (claimsIn s (level + 1) F) =
        (Extract.TraceClaim.ofPoint quarter
          (payloadOf L'.shape.chain _ _ ((s.fields.get? ⟨level, "next_witness", none⟩).getD #[]))
          (payloadOf L'.opening.chain _ _
            ((s.fields.get? ⟨level + 1, "opening_payload", none⟩).getD #[]))
          block position pack (mle (ρ.map Ext.into) (tensorWeight (r2.tail.map Ext.into) η.into))
          (firstIn s ⟨level + 1, "tensor.final_claims", none⟩).into).statement
          (Key.ofSetup (flatOf setup) L'.shape) (OpeningKey.ofSetup (flatOf setup) L'.opening) := by
  obtain ⟨traced, single, -⟩ := layout20_receiver pair fold
  have numbers : ∀ x ∈ [(L'.shape, L.tail)],
      4 * (x.1.innerDegree / 4) = x.1.innerDegree ∧
      2 * (x.1.innerDegree / 4) = 2 ^ Nat.log 2 (x.1.innerDegree / 2) ∧
      x.1.blockLength = 2 ^ Nat.log 2 x.1.blockLength ∧
      Nat.log 2 (x.1.innerDegree / 2) + Nat.log 2 x.1.blockLength ≤ x.2 ∧
      x.1.blocks ≤ 2 ^ (x.2 - (Nat.log 2 (x.1.innerDegree / 2) + Nat.log 2 x.1.blockLength)) := by
    intro x member
    refine key20_segments x ?_
    simp only [List.mem_cons, List.not_mem_nil, or_false] at member pair ⊢
    rcases pair with ⟨rfl, rfl, rfl⟩ | ⟨rfl, rfl, rfl⟩ | ⟨rfl, rfl, rfl⟩ | ⟨rfl, rfl, rfl⟩ <;>
      simp [member]
  obtain ⟨-, width, length, long, blocks⟩ := numbers _ List.mem_cons_self
  exact verifyProof_traceLink pub proof scripted ends terminalSetup accepted decoded at_ shape
    after kept (by rw [traced]; rfl) single L'.opening quarter width length long blocks setup

end Proof

end NightstreamFPrime.Spec.Compression.Akita.Replay
