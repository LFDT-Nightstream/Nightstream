import NightstreamFPrime.Spec.Compression.Akita.Replay.LevelFold
import NightstreamFPrime.Spec.Compression.Akita.Replay.WeightPacking
import NightstreamFPrime.Spec.Compression.Akita.Replay.Stage3

/-! One fold of the replay at a level with coefficient packing and at an offloaded level, against
the checks of the specification (Akita revision `d50f700`: `verify_fold` of `fold/mod.rs`,
`verify_stage3` of `stages/stage3.rs` before `validate_stage2_replay`; `foldCheck` of
`Akita/FoldCheck.lean` for `liftFold` and `liftBatch`, `offloadCheck` of `Extract/Offload.lean`
for `offBatch`).

- `packFinish` gives the values of the end check of a packing level at a point (`foldFused`),
  `packClaim` the claim of `stage2`, and `packFoldRun` is `foldRun` with them.
- `packFoldRun_runs` and `batchFoldRun_runs`: `foldCheck` for `liftedReduction` of
  `packStatementFor` and for `Batch.liftedReduction` of `batchStatementFor` ends in the result of
  `packFoldRun`. The weight of `foldFused` at `r_2` (`weighs`) and its binary-support weight
  (`supported`) are hypotheses: the weight theorem of the packing levels gives them.
- `offloadRun` is the range proof, the fused sum-check with the value `σ` of Stage 3 in the
  weight, and Stage 3 (`stage3Run`). The code checks `stage3.end` before `fused.end`, and
  `offloadCheck` checks the fused end before Stage 3: both reject when one check fails, so the
  result is the same, and the path has the moves of Stage 3 only when the fused end holds.
- `offloadRun_runs`: for each reduction `R` and each split `scan` (132), `offloadCheck` ends in
  the result of `offloadRun`, when the weight of the values is `m_local(r_2) + scale·σ` and the
  weight of Stage 3 is the multilinear extension of `scan.weight`. `batchOffloadRun_runs` is it for
  `Batch.liftedReduction` of `batchStatementFor` over the setup prefix `flat` (the root of
  `key30.aks`; `batch_fused_over`: the claim of Stage 2 does not read the matrices). -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

open NightstreamFPrime.Spec.Compression.Lib NightstreamFPrime.Spec.Compression.Game
open NightstreamFPrime.Spec.Compression (Game)
open NightstreamFPrime.Spec.Compression.Akita

/-- The values of the end check of the fused sum-check of a packing level at the point `r_2`:
`foldFused` of the values `V` of the replay with the point `r_2`. -/
def packFinish (F : FoldInputs) (gs : List PackGroup) (V : PackingValues)
    (setupClaim : Option Ext) (r2 : List Ext) : FusedEnd :=
  foldFused F gs { V with r2 } setupClaim

/-- The values of the checks of a fold for the values `V` of the end of the fused sum-check of the
replay at a packing level and the other values of the fold. -/
def PackingValues.words (V : PackingValues) (tau0 : List Ext) (stages : List Product)
    (image : Ext) (rounds : List (List Ext × Ext)) : FoldWords :=
  ⟨V.alpha, tau0, V.tau1, stages, image, V.binary, V.batch, rounds⟩

/-- The input claim of the fused sum-check of a packing level (`Replayer.stage2Of`): `stage2`
with `rest = γ·s_claim`. -/
def packClaim (V : PackingValues) (data : RhsRun → Array Fq) (runs : List RhsRun) (opening : ℕ)
    (coefficients claims : Array Ext) (image : Ext) : Ext :=
  (stage2 V.tau1 V.alpha data runs opening coefficients claims (V.batch * image + Ext.zero)).fused

/-- The fold of a packing level: `foldRun` with the claim of `stage2` and the end check
`foldFused` of the values `V` of the replay. -/
def packFoldRun (F : FoldInputs) (gs : List PackGroup) (V : PackingValues) (b low : ℕ)
    (tau0 : List Ext) (stages : List Product) (leaf : List (List Ext × Ext)) (image : Ext)
    (rounds : List (List Ext × Ext)) (data : RhsRun → Array Fq) (runs : List RhsRun)
    (opening : ℕ) (coefficients claims : Array Ext) : Option (List Ext × Ext) :=
  foldRun b low (V.words tau0 stages image rounds) leaf
    (packClaim V data runs opening coefficients claims image) (packFinish F gs V none)

/-- One offloaded fold: the range proof, the fused sum-check from `claim` with the end check of
`finish`, and Stage 3 from `σ` with its rounds, the prefix evaluation and the weight
`setupWeight r_2` of the code at the point `r_2` of the fused sum-check. -/
def offloadRun (b low : ℕ) (W : FoldWords) (leaf : List (List Ext × Ext)) (claim : Ext)
    (finish : List Ext → FusedEnd) (σ : Ext) (setupRounds : List (List Ext × Ext))
    (prefixValue : Ext) (setupWeight : List Ext → List Ext → Ext) :
    Option ((List Ext × Ext) × (List Ext × Ext)) :=
  match rangeRun b W.stages leaf W.image (W.point low) with
  | none => none
  | some _ => (fusedRun claim W.rounds finish).bind fun c =>
      (stage3Run σ setupRounds prefixValue (setupWeight c.1)).map fun s => (c, s)

variable (E : Type) [Field E] [Algebra Zq E] [Degree2 E] in
/-- The path of `offloadCheck` for the values of the code. -/
noncomputable def offloadPath (vars batch low b : ℕ) (bin : Bool) (W : FoldWords)
    (leaf : List (List Ext × Ext)) (claim : Ext) (finish : List Ext → FusedEnd) (σ : Ext)
    (setupRounds : List (List Ext × Ext)) (prefixValue : Ext) : List Move :=
  Move.draw (W.alpha.into : E) :: Move.draw (listFn vars (W.tau0.map Ext.into) : Fin vars → E) ::
    Move.draw (listFn batch (W.tau1.map Ext.into) : Fin batch → E) ::
      (treePath (Ext.into (E := E))
          (fun _ _ _ _ => leafPath (Ext.into (E := E)) (rangeLeaf b) leaf W.image)
          W.stages 1 (fun _ => Ext.one) (W.point low) Ext.zero ++
        match rangeRun b W.stages leaf W.image (W.point low) with
        | none => []
        | some _ => binPath bin (W.binary.into : E) ++ Move.draw (W.batch.into : E) ::
            (plainPath (Ext.into (E := E)) 2 claim W.rounds ++
              Move.recv ((finish (W.rounds.map (·.2))).value.into : E) :: Move.recv (σ.into : E) ::
                match fusedRun claim W.rounds finish with
                | none => []
                | some _ => stage3Path E σ setupRounds prefixValue))

variable {E : Type} [Field E] [Fintype E] [DecidableEq E] [Algebra Zq E] [Degree2 E]

omit [Fintype E] [DecidableEq E] in
/-- `supports` of `fusedRun_runs` for values whose virtual weight is `γ·ẽq(r_virt, r_2) + ζ·s`,
with `ζ = s = 0` at a level without compression digits, when `s` is the binary-support weight at
`r_2` at a level with them. -/
theorem supports_of (support : ℕ → Prop) [DecidablePred support] (f : FusedEnd)
    (virtual p : List Ext) (bin : Bool) (ζ s : Ext) (length : p.length = virtual.length)
    (virt : f.virt = eqPoints virtual p) (binary : f.binary = if bin then ζ else Ext.zero)
    (supported : f.support = if bin then s else Ext.zero)
    (value : bin = true → (s.into : E) =
      mle (p.map Ext.into) fun x => if support x then eqAt (virtual.map Ext.into) x else 0) :
    (f.batch.into * f.virt.into + f.binary.into * f.support.into : E) =
      mle (p.map Ext.into) (virtWeight support (virtual.map Ext.into) f.batch.into
        (if bin then ζ.into else 0)) := by
  have split : mle (p.map Ext.into) (virtWeight support (virtual.map Ext.into) f.batch.into
      (if bin then ζ.into else 0)) =
      (f.batch.into : E) * mle (p.map Ext.into) (eqAt (virtual.map Ext.into)) +
        (if bin then ζ.into else 0) * mle (p.map Ext.into)
          (fun x => if support x then eqAt (virtual.map Ext.into) x else 0) := by
    rw [show virtWeight support (virtual.map Ext.into) f.batch.into (if bin then ζ.into else 0) =
      fun x => (f.batch.into : E) * eqAt (virtual.map Ext.into) x +
        (if bin then ζ.into else 0) * (if support x then eqAt (virtual.map Ext.into) x else 0)
      from rfl, mle_add, mle_mul_left, mle_mul_left]
  rw [split, virt, eqPoints_into, mle_eqAt _ _ (by rw [List.length_map, List.length_map, length])]
  congr 1
  rw [binary, supported]
  cases bin
  · simp only [Bool.false_eq_true, ite_false, Ext.into_zero, zero_mul]
  · simp only [ite_true]
    rw [value rfl]

section Packing

variable {S : Shape} {O : OpeningShape S} {fold : Descriptor}

/-- A packing level with one group: `foldCheck` for `liftedReduction` of the statement of the
words of the replay, on the path of the values of the code, ends in the result of `packFoldRun`.
The values `V` are those of `fused.end` (`packingValuesIn`), `leaf` the rounds of `range.leaf`,
whose challenges are the point `r_virt` of `V`. `weighs` and `supported` are the weight and the
binary-support weight of `foldFused` at `r_2`. -/
theorem packFoldRun_runs (β : Module.Basis (Fin O.planes) Zq E) (setup : Array Fq)
    (point : List Ext) (F : FoldInputs) (gs : List PackGroup) (V : PackingValues)
    (tau0 : List Ext) (stages : List Product) (leaf : List (List Ext × Ext)) (image : Ext)
    (rounds : List (List Ext × Ext)) (data : RhsRun → Array Fq) (source : RhsSource)
    (coefficients claims : Array Ext) (challenge : Fin S.blocks → Rq O.carrierDegree) {μ : ℕ}
    (fits : O.cells fold ≤ 2 ^ μ) {digits : ℕ} (room : O.quotientRoom fold digits ≤ 2 ^ μ)
    {batch : ℕ} (rowsFit : O.rowCount + 1 ≤ 2 ^ batch) (w : Table)
    (compressed : V.compressed = O.compressed fold) (virtual : V.virtual = leaf.map (·.2))
    (vars : tau0.length = μ) (tau : V.tau1.length = batch)
    (arities : stages.map (·.arity) = rangeArities S.certifiedBase)
    (shaped : ∀ P ∈ stages, P.rounds.length = μ ∧ ∀ x ∈ P.rounds, x.1.length = P.arity)
    (count : leaf.length = μ) (lengths : ∀ x ∈ leaf, x.1.length = rangeLeaf S.certifiedBase)
    (fusedCount : rounds.length = μ) (fusedLengths : ∀ x ∈ rounds, x.1.length = 3)
    (weighs : ∀ p : List Ext, p.length = μ → ((packFinish F gs V none p).weight.into : E) =
      mle (p.map Ext.into) (((liftedReduction β (packStatementFor O setup point data source
        coefficients claims) challenge μ fits digits room batch rowsFit).rows
          V.alpha.into).batchWeight (V.tau1.map Ext.into)))
    (supported : ∀ p : List Ext, p.length = μ → O.compressed fold = true →
      ((foldSupport (Placement.of gs F.path F.digits) (CellEq.half V.virtual).get
        (CellEq.half p).get).into : E) = mle (p.map Ext.into) fun x =>
          if (O.gaps fold digits).support x then eqAt (V.virtual.map Ext.into) x else 0) :
    (foldCheck (liftedReduction β (packStatementFor O setup point data source coefficients
        claims) challenge μ fits digits room batch rowsFit) S.certifiedBase
        (O.gaps fold digits).support (O.compressed fold) w).Runs
      (foldPath E μ batch O.low S.certifiedBase (O.compressed fold)
        (V.words tau0 stages image rounds) leaf (packClaim V data
          (relation (fun _ : Fin 1 => S) O.openRank O.openDegree O.chain fun _ => source).1
          (relation (fun _ : Fin 1 => S) O.openRank O.openDegree O.chain fun _ => source).2
          coefficients claims image) (packFinish F gs V none))
      ((packFoldRun F gs V S.certifiedBase O.low tau0 stages leaf image rounds data
        (relation (fun _ : Fin 1 => S) O.openRank O.openDegree O.chain fun _ => source).1
        (relation (fun _ : Fin 1 => S) O.openRank O.openDegree O.chain fun _ => source).2
        coefficients claims).map fun o => (o.1.map Ext.into, o.2.into)) := by
  have virtualLength : V.virtual.length = μ := by rw [virtual, List.length_map, count]
  refine foldRun_runs (liftedReduction β (packStatementFor O setup point data source coefficients
    claims) challenge μ fits digits room batch rowsFit) S.certifiedBase (O.gaps fold digits).support
    (O.compressed fold) w (V.words tau0 stages image rounds) leaf _ (packFinish F gs V none) vars
    tau arities shaped count lengths fusedCount fusedLengths ?_ (fun p length => ?_)
    (fun p length => ?_)
  · rw [packClaim, packing_fused_for β setup point data source coefficients claims _ challenge μ fits
      digits room batch rowsFit V.tau1 V.alpha tau, Ext.into_add, Ext.into_mul, Ext.into_zero,
      add_zero]
    rfl
  · show ((packFinish F gs V none p).weight.into + (Ext.zero.into : E) * Ext.zero.into : E) = _
    rw [Ext.into_zero, zero_mul, add_zero]
    exact weighs p length
  · rw [← virtual]
    refine supports_of (O.gaps fold digits).support (packFinish F gs V none p) V.virtual p
      (O.compressed fold) V.binary _ (length.trans virtualLength.symm) rfl (by rw [← compressed]; rfl)
      (by rw [← compressed]; rfl) fun bin => ?_
    exact supported p length bin

end Packing

section Groups

variable {n : ℕ} {B : Batch (Fin n)}

/-- A fold with opening groups: `foldCheck` for `Batch.liftedReduction` of the statement of the
words of the replay, on the path of the values of the code, ends in the result of `packFoldRun`.
`weighs` and `supported` are the weight and the binary-support weight of `foldFused` at
`r_2`. -/
theorem batchFoldRun_runs (β : ∀ g, Module.Basis (Fin (B.planes g)) Zq E) (setup : Array Fq)
    (points : ℕ → List Ext) (F : FoldInputs) (gs : List PackGroup) (V : PackingValues)
    (tau0 : List Ext) (stages : List Product) (leaf : List (List Ext × Ext)) (image : Ext)
    (rounds : List (List Ext × Ext)) (data : RhsRun → Array Fq) (source : Fin n → RhsSource)
    (coefficients claims : Array Ext) (challenge : B.Challenge) {μ : ℕ} (fits : B.cells ≤ 2 ^ μ)
    (base : ℕ) {digits : ℕ} (room : B.quotientRoom digits ≤ 2 ^ μ) {batch : ℕ}
    (rowsFit : B.rowCount + 1 ≤ 2 ^ batch) (w : Table) (compressed : V.compressed = B.compressed)
    (virtual : V.virtual = leaf.map (·.2)) (vars : tau0.length = μ) (tau : V.tau1.length = batch)
    (arities : stages.map (·.arity) = rangeArities base)
    (shaped : ∀ P ∈ stages, P.rounds.length = μ ∧ ∀ x ∈ P.rounds, x.1.length = P.arity)
    (count : leaf.length = μ) (lengths : ∀ x ∈ leaf, x.1.length = rangeLeaf base)
    (fusedCount : rounds.length = μ) (fusedLengths : ∀ x ∈ rounds, x.1.length = 3)
    (weighs : ∀ p : List Ext, p.length = μ → ((packFinish F gs V none p).weight.into : E) =
      mle (p.map Ext.into) (((Batch.liftedReduction β (batchStatementFor B setup points data
        source coefficients claims) challenge μ fits base digits room batch rowsFit).rows
          V.alpha.into).batchWeight (V.tau1.map Ext.into)))
    (supported : ∀ p : List Ext, p.length = μ → B.compressed = true →
      ((foldSupport (Placement.of gs F.path F.digits) (CellEq.half V.virtual).get
        (CellEq.half p).get).into : E) = mle (p.map Ext.into) fun x =>
          if (B.gaps digits).support x then eqAt (V.virtual.map Ext.into) x else 0) :
    (foldCheck (Batch.liftedReduction β (batchStatementFor B setup points data source
        coefficients claims) challenge μ fits base digits room batch rowsFit) base
        (B.gaps digits).support B.compressed w).Runs
      (foldPath E μ batch B.low base B.compressed (V.words tau0 stages image rounds) leaf
        (packClaim V data
          (relation (fun k => B.shape (groupOrder n k)) B.openRank B.openDegree B.chain source).1
          (relation (fun k => B.shape (groupOrder n k)) B.openRank B.openDegree B.chain source).2
          coefficients claims image) (packFinish F gs V none))
      ((packFoldRun F gs V base B.low tau0 stages leaf image rounds data
        (relation (fun k => B.shape (groupOrder n k)) B.openRank B.openDegree B.chain source).1
        (relation (fun k => B.shape (groupOrder n k)) B.openRank B.openDegree B.chain source).2
        coefficients claims).map fun o => (o.1.map Ext.into, o.2.into)) := by
  have virtualLength : V.virtual.length = μ := by rw [virtual, List.length_map, count]
  refine foldRun_runs (Batch.liftedReduction β (batchStatementFor B setup points data source
    coefficients claims) challenge μ fits base digits room batch rowsFit) base
    (B.gaps digits).support B.compressed w (V.words tau0 stages image rounds) leaf _
    (packFinish F gs V none) vars tau arities shaped count lengths fusedCount fusedLengths ?_
    (fun p length => ?_) (fun p length => ?_)
  · rw [packClaim, batch_fused_for β setup points data source coefficients claims _ challenge μ
      fits base digits room batch rowsFit V.tau1 V.alpha tau, Ext.into_add, Ext.into_mul,
      Ext.into_zero, add_zero]
    rfl
  · show ((packFinish F gs V none p).weight.into + (Ext.zero.into : E) * Ext.zero.into : E) = _
    rw [Ext.into_zero, zero_mul, add_zero]
    exact weighs p length
  · rw [← virtual]
    refine supports_of (B.gaps digits).support (packFinish F gs V none p) V.virtual p
      B.compressed V.binary _ (length.trans virtualLength.symm) rfl (by rw [← compressed]; rfl)
      (by rw [← compressed]; rfl) fun bin => ?_
    exact supported p length bin

end Groups

section Offload

/-- `offloadCheck` of the specification, for each reduction `R` and each split `scan` (132), on
the path of the values of the code ends in the result of `offloadRun`. `weighs`: the weight of the
values of the code at `r_2` is `m_local(r_2) + scale·σ`; `setupWeighs`: the weight of Stage 3 of
the code at `ρ` is the multilinear extension of `scan.weight` at `ρ`. -/
theorem offloadRun_runs (R : Reduction E) (b : ℕ) (support : ℕ → Prop) [DecidablePred support]
    (bin : Bool) (w : Table) (ν : ℕ) (flat : Table) (scan : SetupScan R ν flat) (W : FoldWords)
    (leaf : List (List Ext × Ext)) (claim : Ext) (finish : List Ext → FusedEnd) (σ : Ext)
    (setupRounds : List (List Ext × Ext)) (prefixValue : Ext)
    (setupWeight : List Ext → List Ext → Ext) (vars : W.tau0.length = R.vars)
    (batch : W.tau1.length = R.batch) (arities : W.stages.map (·.arity) = rangeArities b)
    (shaped : ∀ S ∈ W.stages, S.rounds.length = R.vars ∧ ∀ x ∈ S.rounds, x.1.length = S.arity)
    (count : leaf.length = R.vars) (lengths : ∀ x ∈ leaf, x.1.length = rangeLeaf b)
    (fusedCount : W.rounds.length = R.vars) (fusedLengths : ∀ x ∈ W.rounds, x.1.length = 3)
    (setupCount : setupRounds.length = ν) (setupLengths : ∀ x ∈ setupRounds, x.1.length = 2)
    (claimed : (claim.into : E) =
      (R.rows W.alpha.into).batchTarget (W.tau1.map Ext.into) + W.batch.into * W.image.into)
    (weighs : ∀ p : List Ext, p.length = R.vars →
      ((finish p).weight.into + (finish p).norm.into * (finish p).normWeight.into : E) =
        scan.part W.alpha.into (W.tau1.map Ext.into) (p.map Ext.into) +
          scan.scale W.alpha.into (W.tau1.map Ext.into) (p.map Ext.into) * σ.into)
    (supports : ∀ p : List Ext, p.length = R.vars →
      ((finish p).batch.into * (finish p).virt.into + (finish p).binary.into *
        (finish p).support.into : E) = mle (p.map Ext.into) (virtWeight support
          ((leaf.map (·.2)).map Ext.into) W.batch.into (if bin then W.binary.into else 0)))
    (setupWeighs : ∀ r2 ρ : List Ext, r2.length = R.vars → ρ.length = ν →
      ((setupWeight r2 ρ).into : E) = mle (ρ.map Ext.into)
        (scan.weight W.alpha.into (W.tau1.map Ext.into) (r2.map Ext.into))) :
    (offloadCheck R b support bin w ν flat scan).Runs
      (offloadPath E R.vars R.batch R.low b bin W leaf claim finish σ setupRounds prefixValue)
      ((offloadRun b R.low W leaf claim finish σ setupRounds prefixValue setupWeight).map
        fun o => ((o.1.1.map Ext.into, o.1.2.into), (o.2.1.map Ext.into, o.2.2.into))) := by
  have long : (W.point R.low).length = R.vars := by
    rw [FoldWords.point, List.length_rotate, vars]
  have range := rangeRun_runs (degree2_extHom (E := E)) (fun a b => same_degree2 (E := E) a b) b
    R.vars support w W.stages leaf W.image (W.point R.low) arities long shaped count lengths
  have fusedLong : (W.rounds.map (·.2)).length = R.vars := by rw [List.length_map, fusedCount]
  have key := holds_iff_final (E := E) (finish (W.rounds.map (·.2)))
    (plainRun claim W.rounds).claim _ _ (weighs _ fusedLong) (supports _ fusedLong)
  have rounds_run := plainRun_runs (degree2_extHom (E := E)) 2
    (combRound R.vars (fusedTables w ((R.rows W.alpha.into).batchWeight (W.tau1.map Ext.into))
      (virtWeight support ((leaf.map (·.2)).map Ext.into) W.batch.into
        (if bin then W.binary.into else 0))) (fusedComb E)) W.rounds [] claim fusedLengths
  rw [fusedCount, claimed] at rounds_run
  have point : (plainRun claim W.rounds).point = W.rounds.map (·.2) :=
    plainRun_point claim W.rounds
  have stage := stage3Run_runs ν flat (scan.weight W.alpha.into (W.tau1.map Ext.into)
    ((W.rounds.map (·.2)).map Ext.into)) σ setupRounds prefixValue
    (setupWeight (W.rounds.map (·.2))) setupCount setupLengths
    (setupWeighs _ _ fusedLong (by rw [plainRun_point_length, setupCount]))
  unfold offloadCheck offloadPath
  refine runs_draw (runs_draw (runs_draw ?_))
  rw [rangePoint_listFn R.low Ext.into vars]
  refine runs_bind range ?_
  unfold offloadRun
  cases ran : rangeRun b W.stages leaf W.image (W.point R.low) with
  | none => rfl
  | some out =>
      obtain rfl := rangeRun_some ran
      refine runs_drawBin (runs_draw ?_)
      rw [ofFn_listFn (by rw [List.length_map, batch])]
      refine runs_bind rounds_run (runs_recv (runs_recv ?_))
      simp only [List.nil_append, point]
      rw [fusedRun_eq]
      by_cases passes : ((finish (W.rounds.map (·.2))).withClaim
          (plainRun claim W.rounds).claim).holds = true
      · rw [ite_eq_left (key.mp passes), ite_eq_left passes, Option.bind_some]
        refine runs_bind_end stage ?_
        cases stage3Run σ setupRounds prefixValue (setupWeight (W.rounds.map (·.2))) <;> rfl
      · rw [ite_eq_right fun equal => passes (key.mpr equal), ite_eq_right passes]
        rfl

variable {n : ℕ} {B : Batch (Fin n)}

omit [DecidableEq E] in
/-- `batch_fused_for` for the statement over the setup vector `S` (`Batch.Statement.over`): the
targets read the payloads and the value, not the matrices. -/
theorem batch_fused_over (β : ∀ g, Module.Basis (Fin (B.planes g)) Zq E) (setup : Array Fq)
    (points : ℕ → List Ext) (data : RhsRun → Array Fq) (source : Fin n → RhsSource)
    (coefficients claims : Array Ext) (rest : Ext) {N : ℕ} (S : Fin N → Zq)
    (challenge : B.Challenge) (μ : ℕ) (fits : B.cells ≤ 2 ^ μ) (base digits : ℕ)
    (room : B.quotientRoom digits ≤ 2 ^ μ) (batch : ℕ) (rowsFit : B.rowCount + 1 ≤ 2 ^ batch)
    (τ : List Ext) (α : Ext) (length : τ.length = batch) :
    ((stage2 τ α data
        (relation (fun k => B.shape (groupOrder n k)) B.openRank B.openDegree B.chain source).1
        (relation (fun k => B.shape (groupOrder n k)) B.openRank B.openDegree B.chain source).2
        coefficients claims rest).fused.into : E) =
      ((Batch.liftedReduction β ((batchStatementFor B setup points data source coefficients
        claims).over S) challenge μ fits base digits room batch rowsFit).rows α.into).batchTarget
        (τ.map Ext.into) + rest.into :=
  batch_fused β _ challenge μ fits base digits room batch rowsFit τ α data source coefficients
    claims rest length
    (fun k => by simp only [Batch.Statement.over, batchStatementFor, Equiv.symm_apply_apply]) rfl
    rfl

/-- The root of `key30.aks`, an offloaded fold with opening groups: `offloadCheck` for
`Batch.liftedReduction` of the statement of the words of the replay over the setup prefix `flat`
of `2^ν` cells (`Batch.Statement.over`, the statement of `Batch.codeScan`), on the path of the
values of the code, ends in the result of `offloadRun` with the values `σ` and `foldFused` with
the value `σ` (`setupClaimIn`). `weighs`, `supported` and `setupWeighs` are the weight of
`foldFused` at `r_2`, its binary-support weight and the weight of Stage 3 at `ρ`. -/
theorem batchOffloadRun_runs (β : ∀ g, Module.Basis (Fin (B.planes g)) Zq E) (setup : Array Fq)
    (points : ℕ → List Ext) (F : FoldInputs) (gs : List PackGroup) (V : PackingValues)
    (tau0 : List Ext) (stages : List Product) (leaf : List (List Ext × Ext)) (image : Ext)
    (rounds : List (List Ext × Ext)) (data : RhsRun → Array Fq) (source : Fin n → RhsSource)
    (coefficients claims : Array Ext) (challenge : B.Challenge) {μ : ℕ} (fits : B.cells ≤ 2 ^ μ)
    (base : ℕ) {digits : ℕ} (room : B.quotientRoom digits ≤ 2 ^ μ) {batch : ℕ}
    (rowsFit : B.rowCount + 1 ≤ 2 ^ batch) (w : Table) (ν : ℕ) (flat : Table)
    (scan : SetupScan (Batch.liftedReduction β ((batchStatementFor B setup points data source
      coefficients claims).over fun p : Fin (2 ^ ν) => flat p) challenge μ fits base digits room
        batch rowsFit) ν flat) (σ : Ext)
    (setupRounds : List (List Ext × Ext)) (prefixValue : Ext)
    (setupWeight : List Ext → List Ext → Ext) (compressed : V.compressed = B.compressed)
    (virtual : V.virtual = leaf.map (·.2)) (vars : tau0.length = μ) (tau : V.tau1.length = batch)
    (arities : stages.map (·.arity) = rangeArities base)
    (shaped : ∀ P ∈ stages, P.rounds.length = μ ∧ ∀ x ∈ P.rounds, x.1.length = P.arity)
    (count : leaf.length = μ) (lengths : ∀ x ∈ leaf, x.1.length = rangeLeaf base)
    (fusedCount : rounds.length = μ) (fusedLengths : ∀ x ∈ rounds, x.1.length = 3)
    (setupCount : setupRounds.length = ν) (setupLengths : ∀ x ∈ setupRounds, x.1.length = 2)
    (weighs : ∀ p : List Ext, p.length = μ → ((packFinish F gs V (some σ) p).weight.into : E) =
      scan.part V.alpha.into (V.tau1.map Ext.into) (p.map Ext.into) +
        scan.scale V.alpha.into (V.tau1.map Ext.into) (p.map Ext.into) * σ.into)
    (supported : ∀ p : List Ext, p.length = μ → B.compressed = true →
      ((foldSupport (Placement.of gs F.path F.digits) (CellEq.half V.virtual).get
        (CellEq.half p).get).into : E) = mle (p.map Ext.into) fun x =>
          if (B.gaps digits).support x then eqAt (V.virtual.map Ext.into) x else 0)
    (setupWeighs : ∀ r2 ρ : List Ext, r2.length = μ → ρ.length = ν →
      ((setupWeight r2 ρ).into : E) = mle (ρ.map Ext.into)
        (scan.weight V.alpha.into (V.tau1.map Ext.into) (r2.map Ext.into))) :
    (offloadCheck (Batch.liftedReduction β ((batchStatementFor B setup points data source
        coefficients claims).over fun p : Fin (2 ^ ν) => flat p) challenge μ fits base digits room
        batch rowsFit) base (B.gaps digits).support B.compressed w ν flat scan).Runs
      (offloadPath E μ batch B.low base B.compressed (V.words tau0 stages image rounds) leaf
        (packClaim V data
          (relation (fun k => B.shape (groupOrder n k)) B.openRank B.openDegree B.chain source).1
          (relation (fun k => B.shape (groupOrder n k)) B.openRank B.openDegree B.chain source).2
          coefficients claims image) (packFinish F gs V (some σ)) σ setupRounds prefixValue)
      ((offloadRun base B.low (V.words tau0 stages image rounds) leaf (packClaim V data
          (relation (fun k => B.shape (groupOrder n k)) B.openRank B.openDegree B.chain source).1
          (relation (fun k => B.shape (groupOrder n k)) B.openRank B.openDegree B.chain source).2
          coefficients claims image) (packFinish F gs V (some σ)) σ setupRounds prefixValue
          setupWeight).map
        fun o => ((o.1.1.map Ext.into, o.1.2.into), (o.2.1.map Ext.into, o.2.2.into))) := by
  have virtualLength : V.virtual.length = μ := by rw [virtual, List.length_map, count]
  refine offloadRun_runs (Batch.liftedReduction β ((batchStatementFor B setup points data source
    coefficients claims).over fun p : Fin (2 ^ ν) => flat p) challenge μ fits base digits room batch
    rowsFit) base (B.gaps digits).support B.compressed w ν flat scan (V.words tau0 stages image
    rounds) leaf _ (packFinish F gs V (some σ)) σ setupRounds prefixValue setupWeight vars tau
    arities shaped count lengths fusedCount fusedLengths setupCount setupLengths ?_
    (fun p length => ?_) (fun p length => ?_) setupWeighs
  · rw [packClaim, batch_fused_over β setup points data source coefficients claims _
      (fun p : Fin (2 ^ ν) => flat p) challenge μ fits base digits room batch rowsFit V.tau1
      V.alpha tau, Ext.into_add, Ext.into_mul, Ext.into_zero, add_zero]
    rfl
  · show ((packFinish F gs V (some σ) p).weight.into + (Ext.zero.into : E) * Ext.zero.into : E) =
      _
    rw [Ext.into_zero, zero_mul, add_zero]
    exact weighs p length
  · rw [← virtual]
    refine supports_of (B.gaps digits).support (packFinish F gs V (some σ) p) V.virtual p
      B.compressed V.binary _ (length.trans virtualLength.symm) rfl (by rw [← compressed]; rfl)
      (by rw [← compressed]; rfl) fun bin => ?_
    exact supported p length bin

end Offload

end NightstreamFPrime.Spec.Compression.Akita.Replay
