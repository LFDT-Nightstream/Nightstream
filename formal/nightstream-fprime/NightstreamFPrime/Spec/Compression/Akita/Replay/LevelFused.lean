import NightstreamFPrime.Spec.Compression.Akita.Replay.LevelStatement

/-! The fused sum-check of one fold of the replay as one function, and its relation to
`fusedCheck` (Akita revision `d50f700`: `verify_stage2` and `validate_stage2_replay` of
`stages/stage2.rs`; ePrint 2026/1983, (125) and Section 7.4).

- `fusedRun` runs the plain rounds of the fused sum-check from its claim (`sumcheckRun`), and then
  the end check `FusedEnd.holds` of the values `finish r_2` at the point `r_2` of the rounds with
  the end claim of the rounds. It ends in `(r_2, v_W)`, or in `none` when the check fails.
- `fusedRun_runs`: `fusedCheck` on the path of the code (`fusedPath`) ends in the result of
  `fusedRun`, when the claim is `target + γ·s_claim` and, at each point `r_2` of `μ` coordinates,
  the weight of the values is `mle r_2 weight` (`weighs`) and their virtual weight is `mle r_2` of
  `virtWeight` (`supports`).
- `traceFinish` gives the values of the end check of a trace level at a point.
  `traceFinish_weighs`, `traceFinish_weighsSq` and `traceFinish_supports` are `weighs` and
  `supports` at a trace level for `traceReduction` of `traceStatementOf`, from `traceWeight_into`,
  `normValue_into` and `supportValue_into`. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

open NightstreamFPrime.Spec.Compression.Lib NightstreamFPrime.Spec.Compression.Game
open NightstreamFPrime.Spec.Compression (Game)
open NightstreamFPrime.Spec.Compression.Akita

/-- The values `f` with the end claim `c` of the rounds. -/
def FusedEnd.withClaim (f : FusedEnd) (c : Ext) : FusedEnd := { f with output := c }

/-- The fused sum-check of a fold: the plain rounds from `claim`, and the end check of the values
`finish r_2` with the end claim of the rounds. -/
def fusedRun (claim : Ext) (rounds : List (List Ext × Ext)) (finish : List Ext → FusedEnd) :
    Option (List Ext × Ext) :=
  match sumcheckRun .plain claim rounds with
  | none => none
  | some ended =>
      if ((finish ended.point).withClaim ended.claim).holds then
        some (ended.point, (finish ended.point).value)
      else none

/-- `fusedRun` at the point of the rounds. -/
theorem fusedRun_eq (claim : Ext) (rounds : List (List Ext × Ext))
    (finish : List Ext → FusedEnd) :
    fusedRun claim rounds finish =
      if ((finish (rounds.map (·.2))).withClaim (plainRun claim rounds).claim).holds then
        some (rounds.map (·.2), (finish (rounds.map (·.2))).value)
      else none := by
  show (if ((finish (plainRun claim rounds).point).withClaim (plainRun claim rounds).claim).holds
    then some ((plainRun claim rounds).point, (finish (plainRun claim rounds).point).value)
    else none) = _
  rw [plainRun_point]

/-- The path of `fusedCheck`: the rounds and the message `v_W`. -/
noncomputable def fusedPath (E : Type) [Field E] [Algebra Zq E] [Degree2 E] (claim : Ext)
    (rounds : List (List Ext × Ext)) (finish : List Ext → FusedEnd) : List Move :=
  plainPath (Ext.into (E := E)) 2 claim rounds ++
    [Move.recv ((finish (rounds.map (·.2))).value.into : E)]

variable {E : Type} [Field E] [Fintype E] [DecidableEq E] [Algebra Zq E] [Degree2 E]

omit [Fintype E] [DecidableEq E] in
/-- The end check of the code with the end claim `c` is the final check of the game for the
public weight `weight` and the virtual weight `virt` of the values at `r_2`. -/
theorem holds_iff_final (f : FusedEnd) (c : Ext) (weight virt : E)
    (weighs : (f.weight.into + f.norm.into * f.normWeight.into : E) = weight)
    (supports : (f.batch.into * f.virt.into + f.binary.into * f.support.into : E) = virt) :
    (f.withClaim c).holds = true ↔
      (c.into : E) = f.value.into * weight + virt * (f.value.into * (f.value.into + 1)) := by
  rw [FusedEnd.holds_iff (E := E)]
  exact Iff.of_eq (by rw [← weighs, ← supports]; rfl)

/-- `fusedCheck` of the specification, on the path of the rounds and of `v_W` of the code, ends in
the result of `fusedRun`: `none` when the end check fails, and otherwise `(r_2, v_W)`. -/
theorem fusedRun_runs (μ : ℕ) (support : ℕ → Prop) [DecidablePred support] (w : Table)
    (weight : ℕ → E) (target : E) (r : List E) (sClaim ζ γ : E) (claim : Ext)
    (rounds : List (List Ext × Ext)) (finish : List Ext → FusedEnd) (count : rounds.length = μ)
    (lengths : ∀ x ∈ rounds, x.1.length = 3) (claimed : (claim.into : E) = target + γ * sClaim)
    (weighs : ∀ p : List Ext, p.length = μ →
      ((finish p).weight.into + (finish p).norm.into * (finish p).normWeight.into : E) =
        mle (p.map Ext.into) weight)
    (supports : ∀ p : List Ext, p.length = μ →
      ((finish p).batch.into * (finish p).virt.into + (finish p).binary.into *
        (finish p).support.into : E) = mle (p.map Ext.into) (virtWeight support r γ ζ)) :
    (fusedCheck μ support w weight target r sClaim ζ γ).Runs (fusedPath E claim rounds finish)
      ((fusedRun claim rounds finish).map fun o => (o.1.map Ext.into, o.2.into)) := by
  have long : (rounds.map (·.2)).length = μ := by rw [List.length_map, count]
  have weighed := weighs (rounds.map (·.2)) long
  have supported := supports (rounds.map (·.2)) long
  have key := holds_iff_final (E := E) (finish (rounds.map (·.2))) (plainRun claim rounds).claim
    _ _ weighed supported
  have rounds_run := plainRun_runs (degree2_extHom (E := E)) 2
    (combRound μ (fusedTables w weight (virtWeight support r γ ζ)) (fusedComb E)) rounds [] claim
    lengths
  rw [count, claimed] at rounds_run
  have point : (plainRun claim rounds).point = rounds.map (·.2) := plainRun_point claim rounds
  unfold fusedCheck fusedPath
  refine runs_bind rounds_run (runs_recv ?_)
  simp only [List.nil_append, point]
  rw [fusedRun_eq]
  by_cases passes : ((finish (rounds.map (·.2))).withClaim (plainRun claim rounds).claim).holds =
    true
  · rw [ite_eq_left (key.mp passes), ite_eq_left passes]
    rfl
  · rw [ite_eq_right fun equal => passes (key.mpr equal), ite_eq_right passes]
    rfl

section Trace

variable {S : Shape} {O : OpeningShape S} {fold : Descriptor}

variable (O fold) in
/-- The values of the end check of the fused sum-check of a trace level at the point `r_2`: the
values `V` of the replay with the point `r_2` (`traceFused`). -/
def traceFinish (setup : Array Fq) (challenges : List (Array Fq)) (V : TraceValues)
    (r2 : List Ext) : FusedEnd :=
  traceFused O fold setup challenges { V with r2 }

omit [DecidableEq E] in
/-- `weighs` of `fusedRun_runs` at a trace level on the coefficient route: the weight of the
values of the code at `r_2` is `m_{τ_1}(r_2)` of `traceReduction` of `traceStatementOf`. -/
theorem traceFinish_weighs (quarter : 4 * (S.innerDegree / 4) = S.innerDegree)
    (two : ((2 * (S.innerDegree / 4) : ℕ) : Zq) ≠ 0) (setup : Array Fq)
    (challenges : List (Array Fq)) (V : TraceValues) (payload : S.Payload)
    (openPayload : O.Payload) (value : E) (challenge : Fin S.blocks → Rq S.innerDegree)
    (challenged : ∀ i, challenge i =
      Poly.toRq S.innerDegree (challenges.getD i (Poly.zero S.innerDegree))) {μ : ℕ}
    (fits : O.cells fold ≤ 2 ^ μ) {batch : ℕ} (rowsFit : O.traceRowCount + 1 ≤ 2 ^ batch)
    (tau : V.tau1.length = batch) (plain : V.euclidean = false) (p : List Ext)
    (length : p.length = μ) :
    ((traceFinish O fold setup challenges V p).weight.into +
        (traceFinish O fold setup challenges V p).norm.into *
          (traceFinish O fold setup challenges V p).normWeight.into : E) =
      mle (p.map Ext.into) (((traceReduction (traceStatementOf O quarter setup V payload
        openPayload value) challenge μ fits batch rowsFit).rows V.alpha.into).batchWeight
          (V.tau1.map Ext.into)) := by
  have zero : (traceFinish O fold setup challenges V p).norm = Ext.zero := by
    simp only [traceFinish, traceFused, plain, Bool.false_eq_true, ite_false]
  rw [zero, Ext.into_zero, zero_mul, add_zero]
  exact traceWeight_into quarter two
    ⟨setup, challenges, V.tau1, V.point, tensorFactor V.tail V.eta V.point⟩ _ challenge challenged
    (Key.ofSetup_shared _ S) (OpeningKey.ofSetup_shared _ O) rfl (fun _ => rfl) (fun _ => rfl) rfl
    rfl fits rowsFit V.alpha p length tau

omit [DecidableEq E] in
/-- `weighs` of `fusedRun_runs` at a trace level on the Euclidean route: the weight of the values
of the code at `r_2` is `m_{τ_1}(r_2) + η_norm·K_{r_virt}(r_2)`, the weight of `foldCheckSq`. -/
theorem traceFinish_weighsSq (quarter : 4 * (S.innerDegree / 4) = S.innerDegree)
    (two : ((2 * (S.innerDegree / 4) : ℕ) : Zq) ≠ 0) (setup : Array Fq)
    (challenges : List (Array Fq)) (V : TraceValues) (payload : S.Payload)
    (openPayload : O.Payload) (value : E) (challenge : Fin S.blocks → Rq S.innerDegree)
    (challenged : ∀ i, challenge i =
      Poly.toRq S.innerDegree (challenges.getD i (Poly.zero S.innerDegree))) {μ : ℕ}
    (fits : O.cells fold ≤ 2 ^ μ) {batch : ℕ} (rowsFit : O.traceRowCount + 1 ≤ 2 ^ batch)
    (digits : 0 < fold.responseDigits) (virtualLength : V.virtual.length = μ)
    (tau : V.tau1.length = batch) (euclidean : V.euclidean = true) (p : List Ext)
    (length : p.length = μ) :
    ((traceFinish O fold setup challenges V p).weight.into +
        (traceFinish O fold setup challenges V p).norm.into *
          (traceFinish O fold setup challenges V p).normWeight.into : E) =
      mle (p.map Ext.into) fun x =>
        ((traceReduction (traceStatementOf O quarter setup V payload openPayload value) challenge
          μ fits batch rowsFit).rows V.alpha.into).batchWeight (V.tau1.map Ext.into) x +
          V.norm.into * normWeight μ (S.normKernel fold) (V.virtual.map Ext.into) x := by
  have norm : (traceFinish O fold setup challenges V p).norm = V.norm := by
    simp only [traceFinish, traceFused, euclidean, ite_true]
  have normed : ((traceFinish O fold setup challenges V p).normWeight.into : E) =
      mle (p.map Ext.into) (normWeight μ (S.normKernel fold) (V.virtual.map Ext.into)) := by
    show ((if V.euclidean then normValue S fold (CellEq.half V.virtual).get (CellEq.half p).get
      else Ext.zero).into : E) = _
    rw [ite_eq_left euclidean]
    exact normValue_into ((O.responseCells_le_cells fold).trans fits) digits _ _
      (by rw [List.length_map, length])
      (fun u small => CellEq.get_half V.virtual (by rw [virtualLength]; exact small))
      (fun x small => CellEq.get_half p (by rw [length]; exact small))
  rw [norm, normed, mle_add, mle_mul_left]
  congr 1
  exact traceWeight_into quarter two
    ⟨setup, challenges, V.tau1, V.point, tensorFactor V.tail V.eta V.point⟩ _ challenge challenged
    (Key.ofSetup_shared _ S) (OpeningKey.ofSetup_shared _ O) rfl (fun _ => rfl) (fun _ => rfl) rfl
    rfl fits rowsFit V.alpha p length tau

omit [Fintype E] [DecidableEq E] in
/-- `supports` of `fusedRun_runs` at a trace level: the virtual weight of the values of the code at
`r_2` is `mle r_2` of `virtWeight` for the binary support of the level, the point `r_virt` of the
replay, `γ` and `ζ_bin` (zero at a level without compression digits). -/
theorem traceFinish_supports (setup : Array Fq) (challenges : List (Array Fq)) (V : TraceValues)
    {μ : ℕ} (fits : O.cells fold ≤ 2 ^ μ) (virtualLength : V.virtual.length = μ)
    (compressed : V.compressed = O.compressed fold) (p : List Ext) (length : p.length = μ) :
    ((traceFinish O fold setup challenges V p).batch.into *
        (traceFinish O fold setup challenges V p).virt.into +
      (traceFinish O fold setup challenges V p).binary.into *
        (traceFinish O fold setup challenges V p).support.into : E) =
      mle (p.map Ext.into) (virtWeight (O.binarySupport fold) (V.virtual.map Ext.into)
        V.batch.into (if O.compressed fold then V.binary.into else 0)) := by
  have virt : ((traceFinish O fold setup challenges V p).virt.into : E) =
      mle (p.map Ext.into) (eqAt (V.virtual.map Ext.into)) := by
    show ((eqPoints V.virtual p).into : E) = _
    rw [eqPoints_into, mle_eqAt _ _ (by rw [List.length_map, List.length_map, length,
      virtualLength])]
  have split : mle (p.map Ext.into) (virtWeight (O.binarySupport fold) (V.virtual.map Ext.into)
      V.batch.into (if O.compressed fold then V.binary.into else 0)) =
      (V.batch.into : E) * mle (p.map Ext.into) (eqAt (V.virtual.map Ext.into)) +
        (if O.compressed fold then V.binary.into else 0) * mle (p.map Ext.into)
          (fun x => if O.binarySupport fold x then eqAt (V.virtual.map Ext.into) x else 0) := by
    rw [show virtWeight (O.binarySupport fold) (V.virtual.map Ext.into) V.batch.into
        (if O.compressed fold then V.binary.into else 0) = fun x =>
      (V.batch.into : E) * eqAt (V.virtual.map Ext.into) x +
        (if O.compressed fold then V.binary.into else 0) *
          (if O.binarySupport fold x then eqAt (V.virtual.map Ext.into) x else 0) from rfl,
      mle_add, mle_mul_left, mle_mul_left]
  rw [split, virt]
  congr 1
  show ((if V.compressed then V.binary else Ext.zero).into : E) *
    ((if V.compressed then supportValue O fold (CellEq.half V.virtual).get (CellEq.half p).get
      else Ext.zero).into : E) = _
  rw [compressed]
  cases O.compressed fold
  · simp only [Bool.false_eq_true, ite_false, Ext.into_zero, zero_mul]
  · simp only [ite_true]
    rw [supportValue_into fits _ _ (by rw [List.length_map, length])
      (fun u small => CellEq.get_half V.virtual (by rw [virtualLength]; exact small))
      (fun x small => CellEq.get_half p (by rw [length]; exact small))]

end Trace

end NightstreamFPrime.Spec.Compression.Akita.Replay
