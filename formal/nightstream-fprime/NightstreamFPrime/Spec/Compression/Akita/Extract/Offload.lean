import NightstreamFPrime.Spec.Compression.Akita.Extract.LiftSource

/-! The checks of an offloaded fold of Akita (ePrint 2026/1983, revision of 2026-10-15, Section 9
with (132) to (135), and Lemma 9.1 with its proof in Appendix D.2).

The verifier of an offloaded fold does not read the setup in the final check of the fused
sum-check. The public weight at the point `r_2` is `m_local(r_2) + σ_S` (132): the proof has the
value `σ_S`, and Stage 3 is a sum-check for `σ_S = Σ_p S^♭(p)·ω_setup(p)` (133), (134) that ends
in the claim `s_{ρ_S} = S̃^♭(ρ_S)` for the next level (135).

- `setupCheck` is Stage 3 for the setup prefix `flat` and a weight. `setupCheck_avoids` states
  that on a path that avoids the bad sets a true final claim gives (133).
- `SetupScan R ν flat` has the split (132) of the public weight of the reduction `R` in the form
  of the pinned Akita code: the local part `m_local`, the factor `scale` of the claim of Stage 3
  and the weight of Stage 3, with the proof of the split. `SetupScan.direct` is the form of the
  paper, and `SetupScan.direct_value` is the relation between the two.
- `offloadCheck` is `foldCheck` with the value `σ_S` in the final check and with Stage 3 after
  it. It ends in the two claims `(r_2, v_W)` and `(ρ_S, s_{ρ_S})` of Akita (136).
  `offloadCheck_avoids` is `foldCheck_avoids` for it: Akita Lemma 9.1 for a table of the setup
  prefix that is `flat`.
- `offBatch` is the level `liftBatch` with these checks, and `offBatch_valid` gives `Tier.Valid`.

The weights of `SetupScan` are parameters. Lean has no proof that the weight of the Akita code
has the split (132) for `Batch.liftedReduction`. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita

open NightstreamFPrime.Spec.Compression.Lib NightstreamFPrime.Spec.Compression.Game Polynomial

local instance : Fact (Nat.Prime fieldModulus) := ⟨FieldPrime.fieldModulus_natPrime⟩

variable {E : Type} [Field E] [Fintype E] [DecidableEq E] [Algebra Zq E]

/-- Stage 3 of an offloaded fold (Akita (134), (135)): the sum-check of degree 2 with `ν_S`
rounds for the claim `σ_S = Σ_p S̃^♭(p)·ω̃_setup(p)`, and the final check
`s_{ρ_S}·ω̃_setup(ρ_S)` with the value `s_{ρ_S}` of the proof. It ends in `none` when a check
fails, and otherwise in the claim `(ρ_S, s_{ρ_S})`. -/
noncomputable def setupCheck (ν : ℕ) (flat : Table) (weight : ℕ → E) (σ : E) :
    Game (Option (List E × E)) :=
  (sumcheck 2 (combRound ν ![lifted E flat, weight] productComb) ν [] σ).bind fun
    | none => .pure none
    | some (ρ, final) => .recv E fun s =>
        if final = s * mle ρ weight then .pure (some (ρ, s)) else .pure none

/-- The soundness of Stage 3 (the proof of Akita Lemma 9.1): on a path that avoids the bad sets,
a final claim `s_{ρ_S} = S̃^♭(ρ_S)` that is true for the setup prefix gives (133). -/
theorem setupCheck_avoids (ν : ℕ) (flat : Table) (weight : ℕ → E) (σ : E) :
    (setupCheck ν flat weight σ).Avoids fun result =>
      ∀ ρ s, result = some (ρ, s) → s = mle ρ (lifted E flat) →
        σ = cubeSum ν fun p => lifted E flat p * weight p := by
  refine (avoids_bind _ _ _).mpr ?_
  refine ((sumcheck_avoids 2 ν _ _ (combRound_truth ν 2 ![lifted E flat, weight] productComb
    productComb_degree) ν [] _ (by simp)).and_ends (sumcheck_ends 2 _ ν [] _)).mono ?_
  rintro (_ | ⟨ρ, final⟩) ⟨sound, ends⟩
  · intro ρ s equal
    simp at equal
  intro s
  dsimp only
  split
  swap
  · intro ρ s equal
    simp at equal
  rename_i checked
  intro ρ' s' equal opening
  simp only [Option.some.injEq, Prod.mk.injEq] at equal
  obtain ⟨rfl, rfl⟩ := equal
  have count : ρ.length = ν := by
    obtain ⟨cs, prefixed, count⟩ := ends ρ final rfl
    rw [prefixed]
    simpa using count
  have claim := not_imp_not.mp (sound ρ final rfl) (by
    have last := product_truth_final ρ (![lifted E flat, weight] : Fin 2 → ℕ → E)
    rw [count] at last
    rw [last, checked, opening]
    rfl)
  rw [product_truth_nil] at claim
  exact claim

/-- The split (132) of the public weight of the reduction `R` for the setup prefix `flat` of
`2^ν` cells, in the form of the pinned Akita code
(`eval_flat_at_point_with_deferred_setup` in `crates/akita-verifier/src/relation/evaluation.rs`):
`part α τ_1 r_2` is `m_local(r_2)`, `scale α τ_1 r_2` is the factor of the claim of Stage 3, which
in the code is the evaluation of the powers of `α` at the coefficient coordinates of `r_2`
(`common_alpha_evaluation`), and `weight α τ_1 r_2` is the weight of Stage 3. The verifier computes
the three from the point `α` of the ring reduction, the batching point `τ_1` and the point
`r_2`. -/
structure SetupScan (R : Reduction E) (ν : ℕ) (flat : Table) where
  part : E → List E → List E → E
  scale : E → List E → List E → E
  weight : E → List E → List E → ℕ → E
  split : ∀ (α : E) (τ : Fin R.batch → E) (r2 : List E), r2.length = R.vars →
    mle r2 ((R.rows α).batchWeight (List.ofFn τ)) =
      part α (List.ofFn τ) r2 + scale α (List.ofFn τ) r2 *
        cubeSum ν fun p => lifted E flat p * weight α (List.ofFn τ) r2 p

/-- The form of Akita (132) and (133) of a split: the factor 1, and the weight `ω_setup` that is
the factor of the code times the weight of the code. -/
def SetupScan.direct {R : Reduction E} {ν : ℕ} {flat : Table} (scan : SetupScan R ν flat) :
    SetupScan R ν flat where
  part := scan.part
  scale _ _ _ := 1
  weight α τ r2 p := scan.scale α τ r2 * scan.weight α τ r2 p
  split α τ r2 length := by
    rw [scan.split α τ r2 length, one_mul, ← cubeSum_mul_left]
    exact congrArg _ (congrArg _ (funext fun p => by ring))

omit [DecidableEq E] in
/-- The relation between the two forms: for a value `σ` of Stage 3 of the code, the value
`σ_S` of Akita (132) is `scale·σ`. It gives the same public weight in the final check of the
fused sum-check, and it is the sum (133) for the weight of `SetupScan.direct` when `σ` is the sum
for the weight of the code. -/
theorem SetupScan.direct_value {R : Reduction E} {ν : ℕ} {flat : Table}
    (scan : SetupScan R ν flat) (α : E) (τ r2 : List E) (σ : E) :
    scan.direct.part α τ r2 + scan.direct.scale α τ r2 * (scan.scale α τ r2 * σ) =
        scan.part α τ r2 + scan.scale α τ r2 * σ ∧
      (σ = cubeSum ν (fun p => lifted E flat p * scan.weight α τ r2 p) →
        scan.scale α τ r2 * σ =
          cubeSum ν fun p => lifted E flat p * scan.direct.weight α τ r2 p) := by
  refine ⟨by simp [SetupScan.direct], fun sum => ?_⟩
  rw [sum, ← cubeSum_mul_left]
  exact congrArg _ (funext fun p => by simp only [SetupScan.direct]; ring)

/-- The checks of an offloaded fold for the flat witness `w` of the next level: `foldCheck` with
the public weight `m_local(r_2) + scale·σ` in the final check of the fused sum-check, for the
value `σ` of the proof, and then Stage 3 `setupCheck` for the weight of the code. With
`SetupScan.direct` it is the check of Akita (132) for `σ_S = scale·σ`. It ends in `none` when a check
fails, and otherwise in the claims `(r_2, v_W)` and `(ρ_S, s_{ρ_S})` for the next level. -/
noncomputable def offloadCheck (R : Reduction E) (b : ℕ) (support : ℕ → Prop)
    [DecidablePred support] (bin : Bool) (w : Table) (ν : ℕ) (flat : Table) (scan : SetupScan R ν
      flat) :
    Game (Option ((List E × E) × (List E × E))) :=
  .draw E (R.bad w) fun α =>
  .draw (Fin R.vars → E) (rangeBad b R.vars w) fun τ0 =>
  .draw (Fin R.batch → E) ((R.rows α).batchBad R.vars R.batch w) fun τ1 =>
  (rangeCheck b R.vars support w (List.ofFn τ0)).bind fun
    | none => .pure none
    | some (r, sClaim) =>
        drawBin bin (zeros (linePoly (mle (List.ofFn τ1) ((R.rows α).residual R.vars w))
          (mle r (supportTable support w)))) fun ζ =>
        .draw E (zeros (linePoly (mle (List.ofFn τ1) ((R.rows α).residual R.vars w) +
          ζ * mle r (supportTable support w)) (mle r (derived E w) - sClaim))) fun γ =>
          (sumcheck 3 (combRound R.vars (fusedTables w ((R.rows α).batchWeight (List.ofFn τ1))
            (virtWeight support r γ ζ)) (fusedComb E)) R.vars []
            ((R.rows α).batchTarget (List.ofFn τ1) + γ * sClaim)).bind fun
            | none => .pure none
            | some (r2, final) => .recv E fun v => .recv E fun σ =>
                if final = v * (scan.part α (List.ofFn τ1) r2 +
                      scan.scale α (List.ofFn τ1) r2 * σ) +
                    mle r2 (virtWeight support r γ ζ) * (v * (v + 1)) then
                  (setupCheck ν flat (scan.weight α (List.ofFn τ1) r2) σ).bind fun
                    | none => .pure none
                    | some claim => .pure (some ((r2, v), claim))
                else .pure none

/-- Akita Lemma 9.1 with `foldCheck_avoids`: on a path of `offloadCheck` that avoids the bad sets,
two final claims that are true, `v_W = w̃(r_2)` for the committed witness and
`s_{ρ_S} = S̃^♭(ρ_S)` for the setup prefix, give the relation of the ring reduction, the digit
range statement and the binary statement. -/
theorem offloadCheck_avoids (R : Reduction E) (b : ℕ) (support : ℕ → Prop)
    [DecidablePred support] (bin : Bool) (empty : bin = false → ∀ x, ¬ support x) (w : Table)
    (ν : ℕ) (flat : Table) (scan : SetupScan R ν flat) :
    (offloadCheck R b support bin w ν flat scan).Avoids fun result =>
      ∀ r2 v ρ s, result = some ((r2, v), (ρ, s)) → v = mle r2 (lifted E w) →
        s = mle ρ (lifted E flat) →
        R.Holds w ∧ RangeHolds b R.vars w ∧ BinaryOn support R.vars w := by
  intro α reduce τ0 point τ1 batch
  refine (avoids_bind _ _ _).mpr ?_
  refine (rangeCheck_avoids b R.vars support w (List.ofFn τ0) List.length_ofFn).mono ?_
  rintro (_ | ⟨r, sClaim⟩) ranged
  · intro r2 v ρ s equal
    simp at equal
  obtain ⟨virt, rangeSound⟩ := ranged (r, sClaim) rfl
  refine drawBin_avoids empty fun ζ outsideζ γ outsideγ => ?_
  refine (avoids_bind _ _ _).mpr ?_
  refine ((sumcheck_avoids 3 R.vars _ _ (fused_roundTruth R.vars w _ _) R.vars [] _
    (by simp)).and_ends (sumcheck_ends 3 _ R.vars [] _)).mono ?_
  rintro (_ | ⟨r2, value2⟩) ⟨fusedSound, fusedEnds⟩
  · intro r2 v ρ s equal
    simp at equal
  intro v σ
  dsimp only
  split
  swap
  · intro r2 v ρ s equal
    simp at equal
  rename_i final
  refine (avoids_bind _ _ _).mpr ((setupCheck_avoids ν flat _ σ).mono ?_)
  rintro (_ | ⟨ρ, s⟩) scanned
  · intro r2 v ρ s equal
    simp at equal
  intro r2' v' ρ' s' equal opening setup
  simp only [Option.some.injEq, Prod.mk.injEq] at equal
  obtain ⟨⟨rfl, rfl⟩, rfl, rfl⟩ := equal
  have length2 : r2.length = R.vars := by
    obtain ⟨cs, prefixed, count⟩ := fusedEnds r2 value2 rfl
    rw [prefixed]
    simpa using count
  rw [scanned ρ s rfl setup, ← scan.split α τ1 r2 length2] at final
  have fusedTrue := not_imp_not.mp (fusedSound r2 value2 rfl) (by
    have last := fusedTruth_final r2 w ((R.rows α).batchWeight (List.ofFn τ1))
      (virtWeight support r γ ζ)
    rw [length2] at last
    rw [last, final, opening])
  obtain ⟨relation, opened, binary⟩ := fused_sound virt
    ((R.rows α).batch_residual R.vars w (List.ofFn τ1)) outsideζ outsideγ fusedTrue
  exact ⟨R.sound w α reduce ((R.rows α).holds_of_batched R.vars batch
    (((R.rows α).batched_iff R.vars w _).mpr relation)),
    rangeHolds_of_sum point (rangeSound opened), binary⟩

namespace Extract

open NightstreamFPrime.Spec.Compression (Game)
open NightstreamFPrime.Spec.Checkpoint

attribute [local instance low] Classical.propDecidable

/-- The bounds of the draws of `setupCheck`: `2/|E|` for each round (Akita (134)). -/
noncomputable def setupCosts (E : Type) [Fintype E] (ν : ℕ) : List ℝ :=
  List.replicate ν (((2 : ℕ) : ℝ) / Fintype.card E)

/-- The losses of the sites of the draws of `setupCheck` in the Akita code: the Stage 3 rounds of
the degree `SETUP_SUMCHECK_DEGREE = 2` (`transcript_grinding/plan.rs`, `append_nonterminal`). -/
def setupLosses (ν : ℕ) : List ℕ := List.replicate ν 2

omit [Field E] [DecidableEq E] [Algebra Zq E] in
theorem setupLosses_length (ν : ℕ) : (setupLosses ν).length = (setupCosts E ν).length := by
  simp [setupLosses, setupCosts]

omit [Field E] [DecidableEq E] [Algebra Zq E] in
theorem setupCosts_sum (ν : ℕ) :
    (setupCosts E ν).sum = ((2 * ν : ℕ) : ℝ) / Fintype.card E := by
  simp only [setupCosts, List.sum_replicate, nsmul_eq_mul]
  push_cast
  ring

/-- The verifier of `offloadCheck` is the same for each witness. -/
theorem offloadCheck_bare (R : Reduction E) (b : ℕ) (support : ℕ → Prop) [DecidablePred support]
    (bin : Bool) (w other : Table) (ν : ℕ) (flat : Table) (scan : SetupScan R ν flat) :
    bare (offloadCheck R b support bin w ν flat scan) =
      bare (offloadCheck R b support bin other ν flat scan) := by
  unfold offloadCheck
  refine bare_draw_congr fun α => bare_draw_congr fun τ0 => bare_draw_congr fun τ1 =>
    bare_bind_congr (rangeCheck_bare b R.vars support w other _) fun result => ?_
  rcases result with _ | ⟨r, sClaim⟩
  · rfl
  · exact drawBin_bare_congr bin fun ζ => bare_draw_congr fun γ =>
      bare_bind_congr (sumcheck_bare 3 _ _ R.vars [] _) fun _ => rfl

variable {Coord : Type} [Fintype Coord] [DecidableEq Coord] [Nonempty Coord]

/-- Each draw of `setupCheck` has the bound `2/|E|`. -/
theorem setupCheck_staged (ν : ℕ) (flat : Table) (weight : ℕ → E) (σ : E) :
    Staged (Coord → E) (setupCheck ν flat weight σ) (setupCosts E ν) := by
  have rounds := sumcheck_staged (Reply := Coord → E) exists_fair_slot 2 ν _
    (combRound_truth ν 2 ![lifted E flat, weight] productComb productComb_degree).degree []
    (fun result rest => StagedTo (Coord → E) (match result with
      | none => Game.pure none
      | some (ρ, final) => Game.recv E fun s =>
          if final = s * mle ρ weight then Game.pure (some (ρ, s)) else Game.pure none) rest
        fun _ _ => True)
    (fun _ => trivial)
    (fun final s => by
      dsimp only
      split <;> trivial) ν [] σ (by simp)
  rw [List.append_nil] at rounds
  exact (stagedTo_bind _ _ _ _).mpr rounds

/-- Each draw of `offloadCheck` has its bound: the bounds `foldCosts` of `foldCheck`, and then
the bounds `setupCosts` of Stage 3. -/
theorem offloadCheck_staged (R : Reduction E) (base : ℕ) (support : ℕ → Prop)
    [DecidablePred support] (bin : Bool) (w : Table) (ν : ℕ) (flat : Table) (scan : SetupScan R ν
      flat)
    (varsFit : R.vars ≤ Fintype.card Coord) (batchFit : R.batch ≤ Fintype.card Coord) :
    Staged (Coord → E) (offloadCheck R base support bin w ν flat scan)
      (foldCosts E R.badBound R.vars R.batch base bin ++ setupCosts E ν) := by
  have field : ∃ decode : (Coord → E) → E, Fair decode := exists_fair_slot
  unfold offloadCheck foldCosts Staged
  simp only [List.cons_append, List.append_assoc]
  refine stagedTo_draw field (density_le (C := E) (R.bad w) _ (R.bad_card w)) fun α => ?_
  refine stagedTo_draw (exists_fair_slots varsFit) (density_rangeBad_le (E := E) base R.vars w)
    fun τ0 => ?_
  refine stagedTo_draw (exists_fair_slots batchFit)
    ((R.rows α).density_batchBad_le R.vars (R.fits α) w) fun τ1 => ?_
  refine (stagedTo_bind _ _ _ _).mpr (rangeCheck_stagedTo base R.vars support w _ _ _ ?_ ?_)
  · exact fun _ => trivial
  rintro ⟨r, sClaim⟩
  refine drawBin_stagedTo bin (zeros_card_le_of_natDegree_le (linePoly_natDegree _ _)) fun ζ =>
      stagedTo_draw field (density_le _ 1
        (zeros_card_le_of_natDegree_le (linePoly_natDegree _ _))) fun γ => ?_
  refine (stagedTo_bind _ _ _ _).mpr (sumcheck_staged (Reply := Coord → E) field 3 R.vars _
    (fused_roundTruth R.vars w _ (virtWeight support r γ ζ)).degree (setupCosts E ν) _ ?_ ?_
    R.vars [] _ (by simp))
  · exact fun _ => trivial
  · intro final v σ
    dsimp only
    split
    · exact (stagedTo_bind _ _ _ _).mpr ((setupCheck_staged ν flat _ σ).mono fun result _ _ => by
        rcases result with _ | claim <;> trivial)
    · trivial

section Groups

variable {n : ℕ} {B : Batch (Fin n)} {Wire : Type}

/-- The level `liftBatch` of an offloaded fold: the checks are `offloadCheck` for
`Batch.liftedReduction`, with the setup prefix `flat` of `2^ν` cells and the split `scan` of the
public weight. A final claim `((r_2, v_W), (ρ_S, s_{ρ_S}))` is true for a table `w` when
`v_W = w̃(r_2)` and `s_{ρ_S} = S̃^♭(ρ_S)` (Akita (136)). -/
noncomputable def offBatch (β : ∀ g, Module.Basis (Fin (B.planes g)) Zq E)
    (key : (g : Fin n) → Key (B.shape g)) (openKey : B.OpenKey) (claim : Wire → B.Claim E)
    (μ : ℕ) (fits : B.cells ≤ 2 ^ μ) (base digits : ℕ) (room : B.quotientRoom digits ≤ 2 ^ μ)
    (batch : ℕ) (rowsFit : B.rowCount + 1 ≤ 2 ^ batch) (block : B.Block → Coord)
    (decode : (g : Fin n) → E → Option (Rq (B.carrierDegree g))) (read : Reader Wire)
    (mass : Fin n → ℕ) (start ν : ℕ) (flat : Table)
    (scan : ∀ (st : B.Statement E) (challenge : B.Challenge),
      SetupScan (Batch.liftedReduction β st challenge μ fits base digits room batch rowsFit) ν
        flat) :
    Tier B.Block (fun b => Rq (B.carrierDegree b.1)) (B.Claim E)
      (B.LevelBreak key openKey fun g => 2 * (2 * mass g) * (B.fold g).sourceBound (B.shape g))
      B.Witness B.Weak Wire Coord E (Option ((List E × E) × (List E × E))) Table where
  method := Batch.method β key openKey mass
  claim := claim
  block b := some (block b)
  decode := fun b => decode b.1
  read := read
  digits table := B.ofFlat fun x => table ((B.gaps digits).place x)
  check wire challenge table :=
    offloadCheck (Batch.liftedReduction β ((claim wire).statement key openKey)
      (fun g i => challenge ⟨g, i⟩) μ fits base digits room batch rowsFit) base
        (B.gaps digits).support B.compressed table ν flat (scan _ _)
  passes table result := ∃ r2 v ρ s, result = some ((r2, v), (ρ, s)) ∧
    v = mle r2 (lifted E table) ∧ s = mle ρ (lifted E flat)
  costs := foldCosts E (2 * B.largestDegree - 1) μ batch base B.compressed ++ setupCosts E ν
  bits := (foldLosses (2 * B.innerMax - 1) μ batch base B.compressed ++ setupLosses ν).map workBits
  foldBits := foldWorkBits (Fintype.card B.Block) B.familyBits
  start := start

variable {β : ∀ g, Module.Basis (Fin (B.planes g)) Zq E} {key : (g : Fin n) → Key (B.shape g)}
  {openKey : B.OpenKey} {claim : Wire → B.Claim E} {μ : ℕ} {fits : B.cells ≤ 2 ^ μ}
  {base digits : ℕ} {room : B.quotientRoom digits ≤ 2 ^ μ} {batch : ℕ}
  {rowsFit : B.rowCount + 1 ≤ 2 ^ batch} {block : B.Block → Coord}
  {decode : (g : Fin n) → E → Option (Rq (B.carrierDegree g))} {read : Reader Wire}
  {mass : Fin n → ℕ} {start ν : ℕ} {flat : Table}
  {scan : ∀ (st : B.Statement E) (challenge : B.Challenge),
    SetupScan (Batch.liftedReduction β st challenge μ fits base digits room batch rowsFit) ν flat}

/-- An offloaded level with opening groups satisfies the conditions of `Tier.Valid`, with the
hypotheses of `liftBatch_valid`. -/
theorem offBatch_valid {peak : Fin n → ℕ} (injective : Function.Injective block)
    (units : ∀ g, UnitDifferences {challenge | ∃ slot, decode g slot = some challenge})
    (sparse : ∀ g challenge, (∃ slot, decode g slot = some challenge) →
      Sparse (mass g) (peak g) challenge)
    (response : ∀ g, 2 ≤ (B.fold g).responseBase)
    (uniform : ∀ g, (B.shape g).certifiedBase = base) (small : base < fieldModulus)
    (varsFit : μ ≤ Fintype.card Coord) (batchFit : batch ≤ Fintype.card Coord) :
    (offBatch β key openKey claim μ fits base digits room batch rowsFit block decode read mass
      start ν flat scan).Valid where
  injective := (Option.some_injective _).comp injective
  avoid _ := Option.some_ne_none _
  merges := Batch.method_sound units sparse response
  sound wire challenge table :=
    (offloadCheck_avoids _ base (B.gaps digits).support B.compressed
      (B.gaps_support_empty digits) table ν flat (scan _ _)).mono
      fun _ implies passed => by
        obtain ⟨r2, v, ρ, s, ended, opened, setup⟩ := passed
        have checked := implies r2 v ρ s ended opened setup
        exact Batch.satisfies_of_lift β ((claim wire).statement key openKey)
          (fun g i => challenge ⟨g, i⟩) μ fits uniform small digits room batch rowsFit checked.1
          checked.2.1 checked.2.2
  staged wire challenge table :=
    offloadCheck_staged _ base (B.gaps digits).support B.compressed table ν flat (scan _ _)
      varsFit batchFit
  nonnegative cost member := by
    rcases List.mem_append.mp member with fold | setup
    · exact foldCosts_nonnegative _ _ _ _ _ cost fold
    · rw [List.eq_of_mem_replicate setup]
      positivity
  draws := by simp [offBatch, foldCosts]
  aligned := by
    show ((foldLosses (2 * B.innerMax - 1) μ batch base B.compressed ++ setupLosses ν).map
      workBits).length = (foldCosts E (2 * B.largestDegree - 1) μ batch base B.compressed ++
        setupCosts E ν).length
    rw [List.length_map, List.length_append, List.length_append, foldLosses_length (E := E),
      setupLosses_length (E := E)]

omit [Fintype Coord] [DecidableEq Coord] [Nonempty Coord] in
/-- The verifier of the checks of an offloaded level does not read the table. -/
theorem offBatch_uniform (wire : Wire) (challenge : (b : B.Block) → Rq (B.carrierDegree b.1))
    (table other : Table) :
    bare ((offBatch β key openKey claim μ fits base digits room batch rowsFit block decode read
        mass start ν flat scan).check wire challenge table) =
      bare ((offBatch β key openKey claim μ fits base digits room batch rowsFit block decode read
        mass start ν flat scan).check wire challenge other) :=
  offloadCheck_bare _ _ _ _ table other ν flat (scan _ _)

end Groups

end Extract

end NightstreamFPrime.Spec.Compression.Akita
