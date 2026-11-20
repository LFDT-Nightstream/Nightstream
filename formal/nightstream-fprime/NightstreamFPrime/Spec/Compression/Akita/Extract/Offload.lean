import NightstreamFPrime.Spec.Compression.Akita.Extract.LiftSource

/-! The checks of an offloaded fold of Akita (ePrint 2026/1983, revision of 2026-10-15, Section 9
with (132) to (135), and Lemma 9.1 with its proof in Appendix D.2).

The verifier of an offloaded fold does not read the setup in the final check of the fused
sum-check. The public weight at the point `r_2` is `m_local(r_2) + σ_S` (132): the proof has the
value `σ_S`, and Stage 3 is a sum-check for `σ_S = Σ_p S^♭(p)·ω_setup(p)` (133), (134) that ends
in the claim `s_{ρ_S} = S̃^♭(ρ_S)` for the next level (135).

- `setupCheck` is Stage 3 for the setup prefix `flat` and a weight. `setupCheck_avoids` states
  that on a path that avoids the bad sets a true final claim gives (133).
- `SetupScan R ν flat` has the split (132) of the public weight of the reduction `R`: the local
  part `m_local` and the weight `ω_setup` that the verifier computes, with the proof of (132).
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
`2^ν` cells: `part α τ_1 r_2` is `m_local(r_2)` and `weight α τ_1 r_2` is `ω_setup`, which the
verifier computes from the point `α` of the ring reduction, the batching point `τ_1` and the
point `r_2`. -/
structure SetupScan (R : Reduction E) (ν : ℕ) (flat : Table) where
  part : E → List E → List E → E
  weight : E → List E → List E → ℕ → E
  split : ∀ (α : E) (τ : Fin R.batch → E) (r2 : List E), r2.length = R.vars →
    mle r2 ((R.rows α).batchWeight (List.ofFn τ)) =
      part α (List.ofFn τ) r2 + cubeSum ν fun p => lifted E flat p * weight α (List.ofFn τ) r2 p

/-- The checks of an offloaded fold for the flat witness `w` of the next level: `foldCheck` with
the public weight `m_local(r_2) + σ_S` of Akita (132) in the final check of the fused sum-check,
for the value `σ_S` of the proof, and then Stage 3 `setupCheck`. It ends in `none` when a check
fails, and otherwise in the claims `(r_2, v_W)` and `(ρ_S, s_{ρ_S})` for the next level. -/
noncomputable def offloadCheck (R : Reduction E) (b : ℕ) (support : ℕ → Prop)
    [DecidablePred support] (w : Table) (ν : ℕ) (flat : Table) (scan : SetupScan R ν flat) :
    Game (Option ((List E × E) × (List E × E))) :=
  .draw E (R.bad w) fun α =>
  .draw (Fin R.vars → E) (rangeBad b R.vars w) fun τ0 =>
  .draw (Fin R.batch → E) ((R.rows α).batchBad R.vars R.batch w) fun τ1 =>
  (sumcheckWith (b / 2 + 1) (combRound R.vars (rangeTables (List.ofFn τ0) w) (rangeComb E b))
    (fun point => bindBad (R.vars - point.length - 1) (fold point (supportTable support w)))
    R.vars [] 0).bind fun
    | none => .pure none
    | some (r, value) => .recv E fun sClaim =>
        if value = mle r (eqAt (List.ofFn τ0)) * rangeSqIn b sClaim then
          .draw E (zeros (linePoly (mle (List.ofFn τ1) ((R.rows α).residual R.vars w))
            (mle r (supportTable support w)))) fun ζ =>
          .draw E (zeros (linePoly (mle (List.ofFn τ1) ((R.rows α).residual R.vars w) +
            ζ * mle r (supportTable support w)) (mle r (derived E w) - sClaim))) fun γ =>
            (sumcheck 3 (combRound R.vars (fusedTables w ((R.rows α).batchWeight (List.ofFn τ1))
              (virtWeight support r γ ζ)) (fusedComb E)) R.vars []
              ((R.rows α).batchTarget (List.ofFn τ1) + γ * sClaim)).bind fun
              | none => .pure none
              | some (r2, final) => .recv E fun v => .recv E fun σ =>
                  if final = v * (scan.part α (List.ofFn τ1) r2 + σ) +
                      mle r2 (virtWeight support r γ ζ) * (v * (v + 1)) then
                    (setupCheck ν flat (scan.weight α (List.ofFn τ1) r2) σ).bind fun
                      | none => .pure none
                      | some claim => .pure (some ((r2, v), claim))
                  else .pure none
        else .pure none

/-- Akita Lemma 9.1 with `foldCheck_avoids`: on a path of `offloadCheck` that avoids the bad sets,
two final claims that are true, `v_W = w̃(r_2)` for the committed witness and
`s_{ρ_S} = S̃^♭(ρ_S)` for the setup prefix, give the relation of the ring reduction, the digit
range statement and the binary statement. -/
theorem offloadCheck_avoids (R : Reduction E) (b : ℕ) (support : ℕ → Prop)
    [DecidablePred support] (w : Table) (ν : ℕ) (flat : Table) (scan : SetupScan R ν flat) :
    (offloadCheck R b support w ν flat scan).Avoids fun result =>
      ∀ r2 v ρ s, result = some ((r2, v), (ρ, s)) → v = mle r2 (lifted E w) →
        s = mle ρ (lifted E flat) →
        R.Holds w ∧ RangeHolds b R.vars w ∧ BinaryOn support R.vars w := by
  intro α reduce τ0 point τ1 batch
  refine (avoids_bind _ _ _).mpr ?_
  refine (sumcheckWith_avoids (b / 2 + 1) R.vars _ _ (range_roundTruth b R.vars (List.ofFn τ0) w)
    (fun point => bindBad (R.vars - point.length - 1) (fold point (supportTable support w)))
    (fun point => NonzeroOn (R.vars - point.length) (fold point (supportTable support w)))
    (fun point c inside nonzero outside => by
      rw [fold_append_single, List.length_append, List.length_singleton,
        show R.vars - (point.length + 1) = R.vars - point.length - 1 by omega]
      exact nonzeroOn_bindLow
        (by rwa [show R.vars - point.length - 1 + 1 = R.vars - point.length by omega]) outside)
    R.vars [] 0 (by simp)).mono ?_
  rintro (_ | ⟨r, value⟩) ranged
  · intro r2 v ρ s equal
    simp at equal
  intro sClaim
  dsimp only
  split
  swap
  · intro r2 v ρ s equal
    simp at equal
  rename_i checked
  intro ζ outsideζ γ outsideγ
  obtain ⟨length, rangeSound, supportKept⟩ := ranged r value rfl
  simp only [List.length_nil, zero_add] at length
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
  have virt := fusedTruth_virt support w ((R.rows α).batchWeight (List.ofFn τ1)) r γ ζ
  rw [length] at virt
  have batched : ∑ x ∈ Finset.range (2 ^ R.vars),
      lifted E w x * (R.rows α).batchWeight (List.ofFn τ1) x -
        (R.rows α).batchTarget (List.ofFn τ1) =
      mle (List.ofFn τ1) ((R.rows α).residual R.vars w) :=
    (R.rows α).batch_residual R.vars w (List.ofFn τ1)
  have root : mle (List.ofFn τ1) ((R.rows α).residual R.vars w) +
      ζ * mle r (supportTable support w) + γ * (mle r (derived E w) - sClaim) = 0 := by
    rw [← batched]
    linear_combination (fusedTrue.trans virt).symm
  obtain ⟨first, second⟩ := eq_zero_of_not_mem_zeros_linePoly outsideγ root
  obtain ⟨relation, binary⟩ := eq_zero_of_not_mem_zeros_linePoly outsideζ first
  have rangeTrue := not_imp_not.mp rangeSound (by
    have last := rangeTruth_final b (List.ofFn τ0) r w
    rw [length] at last
    rw [last, checked, sub_eq_zero.mp second])
  have sum := rangeTruth_nil b (List.ofFn τ0) w
  rw [List.length_ofFn] at sum
  have zeroTable : ¬ NonzeroOn R.vars (supportTable (E := E) support w) := by
    intro nonzero
    have kept := supportKept (by simpa [fold] using nonzero)
    rw [length, Nat.sub_self] at kept
    obtain ⟨i, lt, cell⟩ := kept
    obtain rfl : i = 0 := by simpa using lt
    exact cell binary
  refine ⟨R.sound w α reduce ((R.rows α).holds_of_batched R.vars batch
    (((R.rows α).batched_iff R.vars w _).mpr relation)),
    rangeHolds_of_sum point (by rw [← sum, ← rangeTrue]), fun x small member => ?_⟩
  by_contra cell
  refine zeroTable ⟨x, small, ?_⟩
  rw [supportTable, ite_eq_left member, derived]
  exact fun zero => cell ((algebraMap Zq E).injective (zero.trans (map_zero _).symm))

namespace Extract

open NightstreamFPrime.Spec.Compression (Game)
open NightstreamFPrime.Spec.Checkpoint

attribute [local instance low] Classical.propDecidable

/-- The bounds of the draws of `setupCheck`: `2/|E|` for each round (Akita (134)). -/
noncomputable def setupCosts (E : Type) [Fintype E] (ν : ℕ) : List ℝ :=
  List.replicate ν (((2 : ℕ) : ℝ) / Fintype.card E)

omit [Field E] [DecidableEq E] [Algebra Zq E] in
theorem setupCosts_sum (ν : ℕ) :
    (setupCosts E ν).sum = ((2 * ν : ℕ) : ℝ) / Fintype.card E := by
  simp only [setupCosts, List.sum_replicate, nsmul_eq_mul]
  push_cast
  ring

/-- The verifier of `offloadCheck` is the same for each witness. -/
theorem offloadCheck_bare (R : Reduction E) (b : ℕ) (support : ℕ → Prop) [DecidablePred support]
    (w other : Table) (ν : ℕ) (flat : Table) (scan : SetupScan R ν flat) :
    bare (offloadCheck R b support w ν flat scan) =
      bare (offloadCheck R b support other ν flat scan) := by
  unfold offloadCheck
  refine bare_draw_congr fun α => bare_draw_congr fun τ0 => bare_draw_congr fun τ1 =>
    bare_bind_congr (sumcheckWith_bare _ _ _ _ _ R.vars [] 0) fun result => ?_
  rcases result with _ | ⟨r, value⟩
  · rfl
  · refine bare_recv_congr fun sClaim => ?_
    by_cases checked : value = mle r (eqAt (List.ofFn τ0)) * rangeSqIn b sClaim
    · rw [ite_eq_left checked, ite_eq_left checked]
      exact bare_draw_congr fun ζ => bare_draw_congr fun γ =>
        bare_bind_congr (sumcheck_bare 3 _ _ R.vars [] _) fun _ => rfl
    · rw [ite_eq_right checked, ite_eq_right checked]

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
    [DecidablePred support] (w : Table) (ν : ℕ) (flat : Table) (scan : SetupScan R ν flat)
    (varsFit : R.vars ≤ Fintype.card Coord) (batchFit : R.batch ≤ Fintype.card Coord) :
    Staged (Coord → E) (offloadCheck R base support w ν flat scan)
      (foldCosts E R.badBound R.vars R.batch base ++ setupCosts E ν) := by
  have field : ∃ decode : (Coord → E) → E, Fair decode := exists_fair_slot
  unfold offloadCheck foldCosts Staged
  simp only [List.cons_append, List.append_assoc]
  refine stagedTo_draw field (density_le (C := E) (R.bad w) _ (R.bad_card w)) fun α => ?_
  refine stagedTo_draw (exists_fair_slots varsFit) (density_rangeBad_le (E := E) base R.vars w)
    fun τ0 => ?_
  refine stagedTo_draw (exists_fair_slots batchFit)
    ((R.rows α).density_batchBad_le R.vars (R.fits α) w) fun τ1 => ?_
  refine (stagedTo_bind _ _ _ _).mpr (sumcheckWith_staged field (base / 2 + 1) R.vars 1 _
    (range_roundTruth base R.vars (List.ofFn τ0) w).degree _ (fun _ => bindBad_card_le _ _) _ _
    ?_ ?_ R.vars [] 0 (by simp))
  · exact fun _ => trivial
  rintro ⟨r, value⟩ sClaim
  dsimp only
  split
  swap
  · trivial
  refine stagedTo_draw field (density_le _ 1
    (zeros_card_le_of_natDegree_le (linePoly_natDegree _ _))) fun ζ =>
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
    (μ : ℕ) (fits : B.cells < 2 ^ μ) (base digits : ℕ) (room : B.quotientRoom digits ≤ 2 ^ μ)
    (batch : ℕ) (rowsFit : B.rowCount + 1 + 1 ≤ 2 ^ batch) (block : B.Block → Coord)
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
  block := block
  decode := fun b => decode b.1
  read := read
  digits := B.ofFlat
  check wire challenge table :=
    offloadCheck (Batch.liftedReduction β ((claim wire).statement key openKey)
      (fun g i => challenge ⟨g, i⟩) μ fits base digits room batch rowsFit) base B.binarySupport
      table ν flat (scan _ _)
  passes table result := ∃ r2 v ρ s, result = some ((r2, v), (ρ, s)) ∧
    v = mle r2 (lifted E table) ∧ s = mle ρ (lifted E flat)
  costs := foldCosts E (2 * B.largestDegree - 1) μ batch base ++ setupCosts E ν
  start := start

variable {β : ∀ g, Module.Basis (Fin (B.planes g)) Zq E} {key : (g : Fin n) → Key (B.shape g)}
  {openKey : B.OpenKey} {claim : Wire → B.Claim E} {μ : ℕ} {fits : B.cells < 2 ^ μ}
  {base digits : ℕ} {room : B.quotientRoom digits ≤ 2 ^ μ} {batch : ℕ}
  {rowsFit : B.rowCount + 1 + 1 ≤ 2 ^ batch} {block : B.Block → Coord}
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
  injective := injective
  merges := Batch.method_sound units sparse response
  sound wire challenge table :=
    (offloadCheck_avoids _ base B.binarySupport table ν flat (scan _ _)).mono
      fun _ implies passed => by
        obtain ⟨r2, v, ρ, s, ended, opened, setup⟩ := passed
        have checked := implies r2 v ρ s ended opened setup
        exact Batch.satisfies_of_locked β ((claim wire).statement key openKey)
          (fun g i => challenge ⟨g, i⟩) μ fits uniform small
          (LiftRelation.holds_flat _ checked.1) checked.2.1 checked.2.2
  staged wire challenge table :=
    offloadCheck_staged _ base B.binarySupport table ν flat (scan _ _) varsFit batchFit
  nonnegative cost member := by
    rcases List.mem_append.mp member with fold | setup
    · exact foldCosts_nonnegative _ _ _ _ cost fold
    · rw [List.eq_of_mem_replicate setup]
      positivity
  draws := by simp [offBatch, foldCosts]

omit [Fintype Coord] [DecidableEq Coord] [Nonempty Coord] in
/-- The verifier of the checks of an offloaded level does not read the table. -/
theorem offBatch_uniform (wire : Wire) (challenge : (b : B.Block) → Rq (B.carrierDegree b.1))
    (table other : Table) :
    bare ((offBatch β key openKey claim μ fits base digits room batch rowsFit block decode read
        mass start ν flat scan).check wire challenge table) =
      bare ((offBatch β key openKey claim μ fits base digits room batch rowsFit block decode read
        mass start ν flat scan).check wire challenge other) :=
  offloadCheck_bare _ _ _ table other ν flat (scan _ _)

end Groups

end Extract

end NightstreamFPrime.Spec.Compression.Akita
