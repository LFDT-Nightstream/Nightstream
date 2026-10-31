import CompressionLeanRust.Verifier

/-! Small model verifiers for the statement of `Verifier.lean`. `honest` makes each check of
`shrink/mod.rs::verify` and meets `Simulates`, so the statement can be met. Each other model
drops one thing and fails the statement, so the statement is not vacuous:

- `acceptsAll` makes the channel calls and accepts: not `Sound`.
- `skipsOpening` accepts when the WHIR opening fails: not `Sound`.
- `skipsFinal` skips the final check of the inner rounds: not `Sound`.
- `skipsObserve` draws the batching challenge without observing the matrix values: not `Sound`.
- `skipsPublics` skips the check that the statement is the program's public words: not `Sound`.
- `skipsCounts` skips the check that the program's counts are the shape's: not `Sound`.
- `rejectsAll` rejects: not `Complete`. -/

namespace CompressionLeanRust.Verifier

open Aeneas Aeneas.Std Result
open neo_verified
open CompressionLean CompressionLean.Shrink CompressionLean.Lib
open NightstreamFPrime.Spec (goldilocksModulus)
open Polynomial

local notation "ι" => algebraMap Gl Cubic

/-! ## The checks of `verify` -/

/-- The statement as a CCS, for its public value. -/
def stmtCcs (T : Template Gl) (s : Counts) (stmt : List field.Gl) : Ccs Gl :=
  ⟨Layout.rowVars T s, Layout.cellVars T s, [], stmt.map glOf⟩

/-- The final check: the inner rounds end at the matrix value times `z`. -/
def InnerOk (T : Template Gl) (ev : Evaluate) (s : Counts) (stmt : List field.Gl) (π : Rounds) (d : Draws)
    (w : Cubic) : Prop :=
  innerLast π d = ((ev d.rx d.ry d.rho).1 + blockPart T s (d.rx.map cubicOf) (d.ry.map cubicOf) (cubicOf d.rho)) *
    zValue (stmtCcs T s stmt) (d.ry.map cubicOf) w

/-- The checks of `verify` after its channel calls; the final one only when `final` is `true`. -/
def Checks (T : Template Gl) (ev : Evaluate) (s : Counts) (stmt : List field.Gl) (π : Rounds) (d : Draws)
    (w : Cubic) (final : Bool) : Prop :=
  RoundsFit T s π ∧ OuterOk π d ∧ (ev d.rx d.ry d.rho).2.1 = s ∧ (ev d.rx d.ry d.rho).2.2 = stmt.map glOf ∧
    (final = true → InnerOk T ev s stmt π d w)

/-- `schedule` without observing the matrix values before the batching draw. -/
def scheduleBlind {σ : Type} (ch : Channel σ) (m n : ℕ) (stmt : List field.Gl) (π : Rounds) (st : σ) :
    Result (Draws × σ) := do
  let st ← ch.observeWords st stmt
  let st ← ch.commit st
  let tau ← samples ch m st
  let rx ← sends ch π.outer tau.2
  let rho ← ch.sample rx.2
  let ry ← sends ch π.inner rho.2
  let opened ← ch.openAt ry.2 (ry.1.take n)
  ok (⟨tau.1, rx.1, rho.1, ry.1, opened.1⟩, opened.2)

open Classical in
/-- A model of `verify`: the channel calls of `calls`, then the checks. With `opening`, a failed
opening rejects; without, it reads as zero. With `final`, the final check is made. -/
noncomputable def model (T : Template Gl)
    (calls : ∀ {σ : Type}, Channel σ → ℕ → ℕ → List field.Gl → Rounds → σ → Result (Draws × σ))
    (opening final : Bool) : VerifyFn := fun ch ev s stmt π st => do
  let run ← calls ch (Layout.rowVars T s) (Layout.cellVars T s) stmt π st
  match run.1.opened with
  | none => if opening then ok (false, run.2) else ok (decide (Checks T ev s stmt π run.1 0 final), run.2)
  | some w => ok (decide (Checks T ev s stmt π run.1 (cubicOf w) final), run.2)

noncomputable def honest (T : Template Gl) : VerifyFn := model T schedule true true

noncomputable def skipsOpening (T : Template Gl) : VerifyFn := model T schedule false true

noncomputable def skipsFinal (T : Template Gl) : VerifyFn := model T schedule true false

noncomputable def skipsObserve (T : Template Gl) : VerifyFn := model T scheduleBlind true true

/-- The shape run `ev` with the statement as its public words. -/
def withPublics (ev : Evaluate) (stmt : List field.Gl) : Evaluate :=
  fun rx ry ρ => ((ev rx ry ρ).1, (ev rx ry ρ).2.1, stmt.map glOf)

/-- `honest` without the check that the statement is the program's public words. -/
noncomputable def skipsPublics (T : Template Gl) : VerifyFn := fun ch ev s stmt π st =>
  honest T ch (withPublics ev stmt) s stmt π st

/-- The shape run `ev` with the shape's counts `s` as its counts. -/
def withCounts (ev : Evaluate) (s : Counts) : Evaluate :=
  fun rx ry ρ => ((ev rx ry ρ).1, s, (ev rx ry ρ).2.2)

/-- `honest` without the check that the program's counts are the shape's. -/
noncomputable def skipsCounts (T : Template Gl) : VerifyFn := fun ch ev s stmt π st =>
  honest T ch (withCounts ev s) s stmt π st

def acceptsAll (T : Template Gl) : VerifyFn := fun ch _ s stmt π st => do
  let run ← schedule ch (Layout.rowVars T s) (Layout.cellVars T s) stmt π st
  ok (true, run.2)

def rejectsAll : VerifyFn := fun _ _ _ _ _ st => ok (false, st)

/-! ## Facts about the plays -/

theorem zValue_congr {C C' : Ccs Gl} (cells : C.cellVars = C'.cellVars) (statement : C.statement = C'.statement)
    (ry : List Cubic) (opened : Cubic) : zValue C ry opened = zValue C' ry opened := by
  simp only [zValue, Ccs.publicValue, Ccs.publicTable, cells, statement]

open Classical in
/-- The checks of `verify`, with a sound opening, are `Final` of the run's transcript. -/
theorem final_of_checks {I : Type} {T : Template Gl} {s : Counts} {prog : I → Run Gl} {i : I} {stmt : List field.Gl}
    {π : Rounds} {d : Draws} {w : field.Ext} {cands : List (ℕ → Gl)}
    (counts : (prog i).counts = s) (publics : (prog i).publics = stmt.map glOf)
    (opens : ∃ c ∈ cands, cubicOf w = mle ((d.ry.take (Layout.cellVars T s)).map cubicOf) fun x => ι (c x))
    (inner : InnerOk T (evalOf T s (prog i)) s stmt π d (cubicOf w)) :
    Final ((prog i).ccs T s) ((prog i).counts = s) cands (d.ry.map cubicOf)
      (verifierMatrices T s (prog i) (d.rx.map cubicOf) (d.ry.map cubicOf) (cubicOf d.rho)) (innerLast π d)
      (cubicOf w) := by
  refine ⟨counts, ?_, ?_⟩
  · obtain ⟨c, member, value⟩ := opens
    exact ⟨c, member, by rw [value, List.map_take]; rfl⟩
  · rw [inner, zValue_congr (C := stmtCcs T s stmt) (C' := (prog i).ccs T s) rfl publics.symm]
    rfl

/-- `Final` of the run's transcript gives the counts check and the final check. -/
theorem checks_of_final {I : Type} {T : Template Gl} {s : Counts} {prog : I → Run Gl} {i : I} {stmt : List field.Gl}
    {π : Rounds} {d : Draws} {w : field.Ext} {cands : List (ℕ → Gl)} (publics : (prog i).publics = stmt.map glOf)
    (final : Final ((prog i).ccs T s) ((prog i).counts = s) cands (d.ry.map cubicOf)
      (verifierMatrices T s (prog i) (d.rx.map cubicOf) (d.ry.map cubicOf) (cubicOf d.rho)) (innerLast π d)
      (cubicOf w)) :
    (prog i).counts = s ∧ InnerOk T (evalOf T s (prog i)) s stmt π d (cubicOf w) := by
  obtain ⟨counts, -, last⟩ := final
  refine ⟨counts, ?_⟩
  rw [InnerOk, last, zValue_congr (C := (prog i).ccs T s) (C' := stmtCcs T s stmt) rfl publics]
  rfl

/-! ## `honest` meets the statement -/

theorem honest_sound (T : Template Gl) : Sound (honest T) T := by
  intro I _ prog s L i stmt π cands oracle opening h whir accepts
  obtain ⟨d, run, len_tau, len_rx, len_ry, -, h', hopened⟩ :=
    schedule_recorder oracle opening (Layout.rowVars T s) (Layout.cellVars T s) stmt π
  simp only [honest, model, run, bind_ok] at accepts
  cases hw : d.opened with
  | none =>
      rw [hw] at accepts
      simp at accepts
  | some w =>
      rw [hw] at accepts
      simp only [Result.ok.injEq, Prod.mk.injEq, decide_eq_true_eq] at accepts
      obtain ⟨⟨fit, outerOk, counts, publics, inner⟩, rfl⟩ := accepts
      refine ⟨d, w, run, hw, publics, ?_⟩
      have path := run_path T s prog L i cands π d (cubicOf w) ⟨fit.1, fit.2.2.1, len_tau, len_rx, len_ry⟩ outerOk
      rw [ite_eq_left (final_of_checks counts publics (whir h' _ w (hw ▸ hopened).symm) (inner rfl))] at path
      exact ⟨_, path⟩

/-- No block, no glue row or cell, and no statement word. -/
def emptyCounts : Counts := ⟨0, 0, 0, 0⟩

def emptyRun : Run Gl := ⟨[], 0, 0, []⟩

theorem emptyRun_counts : emptyRun.counts = emptyCounts := by
  simp [Run.counts, Run.rows, emptyRun, emptyCounts]

theorem rowVars_empty (T : Template Gl) : Layout.rowVars T emptyCounts = 1 := by
  simp [Layout.rowVars, variablesFor, partsEnd, emptyCounts]

theorem cellVars_empty (T : Template Gl) : Layout.cellVars T emptyCounts = 1 := by
  simp [Layout.cellVars, variablesFor, partsEnd, emptyCounts]

/-- One outer round and two inner rounds of zeros. -/
def zeroRounds : Rounds :=
  ⟨[List.replicate 8 field.EXT_ZERO], fun _ => field.EXT_ZERO,
    [[field.EXT_ZERO, field.EXT_ZERO], [field.EXT_ZERO, field.EXT_ZERO]]⟩

def zeroTable : ℕ → Gl := fun _ => 0

def zeroOracle : History → field.Ext := fun _ => field.EXT_ZERO

def zeroOpening : History → List Cubic → Option field.Ext := fun _ _ => some field.EXT_ZERO

def failedOpening : History → List Cubic → Option field.Ext := fun _ _ => none

theorem whir_zero : WhirSound [zeroTable] zeroOpening := by
  intro h point v opened
  simp only [zeroOpening, Option.some.injEq] at opened
  subst opened
  refine ⟨zeroTable, List.mem_singleton_self _, ?_⟩
  simp [zeroTable, cubicOf_zero, mle_const]

theorem whir_failed (cands : List (ℕ → Gl)) : WhirSound cands failedOpening := by
  intro h point v opened
  simp [failedOpening] at opened

theorem messageOf_zeros (d k : ℕ) : messageOf d (List.replicate k field.EXT_ZERO) = fun _ => 0 := by
  funext j
  simp only [messageOf, List.map_replicate, cubicOf_zero, List.getD_eq_getElem?_getD, List.getElem?_replicate]
  split <;> rfl

theorem roundAt_zero (d : ℕ) (r : Cubic) : roundAt d 0 (fun _ => 0) r = 0 := by
  have values : Lib.roundValues d (0 : Cubic) (fun _ => 0) = 0 := by
    funext k
    simp [Lib.roundValues]
  simp [roundAt, values]

theorem claimAfter_zeros (d : ℕ) :
    ∀ rs : List ((Fin (d + 1) → Cubic) × Cubic), (∀ p ∈ rs, p.1 = fun _ => 0) → claimAfter d rs 0 = 0
  | [], _ => rfl
  | p :: rs, zero => by
      have head : p.1 = fun _ => 0 := zero p List.mem_cons_self
      have rest := claimAfter_zeros d rs fun q member => zero q (List.mem_cons_of_mem _ member)
      simp only [claimAfter, List.foldl_cons, head, roundAt_zero] at rest ⊢
      exact rest

theorem constraint_zero : constraint (fun _ : Fin 8 => (0 : Cubic)) = 0 := by
  simp [constraint]

theorem batch_zero (ρ : Cubic) : batch ρ (fun _ : Fin 8 => (0 : Cubic)) = 0 := by
  simp [batch]

theorem matrices_empty (T : Template Gl) (rx ry : List field.Ext) (ρ : field.Ext) :
    (evalOf T emptyCounts emptyRun rx ry ρ).1 +
      blockPart T emptyCounts (rx.map cubicOf) (ry.map cubicOf) (cubicOf ρ) = 0 := by
  simp [evalOf, emptyRun, glueTotal, blockPart, emptyCounts]

theorem valuesOf_zero : valuesOf zeroRounds = fun _ => 0 := by
  funext j
  simp [valuesOf, zeroRounds, cubicOf_zero]

theorem zero_fit (T : Template Gl) : RoundsFit T emptyCounts zeroRounds :=
  ⟨by simp [zeroRounds, rowVars_empty], by simp [zeroRounds], by simp [zeroRounds, cellVars_empty],
    by simp [zeroRounds]⟩

theorem zero_outer (d : Draws) : OuterOk zeroRounds d := by
  have pairs : ∀ p ∈ outerPairs zeroRounds d, p.1 = fun _ => 0 := by
    intro p member
    simp only [outerPairs, zeroRounds, List.mem_map] at member
    obtain ⟨q, inZip, rfl⟩ := member
    have := (List.of_mem_zip inZip).1
    simp only [List.mem_singleton] at this
    rw [this, messageOf_zeros]
  rw [OuterOk, outerLast, claimAfter_zeros 7 _ pairs, valuesOf_zero, constraint_zero, mul_zero]

theorem zero_inner (T : Template Gl) (stmt : List field.Gl) (d : Draws) (w : Cubic) :
    InnerOk T (evalOf T emptyCounts emptyRun) emptyCounts stmt zeroRounds d w := by
  have pairs : ∀ p ∈ innerPairs zeroRounds d, p.1 = fun _ => 0 := by
    intro p member
    simp only [innerPairs, zeroRounds, List.mem_map] at member
    obtain ⟨q, inZip, rfl⟩ := member
    have := (List.of_mem_zip inZip).1
    simp only [List.mem_cons, List.not_mem_nil, or_false] at this
    rcases this with same | same <;>
      rw [same, show [field.EXT_ZERO, field.EXT_ZERO] = List.replicate 2 field.EXT_ZERO from rfl, messageOf_zeros]
  rw [InnerOk, innerLast, valuesOf_zero, batch_zero, claimAfter_zeros 1 _ pairs, matrices_empty, zero_mul]

theorem zero_checks (T : Template Gl) (d : Draws) (w : Cubic) (final : Bool) :
    Checks T (evalOf T emptyCounts emptyRun) emptyCounts [] zeroRounds d w final :=
  ⟨zero_fit T, zero_outer d, emptyRun_counts, rfl, fun _ => zero_inner T [] d w⟩

theorem small_empty (T : Template Gl) : Small T emptyCounts [] := by
  refine ⟨?_, ?_, ?_, by simp⟩
  · simp [partsEnd, emptyCounts]
  · simp [partsEnd, emptyCounts]
  · rw [Layout.one, cellVars_empty]
    simp [emptyCounts]

theorem honest_accepts_zero (T : Template Gl) :
    ∃ h, honest T (recorder zeroOracle zeroOpening) (evalOf T emptyCounts emptyRun) emptyCounts [] zeroRounds [] =
      ok (true, h) := by
  obtain ⟨d, run, -, -, -, -, h', hopened⟩ :=
    schedule_recorder zeroOracle zeroOpening (Layout.rowVars T emptyCounts) (Layout.cellVars T emptyCounts) [] zeroRounds
  refine ⟨scheduled [] zeroRounds (Layout.cellVars T emptyCounts) d, ?_⟩
  simp only [honest, model, run, bind_ok, hopened, zeroOpening, Result.ok.injEq, Prod.mk.injEq, decide_eq_true_eq, and_true]
  exact zero_checks T d _ true

/-- An honest run exists for every template. -/
theorem zero_honest (T : Template Gl) :
    ∃ (prog : Unit → Run Gl) (s : Counts) (L : ℕ) (stmt : List field.Gl) (π : Rounds)
      (cands : {l : List (ℕ → Gl) // l.length ≤ L}) (oracle : History → field.Ext)
      (opening : History → List Cubic → Option field.Ext),
      WhirSound cands.1 opening ∧ Honest T prog s L () stmt π cands oracle opening := by
  obtain ⟨h, accepts⟩ := honest_accepts_zero T
  obtain ⟨d, w, run, hw, publics, tr, reach⟩ := honest_sound T (fun _ : Unit => emptyRun) emptyCounts 1 () []
    zeroRounds ⟨[zeroTable], by simp⟩ zeroOracle zeroOpening h whir_zero accepts
  exact ⟨fun _ => emptyRun, emptyCounts, 1, [], zeroRounds, ⟨[zeroTable], by simp⟩, zeroOracle, zeroOpening, whir_zero,
    small_empty T, by simp, publics, zero_fit T, d, h, w, run, hw, tr, reach⟩

theorem honest_complete (T : Template Gl) : Complete (honest T) T := by
  refine ⟨zero_honest T, ?_⟩
  intro I _ prog s L i stmt π cands oracle opening honestRun
  obtain ⟨-, -, publics, fit, d, h, w, run, hw, tr, reach⟩ := honestRun
  obtain ⟨d', run', len_tau, len_rx, len_ry, -, -⟩ :=
    schedule_recorder oracle opening (Layout.rowVars T s) (Layout.cellVars T s) stmt π
  rw [run'] at run
  simp only [Result.ok.injEq, Prod.mk.injEq] at run
  obtain ⟨rfl, rfl⟩ := run
  have len : Lengths T s π d' := ⟨fit.1, fit.2.2.1, len_tau, len_rx, len_ry⟩
  have outerOk := run_outer_ok len reach
  have path := run_path T s prog L i cands π d' (cubicOf w) len outerOk
  have same := reaches_unique path reach
  split_ifs at same with final
  obtain ⟨counts, inner⟩ := checks_of_final publics final
  refine ⟨scheduled stmt π (Layout.cellVars T s) d', ?_⟩
  simp only [honest, model, run', bind_ok, hw, Result.ok.injEq, Prod.mk.injEq, decide_eq_true_eq, and_true]
  exact ⟨fit, outerOk, counts, publics, fun _ => inner⟩

theorem honest_simulates (T : Template Gl) : Simulates (honest T) T :=
  ⟨honest_sound T, honest_complete T⟩

/-! ## Each weaker model fails the statement -/

/-- The draws of the zero oracle are zeros. -/
theorem zero_draws {d : Draws} (drawn : ∀ c ∈ d.tau ++ d.rx ++ d.rho :: d.ry, ∃ h', c = zeroOracle h') :
    ∀ c ∈ d.tau ++ d.rx ++ d.rho :: d.ry, c = field.EXT_ZERO := by
  intro c member
  obtain ⟨_, rfl⟩ := drawn c member
  rfl

/-- `skipsOpening` accepts the zero run when the opening fails, which `Sound` forbids. -/
theorem skipsOpening_not_sound (T : Template Gl) : ¬ Sound (skipsOpening T) T := by
  intro sound
  obtain ⟨d, run, -, -, -, -, h', hopened⟩ := schedule_recorder zeroOracle failedOpening
    (Layout.rowVars T emptyCounts) (Layout.cellVars T emptyCounts) [] zeroRounds
  have accepts : skipsOpening T (recorder zeroOracle failedOpening) (evalOf T emptyCounts emptyRun) emptyCounts []
      zeroRounds [] = ok (true, scheduled [] zeroRounds (Layout.cellVars T emptyCounts) d) := by
    simp only [skipsOpening, model, run, bind_ok, hopened, failedOpening, Bool.false_eq_true, ite_false, Result.ok.injEq, Prod.mk.injEq, decide_eq_true_eq, and_true]
    exact zero_checks T d _ true
  obtain ⟨d', w, run', hw, -⟩ := sound (fun _ : Unit => emptyRun) emptyCounts 0 () [] zeroRounds ⟨[], by simp⟩
    zeroOracle failedOpening _ (whir_failed []) accepts
  rw [run] at run'
  simp only [Result.ok.injEq, Prod.mk.injEq] at run'
  obtain ⟨rfl, -⟩ := run'
  rw [hopened] at hw
  simp [failedOpening] at hw

/-- The zero run with a nonzero constraint value: its outer check fails. -/
def badRounds : Rounds := { zeroRounds with values := fun j => if j = 0 then field.EXT_ONE else field.EXT_ZERO }

theorem bad_outer {d : Draws} (tau : d.tau = [field.EXT_ZERO]) (rx : d.rx = [field.EXT_ZERO]) :
    ¬ OuterOk badRounds d := by
  have pairs : ∀ p ∈ outerPairs badRounds d, p.1 = fun _ => 0 := by
    intro p member
    simp only [outerPairs, badRounds, zeroRounds, List.mem_map] at member
    obtain ⟨q, inZip, rfl⟩ := member
    have := (List.of_mem_zip inZip).1
    simp only [List.mem_singleton] at this
    rw [this, messageOf_zeros]
  have values : valuesOf badRounds = fun j => if j = 0 then 1 else 0 := by
    funext j
    by_cases zero : j = 0 <;> simp [valuesOf, badRounds, zero, cubicOf_one, cubicOf_zero]
  rw [OuterOk, outerLast, claimAfter_zeros 7 _ pairs, values, tau, rx]
  simp [eqEval, constraint, cubicOf_zero]

/-- `acceptsAll` accepts a run whose outer check fails, which `Sound` forbids. -/
theorem acceptsAll_not_sound (T : Template Gl) : ¬ Sound (acceptsAll T) T := by
  intro sound
  obtain ⟨d, run, len_tau, len_rx, len_ry, drawn, -⟩ := schedule_recorder zeroOracle zeroOpening
    (Layout.rowVars T emptyCounts) (Layout.cellVars T emptyCounts) [] badRounds
  have accepts : acceptsAll T (recorder zeroOracle zeroOpening) (evalOf T emptyCounts emptyRun) emptyCounts []
      badRounds [] = ok (true, scheduled [] badRounds (Layout.cellVars T emptyCounts) d) := by
    simp only [acceptsAll, run, bind_ok]
  obtain ⟨d', w, run', hw, -, tr, reach⟩ := sound (fun _ : Unit => emptyRun) emptyCounts 1 () [] badRounds
    ⟨[zeroTable], by simp⟩ zeroOracle zeroOpening _ whir_zero accepts
  rw [run] at run'
  simp only [Result.ok.injEq, Prod.mk.injEq] at run'
  obtain ⟨rfl, -⟩ := run'
  have zeros := zero_draws drawn
  have tau : d.tau = [field.EXT_ZERO] := by
    rw [rowVars_empty] at len_tau
    exact List.eq_replicate_iff.mpr ⟨len_tau, fun c member => zeros c (by simp [member])⟩
  have rx : d.rx = [field.EXT_ZERO] := by
    exact List.eq_replicate_iff.mpr ⟨by rw [len_rx], fun c member => zeros c (by simp [member])⟩
  have len : Lengths T emptyCounts badRounds d :=
    ⟨by simp [badRounds, zeroRounds, rowVars_empty], by simp [badRounds, zeroRounds, cellVars_empty], len_tau, len_rx,
      len_ry⟩
  exact bad_outer tau rx (run_outer_ok len reach)

/-- The zero run with a last inner message of one: the inner rounds end at one. -/
def oneRounds : Rounds :=
  { zeroRounds with inner := [[field.EXT_ZERO, field.EXT_ZERO], [field.EXT_ONE, field.EXT_ZERO]] }

theorem roundAt_node_zero (d : ℕ) (small : d + 2 ≤ goldilocksModulus) (claim : Cubic) (m : Fin (d + 1) → Cubic) :
    roundAt d claim m 0 = m 0 := by
  have distinct : Lib.NodesDistinct Cubic d :=
    fun a b same => Fin.ext (cubic_natCast_injective (d + 2) small a b a.isLt b.isLt same)
  have := Lib.roundPoly_eval_node distinct claim m 0
  simpa [roundAt, Lib.roundPoly, Lib.node, Lib.roundValues] using this

theorem one_inner {d : Draws} (ry : d.ry = [field.EXT_ZERO, field.EXT_ZERO]) :
    innerLast oneRounds d = 1 := by
  have values : valuesOf oneRounds = fun _ => 0 := by
    funext j
    simp [valuesOf, oneRounds, zeroRounds, cubicOf_zero]
  have first : messageOf 1 [field.EXT_ZERO, field.EXT_ZERO] = fun _ => 0 :=
    messageOf_zeros 1 2
  have second : messageOf 1 [field.EXT_ONE, field.EXT_ZERO] 0 = 1 := by
    simp [messageOf, cubicOf_one]
  rw [innerLast, values, batch_zero, innerPairs, ry]
  simp only [oneRounds, zeroRounds, List.zip_cons_cons, List.zip_nil_right, List.map_cons, List.map_nil, claimAfter,
    List.foldl_cons, List.foldl_nil, first, cubicOf_zero, roundAt_zero]
  rw [roundAt_node_zero 1 (by norm_num [goldilocksModulus]), second]

/-- `skipsFinal` accepts a run whose final check fails, which `Sound` forbids. -/
theorem skipsFinal_not_sound (T : Template Gl) : ¬ Sound (skipsFinal T) T := by
  intro sound
  obtain ⟨d, run, len_tau, len_rx, len_ry, drawn, h', hopened⟩ := schedule_recorder zeroOracle zeroOpening
    (Layout.rowVars T emptyCounts) (Layout.cellVars T emptyCounts) [] oneRounds
  have fit : RoundsFit T emptyCounts oneRounds :=
    ⟨by simp [oneRounds, zeroRounds, rowVars_empty], by simp [oneRounds, zeroRounds],
      by simp [oneRounds, zeroRounds, cellVars_empty], by simp [oneRounds, zeroRounds]⟩
  have accepts : skipsFinal T (recorder zeroOracle zeroOpening) (evalOf T emptyCounts emptyRun) emptyCounts []
      oneRounds [] = ok (true, scheduled [] oneRounds (Layout.cellVars T emptyCounts) d) := by
    have checks : Checks T (evalOf T emptyCounts emptyRun) emptyCounts [] oneRounds d (cubicOf field.EXT_ZERO) false :=
      ⟨fit, zero_outer d, emptyRun_counts, rfl, by simp⟩
    simp only [skipsFinal, model, run, bind_ok, hopened, zeroOpening, Result.ok.injEq, Prod.mk.injEq, decide_eq_true_eq, and_true]
    exact checks
  obtain ⟨d', w, run', hw, -, tr, reach⟩ := sound (fun _ : Unit => emptyRun) emptyCounts 1 () [] oneRounds
    ⟨[zeroTable], by simp⟩ zeroOracle zeroOpening _ whir_zero accepts
  rw [run] at run'
  simp only [Result.ok.injEq, Prod.mk.injEq] at run'
  obtain ⟨rfl, -⟩ := run'
  have zeros := zero_draws drawn
  have ry : d.ry = [field.EXT_ZERO, field.EXT_ZERO] :=
    List.eq_replicate_iff.mpr ⟨by rw [len_ry], fun c member => zeros c (by simp [member])⟩
  have len : Lengths T emptyCounts oneRounds d := ⟨fit.1, fit.2.2.1, len_tau, len_rx, len_ry⟩
  have path := run_path T emptyCounts (fun _ : Unit => emptyRun) 1 () ⟨[zeroTable], by simp⟩ oneRounds d (cubicOf w) len
    (zero_outer d)
  have same := reaches_unique path reach
  split_ifs at same with final
  obtain ⟨-, -, last⟩ := final
  rw [one_inner ry] at last
  have zero : verifierMatrices T emptyCounts emptyRun (d.rx.map cubicOf) (d.ry.map cubicOf) (cubicOf d.rho) = 0 :=
    matrices_empty T d.rx d.ry d.rho
  rw [zero, zero_mul] at last
  exact one_ne_zero last

/-- The history of `scheduleBlind` on `recorder`: `scheduled` without the observation of the matrix values. -/
noncomputable def scheduledBlind (stmt : List field.Gl) (π : Rounds) (n : ℕ) (d : Draws) : History :=
  [.words (stmt.map glOf), .commit] ++ d.tau.map (fun c => .sample (cubicOf c)) ++ sent π.outer d.rx ++
    [.sample (cubicOf d.rho)] ++ sent π.inner d.ry ++ [.opened ((d.ry.take n).map cubicOf) (d.opened.map cubicOf)]

theorem scheduleBlind_recorder (oracle : History → field.Ext) (opening : History → List Cubic → Option field.Ext)
    (m n : ℕ) (stmt : List field.Gl) (π : Rounds) :
    ∃ d, scheduleBlind (recorder oracle opening) m n stmt π [] = ok (d, scheduledBlind stmt π n d) ∧
      d.tau.length = m ∧ d.rx.length = π.outer.length ∧ d.ry.length = π.inner.length ∧
      ∃ h', d.opened = opening h' ((d.ry.take n).map cubicOf) := by
  obtain ⟨tau, run1, len1, -⟩ := samples_recorder oracle opening m ([] ++ [.words (stmt.map glOf)] ++ [.commit])
  obtain ⟨rx, run2, len2, -⟩ := sends_recorder oracle opening π.outer
    ([] ++ [.words (stmt.map glOf)] ++ [.commit] ++ tau.map fun c => .sample (cubicOf c))
  let h3 := [] ++ [.words (stmt.map glOf)] ++ [.commit] ++ (tau.map fun c => .sample (cubicOf c)) ++ sent π.outer rx
  obtain ⟨ry, run3, len3, -⟩ := sends_recorder oracle opening π.inner (h3 ++ [.sample (cubicOf (oracle h3))])
  let h4 := h3 ++ [.sample (cubicOf (oracle h3))] ++ sent π.inner ry
  refine ⟨⟨tau, rx, oracle h3, ry, opening h4 ((ry.take n).map cubicOf)⟩, ?_, len1, len2, len3, h4, rfl⟩
  simp only [scheduleBlind, recorder_observeWords, recorder_commit, bind_ok]
  rw [run1]
  simp only [bind_ok]
  rw [run2]
  simp only [bind_ok, recorder_sample]
  rw [run3]
  simp only [bind_ok, recorder_openAt, scheduledBlind, h3, h4, List.nil_append, List.append_assoc, List.cons_append]

theorem sent_length (ms : List (List field.Ext)) (cs : List field.Ext) :
    (sent ms cs).length = 2 * (ms.zip cs).length := by
  simp [sent, List.length_flatMap, List.sum_replicate, Nat.mul_comm]

/-- `skipsObserve` accepts the zero run, but its channel calls are not `schedule`'s, which `Sound`
forbids: the batching draw would not bind the matrix values. -/
theorem skipsObserve_not_sound (T : Template Gl) : ¬ Sound (skipsObserve T) T := by
  intro sound
  obtain ⟨d, run, len_tau, len_rx, len_ry, h', hopened⟩ := scheduleBlind_recorder zeroOracle zeroOpening
    (Layout.rowVars T emptyCounts) (Layout.cellVars T emptyCounts) [] zeroRounds
  have accepts : skipsObserve T (recorder zeroOracle zeroOpening) (evalOf T emptyCounts emptyRun) emptyCounts []
      zeroRounds [] = ok (true, scheduledBlind [] zeroRounds (Layout.cellVars T emptyCounts) d) := by
    simp only [skipsObserve, model, run, bind_ok, hopened, zeroOpening, Result.ok.injEq, Prod.mk.injEq, decide_eq_true_eq, and_true]
    exact zero_checks T d _ true
  obtain ⟨d', w, run', -⟩ := sound (fun _ : Unit => emptyRun) emptyCounts 1 () [] zeroRounds ⟨[zeroTable], by simp⟩
    zeroOracle zeroOpening _ whir_zero accepts
  obtain ⟨d'', run'', len_tau', len_rx', len_ry', -, -⟩ := schedule_recorder zeroOracle zeroOpening
    (Layout.rowVars T emptyCounts) (Layout.cellVars T emptyCounts) [] zeroRounds
  rw [run''] at run'
  simp only [Result.ok.injEq, Prod.mk.injEq] at run'
  have lengths := congrArg List.length run'.2
  simp only [scheduled, scheduledBlind, List.length_append, List.length_cons, List.length_nil, List.length_map,
    sent_length, List.length_zip, len_tau, len_rx, len_ry, len_tau', len_rx', len_ry'] at lengths
  omega

/-- `skipsPublics` accepts the zero run with the statement `[0]`, but the empty program has no
public word, which `Sound` forbids. -/
theorem skipsPublics_not_sound (T : Template Gl) : ¬ Sound (skipsPublics T) T := by
  intro sound
  obtain ⟨d, run, -, -, -, -, h', hopened⟩ := schedule_recorder zeroOracle zeroOpening
    (Layout.rowVars T emptyCounts) (Layout.cellVars T emptyCounts) [field.GL_ZERO] zeroRounds
  have accepts : skipsPublics T (recorder zeroOracle zeroOpening) (evalOf T emptyCounts emptyRun) emptyCounts
      [field.GL_ZERO] zeroRounds [] = ok (true, scheduled [field.GL_ZERO] zeroRounds (Layout.cellVars T emptyCounts) d) := by
    simp only [skipsPublics, honest, model, run, bind_ok, hopened, zeroOpening, Result.ok.injEq, Prod.mk.injEq,
      decide_eq_true_eq, and_true]
    exact ⟨zero_fit T, zero_outer d, emptyRun_counts, rfl, fun _ => zero_inner T [field.GL_ZERO] d _⟩
  obtain ⟨-, -, -, -, publics, -⟩ := sound (fun _ : Unit => emptyRun) emptyCounts 1 () [field.GL_ZERO] zeroRounds
    ⟨[zeroTable], by simp⟩ zeroOracle zeroOpening _ whir_zero accepts
  simp [emptyRun] at publics

/-- The empty run with one glue cell. -/
def cellRun : Run Gl := ⟨[], 1, 0, []⟩

/-- `skipsCounts` accepts the zero run for the empty shape, but the program has one glue cell, so
`shrinkGame` rejects the play, which `Sound` forbids. -/
theorem skipsCounts_not_sound (T : Template Gl) : ¬ Sound (skipsCounts T) T := by
  intro sound
  obtain ⟨d, run, len_tau, len_rx, len_ry, -, h', hopened⟩ := schedule_recorder zeroOracle zeroOpening
    (Layout.rowVars T emptyCounts) (Layout.cellVars T emptyCounts) [] zeroRounds
  have accepts : skipsCounts T (recorder zeroOracle zeroOpening) (evalOf T emptyCounts cellRun) emptyCounts []
      zeroRounds [] = ok (true, scheduled [] zeroRounds (Layout.cellVars T emptyCounts) d) := by
    simp only [skipsCounts, honest, model, run, bind_ok, hopened, zeroOpening, Result.ok.injEq, Prod.mk.injEq,
      decide_eq_true_eq, and_true]
    exact ⟨zero_fit T, zero_outer d, rfl, rfl, fun _ => zero_inner T [] d _⟩
  obtain ⟨d', w, run', -, -, tr, reach⟩ := sound (fun _ : Unit => cellRun) emptyCounts 1 () [] zeroRounds
    ⟨[zeroTable], by simp⟩ zeroOracle zeroOpening _ whir_zero accepts
  rw [run] at run'
  simp only [Result.ok.injEq, Prod.mk.injEq] at run'
  obtain ⟨rfl, -⟩ := run'
  have path := run_path T emptyCounts (fun _ : Unit => cellRun) 1 () ⟨[zeroTable], by simp⟩ zeroRounds d (cubicOf w)
    ⟨(zero_fit T).1, (zero_fit T).2.2.1, len_tau, len_rx, len_ry⟩ (zero_outer d)
  have same := reaches_unique path reach
  split_ifs at same with final
  obtain ⟨counts, -⟩ := final
  simp [Run.counts, Run.rows, cellRun, emptyCounts] at counts

/-- `rejectsAll` is `Sound` but not `Complete`. -/
theorem rejectsAll_not_complete (T : Template Gl) : ¬ Complete rejectsAll T := by
  rintro ⟨-, all⟩
  obtain ⟨prog, s, L, stmt, π, cands, oracle, opening, -, honestRun⟩ := zero_honest T
  obtain ⟨h, accepts⟩ := all prog s L () stmt π cands oracle opening honestRun
  simp [rejectsAll] at accepts

theorem rejectsAll_sound (T : Template Gl) : Sound rejectsAll T := by
  intro I _ prog s L i stmt π cands oracle opening h whir accepts
  simp [rejectsAll] at accepts

end CompressionLeanRust.Verifier
