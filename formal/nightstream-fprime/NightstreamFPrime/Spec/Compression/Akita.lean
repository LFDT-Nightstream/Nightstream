import NightstreamFPrime.Spec.Algebra
import NightstreamFPrime.Spec.Compression.Akita.Rows
import NightstreamFPrime.Spec.Compression.Game.Law

/-! Akita (ePrint 2026/1983, revision of 2026-10-15) as a polynomial commitment scheme, and the
premises of `[AKITA]` (`SECURITY_MODEL.md`, trust item 7).

- `Pcs` has the parts of Akita Definition 3.5 that the security notions read. `KnowledgeSound` is
  Akita Definition 3.7 for a batch of claims (Akita (173)).
- `Schedule.Admissible` is Akita Definition 9.2. The conditions that a schedule file decides are
  those of `Row.Conditions` (`Akita/Rows.lean`). A schedule with the row of `key20.aks` is
  admissible when it has the other conditions (`Schedule.admissible_key20_iff`), and no schedule
  with a row of `key30.aks` is admissible (`Schedule.not_admissible_key30`).
- `Theorem1046` is Akita Theorem 10.46: under an admissible schedule, each prover with
  `Q ≤ Q_max` queries to `H_FS` has an extractor with the error
  `(Q + 1) · ε_int + Δ_setup + Σ_I Adv_I(B_I)`. Akita's terminal and its root commitments enter
  the transcript as digests, so Akita Lemma 10.47 adds `ε_hash` (`Schedule.knowledgeError`).
- `Use` is the one Akita opening of a final proof (SPEC.md Algorithm 10, step 11). Its premises
  are `Use.Knowledge` (Theorem 10.46 with its conditions), `Use.Binding` (the commitment of the
  compression key opens only to the key polynomial, which Theorem 10.46 does not give) and
  `Use.Composition` (an extracted win is a win of the game with ideal openings, which no result of
  Akita gives, because the prover commits `V` after the challenge `τ`).
- `Use.wins_le`: under these premises, the prover with Akita openings wins at most `Use.error`
  more often than the game. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita

open scoped ENNReal
open NightstreamFPrime.Spec.Compression.Game (event_ne_top map_fst_bind)

/-- The parts of a multilinear polynomial commitment scheme (Akita Definition 3.5) that its
security notions read: the evaluation `f(r)` of a table, the decommitment relation `Rel_com`, and
the verifier of a batch of claims `(cm, r, v)` (Akita (172)). -/
structure Pcs where
  Params : Type
  Commitment : Type
  Table : Type
  State : Type
  Point : Type
  Value : Type
  Proof : Type
  eval : Table → Point → Value
  relCom : Params → Commitment → Table → State → Prop
  verify : Params → (n : ℕ) → (Fin n → Commitment × Point × Value) → Proof → Prop

/-- An opening claim `(cm, r, v)`. -/
abbrev Pcs.Claim (S : Pcs) : Type := S.Commitment × S.Point × S.Value

/-- Akita (27) for each claim, which is the batch relation (173) with one claim for each
commitment: the witness decommits the commitment, and its table has the value at the point. -/
def Pcs.Holds (S : Pcs) (pp : S.Params) {n : ℕ} (claims : Fin n → S.Claim)
    (witness : Fin n → S.Table × S.State) : Prop :=
  ∀ i, S.relCom pp (claims i).1 (witness i).1 (witness i).2 ∧
    S.eval (witness i).1 (claims i).2.1 = (claims i).2.2

/-- A prover of the knowledge experiment of Akita Definition 3.7 for a batch of `n` claims: the
law of its tape, which fixes its coins and the answers of the random oracle; its claims and proof;
its number `Q` of queries to `H_FS`; and, for the 256-bit digests of transcript entries (Akita
Lemma 10.47), its number of digest queries and the number of digest positions in all queries. The
counts include the queries of the verifier on the output. -/
structure Prover (S : Pcs) (n : ℕ) where
  Tape : Type
  tape : PMF Tape
  output : Tape → (Fin n → S.Claim) × S.Proof
  queries : ℕ
  digestQueries : ℕ
  digestPositions : ℕ

/-- An extractor with resettable black-box access to a prover (Akita Definition 3.7): from the
prover's tape, which fixes the execution that it runs again, and its own coins, a witness or `⊥`. -/
structure Extractor {S : Pcs} {n : ℕ} (prover : Prover S n) where
  Coins : Type
  coins : PMF Coins
  run : prover.Tape → Coins → Option (Fin n → S.Table × S.State)

namespace Extractor

variable {S : Pcs} {n : ℕ} {prover : Prover S n} (extractor : Extractor prover)

/-- The law of the prover's tape and the extractor's coins. -/
noncomputable def law : PMF (prover.Tape × extractor.Coins) :=
  prover.tape.bind fun tape => extractor.coins.map (tape, ·)

/-- The probability of an event of the tape and the coins. -/
noncomputable def chance (event : prover.Tape × extractor.Coins → Prop) : ℝ :=
  (extractor.law.toOuterMeasure {draw | event draw}).toReal

/-- The extractor returns a witness for the prover's claims. -/
def Extracts (pp : S.Params) (draw : prover.Tape × extractor.Coins) : Prop :=
  ∃ witness, extractor.run draw.1 draw.2 = some witness ∧
    S.Holds pp (prover.output draw.1).1 witness

/-- The event of Akita Definition 3.7: the verifier accepts, and the extractor returns `⊥` or no
witness. -/
def Fails (pp : S.Params) (draw : prover.Tape × extractor.Coins) : Prop :=
  S.verify pp n (prover.output draw.1).1 (prover.output draw.1).2 ∧ ¬ extractor.Extracts pp draw

end Extractor

/-- Akita Definition 3.7 for a batch: each admitted prover `p` has an efficient extractor that
fails with probability at most `κ p`. `Efficient` is expected polynomial time. -/
def KnowledgeSound (S : Pcs) (n : ℕ) (pp : S.Params) (Admitted : Prover S n → Prop)
    (Efficient : (prover : Prover S n) → Extractor prover → Prop) (κ : Prover S n → ℝ) : Prop :=
  ∀ prover, Admitted prover → ∃ extractor, Efficient prover extractor ∧
    extractor.chance (extractor.Fails pp) ≤ κ prover

/-- The values of an Akita schedule that Akita Definition 9.2 and Theorem 10.46 read: the row of
its schedule file, the profile `(λ_FS, Q_max)`, the worst-case count of honest `H_FS` queries,
`ε_int` (Akita (194)), `N_setup` (Akita (34)), the draw width `w` of the setup expansion
(Akita (36)), and the count of Module-SIS instances `I(sch)`. A schedule file holds none of these
numbers. `rootBatching`, `compatible`, `extraction` and `msisFloor` are the conditions of Akita
Section 9.3 that the file does not decide: the root batching conditions of I1; C1, C2 and C4 to
C6; E1 to E3; and S1. They read the challenge families, the samplers, the setup registry and the
Module-SIS instances, which Lean does not model. -/
structure Schedule where
  row : Row
  lambdaFS : ℕ
  queryMax : ℕ
  honestQueries : ℕ
  interactiveError : ℝ
  setupLength : ℕ
  drawBits : ℕ
  instances : ℕ
  rootBatching : Prop
  compatible : Prop
  extraction : Prop
  msisFloor : Prop

namespace Schedule

variable (sch : Schedule)

/-- The conditions of Akita Definition 9.2 that the schedule file does not decide: those of
`rootBatching`, `compatible`, `extraction` and `msisFloor`, S2 (at most `Q_max` honest queries)
and S3 (the budget of Akita (195)). -/
def Rest : Prop :=
  sch.rootBatching ∧ sch.compatible ∧ sch.extraction ∧ sch.msisFloor ∧
    sch.honestQueries ≤ sch.queryMax ∧
    (sch.queryMax + 1 : ℝ) * sch.interactiveError ≤ 1 / 2 ^ sch.lambdaFS

/-- Akita Definition 9.2: a schedule in the family F1 to F3 with items (i) to (iv). -/
def Admissible : Prop := sch.row.Conditions ∧ sch.Rest

/-- A schedule with the row of `key20.aks` is admissible exactly when it has the conditions that
the file does not decide. -/
theorem admissible_key20_iff (file : sch.row = key20) : sch.Admissible ↔ sch.Rest := by
  unfold Admissible
  rw [file]
  exact and_iff_right key20_conditions

/-- No schedule with a row of `key30.aks` is admissible: the row is not in the family F1. -/
theorem not_admissible_key30 (file : sch.row = key30W27 ∨ sch.row = key30W26) :
    ¬ sch.Admissible := by
  rintro ⟨⟨⟨methods, -⟩, -⟩, -⟩
  rcases file with file | file <;> rw [file] at methods
  exacts [key30W27_conditions.1 methods, key30W26_conditions.1 methods]

/-- `N_setup · q / 2^w`, the bound on `Δ_setup` of Akita Theorem 10.46. -/
noncomputable def setupGap : ℝ := sch.setupLength * fieldModulus / 2 ^ sch.drawBits

/-- The error of Akita (210) for the prover, `(Q + 1) · ε_int + Δ_setup + Σ_I Adv_I(B_I)`, plus
`(Q_H² + Q_H · V_H) / 2^256` for `Q_H` digest queries and `V_H` digest positions. The last term
bounds `ε_hash` of Akita (212), because each digest has 256 bits. `advantage I` is `Adv_I(B_I)`. -/
noncomputable def knowledgeError {S : Pcs} {n : ℕ} (prover : Prover S n)
    (advantage : Fin sch.instances → ℝ) : ℝ :=
  (prover.queries + 1) * sch.interactiveError + sch.setupGap + ∑ I, advantage I +
    (prover.digestQueries ^ 2 + prover.digestQueries * prover.digestPositions) / 2 ^ 256

end Schedule

/-- Akita Theorem 10.46 for the parameters `pp` of the schedule `sch`, with Akita Lemma 10.47 for
the digests of transcript entries: under an admissible schedule, each admitted prover with
`Q ≤ Q_max` queries to `H_FS` has an efficient extractor with the error `Schedule.knowledgeError`.
`advantage p I` is `Adv_I(B_I)` of the Module-SIS algorithm `B_I` that the theorem gives for `p`.
The theorem gives neither completeness nor binding. -/
def Theorem1046 (S : Pcs) (n : ℕ) (sch : Schedule) (pp : S.Params) (Admitted : Prover S n → Prop)
    (Efficient : (prover : Prover S n) → Extractor prover → Prop)
    (advantage : Prover S n → Fin sch.instances → ℝ) : Prop :=
  sch.Admissible → KnowledgeSound S n pp
    (fun prover => Admitted prover ∧ prover.queries ≤ sch.queryMax) Efficient fun prover =>
      sch.knowledgeError prover (advantage prover)

/-- The one Akita opening of a final proof: a batch of `claims` claims under `schedule` and
`params`, whose claim `key` is on the commitment `keyCommitment` of the compression key, with the
key polynomial `keyTable`; the admitted provers and the efficient extractors; and two error terms
for each prover, `Adv_I(B_I)` of Theorem 10.46 (`advantage`) and the binding error
(`bindingError`). -/
structure Use where
  scheme : Pcs
  claims : ℕ
  schedule : Schedule
  params : scheme.Params
  key : Fin claims
  keyCommitment : scheme.Commitment
  keyTable : scheme.Table
  Admitted : Prover scheme claims → Prop
  Efficient : (prover : Prover scheme claims) → Extractor prover → Prop
  advantage : Prover scheme claims → Fin schedule.instances → ℝ
  bindingError : Prover scheme claims → ℝ

namespace Use

variable (u : Use)

/-- Akita Theorem 10.46 for the prover `prover`, with the conditions of the theorem. -/
structure Knowledge (prover : Prover u.scheme u.claims) : Prop where
  theorem1046 :
    Theorem1046 u.scheme u.claims u.schedule u.params u.Admitted u.Efficient u.advantage
  admissible : u.schedule.Admissible
  admitted : u.Admitted prover
  budget : prover.queries ≤ u.schedule.queryMax

/-- The Akita verifier accepts the output on `tape`, and its claim `key` is on the commitment of
the compression key. -/
def Opens (prover : Prover u.scheme u.claims) (tape : prover.Tape) : Prop :=
  u.scheme.verify u.params u.claims (prover.output tape).1 (prover.output tape).2 ∧
    ((prover.output tape).1 u.key).1 = u.keyCommitment

/-- The extractor returns a decommitment of the commitment of the compression key to a table
that is not the key polynomial. -/
def KeyBroken {prover : Prover u.scheme u.claims} (extractor : Extractor prover)
    (draw : prover.Tape × extractor.Coins) : Prop :=
  ∃ witness, extractor.run draw.1 draw.2 = some witness ∧
    u.scheme.relCom u.params u.keyCommitment (witness u.key).1 (witness u.key).2 ∧
    (witness u.key).1 ≠ u.keyTable

/-- Binding of the commitment of the compression key to the key polynomial: an admitted prover
with an efficient extractor breaks it with probability at most `bindingError`. -/
def Binding : Prop :=
  ∀ prover (extractor : Extractor prover), u.Admitted prover → u.Efficient prover extractor →
    extractor.chance (u.KeyBroken extractor) ≤ u.bindingError prover

/-- `event` holds, and the extractor returns a witness for the claims with the key polynomial. -/
def Extracted {prover : Prover u.scheme u.claims} (event : prover.Tape → Prop)
    (extractor : Extractor prover) (draw : prover.Tape × extractor.Coins) : Prop :=
  event draw.1 ∧ ∃ witness, extractor.run draw.1 draw.2 = some witness ∧
    u.scheme.Holds u.params (prover.output draw.1).1 witness ∧ (witness u.key).1 = u.keyTable

/-- The composition of the Akita opening with the game of ideal openings, for the event `event`
of the prover with Akita openings and the probability `ideal` of its counterpart in the game: for
each efficient extractor, `Extracted` has probability at most `ideal`. No result of Akita gives
it: Akita Theorem 10.45 needs each commitment in the first message, and the prover commits `V`
after the challenge `τ` (SPEC.md section 5.4). -/
def Composition (prover : Prover u.scheme u.claims) (event : prover.Tape → Prop) (ideal : ℝ) :
    Prop :=
  ∀ extractor, u.Efficient prover extractor → extractor.chance (u.Extracted event extractor) ≤ ideal

/-- The error of `[AKITA]` for the prover: the error of Akita Theorem 10.46 with Lemma 10.47, and
the binding error. -/
noncomputable def error (prover : Prover u.scheme u.claims) : ℝ :=
  u.schedule.knowledgeError prover (u.advantage prover) + u.bindingError prover

/-- A win of the prover with Akita openings is a failure of the extractor of Theorem 10.46, a
break of the binding of the key, or an extracted win, which `composition` bounds by `ideal`. -/
theorem wins_le (prover : Prover u.scheme u.claims) (event : prover.Tape → Prop) (ideal : ℝ)
    (knowledge : u.Knowledge prover) (binding : u.Binding)
    (accepts : ∀ tape, event tape → u.Opens prover tape)
    (composition : u.Composition prover event ideal) :
    (prover.tape.toOuterMeasure {tape | event tape}).toReal ≤ ideal + u.error prover := by
  obtain ⟨extractor, efficient, fails⟩ := knowledge.theorem1046 knowledge.admissible prover
    ⟨knowledge.admitted, knowledge.budget⟩
  beta_reduce at fails
  have broken := binding prover extractor knowledge.admitted efficient
  have extracted := composition extractor efficient
  have marginal : prover.tape.toOuterMeasure {tape | event tape} =
      extractor.law.toOuterMeasure {draw | event draw.1} := by
    have mapped := congrArg (fun law => PMF.toOuterMeasure law {tape | event tape})
      (map_fst_bind prover.tape fun _ => extractor.coins)
    simp only [PMF.toOuterMeasure_map_apply] at mapped
    exact mapped.symm
  have cover : {draw | event draw.1} ⊆ ({draw | extractor.Fails u.params draw} ∪
      {draw | u.KeyBroken extractor draw}) ∪ {draw | u.Extracted event extractor draw} := by
    intro draw won
    obtain ⟨verified, keyed⟩ := accepts draw.1 won
    by_cases extracts : extractor.Extracts u.params draw
    · obtain ⟨witness, ran, holds⟩ := extracts
      by_cases same : (witness u.key).1 = u.keyTable
      · exact Or.inr ⟨won, witness, ran, holds, same⟩
      · have opens := (holds u.key).1
        rw [keyed] at opens
        exact Or.inl (Or.inr ⟨witness, ran, opens, same⟩)
    · exact Or.inl (Or.inl ⟨verified, extracts⟩)
  have split : (extractor.law.toOuterMeasure {draw | event draw.1}).toReal ≤
      extractor.chance (extractor.Fails u.params) + extractor.chance (u.KeyBroken extractor) +
        extractor.chance (u.Extracted event extractor) := by
    unfold Extractor.chance
    rw [← ENNReal.toReal_add (event_ne_top _ _) (event_ne_top _ _),
      ← ENNReal.toReal_add (ENNReal.add_ne_top.mpr ⟨event_ne_top _ _, event_ne_top _ _⟩)
        (event_ne_top _ _)]
    exact ENNReal.toReal_mono
      (ENNReal.add_ne_top.mpr ⟨ENNReal.add_ne_top.mpr ⟨event_ne_top _ _, event_ne_top _ _⟩,
        event_ne_top _ _⟩)
      ((MeasureTheory.measure_mono cover).trans ((MeasureTheory.measure_union_le _ _).trans
        (add_le_add (MeasureTheory.measure_union_le _ _) le_rfl)))
  rw [marginal]
  unfold error
  linarith

end Use

end NightstreamFPrime.Spec.Compression.Akita
