import NightstreamFPrime.Spec.Algebra
import NightstreamFPrime.Spec.Compression.Akita.Rows
import NightstreamFPrime.Spec.Compression.Game.Law

/-! Akita (ePrint 2026/1983, revision of 2026-11-23) as a polynomial commitment scheme, and the
binding of the commitment of the compression key (`SECURITY_MODEL.md`, trust item 7).

- `Pcs` has the parts of Akita Definition 3.5 that the security notions read. `Prover` and
  `Extractor` are those of Akita Definition 3.7.
- `Schedule.Admissible` is Akita Definition 9.2. The conditions that a schedule file decides are
  those of `Row.Conditions` (`Akita/Rows.lean`). A schedule with the row of `key20.aks` is
  admissible when it has the other conditions (`Schedule.admissible_key20_iff`), and no schedule
  with a row of `key30.aks` is admissible (`Schedule.not_admissible_key30`).
- `Use` is the one Akita opening of a final proof (SPEC.md Algorithm 10, step 11). `Use.Binding`
  states that the commitment of the compression key opens only to the key polynomial, which
  `Akita/Binding.lean` reduces to Module-SIS. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita

open scoped ENNReal

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

end Extractor

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

end Schedule

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

end Use

end NightstreamFPrime.Spec.Compression.Akita
