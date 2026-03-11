import SuperNeo.MLE
import SuperNeo.EqPoly
import Mathlib

/-!
SumCheck protocol scaffold.

This module provides:
- protocol objects (`SumCheckInstance`, `SumCheckTranscript`),
- standalone SumCheck scaffold semantics and honest-transcript closure,
- explicit soundness/completeness assumption boundaries.
-/

namespace SuperNeo

structure SumCheckInstance where
  rounds : Nat
  maxDegree : Nat
  domainSize : Nat
  claimedValue : F

structure SumCheckTranscript where
  challenges : Array F
  roundPolys : Array (Array F)

/-- Evaluate a univariate polynomial (coefficient form, low degree first). -/
def sumcheckEvalPoly (poly : Array F) (x : F) : F :=
  poly.foldr (fun c acc => c + x * acc) 0

/-- Basic transcript well-formedness against an instance. -/
def sumcheckRoundConsistent
  (inst : SumCheckInstance)
  (tr : SumCheckTranscript) : Prop :=
  tr.challenges.size = inst.rounds ∧
  tr.roundPolys.size = inst.rounds

/-- Each round polynomial has the expected coefficient length. -/
def sumcheckRoundPolyShape
  (inst : SumCheckInstance)
  (poly : Array F) : Prop :=
  poly.size = inst.maxDegree + 1

/-- Every round polynomial satisfies the expected shape. -/
def sumcheckRoundShapes
  (inst : SumCheckInstance)
  (tr : SumCheckTranscript) : Prop :=
  ∀ i : Fin tr.roundPolys.size,
    sumcheckRoundPolyShape inst tr.roundPolys[i.1]

/--
Round-transition consistency:
- each next-round polynomial satisfies the `0/1` sum equation against the
  previous round challenge evaluation,
- matching Definition-6 style fold transitions.
-/
def sumcheckFoldConsistent
  (tr : SumCheckTranscript) : Prop :=
  tr.challenges.size = tr.roundPolys.size ∧
  ∀ i : Nat,
    i + 1 < tr.roundPolys.size →
      sumcheckEvalPoly (tr.roundPolys[i + 1]!) 0 +
          sumcheckEvalPoly (tr.roundPolys[i + 1]!) 1 =
        sumcheckEvalPoly (tr.roundPolys[i]!) (tr.challenges[i]!)

/--
Initial round-sum consistency: the first round polynomial opened at `{0, 1}`
must sum to the claimed value.
-/
def sumcheckInitialRoundConsistent
  (inst : SumCheckInstance)
  (tr : SumCheckTranscript) : Prop :=
  if _hZero : tr.roundPolys.size = 0 then
    True
  else
    sumcheckEvalPoly (tr.roundPolys[0]!) 0 + sumcheckEvalPoly (tr.roundPolys[0]!) 1 =
      inst.claimedValue

/--
Legacy scaffold helper: final claim check against the first round polynomial's constant term.

This helper is no longer part of `sumcheckAcceptedCore`; endpoint consistency is
tracked separately by `sumcheckFinalOracleConsistent` and `sumcheckAcceptedForTable`.
-/
def sumcheckFinalClaimConsistent
  (inst : SumCheckInstance)
  (tr : SumCheckTranscript) : Prop :=
  if _hZero : tr.roundPolys.size = 0 then
    True
  else
    tr.roundPolys[0]!.getD 0 0 = inst.claimedValue

/-- Static parameter consistency required by the SumCheck instance. -/
def sumcheckParameterConsistent (inst : SumCheckInstance) : Prop :=
  inst.maxDegree ≤ inst.domainSize

/--
Internal completeness restriction of the current table-based SumCheck model.

For positive-round instances, the honest prover needs room for linear round
polynomials, so `maxDegree` must be positive. This is a modeling restriction of
the current standalone scaffold, not a full restatement of Definition 6.
-/
def sumcheckDegreeCompatible (inst : SumCheckInstance) : Prop :=
  inst.rounds = 0 ∨ 0 < inst.maxDegree

/-- Core scaffold acceptance checks (without endpoint-oracle consistency). -/
def sumcheckAcceptedCore
  (inst : SumCheckInstance)
  (tr : SumCheckTranscript) : Prop :=
  sumcheckParameterConsistent inst ∧
  sumcheckDegreeCompatible inst ∧
  sumcheckRoundConsistent inst tr ∧
  sumcheckRoundShapes inst tr ∧
  sumcheckInitialRoundConsistent inst tr ∧
  sumcheckFoldConsistent tr

/-- Legacy scaffold claim shape kept for compatibility with honest-transcript closure helpers. -/
def sumcheckBaseClaimTrue (inst : SumCheckInstance) : Prop :=
  sumcheckParameterConsistent inst ∧
  sumcheckDegreeCompatible inst

/--
Verifier-facing SumCheck claim object.

This packages an explicit transcript witness plus verifier-side predicates.
It is retained as an executable witness surface for reductions and checks.
-/
structure SumCheckClaim (inst : SumCheckInstance) where
  transcript : SumCheckTranscript
  parameterConsistent : sumcheckParameterConsistent inst
  degreeCompatible : sumcheckDegreeCompatible inst
  roundConsistent : sumcheckRoundConsistent inst transcript
  roundShapes : sumcheckRoundShapes inst transcript
  initialRound : sumcheckInitialRoundConsistent inst transcript
  foldConsistent : sumcheckFoldConsistent transcript

/-- Accepted transcript can be re-packaged as a claim object. -/
def sumcheckClaimOfAccepted
  {inst : SumCheckInstance}
  {tr : SumCheckTranscript}
  (hAcc : sumcheckAcceptedCore inst tr) : SumCheckClaim inst :=
  { transcript := tr
    parameterConsistent := hAcc.1
    degreeCompatible := hAcc.2.1
    roundConsistent := hAcc.2.2.1
    roundShapes := hAcc.2.2.2.1
    initialRound := hAcc.2.2.2.2.1
    foldConsistent := hAcc.2.2.2.2.2 }

/-- Claim-object witness implies core accepted transcript checks. -/
theorem SumCheckClaim.accepted
  {inst : SumCheckInstance}
  (c : SumCheckClaim inst) :
  sumcheckAcceptedCore inst c.transcript := by
  exact ⟨c.parameterConsistent, c.degreeCompatible, c.roundConsistent, c.roundShapes,
    c.initialRound, c.foldConsistent⟩

/--
Boolean-cube table sum surface used in the paper-facing SumCheck statement.

Interpretation: the table is indexed by `{0,1}^ell` (in fixed index order),
and `sumcheckTableSum` is the hypercube sum over all indices.
-/
def sumcheckTableSum (table : Array F) : F :=
  table.foldr (fun v acc => v + acc) 0

/--
Paper-facing SumCheck statement object.

This captures the core Definition-6 style claim that the asserted value equals
the sum over all hypercube-indexed table values.
-/
structure SumCheckStatement (inst : SumCheckInstance) where
  parameterConsistent : sumcheckParameterConsistent inst
  degreeCompatible : sumcheckDegreeCompatible inst
  table : Array F
  tableSize : table.size = 2 ^ inst.rounds
  hypercubeSumEqClaimed : sumcheckTableSum table = inst.claimedValue


/-- Final-round oracle consistency for a statement witness and transcript. -/
def sumcheckFinalOracleConsistent
  (inst : SumCheckInstance)
  (stmt : SumCheckStatement inst)
  (tr : SumCheckTranscript) : Prop :=
  sumcheckRoundConsistent inst tr ∧
  if _hZero : inst.rounds = 0 then
    mleByFolding stmt.table #[] = inst.claimedValue
  else
    sumcheckEvalPoly (tr.roundPolys[inst.rounds - 1]!) (tr.challenges[inst.rounds - 1]!) =
      mleByFolding stmt.table tr.challenges

/-- Table-indexed final oracle consistency with transcript-shape safety. -/
def sumcheckFinalOracleConsistentWithTable
  (inst : SumCheckInstance)
  (table : Array F)
  (tr : SumCheckTranscript) : Prop :=
  table.size = 2 ^ inst.rounds ∧
  sumcheckRoundConsistent inst tr ∧
  if _hZero : inst.rounds = 0 then
    mleByFolding table #[] = inst.claimedValue
  else
    sumcheckEvalPoly (tr.roundPolys[inst.rounds - 1]!) (tr.challenges[inst.rounds - 1]!) =
      mleByFolding table tr.challenges

theorem sumcheckFinalOracleConsistent_iff_withTable
  {inst : SumCheckInstance}
  {stmt : SumCheckStatement inst}
  {tr : SumCheckTranscript} :
  sumcheckFinalOracleConsistent inst stmt tr ↔
    sumcheckFinalOracleConsistentWithTable inst stmt.table tr := by
  constructor
  · intro h
    exact ⟨stmt.tableSize, h⟩
  · intro h
    exact h.2

/-- Verifier acceptance predicate for the standalone SumCheck protocol. -/
def sumcheckAccepted
  (inst : SumCheckInstance)
  (tr : SumCheckTranscript) : Prop :=
  sumcheckAcceptedCore inst tr

/--
Constructively closed acceptance surface.

This is the historical scaffolded acceptance notion: verifier acceptance plus a
same-transcript endpoint witness. It is kept explicitly named so downstream
closure packages can continue to compile while the core `sumcheckAccepted`
surface stays a plain verifier-side scaffold predicate.
-/
def sumcheckAcceptedClosed
  (inst : SumCheckInstance)
  (tr : SumCheckTranscript) : Prop :=
  sumcheckAccepted inst tr ∧
  ∃ stmt : SumCheckStatement inst, sumcheckFinalOracleConsistent inst stmt tr

/--
Verifier acceptance for a fixed table witness.

This separates verifier acceptance from existential statement packaging and is
the preferred surface for probabilistic soundness games.
-/
def sumcheckAcceptedForTable
  (inst : SumCheckInstance)
  (table : Array F)
  (tr : SumCheckTranscript) : Prop :=
  sumcheckAccepted inst tr ∧
  sumcheckFinalOracleConsistentWithTable inst table tr

theorem sumcheckAcceptedForTable_of_acceptedClosed
  {inst : SumCheckInstance}
  {tr : SumCheckTranscript}
  (hAcc : sumcheckAcceptedClosed inst tr) :
  ∃ table : Array F,
    sumcheckAcceptedForTable inst table tr ∧
    sumcheckTableSum table = inst.claimedValue := by
  rcases hAcc.2 with ⟨stmt, hFinal⟩
  refine ⟨stmt.table, ?_, stmt.hypercubeSumEqClaimed⟩
  refine ⟨hAcc.1, ?_⟩
  exact (sumcheckFinalOracleConsistent_iff_withTable).mp hFinal

theorem sumcheckAccepted_of_acceptedForTable
  {inst : SumCheckInstance}
  {table : Array F}
  {tr : SumCheckTranscript}
  (hAccTable : sumcheckAcceptedForTable inst table tr) :
  sumcheckAccepted inst tr := by
  exact hAccTable.1

theorem sumcheckAcceptedClosed_of_acceptedForTable
  {inst : SumCheckInstance}
  {table : Array F}
  {tr : SumCheckTranscript}
  (hAccTable : sumcheckAcceptedForTable inst table tr)
  (hSum : sumcheckTableSum table = inst.claimedValue) :
  sumcheckAcceptedClosed inst tr := by
  rcases hAccTable with ⟨hCore, hFinalTable⟩
  have hParam : sumcheckParameterConsistent inst := hCore.1
  have hDegree : sumcheckDegreeCompatible inst := hCore.2.1
  have hTableSize : table.size = 2 ^ inst.rounds := hFinalTable.1
  let stmt : SumCheckStatement inst :=
    { parameterConsistent := hParam
      degreeCompatible := hDegree
      table := table
      tableSize := hTableSize
      hypercubeSumEqClaimed := hSum }
  have hFinal :
      sumcheckFinalOracleConsistent inst stmt tr := by
    exact hFinalTable.2
  refine ⟨hCore, ?_⟩
  exact ⟨stmt, hFinal⟩

/--
Paper-facing statement/transcript consistency:
accepted transcript checks plus final-round oracle consistency against a concrete
table witness.
-/
def sumcheckStatementTranscriptConsistent
  (inst : SumCheckInstance)
  (stmt : SumCheckStatement inst)
  (tr : SumCheckTranscript) : Prop :=
  sumcheckAccepted inst tr ∧
  sumcheckFinalOracleConsistent inst stmt tr

/--
Paper-faithful Definition-6 theorem object for the standalone SumCheck core.

This packages one concrete claim statement together with an honest-prover
transcript constructor that works for arbitrary verifier coins of the prescribed
round length and satisfies verifier acceptance plus final-oracle consistency.
-/
structure SumCheckDefinition6Statement (inst : SumCheckInstance) where
  statement : SumCheckStatement inst
  honestTranscript :
    ∀ challenges : Array F, challenges.size = inst.rounds → SumCheckTranscript
  transcriptChallenges :
    ∀ challenges hSize,
      (honestTranscript challenges hSize).challenges = challenges
  accepted :
    ∀ challenges hSize,
      sumcheckAccepted inst (honestTranscript challenges hSize)
  finalOracle :
    ∀ challenges hSize,
      sumcheckFinalOracleConsistent inst statement (honestTranscript challenges hSize)

/-- Canonical paper-facing Definition-6 claim-truth surface. -/
def sumcheckClaimTrue (inst : SumCheckInstance) : Prop :=
  Nonempty (SumCheckDefinition6Statement inst)

/-- Compatibility alias for the standalone paper-facing claim-truth surface. -/
def sumcheckPaperClaimTrue (inst : SumCheckInstance) : Prop :=
  sumcheckClaimTrue inst

/--
Constructively closed claim-truth surface:
there exists a paper statement and a transcript that are jointly consistent.
-/
def sumcheckClaimTrueClosed (inst : SumCheckInstance) : Prop :=
  ∃ stmt tr, sumcheckStatementTranscriptConsistent inst stmt tr

/-- Soundness boundary: acceptance implies claim truth. -/
def SumcheckSoundnessAssumption : Prop :=
  ∀ inst tr,
    sumcheckAccepted inst tr →
    sumcheckClaimTrue inst

/-- Completeness boundary: true claims have an accepting transcript. -/
def SumcheckCompletenessAssumption : Prop :=
  ∀ inst,
    sumcheckClaimTrue inst →
    ∃ tr, sumcheckAccepted inst tr

/-- Structured SumCheck assumption bundle used by protocol composition. -/
structure SumCheckAssumptions where
  soundness : SumcheckSoundnessAssumption
  completeness : SumcheckCompletenessAssumption

/-!
Paper-facing probabilistic soundness bound surface.

We expose the standard SumCheck bound shape `ℓ·d / |K|` as a pair
`(numerator, denominator)` on the instance:
- numerator: `rounds * maxDegree`
- denominator: `domainSize`
-/

/-- SumCheck soundness-bound numerator `ℓ·d` (`ℓ = rounds`, `d = maxDegree`). -/
def sumcheckLundSoundnessNumerator (inst : SumCheckInstance) : Nat :=
  inst.rounds * inst.maxDegree

/-- SumCheck soundness-bound denominator `|K|` (modeled as `domainSize`). -/
def sumcheckLundSoundnessDenominator (inst : SumCheckInstance) : Nat :=
  inst.domainSize

/-- SumCheck soundness-bound surface `(ℓ·d, |K|)` corresponding to `ℓ·d / |K|`. -/
def sumcheckLundSoundnessBound (inst : SumCheckInstance) : Nat × Nat :=
  (sumcheckLundSoundnessNumerator inst, sumcheckLundSoundnessDenominator inst)

theorem sumcheckAccepted_rounds_eq
  {inst : SumCheckInstance}
  {tr : SumCheckTranscript}
  (hAcc : sumcheckAccepted inst tr) :
  tr.roundPolys.size = inst.rounds :=
  hAcc.2.2.1.2

theorem sumcheckAccepted_challenges_eq
  {inst : SumCheckInstance}
  {tr : SumCheckTranscript}
  (hAcc : sumcheckAccepted inst tr) :
  tr.challenges.size = tr.roundPolys.size := by
  rcases hAcc with
    ⟨_hParam, _hEdge, hRound, _hShape, _hInit, hFold⟩
  calc
    tr.challenges.size = inst.rounds := hRound.1
    _ = tr.roundPolys.size := hRound.2.symm

theorem sumcheckAccepted_fold_step
  {inst : SumCheckInstance}
  {tr : SumCheckTranscript}
  (hAcc : sumcheckAccepted inst tr)
  {i : Nat}
  (hi : i + 1 < tr.roundPolys.size) :
  sumcheckEvalPoly (tr.roundPolys[i + 1]!) 0 +
      sumcheckEvalPoly (tr.roundPolys[i + 1]!) 1 =
    sumcheckEvalPoly (tr.roundPolys[i]!) (tr.challenges[i]!) := by
  rcases hAcc with
    ⟨_hParam, _hEdge, _hRound, _hShape, _hInit, hFold⟩
  exact hFold.2 i hi

theorem sumcheckAccepted_initial_round
  {inst : SumCheckInstance}
  {tr : SumCheckTranscript}
  (hAcc : sumcheckAccepted inst tr) :
  sumcheckInitialRoundConsistent inst tr := by
  rcases hAcc with
    ⟨_hParam, _hEdge, _hRound, _hShape, hInit, _hFold⟩
  exact hInit

theorem sumcheckAccepted_round_sum_step
  {inst : SumCheckInstance}
  {tr : SumCheckTranscript}
  (hAcc : sumcheckAccepted inst tr)
  {i : Nat}
  (hi : i + 1 < tr.roundPolys.size) :
  sumcheckEvalPoly (tr.roundPolys[i + 1]!) 0 +
      sumcheckEvalPoly (tr.roundPolys[i + 1]!) 1 =
    sumcheckEvalPoly (tr.roundPolys[i]!) (tr.challenges[i]!) := by
  rcases hAcc with
    ⟨_hParam, _hEdge, _hRound, _hShape, _hInit, hFold⟩
  exact hFold.2 i hi

theorem sumcheckAccepted_parameter_consistent
  {inst : SumCheckInstance}
  {tr : SumCheckTranscript}
  (hAcc : sumcheckAccepted inst tr) :
  sumcheckParameterConsistent inst := by
  exact hAcc.1

theorem sumcheckAccepted_degree_compatible
  {inst : SumCheckInstance}
  {tr : SumCheckTranscript}
  (hAcc : sumcheckAccepted inst tr) :
  sumcheckDegreeCompatible inst := by
  exact hAcc.2.1

theorem sumcheckDegreeCompatible.maxDegree_pos_of_rounds_pos
  {inst : SumCheckInstance}
  (hCompat : sumcheckDegreeCompatible inst)
  (hRoundsPos : 0 < inst.rounds) :
  0 < inst.maxDegree := by
  rcases hCompat with hZero | hPos
  · omega
  · exact hPos

private def sumcheckResidualTableZero
  (table : Array F) : Nat → Array F
  | 0 => table
  | i + 1 => foldLayer (sumcheckResidualTableZero table i) 0

private theorem sumcheckResidualTableZero_size
  {rounds : Nat}
  (table : Array F)
  (hSize : table.size = 2 ^ rounds) :
  ∀ i, i ≤ rounds → (sumcheckResidualTableZero table i).size = 2 ^ (rounds - i) := by
  intro i hi
  induction i with
  | zero =>
      simpa [sumcheckResidualTableZero] using hSize
  | succ i ih =>
      have hiLt : i < rounds := by omega
      have hPrev := ih (Nat.le_of_lt hiLt)
      calc
        (sumcheckResidualTableZero table (i + 1)).size
            = (sumcheckResidualTableZero table i).size / 2 := by
                simp [sumcheckResidualTableZero]
        _ = (2 ^ (rounds - i)) / 2 := by rw [hPrev]
        _ = 2 ^ (rounds - (i + 1)) := by
              have hExp : rounds - i = (rounds - (i + 1)) + 1 := by omega
              rw [hExp, Nat.pow_succ]
              simp

private def sumcheckArraySum (vals : Array F) : F :=
  Finset.sum (Finset.range vals.size) (fun i => vals[i]!)

@[simp] private theorem sumcheckArraySum_empty :
  sumcheckArraySum (#[] : Array F) = 0 := by
  simp [sumcheckArraySum]

@[simp] private theorem sumcheckArraySum_ofFn
  (n : Nat)
  (f : Fin n → F) :
  sumcheckArraySum (Array.ofFn f) =
    Finset.sum (Finset.range n) (fun i => if h : i < n then f ⟨i, h⟩ else 0) := by
  rw [sumcheckArraySum]
  simp only [Array.size_ofFn]
  apply Finset.sum_congr rfl
  intro i hi
  have hiNat : i < n := Finset.mem_range.mp hi
  have hi' : i < (Array.ofFn f).size := by simpa using hi
  rw [getElem!_pos (c := Array.ofFn f) (i := i) hi']
  simpa [hiNat] using (Array.getElem_ofFn (f := f) (i := i) hi')

private theorem sumcheckSum_range_pairs
  (n : Nat)
  (f : Nat → F) :
  Finset.sum (Finset.range n) (fun j => f (2 * j) + f (2 * j + 1)) =
    Finset.sum (Finset.range (2 * n)) f := by
  induction n with
  | zero =>
      simp
  | succ n ih =>
      calc
        Finset.sum (Finset.range (n + 1)) (fun j => f (2 * j) + f (2 * j + 1))
            = Finset.sum (Finset.range n) (fun j => f (2 * j) + f (2 * j + 1)) +
                (f (2 * n) + f (2 * n + 1)) := by
                  rw [Finset.sum_range_succ]
        _ = Finset.sum (Finset.range (2 * n)) f + (f (2 * n) + f (2 * n + 1)) := by
              rw [ih]
        _ = Finset.sum (Finset.range (2 * n)) f + f (2 * n) + f (2 * n + 1) := by
              abel
        _ = Finset.sum (Finset.range (2 * n + 1)) f + f (2 * n + 1) := by
              rw [Finset.sum_range_succ]
        _ = Finset.sum (Finset.range (2 * (n + 1))) f := by
              have hEq : 2 * (n + 1) = 2 * n + 1 + 1 := by omega
              simpa [hEq] using (Finset.sum_range_succ (f := f) (n := 2 * n + 1)).symm

private theorem sumcheckArraySum_even_add_odd
  (vals : Array F)
  {n : Nat}
  (hSize : vals.size = 2 * n) :
  sumcheckArraySum vals =
    sumcheckArraySum (Array.ofFn (fun j : Fin n => vals[2 * j.1]!)) +
      sumcheckArraySum (Array.ofFn (fun j : Fin n => vals[2 * j.1 + 1]!)) := by
  have hEven :
      sumcheckArraySum (Array.ofFn (fun j : Fin n => vals[2 * j.1]!)) =
        Finset.sum (Finset.range n) (fun j => vals[2 * j]!) := by
    rw [sumcheckArraySum_ofFn]
    apply Finset.sum_congr rfl
    intro x hx
    have hx' : x < n := Finset.mem_range.mp hx
    simp [hx']
  have hOdd :
      sumcheckArraySum (Array.ofFn (fun j : Fin n => vals[2 * j.1 + 1]!)) =
        Finset.sum (Finset.range n) (fun j => vals[2 * j + 1]!) := by
    rw [sumcheckArraySum_ofFn]
    apply Finset.sum_congr rfl
    intro x hx
    have hx' : x < n := Finset.mem_range.mp hx
    simp [hx']
  calc
    sumcheckArraySum vals
        = Finset.sum (Finset.range (2 * n)) (fun k => vals[k]!) := by
              simp [sumcheckArraySum, hSize]
    _ = Finset.sum (Finset.range n) (fun j => vals[2 * j]! + vals[2 * j + 1]!) := by
          symm
          exact sumcheckSum_range_pairs n (fun k => vals[k]!)
    _ = Finset.sum (Finset.range n) (fun j => vals[2 * j]!) +
          Finset.sum (Finset.range n) (fun j => vals[2 * j + 1]!) := by
          rw [Finset.sum_add_distrib]
    _ = sumcheckArraySum (Array.ofFn (fun j : Fin n => vals[2 * j.1]!)) +
          sumcheckArraySum (Array.ofFn (fun j : Fin n => vals[2 * j.1 + 1]!)) := by
          rw [hEven, hOdd]

private theorem sumcheckArraySum_foldLayer
  (vals : Array F)
  (ri : F) :
  sumcheckArraySum (foldLayer vals ri) =
    sumcheckArraySum (Array.ofFn (fun j : Fin (vals.size / 2) => vals[2 * j.1]!)) * ((1 : F) - ri) +
      sumcheckArraySum (Array.ofFn (fun j : Fin (vals.size / 2) => vals[2 * j.1 + 1]!)) * ri := by
  have hEven :
      sumcheckArraySum (Array.ofFn (fun j : Fin (vals.size / 2) => vals[2 * j.1]!)) =
        Finset.sum (Finset.range (vals.size / 2)) (fun j => vals[2 * j]!) := by
    rw [sumcheckArraySum_ofFn]
    apply Finset.sum_congr rfl
    intro x hx
    have hx' : x < vals.size / 2 := Finset.mem_range.mp hx
    simp [hx']
  have hOdd :
      sumcheckArraySum (Array.ofFn (fun j : Fin (vals.size / 2) => vals[2 * j.1 + 1]!)) =
        Finset.sum (Finset.range (vals.size / 2)) (fun j => vals[2 * j + 1]!) := by
    rw [sumcheckArraySum_ofFn]
    apply Finset.sum_congr rfl
    intro x hx
    have hx' : x < vals.size / 2 := Finset.mem_range.mp hx
    simp [hx']
  calc
    sumcheckArraySum (foldLayer vals ri)
        = Finset.sum (Finset.range (vals.size / 2))
            (fun j => vals[2 * j]! * ((1 : F) - ri) + vals[2 * j + 1]! * ri) := by
              rw [foldLayer, sumcheckArraySum_ofFn]
              apply Finset.sum_congr rfl
              intro x hx
              have hx' : x < vals.size / 2 := Finset.mem_range.mp hx
              simp [hx']
    _ = Finset.sum (Finset.range (vals.size / 2)) (fun j => vals[2 * j]! * ((1 : F) - ri)) +
          Finset.sum (Finset.range (vals.size / 2)) (fun j => vals[2 * j + 1]! * ri) := by
            rw [Finset.sum_add_distrib]
    _ = Finset.sum (Finset.range (vals.size / 2)) (fun j => vals[2 * j]!) * ((1 : F) - ri) +
          Finset.sum (Finset.range (vals.size / 2)) (fun j => vals[2 * j + 1]!) * ri := by
            rw [← Finset.sum_mul, ← Finset.sum_mul]
    _ = sumcheckArraySum (Array.ofFn (fun j : Fin (vals.size / 2) => vals[2 * j.1]!)) * ((1 : F) - ri) +
          sumcheckArraySum (Array.ofFn (fun j : Fin (vals.size / 2) => vals[2 * j.1 + 1]!)) * ri := by
            rw [hEven, hOdd]

private theorem sumcheckTableSum_eq_arraySum
  (vals : Array F) :
  sumcheckTableSum vals = sumcheckArraySum vals := by
  unfold sumcheckTableSum sumcheckArraySum
  rw [← Array.foldr_toList]
  rw [← List.sum_eq_foldr]
  rw [show vals.toList = List.ofFn (fun i : Fin vals.size => vals[i.1]!) by
    simpa using (List.ofFn_getElem vals.toList).symm]
  rw [List.sum_ofFn, Fin.sum_univ_eq_sum_range]

private def sumcheckStatementHonestClaim
  (table : Array F)
  (i : Nat) : F :=
  sumcheckTableSum (sumcheckResidualTableZero table i)

private def sumcheckStatementHonestRoundPoly
  (inst : SumCheckInstance)
  (table : Array F)
  (i : Nat) : Array F :=
  let v0 := sumcheckStatementHonestClaim table (i + 1)
  let v1 := sumcheckTableSum (foldLayer (sumcheckResidualTableZero table i) 1)
  if h : inst.maxDegree = 0 then
    #[v0]
  else
    #[v0, v1 - v0] ++ Array.replicate (inst.maxDegree - 1) (0 : F)

def sumcheckHonestTranscript
  {inst : SumCheckInstance}
  (stmt : SumCheckStatement inst) : SumCheckTranscript :=
  { challenges := Array.replicate inst.rounds 0
    roundPolys := Array.ofFn (fun i : Fin inst.rounds =>
      sumcheckStatementHonestRoundPoly inst stmt.table i.1) }

/-!
Honest-transcript construction for the current finite-array SumCheck scaffold.

Attribution:
- The transcript-construction shape follows the standard SumCheck honest-prover
  pattern used in ArkLib's Sumcheck development, adapted to this simplified
  theorem-facing model.
-/

/-- Legacy synthetic honest round polynomial kept only during migration. -/
def sumcheckLegacyHonestRoundPoly (inst : SumCheckInstance) : Array F :=
  if h : inst.maxDegree = 0 then
    #[inst.claimedValue]
  else
    #[inst.claimedValue, -inst.claimedValue] ++
      Array.replicate (inst.maxDegree - 1) (0 : F)

/-- Legacy synthetic honest transcript kept only during migration. -/
def sumcheckLegacyHonestTranscript (inst : SumCheckInstance) : SumCheckTranscript :=
  { challenges := Array.replicate inst.rounds 0
    roundPolys := Array.replicate inst.rounds (sumcheckLegacyHonestRoundPoly inst) }

private theorem sumcheckFoldr_zeros_one (n : Nat) :
    Array.foldr (fun c acc => c + (1 : F) * acc) 0
      (Array.replicate n (0 : F)) n = 0 := by
  induction n with
  | zero =>
      simp [Array.foldr]
  | succ n ih =>
      have hStart :
          Array.foldr (fun c acc => c + (1 : F) * acc) 0
              (Array.replicate (n + 1) (0 : F)) (n + 1) =
            Array.foldr (fun c acc => c + (1 : F) * acc) 0
              (Array.replicate (n + 1) (0 : F)) := by
        simp [Array.size_replicate]
      rw [hStart]
      rw [Array.replicate_succ, Array.foldr_push]
      have hBase : (0 : F) + (1 : F) * (0 : F) = (0 : F) := by
        apply Fin.ext
        simp
      rw [hBase]
      have hStart' :
          Array.foldr (fun c acc => c + (1 : F) * acc) 0
              (Array.replicate n (0 : F)) =
            Array.foldr (fun c acc => c + (1 : F) * acc) 0
              (Array.replicate n (0 : F)) n := by
        simp [Array.size_replicate]
      rw [hStart']
      exact ih

private theorem sumcheckFoldr_zeros_zero (n : Nat) :
    Array.foldr (fun c acc => c + (0 : F) * acc) 0
      (Array.replicate n (0 : F)) n = 0 := by
  induction n with
  | zero =>
      simp [Array.foldr]
  | succ n ih =>
      have hStart :
          Array.foldr (fun c acc => c + (0 : F) * acc) 0
              (Array.replicate (n + 1) (0 : F)) (n + 1) =
            Array.foldr (fun c acc => c + (0 : F) * acc) 0
              (Array.replicate (n + 1) (0 : F)) := by
        simp [Array.size_replicate]
      rw [hStart]
      rw [Array.replicate_succ, Array.foldr_push]
      have hBase : (0 : F) + (0 : F) * (0 : F) = (0 : F) := by
        apply Fin.ext
        simp
      rw [hBase]
      have hStart' :
          Array.foldr (fun c acc => c + (0 : F) * acc) 0
              (Array.replicate n (0 : F)) =
            Array.foldr (fun c acc => c + (0 : F) * acc) 0
              (Array.replicate n (0 : F)) n := by
        simp [Array.size_replicate]
      rw [hStart']
      exact ih

private theorem sumcheckFoldr_zeros_add (n : Nat) :
    Array.foldr (fun c acc => c + acc) 0
      (Array.replicate n (0 : F)) n = 0 := by
  induction n with
  | zero =>
      simp [Array.foldr]
  | succ n ih =>
      have hStart :
          Array.foldr (fun c acc => c + acc) 0
              (Array.replicate (n + 1) (0 : F)) (n + 1) =
            Array.foldr (fun c acc => c + acc) 0
              (Array.replicate (n + 1) (0 : F)) := by
        simp [Array.size_replicate]
      rw [hStart]
      rw [Array.replicate_succ, Array.foldr_push]
      simp
      exact ih

theorem sumcheckLegacyHonestRoundPoly_size
  (inst : SumCheckInstance) :
  (sumcheckLegacyHonestRoundPoly inst).size = inst.maxDegree + 1 := by
  unfold sumcheckLegacyHonestRoundPoly
  by_cases h : inst.maxDegree = 0
  · simp [h]
  · simp [h]
    omega

theorem sumcheckLegacyHonestRoundPoly_getD0
  (inst : SumCheckInstance) :
  (sumcheckLegacyHonestRoundPoly inst).getD 0 0 = inst.claimedValue := by
  unfold sumcheckLegacyHonestRoundPoly
  by_cases h : inst.maxDegree = 0
  · simp [h]
  · simp [h, Array.getD]

theorem sumcheckLegacyHonestRoundPoly_eval_zero
  (inst : SumCheckInstance) :
  sumcheckEvalPoly (sumcheckLegacyHonestRoundPoly inst) 0 = inst.claimedValue := by
  unfold sumcheckLegacyHonestRoundPoly
  by_cases h : inst.maxDegree = 0
  · simp [h, sumcheckEvalPoly]
  · apply Fin.ext
    simp [h, sumcheckEvalPoly, sumcheckFoldr_zeros_zero,
      Nat.mod_eq_of_lt inst.claimedValue.isLt]

theorem sumcheckLegacyHonestRoundPoly_eval_one_zeroDegree
  (inst : SumCheckInstance)
  (hDeg : inst.maxDegree = 0) :
  sumcheckEvalPoly (sumcheckLegacyHonestRoundPoly inst) 1 = inst.claimedValue := by
  unfold sumcheckLegacyHonestRoundPoly
  simp [hDeg, sumcheckEvalPoly]

theorem sumcheckLegacyHonestRoundPoly_eval_one_nonzeroDegree
  (inst : SumCheckInstance)
  (hDeg : inst.maxDegree ≠ 0) :
  sumcheckEvalPoly (sumcheckLegacyHonestRoundPoly inst) 1 = 0 := by
  unfold sumcheckLegacyHonestRoundPoly
  apply Fin.ext
  simp [hDeg, sumcheckEvalPoly, sumcheckFoldr_zeros_add]

theorem sumcheckLegacyHonestTranscript_roundPoly_get!
  (inst : SumCheckInstance)
  {i : Nat}
  (hi : i < inst.rounds) :
  (sumcheckLegacyHonestTranscript inst).roundPolys[i]! = sumcheckLegacyHonestRoundPoly inst := by
  simp [sumcheckLegacyHonestTranscript, hi]

theorem sumcheckLegacyHonestTranscript_challenge_get!
  (inst : SumCheckInstance)
  {i : Nat}
  (hi : i < inst.rounds) :
  (sumcheckLegacyHonestTranscript inst).challenges[i]! = 0 := by
  simp [sumcheckLegacyHonestTranscript, hi]

private theorem sumcheckStatementHonestRoundPoly_size
  (inst : SumCheckInstance)
  (table : Array F)
  (i : Nat) :
  (sumcheckStatementHonestRoundPoly inst table i).size = inst.maxDegree + 1 := by
  unfold sumcheckStatementHonestRoundPoly
  by_cases h : inst.maxDegree = 0
  · simp [h]
  · simp [h]
    omega

private theorem sumcheckStatementHonestRoundPoly_eval_zero
  (inst : SumCheckInstance)
  (table : Array F)
  (i : Nat) :
  sumcheckEvalPoly (sumcheckStatementHonestRoundPoly inst table i) 0 =
    sumcheckStatementHonestClaim table (i + 1) := by
  unfold sumcheckStatementHonestRoundPoly
  by_cases h : inst.maxDegree = 0
  · simp [h, sumcheckEvalPoly]
  · apply Fin.ext
    simp [h, sumcheckEvalPoly, sumcheckFoldr_zeros_zero]

private theorem sumcheckStatementHonestRoundPoly_eval_one
  (inst : SumCheckInstance)
  (table : Array F)
  (i : Nat)
  (hDegPos : 0 < inst.maxDegree) :
  sumcheckEvalPoly (sumcheckStatementHonestRoundPoly inst table i) 1 =
    sumcheckTableSum (foldLayer (sumcheckResidualTableZero table i) 1) := by
  have hDeg : inst.maxDegree ≠ 0 := Nat.ne_of_gt hDegPos
  unfold sumcheckStatementHonestRoundPoly
  have hMain :
      sumcheckStatementHonestClaim table (i + 1) +
          (sumcheckTableSum (foldLayer (sumcheckResidualTableZero table i) 1) -
            sumcheckStatementHonestClaim table (i + 1)) =
        sumcheckTableSum (foldLayer (sumcheckResidualTableZero table i) 1) := by
    abel_nf
  simpa [hDeg, sumcheckEvalPoly, sumcheckFoldr_zeros_add] using hMain

theorem sumcheckHonestTranscript_roundPoly_get!
  {inst : SumCheckInstance}
  (stmt : SumCheckStatement inst)
  {i : Nat}
  (hi : i < inst.rounds) :
  (sumcheckHonestTranscript stmt).roundPolys[i]! =
    sumcheckStatementHonestRoundPoly inst stmt.table i := by
  simp [sumcheckHonestTranscript, hi]

theorem sumcheckHonestTranscript_challenge_get!
  {inst : SumCheckInstance}
  (stmt : SumCheckStatement inst)
  {i : Nat}
  (hi : i < inst.rounds) :
  (sumcheckHonestTranscript stmt).challenges[i]! = 0 := by
  simp [sumcheckHonestTranscript, hi]

private theorem sumcheckTableSum_foldLayer_split
  (vals : Array F)
  (hEven : vals.size = 2 * (vals.size / 2)) :
  sumcheckTableSum (foldLayer vals 0) + sumcheckTableSum (foldLayer vals 1) =
    sumcheckTableSum vals := by
  let evenVals := Array.ofFn (fun j : Fin (vals.size / 2) => vals[2 * j.1]!)
  let oddVals := Array.ofFn (fun j : Fin (vals.size / 2) => vals[2 * j.1 + 1]!)
  have h0 :
      sumcheckTableSum (foldLayer vals 0) = sumcheckArraySum evenVals := by
    rw [sumcheckTableSum_eq_arraySum, sumcheckArraySum_foldLayer]
    simp [evenVals, oddVals]
  have h1 :
      sumcheckTableSum (foldLayer vals 1) = sumcheckArraySum oddVals := by
    rw [sumcheckTableSum_eq_arraySum, sumcheckArraySum_foldLayer]
    simp [evenVals, oddVals]
  have hSum :
      sumcheckTableSum vals = sumcheckArraySum evenVals + sumcheckArraySum oddVals := by
    rw [sumcheckTableSum_eq_arraySum]
    simpa [evenVals, oddVals] using sumcheckArraySum_even_add_odd vals hEven
  calc
    sumcheckTableSum (foldLayer vals 0) + sumcheckTableSum (foldLayer vals 1)
        = sumcheckArraySum evenVals + sumcheckArraySum oddVals := by rw [h0, h1]
    _ = sumcheckTableSum vals := hSum.symm

private theorem sumcheckResidualTableZero_succ
  (table : Array F)
  (n : Nat) :
  sumcheckResidualTableZero table (n + 1) =
    sumcheckResidualTableZero (foldLayer table 0) n := by
  induction n generalizing table with
  | zero =>
      simp [sumcheckResidualTableZero]
  | succ n ih =>
      calc
        sumcheckResidualTableZero table (n + 1 + 1)
            = foldLayer (sumcheckResidualTableZero table (n + 1)) 0 := by
                simp [sumcheckResidualTableZero]
        _ = foldLayer (sumcheckResidualTableZero (foldLayer table 0) n) 0 := by
              rw [ih]
        _ = sumcheckResidualTableZero (foldLayer table 0) (n + 1) := by
              simp [sumcheckResidualTableZero]

private theorem sumcheckMleByFolding_zeroChallenges_eq_residual
  (table : Array F) :
  ∀ n,
    mleByFolding table (Array.replicate n (0 : F)) =
      mleByFolding (sumcheckResidualTableZero table n) #[] := by
  intro n
  induction n generalizing table with
  | zero =>
      simp [sumcheckResidualTableZero, mleByFolding]
  | succ n ih =>
      have hNe : (Array.replicate (n + 1) (0 : F)).size ≠ 0 := by simp
      calc
        mleByFolding table (Array.replicate (n + 1) (0 : F))
            = mleByFolding (foldLayer table 0)
                ((Array.replicate (n + 1) (0 : F)).extract 1 (n + 1)) := by
                  simpa using mleByFolding_step table (Array.replicate (n + 1) (0 : F)) hNe
        _ = mleByFolding (foldLayer table 0) (Array.replicate n (0 : F)) := by
              simp
        _ = mleByFolding (sumcheckResidualTableZero table (n + 1)) #[] := by
              rw [sumcheckResidualTableZero_succ]
              exact ih (foldLayer table 0)

theorem sumcheckTableSum_size_one
  (vals : Array F)
  (hSize : vals.size = 1) :
  sumcheckTableSum vals = vals[0]! := by
  rw [sumcheckTableSum_eq_arraySum]
  simp [sumcheckArraySum, hSize]

private theorem sumcheckStatementHonestClaim_full_zeroChallenges
  {inst : SumCheckInstance}
  (stmt : SumCheckStatement inst) :
  sumcheckStatementHonestClaim stmt.table inst.rounds =
    mleByFolding stmt.table (Array.replicate inst.rounds (0 : F)) := by
  have hResidualSize :
      (sumcheckResidualTableZero stmt.table inst.rounds).size = 1 := by
    simpa using
      (sumcheckResidualTableZero_size stmt.table stmt.tableSize inst.rounds le_rfl)
  have hFold :
      mleByFolding stmt.table (Array.replicate inst.rounds (0 : F)) =
        mleByFolding (sumcheckResidualTableZero stmt.table inst.rounds) #[] :=
    sumcheckMleByFolding_zeroChallenges_eq_residual stmt.table inst.rounds
  have hNe : (sumcheckResidualTableZero stmt.table inst.rounds).size ≠ 0 := by
    simpa [hResidualSize]
  calc
    sumcheckStatementHonestClaim stmt.table inst.rounds
        = sumcheckTableSum (sumcheckResidualTableZero stmt.table inst.rounds) := rfl
    _ = (sumcheckResidualTableZero stmt.table inst.rounds)[0]! := by
          exact sumcheckTableSum_size_one _ hResidualSize
    _ = mleByFolding (sumcheckResidualTableZero stmt.table inst.rounds) #[] := by
          symm
          exact mleByFolding_empty (sumcheckResidualTableZero stmt.table inst.rounds) hNe
    _ = mleByFolding stmt.table (Array.replicate inst.rounds (0 : F)) := hFold.symm

/-!
Paper-statement witness table used to derive `sumcheckPaperClaimTrue` from
accepted transcripts and from legacy base-claim assumptions.
-/

def sumcheckWitnessTable (inst : SumCheckInstance) : Array F :=
  (inst.claimedValue :: List.replicate (2 ^ inst.rounds - 1) (0 : F)).toArray

private theorem sumcheckListFoldr_zeros_add (n : Nat) :
    List.foldr (fun v acc => v + acc) 0 (List.replicate n (0 : F)) = 0 := by
  induction n with
  | zero =>
      simp
  | succ n ih =>
      simp [List.replicate_succ, ih]

theorem sumcheckWitnessTable_size
    (inst : SumCheckInstance) :
    (sumcheckWitnessTable inst).size = 2 ^ inst.rounds := by
  have hPowPos : 0 < 2 ^ inst.rounds := by
    exact Nat.pow_pos (a := 2) (n := inst.rounds) (by decide : 0 < (2 : Nat))
  have hPowGe1 : 1 ≤ 2 ^ inst.rounds := Nat.succ_le_of_lt hPowPos
  calc
    (sumcheckWitnessTable inst).size
        = (2 ^ inst.rounds - 1) + 1 := by
            simp [sumcheckWitnessTable]
    _ = 2 ^ inst.rounds := by
        exact Nat.sub_add_cancel hPowGe1

theorem sumcheckWitnessTable_sum
    (inst : SumCheckInstance) :
    sumcheckTableSum (sumcheckWitnessTable inst) = inst.claimedValue := by
  unfold sumcheckTableSum sumcheckWitnessTable
  rw [← Array.foldr_toList (f := fun v acc : F => v + acc)
      (xs := (inst.claimedValue :: List.replicate (2 ^ inst.rounds - 1) (0 : F)).toArray)]
  have hAdd : inst.claimedValue + 0 = inst.claimedValue := by
    apply Fin.ext
    simp [Nat.mod_eq_of_lt inst.claimedValue.isLt]
  simpa [sumcheckListFoldr_zeros_add] using hAdd

private theorem sumcheckFoldl_range_head_nonzero
    (n : Nat) (t : Nat → F) (c : F)
    (h0 : t 0 = c)
    (hSucc : ∀ i, i < n → t (i + 1) = 0) :
    (List.range (n + 1)).foldl (fun acc i => acc + t i) 0 = c := by
  induction n generalizing t c with
  | zero =>
      have hAdd : (0 : F) + c = c := by
        apply Fin.ext
        simp [Nat.mod_eq_of_lt c.isLt]
      simpa [h0] using hAdd
  | succ n ih =>
      have hRange : List.range (n + 2) = List.range (n + 1) ++ [n + 1] := by
        simpa [Nat.succ_eq_add_one, Nat.add_assoc] using (List.range_succ (n := n + 1))
      rw [hRange, List.foldl_append, List.foldl_cons]
      have hPrefix :
          (List.range (n + 1)).foldl (fun acc i => acc + t i) 0 = c := by
        apply ih (t := t) (c := c) h0
        intro i hi
        exact hSucc i (Nat.lt_trans hi (Nat.lt_succ_self n))
      rw [hPrefix]
      have hLast : t (n + 1) = 0 := hSucc n (Nat.lt_succ_self n)
      rw [hLast]
      have hAdd : c + 0 = c := by
        apply Fin.ext
        simp [Nat.mod_eq_of_lt c.isLt]
      exact hAdd

private theorem bitsToFieldArray_zero_eq_replicate (n : Nat) :
    bitsToFieldArray n 0 = Array.replicate n (0 : F) := by
  apply Array.ext
  · simp [bitsToFieldArray]
  · intro i hi1 hi2
    simp [bitsToFieldArray]

private theorem eqPoly_bits_zero_replicate_zero (n : Nat) :
    eqPoly (bitsToFieldArray n 0) (Array.replicate n (0 : F)) = (1 : F) := by
  have hSize : (bitsToFieldArray n 0).size = (Array.replicate n (0 : F)).size := by
    simp [bitsToFieldArray]
  have hEq : bitsToFieldArray n 0 = Array.replicate n (0 : F) :=
    bitsToFieldArray_zero_eq_replicate n
  have hx : IsBitVec (bitsToFieldArray n 0) := by
    intro i
    have hz : (Array.replicate n (0 : F))[i] = 0 := by simp
    exact Or.inl (by simpa [hEq] using hz)
  have hy : IsBitVec (Array.replicate n (0 : F)) := by
    intro i
    exact Or.inl (by simp)
  have hDelta := eqPoly_eq_delta_of_isBitVec hSize hx hy
  simpa [hEq] using hDelta

private theorem f_zero_mul_any (x : F) : (0 : F) * x = 0 := by
  apply Fin.ext
  change ((0 * x.val) % Goldilocks.q) = 0
  simp

private theorem f_mul_one_any (x : F) : x * (1 : F) = x := by
  apply Fin.ext
  change ((x.val * 1) % Goldilocks.q) = x.val
  simp [Nat.mod_eq_of_lt x.isLt]

private theorem sumcheckWitnessTable_mleByInnerProduct_zeroChallenges
    (inst : SumCheckInstance) :
    mleByInnerProduct (sumcheckWitnessTable inst)
      (Array.replicate inst.rounds (0 : F)) = inst.claimedValue := by
  unfold mleByInnerProduct mleInnerProductForm
  have hPowPos : 0 < 2 ^ inst.rounds := by
    exact Nat.pow_pos (a := 2) (n := inst.rounds) (by decide : 0 < (2 : Nat))
  have hPowGe1 : 1 ≤ 2 ^ inst.rounds := Nat.succ_le_of_lt hPowPos
  have hSize' : (sumcheckWitnessTable inst).size = (2 ^ inst.rounds - 1) + 1 := by
    calc
      (sumcheckWitnessTable inst).size = 2 ^ inst.rounds := sumcheckWitnessTable_size inst
      _ = (2 ^ inst.rounds - 1) + 1 := (Nat.sub_add_cancel hPowGe1).symm
  let t : Nat → F := fun i =>
    (sumcheckWitnessTable inst)[i]! *
      eqPoly (bitsToFieldArray inst.rounds i) (Array.replicate inst.rounds (0 : F))
  have hFold :
      (List.range ((2 ^ inst.rounds - 1) + 1)).foldl (fun acc i => acc + t i) 0 =
        inst.claimedValue := by
    apply sumcheckFoldl_range_head_nonzero
      (n := (2 ^ inst.rounds - 1)) (t := t) (c := inst.claimedValue)
    · unfold t
      have hGet0 : (sumcheckWitnessTable inst)[0]! = inst.claimedValue := by
        unfold sumcheckWitnessTable
        simp
      calc
        (sumcheckWitnessTable inst)[0]! *
            eqPoly (bitsToFieldArray inst.rounds 0) (Array.replicate inst.rounds (0 : F))
            = inst.claimedValue * (1 : F) := by
                rw [hGet0, eqPoly_bits_zero_replicate_zero]
        _ = inst.claimedValue := f_mul_one_any inst.claimedValue
    · intro i hi
      unfold t
      have hGet : (sumcheckWitnessTable inst)[i + 1]! = (0 : F) := by
        unfold sumcheckWitnessTable
        simp [hi]
      calc
        (sumcheckWitnessTable inst)[i + 1]! *
            eqPoly (bitsToFieldArray inst.rounds (i + 1))
              (Array.replicate inst.rounds (0 : F))
            = (0 : F) *
                eqPoly (bitsToFieldArray inst.rounds (i + 1))
                  (Array.replicate inst.rounds (0 : F)) := by rw [hGet]
        _ = 0 := f_zero_mul_any _
  simpa [t, hSize'] using hFold

private theorem sumcheckWitnessTable_mleByFolding_zeroChallenges
    (inst : SumCheckInstance) :
    mleByFolding (sumcheckWitnessTable inst)
      (Array.replicate inst.rounds (0 : F)) = inst.claimedValue := by
  have hSize :
      (sumcheckWitnessTable inst).size =
        2 ^ (Array.replicate inst.rounds (0 : F)).size := by
    simp [sumcheckWitnessTable_size]
  have hBridge :
      mleByInnerProduct (sumcheckWitnessTable inst) (Array.replicate inst.rounds (0 : F)) =
        mleByFolding (sumcheckWitnessTable inst) (Array.replicate inst.rounds (0 : F)) :=
    mleByInnerProduct_eq_mleByFolding_of_size (v := sumcheckWitnessTable inst)
      (r := Array.replicate inst.rounds (0 : F)) hSize
  exact hBridge.symm.trans (sumcheckWitnessTable_mleByInnerProduct_zeroChallenges inst)

def sumcheckWitnessStatement
    (inst : SumCheckInstance)
    (hClaim : sumcheckBaseClaimTrue inst) : SumCheckStatement inst :=
  let hParam := hClaim.1
  let hDegree := hClaim.2
  { parameterConsistent := hParam
    degreeCompatible := hDegree
    table := sumcheckWitnessTable inst
    tableSize := sumcheckWitnessTable_size inst
    hypercubeSumEqClaimed := sumcheckWitnessTable_sum inst }

private theorem sumcheckHonestTranscript_acceptedCore_of_statement
  {inst : SumCheckInstance}
  (stmt : SumCheckStatement inst) :
  sumcheckAcceptedCore inst (sumcheckHonestTranscript stmt) := by
  refine ⟨stmt.parameterConsistent, stmt.degreeCompatible, ?_, ?_, ?_, ?_⟩
  · exact ⟨by simp [sumcheckHonestTranscript], by simp [sumcheckHonestTranscript]⟩
  · intro i
    simpa [sumcheckRoundPolyShape, sumcheckHonestTranscript] using
      sumcheckStatementHonestRoundPoly_size inst stmt.table i.1
  · unfold sumcheckInitialRoundConsistent
    by_cases hRounds : inst.rounds = 0
    · have hSize :
          (sumcheckHonestTranscript stmt).roundPolys.size = 0 := by
        simp [sumcheckHonestTranscript, hRounds]
      simp [sumcheckInitialRoundConsistent, hSize]
    · have hSize :
          (sumcheckHonestTranscript stmt).roundPolys.size ≠ 0 := by
        simp [sumcheckHonestTranscript, hRounds]
      have hRoundsPos : 0 < inst.rounds := Nat.pos_of_ne_zero hRounds
      have hDegPos :=
        sumcheckDegreeCompatible.maxDegree_pos_of_rounds_pos stmt.degreeCompatible hRoundsPos
      have hPoly0 :
          (sumcheckHonestTranscript stmt).roundPolys[0]! =
            sumcheckStatementHonestRoundPoly inst stmt.table 0 :=
        sumcheckHonestTranscript_roundPoly_get! stmt hRoundsPos
      have hEven : (sumcheckResidualTableZero stmt.table 0).size =
          2 * ((sumcheckResidualTableZero stmt.table 0).size / 2) := by
          have hSize0 :
              (sumcheckResidualTableZero stmt.table 0).size = 2 ^ inst.rounds := by
            simpa [sumcheckResidualTableZero] using stmt.tableSize
          rw [hSize0]
          rcases Nat.exists_eq_succ_of_ne_zero hRounds with ⟨n, hn⟩
          rw [hn]
          simp [Nat.pow_succ, Nat.mul_comm, Nat.mul_left_comm, Nat.mul_assoc]
      have hInit :
        sumcheckEvalPoly (sumcheckHonestTranscript stmt).roundPolys[0]! 0 +
            sumcheckEvalPoly (sumcheckHonestTranscript stmt).roundPolys[0]! 1
            =
          sumcheckStatementHonestClaim stmt.table 1 +
            sumcheckTableSum (foldLayer (sumcheckResidualTableZero stmt.table 0) 1) := by
              rw [hPoly0,
                sumcheckStatementHonestRoundPoly_eval_zero,
                sumcheckStatementHonestRoundPoly_eval_one _ _ _ hDegPos]
      have hClaim :
          sumcheckEvalPoly (sumcheckHonestTranscript stmt).roundPolys[0]! 0 +
              sumcheckEvalPoly (sumcheckHonestTranscript stmt).roundPolys[0]! 1 =
            inst.claimedValue := by
        calc
          sumcheckEvalPoly (sumcheckHonestTranscript stmt).roundPolys[0]! 0 +
              sumcheckEvalPoly (sumcheckHonestTranscript stmt).roundPolys[0]! 1
              =
            sumcheckStatementHonestClaim stmt.table 1 +
              sumcheckTableSum (foldLayer (sumcheckResidualTableZero stmt.table 0) 1) := hInit
          _ = sumcheckTableSum stmt.table := by
              unfold sumcheckStatementHonestClaim
              simpa [sumcheckResidualTableZero] using
                sumcheckTableSum_foldLayer_split stmt.table hEven
          _ = inst.claimedValue := stmt.hypercubeSumEqClaimed
      simpa [sumcheckInitialRoundConsistent, hSize] using hClaim
  · refine ⟨by simp [sumcheckHonestTranscript], ?_⟩
    intro i hi
    have hiNext : i + 1 < inst.rounds := by
      simpa [sumcheckHonestTranscript] using hi
    have hiCur : i < inst.rounds := Nat.lt_trans (Nat.lt_succ_self i) hiNext
    have hRoundsPos : 0 < inst.rounds := Nat.lt_trans (Nat.zero_lt_succ _) hiNext
    have hDegPos :=
      sumcheckDegreeCompatible.maxDegree_pos_of_rounds_pos stmt.degreeCompatible hRoundsPos
    have hPolyCur :
        (sumcheckHonestTranscript stmt).roundPolys[i]! =
          sumcheckStatementHonestRoundPoly inst stmt.table i :=
      sumcheckHonestTranscript_roundPoly_get! stmt hiCur
    have hPolyNext :
        (sumcheckHonestTranscript stmt).roundPolys[i + 1]! =
          sumcheckStatementHonestRoundPoly inst stmt.table (i + 1) :=
      sumcheckHonestTranscript_roundPoly_get! stmt hiNext
    have hChal :
        (sumcheckHonestTranscript stmt).challenges[i]! = 0 :=
      sumcheckHonestTranscript_challenge_get! stmt hiCur
    have hEven :
        (sumcheckResidualTableZero stmt.table (i + 1)).size =
          2 * ((sumcheckResidualTableZero stmt.table (i + 1)).size / 2) := by
      have hSize :
          (sumcheckResidualTableZero stmt.table (i + 1)).size =
            2 ^ (inst.rounds - (i + 1)) := by
        exact sumcheckResidualTableZero_size stmt.table stmt.tableSize (i + 1) hiNext.le
      rw [hSize]
      have hDiffPos : 0 < inst.rounds - (i + 1) := by omega
      rcases Nat.exists_eq_succ_of_ne_zero (Nat.ne_of_gt hDiffPos) with ⟨n, hn⟩
      rw [hn]
      simp [Nat.pow_succ, Nat.mul_comm]
    calc
      sumcheckEvalPoly (sumcheckHonestTranscript stmt).roundPolys[i + 1]! 0 +
          sumcheckEvalPoly (sumcheckHonestTranscript stmt).roundPolys[i + 1]! 1
          =
        sumcheckStatementHonestClaim stmt.table (i + 2) +
          sumcheckTableSum (foldLayer (sumcheckResidualTableZero stmt.table (i + 1)) 1) := by
            rw [hPolyNext,
              sumcheckStatementHonestRoundPoly_eval_zero,
              sumcheckStatementHonestRoundPoly_eval_one _ _ _ hDegPos]
      _ = sumcheckStatementHonestClaim stmt.table (i + 1) := by
            unfold sumcheckStatementHonestClaim
            simpa [sumcheckResidualTableZero] using
              sumcheckTableSum_foldLayer_split
                (sumcheckResidualTableZero stmt.table (i + 1)) hEven
      _ = sumcheckEvalPoly (sumcheckHonestTranscript stmt).roundPolys[i]! (sumcheckHonestTranscript stmt).challenges[i]! := by
            rw [hPolyCur, hChal, sumcheckStatementHonestRoundPoly_eval_zero]

theorem sumcheckHonestTranscript_accepted_of_statement
  {inst : SumCheckInstance}
  (stmt : SumCheckStatement inst) :
  sumcheckAccepted inst (sumcheckHonestTranscript stmt) := by
  exact sumcheckHonestTranscript_acceptedCore_of_statement stmt

theorem sumcheckFinalOracleConsistent_of_statement_constructive
  {inst : SumCheckInstance}
  (stmt : SumCheckStatement inst) :
  sumcheckFinalOracleConsistent inst stmt (sumcheckHonestTranscript stmt) := by
  have hCore : sumcheckAcceptedCore inst (sumcheckHonestTranscript stmt) :=
    sumcheckHonestTranscript_acceptedCore_of_statement stmt
  refine ⟨hCore.2.2.1, ?_⟩
  by_cases hZero : inst.rounds = 0
  · simp [sumcheckFinalOracleConsistent, hZero]
    have hSize : stmt.table.size = 1 := by simpa [hZero] using stmt.tableSize
    have hNe : stmt.table.size ≠ 0 := by simpa [hSize]
    calc
      mleByFolding stmt.table #[] = stmt.table[0]! := by
        exact mleByFolding_empty stmt.table hNe
      _ = sumcheckTableSum stmt.table := by
            symm
            exact sumcheckTableSum_size_one stmt.table hSize
      _ = inst.claimedValue := stmt.hypercubeSumEqClaimed
  · simp [sumcheckFinalOracleConsistent, hZero]
    have hLast : inst.rounds - 1 < inst.rounds := by omega
    have hPoly :
        (sumcheckHonestTranscript stmt).roundPolys[inst.rounds - 1]! =
          sumcheckStatementHonestRoundPoly inst stmt.table (inst.rounds - 1) :=
      sumcheckHonestTranscript_roundPoly_get! stmt hLast
    have hChal :
        (sumcheckHonestTranscript stmt).challenges[inst.rounds - 1]! = 0 :=
      sumcheckHonestTranscript_challenge_get! stmt hLast
    calc
      sumcheckEvalPoly (sumcheckHonestTranscript stmt).roundPolys[inst.rounds - 1]!
          (sumcheckHonestTranscript stmt).challenges[inst.rounds - 1]!
          =
        sumcheckStatementHonestClaim stmt.table inst.rounds := by
              have hEq : inst.rounds - 1 + 1 = inst.rounds := by
                exact Nat.sub_add_cancel (Nat.succ_le_of_lt (Nat.pos_of_ne_zero hZero))
              rw [hPoly, hChal, sumcheckStatementHonestRoundPoly_eval_zero]
              simp [hEq]
      _ = mleByFolding stmt.table (Array.replicate inst.rounds (0 : F)) := by
            exact sumcheckStatementHonestClaim_full_zeroChallenges stmt
      _ = mleByFolding stmt.table (sumcheckHonestTranscript stmt).challenges := by
            simp [sumcheckHonestTranscript]

theorem sumcheckStatementTranscriptConsistent_of_statement_constructive
  {inst : SumCheckInstance}
  (stmt : SumCheckStatement inst) :
  sumcheckStatementTranscriptConsistent inst stmt (sumcheckHonestTranscript stmt) := by
  exact ⟨sumcheckHonestTranscript_accepted_of_statement stmt,
    sumcheckFinalOracleConsistent_of_statement_constructive stmt⟩

theorem sumcheckCompleteness_from_baseClaim_constructive
  (inst : SumCheckInstance)
  (hClaim : sumcheckBaseClaimTrue inst) :
  ∃ tr, sumcheckAcceptedClosed inst tr := by
  let stmt := sumcheckWitnessStatement inst hClaim
  refine ⟨sumcheckHonestTranscript stmt, ?_⟩
  refine ⟨sumcheckHonestTranscript_accepted_of_statement stmt, ?_⟩
  exact ⟨stmt, sumcheckFinalOracleConsistent_of_statement_constructive stmt⟩

theorem sumcheckHonestTranscript_accepted_of_baseClaim
  (inst : SumCheckInstance)
  (hClaim : sumcheckBaseClaimTrue inst) :
  sumcheckAccepted inst (sumcheckHonestTranscript (sumcheckWitnessStatement inst hClaim)) := by
  exact sumcheckHonestTranscript_accepted_of_statement (sumcheckWitnessStatement inst hClaim)

theorem sumcheckFinalOracleConsistent_of_baseClaim_constructive
  (inst : SumCheckInstance)
  (hClaim : sumcheckBaseClaimTrue inst) :
  sumcheckFinalOracleConsistent inst
    (sumcheckWitnessStatement inst hClaim)
    (sumcheckHonestTranscript (sumcheckWitnessStatement inst hClaim)) := by
  exact sumcheckFinalOracleConsistent_of_statement_constructive (sumcheckWitnessStatement inst hClaim)

theorem sumcheckStatementTranscriptConsistent_of_baseClaim_constructive
  (inst : SumCheckInstance)
  (hClaim : sumcheckBaseClaimTrue inst) :
  ∃ stmt tr, sumcheckStatementTranscriptConsistent inst stmt tr := by
  let stmt := sumcheckWitnessStatement inst hClaim
  exact ⟨stmt, sumcheckHonestTranscript stmt,
    sumcheckStatementTranscriptConsistent_of_statement_constructive stmt⟩

/--
Accepted transcripts induce constructive statement/transcript consistency
existence at the same transcript.
-/
theorem sumcheckStatementTranscriptConsistent_exists_of_acceptedClosed
  {inst : SumCheckInstance}
  {tr : SumCheckTranscript}
  (hAcc : sumcheckAcceptedClosed inst tr) :
  ∃ stmt, sumcheckStatementTranscriptConsistent inst stmt tr := by
  rcases hAcc.2 with ⟨stmt, hFinal⟩
  exact ⟨stmt, ⟨hAcc.1, hFinal⟩⟩

/--
Same-transcript consistency closure:
if an accepted transcript also has a statement witness for the final-oracle
endpoint, then the pair is statement/transcript consistent.
-/
theorem sumcheckStatementTranscriptConsistent_of_accepted_sameTranscript
  {inst : SumCheckInstance}
  {tr : SumCheckTranscript}
  (hAcc : sumcheckAccepted inst tr)
  {stmt : SumCheckStatement inst}
  (hFinal : sumcheckFinalOracleConsistent inst stmt tr) :
  sumcheckStatementTranscriptConsistent inst stmt tr := by
  exact ⟨hAcc, hFinal⟩

/--
Same-transcript existential closure:
accepted transcript evidence yields
`∃ stmt, sumcheckStatementTranscriptConsistent inst stmt tr` at that same
transcript.
-/
theorem sumcheckStatementTranscriptConsistent_exists_of_acceptedClosed_sameTranscript
  {inst : SumCheckInstance}
  {tr : SumCheckTranscript}
  (hAcc : sumcheckAcceptedClosed inst tr) :
  ∃ stmt, sumcheckStatementTranscriptConsistent inst stmt tr := by
  exact sumcheckStatementTranscriptConsistent_exists_of_acceptedClosed hAcc

/--
Same-transcript claim-truth closure:
accepted transcript evidence yields `sumcheckClaimTrue`.
-/
theorem sumcheckClaimTrueClosed_of_acceptedClosed
  {inst : SumCheckInstance}
  {tr : SumCheckTranscript}
  (hAcc : sumcheckAcceptedClosed inst tr) :
  sumcheckClaimTrueClosed inst := by
  rcases sumcheckStatementTranscriptConsistent_exists_of_acceptedClosed hAcc with
    ⟨stmt, hCons⟩
  exact ⟨stmt, tr, hCons⟩

/-- Lift legacy base-claim witness into the paper-facing claim-truth predicate. -/
theorem sumcheckClaimTrueClosed_of_baseClaim_constructive
  {inst : SumCheckInstance}
  (hClaim : sumcheckBaseClaimTrue inst) :
  sumcheckClaimTrueClosed inst := by
  rcases sumcheckStatementTranscriptConsistent_of_baseClaim_constructive inst hClaim with
    ⟨stmt, tr, hCons⟩
  exact ⟨stmt, tr, hCons⟩

theorem sumcheckAccepted_not_of_challenge_size_ne
  {inst : SumCheckInstance}
  {tr : SumCheckTranscript}
  (hNe : tr.challenges.size ≠ inst.rounds) :
  ¬ sumcheckAccepted inst tr := by
  intro hAcc
  exact hNe hAcc.2.2.1.1

theorem sumcheckAccepted_not_of_roundpoly_size_ne
  {inst : SumCheckInstance}
  {tr : SumCheckTranscript}
  (hNe : tr.roundPolys.size ≠ inst.rounds) :
  ¬ sumcheckAccepted inst tr := by
  intro hAcc
  exact hNe hAcc.2.2.1.2

theorem sumcheckAccepted_not_of_bad_round_shape
  {inst : SumCheckInstance}
  {tr : SumCheckTranscript}
  (hBad : ∃ i : Fin tr.roundPolys.size,
    ¬ sumcheckRoundPolyShape inst tr.roundPolys[i.1]) :
  ¬ sumcheckAccepted inst tr := by
  intro hAcc
  rcases hAcc with
    ⟨_hParam, _hEdge, _hRound, hShapes, _hInit, _hFold⟩
  rcases hBad with ⟨i, hiBad⟩
  exact hiBad (hShapes i)

/-- Acceptance rejects transcripts with no same-transcript final-oracle witness. -/
theorem sumcheckAccepted_not_of_no_final_oracle_witness
  {inst : SumCheckInstance}
  {tr : SumCheckTranscript}
  (hBad : ∀ stmt : SumCheckStatement inst, ¬ sumcheckFinalOracleConsistent inst stmt tr) :
  ¬ sumcheckAcceptedClosed inst tr := by
  intro hAcc
  rcases hAcc.2 with ⟨stmt, hFinal⟩
  exact hBad stmt hFinal

theorem sumcheckAccepted_not_of_bad_initial_round
  {inst : SumCheckInstance}
  {tr : SumCheckTranscript}
  (hBad : ¬ sumcheckInitialRoundConsistent inst tr) :
  ¬ sumcheckAccepted inst tr := by
  intro hAcc
  rcases hAcc with
    ⟨_hParam, _hEdge, _hRound, _hShapes, hInit, _hFold⟩
  exact hBad hInit

theorem sumcheckClaimTrue_of_soundness
  {inst : SumCheckInstance}
  {tr : SumCheckTranscript}
  (hSound : SumcheckSoundnessAssumption)
  (hAcc : sumcheckAccepted inst tr) :
  sumcheckClaimTrue inst := by
  exact hSound inst tr hAcc

/-- Re-export under paper-facing names. -/
abbrev SumCheckAccepted := sumcheckAccepted
abbrev SumCheckClaimTrue := sumcheckClaimTrue
abbrev SumCheckRoundConsistent := sumcheckRoundConsistent

end SuperNeo
