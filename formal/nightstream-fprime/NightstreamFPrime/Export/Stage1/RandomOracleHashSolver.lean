import NightstreamFPrime.Export.Stage1.RandomOracleSetup

/-!
Owns the state-hash collision finders: algorithms with a strict budget of
adversary runs whose success bounds `RandomOracleSetup.hashCollisions`.

Inputs: an adversary for each value of the setup chunks, its claim and prior
preimage, the verifier-owned context digest, and a budget `T`.

Outputs:
- `HashCollision`: two prior preimages with different encodings and one
  state hash (`PiCCSSecurity.StateHashCollision`);
- `lineFinder`: the finder of one `Π_RLC` coordinate: the base run, then
  retries that resample the base run's answer at that coordinate's point until
  one hits the point again (Lemma 5). It compares the two prior preimages;
- `hashSolve`: one finder for each of the 18 targets: a line finder for each
  of the 17 coordinates, or the binding reduction
  (`RandomOracleReduction.reduction`) on its two finished runs. Each finder
  is a program that holds no adversary: it makes at most `T` runs on every
  tape (`hashRuns_length_le`), each run depends only on the records of the
  earlier runs (`hashRuns_take_congr`), and the output only on the records of
  the runs made (`hashSolve_congr`);
- `hashAdvantage`: its success when it draws the chunks, the target and the
  tape uniformly;
- `hashCollisions_le_solver`:
  `𝔼 hashCollisions ≤ 18 · hashAdvantage T + (52 Q + 959) / T`. Each line
  finder takes at most `Q + 18` expected runs
  (`RandomOracleExtraction.expected_retries_at_le`), and the binding reduction
  `35 Q + 653` (`RandomOracleSolver.expected_runs_le`).

`hashCollisions` is a sum over the 18 targets, so each target needs its own
finder; one finder that stops at the first collision would bound only the
largest term.

Does not own: the hardness of the state hash.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.RandomOracleHashSolver

open scoped BigOperators
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.RandomOracle
open NightstreamFPrime.Spec.RandomOracleBudget
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.AjtaiSetupV1.Programming
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.TranscriptCoverage
open NightstreamFPrime.Lifecycle.RandomOracleTest
open NightstreamFPrime.Lifecycle.RandomOracleExtraction
open NightstreamFPrime.Lifecycle.RandomOracleUniqueness
open NightstreamFPrime.Lifecycle.RandomOracleReduction
open NightstreamFPrime.Lifecycle.RandomOracleSolver
open NightstreamFPrime.Export.Stage1.RandomOracleLink
open NightstreamFPrime.Export.Stage1.RandomOracleSetup
open NightstreamFPrime.Layout.Stage1.PiCCSSecurity
open StrongReduction ConcreteCarrier

attribute [local instance low] Classical.propDecidable

/-- The line finder between runs. -/
inductive LineState (Run : Type)
  | start
  | retry (base : Run)
  | finished (base other : Run)
  | failed

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth}

/-- Two prior preimages with different encodings and one state hash. -/
def HashCollision :=
  {pair : Lifecycle.HashPreimage (logicalWidth := logicalWidth) (publicFits := publicFits) ×
      Lifecycle.HashPreimage (logicalWidth := logicalWidth) (publicFits := publicFits) //
    StateHashCollision pair.1 pair.2}

variable (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
  (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
  {Output : Type}
  (claim : Output → Claim relation)
  (prior : Output → Lifecycle.HashPreimage (logicalWidth := logicalWidth) (publicFits := publicFits))
  (contextDigest : KeyDigest)

local notation "Linked" => linkedClaim relation claim prior contextDigest
local notation "Oracle" => Point logicalWidth publicFits (ProductionKey.degreeBound relation) → Answer
local notation "Record" => RunRecord (Point logicalWidth publicFits (ProductionKey.degreeBound relation)) Output
local notation "Coordinate" => Fin Nifs.PaperProfile.arity.total
local notation "Made" => Run Oracle Record
local notation "Retried" => Forks Coordinate Answer Record

/-- Tapes of oracles are finite (named here: instance search exceeds its size
limit on this type). -/
noncomputable local instance (budget : ℕ) : Fintype (Fin budget → Oracle) := Pi.instFintype

/-! ## The finders -/

/-- The prior preimages of two runs, when they collide. -/
noncomputable def collisionOf (base other : Made) :
    Option (HashCollision (logicalWidth := logicalWidth) (publicFits := publicFits)) :=
  if collides : StateHashCollision (prior base.record.output) (prior other.record.output) then
    some ⟨(prior base.record.output, prior other.record.output), collides⟩
  else none

/-- The oracle of the line finder's next run: a retry resamples the base run's
answer at its `index`-th `Π_RLC` point. -/
noncomputable def lineQuery (index : Coordinate) : LineState Made → Oracle → Oracle
  | .retry base, fresh =>
      Function.update base.oracle (rhoPointAt relation Linked base index) (fresh (rhoPointAt relation Linked base index))
  | _, fresh => fresh

/-- **The line finder of coordinate `index`.** One adversary run per step; it
halts when a retry hits the base run's point again, or when the base run
fails. -/
noncomputable def lineFinder (index : Coordinate) : RunMachine (LineState Made) Oracle Oracle Record where
  query := lineQuery relation claim prior contextDigest index
  next
    | .start, fresh, record =>
        if SucceedsOf relation ajtai fresh (Linked record.output) then .retry ⟨fresh, record⟩ else .failed
    | .retry base, fresh, record =>
        if SucceedsOf relation ajtai (lineQuery relation claim prior contextDigest index (.retry base) fresh)
              (Linked record.output) ∧
            rhoPointOf relation (Linked record.output) index = rhoPointAt relation Linked base index
        then .finished base ⟨lineQuery relation claim prior contextDigest index (.retry base) fresh, record⟩
        else .retry base
    | state, _, _ => state
  halted
    | .finished .. => true
    | .failed => true
    | _ => false

/-- The line finder's output on the two runs it finished with. -/
noncomputable def lineOutput : LineState Made → Option (HashCollision (logicalWidth := logicalWidth) (publicFits := publicFits))
  | .finished base other => collisionOf relation prior base other
  | _ => none

/-- The binding reduction's output as a finder, on the two runs it finished with. -/
noncomputable def rerunOutput :
    State Made Retried Coordinate → Option (HashCollision (logicalWidth := logicalWidth) (publicFits := publicFits))
  | .finished base _ other _ => collisionOf relation prior base other
  | _ => none

variable (adversary : OracleComp (Point logicalWidth publicFits (ProductionKey.degreeBound relation))
  Answer Output)

/-- **The collision finder of `target`**, with at most `budget` adversary
runs: the line finder of a coordinate, or the binding reduction. -/
noncomputable def hashSolve (target : Option Coordinate) (budget : ℕ) (tape : Fin budget → Oracle) :
    Option (HashCollision (logicalWidth := logicalWidth) (publicFits := publicFits)) :=
  match target with
  | some index => lineOutput relation prior
      (((lineFinder relation ajtai claim prior contextDigest index).program budget .start tape).run adversary.record)
  | none => rerunOutput relation prior (finalState relation ajtai Linked adversary budget tape)

/-- The oracles of the runs that `hashSolve` makes, in order. -/
noncomputable def hashRuns (target : Option Coordinate) (budget : ℕ) (tape : Fin budget → Oracle) : List Oracle :=
  match target with
  | some index => ((lineFinder relation ajtai claim prior contextDigest index).program budget .start tape).queries
      adversary.record
  | none => runs relation ajtai Linked adversary budget tape

/-- **Strict work.** At most `budget` adversary runs on every tape. -/
theorem hashRuns_length_le (target : Option Coordinate) (budget : ℕ) (tape : Fin budget → Oracle) :
    (hashRuns relation ajtai claim prior contextDigest adversary target budget tape).length ≤ budget := by
  cases target with
  | some index =>
      exact ((lineFinder relation ajtai claim prior contextDigest index).program_bound budget .start tape).queries_length_le _
  | none => exact runs_length_le relation ajtai Linked adversary budget tape

/-- **Causal runs.** The first `step + 1` runs depend on the adversary only
through the records of the first `step`. -/
theorem hashRuns_take_congr
    (other : OracleComp (Point logicalWidth publicFits (ProductionKey.degreeBound relation)) Answer Output)
    (target : Option Coordinate) (budget : ℕ) (tape : Fin budget → Oracle) (step : ℕ)
    (agree : ∀ oracle ∈ (hashRuns relation ajtai claim prior contextDigest adversary target budget tape).take step,
      adversary.record oracle = other.record oracle) :
    (hashRuns relation ajtai claim prior contextDigest other target budget tape).take (step + 1) =
      (hashRuns relation ajtai claim prior contextDigest adversary target budget tape).take (step + 1) := by
  cases target with
  | some index => exact OracleComp.take_succ_queries_congr _ _ _ step agree
  | none => exact runs_take_congr relation ajtai Linked adversary other budget tape step agree

/-- **Causal output.** The output depends on the adversary only through the
records of the runs made. -/
theorem hashSolve_congr
    (other : OracleComp (Point logicalWidth publicFits (ProductionKey.degreeBound relation)) Answer Output)
    (target : Option Coordinate) (budget : ℕ) (tape : Fin budget → Oracle)
    (agree : ∀ oracle ∈ hashRuns relation ajtai claim prior contextDigest adversary target budget tape,
      adversary.record oracle = other.record oracle) :
    hashSolve relation ajtai claim prior contextDigest other target budget tape =
      hashSolve relation ajtai claim prior contextDigest adversary target budget tape := by
  cases target with
  | some index =>
      simp only [hashSolve]
      rw [OracleComp.run_congr _ _ _ agree]
  | none =>
      simp only [hashSolve, finalState]
      rw [OracleComp.run_congr _ _ _ agree]

/-- The chance that the finder of `target` returns a collision. -/
noncomputable def targetChance (target : Option Coordinate) (budget : ℕ) : ℝ :=
  𝔼 tape : Fin budget → Oracle,
    if (hashSolve relation ajtai claim prior contextDigest adversary target budget tape).isSome then 1 else 0

/-! ## The line finder's laws -/

private theorem collision_indicator (base other : Made) :
    (if (collisionOf relation prior base other).isSome then (1 : ℝ) else 0) =
      if StateHashCollision (prior base.record.output) (prior other.record.output) then 1 else 0 := by
  by_cases collides : StateHashCollision (prior base.record.output) (prior other.record.output) <;>
    simp [collisionOf, collides]

/-- A retry loop's outcome law gives the base run's collision chance. -/
private theorem lineAverage_collision (index : Coordinate) (base : Oracle) :
    lineAverage (retrySets relation ajtai adversary Linked base index) (fun answer =>
        if StateHashCollision (prior (adversary.run base))
          (prior (adversary.run (Function.update base (rhoPoint relation adversary Linked index base) answer)))
        then 1 else 0) =
      hashMismatchChance relation ajtai adversary claim prior contextDigest index base := by
  have numerator : (𝔼 answer, if answer ∈ retrySets relation ajtai adversary Linked base index then
        (if StateHashCollision (prior (adversary.run base))
          (prior (adversary.run (Function.update base (rhoPoint relation adversary Linked index base) answer)))
        then (1 : ℝ) else 0) else 0) =
      mass (retrySets relation ajtai adversary Linked base index ∩
        {answer | StateHashCollision (prior (adversary.run base))
          (prior (adversary.run (Function.update base (rhoPoint relation adversary Linked index base) answer)))}) :=
    Finset.expect_congr rfl fun answer _ => by
      by_cases inside : answer ∈ retrySets relation ajtai adversary Linked base index
      · rw [if_pos inside]
        by_cases collides : StateHashCollision (prior (adversary.run base))
            (prior (adversary.run (Function.update base (rhoPoint relation adversary Linked index base) answer)))
        · rw [if_pos collides, if_pos ⟨inside, collides⟩]
        · rw [if_neg collides, if_neg fun both => collides both.2]
      · rw [if_neg inside, if_neg fun both => inside both.1]
  exact congrArg (· / mass (retrySets relation ajtai adversary Linked base index)) numerator

private noncomputable def lineSuccess (index : Coordinate) : LineState Made → ℝ
  | .start => 𝔼 oracle, if Succeeds relation ajtai adversary Linked oracle then
      hashMismatchChance relation ajtai adversary claim prior contextDigest index oracle else 0
  | .retry base => hashMismatchChance relation ajtai adversary claim prior contextDigest index base.oracle
  | .finished base other => if (collisionOf relation prior base other).isSome then 1 else 0
  | .failed => 0

private noncomputable def lineWork (index : Coordinate) : LineState Made → ℝ
  | .start => 1 + 𝔼 oracle, if Succeeds relation ajtai adversary Linked oracle then
      retryCount (Hits relation ajtai adversary Linked index) (rhoPoint relation adversary Linked index oracle) oracle
    else 0
  | .retry base => retryCount (Hits relation ajtai adversary Linked index)
      (rhoPoint relation adversary Linked index base.oracle) base.oracle
  | _ => 0

/-- A pending retry loop stores the adversary's record of a base run that
succeeds. -/
private def LineGood : LineState Made → Prop
  | .retry base => base = runOf relation adversary base.oracle ∧ Succeeds relation ajtai adversary Linked base.oracle
  | _ => True

private theorem hashMismatchChance_le_one (index : Coordinate) (base : Oracle) :
    hashMismatchChance relation ajtai adversary claim prior contextDigest index base ≤ 1 :=
  div_le_one_of_le₀ (mass_mono Set.inter_subset_left) (mass_nonnegative _)

private theorem hashMismatchChance_nonnegative (index : Coordinate) (base : Oracle) :
    0 ≤ hashMismatchChance relation ajtai adversary claim prior contextDigest index base :=
  div_nonneg (mass_nonnegative _) (mass_nonnegative _)

/-- The line finder's potentials: its eventual chance of a collision and its
expected remaining runs, exactly. -/
private noncomputable def lineBounds (index : Coordinate) :
    ((lineFinder relation ajtai claim prior contextDigest index).rewinding adversary.record).Bounds
      (fun state => (lineOutput relation prior state).isSome) where
  good := LineGood relation ajtai claim prior contextDigest adversary
  success := lineSuccess relation ajtai claim prior contextDigest adversary index
  work := lineWork relation ajtai claim prior contextDigest adversary index
  good_step := by
    intro state good unhalted fresh
    cases state with
    | start =>
        show LineGood relation ajtai claim prior contextDigest adversary
          (if SucceedsOf relation ajtai fresh (Linked (adversary.record fresh).output) then
            .retry ⟨fresh, adversary.record fresh⟩ else .failed)
        split
        · rename_i succeeds
          exact ⟨rfl, succeeds⟩
        · trivial
    | retry base =>
        show LineGood relation ajtai claim prior contextDigest adversary (if _ then _ else _)
        split
        · trivial
        · exact good
    | finished => simp [RunMachine.rewinding, lineFinder] at unhalted
    | failed => simp [RunMachine.rewinding, lineFinder] at unhalted
  success_le_one := by
    intro state _
    cases state with
    | start =>
        refine (Finset.expect_le_expect fun oracle _ => ?_).trans_eq (Finset.expect_const Finset.univ_nonempty _)
        split
        · exact hashMismatchChance_le_one relation ajtai claim prior contextDigest adversary index oracle
        · exact zero_le_one
    | retry base => exact hashMismatchChance_le_one relation ajtai claim prior contextDigest adversary index _
    | finished => show (if _ then (1 : ℝ) else 0) ≤ 1; split <;> norm_num
    | failed => show (0 : ℝ) ≤ 1; norm_num
  success_halted := by
    intro state _ halted
    cases state with
    | finished => exact le_rfl
    | failed => show (0 : ℝ) ≤ _; simp [lineOutput]
    | start => simp [RunMachine.rewinding, lineFinder] at halted
    | retry => simp [RunMachine.rewinding, lineFinder] at halted
  success_step := by
    intro state good unhalted
    cases state with
    | start =>
        refine le_of_eq (Finset.expect_congr rfl fun fresh _ => ?_)
        show _ = lineSuccess relation ajtai claim prior contextDigest adversary index
          (if SucceedsOf relation ajtai fresh (Linked (adversary.record fresh).output) then
            .retry ⟨fresh, adversary.record fresh⟩ else .failed)
        rw [apply_ite (lineSuccess relation ajtai claim prior contextDigest adversary index)]
        exact if_congr Iff.rfl rfl rfl
    | retry base =>
        obtain ⟨recorded, succeeds⟩ := good
        rw [recorded]
        let run := base.oracle
        let point := rhoPoint relation adversary Linked index run
        let set := retrySets relation ajtai adversary Linked run index
        let value : Answer → ℝ := fun answer =>
          if StateHashCollision (prior (adversary.run run))
            (prior (adversary.run (Function.update run point answer))) then 1 else 0
        have positive : mass set ≠ 0 := mass_ne_zero (base_mem_retrySets relation ajtai adversary Linked succeeds index)
        have pointwise (fresh : Oracle) :
            lineSuccess relation ajtai claim prior contextDigest adversary index
                (((lineFinder relation ajtai claim prior contextDigest index).rewinding adversary.record).step
                  (.retry (runOf relation adversary run)) fresh) =
              (fun answer => if answer ∈ set then value answer else lineAverage set value) (fresh point) := by
          show lineSuccess relation ajtai claim prior contextDigest adversary index (if _ then _ else _) = _
          rw [apply_ite (lineSuccess relation ajtai claim prior contextDigest adversary index)]
          exact if_congr Iff.rfl (collision_indicator relation prior _ _)
            (lineAverage_collision relation ajtai claim prior contextDigest adversary index run).symm
        calc
          _ = lineAverage set value := (lineAverage_collision relation ajtai claim prior contextDigest adversary index run).symm
          _ = _ := (lineAverage_step set positive value).symm
          _ = _ := (expect_eval point _).symm
          _ = _ := Finset.expect_congr rfl fun fresh _ => (pointwise fresh).symm
          _ ≤ _ := le_rfl
    | finished => simp [RunMachine.rewinding, lineFinder] at unhalted
    | failed => simp [RunMachine.rewinding, lineFinder] at unhalted
  work_nonnegative := by
    intro state _
    cases state with
    | start =>
        refine add_nonneg zero_le_one (Finset.expect_nonneg fun oracle _ => ?_)
        split
        · exact div_nonneg zero_le_one (mass_nonnegative _)
        · exact le_rfl
    | retry => exact div_nonneg zero_le_one (mass_nonnegative _)
    | finished => exact le_rfl
    | failed => exact le_rfl
  work_step := by
    intro state good unhalted
    cases state with
    | start =>
        refine le_of_eq (congrArg (1 + ·) (Finset.expect_congr rfl fun fresh _ => ?_))
        show lineWork relation ajtai claim prior contextDigest adversary index
          (if SucceedsOf relation ajtai fresh (Linked (adversary.record fresh).output) then
            .retry ⟨fresh, adversary.record fresh⟩ else .failed) = _
        rw [apply_ite (lineWork relation ajtai claim prior contextDigest adversary index)]
        exact if_congr Iff.rfl rfl rfl
    | retry base =>
        obtain ⟨recorded, succeeds⟩ := good
        rw [recorded]
        let run := base.oracle
        let point := rhoPoint relation adversary Linked index run
        let set := retrySets relation ajtai adversary Linked run index
        have positive : mass set ≠ 0 := mass_ne_zero (base_mem_retrySets relation ajtai adversary Linked succeeds index)
        have empty : lineAverage set (fun _ => 0) = 0 := by simp [lineAverage]
        let attempt : Answer → ℝ := fun answer =>
          if answer ∈ set then 0 + 0 else 1 / mass set + 0 + lineAverage set fun _ => 0
        have pointwise (fresh : Oracle) :
            lineWork relation ajtai claim prior contextDigest adversary index
                (((lineFinder relation ajtai claim prior contextDigest index).rewinding adversary.record).step
                  (.retry (runOf relation adversary run)) fresh) = attempt (fresh point) := by
          show lineWork relation ajtai claim prior contextDigest adversary index (if _ then _ else _) = _
          rw [apply_ite (lineWork relation ajtai claim prior contextDigest adversary index)]
          refine if_congr Iff.rfl (by simp only [lineWork, add_zero]) ?_
          rw [empty, add_zero, add_zero]
          rfl
        calc
          _ = 1 + 𝔼 fresh : Oracle, attempt (fresh point) :=
            congrArg (1 + ·) (Finset.expect_congr rfl fun fresh _ => pointwise fresh)
          _ = 1 + 𝔼 answer, attempt answer := congrArg (1 + ·) (expect_eval point attempt)
          _ = 1 / mass set + 0 + lineAverage set (fun _ => 0) := line_work_step set positive 0 fun _ => 0
          _ = 1 / mass set := by rw [empty, add_zero, add_zero]
          _ ≤ _ := le_rfl
    | finished => simp [RunMachine.rewinding, lineFinder] at unhalted
    | failed => simp [RunMachine.rewinding, lineFinder] at unhalted

/-- **The Markov loss of one line finder.** -/
private theorem line_le {queries : Nat} (bounded : adversary.QueryBound queries) (index : Coordinate)
    (budget : ℕ) :
    budget * (𝔼 oracle, (if Succeeds relation ajtai adversary Linked oracle then
          hashMismatchChance relation ajtai adversary claim prior contextDigest index oracle else 0) -
        targetChance relation ajtai claim prior contextDigest adversary (some index) budget) ≤
      queries + 18 := by
  have loss := (lineBounds relation ajtai claim prior contextDigest adversary index).budget_mul_le budget
    (state := .start) trivial
  have chance : ((lineFinder relation ajtai claim prior contextDigest index).rewinding adversary.record).chance
      (fun state => (lineOutput relation prior state).isSome) budget .start =
        targetChance relation ajtai claim prior contextDigest adversary (some index) budget :=
    Finset.expect_congr rfl fun tape _ => by
      rw [← ((lineFinder relation ajtai claim prior contextDigest index).run_program adversary.record budget
        .start tape).1]
      rfl
  have retries := expected_retries_at_le relation ajtai adversary Linked bounded index
  simp only [Nifs.PaperProfile.arity_total] at retries
  push_cast at retries
  rw [chance] at loss
  refine loss.trans ?_
  show 1 + _ ≤ _
  linarith

/-! ## The bound -/

/-- **Per key.** With budget `T`, the 18 finders lose at most
`(52 Q + 959) / T` of `hashCollisions`. -/
private theorem collisions_le_targets {queries : Nat} (bounded : adversary.QueryBound queries) (budget : ℕ) :
    budget * ((𝔼 oracle, (if Succeeds relation ajtai adversary Linked oracle then
          ∑ index, hashMismatchChance relation ajtai adversary claim prior contextDigest index oracle else 0) +
        hashRunningChance relation ajtai adversary claim prior contextDigest) -
      ∑ target, targetChance relation ajtai claim prior contextDigest adversary target budget) ≤
      52 * queries + 959 := by
  have rerun := reductionChance_le_program relation ajtai adversary Linked bounded
    (fun state => (rerunOutput relation prior state).isSome) budget
  have outcome : (fun base retries other otherRetries =>
      (rerunOutput relation prior (.finished (runOf relation adversary base)
        (forksOf relation adversary Linked base retries) (runOf relation adversary other)
        (forksOf relation adversary Linked other otherRetries))).isSome = true) =
      fun base _ other _ => StateHashCollision (prior (adversary.run base)) (prior (adversary.run other)) := by
    funext base retries other otherRetries
    by_cases collides : StateHashCollision (prior (adversary.run base)) (prior (adversary.run other)) <;>
      simp [rerunOutput, collisionOf, runOf, OracleComp.record, collides]
  rw [outcome] at rerun
  change budget * (hashRunningChance relation ajtai adversary claim prior contextDigest -
    targetChance relation ajtai claim prior contextDigest adversary none budget) ≤ _ at rerun
  have lines (index : Coordinate) := line_le relation ajtai claim prior contextDigest adversary bounded index budget
  have split : 𝔼 oracle, (if Succeeds relation ajtai adversary Linked oracle then
        ∑ index, hashMismatchChance relation ajtai adversary claim prior contextDigest index oracle else 0) =
      ∑ index, 𝔼 oracle, (if Succeeds relation ajtai adversary Linked oracle then
        hashMismatchChance relation ajtai adversary claim prior contextDigest index oracle else 0) := by
    rw [← Finset.expect_sum_comm]
    refine Finset.expect_congr rfl fun oracle _ => ?_
    split <;> simp
  have summed := Finset.sum_le_sum fun index (_ : index ∈ Finset.univ) => lines index
  have total : ((Nifs.PaperProfile.arity).total : ℝ) = 17 := by
    rw [Nifs.PaperProfile.arity_total]
    norm_num
  rw [← Finset.mul_sum, Finset.sum_sub_distrib, Finset.sum_const, Finset.card_univ, Fintype.card_fin,
    nsmul_eq_mul, total] at summed
  rw [split, Fintype.sum_option]
  linarith

/-- The finders' success over uniform chunks, a uniform target and a uniform
tape. -/
noncomputable def hashAdvantage
    (adversary : (SetupIndex logicalWidth publicFits → Chunk) →
      OracleComp (Point logicalWidth publicFits (ProductionKey.degreeBound relation)) Answer Output)
    (budget : ℕ) : ℝ :=
  𝔼 chunks : SetupIndex logicalWidth publicFits → Chunk, 𝔼 target : Option Coordinate,
    targetChance relation (setupKey chunks) claim prior contextDigest (adversary chunks) target budget

/-- **The state-hash term at strict work.** The 18 finders with a strict
budget of `T` adversary runs bound the state-hash collision chances of the
knowledge theorem, with a loss of `(52 Q + 959) / T`. -/
theorem hashCollisions_le_solver
    (adversary : (SetupIndex logicalWidth publicFits → Chunk) →
      OracleComp (Point logicalWidth publicFits (ProductionKey.degreeBound relation)) Answer Output)
    {queries : Nat} (bounded : ∀ chunks, (adversary chunks).QueryBound queries) {budget : ℕ}
    (positive : 0 < budget) :
    𝔼 chunks, hashCollisions relation adversary claim prior contextDigest chunks ≤
      18 * hashAdvantage relation claim prior contextDigest adversary budget + (52 * queries + 959) / budget := by
  have positiveReal : (0 : ℝ) < budget := by exact_mod_cast positive
  have card : (Fintype.card (Option Coordinate) : ℝ) = 18 := by
    rw [Fintype.card_option, Fintype.card_fin, Nifs.PaperProfile.arity_total]
    norm_num
  have each (chunks : SetupIndex logicalWidth publicFits → Chunk) :
      hashCollisions relation adversary claim prior contextDigest chunks ≤
        18 * 𝔼 target : Option Coordinate,
            targetChance relation (setupKey chunks) claim prior contextDigest (adversary chunks) target budget +
          (52 * queries + 959) / budget := by
    have bound := collisions_le_targets relation (setupKey chunks) claim prior contextDigest (adversary chunks)
      (bounded chunks) budget
    rw [Finset.expect_eq_sum_div_card, Finset.card_univ, card]
    unfold hashCollisions
    rw [← sub_le_iff_le_add', le_div_iff₀ positiveReal]
    linarith
  unfold hashAdvantage
  calc
    _ ≤ 𝔼 chunks : SetupIndex logicalWidth publicFits → Chunk,
          (18 * 𝔼 target : Option Coordinate,
              targetChance relation (setupKey chunks) claim prior contextDigest (adversary chunks) target budget +
            (52 * queries + 959) / budget) :=
      Finset.expect_le_expect fun chunks _ => each chunks
    _ = _ := by
      rw [Finset.expect_add_distrib, Finset.expect_const Finset.univ_nonempty, ← Finset.mul_expect]

end NightstreamFPrime.Export.Stage1.RandomOracleHashSolver
