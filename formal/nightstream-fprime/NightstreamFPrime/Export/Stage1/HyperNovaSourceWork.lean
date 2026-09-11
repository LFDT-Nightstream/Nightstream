import NightstreamFPrime.Export.Stage1.HyperNovaVisitedSecurity
import NightstreamFPrime.Export.Stage1.HyperNovaHistoryWork

/-!
Declared source work follows the operational history, including calls after
false analytical marks and calls that abort. The source kernel always uses
its original recursive visit with mark true. Only `ready` controls whether a
source call is made; `goodActive` has no role in this clock.

The additive recurrence uses each call's existing expected clock and its
actual source-result law. It does not turn an expected cost into an observed
random cost. No machine-time, decoder-cost, or source-success assumption is
introduced. The separate orchestration allowance remains result-list length
plus one.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.HyperNovaSourceWork

open scoped BigOperators ENNReal
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.Nifs
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open StrongReduction
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.Nifs
open PiRLC.PaperForkExtractionWork (Result Primitives)
open PiRLC.CoordinateForkLaw (Challenge)
open HyperNovaHistory (Statement Envelope Payload SourceResult)
open HyperNovaVisitedLaw (Visit ready stopped transition visitedLaw)
open HyperNovaGuardedSourceLaw (inputs)
open Poseidon2HashChainV1Setup (productionAjtaiKey)
open PiDECInputCheck (relation)

attribute [local instance] Classical.propDecidable

/-- The operational kernel ignores the analytical mark. This changes no
terminal data, public input, witness, or transcript. -/
def callContext (visit : Visit) : Visit := (visit.1, true)

private def abortPrefix {State : Type*} : InteractivePrefix.Prover State productionShape 9 where
  rounds := fun _ _ _ => none
  output := fun _ _ _ => none

private theorem abortPrefix_run {State : Type*}
    (alpha : CubePoint K productionShape.cubeVariables) (gamma : K)
    (point : CubePoint K productionShape.cubeVariables) :
    InteractivePrefix.run (abortPrefix (State := State)) alpha gamma point = none := by
  unfold InteractivePrefix.run
  split <;> rfl

private noncomputable def mean {Sample : Type*} (distribution : PMF Sample)
    (value : Sample → ℝ≥0∞) : ℝ≥0∞ :=
  ∑' sample, distribution sample * value sample

private theorem mean_pure {Sample : Type*} (sample : Sample) (value : Sample → ℝ≥0∞) :
    mean (PMF.pure sample) value = value sample := by
  unfold mean
  rw [tsum_eq_single sample]
  · rw [PMF.pure_apply_self, one_mul]
  · intro other different
    rw [PMF.pure_apply_of_ne sample other different, zero_mul]

private theorem mean_bind {Sample Output : Type*} (distribution : PMF Sample)
    (next : Sample → PMF Output) (value : Output → ℝ≥0∞) :
    mean (distribution.bind next) value = mean distribution (fun sample => mean (next sample) value) := by
  unfold mean
  simp only [PMF.bind_apply, ← ENNReal.tsum_mul_right]
  rw [ENNReal.tsum_comm]
  apply tsum_congr
  intro sample
  simp only [mul_assoc, ENNReal.tsum_mul_left]

private theorem mean_map {Sample Output : Type*} (distribution : PMF Sample)
    (map : Sample → Output) (value : Output → ℝ≥0∞) :
    mean (distribution.map map) value = mean distribution (fun sample => value (map sample)) := by
  rw [PMF.map, mean_bind]
  simp only [Function.comp_apply, mean_pure]

private theorem mean_add {Sample : Type*} (distribution : PMF Sample)
    (left right : Sample → ℝ≥0∞) :
    mean distribution (fun sample => left sample + right sample) =
      mean distribution left + mean distribution right := by
  simp only [mean, mul_add, ENNReal.tsum_add]

private noncomputable def expectedAt
    (next : Visit → PMF Visit) (toll : Visit → ℝ≥0∞) : Nat → Visit → ℝ≥0∞
  | 0, _ => 0
  | remaining + 1, visit => toll visit + mean (next visit) (expectedAt next toll remaining)

private def remaining (visit : Visit) : Nat :=
  match visit.1 with
  | none => 0
  | some (statement, _) => statement.iteration

private theorem transition_inactive (source : Statement → Payload → PMF SourceResult)
    (visit : Visit) (inactive : ¬ ready visit) : transition source visit = PMF.pure stopped := by
  simp only [transition, HyperNovaVisitedLaw.draw, if_neg inactive,
    PMF.pure_map, HyperNovaVisitedLaw.advance, if_neg inactive]

private theorem expectedAt_stopped (source : Statement → Payload → PMF SourceResult)
    (toll : Visit → ℝ≥0∞) (zero : toll stopped = 0) (depth : Nat) :
    expectedAt (transition source) toll depth stopped = 0 := by
  induction depth with
  | zero => rfl
  | succ depth induction =>
      rw [expectedAt, zero, transition_inactive source stopped (by exact id), mean_pure, induction, zero_add]

private theorem expectedAt_inactive (source : Statement → Payload → PMF SourceResult)
    (toll : Visit → ℝ≥0∞) (zero : ∀ visit, ¬ ready visit → toll visit = 0)
    (depth : Nat) (visit : Visit) (inactive : ¬ ready visit) :
    expectedAt (transition source) toll depth visit = 0 := by
  cases depth with
  | zero => rfl
  | succ depth =>
      rw [expectedAt, zero visit inactive, transition_inactive source visit inactive, mean_pure,
        expectedAt_stopped source toll (zero stopped (by exact id)), zero_add]

private theorem remaining_of_active (visit : Visit) (active : ready visit) : 0 < remaining visit := by
  rcases visit with ⟨current, mark⟩
  cases current with
  | none => exact False.elim active
  | some input =>
      rcases input with ⟨statement, proof⟩
      cases proof with
      | bottom => exact False.elim active
      | recursive payload => exact Nat.pos_of_ne_zero active.1

private theorem next_remaining_lt (source : Statement → Payload → PMF SourceResult)
    (visit next : Visit) (active : ready visit)
    (supported : next ∈ (transition source visit).support) : remaining next < remaining visit := by
  rw [transition] at supported
  rcases (PMF.mem_support_map_iff _ _ _).mp supported with ⟨result, _produced, same⟩
  rw [← same]
  rcases visit with ⟨current, mark⟩
  cases current with
  | none => exact False.elim active
  | some input =>
      rcases input with ⟨statement, proof⟩
      cases proof with
      | bottom => exact False.elim active
      | recursive payload =>
          cases result with
          | none =>
              simp only [HyperNovaVisitedLaw.advance, if_pos active, remaining, stopped]
              exact Nat.pos_of_ne_zero active.1
          | some values =>
              have counter := active.2.1
              simp only [HyperNovaVisitedLaw.advance, if_pos active, remaining,
                HyperNovaHistory.predecessorStatement]
              omega

private theorem expectedAt_stable (source : Statement → Payload → PMF SourceResult)
    (toll : Visit → ℝ≥0∞) (zero : ∀ visit, ¬ ready visit → toll visit = 0)
    (depth larger : Nat) (visit : Visit) (bound : remaining visit ≤ depth) (largerBound : depth ≤ larger) :
    expectedAt (transition source) toll larger visit = expectedAt (transition source) toll depth visit := by
  induction depth generalizing larger visit with
  | zero =>
      have inactive : ¬ ready visit := fun active => by
        have positive := remaining_of_active visit active
        omega
      rw [expectedAt_inactive source toll zero larger visit inactive]
      rfl
  | succ depth induction =>
      by_cases active : ready visit
      · cases larger with
        | zero => omega
        | succ larger =>
            rw [expectedAt, expectedAt]
            apply congrArg (toll visit + ·)
            unfold mean
            apply tsum_congr
            intro next
            by_cases absent : transition source visit next = 0
            · simp only [absent, zero_mul]
            · apply congrArg (transition source visit next * ·)
              apply induction larger next
              · have decreases := next_remaining_lt source visit next active
                  (((transition source visit).mem_support_iff next).mpr absent)
                omega
              · omega
      · rw [expectedAt_inactive source toll zero larger visit active,
          expectedAt_inactive source toll zero (depth + 1) visit active]

private theorem expectedAt_mean_succ (source : Statement → Payload → PMF SourceResult)
    (toll : Visit → ℝ≥0∞) (contexts : PMF Visit) (depth : Nat) :
    mean contexts (expectedAt (transition source) toll (depth + 1)) =
      mean contexts toll + mean (contexts.bind (transition source)) (expectedAt (transition source) toll depth) := by
  change mean contexts (fun visit => toll visit +
    mean (transition source visit) (expectedAt (transition source) toll depth)) = _
  rw [mean_add, mean_bind]

private theorem expectedAt_mean_eq_visitedSum
    (source : Statement → Payload → PMF SourceResult) (toll : Visit → ℝ≥0∞)
    (initial : PMF (Statement × Envelope)) (depth step : Nat) :
    mean (visitedLaw source initial step) (expectedAt (transition source) toll depth) =
      ∑ j : Fin depth, mean (visitedLaw source initial (step + j.val)) toll := by
  induction depth generalizing step with
  | zero => simp only [expectedAt, mean, mul_zero, tsum_zero, Fin.sum_univ_zero]
  | succ depth induction =>
      rw [expectedAt_mean_succ, ← HyperNovaVisitedLaw.visitedLaw_succ, induction,
        Fin.sum_univ_succ]
      simp only [Fin.val_zero, Nat.add_zero, Fin.val_succ]
      congr 1
      apply Finset.sum_congr rfl
      intro j _member
      congr 2
      omega

variable {State Tape : Type*}
  (originalFirstPhase : Visit → InteractivePrefix.Prover State productionShape 9)
  (continuation : ∀ visit (coins : PublicCoins K productionShape)
    (output : FullOutputCoordinates.FullOutput K productionShape), State →
      WeakExtraction.Continuation Tape relation productionAjtaiKey
        (PiCCSInputCheck.running (inputs visit)) (PiCCSInputCheck.fresh (inputs visit)) coins output)

/-- Call the fixed causal prefix at the actual operational source context.
Inactive visits supply an explicit abort before the checker or suffix. -/
noncomputable def operationalPrefix (visit : Visit) :
    InteractivePrefix.Prover State productionShape 9 :=
  if ready visit then originalFirstPhase (callContext visit) else abortPrefix

/-- The same fixed continuation supplies every operational call. Resetting
the analytical mark preserves inputs definitionally. -/
def operationalContinuation (visit : Visit) (coins : PublicCoins K productionShape)
    (output : FullOutputCoordinates.FullOutput K productionShape) (state : State) :
    WeakExtraction.Continuation Tape relation productionAjtaiKey
      (PiCCSInputCheck.running (inputs visit)) (PiCCSInputCheck.fresh (inputs visit)) coins output :=
  continuation (callContext visit) coins output state

/-- An inactive prefix aborts independently of any checker or suffix. -/
theorem operationalPrefix_abort (visit : Visit) (inactive : ¬ ready visit)
    (check : Probe K productionShape → Bool)
    (alpha : CubePoint K productionShape.cubeVariables) (gamma : K)
    (point : CubePoint K productionShape.cubeVariables) :
    InteractivePrefix.run (InteractivePrefix.checked (operationalPrefix originalFirstPhase visit) check)
      alpha gamma point = none := by
  rw [InteractivePrefix.run_checked]
  simp only [operationalPrefix, if_neg inactive, abortPrefix_run, Option.filter_none]

variable [DecidableEq RingF]
  [Fintype (Challenge (ProductionKey.key relation productionAjtaiKey).piRlcAlgebra)]
  [Nonempty (Challenge (ProductionKey.key relation productionAjtaiKey).piRlcAlgebra)]
  (program : Primitives RingF PiRLCExtractionPrimitives.Assignment)

private theorem atContext_active (visit : Visit) (active : ready visit) :
    HyperNovaSourceLaw.atContext inputs (operationalPrefix originalFirstPhase)
      (operationalContinuation continuation) program visit =
      HyperNovaSourceLaw.atContext inputs originalFirstPhase continuation program (callContext visit) := by
  have checkEq : SupportedExtraction.publicCheck
      (fun context => PiCCSInputCheck.running (inputs context)) (callContext visit) =
      SupportedExtraction.publicCheck
        (fun context => PiCCSInputCheck.running (inputs context)) visit := rfl
  unfold HyperNovaSourceLaw.atContext HyperNovaSourceLaw.law
  simp only [HyperNovaGuardedSourceLaw.sequential_atContext, PMF.map_bind, PMF.map_comp,
    InteractiveComposition.firstPhase, operationalPrefix, if_pos active, checkEq]
  apply congrArg (PMF.bind (VerifierCoinLaw.law productionShape))
  funext request
  cases returned : InteractivePrefix.run
      (InteractivePrefix.checked (originalFirstPhase (callContext visit))
        (SupportedExtraction.publicCheck (fun context => PiCCSInputCheck.running (inputs context)) visit))
      (VerifierCoinSpace.coins request).alpha (VerifierCoinSpace.coins request).gamma
      (VerifierCoinSpace.coins request).roundPoint <;>
    simp only [returned, PMF.pure_map, PMF.map_comp]
  all_goals rfl

private theorem atContext_inactive (visit : Visit) (inactive : ¬ ready visit) :
    HyperNovaSourceLaw.atContext inputs (operationalPrefix originalFirstPhase)
      (operationalContinuation continuation) program visit = PMF.pure none := by
  unfold HyperNovaSourceLaw.atContext HyperNovaSourceLaw.law
  simp only [HyperNovaGuardedSourceLaw.sequential_atContext, PMF.map_bind, PMF.map_comp,
    InteractiveComposition.firstPhase,
    operationalPrefix_abort originalFirstPhase visit inactive, PMF.pure_map, PMF.bind_const]
  rfl

/-- The operational clock's prefix and continuation produce exactly the draw
used by the existing history transition. False marks still execute the same
source kernel; inactive contexts return the unique absent result. -/
theorem operational_atContext_eq_draw (visit : Visit) :
    HyperNovaSourceLaw.atContext inputs (operationalPrefix originalFirstPhase)
      (operationalContinuation continuation) program visit =
      HyperNovaVisitedLaw.draw
        (HyperNovaGuardedSourceLaw.source originalFirstPhase continuation program) visit := by
  by_cases active : ready visit
  · rw [atContext_active originalFirstPhase continuation program visit active]
    rcases visit with ⟨current, mark⟩
    cases current with
    | none => exact False.elim active
    | some input =>
        rcases input with ⟨statement, proof⟩
        cases proof with
        | bottom => exact False.elim active
        | recursive payload =>
            rw [HyperNovaVisitedLaw.draw, if_pos active]
            rfl
  · rw [atContext_inactive originalFirstPhase continuation program visit active]
    simp only [HyperNovaVisitedLaw.draw, if_neg active]

variable
  (prefixClock : Visit → CubePoint K productionShape.cubeVariables → K →
    CubePoint K productionShape.cubeVariables → Nat)
  (sourceCheckClock : Visit → PiCCSStoredSourceProbability.CheckClock)
  (accessClock : Visit → PiCCSStoredSourceProbability.AccessClock)

/-- The existing checked prefix call with its fixed declared clock. Inactive
visits do not call the prefix and receive zero prefix work. -/
noncomputable def call :=
  NifsClosure.prefixCall
    (InteractiveComposition.firstPhase (operationalPrefix originalFirstPhase)
      (SupportedExtraction.publicCheck (fun visit => PiCCSInputCheck.running (inputs visit))))
    (fun visit alpha gamma point =>
      if ready visit then prefixClock (callContext visit) alpha gamma point else 0)

/-- The selected stored checker and accessors retain their original source
context's clocks. The analytical mark never changes the costed operation. -/
def sourceProgram (visit : Visit) :=
  PiCCSStoredSourceProbability.sourceProgram (inputs visit)
    (sourceCheckClock (callContext visit)) (accessClock (callContext visit))

/-- The existing retry, decode, checker and projection clock. Its inactive
branch includes the existing outer-return step; actual source work masks
that branch to zero below. -/
noncomputable def totalClock :=
  InteractiveWork.totalClock relation productionAjtaiKey
    (fun visit => PiCCSInputCheck.running (inputs visit))
    (fun visit => PiCCSInputCheck.fresh (inputs visit))
    (operationalContinuation continuation) (call originalFirstPhase prefixClock) program
    (sourceProgram sourceCheckClock accessClock)

/-- Expected work of the actual source call at this operational visit.
Inactive states make no call. Failed and false-mark active calls are counted. -/
noncomputable def callWork (visit : Visit) : ℝ :=
  if ready visit then
    StrongProbability.verifierMean
      (totalClock originalFirstPhase continuation program prefixClock sourceCheckClock accessClock visit)
  else 0

/-- The costed call returns the same checked receipt used by the proved
operational source law. No call-value equality is a caller premise. -/
theorem call_value (visit : Visit)
    (alpha : CubePoint K productionShape.cubeVariables) (gamma : K)
    (point : CubePoint K productionShape.cubeVariables) :
    ((call originalFirstPhase prefixClock) visit alpha gamma point).value =
      InteractivePrefix.run
        (InteractiveComposition.firstPhase (operationalPrefix originalFirstPhase)
          (SupportedExtraction.publicCheck (fun visit => PiCCSInputCheck.running (inputs visit))) visit)
        alpha gamma point := rfl

/-- Nonnegativity follows from the existing executed clock, including retries
and the actual selected postprocessing mean. -/
theorem totalClock_nonnegative (visit : Visit)
    (alpha : CubePoint K productionShape.cubeVariables) (gamma : K)
    (point : CubePoint K productionShape.cubeVariables) :
    0 ≤ totalClock originalFirstPhase continuation program prefixClock sourceCheckClock accessClock
      visit alpha gamma point :=
  PaperCompositionWork.totalClock_nonnegative _ _ _ _ _ _ _ _ _ _ _

/-- Masking an inactive non-call contributes zero, which is bounded by the
existing total-clock return step. Active calls retain their whole mean. -/
theorem callWork_le_totalMean (visit : Visit) :
    callWork originalFirstPhase continuation program prefixClock sourceCheckClock accessClock visit ≤
      StrongProbability.verifierMean
        (totalClock originalFirstPhase continuation program prefixClock sourceCheckClock accessClock visit) := by
  by_cases active : ready visit
  · simp only [callWork, if_pos active, le_refl]
  · rw [callWork, if_neg active]
    rw [← StrongProbability.verifierMean_const (shape := productionShape) 0]
    exact StrongProbability.verifierMean_mono _ _
      (totalClock_nonnegative originalFirstPhase continuation program prefixClock sourceCheckClock accessClock visit)

/-- The actual history's expected source work. Its recursion bound is each
initial statement's own iteration count. It uses the same operational kernel
as HistoryLaw, and retains failed calls and false-mark paths. -/
noncomputable def expectedSourceWork (initial : PMF (Statement × Envelope)) : ℝ≥0∞ :=
  let next := fun visit =>
    (HyperNovaSourceLaw.atContext inputs (operationalPrefix originalFirstPhase)
      (operationalContinuation continuation) program visit).map (HyperNovaVisitedLaw.advance visit)
  let toll := fun visit => ENNReal.ofReal
    (callWork originalFirstPhase continuation program prefixClock sourceCheckClock accessClock visit)
  mean initial fun input =>
    expectedAt next toll input.1.iteration (HyperNovaVisitedLaw.initialVisit input)

/-- The additive recurrence equals the sum of expected call work on the
exact unconditional visited laws. The symbolic depth bound adds no call and
truncates no history. No work/return independence or source-success premise
is used: expectations add even when call duration and output are correlated. -/
theorem expectedSourceWork_eq_visitedSum (initial : PMF (Statement × Envelope))
    (depth : Nat) (depthBound : ∀ input ∈ initial.support, input.1.iteration ≤ depth) :
    expectedSourceWork originalFirstPhase continuation program prefixClock sourceCheckClock accessClock initial =
      ∑ j : Fin depth,
        ∑' visit, (visitedLaw (HyperNovaGuardedSourceLaw.source originalFirstPhase continuation program)
          initial j.val) visit * ENNReal.ofReal
            (callWork originalFirstPhase continuation program prefixClock sourceCheckClock accessClock visit) := by
  let source := HyperNovaGuardedSourceLaw.source originalFirstPhase continuation program
  let toll := fun visit => ENNReal.ofReal
    (callWork originalFirstPhase continuation program prefixClock sourceCheckClock accessClock visit)
  have zero (visit : Visit) (inactive : ¬ ready visit) : toll visit = 0 := by
    simp only [toll, callWork, if_neg inactive, ENNReal.ofReal_zero]
  unfold expectedSourceWork
  simp_rw [operational_atContext_eq_draw originalFirstPhase continuation program]
  change mean initial
    (fun input => expectedAt (transition source) toll input.1.iteration (HyperNovaVisitedLaw.initialVisit input)) = _
  calc
    _ = mean initial (fun input => expectedAt (transition source) toll depth (HyperNovaVisitedLaw.initialVisit input)) := by
      unfold mean
      apply tsum_congr
      intro input
      by_cases absent : initial input = 0
      · simp only [absent, zero_mul]
      · apply congrArg (initial input * ·)
        exact (expectedAt_stable source toll zero input.1.iteration depth
          (HyperNovaVisitedLaw.initialVisit input) (by exact le_rfl)
          (depthBound input ((initial.mem_support_iff input).mpr absent))).symm
    _ = mean (visitedLaw source initial 0) (expectedAt (transition source) toll depth) := by
      rw [HyperNovaVisitedLaw.visitedLaw_zero, mean_map]
    _ = _ := by
      simpa only [Nat.zero_add] using expectedAt_mean_eq_visitedSum source toll initial depth 0

private theorem totalMean_nonnegative (visit : Visit) :
    0 ≤ StrongProbability.verifierMean
      (totalClock originalFirstPhase continuation program prefixClock sourceCheckClock accessClock visit) := by
  rw [← StrongProbability.verifierMean_const (shape := productionShape) 0]
  exact StrongProbability.verifierMean_mono _ _
    (totalClock_nonnegative originalFirstPhase continuation program prefixClock sourceCheckClock accessClock visit)

private theorem clockMean_nonnegative (contexts : PMF Visit) :
    0 ≤ StrongProbability.clockMean contexts
      (totalClock originalFirstPhase continuation program prefixClock sourceCheckClock accessClock) := by
  change 0 ≤ ∑' visit, (contexts visit).toReal * StrongProbability.verifierMean
    (totalClock originalFirstPhase continuation program prefixClock sourceCheckClock accessClock visit)
  exact tsum_nonneg fun visit => mul_nonneg ENNReal.toReal_nonneg
    (totalMean_nonnegative originalFirstPhase continuation program prefixClock sourceCheckClock accessClock visit)

private theorem callWork_mean_le (contexts : PMF Visit) (ceiling : ℝ)
    (summable : Summable fun visit => (contexts visit).toReal * StrongProbability.verifierMean
      (totalClock originalFirstPhase continuation program prefixClock sourceCheckClock accessClock visit))
    (bound : StrongProbability.clockMean contexts
      (totalClock originalFirstPhase continuation program prefixClock sourceCheckClock accessClock) ≤ ceiling) :
    mean contexts (fun visit => ENNReal.ofReal
      (callWork originalFirstPhase continuation program prefixClock sourceCheckClock accessClock visit)) ≤
      ENNReal.ofReal ceiling := by
  let value := fun visit => StrongProbability.verifierMean
    (totalClock originalFirstPhase continuation program prefixClock sourceCheckClock accessClock visit)
  have nonnegative : ∀ visit, 0 ≤ value visit :=
    totalMean_nonnegative originalFirstPhase continuation program prefixClock sourceCheckClock accessClock
  have fullMean : mean contexts (fun visit => ENNReal.ofReal (value visit)) =
      ENNReal.ofReal (StrongProbability.clockMean contexts
        (totalClock originalFirstPhase continuation program prefixClock sourceCheckClock accessClock)) := by
    change (∑' visit, contexts visit * ENNReal.ofReal (value visit)) =
      ENNReal.ofReal (∑' visit, (contexts visit).toReal * value visit)
    rw [ENNReal.ofReal_tsum_of_nonneg
      (fun visit => mul_nonneg ENNReal.toReal_nonneg (nonnegative visit)) summable]
    apply tsum_congr
    intro visit
    rw [ENNReal.ofReal_mul ENNReal.toReal_nonneg,
      ENNReal.ofReal_toReal (contexts.apply_ne_top visit)]
  calc
    _ ≤ mean contexts (fun visit => ENNReal.ofReal (value visit)) := by
      apply ENNReal.tsum_le_tsum
      intro visit
      apply mul_le_mul_right
      exact ENNReal.ofReal_le_ofReal
        (callWork_le_totalMean originalFirstPhase continuation program prefixClock sourceCheckClock accessClock visit)
    _ = _ := fullMean
    _ ≤ _ := ENNReal.ofReal_le_ofReal bound

/-- The existing InteractiveWork one-call polynomial, with the selected
carrier and the same declared primitive/access clocks. This is only a named
copy of that theorem's expression, not a new counting rule. -/
noncomputable def sourceWorkPolynomial (base primitive access : Polynomial ℝ) : Polynomial ℝ :=
  Polynomial.C ((PaperProfile.arity.total : ℝ) + 1) * base +
    Polynomial.C (PaperProfile.arity.total : ℝ) * (primitive + Polynomial.C 3) +
    Polynomial.C (productionShape.freshCount : ℝ) *
      (Polynomial.C (WitnessProjection.privateWidth PiCCSStoredWitnessCheck.carrier : ℝ) *
        (access + Polynomial.C 6) + Polynomial.C 9) +
    Polynomial.C (productionShape.runningCount : ℝ) *
      (Polynomial.C (PiCCSStoredWitnessCheck.carrier.carrierWidth : ℝ) *
        (access + Polynomial.C 6) + Polynomial.C 9) + Polynomial.C 13

/-- The selected fixed raw-call family has finite expected source work on the
actual history law under its unconditional visited base-moment bounds. The
existing D+1 orchestration allowance is added once. Inactive source visits
cost zero; active aborts and all false-mark calls retain their actual clocks.
The depth is symbolic and fixed independently of the security parameter.
No FS model, successful trace, source-law equality, or work-correctness
premise replaces a local implementation proof. -/
theorem expected_work_polynomial_bound
    (tapes : Visit → PublicCoins K productionShape →
      FullOutputCoordinates.FullOutput K productionShape → State → PMF Tape)
    (rawCall : Visit → PublicCoins K productionShape →
      FullOutputCoordinates.FullOutput K productionShape → State →
        PaperWeakOracle.Call (Tape := Tape) (arity := PaperProfile.arity) NifsExtractionProvider.rlc)
    (checkClock : Visit → PublicCoins K productionShape →
      FullOutputCoordinates.FullOutput K productionShape → State → NifsExtractionProvider.CheckClock)
    (storageClock : Visit → PublicCoins K productionShape →
      FullOutputCoordinates.FullOutput K productionShape → State → NifsExtractionProvider.StorageClock)
    (parentClock : Visit → PublicCoins K productionShape →
      FullOutputCoordinates.FullOutput K productionShape → State → NifsExtractionProvider.ParentClock)
    (storageBound : Visit → PublicCoins K productionShape →
      FullOutputCoordinates.FullOutput K productionShape → State → Nat)
    (storageBounded : ∀ visit coins output state assignments,
      storageClock visit coins output state assignments ≤ storageBound visit coins output state)
    (suffixSummable : ∀ visit coins output state vector, Summable fun tape =>
      (tapes visit coins output state tape).toReal *
        (PaperWeakOracle.baseWork NifsExtractionProvider.rlc
          (NifsExtractionProvider.suffixProgram (NifsExtractionProvider.batchAt inputs visit coins output)
            (checkClock visit coins output state) (storageClock visit coins output state))
          (rawCall visit coins output state) vector tape : ℝ))
    (initial : PMF (Statement × Envelope)) (depth : Nat)
    (depthBound : ∀ input ∈ initial.support, input.1.iteration ≤ depth)
    (scalarSubClock : RingF → RingF → Nat) (inverseAdapterClock : RingF → Nat)
    (assignmentSubClock : PiRLCExtractionPrimitives.Assignment → PiRLCExtractionPrimitives.Assignment → Nat)
    (scalarActionClock : RingF → PiRLCExtractionPrimitives.Assignment → Nat)
    (lowNorm : Phi81StrongSet.LowNormInvertibility)
    (bounds : PiRLC.PaperForkExtractionWork.PrimitiveBounds)
    (bounded : PiRLC.PaperForkExtractionWork.Bounded
      (PaperExtractionAlgebra.extractionAlgebra productionAjtaiKey).ring
      (PiRLCExtractionPrimitives.program scalarSubClock inverseAdapterClock
        assignmentSubClock scalarActionClock) bounds)
    (accessBound : Nat)
    (accessBounded : ∀ visit witness source column, accessClock visit witness source column ≤ accessBound)
    (securityParameter : Nat) (basePolynomial : Fin depth → Polynomial ℝ)
    (primitivePolynomial accessPolynomial : Polynomial ℝ)
    (primitivePPT : (bounds.coordinateWork : ℝ) ≤ primitivePolynomial.eval (securityParameter : ℝ))
    (accessPPT : (accessBound : ℝ) ≤ accessPolynomial.eval (securityParameter : ℝ)) :
    let continued := NifsProviderLaw.continuation inputs tapes rawCall checkClock storageClock parentClock
      storageBound storageBounded suffixSummable
    let primitives := PiRLCExtractionPrimitives.program scalarSubClock inverseAdapterClock
      assignmentSubClock scalarActionClock
    let source := HyperNovaGuardedSourceLaw.source originalFirstPhase continued primitives
    let visits := fun j : Fin depth => visitedLaw source initial j.val
    let base := InteractiveWork.baseClock relation productionAjtaiKey
      (fun visit => PiCCSInputCheck.running (inputs visit))
      (fun visit => PiCCSInputCheck.fresh (inputs visit))
      (operationalContinuation continued) (call originalFirstPhase prefixClock) primitives
      (sourceProgram sourceCheckClock accessClock)
    (∀ j : Fin depth, Summable fun visit => (visits j visit).toReal *
      StrongProbability.verifierMean (base visit)) →
    (∀ j : Fin depth, StrongProbability.clockMean (visits j) base ≤
      (basePolynomial j).eval (securityParameter : ℝ)) →
    expectedSourceWork originalFirstPhase continued primitives prefixClock sourceCheckClock accessClock initial ≠ ∞ ∧
    Summable (fun sample => ((HyperNovaHistoryLaw.law source initial) sample).toReal *
      (HyperNovaHistoryWork.controlAllowance sample.2.2 : ℝ)) ∧
    (expectedSourceWork originalFirstPhase continued primitives prefixClock sourceCheckClock accessClock initial).toReal +
        (∑' sample, ((HyperNovaHistoryLaw.law source initial) sample).toReal *
          (HyperNovaHistoryWork.controlAllowance sample.2.2 : ℝ)) ≤
      ((∑ j : Fin depth, sourceWorkPolynomial (basePolynomial j) primitivePolynomial accessPolynomial) +
        Polynomial.C ((depth + 1 : Nat) : ℝ)).eval (securityParameter : ℝ) := by
  dsimp only
  intro baseSummable basePPT
  let continued := NifsProviderLaw.continuation inputs tapes rawCall checkClock storageClock parentClock
    storageBound storageBounded suffixSummable
  let primitives := PiRLCExtractionPrimitives.program scalarSubClock inverseAdapterClock
    assignmentSubClock scalarActionClock
  let source := HyperNovaGuardedSourceLaw.source originalFirstPhase continued primitives
  let visits := fun j : Fin depth => visitedLaw source initial j.val
  let polynomial := fun j : Fin depth =>
    sourceWorkPolynomial (basePolynomial j) primitivePolynomial accessPolynomial
  have each (j : Fin depth) :
      mean (visits j) (fun visit => ENNReal.ofReal
        (callWork originalFirstPhase continued primitives prefixClock sourceCheckClock accessClock visit)) ≤
        ENNReal.ofReal ((polynomial j).eval (securityParameter : ℝ)) ∧
      0 ≤ (polynomial j).eval (securityParameter : ℝ) := by
    have checked := InteractiveWork.expected_work_polynomial_bound relation productionAjtaiKey
      (fun visit => PiCCSInputCheck.running (inputs visit))
      (fun visit => PiCCSInputCheck.fresh (inputs visit))
      (operationalContinuation continued) (call originalFirstPhase prefixClock) primitives
      (sourceProgram sourceCheckClock accessClock)
      (PaperExtractionAlgebra.extractionAlgebra productionAjtaiKey)
      (Phi81Relation.PiRLCAlgebra.ForkStrongSet.strongSetUnits lowNorm)
      (PiRLCExtractionPrimitives.program_correct scalarSubClock inverseAdapterClock
        assignmentSubClock scalarActionClock)
      bounds bounded accessBound
      (fun visit witness source column => accessBounded (callContext visit) witness source column)
      (visits j) (baseSummable j) securityParameter (basePolynomial j) primitivePolynomial accessPolynomial
      (basePPT j) primitivePPT accessPPT
    have clockBound : StrongProbability.clockMean (visits j)
        (totalClock originalFirstPhase continued primitives prefixClock sourceCheckClock accessClock) ≤
        (polynomial j).eval (securityParameter : ℝ) := by
      simpa only [totalClock, polynomial, sourceWorkPolynomial] using checked.2
    exact ⟨callWork_mean_le originalFirstPhase continued primitives prefixClock sourceCheckClock accessClock
        (visits j) _ checked.1 clockBound,
      (clockMean_nonnegative originalFirstPhase continued primitives prefixClock sourceCheckClock accessClock
        (visits j)).trans clockBound⟩
  have sourceBound :
      expectedSourceWork originalFirstPhase continued primitives prefixClock sourceCheckClock accessClock initial ≤
        ENNReal.ofReal (∑ j : Fin depth, (polynomial j).eval (securityParameter : ℝ)) := by
    rw [expectedSourceWork_eq_visitedSum originalFirstPhase continued primitives prefixClock
      sourceCheckClock accessClock initial depth depthBound]
    calc
      _ ≤ ∑ j : Fin depth, ENNReal.ofReal ((polynomial j).eval (securityParameter : ℝ)) :=
        Finset.sum_le_sum fun j _ => (each j).1
      _ = _ := (ENNReal.ofReal_sum_of_nonneg (fun j _ => (each j).2)).symm
  have finite := ne_top_of_le_ne_top ENNReal.ofReal_ne_top sourceBound
  have realSource := ENNReal.toReal_mono ENNReal.ofReal_ne_top sourceBound
  rw [ENNReal.toReal_ofReal (Finset.sum_nonneg fun j _ => (each j).2)] at realSource
  have control := HyperNovaHistoryWork.expected_control_allowance_le source initial depth depthBound
  refine ⟨finite, control.1, ?_⟩
  calc
    _ ≤ (∑ j : Fin depth, (polynomial j).eval (securityParameter : ℝ)) + ((depth + 1 : Nat) : ℝ) :=
      add_le_add realSource control.2
    _ = _ := by
      rw [Polynomial.eval_add, Polynomial.eval_finsetSum, Polynomial.eval_C]

end NightstreamFPrime.Export.Stage1.HyperNovaSourceWork
