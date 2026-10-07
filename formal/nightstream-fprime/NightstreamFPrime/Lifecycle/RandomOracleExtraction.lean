import NightstreamFPrime.Lifecycle.RandomOracleTest
import NightstreamFPrime.Lifecycle.PaperExtractionAlgebra
import NightstreamFPrime.Spec.Folding.Nifs.PaperStrongInterface
import NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.ForkStrongSet
import NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcSampler.Law
import NightstreamFPrime.Spec.Phi81StrongSet.LowNormInvertibility

/-!
Owns the `Π_RLC` extraction of the production NIFS key in the random-oracle
model (Lemma 5 of ROM_KNOWLEDGE_SOUNDNESS.md).

Inputs: an adaptive oracle adversary that outputs a claim (running and fresh
statements, a NIFS proof, and witnesses for the sixteen children), the
production key, and one retry answer per `Π_RLC` coordinate.

Outputs:
- `Accepts`: the key's NIFS verifier with every coin read from the oracle,
  and valid witnesses for the exact children it returns;
- `completeFork`: a base acceptance and one valid retry per coordinate form
  the paper's complete coordinate fork over the base batch, so
  `extracted_ambient` opens every `Π_CCS` output of the base probe;
- `fork_failure_le`: the retries fail to form that fork with probability at
  most `17 * (Q + 17) * ε_sample`, plus the chance that a retry changes the
  running statement (`mismatchChance`, a state-hash collision at the Export
  layer);
- `expected_retries_le`: at most `17 * (Q + 17)` expected retries.

A retry resamples the oracle answer at one `Π_RLC` point and reruns the
adversary until that point succeeds again (`RandomOracle.retryWeight`).
The adversary may query many candidate outputs; each queried point is
charged once (`RandomOracle.repeat_le`).

The fork algebra is the existing deterministic Appendix D.5 extractor
(`PaperForkExtraction.completeFork_implies_correctedAmbientHolds`).

Does not own: the `Π_CCS` test error (`RandomOracleTest`), uniqueness across
two extractions, or the binding reduction.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Lifecycle.RandomOracleExtraction

open scoped BigOperators
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.RandomOracle
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.TranscriptCoverage
open NightstreamFPrime.Lifecycle.RandomOracleTest
open StrongReduction ConcreteCarrier
open PiRLC.PaperForkExtraction

attribute [local instance low] Classical.propDecidable

/-! ## `Π_RLC` reads -/

/-- The `Π_RLC` challenge in an answer: the sampled strong-set scalar. -/
def readRho (answer : Answer) : RingF :=
  Phi81StrongSet.embedScalar (Spec.Folding.Nifs.NonInteractive.PiRlcSampler.sample answer)

theorem readRho_valid (answer : Answer) :
    Phi81Relation.PiRLCAlgebra.Challenge.challengeValid (readRho answer) :=
  ⟨_, rfl⟩

theorem readRho_eq_iff (left right : Answer) :
    readRho left = readRho right ↔
      Spec.Folding.Nifs.NonInteractive.PiRlcSampler.sample left =
        Spec.Folding.Nifs.NonInteractive.PiRlcSampler.sample right :=
  Phi81StrongSet.embedScalar_injective.eq_iff

/-- The mass of one sampler fiber: the uniform scalar mass plus the sampler's
statistical distance. -/
noncomputable def sampleError : ℝ :=
  1 / Spec.Folding.Nifs.NonInteractive.PiRlcSampler.scalarCount +
    Spec.Folding.Nifs.NonInteractive.PiRlcSampler.distance

theorem sampleError_nonnegative : 0 ≤ sampleError := by
  unfold sampleError
  have := Spec.Folding.Nifs.NonInteractive.PiRlcSampler.distance_nonnegative
  positivity

/-- Every sampler fiber has mass at most `sampleError`
(`PiRlcSampler.expect_difference_abs_le`). -/
theorem fiber_mass_le (scalar : Phi81StrongSet.Scalar) :
    mass {answer : Answer | Spec.Folding.Nifs.NonInteractive.PiRlcSampler.sample answer = scalar} ≤
      sampleError := by
  open Spec.Folding.Nifs.NonInteractive.PiRlcSampler in
  have card : Fintype.card Answer = drawCount := by
    rw [Fintype.card_fun, Fintype.card_fin, Fintype.card_fin]
    rfl
  let test : Phi81StrongSet.Scalar → ℝ := fun value => if value = scalar then 1 else 0
  have sampled : mass {answer : Answer | sample answer = scalar} = sampledExpect test := by
    unfold mass sampledExpect
    rw [Finset.expect_eq_sum_div_card, Finset.card_univ, card]
    exact congrArg (· / (drawCount : ℝ))
      (Finset.sum_congr rfl fun answer _ => by
        by_cases same : sample answer = scalar <;> simp [test, same])
  have uniform : uniformExpect test = 1 / scalarCount := by
    unfold uniformExpect
    simp [test]
  have bound := (abs_le.mp (expect_difference_abs_le test (fun _ => by
    simp only [test]
    split <;> norm_num) (fun _ => by
    simp only [test]
    split <;> norm_num))).2
  rw [sampled]
  unfold sampleError
  linarith

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth}

/-- The `Π_RLC` challenges of one execution under `oracle`. -/
noncomputable def rho {degree : Nat} (oracle : Point logicalWidth publicFits degree → Answer)
    (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
    (proof : Proof degree) : Fin (Nifs.PaperProfile.arity).total → RingF :=
  fun index => readRho (oracle (point fresh proof (.rho index)))

/-! ## Executions that absorb the same messages -/

/-- Two proofs with the same absorbed messages: the round polynomials and the
complete `y′`. They may differ only in the `Π_DEC` child messages. -/
def SameAbsorbed {degree : Nat} (left right : Proof degree) : Prop :=
  left.piCcsRounds = right.piCcsRounds ∧ left.piCcsOutput = right.piCcsOutput

theorem challengeCalls_congr {degree : Nat}
    (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
    {left right : Proof degree} (same : SameAbsorbed left right) (challenge : Challenge) :
    challengeCalls fresh left challenge = challengeCalls fresh right challenge := by
  have messagesEq : messages left = messages right := by
    funext round
    simp only [messages, same.1]
  cases challenge <;>
    simp only [challengeCalls, proverCalls, roundPrefixCalls, outputCalls, messagesEq, same.2]

theorem point_congr {degree : Nat}
    (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
    {left right : Proof degree} (same : SameAbsorbed left right) (challenge : Challenge) :
    point fresh left challenge = point fresh right challenge :=
  Subtype.ext (challengeCalls_congr fresh same challenge)

/-- Equal `Π_RLC` points identify the fresh statement and every absorbed
message (`TranscriptCoverage.proverCalls_identify`). -/
theorem rhoPoint_identifies {degree : Nat}
    {fresh fresh' : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits)}
    {proof proof' : Proof degree} (index : Fin (Nifs.PaperProfile.arity).total)
    (same : point fresh proof (.rho index) = point fresh' proof' (.rho index)) :
    fresh = fresh' ∧ SameAbsorbed proof proof' := by
  have agree := proverCalls_identify (.rho index)
    (List.append_cancel_right (congrArg Subtype.val same))
  refine ⟨agree.1, ?_, ?_⟩
  · rw [← agree.2]
  · rw [← agree.2]

variable (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
  (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))

local notation "Degree" => ProductionKey.degreeBound relation
local notation "Oracle" => Point logicalWidth publicFits (ProductionKey.degreeBound relation) → Answer

/-! ## The oracle verifier -/

/-- The `Π_RLC` input batch of the oracle probe: the key's `Π_CCS` outputs at
the oracle round point. -/
noncomputable def batch (oracle : Oracle)
    (running : Running (logicalWidth := logicalWidth) (publicFits := publicFits))
    (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
    (proof : Proof Degree) :=
  Spec.Folding.Nifs.PaperStrongInterface.piRlcBatchForProbe (ProductionKey.key relation ajtai)
    running fresh (oracleProbe relation ajtai running oracle fresh proof)

/-- The `Π_DEC` attempt over the parent that the oracle challenges select. -/
noncomputable def attempt (oracle : Oracle)
    (running : Running (logicalWidth := logicalWidth) (publicFits := publicFits))
    (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
    (proof : Proof Degree) :=
  (ProductionKey.key relation ajtai).piDecAttemptForParent proof
    (PiRLC.combinedOutput (ProductionKey.key relation ajtai).piRlcAlgebra
      (batch relation ajtai oracle running fresh proof).system
      (batch relation ajtai oracle running fresh proof).point
      (batch relation ajtai oracle running fresh proof).inputs (rho oracle fresh proof))

/-- The key's NIFS verifier with every coin read from `oracle` accepts, and
`children` opens each exact child it returns. -/
def Accepts (oracle : Oracle)
    (running : Running (logicalWidth := logicalWidth) (publicFits := publicFits))
    (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
    (proof : Proof Degree)
    (children : Fin productionShape.runningCount →
      PaperAlgebra.Assignment (logicalWidth := logicalWidth) (publicFits := publicFits)) : Prop :=
  (oracleProbe relation ajtai running oracle fresh proof).FixedWidthAccepted extensionOps K.embed
      ((ProductionKey.key relation ajtai).statement running fresh) Degree ∧
    PiDEC.PaperVerifier.Accepted (ProductionKey.key relation ajtai).piDecAlgebra
      (ProductionKey.key relation ajtai).piDecPublicInputSplit
      (ProductionKey.key relation ajtai).piDecEvaluationArity
      (attempt relation ajtai oracle running fresh proof) ∧
    ((ProductionKey.key relation ajtai).piDecPublicInputSplit.checked
      (attempt relation ajtai oracle running fresh proof).parent.publicInput).isSome ∧
    ∀ child, CE.Holds (ProductionKey.key relation ajtai).piRlcSemantics
      (ProductionKey.key relation ajtai).params
      (PiDEC.PaperVerifier.children (ProductionKey.key relation ajtai).piDecPublicInputSplit
        (attempt relation ajtai oracle running fresh proof) child)
      (children (Fin.cast (ProductionKey.key relation ajtai).outputCount_eq child))

/-- The response that an accepted execution supplies to `Π_RLC`: its
challenges and the recomposed `Π_DEC` parent witness. -/
noncomputable def response (oracle : Oracle)
    (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
    (proof : Proof Degree)
    (children : Fin productionShape.runningCount →
      PaperAlgebra.Assignment (logicalWidth := logicalWidth) (publicFits := publicFits)) :
    Response (PaperAlgebra.Assignment (logicalWidth := logicalWidth) (publicFits := publicFits))
      RingF (ProductionKey.key relation ajtai).params (ProductionKey.key relation ajtai).arity where
  challenges := rho oracle fresh proof
  assignment := (ProductionKey.key relation ajtai).piDecAlgebra.recomposeAssignment fun child =>
    children (Fin.cast (ProductionKey.key relation ajtai).outputCount_eq child)

/-- An accepted execution's response opens its own `Π_RLC` parent
(`Π_DEC` reduction of knowledge). -/
theorem response_success (oracle : Oracle)
    (running : Running (logicalWidth := logicalWidth) (publicFits := publicFits))
    (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
    (proof : Proof Degree)
    (children : Fin productionShape.runningCount →
      PaperAlgebra.Assignment (logicalWidth := logicalWidth) (publicFits := publicFits))
    (accepts : Accepts relation ajtai oracle running fresh proof children) :
    (response relation ajtai oracle fresh proof children).Success
      (ProductionKey.key relation ajtai).piRlcSemantics (ProductionKey.key relation ajtai).params
      (ProductionKey.key relation ajtai).piRlcAlgebra
      (batch relation ajtai oracle running fresh proof) :=
  PiDEC.PaperVerifier.reduce_knowledge (ProductionKey.key relation ajtai).piRlcSemantics
    (ProductionKey.key relation ajtai).params (ProductionKey.key relation ajtai).piDecAlgebra
    (ProductionKey.key relation ajtai).piDecPublicInputSplit
    (ProductionKey.key relation ajtai).piDecEvaluationArity
    (attempt relation ajtai oracle running fresh proof) _
    (ProductionKey.key relation ajtai).kPositive accepts.2.1 accepts.2.2.2

/-! ## One changed `Π_RLC` answer -/

omit relation ajtai in
/-- A read at another challenge of the same execution ignores the changed
answer. -/
theorem update_ne {degree : Nat} (oracle : Point logicalWidth publicFits degree → Answer)
    (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
    (proof : Proof degree) {changed other : Challenge} (different : other ≠ changed)
    (answer : Answer) :
    Function.update oracle (point fresh proof changed) answer (point fresh proof other) =
      oracle (point fresh proof other) :=
  Function.update_of_ne (fun same => different
    (challengeCalls_injective fresh proof (congrArg Subtype.val same))) _ _

omit relation ajtai in
/-- The `Π_CCS` coins ignore every `Π_RLC` answer. -/
theorem coins_update_rho {degree : Nat} (oracle : Point logicalWidth publicFits degree → Answer)
    (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
    (proof : Proof degree) (index : Fin (Nifs.PaperProfile.arity).total) (answer : Answer) :
    coins (Function.update oracle (point fresh proof (.rho index)) answer) fresh proof =
      coins oracle fresh proof := by
  have alpha (coordinate : Fin productionShape.cubeVariables) :
      Function.update oracle (point fresh proof (.rho index)) answer
          (point fresh proof (.alpha coordinate)) = oracle (point fresh proof (.alpha coordinate)) :=
    update_ne oracle fresh proof (by simp) answer
  have gamma : Function.update oracle (point fresh proof (.rho index)) answer
      (point fresh proof .gamma) = oracle (point fresh proof .gamma) :=
    update_ne oracle fresh proof (by simp) answer
  have round (round : Fin productionShape.cubeVariables) :
      Function.update oracle (point fresh proof (.rho index)) answer
          (point fresh proof (.round round)) = oracle (point fresh proof (.round round)) :=
    update_ne oracle fresh proof (by simp) answer
  simp only [coins, coinsFrom, read_challengeCalls, alpha, gamma, round]

omit relation ajtai in
theorem rho_update {degree : Nat} (oracle : Point logicalWidth publicFits degree → Answer)
    (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
    (proof : Proof degree) (index other : Fin (Nifs.PaperProfile.arity).total) (answer : Answer) :
    rho (Function.update oracle (point fresh proof (.rho index)) answer) fresh proof other =
      if other = index then readRho answer else rho oracle fresh proof other := by
  by_cases same : other = index
  · subst same
    simp [rho]
  · rw [if_neg same]
    exact congrArg readRho (update_ne oracle fresh proof (by simpa using same) answer)

omit relation ajtai in
theorem coins_congr {degree : Nat} (oracle : Point logicalWidth publicFits degree → Answer)
    (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
    {left right : Proof degree} (same : SameAbsorbed left right) :
    coins oracle fresh left = coins oracle fresh right := by
  simp only [coins, coinsFrom, challengeCalls_congr fresh same]

omit relation ajtai in
theorem rho_congr {degree : Nat} (oracle : Point logicalWidth publicFits degree → Answer)
    (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
    {left right : Proof degree} (same : SameAbsorbed left right) :
    rho oracle fresh left = rho oracle fresh right := by
  funext index
  simp only [rho, point_congr fresh same]

theorem oracleProbe_congr (oracle : Oracle)
    (running : Running (logicalWidth := logicalWidth) (publicFits := publicFits))
    (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
    {left right : Proof Degree} (same : SameAbsorbed left right) :
    oracleProbe relation ajtai running oracle fresh left =
      oracleProbe relation ajtai running oracle fresh right := by
  simp only [oracleProbe, coins_congr oracle fresh same,
    Spec.Folding.Nifs.PaperNonInteractive.Key.piCcsProbe,
    Spec.Folding.Nifs.PaperNonInteractive.Key.piCcsCertificate, same.1, same.2]

/-- A retry that keeps the absorbed messages sees the base batch: the
`Π_CCS` probe ignores the changed `Π_RLC` answer. -/
theorem batch_update (oracle : Oracle)
    (running : Running (logicalWidth := logicalWidth) (publicFits := publicFits))
    (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
    {left right : Proof Degree} (same : SameAbsorbed left right)
    (index : Fin (Nifs.PaperProfile.arity).total) (answer : Answer) :
    batch relation ajtai (Function.update oracle (point fresh left (.rho index)) answer)
        running fresh right =
      batch relation ajtai oracle running fresh left := by
  have probe : oracleProbe relation ajtai running
      (Function.update oracle (point fresh left (.rho index)) answer) fresh right =
        oracleProbe relation ajtai running oracle fresh left := by
    rw [← oracleProbe_congr relation ajtai _ running fresh same]
    simp only [oracleProbe, coins_update_rho]
  simp only [batch, probe]

/-! ## The adversary's coordinate forks -/

/-- What the adversary outputs: both statements, a NIFS proof, and witnesses
for the children it claims. -/
structure Claim where
  running : Running (logicalWidth := logicalWidth) (publicFits := publicFits)
  fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits)
  proof : Proof (ProductionKey.degreeBound relation)
  children : Fin productionShape.runningCount →
    PaperAlgebra.Assignment (logicalWidth := logicalWidth) (publicFits := publicFits)

variable {Output : Type}
  (adversary : OracleComp (Point logicalWidth publicFits (ProductionKey.degreeBound relation))
    Answer Output)
  (claim : Output → Claim relation)

/-- The claim the adversary outputs under `oracle`. -/
noncomputable def claimed (oracle : Oracle) : Claim relation :=
  claim (adversary.run oracle)

/-- The oracle verifier accepts the adversary's claim. -/
def Succeeds (oracle : Oracle) : Prop :=
  Accepts relation ajtai oracle (claimed relation adversary claim oracle).running
    (claimed relation adversary claim oracle).fresh (claimed relation adversary claim oracle).proof
    (claimed relation adversary claim oracle).children

/-- The oracle point of the claim's `index`-th `Π_RLC` challenge. -/
noncomputable def rhoPoint (index : Fin (Nifs.PaperProfile.arity).total) (oracle : Oracle) :
    Point logicalWidth publicFits Degree :=
  point (claimed relation adversary claim oracle).fresh (claimed relation adversary claim oracle).proof
    (.rho index)

/-- Coordinate `index` succeeds at `target`. -/
def Hits (index : Fin (Nifs.PaperProfile.arity).total) (target : Point logicalWidth publicFits Degree)
    (oracle : Oracle) : Prop :=
  Succeeds relation ajtai adversary claim oracle ∧ rhoPoint relation adversary claim index oracle = target

/-- The adversary followed by the verifier's `Π_RLC` queries. -/
noncomputable def normalized :
    OracleComp (Point logicalWidth publicFits Degree) Answer Output :=
  adversary.thenQueries fun output =>
    List.ofFn fun index => point (claim output).fresh (claim output).proof (.rho index)

theorem normalized_bound {queries : Nat} (bounded : adversary.QueryBound queries) :
    (normalized relation adversary claim).QueryBound (queries + (Nifs.PaperProfile.arity).total) :=
  bounded.thenQueries fun _ => by simp

/-- A coordinate succeeds only at a point the normalized adversary queries. -/
theorem hits_queried (index : Fin (Nifs.PaperProfile.arity).total)
    (target : Point logicalWidth publicFits Degree) (oracle : Oracle)
    (hits : Hits relation ajtai adversary claim index target oracle) :
    target ∈ (normalized relation adversary claim).queries oracle := by
  apply OracleComp.mem_queries_thenQueries
  rw [← hits.2]
  exact List.mem_ofFn.mpr ⟨index, rfl⟩

/-- The oracle a retry at coordinate `index` runs: the base oracle with that
coordinate's answer resampled. -/
noncomputable def forked (oracle : Oracle) (retries : Fin (Nifs.PaperProfile.arity).total → Answer)
    (index : Fin (Nifs.PaperProfile.arity).total) : Oracle :=
  Function.update oracle (rhoPoint relation adversary claim index oracle) (retries index)

/-- A retry forms a fork: it succeeds at the same point, with a different
`Π_RLC` scalar, and keeps the running statement. -/
def ForkValid (oracle : Oracle) (retries : Fin (Nifs.PaperProfile.arity).total → Answer)
    (index : Fin (Nifs.PaperProfile.arity).total) : Prop :=
  retries index ∈ lineSet (Hits relation ajtai adversary claim index)
      (rhoPoint relation adversary claim index oracle) oracle ∧
    Spec.Folding.Nifs.NonInteractive.PiRlcSampler.sample (retries index) ≠
      Spec.Folding.Nifs.NonInteractive.PiRlcSampler.sample
        (oracle (rhoPoint relation adversary claim index oracle)) ∧
    (claimed relation adversary claim (forked relation adversary claim oracle retries index)).running =
      (claimed relation adversary claim oracle).running

/-- A valid fork outputs the base fresh statement and the base absorbed
messages. -/
theorem fork_identifies {oracle : Oracle} {retries : Fin (Nifs.PaperProfile.arity).total → Answer}
    {index : Fin (Nifs.PaperProfile.arity).total}
    (valid : ForkValid relation ajtai adversary claim oracle retries index) :
    (claimed relation adversary claim oracle).fresh =
        (claimed relation adversary claim (forked relation adversary claim oracle retries index)).fresh ∧
      SameAbsorbed (claimed relation adversary claim oracle).proof
        (claimed relation adversary claim (forked relation adversary claim oracle retries index)).proof :=
  rhoPoint_identifies index valid.1.2.symm

/-- The fork's response opens the base batch. -/
theorem fork_success {oracle : Oracle} {retries : Fin (Nifs.PaperProfile.arity).total → Answer}
    {index : Fin (Nifs.PaperProfile.arity).total}
    (valid : ForkValid relation ajtai adversary claim oracle retries index) :
    let base := claimed relation adversary claim oracle
    let fork := claimed relation adversary claim (forked relation adversary claim oracle retries index)
    (response relation ajtai (forked relation adversary claim oracle retries index) fork.fresh fork.proof
        fork.children).Success
      (ProductionKey.key relation ajtai).piRlcSemantics (ProductionKey.key relation ajtai).params
      (ProductionKey.key relation ajtai).piRlcAlgebra
      (batch relation ajtai oracle base.running base.fresh base.proof) := by
  intro base fork
  have success : (response relation ajtai (forked relation adversary claim oracle retries index)
      fork.fresh fork.proof fork.children).Success
        (ProductionKey.key relation ajtai).piRlcSemantics (ProductionKey.key relation ajtai).params
        (ProductionKey.key relation ajtai).piRlcAlgebra
        (batch relation ajtai (forked relation adversary claim oracle retries index)
          fork.running fork.fresh fork.proof) :=
    response_success relation ajtai _ fork.running fork.fresh fork.proof fork.children valid.1.1
  obtain ⟨freshEq, same⟩ := fork_identifies relation ajtai adversary claim valid
  have batchEq : batch relation ajtai (forked relation adversary claim oracle retries index)
      fork.running fork.fresh fork.proof = batch relation ajtai oracle base.running base.fresh base.proof := by
    have runningEq : fork.running = base.running := valid.2.2
    rw [runningEq, ← freshEq]
    exact batch_update relation ajtai oracle base.running base.fresh same index (retries index)
  rw [batchEq] at success
  exact success

/-- The fork's challenges: the base challenges with coordinate `index`
replaced by the retry's scalar. -/
theorem fork_rho {oracle : Oracle} {retries : Fin (Nifs.PaperProfile.arity).total → Answer}
    {index : Fin (Nifs.PaperProfile.arity).total}
    (valid : ForkValid relation ajtai adversary claim oracle retries index)
    (other : Fin (Nifs.PaperProfile.arity).total) :
    let base := claimed relation adversary claim oracle
    let fork := claimed relation adversary claim (forked relation adversary claim oracle retries index)
    rho (forked relation adversary claim oracle retries index) fork.fresh fork.proof other =
      if other = index then readRho (retries index) else rho oracle base.fresh base.proof other := by
  intro base fork
  obtain ⟨freshEq, same⟩ := fork_identifies relation ajtai adversary claim valid
  rw [← freshEq, ← rho_congr _ base.fresh same]
  exact rho_update oracle base.fresh base.proof index other (retries index)

/-- One base acceptance and a valid fork at every coordinate form the paper's
complete coordinate fork over the base batch. -/
noncomputable def completeFork (oracle : Oracle)
    (retries : Fin (Nifs.PaperProfile.arity).total → Answer)
    (succeeds : Succeeds relation ajtai adversary claim oracle)
    (valid : ∀ index, ForkValid relation ajtai adversary claim oracle retries index) :
    CompleteFork (ProductionKey.key relation ajtai).piRlcSemantics
      (ProductionKey.key relation ajtai).params (ProductionKey.key relation ajtai).piRlcAlgebra
      (batch relation ajtai oracle (claimed relation adversary claim oracle).running
        (claimed relation adversary claim oracle).fresh (claimed relation adversary claim oracle).proof) where
  base := response relation ajtai oracle (claimed relation adversary claim oracle).fresh
    (claimed relation adversary claim oracle).proof (claimed relation adversary claim oracle).children
  forks index := response relation ajtai (forked relation adversary claim oracle retries index)
    (claimed relation adversary claim (forked relation adversary claim oracle retries index)).fresh
    (claimed relation adversary claim (forked relation adversary claim oracle retries index)).proof
    (claimed relation adversary claim (forked relation adversary claim oracle retries index)).children
  baseSuccess := response_success relation ajtai oracle _ _ _ _ succeeds
  forkSuccess index := fork_success relation ajtai adversary claim (valid index)
  baseStrong _ := readRho_valid _
  forkStrong _ _ := readRho_valid _
  agreeExcept index other different := by
    change rho oracle _ _ other = rho (forked relation adversary claim oracle retries index) _ _ other
    rw [fork_rho relation ajtai adversary claim (valid index) other]
    split_ifs with same
    · exact absurd same different
    · rfl
  changed index := by
    change rho oracle _ _ index ≠ rho (forked relation adversary claim oracle retries index) _ _ index
    rw [fork_rho relation ajtai adversary claim (valid index) index]
    simp only [if_true]
    exact fun same => (valid index).2.1 ((readRho_eq_iff _ _).mp same).symm

/-- The `Π_CCS` output witness that the complete fork extracts. -/
noncomputable def extractedWitness (oracle : Oracle)
    (retries : Fin (Nifs.PaperProfile.arity).total → Answer)
    (succeeds : Succeeds relation ajtai adversary claim oracle)
    (valid : ∀ index, ForkValid relation ajtai adversary claim oracle retries index) :
    OutputWitness productionShape (Phi81CarrierLayout.carrierWidth logicalWidth) :=
  Spec.Folding.Nifs.PaperStrongInterface.outputWitnessOfAssignments (ProductionKey.key relation ajtai)
    fun coordinate => extractedAssignment (PaperExtractionAlgebra.extractionAlgebra ajtai)
      (Spec.Phi81Relation.PiRLCAlgebra.ForkStrongSet.strongSetUnits
        Spec.Phi81StrongSet.lowNormInvertibility)
      (completeFork relation ajtai adversary claim oracle retries succeeds valid) coordinate

/-- The extracted witness opens every `Π_CCS` output of the base oracle
probe, as Lemma 4 requires. -/
theorem extracted_ambient (oracle : Oracle)
    (retries : Fin (Nifs.PaperProfile.arity).total → Answer)
    (succeeds : Succeeds relation ajtai adversary claim oracle)
    (valid : ∀ index, ForkValid relation ajtai adversary claim oracle retries index) :
    let base := claimed relation adversary claim oracle
    AmbientOutputHolds extensionOps K.embed (PaperAlgebra.openingMaps ajtai) productionGlobalParams
      ((ProductionKey.key relation ajtai).statement base.running base.fresh)
      (oracleProbe relation ajtai base.running oracle base.fresh base.proof)
      (extractedWitness relation ajtai adversary claim oracle retries succeeds valid) := by
  intro base
  exact Spec.Folding.Nifs.PaperStrongInterface.outputWitnessOfAssignments_ambient
    (ProductionKey.key relation ajtai) base.running base.fresh _ _
    (completeFork_implies_correctedAmbientHolds (ProductionKey.key relation ajtai).piRlcSemantics
      (ProductionKey.key relation ajtai).params (ProductionKey.key relation ajtai).arity
      (ProductionKey.key relation ajtai).piRlcAlgebra (PaperExtractionAlgebra.extractionAlgebra ajtai)
      (Spec.Phi81Relation.PiRLCAlgebra.ForkStrongSet.strongSetUnits
        Spec.Phi81StrongSet.lowNormInvertibility)
      _ (completeFork relation ajtai adversary claim oracle retries succeeds valid))

/-! ## Loss and work -/

/-- The retry law's sets: coordinate `index` retries on its line. -/
def retrySets (oracle : Oracle) (index : Fin (Nifs.PaperProfile.arity).total) : Set Answer :=
  lineSet (Hits relation ajtai adversary claim index) (rhoPoint relation adversary claim index oracle) oracle

/-- The chance that a retry at coordinate `index` changes the running
statement. With the prior link this is a state-hash collision. -/
noncomputable def mismatchChance (index : Fin (Nifs.PaperProfile.arity).total) (oracle : Oracle) : ℝ :=
  mass (retrySets relation ajtai adversary claim oracle index ∩
      {answer | (claimed relation adversary claim
        (Function.update oracle (rhoPoint relation adversary claim index oracle) answer)).running ≠
          (claimed relation adversary claim oracle).running}) /
    mass (retrySets relation ajtai adversary claim oracle index)

/-- Retry answers at `index` that do not form a valid fork. -/
def badRetries (oracle : Oracle) (index : Fin (Nifs.PaperProfile.arity).total) : Set Answer :=
  {answer | ¬ ForkValid relation ajtai adversary claim oracle (fun _ => answer) index}

theorem forkValid_iff (oracle : Oracle) (retries : Fin (Nifs.PaperProfile.arity).total → Answer)
    (index : Fin (Nifs.PaperProfile.arity).total) :
    ForkValid relation ajtai adversary claim oracle retries index ↔
      retries index ∉ badRetries relation ajtai adversary claim oracle index := by
  simp [badRetries, ForkValid, forked]

private theorem mass_union_le (left right : Set Answer) :
    mass (left ∪ right) ≤ mass left + mass right := by
  unfold mass
  rw [← Finset.expect_add_distrib]
  exact Finset.expect_le_expect fun answer _ => by
    by_cases inLeft : answer ∈ left <;> by_cases inRight : answer ∈ right <;>
      simp [inLeft, inRight]

theorem mass_ne_zero {set : Set Answer} {answer : Answer} (inside : answer ∈ set) :
    mass set ≠ 0 := by
  unfold mass
  apply ne_of_gt
  rw [Finset.expect_eq_sum_div_card]
  apply div_pos _ (by exact_mod_cast Finset.card_pos.mpr Finset.univ_nonempty)
  calc
    (0 : ℝ) < if answer ∈ set then 1 else 0 := by simp [inside]
    _ ≤ _ := Finset.single_le_sum (f := fun answer => if answer ∈ set then (1 : ℝ) else 0)
      (fun _ _ => by split <;> norm_num) (Finset.mem_univ answer)

/-- A successful base answers inside its own line. -/
theorem base_mem_retrySets {oracle : Oracle} (succeeds : Succeeds relation ajtai adversary claim oracle)
    (index : Fin (Nifs.PaperProfile.arity).total) :
    oracle (rhoPoint relation adversary claim index oracle) ∈
      retrySets relation ajtai adversary claim oracle index := by
  change Hits relation ajtai adversary claim index _ (Function.update oracle _ _)
  rw [Function.update_eq_self]
  exact ⟨succeeds, rfl⟩

/-- Sum over every point of the term at the claim's own `Π_RLC` point. -/
private theorem sum_hits (index : Fin (Nifs.PaperProfile.arity).total)
    (term : Point logicalWidth publicFits Degree → Oracle → ℝ) (oracle : Oracle) :
    (∑ target, if Hits relation ajtai adversary claim index target oracle then term target oracle else 0) =
      if Succeeds relation ajtai adversary claim oracle then
        term (rhoPoint relation adversary claim index oracle) oracle else 0 := by
  by_cases succeeds : Succeeds relation ajtai adversary claim oracle
  · simp only [Hits, succeeds, true_and, if_true]
    exact (Finset.sum_ite_eq Finset.univ _ _).trans (if_pos (Finset.mem_univ _))
  · simp [Hits, succeeds]

/-- Lemma 5 loss: the retries fail to form a complete fork with probability
at most `17 * (Q + 17) * sampleError`, plus the running-statement mismatch
chance. -/
theorem fork_failure_le {queries : Nat} (bounded : adversary.QueryBound queries) :
    𝔼 oracle, (if Succeeds relation ajtai adversary claim oracle then
        ∑ retries, retryWeight (retrySets relation ajtai adversary claim oracle) retries *
          (if ∀ index, ForkValid relation ajtai adversary claim oracle retries index then 0 else 1)
      else 0) ≤
      (Nifs.PaperProfile.arity).total *
          (((queries + (Nifs.PaperProfile.arity).total : Nat) : ℝ) * sampleError) +
        𝔼 oracle, (if Succeeds relation ajtai adversary claim oracle then
          ∑ index, mismatchChance relation ajtai adversary claim index oracle else 0) := by
  have pointwise (oracle : Oracle) :
      (if Succeeds relation ajtai adversary claim oracle then
        ∑ retries, retryWeight (retrySets relation ajtai adversary claim oracle) retries *
          (if ∀ index, ForkValid relation ajtai adversary claim oracle retries index then 0 else 1)
        else 0) ≤
      (∑ index, if Succeeds relation ajtai adversary claim oracle then
          repeatChance (Hits relation ajtai adversary claim index)
            Spec.Folding.Nifs.NonInteractive.PiRlcSampler.sample
            (rhoPoint relation adversary claim index oracle) oracle else 0) +
        (if Succeeds relation ajtai adversary claim oracle then
          ∑ index, mismatchChance relation ajtai adversary claim index oracle else 0) := by
    by_cases succeeds : Succeeds relation ajtai adversary claim oracle
    · simp only [if_pos succeeds]
      have failure (retries : Fin (Nifs.PaperProfile.arity).total → Answer) :
          (if ∀ index, ForkValid relation ajtai adversary claim oracle retries index then (0 : ℝ) else 1) ≤
            ∑ index, if retries index ∈ badRetries relation ajtai adversary claim oracle index
              then 1 else 0 := by
        by_cases every : ∀ index, ForkValid relation ajtai adversary claim oracle retries index
        · rw [if_pos every]
          exact Finset.sum_nonneg fun _ _ => by split <;> norm_num
        · rw [if_neg every]
          obtain ⟨index, invalid⟩ := not_forall.mp every
          have inside : retries index ∈ badRetries relation ajtai adversary claim oracle index :=
            by_contra fun outside =>
              invalid ((forkValid_iff relation ajtai adversary claim oracle retries index).mpr outside)
          have single := Finset.single_le_sum
            (f := fun index => if retries index ∈ badRetries relation ajtai adversary claim oracle index
              then (1 : ℝ) else 0) (fun _ _ => by split <;> norm_num) (Finset.mem_univ index)
          simpa [inside] using single
      refine (Finset.sum_le_sum fun retries _ => mul_le_mul_of_nonneg_left (failure retries)
        (retryWeight_nonnegative _ retries)).trans ?_
      rw [expected_bad_retries _ (fun index => mass_ne_zero (base_mem_retrySets relation ajtai
        adversary claim succeeds index)) (badRetries relation ajtai adversary claim oracle)]
      rw [← Finset.sum_add_distrib]
      refine Finset.sum_le_sum fun index _ => ?_
      rw [repeatChance, mismatchChance]
      unfold retrySets
      rw [← add_div]
      apply div_le_div_of_nonneg_right _ (mass_nonnegative _)
      refine (mass_mono ?_).trans (mass_union_le _ _)
      rintro answer ⟨inside, invalid⟩
      simp only [badRetries, Set.mem_setOf_eq, ForkValid, forked, not_and_or, not_not] at invalid
      rcases invalid with outside | repeated | mismatch
      · exact absurd inside outside
      · exact Or.inl ⟨inside, repeated⟩
      · exact Or.inr ⟨inside, mismatch⟩
    · simp [succeeds]
  calc
    _ ≤ 𝔼 oracle, ((∑ index, if Succeeds relation ajtai adversary claim oracle then
            repeatChance (Hits relation ajtai adversary claim index)
              Spec.Folding.Nifs.NonInteractive.PiRlcSampler.sample
              (rhoPoint relation adversary claim index oracle) oracle else 0) +
          (if Succeeds relation ajtai adversary claim oracle then
            ∑ index, mismatchChance relation ajtai adversary claim index oracle else 0)) :=
      Finset.expect_le_expect fun oracle _ => pointwise oracle
    _ = (∑ index, 𝔼 oracle, ∑ target,
            if Hits relation ajtai adversary claim index target oracle then
              repeatChance (Hits relation ajtai adversary claim index)
                Spec.Folding.Nifs.NonInteractive.PiRlcSampler.sample target oracle else 0) +
          𝔼 oracle, (if Succeeds relation ajtai adversary claim oracle then
            ∑ index, mismatchChance relation ajtai adversary claim index oracle else 0) := by
      rw [Finset.expect_add_distrib, Finset.expect_sum_comm]
      simp only [sum_hits]
    _ ≤ ∑ _index : Fin (Nifs.PaperProfile.arity).total,
            ((queries + (Nifs.PaperProfile.arity).total : Nat) : ℝ) * sampleError +
          𝔼 oracle, (if Succeeds relation ajtai adversary claim oracle then
            ∑ index, mismatchChance relation ajtai adversary claim index oracle else 0) := by
      refine add_le_add (Finset.sum_le_sum fun index _ => ?_) le_rfl
      exact repeat_le (Hits relation ajtai adversary claim index)
        Spec.Folding.Nifs.NonInteractive.PiRlcSampler.sample sampleError fiber_mass_le
        sampleError_nonnegative (normalized_bound relation adversary claim bounded)
        (hits_queried relation ajtai adversary claim index)
    _ = _ := by
      rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]

/-- Lemma 5 work: at most `17 * (Q + 17)` expected retries. -/
theorem expected_retries_le {queries : Nat} (bounded : adversary.QueryBound queries) :
    𝔼 oracle, (if Succeeds relation ajtai adversary claim oracle then
        ∑ index, retryCount (Hits relation ajtai adversary claim index)
          (rhoPoint relation adversary claim index oracle) oracle else 0) ≤
      (Nifs.PaperProfile.arity).total * ((queries + (Nifs.PaperProfile.arity).total : Nat) : ℝ) := by
  calc
    _ = ∑ index, 𝔼 oracle, ∑ target,
          if Hits relation ajtai adversary claim index target oracle then
            retryCount (Hits relation ajtai adversary claim index) target oracle else 0 := by
      rw [← Finset.expect_sum_comm]
      refine Finset.expect_congr rfl fun oracle _ => ?_
      simp only [sum_hits]
      split <;> simp
    _ ≤ ∑ _index : Fin (Nifs.PaperProfile.arity).total,
          ((queries + (Nifs.PaperProfile.arity).total : Nat) : ℝ) :=
      Finset.sum_le_sum fun index _ => retries_le (Hits relation ajtai adversary claim index)
        (normalized_bound relation adversary claim bounded)
        (hits_queried relation ajtai adversary claim index)
    _ = _ := by
      rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]

end NightstreamFPrime.Lifecycle.RandomOracleExtraction
