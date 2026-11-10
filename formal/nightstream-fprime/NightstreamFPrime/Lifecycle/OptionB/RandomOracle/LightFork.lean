import NightstreamFPrime.Lifecycle.OptionB.RandomOracle.ForkLink
import NightstreamFPrime.Lifecycle.OptionB.RandomOracle.LightTest
import NightstreamFPrime.Spec.Folding.PiRLC.NormFreeFork
import NightstreamFPrime.Spec.Phi81StrongSet.LowNormInvertibility

/-! The light fold's fork step in the random-oracle model: `Fork` on the light schedule, which
has one `ρ` challenge. `verify` is `Light.verify` with the Π_CCS coins and `ρ` read from the
oracle (`LightTest.oracleProbe`, `challenge`), and `folds` is `slotFolds` at a light step with
`verify` in its place. An adversary outputs an option-B attempt; the attempt succeeds in the
light game (`game`) when its fold has the game's circuit key and `ForkLink.Succeeds` holds with
`folds`: the shape of the walk's success event with oracle coins in place of the sponge.

A successful attempt passes the light Π_CCS check on the oracle coins, and its witness opens the
light Π_RLC output at the attempt's bound (`succeeds_holds`). A valid retry has the base run's
fresh claim, start slot and proof, and the oracle differs only at the `ρ` point
(`retry_holds`). So the base and retry runs give a light Π_RLC fork without norms on the base
run's inputs (`fork`, a `Light.Rlc.Fork` of `normFreeAlgebra`: the input of SuperNeo v1.2
Lemma 8), whose base witness opens at the base bound (`fork_baseNorm`). The fork extracts the fresh and running
inputs at the ambient bound (`extracted_ambient`); with a fresh witness of norm below `b`, the
running witness opens at the base bound plus `T · (b − 1)` (`extracted_running_normBounded`,
Cyclo Theorem 3, item 2).

With `rhoCount = 1`: an invalid retry has probability at most `(Q + 1) · sampleError` plus the
prior-state collision chance (`fork_failure_le`), and the expected number of reruns is at most
`Q + 1` (`expected_retries_le`). -/

set_option autoImplicit false

namespace NightstreamFPrime.Lifecycle.OptionB.RandomOracle.LightFork

open scoped BigOperators
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.RandomOracle
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.StrongReduction
open NightstreamFPrime.Spec.HyperNova.Construction2
open NightstreamFPrime.Spec.HyperNova.Construction2.OptionBFolds
  (toLightRunning fromLight freshToLight evaluationsToLight)
open NightstreamFPrime.Spec.Folding.PiRLC.NormFreeFork (normFree normFreeAlgebra normFreeLaws)
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.OptionB
open NightstreamFPrime.Lifecycle.RandomOracle
open NightstreamFPrime.Lifecycle.RandomOracle.Test (Answer Point point)
open NightstreamFPrime.Lifecycle.OptionB.RandomOracle.ForkLink (Attempt Folds)

attribute [local instance low] Classical.propDecidable

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth}

section Types

variable (logicalWidth publicFits)

abbrev Oracle := Point (LightSchedule.schedule logicalWidth publicFits) 8 → Answer

abbrev Running :=
  Nifs.PaperNonInteractive.Running K PaperAlgebra.Commitment
    (PaperAlgebra.PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits)) LightShape

abbrev Statement := (LightSchedule.schedule logicalWidth publicFits).Statement

abbrev Claim :=
  CE.Instance
    (RelationSource LightShape (Phi81CarrierLayout.carrierWidth logicalWidth)
      (Phi81ColumnLayout.blockCount (Phi81CarrierLayout.carrierWidth logicalWidth)))
    (PaperAlgebra.PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits))
    (CubePoint K LightShape.cubeVariables) (EvaluationFamily K LightShape) PaperAlgebra.Commitment

end Types

/-- The light schedule's one `ρ` coordinate. -/
def first : Fin (LightSchedule.schedule logicalWidth publicFits).rhoCount :=
  ⟨0, Nat.zero_lt_one⟩

/-- The light proof that the light game reads from a refresh proof. A refresh proof never
passes the light game's fold verifiers, so the value does not matter. -/
def placeholder : Light.Proof LightShape 8 :=
  ⟨fun _ => ⟨List.replicate (8 + 1) K.zero, by simp⟩,
    ⟨fun _ _ => K.zero, fun _ _ _ => K.zero⟩⟩

def proofOf : StepProof → Light.Proof LightShape 8
  | .light proof => proof
  | .refresh _ => placeholder

/-- A slot claim as the light fold's running claim (`OptionBFolds.toLightRunning`). -/
def runningOf (claim : SlotClaim logicalWidth publicFits) : Running logicalWidth publicFits where
  point := claim.point
  commitments := fun _ => claim.commitment
  publicInputs := fun _ => claim.publicInput
  evaluations := fun _ => evaluationsToLight claim.evaluations

theorem toLightRunning_toClaim (system : PaperAlgebra.Structure logicalWidth)
    (claim : SlotClaim logicalWidth publicFits) :
    toLightRunning (toClaim system claim) = some (runningOf claim) := by
  simp [toLightRunning, toClaim, runningOf]

theorem freshToLight_injective {fresh fresh' : StepFresh logicalWidth publicFits}
    (same : freshToLight fresh = freshToLight fresh') : fresh = fresh' := by
  cases fresh
  cases fresh'
  simp only [freshToLight] at same
  obtain ⟨rfl, rfl⟩ := same
  rfl

theorem holds_normFree {Structure Assignment PublicInput Location Evaluation Commitment : Type}
    {semantics : RelationSemantics Structure Assignment PublicInput Location Evaluation Commitment}
    {bound : Nat} (bound' : Nat)
    {statement : CE.Instance Structure PublicInput Location Evaluation Commitment}
    {assignment : Assignment} (holds : Light.Rlc.Holds semantics bound statement assignment) :
    Light.Rlc.Holds (normFree semantics) bound' statement assignment :=
  ⟨⟨holds.1.1, holds.1.2.1, trivial⟩, holds.2⟩

section Verifier

variable (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
  (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))

/-- The light `ρ` that the transcript at `fresh` and `proof` reads from `oracle`. -/
noncomputable def challenge (oracle : Oracle logicalWidth publicFits)
    (fresh : Statement logicalWidth publicFits) (proof : Light.Proof LightShape 8) : RingF :=
  Fork.rho oracle fresh (LightSchedule.messages proof) first

noncomputable def outputs (oracle : Oracle logicalWidth publicFits)
    (running : Running logicalWidth publicFits) (fresh : Statement logicalWidth publicFits)
    (proof : Light.Proof LightShape 8) :=
  ((keys relation ajtai).light.statement running fresh).publicOutput
    (LightTest.oracleProbe relation ajtai running oracle fresh proof)

/-- The light Π_RLC output (`Light.Key.outputFor`) on the oracle probe, at `ρ = rho`. -/
noncomputable def outputFor (oracle : Oracle logicalWidth publicFits)
    (running : Running logicalWidth publicFits) (fresh : Statement logicalWidth publicFits)
    (proof : Light.Proof LightShape 8) (rho : RingF) : Claim logicalWidth publicFits :=
  Light.Rlc.output (keys relation ajtai).light.piRlcAlgebra (keys relation ajtai).light.one rho
    (keys relation ajtai).light.relationSource
    (LightTest.oracleProbe relation ajtai running oracle fresh proof).coins.roundPoint
    (outputs relation ajtai oracle running fresh proof (keys relation ajtai).light.freshSource)
    (outputs relation ajtai oracle running fresh proof (keys relation ajtai).light.runningSource)

/-- `Light.verify` in the random-oracle model: the light Π_CCS check and the light Π_RLC output
with the coins and `ρ` read from `oracle`. -/
noncomputable def verify (oracle : Oracle logicalWidth publicFits)
    (running : Running logicalWidth publicFits) (fresh : Statement logicalWidth publicFits)
    (proof : Light.Proof LightShape 8) : Option (Claim logicalWidth publicFits) :=
  if Light.PiCCS.check (keys relation ajtai).light.extensionOps (keys relation ajtai).light.lift
      ((keys relation ajtai).light.statement running fresh)
      (ProductionKey.degreeBound relation)
      (LightTest.oracleProbe relation ajtai running oracle fresh proof) then
    some (outputFor relation ajtai oracle running fresh proof (challenge oracle fresh proof))
  else
    none

/-- A new answer at the `ρ` point leaves the oracle probe unchanged. -/
theorem oracleProbe_update (oracle : Oracle logicalWidth publicFits)
    (running : Running logicalWidth publicFits) (fresh : Statement logicalWidth publicFits)
    (proof : Light.Proof LightShape 8) (answer : Answer) :
    LightTest.oracleProbe relation ajtai running
        (Function.update oracle (point fresh (LightSchedule.messages proof) (.rho first)) answer :
          Oracle logicalWidth publicFits)
        fresh proof =
      LightTest.oracleProbe relation ajtai running oracle fresh proof := by
  have coinsEq := Fork.coins_update_rho oracle fresh (LightSchedule.messages proof) first answer
  unfold LightTest.oracleProbe
  exact congrArg (fun coins =>
    ({ (keys relation ajtai).light.probe running fresh proof with coins := coins } :
      Probe K LightShape)) coinsEq

theorem outputFor_update (oracle : Oracle logicalWidth publicFits)
    (running : Running logicalWidth publicFits) (fresh : Statement logicalWidth publicFits)
    (proof : Light.Proof LightShape 8) (answer : Answer) (rho : RingF) :
    outputFor relation ajtai
        (Function.update oracle (point fresh (LightSchedule.messages proof) (.rho first)) answer)
        running fresh proof rho =
      outputFor relation ajtai oracle running fresh proof rho := by
  simp only [outputFor, outputs, oracleProbe_update]

theorem challenge_update (oracle : Oracle logicalWidth publicFits)
    (fresh : Statement logicalWidth publicFits) (proof : Light.Proof LightShape 8)
    (answer : Answer) :
    challenge
        (Function.update oracle (point fresh (LightSchedule.messages proof) (.rho first)) answer)
        fresh proof =
      Fork.readRho answer := by
  simp [challenge, Fork.rho]

end Verifier

variable (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))

/-- The option-B fold verifiers of the light fold in the random-oracle model: `slotFolds` at a
light step with `verify` in place of `Light.verify`. Every other fold rejects. -/
noncomputable def folds (oracle : Oracle logicalWidth publicFits) : Folds logicalWidth publicFits
  | .light, key, claim, fresh, .light proof =>
      ((toLightRunning (toClaim (key.system ajtai) claim)).bind fun running =>
        (verify key.relation ajtai oracle running (freshToLight fresh) proof).map fromLight).bind
        fromClaim
  | _, _, _, _, _ => none

/-- A light fold that `folds` accepts: a light proof that passes the light Π_CCS check on the
oracle coins, and the next slot is the light Π_RLC output at the oracle's `ρ`. -/
theorem folds_eq_some {oracle : Oracle logicalWidth publicFits} {step : Folding.Schedule.Step}
    {key : CircuitKey logicalWidth publicFits} {claim : SlotClaim logicalWidth publicFits}
    {fresh : StepFresh logicalWidth publicFits} {proof : StepProof}
    {next : SlotClaim logicalWidth publicFits}
    (accepted : folds ajtai oracle step key claim fresh proof = some next) :
    proof = .light (proofOf proof) ∧
      Light.PiCCS.Accepted (keys key.relation ajtai).light.extensionOps
        (keys key.relation ajtai).light.lift
        ((keys key.relation ajtai).light.statement (runningOf claim) (freshToLight fresh))
        (ProductionKey.degreeBound key.relation)
        (LightTest.oracleProbe key.relation ajtai (runningOf claim) oracle (freshToLight fresh)
          (proofOf proof)) ∧
      toClaim (key.system ajtai) next =
        fromLight (outputFor key.relation ajtai oracle (runningOf claim) (freshToLight fresh)
          (proofOf proof) (challenge oracle (freshToLight fresh) (proofOf proof))) := by
  cases step with
  | refresh => cases proof <;> exact absurd accepted (by simp [folds])
  | light =>
      cases proof with
      | refresh _ => exact absurd accepted (by simp [folds])
      | light proof =>
          simp only [folds, toLightRunning_toClaim, Option.bind_some] at accepted
          unfold verify at accepted
          split at accepted
          · next checked =>
              simp only [Option.map_some, Option.bind_some] at accepted
              exact ⟨rfl, checked, (toClaim_of_fromClaim rfl rfl accepted).symm⟩
          · simp at accepted

/-- The light fork game with circuit key `key`. -/
noncomputable def game (key : CircuitKey logicalWidth publicFits) :=
  ForkLink.game ajtai key (folds ajtai) (fun attempt => freshToLight attempt.link.fold.fresh)
    (fun attempt => LightSchedule.messages (proofOf attempt.link.fold.proof))

/-- A successful light attempt carries a light proof that passes the light Π_CCS check on the
oracle coins, and its witness opens the light Π_RLC output at the attempt's bound. -/
theorem succeeds_holds {key : CircuitKey logicalWidth publicFits}
    {oracle : Oracle logicalWidth publicFits} {attempt : Attempt logicalWidth publicFits}
    (succeeds : (game ajtai key).Succeeds oracle attempt) :
    attempt.link.fold.proof = .light (proofOf attempt.link.fold.proof) ∧
      Light.PiCCS.Accepted (keys key.relation ajtai).light.extensionOps
        (keys key.relation ajtai).light.lift
        ((keys key.relation ajtai).light.statement (runningOf attempt.link.start.1)
          (freshToLight attempt.link.fold.fresh))
        (ProductionKey.degreeBound key.relation)
        (LightTest.oracleProbe key.relation ajtai (runningOf attempt.link.start.1) oracle
          (freshToLight attempt.link.fold.fresh) (proofOf attempt.link.fold.proof)) ∧
      Light.Rlc.Holds (keys key.relation ajtai).light.piRlcSemantics attempt.bound
        (outputFor key.relation ajtai oracle (runningOf attempt.link.start.1)
          (freshToLight attempt.link.fold.fresh) (proofOf attempt.link.fold.proof)
          (challenge oracle (freshToLight attempt.link.fold.fresh)
            (proofOf attempt.link.fold.proof)))
        attempt.witness := by
  obtain ⟨keyEq, _, _, accepted, holds, _⟩ := succeeds
  obtain ⟨_, folded, _⟩ := (BoundedDepth.accepts_iff _ _ _ _ _ _ _ _ _).mp accepted
  obtain ⟨proofEq, checked, claimEq⟩ := folds_eq_some ajtai folded
  subst keyEq
  refine ⟨proofEq, checked, ?_⟩
  have opened : Light.Rlc.Holds (PaperAlgebra.semantics ajtai) attempt.bound
      (fromLight (outputFor attempt.link.fold.key.relation ajtai oracle
        (runningOf attempt.link.start.1) (freshToLight attempt.link.fold.fresh)
        (proofOf attempt.link.fold.proof)
        (challenge oracle (freshToLight attempt.link.fold.fresh)
          (proofOf attempt.link.fold.proof))))
      attempt.witness := by
    rw [← claimEq]
    exact holds
  exact (OptionBKeys.boundHolds_fromLight_iff _ _ _).mp opened

section Fork

variable {key : CircuitKey logicalWidth publicFits}
  {adversary : OracleComp (Point (LightSchedule.schedule logicalWidth publicFits) 8) Answer
    (Attempt logicalWidth publicFits)}
  {oracle : Oracle logicalWidth publicFits}
  {retries : Fin (LightSchedule.schedule logicalWidth publicFits).rhoCount → Answer}

local notation "Base" => adversary.run oracle
local notation "Retry" =>
  adversary.run (Fork.forked (game ajtai key) adversary oracle retries first)

/-- A valid retry's witness opens the light Π_RLC output on the base run's inputs at the
retry's answer. -/
theorem retry_holds (valid : Fork.ForkValid (game ajtai key) adversary oracle retries first) :
    Light.Rlc.Holds (keys key.relation ajtai).light.piRlcSemantics (Retry).bound
      (outputFor key.relation ajtai oracle (runningOf (Base).link.start.1)
        (freshToLight (Base).link.fold.fresh) (proofOf (Base).link.fold.proof)
        (Fork.readRho (retries first)))
      (Retry).witness := by
  have holds := (succeeds_holds ajtai valid.succeeds).2.2
  have statementEq : freshToLight (Retry).link.fold.fresh = freshToLight (Base).link.fold.fresh :=
    valid.identifies.1
  have proofEq : proofOf (Retry).link.fold.proof = proofOf (Base).link.fold.proof :=
    congrArg (fun messages : (LightSchedule.schedule logicalWidth publicFits).Messages 8 =>
      (⟨messages.rounds, messages.output⟩ : Light.Proof LightShape 8)) valid.identifies.2
  have startEq : (Retry).link.start = (Base).link.start := valid.start_eq
  rw [statementEq, proofEq, startEq] at holds
  have outputEq : outputFor key.relation ajtai
      (Fork.forked (game ajtai key) adversary oracle retries first)
      (runningOf (Base).link.start.1) (freshToLight (Base).link.fold.fresh)
      (proofOf (Base).link.fold.proof)
      (challenge (Fork.forked (game ajtai key) adversary oracle retries first)
        (freshToLight (Base).link.fold.fresh) (proofOf (Base).link.fold.proof)) =
      outputFor key.relation ajtai oracle (runningOf (Base).link.start.1)
        (freshToLight (Base).link.fold.fresh) (proofOf (Base).link.fold.proof)
        (Fork.readRho (retries first)) :=
    (outputFor_update key.relation ajtai oracle (runningOf (Base).link.start.1)
      (freshToLight (Base).link.fold.fresh) (proofOf (Base).link.fold.proof) (retries first)
      _).trans
      (congrArg (outputFor key.relation ajtai oracle (runningOf (Base).link.start.1)
        (freshToLight (Base).link.fold.fresh) (proofOf (Base).link.fold.proof))
        (challenge_update oracle (freshToLight (Base).link.fold.fresh)
          (proofOf (Base).link.fold.proof) (retries first)))
  rw [outputEq] at holds
  exact holds

/-- The light Π_RLC fork without norms of a base run and its valid retry (SuperNeo v1.2
Lemma 8): two outputs on the base run's inputs with different `ρ`. -/
abbrev RlcFork (key : CircuitKey logicalWidth publicFits) (oracle : Oracle logicalWidth publicFits)
    (attempt : Attempt logicalWidth publicFits) :=
  Light.Rlc.Fork (normFreeAlgebra (keys key.relation ajtai).light.piRlcAlgebra)
    (keys key.relation ajtai).light.one attempt.bound
    (keys key.relation ajtai).light.relationSource
    (LightTest.oracleProbe key.relation ajtai (runningOf attempt.link.start.1) oracle
      (freshToLight attempt.link.fold.fresh) (proofOf attempt.link.fold.proof)).coins.roundPoint
    (outputs key.relation ajtai oracle (runningOf attempt.link.start.1)
      (freshToLight attempt.link.fold.fresh) (proofOf attempt.link.fold.proof)
      (keys key.relation ajtai).light.freshSource)
    (outputs key.relation ajtai oracle (runningOf attempt.link.start.1)
      (freshToLight attempt.link.fold.fresh) (proofOf attempt.link.fold.proof)
      (keys key.relation ajtai).light.runningSource)

/-- The fork of a successful base run and a valid retry. -/
noncomputable def fork (succeeds : Fork.Succeeds (game ajtai key) adversary oracle)
    (valid : ∀ index, Fork.ForkValid (game ajtai key) adversary oracle retries index) :
    RlcFork ajtai key oracle Base where
  baseChallenge :=
    challenge oracle (freshToLight (Base).link.fold.fresh) (proofOf (Base).link.fold.proof)
  forkChallenge := Fork.readRho (retries first)
  baseAssignment := (Base).witness
  forkAssignment := (Retry).witness
  baseValid := Fork.readRho_valid _
  forkValid := Fork.readRho_valid _
  changed := by
    have changed := (valid first).changed
    rw [(valid first).rho_eq first, ite_eq_left rfl] at changed
    exact changed
  baseHolds := holds_normFree _ (succeeds_holds ajtai succeeds).2.2
  forkHolds := holds_normFree _ (retry_holds ajtai (valid first))

/-- The fork's base witness opens at the base attempt's bound. -/
theorem fork_baseNorm (succeeds : Fork.Succeeds (game ajtai key) adversary oracle)
    (valid : ∀ index, Fork.ForkValid (game ajtai key) adversary oracle retries index) :
    (keys key.relation ajtai).light.piRlcSemantics.normBounded (Base).bound
      (fork ajtai succeeds valid).baseAssignment :=
  (succeeds_holds ajtai succeeds).2.2.1.2.2

/-- SuperNeo v1.2 Lemma 8's strong set for the production light key. -/
noncomputable abbrev strongSet :=
  Phi81Relation.PiRLCAlgebra.ForkStrongSet.strongSetUnits Phi81StrongSet.lowNormInvertibility

/-- The output witness that the fork extracts: the fresh witness `(ρ − ρ')⁻¹ (w − w')` and
the running witness `w − ρ · fresh` (`Light.Rlc.Fork.freshAssignment`, `runningAssignment`). -/
noncomputable def extractedWitness (succeeds : Fork.Succeeds (game ajtai key) adversary oracle)
    (valid : ∀ index, Fork.ForkValid (game ajtai key) adversary oracle retries index) :
    OutputWitness LightShape (Phi81CarrierLayout.carrierWidth logicalWidth) where
  assignments := fun source =>
    if source.val = 0 then
      (fork ajtai succeeds valid).freshAssignment (normFreeLaws (lightExtraction ajtai)) strongSet
    else
      (fork ajtai succeeds valid).runningAssignment (normFreeLaws (lightExtraction ajtai))
        strongSet

/-- SuperNeo v1.2 Lemma 8 on the fork: the extracted witnesses open the base run's fresh and
running Π_CCS outputs at the ambient bound (`Light.Rlc.fork_extracts`). -/
theorem extracted_ambient (succeeds : Fork.Succeeds (game ajtai key) adversary oracle)
    (valid : ∀ index, Fork.ForkValid (game ajtai key) adversary oracle retries index) :
    PiRLC.PaperCorrections.CorrectedAmbientHolds (keys key.relation ajtai).light.piRlcSemantics
        (keys key.relation ajtai).light.params
        (outputs key.relation ajtai oracle (runningOf (Base).link.start.1)
          (freshToLight (Base).link.fold.fresh) (proofOf (Base).link.fold.proof)
          (keys key.relation ajtai).light.freshSource)
        ((fork ajtai succeeds valid).freshAssignment (normFreeLaws (lightExtraction ajtai))
          strongSet) ∧
      PiRLC.PaperCorrections.CorrectedAmbientHolds (keys key.relation ajtai).light.piRlcSemantics
        (keys key.relation ajtai).light.params
        (outputs key.relation ajtai oracle (runningOf (Base).link.start.1)
          (freshToLight (Base).link.fold.fresh) (proofOf (Base).link.fold.proof)
          (keys key.relation ajtai).light.runningSource)
        ((fork ajtai succeeds valid).runningAssignment (normFreeLaws (lightExtraction ajtai))
          strongSet) := by
  obtain ⟨freshFree, runningFree⟩ := Light.Rlc.fork_extracts
    (normFreeLaws (lightExtraction ajtai)) strongSet rfl (fork ajtai succeeds valid)
    rfl rfl rfl rfl 1 rfl rfl
  exact ⟨⟨⟨freshFree.1.1, freshFree.1.2.1, (lightExtraction ajtai).correctedNormCoverage _⟩,
      freshFree.2⟩,
    ⟨⟨runningFree.1.1, runningFree.1.2.1, (lightExtraction ajtai).correctedNormCoverage _⟩,
      runningFree.2⟩⟩

/-- Cyclo (ePrint 2026/359) Theorem 3, item 2, on the fork: with a fresh witness of norm below
`b`, the extracted running witness opens at the base bound plus `T · (b − 1)`. -/
theorem extracted_running_normBounded
    (succeeds : Fork.Succeeds (game ajtai key) adversary oracle)
    (valid : ∀ index, Fork.ForkValid (game ajtai key) adversary oracle retries index)
    (freshBounded : (keys key.relation ajtai).light.piRlcSemantics.normBounded
      (keys key.relation ajtai).light.params.b
      ((fork ajtai succeeds valid).freshAssignment (normFreeLaws (lightExtraction ajtai))
        strongSet)) :
    (keys key.relation ajtai).light.piRlcSemantics.normBounded
      ((Base).bound + (keys key.relation ajtai).light.params.expansionT *
        ((keys key.relation ajtai).light.params.b - 1))
      ((fork ajtai succeeds valid).runningAssignment (normFreeLaws (lightExtraction ajtai))
        strongSet) :=
  (lightExtractionNormLaw ajtai).sub_bound (Base).bound (fork ajtai succeeds valid).baseChallenge
    (fork ajtai succeeds valid).baseAssignment _ (fork ajtai succeeds valid).baseValid
    (fork_baseNorm ajtai succeeds valid) freshBounded

end Fork

section Bounds

variable (key : CircuitKey logicalWidth publicFits)
  (adversary : OracleComp (Point (LightSchedule.schedule logicalWidth publicFits) 8) Answer
    (Attempt logicalWidth publicFits))

/-- The chance that the light retry gives a collision of the base and retry prior states. -/
noncomputable def collisionChance
    (index : Fin (LightSchedule.schedule logicalWidth publicFits).rhoCount)
    (oracle : Oracle logicalWidth publicFits) : ℝ :=
  ForkLink.collisionChance ajtai key (folds ajtai)
    (fun attempt => freshToLight attempt.link.fold.fresh)
    (fun attempt => LightSchedule.messages (proofOf attempt.link.fold.proof)) adversary index oracle

/-- `Fork.fork_failure_le` for the light fold: a successful run whose retry is invalid has
probability at most `(Q + 1) · sampleError` plus the expected prior-state collision chance. -/
theorem fork_failure_le {queries : Nat} (bounded : adversary.QueryBound queries) :
    𝔼 oracle, (if Fork.Succeeds (game ajtai key) adversary oracle then
        ∑ retries, retryWeight (Fork.retrySets (game ajtai key) adversary oracle) retries *
          (if ∀ index, Fork.ForkValid (game ajtai key) adversary oracle retries index
            then 0 else 1)
      else 0) ≤
      ((queries + 1 : Nat) : ℝ) * Fork.sampleError +
        𝔼 oracle, (if Fork.Succeeds (game ajtai key) adversary oracle then
          ∑ index, collisionChance ajtai key adversary index oracle else 0) := by
  have count : ((LightSchedule.schedule logicalWidth publicFits).rhoCount : ℝ) *
      (((queries + (LightSchedule.schedule logicalWidth publicFits).rhoCount : Nat) : ℝ) *
        Fork.sampleError) =
      ((queries + 1 : Nat) : ℝ) * Fork.sampleError := by
    show ((1 : Nat) : ℝ) * (((queries + 1 : Nat) : ℝ) * Fork.sampleError) = _
    rw [Nat.cast_one, one_mul]
  have failure := ForkLink.fork_failure_le ajtai key (folds ajtai)
    (fun attempt => LightSchedule.messages (proofOf attempt.link.fold.proof)) adversary
    (fun same => freshToLight_injective same) bounded
  rw [count] at failure
  exact failure

/-- `Fork.expected_retries_le` for the light fold: at most `Q + 1` expected reruns. -/
theorem expected_retries_le {queries : Nat} (bounded : adversary.QueryBound queries) :
    𝔼 oracle, (if Fork.Succeeds (game ajtai key) adversary oracle then
        ∑ index, retryCount (Fork.Hits (game ajtai key) adversary index)
          (Fork.rhoPoint (game ajtai key) adversary index oracle) oracle else 0) ≤
      ((queries + 1 : Nat) : ℝ) := by
  have count : ((LightSchedule.schedule logicalWidth publicFits).rhoCount : ℝ) *
      ((queries + (LightSchedule.schedule logicalWidth publicFits).rhoCount : Nat) : ℝ) =
      ((queries + 1 : Nat) : ℝ) := by
    show ((1 : Nat) : ℝ) * ((queries + 1 : Nat) : ℝ) = _
    rw [Nat.cast_one, one_mul]
  have retries := Fork.expected_retries_le (game ajtai key) adversary bounded
  rw [count] at retries
  exact retries

end Bounds

end NightstreamFPrime.Lifecycle.OptionB.RandomOracle.LightFork
