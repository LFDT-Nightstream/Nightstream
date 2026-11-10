import NightstreamFPrime.Lifecycle.OptionB.RandomOracle.ForkLink
import NightstreamFPrime.Lifecycle.OptionB.RandomOracle.RefreshTest
import NightstreamFPrime.Spec.Folding.PiRLC.NormFreeFork
import NightstreamFPrime.Spec.Phi81StrongSet.LowNormInvertibility
import NightstreamFPrime.Spec.Folding.Nifs.PaperStrongInterface

/-! The refresh fold's fork step in the random-oracle model: `Fork` on the refresh schedule,
which has 17 `ρ` challenges. `verify` is `Refresh.verify` with the Π_CCS coins and the `ρ`
challenges read from the oracle (`RefreshTest.oracleProbe`, `Fork.rho`); the Π_RLC parent is the
combined output of the batch on the oracle probe (`batch`, `parentFor`). `folds` is `slotFolds`
at a refresh step with `verify` in its place. An adversary outputs an option-B attempt; the
attempt succeeds in the refresh game (`game`) when its fold has the game's circuit key and
`ForkLink.Succeeds` holds with `folds`.

A successful attempt passes the Π_DEC input check and the refresh Π_CCS check on the oracle
coins, and its witness opens the parent at the attempt's bound (`succeeds_holds`). The oracle
probe reads only the split and the Π_CCS messages of a proof (`oracleProbe_congr`), so a valid
retry at coordinate `i` has the base run's batch, and its witness opens the parent at the base
run's challenges with coordinate `i` replaced (`retry_holds`). The base run and the 17 retries
are SuperNeo v1.2 Appendix B.3's complete fork without norms on the base batch (`fork`). It
extracts the 17 Π_CCS outputs at the ambient bound (`extractedWitness`, `extracted_ambient`,
through `NormFreeFork.correctedAmbientHolds`).

With `rhoCount = 17`: an invalid retry family has probability at most
`17 · (Q + 17) · sampleError` plus the prior-state collision chance (`fork_failure_le`), and the
expected number of reruns is at most `17 · (Q + 17)` (`expected_retries_le`). -/

set_option autoImplicit false

namespace NightstreamFPrime.Lifecycle.OptionB.RandomOracle.RefreshFork

open scoped BigOperators
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.RandomOracle
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.StrongReduction
open NightstreamFPrime.Spec.HyperNova.Construction2
open NightstreamFPrime.Spec.Folding.PiRLC.PaperForkExtraction
  (CompleteFork Response extractedAssignment)
open NightstreamFPrime.Spec.Folding.PiRLC.NormFreeFork (normFree normFreeAlgebra normFreeLaws)
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.OptionB
open NightstreamFPrime.Lifecycle.RandomOracle
open NightstreamFPrime.Lifecycle.RandomOracle.Test (Answer Point)
open NightstreamFPrime.Lifecycle.OptionB.RandomOracle.ForkLink (Attempt Folds)

attribute [local instance low] Classical.propDecidable

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth}

section Types

variable (logicalWidth publicFits)

abbrev Oracle := Point (RefreshSchedule.schedule logicalWidth publicFits) 8 → Answer

abbrev Accumulator :=
  Refresh.Accumulator PaperAlgebra.Commitment
    (PaperAlgebra.PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits))
    productionShape (Phi81CarrierLayout.carrierWidth logicalWidth)
    (Phi81ColumnLayout.blockCount (Phi81CarrierLayout.carrierWidth logicalWidth))

end Types

abbrev RefreshProof := Refresh.Proof PaperAlgebra.Commitment productionShape 8

/-- The refresh proof that the refresh game reads from a light proof. A light proof never passes
the refresh game's fold verifiers, so the value does not matter. -/
def placeholder : RefreshProof :=
  let family : EvaluationFamily K productionShape := ⟨fun _ => K.zero, fun _ _ => K.zero⟩
  ⟨⟨fun _ => Phi81Relation.PiRLCAlgebra.Commitment.commitmentZero, fun _ => family⟩,
    ⟨fun _ => ⟨List.replicate (8 + 1) K.zero, by simp⟩,
      ⟨fun _ _ => K.zero, fun _ _ _ => K.zero⟩,
      fun _ => Phi81Relation.PiRLCAlgebra.Commitment.commitmentZero, fun _ => family⟩⟩

def proofOf : StepProof → RefreshProof
  | .refresh proof => proof
  | .light _ => placeholder

section Verifier

variable (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
  (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))

/-- The refresh Π_RLC batch on the oracle probe: the 17 Π_CCS outputs at the oracle's round
point (`PaperStrongInterface.piRlcBatchForProbe`). -/
noncomputable def batch (oracle : Oracle logicalWidth publicFits)
    (accumulator : Accumulator logicalWidth publicFits) (fresh : StepFresh logicalWidth publicFits)
    (proof : RefreshProof) :=
  Nifs.PaperStrongInterface.piRlcBatchForProbe (refreshKey relation ajtai)
    (Refresh.children (refreshKey relation ajtai) accumulator proof.split) fresh
    (RefreshTest.oracleProbe relation ajtai accumulator oracle fresh proof)

/-- The refresh Π_RLC parent at challenges `rho` (`Key.parentForChallenges` on the oracle
probe). -/
noncomputable def parentFor (oracle : Oracle logicalWidth publicFits)
    (accumulator : Accumulator logicalWidth publicFits) (fresh : StepFresh logicalWidth publicFits)
    (proof : RefreshProof) (rho : Fin (refreshKey relation ajtai).arity.total → RingF) :
    Accumulator logicalWidth publicFits :=
  PiRLC.combinedOutput (refreshKey relation ajtai).piRlcAlgebra
    (batch relation ajtai oracle accumulator fresh proof).system
    (batch relation ajtai oracle accumulator fresh proof).point
    (batch relation ajtai oracle accumulator fresh proof).inputs rho

/-- `Refresh.verify` in the random-oracle model: the Π_DEC input check, the refresh Π_CCS check
on the oracle coins, and the Π_RLC parent at the oracle's 17 `ρ` challenges. -/
noncomputable def verify (oracle : Oracle logicalWidth publicFits)
    (accumulator : Accumulator logicalWidth publicFits) (fresh : StepFresh logicalWidth publicFits)
    (proof : RefreshProof) : Option (Accumulator logicalWidth publicFits) :=
  if Refresh.inputCheck (refreshKey relation ajtai) accumulator proof.split = true ∧
      (RefreshTest.oracleProbe relation ajtai accumulator oracle fresh proof).FixedWidthAccepted
        (refreshKey relation ajtai).extensionOps (refreshKey relation ajtai).lift
        ((refreshKey relation ajtai).statement
          (Refresh.children (refreshKey relation ajtai) accumulator proof.split) fresh)
        (ProductionKey.degreeBound relation) then
    some (parentFor relation ajtai oracle accumulator fresh proof
      (Fork.rho oracle (fresh, proof.split) (RefreshSchedule.messages proof.nifs)))
  else
    none

/-- The oracle probe reads only the split and the Π_CCS messages of a proof. -/
theorem oracleProbe_congr (oracle : Oracle logicalWidth publicFits)
    (accumulator : Accumulator logicalWidth publicFits) (fresh : StepFresh logicalWidth publicFits)
    {proof proof' : RefreshProof} (split : proof.split = proof'.split)
    (messages : RefreshSchedule.messages (logicalWidth := logicalWidth) (publicFits := publicFits)
      proof.nifs = RefreshSchedule.messages proof'.nifs) :
    RefreshTest.oracleProbe relation ajtai accumulator oracle fresh proof =
      RefreshTest.oracleProbe relation ajtai accumulator oracle fresh proof' := by
  obtain ⟨split₀, ⟨rounds, output, commitments, evaluations⟩⟩ := proof
  obtain ⟨split₁, ⟨rounds', output', commitments', evaluations'⟩⟩ := proof'
  have roundsEq : rounds = rounds' := congrArg Schedule.Messages.rounds messages
  have outputEq : output = output' := congrArg Schedule.Messages.output messages
  have splitEq : split₀ = split₁ := split
  subst roundsEq outputEq splitEq
  rfl

theorem batch_congr (oracle : Oracle logicalWidth publicFits)
    (accumulator : Accumulator logicalWidth publicFits) (fresh : StepFresh logicalWidth publicFits)
    {proof proof' : RefreshProof} (split : proof.split = proof'.split)
    (messages : RefreshSchedule.messages (logicalWidth := logicalWidth) (publicFits := publicFits)
      proof.nifs = RefreshSchedule.messages proof'.nifs) :
    batch relation ajtai oracle accumulator fresh proof =
      batch relation ajtai oracle accumulator fresh proof' := by
  unfold batch
  rw [oracleProbe_congr relation ajtai oracle accumulator fresh split messages, split]

/-- A new answer at a `ρ` point leaves the batch unchanged. -/
theorem batch_update (oracle : Oracle logicalWidth publicFits)
    (accumulator : Accumulator logicalWidth publicFits) (fresh : StepFresh logicalWidth publicFits)
    (proof : RefreshProof) (index : Fin (RefreshSchedule.schedule logicalWidth publicFits).rhoCount)
    (answer : Answer) :
    batch relation ajtai
        (Function.update oracle
          (Test.point (fresh, proof.split) (RefreshSchedule.messages proof.nifs) (.rho index))
          answer)
        accumulator fresh proof =
      batch relation ajtai oracle accumulator fresh proof := by
  have coinsEq := Fork.coins_update_rho oracle (fresh, proof.split)
    (RefreshSchedule.messages proof.nifs) index answer
  have probeEq : RefreshTest.oracleProbe relation ajtai accumulator
      (Function.update oracle
        (Test.point (fresh, proof.split) (RefreshSchedule.messages proof.nifs) (.rho index))
        answer : Oracle logicalWidth publicFits) fresh proof =
      RefreshTest.oracleProbe relation ajtai accumulator oracle fresh proof := by
    unfold RefreshTest.oracleProbe
    exact congrArg (fun coins =>
      ({ (refreshKey relation ajtai).piCcsProbe
          (Refresh.children (refreshKey relation ajtai) accumulator proof.split) fresh proof.nifs
        with coins := coins } : Probe K productionShape)) coinsEq
  unfold batch
  rw [probeEq]

end Verifier

variable (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))

/-- The option-B fold verifiers of the refresh fold in the random-oracle model: `slotFolds` at a
refresh step with `verify` in place of `Refresh.verify`. Every other fold rejects. -/
noncomputable def folds (oracle : Oracle logicalWidth publicFits) : Folds logicalWidth publicFits
  | .refresh, key, claim, fresh, .refresh proof =>
      (verify key.relation ajtai oracle (toClaim (key.system ajtai) claim) fresh proof).bind
        fromClaim
  | _, _, _, _, _ => none

/-- A refresh fold that `folds` accepts: a refresh proof that passes the Π_DEC input check and
the refresh Π_CCS check on the oracle coins, and the next slot is the parent at the oracle's
`ρ` challenges. -/
theorem folds_eq_some {oracle : Oracle logicalWidth publicFits} {step : Folding.Schedule.Step}
    {key : CircuitKey logicalWidth publicFits} {claim : SlotClaim logicalWidth publicFits}
    {fresh : StepFresh logicalWidth publicFits} {proof : StepProof}
    {next : SlotClaim logicalWidth publicFits}
    (accepted : folds ajtai oracle step key claim fresh proof = some next) :
    proof = .refresh (proofOf proof) ∧
      Refresh.inputCheck (refreshKey key.relation ajtai) (toClaim (key.system ajtai) claim)
        (proofOf proof).split = true ∧
      (RefreshTest.oracleProbe key.relation ajtai (toClaim (key.system ajtai) claim) oracle fresh
          (proofOf proof)).FixedWidthAccepted
        (refreshKey key.relation ajtai).extensionOps (refreshKey key.relation ajtai).lift
        ((refreshKey key.relation ajtai).statement
          (Refresh.children (refreshKey key.relation ajtai) (toClaim (key.system ajtai) claim)
            (proofOf proof).split) fresh)
        (ProductionKey.degreeBound key.relation) ∧
      toClaim (key.system ajtai) next =
        parentFor key.relation ajtai oracle (toClaim (key.system ajtai) claim) fresh
          (proofOf proof)
          (Fork.rho oracle (fresh, (proofOf proof).split)
            (RefreshSchedule.messages (proofOf proof).nifs)) := by
  cases step with
  | light => cases proof <;> exact absurd accepted (by simp [folds])
  | refresh =>
      cases proof with
      | light _ => exact absurd accepted (by simp [folds])
      | refresh proof =>
          simp only [folds] at accepted
          unfold verify at accepted
          split at accepted
          · next checked =>
              simp only [Option.bind_some] at accepted
              exact ⟨rfl, checked.1, checked.2, (toClaim_of_fromClaim rfl rfl accepted).symm⟩
          · simp at accepted

/-- The refresh fork game with circuit key `key`. -/
noncomputable def game (key : CircuitKey logicalWidth publicFits) :=
  ForkLink.game ajtai key (folds ajtai)
    (fun attempt => (attempt.link.fold.fresh, (proofOf attempt.link.fold.proof).split))
    (fun attempt => RefreshSchedule.messages (proofOf attempt.link.fold.proof).nifs)

/-- A successful refresh attempt carries a refresh proof that passes the Π_DEC input check and
the refresh Π_CCS check on the oracle coins, and its witness opens the parent at the attempt's
bound. -/
theorem succeeds_holds {key : CircuitKey logicalWidth publicFits}
    {oracle : Oracle logicalWidth publicFits} {attempt : Attempt logicalWidth publicFits}
    (succeeds : (game ajtai key).Succeeds oracle attempt) :
    attempt.link.fold.proof = .refresh (proofOf attempt.link.fold.proof) ∧
      Refresh.inputCheck (refreshKey key.relation ajtai)
        (toClaim (key.system ajtai) attempt.link.start.1)
        (proofOf attempt.link.fold.proof).split = true ∧
      (RefreshTest.oracleProbe key.relation ajtai (toClaim (key.system ajtai) attempt.link.start.1)
          oracle attempt.link.fold.fresh (proofOf attempt.link.fold.proof)).FixedWidthAccepted
        (refreshKey key.relation ajtai).extensionOps (refreshKey key.relation ajtai).lift
        ((refreshKey key.relation ajtai).statement
          (Refresh.children (refreshKey key.relation ajtai)
            (toClaim (key.system ajtai) attempt.link.start.1)
            (proofOf attempt.link.fold.proof).split)
          attempt.link.fold.fresh)
        (ProductionKey.degreeBound key.relation) ∧
      Light.Rlc.Holds (refreshKey key.relation ajtai).piRlcSemantics attempt.bound
        (parentFor key.relation ajtai oracle (toClaim (key.system ajtai) attempt.link.start.1)
          attempt.link.fold.fresh (proofOf attempt.link.fold.proof)
          (Fork.rho oracle (attempt.link.fold.fresh, (proofOf attempt.link.fold.proof).split)
            (RefreshSchedule.messages (proofOf attempt.link.fold.proof).nifs)))
        attempt.witness := by
  obtain ⟨keyEq, _, _, accepted, holds, _⟩ := succeeds
  obtain ⟨_, folded, _⟩ := (BoundedDepth.accepts_iff _ _ _ _ _ _ _ _ _).mp accepted
  obtain ⟨proofEq, input, checked, claimEq⟩ := folds_eq_some ajtai folded
  subst keyEq
  refine ⟨proofEq, input, checked, ?_⟩
  rw [← claimEq]
  exact holds

section Fork

variable {key : CircuitKey logicalWidth publicFits}
  {adversary : OracleComp (Point (RefreshSchedule.schedule logicalWidth publicFits) 8) Answer
    (Attempt logicalWidth publicFits)}
  {oracle : Oracle logicalWidth publicFits}
  {retries : Fin (RefreshSchedule.schedule logicalWidth publicFits).rhoCount → Answer}

local notation "Base" => adversary.run oracle

/-- The complete Π_RLC fork without norms on the base run's batch (SuperNeo v1.2 Appendix B.3):
the base response and one response per `ρ` coordinate. -/
abbrev RlcFork (key : CircuitKey logicalWidth publicFits) (oracle : Oracle logicalWidth publicFits)
    (attempt : Attempt logicalWidth publicFits) :=
  CompleteFork (normFree (refreshKey key.relation ajtai).piRlcSemantics)
    (refreshKey key.relation ajtai).params
    (normFreeAlgebra (refreshKey key.relation ajtai).piRlcAlgebra)
    (batch key.relation ajtai oracle (toClaim (key.system ajtai) attempt.link.start.1)
      attempt.link.fold.fresh (proofOf attempt.link.fold.proof))

/-- A valid retry at `index` opens the parent of the base run's batch at the retry's
challenges. -/
theorem retry_holds {index : Fin (RefreshSchedule.schedule logicalWidth publicFits).rhoCount}
    (valid : Fork.ForkValid (game ajtai key) adversary oracle retries index) :
    Light.Rlc.Holds (refreshKey key.relation ajtai).piRlcSemantics
      (adversary.run (Fork.forked (game ajtai key) adversary oracle retries index)).bound
      (parentFor key.relation ajtai oracle (toClaim (key.system ajtai) (Base).link.start.1)
        (Base).link.fold.fresh (proofOf (Base).link.fold.proof)
        (Fork.rho (Fork.forked (game ajtai key) adversary oracle retries index)
          ((game ajtai key).statement
            (adversary.run (Fork.forked (game ajtai key) adversary oracle retries index)))
          ((game ajtai key).messages
            (adversary.run (Fork.forked (game ajtai key) adversary oracle retries index)))))
      (adversary.run (Fork.forked (game ajtai key) adversary oracle retries index)).witness := by
  have holds := (succeeds_holds ajtai valid.succeeds).2.2.2
  obtain ⟨statementEq, messagesEq⟩ := valid.identifies
  have freshEq :
      (adversary.run
          (Fork.forked (game ajtai key) adversary oracle retries index)).link.fold.fresh =
        (Base).link.fold.fresh :=
    congrArg Prod.fst statementEq
  have splitEq : (proofOf (adversary.run
      (Fork.forked (game ajtai key) adversary oracle retries index)).link.fold.proof).split =
        (proofOf (Base).link.fold.proof).split :=
    congrArg Prod.snd statementEq
  have messagesEq' : RefreshSchedule.messages (proofOf (adversary.run
      (Fork.forked (game ajtai key) adversary oracle retries index)).link.fold.proof).nifs =
        RefreshSchedule.messages (proofOf (Base).link.fold.proof).nifs :=
    messagesEq
  have startEq :
      (adversary.run (Fork.forked (game ajtai key) adversary oracle retries index)).link.start =
        (Base).link.start :=
    valid.start_eq
  have batchEq : batch key.relation ajtai
      (Fork.forked (game ajtai key) adversary oracle retries index)
      (toClaim (key.system ajtai)
        (adversary.run (Fork.forked (game ajtai key) adversary oracle retries index)).link.start.1)
      (adversary.run (Fork.forked (game ajtai key) adversary oracle retries index)).link.fold.fresh
      (proofOf
        (adversary.run
          (Fork.forked (game ajtai key) adversary oracle retries index)).link.fold.proof) =
      batch key.relation ajtai oracle (toClaim (key.system ajtai) (Base).link.start.1)
        (Base).link.fold.fresh (proofOf (Base).link.fold.proof) := by
    rw [batch_congr key.relation ajtai _ _ _ splitEq messagesEq', freshEq, startEq]
    exact batch_update key.relation ajtai oracle (toClaim (key.system ajtai) (Base).link.start.1)
      (Base).link.fold.fresh (proofOf (Base).link.fold.proof) index (retries index)
  unfold parentFor at holds ⊢
  rw [batchEq] at holds
  exact holds

/-- The complete fork of a successful base run and a valid retry family. -/
noncomputable def fork (succeeds : Fork.Succeeds (game ajtai key) adversary oracle)
    (valid : ∀ index, Fork.ForkValid (game ajtai key) adversary oracle retries index) :
    RlcFork ajtai key oracle Base where
  base := ⟨Fork.rho oracle ((game ajtai key).statement Base) ((game ajtai key).messages Base),
    (Base).witness⟩
  forks index :=
    ⟨Fork.rho (Fork.forked (game ajtai key) adversary oracle retries index)
        ((game ajtai key).statement
          (adversary.run (Fork.forked (game ajtai key) adversary oracle retries index)))
        ((game ajtai key).messages
          (adversary.run (Fork.forked (game ajtai key) adversary oracle retries index))),
      (adversary.run (Fork.forked (game ajtai key) adversary oracle retries index)).witness⟩
  baseSuccess :=
    let holds := (succeeds_holds ajtai succeeds).2.2.2
    PiRLC.NormFreeFork.success_at _ _ holds.1 holds.2.1 holds.2.2
  forkSuccess index :=
    let holds := retry_holds ajtai (valid index)
    PiRLC.NormFreeFork.success_at _ _ holds.1 holds.2.1 holds.2.2
  baseStrong _ := Fork.readRho_valid _
  forkStrong _ _ := Fork.readRho_valid _
  agreeExcept index _ different := (valid index).agree different
  changed index := (valid index).changed

/-- SuperNeo v1.2 Lemma 8's strong set for the production refresh key. -/
noncomputable abbrev strongSet :=
  Phi81Relation.PiRLCAlgebra.ForkStrongSet.strongSetUnits Phi81StrongSet.lowNormInvertibility

/-- The Π_CCS output witness that the fork extracts (`Refresh.forkWitness` on the oracle
probe). -/
noncomputable def extractedWitness (succeeds : Fork.Succeeds (game ajtai key) adversary oracle)
    (valid : ∀ index, Fork.ForkValid (game ajtai key) adversary oracle retries index) :
    OutputWitness productionShape (Phi81CarrierLayout.carrierWidth logicalWidth) :=
  Nifs.PaperStrongInterface.outputWitnessOfAssignments (refreshKey key.relation ajtai)
    fun coordinate => extractedAssignment
      (normFreeLaws (PaperExtractionAlgebra.extractionAlgebra ajtai)) strongSet
      (fork ajtai succeeds valid) coordinate

/-- SuperNeo v1.2 Lemma 8 on the fork: the extracted witness opens the base run's 17 Π_CCS
outputs on the oracle probe at the ambient bound. -/
theorem extracted_ambient (succeeds : Fork.Succeeds (game ajtai key) adversary oracle)
    (valid : ∀ index, Fork.ForkValid (game ajtai key) adversary oracle retries index) :
    AmbientOutputHolds (refreshKey key.relation ajtai).extensionOps
      (refreshKey key.relation ajtai).lift (refreshKey key.relation ajtai).openingMaps
      (refreshKey key.relation ajtai).params
      ((refreshKey key.relation ajtai).statement
        (Refresh.children (refreshKey key.relation ajtai)
          (toClaim (key.system ajtai) (Base).link.start.1) (proofOf (Base).link.fold.proof).split)
        (Base).link.fold.fresh)
      (RefreshTest.oracleProbe key.relation ajtai (toClaim (key.system ajtai) (Base).link.start.1)
        oracle (Base).link.fold.fresh (proofOf (Base).link.fold.proof))
      (extractedWitness ajtai succeeds valid) :=
  Nifs.PaperStrongInterface.outputWitnessOfAssignments_ambient (refreshKey key.relation ajtai)
    _ _ _ _ (PiRLC.NormFreeFork.correctedAmbientHolds
      (PaperExtractionAlgebra.extractionAlgebra ajtai) strongSet (fork ajtai succeeds valid))

end Fork

section Bounds

variable (key : CircuitKey logicalWidth publicFits)
  (adversary : OracleComp (Point (RefreshSchedule.schedule logicalWidth publicFits) 8) Answer
    (Attempt logicalWidth publicFits))

/-- The chance that a refresh retry gives a collision of the base and retry prior states. -/
noncomputable def collisionChance
    (index : Fin (RefreshSchedule.schedule logicalWidth publicFits).rhoCount)
    (oracle : Oracle logicalWidth publicFits) : ℝ :=
  ForkLink.collisionChance ajtai key (folds ajtai)
    (fun attempt => (attempt.link.fold.fresh, (proofOf attempt.link.fold.proof).split))
    (fun attempt => RefreshSchedule.messages (proofOf attempt.link.fold.proof).nifs) adversary
    index oracle

/-- `Fork.fork_failure_le` for the refresh fold: a successful run whose retry family is invalid
has probability at most `17 · (Q + 17) · sampleError` plus the expected prior-state collision
chance. -/
theorem fork_failure_le {queries : Nat} (bounded : adversary.QueryBound queries) :
    𝔼 oracle, (if Fork.Succeeds (game ajtai key) adversary oracle then
        ∑ retries, retryWeight (Fork.retrySets (game ajtai key) adversary oracle) retries *
          (if ∀ index, Fork.ForkValid (game ajtai key) adversary oracle retries index
            then 0 else 1)
      else 0) ≤
      17 * (((queries + 17 : Nat) : ℝ) * Fork.sampleError) +
        𝔼 oracle, (if Fork.Succeeds (game ajtai key) adversary oracle then
          ∑ index, collisionChance ajtai key adversary index oracle else 0) := by
  have count : ((RefreshSchedule.schedule logicalWidth publicFits).rhoCount : ℝ) *
      (((queries + (RefreshSchedule.schedule logicalWidth publicFits).rhoCount : Nat) : ℝ) *
        Fork.sampleError) =
      17 * (((queries + 17 : Nat) : ℝ) * Fork.sampleError) := by
    show ((17 : Nat) : ℝ) * (((queries + 17 : Nat) : ℝ) * Fork.sampleError) = _
    norm_num
  have failure := ForkLink.fork_failure_le ajtai key (folds ajtai)
    (statement := fun attempt => (attempt.link.fold.fresh, (proofOf attempt.link.fold.proof).split))
    (fun attempt => RefreshSchedule.messages (proofOf attempt.link.fold.proof).nifs) adversary
    (fun same => congrArg Prod.fst same) bounded
  rw [count] at failure
  exact failure

/-- `Fork.expected_retries_le` for the refresh fold: at most `17 · (Q + 17)` expected
reruns. -/
theorem expected_retries_le {queries : Nat} (bounded : adversary.QueryBound queries) :
    𝔼 oracle, (if Fork.Succeeds (game ajtai key) adversary oracle then
        ∑ index, retryCount (Fork.Hits (game ajtai key) adversary index)
          (Fork.rhoPoint (game ajtai key) adversary index oracle) oracle else 0) ≤
      17 * ((queries + 17 : Nat) : ℝ) := by
  have count : ((RefreshSchedule.schedule logicalWidth publicFits).rhoCount : ℝ) *
      ((queries + (RefreshSchedule.schedule logicalWidth publicFits).rhoCount : Nat) : ℝ) =
      17 * ((queries + 17 : Nat) : ℝ) := by
    show ((17 : Nat) : ℝ) * ((queries + 17 : Nat) : ℝ) = _
    norm_num
  have retries := Fork.expected_retries_le (game ajtai key) adversary bounded
  rw [count] at retries
  exact retries

end Bounds

end NightstreamFPrime.Lifecycle.OptionB.RandomOracle.RefreshFork
