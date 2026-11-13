import NightstreamFPrime.Lifecycle.BoundedDepth.RandomOracle.RomModel
import NightstreamFPrime.Spec.Folding.Nifs.PaperStrongInterface

/-! The random-oracle fold verifiers are the deployed ones at the sponge oracle.

An oracle of a schedule is deployed (`DeployedOn`) when, at each address of the schedule, its
value is the Poseidon2 sponge read: `decodeK` of the answer is the `K` coin of the state after
the address's calls, and `readRho` of the answer is the `ρ` read of that state. The sponge as an
oracle (`spongeOracle`) is deployed (`spongeOracle_deployed`). A bounded-depth oracle is deployed
(`Deployed`) when its light and refresh parts are.

On a deployed oracle, the oracle coins are the sponge coins (`coins_eq`, through
`LightSchedule.probe_coins` and `RefreshSchedule.piCcsProbe_coins`), and the oracle `ρ`
challenges are the sponge challenges (through `LightSchedule.challenge_seal` and
`RefreshSchedule.rho_seal`). So the light verifier is `Light.verify` of the light key of
`BoundedDepth.keys` (`light_verify_eq`), the refresh verifier is `Refresh.verify` of
`BoundedDepth.refreshKey` (`refresh_verify_eq`), the bounded-depth folds are `slotFolds`
(`folds_eq_slotFolds`), and the success event is `ForkLink.Succeeds` with `slotFolds`
(`succeeds_iff`), which is the walk's success event (`Export.BoundedDepth.RomWalk`).

A deployed oracle is one fixed function. This is a match of events at that function, not a
statement about probabilities. -/

set_option autoImplicit false

namespace NightstreamFPrime.Lifecycle.BoundedDepth.RandomOracle.Fidelity

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.StrongReduction
open NightstreamFPrime.Spec.HyperNova.Construction2
open NightstreamFPrime.Spec.HyperNova.Construction2.BoundedDepthFolds
  (toLightRunning fromLight freshToLight)
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.BoundedDepth
open NightstreamFPrime.Lifecycle.RandomOracle
open NightstreamFPrime.Lifecycle.BoundedDepth.RandomOracle.ForkLink (Attempt)
open NightstreamFPrime.Lifecycle.BoundedDepth.RandomOracle.RomModel (Oracle)

section Generic

variable {schedule : Schedule} {width : Nat}

/-- The value of one challenge that the transcript at `statement` and `messages` reads from
`oracle`. -/
noncomputable def oracleValue (oracle : Test.Point schedule width → Test.Answer)
    (statement : schedule.Statement) (messages : schedule.Messages width) :
    (challenge : schedule.Challenge) → challenge.Value
  | .alpha coordinate => Test.decodeK (oracle (Test.point statement messages (.alpha coordinate)))
  | .gamma => Test.decodeK (oracle (Test.point statement messages .gamma))
  | .round index => Test.decodeK (oracle (Test.point statement messages (.round index)))
  | .rho index => Fork.readRho (oracle (Test.point statement messages (.rho index)))

/-- The oracle reads each address of the schedule as the Poseidon2 sponge does. -/
def DeployedOn (oracle : Test.Point schedule width → Test.Answer) : Prop :=
  ∀ statement messages challenge, oracleValue oracle statement messages challenge =
    challenge.read (Sponge.run Transcript.initialState
      (schedule.challengeCalls statement messages challenge))

private def stateAnswer (slot : Sponge.Slot) (state : Transcript.State) : Test.Answer :=
  fun word => state.getD (2 * slot.val + word.val) 0

/-- The sponge as an oracle: the state after a point's calls, read at the point's lane pair. -/
noncomputable def spongeOracle : Test.Point schedule width → Test.Answer := fun target =>
  stateAnswer target.2 (Sponge.run Transcript.initialState target.1.val)

theorem spongeOracle_deployed : DeployedOn (spongeOracle (schedule := schedule) (width := width)) := by
  intro statement messages challenge
  cases challenge with
  | rho index =>
      show Fork.readRho (stateAnswer (Sponge.slotOf 0) (Sponge.run Transcript.initialState
          (schedule.challengeCalls statement messages (.rho index)))) =
        Sponge.readRho _
      have block : stateAnswer (Sponge.slotOf 0) (Sponge.run Transcript.initialState
          (schedule.challengeCalls statement messages (.rho index))) =
          Spec.Folding.Nifs.NonInteractive.PiRlcSampler.Transcript.block
            (Sponge.run Transcript.initialState
              (schedule.challengeCalls statement messages (.rho index))) := by
        funext word
        simp [stateAnswer, Sponge.slotOf,
          Spec.Folding.Nifs.NonInteractive.PiRlcSampler.Transcript.block]
      rw [block]
      rfl
  | _ => rfl

variable {oracle : Test.Point schedule width → Test.Answer}

/-- On a deployed oracle, the oracle coins are the sponge coins. -/
theorem coins_eq (deployed : DeployedOn oracle) (statement : schedule.Statement)
    (messages : schedule.Messages width) :
    Test.coins oracle statement messages =
      schedule.coinsFrom Schedule.spongeRead statement messages := by
  have alpha (coordinate : Fin schedule.shape.cubeVariables) :
      Test.read oracle (schedule.challengeAddress statement messages (.alpha coordinate)) =
        Schedule.spongeRead (schedule.challengeAddress statement messages (.alpha coordinate)) :=
    (Test.read_challengeAddress oracle statement messages _).trans
      (deployed statement messages (.alpha coordinate))
  have gamma : Test.read oracle (schedule.challengeAddress statement messages .gamma) =
      Schedule.spongeRead (schedule.challengeAddress statement messages .gamma) :=
    (Test.read_challengeAddress oracle statement messages _).trans
      (deployed statement messages .gamma)
  have round (index : Fin schedule.shape.cubeVariables) :
      Test.read oracle (schedule.challengeAddress statement messages (.round index)) =
        Schedule.spongeRead (schedule.challengeAddress statement messages (.round index)) :=
    (Test.read_challengeAddress oracle statement messages _).trans
      (deployed statement messages (.round index))
  simp only [Test.coins, Schedule.coinsFrom, alpha, gamma, round]

/-- On a deployed oracle, the oracle `ρ` challenges are the sponge reads. -/
theorem rho_eq (deployed : DeployedOn oracle) (statement : schedule.Statement)
    (messages : schedule.Messages width) (index : Fin schedule.rhoCount) :
    Fork.rho oracle statement messages index =
      Sponge.readRho (Sponge.run Transcript.initialState
        (schedule.challengeCalls statement messages (.rho index))) :=
  deployed statement messages (.rho index)

end Generic

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤ RingCarrierLayout.carrierWidth logicalWidth}

/-- Both parts of a bounded-depth oracle read their schedule as the Poseidon2 sponge does. -/
def Deployed (oracle : Oracle logicalWidth publicFits) : Prop :=
  DeployedOn oracle.1 ∧ DeployedOn oracle.2

/-- The sponge as a bounded-depth oracle. -/
noncomputable def deployedOracle : Oracle logicalWidth publicFits :=
  (spongeOracle, spongeOracle)

theorem deployedOracle_deployed : Deployed (deployedOracle (logicalWidth := logicalWidth)
    (publicFits := publicFits)) :=
  ⟨spongeOracle_deployed, spongeOracle_deployed⟩

section Light

variable (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
  (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
  {oracle : LightFork.Oracle logicalWidth publicFits}
  (running : LightFork.Running logicalWidth publicFits)
  (fresh : LightFork.Statement logicalWidth publicFits) (proof : Light.Proof LightShape 8)

theorem light_probe_eq (deployed : DeployedOn oracle) :
    LightTest.oracleProbe relation ajtai running oracle fresh proof =
      (keys relation ajtai).light.probe running fresh proof := by
  have coinsEq : Test.coins oracle fresh (LightSchedule.messages proof) =
      ((keys relation ajtai).light.probe running fresh proof).coins :=
    (coins_eq deployed fresh (LightSchedule.messages proof)).trans
      (LightSchedule.probe_coins relation ajtai running fresh proof).symm
  unfold LightTest.oracleProbe
  exact congrArg (fun coins =>
    ({ (keys relation ajtai).light.probe running fresh proof with coins := coins } :
      Probe K LightShape)) coinsEq

theorem light_challenge_eq (deployed : DeployedOn oracle) :
    (keys relation ajtai).light.challenge running fresh proof =
      some (LightFork.challenge oracle fresh proof) := by
  rw [LightSchedule.challenge_seal relation ajtai running fresh proof]
  exact congrArg some (rho_eq deployed fresh (LightSchedule.messages proof) LightFork.first).symm

/-- On a deployed oracle, the random-oracle light verifier is `Light.verify` of the light key. -/
theorem light_verify_eq (deployed : DeployedOn oracle) :
    LightFork.verify relation ajtai oracle running fresh proof =
      Light.verify (keys relation ajtai).light running fresh proof := by
  have outputEq (rho : RingF) :
      LightFork.outputFor relation ajtai oracle running fresh proof rho =
        (keys relation ajtai).light.outputFor running fresh proof rho := by
    unfold LightFork.outputFor LightFork.outputs Light.Key.outputFor
    rw [light_probe_eq relation ajtai running fresh proof deployed]
  unfold LightFork.verify Light.verify
  rw [light_probe_eq relation ajtai running fresh proof deployed,
    light_challenge_eq relation ajtai running fresh proof deployed, Option.map_some, outputEq]

end Light

section Refresh

variable (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
  (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
  {oracle : RefreshFork.Oracle logicalWidth publicFits}
  (accumulator : RefreshFork.Accumulator logicalWidth publicFits)
  (fresh : StepFresh logicalWidth publicFits) (proof : RefreshFork.RefreshProof)

theorem refresh_probe_eq (deployed : DeployedOn oracle) :
    RefreshTest.oracleProbe relation ajtai accumulator oracle fresh proof =
      (refreshKey relation ajtai).piCcsProbe
        (Refresh.children (refreshKey relation ajtai) accumulator proof.split) fresh proof.nifs := by
  have coinsEq : Test.coins oracle (fresh, proof.split) (RefreshSchedule.messages proof.nifs) =
      ((refreshKey relation ajtai).piCcsProbe
        (Refresh.children (refreshKey relation ajtai) accumulator proof.split) fresh
        proof.nifs).coins :=
    (coins_eq deployed _ _).trans (RefreshSchedule.piCcsProbe_coins relation ajtai
      (Refresh.children (refreshKey relation ajtai) accumulator proof.split) fresh proof.nifs).symm
  unfold RefreshTest.oracleProbe
  exact congrArg (fun coins =>
    ({ (refreshKey relation ajtai).piCcsProbe
        (Refresh.children (refreshKey relation ajtai) accumulator proof.split) fresh proof.nifs
      with coins := coins } : Probe K productionShape)) coinsEq

theorem refresh_parent_eq (deployed : DeployedOn oracle) :
    (refreshKey relation ajtai).parent
        (Refresh.children (refreshKey relation ajtai) accumulator proof.split) fresh proof.nifs =
      some (RefreshFork.parentFor relation ajtai oracle accumulator fresh proof
        (Fork.rho oracle (fresh, proof.split) (RefreshSchedule.messages proof.nifs))) := by
  have challenges : Fork.rho oracle (fresh, proof.split) (RefreshSchedule.messages proof.nifs) =
      fun index => Sponge.readRho (Sponge.run Transcript.initialState
        ((RefreshSchedule.schedule logicalWidth publicFits).challengeCalls (fresh, proof.split)
          (RefreshSchedule.messages proof.nifs) (.rho index))) :=
    funext fun index => rho_eq deployed _ _ index
  unfold Nifs.PaperNonInteractive.Key.parent
  rw [RefreshSchedule.rho_seal relation ajtai
    (Refresh.children (refreshKey relation ajtai) accumulator proof.split) fresh proof.nifs,
    challenges]
  refine congrArg some ?_
  unfold RefreshFork.parentFor RefreshFork.batch
  rw [refresh_probe_eq relation ajtai accumulator fresh proof deployed]
  rfl

/-- On a deployed oracle, the random-oracle refresh verifier is `Refresh.verify` of
`BoundedDepth.refreshKey`. -/
theorem refresh_verify_eq (deployed : DeployedOn oracle) :
    RefreshFork.verify relation ajtai oracle accumulator fresh proof =
      Refresh.verify (refreshKey relation ajtai) accumulator fresh proof := by
  have checked := Nifs.PaperNonInteractive.piCcsCheck_eq_true_iff_fixedWidthAccepted
    (refreshKey relation ajtai)
    (Refresh.children (refreshKey relation ajtai) accumulator proof.split) fresh proof.nifs
  rw [← refresh_probe_eq relation ajtai accumulator fresh proof deployed] at checked
  unfold RefreshFork.verify Refresh.verify
  rw [← refresh_parent_eq relation ajtai accumulator fresh proof deployed]
  by_cases passes : Refresh.inputCheck (refreshKey relation ajtai) accumulator proof.split = true ∧
      (RefreshTest.oracleProbe relation ajtai accumulator oracle fresh proof).FixedWidthAccepted
        (refreshKey relation ajtai).extensionOps (refreshKey relation ajtai).lift
        ((refreshKey relation ajtai).statement
          (Refresh.children (refreshKey relation ajtai) accumulator proof.split) fresh)
        (ProductionKey.degreeBound relation)
  · have both : (Refresh.inputCheck (refreshKey relation ajtai) accumulator proof.split &&
        Nifs.PaperNonInteractive.piCcsCheck (refreshKey relation ajtai)
          (Refresh.children (refreshKey relation ajtai) accumulator proof.split) fresh
          proof.nifs) = true := by
      rw [Bool.and_eq_true]
      exact ⟨passes.1, checked.mpr passes.2⟩
    rw [ite_eq_left passes, ite_eq_left both]
  · have neither : ¬ (Refresh.inputCheck (refreshKey relation ajtai) accumulator proof.split &&
        Nifs.PaperNonInteractive.piCcsCheck (refreshKey relation ajtai)
          (Refresh.children (refreshKey relation ajtai) accumulator proof.split) fresh
          proof.nifs) = true := by
      rw [Bool.and_eq_true]
      exact fun both => passes ⟨both.1, checked.mp both.2⟩
    rw [ite_eq_right passes, ite_eq_right neither]

end Refresh

section Folds

variable (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))

/-- On a deployed oracle, the random-oracle bounded-depth folds are `slotFolds`. -/
theorem folds_eq_slotFolds {oracle : Oracle logicalWidth publicFits} (deployed : Deployed oracle) :
    RomModel.folds ajtai oracle = slotFolds ajtai := by
  funext step key claim fresh proof
  cases step with
  | light =>
      cases proof with
      | light proof =>
          have inner : (fun running => (LightFork.verify key.relation ajtai oracle.1 running
                (freshToLight fresh) proof).map fromLight) =
              fun running => (Light.verify (keys key.relation ajtai).light running
                (freshToLight fresh) proof).map fromLight :=
            funext fun running => congrArg (Option.map fromLight)
              (light_verify_eq key.relation ajtai running (freshToLight fresh) proof deployed.1)
          exact congrArg (fun next =>
            ((toLightRunning (toClaim (key.system ajtai) claim)).bind next).bind fromClaim) inner
      | refresh _ => rfl
  | refresh =>
      cases proof with
      | light _ => rfl
      | refresh proof =>
          show (RefreshFork.verify key.relation ajtai oracle.2 (toClaim (key.system ajtai) claim)
              fresh proof).bind fromClaim = _
          rw [refresh_verify_eq key.relation ajtai _ fresh proof deployed.2]
          rfl

/-- On a deployed oracle, the random-oracle success event is `ForkLink.Succeeds` with
`slotFolds`. -/
theorem succeeds_iff {oracle : Oracle logicalWidth publicFits} (deployed : Deployed oracle)
    (attempt : Attempt logicalWidth publicFits) :
    RomModel.Succeeds ajtai (oracle, attempt) ↔ ForkLink.Succeeds ajtai (slotFolds ajtai) attempt := by
  unfold RomModel.Succeeds
  rw [folds_eq_slotFolds ajtai deployed]

end Folds

end NightstreamFPrime.Lifecycle.BoundedDepth.RandomOracle.Fidelity
