import NightstreamFPrime.Lifecycle.BoundedDepth.RandomOracle.Fidelity
import NightstreamFPrime.Spec.Folding.Light.Completeness
import NightstreamFPrime.Spec.Folding.Refresh.Completeness
import NightstreamFPrime.Lifecycle.BoundedDepth.RandomOracle.LightUniqueness

/-! The random-oracle success event (`RomModel.Succeeds`) holds for honest folds on a deployed
oracle (`Fidelity.Deployed`), so the success event of `Assumption1.romAssumption1` is not
empty.

`light_honest`: from the premises of Cyclo Theorem 3, item 1 (`Spec.Folding.Light.complete`) on
the slot's claim and the fresh instance, a prior state linked to the fresh instance, a slot
below the light cap, and an output bound within the chain bound, the honest light proof gives a
successful light attempt. `refresh_honest`: from the premises of SuperNeo v1.2 Theorem 13 and
Lemmas 10 and 11 (`Spec.Folding.Refresh.complete`) and a linked prior state, the honest
refresh proof gives a successful refresh attempt at the bound `B`.

Both go through the deployed verifiers (`Fidelity.light_verify_eq`,
`Fidelity.refresh_verify_eq`). On an oracle that is not deployed, the honest prover would have to
compute its sum-check messages from the oracle coins; the completeness theorems above are
stated for the sponge transcript of the key, so this module does not cover that case. -/

set_option autoImplicit false

namespace NightstreamFPrime.Lifecycle.BoundedDepth.RandomOracle.Honest

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

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤ RingCarrierLayout.carrierWidth logicalWidth}
  (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))

/-- A claim that opens under the semantics has one evaluation family, so it is a slot claim. -/
theorem slot_of_holds {claim : StepClaim logicalWidth publicFits}
    {witness : PaperAlgebra.Assignment (logicalWidth := logicalWidth) (publicFits := publicFits)}
    {bound : Nat} (holds : Light.Rlc.Holds (semantics ajtai) bound claim witness) :
    ∃ next, fromClaim claim = some next := by
  have size : claim.evaluations.size = 1 := by
    rw [← holds.2.2]
    rfl
  unfold fromClaim
  rw [dite_eq_left size]
  exact ⟨_, rfl⟩

/-- The honest light fold succeeds on a deployed oracle. -/
theorem light_honest {oracle : RomModel.Oracle logicalWidth publicFits}
    (deployed : Fidelity.Deployed oracle) (key : CircuitKey logicalWidth publicFits)
    (prior : PriorPreimage logicalWidth publicFits) (start : SlotData logicalWidth publicFits)
    (circuit : Fin 2) (lightStep : Spec.HyperNova.Construction2.BoundedDepth.circuitStep circuit = .light)
    (room : start.2 < Folding.Schedule.productionLightFoldCap)
    (fresh : StepFresh logicalWidth publicFits)
    (witness : OutputWitness LightShape (RingCarrierLayout.carrierWidth logicalWidth))
    (bound : Nat)
    (within : bound + (keys key.relation ajtai).light.params.expansionT *
        ((keys key.relation ajtai).light.params.b - 1) ≤
      BoundedDepthReverse.chainBound productionGlobalParams
        Folding.Schedule.productionLightFoldCap)
    (linked : ∀ fold : StepFold logicalWidth publicFits, fold.fresh = fresh →
      PriorLink prior start fold)
    (freshOpening : Opening.Holds
      (paperRelationSemantics (shape := LightShape)
        (blockCount := ConcreteRingColumnLayout.blockCount (RingCarrierLayout.carrierWidth logicalWidth))
        (keys key.relation ajtai).light.baseOps (keys key.relation ajtai).light.extensionOps
        (keys key.relation ajtai).light.lift (keys key.relation ajtai).light.openingMaps)
      (keys key.relation ajtai).light.params.b
      (((keys key.relation ajtai).light.statement (LightFork.runningOf start.1)
        (freshToLight fresh)).commitments (keys key.relation ajtai).light.freshSource)
      (((keys key.relation ajtai).light.statement (LightFork.runningOf start.1)
        (freshToLight fresh)).publicInputs (keys key.relation ajtai).light.freshSource)
      (witness.assignments (keys key.relation ajtai).light.freshSource))
    (runningOpening : Opening.Holds
      (paperRelationSemantics (shape := LightShape)
        (blockCount := ConcreteRingColumnLayout.blockCount (RingCarrierLayout.carrierWidth logicalWidth))
        (keys key.relation ajtai).light.baseOps (keys key.relation ajtai).light.extensionOps
        (keys key.relation ajtai).light.lift (keys key.relation ajtai).light.openingMaps)
      bound
      (((keys key.relation ajtai).light.statement (LightFork.runningOf start.1)
        (freshToLight fresh)).commitments (keys key.relation ajtai).light.runningSource)
      (((keys key.relation ajtai).light.statement (LightFork.runningOf start.1)
        (freshToLight fresh)).publicInputs (keys key.relation ajtai).light.runningSource)
      (witness.assignments (keys key.relation ajtai).light.runningSource))
    (truth : LightUniqueness.Truth ajtai key.relation (LightFork.runningOf start.1)
      (freshToLight fresh) witness) :
    ∃ (proof : Light.Proof LightShape 8) (output : SlotData logicalWidth publicFits)
      (outputWitness : PaperAlgebra.Assignment (logicalWidth := logicalWidth)
        (publicFits := publicFits)),
      RomModel.Succeeds ajtai (oracle,
        ⟨⟨prior, start, ⟨circuit, key, start.1, fresh, .light proof, output⟩⟩,
          bound + (keys key.relation ajtai).light.params.expansionT *
            ((keys key.relation ajtai).light.params.b - 1), outputWitness⟩) := by
  obtain ⟨rounds, honest⟩ := Light.complete (keys key.relation ajtai).light
    (LightFork.runningOf start.1) (freshToLight fresh) witness bound freshOpening runningOpening
    truth
  let proof : Light.Proof LightShape 8 :=
    { piCcsRounds := rounds
      piCcsOutput := FullOutputCoordinates.FullOutput.honestAt
        (keys key.relation ajtai).light.baseOps (keys key.relation ajtai).light.extensionOps
        (keys key.relation ajtai).light.lift
        (((keys key.relation ajtai).light.statement (LightFork.runningOf start.1)
          (freshToLight fresh)).sourceConnectedInputs witness)
        ((keys key.relation ajtai).light.coins (LightFork.runningOf start.1) (freshToLight fresh)
          rounds).roundPoint }
  have sealed := Fidelity.light_challenge_eq key.relation ajtai (LightFork.runningOf start.1)
    (freshToLight fresh) proof deployed.1
  obtain ⟨result, verified, holds⟩ := honest _ sealed
  have romVerified : LightFork.verify key.relation ajtai oracle.1 (LightFork.runningOf start.1)
      (freshToLight fresh) proof = some result :=
    (Fidelity.light_verify_eq key.relation ajtai (LightFork.runningOf start.1) (freshToLight fresh)
      proof deployed.1).trans verified
  have resultEq : result = LightFork.outputFor key.relation ajtai oracle.1
      (LightFork.runningOf start.1) (freshToLight fresh) proof
      (LightFork.challenge oracle.1 (freshToLight fresh) proof) := by
    unfold LightFork.verify at romVerified
    split at romVerified
    · exact (Option.some.inj romVerified).symm
    · cases romVerified
  obtain ⟨outputWitness, slotHolds⟩ : ∃ outputWitness, Light.Rlc.Holds (semantics ajtai)
      (bound + (keys key.relation ajtai).light.params.expansionT *
        ((keys key.relation ajtai).light.params.b - 1))
      (fromLight result) outputWitness :=
    ⟨_, (BoundedDepthKeys.boundHolds_fromLight_iff _ _ _).mpr holds⟩
  obtain ⟨next, read⟩ := slot_of_holds ajtai slotHolds
  have claimEq : fromLight result = toClaim (key.system ajtai) next := by
    rw [resultEq] at read ⊢
    exact toClaim_of_fromClaim rfl rfl read
  have folded : RomModel.folds ajtai oracle (Spec.HyperNova.Construction2.BoundedDepth.circuitStep circuit) key start.1 fresh
      (.light proof) = some next := by
    rw [lightStep]
    calc
      _ = ((LightFork.verify key.relation ajtai oracle.1 (LightFork.runningOf start.1)
            (freshToLight fresh) proof).map fromLight).bind fromClaim := by
        show ((toLightRunning (toClaim (key.system ajtai) start.1)).bind fun running =>
            (LightFork.verify key.relation ajtai oracle.1 running (freshToLight fresh)
              proof).map fromLight).bind fromClaim = _
        rw [LightFork.toLightRunning_toClaim]
        rfl
      _ = ((some result).map fromLight).bind fromClaim := by rw [romVerified]
      _ = some next := read
  refine ⟨proof, (next, Folding.Schedule.nextDepth start.2 (Spec.HyperNova.Construction2.BoundedDepth.circuitStep circuit)),
    outputWitness, linked _ rfl, rfl, ?_, ?_, within⟩
  · exact (Spec.HyperNova.Construction2.BoundedDepth.accepts_iff _ _ _ _ _ _ _ _ _).mpr ⟨fun _ => room, folded, rfl⟩
  · show Light.Rlc.Holds (semantics ajtai) _ (toClaim (key.system ajtai) next) _
    rw [← claimEq]
    exact slotHolds

/-- The honest refresh fold succeeds on a deployed oracle, at the bound `B`. -/
theorem refresh_honest {oracle : RomModel.Oracle logicalWidth publicFits}
    (deployed : Fidelity.Deployed oracle) (key : CircuitKey logicalWidth publicFits)
    (prior : PriorPreimage logicalWidth publicFits) (start : SlotData logicalWidth publicFits)
    (circuit : Fin 2) (refreshStep : Spec.HyperNova.Construction2.BoundedDepth.circuitStep circuit = .refresh)
    (fresh : StepFresh logicalWidth publicFits)
    (assignment : PaperAlgebra.Assignment (logicalWidth := logicalWidth) (publicFits := publicFits))
    (freshWitness : Fin productionShape.freshCount →
      PaperAlgebra.Assignment (logicalWidth := logicalWidth) (publicFits := publicFits))
    (linked : ∀ fold : StepFold logicalWidth publicFits, fold.fresh = fresh →
      PriorLink prior start fold)
    (accumulatorHolds : CE.Holds (refreshKey key.relation ajtai).piRlcSemantics
      (refreshKey key.relation ajtai).params (toClaim (key.system ajtai) start.1) assignment)
    (freshHolds : ∀ source, CCS.Holds
      (paperRelationSemantics (refreshKey key.relation ajtai).baseOps
        (refreshKey key.relation ajtai).extensionOps (refreshKey key.relation ajtai).lift
        (refreshKey key.relation ajtai).openingMaps)
      (refreshKey key.relation ajtai).params
      (SourceMembership.freshInstance
        ((refreshKey key.relation ajtai).statement
          (Refresh.children (refreshKey key.relation ajtai) (toClaim (key.system ajtai) start.1)
            (Refresh.honestSplit (refreshKey key.relation ajtai)
              (toClaim (key.system ajtai) start.1) assignment)) fresh)
        source)
      (freshWitness source)) :
    ∃ (proof : RefreshFork.RefreshProof) (output : SlotData logicalWidth publicFits)
      (outputWitness : PaperAlgebra.Assignment (logicalWidth := logicalWidth)
        (publicFits := publicFits)),
      RomModel.Succeeds ajtai (oracle,
        ⟨⟨prior, start, ⟨circuit, key, start.1, fresh, .refresh proof, output⟩⟩,
          productionGlobalParams.bigB, outputWitness⟩) := by
  obtain ⟨_, _, honest⟩ := Refresh.complete (refreshKey key.relation ajtai)
    (toClaim (key.system ajtai) start.1) fresh assignment freshWitness rfl rfl accumulatorHolds
    freshHolds
  obtain ⟨proof, parent, parentWitness, _, _, verified, parentHolds⟩ := honest _ rfl
  have romVerified : RefreshFork.verify key.relation ajtai oracle.2
      (toClaim (key.system ajtai) start.1) fresh proof = some parent :=
    (Fidelity.refresh_verify_eq key.relation ajtai (toClaim (key.system ajtai) start.1) fresh proof
      deployed.2).trans verified
  have parentEq : parent = RefreshFork.parentFor key.relation ajtai oracle.2
      (toClaim (key.system ajtai) start.1) fresh proof
      (Fork.rho oracle.2 (fresh, proof.split) (RefreshSchedule.messages proof.nifs)) := by
    unfold RefreshFork.verify at romVerified
    split at romVerified
    · exact (Option.some.inj romVerified).symm
    · cases romVerified
  subst parentEq
  have slotHolds : Light.Rlc.Holds (semantics ajtai) productionGlobalParams.bigB
      (RefreshFork.parentFor key.relation ajtai oracle.2 (toClaim (key.system ajtai) start.1)
        fresh proof
        (Fork.rho oracle.2 (fresh, proof.split) (RefreshSchedule.messages proof.nifs)))
      parentWitness :=
    parentHolds
  obtain ⟨next, read⟩ := slot_of_holds ajtai slotHolds
  have claimEq : RefreshFork.parentFor key.relation ajtai oracle.2
      (toClaim (key.system ajtai) start.1) fresh proof
      (Fork.rho oracle.2 (fresh, proof.split) (RefreshSchedule.messages proof.nifs)) =
        toClaim (key.system ajtai) next :=
    toClaim_of_fromClaim rfl rfl read
  have folded : RomModel.folds ajtai oracle (Spec.HyperNova.Construction2.BoundedDepth.circuitStep circuit) key start.1 fresh
      (.refresh proof) = some next := by
    rw [refreshStep]
    calc
      _ = (RefreshFork.verify key.relation ajtai oracle.2 (toClaim (key.system ajtai) start.1)
            fresh proof).bind fromClaim := rfl
      _ = (some (RefreshFork.parentFor key.relation ajtai oracle.2
            (toClaim (key.system ajtai) start.1) fresh proof
            (Fork.rho oracle.2 (fresh, proof.split) (RefreshSchedule.messages proof.nifs)))).bind
              fromClaim := by rw [romVerified]
      _ = some next := read
  refine ⟨proof, (next, Folding.Schedule.nextDepth start.2 (Spec.HyperNova.Construction2.BoundedDepth.circuitStep circuit)),
    parentWitness, linked _ rfl, rfl, ?_, ?_, ?_⟩
  · refine (Spec.HyperNova.Construction2.BoundedDepth.accepts_iff _ _ _ _ _ _ _ _ _).mpr ⟨fun light => ?_, folded, rfl⟩
    rw [refreshStep] at light
    cases light
  · show Light.Rlc.Holds (semantics ajtai) _ (toClaim (key.system ajtai) next) _
    rw [← claimEq]
    exact slotHolds
  · unfold BoundedDepthReverse.chainBound
    exact Nat.le_add_right _ _

end NightstreamFPrime.Lifecycle.BoundedDepth.RandomOracle.Honest
