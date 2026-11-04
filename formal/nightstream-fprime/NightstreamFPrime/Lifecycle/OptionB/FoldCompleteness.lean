import NightstreamFPrime.Lifecycle.OptionB.RandomOracle.Honest

/-! Completeness of the option-B folds at the production keys (CW-10.3). An honest light fold
(`Spec.Folding.Light.complete`, Cyclo Theorem 3, item 1) and an honest refresh fold
(`Spec.Folding.Refresh.complete`, SuperNeo v1.2 Theorem 13 with Lemmas 10 and 11) are accepted by
`OptionBFolds.folds` at `keys`, and the output is a slot claim (`light_folds`, `refresh_folds`).
This is the `accepted` premise of `Layout.OptionB.LightAssemblerCompleteness.complete` and
`Layout.OptionB.RefreshAssemblerCompleteness.complete`.

The light challenge always exists: the Poseidon2 sponge is a deployed oracle
(`Fidelity.deployedOracle_deployed`), and on it the light key's challenge is the oracle's answer
(`Fidelity.light_challenge_eq`). The deployed verifiers also name the folded claim in closed form
(`Fidelity.light_verify_eq`, `Fidelity.refresh_verify_eq`), so the claim reads back as a slot
(`Honest.slot_of_holds`, `toClaim_of_fromClaim`). No probability is involved. -/

set_option autoImplicit false

namespace NightstreamFPrime.Lifecycle.OptionB.FoldCompleteness

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.StrongReduction
open NightstreamFPrime.Spec.HyperNova.Construction2
open NightstreamFPrime.Spec.HyperNova.Construction2.OptionBFolds
  (toLightRunning fromLight freshToLight)
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.OptionB
open NightstreamFPrime.Lifecycle.RandomOracle
open NightstreamFPrime.Lifecycle.OptionB.RandomOracle

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth}
  (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))

/-- The honest light fold of a slot claim is accepted at the production keys, and its output is a
slot claim that holds at the input bound plus `T(b − 1)`. -/
theorem light_folds (key : CircuitKey logicalWidth publicFits)
    (start : SlotClaim logicalWidth publicFits) (fresh : StepFresh logicalWidth publicFits)
    (witness : OutputWitness LightShape (Phi81CarrierLayout.carrierWidth logicalWidth))
    (bound : Nat)
    (freshOpening : Opening.Holds
      (paperRelationSemantics (shape := LightShape)
        (blockCount := Phi81ColumnLayout.blockCount (Phi81CarrierLayout.carrierWidth logicalWidth))
        (keys key.relation ajtai).light.baseOps (keys key.relation ajtai).light.extensionOps
        (keys key.relation ajtai).light.lift (keys key.relation ajtai).light.openingMaps)
      (keys key.relation ajtai).light.params.b
      (((keys key.relation ajtai).light.statement (LightFork.runningOf start)
        (freshToLight fresh)).commitments (keys key.relation ajtai).light.freshSource)
      (((keys key.relation ajtai).light.statement (LightFork.runningOf start)
        (freshToLight fresh)).publicInputs (keys key.relation ajtai).light.freshSource)
      (witness.assignments (keys key.relation ajtai).light.freshSource))
    (runningOpening : Opening.Holds
      (paperRelationSemantics (shape := LightShape)
        (blockCount := Phi81ColumnLayout.blockCount (Phi81CarrierLayout.carrierWidth logicalWidth))
        (keys key.relation ajtai).light.baseOps (keys key.relation ajtai).light.extensionOps
        (keys key.relation ajtai).light.lift (keys key.relation ajtai).light.openingMaps)
      bound
      (((keys key.relation ajtai).light.statement (LightFork.runningOf start)
        (freshToLight fresh)).commitments (keys key.relation ajtai).light.runningSource)
      (((keys key.relation ajtai).light.statement (LightFork.runningOf start)
        (freshToLight fresh)).publicInputs (keys key.relation ajtai).light.runningSource)
      (witness.assignments (keys key.relation ajtai).light.runningSource))
    (truth : LightUniqueness.Truth ajtai key.relation (LightFork.runningOf start)
      (freshToLight fresh) witness) :
    ∃ (proof : Light.Proof LightShape 8) (folded : SlotClaim logicalWidth publicFits)
      (foldedWitness : PaperAlgebra.Assignment (logicalWidth := logicalWidth)
        (publicFits := publicFits)),
      OptionBFolds.folds .light (keys key.relation ajtai) (toClaim (key.system ajtai) start) fresh
          (.light proof) = some (toClaim (key.system ajtai) folded) ∧
        Light.Rlc.Holds (semantics ajtai)
          (bound + (keys key.relation ajtai).light.params.expansionT *
            ((keys key.relation ajtai).light.params.b - 1))
          (toClaim (key.system ajtai) folded) foldedWitness := by
  let oracle := (Fidelity.deployedOracle (logicalWidth := logicalWidth)
    (publicFits := publicFits)).1
  have deployed : Fidelity.DeployedOn oracle :=
    (Fidelity.deployedOracle_deployed (logicalWidth := logicalWidth) (publicFits := publicFits)).1
  obtain ⟨rounds, honest⟩ := Light.complete (keys key.relation ajtai).light
    (LightFork.runningOf start) (freshToLight fresh) witness bound freshOpening runningOpening truth
  let proof : Light.Proof LightShape 8 :=
    { piCcsRounds := rounds
      piCcsOutput := FullOutputCoordinates.FullOutput.honestAt
        (keys key.relation ajtai).light.baseOps (keys key.relation ajtai).light.extensionOps
        (keys key.relation ajtai).light.lift
        (((keys key.relation ajtai).light.statement (LightFork.runningOf start)
          (freshToLight fresh)).sourceConnectedInputs witness)
        ((keys key.relation ajtai).light.coins (LightFork.runningOf start) (freshToLight fresh)
          rounds).roundPoint }
  have sealed := Fidelity.light_challenge_eq key.relation ajtai (LightFork.runningOf start)
    (freshToLight fresh) proof deployed
  obtain ⟨result, verified, holds⟩ := honest _ sealed
  have romVerified : LightFork.verify key.relation ajtai oracle (LightFork.runningOf start)
      (freshToLight fresh) proof = some result :=
    (Fidelity.light_verify_eq key.relation ajtai (LightFork.runningOf start) (freshToLight fresh)
      proof deployed).trans verified
  have resultEq : result = LightFork.outputFor key.relation ajtai oracle
      (LightFork.runningOf start) (freshToLight fresh) proof
      (LightFork.challenge oracle (freshToLight fresh) proof) := by
    unfold LightFork.verify at romVerified
    split at romVerified
    · exact (Option.some.inj romVerified).symm
    · cases romVerified
  obtain ⟨foldedWitness, slotHolds⟩ : ∃ foldedWitness, Light.Rlc.Holds (semantics ajtai)
      (bound + (keys key.relation ajtai).light.params.expansionT *
        ((keys key.relation ajtai).light.params.b - 1))
      (fromLight result) foldedWitness :=
    ⟨_, (OptionBKeys.boundHolds_fromLight_iff _ _ _).mpr holds⟩
  obtain ⟨folded, read⟩ := Honest.slot_of_holds ajtai slotHolds
  have claimEq : fromLight result = toClaim (key.system ajtai) folded := by
    rw [resultEq] at read ⊢
    exact toClaim_of_fromClaim rfl rfl read
  refine ⟨proof, folded, foldedWitness, ?_, claimEq ▸ slotHolds⟩
  show ((toLightRunning (toClaim (key.system ajtai) start)).bind fun running =>
      (Light.verify (keys key.relation ajtai).light running (freshToLight fresh) proof).map
        fromLight) = _
  have verifiedProof : Light.verify (keys key.relation ajtai).light (LightFork.runningOf start)
      (freshToLight fresh) proof = some result := verified
  rw [LightFork.toLightRunning_toClaim, Option.bind_some, verifiedProof, Option.map_some, claimEq]

/-- The honest refresh fold of a slot claim is accepted at the production keys, and its output is
a slot claim that holds below `B`. -/
theorem refresh_folds (key : CircuitKey logicalWidth publicFits)
    (start : SlotClaim logicalWidth publicFits) (fresh : StepFresh logicalWidth publicFits)
    (assignment : PaperAlgebra.Assignment (logicalWidth := logicalWidth) (publicFits := publicFits))
    (freshWitness : Fin productionShape.freshCount →
      PaperAlgebra.Assignment (logicalWidth := logicalWidth) (publicFits := publicFits))
    (accumulatorHolds : CE.Holds (refreshKey key.relation ajtai).piRlcSemantics
      (refreshKey key.relation ajtai).params (toClaim (key.system ajtai) start) assignment)
    (freshHolds : ∀ source, CCS.Holds
      (paperRelationSemantics (refreshKey key.relation ajtai).baseOps
        (refreshKey key.relation ajtai).extensionOps (refreshKey key.relation ajtai).lift
        (refreshKey key.relation ajtai).openingMaps)
      (refreshKey key.relation ajtai).params
      (SourceMembership.freshInstance
        ((refreshKey key.relation ajtai).statement
          (Refresh.children (refreshKey key.relation ajtai) (toClaim (key.system ajtai) start)
            (Refresh.honestSplit (refreshKey key.relation ajtai)
              (toClaim (key.system ajtai) start) assignment)) fresh)
        source)
      (freshWitness source)) :
    ∃ (proof : RefreshFork.RefreshProof) (folded : SlotClaim logicalWidth publicFits)
      (foldedWitness : PaperAlgebra.Assignment (logicalWidth := logicalWidth)
        (publicFits := publicFits)),
      OptionBFolds.folds .refresh (keys key.relation ajtai) (toClaim (key.system ajtai) start)
          fresh (.refresh proof) = some (toClaim (key.system ajtai) folded) ∧
        Light.Rlc.Holds (semantics ajtai) productionGlobalParams.bigB
          (toClaim (key.system ajtai) folded) foldedWitness := by
  let oracle := (Fidelity.deployedOracle (logicalWidth := logicalWidth)
    (publicFits := publicFits)).2
  have deployed : Fidelity.DeployedOn oracle :=
    (Fidelity.deployedOracle_deployed (logicalWidth := logicalWidth) (publicFits := publicFits)).2
  obtain ⟨_, _, honest⟩ := Refresh.complete (refreshKey key.relation ajtai)
    (toClaim (key.system ajtai) start) fresh assignment freshWitness rfl rfl accumulatorHolds
    freshHolds
  obtain ⟨proof, parent, parentWitness, _, _, verified, parentHolds⟩ := honest _ rfl
  have romVerified : RefreshFork.verify key.relation ajtai oracle
      (toClaim (key.system ajtai) start) fresh proof = some parent :=
    (Fidelity.refresh_verify_eq key.relation ajtai (toClaim (key.system ajtai) start) fresh proof
      deployed).trans verified
  have parentEq : parent = RefreshFork.parentFor key.relation ajtai oracle
      (toClaim (key.system ajtai) start) fresh proof
      (Fork.rho oracle (fresh, proof.split) (RefreshSchedule.messages proof.nifs)) := by
    unfold RefreshFork.verify at romVerified
    split at romVerified
    · exact (Option.some.inj romVerified).symm
    · cases romVerified
  subst parentEq
  have slotHolds : Light.Rlc.Holds (semantics ajtai) productionGlobalParams.bigB
      (RefreshFork.parentFor key.relation ajtai oracle (toClaim (key.system ajtai) start) fresh proof
        (Fork.rho oracle (fresh, proof.split) (RefreshSchedule.messages proof.nifs)))
      parentWitness :=
    parentHolds
  obtain ⟨folded, read⟩ := Honest.slot_of_holds ajtai slotHolds
  have claimEq : RefreshFork.parentFor key.relation ajtai oracle (toClaim (key.system ajtai) start)
      fresh proof (Fork.rho oracle (fresh, proof.split) (RefreshSchedule.messages proof.nifs)) =
        toClaim (key.system ajtai) folded :=
    toClaim_of_fromClaim rfl rfl read
  refine ⟨proof, folded, parentWitness, ?_, claimEq ▸ slotHolds⟩
  show Refresh.verify (keys key.relation ajtai).refresh (toClaim (key.system ajtai) start) fresh
    proof = _
  rw [show (keys key.relation ajtai).refresh = refreshKey key.relation ajtai from rfl, verified,
    claimEq]

end NightstreamFPrime.Lifecycle.OptionB.FoldCompleteness
