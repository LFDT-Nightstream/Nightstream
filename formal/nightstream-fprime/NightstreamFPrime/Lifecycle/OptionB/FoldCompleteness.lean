import NightstreamFPrime.Lifecycle.OptionB.RandomOracle.Honest
import NightstreamFPrime.Spec.Folding.Light.SourceTruth
import NightstreamFPrime.Spec.Folding.Refresh.Bound
import NightstreamFPrime.Lifecycle.OptionB.BlockSupport

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
(`Honest.slot_of_holds`, `toClaim_of_fromClaim`). No probability is involved.

Both folds keep block support (`BlockSupport.Supported`): the folded witness is zero on each ring
block on which the running witness and the fresh witness are zero (`lightWitness_supported`,
`Refresh.parent_supported` with `BlockSupport.supportLaw`). -/

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
open NightstreamFPrime.Lifecycle.OptionB.BlockSupport

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth}
  (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))

/-- The light Π_RLC witness `ρ · fresh + running` of supported witnesses is supported. -/
theorem lightWitness_supported (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (width : Nat) (challenge : RingF)
    {fresh running : PaperAlgebra.Assignment (logicalWidth := logicalWidth)
      (publicFits := publicFits)}
    (freshSupported : Supported width fresh) (runningSupported : Supported width running) :
    Supported width (Light.Rlc.witness (keys relation ajtai).light.piRlcAlgebra
      (keys relation ajtai).light.one challenge fresh running) := by
  show Supported width (Phi81Relation.EvaluationHomomorphism.PiRLCFinite.combineAssignments
    (Light.Rlc.pair challenge (keys relation ajtai).light.one) (Light.Rlc.pair fresh running))
  apply combineAssignments_supported
  intro index
  unfold Light.Rlc.pair
  split
  · exact freshSupported
  · exact runningSupported

/-- The honest light fold of a slot claim is accepted at the production keys, and its output is a
slot claim that holds at the input bound plus `T(b − 1)`. Its witness is zero on each ring block
on which both source witnesses are zero. -/
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
          (toClaim (key.system ajtai) folded) foldedWitness ∧
        ∀ width, Supported width (witness.assignments (keys key.relation ajtai).light.freshSource) →
          Supported width (witness.assignments (keys key.relation ajtai).light.runningSource) →
          Supported width foldedWitness := by
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
  obtain ⟨foldedWitness, slotHolds, foldedSupported⟩ : ∃ foldedWitness,
      Light.Rlc.Holds (semantics ajtai)
        (bound + (keys key.relation ajtai).light.params.expansionT *
          ((keys key.relation ajtai).light.params.b - 1))
        (fromLight result) foldedWitness ∧
      ∀ width, Supported width (witness.assignments (keys key.relation ajtai).light.freshSource) →
        Supported width (witness.assignments (keys key.relation ajtai).light.runningSource) →
        Supported width foldedWitness :=
    ⟨_, (OptionBKeys.boundHolds_fromLight_iff _ _ _).mpr holds,
      fun width freshSupported runningSupported =>
        lightWitness_supported ajtai key.relation width _ freshSupported runningSupported⟩
  obtain ⟨folded, read⟩ := Honest.slot_of_holds ajtai slotHolds
  have claimEq : fromLight result = toClaim (key.system ajtai) folded := by
    rw [resultEq] at read ⊢
    exact toClaim_of_fromClaim rfl rfl read
  refine ⟨proof, folded, foldedWitness, ?_, claimEq ▸ slotHolds, foldedSupported⟩
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

/-! ### From the terminal's claims -/

/-- The light output witness of a running witness and a fresh witness; `LightUniqueness.answerOf`
reads them back. -/
def witnessOf (running fresh : PaperAlgebra.Assignment (logicalWidth := logicalWidth)
    (publicFits := publicFits)) :
    OutputWitness LightShape (Phi81CarrierLayout.carrierWidth logicalWidth) :=
  ⟨fun source => if source.val = 0 then fresh else running⟩

/-- A slot claim that holds at `bound` and a fresh instance that holds under the slot's relation
give the premises of `light_folds`, the converse of `LightUniqueness.holds_of_knowledge`. -/
theorem light_premises (key : CircuitKey logicalWidth publicFits)
    (claim : SlotClaim logicalWidth publicFits) (fresh : StepFresh logicalWidth publicFits)
    (bound : Nat)
    (running freshWitness : PaperAlgebra.Assignment (logicalWidth := logicalWidth)
      (publicFits := publicFits))
    (slot : ForkLink.SlotHolds ajtai key bound claim running)
    (freshHolds : UniquenessLink.FreshHolds ajtai key fresh freshWitness) :
    Opening.Holds
        (paperRelationSemantics (shape := LightShape)
          (blockCount := Phi81ColumnLayout.blockCount (Phi81CarrierLayout.carrierWidth logicalWidth))
          (keys key.relation ajtai).light.baseOps (keys key.relation ajtai).light.extensionOps
          (keys key.relation ajtai).light.lift (keys key.relation ajtai).light.openingMaps)
        (keys key.relation ajtai).light.params.b
        (((keys key.relation ajtai).light.statement (LightFork.runningOf claim)
          (freshToLight fresh)).commitments (keys key.relation ajtai).light.freshSource)
        (((keys key.relation ajtai).light.statement (LightFork.runningOf claim)
          (freshToLight fresh)).publicInputs (keys key.relation ajtai).light.freshSource)
        ((witnessOf running freshWitness).assignments (keys key.relation ajtai).light.freshSource) ∧
      Opening.Holds
        (paperRelationSemantics (shape := LightShape)
          (blockCount := Phi81ColumnLayout.blockCount (Phi81CarrierLayout.carrierWidth logicalWidth))
          (keys key.relation ajtai).light.baseOps (keys key.relation ajtai).light.extensionOps
          (keys key.relation ajtai).light.lift (keys key.relation ajtai).light.openingMaps)
        bound
        (((keys key.relation ajtai).light.statement (LightFork.runningOf claim)
          (freshToLight fresh)).commitments (keys key.relation ajtai).light.runningSource)
        (((keys key.relation ajtai).light.statement (LightFork.runningOf claim)
          (freshToLight fresh)).publicInputs (keys key.relation ajtai).light.runningSource)
        ((witnessOf running freshWitness).assignments
          (keys key.relation ajtai).light.runningSource) ∧
      LightUniqueness.Truth ajtai key.relation (LightFork.runningOf claim) (freshToLight fresh)
        (witnessOf running freshWitness) := by
  let light := (keys key.relation ajtai).light
  have freshOpening := (light.openingAgreement _ _ _ _).mpr freshHolds.1
  have runningOpening := (light.openingAgreement _ _ _ _).mpr slot.1
  have lightSlot := (OptionBKeys.boundHolds_claimToLight_iff _ _ _).mpr slot
  have runningEvaluations : (paperRelationSemantics (shape := LightShape)
        (blockCount := Phi81ColumnLayout.blockCount (Phi81CarrierLayout.carrierWidth logicalWidth))
        light.baseOps light.extensionOps light.lift light.openingMaps).evaluations
        (light.statement (LightFork.runningOf claim) (freshToLight fresh)).relationSource
        ((witnessOf running freshWitness).assignments
          (UnifiedSources.runningSourceIndex ⟨0, by decide⟩))
        (light.statement (LightFork.runningOf claim) (freshToLight fresh)).priorPoint =
      #[SourceMembership.runningEvaluation
        (light.statement (LightFork.runningOf claim) (freshToLight fresh)) ⟨0, by decide⟩] := by
    refine (light.evaluationAgreement running claim.point).2.symm.trans (lightSlot.2.2.trans ?_)
    simp only [OptionBKeys.claimToLight, toClaim, List.map_toArray, List.map_cons, List.map_nil]
    rfl
  refine ⟨freshOpening, runningOpening, ?_⟩
  apply Light.PiCCS.semanticTruth_of_sources light.extensionOps light.extensionLaws light.lift
    light.openingMaps light.params rfl
  · intro index
    obtain rfl : index = ⟨0, by decide⟩ := Fin.ext (by
      have bound := index.isLt
      change index.val < 1 at bound
      show index.val = 0
      omega)
    exact ⟨freshOpening, freshHolds.2⟩
  · intro index
    obtain rfl : index = ⟨0, by decide⟩ := Fin.ext (by
      have bound := index.isLt
      change index.val < 1 at bound
      show index.val = 0
      omega)
    exact runningEvaluations

/-- The honest light fold of a slot claim that holds at `bound`, with a fresh instance that holds
under the slot's relation, is accepted, and its output holds at `bound + T(b − 1)`. Its witness is
zero on each ring block on which the running and fresh witnesses are zero. -/
theorem light_folds_of_holds (key : CircuitKey logicalWidth publicFits)
    (claim : SlotClaim logicalWidth publicFits) (fresh : StepFresh logicalWidth publicFits)
    (bound : Nat)
    (running freshWitness : PaperAlgebra.Assignment (logicalWidth := logicalWidth)
      (publicFits := publicFits))
    (slot : ForkLink.SlotHolds ajtai key bound claim running)
    (freshHolds : UniquenessLink.FreshHolds ajtai key fresh freshWitness) :
    ∃ (proof : Light.Proof LightShape 8) (folded : SlotClaim logicalWidth publicFits)
      (foldedWitness : PaperAlgebra.Assignment (logicalWidth := logicalWidth)
        (publicFits := publicFits)),
      OptionBFolds.folds .light (keys key.relation ajtai) (toClaim (key.system ajtai) claim) fresh
          (.light proof) = some (toClaim (key.system ajtai) folded) ∧
        Light.Rlc.Holds (semantics ajtai)
          (bound + (keys key.relation ajtai).light.params.expansionT *
            ((keys key.relation ajtai).light.params.b - 1))
          (toClaim (key.system ajtai) folded) foldedWitness ∧
        ∀ width, Supported width running → Supported width freshWitness →
          Supported width foldedWitness := by
  obtain ⟨freshOpening, runningOpening, truth⟩ :=
    light_premises ajtai key claim fresh bound running freshWitness slot freshHolds
  obtain ⟨proof, folded, foldedWitness, accepted, foldedHolds, foldedSupported⟩ :=
    light_folds ajtai key claim fresh (witnessOf running freshWitness) bound freshOpening
      runningOpening truth
  exact ⟨proof, folded, foldedWitness, accepted, foldedHolds,
    fun width runningSupported freshSupported =>
      foldedSupported width freshSupported runningSupported⟩

/-- The honest refresh fold of a slot claim that holds, with a fresh instance that holds under
the slot's relation, is accepted, and its output holds below `B`. -/
theorem refresh_folds_of_holds (key : CircuitKey logicalWidth publicFits)
    (claim : SlotClaim logicalWidth publicFits) (fresh : StepFresh logicalWidth publicFits)
    (assignment freshWitness : PaperAlgebra.Assignment (logicalWidth := logicalWidth)
      (publicFits := publicFits))
    (slot : CE.Holds (semantics ajtai) productionGlobalParams (toClaim (key.system ajtai) claim)
      assignment)
    (freshHolds : UniquenessLink.FreshHolds ajtai key fresh freshWitness) :
    ∃ (proof : RefreshFork.RefreshProof) (folded : SlotClaim logicalWidth publicFits)
      (foldedWitness : PaperAlgebra.Assignment (logicalWidth := logicalWidth)
        (publicFits := publicFits)),
      OptionBFolds.folds .refresh (keys key.relation ajtai) (toClaim (key.system ajtai) claim)
          fresh (.refresh proof) = some (toClaim (key.system ajtai) folded) ∧
        Light.Rlc.Holds (semantics ajtai) productionGlobalParams.bigB
          (toClaim (key.system ajtai) folded) foldedWitness :=
  refresh_folds ajtai key claim fresh assignment (fun _ => freshWitness) slot fun source => by
    obtain rfl : source = ⟨0, by decide⟩ := Fin.ext (by
      have bound := source.isLt
      change source.val < 1 at bound
      show source.val = 0
      omega)
    exact ⟨((refreshKey key.relation ajtai).openingAgreement _ _ _ _).mpr freshHolds.1,
      freshHolds.2⟩

/-- The norm law of the production Π_RLC combination: `count` sources of norm below `b` give a
combination below `count · T · (b − 1) + 1`, as `Refresh.production_countNormLaw`. -/
theorem countNormLaw : Refresh.CountNormLaw (PaperAlgebra.piRlcAlgebra ajtai) := by
  constructor
  intro count challenges assignments valid bounded column
  have fresh : ∀ index, Phi81Relation.assignmentNormBounded 2 (assignments index) := by
    intro index
    simpa [PaperAlgebra.semantics, productionGlobalParams] using bounded index
  have finite := Phi81Relation.PiRLCAlgebra.Norm.Finite.combineAssignments_le challenges
    assignments valid fresh column
  change centeredMagnitude
      (Phi81Relation.EvaluationHomomorphism.PiRLCFinite.combineAssignments challenges
        assignments column) < count * 216 * (2 - 1) + 1
  omega

/-- The honest refresh fold of a slot claim that holds, with a fresh instance that holds under
the slot's relation, is accepted, and its output holds within the depth-0 accumulator bound
(`Refresh.complete_bounded`, SuperNeo v1.2 Definition 22). Its witness is zero on each ring block
on which the slot's and the fresh witnesses are zero. -/
theorem refresh_folds_bounded (key : CircuitKey logicalWidth publicFits)
    (claim : SlotClaim logicalWidth publicFits) (fresh : StepFresh logicalWidth publicFits)
    (assignment freshWitness : PaperAlgebra.Assignment (logicalWidth := logicalWidth)
      (publicFits := publicFits))
    (slot : CE.Holds (semantics ajtai) productionGlobalParams (toClaim (key.system ajtai) claim)
      assignment)
    (freshHolds : UniquenessLink.FreshHolds ajtai key fresh freshWitness) :
    ∃ (proof : RefreshFork.RefreshProof) (folded : SlotClaim logicalWidth publicFits)
      (foldedWitness : PaperAlgebra.Assignment (logicalWidth := logicalWidth)
        (publicFits := publicFits)),
      OptionBFolds.folds .refresh (keys key.relation ajtai) (toClaim (key.system ajtai) claim)
          fresh (.refresh proof) = some (toClaim (key.system ajtai) folded) ∧
        CE.Holds (semantics ajtai) productionGlobalParams (toClaim (key.system ajtai) folded)
          foldedWitness ∧
        (semantics ajtai).normBounded (productionGlobalParams.accumulatorBound 0 + 1)
          foldedWitness ∧
        ∀ width, Supported width assignment → Supported width freshWitness →
          Supported width foldedWitness := by
  let oracle := (Fidelity.deployedOracle (logicalWidth := logicalWidth)
    (publicFits := publicFits)).2
  have deployed : Fidelity.DeployedOn oracle :=
    (Fidelity.deployedOracle_deployed (logicalWidth := logicalWidth) (publicFits := publicFits)).2
  obtain ⟨_, _, honest⟩ := Refresh.complete_bounded (refreshKey key.relation ajtai)
    (countNormLaw ajtai) rfl (toClaim (key.system ajtai) claim) fresh assignment
    (fun _ => freshWitness) rfl rfl slot fun source => by
      obtain rfl : source = ⟨0, by decide⟩ := Fin.ext (by
        have bound := source.isLt
        change source.val < 1 at bound
        show source.val = 0
        omega)
      exact ⟨((refreshKey key.relation ajtai).openingAgreement _ _ _ _).mpr freshHolds.1,
        freshHolds.2⟩
  obtain ⟨proof, parent, parentWitness, _, _, verified, parentHolds, parentNorm, witnessEq⟩ :=
    honest _ rfl
  have romVerified : RefreshFork.verify key.relation ajtai oracle
      (toClaim (key.system ajtai) claim) fresh proof = some parent :=
    (Fidelity.refresh_verify_eq key.relation ajtai (toClaim (key.system ajtai) claim) fresh proof
      deployed).trans verified
  have parentEq : parent = RefreshFork.parentFor key.relation ajtai oracle
      (toClaim (key.system ajtai) claim) fresh proof
      (Fork.rho oracle (fresh, proof.split) (RefreshSchedule.messages proof.nifs)) := by
    unfold RefreshFork.verify at romVerified
    split at romVerified
    · exact (Option.some.inj romVerified).symm
    · cases romVerified
  subst parentEq
  have slotHolds : Light.Rlc.Holds (semantics ajtai) productionGlobalParams.bigB
      (RefreshFork.parentFor key.relation ajtai oracle (toClaim (key.system ajtai) claim) fresh proof
        (Fork.rho oracle (fresh, proof.split) (RefreshSchedule.messages proof.nifs)))
      parentWitness :=
    parentHolds
  obtain ⟨folded, read⟩ := Honest.slot_of_holds ajtai slotHolds
  have claimEq : RefreshFork.parentFor key.relation ajtai oracle (toClaim (key.system ajtai) claim)
      fresh proof (Fork.rho oracle (fresh, proof.split) (RefreshSchedule.messages proof.nifs)) =
        toClaim (key.system ajtai) folded :=
    toClaim_of_fromClaim rfl rfl read
  refine ⟨proof, folded, parentWitness, ?_, claimEq ▸ parentHolds, parentNorm,
    fun width assignmentSupported freshSupported => ?_⟩
  swap
  · rw [witnessEq]
    exact Refresh.parent_supported (refreshKey key.relation ajtai) (supportLaw ajtai width)
      assignment (fun _ => freshWitness) _ assignmentSupported fun _ => freshSupported
  show Refresh.verify (keys key.relation ajtai).refresh (toClaim (key.system ajtai) claim) fresh
    proof = _
  rw [show (keys key.relation ajtai).refresh = refreshKey key.relation ajtai from rfl, verified,
    claimEq]

end NightstreamFPrime.Lifecycle.OptionB.FoldCompleteness
