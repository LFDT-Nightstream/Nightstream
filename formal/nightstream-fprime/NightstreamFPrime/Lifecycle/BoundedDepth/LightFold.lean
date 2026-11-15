import NightstreamFPrime.Lifecycle.BoundedDepth.LightPiCCSPhase
import NightstreamFPrime.Lifecycle.BoundedDepth.LightSampler
import NightstreamFPrime.Lifecycle.BoundedDepth.LightPublicInput
import NightstreamFPrime.Lifecycle.BoundedDepth.LightEvaluation
import NightstreamFPrime.Lifecycle.BoundedDepth.Step
import NightstreamFPrime.Lifecycle.BoundedDepth.LightCommitment

/-! The light fold (CW-2.5b). The light Π_CCS phase, one Π_RLC challenge sampler and the three
Π_RLC combinations make the bounded-depth light fold accept the slot's claim, with the output slot
claim that the circuit's words carry (Cyclo, ePrint 2026/359, contribution 2, on SuperNeo v1.2
Sections 7.3 and 7.4). This is the fold term of the light case of `stepHolds_iff`. -/

namespace NightstreamFPrime.Lifecycle.BoundedDepth.LightFold

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.StrongReduction
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Lifecycle.PiCCS.v1_2
open NightstreamFPrime.Lifecycle.PiRLC.v1_2

variable {logicalWidth : Nat}
  {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
    RingCarrierLayout.carrierWidth logicalWidth}

/-- A slot claim as the light fold's running input (`BoundedDepthFolds.toLightRunning`). -/
def lightRunning (slot : SlotClaim logicalWidth publicFits) :
    Spec.Folding.Nifs.PaperNonInteractive.Running K PaperAlgebra.Commitment
      (PaperAlgebra.PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits))
      LightShape where
  point := slot.point
  commitments := fun _ => slot.commitment
  publicInputs := fun _ => slot.publicInput
  evaluations := fun _ =>
    HyperNova.Construction2.BoundedDepthFolds.evaluationsToLight slot.evaluations

theorem toLightRunning_toClaim (system : PaperAlgebra.Structure logicalWidth)
    (slot : SlotClaim logicalWidth publicFits) :
    HyperNova.Construction2.BoundedDepthFolds.toLightRunning (toClaim system slot) =
      some (lightRunning slot) := by
  simp [HyperNova.Construction2.BoundedDepthFolds.toLightRunning, toClaim, lightRunning]

/-- The light Π_CCS sources: the fresh instance first, then the running claim
(`Light.Key.freshSource`, `Light.Key.runningSource`). -/
def freshSource : Fin LightShape.sourceCount := ⟨0, by decide⟩

def runningSource : Fin LightShape.sourceCount := ⟨1, by decide⟩

/-- A source's evaluation family in the Π_CCS output (`StrongReduction.Statement.publicOutput`). -/
def sourceFamily (output : FullOutputCoordinates.FullOutput K LightShape)
    (source : Fin LightShape.sourceCount) : EvaluationFamily K LightShape :=
  ⟨output.padCoordinate source, output.matrixCoordinate source⟩

/-- The evaluation family that the ring blocks of `LightEvaluation` carry. -/
def familyOf (blocks : Fin LightEvaluation.blockCount → RingK) :
    EvaluationFamily K productionShape where
  pad := blocks ⟨0, by decide⟩
  matrix := fun matrix => blocks ⟨matrix.val + 1, Nat.succ_lt_succ matrix.isLt⟩

theorem familyOf_familyBlocks (family : EvaluationFamily K LightShape) :
    familyOf (LightEvaluation.familyBlocks family) =
      HyperNova.Construction2.BoundedDepthFolds.evaluationsFromLight family := by
  simp [familyOf, LightEvaluation.familyBlocks,
    HyperNova.Construction2.BoundedDepthFolds.evaluationsFromLight]
  rfl

/-- The Π_RLC gadgets of the light fold and their offsets. -/
structure RlcLeaves (logicalWidth : Nat)
    (publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
      RingCarrierLayout.carrierWidth logicalWidth) where
  sampler : Sampler.Interface
  samplerOffset : Nat
  commitment : LightCommitment.Interface
  commitmentOffset : Nat
  publicInput : LightPublicInput.Interface logicalWidth publicFits
  publicInputOffset : Nat
  evaluation : LightEvaluation.Interface
  evaluationOffset : Nat

/-- Each Π_RLC gadget's rows hold; the sampler draws coordinate 0 (`lightChallenge`). -/
structure RlcSpecHolds (leaves : RlcLeaves logicalWidth publicFits) (env : Env) : Prop where
  sampler : Sampler.SpecHolds leaves.sampler 0 leaves.samplerOffset env
  commitment : CombinationStep.SpecHolds (LightCommitment.stepInterface leaves.commitment)
    leaves.commitmentOffset env
  publicInput : CombinationStep.SpecHolds (LightPublicInput.stepInterface leaves.publicInput)
    leaves.publicInputOffset env
  evaluation : CombinationStep.SpecHolds (LightEvaluation.stepInterface leaves.evaluation)
    leaves.evaluationOffset env

/-- The sampled `ρ`. -/
def challengeWords (leaves : RlcLeaves logicalWidth publicFits) (env : Env) : RingF :=
  fun position => (Sampler.outputChallenge leaves.samplerOffset position).eval env

/-- The wires of the light Π_RLC: the sampler starts from the state after the Π_CCS output
binding, each combination reads `ρ`, the running inputs are the slot's claim and the Π_CCS
output's running family, and the fresh inputs are the fresh instance and the output's fresh
family. -/
structure RlcWired {degreeBound : Nat}
    (piccs : LightPiCCSPhase.Leaves logicalWidth degreeBound publicFits)
    (leaves : RlcLeaves logicalWidth publicFits) (slot : SlotClaim logicalWidth publicFits)
    (fresh : StepFresh logicalWidth publicFits)
    (output : FullOutputCoordinates.FullOutput K LightShape) (env : Env) : Prop where
  samplerStart : Sampler.evalState env (leaves.sampler.initialState leaves.samplerOffset) =
    List.ofFn (Layer.evalState env (OutputBinding.finalState piccs.output piccs.outputOffset))
  commitmentChallenge : LightCommitment.evalChallenge leaves.commitment leaves.commitmentOffset
    env = challengeWords leaves env
  commitmentRunning : LightCommitment.evalRunning leaves.commitment leaves.commitmentOffset env =
    slot.commitment
  commitmentFresh : LightCommitment.evalFresh leaves.commitment leaves.commitmentOffset env =
    fresh.commitments ⟨0, by decide⟩
  publicInputChallenge : LightPublicInput.evalChallenge leaves.publicInput
    leaves.publicInputOffset env = challengeWords leaves env
  publicInputRunning : LightPublicInput.evalRunning leaves.publicInput leaves.publicInputOffset
    env = slot.publicInput
  publicInputFresh : LightPublicInput.evalFresh leaves.publicInput leaves.publicInputOffset env =
    fresh.publicInputs ⟨0, by decide⟩
  evaluationChallenge : LightEvaluation.evalChallenge leaves.evaluation leaves.evaluationOffset
    env = challengeWords leaves env
  evaluationRunning : LightEvaluation.evalRunning leaves.evaluation leaves.evaluationOffset env =
    LightEvaluation.familyBlocks (sourceFamily output runningSource)
  evaluationFresh : LightEvaluation.evalFresh leaves.evaluation leaves.evaluationOffset env =
    LightEvaluation.familyBlocks (sourceFamily output freshSource)

/-- The slot claim that the circuit's words carry after a light fold. -/
def nextSlot {degreeBound : Nat}
    (piccs : LightPiCCSPhase.Leaves logicalWidth degreeBound publicFits)
    (leaves : RlcLeaves logicalWidth publicFits) (env : Env) :
    SlotClaim logicalWidth publicFits where
  commitment := LightCommitment.evalOutput leaves.commitmentOffset env
  publicInput := LightPublicInput.evalOutput leaves.publicInputOffset env
  point := RoundTranscript.evalRoundPoint piccs.rounds piccs.roundsOffset env
  evaluations := familyOf (LightEvaluation.evalOutput leaves.evaluationOffset env)

private theorem array_eq_singleton {α : Type} {array : Array α} (default : α)
    (size : array.size = 1) : array = #[array.getD 0 default] := by
  apply Array.ext
  · simpa using size
  · intro index inside _
    have zero : index = 0 := by omega
    subst zero
    simp [Array.getD, size]

private theorem point_eq {count : Nat} {left right : CubePoint K count}
    (coordinates : left.coordinates = right.coordinates) : left = right := by
  cases left
  cases right
  simp_all

/-- The light fold term of `stepHolds_iff`: the rows and wires make the bounded-depth light fold
accept the slot's claim with the next slot's claim. -/
theorem spec_implies_fold (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (slot : SlotClaim logicalWidth publicFits) (fresh : StepFresh logicalWidth publicFits)
    (proof : Spec.Folding.Light.Proof LightShape (ProductionKey.degreeBound relation))
    (piccs : LightPiCCSPhase.Leaves logicalWidth (ProductionKey.degreeBound relation) publicFits)
    (leaves : RlcLeaves logicalWidth publicFits) (env : Env)
    (piccsSpec : LightPiCCSPhase.SpecHolds relation piccs env)
    (piccsWired : LightPiCCSPhase.Wired relation ajtai (lightRunning slot)
      (HyperNova.Construction2.BoundedDepthFolds.freshToLight fresh) proof piccs env)
    (rlcSpec : RlcSpecHolds leaves env)
    (rlcWired : RlcWired piccs leaves slot fresh proof.piCcsOutput env) :
    HyperNova.Construction2.BoundedDepthFolds.folds .light (keys relation ajtai)
        (toClaim (refreshKey relation ajtai).relationSource slot) fresh (.light proof) =
      some (toClaim (refreshKey relation ajtai).relationSource (nextSlot piccs leaves env)) := by
  have phase := LightPiCCSPhase.spec_implies_phaseHolds relation ajtai (lightRunning slot)
    (HyperNova.Construction2.BoundedDepthFolds.freshToLight fresh) proof piccs env piccsSpec
    piccsWired
  have challengeEq : challengeWords leaves env =
      lightChallenge ((keys relation ajtai).light.absorbPiCcsOutput
        (LightPiCCS.lightCoins relation ajtai (lightRunning slot)
          (HyperNova.Construction2.BoundedDepthFolds.freshToLight fresh)
          proof.piCcsRounds).finalState proof.piCcsOutput) := by
    refine (LightSampler.challenge_eq leaves.sampler env leaves.samplerOffset
      rlcSpec.sampler).trans ?_
    rw [rlcWired.samplerStart, phase.outgoingState]
  have verified : Spec.Folding.Light.verify (keys relation ajtai).light (lightRunning slot)
      (HyperNova.Construction2.BoundedDepthFolds.freshToLight fresh) proof =
      some ((keys relation ajtai).light.outputFor (lightRunning slot)
        (HyperNova.Construction2.BoundedDepthFolds.freshToLight fresh) proof
        (challengeWords leaves env)) := by
    unfold Spec.Folding.Light.verify
    simp only [phase.accepted, ↓reduceIte]
    rw [challengeEq]
    rfl
  rw [HyperNova.Construction2.BoundedDepthFolds.light_eq_some_iff]
  refine ⟨lightRunning slot, _, toLightRunning_toClaim _ slot, verified, ?_⟩
  let key := (keys relation ajtai).light
  let lightFresh := HyperNova.Construction2.BoundedDepthFolds.freshToLight fresh
  let probe := key.probe (lightRunning slot) lightFresh proof
  let outputs := (key.statement (lightRunning slot) lightFresh).publicOutput probe
  have commitmentEq := LightCommitment.output_eq relation ajtai leaves.commitment
    leaves.commitmentOffset env rlcSpec.commitment key.relationSource probe.coins.roundPoint
    (outputs key.freshSource) (outputs key.runningSource) (challengeWords leaves env)
    rlcWired.commitmentChallenge rlcWired.commitmentRunning rlcWired.commitmentFresh
  have publicInputEq := LightPublicInput.output_eq relation ajtai leaves.publicInput
    leaves.publicInputOffset env rlcSpec.publicInput key.relationSource probe.coins.roundPoint
    (outputs key.freshSource) (outputs key.runningSource) (challengeWords leaves env)
    rlcWired.publicInputChallenge rlcWired.publicInputRunning rlcWired.publicInputFresh
  have evaluationEq := LightEvaluation.output_eq relation ajtai leaves.evaluation
    leaves.evaluationOffset env rlcSpec.evaluation key.relationSource probe.coins.roundPoint
    (outputs key.freshSource) (outputs key.runningSource) rfl (challengeWords leaves env)
    rlcWired.evaluationChallenge rlcWired.evaluationRunning rlcWired.evaluationFresh
  have sizeEq : (Spec.Folding.Light.Rlc.output key.piRlcAlgebra key.one
      (challengeWords leaves env) key.relationSource probe.coins.roundPoint
      (outputs key.freshSource) (outputs key.runningSource)).evaluations.size = 1 :=
    (lightExtraction ajtai).combineEvaluations_size _ _ 1 (by decide) fun source => by
      unfold Spec.Folding.Light.Rlc.pair
      split <;> rfl
  change toClaim _ (nextSlot piccs leaves env) =
    HyperNova.Construction2.BoundedDepthFolds.fromLight (Spec.Folding.Light.Rlc.output
      key.piRlcAlgebra key.one (challengeWords leaves env) key.relationSource
      probe.coins.roundPoint (outputs key.freshSource) (outputs key.runningSource))
  simp only [toClaim, HyperNova.Construction2.BoundedDepthFolds.fromLight, nextSlot]
  congr 1
  · exact point_eq phase.roundPoint
  · rw [array_eq_singleton (lightExtraction ajtai).evaluationModule.zero sizeEq, evaluationEq,
      familyOf_familyBlocks]
    simp only [List.map_toArray, List.map_cons, List.map_nil]
    rfl

/-- An accepted light fold passed the light Π_CCS check: `Light.verify` returns nothing
otherwise. -/
theorem check_of_folds (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (system : PaperAlgebra.Structure logicalWidth) (slot : SlotClaim logicalWidth publicFits)
    (fresh : StepFresh logicalWidth publicFits)
    (proof : Spec.Folding.Light.Proof LightShape (ProductionKey.degreeBound relation))
    (result : StepClaim logicalWidth publicFits)
    (accepted : HyperNova.Construction2.BoundedDepthFolds.folds .light (keys relation ajtai)
      (toClaim system slot) fresh (.light proof) = some result) :
    Spec.Folding.Light.PiCCS.check (keys relation ajtai).light.extensionOps
      (keys relation ajtai).light.lift
      ((keys relation ajtai).light.statement (lightRunning slot)
        (HyperNova.Construction2.BoundedDepthFolds.freshToLight fresh))
      (ProductionKey.degreeBound relation)
      ((keys relation ajtai).light.probe (lightRunning slot)
        (HyperNova.Construction2.BoundedDepthFolds.freshToLight fresh) proof) = true := by
  simp only [HyperNova.Construction2.BoundedDepthFolds.folds, toLightRunning_toClaim,
    Option.bind_some] at accepted
  unfold Spec.Folding.Light.verify at accepted
  by_cases checked : Spec.Folding.Light.PiCCS.check (keys relation ajtai).light.extensionOps
      (keys relation ajtai).light.lift
      ((keys relation ajtai).light.statement (lightRunning slot)
        (HyperNova.Construction2.BoundedDepthFolds.freshToLight fresh))
      (ProductionKey.degreeBound relation)
      ((keys relation ajtai).light.probe (lightRunning slot)
        (HyperNova.Construction2.BoundedDepthFolds.freshToLight fresh) proof) = true
  · exact checked
  · simp only [checked] at accepted
    simp at accepted

/-- `toClaim` keeps every field of the slot claim. -/
theorem toClaim_injective (system : PaperAlgebra.Structure logicalWidth)
    {left right : SlotClaim logicalWidth publicFits}
    (same : toClaim system left = toClaim system right) : left = right := by
  have back := congrArg fromClaim same
  rw [fromClaim_toClaim, fromClaim_toClaim] at back
  exact Option.some.inj back

end NightstreamFPrime.Lifecycle.BoundedDepth.LightFold
