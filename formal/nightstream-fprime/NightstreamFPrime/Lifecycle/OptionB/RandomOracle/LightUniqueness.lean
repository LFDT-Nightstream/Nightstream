import NightstreamFPrime.Lifecycle.OptionB.RandomOracle.LightFork
import NightstreamFPrime.Lifecycle.OptionB.RandomOracle.UniquenessLink
import NightstreamFPrime.Lifecycle.Nifs.BindingBridge
import NightstreamFPrime.Spec.Folding.Light.Security
import NightstreamFPrime.Spec.Folding.Light.RunningBinding
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.SourceMembership
import NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.ChainBinding

/-! The light fold's uniqueness step in the random-oracle model: `Uniqueness` on the light game
`LightFork.game`.

Extraction. A valid pair's fork (`LightFork.fork`) gives the light knowledge outcome on the
oracle probe when the extracted witness makes the light Π_CCS source claims true (`Truth`,
`Light.knowledge_of_truth`). The outcome opens the input claim at the base bound plus
`T · (b − 1)` and the fresh instance under the circuit key's relation (`holds_of_knowledge`), so
the pair gives the walk's `Extracted` (`extracted_of_truth`). Otherwise the pair fails
(`Fails`): its source claims are false.

Agreement. A valid resampled run with the failing pair's statement, start claim and extracted
witness passes the light Π_CCS check on its oracle probe, and the witness opens that probe's
outputs (`Light.fork_ambientOutputHolds`). So it is a light false acceptance of the failing
pair's data, a hit of the bad sets of `LightTest.source` (`agree`).

Binding. Two valid pairs with one statement and one start claim whose extracted witnesses
differ give a kernel vector of the Ajtai key below the chain norm (`rerunKernel`,
`rerunKernel_isSome`): different fresh witnesses through `Light.Rlc.collisionAbove`, one fresh
and different running witnesses through `Light.Rlc.runningCollisionAbove`, both on the forks at
the chain bound (`boundedFork`), then `Binding.chainCollision_to_shortKernel`. So the collision
chance is at most the kernel chance (`collisionChance_le_kernelChance`). -/

set_option autoImplicit false

namespace NightstreamFPrime.Lifecycle.OptionB.RandomOracle.LightUniqueness

open scoped BigOperators
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.RandomOracle
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.StrongReduction
open NightstreamFPrime.Spec.HyperNova.Construction2
open NightstreamFPrime.Spec.HyperNova.Construction2.OptionBFolds (freshToLight evaluationsToLight)
open NightstreamFPrime.Spec.Folding.PiRLC.NormFreeFork (normFreeLaws)
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.OptionB
open NightstreamFPrime.Lifecycle.RandomOracle
open NightstreamFPrime.Lifecycle.RandomOracle.Test (Answer Point)
open NightstreamFPrime.Lifecycle.OptionB.RandomOracle.ForkLink (Attempt)

attribute [local instance low] Classical.propDecidable

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth}
  (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))

variable (logicalWidth publicFits) in
/-- A retry family of the light fork: one answer at the one `ρ` point. -/
abbrev Retries := Fin (LightSchedule.schedule logicalWidth publicFits).rhoCount → Answer

local notation "LightWitness" =>
  OutputWitness LightShape (Phi81CarrierLayout.carrierWidth logicalWidth)
local notation "Oracle" => LightFork.Oracle logicalWidth publicFits

section Knowledge

/-- The light Π_CCS source claims on `running` and `fresh` are true for `witness`. -/
def Truth (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (running : LightFork.Running logicalWidth publicFits)
    (fresh : LightFork.Statement logicalWidth publicFits)
    (witness : OutputWitness LightShape (Phi81CarrierLayout.carrierWidth logicalWidth)) : Prop :=
  Light.PiCCS.SemanticTruth
    ((((keys relation ajtai).light.statement running fresh).sourceConnectedInputs
      witness).toUnifiedInputs (keys relation ajtai).light.baseOps)
    (keys relation ajtai).light.baseOps (keys relation ajtai).light.extensionOps
    (keys relation ajtai).light.lift

/-- The answer of a light output witness: the running witness, then the fresh witness. -/
def answerOf (witness : OutputWitness LightShape (Phi81CarrierLayout.carrierWidth logicalWidth)) :
    PaperAlgebra.Assignment (logicalWidth := logicalWidth) (publicFits := publicFits) ×
      PaperAlgebra.Assignment (logicalWidth := logicalWidth) (publicFits := publicFits) :=
  (witness.assignments ⟨1, by decide⟩, witness.assignments ⟨0, by decide⟩)

/-- The light knowledge outcome opens the slot claim at the bound plus `T · (b − 1)`, with true
evaluations, and the fresh instance under the circuit key's relation. -/
theorem holds_of_knowledge {key : CircuitKey logicalWidth publicFits}
    {claim : SlotClaim logicalWidth publicFits} {fresh : StepFresh logicalWidth publicFits}
    {bound : Nat} {witness : LightWitness}
    (knowledge : Light.Knowledge (keys key.relation ajtai).light (LightFork.runningOf claim)
      (freshToLight fresh) bound witness) :
    ForkLink.SlotHolds ajtai key
        (Folding.Schedule.extractedBefore productionGlobalParams .light bound) claim
        (answerOf witness).1 ∧
      UniquenessLink.FreshHolds ajtai key fresh (answerOf witness).2 := by
  obtain ⟨freshOpening, runningOpening, truth⟩ := knowledge
  have evaluations := (SourceMembership.runningEvaluations_eq_iff
    (keys key.relation ajtai).light.extensionOps (keys key.relation ajtai).light.extensionLaws
    (keys key.relation ajtai).light.lift (keys key.relation ajtai).light.openingMaps
    ((keys key.relation ajtai).light.statement (LightFork.runningOf claim) (freshToLight fresh))
    witness ⟨0, by decide⟩).mpr
    ⟨fun coefficient => truth.2.2.1 ⟨⟨0, by decide⟩, coefficient⟩,
      fun matrix coefficient => truth.2.2.2 ⟨⟨0, by decide⟩, matrix, coefficient⟩⟩
  refine ⟨?_, ?_⟩
  · rw [ForkLink.SlotHolds, ← OptionBKeys.boundHolds_claimToLight_iff]
    refine ⟨runningOpening, trivial, ?_⟩
    refine ((keys key.relation ajtai).light.evaluationAgreement (answerOf witness).1
      claim.point).2.trans (evaluations.trans ?_)
    simp only [OptionBKeys.claimToLight, toClaim, List.map_toArray, List.map_cons, List.map_nil]
    rfl
  · exact ⟨((keys key.relation ajtai).light.openingAgreement _ _ _ _).mp freshOpening,
      truth.1 ⟨0, by decide⟩⟩

/-- A light fold that the light game's fold verifiers accept is a light step. -/
theorem folds_light {oracle : Oracle} {step : Folding.Schedule.Step}
    {key : CircuitKey logicalWidth publicFits} {claim : SlotClaim logicalWidth publicFits}
    {fresh : StepFresh logicalWidth publicFits} {proof : StepProof}
    {next : SlotClaim logicalWidth publicFits}
    (accepted : LightFork.folds ajtai oracle step key claim fresh proof = some next) :
    step = .light := by
  cases step with
  | light => rfl
  | refresh => cases proof <;> exact absurd accepted (by simp [LightFork.folds])

variable {key : CircuitKey logicalWidth publicFits}
  {adversary : OracleComp (Point (LightSchedule.schedule logicalWidth publicFits) 8) Answer
    (Attempt logicalWidth publicFits)}
  {oracle : LightFork.Oracle logicalWidth publicFits} {retries : Retries logicalWidth publicFits}

/-- The light knowledge outcome on the oracle probe gives the walk's `Extracted`: a valid pair
whose extracted witness makes the light source claims true opens the input claim at the base
bound plus `T · (b − 1)` and the fresh instance. -/
theorem extracted_of_truth (succeeds : Fork.Succeeds (LightFork.game ajtai key) adversary oracle)
    (valid : ∀ index, Fork.ForkValid (LightFork.game ajtai key) adversary oracle retries index)
    (truth : Truth ajtai key.relation (LightFork.runningOf (adversary.run oracle).link.start.1)
      (freshToLight (adversary.run oracle).link.fold.fresh)
      (LightFork.extractedWitness ajtai succeeds valid)) :
    UniquenessLink.Extracted ajtai (adversary.run oracle)
      (answerOf (LightFork.extractedWitness ajtai succeeds valid)) := by
  have knowledge := Light.knowledge_of_truth (keys key.relation ajtai).light
    (LightFork.runningOf (adversary.run oracle).link.start.1)
    (freshToLight (adversary.run oracle).link.fold.fresh)
    (LightTest.oracleProbe key.relation ajtai
      (LightFork.runningOf (adversary.run oracle).link.start.1) oracle
      (freshToLight (adversary.run oracle).link.fold.fresh)
      (LightFork.proofOf (adversary.run oracle).link.fold.proof))
    (adversary.run oracle).bound (lightExtraction ajtai) LightFork.strongSet rfl rfl
    (lightExtractionNormLaw ajtai) (LightFork.fork ajtai succeeds valid)
    (LightFork.fork_baseNorm ajtai succeeds valid) truth
  have keyEq : (adversary.run oracle).link.fold.key = key := succeeds.1
  have inputEq : (adversary.run oracle).link.fold.input = (adversary.run oracle).link.start.1 :=
    succeeds.2.2.1
  have stepLight : BoundedDepth.circuitStep (adversary.run oracle).link.fold.circuit = .light :=
    folds_light ajtai ((BoundedDepth.accepts_iff _ _ _ _ _ _ _ _ _).mp succeeds.2.2.2.1).2.1
  show ForkLink.SlotHolds ajtai (adversary.run oracle).link.fold.key
      (Folding.Schedule.extractedBefore productionGlobalParams
        (BoundedDepth.circuitStep (adversary.run oracle).link.fold.circuit)
        (adversary.run oracle).bound)
      (adversary.run oracle).link.fold.input
      (answerOf (LightFork.extractedWitness ajtai succeeds valid)).1 ∧
    UniquenessLink.FreshHolds ajtai (adversary.run oracle).link.fold.key
      (adversary.run oracle).link.fold.fresh
      (answerOf (LightFork.extractedWitness ajtai succeeds valid)).2
  rw [stepLight, inputEq, keyEq]
  exact holds_of_knowledge ajtai knowledge

end Knowledge

section Uniqueness

variable (key : CircuitKey logicalWidth publicFits)
  (adversary : OracleComp (Point (LightSchedule.schedule logicalWidth publicFits) 8) Answer
    (Attempt logicalWidth publicFits))

/-- The witness that a valid pair extracts. -/
noncomputable def witnessOf (oracle : LightFork.Oracle logicalWidth publicFits)
    (retries : Fin (LightSchedule.schedule logicalWidth publicFits).rhoCount → Answer) :
    Option (OutputWitness LightShape (Phi81CarrierLayout.carrierWidth logicalWidth)) :=
  if valid : Uniqueness.Valid (LightFork.game ajtai key) adversary oracle retries then
    some (LightFork.extractedWitness ajtai valid.1 valid.2)
  else none

def zeroWitness : OutputWitness LightShape (Phi81CarrierLayout.carrierWidth logicalWidth) :=
  ⟨fun _ _ => 0⟩

/-- The claimed light Π_CCS data of a pair: the base run's start claim, read through the pair's
extracted witness. -/
noncomputable def source
    (pair : LightFork.Oracle logicalWidth publicFits ×
      (Fin (LightSchedule.schedule logicalWidth publicFits).rhoCount → Answer))
    (statement : LightFork.Statement logicalWidth publicFits) :
    ProtocolPolynomial.Data K (LightSchedule.schedule logicalWidth publicFits).shape :=
  LightTest.source key.relation ajtai (LightFork.runningOf (adversary.run pair.1).link.start.1)
    ((witnessOf ajtai key adversary pair.1 pair.2).getD zeroWitness) statement

theorem source_degreeCovers
    (pair : LightFork.Oracle logicalWidth publicFits ×
      (Fin (LightSchedule.schedule logicalWidth publicFits).rhoCount → Answer))
    (statement : LightFork.Statement logicalWidth publicFits) :
    (source ajtai key adversary pair statement).toVerifierInput.sumcheckDegreeBound ≤ 8 :=
  LightTest.source_degreeCovers key.relation ajtai _ _ statement

/-- A valid pair whose extracted witness makes a light source claim false. -/
def Fails (oracle : LightFork.Oracle logicalWidth publicFits)
    (retries : Fin (LightSchedule.schedule logicalWidth publicFits).rhoCount → Answer) : Prop :=
  ∃ valid : Uniqueness.Valid (LightFork.game ajtai key) adversary oracle retries,
    ¬ Truth ajtai key.relation (LightFork.runningOf (adversary.run oracle).link.start.1)
      (freshToLight (adversary.run oracle).link.fold.fresh)
      (LightFork.extractedWitness ajtai valid.1 valid.2)

variable {key adversary}

theorem witnessOf_eq {oracle : Oracle} {retries : Retries logicalWidth publicFits}
    (valid : Uniqueness.Valid (LightFork.game ajtai key) adversary oracle retries) :
    witnessOf ajtai key adversary oracle retries =
      some (LightFork.extractedWitness ajtai valid.1 valid.2) :=
  dite_eq_left valid

/-- Agreement: a valid resampled run with a failing pair's statement, start claim and extracted
witness is a light false acceptance of the failing pair's data. -/
theorem agree {oracle : Oracle} {retries : Retries logicalWidth publicFits} {other : Oracle}
    {otherRetries : Retries logicalWidth publicFits}
    (fails : Fails ajtai key adversary oracle retries)
    (otherValid : Uniqueness.Valid (LightFork.game ajtai key) adversary other otherRetries)
    (sameStatement : (LightFork.game ajtai key).statement (adversary.run other) =
      (LightFork.game ajtai key).statement (adversary.run oracle))
    (sameStart : (LightFork.game ajtai key).start (adversary.run other) =
      (LightFork.game ajtai key).start (adversary.run oracle))
    (sameWitness : witnessOf ajtai key adversary other otherRetries =
      witnessOf ajtai key adversary oracle retries) :
    Uniqueness.FalseFor (LightFork.game ajtai key) adversary (source ajtai key adversary)
      (oracle, retries) other := by
  obtain ⟨valid, untrue⟩ := fails
  have witnessEq : LightFork.extractedWitness ajtai otherValid.1 otherValid.2 =
      LightFork.extractedWitness ajtai valid.1 valid.2 := by
    rw [witnessOf_eq ajtai otherValid, witnessOf_eq ajtai valid] at sameWitness
    exact Option.some.inj sameWitness
  have sourceEq : source ajtai key adversary (oracle, retries) =
      LightTest.source key.relation ajtai (LightFork.runningOf (adversary.run oracle).link.start.1)
        (LightFork.extractedWitness ajtai valid.1 valid.2) := by
    funext statement
    simp only [source, witnessOf_eq ajtai valid, Option.getD_some]
  have startEq : (adversary.run other).link.start = (adversary.run oracle).link.start := sameStart
  have freshEq : freshToLight (adversary.run other).link.fold.fresh =
      freshToLight (adversary.run oracle).link.fold.fresh := sameStatement
  have checked := (LightFork.succeeds_holds ajtai otherValid.1).2.1
  have ambient : AmbientOutputHolds (keys key.relation ajtai).light.extensionOps
      (keys key.relation ajtai).light.lift (keys key.relation ajtai).light.openingMaps
      (keys key.relation ajtai).light.params
      ((keys key.relation ajtai).light.statement
        (LightFork.runningOf (adversary.run other).link.start.1)
        (freshToLight (adversary.run other).link.fold.fresh))
      (LightTest.oracleProbe key.relation ajtai
        (LightFork.runningOf (adversary.run other).link.start.1) other
        (freshToLight (adversary.run other).link.fold.fresh)
        (LightFork.proofOf (adversary.run other).link.fold.proof))
      (LightFork.extractedWitness ajtai otherValid.1 otherValid.2) :=
    Light.fork_ambientOutputHolds (keys key.relation ajtai).light _ _ _ _ (lightExtraction ajtai)
      LightFork.strongSet rfl (LightFork.fork ajtai otherValid.1 otherValid.2)
  rw [witnessEq, startEq] at ambient
  rw [startEq] at checked
  have false_ : LightTest.FalseAcceptance key.relation ajtai
      (LightFork.runningOf (adversary.run oracle).link.start.1)
      (LightFork.extractedWitness ajtai valid.1 valid.2) other
      (freshToLight (adversary.run other).link.fold.fresh)
      (LightFork.proofOf (adversary.run other).link.fold.proof) := by
    refine ⟨checked, ambient, ?_⟩
    rw [freshEq]
    exact untrue
  unfold Uniqueness.FalseFor
  rw [sourceEq]
  exact LightTest.falseAcceptance_hits key.relation ajtai _ _ other _ _ false_

end Uniqueness

section Kernel

/-- An opening of a light output at a bound opens at every larger bound. -/
theorem lightHolds_mono {relation : ProductionKey.LogicalRelation logicalWidth publicFits}
    {claim : LightFork.Claim logicalWidth publicFits}
    {witness : PaperAlgebra.Assignment (logicalWidth := logicalWidth) (publicFits := publicFits)}
    {bound bound' : Nat}
    (holds : Light.Rlc.Holds (keys relation ajtai).light.piRlcSemantics bound claim witness)
    (larger : bound ≤ bound') :
    Light.Rlc.Holds (keys relation ajtai).light.piRlcSemantics bound' claim witness :=
  ⟨⟨holds.1.1, holds.1.2.1, fun column => Nat.lt_of_lt_of_le (holds.1.2.2 column) larger⟩,
    holds.2⟩

/-- Relaxed binding on the Ajtai key for the light key's extraction algebra. -/
theorem compatible (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    PiRLC.PaperForkBinding.Compatible (algebra := (keys relation ajtai).light.piRlcAlgebra)
      (lightExtraction ajtai)
      (Phi81Relation.PiRLCAlgebra.Binding.relaxedOps (shape := FullShape logicalWidth publicFits)
        (rows := productionProfile.commitmentWidth)) :=
  { differenceValid := (Nifs.BindingBridge.compatible relation ajtai).differenceValid
    assignmentAction := (Nifs.BindingBridge.compatible relation ajtai).assignmentAction
    commitmentAction := (Nifs.BindingBridge.compatible relation ajtai).commitmentAction
    differenceBounded := (Nifs.BindingBridge.compatible relation ajtai).differenceBounded }

/-- A light collision, read on the Ajtai key's relation. -/
def lightCollision {relation : ProductionKey.LogicalRelation logicalWidth publicFits}
    {bound : Nat} {commitment : PaperAlgebra.Commitment}
    (collision : PiRLC.ChainBinding.CollisionAt (keys relation ajtai).light.piRlcSemantics
      productionGlobalParams
      (Phi81Relation.PiRLCAlgebra.Binding.relaxedOps (shape := FullShape logicalWidth publicFits)
        (rows := productionProfile.commitmentWidth)) bound commitment) :
    PiRLC.ChainBinding.CollisionAt
      (Phi81Relation.relationSemantics (Phi81Relation.PiRLCAlgebra.Commitment.commit ajtai))
      productionGlobalParams Phi81Relation.PiRLCAlgebra.Binding.relaxedOps bound commitment where
  collision :=
    { delta₁ := collision.collision.delta₁
      delta₂ := collision.collision.delta₂
      opening₁ := collision.collision.opening₁
      opening₂ := collision.collision.opening₂
      delta₁Valid := collision.collision.delta₁Valid
      delta₂Valid := collision.collision.delta₂Valid
      firstEquation := collision.collision.firstEquation
      secondEquation := collision.collision.secondEquation
      firstNorm := collision.collision.firstNorm
      secondNorm := collision.collision.secondNorm
      crossDifferent := collision.collision.crossDifferent }
  firstNorm := collision.firstNorm
  secondNorm := collision.secondNorm

/-- The commitment of a light Π_CCS output depends only on the running claim and the fresh
claim. -/
theorem outputs_commitment_eq {relation : ProductionKey.LogicalRelation logicalWidth publicFits}
    {oracle other : Oracle} {running running' : LightFork.Running logicalWidth publicFits}
    {fresh fresh' : LightFork.Statement logicalWidth publicFits}
    {proof proof' : Light.Proof LightShape 8} {source : Fin LightShape.sourceCount}
    (runningEq : running = running') (freshEq : fresh = fresh') :
    (LightFork.outputs relation ajtai oracle running fresh proof source).commitment =
      (LightFork.outputs relation ajtai other running' fresh' proof' source).commitment := by
  subst runningEq freshEq
  rfl

variable {key : CircuitKey logicalWidth publicFits}
  {adversary : OracleComp (Point (LightSchedule.schedule logicalWidth publicFits) 8) Answer
    (Attempt logicalWidth publicFits)}

/-- The fork of a valid pair at the chain bound, which bounds the base and retry bounds. -/
noncomputable def boundedFork {oracle : Oracle} {retries : Retries logicalWidth publicFits}
    (succeeds : Fork.Succeeds (LightFork.game ajtai key) adversary oracle)
    (valid : ∀ index, Fork.ForkValid (LightFork.game ajtai key) adversary oracle retries index) :
    Light.Rlc.Fork (keys key.relation ajtai).light.piRlcAlgebra (keys key.relation ajtai).light.one
      (BoundedDepthReverse.chainBound productionGlobalParams
        Folding.Schedule.productionLightFoldCap)
      (keys key.relation ajtai).light.relationSource
      (LightTest.oracleProbe key.relation ajtai
        (LightFork.runningOf (adversary.run oracle).link.start.1) oracle
        (freshToLight (adversary.run oracle).link.fold.fresh)
        (LightFork.proofOf (adversary.run oracle).link.fold.proof)).coins.roundPoint
      (LightFork.outputs key.relation ajtai oracle
        (LightFork.runningOf (adversary.run oracle).link.start.1)
        (freshToLight (adversary.run oracle).link.fold.fresh)
        (LightFork.proofOf (adversary.run oracle).link.fold.proof)
        (keys key.relation ajtai).light.freshSource)
      (LightFork.outputs key.relation ajtai oracle
        (LightFork.runningOf (adversary.run oracle).link.start.1)
        (freshToLight (adversary.run oracle).link.fold.fresh)
        (LightFork.proofOf (adversary.run oracle).link.fold.proof)
        (keys key.relation ajtai).light.runningSource) where
  baseChallenge := (LightFork.fork ajtai succeeds valid).baseChallenge
  forkChallenge := (LightFork.fork ajtai succeeds valid).forkChallenge
  baseAssignment := (LightFork.fork ajtai succeeds valid).baseAssignment
  forkAssignment := (LightFork.fork ajtai succeeds valid).forkAssignment
  baseValid := (LightFork.fork ajtai succeeds valid).baseValid
  forkValid := (LightFork.fork ajtai succeeds valid).forkValid
  changed := (LightFork.fork ajtai succeeds valid).changed
  baseHolds := lightHolds_mono ajtai (LightFork.succeeds_holds ajtai succeeds).2.2
    (UniquenessLink.bound_le_chain ajtai key _ _ _ adversary succeeds)
  forkHolds := lightHolds_mono ajtai (LightFork.retry_holds ajtai (valid LightFork.first))
    (UniquenessLink.bound_le_chain ajtai key _ _ _ adversary (valid LightFork.first).succeeds)

variable (key adversary) in
/-- The kernel vector that two valid pairs with one statement and one start claim give when
their extracted witnesses differ. -/
noncomputable def rerunKernel (oracle : Oracle) (retries : Retries logicalWidth publicFits)
    (other : Oracle)
    (otherRetries : Retries logicalWidth publicFits) :
    Option (Phi81Relation.PiRLCAlgebra.Binding.ShortKernelVector ajtai
      (Folding.Schedule.msisChainBound productionGlobalParams
        Folding.Schedule.productionLightFoldCap)) :=
  if paired : Uniqueness.Valid (LightFork.game ajtai key) adversary oracle retries ∧
      Uniqueness.Valid (LightFork.game ajtai key) adversary other otherRetries ∧
      freshToLight (adversary.run other).link.fold.fresh =
        freshToLight (adversary.run oracle).link.fold.fresh ∧
      (adversary.run other).link.start = (adversary.run oracle).link.start then
    have freshCommitment :
        (LightFork.outputs key.relation ajtai oracle
          (LightFork.runningOf (adversary.run oracle).link.start.1)
          (freshToLight (adversary.run oracle).link.fold.fresh)
          (LightFork.proofOf (adversary.run oracle).link.fold.proof)
          (keys key.relation ajtai).light.freshSource).commitment =
        (LightFork.outputs key.relation ajtai other
          (LightFork.runningOf (adversary.run other).link.start.1)
          (freshToLight (adversary.run other).link.fold.fresh)
          (LightFork.proofOf (adversary.run other).link.fold.proof)
          (keys key.relation ajtai).light.freshSource).commitment :=
      outputs_commitment_eq ajtai
        (congrArg (fun start => LightFork.runningOf start.1) paired.2.2.2.symm) paired.2.2.1.symm
    have runningCommitment :
        (LightFork.outputs key.relation ajtai oracle
          (LightFork.runningOf (adversary.run oracle).link.start.1)
          (freshToLight (adversary.run oracle).link.fold.fresh)
          (LightFork.proofOf (adversary.run oracle).link.fold.proof)
          (keys key.relation ajtai).light.runningSource).commitment =
        (LightFork.outputs key.relation ajtai other
          (LightFork.runningOf (adversary.run other).link.start.1)
          (freshToLight (adversary.run other).link.fold.fresh)
          (LightFork.proofOf (adversary.run other).link.fold.proof)
          (keys key.relation ajtai).light.runningSource).commitment :=
      outputs_commitment_eq ajtai
        (congrArg (fun start => LightFork.runningOf start.1) paired.2.2.2.symm) paired.2.2.1.symm
    if freshDifferent :
        (boundedFork ajtai paired.1.1 paired.1.2).freshAssignment (lightExtraction ajtai)
            LightFork.strongSet ≠
          (boundedFork ajtai paired.2.1.1 paired.2.1.2).freshAssignment (lightExtraction ajtai)
            LightFork.strongSet then
      some (Phi81Relation.PiRLCAlgebra.Binding.chainCollision_to_shortKernel ajtai _ le_rfl
        (lightCollision ajtai (relation := key.relation)
          (Light.Rlc.collisionAbove (lightExtraction ajtai) _
          (compatible ajtai key.relation) LightFork.strongSet rfl
          (fun left right => Nifs.BindingBridge.differenceBounded key.relation ajtai _ left right)
          (boundedFork ajtai paired.1.1 paired.1.2) (boundedFork ajtai paired.2.1.1 paired.2.1.2)
          freshCommitment freshDifferent)))
    else if runningDifferent :
        (boundedFork ajtai paired.1.1 paired.1.2).runningAssignment (lightExtraction ajtai)
            LightFork.strongSet ≠
          (boundedFork ajtai paired.2.1.1 paired.2.1.2).runningAssignment (lightExtraction ajtai)
            LightFork.strongSet then
      some (Phi81Relation.PiRLCAlgebra.Binding.chainCollision_to_shortKernel ajtai _ le_rfl
        (lightCollision ajtai (relation := key.relation)
          (Light.Rlc.runningCollisionAbove (lightExtraction ajtai) _
          (compatible ajtai key.relation) LightFork.strongSet rfl
          (fun left right => Nifs.BindingBridge.differenceBounded key.relation ajtai _ left right)
          (boundedFork ajtai paired.1.1 paired.1.2) (boundedFork ajtai paired.2.1.1 paired.2.1.2)
          freshCommitment runningCommitment (not_not.mp freshDifferent) runningDifferent)))
    else none
  else none

/-- Two valid pairs with one statement whose start claims agree and whose extracted witnesses
differ give a kernel vector. -/
theorem rerunKernel_isSome {oracle : Oracle} {retries : Retries logicalWidth publicFits}
    {other : Oracle}
    {otherRetries : Retries logicalWidth publicFits}
    (valid : Uniqueness.Valid (LightFork.game ajtai key) adversary oracle retries)
    (otherValid : Uniqueness.Valid (LightFork.game ajtai key) adversary other otherRetries)
    (sameStatement : freshToLight (adversary.run other).link.fold.fresh =
      freshToLight (adversary.run oracle).link.fold.fresh)
    (collides : Uniqueness.Collides (LightFork.game ajtai key) adversary
      (witnessOf ajtai key adversary) oracle retries other otherRetries) :
    (rerunKernel ajtai key adversary oracle retries other otherRetries).isSome := by
  have paired : Uniqueness.Valid (LightFork.game ajtai key) adversary oracle retries ∧
      Uniqueness.Valid (LightFork.game ajtai key) adversary other otherRetries ∧
      freshToLight (adversary.run other).link.fold.fresh =
        freshToLight (adversary.run oracle).link.fold.fresh ∧
      (adversary.run other).link.start = (adversary.run oracle).link.start :=
    ⟨valid, otherValid, sameStatement, collides.1⟩
  unfold rerunKernel
  rw [dite_eq_left paired]
  by_cases freshDifferent :
      (boundedFork ajtai paired.1.1 paired.1.2).freshAssignment (lightExtraction ajtai)
          LightFork.strongSet ≠
        (boundedFork ajtai paired.2.1.1 paired.2.1.2).freshAssignment (lightExtraction ajtai)
          LightFork.strongSet
  · rw [dite_eq_left freshDifferent]
    rfl
  · rw [dite_eq_right freshDifferent]
    have runningDifferent :
        (boundedFork ajtai paired.1.1 paired.1.2).runningAssignment (lightExtraction ajtai)
            LightFork.strongSet ≠
          (boundedFork ajtai paired.2.1.1 paired.2.1.2).runningAssignment (lightExtraction ajtai)
            LightFork.strongSet := by
      intro same
      apply collides.2
      rw [witnessOf_eq ajtai otherValid, witnessOf_eq ajtai valid]
      refine congrArg some ?_
      unfold LightFork.extractedWitness
      refine congrArg OutputWitness.mk (funext fun source => ?_)
      by_cases zero : source.val = 0
      · rw [ite_eq_left zero, ite_eq_left zero]
        exact (not_not.mp freshDifferent).symm
      · rw [ite_eq_right zero, ite_eq_right zero]
        exact same.symm
    rw [dite_eq_left runningDifferent]
    rfl

variable (key adversary)

/-- The chance that a valid resampled pair at the base fork index gives a kernel vector with
the base pair. -/
noncomputable def kernelChance : ℝ :=
  Uniqueness.reductionChance (LightFork.game ajtai key) adversary
    fun oracle retries other otherRetries =>
      (rerunKernel ajtai key adversary oracle retries other otherRetries).isSome

theorem collisionChance_le_kernelChance :
    Uniqueness.collisionChance (LightFork.game ajtai key) adversary
        (witnessOf ajtai key adversary) ≤
      kernelChance ajtai key adversary := by
  apply Uniqueness.reductionChance_mono
  intro oracle retries valid sample otherRetries forked otherValid collides
  exact rerunKernel_isSome ajtai valid otherValid
    (Uniqueness.statement_eq_of_fork _ adversary oracle sample forked) collides

end Kernel

end NightstreamFPrime.Lifecycle.OptionB.RandomOracle.LightUniqueness
