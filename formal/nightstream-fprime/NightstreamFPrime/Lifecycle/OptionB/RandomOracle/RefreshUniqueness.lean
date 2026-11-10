import NightstreamFPrime.Lifecycle.OptionB.RandomOracle.RefreshFork
import NightstreamFPrime.Lifecycle.OptionB.RandomOracle.UniquenessLink
import NightstreamFPrime.Lifecycle.Nifs.BindingBridge
import NightstreamFPrime.Spec.Folding.PiRLC.ChainBinding
import NightstreamFPrime.Spec.Folding.Nifs.PaperStrongInterface
import NightstreamFPrime.Spec.Folding.Refresh.Security
import NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.ChainBinding

/-! The refresh fold's uniqueness step in the random-oracle model: `Uniqueness` on the refresh
game `RefreshFork.game`.

Extraction. A valid pair's complete fork (`RefreshFork.fork`) extracts the 17 Π_CCS output
witnesses (`RefreshFork.extractedWitness`). When they satisfy the 17 source claims (`Truth`), the
Π_DEC input check that the successful run passes recomposes the 16 children into a witness of
the input accumulator below `B`, and the fresh witness satisfies the CCS
(`Refresh.knowledge_of_sourceHolds`). That is the walk's `Extracted` at the refresh bound `B`
(`holds_of_knowledge`, `extracted_of_truth`). Otherwise the pair fails (`Fails`).

Agreement. A valid resampled run with the failing pair's statement (the fresh claim and the
split), start claim and extracted witness passes the refresh Π_CCS check on its oracle probe,
and the witness opens that probe's outputs (`RefreshFork.extracted_ambient`). So it is a refresh
false acceptance of the failing pair's data, a hit of the bad sets of `RefreshTest.source`
(`agree`).

Binding. Two valid pairs with one statement and one start claim give forks on batches with one
commitment vector (`batch_phi`). When their extracted witnesses differ at a coordinate,
`PiRLC.ChainBinding.forkCollision` at the chain bound, which bounds every response
(`responsesBounded`), and `Binding.chainCollision_to_shortKernel` give a kernel vector of the
Ajtai key below the chain norm (`rerunKernel`). So the collision chance is at most the kernel
chance (`collisionChance_le_kernelChance`). -/

set_option autoImplicit false

namespace NightstreamFPrime.Lifecycle.OptionB.RandomOracle.RefreshUniqueness

open scoped BigOperators
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.RandomOracle
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.StrongReduction
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.UnifiedSources
open NightstreamFPrime.Spec.HyperNova.Construction2
open NightstreamFPrime.Spec.Folding.PiRLC.PaperForkExtraction (extractedAssignment)
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
/-- A retry family of the refresh fork: one answer per `ρ` coordinate. -/
abbrev Retries := Fin (RefreshSchedule.schedule logicalWidth publicFits).rhoCount → Answer

local notation "Oracle" => RefreshFork.Oracle logicalWidth publicFits

section Knowledge

/-- The 17 refresh Π_CCS source claims on the children of `accumulator` and `fresh` hold for
`witness`, with `b = 2` on every source. -/
def Truth (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (accumulator : RefreshFork.Accumulator logicalWidth publicFits)
    (fresh : StepFresh logicalWidth publicFits)
    (split : Refresh.Split PaperAlgebra.Commitment productionShape)
    (witness : OutputWitness productionShape (Phi81CarrierLayout.carrierWidth logicalWidth)) :
    Prop :=
  SourceHolds (refreshKey relation ajtai).extensionOps (refreshKey relation ajtai).lift
    (refreshKey relation ajtai).openingMaps (refreshKey relation ajtai).params
    ((refreshKey relation ajtai).statement
      (Refresh.children (refreshKey relation ajtai) accumulator split) fresh) witness

/-- The answer of a refresh output witness: the recomposed children, then the fresh witness. -/
noncomputable def answerOf (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (witness : OutputWitness productionShape (Phi81CarrierLayout.carrierWidth logicalWidth)) :
    PaperAlgebra.Assignment (logicalWidth := logicalWidth) (publicFits := publicFits) ×
      PaperAlgebra.Assignment (logicalWidth := logicalWidth) (publicFits := publicFits) :=
  ((refreshKey relation ajtai).piDecAlgebra.recomposeAssignment fun child =>
      witness.assignments
        (runningSourceIndex (Fin.cast (refreshKey relation ajtai).outputCount_eq child)),
    witness.assignments (freshSourceIndex ⟨0, by decide⟩))

/-- SuperNeo v1.2 Theorem 13 and Lemma 4 on a slot claim: an accepted Π_DEC input check and true
source claims open the slot claim below `B` and the fresh instance under the circuit key's
relation. -/
theorem holds_of_knowledge {key : CircuitKey logicalWidth publicFits}
    {claim : SlotClaim logicalWidth publicFits} {fresh : StepFresh logicalWidth publicFits}
    {split : Refresh.Split PaperAlgebra.Commitment productionShape} (bound : Nat)
    {witness : OutputWitness productionShape (Phi81CarrierLayout.carrierWidth logicalWidth)}
    (inputAccepted : Refresh.inputCheck (refreshKey key.relation ajtai)
      (toClaim (key.system ajtai) claim) split = true)
    (truth : Truth ajtai key.relation (toClaim (key.system ajtai) claim) fresh split witness) :
    ForkLink.SlotHolds ajtai key
        (Folding.Schedule.extractedBefore productionGlobalParams .refresh bound) claim
        (answerOf ajtai key.relation witness).1 ∧
      UniquenessLink.FreshHolds ajtai key fresh (answerOf ajtai key.relation witness).2 := by
  obtain ⟨freshHolds, accumulatorHolds⟩ := Refresh.knowledge_of_sourceHolds
    (refreshKey key.relation ajtai) (toClaim (key.system ajtai) claim) fresh split rfl
    inputAccepted witness truth
  have opened := freshHolds ⟨0, by decide⟩
  exact ⟨accumulatorHolds,
    ⟨((refreshKey key.relation ajtai).openingAgreement _ _ _ _).mp opened.1, opened.2⟩⟩

/-- A fold that the refresh game's fold verifiers accept is a refresh step. -/
theorem folds_refresh {oracle : Oracle} {step : Folding.Schedule.Step}
    {key : CircuitKey logicalWidth publicFits} {claim : SlotClaim logicalWidth publicFits}
    {fresh : StepFresh logicalWidth publicFits} {proof : StepProof}
    {next : SlotClaim logicalWidth publicFits}
    (accepted : RefreshFork.folds ajtai oracle step key claim fresh proof = some next) :
    step = .refresh := by
  cases step with
  | refresh => rfl
  | light => cases proof <;> exact absurd accepted (by simp [RefreshFork.folds])

variable {key : CircuitKey logicalWidth publicFits}
  {adversary : OracleComp (Point (RefreshSchedule.schedule logicalWidth publicFits) 8) Answer
    (Attempt logicalWidth publicFits)}
  {oracle : RefreshFork.Oracle logicalWidth publicFits} {retries : Retries logicalWidth publicFits}

/-- A valid pair whose extracted witness satisfies the source claims gives the walk's
`Extracted` at the refresh bound `B`. -/
theorem extracted_of_truth
    (succeeds : Fork.Succeeds (RefreshFork.game ajtai key) adversary oracle)
    (valid : ∀ index, Fork.ForkValid (RefreshFork.game ajtai key) adversary oracle retries index)
    (truth : Truth ajtai key.relation
      (toClaim (key.system ajtai) (adversary.run oracle).link.start.1)
      (adversary.run oracle).link.fold.fresh
      (RefreshFork.proofOf (adversary.run oracle).link.fold.proof).split
      (RefreshFork.extractedWitness ajtai succeeds valid)) :
    UniquenessLink.Extracted ajtai (adversary.run oracle)
      (answerOf ajtai key.relation (RefreshFork.extractedWitness ajtai succeeds valid)) := by
  have holds := holds_of_knowledge ajtai (adversary.run oracle).bound
    (RefreshFork.succeeds_holds ajtai succeeds).2.1 truth
  have keyEq : (adversary.run oracle).link.fold.key = key := succeeds.1
  have inputEq : (adversary.run oracle).link.fold.input = (adversary.run oracle).link.start.1 :=
    succeeds.2.2.1
  have stepRefresh :
      BoundedDepth.circuitStep (adversary.run oracle).link.fold.circuit = .refresh :=
    folds_refresh ajtai ((BoundedDepth.accepts_iff _ _ _ _ _ _ _ _ _).mp succeeds.2.2.2.1).2.1
  show ForkLink.SlotHolds ajtai (adversary.run oracle).link.fold.key
      (Folding.Schedule.extractedBefore productionGlobalParams
        (BoundedDepth.circuitStep (adversary.run oracle).link.fold.circuit)
        (adversary.run oracle).bound)
      (adversary.run oracle).link.fold.input
      (answerOf ajtai key.relation (RefreshFork.extractedWitness ajtai succeeds valid)).1 ∧
    UniquenessLink.FreshHolds ajtai (adversary.run oracle).link.fold.key
      (adversary.run oracle).link.fold.fresh
      (answerOf ajtai key.relation (RefreshFork.extractedWitness ajtai succeeds valid)).2
  rw [stepRefresh, inputEq, keyEq]
  exact holds

end Knowledge

section Uniqueness

variable (key : CircuitKey logicalWidth publicFits)
  (adversary : OracleComp (Point (RefreshSchedule.schedule logicalWidth publicFits) 8) Answer
    (Attempt logicalWidth publicFits))

/-- The witness that a valid pair extracts. -/
noncomputable def witnessOf (oracle : RefreshFork.Oracle logicalWidth publicFits)
    (retries : Retries logicalWidth publicFits) :
    Option (OutputWitness productionShape (Phi81CarrierLayout.carrierWidth logicalWidth)) :=
  if valid : Uniqueness.Valid (RefreshFork.game ajtai key) adversary oracle retries then
    some (RefreshFork.extractedWitness ajtai valid.1 valid.2)
  else none

def zeroWitness : OutputWitness productionShape (Phi81CarrierLayout.carrierWidth logicalWidth) :=
  ⟨fun _ _ => 0⟩

/-- The claimed refresh Π_CCS data of a pair: the children of the base run's start claim, read
through the pair's extracted witness. -/
noncomputable def source
    (pair : RefreshFork.Oracle logicalWidth publicFits × Retries logicalWidth publicFits)
    (statement : (RefreshSchedule.schedule logicalWidth publicFits).Statement) :
    ProtocolPolynomial.Data K (RefreshSchedule.schedule logicalWidth publicFits).shape :=
  RefreshTest.source key.relation ajtai
    (toClaim (key.system ajtai) (adversary.run pair.1).link.start.1)
    ((witnessOf ajtai key adversary pair.1 pair.2).getD zeroWitness) statement

theorem source_degreeCovers
    (pair : RefreshFork.Oracle logicalWidth publicFits × Retries logicalWidth publicFits)
    (statement : (RefreshSchedule.schedule logicalWidth publicFits).Statement) :
    (source ajtai key adversary pair statement).toVerifierInput.sumcheckDegreeBound ≤ 8 :=
  RefreshTest.source_degreeCovers key.relation ajtai _ _ statement

/-- A valid pair whose extracted witness makes a refresh source claim false. -/
def Fails (oracle : RefreshFork.Oracle logicalWidth publicFits)
    (retries : Retries logicalWidth publicFits) : Prop :=
  ∃ valid : Uniqueness.Valid (RefreshFork.game ajtai key) adversary oracle retries,
    ¬ Truth ajtai key.relation (toClaim (key.system ajtai) (adversary.run oracle).link.start.1)
      (adversary.run oracle).link.fold.fresh
      (RefreshFork.proofOf (adversary.run oracle).link.fold.proof).split
      (RefreshFork.extractedWitness ajtai valid.1 valid.2)

variable {key adversary}

theorem witnessOf_eq {oracle : Oracle} {retries : Retries logicalWidth publicFits}
    (valid : Uniqueness.Valid (RefreshFork.game ajtai key) adversary oracle retries) :
    witnessOf ajtai key adversary oracle retries =
      some (RefreshFork.extractedWitness ajtai valid.1 valid.2) :=
  dite_eq_left valid

/-- Agreement: a valid resampled run with a failing pair's statement, start claim and extracted
witness is a refresh false acceptance of the failing pair's data. -/
theorem agree {oracle : Oracle} {retries : Retries logicalWidth publicFits} {other : Oracle}
    {otherRetries : Retries logicalWidth publicFits}
    (fails : Fails ajtai key adversary oracle retries)
    (otherValid : Uniqueness.Valid (RefreshFork.game ajtai key) adversary other otherRetries)
    (sameStatement : (RefreshFork.game ajtai key).statement (adversary.run other) =
      (RefreshFork.game ajtai key).statement (adversary.run oracle))
    (sameStart : (RefreshFork.game ajtai key).start (adversary.run other) =
      (RefreshFork.game ajtai key).start (adversary.run oracle))
    (sameWitness : witnessOf ajtai key adversary other otherRetries =
      witnessOf ajtai key adversary oracle retries) :
    Uniqueness.FalseFor (RefreshFork.game ajtai key) adversary (source ajtai key adversary)
      (oracle, retries) other := by
  obtain ⟨valid, untrue⟩ := fails
  have witnessEq : RefreshFork.extractedWitness ajtai otherValid.1 otherValid.2 =
      RefreshFork.extractedWitness ajtai valid.1 valid.2 := by
    rw [witnessOf_eq ajtai otherValid, witnessOf_eq ajtai valid] at sameWitness
    exact Option.some.inj sameWitness
  have sourceEq : source ajtai key adversary (oracle, retries) =
      RefreshTest.source key.relation ajtai
        (toClaim (key.system ajtai) (adversary.run oracle).link.start.1)
        (RefreshFork.extractedWitness ajtai valid.1 valid.2) := by
    funext statement
    simp only [source, witnessOf_eq ajtai valid, Option.getD_some]
  have startEq : (adversary.run other).link.start = (adversary.run oracle).link.start := sameStart
  have freshEq : (adversary.run other).link.fold.fresh = (adversary.run oracle).link.fold.fresh :=
    congrArg Prod.fst sameStatement
  have splitEq : (RefreshFork.proofOf (adversary.run other).link.fold.proof).split =
      (RefreshFork.proofOf (adversary.run oracle).link.fold.proof).split :=
    congrArg Prod.snd sameStatement
  have checked := (RefreshFork.succeeds_holds ajtai otherValid.1).2.2.1
  have ambient := RefreshFork.extracted_ambient ajtai otherValid.1 otherValid.2
  rw [witnessEq, startEq] at ambient
  rw [startEq] at checked
  have false_ : RefreshTest.FalseAcceptance key.relation ajtai
      (toClaim (key.system ajtai) (adversary.run oracle).link.start.1)
      (RefreshFork.extractedWitness ajtai valid.1 valid.2) other
      (adversary.run other).link.fold.fresh
      (RefreshFork.proofOf (adversary.run other).link.fold.proof) := by
    refine ⟨checked, ambient, ?_⟩
    rw [freshEq, splitEq]
    exact untrue
  unfold Uniqueness.FalseFor
  rw [sourceEq]
  exact RefreshTest.falseAcceptance_hits key.relation ajtai _ _ other _ _ false_

end Uniqueness

section Kernel

/-- The commitments of the refresh batch depend only on the accumulator, the fresh claim and
the split. -/
theorem batch_phi {relation : ProductionKey.LogicalRelation logicalWidth publicFits}
    {oracle other : Oracle}
    {accumulator accumulator' : RefreshFork.Accumulator logicalWidth publicFits}
    {fresh fresh' : StepFresh logicalWidth publicFits} {proof proof' : RefreshFork.RefreshProof}
    (accumulatorEq : accumulator = accumulator') (freshEq : fresh = fresh')
    (splitEq : proof.split = proof'.split) :
    PiRLC.phi (RefreshFork.batch relation ajtai oracle accumulator fresh proof).inputs =
      PiRLC.phi (RefreshFork.batch relation ajtai other accumulator' fresh' proof').inputs := by
  subst accumulatorEq freshEq
  funext coordinate
  show ((refreshKey relation ajtai).statement
      (Refresh.children (refreshKey relation ajtai) accumulator proof.split) fresh).commitments _ =
    ((refreshKey relation ajtai).statement
      (Refresh.children (refreshKey relation ajtai) accumulator proof'.split) fresh).commitments _
  rw [splitEq]

variable {key : CircuitKey logicalWidth publicFits}
  {adversary : OracleComp (Point (RefreshSchedule.schedule logicalWidth publicFits) 8) Answer
    (Attempt logicalWidth publicFits)}

/-- Every response of a valid pair's fork opens at the chain bound. -/
theorem responsesBounded {oracle : Oracle} {retries : Retries logicalWidth publicFits}
    (succeeds : Fork.Succeeds (RefreshFork.game ajtai key) adversary oracle)
    (valid : ∀ index,
      Fork.ForkValid (RefreshFork.game ajtai key) adversary oracle retries index) :
    PiRLC.ChainBinding.ResponsesBounded
      (BoundedDepthReverse.chainBound productionGlobalParams
        Folding.Schedule.productionLightFoldCap)
      (RefreshFork.fork ajtai succeeds valid) := by
  refine ⟨fun column => Nat.lt_of_lt_of_le
    ((RefreshFork.succeeds_holds ajtai succeeds).2.2.2.1.2.2 column)
    (UniquenessLink.bound_le_chain ajtai key _ _ _ adversary succeeds), fun index column => ?_⟩
  exact Nat.lt_of_lt_of_le
    ((RefreshFork.succeeds_holds ajtai (valid index).succeeds).2.2.2.1.2.2 column)
    (UniquenessLink.bound_le_chain ajtai key _ _ _ adversary (valid index).succeeds)

variable (key adversary) in
/-- The kernel vector that two valid pairs with one statement and one start claim give when
their extracted witnesses differ. -/
noncomputable def rerunKernel (oracle : Oracle) (retries : Retries logicalWidth publicFits)
    (other : Oracle) (otherRetries : Retries logicalWidth publicFits) :
    Option (Phi81Relation.PiRLCAlgebra.Binding.ShortKernelVector ajtai
      (Folding.Schedule.msisChainBound productionGlobalParams
        Folding.Schedule.productionLightFoldCap)) :=
  if paired : Uniqueness.Valid (RefreshFork.game ajtai key) adversary oracle retries ∧
      Uniqueness.Valid (RefreshFork.game ajtai key) adversary other otherRetries ∧
      (RefreshFork.game ajtai key).statement (adversary.run other) =
        (RefreshFork.game ajtai key).statement (adversary.run oracle) ∧
      (adversary.run other).link.start = (adversary.run oracle).link.start then
    if different : ∃ coordinate,
        extractedAssignment (normFreeLaws (PaperExtractionAlgebra.extractionAlgebra ajtai))
            RefreshFork.strongSet (RefreshFork.fork ajtai paired.1.1 paired.1.2) coordinate ≠
          extractedAssignment (normFreeLaws (PaperExtractionAlgebra.extractionAlgebra ajtai))
            RefreshFork.strongSet (RefreshFork.fork ajtai paired.2.1.1 paired.2.1.2) coordinate then
      some (Phi81Relation.PiRLCAlgebra.Binding.chainCollision_to_shortKernel ajtai _ le_rfl
        (UniquenessLink.ajtaiCollision ajtai (PiRLC.ChainBinding.forkCollision
          (PaperExtractionAlgebra.extractionAlgebra ajtai) _
          (Nifs.BindingBridge.compatible key.relation ajtai) RefreshFork.strongSet
          (fun left right => Nifs.BindingBridge.differenceBounded key.relation ajtai _ left right)
          (RefreshFork.fork ajtai paired.1.1 paired.1.2)
          (RefreshFork.fork ajtai paired.2.1.1 paired.2.1.2)
          (responsesBounded ajtai paired.1.1 paired.1.2)
          (responsesBounded ajtai paired.2.1.1 paired.2.1.2)
          (batch_phi ajtai (congrArg (fun start => toClaim (key.system ajtai) start.1)
              paired.2.2.2.symm) (congrArg Prod.fst paired.2.2.1.symm)
            (congrArg Prod.snd paired.2.2.1.symm))
          (Classical.choose different) (Classical.choose_spec different))))
    else none
  else none

/-- Two valid pairs with one statement whose start claims agree and whose extracted witnesses
differ give a kernel vector. -/
theorem rerunKernel_isSome {oracle : Oracle} {retries : Retries logicalWidth publicFits}
    {other : Oracle} {otherRetries : Retries logicalWidth publicFits}
    (valid : Uniqueness.Valid (RefreshFork.game ajtai key) adversary oracle retries)
    (otherValid : Uniqueness.Valid (RefreshFork.game ajtai key) adversary other otherRetries)
    (sameStatement : (RefreshFork.game ajtai key).statement (adversary.run other) =
      (RefreshFork.game ajtai key).statement (adversary.run oracle))
    (collides : Uniqueness.Collides (RefreshFork.game ajtai key) adversary
      (witnessOf ajtai key adversary) oracle retries other otherRetries) :
    (rerunKernel ajtai key adversary oracle retries other otherRetries).isSome := by
  have paired : Uniqueness.Valid (RefreshFork.game ajtai key) adversary oracle retries ∧
      Uniqueness.Valid (RefreshFork.game ajtai key) adversary other otherRetries ∧
      (RefreshFork.game ajtai key).statement (adversary.run other) =
        (RefreshFork.game ajtai key).statement (adversary.run oracle) ∧
      (adversary.run other).link.start = (adversary.run oracle).link.start :=
    ⟨valid, otherValid, sameStatement, collides.1⟩
  have different : ∃ coordinate,
      extractedAssignment (normFreeLaws (PaperExtractionAlgebra.extractionAlgebra ajtai))
          RefreshFork.strongSet (RefreshFork.fork ajtai paired.1.1 paired.1.2) coordinate ≠
        extractedAssignment (normFreeLaws (PaperExtractionAlgebra.extractionAlgebra ajtai))
          RefreshFork.strongSet (RefreshFork.fork ajtai paired.2.1.1 paired.2.1.2) coordinate := by
    by_contra same
    simp only [not_exists, not_not] at same
    apply collides.2
    rw [witnessOf_eq ajtai otherValid, witnessOf_eq ajtai valid]
    exact congrArg (fun values => some (Nifs.PaperStrongInterface.outputWitnessOfAssignments
      (refreshKey key.relation ajtai) values)) (funext same).symm
  unfold rerunKernel
  rw [dite_eq_left paired, dite_eq_left different]
  rfl

variable (key adversary)

/-- The chance that a valid resampled pair at the base fork index gives a kernel vector with
the base pair. -/
noncomputable def kernelChance : ℝ :=
  Uniqueness.reductionChance (RefreshFork.game ajtai key) adversary
    fun oracle retries other otherRetries =>
      (rerunKernel ajtai key adversary oracle retries other otherRetries).isSome

theorem collisionChance_le_kernelChance :
    Uniqueness.collisionChance (RefreshFork.game ajtai key) adversary
        (witnessOf ajtai key adversary) ≤
      kernelChance ajtai key adversary := by
  apply Uniqueness.reductionChance_mono
  intro oracle retries valid sample otherRetries forked otherValid collides
  exact rerunKernel_isSome ajtai valid otherValid
    (Uniqueness.statement_eq_of_fork _ adversary oracle sample forked) collides

end Kernel

end NightstreamFPrime.Lifecycle.OptionB.RandomOracle.RefreshUniqueness
