import NightstreamFPrime.Spec.Folding.Light.Verifier
import NightstreamFPrime.Spec.Folding.Light.Strong
import NightstreamFPrime.Spec.Folding.Light.Binding

/-! SuperNeo v1.2 Theorem 12 for the light fold, with Cyclo (ePrint 2026/359) Theorem 3,
item 2. When `verify` accepts and two outputs differ only in `ρ`, the extracted fresh witness
satisfies the CCS with norm below `b`, the extracted accumulator opens at the output bound
plus `T · (b − 1)` with its claims true at the prior point, or a Π_CCS bad event occurred.
`accepted_implies_securityOutcome_probe` states this for any probe whose light Π_CCS check
accepts, such as the probe of a random-oracle verifier. Its parts are `fork_ambientOutputHolds`
(the extracted witnesses open the probe's outputs) and `knowledge_of_truth` (true source claims
give the knowledge outcome). -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Folding.Light

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.StrongReduction
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open NightstreamFPrime.Spec.Folding.PiRLC.PaperForkExtraction (ExtractionAlgebra StrongSetUnits)
open NightstreamFPrime.Spec.Folding.Nifs.PaperNonInteractive (Fresh Running)
open NightstreamFPrime.Spec.Folding.PiRLC.NormFreeFork (normFreeAlgebra normFreeLaws)

universe uCommitment uPublicInput uScalar uState

variable {Commitment : Type uCommitment} {PublicInput : Type uPublicInput}
  {Scalar : Type uScalar} {State : Type uState} {shape : Shape}
  {columns blockCount width : Nat}

namespace Key

variable (key : Key Commitment PublicInput Scalar State shape columns blockCount width)

def outputs (running : Running K Commitment PublicInput shape)
    (fresh : Fresh Commitment PublicInput shape) (proof : Proof shape width) :=
  (key.statement running fresh).publicOutput (key.probe running fresh proof)

/-- Two accepted outputs of one light proof that differ only in `ρ`. -/
abbrev Fork (running : Running K Commitment PublicInput shape)
    (fresh : Fresh Commitment PublicInput shape) (proof : Proof shape width) (bound : Nat) :=
  Rlc.Fork key.piRlcAlgebra key.one bound key.relationSource
    (key.probe running fresh proof).coins.roundPoint
    (key.outputs running fresh proof key.freshSource)
    (key.outputs running fresh proof key.runningSource)

def extractedWitness (running : Running K Commitment PublicInput shape)
    (fresh : Fresh Commitment PublicInput shape) (proof : Proof shape width) (bound : Nat)
    (laws : ExtractionAlgebra key.piRlcSemantics key.params key.piRlcAlgebra)
    (strongSet : StrongSetUnits laws.ring key.piRlcAlgebra.challengeValid)
    (fork : key.Fork running fresh proof bound) : OutputWitness shape columns where
  assignments := fun source =>
    if source.val = 0 then fork.freshAssignment laws strongSet
    else fork.runningAssignment laws strongSet

end Key

/-- The knowledge outcome: both inputs of the light fold, with the light norm bounds. -/
structure Knowledge
    (key : Key Commitment PublicInput Scalar State shape columns blockCount width)
    (running : Running K Commitment PublicInput shape)
    (fresh : Fresh Commitment PublicInput shape)
    (bound : Nat) (witness : OutputWitness shape columns) : Prop where
  freshOpening : Opening.Holds
    (paperRelationSemantics (shape := shape) (blockCount := blockCount)
      key.baseOps key.extensionOps key.lift key.openingMaps)
    key.params.b ((key.statement running fresh).commitments key.freshSource)
    ((key.statement running fresh).publicInputs key.freshSource)
    (witness.assignments key.freshSource)
  runningOpening : Opening.Holds key.piRlcSemantics
    (bound + key.params.expansionT * (key.params.b - 1))
    ((key.statement running fresh).commitments key.runningSource)
    ((key.statement running fresh).publicInputs key.runningSource)
    (witness.assignments key.runningSource)
  truth : PiCCS.SemanticTruth
    (((key.statement running fresh).sourceConnectedInputs witness).toUnifiedInputs key.baseOps)
    key.baseOps key.extensionOps key.lift

private theorem paperAmbient
    (key : Key Commitment PublicInput Scalar State shape columns blockCount width)
    (statement : CE.Instance (RelationSource shape columns blockCount) PublicInput
      (CubePoint K shape.cubeVariables) (EvaluationFamily K shape) Commitment)
    (system : statement.constraintSystem = key.relationSource)
    (assignment : Assignment F columns)
    (holds : PiRLC.PaperCorrections.CorrectedAmbientHolds key.piRlcSemantics key.params
      statement assignment) :
    PiRLC.PaperCorrections.CorrectedAmbientHolds
      (paperRelationSemantics key.baseOps key.extensionOps key.lift key.openingMaps)
      key.params statement assignment := by
  refine ⟨(key.openingAgreement _ _ _ _).mpr holds.1, trivial, ?_⟩
  have evaluations := holds.2.2
  rw [system] at evaluations ⊢
  exact (key.evaluationAgreement assignment statement.point).2.symm.trans evaluations

/-- The output witness that a light fork without norms extracts: the fresh witness at the fresh
source and the running witness at the running source. -/
def Key.normFreeWitness
    (key : Key Commitment PublicInput Scalar State shape columns blockCount width)
    {bound : Nat} {point : CubePoint K shape.cubeVariables}
    {freshOutput runningOutput : CE.Instance (RelationSource shape columns blockCount) PublicInput
      (CubePoint K shape.cubeVariables) (EvaluationFamily K shape) Commitment}
    (laws : ExtractionAlgebra key.piRlcSemantics key.params key.piRlcAlgebra)
    (strongSet : StrongSetUnits laws.ring key.piRlcAlgebra.challengeValid)
    (fork : Rlc.Fork (normFreeAlgebra key.piRlcAlgebra) key.one bound key.relationSource point
      freshOutput runningOutput) : OutputWitness shape columns where
  assignments := fun source =>
    if source.val = 0 then fork.freshAssignment (normFreeLaws laws) strongSet
    else fork.runningAssignment (normFreeLaws laws) strongSet

section Probe

variable (key : Key Commitment PublicInput Scalar State shape columns blockCount width)
  (running : Running K Commitment PublicInput shape)
  (fresh : Fresh Commitment PublicInput shape) (probe : Probe K shape) (bound : Nat)
  (laws : ExtractionAlgebra key.piRlcSemantics key.params key.piRlcAlgebra)
  (strongSet : StrongSetUnits laws.ring key.piRlcAlgebra.challengeValid)

/-- A light fork without norms on the Π_CCS outputs of `probe`, which can be any probe, such as
one whose coins a random oracle gives. -/
abbrev ProbeFork :=
  Rlc.Fork (normFreeAlgebra key.piRlcAlgebra) key.one bound key.relationSource
    probe.coins.roundPoint ((key.statement running fresh).publicOutput probe key.freshSource)
    ((key.statement running fresh).publicOutput probe key.runningSource)

theorem source_eq_fresh_or_running (source : Fin shape.sourceCount) :
    source = key.freshSource ∨ source = key.runningSource := by
  have below : source.val < 2 := by
    have := source.isLt
    simp only [Shape.sourceCount, key.freshCount_eq, key.runningCount_eq] at this
    exact this
  by_cases first : source.val = 0
  · exact Or.inl (Fin.ext first)
  · exact Or.inr (Fin.ext (by simp only [Key.runningSource]; omega))

/-- SuperNeo v1.2 Lemma 8 on a light fork without norms over any probe: the extracted witnesses
open the probe's Π_CCS outputs at the ambient bound. -/
theorem fork_ambientOutputHolds (oneEq : key.one = laws.ring.one)
    (fork : ProbeFork key running fresh probe bound) :
    AmbientOutputHolds key.extensionOps key.lift key.openingMaps key.params
      (key.statement running fresh) probe (key.normFreeWitness laws strongSet fork) := by
  obtain ⟨freshFree, runningFree⟩ := Rlc.fork_extracts (normFreeLaws laws) strongSet oneEq fork
    rfl rfl rfl rfl 1 rfl rfl
  intro source
  rcases source_eq_fresh_or_running key source with same | same <;> subst same
  · have value : (key.normFreeWitness laws strongSet fork).assignments key.freshSource =
        fork.freshAssignment (normFreeLaws laws) strongSet := by
      simp [Key.normFreeWitness, Key.freshSource]
    rw [value]
    exact paperAmbient key _ rfl _
      ⟨⟨freshFree.1.1, freshFree.1.2.1, laws.correctedNormCoverage _⟩, freshFree.2⟩
  · have value : (key.normFreeWitness laws strongSet fork).assignments key.runningSource =
        fork.runningAssignment (normFreeLaws laws) strongSet := by
      simp [Key.normFreeWitness, Key.runningSource]
    rw [value]
    exact paperAmbient key _ rfl _
      ⟨⟨runningFree.1.1, runningFree.1.2.1, laws.correctedNormCoverage _⟩, runningFree.2⟩

/-- SuperNeo v1.2 Theorem 12's knowledge outcome for the light fold, with Cyclo Theorem 3,
item 2: when the extracted witnesses make the source claims true, the fresh witness opens below
`b` and the running witness opens at `bound + T · (b − 1)`. -/
theorem knowledge_of_truth (freshBound : key.params.b = 2) (oneEq : key.one = laws.ring.one)
    (normLaw : Rlc.ExtractionNormLaw laws) (fork : ProbeFork key running fresh probe bound)
    (baseNorm : key.piRlcSemantics.normBounded bound fork.baseAssignment)
    (truth : PiCCS.SemanticTruth
      (((key.statement running fresh).sourceConnectedInputs
        (key.normFreeWitness laws strongSet fork)).toUnifiedInputs key.baseOps)
      key.baseOps key.extensionOps key.lift) :
    Knowledge key running fresh bound (key.normFreeWitness laws strongSet fork) := by
  have ambient := fork_ambientOutputHolds key running fresh probe bound laws strongSet oneEq fork
  have freshValue : (key.normFreeWitness laws strongSet fork).assignments key.freshSource =
      fork.freshAssignment (normFreeLaws laws) strongSet := by
    simp [Key.normFreeWitness, Key.freshSource]
  have runningValue : (key.normFreeWitness laws strongSet fork).assignments key.runningSource =
      fork.runningAssignment (normFreeLaws laws) strongSet := by
    simp [Key.normFreeWitness, Key.runningSource]
  have freshNorm : ∀ column, centeredMagnitude
      ((key.normFreeWitness laws strongSet fork).assignments key.freshSource column) <
        key.params.b := by
    intro column
    rw [freshBound]
    exact truth.2.1 key.freshSource (by
      simp only [Key.freshSource, key.freshCount_eq]
      omega) column
  have freshPaperOpening : Opening.Holds
      (paperRelationSemantics (shape := shape) (blockCount := blockCount)
        key.baseOps key.extensionOps key.lift key.openingMaps)
      key.params.b ((key.statement running fresh).commitments key.freshSource)
      ((key.statement running fresh).publicInputs key.freshSource)
      ((key.normFreeWitness laws strongSet fork).assignments key.freshSource) :=
    ⟨(ambient key.freshSource).1.1, (ambient key.freshSource).1.2.1, freshNorm⟩
  have freshPiRlcNorm : key.piRlcSemantics.normBounded key.params.b
      (fork.freshAssignment (normFreeLaws laws) strongSet) := by
    rw [← freshValue]
    exact ((key.openingAgreement _ _ _ _).mp freshPaperOpening).2.2
  have runningNorm : key.piRlcSemantics.normBounded
      (bound + key.params.expansionT * (key.params.b - 1))
      (fork.runningAssignment (normFreeLaws laws) strongSet) :=
    normLaw.sub_bound bound fork.baseChallenge fork.baseAssignment _ fork.baseValid baseNorm
      freshPiRlcNorm
  have runningAmbient := (key.openingAgreement _ _ _ _).mp (ambient key.runningSource).1
  refine ⟨freshPaperOpening, ?_, truth⟩
  rw [runningValue] at runningAmbient ⊢
  exact ⟨runningAmbient.1, runningAmbient.2.1, runningNorm⟩

/-- SuperNeo v1.2 Theorem 12 for the light fold, with Cyclo Theorem 3, item 2, on any probe whose
light Π_CCS check accepts and a fork without norms on its outputs whose base output opens at
`bound`. -/
theorem accepted_implies_securityOutcome_probe
    (checked : PiCCS.check key.extensionOps key.lift (key.statement running fresh) width probe =
      true)
    (freshBound : key.params.b = 2) (challengeSetSize : Nat) (oneEq : key.one = laws.ring.one)
    (normLaw : Rlc.ExtractionNormLaw laws) (fork : ProbeFork key running fresh probe bound)
    (baseNorm : key.piRlcSemantics.normBounded bound fork.baseAssignment) :
    Knowledge key running fresh bound (key.normFreeWitness laws strongSet fork) ∨
      SignedCoefficientObject.MixingRoot key.extensionOps
        ((PiCCS.protocolData key.baseOps key.lift
          (((key.statement running fresh).sourceConnectedInputs
            (key.normFreeWitness laws strongSet fork)).toUnifiedInputs
            key.baseOps)).toJointData key.extensionOps)
        probe.coins.alpha probe.coins.gamma ∨
      PiCCS.SumCheckFailure key.extensionOps key.lift (key.statement running fresh) width
        challengeSetSize probe (key.normFreeWitness laws strongSet fork) := by
  rcases PiCCS.accepted_extracts_or_badEvent key.baseLaws key.baseZero key.noZeroDivisors
      key.extensionOps key.extensionLaws key.extensionZeroLaws key.lift key.liftLaws
      key.openingMaps key.params (key.statement running fresh) key.constantLaw width
      (Nat.le_of_eq (key.statement_sumcheckDegreeBound_eq running fresh)) challengeSetSize
      probe (key.normFreeWitness laws strongSet fork)
      (fork_ambientOutputHolds key running fresh probe bound laws strongSet oneEq fork) checked with
    truth | mixingRoot | sumCheck
  · exact Or.inl (knowledge_of_truth key running fresh probe bound laws strongSet freshBound oneEq
      normLaw fork baseNorm truth)
  · exact Or.inr (Or.inl mixingRoot)
  · exact Or.inr (Or.inr sumCheck)

end Probe

/-- SuperNeo v1.2 Theorem 12 for the light fold, with Cyclo Theorem 3, item 2, from a fork whose
outputs satisfy no norm bound and whose base output opens at `bound`. The extracted accumulator
opens at `bound + T · (b − 1)`; the forked output's norm does not enter. -/
theorem accepted_implies_securityOutcome_normFree
    (key : Key Commitment PublicInput Scalar State shape columns blockCount width)
    (running : Running K Commitment PublicInput shape)
    (fresh : Fresh Commitment PublicInput shape)
    (proof : Proof shape width)
    (result : CE.Instance (RelationSource shape columns blockCount) PublicInput
      (CubePoint K shape.cubeVariables) (EvaluationFamily K shape) Commitment)
    (accepted : verify key running fresh proof = some result)
    (freshBound : key.params.b = 2)
    (challengeSetSize bound : Nat)
    (laws : ExtractionAlgebra key.piRlcSemantics key.params key.piRlcAlgebra)
    (strongSet : StrongSetUnits laws.ring key.piRlcAlgebra.challengeValid)
    (oneEq : key.one = laws.ring.one)
    (normLaw : Rlc.ExtractionNormLaw laws)
    (fork : Rlc.Fork (normFreeAlgebra key.piRlcAlgebra) key.one bound key.relationSource
      (key.probe running fresh proof).coins.roundPoint (key.outputs running fresh proof key.freshSource)
      (key.outputs running fresh proof key.runningSource))
    (baseNorm : key.piRlcSemantics.normBounded bound fork.baseAssignment) :
    let witness : OutputWitness shape columns :=
      ⟨fun source => if source.val = 0 then fork.freshAssignment (normFreeLaws laws) strongSet
        else fork.runningAssignment (normFreeLaws laws) strongSet⟩
    Knowledge key running fresh bound witness ∨
      SignedCoefficientObject.MixingRoot key.extensionOps
        ((PiCCS.protocolData key.baseOps key.lift
          (((key.statement running fresh).sourceConnectedInputs witness).toUnifiedInputs
            key.baseOps)).toJointData key.extensionOps)
        (key.probe running fresh proof).coins.alpha
        (key.probe running fresh proof).coins.gamma ∨
      PiCCS.SumCheckFailure key.extensionOps key.lift (key.statement running fresh) width
        challengeSetSize (key.probe running fresh proof) witness := by
  have checked : PiCCS.check key.extensionOps key.lift (key.statement running fresh) width
      (key.probe running fresh proof) = true := by
    by_contra rejected
    simp [verify, rejected] at accepted
  exact accepted_implies_securityOutcome_probe key running fresh (key.probe running fresh proof)
    bound laws strongSet checked freshBound challengeSetSize oneEq normLaw fork baseNorm

/-- SuperNeo v1.2 Theorem 12 for the light fold, with Cyclo Theorem 3, item 2. -/
theorem accepted_implies_securityOutcome
    (key : Key Commitment PublicInput Scalar State shape columns blockCount width)
    (running : Running K Commitment PublicInput shape)
    (fresh : Fresh Commitment PublicInput shape)
    (proof : Proof shape width)
    (result : CE.Instance (RelationSource shape columns blockCount) PublicInput
      (CubePoint K shape.cubeVariables) (EvaluationFamily K shape) Commitment)
    (accepted : verify key running fresh proof = some result)
    (freshBound : key.params.b = 2)
    (challengeSetSize bound : Nat)
    (laws : ExtractionAlgebra key.piRlcSemantics key.params key.piRlcAlgebra)
    (strongSet : StrongSetUnits laws.ring key.piRlcAlgebra.challengeValid)
    (oneEq : key.one = laws.ring.one)
    (normLaw : Rlc.ExtractionNormLaw laws)
    (fork : key.Fork running fresh proof bound) :
    let witness := key.extractedWitness running fresh proof bound laws strongSet fork
    Knowledge key running fresh bound witness ∨
      SignedCoefficientObject.MixingRoot key.extensionOps
        ((PiCCS.protocolData key.baseOps key.lift
          (((key.statement running fresh).sourceConnectedInputs witness).toUnifiedInputs
            key.baseOps)).toJointData key.extensionOps)
        (key.probe running fresh proof).coins.alpha
        (key.probe running fresh proof).coins.gamma ∨
      PiCCS.SumCheckFailure key.extensionOps key.lift (key.statement running fresh) width
        challengeSetSize (key.probe running fresh proof) witness :=
  accepted_implies_securityOutcome_normFree key running fresh proof result accepted freshBound
    challengeSetSize bound laws strongSet oneEq normLaw fork.forgetNorms fork.baseHolds.1.2.2

end NightstreamFPrime.Spec.Folding.Light
