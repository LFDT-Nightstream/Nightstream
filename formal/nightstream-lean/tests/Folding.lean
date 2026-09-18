import Nightstream.SuperNeo.Folding.Composition

/-! Non-vacuity and adversarial tests for the batch-faithful folding theorems. -/

namespace NightstreamTests.Folding

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Folding

def unitParams : GlobalParams where
  q := 97
  b := 2
  k := 2
  maxFresh := 1
  expansionT := 1
  rlc_bound := by decide

def bootstrapArity : BatchArity unitParams :=
  BatchArity.bootstrap unitParams 1 (by decide) (by decide)

def activeArity : BatchArity unitParams :=
  BatchArity.active unitParams 1 (by decide) (by decide)

example : bootstrapArity.total = 1 := by decide
example : activeArity.total = 3 := by decide

def unitSemantics : RelationSemantics Unit Unit Unit Unit Unit Unit where
  commit := fun _ => ()
  projectPublicInput := fun _ => ()
  normBounded := fun _ _ => True
  ccsSatisfied := fun _ _ => True
  evaluationPointValid := fun _ _ => True
  evaluations := fun _ _ _ => #[()]

def unitCCS : CCS.Instance Unit Unit Unit where
  constraintSystem := ()
  commitment := ()
  publicInput := ()
  stage := .fresh

def unitCE (stage : NormStage) : CE.Instance Unit Unit Unit Unit Unit where
  constraintSystem := ()
  commitment := ()
  publicInput := ()
  point := ()
  evaluations := #[()]
  stage := stage

def unitSumOps : SumCheck.Ops Unit Unit where
  zero := ()
  one := ()
  add := fun _ _ => ()

def unitSumcheck : SumCheck.Instance Unit Unit where
  claimedInitial := ()
  trueInitial := ()
  terminal := ()
  rounds := []
  maxDegree := 0
  challengeSetSize := 97

def unitCCSInputs : PiCCS.InputProduct Unit Unit Unit Unit Unit unitParams activeArity where
  fresh := fun _ => unitCCS
  running := fun _ => unitCE .fresh

def unitCCSOutputs : Fin activeArity.total → CE.Instance Unit Unit Unit Unit Unit :=
  fun _ => unitCE .fresh

def unitCCSAttempt : PiCCS.Attempt
    Unit Unit Unit Unit Unit Unit Unit unitParams activeArity where
  inputs := unitCCSInputs
  outputs := unitCCSOutputs
  fe := unitSumcheck
  nc := unitSumcheck

theorem unitSourceFresh (i) :
    (unitCCSInputs.source i).stage = .fresh := by
  exact unitCCSInputs.sourceCases (fun source => source.stage = NormStage.fresh)
    (fun _ => rfl) (fun _ => rfl) i

theorem unitSourceValid (i) :
    (unitCCSInputs.source i).Holds unitSemantics unitParams () := by
  exact unitCCSInputs.sourceCases
    (fun source => source.Holds unitSemantics unitParams ())
    (fun _ => ⟨⟨rfl, rfl, trivial⟩, trivial⟩)
    (fun _ => ⟨⟨rfl, rfl, trivial⟩, trivial, rfl⟩) i

theorem unitPayloadTruth (i) :
    PiCCS.Source.PayloadTruth unitSemantics (unitCCSInputs.source i) () := by
  exact unitCCSInputs.sourceCases
    (fun source => PiCCS.Source.PayloadTruth unitSemantics source ())
    (fun _ => trivial) (fun _ => ⟨trivial, rfl⟩) i

theorem unitCCSAccepted : PiCCS.Accepted unitSumOps unitCCSAttempt := by
  exact ⟨{
    sourceFresh := unitSourceFresh
    outputFresh := fun _ => rfl
    sameStructure := fun _ => rfl
    sameCommitment := fun _ => rfl
    samePublicInput := fun _ => rfl
    sharedOutputPoint := fun _ _ => rfl
  }, by simp [SumCheck.Accepted, SumCheck.Chain, unitCCSAttempt, unitSumcheck],
    by simp [SumCheck.Accepted, SumCheck.Chain, unitCCSAttempt, unitSumcheck]⟩

theorem unitCCSArithmetization (assignments : Fin activeArity.total → Unit) :
    PiCCS.Arithmetization unitSemantics unitParams unitSumOps
      unitCCSAttempt assignments := by
  constructor
  · simp [SumCheck.TruthPath, SumCheck.Chain, unitCCSAttempt, unitSumcheck]
  · simp [SumCheck.TruthPath, SumCheck.Chain, unitCCSAttempt, unitSumcheck]
  · constructor
    · intro _ i
      simpa using unitPayloadTruth i
    · intro _
      rfl
  · constructor
    · intro _ _
      trivial
    · intro _
      rfl

theorem unitOutputsAmbientValid (assignments : Fin activeArity.total → Unit) :
    ∀ i, CE.Holds unitSemantics unitParams
      (PiCCS.relaxedOutput (unitCCSAttempt.outputs i)) (assignments i) := by
  intro i
  exact ⟨⟨rfl, rfl, trivial⟩, trivial, rfl⟩

example :
    (∀ i, (unitCCSAttempt.inputs.source i).Holds unitSemantics unitParams ()) ∨
      Nonempty (PiCCS.BadChallenge unitCCSAttempt) :=
  PiCCS.strong_extract_or_bad_challenge unitSemantics unitParams unitSumOps
    unitCCSAttempt (fun _ => ()) unitCCSAccepted
    (unitCCSArithmetization (fun _ => ()))
    (unitOutputsAmbientValid (fun _ => ()))

/-- Product completeness covers the paper's active `K+k` arity. -/
example : ∀ i, CE.Holds unitSemantics unitParams
    (PiCCS.honestOutputs unitSemantics unitCCSInputs (fun _ => ()) () i) () :=
  PiCCS.product_complete unitSemantics unitParams activeArity unitCCSInputs
    (fun _ => ()) () unitSourceFresh unitSourceValid (fun _ => trivial)

/-! A false joint FE claim must expose a phase-tagged bad challenge. -/

def boolSemantics : RelationSemantics Unit Bool Unit Unit Unit Unit where
  commit := fun _ => ()
  projectPublicInput := fun _ => ()
  normBounded := fun _ _ => True
  ccsSatisfied := fun _ assignment => assignment = true
  evaluationPointValid := fun _ _ => True
  evaluations := fun _ _ _ => #[()]

def expectedPolynomial : Nat → Nat
  | 0 => 2
  | 1 => 3
  | _ => 7

def forgedPolynomial : Nat → Nat
  | 0 => 4
  | 1 => 4
  | _ => 7

def natSumOps : SumCheck.Ops Nat Nat where
  zero := 0
  one := 1
  add := Nat.add

def forgedSumcheck : SumCheck.Instance Nat Nat where
  claimedInitial := 8
  trueInitial := 5
  terminal := 7
  rounds := [{
    claimed := forgedPolynomial
    expected := expectedPolynomial
    challenge := 2
    degree := 2
  }]
  maxDegree := 2
  challengeSetSize := 97

def trueNatSumcheck : SumCheck.Instance Nat Nat where
  claimedInitial := 0
  trueInitial := 0
  terminal := 0
  rounds := []
  maxDegree := 0
  challengeSetSize := 97

def boolCCSInputs : PiCCS.InputProduct Unit Unit Unit Unit Unit unitParams bootstrapArity where
  fresh := fun _ => unitCCS
  running := fun i => Fin.elim0 i

def forgedCCSAttempt : PiCCS.Attempt
    Unit Unit Unit Unit Unit Nat Nat unitParams bootstrapArity where
  inputs := boolCCSInputs
  outputs := fun _ => unitCE .fresh
  fe := forgedSumcheck
  nc := trueNatSumcheck

theorem forgedCCSAccepted : PiCCS.Accepted natSumOps forgedCCSAttempt := by
  exact ⟨{
    sourceFresh := fun i => boolCCSInputs.sourceCases
      (fun source => source.stage = NormStage.fresh)
      (fun _ => rfl) (fun empty => Fin.elim0 empty) i
    outputFresh := fun _ => rfl
    sameStructure := fun _ => rfl
    sameCommitment := fun _ => rfl
    samePublicInput := fun _ => rfl
    sharedOutputPoint := fun _ _ => rfl
  }, by
    simp [SumCheck.Accepted, SumCheck.Chain, forgedCCSAttempt, forgedSumcheck,
      forgedPolynomial, natSumOps], by
    simp [SumCheck.Accepted, SumCheck.Chain, forgedCCSAttempt, trueNatSumcheck]⟩

theorem forgedCCSArithmetization :
    PiCCS.Arithmetization boolSemantics unitParams natSumOps forgedCCSAttempt
      (fun _ => false) := by
  constructor
  · simp [SumCheck.TruthPath, SumCheck.Chain, forgedCCSAttempt, forgedSumcheck,
      expectedPolynomial, natSumOps]
  · simp [SumCheck.TruthPath, SumCheck.Chain, forgedCCSAttempt, trueNatSumcheck]
  · constructor
    · intro claim
      simp [SumCheck.Claim.True, forgedCCSAttempt, forgedSumcheck] at claim
    · intro payloads
      have falsePayload := payloads ⟨0, by decide⟩
      have sourceZero : boolCCSInputs.source ⟨0, by decide⟩ = .ccs unitCCS := rfl
      change PiCCS.Source.PayloadTruth boolSemantics
        (boolCCSInputs.source ⟨0, by decide⟩) false at falsePayload
      rw [sourceZero] at falsePayload
      simp [PiCCS.Source.PayloadTruth, boolSemantics] at falsePayload
  · constructor
    · intro _ _
      trivial
    · intro _
      rfl

theorem forgedOutputsAmbientValid : ∀ i,
    CE.Holds boolSemantics unitParams
      (PiCCS.relaxedOutput (forgedCCSAttempt.outputs i)) false := by
  intro i
  exact ⟨⟨rfl, rfl, trivial⟩, trivial, rfl⟩

example : Nonempty (PiCCS.BadChallenge forgedCCSAttempt) := by
  rcases PiCCS.strong_extract_or_bad_challenge boolSemantics unitParams natSumOps
      forgedCCSAttempt (fun _ => false) forgedCCSAccepted
      forgedCCSArithmetization forgedOutputsAmbientValid with valid | bad
  · have sourceValid := valid ⟨0, by decide⟩
    exact False.elim (by
      simpa [PiCCS.Source.Holds, boolSemantics, boolCCSInputs,
        PiCCS.InputProduct.source] using sourceValid.2)
  · exact bad

/-! Lawful toy Π_RLC/Π_DEC instantiations exercise the universal chain. -/

def unitRLCAlgebra : PiRLC.Algebra Unit Unit Unit Unit Unit Unit Unit
    unitSemantics unitParams where
  challengeValid := fun _ => True
  combineAssignment := fun _ _ => ()
  combineCommitment := fun _ _ => ()
  combinePublicInput := fun _ _ => ()
  combineEvaluations := fun _ _ => #[()]
  commit_hom := by intros; rfl
  publicInput_hom := by intros; rfl
  evaluations_hom := by intros; rfl
  norm_growth := by intros; trivial

def unitRLCInputs : Fin activeArity.total → CE.Instance Unit Unit Unit Unit Unit :=
  fun _ => unitCE .fresh

def unitRLCChallenges : Fin activeArity.total → Unit := fun _ => ()
def unitRLCAssignments : Fin activeArity.total → Unit := fun _ => ()

def unitRLCOutput : CE.Instance Unit Unit Unit Unit Unit :=
  PiRLC.combinedOutput unitRLCAlgebra () () unitRLCInputs unitRLCChallenges

def unitRLCAttempt : PiRLC.Attempt
    Unit Unit Unit Unit Unit Unit unitParams activeArity where
  inputs := unitRLCInputs
  challenges := unitRLCChallenges
  output := unitRLCOutput

theorem unitInputValid (i) :
    CE.Holds unitSemantics unitParams (unitRLCInputs i) (unitRLCAssignments i) := by
  exact ⟨⟨rfl, rfl, trivial⟩, trivial, rfl⟩

theorem unitRLCAccepted : PiRLC.Accepted unitRLCAlgebra unitRLCAttempt := by
  exact {
    inputFresh := fun _ => rfl
    sameStructure := fun _ => rfl
    samePoint := fun _ => rfl
    challengesValid := fun _ => trivial
    outputCombined := rfl
    commitmentEquation := rfl
    publicInputEquation := rfl
    evaluationEquation := rfl
  }

example : CE.Holds unitSemantics unitParams unitRLCOutput () :=
  PiRLC.combinedOutput_holds unitSemantics unitParams unitRLCAlgebra activeArity () ()
    unitRLCInputs unitRLCChallenges unitRLCAssignments
    (fun _ => rfl) (fun _ => rfl) (fun _ => rfl) (fun _ => trivial)
    unitInputValid trivial

def malformedRLCAttempt : PiRLC.Attempt
    Unit Unit Unit Unit Unit Unit unitParams activeArity :=
  { unitRLCAttempt with output := { unitRLCOutput with stage := .fresh } }

example : ¬ PiRLC.Accepted unitRLCAlgebra malformedRLCAttempt := by
  intro accepted
  cases accepted.outputCombined

def unitDECAlgebra : PiDEC.Algebra Unit Unit Unit Unit Unit Unit
    unitSemantics unitParams where
  splitAssignment := fun _ _ => ()
  recomposeAssignment := fun _ => ()
  recomposeCommitment := fun _ => ()
  recomposePublicInput := fun _ => ()
  recomposeEvaluations := fun _ => #[()]
  split_recompose := by intros; rfl
  split_norm := by intros; trivial
  recompose_norm := by intros; trivial
  commit_hom := by intros; rfl
  publicInput_hom := by intros; rfl
  evaluations_hom := by intros; rfl

def unitDECAttempt : PiDEC.Attempt Unit Unit Unit Unit Unit unitParams where
  parent := unitRLCOutput
  children := PiDEC.childrenOf unitDECAlgebra unitRLCOutput ()

theorem unitParentValid : CE.Holds unitSemantics unitParams unitRLCOutput () := by
  exact ⟨⟨rfl, rfl, trivial⟩, trivial, rfl⟩

theorem unitDECAccepted : PiDEC.Accepted unitDECAlgebra unitDECAttempt :=
  (PiDEC.complete unitSemantics unitParams unitDECAlgebra unitRLCOutput () rfl
    unitParentValid).1

theorem unitFinalValid (i) :
    CE.Holds unitSemantics unitParams (unitDECAttempt.children i) () :=
  (PiDEC.complete unitSemantics unitParams unitDECAlgebra unitRLCOutput () rfl
    unitParentValid).2 i

def noSamplingFailure : PiRLC.SamplingBoundary activeArity.total where
  Failure := False
  classify := False.elim

def unitExtracted :
    PiRLC.ExtractedAmbient unitSemantics unitParams unitRLCAttempt.inputs where
  assignments := fun _ => ()
  valid := fun _ => ⟨⟨rfl, rfl, trivial⟩, trivial, rfl⟩

def unitExtractor : Composition.WeakExtractor unitSemantics unitParams
    unitRLCAlgebra unitRLCAttempt noSamplingFailure where
  run := fun _ _ _ _ => .extracted unitExtracted

def unitBindingOps : PiRLC.RelaxedBindingOps Unit Unit Unit where
  scaleAssignment := fun _ _ => ()
  scaleCommitment := fun _ _ => ()
  differenceChallenge := fun _ => True

def unitUniqueness : PiRLC.UniquenessBridge unitSemantics unitParams
    unitBindingOps (n := activeArity.total) where
  disagreement_to_collision := by
    intro _ _ leftAssignments rightAssignments _ _ _ different
    exfalso
    apply different
    funext i
    exact Subsingleton.elim (leftAssignments i) (rightAssignments i)

theorem noUnitBadEvent :
    ¬ Composition.BadEvent unitSemantics unitParams unitBindingOps noSamplingFailure
      unitCCSAttempt unitRLCAttempt.inputs := by
  intro bad
  rcases bad with sumcheckBad | samplingOrBinding
  · rcases sumcheckBad with ⟨tagged⟩
    cases tagged with
    | fe round evidence =>
        rcases evidence with ⟨roundInTranscript, _⟩
        simp [unitCCSAttempt, unitSumcheck] at roundInTranscript
    | nc round evidence =>
        rcases evidence with ⟨roundInTranscript, _⟩
        simp [unitCCSAttempt, unitSumcheck] at roundInTranscript
  · rcases samplingOrBinding with sampling | binding
    · exact sampling
    · rcases binding with ⟨_, ⟨collision⟩⟩
      exact collision.crossDifferent rfl

/-- The full composed theorem takes the knowledge branch in a lawful active model. -/
example : Composition.InputsValid unitSemantics unitParams unitCCSAttempt := by
  have result := Composition.fold_knowledge_or_bad_event unitSemantics unitParams
    unitSumOps unitRLCAlgebra unitDECAlgebra unitBindingOps activeArity
    noSamplingFailure unitCCSAttempt unitRLCAttempt unitDECAttempt
    (fun _ => ()) (by decide) (fun _ => rfl) rfl unitCCSAccepted
    unitRLCAccepted unitDECAccepted unitFinalValid unitExtractor unitUniqueness
    (fun left _ _ _ _ => unitCCSArithmetization left)
  rcases result with valid | bad
  · exact valid
  · exact False.elim (noUnitBadEvent bad)

end NightstreamTests.Folding
