import NightstreamFPrime.Spec.Folding.Light.Rlc
import NightstreamFPrime.Spec.Folding.PiRLC.PaperForkExtraction

/-! SuperNeo v1.2 Lemma 8 for the light Π_RLC. Two accepted outputs that differ only in the
fresh challenge give the fresh witness `(ρ − ρ')⁻¹ (w − w')` and the accumulator witness
`w − ρ · fresh`. The accumulator keeps slack 1 (Cyclo, ePrint 2026/359, Theorem 3). -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Folding.Light.Rlc

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiRLC.PaperForkAlgebra
open NightstreamFPrime.Spec.Folding.PiRLC.PaperForkExtraction
  (ExtractionAlgebra StrongSetUnits LinearMapLaws)

universe uStructure uAssignment uPublicInput uPoint uEvaluation uCommitment uScalar uSource
  uTarget

section Algebra

variable {Scalar : Type uScalar} {Target : Type uTarget}

theorem linearCombination_pair (ring : CommutativeRingOps Scalar)
    (module : ModuleOps Scalar Target) (challenge one : Scalar) (fresh running : Target) :
    linearCombination ring module (pair challenge one) (pair fresh running) =
      module.add (module.smul challenge fresh)
        (module.add (module.smul one running) module.zero) := by
  simp [linearCombination, pair]

theorem agree_pair (challenge other one : Scalar) :
    AgreeExcept (0 : Fin 2) (pair challenge one) (pair other one) := by
  intro index notZero
  have valueNotZero : index.val ≠ 0 := fun value => notZero (Fin.ext value)
  simp [pair, valueNotZero]

theorem sub_add_cancel_left (ring : CommutativeRingOps Scalar)
    (module : ModuleOps Scalar Target) (laws : ModuleLaws ring module)
    (shift value : Target) :
    module.sub (module.add shift value) shift = value := by
  unfold ModuleOps.sub
  rw [laws.add_comm shift value, laws.add_assoc, laws.add_neg, laws.add_zero]

theorem fork_map {Source : Type uSource}
    (ring : CommutativeRingOps Scalar) (ringLaws : CommutativeRingLaws ring)
    (sourceModule : ModuleOps Scalar Source) (targetModule : ModuleOps Scalar Target)
    (targetLaws : ModuleLaws ring targetModule) (map : Source → Target)
    (mapLaws : LinearMapLaws sourceModule targetModule map)
    (baseChallenge forkChallenge : Scalar)
    (unit : UnitWitness ring (ring.sub baseChallenge forkChallenge))
    (fresh running : Target) (base fork : Source)
    (baseImage : map base =
      linearCombination ring targetModule (pair baseChallenge ring.one) (pair fresh running))
    (forkImage : map fork =
      linearCombination ring targetModule (pair forkChallenge ring.one) (pair fresh running)) :
    map (sourceModule.smul unit.inverse (sourceModule.sub base fork)) = fresh ∧
      map (sourceModule.sub base (sourceModule.smul baseChallenge
        (sourceModule.smul unit.inverse (sourceModule.sub base fork)))) = running := by
  have freshImage : map (sourceModule.smul unit.inverse (sourceModule.sub base fork)) = fresh := by
    rw [mapLaws.map_smul, mapLaws.map_sub, baseImage, forkImage,
      coordinateIsolation ring targetModule ringLaws targetLaws _ _ _ (0 : Fin 2)
        (agree_pair baseChallenge forkChallenge ring.one)]
    exact inverseActionCancellation ring targetModule targetLaws _ unit fresh
  refine ⟨freshImage, ?_⟩
  rw [mapLaws.map_sub, mapLaws.map_smul, freshImage, baseImage, linearCombination_pair,
    targetLaws.one_smul, targetLaws.add_zero]
  exact sub_add_cancel_left ring targetModule targetLaws _ _

end Algebra

variable {Structure : Type uStructure} {Assignment : Type uAssignment}
  {PublicInput : Type uPublicInput} {Point : Type uPoint} {Evaluation : Type uEvaluation}
  {Commitment : Type uCommitment} {Scalar : Type uScalar}
  {semantics : RelationSemantics Structure Assignment PublicInput Point Evaluation Commitment}
  {params : GlobalParams}

/-- Two accepted light outputs with the same inputs and different fresh challenges. -/
structure Fork
    (algebra : PiRLC.Algebra Structure Assignment PublicInput Point Evaluation Commitment Scalar
      semantics params)
    (one : Scalar) (bound : Nat) (system : Structure) (point : Point)
    (fresh running : CE.Instance Structure PublicInput Point Evaluation Commitment) where
  baseChallenge : Scalar
  forkChallenge : Scalar
  baseAssignment : Assignment
  forkAssignment : Assignment
  baseValid : algebra.challengeValid baseChallenge
  forkValid : algebra.challengeValid forkChallenge
  changed : baseChallenge ≠ forkChallenge
  baseHolds : Holds semantics bound
    (output algebra one baseChallenge system point fresh running) baseAssignment
  forkHolds : Holds semantics bound
    (output algebra one forkChallenge system point fresh running) forkAssignment

namespace Fork

variable {algebra : PiRLC.Algebra Structure Assignment PublicInput Point Evaluation Commitment
    Scalar semantics params}
  {one : Scalar} {bound : Nat} {system : Structure} {point : Point}
  {fresh running : CE.Instance Structure PublicInput Point Evaluation Commitment}

def unit (laws : ExtractionAlgebra semantics params algebra)
    (strongSet : StrongSetUnits laws.ring algebra.challengeValid)
    (fork : Fork algebra one bound system point fresh running) :
    UnitWitness laws.ring (laws.ring.sub fork.baseChallenge fork.forkChallenge) :=
  strongSet.differenceUnit fork.baseValid fork.forkValid fork.changed

def freshAssignment (laws : ExtractionAlgebra semantics params algebra)
    (strongSet : StrongSetUnits laws.ring algebra.challengeValid)
    (fork : Fork algebra one bound system point fresh running) : Assignment :=
  laws.assignmentModule.smul (fork.unit laws strongSet).inverse
    (laws.assignmentModule.sub fork.baseAssignment fork.forkAssignment)

def runningAssignment (laws : ExtractionAlgebra semantics params algebra)
    (strongSet : StrongSetUnits laws.ring algebra.challengeValid)
    (fork : Fork algebra one bound system point fresh running) : Assignment :=
  laws.assignmentModule.sub fork.baseAssignment
    (laws.assignmentModule.smul fork.baseChallenge (fork.freshAssignment laws strongSet))

end Fork

/-- SuperNeo v1.2 Lemma 8 for one fresh source and one running source with coefficient 1. -/
theorem fork_extracts
    {algebra : PiRLC.Algebra Structure Assignment PublicInput Point Evaluation Commitment
      Scalar semantics params}
    {one : Scalar} {bound : Nat} {system : Structure} {point : Point}
    {fresh running : CE.Instance Structure PublicInput Point Evaluation Commitment}
    (laws : ExtractionAlgebra semantics params algebra)
    (strongSet : StrongSetUnits laws.ring algebra.challengeValid)
    (oneEq : one = laws.ring.one)
    (fork : Fork algebra one bound system point fresh running)
    (freshSystem : fresh.constraintSystem = system) (freshPoint : fresh.point = point)
    (runningSystem : running.constraintSystem = system) (runningPoint : running.point = point)
    (count : Nat) (freshSize : fresh.evaluations.size = count)
    (runningSize : running.evaluations.size = count) :
    PiRLC.PaperCorrections.CorrectedAmbientHolds semantics params fresh
        (fork.freshAssignment laws strongSet) ∧
      PiRLC.PaperCorrections.CorrectedAmbientHolds semantics params running
        (fork.runningAssignment laws strongSet) := by
  subst oneEq
  let unit := fork.unit laws strongSet
  have sizes : ∀ source, (pair fresh.evaluations running.evaluations source).size = count := by
    intro source
    by_cases first : source.val = 0 <;> simp [pair, first, freshSize, runningSize]
  have commitImages := fork_map laws.ring laws.ringLaws laws.assignmentModule
    laws.commitmentModule laws.commitmentLaws semantics.commit laws.commitMap
    fork.baseChallenge fork.forkChallenge unit fresh.commitment running.commitment
    fork.baseAssignment fork.forkAssignment
    (fork.baseHolds.1.1.trans (laws.combineCommitment_eq _ _))
    (fork.forkHolds.1.1.trans (laws.combineCommitment_eq _ _))
  have publicImages := fork_map laws.ring laws.ringLaws laws.assignmentModule
    laws.publicInputModule laws.publicInputLaws semantics.projectPublicInput laws.publicInputMap
    fork.baseChallenge fork.forkChallenge unit fresh.publicInput running.publicInput
    fork.baseAssignment fork.forkAssignment
    (fork.baseHolds.1.2.1.trans (laws.combinePublicInput_eq _ _))
    (fork.forkHolds.1.2.1.trans (laws.combinePublicInput_eq _ _))
  have baseEq : semantics.evaluations system fork.baseAssignment point =
      algebra.combineEvaluations (pair fork.baseChallenge laws.ring.one)
        (pair fresh.evaluations running.evaluations) := fork.baseHolds.2.2
  have forkEq : semantics.evaluations system fork.forkAssignment point =
      algebra.combineEvaluations (pair fork.forkChallenge laws.ring.one)
        (pair fresh.evaluations running.evaluations) := fork.forkHolds.2.2
  have getDPair : ∀ index,
      (fun source => (pair fresh.evaluations running.evaluations source).getD index
        laws.evaluationModule.zero) =
      pair (fresh.evaluations.getD index laws.evaluationModule.zero)
        (running.evaluations.getD index laws.evaluationModule.zero) := by
    intro index
    funext source
    by_cases first : source.val = 0 <;> simp [pair, first]
  have evaluationImages := fun index => fork_map laws.ring laws.ringLaws laws.assignmentModule
    laws.evaluationModule laws.evaluationLaws
    (fun assignment =>
      (semantics.evaluations system assignment point).getD index laws.evaluationModule.zero)
    (laws.evaluationsMap system point index)
    fork.baseChallenge fork.forkChallenge unit
    (fresh.evaluations.getD index laws.evaluationModule.zero)
    (running.evaluations.getD index laws.evaluationModule.zero)
    fork.baseAssignment fork.forkAssignment
    (by
      change (semantics.evaluations system fork.baseAssignment point).getD index
        laws.evaluationModule.zero = _
      rw [baseEq, laws.combineEvaluations_getD _ _ count index (by decide) sizes, getDPair])
    (by
      change (semantics.evaluations system fork.forkAssignment point).getD index
        laws.evaluationModule.zero = _
      rw [forkEq, laws.combineEvaluations_getD _ _ count index (by decide) sizes, getDPair])
  have arrayEq : ∀ (assignment : Assignment) (target : Array Evaluation),
      target.size = count →
      (∀ index, (semantics.evaluations system assignment point).getD index
          laws.evaluationModule.zero = target.getD index laws.evaluationModule.zero) →
      semantics.evaluations system assignment point = target := by
    intro assignment target targetSize agree
    apply Array.ext
    · rw [laws.semanticEvaluations_size_eq system point assignment fork.baseAssignment, baseEq,
        laws.combineEvaluations_size _ _ count (by decide) sizes, targetSize]
    · intro index left right
      simpa [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem left,
        Array.getElem?_eq_getElem right] using agree index
  refine ⟨⟨⟨commitImages.1, publicImages.1, laws.correctedNormCoverage _⟩, ?_, ?_⟩,
    ⟨⟨commitImages.2, publicImages.2, laws.correctedNormCoverage _⟩, ?_, ?_⟩⟩
  · rw [freshSystem, freshPoint]
    exact fork.baseHolds.2.1
  · rw [freshSystem, freshPoint]
    exact arrayEq _ _ freshSize (fun index => (evaluationImages index).1)
  · rw [runningSystem, runningPoint]
    exact fork.baseHolds.2.1
  · rw [runningSystem, runningPoint]
    exact arrayEq _ _ runningSize (fun index => (evaluationImages index).2)

/-- Cyclo (ePrint 2026/359) Theorem 3, item 2: the accumulator keeps slack 1. -/
structure ExtractionNormLaw
    {algebra : PiRLC.Algebra Structure Assignment PublicInput Point Evaluation Commitment
      Scalar semantics params}
    (laws : ExtractionAlgebra semantics params algebra) : Prop where
  sub_bound : ∀ (bound : Nat) (challenge : Scalar) (accumulated fresh : Assignment),
    algebra.challengeValid challenge →
    semantics.normBounded bound accumulated →
    semantics.normBounded params.b fresh →
    semantics.normBounded (bound + params.expansionT * (params.b - 1))
      (laws.assignmentModule.sub accumulated (laws.assignmentModule.smul challenge fresh))

theorem runningAssignment_normBounded
    {algebra : PiRLC.Algebra Structure Assignment PublicInput Point Evaluation Commitment
      Scalar semantics params}
    {one : Scalar} {bound : Nat} {system : Structure} {point : Point}
    {fresh running : CE.Instance Structure PublicInput Point Evaluation Commitment}
    (laws : ExtractionAlgebra semantics params algebra)
    (strongSet : StrongSetUnits laws.ring algebra.challengeValid)
    (normLaw : ExtractionNormLaw laws)
    (fork : Fork algebra one bound system point fresh running)
    (freshBounded : semantics.normBounded params.b (fork.freshAssignment laws strongSet)) :
    semantics.normBounded (bound + params.expansionT * (params.b - 1))
      (fork.runningAssignment laws strongSet) :=
  normLaw.sub_bound bound fork.baseChallenge fork.baseAssignment
    (fork.freshAssignment laws strongSet) fork.baseValid fork.baseHolds.1.2.2 freshBounded

end NightstreamFPrime.Spec.Folding.Light.Rlc
