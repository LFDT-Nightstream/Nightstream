import NightstreamFPrime.Spec.Folding.Refresh.Completeness
import NightstreamFPrime.Spec.Folding.Light.Bound
import NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.Algebra

/-! The honest refresh parent is the Π_RLC combination of its 17 sources, so its norm is at most
`17 · T · (b − 1)`, the accumulator bound at depth 0 (SuperNeo v1.2 Definition 22). Light folds
after a refresh start from this bound. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Folding.Refresh

open NightstreamFPrime.Spec
open _root_.NightstreamFPrime.Spec.SumCheck.Finite
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.StrongReduction
open NightstreamFPrime.Spec.Folding.Nifs.PaperNonInteractive (Key Fresh Running)

universe uStructure uAssignment uPublicInput uPoint uEvaluation uCommitment uScalar uState

/-- SuperNeo v1.2 Definition 22, per source: `n` combined sources of norm below `b` stay within
`n · T · (b − 1)`. -/
structure CountNormLaw
    {Structure : Type uStructure} {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput} {Point : Type uPoint} {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment} {Scalar : Type uScalar}
    {semantics : RelationSemantics Structure Assignment PublicInput Point Evaluation Commitment}
    {params : GlobalParams}
    (algebra : PiRLC.Algebra Structure Assignment PublicInput Point Evaluation Commitment Scalar
      semantics params) : Prop where
  count_bound : ∀ {count : Nat} (challenges : Fin count → Scalar)
      (assignments : Fin count → Assignment),
    (∀ index, algebra.challengeValid (challenges index)) →
    (∀ index, semantics.normBounded params.b (assignments index)) →
    semantics.normBounded (count * params.expansionT * (params.b - 1) + 1)
      (algebra.combineAssignment challenges assignments)

theorem production_countNormLaw {shape : Phi81Relation.Shape} {verifierRows : Nat}
    (key : Phi81Relation.PiRLCAlgebra.Commitment.Key shape verifierRows) :
    CountNormLaw (Phi81Relation.PiRLCAlgebra.Algebra.concrete key) := by
  constructor
  intro count challenges assignments valid bounded column
  have fresh : ∀ index, Phi81Relation.assignmentNormBounded 2 (assignments index) := by
    intro index
    simpa [Phi81Relation.relationSemantics, productionGlobalParams] using bounded index
  have finite := Phi81Relation.PiRLCAlgebra.Norm.Finite.combineAssignments_le challenges
    assignments valid fresh column
  change centeredMagnitude
      (Phi81Relation.EvaluationHomomorphism.PiRLCFinite.combineAssignments challenges
        assignments column) < count * 216 * (2 - 1) + 1
  omega

variable {Commitment : Type uCommitment} {PublicInput : Type uPublicInput}
  {Scalar : Type uScalar} {State : Type uState} {shape : Shape}
  {columns blockCount degreeBound : Nat}

/-- The parent of honest Π_CCS outputs is the combination of their witnesses, within the
depth-0 accumulator bound. -/
theorem parent_bounded
    (key : Key K Commitment PublicInput Scalar State shape columns blockCount degreeBound)
    (normLaw : CountNormLaw key.piRlcAlgebra)
    (freshOne : shape.freshCount = 1)
    (running : Running K Commitment PublicInput shape)
    (fresh : Fresh Commitment PublicInput shape)
    (nifs : Nifs.PaperNonInteractive.Proof K Commitment shape degreeBound)
    (challenges : Fin key.arity.total → Scalar)
    (valid : ∀ index, key.piRlcAlgebra.challengeValid (challenges index))
    (witness : OutputWitness shape columns)
    (outputsHold : ∀ source, CE.Holds key.piRlcSemantics key.params
      ((key.statement running fresh).publicOutput (key.piCcsProbe running fresh nifs) source)
      (witness.assignments source)) :
    let parentWitness := PiRLC.combinedWitness key.piRlcAlgebra challenges
      (fun index => witness.assignments (Fin.cast key.total_eq_sourceCount index))
    CE.Holds key.piRlcSemantics key.params
        (key.parentForChallenges running fresh nifs challenges) parentWitness ∧
      key.piRlcSemantics.normBounded (key.params.accumulatorBound 0 + 1) parentWitness := by
  intro parentWitness
  have total : key.arity.total = 1 + key.params.k := by
    rw [key.total_eq_sourceCount]
    simp only [Shape.sourceCount, freshOne, key.outputCount_eq]
  have sourceHolds : ∀ index, CE.Holds key.piRlcSemantics key.params
      (key.piCcsOutputs running fresh nifs index)
      (witness.assignments (Fin.cast key.total_eq_sourceCount index)) :=
    fun index => outputsHold (Fin.cast key.total_eq_sourceCount index)
  have first : 0 < key.arity.total := key.arity.totalPositive
  refine ⟨PiRLC.combinedOutput_holds key.piRlcSemantics key.params key.piRlcAlgebra key.arity
    key.relationSource (key.piCcsExecution running fresh nifs).coins.roundPoint
    (key.piCcsOutputs running fresh nifs) challenges _ (fun _ => rfl) (fun _ => rfl)
    (fun _ => rfl) valid sourceHolds (sourceHolds ⟨0, first⟩).2.1, ?_⟩
  have bounded := normLaw.count_bound challenges
    (fun index => witness.assignments (Fin.cast key.total_eq_sourceCount index)) valid
    (fun index => (sourceHolds index).1.2.2)
  have numbers : key.arity.total * key.params.expansionT * (key.params.b - 1) + 1 =
      key.params.accumulatorBound 0 + 1 := by
    rw [total]
    simp [GlobalParams.accumulatorBound]
  rw [numbers] at bounded
  exact bounded

/-- Refresh completeness with the depth-0 bound on the parent: light folds after an honest
refresh start from `accumulatorBound 0`. -/
theorem complete_bounded
    (key : Key K Commitment PublicInput Scalar State shape columns blockCount degreeBound)
    (normLaw : CountNormLaw key.piRlcAlgebra)
    (freshOne : shape.freshCount = 1)
    (accumulator : Accumulator Commitment PublicInput shape columns blockCount)
    (fresh : Fresh Commitment PublicInput shape)
    (assignment : PaperLinearAlgebra.Assignment F columns)
    (freshWitness : Fin shape.freshCount -> PaperLinearAlgebra.Assignment F columns)
    (system : accumulator.constraintSystem = key.relationSource)
    (combined : accumulator.stage = .combined)
    (accumulatorHolds : CE.Holds key.piRlcSemantics key.params accumulator assignment)
    (freshHolds : ∀ source, CCS.Holds
      (paperRelationSemantics key.baseOps key.extensionOps key.lift key.openingMaps) key.params
      (SourceMembership.freshInstance
        (key.statement (children key accumulator (honestSplit key accumulator assignment)) fresh)
        source)
      (freshWitness source)) :
    let running := children key accumulator (honestSplit key accumulator assignment)
    ∃ (messages : Fin shape.cubeVariables → FixedPolynomial K degreeBound)
      (fullOutput : FullOutputCoordinates.FullOutput K shape),
      let coins := FiatShamir.derive key.oracle.transcript
        ({ priorState := key.publicInputState running fresh
           input := (key.statement running fresh).verifierInput key.lift } :
          PiCCS.TranscriptReplay.Statement K State shape)
        { rounds := fun round => (messages round).toMessage }
      ∀ rho : Fin key.arity.total → Scalar,
        key.piRlcResponse (key.absorbPiCcsOutput coins.finalState fullOutput) = some rho →
        ∃ (proof : Proof Commitment shape degreeBound)
          (parent : Accumulator Commitment PublicInput shape columns blockCount)
          (parentWitness : PaperLinearAlgebra.Assignment F columns),
          proof.split = honestSplit key accumulator assignment ∧
          proof.nifs.piCcsRounds = messages ∧
          verify key accumulator fresh proof = some parent ∧
          CE.Holds key.piRlcSemantics key.params parent parentWitness ∧
          key.piRlcSemantics.normBounded (key.params.accumulatorBound 0 + 1) parentWitness := by
  intro running
  let split := honestSplit key accumulator assignment
  let witness := honestWitness key assignment freshWitness
  obtain ⟨inputAccepted, valid⟩ := honest_inputs key accumulator fresh assignment freshWitness
    system combined accumulatorHolds freshHolds
  obtain ⟨messages, fullOutput, _, openings, suffix⟩ :=
    Nifs.PaperNonInteractive.Completeness.exists_honest_proof_of_sampler_success key running fresh
      witness valid
  refine ⟨messages, fullOutput, ?_⟩
  intro _ rho sampled
  obtain ⟨nifs, result, _, roundsEq, outputEq, challengesEq, verified, _⟩ := suffix rho sampled
  have checks := (Nifs.PaperNonInteractive.verify_eq_some_iff key running fresh nifs result).mp
    verified
  have parentEq : key.parent running fresh nifs =
      some (key.parentForChallenges running fresh nifs rho) := by
    simp [Nifs.PaperNonInteractive.Key.parent, challengesEq]
  have outputsHold : ∀ source, CE.Holds key.piRlcSemantics key.params
      ((key.statement running fresh).publicOutput (key.piCcsProbe running fresh nifs) source)
      (witness.assignments source) := by
    intro source
    have opened := openings source
    simp only [StrongReduction.Statement.publicOutput] at opened ⊢
    rw [Nifs.PaperNonInteractive.Key.piCcsProbe, key.piCcsExecution_coins_eq_derive, roundsEq,
      outputEq]
    exact opened
  obtain ⟨parentHolds, parentBounded⟩ := parent_bounded key normLaw freshOne running fresh nifs
    rho (fun index => key.piRlcResponseValid _ rho sampled index) witness outputsHold
  refine ⟨{ split := split, nifs := nifs }, key.parentForChallenges running fresh nifs rho, _,
    rfl, roundsEq, ?_, parentHolds, parentBounded⟩
  exact (verify_eq_some_iff key accumulator fresh _ _).mpr ⟨inputAccepted, checks.1, parentEq⟩

end NightstreamFPrime.Spec.Folding.Refresh
