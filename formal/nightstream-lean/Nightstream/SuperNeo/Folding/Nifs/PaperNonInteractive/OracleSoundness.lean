import Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.CoordinateForkBridge
import Nightstream.SuperNeo.InteractiveReduction.ProbabilityCalculus

/-!
Conditional probability theorem for the frozen paper non-interactive NIFS.

Source: SuperNeo Sections 7.3--7.5 and Appendices D.3--D.6, together with
HyperNova Section 3's non-interactive multi-folding boundary.

Owns: one exact accepted-NIFS-without-target-witness predicate, four exact
nonzero residual interactive predicates, one independently stated bound per
predicate, their composition-ordered union bound, the specialization of the
six-event random-oracle contract to `AlignedForkOutcome`, the exact accepted
execution cover, and the final subtractive soundness inequality with error
`nonInteractiveTotal`.

Does not own: any event bound, an oracle distribution, an adversary or
extractor implementation, commitment binding, SumCheck root probabilities,
Poseidon2, Ajtai, Rust, R1CS, artifacts, minimality, or costs.

Emits constraints: no.

No contract field contains the desired transition or soundness conclusion.
Each field bounds one named predicate over the actual aligned NIFS outcome.
The `Pi_DEC` child-extraction predicate has its own NIFS extraction budget:
Theorem 7's zero intrinsic loss begins only after target witnesses exist.
Rejected transcripts do not consume that extraction budget.
-/

set_option autoImplicit false

namespace Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.InteractiveReduction.Paper
open Nightstream.SuperNeo.InteractiveReduction.FiatShamirContract
open Nightstream.SuperNeo.InteractiveReduction.ProbabilityCalculus

universe uExtension uCommitment uPublicInput uScalar uState uWeight

/-- Exact accepted `Pi_DEC` child-extraction event. Rejected transcripts do
not consume extraction budget. -/
def PiDecChildExtractionEvent
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    [DecidableEq Extension]
    {shape : PiCCS.PaperJoint.Shape}
    {columns blockCount degreeBound : Nat}
    {key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound}
    (outcome : AlignedForkOutcome key) : Prop :=
  verify key outcome.running outcome.fresh outcome.proof =
      some outcome.result /\
    PiDecChildExtractionFailure key outcome.running outcome.fresh outcome.proof

/-- Exact intrinsic `Pi_RLC` coordinate-fork sampling event. -/
def PiRlcForkSamplingEvent
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : PiCCS.PaperJoint.Shape}
    {columns blockCount degreeBound : Nat}
    {key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound}
    (outcome : AlignedForkOutcome key) : Prop :=
  CoordinateForkSamplingFailure outcome

/-- Exact `Pi_CCS` SumCheck bad-challenge event, with its source witness
existentially hidden from the probability predicate. -/
def PiCcsSumCheckEvent
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : PiCCS.PaperJoint.Shape}
    {columns blockCount degreeBound : Nat}
    {key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound}
    (outcome : AlignedForkOutcome key) : Prop :=
  exists sourceWitness : PiCCS.PaperJoint.StrongReduction.OutputWitness
      shape columns,
    PiCcsSumCheckCollision key outcome.running outcome.fresh outcome.proof
      sourceWitness

/-- Exact `Pi_CCS` mixing-polynomial root event. -/
def PiCcsMixingRootEvent
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : PiCCS.PaperJoint.Shape}
    {columns blockCount degreeBound : Nat}
    {key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound}
    (outcome : AlignedForkOutcome key) : Prop :=
  exists sourceWitness : PiCCS.PaperJoint.StrongReduction.OutputWitness
      shape columns,
    PiCcsMixingRoot key outcome.running outcome.fresh outcome.proof
      sourceWitness

/-- Exact parent-opening binding collision emitted by operational `Pi_DEC`. -/
def ParentBindingCollisionEvent
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : PiCCS.PaperJoint.Shape}
    {columns blockCount degreeBound : Nat}
    {key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound}
    (outcome : AlignedForkOutcome key) : Prop :=
  Nonempty (PiDEC.ParentOpeningBindingCollision key.semantics key.params
    (key.parent outcome.running outcome.fresh outcome.proof).commitment)

/-- Exact residual union in NIFS accounting order: target-witness extraction,
`Pi_RLC` fork sampling, `Pi_CCS` SumCheck plus mixing root, then adjusted
binding. -/
def ResidualFailure
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    [DecidableEq Extension]
    {shape : PiCCS.PaperJoint.Shape}
    {columns blockCount degreeBound : Nat}
    {key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound}
    (outcome : AlignedForkOutcome key) : Prop :=
  PiDecChildExtractionEvent outcome ∨
    PiRlcForkSamplingEvent outcome ∨
      (PiCcsSumCheckEvent outcome ∨ PiCcsMixingRootEvent outcome) ∨
        ParentBindingCollisionEvent outcome

/-- Under executable acceptance, the closed residual inductive family and the
explicit nested probability event are extensionally identical. -/
theorem residualBadEvent_iff_residualFailure
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    [DecidableEq Extension]
    {shape : PiCCS.PaperJoint.Shape}
    {columns blockCount degreeBound : Nat}
    {key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound}
    (outcome : AlignedForkOutcome key)
    (accepted :
      verify key outcome.running outcome.fresh outcome.proof =
        some outcome.result) :
    ResidualBadEvent key outcome ↔ ResidualFailure outcome := by
  constructor
  · intro event
    cases event with
    | piRlcForkSampling failure =>
        exact Or.inr (Or.inl failure)
    | piDecChildExtraction failure =>
        exact Or.inl ⟨accepted, failure⟩
    | piCcsMixingRoot sourceWitness root =>
        exact Or.inr (Or.inr (Or.inl (Or.inr ⟨sourceWitness, root⟩)))
    | piCcsSumCheckCollision sourceWitness collision =>
        exact Or.inr
          (Or.inr (Or.inl (Or.inl ⟨sourceWitness, collision⟩)))
    | parentBindingCollision collision =>
        exact Or.inr (Or.inr (Or.inr collision))
  · intro failure
    rcases failure with child | tail
    · exact .piDecChildExtraction child.2
    rcases tail with fork | tail
    · exact .piRlcForkSampling fork
    rcases tail with piCcs | binding
    · rcases piCcs with sumCheck | mixingRoot
      · rcases sumCheck with ⟨sourceWitness, collision⟩
        exact .piCcsSumCheckCollision sourceWitness collision
      · rcases mixingRoot with ⟨sourceWitness, root⟩
        exact .piCcsMixingRoot sourceWitness root
    · exact .parentBindingCollision binding

/-- Independently stated bounds for the five exact residual events. The first
is accepted NIFS execution without target witnesses and belongs to the NIFS
extraction boundary; the remaining four follow `InteractiveErrorBudget`. No
field may bound their union or the desired soundness conclusion directly. -/
structure InteractiveResidualContract
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {Weight : Type uWeight}
    [DecidableEq Extension]
    {shape : PiCCS.PaperJoint.Shape}
    {columns blockCount degreeBound : Nat}
    {key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound}
    {scale : ProbabilityScale Weight}
    (experiment : ProbabilityExperiment scale (AlignedForkOutcome key))
    (extractionBudget : NifsExtractionErrorBudget Weight)
    (budget : InteractiveErrorBudget Weight) : Prop where
  piDecChildExtraction :
    scale.le (experiment.probability PiDecChildExtractionEvent)
      extractionBudget.piDecTargetWitnessFailure
  piRlcForkSampling :
    scale.le (experiment.probability PiRlcForkSamplingEvent)
      budget.piRlcForkSampling
  piCcsSumCheck :
    scale.le (experiment.probability PiCcsSumCheckEvent)
      budget.piCcsSumCheck
  piCcsMixingRoot :
    scale.le (experiment.probability PiCcsMixingRootEvent)
      budget.piCcsSchwartzZippel
  parentBindingCollision :
    scale.le (experiment.probability ParentBindingCollisionEvent)
      budget.adjustedRelaxedBinding

/-- Five exact residual bounds imply the NIFS extraction plus
composition-ordered interactive total. This theorem is bookkeeping only; it
proves none of the contract fields. -/
theorem residualFailure_probability_le_total
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {Weight : Type uWeight}
    [DecidableEq Extension]
    {shape : PiCCS.PaperJoint.Shape}
    {columns blockCount degreeBound : Nat}
    {key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound}
    {scale : ProbabilityScale Weight}
    (scaleLaws : ScaleLaws scale)
    (experiment : ProbabilityExperiment scale (AlignedForkOutcome key))
    (unionLaw : UnionBound experiment)
    (extractionBudget : NifsExtractionErrorBudget Weight)
    (budget : InteractiveErrorBudget Weight)
    (contract :
      InteractiveResidualContract experiment extractionBudget budget) :
    scale.le (experiment.probability ResidualFailure)
      (nifsInteractiveTotal scale extractionBudget budget) := by
  have sumCheckMixing :
      scale.le
        (experiment.probability fun outcome =>
          PiCcsSumCheckEvent outcome ∨ PiCcsMixingRootEvent outcome)
        (scale.add budget.piCcsSumCheck
          budget.piCcsSchwartzZippel) :=
    scale.le_trans
      (unionLaw.unionBound PiCcsSumCheckEvent PiCcsMixingRootEvent)
      (scaleLaws.add_mono contract.piCcsSumCheck
        contract.piCcsMixingRoot)
  have piCcsBinding :
      scale.le
        (experiment.probability fun outcome =>
          (PiCcsSumCheckEvent outcome ∨ PiCcsMixingRootEvent outcome) ∨
            ParentBindingCollisionEvent outcome)
        (scale.add
          (scale.add budget.piCcsSumCheck budget.piCcsSchwartzZippel)
          budget.adjustedRelaxedBinding) :=
    scale.le_trans
      (unionLaw.unionBound
        (fun outcome =>
          PiCcsSumCheckEvent outcome ∨ PiCcsMixingRootEvent outcome)
        ParentBindingCollisionEvent)
      (scaleLaws.add_mono sumCheckMixing
        contract.parentBindingCollision)
  have forkTail :
      scale.le
        (experiment.probability fun outcome =>
          PiRlcForkSamplingEvent outcome ∨
            (PiCcsSumCheckEvent outcome ∨ PiCcsMixingRootEvent outcome) ∨
              ParentBindingCollisionEvent outcome)
        (scale.add budget.piRlcForkSampling
          (scale.add
            (scale.add budget.piCcsSumCheck budget.piCcsSchwartzZippel)
            budget.adjustedRelaxedBinding)) :=
    scale.le_trans
      (unionLaw.unionBound PiRlcForkSamplingEvent
        (fun outcome =>
          (PiCcsSumCheckEvent outcome ∨ PiCcsMixingRootEvent outcome) ∨
            ParentBindingCollisionEvent outcome))
      (scaleLaws.add_mono contract.piRlcForkSampling piCcsBinding)
  exact
    scale.le_trans
      (unionLaw.unionBound PiDecChildExtractionEvent
        (fun outcome =>
          PiRlcForkSamplingEvent outcome ∨
            (PiCcsSumCheckEvent outcome ∨ PiCcsMixingRootEvent outcome) ∨
              ParentBindingCollisionEvent outcome))
      (by
        simpa [nifsInteractiveTotal,
          InteractiveErrorBudget.strongWeakTotal] using
          scaleLaws.add_mono contract.piDecChildExtraction forkTail)

/-- The permitted explicit random-oracle contract, specialized to the exact
typed NIFS outcome and its concrete six predicates. -/
abbrev NifsExplicitRandomOracleContract
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {Weight : Type uWeight}
    [DecidableEq Extension]
    {shape : PiCCS.PaperJoint.Shape}
    {columns blockCount degreeBound : Nat}
    {key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound}
    {scale : ProbabilityScale Weight}
    (experiment : ProbabilityExperiment scale (AlignedForkOutcome key))
    (budget : FiatShamirErrorBudget Weight) :=
  ExplicitRandomOracleContract experiment (nifsEventPredicates key) budget

/-- Executable acceptance for one aligned experiment outcome. -/
def AcceptedOutcome
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    [DecidableEq Extension]
    {shape : PiCCS.PaperJoint.Shape}
    {columns blockCount degreeBound : Nat}
    {key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound}
    (outcome : AlignedForkOutcome key) : Prop :=
  verify key outcome.running outcome.fresh outcome.proof = some outcome.result

/-- Independently stated paper transition for one aligned outcome. -/
def TransitionOutcome
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    [DecidableEq Extension]
    {shape : PiCCS.PaperJoint.Shape}
    {columns blockCount degreeBound : Nat}
    {key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound}
    (outcome : AlignedForkOutcome key) : Prop :=
  Transition key outcome.running outcome.fresh outcome.result

/-- Complete non-interactive failure union: interactive residual events first,
then the six Fiat--Shamir events in transcript order. -/
def NonInteractiveFailure
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    [DecidableEq Extension]
    {shape : PiCCS.PaperJoint.Shape}
    {columns blockCount degreeBound : Nat}
    {key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound}
    (outcome : AlignedForkOutcome key) : Prop :=
  ResidualFailure outcome ∨ AnyFailure (nifsEventPredicates key) outcome

/-- Every accepted aligned outcome is a paper transition or inhabits the exact
eleven-event non-interactive failure union. -/
theorem acceptedOutcome_implies_transition_or_failure
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    [DecidableEq Extension]
    {shape : PiCCS.PaperJoint.Shape}
    {columns blockCount degreeBound : Nat}
    {key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound}
    (laws : PiRLC.PaperForkExtraction.ExtractionAlgebra
      key.nifsPiRlcContext.semantics key.nifsPiRlcContext.params
      key.nifsPiRlcContext.algebra)
    (strongSet : PiRLC.PaperForkExtraction.StrongSetUnits laws.ring
      key.nifsPiRlcContext.algebra.challengeValid)
    (outcome : AlignedForkOutcome key)
    (accepted : AcceptedOutcome outcome) :
    TransitionOutcome outcome ∨ NonInteractiveFailure outcome := by
  have acceptedExact :
      verify key outcome.running outcome.fresh outcome.proof =
        some outcome.result := by
    simpa [AcceptedOutcome] using accepted
  rcases verify_sound_or_residual_or_multiFork laws strongSet outcome
      acceptedExact with transition | residual | programming
  · exact Or.inl (by simpa [TransitionOutcome] using transition)
  · exact Or.inr (Or.inl
      ((residualBadEvent_iff_residualFailure outcome acceptedExact).1
        residual))
  · exact Or.inr (Or.inr
      ((anyFailure_iff_exists_event (nifsEventPredicates key) outcome).2
        ⟨.multiForkProgrammingFailure, programming⟩))

/-- The interactive and random-oracle contracts bound the exact complete
failure union by `nonInteractiveTotal`. -/
theorem nonInteractiveFailure_probability_le_total
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {Weight : Type uWeight}
    [DecidableEq Extension]
    {shape : PiCCS.PaperJoint.Shape}
    {columns blockCount degreeBound : Nat}
    {key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound}
    {scale : ProbabilityScale Weight}
    (scaleLaws : ScaleLaws scale)
    (experiment : ProbabilityExperiment scale (AlignedForkOutcome key))
    (unionLaw : UnionBound experiment)
    (extractionBudget : NifsExtractionErrorBudget Weight)
    (interactiveBudget : InteractiveErrorBudget Weight)
    (fiatShamirBudget : FiatShamirErrorBudget Weight)
    (interactiveContract :
      InteractiveResidualContract experiment extractionBudget
        interactiveBudget)
    (randomOracleContract :
      NifsExplicitRandomOracleContract experiment fiatShamirBudget) :
    scale.le (experiment.probability NonInteractiveFailure)
      (nonInteractiveTotal scale extractionBudget interactiveBudget
        fiatShamirBudget) := by
  have interactiveBound :=
    residualFailure_probability_le_total scaleLaws experiment unionLaw
      extractionBudget interactiveBudget interactiveContract
  have fiatShamirBound :=
    anyFailure_probability_le_total scaleLaws experiment unionLaw
      (nifsEventPredicates key) fiatShamirBudget randomOracleContract
  exact
    scale.le_trans
      (unionLaw.unionBound ResidualFailure
        (AnyFailure (nifsEventPredicates key)))
      (by
        simpa [NonInteractiveFailure, nonInteractiveTotal] using
          scaleLaws.add_mono interactiveBound fiatShamirBound)

/-- Headline conditional non-interactive NIFS soundness. The probability of
accepted execution, minus precisely the interactive and Fiat--Shamir budgets,
is at most the probability of the independently stated paper transition. -/
theorem accepted_probability_sub_total_le_transition
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {Weight : Type uWeight}
    [DecidableEq Extension]
    {shape : PiCCS.PaperJoint.Shape}
    {columns blockCount degreeBound : Nat}
    {key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound}
    {scale : ProbabilityScale Weight}
    (laws : PiRLC.PaperForkExtraction.ExtractionAlgebra
      key.nifsPiRlcContext.semantics key.nifsPiRlcContext.params
      key.nifsPiRlcContext.algebra)
    (strongSet : PiRLC.PaperForkExtraction.StrongSetUnits laws.ring
      key.nifsPiRlcContext.algebra.challengeValid)
    (scaleLaws : ScaleLaws scale)
    (experiment : ProbabilityExperiment scale (AlignedForkOutcome key))
    (unionLaw : UnionBound experiment)
    (extractionBudget : NifsExtractionErrorBudget Weight)
    (interactiveBudget : InteractiveErrorBudget Weight)
    (fiatShamirBudget : FiatShamirErrorBudget Weight)
    (interactiveContract :
      InteractiveResidualContract experiment extractionBudget
        interactiveBudget)
    (randomOracleContract :
      NifsExplicitRandomOracleContract experiment fiatShamirBudget) :
    scale.le
      (scale.subtract (experiment.probability AcceptedOutcome)
        (nonInteractiveTotal scale extractionBudget interactiveBudget
          fiatShamirBudget))
      (experiment.probability TransitionOutcome) := by
  apply loss_le_of_cover scaleLaws experiment unionLaw AcceptedOutcome
    TransitionOutcome NonInteractiveFailure
    (nonInteractiveTotal scale extractionBudget interactiveBudget
      fiatShamirBudget)
  · exact acceptedOutcome_implies_transition_or_failure laws strongSet
  · exact nonInteractiveFailure_probability_le_total scaleLaws experiment
      unionLaw extractionBudget interactiveBudget fiatShamirBudget
      interactiveContract randomOracleContract

end Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive
