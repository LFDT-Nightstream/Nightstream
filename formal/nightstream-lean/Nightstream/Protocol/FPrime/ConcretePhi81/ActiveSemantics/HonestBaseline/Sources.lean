import Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.PiCcsSources
import Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra.Norm.Centered
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Semantics.Paper

/-!
Production-shaped fixed-carrier model source fixture for the intended
fixed-active 270-coordinate profile.

Owns: one explicit `rowVariables = 1`, legacy-width 257, matrix-count 3,
fresh-count 1, running-count 14 source product; the sparse degree-two R1CS
polynomial `u0 * u1 - u2`; zero matrices, assignments, prior point, and
carried claims; and a kernel proof of the three independent paper obligations.

Does not own: a production-derived artifact, Rust decoding, verifier
acceptance, transcript execution, running-parent authority, active F-prime
context construction, commitments, R1CS lowering, costs, or row removal.

Emits constraints: no.

Authority boundary: every value in this file is model data defined from the
independent CCS/Split-NC semantic types. The numerical profile is the intended
fixed-active model profile. This file proves neither equality with the current
production relation nor production provenance; no production file, trace,
digest, acceptance bit, or artifact is treated as semantic evidence.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `fprime.active.honest_baseline.sources.profile` | the intended `1 / 257 / 3`, `1 + 14` source profile and five complete public rings give one 270-coordinate carrier | explicit model fixture | `profile_exact` |
| `fprime.active.honest_baseline.sources.polynomial` | sparse syntax is exactly `u0 * u1 - u2`, with monomial degrees two and one | explicit model fixture | `r1csPolynomial`, `monomial_degrees_exact` |
| `fprime.active.honest_baseline.sources.matrix` | all three legacy matrices are zero | explicit model fixture | `legacyStructure` |
| `fprime.active.honest_baseline.sources.assignment` | the fresh source and all fourteen running sources are complete zero assignments | explicit model fixture | `inputs` |
| `fprime.active.honest_baseline.sources.fresh_ccs` | the zero source satisfies `u0 * u1 - u2 = 0` at both Boolean rows | derived | `freshCcsHolds` |
| `fprime.active.honest_baseline.sources.norm` | every one of the fifteen complete source assignments has centered magnitude below two | derived | `allSourceNormsHold` |
| `fprime.active.honest_baseline.sources.carried` | all `14 * 3 * 54` zero claims equal their source-derived prior evaluations | derived | `carriedEvaluationsHold` |
| `fprime.active.honest_baseline.sources.paper` | the independent paper source statement is inhabited at the intended fixed-active model profile | derived | `paperHolds` |
-/

namespace Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.HonestBaseline.Sources

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270
open Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.PiCcsSources
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Semantics
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources

/-- Exact intended fixed-active model dimensions. The thirteen fixed public zeros
are inserted by `PiCcsSources.Inputs.data`, giving logical width 270. -/
def dimensions : Dimensions where
  rowVariables := 1
  legacyLogicalWidth := 257
  matrixCount := 3
  legacyPublicFits := by decide

/-- Exact batch-shaped independent semantic carrier for one fresh and
fourteen running sources. -/
def shape : SemanticShape := semanticShape dimensions 1 14

/-- The product monomial `u0 * u1`. -/
def productMonomial : CCSResidualTable.Monomial F 3 where
  coefficient := 1
  exponents := fun index => if index.val < 2 then 1 else 0

/-- The signed output monomial `-u2`. -/
def outputMonomial : CCSResidualTable.Monomial F 3 where
  coefficient := -1
  exponents := fun index => if index.val = 2 then 1 else 0

/-- The product term has total degree two. -/
theorem productMonomial_degree : productMonomial.totalDegree = 2 := by
  decide

/-- The signed output term has total degree one. -/
theorem outputMonomial_degree : outputMonomial.totalDegree = 1 := by
  decide

/-- Sparse degree-two R1CS polynomial `u0 * u1 - u2`. The declared bound is
three because the paper structure records the strict inequality
`term.totalDegree < degreeBound`. -/
def r1csPolynomial : CCSResidualTable.ConstraintPolynomial F 3 where
  degreeBound := 3
  terms := [productMonomial, outputMonomial]
  termsBelowDegree := by
    intro term member
    simp only [List.mem_cons, List.not_mem_nil, or_false] at member
    rcases member with rfl | rfl
    · rw [productMonomial_degree]
      decide
    · rw [outputMonomial_degree]
      decide

/-- The syntax-derived monomial degrees are exactly two and one. -/
theorem monomial_degrees_exact :
    productMonomial.totalDegree = 2 /\
      outputMonomial.totalDegree = 1 := by
  exact ⟨productMonomial_degree, outputMonomial_degree⟩

/-- Three zero legacy matrices paired with the explicit R1CS polynomial. -/
def legacyStructure : LegacyBatchStructure dimensions 1 14 where
  matrices := fun _ _ _ => 0
  constraintPolynomial := r1csPolynomial

/-- Canonical zero assignment on the legacy 257-coordinate source. -/
def zeroLegacyAssignment : LegacyAssignment dimensions := fun _ => 0

/-- Canonical zero assignment on the complete 270-coordinate source. -/
def zeroRunningAssignment : Assignment dimensions.shape := fun _ => 0

/-- The zero extension-field point in the single Boolean-row variable. -/
def priorPoint : Point dimensions.shape where
  coordinates := [K.zero]
  dimension := rfl

/-- One model-level source product at the intended fixed-active profile. Claims
are explicit zero values; they are proved correct below rather than assumed as
callbacks. -/
def inputs : Inputs dimensions 1 14 where
  legacyStructure := legacyStructure
  freshAssignments := fun _ => zeroLegacyAssignment
  runningAssignments := fun _ => zeroRunningAssignment
  priorPoint := priorPoint
  claimedCoefficient := fun _ => K.zero

/-- Sole source data consumed by all three independent paper obligations. -/
def data : Data shape := inputs.data

/-- The explicit R1CS polynomial evaluates to zero at the zero image vector. -/
private theorem r1csPolynomial_at_zero :
    CCSResidualTable.evaluatePolynomial ConcreteCarrier.baseOps r1csPolynomial
        (fun _ => 0) = 0 := by
  rfl

/-- The legacy fresh zero assignment satisfies the explicit sparse relation. -/
private theorem legacyFreshConstraintSatisfied
    (source : Fin 1) :
    CCSResidualTable.ConstraintSatisfied ConcreteCarrier.baseOps
      legacyStructure (inputs.freshAssignments source) := by
  intro vertex
  change CCSResidualTable.residualAt ConcreteCarrier.baseOps legacyStructure
      zeroLegacyAssignment vertex = 0
  unfold CCSResidualTable.residualAt
  have imagesZero :
      CCSResidualTable.matrixImagesAt ConcreteCarrier.baseOps legacyStructure
          zeroLegacyAssignment vertex = fun _ => 0 := by
    funext matrix
    exact PaperLinearAlgebra.matrixVectorAt_zero ConcreteCarrier.baseOps
      ConcreteCarrier.baseLaws (legacyStructure.matrices matrix) vertex
  rw [imagesZero]
  exact r1csPolynomial_at_zero

/-- The aligned-and-completed fresh assignment is pointwise zero. -/
private theorem completedFreshAssignment_zero
    (fresh : Fin 1) :
    inputs.data.freshAssignment fresh = fun _ => 0 := by
  rw [Inputs.data_freshAssignment_eq]
  funext column
  unfold FPrimeCarrier270.assignment Phi81CarrierLayout.extendAssignment
  split
  · simp [FPrimeCarrier270.alignedLogicalAssignment, inputs,
      zeroLegacyAssignment]
  · rfl

/-- Every authoritative source in the intended fixed-active model fixture is
the complete zero assignment. -/
theorem data_assignment_zero
    (source : Fin shape.sourceCount) :
    data.assignment source = fun _ => 0 := by
  change inputs.data.assignment source = fun _ => 0
  funext column
  rcases Data.source_eq_fresh_or_running source with
    ⟨fresh, rfl⟩ | ⟨running, rfl⟩
  · calc
      inputs.data.assignment (Data.freshIndex fresh) column =
          inputs.data.freshAssignment fresh column :=
        congrFun (Data.assignment_freshIndex inputs.data fresh) column
      _ = 0 := congrFun (completedFreshAssignment_zero fresh) column
  · calc
      inputs.data.assignment (Data.runningIndex running) column =
          inputs.data.runningAssignments running column :=
        congrFun (Data.assignment_runningIndex inputs.data running) column
      _ = 0 := rfl

/-- The fixture has the intended fixed-active model dimensions and a complete
five-ring 270-coordinate carrier. This is a model shape fact, not a theorem
about current production shape or provenance. -/
theorem profile_exact :
    dimensions.rowVariables = 1 /\
      dimensions.legacyLogicalWidth = 257 /\
      dimensions.matrixCount = 3 /\
      shape.freshCount = 1 /\
      shape.runningCount = 14 /\
      dimensions.shape.publicRingColumns = 5 /\
      shape.logicalWidth = 270 /\
      shape.carrierWidth = 270 := by
  decide

/-- The unique fresh zero source satisfies the explicit degree-two R1CS
polynomial at every Boolean row. -/
theorem freshCcsHolds : Paper.FreshCcsHolds data := by
  have truth : Semantics.Fe.FreshTruth inputs.data :=
    (Inputs.freshTruth_iff_legacy inputs).2 legacyFreshConstraintSatisfied
  simpa only [data, shape] using truth

/-- Every fresh and running coordinate is zero, hence satisfies the strict
paper norm bound. -/
theorem allSourceNormsHold : Paper.AllSourceNormsHold data := by
  intro source column
  rw [congrFun (data_assignment_zero source) column,
    Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra.Norm.Centered.centeredMagnitude_zero]
  decide

/-- Every carried claim is zero and the independently computed evaluation of
the zero matrices at the zero running assignments is zero. -/
theorem carriedEvaluationsHold : Paper.CarriedEvaluationsHold data := by
  intro coordinate
  unfold CarriedEvaluationResidual.EvaluationClaimHolds
  have assignmentZero :
      data.carriedData.assignments coordinate.running =
        fun _ => ConcreteCarrier.baseOps.zero := by
    calc
      data.carriedData.assignments coordinate.running =
          data.assignment (Data.runningIndex coordinate.running) :=
        data.carriedData_assignment_eq coordinate.running
      _ = fun _ => ConcreteCarrier.baseOps.zero := by
        simpa only [ConcreteCarrier.baseOps] using
          data_assignment_zero (Data.runningIndex coordinate.running)
  have computedZero :
      CarriedEvaluationResidual.computedCoefficient
          ConcreteCarrier.baseOps ConcreteCarrier.extensionOps K.embed
          data.carriedData coordinate = K.zero := by
    simpa only [ConcreteCarrier.extensionOps] using
      CarriedEvaluationResidual.computedCoefficient_eq_zero_of_assignment_zero
        ConcreteCarrier.baseOps ConcreteCarrier.baseLaws
        ConcreteCarrier.extensionOps ConcreteCarrier.extensionLaws K.embed
        ConcreteCarrier.embed_zero data.carriedData coordinate assignmentZero
  calc
    data.carriedData.claimedCoefficient coordinate = K.zero := rfl
    _ = CarriedEvaluationResidual.computedCoefficient
          ConcreteCarrier.baseOps ConcreteCarrier.extensionOps K.embed
          data.carriedData coordinate := computedZero.symm

/-- The independent paper source semantics is non-vacuously inhabited at the
intended fixed-active 270-coordinate model profile. -/
theorem paperHolds : Paper.Holds data :=
  ⟨freshCcsHolds, allSourceNormsHold, carriedEvaluationsHold⟩

end Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.HonestBaseline.Sources
