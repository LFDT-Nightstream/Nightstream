import Nightstream.SuperNeo.Concrete.Phi81Relation.Semantics
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Semantics
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Fe.SourceRefinement
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.ProductAlignment

/-!
Authority bridge from the public `Pi_CCS` input product to the independent
Split-NC source family.

Protocol: SuperNeo `Pi_CCS`.
Phase: input opening and prior-running-claim authority before transcript
replay.
Constraint family: source structure, commitment, public input, stage, prior
point, and prior evaluation arrays; this file emits no rows.

Owns: the exact prior-evaluation array encoded by `Sources.Data`; separate
fresh and running source-binding records; derivation of actual CCS/CE
membership from those public bindings and the independent Section 7.3
statement; and one unified source-assignment vector in product order.

Does not own: transcript acceptance, new output claims, commitment binding as
a cryptographic assumption, PiRLC, PiDEC, Rust, R1CS, rows, costs, or row
removal.

Emits constraints: no.

Authority boundary: `Sources.Data` alone cannot authorize a public NIFS input.
Each public statement must independently copy the source-derived relation,
commitment opening, public projection, and fresh stage. Running statements
must additionally expose exactly the semantic prior point and complete prior
evaluation array. A transcript digest is not accepted as a substitute for any
field.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_ccs.input.fresh.structure` | fresh CCS uses the sole source-derived relation | checked | `FreshSourceBound.structure` |
| `nifs.pi_ccs.input.fresh.opening` | commitment/public input open the aligned fresh assignment | checked | `FreshSourceBound.commitment`, `publicInput` |
| `nifs.pi_ccs.input.fresh.stage` | source is at the strict fresh bound | checked | `FreshSourceBound.stage` |
| `nifs.pi_ccs.input.running.structure` | running CE uses the same relation | checked | `RunningSourceBound.structure` |
| `nifs.pi_ccs.input.running.opening` | commitment/public input open the full running assignment | checked | `RunningSourceBound.commitment`, `publicInput` |
| `nifs.pi_ccs.input.running.prior` | point and all matrix/lane claims equal the semantic prior carrier | checked | `RunningSourceBound.point`, `evaluations` |
| `nifs.pi_ccs.input.running.stage` | running input is also at the strict fresh bound | checked | `RunningSourceBound.stage` |
| `nifs.pi_ccs.input.membership` | bindings plus paper truth imply every public source really opens | derived | `allSourcesHold` |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.InputAuthority

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Fe.SourceRefinement

universe uCommitment

/-- Complete matrix-indexed prior evaluation claims for one running source,
in the same canonical array order used by `CE.Instance`. -/
def priorEvaluations
    {shape : SemanticShape}
    (data : Data shape)
    (running : Fin shape.runningCount) : Array Phi81Relation.Evaluation :=
  Array.ofFn fun (matrix : Fin shape.matrixCount)
      (lane : Fin ringDegree) =>
    (PublicInput.ofSources data).claimedYRing running matrix lane

@[simp] theorem priorEvaluations_size
    {shape : SemanticShape}
    (data : Data shape)
    (running : Fin shape.runningCount) :
    (priorEvaluations data running).size = shape.matrixCount := by
  simp [priorEvaluations, SemanticShape.paperShape]

@[simp] theorem priorEvaluations_get
    {shape : SemanticShape}
    (data : Data shape)
    (running : Fin shape.runningCount)
    (matrix : Fin shape.matrixCount) :
    (priorEvaluations data running)[matrix.val]'(by
      simpa only [priorEvaluations, Array.size_ofFn] using matrix.isLt) =
      fun lane =>
        (PublicInput.ofSources data).claimedYRing running matrix lane := by
  simp [priorEvaluations]

/-- Independent carried truth makes the public prior claim array exactly the
concrete relation evaluation array for the same running assignment and prior
point. -/
theorem relationEvaluations_eq_priorEvaluations_of_carriedTruth
    {shape : SemanticShape}
    (publicRingColumns : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (data : Data shape)
    (running : Fin shape.runningCount)
    (truth : SplitNc.Semantics.Paper.CarriedEvaluationsHold data) :
    Phi81Relation.evaluations
        (Phi81Relation.Structure.ofSourceData
          publicRingColumns publicFits data)
        (data.runningAssignments running) data.priorPoint =
      priorEvaluations data running := by
  apply Array.ext
  · simp [Phi81Relation.evaluations, priorEvaluations,
      Phi81Relation.Shape.ofSemantic, SemanticShape.paperShape]
  · intro index canonicalLt claimedLt
    have indexLt : index < shape.matrixCount := by
      simpa only [Phi81Relation.evaluations, Array.size_ofFn] using canonicalLt
    let matrix : Fin shape.matrixCount := ⟨index, indexLt⟩
    have claimsEq :=
      claimedYRing_eq_sourceYRingAt_of_carriedTruth data truth
    funext lane
    calc
      ((Phi81Relation.evaluations
          (Phi81Relation.Structure.ofSourceData
            publicRingColumns publicFits data)
          (data.runningAssignments running)
          data.priorPoint)[index]'canonicalLt) lane =
          Verifier.Polynomial.Fe.sourceYRingAt
            data data.priorPoint (Data.runningIndex running) matrix lane := by
        calc
          ((Phi81Relation.evaluations
              (Phi81Relation.Structure.ofSourceData
                publicRingColumns publicFits data)
              (data.runningAssignments running)
              data.priorPoint)[index]'canonicalLt) lane =
              Phi81Relation.matrixEvaluation
                (Phi81Relation.Structure.ofSourceData
                  publicRingColumns publicFits data)
                (data.runningAssignments running)
                data.priorPoint matrix lane := by
            simpa [matrix] using congrFun
              (Phi81Relation.evaluations_get
                (Phi81Relation.Structure.ofSourceData
                  publicRingColumns publicFits data)
                (data.runningAssignments running)
                data.priorPoint matrix) lane
          _ = Verifier.Polynomial.Fe.sourceYRingAt
                data data.priorPoint (Data.runningIndex running)
                matrix lane := by
            unfold Verifier.Polynomial.Fe.sourceYRingAt
            rw [data.assignment_runningIndex]
            rfl
      _ = (PublicInput.ofSources data).claimedYRing running matrix lane := by
        symm
        exact congrFun (congrFun (congrFun claimsEq running) matrix) lane
      _ = ((priorEvaluations data running)[index]'claimedLt) lane := by
        symm
        exact congrFun (priorEvaluations_get data running matrix) lane

/-- Public authority for one fresh CCS source. -/
structure FreshSourceBound
    {shape : SemanticShape}
    {params : GlobalParams}
    {arity : BatchArity params}
    {Commitment : Type uCommitment}
    (publicRingColumns : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (commit : SourceAssignment shape -> Commitment)
    (data : Data shape)
    (alignment : SourceAlignment shape params arity)
    (input :
      SourceProduct shape publicRingColumns publicFits Commitment params arity)
    (source : Fin arity.freshCount) : Prop where
  constraintSystem :
    (input.fresh source).constraintSystem =
      Phi81Relation.Structure.ofSourceData publicRingColumns publicFits data
  commitment :
    commit (data.freshAssignment (alignment.semanticFreshIndex source)) =
      (input.fresh source).commitment
  publicInput :
    sourcePublicInput publicRingColumns publicFits
        (data.freshAssignment (alignment.semanticFreshIndex source)) =
      (input.fresh source).publicInput
  stage : (input.fresh source).stage = .fresh

/-- Public authority for one running CE source. -/
structure RunningSourceBound
    {shape : SemanticShape}
    {params : GlobalParams}
    {arity : BatchArity params}
    {Commitment : Type uCommitment}
    (publicRingColumns : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (commit : SourceAssignment shape -> Commitment)
    (data : Data shape)
    (alignment : SourceAlignment shape params arity)
    (input :
      SourceProduct shape publicRingColumns publicFits Commitment params arity)
    (source : Fin (arity.mode.count params)) : Prop where
  constraintSystem :
    (input.running source).constraintSystem =
      Phi81Relation.Structure.ofSourceData publicRingColumns publicFits data
  commitment :
    commit
        (data.runningAssignments
          (alignment.semanticRunningIndex source)) =
      (input.running source).commitment
  publicInput :
    sourcePublicInput publicRingColumns publicFits
        (data.runningAssignments
          (alignment.semanticRunningIndex source)) =
      (input.running source).publicInput
  point : (input.running source).point = data.priorPoint
  evaluations :
    (input.running source).evaluations =
      priorEvaluations data (alignment.semanticRunningIndex source)
  stage : (input.running source).stage = .fresh

/-- Complete public input-product authority, preserving the fresh/running
partition rather than merely the total source count. -/
structure BoundToSources
    {shape : SemanticShape}
    {params : GlobalParams}
    {arity : BatchArity params}
    {Commitment : Type uCommitment}
    (publicRingColumns : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (commit : SourceAssignment shape -> Commitment)
    (data : Data shape)
    (alignment : SourceAlignment shape params arity)
    (input :
      SourceProduct shape publicRingColumns publicFits Commitment params arity) :
    Prop where
  fresh : ∀ source,
    FreshSourceBound publicRingColumns publicFits commit data alignment input
      source
  running : ∀ source,
    RunningSourceBound publicRingColumns publicFits commit data alignment input
      source

namespace BoundToSources

/-- Every public input source is visibly at the fresh norm stage. -/
theorem sourceFresh
    {shape : SemanticShape}
    {params : GlobalParams}
    {arity : BatchArity params}
    {Commitment : Type uCommitment}
    (publicRingColumns : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (commit : SourceAssignment shape -> Commitment)
    (data : Data shape)
    (alignment : SourceAlignment shape params arity)
    (input :
      SourceProduct shape publicRingColumns publicFits Commitment params arity)
    (bound :
      BoundToSources publicRingColumns publicFits commit data alignment input) :
    ∀ source, (input.source source).stage = .fresh := by
  intro source
  refine Fin.addCases ?_ ?_ source
  · intro fresh
    simpa [PiCCS.InputProduct.source, PiCCS.Source.stage] using
      (bound.fresh fresh).stage
  · intro running
    simpa [PiCCS.InputProduct.source, PiCCS.Source.stage] using
      (bound.running running).stage

/-- Every public input source uses the sole source-derived relation
structure. -/
theorem sourceStructure
    {shape : SemanticShape}
    {params : GlobalParams}
    {arity : BatchArity params}
    {Commitment : Type uCommitment}
    (publicRingColumns : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (commit : SourceAssignment shape -> Commitment)
    (data : Data shape)
    (alignment : SourceAlignment shape params arity)
    (input :
      SourceProduct shape publicRingColumns publicFits Commitment params arity)
    (bound :
      BoundToSources publicRingColumns publicFits commit data alignment input) :
    ∀ source,
      (input.source source).constraintSystem =
        Phi81Relation.Structure.ofSourceData
          publicRingColumns publicFits data := by
  intro source
  refine Fin.addCases ?_ ?_ source
  · intro fresh
    simpa [PiCCS.InputProduct.source, PiCCS.Source.constraintSystem] using
      (bound.fresh fresh).constraintSystem
  · intro running
    simpa [PiCCS.InputProduct.source, PiCCS.Source.constraintSystem] using
      (bound.running running).constraintSystem

/-- Every source commitment is the commitment of the authoritative assignment
at the same partition-preserving index. -/
theorem sourceCommitment
    {shape : SemanticShape}
    {params : GlobalParams}
    {arity : BatchArity params}
    {Commitment : Type uCommitment}
    (publicRingColumns : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (commit : SourceAssignment shape -> Commitment)
    (data : Data shape)
    (alignment : SourceAlignment shape params arity)
    (input :
      SourceProduct shape publicRingColumns publicFits Commitment params arity)
    (bound :
      BoundToSources publicRingColumns publicFits commit data alignment input) :
    ∀ source,
      commit (data.assignment (alignment.semanticIndex source)) =
        (input.source source).commitment := by
  intro source
  refine Fin.addCases ?_ ?_ source
  · intro fresh
    rw [alignment.semanticIndex_fresh, data.assignment_freshIndex]
    simpa [PiCCS.InputProduct.source, PiCCS.Source.commitment] using
      (bound.fresh fresh).commitment
  · intro running
    rw [alignment.semanticIndex_running, data.assignment_runningIndex]
    simpa [PiCCS.InputProduct.source, PiCCS.Source.commitment] using
      (bound.running running).commitment

/-- Every source public input is the exact projection of the authoritative
assignment at the same partition-preserving index. -/
theorem sourcePublicInput
    {shape : SemanticShape}
    {params : GlobalParams}
    {arity : BatchArity params}
    {Commitment : Type uCommitment}
    (publicRingColumns : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (commit : SourceAssignment shape -> Commitment)
    (data : Data shape)
    (alignment : SourceAlignment shape params arity)
    (input :
      SourceProduct shape publicRingColumns publicFits Commitment params arity)
    (bound :
      BoundToSources publicRingColumns publicFits commit data alignment input) :
    ∀ source,
      Verifier.sourcePublicInput publicRingColumns publicFits
          (data.assignment (alignment.semanticIndex source)) =
        (input.source source).publicInput := by
  intro source
  refine Fin.addCases ?_ ?_ source
  · intro fresh
    rw [alignment.semanticIndex_fresh, data.assignment_freshIndex]
    simpa [PiCCS.InputProduct.source, PiCCS.Source.publicInput] using
      (bound.fresh fresh).publicInput
  · intro running
    rw [alignment.semanticIndex_running, data.assignment_runningIndex]
    simpa [PiCCS.InputProduct.source, PiCCS.Source.publicInput] using
      (bound.running running).publicInput

end BoundToSources

/-- The authoritative assignment vector in exact public-product order. -/
def productAssignments
    {shape : SemanticShape}
    {params : GlobalParams}
    {arity : BatchArity params}
    (data : Data shape)
    (alignment : SourceAlignment shape params arity) :
    Fin arity.total -> SourceAssignment shape :=
  fun source => data.assignment (alignment.semanticIndex source)

@[simp] theorem productAssignments_fresh
    {shape : SemanticShape}
    {params : GlobalParams}
    {arity : BatchArity params}
    (data : Data shape)
    (alignment : SourceAlignment shape params arity)
    (source : Fin arity.freshCount) :
    productAssignments data alignment
        (Fin.castAdd (arity.mode.count params) source) =
      data.freshAssignment (alignment.semanticFreshIndex source) := by
  unfold productAssignments
  rw [alignment.semanticIndex_fresh, data.assignment_freshIndex]

@[simp] theorem productAssignments_running
    {shape : SemanticShape}
    {params : GlobalParams}
    {arity : BatchArity params}
    (data : Data shape)
    (alignment : SourceAlignment shape params arity)
    (source : Fin (arity.mode.count params)) :
    productAssignments data alignment
        (Fin.natAdd arity.freshCount source) =
      data.runningAssignments (alignment.semanticRunningIndex source) := by
  unfold productAssignments
  rw [alignment.semanticIndex_running, data.assignment_runningIndex]

/-- Paper norm truth is exactly the fresh-stage norm opening needed by every
source in public-product order. -/
theorem productAssignments_normFresh
    {shape : SemanticShape}
    {params : GlobalParams}
    {arity : BatchArity params}
    (data : Data shape)
    (alignment : SourceAlignment shape params arity)
    (freshBound_eq_two : NormStage.bound params .fresh = 2)
    (paper : SplitNc.Semantics.Paper.Holds data) :
    ∀ source column,
      centeredMagnitude
          (productAssignments data alignment source column) <
        NormStage.bound params .fresh := by
  intro source column
  rw [freshBound_eq_two]
  exact paper.2.1 (alignment.semanticIndex source) column

/-- One bound fresh public source is genuine CCS membership whenever the
independent paper statement holds. -/
theorem freshSource_holds
    {shape : SemanticShape}
    {params : GlobalParams}
    {arity : BatchArity params}
    {Commitment : Type uCommitment}
    (publicRingColumns : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (commit : SourceAssignment shape -> Commitment)
    (data : Data shape)
    (alignment : SourceAlignment shape params arity)
    (input :
      SourceProduct shape publicRingColumns publicFits Commitment params arity)
    (freshBound_eq_two : NormStage.bound params .fresh = 2)
    (paper : SplitNc.Semantics.Paper.Holds data)
    (source : Fin arity.freshCount)
    (bound :
      FreshSourceBound publicRingColumns publicFits commit data alignment input
        source) :
    CCS.Holds
      (productSemantics publicRingColumns publicFits commit) params
      (input.fresh source)
      (data.freshAssignment (alignment.semanticFreshIndex source)) := by
  refine ⟨⟨bound.commitment, ?_, ?_⟩, ?_⟩
  · simpa [Phi81Relation.publicInputMatches, sourcePublicInput] using
      bound.publicInput
  · change ∀ column,
      centeredMagnitude
          (data.freshAssignment
            (alignment.semanticFreshIndex source) column) <
        NormStage.bound params (input.fresh source).stage
    rw [bound.stage, freshBound_eq_two]
    intro column
    simpa only [data.assignment_freshIndex] using
      paper.2.1
        (Data.freshIndex (alignment.semanticFreshIndex source)) column
  · rw [bound.constraintSystem]
    exact paper.1 (alignment.semanticFreshIndex source)

/-- One bound running public source is genuine CE membership whenever the
independent paper statement holds. -/
theorem runningSource_holds
    {shape : SemanticShape}
    {params : GlobalParams}
    {arity : BatchArity params}
    {Commitment : Type uCommitment}
    (publicRingColumns : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (commit : SourceAssignment shape -> Commitment)
    (data : Data shape)
    (alignment : SourceAlignment shape params arity)
    (input :
      SourceProduct shape publicRingColumns publicFits Commitment params arity)
    (freshBound_eq_two : NormStage.bound params .fresh = 2)
    (paper : SplitNc.Semantics.Paper.Holds data)
    (source : Fin (arity.mode.count params))
    (bound :
      RunningSourceBound publicRingColumns publicFits commit data alignment
        input source) :
    CE.Holds
      (productSemantics publicRingColumns publicFits commit) params
      (input.running source)
      (data.runningAssignments (alignment.semanticRunningIndex source)) := by
  refine ⟨⟨bound.commitment, ?_, ?_⟩, ?_, ?_⟩
  · simpa [Phi81Relation.publicInputMatches, sourcePublicInput] using
      bound.publicInput
  · change ∀ column,
      centeredMagnitude
          (data.runningAssignments
            (alignment.semanticRunningIndex source) column) <
        NormStage.bound params (input.running source).stage
    rw [bound.stage, freshBound_eq_two]
    intro column
    simpa only [data.assignment_runningIndex] using
      paper.2.1
        (Data.runningIndex (alignment.semanticRunningIndex source)) column
  · exact (input.running source).point.dimension
  · calc
      Phi81Relation.evaluations
          (input.running source).constraintSystem
          (data.runningAssignments (alignment.semanticRunningIndex source))
          (input.running source).point =
          Phi81Relation.evaluations
            (Phi81Relation.Structure.ofSourceData
              publicRingColumns publicFits data)
            (data.runningAssignments
              (alignment.semanticRunningIndex source))
            data.priorPoint := by
        rw [bound.constraintSystem, bound.point]
      _ = priorEvaluations data (alignment.semanticRunningIndex source) :=
        relationEvaluations_eq_priorEvaluations_of_carriedTruth
          publicRingColumns publicFits data
          (alignment.semanticRunningIndex source) paper.2.2
      _ = (input.running source).evaluations := bound.evaluations.symm

/-- Every public source in product order genuinely opens the independent
semantic assignment at the strict fresh relation bound. -/
theorem allSourcesHold
    {shape : SemanticShape}
    {params : GlobalParams}
    {arity : BatchArity params}
    {Commitment : Type uCommitment}
    (publicRingColumns : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (commit : SourceAssignment shape -> Commitment)
    (data : Data shape)
    (alignment : SourceAlignment shape params arity)
    (input :
      SourceProduct shape publicRingColumns publicFits Commitment params arity)
    (freshBound_eq_two : NormStage.bound params .fresh = 2)
    (paper : SplitNc.Semantics.Paper.Holds data)
    (bound :
      BoundToSources publicRingColumns publicFits commit data alignment input) :
    ∀ source,
      (input.source source).Holds
        (productSemantics publicRingColumns publicFits commit) params
        (productAssignments data alignment source) := by
  intro source
  refine Fin.addCases ?_ ?_ source
  · intro fresh
    simpa [PiCCS.InputProduct.source, PiCCS.Source.Holds] using
      freshSource_holds publicRingColumns publicFits commit data alignment
        input freshBound_eq_two paper fresh (bound.fresh fresh)
  · intro running
    simpa [PiCCS.InputProduct.source, PiCCS.Source.Holds] using
      runningSource_holds publicRingColumns publicFits commit data alignment
        input freshBound_eq_two paper running (bound.running running)

end Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.InputAuthority
