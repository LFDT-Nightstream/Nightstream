import Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.SelectedNifs.SemanticMinimality.Baseline

/-!
Canonical public context for independent `Pi_CCS` source-data counterexamples.

Assurance tier: model-level.

Owns: construction of a complete fixed-active context from one independent
source `Data` value when its running assignments are zero and its carried
claims are true; exact source-product binding; and checked incoming-parent
authority for the resulting canonical zero opening.

Does not own: any paper obligation about fresh CCS or source norms, malformed
fixtures, executable transcripts, Rust/R1CS refinement, costs, or row removal.

Emits constraints: no.

Authority boundary: the helper computes statements from source data. The two
semantic premises are explicit theorem inputs and cannot be replaced by an
accepted certificate, digest, or implementation result.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `fprime.active.nifs.pi_ccs.counterexample.context.system` | derive the sole Phi81 structure from source data | computed | `system` |
| `fprime.active.nifs.pi_ccs.counterexample.context.product` | bind fresh and running statements to the same data | checked/derived | `sourceBound` |
| `fprime.active.nifs.pi_ccs.counterexample.context.parent` | install a valid combined zero parent and canonical children | computed/derived | `parent`, `children`, `runningAccepted` |
| `fprime.active.nifs.pi_ccs.counterexample.context.input` | bind both public input surfaces | derived | `semanticInput` |
-/

namespace Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.SelectedNifs.SemanticMinimality.PiCcs.CanonicalContext

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.Nifs
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.HonestBaseline

/-- Sole relation structure derived from the supplied independent data. -/
def system (data : Sources.Data Sources.shape) :
    RelationStructure Sources.shape Context.publicRingColumns
      Context.publicFits :=
  Phi81Relation.Structure.ofSourceData Context.publicRingColumns
    Context.publicFits data

/-- Canonical index of the sole fixed-active fresh source. -/
def firstFresh : Fin FixedActive.arity.freshCount :=
  ⟨0, FixedActive.arity.freshPositive⟩

/-- Canonical fresh public statement for the sole fixed-active fresh source. -/
def freshStatement (data : Sources.Data Sources.shape) :
    Phi81Relation.CCSStatement
      (RelationShape Sources.shape Context.publicRingColumns Context.publicFits)
      (CommitmentValue Context.verifierRows) :=
  Phi81Relation.canonicalCCSStatement (ConcretePhi81.commit Context.key)
    (system data) .fresh (data.freshAssignment firstFresh)

/-- Canonical combined parent at the source-owned prior point and zero
opening. -/
def parent (data : Sources.Data Sources.shape) :
    Phi81Relation.CEStatement
      (RelationShape Sources.shape Context.publicRingColumns Context.publicFits)
      (CommitmentValue Context.verifierRows) :=
  Phi81Relation.canonicalCEStatement (ConcretePhi81.commit Context.key)
    (system data) .combined data.priorPoint Context.zeroAssignment

/-- Canonical radix children of the combined zero parent. -/
def children (data : Sources.Data Sources.shape) :
    Fin productionGlobalParams.k ->
      Phi81Relation.CEStatement
        (RelationShape Sources.shape Context.publicRingColumns
          Context.publicFits)
        (CommitmentValue Context.verifierRows) :=
  PiDEC.childrenOf (ConcretePhi81.decAlgebra Context.key) (parent data)
    Context.zeroAssignment

/-- Complete public source product in the fixed-active partition. -/
def input (data : Sources.Data Sources.shape) :
    SourceProduct Sources.shape Context.publicRingColumns Context.publicFits
      (CommitmentValue Context.verifierRows) productionGlobalParams
      FixedActive.arity where
  fresh := fun _ => freshStatement data
  running := children data

/-- Complete verifier context computed from the supplied source data. -/
def context (data : Sources.Data Sources.shape) :
    FixedActive.Context Sources.shape Unit Context.publicRingColumns
      Context.publicFits Context.verifierRows :=
  { Context.context with
    input := input data
    runningParent := some (parent data)
    piCcsInput := PublicInput.ofSources data }

@[simp] theorem context_input_fresh
    (data : Sources.Data Sources.shape)
    (source : Fin FixedActive.arity.freshCount) :
    (context data).input.fresh source = freshStatement data := by
  rfl

@[simp] theorem context_input_running
    (data : Sources.Data Sources.shape)
    (source : Fin (FixedActive.arity.mode.count productionGlobalParams)) :
    (context data).input.running source = children data source := by
  rfl

/-- The canonical combined parent has its explicit zero opening. -/
theorem parentHolds (data : Sources.Data Sources.shape) :
    CE.Holds (ConcretePhi81.semantics Context.key) productionGlobalParams
      (parent data) Context.zeroAssignment := by
  apply Phi81Relation.canonicalCE_holds
  intro column
  simp [Context.zeroAssignment, productionGlobalParams, NormStage.bound,
    GlobalParams.bigB,
    Phi81Relation.PiRLCAlgebra.Norm.Centered.centeredMagnitude_zero]

/-- The installed parent and children satisfy strict active incoming
authority. -/
theorem runningAccepted (data : Sources.Data Sources.shape) :
    ConcretePhi81.RunningAuthority.Accepted (context data) := by
  apply
    Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.HonestBaseline.RunningAuthority.accepted_of_combinedOpening
      (context data) Context.zeroAssignment (parent data)
  · rfl
  · exact {
      parentCombined := rfl
      parentValid := parentHolds data
      childrenEq := rfl
    }

/-- Bind the complete public product to the same source data. Running truth is
used only to identify prior claim arrays with canonical zero-opening
evaluations. -/
theorem sourceBound
    (data : Sources.Data Sources.shape)
    (runningZero : ∀ source, data.runningAssignments source =
      Context.zeroAssignment)
    (carried : Semantics.Paper.CarriedEvaluationsHold data) :
    InputAuthority.BoundToSources Context.publicRingColumns Context.publicFits
      (ConcretePhi81.commit Context.key) data Context.alignment
      (context data).input := by
  refine { fresh := ?_, running := ?_ }
  · intro source
    have source_eq : source = firstFresh := by
      apply Fin.ext
      have source_lt : source.val < 1 := by
        simpa only [FixedActive.arity_freshCount] using source.isLt
      change source.val = 0
      omega
    subst source
    refine {
      constraintSystem := ?_
      commitment := ?_
      publicInput := ?_
      stage := ?_
    }
    · rw [context_input_fresh]
      rfl
    · rw [context_input_fresh]
      rfl
    · rw [context_input_fresh]
      rfl
    · rw [context_input_fresh]
      rfl
  · intro source
    refine {
      constraintSystem := ?_
      commitment := ?_
      publicInput := ?_
      point := ?_
      evaluations := ?_
      stage := ?_
    }
    · rw [context_input_running]
      rfl
    · rw [context_input_running, runningZero]
      change
        ConcretePhi81.commit Context.key Context.zeroAssignment =
          (children data source).commitment
      unfold children PiDEC.childrenOf
      rw [Context.splitZero]
      rfl
    · rw [context_input_running, runningZero]
      change
        Phi81Relation.projectPublicInput Context.zeroAssignment =
          (children data source).publicInput
      unfold children PiDEC.childrenOf
      rw [Context.splitZero]
      rfl
    · rw [context_input_running]
      rfl
    · rw [context_input_running]
      unfold children PiDEC.childrenOf
      rw [Context.splitZero]
      have carriedEvaluation :=
        InputAuthority.relationEvaluations_eq_priorEvaluations_of_carriedTruth
          Context.publicRingColumns Context.publicFits data source carried
      rw [runningZero] at carriedEvaluation
      simpa only [parent, Phi81Relation.canonicalCEStatement] using
        carriedEvaluation
    · rw [context_input_running]
      rfl

/-- Both public input surfaces bind to the supplied independent data. -/
theorem semanticInput
    (data : Sources.Data Sources.shape)
    (runningZero : ∀ source, data.runningAssignments source =
      Context.zeroAssignment)
    (carried : Semantics.Paper.CarriedEvaluationsHold data) :
    ConcretePhi81.SemanticInput (context data) data where
  publicInput := rfl
  sources := sourceBound data runningZero carried

end Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.SelectedNifs.SemanticMinimality.PiCcs.CanonicalContext
