import Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.SelectedNifs.SemanticMinimality.PiCcs
import Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.SelectedNifs.SemanticMinimality.PiCcs.CarriedEvaluations
import Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.SelectedNifs.SemanticMinimality.PiCcs.PaperRelation
import Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.SelectedNifs.PaperSemanticMinimality.SourceBinding
import Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra.Challenge
import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.PaperProfile.ObligationPlan

/-!
Closed removal witnesses for the corrected six-leaf fixed-active paper plan.

Assurance tier: model-level.

Owns: one accepted candidate for the exact operational SuperNeo paper
profile; a canonical `Pi_DEC` acceptance lemma that needs source binding but
does not assume source relation truth, source norm truth, carried-evaluation
truth, or challenge validity; one counterexample for each of the six retained
leaves; and inclusion-minimal soundness of that fixed plan.

Does not own: SumCheck or Fiat--Shamir security, transcript replay,
child-opening extraction, Rust/R1CS refinement, physical rows, costs, global
gate minimality, or row removal.

Emits constraints: no.

Authority boundary: every counterexample fixes the profile, public source,
independent source data, row point, complete challenge vector, and public
target outside the target proposition. The first five counterexamples retain
the exact operational `Pi_DEC` verifier, including verifier-computed child
public inputs and exact evaluation arity.

| Stable stage path | Mathematical obligation | Authority class |
|---|---|---|
| `f_prime.nifs.paper.selected.minimality` | each of the six fixed-active paper obligations has a kernel-checked removal witness and the retained plan is inclusion-minimal | derived/necessity |
-/

namespace Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.SelectedNifs.PaperSemanticMinimality

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
open Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.SelectedNifs.SemanticMinimality

abbrev PaperLeaf := FixedActive.PaperProfile.ObligationPlan.Leaf

abbrev PaperCandidate :=
  FixedActive.PaperProfile.ObligationPlan.Candidate Sources.shape
    Context.publicRingColumns Context.publicFits Context.verifierRows

abbrev paperSemantics : PaperLeaf -> PaperCandidate -> Prop :=
  FixedActive.PaperProfile.ObligationPlan.semantics

abbrev paperTarget : PaperCandidate -> Prop :=
  FixedActive.PaperProfile.ObligationPlan.target

abbrev paperChecks : List PaperLeaf :=
  FixedActive.PaperProfile.ObligationPlan.checks

/-! ## Canonical operational `Pi_DEC` acceptance -/

/-- Exact operational `Pi_DEC` accepts the canonical private split whenever
the complete public source product is bound to the fixed source assignments.
No source relation, norm, carried-evaluation, or challenge-validity premise is
used: those are independent leaves in the six-leaf plan. -/
theorem canonicalPiDecAccepted_of_inputBound
    (profile : FixedActive.PaperProfile.Profile Sources.shape
      Context.publicRingColumns Context.publicFits Context.verifierRows)
    (source : FixedActive.PaperProfile.Source Sources.shape
      Context.publicRingColumns Context.publicFits Context.verifierRows)
    (data : Sources.Data Sources.shape)
    (witness : FixedActive.PaperProfile.Witness Sources.shape)
    (input : FixedActive.PaperProfile.InputBound profile source data) :
    PiDEC.PaperVerifier.OutputAccepted
      (FixedActive.PaperProfile.decAlgebra profile)
      (FixedActive.PaperProfile.decPublicInputSplit profile)
      (FixedActive.PaperProfile.decEvaluationArity profile)
      (FixedActive.PaperProfile.parentOf profile source data witness)
      (FixedActive.PaperProfile.childrenOf profile source data witness) := by
  let algebra := FixedActive.PaperProfile.decAlgebra profile
  let parent := FixedActive.PaperProfile.parentOf profile source data witness
  let assignment :=
    FixedActive.PaperProfile.combinedAssignment profile data witness
  let output := FixedActive.PaperProfile.childrenOf profile source data witness
  have sourceCommitment : forall index,
      (FixedActive.PaperProfile.semantics profile).commit
          (FixedActive.PaperProfile.assignments profile data index) =
        (FixedActive.PaperProfile.outputs profile source data witness index).commitment := by
    intro index
    simpa [FixedActive.PaperProfile.outputs, PiCCS.honestOutputs,
      PiCCS.honestOutput, FixedActive.PaperProfile.assignments] using
      (InputAuthority.BoundToSources.sourceCommitment
        Context.publicRingColumns Context.publicFits
        (FixedActive.PaperProfile.commit profile) data profile.alignment source
        input index)
  have sourcePublicInput : forall index,
      (FixedActive.PaperProfile.semantics profile).projectPublicInput
          (FixedActive.PaperProfile.assignments profile data index) =
        (FixedActive.PaperProfile.outputs profile source data witness index).publicInput := by
    intro index
    simpa [FixedActive.PaperProfile.outputs, PiCCS.honestOutputs,
      PiCCS.honestOutput, FixedActive.PaperProfile.assignments,
      FixedActive.PaperProfile.semantics, productSemantics,
      Phi81Relation.relationSemantics, sourcePublicInput] using
      (InputAuthority.BoundToSources.sourcePublicInput
        Context.publicRingColumns Context.publicFits
        (FixedActive.PaperProfile.commit profile) data profile.alignment source
        input index)
  have sourceStructure : forall index,
      (source.source index).constraintSystem =
        FixedActive.PaperProfile.systemOf profile data := by
    intro index
    simpa [FixedActive.PaperProfile.systemOf] using
      (InputAuthority.BoundToSources.sourceStructure
        Context.publicRingColumns Context.publicFits
        (FixedActive.PaperProfile.commit profile) data profile.alignment source
        input index)
  have sourceEvaluations : forall index,
      (FixedActive.PaperProfile.semantics profile).evaluations
          (FixedActive.PaperProfile.systemOf profile data)
          (FixedActive.PaperProfile.assignments profile data index)
          witness.point =
        (FixedActive.PaperProfile.outputs profile source data witness index).evaluations := by
    intro index
    simp only [FixedActive.PaperProfile.outputs, PiCCS.honestOutputs,
      PiCCS.honestOutput]
    rw [sourceStructure index]
  have parentCommitment :
      (FixedActive.PaperProfile.semantics profile).commit assignment =
        parent.commitment := by
    calc
      (FixedActive.PaperProfile.semantics profile).commit assignment =
          (FixedActive.PaperProfile.rlcAlgebra profile).combineCommitment
            witness.challenges
            (fun index =>
              (FixedActive.PaperProfile.semantics profile).commit
                (FixedActive.PaperProfile.assignments profile data index)) :=
        (FixedActive.PaperProfile.rlcAlgebra profile).commit_hom
          witness.challenges
          (FixedActive.PaperProfile.assignments profile data)
      _ = (FixedActive.PaperProfile.rlcAlgebra profile).combineCommitment
            witness.challenges
            (fun index =>
              (FixedActive.PaperProfile.outputs profile source data witness
                index).commitment) := by
        apply congrArg
        funext index
        exact sourceCommitment index
      _ = parent.commitment := by
        rfl
  have parentPublicInput :
      (FixedActive.PaperProfile.semantics profile).projectPublicInput assignment =
        parent.publicInput := by
    calc
      (FixedActive.PaperProfile.semantics profile).projectPublicInput assignment =
          (FixedActive.PaperProfile.rlcAlgebra profile).combinePublicInput
            witness.challenges
            (fun index =>
              (FixedActive.PaperProfile.semantics profile).projectPublicInput
                (FixedActive.PaperProfile.assignments profile data index)) :=
        (FixedActive.PaperProfile.rlcAlgebra profile).publicInput_hom
          witness.challenges
          (FixedActive.PaperProfile.assignments profile data)
      _ = (FixedActive.PaperProfile.rlcAlgebra profile).combinePublicInput
            witness.challenges
            (fun index =>
              (FixedActive.PaperProfile.outputs profile source data witness
                index).publicInput) := by
        apply congrArg
        funext index
        exact sourcePublicInput index
      _ = parent.publicInput := by
        rfl
  have parentEvaluations :
      (FixedActive.PaperProfile.semantics profile).evaluations
          parent.constraintSystem assignment parent.point =
        parent.evaluations := by
    calc
      (FixedActive.PaperProfile.semantics profile).evaluations
          parent.constraintSystem assignment parent.point =
          (FixedActive.PaperProfile.semantics profile).evaluations
            (FixedActive.PaperProfile.systemOf profile data) assignment
            witness.point := by
        rfl
      _ = (FixedActive.PaperProfile.rlcAlgebra profile).combineEvaluations
            witness.challenges
            (fun index =>
              (FixedActive.PaperProfile.semantics profile).evaluations
                (FixedActive.PaperProfile.systemOf profile data)
                (FixedActive.PaperProfile.assignments profile data index)
                witness.point) :=
        (FixedActive.PaperProfile.rlcAlgebra profile).evaluations_hom
          (FixedActive.PaperProfile.systemOf profile data) witness.point
          witness.challenges
          (FixedActive.PaperProfile.assignments profile data)
      _ = (FixedActive.PaperProfile.rlcAlgebra profile).combineEvaluations
            witness.challenges
            (fun index =>
              (FixedActive.PaperProfile.outputs profile source data witness
                index).evaluations) := by
        apply congrArg
        funext index
        exact sourceEvaluations index
      _ = parent.evaluations := by
        rfl
  have computed :
      PiDEC.PaperVerifier.children
          (FixedActive.PaperProfile.decPublicInputSplit profile)
          (PiDEC.PaperVerifier.attemptForOutput parent output) = output := by
    simpa [algebra, parent, assignment, output,
      FixedActive.PaperProfile.childrenOf,
      PiDEC.PaperVerifier.attemptForOutput,
      PiDEC.PaperVerifier.messagesOf,
      PiDEC.PaperVerifier.honestAttempt,
      PiDEC.PaperVerifier.honestMessages,
      PiDEC.childrenOf] using
      (PiDEC.PaperVerifier.honestChildren_eq_childrenOf algebra
        (FixedActive.PaperProfile.decPublicInputSplit profile) parent assignment
        parentPublicInput)
  refine {
    outputComputed := computed
    checks := {
      parentCombined := rfl
      parentEvaluationSize := ?_
      messageEvaluationSize := ?_
      commitmentEquation := ?_
      evaluationEquation := ?_
    }
  }
  · change parent.evaluations.size =
      (FixedActive.PaperProfile.decEvaluationArity profile).count
        parent.constraintSystem
    rw [← parentEvaluations]
    exact
      (FixedActive.PaperProfile.decEvaluationArity profile).evaluations_size
        parent.constraintSystem assignment parent.point
  · intro child
    simpa [output, FixedActive.PaperProfile.childrenOf, PiDEC.childrenOf,
      PiDEC.PaperVerifier.attemptForOutput,
      PiDEC.PaperVerifier.messagesOf] using
      (FixedActive.PaperProfile.decEvaluationArity profile).evaluations_size
        parent.constraintSystem (algebra.splitAssignment assignment child)
        parent.point
  · change parent.commitment = algebra.recomposeCommitment
      (fun child => (output child).commitment)
    calc
      parent.commitment =
          (FixedActive.PaperProfile.semantics profile).commit assignment :=
        parentCommitment.symm
      _ = (FixedActive.PaperProfile.semantics profile).commit
          (algebra.recomposeAssignment
            (algebra.splitAssignment assignment)) := by
        rw [algebra.split_recompose assignment]
      _ = algebra.recomposeCommitment
          (fun child =>
            (FixedActive.PaperProfile.semantics profile).commit
              (algebra.splitAssignment assignment child)) :=
        algebra.commit_hom (algebra.splitAssignment assignment)
      _ = algebra.recomposeCommitment
          (fun child => (output child).commitment) := by
        rfl
  · change parent.evaluations = algebra.recomposeEvaluations
      (fun child => (output child).evaluations)
    calc
      parent.evaluations =
          (FixedActive.PaperProfile.semantics profile).evaluations
            parent.constraintSystem assignment parent.point :=
        parentEvaluations.symm
      _ = (FixedActive.PaperProfile.semantics profile).evaluations
          parent.constraintSystem
          (algebra.recomposeAssignment
            (algebra.splitAssignment assignment)) parent.point := by
        rw [algebra.split_recompose assignment]
      _ = algebra.recomposeEvaluations
          (fun child =>
            (FixedActive.PaperProfile.semantics profile).evaluations
              parent.constraintSystem
              (algebra.splitAssignment assignment child) parent.point) :=
        algebra.evaluations_hom parent.constraintSystem parent.point
          (algebra.splitAssignment assignment)
      _ = algebra.recomposeEvaluations
          (fun child => (output child).evaluations) := by
        rfl

/-! ## Accepted fixed baseline -/

def paperBaselineProfile := SourceBinding.profile

def paperBaselineWitness : FixedActive.PaperProfile.Witness Sources.shape :=
  SourceBinding.witness

def paperBaselineTarget :=
  FixedActive.PaperProfile.childrenOf paperBaselineProfile Context.context.input
    Sources.data paperBaselineWitness

def paperBaselineCandidate : PaperCandidate := {
  profile := paperBaselineProfile
  source := Context.context.input
  data := Sources.data
  witness := paperBaselineWitness
  target := paperBaselineTarget
}

theorem paperBaselineTargetHolds : paperTarget paperBaselineCandidate := by
  exact FixedActive.PaperProfile.completeRealization paperBaselineProfile
    Context.context.input Sources.data Sources.paperHolds Context.sourceBound
    paperBaselineWitness (by
      intro coordinate
      exact (Context.samplerBound ()).challengeValid coordinate)

theorem paperBaselineAccepted :
    CheckPlan.Accepts paperSemantics paperChecks paperBaselineCandidate :=
  (FixedActive.PaperProfile.ObligationPlan.accepts_iff_target
    paperBaselineCandidate).mpr paperBaselineTargetHolds

/-! ## Fresh-CCS necessity -/

def falseCcsPaperProfile := FixedActive.paperProfileOf falseCcsContext

def falseCcsPaperWitness : FixedActive.PaperProfile.Witness Sources.shape :=
  FixedActive.paperWitnessOf falseCcsWitness

def falseCcsPaperTarget :=
  FixedActive.PaperProfile.childrenOf falseCcsPaperProfile
    falseCcsContext.input falseCcsData falseCcsPaperWitness

def falseCcsPaperCandidate : PaperCandidate := {
  profile := falseCcsPaperProfile
  source := falseCcsContext.input
  data := falseCcsData
  witness := falseCcsPaperWitness
  target := falseCcsPaperTarget
}

theorem falseCcsPaperPiDecAccepted :
    paperSemantics .piDecAcceptance falseCcsPaperCandidate := by
  exact canonicalPiDecAccepted_of_inputBound falseCcsPaperProfile
    falseCcsContext.input falseCcsData falseCcsPaperWitness
    (PiCcs.CanonicalContext.sourceBound falseCcsData falseCcs_runningZero
      falseCcs_carriedEvaluationsHold)

theorem falseCcsPaperWeakened :
    CheckPlan.Accepts paperSemantics
      (CheckPlan.without paperChecks .freshCcs) falseCcsPaperCandidate := by
  intro leaf member
  have retained := (CheckPlan.mem_without_iff.mp member).2
  cases leaf with
  | freshCcs => exact (retained rfl).elim
  | allSourceNorm => exact falseCcs_allSourceNormsHold
  | carriedEvaluations => exact falseCcs_carriedEvaluationsHold
  | sourceBinding =>
      exact PiCcs.CanonicalContext.sourceBound falseCcsData
        falseCcs_runningZero falseCcs_carriedEvaluationsHold
  | challengeStrongSet =>
      intro coordinate
      simpa [falseCcsPaperCandidate, falseCcsPaperProfile,
        falseCcsPaperWitness, FixedActive.paperProfileOf,
        FixedActive.paperWitnessOf, falseCcsContext,
        PiCcs.CanonicalContext.context] using
        (Context.samplerBound ()).challengeValid coordinate
  | piDecAcceptance => exact falseCcsPaperPiDecAccepted

theorem falseCcsPaperRejected :
    ¬ paperTarget falseCcsPaperCandidate := by
  intro realized
  exact falseCcs_not_freshCcs realized.paper.1

theorem freshCcs_necessary :
    CheckPlan.NecessaryForSoundness paperSemantics paperTarget paperChecks
      .freshCcs := by
  exact ⟨falseCcsPaperCandidate, falseCcsPaperWeakened,
    falseCcsPaperRejected⟩

/-! ## All-source-norm necessity -/

def highNormPaperProfile := FixedActive.paperProfileOf highNormContext

def highNormPaperWitness : FixedActive.PaperProfile.Witness Sources.shape :=
  FixedActive.paperWitnessOf highNormWitness

def highNormPaperTarget :=
  FixedActive.PaperProfile.childrenOf highNormPaperProfile highNormContext.input
    highNormData highNormPaperWitness

def highNormPaperCandidate : PaperCandidate := {
  profile := highNormPaperProfile
  source := highNormContext.input
  data := highNormData
  witness := highNormPaperWitness
  target := highNormPaperTarget
}

theorem highNormPaperPiDecAccepted :
    paperSemantics .piDecAcceptance highNormPaperCandidate := by
  exact canonicalPiDecAccepted_of_inputBound highNormPaperProfile
    highNormContext.input highNormData highNormPaperWitness
    (PiCcs.CanonicalContext.sourceBound highNormData highNorm_runningZero
      highNorm_carriedEvaluationsHold)

theorem highNormPaperWeakened :
    CheckPlan.Accepts paperSemantics
      (CheckPlan.without paperChecks .allSourceNorm) highNormPaperCandidate := by
  intro leaf member
  have retained := (CheckPlan.mem_without_iff.mp member).2
  cases leaf with
  | freshCcs => exact highNorm_freshCcsHolds
  | allSourceNorm => exact (retained rfl).elim
  | carriedEvaluations => exact highNorm_carriedEvaluationsHold
  | sourceBinding =>
      exact PiCcs.CanonicalContext.sourceBound highNormData
        highNorm_runningZero highNorm_carriedEvaluationsHold
  | challengeStrongSet =>
      intro coordinate
      simpa [highNormPaperCandidate, highNormPaperProfile,
        highNormPaperWitness, FixedActive.paperProfileOf,
        FixedActive.paperWitnessOf, highNormContext,
        PiCcs.CanonicalContext.context] using
        (Context.samplerBound ()).challengeValid coordinate
  | piDecAcceptance => exact highNormPaperPiDecAccepted

theorem highNormPaperRejected :
    ¬ paperTarget highNormPaperCandidate := by
  intro realized
  exact highNorm_not_allSourceNorms realized.paper.2.1

theorem allSourceNorm_necessary :
    CheckPlan.NecessaryForSoundness paperSemantics paperTarget paperChecks
      .allSourceNorm := by
  exact ⟨highNormPaperCandidate, highNormPaperWeakened,
    highNormPaperRejected⟩

/-! ## Carried-evaluation necessity -/

def forgedCarriedPaperProfile :=
  FixedActive.paperProfileOf forgedCarriedContext

def forgedCarriedPaperWitness :
    FixedActive.PaperProfile.Witness Sources.shape :=
  FixedActive.paperWitnessOf forgedCarriedWitness

def forgedCarriedPaperTarget :=
  FixedActive.PaperProfile.childrenOf forgedCarriedPaperProfile
    forgedCarriedContext.input forgedCarriedData forgedCarriedPaperWitness

def forgedCarriedPaperCandidate : PaperCandidate := {
  profile := forgedCarriedPaperProfile
  source := forgedCarriedContext.input
  data := forgedCarriedData
  witness := forgedCarriedPaperWitness
  target := forgedCarriedPaperTarget
}

theorem forgedCarriedPaperPiDecAccepted :
    paperSemantics .piDecAcceptance forgedCarriedPaperCandidate := by
  exact canonicalPiDecAccepted_of_inputBound forgedCarriedPaperProfile
    forgedCarriedContext.input forgedCarriedData forgedCarriedPaperWitness
    forgedCarried_sourceBound

theorem forgedCarriedPaperWeakened :
    CheckPlan.Accepts paperSemantics
      (CheckPlan.without paperChecks .carriedEvaluations)
      forgedCarriedPaperCandidate := by
  intro leaf member
  have retained := (CheckPlan.mem_without_iff.mp member).2
  cases leaf with
  | freshCcs => exact forgedCarried_freshCcsHolds
  | allSourceNorm => exact forgedCarried_allSourceNormsHold
  | carriedEvaluations => exact (retained rfl).elim
  | sourceBinding => exact forgedCarried_sourceBound
  | challengeStrongSet =>
      intro coordinate
      simpa [forgedCarriedPaperCandidate, forgedCarriedPaperProfile,
        forgedCarriedPaperWitness, FixedActive.paperProfileOf,
        FixedActive.paperWitnessOf, forgedCarriedContext] using
        (Context.samplerBound ()).challengeValid coordinate
  | piDecAcceptance => exact forgedCarriedPaperPiDecAccepted

theorem forgedCarriedPaperRejected :
    ¬ paperTarget forgedCarriedPaperCandidate := by
  intro realized
  exact forgedCarried_not_evaluations realized.paper.2.2

theorem carriedEvaluations_necessary :
    CheckPlan.NecessaryForSoundness paperSemantics paperTarget paperChecks
      .carriedEvaluations := by
  exact ⟨forgedCarriedPaperCandidate, forgedCarriedPaperWeakened,
    forgedCarriedPaperRejected⟩

/-! ## Complete-source-binding necessity -/

def mismatchedSourcePaperCandidate : PaperCandidate := {
  profile := paperBaselineProfile
  source := mismatchedSourceProduct
  data := Sources.data
  witness := paperBaselineWitness
  target := paperBaselineTarget
}

theorem mismatchedSourcePaper_not_bound :
    ¬ paperSemantics .sourceBinding mismatchedSourcePaperCandidate := by
  intro bound
  have stage := (bound.fresh (Fin.last 0)).stage
  change NormStage.combined = NormStage.fresh at stage
  cases stage

theorem mismatchedSource_parent_eq_baseline :
    FixedActive.PaperProfile.parentOf paperBaselineProfile
        mismatchedSourceProduct Sources.data paperBaselineWitness =
      FixedActive.PaperProfile.parentOf paperBaselineProfile
        Context.context.input Sources.data paperBaselineWitness := by
  exact SourceBinding.parent_eq

theorem mismatchedSourcePaperPiDecAccepted :
    paperSemantics .piDecAcceptance mismatchedSourcePaperCandidate := by
  change PiDEC.PaperVerifier.OutputAccepted
    (FixedActive.PaperProfile.decAlgebra paperBaselineProfile)
    (FixedActive.PaperProfile.decPublicInputSplit paperBaselineProfile)
    (FixedActive.PaperProfile.decEvaluationArity paperBaselineProfile)
    (FixedActive.PaperProfile.parentOf paperBaselineProfile
      mismatchedSourceProduct Sources.data paperBaselineWitness)
    paperBaselineTarget
  rw [mismatchedSource_parent_eq_baseline]
  exact paperBaselineTargetHolds.piDecAccepted

theorem mismatchedSourcePaperWeakened :
    CheckPlan.Accepts paperSemantics
      (CheckPlan.without paperChecks .sourceBinding)
      mismatchedSourcePaperCandidate := by
  intro leaf member
  have retained := (CheckPlan.mem_without_iff.mp member).2
  cases leaf with
  | freshCcs => exact Sources.freshCcsHolds
  | allSourceNorm => exact Sources.allSourceNormsHold
  | carriedEvaluations => exact Sources.carriedEvaluationsHold
  | sourceBinding => exact (retained rfl).elim
  | challengeStrongSet =>
      intro coordinate
      exact (Context.samplerBound ()).challengeValid coordinate
  | piDecAcceptance => exact mismatchedSourcePaperPiDecAccepted

theorem mismatchedSourcePaperRejected :
    ¬ paperTarget mismatchedSourcePaperCandidate := by
  intro realized
  exact mismatchedSourcePaper_not_bound realized.input

theorem sourceBinding_necessary :
    CheckPlan.NecessaryForSoundness paperSemantics paperTarget paperChecks
      .sourceBinding := by
  exact ⟨mismatchedSourcePaperCandidate, mismatchedSourcePaperWeakened,
    mismatchedSourcePaperRejected⟩

/-! ## Challenge-set necessity -/

def outsidePaperWitness : FixedActive.PaperProfile.Witness Sources.shape where
  point := paperBaselineWitness.point
  challenges := fun _ => Phi81StrongSet.outsideChallenge

def outsidePaperTarget :=
  FixedActive.PaperProfile.childrenOf paperBaselineProfile Context.context.input
    Sources.data outsidePaperWitness

def outsidePaperCandidate : PaperCandidate := {
  profile := paperBaselineProfile
  source := Context.context.input
  data := Sources.data
  witness := outsidePaperWitness
  target := outsidePaperTarget
}

theorem outsidePaperPiDecAccepted :
    paperSemantics .piDecAcceptance outsidePaperCandidate := by
  exact canonicalPiDecAccepted_of_inputBound paperBaselineProfile
    Context.context.input Sources.data outsidePaperWitness Context.sourceBound

theorem outsidePaperWeakened :
    CheckPlan.Accepts paperSemantics
      (CheckPlan.without paperChecks .challengeStrongSet)
      outsidePaperCandidate := by
  intro leaf member
  have retained := (CheckPlan.mem_without_iff.mp member).2
  cases leaf with
  | freshCcs => exact Sources.freshCcsHolds
  | allSourceNorm => exact Sources.allSourceNormsHold
  | carriedEvaluations => exact Sources.carriedEvaluationsHold
  | sourceBinding => exact Context.sourceBound
  | challengeStrongSet => exact (retained rfl).elim
  | piDecAcceptance => exact outsidePaperPiDecAccepted

theorem outsidePaperRejected :
    ¬ paperTarget outsidePaperCandidate := by
  intro realized
  let first : Fin FixedActive.arity.total :=
    Fin.last (FixedActive.arity.total - 1)
  have valid := realized.challengesValid first
  exact Phi81StrongSet.outsideChallenge_not_member (by
    simpa [outsidePaperCandidate, outsidePaperWitness, paperBaselineProfile,
      FixedActive.paperProfileOf, FixedActive.PaperProfile.rlcAlgebra] using
      valid)

theorem challengeStrongSet_necessary :
    CheckPlan.NecessaryForSoundness paperSemantics paperTarget paperChecks
      .challengeStrongSet := by
  exact ⟨outsidePaperCandidate, outsidePaperWeakened, outsidePaperRejected⟩

/-! ## Exact operational `Pi_DEC` necessity -/

def firstChild : Fin productionGlobalParams.k := Fin.last 13

def forgedPiDecTarget
    (child : Fin productionGlobalParams.k) :=
  if child = firstChild then
    SemanticFold.ObligationPlan.Necessity.withDifferentStage
      (paperBaselineTarget child)
  else
    paperBaselineTarget child

def forgedPiDecCandidate : PaperCandidate :=
  { paperBaselineCandidate with target := forgedPiDecTarget }

theorem forgedPiDec_not_accepted :
    ¬ paperSemantics .piDecAcceptance forgedPiDecCandidate := by
  intro accepted
  have atFirst := congrFun accepted.outputComputed firstChild
  have stageEqual := congrArg (fun statement => statement.stage) atFirst
  change NormStage.fresh = NormStage.combined at stageEqual
  cases stageEqual

theorem forgedPiDecWeakened :
    CheckPlan.Accepts paperSemantics
      (CheckPlan.without paperChecks .piDecAcceptance)
      forgedPiDecCandidate := by
  intro leaf member
  have retained := (CheckPlan.mem_without_iff.mp member).2
  cases leaf with
  | freshCcs => exact Sources.freshCcsHolds
  | allSourceNorm => exact Sources.allSourceNormsHold
  | carriedEvaluations => exact Sources.carriedEvaluationsHold
  | sourceBinding => exact Context.sourceBound
  | challengeStrongSet =>
      intro coordinate
      exact (Context.samplerBound ()).challengeValid coordinate
  | piDecAcceptance => exact (retained rfl).elim

theorem forgedPiDecRejected :
    ¬ paperTarget forgedPiDecCandidate := by
  intro realized
  exact forgedPiDec_not_accepted realized.piDecAccepted

theorem piDecAcceptance_necessary :
    CheckPlan.NecessaryForSoundness paperSemantics paperTarget paperChecks
      .piDecAcceptance := by
  exact ⟨forgedPiDecCandidate, forgedPiDecWeakened, forgedPiDecRejected⟩

/-! ## Closed ledger -/

def provedNecessaryLeaves : List PaperLeaf :=
  [.freshCcs, .allSourceNorm, .carriedEvaluations, .sourceBinding,
    .challengeStrongSet, .piDecAcceptance]

theorem provedNecessaryLeaves_eq_checks :
    provedNecessaryLeaves = paperChecks := by
  rfl

structure NecessityWitnesses : Prop where
  freshCcs : CheckPlan.NecessaryForSoundness paperSemantics paperTarget
    paperChecks .freshCcs
  allSourceNorm : CheckPlan.NecessaryForSoundness paperSemantics paperTarget
    paperChecks .allSourceNorm
  carriedEvaluations : CheckPlan.NecessaryForSoundness paperSemantics paperTarget
    paperChecks .carriedEvaluations
  sourceBinding : CheckPlan.NecessaryForSoundness paperSemantics paperTarget
    paperChecks .sourceBinding
  challengeStrongSet : CheckPlan.NecessaryForSoundness paperSemantics paperTarget
    paperChecks .challengeStrongSet
  piDecAcceptance : CheckPlan.NecessaryForSoundness paperSemantics paperTarget
    paperChecks .piDecAcceptance

theorem necessityWitnesses : NecessityWitnesses := {
  freshCcs := freshCcs_necessary
  allSourceNorm := allSourceNorm_necessary
  carriedEvaluations := carriedEvaluations_necessary
  sourceBinding := sourceBinding_necessary
  challengeStrongSet := challengeStrongSet_necessary
  piDecAcceptance := piDecAcceptance_necessary
}

theorem inclusionMinimalSound :
    CheckPlan.InclusionMinimalSound paperSemantics paperTarget paperChecks := by
  apply CheckPlan.inclusionMinimalSound_of_witnesses
  · intro candidate accepted
    exact
      (FixedActive.PaperProfile.ObligationPlan.accepts_iff_target candidate).mp
        accepted
  · intro leaf _member
    cases leaf with
    | freshCcs => exact freshCcs_necessary
    | allSourceNorm => exact allSourceNorm_necessary
    | carriedEvaluations => exact carriedEvaluations_necessary
    | sourceBinding => exact sourceBinding_necessary
    | challengeStrongSet => exact challengeStrongSet_necessary
    | piDecAcceptance => exact piDecAcceptance_necessary

end Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.SelectedNifs.PaperSemanticMinimality
