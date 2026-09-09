import NightstreamFPrime.Export.Stage1.PiCCSActionPayloadSupport
import NightstreamFPrime.Export.Stage1.PiCCSAssignmentSoundness

/-!
Owns compilation of the declared PiCCS action payload through the parent's
ordinary source map. The result reads existing sparse forms and adds no
payload variables or copy rows. The active Poseidon plan must still select
this wiring before the independent payload allocation can be removed.
-/

namespace NightstreamFPrime.Export.Stage1.PiCCSPayloadWiring

open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

abbrev Program := Lifecycle.Stage1.Application.Program

/-- The action expressions use source indices before the physical column
permutation; the parent map already owns their physical source forms. -/
def sourceMap {program : Program} {logicalWidth : Nat}
    (geometry : PiCCSOrdinaryRetainedGeometry.Geometry program logicalWidth) :
    SourceCompiler.SourceMap Spartan.SourceColumnCount logicalWidth where
  form := fun column => (PiCCSOrdinaryDirectPlan.sourceMap geometry).form
    ⟨Spartan.sourceToSpartan column.val, Spartan.sourceToSpartan_lt _ column.isLt⟩

theorem sourceMap_preserves {program : Program} {logicalWidth : Nat}
    (geometry : PiCCSOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth) :
    (sourceMap geometry).Preserves assignment
      (Spartan.pullback (PiCCSAssignmentSoundness.decodedEnv geometry assignment)) := by
  intro column
  exact PiCCSAssignmentSoundness.decodedEnv_preserves geometry assignment
    ⟨Spartan.sourceToSpartan column.val, Spartan.sourceToSpartan_lt _ column.isLt⟩

/-- Fail-closed affine compilation of one canonical payload word. The fixed
one-coordinate supplies constants; every other value is a parent source form. -/
def form? {program : Program} {logicalWidth : Nat}
    (geometry : PiCCSOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (index : Fin PiCCSActionPayloadBlock.payloadCount) :
    Option (SparseForm logicalWidth) := do
  let lowered ← SourceCompiler.lowerAffine? Spartan.SourceColumnCount
    (PiCCSActionPayloadBlock.payloadExpression index)
  pure (SourceCompiler.compileCombination (sourceMap geometry)
    (PiCCSOrdinaryRetainedGeometry.oneColumn geometry)
    lowered.combination lowered.bounded)

/-- Every declared payload word has a bounded affine source representation. -/
theorem lowering?_isSome (index : Fin PiCCSActionPayloadBlock.payloadCount) :
    (SourceCompiler.lowerAffine? Spartan.SourceColumnCount
      (PiCCSActionPayloadBlock.payloadExpression index)).isSome := by
  rcases PiCCSActionPayloadSupport.payloadExpression_affine index with
    ⟨lowered, loweredEq⟩
  have supported := R1CS.lowerAffine_varsSatisfy
    (PiCCSActionPayloadBlock.payloadExpression index)
    PiCCSOrdinarySourceSupport.Source
    (PiCCSActionPayloadSupport.payloadExpression_supported index) lowered loweredEq
  have bounded : SourceCompiler.CombinationBounded Spartan.SourceColumnCount
      lowered.combination := by
    intro term member
    exact PiCCSOrdinarySourceSupport.source_lt_sourceColumnCount (supported term member)
  unfold SourceCompiler.lowerAffine?
  rw [loweredEq]
  dsimp only
  cases checked : SourceCompiler.combinationBoundedDecidable
      Spartan.SourceColumnCount lowered.combination with
  | isTrue proof => simp only [checked]; rfl
  | isFalse rejected => exact False.elim (rejected bounded)

def lowering (index : Fin PiCCSActionPayloadBlock.payloadCount) :
    SourceCompiler.AffineSource Spartan.SourceColumnCount :=
  (SourceCompiler.lowerAffine? Spartan.SourceColumnCount
    (PiCCSActionPayloadBlock.payloadExpression index)).get (lowering?_isSome index)

theorem lowering?_eq_some_lowering (index : Fin PiCCSActionPayloadBlock.payloadCount) :
    SourceCompiler.lowerAffine? Spartan.SourceColumnCount
      (PiCCSActionPayloadBlock.payloadExpression index) = some (lowering index) := by
  cases found : SourceCompiler.lowerAffine? Spartan.SourceColumnCount
      (PiCCSActionPayloadBlock.payloadExpression index) with
  | none =>
      have possible := lowering?_isSome index
      simp only [found, Option.isSome_none, Bool.false_eq_true] at possible
  | some value => simp [lowering, found]

theorem lowering_supported (index : Fin PiCCSActionPayloadBlock.payloadCount) :
    (lowering index).combination.VarsSatisfy PiCCSOrdinarySourceSupport.Source := by
  have found := lowering?_eq_some_lowering index
  unfold SourceCompiler.lowerAffine? at found
  cases loweredEq : R1CS.lowerAffine (PiCCSActionPayloadBlock.payloadExpression index) with
  | none => simp [loweredEq] at found
  | some lowered =>
      rw [loweredEq] at found
      dsimp only at found
      cases checked : SourceCompiler.combinationBoundedDecidable
          Spartan.SourceColumnCount lowered.combination with
      | isFalse rejected => simp [checked] at found
      | isTrue bounded =>
          rw [checked] at found
          have same := congrArg
            (fun value : SourceCompiler.AffineSource Spartan.SourceColumnCount =>
              value.combination) (Option.some.inj found)
          change lowered.combination = (lowering index).combination at same
          rw [← same]
          exact R1CS.lowerAffine_varsSatisfy
            (PiCCSActionPayloadBlock.payloadExpression index)
            PiCCSOrdinarySourceSupport.Source
            (PiCCSActionPayloadSupport.payloadExpression_supported index)
            lowered loweredEq

theorem form?_isSome {program : Program} {logicalWidth : Nat}
    (geometry : PiCCSOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (index : Fin PiCCSActionPayloadBlock.payloadCount) :
    (form? geometry index).isSome := by
  unfold form?
  rw [lowering?_eq_some_lowering]
  rfl

/-- Total selection is justified by affine compilation of the declared word. -/
def form {program : Program} {logicalWidth : Nat}
    (geometry : PiCCSOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (index : Fin PiCCSActionPayloadBlock.payloadCount) : SparseForm logicalWidth :=
  (form? geometry index).get (form?_isSome geometry index)

theorem form?_eq_some_form {program : Program} {logicalWidth : Nat}
    (geometry : PiCCSOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (index : Fin PiCCSActionPayloadBlock.payloadCount) :
    form? geometry index = some (form geometry index) := by
  cases found : form? geometry index with
  | none =>
      have possible := form?_isSome geometry index
      simp only [found, Option.isSome_none, Bool.false_eq_true] at possible
  | some value => simp [form, found]

/-- Exact ordered form used by the matrix IR, including the constant entry. -/
theorem form_eq_compileCombination {program : Program} {logicalWidth : Nat}
    (geometry : PiCCSOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (index : Fin PiCCSActionPayloadBlock.payloadCount) :
    form geometry index =
      SourceCompiler.compileCombination (sourceMap geometry)
        (PiCCSOrdinaryRetainedGeometry.oneColumn geometry)
        (lowering index).combination (lowering index).bounded := by
  have found := form?_eq_some_form geometry index
  unfold form? at found
  rw [lowering?_eq_some_lowering] at found
  exact (Option.some.inj found).symm

/-- The compiled payload evaluates to the exact declared action expression
for every assignment. No raw packet, encoding, or coherence premise is used. -/
theorem form?_eval {program : Program} {logicalWidth : Nat}
    (geometry : PiCCSOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (index : Fin PiCCSActionPayloadBlock.payloadCount)
    (form : SparseForm logicalWidth) (found : form? geometry index = some form)
    (assignment : Assignment F logicalWidth)
    (one : assignment (PiCCSOrdinaryRetainedGeometry.oneColumn geometry) = 1) :
    form.eval assignment =
      (PiCCSActionPayloadBlock.payloadExpression index).eval
        (Spartan.pullback (PiCCSAssignmentSoundness.decodedEnv geometry assignment)) := by
  unfold form? at found
  cases loweredEq : SourceCompiler.lowerAffine? Spartan.SourceColumnCount
      (PiCCSActionPayloadBlock.payloadExpression index) with
  | none => simp [loweredEq] at found
  | some lowered =>
      rw [loweredEq] at found
      change some (SourceCompiler.compileCombination (sourceMap geometry)
        (PiCCSOrdinaryRetainedGeometry.oneColumn geometry)
        lowered.combination lowered.bounded) = some form at found
      have equal := Option.some.inj found
      subst form
      rw [SourceCompiler.compileCombination_eval (sourceMap geometry)
        (PiCCSOrdinaryRetainedGeometry.oneColumn geometry)
        lowered.combination lowered.bounded assignment
        (Spartan.pullback (PiCCSAssignmentSoundness.decodedEnv geometry assignment))
        one (sourceMap_preserves geometry assignment)]
      exact SourceCompiler.lowerAffine?_sound
        (PiCCSActionPayloadBlock.payloadExpression index) lowered loweredEq _

/-- Each selected parent form reads the exact action expression for every
assignment with its required constant coordinate. -/
theorem form_eval {program : Program} {logicalWidth : Nat}
    (geometry : PiCCSOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (index : Fin PiCCSActionPayloadBlock.payloadCount)
    (assignment : Assignment F logicalWidth)
    (one : assignment (PiCCSOrdinaryRetainedGeometry.oneColumn geometry) = 1) :
    (form geometry index).eval assignment =
      (PiCCSActionPayloadBlock.payloadExpression index).eval
        (Spartan.pullback (PiCCSAssignmentSoundness.decodedEnv geometry assignment)) :=
  form?_eval geometry index (form geometry index)
    (form?_eq_some_form geometry index) assignment one

/-- Canonical construction reads the same parent values through the existing
source-map preservation proof. The payload has no separate allocation. -/
theorem form_eval_source {program : Program} {logicalWidth : Nat}
    (geometry : PiCCSOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (index : Fin PiCCSActionPayloadBlock.payloadCount)
    (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F)
    (encodes : PiCCSOrdinaryRetainedGeometry.Encodes geometry assignment
      (PiRLCRetainedPreservation.sourceAssignment program base groupValue products))
    (one : assignment (PiCCSOrdinaryRetainedGeometry.oneColumn geometry) = 1) :
    (form geometry index).eval assignment =
      PiCCSActionPayloadBlock.payloadValue program
        (PiRLCRetainedPreservation.sourceAssignment program base groupValue products)
        index := by
  rw [form_eval geometry index assignment one]
  unfold PiCCSActionPayloadBlock.payloadValue
  apply Expr.eval_eq_of_agree_satisfy _ PiCCSOrdinarySourceSupport.Source _ _
    (PiCCSActionPayloadSupport.payloadExpression_supported index)
  intro source supported
  let column : Fin Spartan.spartanColumnCount :=
    ⟨Spartan.sourceToSpartan source, Spartan.sourceToSpartan_lt _
      (PiCCSOrdinarySourceSupport.source_lt_sourceColumnCount supported)⟩
  exact (PiCCSAssignmentSoundness.decodedEnv_preserves geometry assignment column).symm.trans
    ((PiCCSOrdinaryDirectPlan.sourceMap_form_eval_of_target geometry assignment
      base groupValue products encodes column
      (PiCCSOrdinarySourceSupport.source_target source supported)).trans
      (PiCCSPoseidonPreservation.packageEnv_sourceAssignment program base
        groupValue products source
        (PiCCSOrdinarySourceSupport.source_lt_sourceColumnCount supported)).symm)

end NightstreamFPrime.Export.Stage1.PiCCSPayloadWiring
