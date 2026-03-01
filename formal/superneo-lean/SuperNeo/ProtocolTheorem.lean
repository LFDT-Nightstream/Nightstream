import SuperNeo.InteractiveReductions
import SuperNeo.Interp

/-!
Canonical final theorem shape for the SuperNeo scaffold.

This file provides:
- one explicit assumption registry,
- completeness and knowledge-soundness statement shapes,
- one canonical theorem constructor from assumptions.
-/

namespace SuperNeo

/-- Typed boundary for Schwartz-Zippel style polynomial identity control. -/
def schwartzZippelAssumption : Prop :=
  interpolationAssumption

/-- Typed boundary for Ajtai commitment binding. -/
def ajtaiBindingAssumption : Prop :=
  lowNormInvertibilityAssumption Goldilocks.halfQ

/-- Typed boundary for Ajtai relaxed binding. -/
def ajtaiRelaxedBindingAssumption : Prop :=
  lowNormInvertibilityAssumption Goldilocks.halfQ

/-- Canonical final assumption registry. -/
structure FinalTheoremAssumptions (ctx : ProtocolTargetContext) where
  reduction : InteractiveReductionAssumptions ctx
  sumcheckSoundnessBoundary : SumcheckSoundnessAssumption
  sumcheckCompletenessBoundary : SumcheckCompletenessAssumption
  schwartzZippel : schwartzZippelAssumption
  ajtaiBinding : ajtaiBindingAssumption
  ajtaiRelaxedBinding : ajtaiRelaxedBindingAssumption

/-- Final completeness statement shape. -/
def FinalCompletenessStatement
  (ctx : ProtocolTargetContext)
  (_hA : FinalTheoremAssumptions ctx) : Prop :=
  ceRelaxedRelation ctx

/-- Final knowledge-soundness statement shape. -/
def FinalKnowledgeSoundnessStatement
  (ctx : ProtocolTargetContext)
  (_hA : FinalTheoremAssumptions ctx) : Prop :=
  strongCompositionStatement ctx ∧
  schwartzZippelAssumption ∧
  ajtaiBindingAssumption ∧
  ajtaiRelaxedBindingAssumption

/-- Canonical final theorem container. -/
structure FinalTheoremShape
  (ctx : ProtocolTargetContext)
  (hA : FinalTheoremAssumptions ctx) : Prop where
  completeness : FinalCompletenessStatement ctx hA
  knowledgeSoundness : FinalKnowledgeSoundnessStatement ctx hA

/-- Canonical final theorem constructor. -/
theorem finalTheoremShape_of_assumptions
  {ctx : ProtocolTargetContext}
  (hA : FinalTheoremAssumptions ctx) :
  FinalTheoremShape ctx hA := by
  refine ⟨?_, ?_⟩
  · exact (weakComposition_of_assumptions hA.reduction).1
  · exact ⟨strongComposition_of_assumptions hA.reduction,
      hA.schwartzZippel,
      hA.ajtaiBinding,
      hA.ajtaiRelaxedBinding⟩

end SuperNeo
