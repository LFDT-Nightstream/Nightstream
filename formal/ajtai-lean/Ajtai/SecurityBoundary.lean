import Ajtai.EstimatorModel

/-!
Contract: the explicit security boundary for the selected Ajtai width.

Owns: the parameter bundle, the selected protocol-binding parameters, and the
logical composition from a collision extractor plus an MSIS boundary to an
Ajtai binding boundary.

Does not own: a concrete commitment encoding, a collision extractor for the
Nightstream relation, an MSIS game, a probability model, or an MSIS hardness
proof. Those inputs must come from the authoritative protocol integration.

This small interface replaces the dependency on the deprecated
`superneo-lean` proof package. It does not copy that package's abstract
probability wrappers or present them as a concrete security reduction.
-/

namespace Ajtai.SecurityBoundary

open Ajtai.EstimatorModel
open Ajtai.Parameters

/-- Parameters used by the Ajtai binding boundary. -/
structure AjtaiParams where
  ringDim : Nat
  messageLength : Nat
  bindingNormBound : Nat
  relaxedExpansion : Nat

namespace AjtaiParams

def kappa (params : AjtaiParams) : Nat := params.ringDim

def msgLen (params : AjtaiParams) : Nat := params.messageLength

def SideConditions (params : AjtaiParams) : Prop :=
  0 < params.ringDim ∧
    0 < params.messageLength ∧
    0 < params.bindingNormBound ∧
    0 < params.relaxedExpansion

end AjtaiParams

/-- Verifier-selected parameters for the long protocol-binding commitment. -/
def protocolBindingParams : AjtaiParams where
  ringDim := protocolBindingRank
  messageLength := computedMaxRingColumns
  bindingNormBound := 2
  relaxedExpansion := 1

theorem protocolBindingParams_values :
    protocolBindingParams.kappa = 2 ∧
      protocolBindingParams.msgLen = 50_371 ∧
      protocolBindingParams.bindingNormBound = 2 ∧
      protocolBindingParams.relaxedExpansion = 1 := by
  native_decide

theorem protocolBindingParams_sideConditions :
    protocolBindingParams.SideConditions := by
  change 0 < 2 ∧ 0 < computedMaxRingColumns ∧ 0 < 2 ∧ 0 < 1
  rw [computedMaxRingColumns_eq]
  decide

/--
The two concrete events and their extractor implication. The integration layer
must instantiate these fields from one authoritative commitment relation.
-/
structure LatticeReductionLaws (params : AjtaiParams) where
  bindingCollision : Prop
  msisBreakEvent : Prop
  collisionImpliesBreak : bindingCollision → msisBreakEvent

/-- Explicit boundary assumption that the selected MSIS break event is absent. -/
structure MSISHardnessBoundary
    {params : AjtaiParams} (laws : LatticeReductionLaws params) where
  noBreak : ¬ laws.msisBreakEvent

/-- Binding means that the selected concrete collision event is absent. -/
def AjtaiBindingAssumption
    {params : AjtaiParams} (laws : LatticeReductionLaws params) : Prop :=
  ¬ laws.bindingCollision

/-- The extractor field exposes the exact collision-to-MSIS implication. -/
theorem collision_implies_msis_break
    (laws : LatticeReductionLaws protocolBindingParams) :
    laws.bindingCollision → laws.msisBreakEvent :=
  laws.collisionImpliesBreak

/--
Security-reduced composition: a concrete extractor and its matching MSIS
boundary imply binding for the same selected events.
-/
theorem binding_of_msis_boundary
    (laws : LatticeReductionLaws protocolBindingParams)
    (hardness : MSISHardnessBoundary laws) :
    AjtaiBindingAssumption laws := by
  intro collision
  exact hardness.noBreak (laws.collisionImpliesBreak collision)

end Ajtai.SecurityBoundary
