import SuperNeo.ProofSystem.Lattice

/-!
Contract interface for `SuperNeo.ProofSystem.Lattice`.

Spec: `specs/ProofSystem/Lattice.spec.md`
Paper: `./formal/superneo-lean/SuperNeo.pdf.md`
Anchors:
- Definition 4 (binding / relaxed binding), lines 305-315.
- Definition 16 (MSIS), lines 743-744.
- Definition 18 (Ajtai commitment), lines 753-756.
- Theorem 2 boundary package shape, lines 319-321.
-/

namespace SuperNeo
namespace ProofSystem.LatticeInterface

noncomputable section

/-! ## Core Parameters and Shapes -/

abbrev AjtaiParams := SuperNeo.ProofSystem.AjtaiParams
abbrev AjtaiParams_kappa := SuperNeo.ProofSystem.AjtaiParams.kappa
abbrev AjtaiParams_msgLen := SuperNeo.ProofSystem.AjtaiParams.msgLen
abbrev AjtaiParams_matrixFlatLen := SuperNeo.ProofSystem.AjtaiParams.matrixFlatLen
abbrev AjtaiParams_commitmentLen := SuperNeo.ProofSystem.AjtaiParams.commitmentLen
abbrev AjtaiParams_payloadLen := SuperNeo.ProofSystem.AjtaiParams.payloadLen
abbrev AjtaiParams_msisNormBound := SuperNeo.ProofSystem.AjtaiParams.msisNormBound
abbrev AjtaiParams_SideConditions := SuperNeo.ProofSystem.AjtaiParams.SideConditions

abbrev Commitment := SuperNeo.ProofSystem.Commitment
abbrev Opening := SuperNeo.ProofSystem.Opening
abbrev BindingCollision := SuperNeo.ProofSystem.BindingCollision
abbrev RelaxedBindingCollision := SuperNeo.ProofSystem.RelaxedBindingCollision

/-! ## Ring/Vector Surfaces -/

abbrev normInfVec := SuperNeo.ProofSystem.normInfVec
abbrev dotRq := SuperNeo.ProofSystem.dotRq
abbrev matRow := SuperNeo.ProofSystem.matRow
abbrev matVecMul := SuperNeo.ProofSystem.matVecMul
abbrev smulVec := SuperNeo.ProofSystem.smulVec
abbrev subRq := SuperNeo.ProofSystem.subRq
abbrev zeroVec := SuperNeo.ProofSystem.zeroVec
abbrev subVec := SuperNeo.ProofSystem.subVec

abbrev subRq_self := SuperNeo.ProofSystem.subRq_self
abbrev subVec_size := SuperNeo.ProofSystem.subVec_size
abbrev subVec_self := SuperNeo.ProofSystem.subVec_self
abbrev smulVec_size := SuperNeo.ProofSystem.smulVec_size
abbrev matVecMul_size := SuperNeo.ProofSystem.matVecMul_size

/-! ## Commitment/Openings and MSIS -/

abbrev Commitment_WellFormed := SuperNeo.ProofSystem.Commitment.WellFormed
abbrev Commitment_ppMatrixFlat := SuperNeo.ProofSystem.Commitment.ppMatrixFlat
abbrev Commitment_valueVec := SuperNeo.ProofSystem.Commitment.valueVec
abbrev Opening_WellFormed := SuperNeo.ProofSystem.Opening.WellFormed
abbrev Opening_NormSound := SuperNeo.ProofSystem.Opening.NormSound

abbrev opensTo := SuperNeo.ProofSystem.opensTo
abbrev opensToRelaxed := SuperNeo.ProofSystem.opensToRelaxed

abbrev MSISChallenge := SuperNeo.ProofSystem.MSISChallenge
abbrev MSISChallenge_WellFormed := SuperNeo.ProofSystem.MSISChallenge.WellFormed
abbrev MSISSolution := SuperNeo.ProofSystem.MSISSolution
abbrev MSISBreakEvent := SuperNeo.ProofSystem.MSISBreakEvent
abbrev MSISGame := SuperNeo.ProofSystem.MSISGame
abbrev canonicalMSISGame := SuperNeo.ProofSystem.canonicalMSISGame
abbrev MSISAdvantage {params : AjtaiParams} := SuperNeo.ProofSystem.MSISAdvantage (params := params)
abbrev MSISAdvantageBound := SuperNeo.ProofSystem.MSISAdvantageBound
abbrev MSISHardnessAssumption := SuperNeo.ProofSystem.MSISHardnessAssumption
abbrev MSISHardnessBoundary := SuperNeo.ProofSystem.MSISHardnessBoundary
def MSISHardnessBoundary_hardness
  {params : AjtaiParams}
  (h : MSISHardnessBoundary params) : MSISHardnessAssumption params :=
  h.hardness

def MSISHardnessBoundary_hardnessFromFields
  {params : AjtaiParams}
  (h : MSISHardnessBoundary params) : MSISHardnessAssumption params :=
  h.hardnessFromFields

/-! ## Ajtai Security Boundary Surfaces -/

abbrev AjtaiBindingAssumption := SuperNeo.ProofSystem.AjtaiBindingAssumption
abbrev AjtaiRelaxedBindingAssumption := SuperNeo.ProofSystem.AjtaiRelaxedBindingAssumption
abbrev AjtaiBindingGame := SuperNeo.ProofSystem.AjtaiBindingGame
abbrev canonicalAjtaiBindingGame := SuperNeo.ProofSystem.canonicalAjtaiBindingGame
abbrev AjtaiBindingAdvantage {params : AjtaiParams} := SuperNeo.ProofSystem.AjtaiBindingAdvantage (params := params)
abbrev AjtaiBindingAdvantageBound := SuperNeo.ProofSystem.AjtaiBindingAdvantageBound
abbrev AjtaiRelaxedBindingGame := SuperNeo.ProofSystem.AjtaiRelaxedBindingGame
abbrev canonicalAjtaiRelaxedBindingGame := SuperNeo.ProofSystem.canonicalAjtaiRelaxedBindingGame
abbrev AjtaiRelaxedBindingAdvantage {params : AjtaiParams} :=
  SuperNeo.ProofSystem.AjtaiRelaxedBindingAdvantage (params := params)
abbrev AjtaiRelaxedBindingAdvantageBound := SuperNeo.ProofSystem.AjtaiRelaxedBindingAdvantageBound
abbrev AjtaiBindingBoundary := SuperNeo.ProofSystem.AjtaiBindingBoundary
abbrev AjtaiRelaxedBindingBoundary := SuperNeo.ProofSystem.AjtaiRelaxedBindingBoundary

end
end ProofSystem.LatticeInterface
end SuperNeo
