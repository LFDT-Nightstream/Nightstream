import SuperNeo.PiCCS
import SuperNeo.ProofSystem.Types

/-!
Paper-facing Pi_CCS theorem surface.

This is the import point intended for protocol-level users. It exposes compact,
protocol-native names and forwards to the underlying formal theorems.
-/

namespace SuperNeo.ProofSystem.Folding.PiCCS

abbrev ProtocolAssumptions := SuperNeo.PiCCSProtocolAssumption
abbrev StrongProtocolAssumptions := SuperNeo.PiCCSStrongProtocolAssumption
abbrev CheckBundleProtocolAssumptions := SuperNeo.PiCCSCheckBundleProtocolAssumption
abbrev StrongCheckBundleProtocolAssumptions := SuperNeo.PiCCSStrongCheckBundleProtocolAssumption

/-- Pi_CCS soundness relation under the standard protocol-assumption bundle. -/
theorem soundness_relations
  (hProto : ProtocolAssumptions)
  {ctx : PSContext} {claim : PSClaim} {witness : PSWitness}
  {inst : PSSumcheckInstance} {tr : PSSumcheckTranscript}
  (hShape : SuperNeo.ClaimShapeValid claim)
  (hBar : SuperNeo.IsDBarMatrix ctx.bar)
  (hA : SuperNeo.IsDVec claim.a)
  (hB : SuperNeo.IsDVec claim.b)
  (hP10 : SuperNeo.p10CoreProp ctx.bar claim.a claim.b)
  (hAccepted : PSSumcheckAccepted inst tr)
  (hWitness : witness.z = claim.z)
  (hNorm : SuperNeo.normInfCoeffs witness.z < ctx.ceNormBound) :
  PSCCSRelation ctx claim ∧ PSCERelation ctx claim witness :=
  SuperNeo.piCCSStrongIR_relations_of_protocolAssumption
    hProto hShape hBar hA hB hP10 hAccepted hWitness hNorm

/-- Pi_CCS completeness: arithmetic-valid claims admit an accepted transcript. -/
theorem completeness
  (hProto : ProtocolAssumptions)
  {ctx : PSContext} {claim : PSClaim} {inst : PSSumcheckInstance}
  (hArith : SuperNeo.ClaimArithmeticValid ctx claim) :
  ∃ tr : PSSumcheckTranscript, PSSumcheckAccepted inst tr :=
  SuperNeo.piCCSStrongIR_completeness_of_protocolAssumption hProto hArith

/-- Pi_CCS soundness relation under the check-bundle protocol-assumption surface. -/
theorem checkBundle_soundness_relations
  (hProto : CheckBundleProtocolAssumptions)
  {ctx : PSContext} {claim : PSClaim} {witness : PSWitness}
  {inst : PSSumcheckInstance} {tr : PSSumcheckTranscript}
  (hShape : SuperNeo.ClaimShapeValid claim)
  (hBar : SuperNeo.IsDBarMatrix ctx.bar)
  (hA : SuperNeo.IsDVec claim.a)
  (hB : SuperNeo.IsDVec claim.b)
  (hP10 : SuperNeo.p10CoreProp ctx.bar claim.a claim.b)
  (hAccepted : PSSumcheckAccepted inst tr)
  (hWitness : witness.z = claim.z)
  (hNorm : SuperNeo.normInfCoeffs witness.z < ctx.ceNormBound) :
  PSCCSRelation ctx claim ∧ PSCERelation ctx claim witness :=
  SuperNeo.piCCSStrongIR_relations_of_checkBundleProtocolAssumption
    hProto hShape hBar hA hB hP10 hAccepted hWitness hNorm

/-- Pi_CCS completeness under the check-bundle protocol-assumption surface. -/
theorem checkBundle_completeness
  (hProto : CheckBundleProtocolAssumptions)
  {ctx : PSContext} {claim : PSClaim} {inst : PSSumcheckInstance}
  (hArith : SuperNeo.ClaimArithmeticValid ctx claim) :
  ∃ tr : PSSumcheckTranscript, PSSumcheckAccepted inst tr :=
  SuperNeo.piCCSStrongIR_completeness_of_checkBundleProtocolAssumption hProto hArith

/-- Pi_CCS soundness relation under the strong SumCheck acceptance surface. -/
theorem strong_soundness_relations
  (hProto : StrongProtocolAssumptions)
  {ctx : PSContext} {claim : PSClaim} {witness : PSWitness}
  {inst : PSSumcheckInstance} {tr : PSSumcheckTranscript}
  (hShape : SuperNeo.ClaimShapeValid claim)
  (hBar : SuperNeo.IsDBarMatrix ctx.bar)
  (hA : SuperNeo.IsDVec claim.a)
  (hB : SuperNeo.IsDVec claim.b)
  (hP10 : SuperNeo.p10CoreProp ctx.bar claim.a claim.b)
  (hAcceptedStrong : PSSumcheckAcceptedStrong inst tr)
  (hWitness : witness.z = claim.z)
  (hNorm : SuperNeo.normInfCoeffs witness.z < ctx.ceNormBound) :
  PSCCSRelation ctx claim ∧ PSCERelation ctx claim witness :=
  SuperNeo.piCCSStrongIR_relations_of_strongProtocolAssumption
    hProto hShape hBar hA hB hP10 hAcceptedStrong hWitness hNorm

/-- Pi_CCS completeness under the strong protocol-assumption surface. -/
theorem strong_completeness
  (hProto : StrongProtocolAssumptions)
  {ctx : PSContext} {claim : PSClaim} {inst : PSSumcheckInstance}
  (hArith : SuperNeo.ClaimArithmeticValid ctx claim) :
  ∃ tr : PSSumcheckTranscript, PSSumcheckAccepted inst tr :=
  SuperNeo.piCCSStrongIR_completeness_of_strongProtocolAssumption hProto hArith

/-- Pi_CCS soundness relation under the strong check-bundle surface. -/
theorem strong_checkBundle_soundness_relations
  (hProto : StrongCheckBundleProtocolAssumptions)
  {ctx : PSContext} {claim : PSClaim} {witness : PSWitness}
  {inst : PSSumcheckInstance} {tr : PSSumcheckTranscript}
  (hShape : SuperNeo.ClaimShapeValid claim)
  (hBar : SuperNeo.IsDBarMatrix ctx.bar)
  (hA : SuperNeo.IsDVec claim.a)
  (hB : SuperNeo.IsDVec claim.b)
  (hP10 : SuperNeo.p10CoreProp ctx.bar claim.a claim.b)
  (hAcceptedStrong : PSSumcheckAcceptedStrong inst tr)
  (hWitness : witness.z = claim.z)
  (hNorm : SuperNeo.normInfCoeffs witness.z < ctx.ceNormBound) :
  PSCCSRelation ctx claim ∧ PSCERelation ctx claim witness :=
  SuperNeo.piCCSStrongIR_relations_of_strongCheckBundleProtocolAssumption
    hProto hShape hBar hA hB hP10 hAcceptedStrong hWitness hNorm

/-- Pi_CCS completeness under the strong check-bundle protocol-assumption surface. -/
theorem strong_checkBundle_completeness
  (hProto : StrongCheckBundleProtocolAssumptions)
  {ctx : PSContext} {claim : PSClaim} {inst : PSSumcheckInstance}
  (hArith : SuperNeo.ClaimArithmeticValid ctx claim) :
  ∃ tr : PSSumcheckTranscript, PSSumcheckAccepted inst tr :=
  SuperNeo.piCCSStrongIR_completeness_of_strongCheckBundleProtocolAssumption hProto hArith

/-- Bridge theorem from protocol skeleton assumptions to Pi_CCS relations. -/
theorem relations_of_protocol_skeleton
  {ctx : PSContext} {claim : PSClaim} {witness : PSWitness}
  (hShape : SuperNeo.ClaimShapeValid claim)
  (hBar : SuperNeo.IsDBarMatrix ctx.bar)
  (hA : SuperNeo.IsDVec claim.a)
  (hB : SuperNeo.IsDVec claim.b)
  (hP10 : SuperNeo.p10ForClaim ctx claim)
  (hP20 : SuperNeo.p20ForClaim ctx claim)
  (hWitness : witness.z = claim.z)
  (hNorm : SuperNeo.normInfCoeffs witness.z < ctx.ceNormBound) :
  PSCCSRelation ctx claim ∧ PSCERelation ctx claim witness :=
  SuperNeo.piCCSRelations_of_protocolSkeleton
    hShape hBar hA hB hP10 hP20 hWitness hNorm

end SuperNeo.ProofSystem.Folding.PiCCS
