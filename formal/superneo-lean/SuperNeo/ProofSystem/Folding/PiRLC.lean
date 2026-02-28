import SuperNeo.PiRLC
import SuperNeo.ProofSystem.Types
import SuperNeo.ProofSystem.Folding.PiCCS

/-!
Paper-facing Pi_RLC theorem surface.

This module exposes weak-reduction interfaces with compact protocol-native names
and forwards directly to the underlying Pi_RLC theorems.
-/

namespace SuperNeo.ProofSystem.Folding.PiRLC

abbrev WeakAssumptions := SuperNeo.PiRLCWeakRelationAssumption
abbrev ProtocolAssumptions := SuperNeo.PiRLCProtocolAssumption

/-- Weak Pi_RLC relation: two CE inputs fold to one relaxed CE output. -/
theorem weak_relaxed
  (hWeak : WeakAssumptions)
  {ctx : PSContext}
  {claimL claimR claimOut : PSClaim}
  {witL witR witOut : PSWitness}
  (hLeft : PSCERelation ctx claimL witL)
  (hRight : PSCERelation ctx claimR witR) :
  PSCERelaxedRelation ctx claimOut witOut :=
  SuperNeo.piRLCWeakIR_relaxed_of_assumption hWeak hLeft hRight

/-- Weak Pi_RLC relation upgraded to CE using an explicit output norm bound. -/
theorem weak_ce_of_norm
  (hWeak : WeakAssumptions)
  {ctx : PSContext}
  {claimL claimR claimOut : PSClaim}
  {witL witR witOut : PSWitness}
  (hLeft : PSCERelation ctx claimL witL)
  (hRight : PSCERelation ctx claimR witR)
  (hNormOut : SuperNeo.normInfCoeffs witOut.z < ctx.ceNormBound) :
  PSCERelation ctx claimOut witOut :=
  SuperNeo.piRLCWeakIR_ce_of_assumption hWeak hLeft hRight hNormOut

/-- Protocol-bundle variant: output CE follows directly via norm rebind boundary. -/
theorem weak_ce_of_protocol
  (hProto : ProtocolAssumptions)
  {ctx : PSContext}
  {claimL claimR claimOut : PSClaim}
  {witL witR witOut : PSWitness}
  (hLeft : PSCERelation ctx claimL witL)
  (hRight : PSCERelation ctx claimR witR) :
  PSCERelation ctx claimOut witOut :=
  SuperNeo.piRLCWeakIR_ce_of_protocolAssumption hProto hLeft hRight

/-- CEValid-input variant producing a relaxed CE output. -/
theorem weak_relaxed_of_ceValid_pair
  (hWeak : WeakAssumptions)
  {ctx : PSContext}
  {claimL claimR claimOut : PSClaim}
  {witL witR witOut : PSWitness}
  (hLeft : PSCEValid ctx claimL witL)
  (hRight : PSCEValid ctx claimR witR) :
  PSCERelaxedRelation ctx claimOut witOut :=
  SuperNeo.piRLCWeakIR_relaxed_of_ceValid_pair hWeak hLeft hRight

/-- Composition helper: obtain relaxed Pi_RLC output using Pi_CCS assumptions. -/
theorem weak_relaxed_of_piCCS
  (hCCS : SuperNeo.ProofSystem.Folding.PiCCS.ProtocolAssumptions)
  (hWeak : WeakAssumptions)
  {ctx : PSContext}
  {claimL claimR claimOut : PSClaim}
  {witL witR witOut : PSWitness}
  {instL instR : PSSumcheckInstance}
  {trL trR : PSSumcheckTranscript}
  (hShapeL : SuperNeo.ClaimShapeValid claimL)
  (hShapeR : SuperNeo.ClaimShapeValid claimR)
  (hBar : SuperNeo.IsDBarMatrix ctx.bar)
  (hAL : SuperNeo.IsDVec claimL.a)
  (hBL : SuperNeo.IsDVec claimL.b)
  (hAR : SuperNeo.IsDVec claimR.a)
  (hBR : SuperNeo.IsDVec claimR.b)
  (hP10L : SuperNeo.p10CoreProp ctx.bar claimL.a claimL.b)
  (hP10R : SuperNeo.p10CoreProp ctx.bar claimR.a claimR.b)
  (hAcceptedL : PSSumcheckAccepted instL trL)
  (hAcceptedR : PSSumcheckAccepted instR trR)
  (hWitnessL : witL.z = claimL.z)
  (hWitnessR : witR.z = claimR.z)
  (hNormL : SuperNeo.normInfCoeffs witL.z < ctx.ceNormBound)
  (hNormR : SuperNeo.normInfCoeffs witR.z < ctx.ceNormBound) :
  PSCERelaxedRelation ctx claimOut witOut :=
  SuperNeo.piRLCWeakIR_relaxed_of_piCCS_assumptions
    hCCS.1.1 hCCS.2.1 hWeak
    hShapeL hShapeR hBar hAL hBL hAR hBR hP10L hP10R
    hAcceptedL hAcceptedR hWitnessL hWitnessR hNormL hNormR

end SuperNeo.ProofSystem.Folding.PiRLC
