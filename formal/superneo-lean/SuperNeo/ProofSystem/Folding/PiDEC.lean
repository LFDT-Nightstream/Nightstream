import SuperNeo.PiDEC
import SuperNeo.ProofSystem.Types
import SuperNeo.ProofSystem.Folding.PiRLC

/-!
Paper-facing Pi_DEC theorem surface.

This module exposes the final upgrade step from relaxed CE outputs to CE-valid
final targets, with optional invertibility witness extraction.
-/

namespace SuperNeo.ProofSystem.Folding.PiDEC

abbrev UpgradeAssumptions := SuperNeo.PiDECUpgradeAssumption

def FinalTarget (ctx : PSContext) (claim : PSClaim) (wit : PSWitness) : Prop :=
  SuperNeo.PiDECFinalTarget ctx claim wit

/-- Pi_DEC final target from the upgrade assumption and relaxed CE relation. -/
theorem final_of_assumption
  (hDec : UpgradeAssumptions)
  {ctx : PSContext} {claim : PSClaim} {wit : PSWitness}
  (hRelaxed : PSCERelaxedRelation ctx claim wit) :
  FinalTarget ctx claim wit :=
  SuperNeo.piDECFinalTarget_of_assumption hDec hRelaxed

/-- Project CE relation from the Pi_DEC final target path. -/
theorem ce_of_assumption
  (hDec : UpgradeAssumptions)
  {ctx : PSContext} {claim : PSClaim} {wit : PSWitness}
  (hRelaxed : PSCERelaxedRelation ctx claim wit) :
  PSCERelation ctx claim wit :=
  SuperNeo.piDEC_ceRelation_of_assumption hDec hRelaxed

/-- Project decomposition property from the Pi_DEC final target path. -/
theorem decomp_of_assumption
  (hDec : UpgradeAssumptions)
  {ctx : PSContext} {claim : PSClaim} {wit : PSWitness}
  (hRelaxed : PSCERelaxedRelation ctx claim wit) :
  SuperNeo.p20DecompProp claim.zDecomp ctx.bSplit ctx.kSplit :=
  SuperNeo.piDECDecomp_of_assumption hDec hRelaxed

/-- Invertibility witness from CE relation. -/
theorem invertibility_witness_of_ce
  {ctx : PSContext} {claim : PSClaim} {wit : PSWitness}
  (hCE : PSCERelation ctx claim wit) :
  ∃ deltaInv : SuperNeo.Coeffs, SuperNeo.mulRq claim.invDelta deltaInv = SuperNeo.oneRq :=
  SuperNeo.piDECInvertibilityWitness_of_ceRelation hCE

/-- Pi_DEC final target with explicit invertibility witness extraction. -/
theorem final_with_invertibility
  (hDec : UpgradeAssumptions)
  {ctx : PSContext} {claim : PSClaim} {wit : PSWitness}
  (hRelaxed : PSCERelaxedRelation ctx claim wit) :
  ∃ deltaInv : SuperNeo.Coeffs,
    SuperNeo.mulRq claim.invDelta deltaInv = SuperNeo.oneRq ∧ FinalTarget ctx claim wit :=
  SuperNeo.piDECFinal_with_invertibility_of_assumption hDec hRelaxed

/-- Composition hook: Pi_RLC weak output upgraded by Pi_DEC. -/
theorem final_of_piRLC
  (hWeak : SuperNeo.ProofSystem.Folding.PiRLC.WeakAssumptions)
  (hDec : UpgradeAssumptions)
  {ctx : PSContext}
  {claimL claimR claimOut : PSClaim}
  {witL witR witOut : PSWitness}
  (hLeft : PSCERelation ctx claimL witL)
  (hRight : PSCERelation ctx claimR witR) :
  FinalTarget ctx claimOut witOut :=
  SuperNeo.piDEC_of_piRLC_assumptions hWeak hDec hLeft hRight

end SuperNeo.ProofSystem.Folding.PiDEC
