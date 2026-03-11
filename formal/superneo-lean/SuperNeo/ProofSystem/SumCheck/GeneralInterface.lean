import SuperNeo.ProofSystem.SumCheck.General

/-!
Interface for `SuperNeo.ProofSystem.SumCheck.General`.

Spec: `specs/ProofSystem/SumCheck/General.spec.md`
Paper: `./formal/superneo-lean/SuperNeo.pdf.md`
- Definition 6 (The sum-check protocol), lines 352–355
- Section 7.3 (Π_CCS), lines 481–548

This interface file is the typed boundary companion for the implementation.
The public surface is intentionally narrow: core SumCheck game objects,
constructive theorem-package plumbing, and the faithful prefix-dependent
full-field endpoint.
-/

namespace SuperNeo

namespace ProofSystem.SumCheck.GeneralInterface

/-- Canonical implementation module name for this interface. -/
def implementationModule : String := "SuperNeo.ProofSystem.SumCheck.General"

/-- Canonical paper source used for this module-level interface/spec pair. -/
def paperSource : String := "./formal/superneo-lean/SuperNeo.pdf.md"

/-- Paper sections used to ground this module boundary. -/
def paperAnchors : List String := ["§2.1 Sum-check reduction role", "§7.3 Interactive reduction for CCS"]

/-- Public symbol inventory extracted from the implementation module. -/
def exportedSymbolNames : List String :=
  [ "Instance"
  , "Transcript"
  , "Accepted"
  , "ClaimTrue"
  , "SoundnessAssumption"
  , "CompletenessAssumption"
  , "Assumptions"
  , "SoundnessFailureEvent"
  , "SoundnessFailureAdvantage"
  , "SoundnessFailureAdvantageBound"
  , "lundSchwartzZippelSoundnessBound"
  , "CoinProbModel"
  , "OnlineProverStrategy"
  , "SoundnessGame"
  , "SoundnessGame.transcript"
  , "SoundnessGame.acceptsOn"
  , "SoundnessGame.failureEvent"
  , "SoundnessGame.advantage"
  , "SoundnessGame.lundBoundHolds"
  , "fullFieldChallengeDomain_length"
  , "fullFieldCoinSpace_length"
  , "roundFailureUnion"
  , "roundErrorSum"
  , "roundFailureUnionCoins"
  , "pr_roundFailureUnion_le_roundErrorSum"
  , "pr_roundFailureUnionCoins_le_roundErrorSum"
  , "LundSoundnessAssumption"
  , "LundRoundBoundary"
  , "SoundnessGame.lundBoundHolds_of_roundBoundary"
  , "LundRoundBoundaryAssumption"
  , "lundSoundnessAssumption_of_roundBoundary"
  , "LundRoundBoundaryScaled"
  , "LundRoundKernel"
  , "LundRoundBoundaryScaled.of_kernel"
  , "SoundnessGame.lundBoundHolds_of_scaledRoundBoundary"
  , "LundRoundScaledBoundaryAssumption"
  , "lundSoundnessAssumption_of_scaledRoundBoundary"
  , "LundRoundKernelAssumption"
  , "lundRoundScaledBoundaryAssumption_of_kernel"
  , "lundSoundnessAssumption_of_kernel"
  , "RoundByRoundSoundnessBoundary"
  , "RoundByRoundSoundnessBoundary.totalRoundError"
  , "soundnessFailureEvent_not"
  , "soundnessFailureAdvantage_eq_zero_of_soundness"
  , "soundnessFailureAdvantageBound_of_soundness"
  , "RoundByRoundSoundnessBoundary.soundnessFailureAdvantage_le_totalRoundError"
  , "RoundByRoundSoundnessBoundary.soundnessFailureAdvantageBound"
  , "SoundnessErrorBoundary"
  , "TheoremPackage"
  , "TheoremPackage.eps"
  , "TheoremPackage.nonneg"
  , "TheoremPackage.negligible"
  , "TheoremPackage.soundness"
  , "TheoremPackage.completeness"
  , "TheoremPackage.soundnessFailureAdvantage_eq_zero"
  , "TheoremPackage.soundnessFailureAdvantageBound"
  , "theoremPackage_constructive"
  , "soundnessErrorBoundary_zero"
  , "theoremPackage_constructive_zeroError"
  , "accepted_rounds_eq"
  , "accepted_challenges_eq"
  , "accepted_fold_step"
  , "accepted_initial_round"
  , "accepted_round_sum_step"
  , "soundness"
  , "completeness"
  , "SoundnessGame.prefixGapRootSet"
  , "SoundnessGame.prefixGapEvent"
  , "SoundnessGame.prefixGapSchwartzZippelLemmas"
  , "SoundnessGame.lundBoundHolds_of_prefixGapSchwartzZippel"
  , "lundSoundnessAssumptionFullFieldAlignedPosRoundsPosDegree_prefix"
  , "lundSoundnessAssumptionFullFieldAlignedPosRounds_prefix"
  ]

/-- Assumption/boundary-oriented symbols extracted by naming convention. -/
def boundarySymbolNames : List String :=
  [ "SoundnessAssumption"
  , "CompletenessAssumption"
  , "Assumptions"
  , "SoundnessFailureAdvantageBound"
  , "LundSoundnessAssumption"
  , "SoundnessGame.lundBoundHolds"
  , "LundRoundBoundary"
  , "LundRoundBoundaryAssumption"
  , "lundSoundnessAssumption_of_roundBoundary"
  , "LundRoundBoundaryScaled"
  , "LundRoundKernel"
  , "LundRoundScaledBoundaryAssumption"
  , "lundSoundnessAssumption_of_scaledRoundBoundary"
  , "LundRoundKernelAssumption"
  , "lundSoundnessAssumption_of_kernel"
  , "SoundnessErrorBoundary"
  , "TheoremPackage"
  , "SoundnessGame.prefixGapRootSet"
  , "SoundnessGame.prefixGapEvent"
  , "SoundnessGame.prefixGapSchwartzZippelLemmas"
  , "SoundnessGame.lundBoundHolds_of_prefixGapSchwartzZippel"
  , "lundSoundnessAssumptionFullFieldAlignedPosRoundsPosDegree_prefix"
  , "lundSoundnessAssumptionFullFieldAlignedPosRounds_prefix"
  ]

/-- Interface inventory marker for the typed surface exposed here. -/
def interfaceScaffoldReady : Prop := True

theorem interfaceScaffoldReady_true : interfaceScaffoldReady := by
  trivial

end ProofSystem.SumCheck.GeneralInterface

end SuperNeo
