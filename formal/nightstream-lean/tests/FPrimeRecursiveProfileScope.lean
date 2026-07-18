import Nightstream.Implementation.R1CS.Correspondence.FPrimeRecursive.ProfileScope

/-! Focused theorem-shape checks for diagnostic/fixed-point separation. -/

open Nightstream.Implementation.R1CS

#check FPrimeRecursive.ProfileScope.artifactProfile_eq_diagnostic
#check FPrimeRecursive.ProfileScope.artifactFieldCount_eq
#check FPrimeRecursive.ProfileScope.artifactProfile_ne_steadyFixedPoint
#check FPrimeRecursive.ProfileScope.artifactProfile_ne_selectiveShape

#guard FPrimeRecursive.ProfileScope.ScopedProfile.diagnosticArtifact.profile ==
  PiCcsOutputDigest.Profile.diagnosticThreeMatrix
#guard FPrimeRecursive.ProfileScope.ScopedProfile.diagnosticArtifact.profile !=
  FPrimeRecursive.ProfileScope.ScopedProfile.selectiveTarget.profile
#guard FPrimeRecursiveManifest.piCcsOutputFieldCount == 6683
