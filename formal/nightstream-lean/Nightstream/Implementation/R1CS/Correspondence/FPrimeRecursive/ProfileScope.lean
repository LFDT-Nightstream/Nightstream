import Nightstream.Implementation.R1CS.Correspondence.PiCcsOutputDigest.ActiveProfile
import Nightstream.Implementation.R1CS.Correspondence.PiCcsOutputDigest.Profile
import Nightstream.Implementation.R1CS.Ownership.FPrimeRecursive.FPrimeRecursiveManifest

/-!
Typed scope boundary for the generated recursive-F-prime diagnostic manifest.

Assurance tier: artifact-checked profile correspondence.

Owns: the profile recovered by the Rust manifest generator; reconciliation of
its field count with the independent serializer formula; and proof that this
three-matrix artifact cannot instantiate the thirteen-matrix selective shape.

Does not own: output authority, row satisfaction, fixed-point materialization,
the selective compiler, costs, cryptographic soundness, or row removal.

Emits constraints: no.

Authority boundary: generated counts identify which physical fixture was
measured. They never select the verifier relation and cannot become premises
for the active selective profile.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `fprime.diagnostic.profile.recovery` | generated `(15, 3)` is the direct-CCS fixture | artifact-checked | `artifactProfile_eq_diagnostic` |
| `fprime.diagnostic.profile.fields` | generated `6683` agrees with the independent formula | derived | `artifactFieldCount_eq` |
| `fprime.diagnostic.profile.separation` | diagnostic profile differs from the 13-matrix fixed-point target | derived/quarantine | `artifactProfile_ne_steadyFixedPoint` |
| `fprime.diagnostic.profile.selective` | no selective semantic shape decodes to this artifact profile | derived/quarantine | `artifactProfile_ne_selectiveShape` |
-/

namespace Nightstream.Implementation.R1CS.FPrimeRecursive.ProfileScope

/-- Physical relation scopes that must never be interchanged by an untyped
artifact count. The selective case is a semantic target, not an emitted
artifact. -/
inductive Scope where
  | diagnosticDirectCcs3
  | selectiveFixedPoint13
deriving DecidableEq, Repr

/-- A serializer profile indexed by the relation scope that owns it. -/
structure ScopedProfile (scope : Scope) where
  profile : PiCcsOutputDigest.Profile
deriving DecidableEq, Repr

namespace ScopedProfile

/-- Profile values mechanically recovered from the emitted diagnostic rows. -/
def diagnosticArtifact : ScopedProfile .diagnosticDirectCcs3 where
  profile := {
    sourceCount := FPrimeRecursiveManifest.piCcsSourceCount
    matrixCount := FPrimeRecursiveManifest.piCcsMatrixCount
  }

/-- Independently specified target profile. This definition does not claim
that a corresponding physical relation was materialized. -/
def selectiveTarget : ScopedProfile .selectiveFixedPoint13 where
  profile := PiCcsOutputDigest.Profile.steadyFixedPointThirteenMatrix

end ScopedProfile

@[simp] theorem artifactProfile_eq_diagnostic :
    ScopedProfile.diagnosticArtifact.profile =
      PiCcsOutputDigest.Profile.diagnosticThreeMatrix := by
  rfl

@[simp] theorem artifactFieldCount_eq :
    FPrimeRecursiveManifest.piCcsOutputFieldCount =
      PiCcsOutputDigest.Profile.fieldCount
        ScopedProfile.diagnosticArtifact.profile := by
  rfl

theorem artifactProfile_ne_steadyFixedPoint :
    ScopedProfile.diagnosticArtifact.profile ≠
      ScopedProfile.selectiveTarget.profile := by
  decide

/-- The generated direct-CCS fixture cannot be silently reinterpreted as the
independently specified selective fixed-point relation. -/
theorem artifactProfile_ne_selectiveShape
    {rows columns : Nat}
    (profile : PiCcsOutputDigest.ActiveProfile.Selective.Profile rows columns) :
    ScopedProfile.diagnosticArtifact.profile ≠
      PiCcsOutputDigest.Profile.ofSemanticShape
        (PiCcsOutputDigest.ActiveProfile.selectiveShape profile) := by
  intro profileEq
  have matrixCountEq := congrArg
    (fun value : PiCcsOutputDigest.Profile => value.matrixCount) profileEq
  change 3 = 13 at matrixCountEq
  omega

end Nightstream.Implementation.R1CS.FPrimeRecursive.ProfileScope
