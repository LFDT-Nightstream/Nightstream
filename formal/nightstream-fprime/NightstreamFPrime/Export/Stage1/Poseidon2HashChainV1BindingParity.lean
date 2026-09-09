import NightstreamFPrime.Export.Codec
import NightstreamFPrime.Export.Stage1.Poseidon2HashChainV1Package
import NightstreamFPrime.Export.Stage1.Poseidon2HashChainV1Setup
import NightstreamFPrime.Export.Stage1.PerApplicationVerifierContextStreaming

/-!
Owns canonical-source and component-replay fixtures for the
`Poseidon2HashChainV1` verification-key binding. Canonical mode derives its
authority from the current Lean package. Component replay remains conditional
on separate checks of the supplied component digests.
-/

namespace NightstreamFPrime.Export.Stage1.Poseidon2HashChainV1BindingParity

open NightstreamFPrime.Export.Codec
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Spec

def schema : Nat := 1

private def packageIdentityFromParts
    (structural : VerifierContext.Digest4)
    (context : VerifierContext.Descriptor) : VerifierContext.Digest4 :=
  VerifierContext.Digest4.ofList
    (Poseidon2.hash
      (PerApplicationCanonicalPackage.packageIdentityDomain ++
        VerifierContext.framed structural.toList ++
        VerifierContext.framed context.serialize))

/-- Construct the complete binding from separately replayed component digests.
This bounded constructor stays local to the parity emitter. Each digest is
evidence only after a consumer checks its complete authority stream. -/
private def bindingFromComponentDigests
    (structural relation application nifsKey commitmentKey :
      VerifierContext.Digest4) :
    Lifecycle.Stage1.VerificationKey.Binding :=
  let context : VerifierContext.Descriptor :=
    { relation, application, nifsKey, commitmentKey }
  {
    packageIdentity := packageIdentityFromParts structural context
    context := context
  }

theorem bindingFromComponentDigests_eq_production
    (structural relation application nifsKey commitmentKey :
      VerifierContext.Digest4)
    (structuralCanonical : structural =
      PerApplicationCanonicalPackage.structuralPackageIdentity
        Poseidon2HashChainV1Package.application
        Poseidon2HashChainV1Package.fits)
    (relationCanonical : relation =
      VerifierContext.componentDigest 1
        (PerApplicationCanonicalPackage.relationAuthorityWords
          Poseidon2HashChainV1Package.application
          Poseidon2HashChainV1Package.fits))
    (applicationCanonical : application =
      VerifierContext.componentDigest 2
        (PerApplicationCanonicalPackage.applicationAuthorityWords
          Poseidon2HashChainV1Package.application))
    (nifsKeyCanonical : nifsKey =
      VerifierContext.componentDigest 3
        (PerApplicationCanonicalPackage.nifsKeyWords
          Poseidon2HashChainV1Package.fits
          Poseidon2HashChainV1Setup.productionSetup))
    (commitmentKeyCanonical : commitmentKey =
      VerifierContext.componentDigest 4
        (PerApplicationCanonicalPackage.commitmentKeyWords
          Poseidon2HashChainV1Setup.productionSetup)) :
    bindingFromComponentDigests structural relation application nifsKey
        commitmentKey =
      Poseidon2HashChainV1Setup.verificationKeyBinding := by
  subst structural
  subst relation
  subst application
  subst nifsKey
  subst commitmentKey
  rfl

def fieldWordsValue (values : List F) : Value :=
  .array (values.map fun value => .atom value.val)

private def bindingValue (structural : VerifierContext.Digest4)
    (currentBinding : Lifecycle.Stage1.VerificationKey.Binding) : Value :=
  .array [
    .atom schema,
    fieldWordsValue structural.toList,
    fieldWordsValue currentBinding.packageIdentity.toList,
    fieldWordsValue currentBinding.context.serialize,
    fieldWordsValue currentBinding.serialize,
    fieldWordsValue currentBinding.digest]

/-- Schema 1 fields: structural package identity, final package identity,
serialized verifier-context descriptor, complete binding preimage, and final
verification-key digest. This mode replays supplied component digests. -/
def parityValue (structural relation application nifsKey commitmentKey :
    VerifierContext.Digest4) : Value :=
  bindingValue structural
    (bindingFromComponentDigests structural relation application nifsKey commitmentKey)

def parityValueIO (structural relation application nifsKey commitmentKey :
    VerifierContext.Digest4) : IO Value := do
  pure (parityValue structural relation application nifsKey commitmentKey)

private def canonicalValues (_delay : Unit) :
    VerifierContext.Digest4 × Lifecycle.Stage1.VerificationKey.Binding :=
  let structural := PerApplicationStreamingIdentity.structuralPackageIdentityDirect
    Poseidon2HashChainV1Package.application Poseidon2HashChainV1Package.fits
  let applicationDigest :=
    PerApplicationVerifierContextStreaming.applicationComponentDigestDirect
      (PerApplicationPackage.directApplicationPlan Poseidon2HashChainV1Package.application)
  let currentBinding :=
    PerApplicationCanonicalPackage.verificationKeyBindingFromStructuralAndApplicationDigest
      Poseidon2HashChainV1Package.fits Poseidon2HashChainV1Setup.productionSetup
      structural applicationDigest
  (structural, currentBinding)

/-- Canonical mode derives the complete binding from Lean's package and
authority definitions. No caller supplies a digest or a canonicality premise. -/
theorem canonicalValues_eq_production :
    canonicalValues () =
      (PerApplicationCanonicalPackage.structuralPackageIdentity
        Poseidon2HashChainV1Package.application Poseidon2HashChainV1Package.fits,
        Poseidon2HashChainV1Setup.verificationKeyBinding ()) := by
  have applicationDigest_eq :
      PerApplicationVerifierContextStreaming.applicationComponentDigestDirect
          (PerApplicationPackage.directApplicationPlan Poseidon2HashChainV1Package.application) =
        VerifierContext.componentDigest 2
          (PerApplicationCanonicalPackage.applicationAuthorityWords
            Poseidon2HashChainV1Package.application) := by
    rw [PerApplicationVerifierContextStreaming.applicationComponentDigestDirect_eq,
      PerApplicationPackage.directApplicationPlan_eq_applicationPlan]
    rfl
  simp only [canonicalValues,
    PerApplicationStreamingIdentity.structuralPackageIdentityDirect_eq,
    applicationDigest_eq,
    PerApplicationCanonicalPackage.verificationKeyBindingFromStructuralAndApplicationDigest_canonical,
    PerApplicationCanonicalPackage.verificationKeyBindingFromStructural_canonical]
  rfl

/-- Compute the structural package identity once. The existing structural
constructor derives and shares one authority/context value for the package
identity, binding preimage, and binding digest. -/
def canonicalParityValueIO (_delay : Unit) : IO Value := do
  let cached := canonicalValues ()
  pure (bindingValue cached.1 cached.2)

end NightstreamFPrime.Export.Stage1.Poseidon2HashChainV1BindingParity
