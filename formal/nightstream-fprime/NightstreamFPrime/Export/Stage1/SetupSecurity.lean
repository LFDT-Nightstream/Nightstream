import NightstreamFPrime.Export.Stage1.SetupBinding
import NightstreamFPrime.Export.Stage1.PerApplicationSecurity

/-! The approved Nightstream-specific premise is public-seed MSIS hardness
for the selected setup, not SuperNeo's uniform-matrix assumption. This
package freezes `productionSeed`; its assumption concerns that specific
matrix. No uniform-setup estimate is inherited. The complete premise is in
`docs/reviews/nightstream-fprime-requirements/PUBLIC_SEED_MSIS_ASSUMPTION.md`.

This theorem identifies the verifier-selected setup from its expected package
identity, or exposes a named Poseidon2 collision. A prover-supplied expected
identity is not verifier authority. The final Stage 1 security claim must say
"secure assuming public-seed MSIS hardness for the selected setup" and retain
the other explicit security premises. -/

namespace NightstreamFPrime.Export.Stage1.Poseidon2HashChainV1Setup

open NightstreamFPrime.Spec
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Layout.Stage1.PiCCSSecurity

/-- A different seed cannot select a different indexed matrix under the
verifier's expected package identity without a named Poseidon2 collision.
Dimensions, modulus, profile and algorithm version remain those of the
selected package and the complete setup descriptor. -/
theorem packageIdentity_identifies_selected_setup_or_collision
    (candidate : Setup)
    (sameIdentity : PerApplicationCanonicalPackage.packageIdentity
      Poseidon2HashChainV1Package.fits candidate = packageIdentity) :
    candidate = productionSetup ∨
      AuthorityComponentDigestCollision
        (PerApplicationCanonicalPackage.authority Poseidon2HashChainV1Package.fits candidate)
        (PerApplicationCanonicalPackage.authority Poseidon2HashChainV1Package.fits productionSetup) ∨
      PerApplicationSecurity.FinalPackageBindingCollision
        Poseidon2HashChainV1Package.fits Poseidon2HashChainV1Package.fits
        candidate productionSetup := by
  have result := PerApplicationSecurity.packageIdentity_identifies_package_authority_or_collision
    Poseidon2HashChainV1Package.fits Poseidon2HashChainV1Package.fits
    candidate productionSetup sameIdentity
  rcases result with same | structural | component | finalCollision
  · left
    apply authorityWords_injective
    exact congrArg VerifierContext.Authority.commitmentKeyWords same.2
  · exact False.elim (structural.1 rfl)
  · exact Or.inr (Or.inl component)
  · exact Or.inr (Or.inr finalCollision)

end NightstreamFPrime.Export.Stage1.Poseidon2HashChainV1Setup
