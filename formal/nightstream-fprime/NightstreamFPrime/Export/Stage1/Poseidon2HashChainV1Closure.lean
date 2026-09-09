import NightstreamFPrime.Export.Stage1.PerApplicationSecurity
import NightstreamFPrime.Export.Stage1.Poseidon2HashChainV1Setup

/-!
Owns the concrete deterministic and security closure for the verifier-owned
`Poseidon2HashChainV1` package. The application, recursive fit, Ajtai setup,
package identity, verifier-context digest, and full verification-key binding
are fixed definitions. The only
external premise used by the deterministic Lean reduction is the low-norm
invertibility boundary used by SuperNeo's strong-set extraction argument.
The final quantitative claim remains conditional on the owner-recorded
Module-SIS, ChaCha20, wide-reduction, Poseidon2, and Fiat--Shamir/forking
analyses; this package has no adversary or probability model that could state
those analyses as kernel theorems.

This module does not execute or authorize a proof backend.
-/

namespace NightstreamFPrime.Export.Stage1.Poseidon2HashChainV1Closure

open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Spec

abbrev RawValues :=
  PerApplicationCanonicalAssignment.RawValues
    Poseidon2HashChainV1Package.application

def bound (raw : RawValues) : RawValues :=
  PerApplicationVerifierBoundAssignment.bind
    Poseidon2HashChainV1Package.fits
    Poseidon2HashChainV1Setup.productionSetup raw

/-- Accepted rows of the exact verifier-owned package imply its complete
HyperNova step. No production relation or key is caller-selected. -/
theorem rowsZero_implies_stepHoldsFor
    (raw : RawValues)
    (accepted : (PerApplicationFixedPoint.structuralPlan
      Poseidon2HashChainV1Package.application
      Poseidon2HashChainV1Package.fits).RowsZero
        (bound raw).assignment) :
    StepHoldsFor
      (PerApplicationFixedPoint.relation
        Poseidon2HashChainV1Package.application
        Poseidon2HashChainV1Package.fits)
      Poseidon2HashChainV1Setup.productionAjtaiKey
      (PerApplicationCanonicalPackage.verifierContextDigest
        Poseidon2HashChainV1Package.fits
        Poseidon2HashChainV1Setup.productionSetup)
      Poseidon2HashChainV1Package.application
      (PerApplicationDecodedIO.input
        Poseidon2HashChainV1Package.application
        Poseidon2HashChainV1Package.fits (bound raw))
      (PerApplicationDecodedIO.output
        Poseidon2HashChainV1Package.application (bound raw)) := by
  exact PerApplicationFixedPointSoundness.verifierBoundRowsZero_implies_stepHoldsFor
      Poseidon2HashChainV1Package.application
      Poseidon2HashChainV1Package.fits
      Poseidon2HashChainV1Setup.productionSetup raw accepted

/-- Accepted rows of the exact verifier-owned package reach the complete
base-or-recursive SuperNeo security outcome. The base branch performs no NIFS
extraction. The recursive branch uses the fixed package key and the explicit
low-norm invertibility premise. -/
theorem rowsZero_implies_base_or_securityOutcome
    (raw : RawValues)
    (theorem8 : Spec.Phi81StrongSet.LowNormInvertibility)
    (accepted : (PerApplicationFixedPoint.structuralPlan
      Poseidon2HashChainV1Package.application
      Poseidon2HashChainV1Package.fits).RowsZero
        (bound raw).assignment) :
    let input := PerApplicationDecodedIO.input
      Poseidon2HashChainV1Package.application
      Poseidon2HashChainV1Package.fits (bound raw)
    let output := PerApplicationDecodedIO.output
      Poseidon2HashChainV1Package.application (bound raw)
    StepHoldsFor
        (PerApplicationFixedPoint.relation
          Poseidon2HashChainV1Package.application
          Poseidon2HashChainV1Package.fits)
        Poseidon2HashChainV1Setup.productionAjtaiKey
        (PerApplicationCanonicalPackage.verifierContextDigest
          Poseidon2HashChainV1Package.fits
          Poseidon2HashChainV1Setup.productionSetup)
        Poseidon2HashChainV1Package.application input output ∧
      (input.iteration = 0 ∨
        (0 < input.iteration ∧
          Spec.Folding.Nifs.PaperSecurityComposition.SecurityOutcome
            (PerApplicationSecurity.canonicalKey
              Poseidon2HashChainV1Package.fits
              Poseidon2HashChainV1Setup.productionSetup)
            (PerApplicationSecurity.selectedRunning input)
            input.fresh input.nifsProof
            (PerApplicationSecurity.productionExtractionAlgebra
              Poseidon2HashChainV1Package.fits
              Poseidon2HashChainV1Setup.productionSetup)
            (PerApplicationSecurity.productionStrongSet
              Poseidon2HashChainV1Package.fits
              Poseidon2HashChainV1Setup.productionSetup theorem8))) := by
  exact PerApplicationSecurity.verifierBoundRowsZero_implies_base_or_securityOutcome
      Poseidon2HashChainV1Package.fits
      Poseidon2HashChainV1Setup.productionSetup raw theorem8 accepted

/-- The verifier fixes the hash-chain package and setup. A claimed package
with the same verification-key binding must have the same complete package
and authority, or exhibit one named Poseidon2 binding collision. Claimed
values are adversarial inputs, not verifier choices. -/
theorem expectedBindingAndRowsZero_implies_securityOrCollision
    {claimedProgram : Lifecycle.Stage1.Application.Program}
    (claimedFits : PerApplicationFixedPoint.FitsTwoPow28 claimedProgram)
    (claimedSetup : PerApplicationCanonicalPackage.CommitmentSetup
      claimedProgram)
    (raw : PerApplicationCanonicalAssignment.RawValues claimedProgram)
    (theorem8 : Spec.Phi81StrongSet.LowNormInvertibility)
    (bindingEqual : Poseidon2HashChainV1Setup.verificationKeyBinding =
      PerApplicationCanonicalPackage.verificationKeyBinding
        claimedFits claimedSetup)
    (accepted : (PerApplicationFixedPoint.structuralPlan claimedProgram
      claimedFits).RowsZero
        (PerApplicationVerifierBoundAssignment.bind
          claimedFits claimedSetup raw).assignment) :
    ((PerApplicationCanonicalPackage.sealedPackageValue
          Poseidon2HashChainV1Package.application
          Poseidon2HashChainV1Package.fits =
        PerApplicationCanonicalPackage.sealedPackageValue
          claimedProgram claimedFits ∧
      PerApplicationCanonicalPackage.authority
          Poseidon2HashChainV1Package.fits
          Poseidon2HashChainV1Setup.productionSetup =
        PerApplicationCanonicalPackage.authority claimedFits claimedSetup) ∧
      (let claimedBound := PerApplicationVerifierBoundAssignment.bind
          claimedFits claimedSetup raw
       let input := PerApplicationDecodedIO.input
          claimedProgram claimedFits claimedBound
       let output := PerApplicationDecodedIO.output claimedProgram claimedBound
       StepHoldsFor
            (PerApplicationFixedPoint.relation claimedProgram claimedFits)
            (PerApplicationCanonicalPackage.commitmentKey claimedSetup)
            (PerApplicationSecurity.verifierContextDigest
              claimedFits claimedSetup)
            claimedProgram input output ∧
          (input.iteration = 0 ∨
            (0 < input.iteration ∧
              Spec.Folding.Nifs.PaperSecurityComposition.SecurityOutcome
                (PerApplicationSecurity.canonicalKey
                  claimedFits claimedSetup)
                (PerApplicationSecurity.selectedRunning input)
                input.fresh input.nifsProof
                (PerApplicationSecurity.productionExtractionAlgebra
                  claimedFits claimedSetup)
                (PerApplicationSecurity.productionStrongSet
                  claimedFits claimedSetup theorem8))))) ∨
      PerApplicationSecurity.StructuralPackageCollision
        Poseidon2HashChainV1Package.application claimedProgram
        Poseidon2HashChainV1Package.fits claimedFits ∨
      Layout.Stage1.PiCCSSecurity.AuthorityComponentDigestCollision
        (PerApplicationCanonicalPackage.authority
          Poseidon2HashChainV1Package.fits
          Poseidon2HashChainV1Setup.productionSetup)
        (PerApplicationCanonicalPackage.authority claimedFits claimedSetup) ∨
      PerApplicationSecurity.FinalPackageBindingCollision
        Poseidon2HashChainV1Package.fits claimedFits
        Poseidon2HashChainV1Setup.productionSetup claimedSetup := by
  exact PerApplicationSecurity.verificationKeyBindingAndRowsZero_implies_securityOrCollision
      Poseidon2HashChainV1Package.fits claimedFits
      Poseidon2HashChainV1Setup.productionSetup claimedSetup raw theorem8
      bindingEqual accepted

end NightstreamFPrime.Export.Stage1.Poseidon2HashChainV1Closure
