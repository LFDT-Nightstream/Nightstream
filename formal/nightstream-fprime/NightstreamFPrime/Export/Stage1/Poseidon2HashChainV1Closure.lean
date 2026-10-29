import NightstreamFPrime.Export.Stage1.PerApplicationSecurity
import NightstreamFPrime.Export.Stage1.Poseidon2HashChainV1Setup

/-! Conditional on the owner-recorded Module-SIS, wide-reduction, Poseidon2,
Fiat--Shamir/forking and SHAKE128 random-oracle analyses. -/

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

theorem rowsZero_implies_base_or_securityOutcome
    (raw : RawValues)
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
              Poseidon2HashChainV1Setup.productionSetup))) := by
  exact PerApplicationSecurity.verifierBoundRowsZero_implies_base_or_securityOutcome
      Poseidon2HashChainV1Package.fits
      Poseidon2HashChainV1Setup.productionSetup raw accepted

theorem expectedBindingAndRowsZero_implies_securityOrCollision
    {claimedProgram : Lifecycle.Stage1.Application.Program}
    (claimedFits : PerApplicationFixedPoint.FitsTwoPow28 claimedProgram)
    (claimedSetup : PerApplicationCanonicalPackage.CommitmentSetup
      claimedProgram)
    (raw : PerApplicationCanonicalAssignment.RawValues claimedProgram)
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
                  claimedFits claimedSetup))))) ∨
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
      Poseidon2HashChainV1Setup.productionSetup claimedSetup raw
      bindingEqual accepted

end NightstreamFPrime.Export.Stage1.Poseidon2HashChainV1Closure
