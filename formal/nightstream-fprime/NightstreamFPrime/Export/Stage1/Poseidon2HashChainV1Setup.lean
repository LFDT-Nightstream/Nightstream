import NightstreamFPrime.Export.Stage1.Poseidon2HashChainV1Package
import NightstreamFPrime.Export.Stage1.PerApplicationCanonicalPackage
import NightstreamFPrime.Export.Stage1.Poseidon2HashChainV1SetupAuthority

/-!
Owns the verifier-selected Ajtai setup for the canonical
`Poseidon2HashChainV1` package. It derives every key dimension from the
recursive fixed point and fixes the owner-approved production seed. It does
not claim an MSIS estimate or close package conformance.
-/

namespace NightstreamFPrime.Export.Stage1.Poseidon2HashChainV1Setup

open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

def verifierRows : Nat := productionProfile.commitmentWidth

def messageColumns : Nat :=
  Phi81ColumnLayout.blockCount
    (Phi81CarrierLayout.carrierWidth
      (PerApplicationFixedPoint.logicalWidth
        Poseidon2HashChainV1Package.application))

@[simp] theorem verifierRows_eq : verifierRows = 22 := by
  rfl

@[simp] theorem carrierWidth_eq :
    Phi81CarrierLayout.carrierWidth
        (PerApplicationFixedPoint.logicalWidth
          Poseidon2HashChainV1Package.application) =
      254260620 := by
  rw [Poseidon2HashChainV1Package.logicalWidth]
  norm_num [Phi81CarrierLayout.carrierWidth, Phi81ColumnLayout.blockCount,
    ringDegree]

@[simp] theorem messageColumns_eq : messageColumns = 4708530 := by
  unfold messageColumns
  rw [carrierWidth_eq]
  norm_num [Phi81ColumnLayout.blockCount, ringDegree]

abbrev Setup :=
  PerApplicationCanonicalPackage.CommitmentSetup
    Poseidon2HashChainV1Package.application

/-- The exact indexed key accepted by the application-fixed SuperNeo
relation. No expanded key list is constructed. -/
def ajtaiKey (setup : Setup) :
    PaperAlgebra.AjtaiKey
      (logicalWidth := PerApplicationFixedPoint.logicalWidth
        Poseidon2HashChainV1Package.application)
      (publicFits := PerApplicationFixedPoint.publicFits
        Poseidon2HashChainV1Package.application) :=
  PerApplicationCanonicalPackage.commitmentKey setup

/-- Compact static words that must feed the commitment-key component of the
verifier context. -/
def authorityWords (setup : Setup) : List F :=
  PerApplicationCanonicalPackage.commitmentKeyWords setup

@[simp] theorem authorityWords_length (setup : Setup) :
    (authorityWords setup).length = 73 := by
  exact setup.authorityWords_length

def productionSeedBytes : List Nat :=
  Poseidon2HashChainV1SetupAuthority.productionSeedBytes

@[simp] theorem productionSeedBytes_length : productionSeedBytes.length = 32 := by
  rfl

theorem productionSeedBytes_canonical :
    ∀ byte, byte ∈ productionSeedBytes → byte < 256 := by
  exact Poseidon2HashChainV1SetupAuthority.productionSeed.canonical

def productionSeed : AjtaiSetupV1.Seed :=
  Poseidon2HashChainV1SetupAuthority.productionSeed

/-- The sole production commitment setup for this application package. -/
def productionSetup : Setup where
  seed := productionSeed

/-- The semantic key used by `ProductionKey.key`. It is the lazy indexed
projection of `productionSetup`; no key matrix is materialized. -/
def productionAjtaiKey := ajtaiKey productionSetup

/-- The exact compact setup authority hashed by the verifier context. -/
def productionAuthorityWords : List F := authorityWords productionSetup

@[simp] theorem productionAuthorityWords_length :
    productionAuthorityWords.length = 73 := by
  exact authorityWords_length productionSetup

def directProductionAuthorityNats : List Nat :=
  Poseidon2HashChainV1SetupAuthority.authorityNats

theorem directProductionAuthorityNats_eq :
    directProductionAuthorityNats = productionSetup.authorityNats := by
  change directProductionAuthorityNats =
    [AjtaiSetupV1.setupIdBytes.length] ++ AjtaiSetupV1.setupIdBytes ++
      [verifierRows, messageColumns, productionSeedBytes.length] ++
        productionSeedBytes
  rw [verifierRows_eq, messageColumns_eq, productionSeedBytes_length]
  rfl

def directProductionAuthorityWords : List F :=
  directProductionAuthorityNats.map Poseidon2.ofNat

theorem directProductionAuthorityWords_eq :
    directProductionAuthorityWords = productionAuthorityWords := by
  unfold directProductionAuthorityWords productionAuthorityWords
    authorityWords
  rw [directProductionAuthorityNats_eq]
  rfl

def packageIdentity (_delay : Unit := ()) :
    Lifecycle.VerifierContext.Digest4 :=
  PerApplicationCanonicalPackage.packageIdentity
    Poseidon2HashChainV1Package.fits productionSetup

def verifierContextDescriptor (_delay : Unit := ()) :
    Lifecycle.VerifierContext.Descriptor :=
  PerApplicationCanonicalPackage.verifierContextDescriptor
    Poseidon2HashChainV1Package.fits productionSetup

def verificationKeyBinding (_delay : Unit := ()) :
    Lifecycle.Stage1.VerificationKey.Binding :=
  PerApplicationCanonicalPackage.verificationKeyBinding
    Poseidon2HashChainV1Package.fits productionSetup

@[simp] theorem verificationKeyBinding_packageIdentity :
    verificationKeyBinding.packageIdentity = packageIdentity := by
  rfl

@[simp] theorem verificationKeyBinding_context :
    verificationKeyBinding.context = verifierContextDescriptor := by
  rfl

theorem verificationKeyDigest_recomputed :
    verificationKeyBinding.digest =
      (Lifecycle.VerifierContext.Digest4.ofList
        (Poseidon2.hash verificationKeyBinding.serialize)).toList := by
  exact Lifecycle.Stage1.VerificationKey.Binding.digest_recomputed _

end NightstreamFPrime.Export.Stage1.Poseidon2HashChainV1Setup
