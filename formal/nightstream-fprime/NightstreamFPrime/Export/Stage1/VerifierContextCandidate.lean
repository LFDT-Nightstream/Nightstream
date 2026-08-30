import NightstreamFPrime.Lifecycle.PaperAlgebra
import NightstreamFPrime.Lifecycle.VerifierContext

/-!
Owns the small verifier-context recipe for the current Stage 1 package cut.

This module contains no package rows or layout imports. The package-bound
module separately proves that these candidate identity words equal the
canonical package identity. The final Stage 1 integration must replace this
prefix width and rerun every applicable gate on the final package identity.
-/

namespace NightstreamFPrime.Export.Stage1.VerifierContext

open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

/-- Exact logical width of the Pilot + PiCCS + PiRLC + PiDEC package cut. -/
def candidateLogicalWidth : Nat := 27420587

def candidatePublicFits : ringDegree * publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth candidateLogicalWidth := by
  apply Nat.le_trans (m := candidateLogicalWidth)
  · norm_num [candidateLogicalWidth, ringDegree, publicRingColumns]
  · exact Phi81CarrierLayout.logicalWidth_le_carrierWidth
      candidateLogicalWidth

/-- Last validated verifier-owned package identity. Do not change this pin
until the current candidate passes every identity-change gate. -/
def expectedPackageIdentity : Lifecycle.VerifierContext.Digest4 where
  c0 := ⟨5326948389888638380, by norm_num [F, goldilocksModulus]⟩
  c1 := ⟨15945253772729055182, by norm_num [F, goldilocksModulus]⟩
  c2 := ⟨12038831075978321435, by norm_num [F, goldilocksModulus]⟩
  c3 := ⟨4066786242110063495, by norm_num [F, goldilocksModulus]⟩

def packageIdentityWords : List F :=
  expectedPackageIdentity.toList

/-- Unpinned identity candidate recomputed from the current canonical package.
It is used only to produce pre-pin conformance fixtures. -/
def candidatePackageIdentity : Lifecycle.VerifierContext.Digest4 where
  c0 := ⟨5326948389888638380, by norm_num [F, goldilocksModulus]⟩
  c1 := ⟨15945253772729055182, by norm_num [F, goldilocksModulus]⟩
  c2 := ⟨12038831075978321435, by norm_num [F, goldilocksModulus]⟩
  c3 := ⟨4066786242110063495, by norm_num [F, goldilocksModulus]⟩

def candidatePackageIdentityWords : List F :=
  candidatePackageIdentity.toList

/-- Domain of the compact NIFS-key authority description. -/
def nifsKeyDomain : List F :=
  ([78, 105, 103, 104, 116, 115, 116, 114, 101, 97, 109, 47,
    70, 80, 114, 105, 109, 101, 47, 110, 105, 102, 115, 45,
    107, 101, 121, 47, 118, 49, 95, 49] : List Nat).map Poseidon2.ofNat

/-- Package-bound NIFS descriptor for explicit identity words and one
verifier-owned commitment setup serialization. -/
def nifsKeyWordsFor (identityWords commitmentKeyWords : List F) : List F :=
  nifsKeyDomain ++
    Lifecycle.VerifierContext.framed Lifecycle.VerifierContext.profileWords ++
    Lifecycle.VerifierContext.framed Lifecycle.VerifierContext.scheduleWords ++
    Lifecycle.VerifierContext.framed identityWords ++
    Lifecycle.VerifierContext.framed
      (Lifecycle.VerifierContext.componentDigest 4 commitmentKeyWords).toList

def nifsKeyWords (commitmentKeyWords : List F) : List F :=
  nifsKeyWordsFor packageIdentityWords commitmentKeyWords

def authorityForIdentity (identityWords commitmentKeyWords : List F) :
    Lifecycle.VerifierContext.Authority where
  relationWords := identityWords
  applicationWords := identityWords
  nifsKeyWords := nifsKeyWordsFor identityWords commitmentKeyWords
  commitmentKeyWords := commitmentKeyWords

/-- Canonical context authority for the selected package. -/
def authority (commitmentKeyWords : List F) :
    Lifecycle.VerifierContext.Authority :=
  authorityForIdentity packageIdentityWords commitmentKeyWords

@[simp] theorem authority_relationWords (commitmentKeyWords : List F) :
    (authority commitmentKeyWords).relationWords = packageIdentityWords := by
  rfl

@[simp] theorem authority_applicationWords (commitmentKeyWords : List F) :
    (authority commitmentKeyWords).applicationWords = packageIdentityWords := by
  rfl

@[simp] theorem authority_nifsKeyWords (commitmentKeyWords : List F) :
    (authority commitmentKeyWords).nifsKeyWords = nifsKeyWords commitmentKeyWords := by
  rfl

@[simp] theorem authority_commitmentKeyWords (commitmentKeyWords : List F) :
    (authority commitmentKeyWords).commitmentKeyWords = commitmentKeyWords := by
  rfl

theorem descriptor_eq (commitmentKeyWords : List F) :
    Lifecycle.VerifierContext.descriptor (authority commitmentKeyWords) = {
      relation := Lifecycle.VerifierContext.componentDigest 1
        packageIdentityWords
      application := Lifecycle.VerifierContext.componentDigest 2
        packageIdentityWords
      nifsKey := Lifecycle.VerifierContext.componentDigest 3
        (nifsKeyWords commitmentKeyWords)
      commitmentKey := Lifecycle.VerifierContext.componentDigest 4
        commitmentKeyWords } := by
  rfl

/-- Small deterministic seeded-setup descriptor used only by the nonzero
conformance fixture. -/
def fixtureCommitmentKeyWords : List F :=
  ([1, ringDegree, productionProfile.commitmentWidth,
    Phi81ColumnLayout.blockCount
      (Phi81CarrierLayout.carrierWidth candidateLogicalWidth)] : List Nat).map
      Poseidon2.ofNat ++
    (List.range 32).map fun index => Poseidon2.ofNat (index + 1)

def fixtureAuthority : Lifecycle.VerifierContext.Authority :=
  authorityForIdentity candidatePackageIdentityWords fixtureCommitmentKeyWords

end NightstreamFPrime.Export.Stage1.VerifierContext
