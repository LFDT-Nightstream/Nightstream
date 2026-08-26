import NightstreamFPrime.Export.Stage1.Data
import NightstreamFPrime.Lifecycle.VerifierContext

/-!
Owns the canonical verifier-context recipe for the selected Stage 1 package.

The package identity selects the exact Lean-emitted prefix rows. The current
prefix uses that identity in the relation and application component positions;
it does not yet prove that the prefix contains the final logical relation or
application transition. That connection remains a required Stage 1 fixed-point
edge. The NIFS component also binds the fixed profile, digest-only transcript
schedule, package identity, and commitment-key component digest. The final
component hashes the verifier-owned commitment setup serialization itself.

The sealed package identity and the resulting context digest remain distinct.
The package contains public context columns, not fixed context values, so this
recipe is non-self-referential.
-/

namespace NightstreamFPrime.Export.Stage1.VerifierContext

open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

/-- Verifier-owned expected identity for the current package cut. This value
was pinned only after exact matrix equality, independent assignment checking,
complete nonzero parity, and mutation gates passed. Rust recomputes the same
identity from every package field before it can derive a context. -/
def expectedPackageIdentity : Lifecycle.VerifierContext.Digest4 where
  c0 := ⟨4149794454264745319, by norm_num [F, goldilocksModulus]⟩
  c1 := ⟨3860295598124073314, by norm_num [F, goldilocksModulus]⟩
  c2 := ⟨9185184515076867919, by norm_num [F, goldilocksModulus]⟩
  c3 := ⟨6634095431211870257, by norm_num [F, goldilocksModulus]⟩

def packageIdentityWords : List F :=
  expectedPackageIdentity.toList

/-- Trusted implementation-link condition checked by the strict Rust loader.
It is not a semantic axiom and is not used to prove row soundness. -/
def PackageIdentityHolds : Prop :=
  packageIdentityWords = Data.relationIdentifier ()

/-- Domain of the compact NIFS-key authority description. -/
def nifsKeyDomain : List F :=
  ([78, 105, 103, 104, 116, 115, 116, 114, 101, 97, 109, 47,
    70, 80, 114, 105, 109, 101, 47, 110, 105, 102, 115, 45,
    107, 101, 121, 47, 118, 49, 95, 49] : List Nat).map Poseidon2.ofNat

/-- Current package-bound NIFS descriptor and one verifier-owned commitment
setup serialization. The commitment component is compressed once before it
is included here; a collision in that compression is a named context failure.
The later fixed-point theorem must connect this descriptor to `ProductionKey`. -/
def nifsKeyWords (commitmentKeyWords : List F) : List F :=
  nifsKeyDomain ++
    Lifecycle.VerifierContext.framed Lifecycle.VerifierContext.profileWords ++
    Lifecycle.VerifierContext.framed Lifecycle.VerifierContext.scheduleWords ++
    Lifecycle.VerifierContext.framed packageIdentityWords ++
    Lifecycle.VerifierContext.framed
      (Lifecycle.VerifierContext.componentDigest 4 commitmentKeyWords).toList

/-- Canonical context authority for the selected package. The verifier supplies
the canonical serialization of its actual commitment setup, not a digest. -/
def authority (commitmentKeyWords : List F) :
    Lifecycle.VerifierContext.Authority where
  relationWords := packageIdentityWords
  applicationWords := packageIdentityWords
  nifsKeyWords := nifsKeyWords commitmentKeyWords
  commitmentKeyWords := commitmentKeyWords

@[simp] theorem authority_relationWords (commitmentKeyWords : List F) :
    (authority commitmentKeyWords).relationWords = packageIdentityWords := by
  rfl

@[simp] theorem authority_applicationWords (commitmentKeyWords : List F) :
    (authority commitmentKeyWords).applicationWords = packageIdentityWords := by
  rfl

theorem authority_relationWords_of_packageIdentity
    (commitmentKeyWords : List F) (holds : PackageIdentityHolds) :
    (authority commitmentKeyWords).relationWords = Data.relationIdentifier () := by
  exact holds

theorem authority_applicationWords_of_packageIdentity
    (commitmentKeyWords : List F) (holds : PackageIdentityHolds) :
    (authority commitmentKeyWords).applicationWords = Data.relationIdentifier () := by
  exact holds

@[simp] theorem authority_nifsKeyWords (commitmentKeyWords : List F) :
    (authority commitmentKeyWords).nifsKeyWords = nifsKeyWords commitmentKeyWords := by
  rfl

@[simp] theorem authority_commitmentKeyWords (commitmentKeyWords : List F) :
    (authority commitmentKeyWords).commitmentKeyWords = commitmentKeyWords := by
  rfl

/-- The descriptor uses the exact current package identity in both reserved
relation positions, the NIFS recipe, and the supplied setup serialization. -/
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
conformance fixture. Production supplies the actual verifier-owned setup
serialization to `authority`. -/
def fixtureCommitmentKeyWords : List F :=
  ([1, ringDegree, productionProfile.commitmentWidth,
    Phi81ColumnLayout.blockCount
      (Phi81CarrierLayout.carrierWidth Data.logicalWidth)] : List Nat).map
      Poseidon2.ofNat ++
    (List.range 32).map fun index => Poseidon2.ofNat (index + 1)

def fixtureAuthority : Lifecycle.VerifierContext.Authority :=
  authority fixtureCommitmentKeyWords

end NightstreamFPrime.Export.Stage1.VerifierContext
