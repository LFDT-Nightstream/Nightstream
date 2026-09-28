import NightstreamFPrime.Lifecycle.VerifierContext

/-!
Owns the canonical binding preimage required in every production Stage 1
verification key.

The binding joins the final package identity to the verifier-context
descriptor recomputed from raw relation, application, NIFS-key, and
commitment-key authority. It is acyclic because the context descriptor does
not contain the final package identity.

This is not a proof-backend verification key and does not authorize one.
-/

namespace NightstreamFPrime.Lifecycle.Stage1.VerificationKey

open NightstreamFPrime.Spec
open NightstreamFPrime.Lifecycle

/-- Static values that every concrete backend verification key must bind. -/
structure Binding where
  packageIdentity : VerifierContext.Digest4
  context : VerifierContext.Descriptor
deriving DecidableEq

def domain : List F :=
  ([78, 105, 103, 104, 116, 115, 116, 114, 101, 97, 109, 47,
    70, 80, 114, 105, 109, 101, 47, 118, 101, 114, 105, 102, 105, 101,
    114, 45, 107, 101, 121, 47, 118, 49] : List Nat).map Poseidon2.ofNat

def Binding.serialize (binding : Binding) : List F :=
  domain ++ VerifierContext.framed binding.packageIdentity.toList ++
    VerifierContext.framed binding.context.serialize

def Binding.digest4 (binding : Binding) : VerifierContext.Digest4 :=
  VerifierContext.Digest4.ofList (Poseidon2.hash binding.serialize)

def Binding.digest (binding : Binding) : KeyDigest :=
  binding.digest4.toList

/-- Construct the binding only from a recomputed verifier-owned authority and
the final verifier-pinned package identity. -/
def ofAuthority (packageIdentity : VerifierContext.Digest4)
    (authority : VerifierContext.Authority) : Binding where
  packageIdentity := packageIdentity
  context := VerifierContext.descriptor authority

@[simp] theorem ofAuthority_packageIdentity
    (packageIdentity : VerifierContext.Digest4)
    (authority : VerifierContext.Authority) :
    (ofAuthority packageIdentity authority).packageIdentity = packageIdentity := by
  rfl

@[simp] theorem ofAuthority_context
    (packageIdentity : VerifierContext.Digest4)
    (authority : VerifierContext.Authority) :
    (ofAuthority packageIdentity authority).context =
      VerifierContext.descriptor authority := by
  rfl

theorem Binding.serialize_length (binding : Binding) :
    binding.serialize.length = 126 := by
  simp [Binding.serialize, domain, VerifierContext.framed,
    VerifierContext.Digest4.toList,
    VerifierContext.Descriptor.serialize_length]

@[simp] theorem Binding.digest_length (binding : Binding) :
    binding.digest.length = 4 := by
  exact VerifierContext.Digest4.toList_length binding.digest4

/-- Verification-key binding is recomputed from the complete canonical
package-and-context preimage. -/
theorem Binding.digest_recomputed (binding : Binding) :
    binding.digest =
      (VerifierContext.Digest4.ofList
        (Poseidon2.hash binding.serialize)).toList := by
  rfl

end NightstreamFPrime.Lifecycle.Stage1.VerificationKey
