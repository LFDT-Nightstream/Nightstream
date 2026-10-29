import NightstreamFPrime.Lifecycle.VerifierContext

namespace NightstreamFPrime.Lifecycle.Stage1.VerificationKey

open NightstreamFPrime.Spec
open NightstreamFPrime.Lifecycle

structure Binding where
  packageIdentity : VerifierContext.Digest4
  context : VerifierContext.Descriptor
  formulas : VerifierContext.Digest4
deriving DecidableEq

def domain : List F :=
  ([78, 105, 103, 104, 116, 115, 116, 114, 101, 97, 109, 47,
    70, 80, 114, 105, 109, 101, 47, 118, 101, 114, 105, 102, 105, 101,
    114, 45, 107, 101, 121, 47, 118, 49] : List Nat).map Poseidon2.ofNat

def Binding.serialize (binding : Binding) : List F :=
  domain ++ VerifierContext.framed binding.packageIdentity.toList ++
    VerifierContext.framed binding.context.serialize ++
    VerifierContext.framed binding.formulas.toList

def Binding.digest4 (binding : Binding) : VerifierContext.Digest4 :=
  VerifierContext.Digest4.ofList (Poseidon2.hash binding.serialize)

def Binding.digest (binding : Binding) : KeyDigest :=
  binding.digest4.toList

def ofAuthority (packageIdentity : VerifierContext.Digest4)
    (authority : VerifierContext.Authority)
    (formulas : VerifierContext.Digest4) : Binding where
  packageIdentity := packageIdentity
  context := VerifierContext.descriptor authority
  formulas := formulas

@[simp] theorem ofAuthority_packageIdentity
    (packageIdentity : VerifierContext.Digest4)
    (authority : VerifierContext.Authority)
    (formulas : VerifierContext.Digest4) :
    (ofAuthority packageIdentity authority formulas).packageIdentity = packageIdentity := by
  rfl

@[simp] theorem ofAuthority_context
    (packageIdentity : VerifierContext.Digest4)
    (authority : VerifierContext.Authority)
    (formulas : VerifierContext.Digest4) :
    (ofAuthority packageIdentity authority formulas).context =
      VerifierContext.descriptor authority := by
  rfl

@[simp] theorem ofAuthority_formulas
    (packageIdentity : VerifierContext.Digest4)
    (authority : VerifierContext.Authority)
    (formulas : VerifierContext.Digest4) :
    (ofAuthority packageIdentity authority formulas).formulas = formulas := by
  rfl

theorem Binding.serialize_length (binding : Binding) :
    binding.serialize.length = 131 := by
  simp [Binding.serialize, domain, VerifierContext.framed,
    VerifierContext.Digest4.toList,
    VerifierContext.Descriptor.serialize_length]

@[simp] theorem Binding.digest_length (binding : Binding) :
    binding.digest.length = 4 := by
  exact VerifierContext.Digest4.toList_length binding.digest4

theorem Binding.digest_recomputed (binding : Binding) :
    binding.digest =
      (VerifierContext.Digest4.ofList
        (Poseidon2.hash binding.serialize)).toList := by
  rfl

end NightstreamFPrime.Lifecycle.Stage1.VerificationKey
