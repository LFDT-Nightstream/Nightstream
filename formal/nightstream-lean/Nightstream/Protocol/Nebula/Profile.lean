/-!
Contract: exact protocol identity for `PaddedRowIdentityMemoryV2`.

Assurance tier: model-level.

Owns the profile name, version, checked-step factor, and authority-bearing
commitment encoding identifier. These values distinguish relations that must
not share one verifier key.

Does not own a verifier-key codec, generated manifest, or deployed parser.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Protocol.Nebula.Profile

inductive Name where
  | paddedRowIdentityMemoryV2
  | paddedRowIdentityMemoryFieldNative
deriving DecidableEq, Repr

inductive CommitmentEncoding where
  | shiftedTernary41V1
deriving DecidableEq, Repr

structure Identity where
  name : Name
  version : Nat
  checkedStepsPerFreshClaim : Nat
  commitmentEncoding : CommitmentEncoding
deriving DecidableEq, Repr

def v2 : Identity where
  name := .paddedRowIdentityMemoryV2
  version := 2
  checkedStepsPerFreshClaim := 1
  commitmentEncoding := .shiftedTernary41V1

@[simp] theorem v2_version : v2.version = 2 := rfl

@[simp] theorem v2_checkedStepsPerFreshClaim :
    v2.checkedStepsPerFreshClaim = 1 := rfl

@[simp] theorem v2_commitmentEncoding :
    v2.commitmentEncoding = .shiftedTernary41V1 := rfl

end Nightstream.Protocol.Nebula.Profile
