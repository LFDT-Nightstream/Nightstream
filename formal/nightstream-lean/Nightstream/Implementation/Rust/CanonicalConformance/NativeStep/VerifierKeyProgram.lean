import Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.ObservedTrace

/-!
Contract: exact raw-field programs for the Construction-2 verifier-key and
initial-boundary Poseidon2 preimages used by the streaming lifecycle relation.

Owns the Rust tag packing, field order, optional public-width encoding, policy
bits, and sponge geometry. It does not evaluate Poseidon2 or establish input
authority. Those obligations belong to generated-row soundness and the
verifier-owned lifecycle boundary.

Assurance tier: model-level for
`FPRIME-STREAMING-RECURSIVE-VERIFIER-KEY-PROGRAM-V1`, Nightstream b2/k16.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.VerifierKeyProgram

open Nightstream.Implementation.Rust.CanonicalConformance.NativeStep

abbrev DigestFields := Fin 4 -> RawField

def digestFields (digest : DigestFields) : List RawField :=
  List.ofFn digest

/-- Verifier-owned scalar parameters in the exact Rust absorb order. -/
structure Parameters where
  q : Nat
  eta : Nat
  d : Nat
  kappa : Nat
  m : Nat
  b : Nat
  kRho : Nat
  bigB : Nat
  t : Nat
  extensionDegree : Nat
  lambda : Nat
  publicInputLength : Option Nat
deriving Repr, DecidableEq

/-- Four-field inputs consumed across the recursive verifier-key stage. -/
structure Inputs where
  structureDigest : DigestFields
  piCcsHeader : DigestFields
  ajtaiPpDigest : DigestFields
  initialSemanticStateDigest : DigestFields

/-- `pack_bytes_as_fields(b"neo.fold.clean/vk_fs/v4")`. -/
def vkFsTag : List RawField :=
  [23, 0x6c6f662e6f656e, 0x6e61656c632e64,
    0x2f73665f6b762f, 0x3476]

/-- `pack_bytes_as_fields(b"neo.fold.clean/vk_fs_policy/v1")`. -/
def vkFsPolicyTag : List RawField :=
  [30, 0x6c6f662e6f656e, 0x6e61656c632e64,
    0x5f73665f6b762f, 0x2f7963696c6f70, 0x3176]

/-- `pack_bytes_as_fields(b"neo.fold.clean/initial_boundary/v2")`. -/
def initialBoundaryTag : List RawField :=
  [34, 0x6c6f662e6f656e, 0x6e61656c632e64,
    0x616974696e692f, 0x646e756f625f6c, 0x32762f797261]

/-- Rust's two-field `Option<usize>` encoding. -/
def optionalUsizeFields : Option Nat -> List RawField
  | none => [0, 0]
  | some value =>
      [rawFieldOfNat (value % 4294967296 + 1),
        rawFieldOfNat (value / 4294967296)]

/-- Base `vk_fs` preimage before the policy wrapper. -/
def basePreimage (parameters : Parameters) (inputs : Inputs) : List RawField :=
  vkFsTag ++
  digestFields inputs.structureDigest ++
  digestFields inputs.piCcsHeader ++
  digestFields inputs.ajtaiPpDigest ++
  u64Halves parameters.q ++
  [rawFieldOfNat parameters.eta, rawFieldOfNat parameters.d,
    rawFieldOfNat parameters.kappa] ++
  u64Halves parameters.m ++
  [rawFieldOfNat parameters.b, rawFieldOfNat parameters.kRho] ++
  u64Halves parameters.bigB ++
  [rawFieldOfNat parameters.t, rawFieldOfNat parameters.extensionDegree,
    rawFieldOfNat parameters.lambda] ++
  optionalUsizeFields parameters.publicInputLength ++
  digestFields inputs.initialSemanticStateDigest

/-- Policy wrapper used by all supported streaming lifecycle arms. -/
def policyPreimage (baseDigest : DigestFields) : List RawField :=
  vkFsPolicyTag ++ digestFields baseDigest ++ [1, 1, 1]

/-- Initial-boundary preimage recomputed from verifier-owned inputs. -/
def initialBoundaryPreimage
    (structureDigest : DigestFields)
    (publicInputLength : Option Nat) : List RawField :=
  initialBoundaryTag ++ digestFields structureDigest ++
    optionalUsizeFields publicInputLength

/-- Constants allocated contiguously for the base verifier-key hash. -/
def baseConstantFields (parameters : Parameters) : List RawField :=
  vkFsTag ++
  u64Halves parameters.q ++
  [rawFieldOfNat parameters.eta, rawFieldOfNat parameters.d,
    rawFieldOfNat parameters.kappa] ++
  u64Halves parameters.m ++
  [rawFieldOfNat parameters.b, rawFieldOfNat parameters.kRho] ++
  u64Halves parameters.bigB ++
  [rawFieldOfNat parameters.t, rawFieldOfNat parameters.extensionDegree,
    rawFieldOfNat parameters.lambda] ++
  optionalUsizeFields parameters.publicInputLength

/-- Constants allocated contiguously for the policy hash. -/
def policyConstantFields : List RawField :=
  vkFsPolicyTag ++ [1, 1, 1]

/-- Constants allocated contiguously for the initial-boundary hash. -/
def initialBoundaryConstantFields
    (publicInputLength : Option Nat) : List RawField :=
  initialBoundaryTag ++ optionalUsizeFields publicInputLength

/-- Required supported-profile choices. Other key parameters stay explicit. -/
def NightstreamGoldilocksProfile (parameters : Parameters) : Prop :=
  parameters.b = 2 /\
  parameters.kRho = 16 /\
  parameters.bigB = 65536 /\
  parameters.publicInputLength = some 648

@[simp] theorem digestFields_length (digest : DigestFields) :
    (digestFields digest).length = 4 := by
  simp [digestFields]

@[simp] theorem optionalUsizeFields_length (value : Option Nat) :
    (optionalUsizeFields value).length = 2 := by
  cases value <;> rfl

theorem basePreimage_length (parameters : Parameters) (inputs : Inputs) :
    (basePreimage parameters inputs).length = 37 := by
  simp [basePreimage, vkFsTag, u64Halves]

theorem policyPreimage_length (baseDigest : DigestFields) :
    (policyPreimage baseDigest).length = 13 := by
  simp [policyPreimage, vkFsPolicyTag]

theorem initialBoundaryPreimage_length
    (structureDigest : DigestFields) (publicInputLength : Option Nat) :
    (initialBoundaryPreimage structureDigest publicInputLength).length = 12 := by
  simp [initialBoundaryPreimage, initialBoundaryTag]

/-- Goldilocks Poseidon2 uses rate four and one final padding permutation. -/
def poseidon2PermutationCount (preimage : List RawField) : Nat :=
  (preimage.length + 3) / 4 + 1

theorem base_poseidon2_permutations
    (parameters : Parameters) (inputs : Inputs) :
    poseidon2PermutationCount (basePreimage parameters inputs) = 11 := by
  rw [poseidon2PermutationCount, basePreimage_length]

theorem policy_poseidon2_permutations (baseDigest : DigestFields) :
    poseidon2PermutationCount (policyPreimage baseDigest) = 5 := by
  rw [poseidon2PermutationCount, policyPreimage_length]

theorem initialBoundary_poseidon2_permutations
    (structureDigest : DigestFields) (publicInputLength : Option Nat) :
    poseidon2PermutationCount
      (initialBoundaryPreimage structureDigest publicInputLength) = 4 := by
  rw [poseidon2PermutationCount, initialBoundaryPreimage_length]

theorem recursiveVerifierKey_poseidon2_permutations
    (parameters : Parameters) (inputs : Inputs) (baseDigest : DigestFields) :
    poseidon2PermutationCount (basePreimage parameters inputs) +
      poseidon2PermutationCount (policyPreimage baseDigest) +
      poseidon2PermutationCount
        (initialBoundaryPreimage inputs.structureDigest
          parameters.publicInputLength) = 20 := by
  rw [base_poseidon2_permutations, policy_poseidon2_permutations,
    initialBoundary_poseidon2_permutations]

end Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.VerifierKeyProgram
