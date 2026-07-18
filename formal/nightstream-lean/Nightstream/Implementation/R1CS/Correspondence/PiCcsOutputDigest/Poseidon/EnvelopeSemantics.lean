import Nightstream.Implementation.R1CS.Correspondence.PiCcsOutputDigest.Encoding
import Nightstream.Implementation.R1CS.Core.SevenBytePacking

/-!
Independent field-envelope semantics for the terminal `Pi_CCS` output digest.

Assurance tier: executable protocol-primitive semantics. This file fixes the
Poseidon2 preimage after SIS compression without importing generated rows,
production columns, Rust emitters, profiler totals, or a carried digest.

Owns: the outer SIS-digest domain bytes; their seven-byte field packing; the
`Pi_CCS`-output map domain; profile-indexed source-field metadata; binding
rank; and the ordered envelope shape for a 54-coordinate compression output.

Does not own: the two SIS maps, public-seed expansion, Poseidon2 permutation
semantics, sponge lowering, transcript placement, collision resistance, row
necessity, row removal, or cost totals.

Emits constraints: no.

Authority boundary: `envelope` accepts the mathematically derived compression
output, never a digest. A later refinement must prove that production columns
equal this list before any four output lanes receive digest meaning.

| Protocol | Phase | Constraint family | Mathematical obligation |
|---|---|---|---|
| `Pi_CCS` | output digest | domain packing | exact `accumulator/sis/digest/v4` bytes and seven-byte packing |
| `Pi_CCS` | output digest | map metadata | domain `0x50494343535F4F55`, caller-supplied semantic field count, rank two |
| `Pi_CCS` | output digest | compression payload | all 54 rank-one SIS output coordinates in order |
| `Pi_CCS` | output digest | Poseidon2 preimage | exact ten-field profile prefix followed by the 54-coordinate payload |
| `Pi_CCS` | output digest | diagnostic alias | retain the quarantined 6,683-field specialization for its historical artifact |
-/

namespace Nightstream.Implementation.R1CS.PiCcsOutputDigest.Poseidon.EnvelopeSemantics

/-- Exact bytes of `neo.fold.clean/accumulator/sis/digest/v4`. -/
def digestDomainBytes : List Nat :=
  [110, 101, 111, 46, 102, 111, 108, 100, 46, 99, 108, 101, 97,
   110, 47, 97, 99, 99, 117, 109, 117, 108, 97, 116, 111, 114,
   47, 115, 105, 115, 47, 100, 105, 103, 101, 115, 116, 47, 118, 52]

/-- Domain of the primary `Pi_CCS` output-message SIS map. -/
def bindingDomain : Nat := 5785229152774737749

/-- Exact field count of the quarantined three-matrix diagnostic. The active
13-matrix target must use `envelopePrefixFor 23033` instead. -/
def diagnosticSourceFieldCount : Nat := 6683

/-- Rank of the primary binding map. -/
def bindingRank : Nat := 2

/-- Independent ten-field prefix for an explicitly selected serializer
profile. The field count is semantic metadata, never inferred from a digest. -/
def envelopePrefixFor (fieldCount : Nat) : List Nat :=
  Nightstream.Implementation.R1CS.SevenBytePacking.packBytesAsNats
      digestDomainBytes ++
    [bindingDomain, fieldCount, bindingRank]

/-- Complete profile-indexed Poseidon2 input after rank-one compression. -/
def envelopeFor (fieldCount : Nat) (compressionOutput : List Nat) : List Nat :=
  envelopePrefixFor fieldCount ++ compressionOutput

/-- Quarantined three-matrix prefix retained for the historical artifact. -/
def diagnosticEnvelopePrefix : List Nat :=
  envelopePrefixFor diagnosticSourceFieldCount

/-- Quarantined three-matrix envelope retained for the historical artifact. -/
def diagnosticEnvelope (compressionOutput : List Nat) : List Nat :=
  envelopeFor diagnosticSourceFieldCount compressionOutput

/-- Closed byte-string and packing check, independent of production pins. -/
theorem digestDomainTag_eq :
    Nightstream.Implementation.R1CS.SevenBytePacking.packBytesAsNats
        digestDomainBytes =
      [40, 30521782141150574, 31069335676202596,
       33052923221205295, 32421790864400748,
       28542674997834601, 225321120883] := by
  decide

theorem diagnosticEnvelopePrefix_eq :
    diagnosticEnvelopePrefix =
      [40, 30521782141150574, 31069335676202596,
       33052923221205295, 32421790864400748,
       28542674997834601, 225321120883,
       5785229152774737749, 6683, 2] := by
  rw [diagnosticEnvelopePrefix, envelopePrefixFor, digestDomainTag_eq]
  rfl

@[simp] theorem envelopePrefixFor_length (fieldCount : Nat) :
    (envelopePrefixFor fieldCount).length = 10 := by
  simp [envelopePrefixFor, digestDomainTag_eq]

@[simp] theorem diagnosticEnvelopePrefix_length :
    diagnosticEnvelopePrefix.length = 10 := by
  simp [diagnosticEnvelopePrefix]

@[simp] theorem envelopeFor_length (fieldCount : Nat)
    (compressionOutput : List Nat) :
    (envelopeFor fieldCount compressionOutput).length =
      10 + compressionOutput.length := by
  simp [envelopeFor]

@[simp] theorem diagnosticEnvelope_length (compressionOutput : List Nat) :
    (diagnosticEnvelope compressionOutput).length =
      10 + compressionOutput.length := by
  simp [diagnosticEnvelope, envelopeFor]

theorem diagnosticEnvelope_length_of_compression
    (compressionOutput : List Nat)
    (length : compressionOutput.length = 54) :
    (diagnosticEnvelope compressionOutput).length = 64 := by
  simp [length]

end Nightstream.Implementation.R1CS.PiCcsOutputDigest.Poseidon.EnvelopeSemantics
