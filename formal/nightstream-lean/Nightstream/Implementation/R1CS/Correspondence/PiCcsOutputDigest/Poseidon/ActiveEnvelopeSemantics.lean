import Nightstream.Implementation.R1CS.Correspondence.PiCcsOutputDigest.Profile
import Nightstream.Implementation.R1CS.Correspondence.PiCcsOutputDigest.Poseidon.EnvelopeSemantics

/-!
Independent Poseidon2 envelope semantics for the active thirteen-matrix
`Pi_CCS` output profile.

Assurance tier: model-level representation semantics.

Owns: specialization of the generic SIS-digest envelope to the independently
defined 15-source, 13-matrix profile; the exact 23,033-field metadata value;
and the resulting 64-field preimage shape after 54-coordinate compression.

Does not own: production relation selection, source authority, either SIS
map, Poseidon2 lowering, transcript placement, physical columns, costs,
necessity, or row removal.

Emits constraints: no.

Authority boundary: the active field count is computed from `Profile`; it is
not recovered from a historical artifact or accepted from a prover. A later
physical refinement must prove that the stabilized Rust relation selects the
same profile and pins these exact prefix values.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_ccs.output_digest.poseidon.prefix.profile` | specialize to 15 sources and 13 matrices | verifier-selected shape | `sourceFieldCount` |
| `nifs.pi_ccs.output_digest.poseidon.prefix.fields` | place 23,033 in the exact metadata slot | computed | `envelopePrefix_eq` |
| `nifs.pi_ccs.output_digest.poseidon.envelope` | append exactly 54 compressed coordinates | computed | `envelope_length_of_compression` |
-/

namespace Nightstream.Implementation.R1CS.PiCcsOutputDigest.Poseidon.ActiveEnvelopeSemantics

/-- Active serializer size, derived from the independent profile formula. -/
def sourceFieldCount : Nat :=
  Profile.fieldCount Profile.steadyFixedPointThirteenMatrix

/-- Exact active ten-field prefix. -/
def envelopePrefix : List Nat :=
  EnvelopeSemantics.envelopePrefixFor sourceFieldCount

/-- Exact active Poseidon2 preimage after rank-one SIS compression. -/
def envelope (compressionOutput : List Nat) : List Nat :=
  EnvelopeSemantics.envelopeFor sourceFieldCount compressionOutput

@[simp] theorem sourceFieldCount_eq : sourceFieldCount = 23033 := by
  decide

theorem envelopePrefix_eq :
    envelopePrefix =
      [40, 30521782141150574, 31069335676202596,
       33052923221205295, 32421790864400748,
       28542674997834601, 225321120883,
       5785229152774737749, 23033, 2] := by
  rw [envelopePrefix, EnvelopeSemantics.envelopePrefixFor,
    EnvelopeSemantics.digestDomainTag_eq]
  decide

@[simp] theorem envelopePrefix_length : envelopePrefix.length = 10 := by
  simp [envelopePrefix]

@[simp] theorem envelope_length (compressionOutput : List Nat) :
    (envelope compressionOutput).length = 10 + compressionOutput.length := by
  simp [envelope, EnvelopeSemantics.envelopeFor]

theorem envelope_length_of_compression
    (compressionOutput : List Nat)
    (length : compressionOutput.length = 54) :
    (envelope compressionOutput).length = 64 := by
  simp [length]

theorem active_ne_diagnostic_prefix :
    envelopePrefix ≠ EnvelopeSemantics.diagnosticEnvelopePrefix := by
  rw [envelopePrefix_eq, EnvelopeSemantics.diagnosticEnvelopePrefix_eq]
  decide

end Nightstream.Implementation.R1CS.PiCcsOutputDigest.Poseidon.ActiveEnvelopeSemantics
