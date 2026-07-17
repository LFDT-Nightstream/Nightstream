import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Primitives
import Nightstream.SuperNeo.Concrete.Algebra

/-!
Shared carrier transport for production-shaped `Pi_CCS` transcript
refinement.

Assurance tier: executable implementation semantics.

Owns: lossless transport between the independently named semantic
Goldilocks/extension carriers and the concrete transcript field/extension
carriers.

Does not own: transcript order, message widths, Poseidon2 execution, SumCheck
algebra, Rust/R1CS refinement, emitted rows, costs, or row removal.

Emits constraints: no.

Authority boundary: transport changes representation only. It cannot create,
discard, reorder, hash, or authorize a field element.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_ccs.transcript.transport.base` | semantic and implementation fields use the same Goldilocks modulus | derived | `modulus_eq` |
| `nifs.pi_ccs.transcript.transport.base.roundtrip` | base-field transport is lossless in both directions | derived | `toSemanticField_toField`, `toField_toSemanticField` |
| `nifs.pi_ccs.transcript.transport.extension.roundtrip` | `(c0,c1)` transport is lossless and order-preserving | derived | `toK_toExtension`, `toExtension_toK` |
-/

namespace Nightstream.Implementation.R1CS.PiCcsTranscript.Transport

open Nightstream.Implementation.R1CS.PiRlcChallenge.TranscriptMachine
open Nightstream.Implementation.R1CS.PiCcsTranscript.Primitives
open Nightstream.SuperNeo.Concrete

/-- The implementation and semantic carriers independently name the same
production Goldilocks modulus. -/
theorem modulus_eq : goldilocksP = goldilocksModulus := by
  rfl

/-- Transport one semantic Goldilocks residue into the implementation carrier. -/
def toField (value : F) : Field :=
  ⟨value.val, by
    rw [modulus_eq]
    exact value.isLt⟩

/-- Transport one implementation Goldilocks residue into the semantic carrier. -/
def toSemanticField (value : Field) : F :=
  ⟨value.val, by
    rw [← modulus_eq]
    exact value.isLt⟩

/-- Semantic-to-implementation base-field transport is lossless. -/
@[simp] theorem toSemanticField_toField (value : F) :
    toSemanticField (toField value) = value := by
  apply Fin.ext
  rfl

/-- Implementation-to-semantic base-field transport is lossless. -/
@[simp] theorem toField_toSemanticField (value : Field) :
    toField (toSemanticField value) = value := by
  apply Fin.ext
  rfl

/-- Explicit semantic-`K` to implementation-extension transport. -/
def toExtension (value : K) : Extension :=
  { c0 := toField value.c0
    c1 := toField value.c1 }

/-- Explicit implementation-extension to semantic-`K` transport. -/
def toK (value : Extension) : K :=
  { c0 := toSemanticField value.c0
    c1 := toSemanticField value.c1 }

/-- Semantic challenges survive the implementation transport round trip. -/
@[simp] theorem toK_toExtension (value : K) :
    toK (toExtension value) = value := by
  cases value
  simp [toK, toExtension]

/-- Implementation responses survive the semantic transport round trip. -/
@[simp] theorem toExtension_toK (value : Extension) :
    toExtension (toK value) = value := by
  cases value
  simp [toK, toExtension]

end Nightstream.Implementation.R1CS.PiCcsTranscript.Transport
