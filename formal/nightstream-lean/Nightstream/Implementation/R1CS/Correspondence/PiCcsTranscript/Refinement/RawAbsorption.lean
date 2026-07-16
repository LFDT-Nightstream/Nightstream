import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Primitives

/-!
Native-versus-gadget refinement for raw transcript absorption.

Assurance tier: executable implementation refinement. The native transcript
and the variable-slice gadget eagerly normalize a full rate buffer; the
constant gadget absorbs one field at a time and may leave the cursor full.
This module proves that the latter is a delayed representation of the former,
not an independently trusted transcript state.

Owns: the two concrete absorption modes and their exact equivalence at every
subsequent verifier-visible absorb, raw append, or digest boundary.

Does not own: which `Pi_CCS` messages use either mode, generated columns or
rows, Poseidon2-call acceptance, protocol soundness, cost totals, or row
removal.

Emits constraints: no.

Authority boundary: the lazy state is never accepted as a digest. It is usable
only through the proved normalization boundary that recovers native state.

| Protocol | Phase | Rust path | Mathematical guarantee |
|---|---|---|---|
| `Pi_CCS` | raw constant append | `append_fields_raw_const` | lazy element absorption differs only by a pending full-rate permutation |
| `Pi_CCS` | raw variable append | `append_fields_raw_vars` | state equals native eager `append_fields_raw` semantics |
| `Pi_CCS` | next operation | absorb/append/digest | pending normalization is forced before any verifier-visible response |
-/

namespace Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.RawAbsorption

open Nightstream.Implementation.R1CS.PiRlcChallenge.TranscriptMachine
open Nightstream.Implementation.R1CS.PiCcsTranscript.Primitives

/-- State semantics of `TranscriptGadget::append_fields_raw_const`. -/
def gadgetConstantAppend (state : State) (fields : List Field) : State :=
  appendRawLazy state fields

/-- State semantics of `TranscriptGadget::append_fields_raw_vars`. -/
def gadgetVariableAppend (state : State) (fields : List Field) : State :=
  appendRaw state fields

/-- Normalizing before one absorb is observationally redundant because
`absorbElem` itself must normalize a full cursor before overwriting lane zero. -/
theorem absorbElem_normalizeFull (state : State) (value : Field) :
    absorbElem (normalizeFull state) value = absorbElem state value := by
  unfold normalizeFull
  split
  · rename_i full
    have notRoom : not (state.absorbed.val < rate) := by
      simp [full]
    simp [absorbElem, full, permute, rate]
  · rfl

/-- The constant gadget state becomes exactly the native state after explicit
full-rate normalization. -/
theorem constant_normalizes_to_native (state : State) (fields : List Field) :
    normalizeFull (gadgetConstantAppend state fields) =
      appendRaw state fields := by
  rfl

/-- The variable-slice gadget already has native state semantics. -/
theorem variable_eq_native (state : State) (fields : List Field) :
    gadgetVariableAppend state fields = appendRaw state fields := by
  rfl

/-- Any following absorbed word forces the lazy constant path to the same
state as native eager absorption. -/
theorem constant_then_absorb_eq_native (state : State)
    (fields : List Field) (next : Field) :
    absorbElem (gadgetConstantAppend state fields) next =
      absorbElem (appendRaw state fields) next := by
  symm
  exact absorbElem_normalizeFull (gadgetConstantAppend state fields) next

/-- A following raw message cannot distinguish the delayed constant path from
native eager absorption. -/
theorem constant_then_append_eq_native (state : State)
    (fields next : List Field) :
    appendRaw (gadgetConstantAppend state fields) next =
      appendRaw (appendRaw state fields) next := by
  simp only [appendRaw, appendRawLazy, gadgetConstantAppend, absorbAll]
  rw [absorbElem_normalizeFull]

/-- A verifier squeeze observes exactly the native digest and successor state,
even when the preceding constant gadget append ended on a full cursor. -/
theorem constant_then_digest_eq_native (state : State)
    (fields : List Field) :
    digest (gadgetConstantAppend state fields) =
      digest (appendRaw state fields) := by
  simp only [digest]
  rw [constant_then_absorb_eq_native]

end Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.RawAbsorption
