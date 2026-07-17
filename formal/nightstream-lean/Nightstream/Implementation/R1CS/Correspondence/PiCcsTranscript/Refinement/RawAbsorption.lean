import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Primitives

/-!
Native-versus-gadget refinement for raw transcript absorption.

Assurance tier: executable implementation refinement. The native transcript
and the variable-slice gadget eagerly normalize a full rate buffer; the
constant gadget absorbs one field at a time and may leave the cursor full.
This module proves that the latter is a delayed representation of the former,
not an independently trusted transcript state.

Owns: the two concrete absorption modes, normalized-state equivalence, its
basic composition laws, and exact congruence at every subsequent
verifier-visible absorb, raw append, or digest boundary.

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
| `Pi_CCS` | state relation | native/gadget phase boundary | states are equivalent exactly when explicit full-rate normalization agrees |
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

/-- Two transcript states differ by at most the constant gadget's one pending
full-rate permutation. This is the compositional relation used across phase
boundaries; neither side is accepted as a digest without a later observer. -/
def NormalizedEq (left right : State) : Prop :=
  normalizeFull left = normalizeFull right

/-- Normalized-state equivalence is reflexive. -/
theorem normalizedEq_refl (state : State) :
    NormalizedEq state state :=
  rfl

/-- Normalized-state equivalence is symmetric. -/
theorem normalizedEq_symm
    {left right : State}
    (related : NormalizedEq left right) :
    NormalizedEq right left :=
  related.symm

/-- Normalized-state equivalence composes across adjacent transcript phases. -/
theorem normalizedEq_trans
    {left middle right : State}
    (leftMiddle : NormalizedEq left middle)
    (middleRight : NormalizedEq middle right) :
    NormalizedEq left right :=
  leftMiddle.trans middleRight

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

/-- Explicit full-rate normalization is idempotent. -/
theorem normalizeFull_idempotent (state : State) :
    normalizeFull (normalizeFull state) = normalizeFull state := by
  unfold normalizeFull
  split
  · simp [normalizeFull, permute, rate]
  · rfl

/-- Native raw append always returns an explicitly normalized state. -/
theorem normalizeFull_appendRaw (state : State) (fields : List Field) :
    normalizeFull (appendRaw state fields) = appendRaw state fields := by
  unfold appendRaw
  exact normalizeFull_idempotent _

/-- Native raw append cannot observe whether its input was normalized
immediately or carried one pending full-rate permutation. -/
theorem appendRaw_normalizeFull (state : State) (fields : List Field) :
    appendRaw (normalizeFull state) fields = appendRaw state fields := by
  simp only [appendRaw, appendRawLazy, absorbAll]
  rw [absorbElem_normalizeFull]

/-- One subsequent absorbed field collapses normalized-state equivalence to
literal state equality. -/
theorem absorbElem_eq_of_normalizedEq
    {left right : State}
    (related : NormalizedEq left right)
    (value : Field) :
    absorbElem left value = absorbElem right value := by
  change normalizeFull left = normalizeFull right at related
  calc
    absorbElem left value =
        absorbElem (normalizeFull left) value :=
      (absorbElem_normalizeFull left value).symm
    _ = absorbElem (normalizeFull right) value := by rw [related]
    _ = absorbElem right value :=
      absorbElem_normalizeFull right value

/-- One subsequent native raw append collapses normalized-state equivalence
to literal state equality. -/
theorem appendRaw_eq_of_normalizedEq
    {left right : State}
    (related : NormalizedEq left right)
    (fields : List Field) :
    appendRaw left fields = appendRaw right fields := by
  change normalizeFull left = normalizeFull right at related
  calc
    appendRaw left fields =
        appendRaw (normalizeFull left) fields :=
      (appendRaw_normalizeFull left fields).symm
    _ = appendRaw (normalizeFull right) fields := by rw [related]
    _ = appendRaw right fields :=
      appendRaw_normalizeFull right fields

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

/-- A lazy constant append on one state is normalized-equivalent to an eager
native append on any already equivalent state. -/
theorem constantAppend_normalizedEq
    {left right : State}
    (related : NormalizedEq left right)
    (fields : List Field) :
    NormalizedEq
      (gadgetConstantAppend left fields)
      (appendRaw right fields) := by
  change
    normalizeFull (gadgetConstantAppend left fields) =
      normalizeFull (appendRaw right fields)
  rw [constant_normalizes_to_native, normalizeFull_appendRaw]
  exact appendRaw_eq_of_normalizedEq related fields

/-- An eager variable append on one state is literally equal to native append
on any normalized-equivalent state. -/
theorem variableAppend_eq_of_normalizedEq
    {left right : State}
    (related : NormalizedEq left right)
    (fields : List Field) :
    gadgetVariableAppend left fields = appendRaw right fields := by
  exact appendRaw_eq_of_normalizedEq related fields

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

/-- A verifier digest cannot distinguish any two normalized-equivalent
states, including an empty-round phase that leaves the constant gadget full. -/
theorem digest_eq_of_normalizedEq
    {left right : State}
    (related : NormalizedEq left right) :
    digest left = digest right := by
  simp only [digest]
  rw [absorbElem_eq_of_normalizedEq related]

end Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.RawAbsorption
