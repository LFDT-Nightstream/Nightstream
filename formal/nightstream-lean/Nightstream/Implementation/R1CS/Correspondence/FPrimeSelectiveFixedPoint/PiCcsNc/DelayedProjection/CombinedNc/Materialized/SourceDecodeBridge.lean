import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.Semantics

/-!
Kernel bridge from proof-free combined-NC source rows to their typed decoder
semantics.

Owns: the profile-neutral structural predicate for one raw source row,
successful fail-closed decoding under that predicate, and lossless recovery of
the ordered sparse A/B/C row from every successful decoding.

Does not own: generated-row truth, production-profile counts, source-to-final
rewrite refinement, row satisfaction, transcript scheduling, parent or child
authority, commitment binding, costs, or row removal.

Emits constraints: none.

Assurance tier: model-level. Generated certificates may discharge the explicit
premises, but this file imports and evaluates no generated row collection.
-/

/-!
| Stable stage path | Obligation | Authority class |
|---|---|---|
| `f_prime.pi_ccs_nc.delayed.combined.source_decode_bridge` | Connect indexed source-row decoding facts to typed source obligations. | checked artifact |

-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.SourceDecodeBridge

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.Decoder

/-- Exact proof-free sparse-term projection used by artifact certificates. -/
def rawTerms (terms : List RawTerm) : List (Nat × Nat) :=
  terms.map fun term => (term.column, term.coefficient)

/-- Exact proof-free A/B/C projection. Row metadata is checked separately. -/
def rawRow (row : RawSourceRow) : Row where
  a := rawTerms row.a
  b := rawTerms row.b
  c := rawTerms row.c

/-- Canonical, nonzero, in-range conditions accepted by `decodeTerm`. -/
def RawTermValid (columns : Nat) (term : RawTerm) : Prop :=
  term.column < columns ∧
  term.coefficient < goldilocksModulus ∧
  term.coefficient ≠ 0

/-- Profile-neutral fail-closed premises for one source row. The joined term
condition deliberately matches the compact certificate representation. -/
def RawSourceRowValid (row : RawSourceRow) : Prop :=
  row.schemaVersion = supportedSchemaVersion ∧
  0 < row.rows ∧
  0 < row.columns ∧
  row.sourceRow < row.rows ∧
  ∀ term ∈ row.a ++ row.b ++ row.c, RawTermValid row.columns term

private theorem field_ne_zero_of_word_ne_zero {value : Nat}
    (canonical : value < goldilocksModulus)
    (wordNonzero : value ≠ 0) :
    (⟨value, canonical⟩ : F) ≠ 0 := by
  intro equality
  apply wordNonzero
  simpa using congrArg Fin.val equality

/-- One structurally valid proof-free term has a typed fail-closed decoding. -/
theorem decodeTerm_of_valid {columns : Nat} {raw : RawTerm}
    (valid : RawTermValid columns raw) :
    ∃ decoded, decodeTerm columns raw = some decoded := by
  have coefficientNonzero :
      (⟨raw.coefficient, valid.2.1⟩ : F) ≠ 0 :=
    field_ne_zero_of_word_ne_zero valid.2.1 valid.2.2
  exact ⟨
    canonicalDecodedTerm columns raw.column raw.coefficient valid.1 valid.2.1
      coefficientNonzero,
    decodeTerm_canonical columns raw.column raw.coefficient valid.1 valid.2.1
      coefficientNonzero⟩

private def RawTermsValid (columns : Nat) (terms : List RawTerm) : Prop :=
  ∀ term ∈ terms, RawTermValid columns term

private theorem decodeTerms_of_valid {columns : Nat} {raw : List RawTerm}
    (valid : RawTermsValid columns raw) :
    ∃ decoded, decodeTerms columns raw = some decoded := by
  induction raw with
  | nil =>
      exact ⟨[], rfl⟩
  | cons head tail inductionHypothesis =>
      have headValid : RawTermValid columns head :=
        valid head (by simp)
      have tailValid : RawTermsValid columns tail := by
        intro term member
        exact valid term (by simp [member])
      rcases decodeTerm_of_valid headValid with ⟨decodedHead, headDecodes⟩
      rcases inductionHypothesis tailValid with ⟨decodedTail, tailDecodes⟩
      unfold decodeTerms at tailDecodes ⊢
      exact ⟨decodedHead :: decodedTail,
        by simp [headDecodes, tailDecodes]⟩

/-- The explicit structural premises are sufficient for the source-row
decoder. No generated label, interval, digest, or satisfaction claim is used. -/
theorem decodeSourceRow_of_valid {raw : RawSourceRow}
    (valid : RawSourceRowValid raw) :
    ∃ decoded, decodeSourceRow raw = some decoded := by
  have aValid : RawTermsValid raw.columns raw.a := by
    intro term member
    exact valid.2.2.2.2 term (by simp [member])
  have bValid : RawTermsValid raw.columns raw.b := by
    intro term member
    exact valid.2.2.2.2 term (by simp [member])
  have cValid : RawTermsValid raw.columns raw.c := by
    intro term member
    exact valid.2.2.2.2 term (by simp [member])
  rcases decodeTerms_of_valid aValid with ⟨a, aDecodes⟩
  rcases decodeTerms_of_valid bValid with ⟨b, bDecodes⟩
  rcases decodeTerms_of_valid cValid with ⟨c, cDecodes⟩
  refine ⟨
    { rows := raw.rows
      columns := raw.columns
      rowsPositive := valid.2.1
      columnsPositive := valid.2.2.1
      sourceRow := ⟨raw.sourceRow, valid.2.2.2.1⟩
      a
      b
      c }, ?_⟩
  simp [decodeSourceRow, valid.1, valid.2.1, valid.2.2.1,
    valid.2.2.2.1, aDecodes, bDecodes, cDecodes]

/-- Successful term decoding preserves the exact artifact word and column. -/
theorem termAsNatTerm_eq_of_decodeTerm {columns : Nat} {raw : RawTerm}
    {decoded : DecodedTerm columns}
    (decodes : decodeTerm columns raw = some decoded) :
    Semantics.termAsNatTerm decoded = (raw.column, raw.coefficient) := by
  by_cases columnInRange : raw.column < columns
  · by_cases coefficientCanonical : raw.coefficient < goldilocksModulus
    · have coefficientNonzero :
          (⟨raw.coefficient, coefficientCanonical⟩ : F) ≠ 0 := by
        intro zero
        unfold decodeTerm at decodes
        rw [dif_pos columnInRange] at decodes
        unfold Decoder.decodeField at decodes
        rw [Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Decoder.decodeField,
          dif_pos coefficientCanonical] at decodes
        simp [zero] at decodes
      rw [decodeTerm_canonical columns raw.column raw.coefficient columnInRange
        coefficientCanonical coefficientNonzero] at decodes
      have decodedEq :
          decoded = canonicalDecodedTerm columns raw.column raw.coefficient
            columnInRange coefficientCanonical coefficientNonzero :=
        (Option.some.inj decodes).symm
      subst decoded
      rfl
    · unfold decodeTerm at decodes
      rw [dif_pos columnInRange] at decodes
      unfold Decoder.decodeField at decodes
      rw [Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Decoder.decodeField,
        dif_neg coefficientCanonical] at decodes
      simp at decodes
  · unfold decodeTerm at decodes
    rw [dif_neg columnInRange] at decodes
    simp at decodes

private theorem termsAsNatTerms_eq_rawTerms_of_decodeTerms
    {columns : Nat} {raw : List RawTerm}
    {decoded : List (DecodedTerm columns)}
    (decodes : decodeTerms columns raw = some decoded) :
    Semantics.termsAsNatTerms decoded = rawTerms raw := by
  induction raw generalizing decoded with
  | nil =>
      simp [decodeTerms] at decodes
      subst decoded
      rfl
  | cons head tail inductionHypothesis =>
      cases headResult : decodeTerm columns head with
      | none =>
          simp [decodeTerms, headResult] at decodes
      | some decodedHead =>
          cases tailResult : decodeTerms columns tail with
          | none =>
              unfold decodeTerms at tailResult
              simp [decodeTerms, headResult, tailResult] at decodes
          | some decodedTail =>
              unfold decodeTerms at tailResult
              simp [decodeTerms, headResult, tailResult] at decodes
              subst decoded
              have tailDecodes :
                  decodeTerms columns tail = some decodedTail := by
                unfold decodeTerms
                exact tailResult
              change
                Semantics.termAsNatTerm decodedHead ::
                    Semantics.termsAsNatTerms decodedTail =
                  (head.column, head.coefficient) :: rawTerms tail
              rw [termAsNatTerm_eq_of_decodeTerm headResult,
                inductionHypothesis tailDecodes]

/-- Every successful source-row decoding is lossless: its executable R1CS row
is exactly the ordered proof-free A/B/C projection, including duplicates. -/
theorem sourceRowToRow_eq_rawRow_of_decode {raw : RawSourceRow}
    {decoded : DecodedSourceRow}
    (decodes : decodeSourceRow raw = some decoded) :
    Semantics.sourceRowToRow decoded = rawRow raw := by
  by_cases version : raw.schemaVersion = supportedSchemaVersion
  · by_cases rowsPositive : 0 < raw.rows
    · by_cases columnsPositive : 0 < raw.columns
      · by_cases rowInRange : raw.sourceRow < raw.rows
        · cases aResult : decodeTerms raw.columns raw.a with
          | none =>
              simp [decodeSourceRow, version, rowsPositive, columnsPositive,
                rowInRange, aResult] at decodes
          | some a =>
              cases bResult : decodeTerms raw.columns raw.b with
              | none =>
                  simp [decodeSourceRow, version, rowsPositive,
                    columnsPositive, rowInRange, aResult, bResult] at decodes
              | some b =>
                  cases cResult : decodeTerms raw.columns raw.c with
                  | none =>
                      simp [decodeSourceRow, version, rowsPositive,
                        columnsPositive, rowInRange, aResult, bResult,
                        cResult] at decodes
                  | some c =>
                      simp [decodeSourceRow, version, rowsPositive,
                        columnsPositive, rowInRange, aResult, bResult,
                        cResult] at decodes
                      subst decoded
                      simp [Semantics.sourceRowToRow, rawRow,
                        termsAsNatTerms_eq_rawTerms_of_decodeTerms aResult,
                        termsAsNatTerms_eq_rawTerms_of_decodeTerms bResult,
                        termsAsNatTerms_eq_rawTerms_of_decodeTerms cResult]
        · simp [decodeSourceRow, version, rowsPositive, columnsPositive,
            rowInRange] at decodes
      · simp [decodeSourceRow, version, rowsPositive, columnsPositive] at decodes
    · simp [decodeSourceRow, version, rowsPositive] at decodes
  · simp [decodeSourceRow, version] at decodes

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.SourceDecodeBridge
