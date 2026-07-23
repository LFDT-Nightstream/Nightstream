import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.SelectiveArtifactPairs.Core
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.SourceDecodeBridge

/-!
Kernel bridge from proof-free combined-NC selective pair certificates to the
typed decoder and compiler semantics.

Owns: successful fail-closed decoding from structural validity; lossless port
and source-linear-form evaluation; and one-record transport from the compact
coefficient certificate to `RewriteProvenanceMatches` or
`RetainedProvenanceMatches`.

Does not own: generated certificate truth, selected-row satisfaction, source
program execution, selector enforcement, transcript order, parent or raw-child
authority, commitment binding, costs, or row removal.

Emits constraints: none.

Assurance tier: model-level.  Generated leaves may prove the explicit compact
certificate predicates, but this file performs no closed computation over a
generated collection.
-/

/-!
| Stable stage path | Obligation | Authority class |
|---|---|---|
| `f_prime.pi_ccs_nc.delayed.combined.selective_pairs.decode` | Lift compact certificate facts into typed row and source-obligation equalities. | checked artifact |

-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.SelectiveArtifactPairs

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc
open Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized
open Decoder
open Semantics
open SelectiveCompilerBridge

private theorem field_ne_zero_of_word_ne_zero {value : Nat}
    (canonical : value < goldilocksModulus) (wordNonzero : value ≠ 0) :
    (⟨value, canonical⟩ : F) ≠ 0 := by
  intro equality
  apply wordNonzero
  simpa using congrArg Fin.val equality

private theorem fieldResidue_val (value : F) :
    fieldResidue value.val = value := by
  apply Fin.ext
  simp [fieldResidue, Nat.mod_eq_of_lt value.isLt]

private theorem decodeField_value {raw : Nat} {decoded : F}
    (decodes : decodeField raw = some decoded) :
    decoded = fieldResidue raw := by
  unfold decodeField at decodes
  unfold Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Decoder.decodeField at decodes
  split at decodes
  next canonical =>
    simp only [Option.some.injEq] at decodes
    subst decoded
    apply Fin.ext
    simp [fieldResidue, Nat.mod_eq_of_lt canonical]
  next notCanonical => simp at decodes

/-! ## Structural decoding -/

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

theorem decodeGeometricRun_of_valid {columns : Nat} {raw : RawGeometricRun}
    (valid : RawGeometricRunValid columns raw) :
    ∃ decoded, decodeGeometricRun columns raw = some decoded := by
  have initialNonzero : (⟨raw.initial, valid.2.2.1⟩ : F) ≠ 0 :=
    field_ne_zero_of_word_ne_zero valid.2.2.1 valid.2.2.2.1
  have ratioNonzero : (⟨raw.ratio, valid.2.2.2.2.1⟩ : F) ≠ 0 :=
    field_ne_zero_of_word_ne_zero valid.2.2.2.2.1 valid.2.2.2.2.2
  refine ⟨
    { columnStart := raw.columnStart
      length := raw.length
      lengthPositive := valid.1
      endBound := valid.2.1
      initial := ⟨raw.initial, valid.2.2.1⟩
      ratio := ⟨raw.ratio, valid.2.2.2.2.1⟩
      initialNonzero
      ratioNonzero }, ?_⟩
  unfold decodeGeometricRun decodeField
  rw [dif_pos valid.1, dif_pos valid.2.1]
  unfold Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Decoder.decodeField
  rw [dif_pos valid.2.2.1, dif_pos valid.2.2.2.2.1]
  simp [initialNonzero, ratioNonzero]

private theorem decodeTerms_of_valid {columns : Nat} {raw : List RawTerm}
    (valid : ∀ term ∈ raw, RawTermValid columns term) :
    ∃ decoded, decodeTerms columns raw = some decoded := by
  induction raw with
  | nil => exact ⟨[], rfl⟩
  | cons head tail inductionHypothesis =>
      rcases decodeTerm_of_valid (valid head (by simp)) with
        ⟨decodedHead, headDecodes⟩
      rcases inductionHypothesis (by
        intro term member
        exact valid term (by simp [member])) with ⟨decodedTail, tailDecodes⟩
      unfold decodeTerms at tailDecodes ⊢
      exact ⟨decodedHead :: decodedTail, by
        simp [headDecodes, tailDecodes]⟩

private theorem decodeRuns_of_valid {columns : Nat}
    {raw : List RawGeometricRun}
    (valid : ∀ run ∈ raw, RawGeometricRunValid columns run) :
    ∃ decoded, raw.mapM (decodeGeometricRun columns) = some decoded := by
  induction raw with
  | nil => exact ⟨[], rfl⟩
  | cons head tail inductionHypothesis =>
      rcases decodeGeometricRun_of_valid (valid head (by simp)) with
        ⟨decodedHead, headDecodes⟩
      rcases inductionHypothesis (by
        intro run member
        exact valid run (by simp [member])) with ⟨decodedTail, tailDecodes⟩
      exact ⟨decodedHead :: decodedTail, by simp [headDecodes, tailDecodes]⟩

theorem decodePort_of_valid {columns : Nat} {raw : RawPort}
    (valid : RawPortValid columns raw) :
    ∃ decoded, decodePort columns raw = some decoded := by
  rcases decodeTerms_of_valid valid.1 with ⟨explicit, explicitDecodes⟩
  rcases decodeRuns_of_valid valid.2 with ⟨geometric, geometricDecodes⟩
  exact ⟨⟨explicit, geometric⟩, by
    simp [decodePort, explicitDecodes, geometricDecodes]⟩

private theorem decodePorts_of_valid {columns : Nat} {raw : List RawPort}
    (valid : ∀ port ∈ raw, RawPortValid columns port) :
    ∃ decoded, raw.mapM (decodePort columns) = some decoded := by
  induction raw with
  | nil => exact ⟨[], rfl⟩
  | cons head tail inductionHypothesis =>
      rcases decodePort_of_valid (valid head (by simp)) with
        ⟨decodedHead, headDecodes⟩
      rcases inductionHypothesis (by
        intro port member
        exact valid port (by simp [member])) with ⟨decodedTail, tailDecodes⟩
      exact ⟨decodedHead :: decodedTail, by simp [headDecodes, tailDecodes]⟩

private theorem length_eq_of_mapM {alpha beta : Type}
    (decode : alpha → Option beta) {raw : List alpha} {decoded : List beta}
    (decodes : raw.mapM decode = some decoded) :
    decoded.length = raw.length := by
  induction raw generalizing decoded with
  | nil => simp at decodes; subst decoded; rfl
  | cons head tail inductionHypothesis =>
      cases headResult : decode head with
      | none => simp [headResult] at decodes
      | some decodedHead =>
          cases tailResult : tail.mapM decode with
          | none => simp [headResult, tailResult] at decodes
          | some decodedTail =>
              simp [headResult, tailResult] at decodes
              subst decoded
              simp [inductionHypothesis tailResult]

theorem decodeEmittedRow_of_valid {raw : RawEmittedRow}
    (valid : RawEmittedRowValid raw) :
    ∃ decoded, decodeEmittedRow raw = some decoded := by
  have rowsPositive : 0 < raw.rows := by
    rw [valid.2.1]
    decide
  have columnsPositive : 0 < raw.columns := by
    rw [valid.2.2.1]
    decide
  rcases decodePorts_of_valid valid.2.2.2.2.2 with ⟨ports, portsDecodes⟩
  have portsLength : ports.length = selectivePortCount := by
    rw [length_eq_of_mapM (decodePort raw.columns) portsDecodes,
      valid.2.2.2.2.1]
  refine ⟨
    { rows := raw.rows
      columns := raw.columns
      rowsPositive
      columnsPositive
      emittedRow := ⟨raw.emittedRow, valid.2.2.2.1⟩
      runIndex := raw.runIndex
      family := raw.family
      arm := raw.arm
      ports := fun port => ports.get ⟨port.val, by
        rw [portsLength]
        exact port.isLt⟩ }, ?_⟩
  simp [decodeEmittedRow, valid.1, rowsPositive, columnsPositive,
    valid.2.2.2.1, portsDecodes, portsLength]

theorem decodeLinearCombination_of_valid {columns : Nat}
    {raw : RawLinearCombination}
    (valid : RawLinearCombinationValid columns raw) :
    ∃ decoded, decodeLinearCombination columns raw = some decoded := by
  rcases decodeTerms_of_valid valid.2 with ⟨terms, termsDecodes⟩
  refine ⟨{
    constant := ⟨raw.constant, valid.1⟩
    terms }, ?_⟩
  unfold decodeLinearCombination decodeField
  unfold Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Decoder.decodeField
  rw [dif_pos valid.1]
  simp [termsDecodes]

theorem decodeProductFactor_of_valid {columns : Nat} {raw : RawProductFactor}
    (valid : RawProductFactorValid columns raw) :
    ∃ decoded, decodeProductFactor columns raw = some decoded := by
  rcases decodeLinearCombination_of_valid valid.1 with
    ⟨left, leftDecodes⟩
  rcases decodeLinearCombination_of_valid valid.2.1 with
    ⟨right, rightDecodes⟩
  refine ⟨{
    left
    right
    coefficient := ⟨raw.coefficient, valid.2.2⟩ }, ?_⟩
  unfold decodeProductFactor decodeField
  unfold Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Decoder.decodeField
  rw [dif_pos valid.2.2]
  simp [leftDecodes, rightDecodes]

private theorem decodeFactors_of_valid {columns : Nat}
    {raw : List RawProductFactor}
    (valid : ∀ factor ∈ raw, RawProductFactorValid columns factor) :
    ∃ decoded, raw.mapM (decodeProductFactor columns) = some decoded := by
  induction raw with
  | nil => exact ⟨[], rfl⟩
  | cons head tail inductionHypothesis =>
      rcases decodeProductFactor_of_valid (valid head (by simp)) with
        ⟨decodedHead, headDecodes⟩
      rcases inductionHypothesis (by
        intro factor member
        exact valid factor (by simp [member])) with ⟨decodedTail, tailDecodes⟩
      exact ⟨decodedHead :: decodedTail, by simp [headDecodes, tailDecodes]⟩

private theorem decodeRewriteOutput_of_valid {columns : Nat}
    {raw : RawRewriteOutput} (valid : RawRewriteOutputValid columns raw) :
    ∃ decoded, decodeRewriteOutput columns raw = some decoded := by
  cases raw with
  | source value =>
      rcases decodeLinearCombination_of_valid valid with
        ⟨decoded, decodes⟩
      exact ⟨DecodedRewriteOutput.source decoded,
        by simp [decodeRewriteOutput, decodes]⟩
  | derivedProductSum compilerIndex =>
      exact ⟨DecodedRewriteOutput.derivedProductSum compilerIndex, rfl⟩

theorem decodeRewriteStep_of_valid {raw : RawRewriteStep}
    (valid : RawRewriteStepValid raw) :
    ∃ decoded,
      decodeRewriteStep Metadata.sourceRelationRows
        Metadata.sourceRelationColumns Metadata.finalRelationRows raw =
          some decoded := by
  rcases valid with
    ⟨emittedBound, sourceRanges, outputValid, baseValid, factorsValid,
      _factorsLength⟩
  rcases decodeRewriteOutput_of_valid outputValid with
    ⟨output, outputDecodes⟩
  rcases decodeLinearCombination_of_valid baseValid with
    ⟨base, baseDecodes⟩
  rcases decodeFactors_of_valid factorsValid with
    ⟨factors, factorsDecodes⟩
  refine ⟨
    { sourceRowCount := Metadata.sourceRelationRows
      finalRowCount := Metadata.finalRelationRows
      emittedRow := raw.emittedRow
      emittedBound
      rewriteId := raw.rewriteId
      kind := raw.kind
      sourceRows := raw.sourceRows
      sourceRanges
      output
      base
      previous := raw.previous
      factors }, ?_⟩
  unfold decodeRewriteStep
  rw [dif_pos emittedBound, dif_pos sourceRanges]
  simp [outputDecodes, baseDecodes, factorsDecodes]

theorem decodeRetainedStep_of_valid {raw : RawRetainedStep}
    (valid : RawRetainedStepValid raw) :
    ∃ decoded,
      decodeRetainedStep Metadata.sourceRelationRows
        Metadata.sourceRelationColumns Metadata.finalRelationRows raw =
          some decoded := by
  rcases decodeLinearCombination_of_valid valid.2.2.1 with
    ⟨a, aDecodes⟩
  rcases decodeLinearCombination_of_valid valid.2.2.2.1 with
    ⟨b, bDecodes⟩
  rcases decodeLinearCombination_of_valid valid.2.2.2.2 with
    ⟨c, cDecodes⟩
  refine ⟨
    { sourceRowCount := Metadata.sourceRelationRows
      finalRowCount := Metadata.finalRelationRows
      emittedRow := raw.emittedRow
      emittedBound := valid.1
      sourceRow := raw.sourceRow
      sourceBound := valid.2.1
      a
      b
      c }, ?_⟩
  simp [decodeRetainedStep, valid.1, valid.2.1, aDecodes, bDecodes,
    cDecodes]

/-! ## Lossless evaluation -/

private theorem termsAsNatTerms_eq_of_decodeTerms {columns : Nat}
    {raw : List RawTerm} {decoded : List (DecodedTerm columns)}
    (decodes : decodeTerms columns raw = some decoded) :
    termsAsNatTerms decoded = raw.map fun term =>
      (term.column, term.coefficient) := by
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
                termAsNatTerm decodedHead :: termsAsNatTerms decodedTail =
                  (head.column, head.coefficient) ::
                    (tail.map fun term => (term.column, term.coefficient))
              rw [SourceDecodeBridge.termAsNatTerm_eq_of_decodeTerm headResult,
                inductionHypothesis tailDecodes]

private theorem rawTermsLinearForm_eq (raw : List RawTerm) :
    natTermsLinearForm
        (raw.map fun term => (term.column, term.coefficient)) =
      raw.flatMap rawTermLinearForm := by
  induction raw with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      change
        (head.column, fieldResidue head.coefficient) ::
            natTermsLinearForm
              (tail.map fun term => (term.column, term.coefficient)) =
          (head.column, fieldResidue head.coefficient) ::
            tail.flatMap rawTermLinearForm
      rw [inductionHypothesis]

private theorem decodedTermsLinearForm_eq {columns : Nat}
    {raw : List RawTerm} {decoded : List (DecodedTerm columns)}
    (decodes : decodeTerms columns raw = some decoded) :
    natTermsLinearForm (termsAsNatTerms decoded) =
      raw.flatMap rawTermLinearForm := by
  rw [termsAsNatTerms_eq_of_decodeTerms decodes]
  exact rawTermsLinearForm_eq raw

private theorem decodedRunLinearForm_eq {columns : Nat}
    {raw : RawGeometricRun} {decoded : DecodedGeometricRun columns}
    (decodes : decodeGeometricRun columns raw = some decoded) :
    natTermsLinearForm (expandedRunNatTerms decoded) =
      rawGeometricRunLinearForm raw := by
  by_cases lengthPositive : 0 < raw.length
  · by_cases endBound : raw.columnStart + raw.length ≤ columns
    · cases initialResult : decodeField raw.initial with
      | none =>
          simp [decodeGeometricRun, lengthPositive, endBound, initialResult]
            at decodes
      | some initial =>
          cases ratioResult : decodeField raw.ratio with
          | none =>
              simp [decodeGeometricRun, lengthPositive, endBound,
                initialResult, ratioResult] at decodes
          | some ratio =>
              by_cases initialNonzero : initial ≠ 0
              · by_cases ratioNonzero : ratio ≠ 0
                · simp [decodeGeometricRun, lengthPositive, endBound,
                    initialResult, ratioResult, initialNonzero, ratioNonzero]
                    at decodes
                  subst decoded
                  have initialEq := decodeField_value initialResult
                  have ratioEq := decodeField_value ratioResult
                  subst initial
                  subst ratio
                  simp [expandedRunNatTerms, rawGeometricRunLinearForm,
                    natTermsLinearForm, fieldResidue_val,
                    DecodedGeometricRun.column]
                · unfold decodeGeometricRun at decodes
                  rw [dif_pos lengthPositive, dif_pos endBound,
                    initialResult] at decodes
                  rw [ratioResult] at decodes
                  simp [Option.bind, initialNonzero, ratioNonzero] at decodes
                  rcases decodes with ⟨ratioWitness, _⟩
                  exact ratioWitness.elim
              · unfold decodeGeometricRun at decodes
                rw [dif_pos lengthPositive, dif_pos endBound,
                  initialResult] at decodes
                rw [ratioResult] at decodes
                simp [Option.bind, initialNonzero] at decodes
                rcases decodes with ⟨initialWitness, _, _⟩
                exact initialWitness.elim
    · simp [decodeGeometricRun, lengthPositive, endBound] at decodes
  · simp [decodeGeometricRun, lengthPositive] at decodes

private theorem decodedRunsLinearForm_eq {columns : Nat}
    {raw : List RawGeometricRun}
    {decoded : List (DecodedGeometricRun columns)}
    (decodes : raw.mapM (decodeGeometricRun columns) = some decoded) :
    (decoded.flatMap fun run => natTermsLinearForm (expandedRunNatTerms run)) =
      raw.flatMap rawGeometricRunLinearForm := by
  induction raw generalizing decoded with
  | nil => simp at decodes; subst decoded; rfl
  | cons head tail inductionHypothesis =>
      cases headResult : decodeGeometricRun columns head with
      | none => simp [headResult] at decodes
      | some decodedHead =>
          cases tailResult : tail.mapM (decodeGeometricRun columns) with
          | none => simp [headResult, tailResult] at decodes
          | some decodedTail =>
              simp [headResult, tailResult] at decodes
              subst decoded
              simp only [List.flatMap_cons]
              rw [decodedRunLinearForm_eq headResult,
                inductionHypothesis tailResult]

private theorem natTermsLinearForm_append (left right : List (Nat × Nat)) :
    natTermsLinearForm (left ++ right) =
      natTermsLinearForm left ++ natTermsLinearForm right := by
  simp [natTermsLinearForm]

private theorem natTermsLinearForm_flatMapRuns {columns : Nat}
    (runs : List (DecodedGeometricRun columns)) :
    natTermsLinearForm (runs.flatMap expandedRunNatTerms) =
      runs.flatMap fun run => natTermsLinearForm (expandedRunNatTerms run) := by
  induction runs with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      simp only [List.flatMap_cons, natTermsLinearForm_append]
      rw [inductionHypothesis]

theorem portValue_eq_evalRawPort {columns : Nat} {raw : RawPort}
    {decoded : DecodedPort columns}
    (decodes : decodePort columns raw = some decoded)
    (assignment : Nat → Nat) :
    portValue decoded assignment =
      evalLinearForm (fun column => fieldResidue (assignment column))
        (rawPortLinearForm raw) := by
  unfold decodePort at decodes
  cases explicitResult : decodeTerms columns raw.explicit with
  | none => simp [explicitResult] at decodes
  | some explicit =>
      cases geometricResult : raw.geometric.mapM (decodeGeometricRun columns) with
      | none => simp [explicitResult, geometricResult] at decodes
      | some geometric =>
          simp [explicitResult, geometricResult] at decodes
          subst decoded
          unfold portValue expandedNatTerms rawPortLinearForm
          rw [← evalNatTermsLinearForm assignment,
            natTermsLinearForm_append, eval_append,
            decodedTermsLinearForm_eq explicitResult,
            natTermsLinearForm_flatMapRuns,
            decodedRunsLinearForm_eq geometricResult, eval_append]

private theorem portValuesAt_of_decodePorts {columns : Nat}
    {raw : List RawPort} {decoded : List (DecodedPort columns)}
    (decodes : raw.mapM (decodePort columns) = some decoded)
    (assignment : Nat → Nat) (index : Nat) :
    (decoded[index]?.map fun port => portValue port assignment) =
      (raw[index]?.map fun port =>
        evalLinearForm (fun column => fieldResidue (assignment column))
          (rawPortLinearForm port)) := by
  induction raw generalizing decoded assignment index with
  | nil => simp at decodes; subst decoded; simp
  | cons head tail inductionHypothesis =>
      cases headResult : decodePort columns head with
      | none => simp [headResult] at decodes
      | some decodedHead =>
          cases tailResult : tail.mapM (decodePort columns) with
          | none => simp [headResult, tailResult] at decodes
          | some decodedTail =>
              simp [headResult, tailResult] at decodes
              subst decoded
              cases index with
              | zero => simp [portValue_eq_evalRawPort headResult]
              | succ index =>
                  simpa using inductionHypothesis tailResult assignment index

theorem emittedPoint_eq_evalRawPorts {raw : RawEmittedRow}
    {decoded : DecodedEmittedRow}
    (decodes : decodeEmittedRow raw = some decoded)
    (assignment : Nat → Nat) :
    emittedPoint decoded assignment = fun port =>
      evalLinearForm (fun column => fieldResidue (assignment column))
        (rawEmittedPortLinearForm raw port) := by
  by_cases version : raw.schemaVersion = supportedSchemaVersion
  · by_cases rowsPositive : 0 < raw.rows
    · by_cases columnsPositive : 0 < raw.columns
      · by_cases rowInRange : raw.emittedRow < raw.rows
        · cases portsResult : raw.ports.mapM (decodePort raw.columns) with
          | none =>
              simp [decodeEmittedRow, version, rowsPositive, columnsPositive,
                rowInRange, portsResult] at decodes
          | some ports =>
              by_cases portCount : ports.length = selectivePortCount
              · simp [decodeEmittedRow, version, rowsPositive,
                  columnsPositive, rowInRange, portsResult, portCount] at decodes
                subst decoded
                funext port
                unfold emittedPoint DecodedEmittedRow.port
                have decodedPortLt : port.val < ports.length := by
                  rw [portCount]
                  exact port.isLt
                have rawPortLt : port.val < raw.ports.length := by
                  rw [← length_eq_of_mapM (decodePort raw.columns) portsResult,
                    portCount]
                  exact port.isLt
                have atPort := portValuesAt_of_decodePorts portsResult
                  assignment port.val
                rw [List.getElem?_eq_getElem decodedPortLt,
                  List.getElem?_eq_getElem rawPortLt] at atPort
                simp only [Option.map_some, Option.some.injEq] at atPort
                unfold rawEmittedPortLinearForm
                rw [List.getElem?_eq_getElem rawPortLt]
                simpa only using atPort
              · simp [decodeEmittedRow, version, rowsPositive,
                  columnsPositive, rowInRange, portsResult, portCount] at decodes
        · simp [decodeEmittedRow, version, rowsPositive, columnsPositive,
            rowInRange] at decodes
      · simp [decodeEmittedRow, version, rowsPositive, columnsPositive]
          at decodes
    · simp [decodeEmittedRow, version, rowsPositive] at decodes
  · simp [decodeEmittedRow, version] at decodes

private theorem fieldResidue_add (left right : Nat) :
    fieldResidue (left + right) = fieldResidue left + fieldResidue right := by
  apply Fin.ext
  simp [fieldResidue, Fin.val_add, Nat.add_mod]

private theorem fieldResidue_mul (left right : Nat) :
    fieldResidue (left * right) = fieldResidue left * fieldResidue right := by
  apply Fin.ext
  simp [fieldResidue, Fin.val_mul, Nat.mul_mod]

private theorem evalSubstituteCompiler (assignment : Nat → Nat) :
    ∀ terms,
      evalLinearForm (fun column => fieldResidue (assignment column))
          (substituteLinearTerms compilerLinearForms terms) =
        fieldResidue
          (lcEval (SourceAssignment.compilerAssignment assignment) terms) := by
  intro terms
  induction terms with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      change
        evalLinearForm (fun column => fieldResidue (assignment column))
            (scaleLinearForm (fieldResidue head.2)
                (compilerLinearForms head.1) ++
              substituteLinearTerms compilerLinearForms tail) =
          fieldResidue
            (lcEval (SourceAssignment.compilerAssignment assignment)
              (head :: tail))
      rw [eval_append, eval_scale, evalCompilerLinearForm,
        inductionHypothesis]
      change
        fieldResidue head.2 *
              fieldResidue
                (SourceAssignment.compilerAssignment assignment head.1) +
            fieldResidue
              (lcEval (SourceAssignment.compilerAssignment assignment) tail) =
          fieldResidue
            (lcEval (SourceAssignment.compilerAssignment assignment)
              (head :: tail))
      rw [← fieldResidue_mul, ← fieldResidue_add]
      have modulusEq : goldilocksP = goldilocksModulus := by rfl
      apply Fin.ext
      simp only [fieldResidue]
      rw [Program.lcEval_eq_raw_mod, Program.lcEval_eq_raw_mod]
      simp only [Program.rawLcEval]
      rw [modulusEq]
      simp only [Nat.add_mod, Nat.mod_mod]

theorem evalSourceLinearForm (assignment : Nat → Nat)
    (linear : RawLinearCombination) :
    evalLinearForm (fun column => fieldResidue (assignment column))
        (sourceLinearForm linear) =
      fieldResidue
        (lcEval (SourceAssignment.compilerAssignment assignment)
          (SourceAssignment.RawLinearCombination.programTerms linear)) := by
  exact evalSubstituteCompiler assignment
    (SourceAssignment.RawLinearCombination.programTerms linear)

private theorem decodeField_val {raw : Nat} {decoded : F}
    (decodes : decodeField raw = some decoded) :
    decoded.val = raw := by
  unfold decodeField at decodes
  unfold Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Decoder.decodeField at decodes
  split at decodes
  next canonical =>
    simp only [Option.some.injEq] at decodes
    subst decoded
    rfl
  next notCanonical => simp at decodes

/-- Lossless evaluation of one decoded source linear combination on the same
source assignment.  Unlike the emitted-row substitution theorem below, this
lemma performs no compiler-assignment reconstruction. -/
theorem linearCombinationValue_eq_raw
    {columns : Nat} {raw : RawLinearCombination}
    {decoded : DecodedLinearCombination columns}
    (decodes : decodeLinearCombination columns raw = some decoded)
    (assignment : Nat → Nat) :
    linearCombinationValue decoded assignment =
      fieldResidue
        (lcEval assignment
          (SourceAssignment.RawLinearCombination.programTerms raw)) := by
  unfold decodeLinearCombination at decodes
  cases constantResult : decodeField raw.constant with
  | none => simp [constantResult] at decodes
  | some constant =>
      cases termsResult : decodeTerms columns raw.terms with
      | none => simp [constantResult, termsResult] at decodes
      | some terms =>
          simp [constantResult, termsResult] at decodes
          subst decoded
          unfold linearCombinationValue linearCombinationTerms
          rw [termsAsNatTerms_eq_of_decodeTerms termsResult,
            decodeField_val constantResult]
          unfold SourceAssignment.RawLinearCombination.programTerms
          have termsMapEq :
              raw.terms.map (fun term =>
                (term.column, term.coefficient)) =
                raw.terms.map SourceAssignment.RawTerm.asNatTerm := by
            apply congrArg (fun mapper => raw.terms.map mapper)
            funext term
            rfl
          rw [termsMapEq]
          by_cases constantZero : raw.constant = 0
          · simp [constantZero, lcEval, fieldResidue, Nat.mod_mod]
          · simp [constantZero]

/-- Lossless evaluation of one decoded product factor on the same source
assignment. -/
theorem productFactorValue_eq_raw
    {columns : Nat} {raw : RawProductFactor}
    {decoded : DecodedProductFactor columns}
    (decodes : decodeProductFactor columns raw = some decoded)
    (assignment : Nat → Nat) :
    productFactorValue decoded assignment =
      fieldResidue raw.coefficient *
        fieldResidue
          (lcEval assignment
            (SourceAssignment.RawLinearCombination.programTerms raw.left)) *
        fieldResidue
          (lcEval assignment
            (SourceAssignment.RawLinearCombination.programTerms raw.right)) := by
  unfold decodeProductFactor at decodes
  cases leftResult : decodeLinearCombination columns raw.left with
  | none => simp [leftResult] at decodes
  | some left =>
      cases rightResult : decodeLinearCombination columns raw.right with
      | none => simp [leftResult, rightResult] at decodes
      | some right =>
          cases coefficientResult : decodeField raw.coefficient with
          | none => simp [leftResult, rightResult, coefficientResult] at decodes
          | some coefficient =>
              simp [leftResult, rightResult, coefficientResult] at decodes
              subst decoded
              unfold productFactorValue
              rw [linearCombinationValue_eq_raw leftResult,
                linearCombinationValue_eq_raw rightResult,
                decodeField_value coefficientResult]

theorem linearCombinationValue_eq_evalSourceLinearForm
    {columns : Nat} {raw : RawLinearCombination}
    {decoded : DecodedLinearCombination columns}
    (decodes : decodeLinearCombination columns raw = some decoded)
    (assignment : Nat → Nat) :
    linearCombinationValue decoded
        (SourceAssignment.compilerAssignment assignment) =
      evalLinearForm (fun column => fieldResidue (assignment column))
        (sourceLinearForm raw) := by
  unfold decodeLinearCombination at decodes
  cases constantResult : decodeField raw.constant with
  | none => simp [constantResult] at decodes
  | some constant =>
      cases termsResult : decodeTerms columns raw.terms with
      | none => simp [constantResult, termsResult] at decodes
      | some terms =>
          simp [constantResult, termsResult] at decodes
          subst decoded
          rw [evalSourceLinearForm]
          unfold linearCombinationValue linearCombinationTerms
          rw [termsAsNatTerms_eq_of_decodeTerms termsResult,
            decodeField_val constantResult]
          unfold SourceAssignment.RawLinearCombination.programTerms
          have termsMapEq :
              raw.terms.map (fun term => (term.column, term.coefficient)) =
                raw.terms.map SourceAssignment.RawTerm.asNatTerm := by
            apply congrArg (fun mapper => raw.terms.map mapper)
            funext term
            rfl
          rw [termsMapEq]
          by_cases constantZero : raw.constant = 0
          · simp [constantZero, lcEval, fieldResidue, Nat.mod_mod]
          · simp [constantZero]

theorem evalDerivedLinearForm (assignment : Nat → Nat)
    (compilerIndex : Nat) :
    evalLinearForm (fun column => fieldResidue (assignment column))
        (derivedLinearForm compilerIndex) =
      SourceAssignment.derivedValue assignment compilerIndex := by
  cases hslot : SourceAssignment.derivedSlot? compilerIndex with
  | none =>
      simp [derivedLinearForm, SourceAssignment.derivedValue, hslot,
        evalLinearForm]
  | some slot =>
      simp only [derivedLinearForm, SourceAssignment.derivedValue, hslot]
      rw [evalNatTermsLinearForm]

private theorem outputValue_eq_evalOutputLinearForm
    {columns : Nat} {raw : RawRewriteOutput}
    {decoded : DecodedRewriteOutput columns}
    (decodes : decodeRewriteOutput columns raw = some decoded)
    (assignment : Nat → Nat) :
    rewriteOutputValue (SourceAssignment.compilerAssignment assignment)
        (SourceAssignment.derivedValue assignment) decoded =
      evalLinearForm (fun column => fieldResidue (assignment column))
        (outputLinearForm raw) := by
  cases raw with
  | source value =>
      unfold decodeRewriteOutput at decodes
      cases valueResult : decodeLinearCombination columns value with
      | none => simp [valueResult] at decodes
      | some decodedValue =>
          simp [valueResult] at decodes
          subst decoded
          exact linearCombinationValue_eq_evalSourceLinearForm
            valueResult assignment
  | derivedProductSum compilerIndex =>
      simp [decodeRewriteOutput] at decodes
      subst decoded
      exact (evalDerivedLinearForm assignment compilerIndex).symm

private theorem previousValue_eq_evalPreviousLinearForm
    (assignment : Nat → Nat) (previous : Option Nat) :
    rewritePreviousValue (SourceAssignment.derivedValue assignment) previous =
      evalLinearForm (fun column => fieldResidue (assignment column))
        (previousLinearForm previous) := by
  cases previous with
  | none => rfl
  | some compilerIndex =>
      exact (evalDerivedLinearForm assignment compilerIndex).symm

private theorem productFactorValues_eq
    {columns : Nat} {raw : RawProductFactor}
    {decoded : DecodedProductFactor columns}
    (decodes : decodeProductFactor columns raw = some decoded)
    (assignment : Nat → Nat) :
    factorLeftValue (SourceAssignment.compilerAssignment assignment) decoded =
        evalLinearForm (fun column => fieldResidue (assignment column))
          (factorLeftLinearForm raw) ∧
      factorRightValue (SourceAssignment.compilerAssignment assignment) decoded =
        evalLinearForm (fun column => fieldResidue (assignment column))
          (factorRightLinearForm raw) := by
  unfold decodeProductFactor at decodes
  cases leftResult : decodeLinearCombination columns raw.left with
  | none => simp [leftResult] at decodes
  | some left =>
      cases rightResult : decodeLinearCombination columns raw.right with
      | none => simp [leftResult, rightResult] at decodes
      | some right =>
          cases coefficientResult : decodeField raw.coefficient with
          | none => simp [leftResult, rightResult, coefficientResult] at decodes
          | some coefficient =>
              simp [leftResult, rightResult, coefficientResult] at decodes
              subst decoded
              constructor
              · unfold factorLeftValue factorLeftLinearForm
                rw [eval_scale, ←
                  linearCombinationValue_eq_evalSourceLinearForm
                    leftResult assignment,
                  decodeField_value coefficientResult]
              · unfold factorRightValue factorRightLinearForm
                exact linearCombinationValue_eq_evalSourceLinearForm
                  rightResult assignment

private theorem factorValuesAt_eq {columns : Nat}
    {raw : List RawProductFactor}
    {decoded : List (DecodedProductFactor columns)}
    (decodes : raw.mapM (decodeProductFactor columns) = some decoded)
    (assignment : Nat → Nat) (index : Nat) :
    factorLeftValueAt (SourceAssignment.compilerAssignment assignment)
        decoded index =
        evalLinearForm (fun column => fieldResidue (assignment column))
          (factorLeftLinearFormAt raw index) ∧
      factorRightValueAt (SourceAssignment.compilerAssignment assignment)
        decoded index =
        evalLinearForm (fun column => fieldResidue (assignment column))
          (factorRightLinearFormAt raw index) := by
  induction raw generalizing decoded assignment index with
  | nil => simp at decodes; subst decoded; simp [factorLeftValueAt,
      factorRightValueAt, factorLeftLinearFormAt, factorRightLinearFormAt,
      evalLinearForm]
  | cons head tail inductionHypothesis =>
      cases headResult : decodeProductFactor columns head with
      | none => simp [headResult] at decodes
      | some decodedHead =>
          cases tailResult : tail.mapM (decodeProductFactor columns) with
          | none => simp [headResult, tailResult] at decodes
          | some decodedTail =>
              simp [headResult, tailResult] at decodes
              subst decoded
              cases index with
              | zero =>
                  simpa [factorLeftValueAt, factorRightValueAt,
                    factorLeftLinearFormAt, factorRightLinearFormAt] using
                    productFactorValues_eq headResult assignment
              | succ index =>
                  simpa [factorLeftValueAt, factorRightValueAt,
                    factorLeftLinearFormAt, factorRightLinearFormAt] using
                    inductionHypothesis tailResult assignment index

private theorem fin13_cases {predicate : Fin 13 → Prop}
    (case0 : predicate 0) (case1 : predicate 1) (case2 : predicate 2)
    (case3 : predicate 3) (case4 : predicate 4) (case5 : predicate 5)
    (case6 : predicate 6) (case7 : predicate 7) (case8 : predicate 8)
    (case9 : predicate 9) (case10 : predicate 10)
    (case11 : predicate 11) (case12 : predicate 12) :
    ∀ index, predicate index := by
  intro index
  refine Fin.cases case0 ?_ index
  intro index
  refine Fin.cases case1 ?_ index
  intro index
  refine Fin.cases case2 ?_ index
  intro index
  refine Fin.cases case3 ?_ index
  intro index
  refine Fin.cases case4 ?_ index
  intro index
  refine Fin.cases case5 ?_ index
  intro index
  refine Fin.cases case6 ?_ index
  intro index
  refine Fin.cases case7 ?_ index
  intro index
  refine Fin.cases case8 ?_ index
  intro index
  refine Fin.cases case9 ?_ index
  intro index
  refine Fin.cases case10 ?_ index
  intro index
  refine Fin.cases case11 ?_ index
  intro index
  have valueZero : index.val = 0 :=
    Nat.eq_zero_of_le_zero (Nat.le_of_lt_succ index.isLt)
  have indexZero : index = ⟨0, by decide⟩ := by
    apply Fin.ext
    exact valueZero
  subst index
  simpa using case12

private theorem selectorLinearForm_eq_one (assignment : Nat → Nat)
    (selectorOne : assignment Metadata.steadySelectorColumn = 1) :
    evalLinearForm (fun column => fieldResidue (assignment column))
      steadySelectorLinearForm = 1 := by
  unfold steadySelectorLinearForm natTermsLinearForm
  simp only [List.map_cons, List.map_nil]
  simp only [Prod.snd, Prod.fst, evalLinearForm, termValue]
  rw [selectorOne]
  rfl

private theorem negOne_mul (value : F) : (-1 : F) * value = -value := by
  calc
    (-1 : F) * value = -(1 * value) := Lean.Grind.Fin.neg_mul 1 value
    _ = -value := by rw [Fin.one_mul]

theorem evalRewritePortLinearForm_eq_rewritePoint
    {raw : RawRewriteStep}
    {decoded : DecodedRewriteStep Metadata.sourceRelationColumns}
    (decodes :
      decodeRewriteStep Metadata.sourceRelationRows
        Metadata.sourceRelationColumns Metadata.finalRelationRows raw =
          some decoded)
    (assignment : Nat → Nat)
    (selectorOne : assignment Metadata.steadySelectorColumn = 1) :
    (fun port =>
      evalLinearForm (fun column => fieldResidue (assignment column))
        (rewritePortLinearForm raw port)) =
      rewritePoint (SourceAssignment.compilerAssignment assignment)
        (SourceAssignment.derivedValue assignment) decoded := by
  unfold decodeRewriteStep at decodes
  split at decodes
  next emittedBound =>
    split at decodes
    next sourceRanges =>
      cases outputResult : decodeRewriteOutput Metadata.sourceRelationColumns
          raw.output with
      | none => simp [outputResult] at decodes
      | some output =>
          cases baseResult : decodeLinearCombination
              Metadata.sourceRelationColumns raw.base with
          | none => simp [outputResult, baseResult] at decodes
          | some base =>
              cases factorsResult : raw.factors.mapM
                  (decodeProductFactor Metadata.sourceRelationColumns) with
              | none =>
                  simp [outputResult, baseResult, factorsResult] at decodes
              | some factors =>
                  simp [outputResult, baseResult, factorsResult] at decodes
                  subst decoded
                  have outputEq := outputValue_eq_evalOutputLinearForm
                    outputResult assignment
                  have baseEq :=
                    linearCombinationValue_eq_evalSourceLinearForm
                      baseResult assignment
                  have previousEq :=
                    previousValue_eq_evalPreviousLinearForm assignment raw.previous
                  have factors0 := factorValuesAt_eq factorsResult assignment 0
                  have factors1 := factorValuesAt_eq factorsResult assignment 1
                  have factors2 := factorValuesAt_eq factorsResult assignment 2
                  have factors3 := factorValuesAt_eq factorsResult assignment 3
                  have factors4 := factorValuesAt_eq factorsResult assignment 4
                  have cEq :
                      evalLinearForm
                          (fun column => fieldResidue (assignment column))
                          (rewriteCLinearForm raw) =
                        rewriteCValue
                          (SourceAssignment.compilerAssignment assignment)
                          (SourceAssignment.derivedValue assignment)
                          { sourceRowCount := Metadata.sourceRelationRows
                            finalRowCount := Metadata.finalRelationRows
                            emittedRow := raw.emittedRow
                            emittedBound
                            rewriteId := raw.rewriteId
                            kind := raw.kind
                            sourceRows := raw.sourceRows
                            sourceRanges
                            output
                            base
                            previous := raw.previous
                            factors } := by
                    unfold rewriteCLinearForm negateLinearForm rewriteCValue
                    rw [eval_append, eval_append, eval_scale, eval_scale,
                      ← outputEq, ← baseEq, ← previousEq]
                    rw [negOne_mul, negOne_mul]
                    exact Lean.Grind.Fin.add_assoc _ _ _
                  apply funext
                  apply fin13_cases
                  · simpa [rewritePortLinearForm, rewritePoint] using factors0.1.symm
                  · rfl
                  · simpa [rewritePortLinearForm, rewritePoint] using factors0.2.symm
                  · simpa [rewritePortLinearForm, rewritePoint] using factors1.1.symm
                  · simpa [rewritePortLinearForm, rewritePoint] using cEq
                  · simpa [rewritePortLinearForm, rewritePoint] using factors1.2.symm
                  · simpa [rewritePortLinearForm, rewritePoint] using factors2.1.symm
                  · simpa [rewritePortLinearForm, rewritePoint] using
                      selectorLinearForm_eq_one assignment selectorOne
                  · simpa [rewritePortLinearForm, rewritePoint] using factors2.2.symm
                  · simpa [rewritePortLinearForm, rewritePoint] using factors3.1.symm
                  · simpa [rewritePortLinearForm, rewritePoint] using factors3.2.symm
                  · simpa [rewritePortLinearForm, rewritePoint] using factors4.1.symm
                  · simpa [rewritePortLinearForm, rewritePoint] using factors4.2.symm
    next invalidRanges => simp at decodes
  next emittedOutside => simp at decodes

theorem evalRetainedPortLinearForm_eq_retainedPoint
    {raw : RawRetainedStep}
    {decoded : DecodedRetainedStep Metadata.sourceRelationColumns}
    (decodes :
      decodeRetainedStep Metadata.sourceRelationRows
        Metadata.sourceRelationColumns Metadata.finalRelationRows raw =
          some decoded)
    (assignment : Nat → Nat)
    (selectorOne : assignment Metadata.steadySelectorColumn = 1) :
    (fun port =>
      evalLinearForm (fun column => fieldResidue (assignment column))
        (retainedPortLinearForm raw port)) =
      retainedPoint (SourceAssignment.compilerAssignment assignment) decoded := by
  unfold decodeRetainedStep at decodes
  split at decodes
  next emittedBound =>
    split at decodes
    next sourceBound =>
      cases aResult : decodeLinearCombination Metadata.sourceRelationColumns raw.a with
      | none => simp [aResult] at decodes
      | some a =>
          cases bResult : decodeLinearCombination Metadata.sourceRelationColumns raw.b with
          | none => simp [aResult, bResult] at decodes
          | some b =>
              cases cResult : decodeLinearCombination
                  Metadata.sourceRelationColumns raw.c with
              | none => simp [aResult, bResult, cResult] at decodes
              | some c =>
                  simp [aResult, bResult, cResult] at decodes
                  subst decoded
                  have aEq := linearCombinationValue_eq_evalSourceLinearForm
                    aResult assignment
                  have bEq := linearCombinationValue_eq_evalSourceLinearForm
                    bResult assignment
                  have cEq := linearCombinationValue_eq_evalSourceLinearForm
                    cResult assignment
                  apply funext
                  apply fin13_cases
                  · rfl
                  · simpa [retainedPortLinearForm, retainedPoint] using
                      selectorLinearForm_eq_one assignment selectorOne
                  · simpa [retainedPortLinearForm, retainedPoint] using aEq.symm
                  · simpa [retainedPortLinearForm, retainedPoint] using bEq.symm
                  · simpa [retainedPortLinearForm, retainedPoint] using cEq.symm
                  · rfl
                  · rfl
                  · rfl
                  · rfl
                  · rfl
                  · rfl
                  · rfl
                  · rfl
    next sourceOutside => simp at decodes
  next emittedOutside => simp at decodes

theorem emittedRow_eq_of_decodeEmittedRow {raw : RawEmittedRow}
    {decoded : DecodedEmittedRow}
    (decodes : decodeEmittedRow raw = some decoded) :
    decoded.emittedRow.val = raw.emittedRow := by
  by_cases version : raw.schemaVersion = supportedSchemaVersion
  · by_cases rowsPositive : 0 < raw.rows
    · by_cases columnsPositive : 0 < raw.columns
      · by_cases rowInRange : raw.emittedRow < raw.rows
        · cases portsResult : raw.ports.mapM (decodePort raw.columns) with
          | none =>
              simp [decodeEmittedRow, version, rowsPositive, columnsPositive,
                rowInRange, portsResult] at decodes
          | some ports =>
              by_cases portCount : ports.length = selectivePortCount
              · simp [decodeEmittedRow, version, rowsPositive,
                  columnsPositive, rowInRange, portsResult, portCount] at decodes
                subst decoded
                rfl
              · simp [decodeEmittedRow, version, rowsPositive,
                  columnsPositive, rowInRange, portsResult, portCount] at decodes
        · simp [decodeEmittedRow, version, rowsPositive, columnsPositive,
            rowInRange] at decodes
      · simp [decodeEmittedRow, version, rowsPositive, columnsPositive]
          at decodes
    · simp [decodeEmittedRow, version, rowsPositive] at decodes
  · simp [decodeEmittedRow, version] at decodes

theorem rewriteStepRow_eq_of_decode {raw : RawRewriteStep}
    {decoded : DecodedRewriteStep Metadata.sourceRelationColumns}
    (decodes : decodeRewriteStep Metadata.sourceRelationRows
      Metadata.sourceRelationColumns Metadata.finalRelationRows raw =
        some decoded) :
    decoded.emittedRow = raw.emittedRow := by
  by_cases emittedBound : raw.emittedRow < Metadata.finalRelationRows
  · by_cases sourceRanges : ∀ range ∈ raw.sourceRows,
        rowRangeValid Metadata.sourceRelationRows range
    · cases outputResult : decodeRewriteOutput
          Metadata.sourceRelationColumns raw.output with
      | none =>
          simp [decodeRewriteStep, emittedBound, sourceRanges, outputResult]
            at decodes
      | some output =>
          cases baseResult : decodeLinearCombination
              Metadata.sourceRelationColumns raw.base with
          | none =>
              simp [decodeRewriteStep, emittedBound, sourceRanges,
                outputResult, baseResult] at decodes
          | some base =>
              cases factorsResult : raw.factors.mapM
                  (decodeProductFactor Metadata.sourceRelationColumns) with
              | none =>
                  simp [decodeRewriteStep, emittedBound, sourceRanges,
                    outputResult, baseResult, factorsResult] at decodes
              | some factors =>
                  simp [decodeRewriteStep, emittedBound, sourceRanges,
                    outputResult, baseResult, factorsResult] at decodes
                  rcases decodes with ⟨_, decodes⟩
                  subst decoded
                  rfl
    · simp [decodeRewriteStep, emittedBound, sourceRanges] at decodes
  · simp [decodeRewriteStep, emittedBound] at decodes

theorem retainedStepRow_eq_of_decode {raw : RawRetainedStep}
    {decoded : DecodedRetainedStep Metadata.sourceRelationColumns}
    (decodes : decodeRetainedStep Metadata.sourceRelationRows
      Metadata.sourceRelationColumns Metadata.finalRelationRows raw =
        some decoded) :
    decoded.emittedRow = raw.emittedRow := by
  by_cases emittedBound : raw.emittedRow < Metadata.finalRelationRows
  · by_cases sourceBound : raw.sourceRow < Metadata.sourceRelationRows
    · cases aResult : decodeLinearCombination
          Metadata.sourceRelationColumns raw.a with
      | none =>
          simp [decodeRetainedStep, emittedBound, sourceBound, aResult]
            at decodes
      | some a =>
          cases bResult : decodeLinearCombination
              Metadata.sourceRelationColumns raw.b with
          | none =>
              simp [decodeRetainedStep, emittedBound, sourceBound, aResult,
                bResult] at decodes
          | some b =>
              cases cResult : decodeLinearCombination
                  Metadata.sourceRelationColumns raw.c with
              | none =>
                  simp [decodeRetainedStep, emittedBound, sourceBound, aResult,
                    bResult, cResult] at decodes
              | some c =>
                  simp [decodeRetainedStep, emittedBound, sourceBound, aResult,
                    bResult, cResult] at decodes
                  subst decoded
                  rfl
    · simp [decodeRetainedStep, emittedBound, sourceBound] at decodes
  · simp [decodeRetainedStep, emittedBound] at decodes

theorem rewriteStepFactorsLength_eq_of_decode {raw : RawRewriteStep}
    {decoded : DecodedRewriteStep Metadata.sourceRelationColumns}
    (decodes : decodeRewriteStep Metadata.sourceRelationRows
      Metadata.sourceRelationColumns Metadata.finalRelationRows raw =
        some decoded) :
    decoded.factors.length = raw.factors.length := by
  by_cases emittedBound : raw.emittedRow < Metadata.finalRelationRows
  · by_cases sourceRanges : ∀ range ∈ raw.sourceRows,
        rowRangeValid Metadata.sourceRelationRows range
    · cases outputResult : decodeRewriteOutput
          Metadata.sourceRelationColumns raw.output with
      | none =>
          simp [decodeRewriteStep, emittedBound, sourceRanges, outputResult]
            at decodes
      | some output =>
          cases baseResult : decodeLinearCombination
              Metadata.sourceRelationColumns raw.base with
          | none =>
              simp [decodeRewriteStep, emittedBound, sourceRanges,
                outputResult, baseResult] at decodes
          | some base =>
              cases factorsResult : raw.factors.mapM
                  (decodeProductFactor Metadata.sourceRelationColumns) with
              | none =>
                  simp [decodeRewriteStep, emittedBound, sourceRanges,
                    outputResult, baseResult, factorsResult] at decodes
              | some factors =>
                  simp [decodeRewriteStep, emittedBound, sourceRanges,
                    outputResult, baseResult, factorsResult] at decodes
                  rcases decodes with ⟨_, decodes⟩
                  subst decoded
                  exact length_eq_of_mapM
                    (decodeProductFactor Metadata.sourceRelationColumns)
                    factorsResult
    · simp [decodeRewriteStep, emittedBound, sourceRanges] at decodes
  · simp [decodeRewriteStep, emittedBound] at decodes

/-! ## One-record certificate lift -/

theorem rewritePairCertificate_implies_matching
    {pair : RawRewritePair}
    {emitted : DecodedEmittedRow}
    {provenance : DecodedRewriteStep Metadata.sourceRelationColumns}
    (certificate : RewritePairCertificate pair)
    (emittedDecodes : decodeEmittedRow pair.emitted = some emitted)
    (provenanceDecodes :
      decodeRewriteStep Metadata.sourceRelationRows
        Metadata.sourceRelationColumns Metadata.finalRelationRows
          pair.provenance = some provenance)
    (assignment : Nat → Nat)
    (selectorOne : assignment Metadata.steadySelectorColumn = 1) :
    RewriteProvenanceMatches assignment
      (SourceAssignment.compilerAssignment assignment)
      (SourceAssignment.derivedValue assignment) emitted provenance := by
  constructor
  · calc
      emitted.emittedRow.val = pair.emitted.emittedRow :=
        emittedRow_eq_of_decodeEmittedRow emittedDecodes
      _ = pair.provenance.emittedRow := certificate.2.2.1
      _ = provenance.emittedRow :=
        (rewriteStepRow_eq_of_decode provenanceDecodes).symm
  · calc
      emittedPoint emitted assignment =
          (fun port => evalLinearForm
            (fun column => fieldResidue (assignment column))
            (rawEmittedPortLinearForm pair.emitted port)) :=
        emittedPoint_eq_evalRawPorts emittedDecodes assignment
      _ = (fun port => evalLinearForm
            (fun column => fieldResidue (assignment column))
            (rewritePortLinearForm pair.provenance port)) := by
        funext port
        exact rewriteCoefficientShape_semantic certificate.2.2.2
          (fun column => fieldResidue (assignment column)) port
      _ = rewritePoint (SourceAssignment.compilerAssignment assignment)
          (SourceAssignment.derivedValue assignment) provenance :=
        evalRewritePortLinearForm_eq_rewritePoint provenanceDecodes assignment
          selectorOne

theorem retainedPairCertificate_implies_matching
    {pair : RawRetainedPair}
    {emitted : DecodedEmittedRow}
    {provenance : DecodedRetainedStep Metadata.sourceRelationColumns}
    (certificate : RetainedPairCertificate pair)
    (emittedDecodes : decodeEmittedRow pair.emitted = some emitted)
    (provenanceDecodes :
      decodeRetainedStep Metadata.sourceRelationRows
        Metadata.sourceRelationColumns Metadata.finalRelationRows
          pair.provenance = some provenance)
    (assignment : Nat → Nat)
    (selectorOne : assignment Metadata.steadySelectorColumn = 1) :
    RetainedProvenanceMatches assignment
      (SourceAssignment.compilerAssignment assignment) emitted provenance := by
  constructor
  · calc
      emitted.emittedRow.val = pair.emitted.emittedRow :=
        emittedRow_eq_of_decodeEmittedRow emittedDecodes
      _ = pair.provenance.emittedRow := certificate.2.2.1
      _ = provenance.emittedRow :=
        (retainedStepRow_eq_of_decode provenanceDecodes).symm
  · calc
      emittedPoint emitted assignment =
          (fun port => evalLinearForm
            (fun column => fieldResidue (assignment column))
            (rawEmittedPortLinearForm pair.emitted port)) :=
        emittedPoint_eq_evalRawPorts emittedDecodes assignment
      _ = (fun port => evalLinearForm
            (fun column => fieldResidue (assignment column))
            (retainedPortLinearForm pair.provenance port)) := by
        funext port
        exact retainedCoefficientShape_semantic certificate.2.2.2
          (fun column => fieldResidue (assignment column)) port
      _ = retainedPoint (SourceAssignment.compilerAssignment assignment)
          provenance :=
        evalRetainedPortLinearForm_eq_retainedPoint provenanceDecodes assignment
          selectorOne

/-- A single proof-free rewrite certificate constructs both typed decodings,
the exact semantic coefficient match, and the five-factor capacity needed by
the selective compiler bridge. -/
theorem exists_rewriteProvenanceMatches_of_certificate
    {pair : RawRewritePair}
    (certificate : RewritePairCertificate pair)
    (assignment : Nat → Nat)
    (selectorOne : assignment Metadata.steadySelectorColumn = 1) :
    ∃ emitted provenance,
      decodeEmittedRow pair.emitted = some emitted ∧
      decodeRewriteStep Metadata.sourceRelationRows
        Metadata.sourceRelationColumns Metadata.finalRelationRows
          pair.provenance = some provenance ∧
      provenance.factors.length ≤ 5 ∧
      RewriteProvenanceMatches assignment
        (SourceAssignment.compilerAssignment assignment)
        (SourceAssignment.derivedValue assignment) emitted provenance := by
  rcases decodeEmittedRow_of_valid certificate.1 with
    ⟨emitted, emittedDecodes⟩
  rcases decodeRewriteStep_of_valid certificate.2.1 with
    ⟨provenance, provenanceDecodes⟩
  refine ⟨emitted, provenance, emittedDecodes, provenanceDecodes, ?_, ?_⟩
  · rw [rewriteStepFactorsLength_eq_of_decode provenanceDecodes]
    exact certificate.2.1.2.2.2.2.2
  · exact rewritePairCertificate_implies_matching certificate
      emittedDecodes provenanceDecodes assignment selectorOne

/-- A single proof-free retained certificate constructs both typed decodings
and their exact retained-row coefficient match. -/
theorem exists_retainedProvenanceMatches_of_certificate
    {pair : RawRetainedPair}
    (certificate : RetainedPairCertificate pair)
    (assignment : Nat → Nat)
    (selectorOne : assignment Metadata.steadySelectorColumn = 1) :
    ∃ emitted provenance,
      decodeEmittedRow pair.emitted = some emitted ∧
      decodeRetainedStep Metadata.sourceRelationRows
        Metadata.sourceRelationColumns Metadata.finalRelationRows
          pair.provenance = some provenance ∧
      RetainedProvenanceMatches assignment
        (SourceAssignment.compilerAssignment assignment) emitted provenance := by
  rcases decodeEmittedRow_of_valid certificate.1 with
    ⟨emitted, emittedDecodes⟩
  rcases decodeRetainedStep_of_valid certificate.2.1 with
    ⟨provenance, provenanceDecodes⟩
  exact ⟨emitted, provenance, emittedDecodes, provenanceDecodes,
    retainedPairCertificate_implies_matching certificate emittedDecodes
      provenanceDecodes assignment selectorOne⟩

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.SelectiveArtifactPairs
