import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.RewriteBlockSemantics
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.SelectiveArtifactPairs.Certificates
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.SelectiveArtifactPairs.Decode
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.SourceDisposition.Pivots

/-!
Kernel eliminator for the exact production combined-NC rewrite owners.

Owns: bounded `BatchCheck` elimination into typed rewrite witnesses, exact decoded facts, and the `DotTrace.Components` seam.

Does not own: row satisfaction, global execution, transcript/child authority,
commitment binding, costs, or permission to remove rows.

Emits constraints: none; this module proves an existing rewrite block sound.

| Stable stage path | Obligation | Authority class |
|---|---|---|
| `f_prime.pi_ccs_nc.delayed.combined.rewrite_block_soundness` | Show selected rewrite rows imply their corresponding source-definition block. | derived |
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.RewriteBlockSoundness

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.CheckedProgram
open Nightstream.Implementation.R1CS.ProjectionProgram
open Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc
open Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized
open Decoder
open Semantics
open RewriteBatchIndex
open RewriteChain
open RewriteSourceSemantics.ChainAgreement
open RewriteBlockSemantics
open SelectiveArtifactPairs

private theorem boolAnd_left {left right : Bool}
    (both : (left && right) = true) : left = true := by
  cases left <;> cases right <;> simp_all

private theorem boolAnd_right {left right : Bool}
    (both : (left && right) = true) : right = true := by
  cases left <;> cases right <;> simp_all

private theorem find?_eq_some_mem {Alpha : Type}
    (predicate : Alpha → Bool) :
    ∀ (values : List Alpha) {found : Alpha},
      values.find? predicate = some found → found ∈ values := by
  intro values
  induction values with
  | nil => simp [List.find?]
  | cons head tail inductionHypothesis =>
      intro found lookup
      cases test : predicate head with
      | false =>
          exact List.mem_cons_of_mem head
            (inductionHypothesis (by
              simpa [List.find?, test] using lookup))
      | true =>
          have equal : head = found := by
            simpa [List.find?, test] using lookup
          subst found
          exact List.mem_cons_self

private theorem find?_eq_some_matches {Alpha : Type}
    (predicate : Alpha → Bool) :
    ∀ (values : List Alpha) {found : Alpha},
      values.find? predicate = some found → predicate found = true := by
  intro values
  induction values with
  | nil => simp [List.find?]
  | cons head tail inductionHypothesis =>
      intro found lookup
      cases test : predicate head with
      | false =>
          exact inductionHypothesis (by
            simpa [List.find?, test] using lookup)
      | true =>
          have equal : head = found := by
            simpa [List.find?, test] using lookup
          simpa [← equal] using test

/-! ## Five-definition owners -/

inductive DecodedSingletonMatch (definitions : List Definition)
    (raw : RawRewriteStep) : Prop where
  | intro
      (decoded : DecodedRewriteStep Metadata.sourceRelationColumns)
      (output : DecodedLinearCombination Metadata.sourceRelationColumns)
      (decodes :
        decodeRewriteStep Metadata.sourceRelationRows
          Metadata.sourceRelationColumns Metadata.finalRelationRows raw =
            some decoded)
      (outputExact : decoded.output = .source output)
      (exactMatch : ExactChainMatch definitions [decoded] output) :
      DecodedSingletonMatch definitions raw

inductive SmallBatchWitness (batch : Batch) : Prop where
  | intro
      (definitions : List Definition)
      (raws : List RawRewriteStep)
      (definitionsExact :
        sourceDefinitionsForBatch? batch = some definitions)
      (rawsExact : rawStepsFor? batch = some raws)
      (sourceLength : batch.descriptor.sourceRange.stop -
        batch.descriptor.sourceRange.start = 5)
      (definitionCount : definitions.length = 5)
      (compactExact : raws.map CompactStep.ofRaw = batch.steps)
      (sourceRangesExact : ∀ raw ∈ raws,
        raw.sourceRows = [batch.descriptor.sourceRange])
      (rawCount : raws.length = 2)
      (decodedMatches : ∀ raw ∈ raws,
        DecodedSingletonMatch definitions raw)
      (chainsValid : ∀ chain ∈ partitionChains raws,
        RawChainValid chain)
      (chainsTriangular :
        RawChainsTriangularFrom [] (partitionChains raws)) :
      SmallBatchWitness batch

private theorem decodedSingletonMatch_of_check
    {definitions : List Definition} {raw : RawRewriteStep}
    (checked : exactSingletonCheck definitions raw = true) :
    DecodedSingletonMatch definitions raw := by
  rcases exactSingletonCheck_sound checked with
    ⟨decoded, output, decodes, outputExact, exactMatch⟩
  refine .intro decoded output ?_ outputExact exactMatch
  change decodeRewriteStep Metadata.sourceRelationRows
    Metadata.sourceRelationColumns Metadata.finalRelationRows raw =
      some decoded at decodes
  exact decodes

theorem smallBatchWitness_of_check {batch : Batch}
    (checked : SmallBatchCheck batch = true) :
    SmallBatchWitness batch := by
  unfold SmallBatchCheck at checked
  cases definitionsResult : sourceDefinitionsForBatch? batch with
  | none => simp [definitionsResult] at checked
  | some definitions =>
      cases rawsResult : rawStepsFor? batch with
      | none => simp [definitionsResult, rawsResult] at checked
      | some raws =>
          simp only [definitionsResult, rawsResult] at checked
          have triangularCheck := boolAnd_right checked
          have prefix7 := boolAnd_left checked
          have chainCheck := boolAnd_right prefix7
          have prefix6 := boolAnd_left prefix7
          have exactChecks := boolAnd_right prefix6
          have prefix5 := boolAnd_left prefix6
          have rawCountCheck := boolAnd_right prefix5
          have prefix4 := boolAnd_left prefix5
          have sourceRangesCheck := boolAnd_right prefix4
          have prefix3 := boolAnd_left prefix4
          have compactCheck := boolAnd_right prefix3
          have prefix2 := boolAnd_left prefix3
          have definitionCountCheck := boolAnd_right prefix2
          have sourceLengthCheck := boolAnd_left prefix2
          refine .intro definitions raws definitionsResult rawsResult
            (of_decide_eq_true sourceLengthCheck)
            (of_decide_eq_true definitionCountCheck)
            (of_decide_eq_true compactCheck) ?_
            (of_decide_eq_true rawCountCheck) ?_
            (chainShapeCheck_sound chainCheck)
            (RewriteBlockSemantics.triangularCheck_sound triangularCheck)
          · intro raw member
            exact of_decide_eq_true
              ((List.all_eq_true.mp sourceRangesCheck) raw member)
          · intro raw member
            exact decodedSingletonMatch_of_check
              ((List.all_eq_true.mp exactChecks) raw member)

/-! ## Large dot owners -/

/-- Exact raw coefficient contract retained from the bounded large-owner
certificate. Its interpretation is paired with `DotTrace.Components`; no
large source-definition list is evaluated here. -/
def LargeContributionContract (owner : DotOwnerKey)
    (chains : List (List RawRewriteStep)) : Prop :=
  largeContributionsCheck owner chains = true

inductive LargeBatchWitness (batch : Batch) : Prop where
  | intro
      (owner : DotOwnerKey)
      (raws : List RawRewriteStep)
      (ownerExact :
        dotOwners.find? (fun candidate => decide
          (candidate.sourceRange = batch.descriptor.sourceRange)) = some owner)
      (ownerMember : owner ∈ dotOwners)
      (sourceRangeExact : owner.sourceRange = batch.descriptor.sourceRange)
      (rawsExact : rawStepsFor? batch = some raws)
      (compactExact : raws.map CompactStep.ofRaw = batch.steps)
      (sourceRangesExact : ∀ raw ∈ raws,
        raw.sourceRows = [batch.descriptor.sourceRange])
      (chainCount : (partitionChains raws).length = 3)
      (chainLengths : (partitionChains raws).map List.length =
        List.replicate 3 owner.chainLength)
      (targetColumns : (partitionChains raws).mapM rawChainTarget? =
        some owner.targetColumns)
      (chainsValid : ∀ chain ∈ partitionChains raws,
        RawChainValid chain)
      (chainsTriangular :
        RawChainsTriangularFrom [] (partitionChains raws))
      (contributionsExact :
        LargeContributionContract owner (partitionChains raws)) :
      LargeBatchWitness batch

theorem largeBatchWitness_of_check {batch : Batch}
    (checked : LargeBatchCheck batch = true) :
    LargeBatchWitness batch := by
  unfold LargeBatchCheck at checked
  let predicate := fun owner : DotOwnerKey => decide
    (owner.sourceRange = batch.descriptor.sourceRange)
  cases ownerResult : dotOwners.find? predicate with
  | none => simp [predicate, ownerResult] at checked
  | some owner =>
      cases rawsResult : rawStepsFor? batch with
      | none => simp [predicate, ownerResult, rawsResult] at checked
      | some raws =>
          simp only [predicate, ownerResult, rawsResult] at checked
          have contributionsCheck := boolAnd_right checked
          have prefix7 := boolAnd_left checked
          have triangularCheck := boolAnd_right prefix7
          have prefix6 := boolAnd_left prefix7
          have chainCheck := boolAnd_right prefix6
          have prefix5 := boolAnd_left prefix6
          have targetCheck := boolAnd_right prefix5
          have prefix4 := boolAnd_left prefix5
          have chainLengthsCheck := boolAnd_right prefix4
          have prefix3 := boolAnd_left prefix4
          have chainCountCheck := boolAnd_right prefix3
          have prefix2 := boolAnd_left prefix3
          have sourceRangesCheck := boolAnd_right prefix2
          have compactCheck := boolAnd_left prefix2
          have ownerLookup : dotOwners.find? predicate = some owner :=
            ownerResult
          have ownerMatches : predicate owner = true :=
            find?_eq_some_matches predicate dotOwners ownerLookup
          refine .intro owner raws ownerResult
            (find?_eq_some_mem predicate dotOwners ownerLookup)
            (of_decide_eq_true ownerMatches) rawsResult
            (of_decide_eq_true compactCheck) ?_
            (of_decide_eq_true chainCountCheck)
            (of_decide_eq_true chainLengthsCheck)
            (of_decide_eq_true targetCheck)
            (chainShapeCheck_sound chainCheck)
            (RewriteBlockSemantics.triangularCheck_sound triangularCheck)
            contributionsCheck
          intro raw member
          exact of_decide_eq_true
            ((List.all_eq_true.mp sourceRangesCheck) raw member)

/-! ## Certified typed chain recovery -/

private theorem mapM_some_member {Alpha Beta : Type}
    (decode : Alpha → Option Beta) :
    ∀ {inputs : List Alpha} {outputs : List Beta},
      inputs.mapM decode = some outputs →
      ∀ output ∈ outputs, ∃ input ∈ inputs, decode input = some output := by
  intro inputs
  induction inputs with
  | nil =>
      intro outputs decoded
      simp at decoded
      subst outputs
      simp
  | cons head tail inductionHypothesis =>
      intro outputs decoded
      cases headResult : decode head with
      | none => simp [headResult] at decoded
      | some decodedHead =>
          cases tailResult : tail.mapM decode with
          | none => simp [headResult, tailResult] at decoded
          | some decodedTail =>
              simp [headResult, tailResult] at decoded
              subst outputs
              intro output member
              simp only [List.mem_cons] at member
              rcases member with rfl | tailMember
              · exact ⟨head, by simp, headResult⟩
              · rcases inductionHypothesis tailResult output tailMember with
                  ⟨input, inputMember, inputDecodes⟩
                exact ⟨input, by simp [inputMember], inputDecodes⟩

/-- A direct-shard lookup is membership in the exact generated 1,493-record
stream.  The proof never evaluates or searches the concatenated stream. -/
private theorem rawStepAt?_mem_generated {index : Nat}
    {raw : RawRewriteStep} (lookup : rawStepAt? index = some raw) :
    raw ∈ Provenance.RewriteSteps.values := by
  unfold rawStepAt? at lookup
  split at lookup <;>
    try
      { rcases getElem?_eq_some_iff.mp lookup with
          ⟨bound, valueExact⟩
        have localMember := List.getElem_mem bound
        rw [valueExact] at localMember
        simpa only [Provenance.RewriteSteps.values, List.mem_append,
          localMember, true_or, or_true] }
  simp at lookup

private theorem rawStepsAt?_member_generated {offset count : Nat}
    {raws : List RawRewriteStep}
    (lookup : rawStepsAt? offset count = some raws) :
    ∀ raw ∈ raws, raw ∈ Provenance.RewriteSteps.values := by
  intro raw member
  unfold rawStepsAt? at lookup
  rcases mapM_some_member
      (fun index => rawStepAt? (offset + index)) lookup raw member with
    ⟨_index, _indexMember, rawLookup⟩
  exact rawStepAt?_mem_generated rawLookup

/-- Exact source-range lookup never manufactures a rewrite step: every
returned record belongs to the generated production provenance stream. -/
theorem generatedRawMember_of_rawStepsFor {batch : Batch}
    {raws : List RawRewriteStep}
    (lookup : rawStepsFor? batch = some raws) :
    ∀ raw ∈ raws, raw ∈ Provenance.RewriteSteps.values := by
  unfold rawStepsFor? at lookup
  exact rawStepsAt?_member_generated lookup

/-- Pair coverage turns generated-stream membership into the already checked
fail-closed decoder validity predicate. -/
private theorem generatedRaw_valid {raw : RawRewriteStep}
    (member : raw ∈ Provenance.RewriteSteps.values) :
    RawRewriteStepValid raw := by
  have provenanceExact :=
    SelectiveArtifactPairs.Certificates.allPairedProvenanceExact.2
  change raw ∈ Provenance.rewriteSteps at member
  rw [← provenanceExact] at member
  rcases List.mem_map.mp member with ⟨pair, pairMember, pairExact⟩
  subst raw
  exact
    (SelectiveArtifactPairs.Certificates.rewritePairsCertified pair
      pairMember).2.1

inductive DecodedOutputMatches {columns : Nat}
    (rawOutput : RawRewriteOutput)
    (decodedOutput : DecodedRewriteOutput columns) : Prop where
  | source (raw : RawLinearCombination)
      (decoded : DecodedLinearCombination columns)
      (rawExact : rawOutput = .source raw)
      (decodedExact : decodedOutput = .source decoded)
      (decodes : decodeLinearCombination columns raw = some decoded) :
      DecodedOutputMatches rawOutput decodedOutput
  | derived (compilerIndex : Nat)
      (rawExact : rawOutput = .derivedProductSum compilerIndex)
      (decodedExact : decodedOutput = .derivedProductSum compilerIndex) :
      DecodedOutputMatches rawOutput decodedOutput

def DecodedStepShape (raw : RawRewriteStep)
    (decoded : DecodedRewriteStep Metadata.sourceRelationColumns) : Prop :=
  decoded.previous = raw.previous ∧
    DecodedOutputMatches raw.output decoded.output

private theorem decodedOutputMatches_of_decode
    {raw : RawRewriteOutput}
    {decoded : DecodedRewriteOutput Metadata.sourceRelationColumns}
    (decodes : decodeRewriteOutput Metadata.sourceRelationColumns raw =
      some decoded) :
    DecodedOutputMatches raw decoded := by
  cases raw with
  | source output =>
      cases outputResult : decodeLinearCombination
          Metadata.sourceRelationColumns output with
      | none => simp [decodeRewriteOutput, outputResult] at decodes
      | some decodedOutput =>
          simp [decodeRewriteOutput, outputResult] at decodes
          subst decoded
          exact .source output decodedOutput rfl rfl outputResult
  | derivedProductSum compilerIndex =>
      simp [decodeRewriteOutput] at decodes
      subst decoded
      exact .derived compilerIndex rfl rfl

private theorem decodedStepShape_of_decode {raw : RawRewriteStep}
    {decoded : DecodedRewriteStep Metadata.sourceRelationColumns}
    (decodes : decodeRewriteStep Metadata.sourceRelationRows
      Metadata.sourceRelationColumns Metadata.finalRelationRows raw =
        some decoded) :
    DecodedStepShape raw decoded := by
  unfold decodeRewriteStep at decodes
  split at decodes
  next emittedBound =>
    split at decodes
    next sourceRanges =>
      cases outputResult : decodeRewriteOutput
          Metadata.sourceRelationColumns raw.output with
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
                  exact ⟨rfl, decodedOutputMatches_of_decode outputResult⟩
    next sourceRangesFalse => simp at decodes
  next emittedBoundFalse => simp at decodes

/-- Pointwise decoding stays propositional, outside executable certificates. -/
inductive DecodedSteps : List RawRewriteStep →
    List (DecodedRewriteStep Metadata.sourceRelationColumns) → Prop where
  | nil : DecodedSteps [] []
  | cons {raw decoded raws decodedRaws}
      (decodes : decodeRewriteStep Metadata.sourceRelationRows
        Metadata.sourceRelationColumns Metadata.finalRelationRows raw =
          some decoded)
      (tail : DecodedSteps raws decodedRaws) :
      DecodedSteps (raw :: raws) (decoded :: decodedRaws)

private theorem decodedSteps_of_generated {raws : List RawRewriteStep}
    (generated : ∀ raw ∈ raws,
      raw ∈ Provenance.RewriteSteps.values) :
    ∃ decoded, DecodedSteps raws decoded := by
  induction raws with
  | nil => exact ⟨[], .nil⟩
  | cons raw raws inductionHypothesis =>
      rcases SelectiveArtifactPairs.decodeRewriteStep_of_valid
          (generatedRaw_valid (generated raw (by simp))) with
        ⟨decoded, decodes⟩
      rcases inductionHypothesis (by
        intro candidate member
        exact generated candidate (by simp [member])) with
        ⟨decodedRaws, tail⟩
      exact ⟨decoded :: decodedRaws, .cons decodes tail⟩

private theorem sourceChain_from_rawLinked :
    ∀ {previous : Option Nat} {raw : RawRewriteStep}
      {raws : List RawRewriteStep}
      {decoded : DecodedRewriteStep Metadata.sourceRelationColumns}
      {decodedRaws :
        List (DecodedRewriteStep Metadata.sourceRelationColumns)},
      raw.previous = previous →
      DecodedStepShape raw decoded →
      DecodedSteps raws decodedRaws →
      RawChainLinked raw.output raws →
      ∃ rawOutput output,
        rawSourceOutput? (raw :: raws) = some rawOutput ∧
        decodeLinearCombination Metadata.sourceRelationColumns rawOutput =
          some output ∧
        SourceChain previous (decoded :: decodedRaws) output := by
  intro previous raw raws
  induction raws generalizing previous raw with
  | nil =>
      intro decoded decodedRaws previousExact shape decodedTail linked
      cases decodedTail
      rcases shape with ⟨decodedPrevious, outputShape⟩
      cases outputShape with
      | source rawOutput decodedOutput rawExact decodedExact outputDecodes =>
          refine ⟨rawOutput, decodedOutput, ?_, outputDecodes, .terminal
            (decodedPrevious.trans previousExact) decodedExact⟩
          simp [rawSourceOutput?, rawExact]
      | derived _compilerIndex rawExact _decodedExact =>
          rw [rawExact] at linked
          simp only [RawChainLinked] at linked
  | cons next rest inductionHypothesis =>
      intro decoded decodedRaws previousExact shape decodedTail linked
      cases decodedTail with
      | cons nextDecodes remainingDecoded =>
          rcases shape with ⟨decodedPrevious, outputShape⟩
          cases outputShape with
          | source _rawOutput _decodedOutput rawExact _decodedExact
              _outputDecodes =>
              rw [rawExact] at linked
              simp only [RawChainLinked] at linked
          | derived _compilerIndex rawExact decodedExact =>
              rw [rawExact] at linked
              simp only [RawChainLinked] at linked
              rcases linked with ⟨nextPrevious, remainingLinked⟩
              have nextShape := decodedStepShape_of_decode nextDecodes
              rcases inductionHypothesis nextPrevious nextShape
                  remainingDecoded remainingLinked with
                ⟨rawOutput, output, rawOutputExact, outputDecodes,
                  tailChain⟩
              refine ⟨rawOutput, output, ?_, outputDecodes, .derived
                (decodedPrevious.trans previousExact) decodedExact tailChain⟩
              simpa [rawSourceOutput?] using rawOutputExact
structure TypedSourceChainData where
  decoded : List (DecodedRewriteStep Metadata.sourceRelationColumns)
  output : DecodedLinearCombination Metadata.sourceRelationColumns
  rawOutput : RawLinearCombination
structure TypedSourceChain (raws : List RawRewriteStep)
    (data : TypedSourceChainData) : Prop where
  decodes : DecodedSteps raws data.decoded
  chain : SourceChain none data.decoded data.output
  rawOutputExact : rawSourceOutput? raws = some data.rawOutput
  outputDecodes : decodeLinearCombination Metadata.sourceRelationColumns
    data.rawOutput = some data.output
/-- A generated closed chain decodes to an actual typed `SourceChain`. -/
theorem typedSourceChain_of_generated {raws : List RawRewriteStep}
    (generated : ∀ raw ∈ raws,
      raw ∈ Provenance.RewriteSteps.values)
    (valid : RawChainValid raws) :
    ∃ data, TypedSourceChain raws data := by
  cases raws with
  | nil => contradiction
  | cons raw rest =>
      rcases decodedSteps_of_generated generated with
        ⟨decoded, decodes⟩
      cases decoded with
      | nil => cases decodes
      | cons decodedHead decodedRest =>
          cases decodes with
          | cons headDecodes decodedTail =>
              rcases valid with ⟨previousExact, linked⟩
              have headShape := decodedStepShape_of_decode headDecodes
              rcases sourceChain_from_rawLinked previousExact headShape
                  decodedTail linked with
                ⟨rawOutput, output, rawOutputExact, outputDecodes, chain⟩
              let data : TypedSourceChainData :=
                { decoded := decodedHead :: decodedRest
                  output := output
                  rawOutput := rawOutput }
              refine ⟨data, ?_⟩
              exact
                { decodes := .cons headDecodes decodedTail
                  chain := chain
                  rawOutputExact := rawOutputExact
                  outputDecodes := outputDecodes }
/-- Checked large-dot chains have typed production decoder witnesses. -/
theorem LargeBatchWitness.typedSourceChains {batch : Batch}
    (witness : LargeBatchWitness batch) :
    ∃ (owner : DotOwnerKey) (raws : List RawRewriteStep),
      owner.sourceRange = batch.descriptor.sourceRange ∧
      rawStepsFor? batch = some raws ∧
      (partitionChains raws).mapM rawChainTarget? =
        some owner.targetColumns ∧
      ∀ chain ∈ partitionChains raws,
        ∃ data, TypedSourceChain chain data := by
  rcases witness with
    ⟨(owner : DotOwnerKey), raws, _ownerExact, _ownerMember,
      sourceRangeExact, rawsExact, _compactExact, _sourceRangesExact,
      _chainCount, _chainLengths, targetColumns, chainsValid,
      _chainsTriangular, _contributionsExact⟩
  refine ⟨owner, raws, sourceRangeExact, rawsExact, targetColumns, ?_⟩
  intro chain chainMember
  apply typedSourceChain_of_generated
  · intro raw rawMember
    apply generatedRawMember_of_rawStepsFor rawsExact raw
    rw [← partitionChains_flatten raws]
    exact List.mem_flatten.mpr ⟨chain, chainMember, rawMember⟩
  · exact chainsValid chain chainMember
/-! ## Large factor-fold semantics -/

private theorem decodedStep_base_factors {raw : RawRewriteStep}
    {decoded : DecodedRewriteStep Metadata.sourceRelationColumns}
    (decodes : decodeRewriteStep Metadata.sourceRelationRows
      Metadata.sourceRelationColumns Metadata.finalRelationRows raw =
        some decoded) :
    decodeLinearCombination Metadata.sourceRelationColumns raw.base =
        some decoded.base ∧
      raw.factors.mapM
          (decodeProductFactor Metadata.sourceRelationColumns) =
        some decoded.factors := by
  unfold decodeRewriteStep at decodes
  split at decodes
  next emittedBound =>
    split at decodes
    next sourceRanges =>
      cases outputResult : decodeRewriteOutput
          Metadata.sourceRelationColumns raw.output with
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
                  constructor <;> simp [baseResult, factorsResult]
    next sourceRangesFalse => simp at decodes
  next emittedBoundFalse => simp at decodes
private theorem mapM_length_eq {Alpha Beta : Type}
    (decode : Alpha → Option Beta) :
    ∀ {raw : List Alpha} {decoded : List Beta},
      raw.mapM decode = some decoded → decoded.length = raw.length := by
  intro raw
  induction raw with
  | nil =>
      intro decoded decodes
      simp at decodes
      subst decoded
      rfl
  | cons head tail inductionHypothesis =>
      intro decoded decodes
      cases headResult : decode head with
      | none => simp [headResult] at decodes
      | some decodedHead =>
          cases tailResult : tail.mapM decode with
          | none => simp [headResult, tailResult] at decodes
          | some decodedTail =>
              simp [headResult, tailResult] at decodes
              subst decoded
              simp [inductionHypothesis tailResult]
private theorem factorSum_eq_foldr
    {factors : List
      (DecodedProductFactor Metadata.sourceRelationColumns)}
    (assignment : Nat → Nat) (bound : factors.length ≤ 5) :
    SelectiveCompilerBridge.factorSum assignment factors =
      factors.foldr
        (fun factor suffix => productFactorValue factor assignment + suffix) 0 := by
  unfold SelectiveCompilerBridge.factorSum
  cases factors with
  | nil => rfl
  | cons factor0 factors =>
      cases factors with
      | nil => simp [SelectiveCompilerBridge.factorValueAt]
      | cons factor1 factors =>
          cases factors with
          | nil => simp [SelectiveCompilerBridge.factorValueAt]
          | cons factor2 factors =>
              cases factors with
              | nil => simp [SelectiveCompilerBridge.factorValueAt, Lean.Grind.Fin.add_assoc]
              | cons factor3 factors =>
                  cases factors with
                  | nil => simp [SelectiveCompilerBridge.factorValueAt, Lean.Grind.Fin.add_assoc]
                  | cons factor4 factors =>
                      cases factors with
                      | nil => simp [SelectiveCompilerBridge.factorValueAt, Lean.Grind.Fin.add_assoc]
                      | cons factor5 factors => simp at bound
private theorem decodedFactorFold_eq_raw {raw : List RawProductFactor}
    {decoded :
      List (DecodedProductFactor Metadata.sourceRelationColumns)}
    (decodes : raw.mapM
      (decodeProductFactor Metadata.sourceRelationColumns) = some decoded)
    (assignment : Nat → Nat) :
    decoded.foldr
        (fun factor suffix => productFactorValue factor assignment + suffix) 0 =
      raw.foldr
        (fun factor suffix => rawFactorValue assignment factor + suffix) 0 := by
  induction raw generalizing decoded with
  | nil =>
      simp at decodes
      subst decoded
      rfl
  | cons rawFactor rawFactors inductionHypothesis =>
      cases headResult : decodeProductFactor
          Metadata.sourceRelationColumns rawFactor with
      | none => simp [headResult] at decodes
      | some decodedFactor =>
          cases tailResult : rawFactors.mapM
              (decodeProductFactor Metadata.sourceRelationColumns) with
          | none => simp [headResult, tailResult] at decodes
          | some decodedFactors =>
              simp [headResult, tailResult] at decodes
              subst decoded
              simp only [List.foldr_cons]
              rw [inductionHypothesis tailResult]
              have headEq : productFactorValue decodedFactor assignment =
                  rawFactorValue assignment rawFactor := by
                simpa [rawFactorValue] using
                  SelectiveArtifactPairs.productFactorValue_eq_raw headResult
                    assignment
              rw [headEq]
private theorem foldr_weighted_add_append {Alpha : Type}
    (weight : Alpha → ProjectionProgram.F) (left right : List Alpha) :
    (left ++ right).foldr (fun value suffix => weight value + suffix) 0 =
      left.foldr (fun value suffix => weight value + suffix) 0 +
        right.foldr (fun value suffix => weight value + suffix) 0 := by
  induction left with
  | nil => simp only [List.nil_append, List.foldr_nil, Fin.zero_add]
  | cons head tail inductionHypothesis =>
      simp only [List.cons_append, List.foldr_cons]
      rw [inductionHypothesis]
      exact (ProjectionProgram.fadd_assoc _ _ _).symm
private theorem fadd_shuffle (a b c d : ProjectionProgram.F) :
    (a + b) + (c + d) = (a + c) + (b + d) := by
  rw [ProjectionProgram.fadd_assoc, ProjectionProgram.fadd_assoc]
  congr 1
  rw [← ProjectionProgram.fadd_assoc, ProjectionProgram.fadd_comm b c,
    ProjectionProgram.fadd_assoc]
private theorem contributionSum_eq_rawFactorFold
    {raws : List RawRewriteStep}
    {decoded :
      List (DecodedRewriteStep Metadata.sourceRelationColumns)}
    (decodes : DecodedSteps raws decoded)
    (assignment : Nat → Nat)
    (basesZero : ∀ raw ∈ raws,
      raw.base = { constant := 0, terms := [] })
    (valid : ∀ raw ∈ raws, RawRewriteStepValid raw) :
    contributionSum assignment decoded =
      (chainFactors raws).foldr
        (fun factor suffix => rawFactorValue assignment factor + suffix) 0 := by
  induction decodes with
  | nil => rfl
  | @cons raw decodedStep raws decodedSteps stepDecodes tail
      inductionHypothesis =>
      have parts := decodedStep_base_factors stepDecodes
      have baseValue :
          linearCombinationValue decodedStep.base assignment = 0 := by
        rw [SelectiveArtifactPairs.linearCombinationValue_eq_raw parts.1]
        rw [basesZero raw (by simp)]
        simp [SourceAssignment.RawLinearCombination.programTerms,
          lcEval, fieldResidue]
      have decodedFactorBound : decodedStep.factors.length ≤ 5 := by
        rw [mapM_length_eq
          (decodeProductFactor Metadata.sourceRelationColumns) parts.2]
        exact (valid raw (by simp)).2.2.2.2.2
      have headFactorFold := factorSum_eq_foldr assignment decodedFactorBound
      have decodedRawFold := decodedFactorFold_eq_raw parts.2 assignment
      have tailFold := inductionHypothesis
        (fun candidate member => basesZero candidate (by simp [member]))
        (fun candidate member => valid candidate (by simp [member]))
      unfold contributionSum contribution
      rw [baseValue, Fin.zero_add, headFactorFold, decodedRawFold, tailFold]
      unfold chainFactors
      rw [List.flatMap_cons]
      exact (foldr_weighted_add_append (rawFactorValue assignment)
        raw.factors (raws.flatMap RawRewriteStep.factors)).symm
private theorem largeStreams_cases {owner : DotOwnerKey}
    {chains : List (List RawRewriteStep)}
    (streams : LargeContributionStreams owner chains) :
    ∃ qChain c0Chain sumChain,
      chains = [qChain, c0Chain, sumChain] ∧
      (∀ step ∈ qChain,
        step.base = { constant := 0, terms := [] }) ∧
      (∀ step ∈ c0Chain,
        step.base = { constant := 0, terms := [] }) ∧
      (∀ step ∈ sumChain,
        step.base = { constant := 0, terms := [] }) ∧
      rawSourceOutput? qChain = some owner.qTerminalOutput ∧
      rawSourceOutput? c0Chain = some owner.productC0TerminalOutput ∧
      rawSourceOutput? sumChain = some owner.productSumTerminalOutput ∧
      chainFactors qChain =
        (owner.trace.multiplications.map fun multiplication =>
          (DotFactorTriple.ofMultiplication multiplication).qSum) ∧
      chainFactors c0Chain =
        (owner.trace.multiplications.map fun multiplication =>
          (DotFactorTriple.ofMultiplication multiplication).outputC0) ∧
      chainFactors sumChain =
        (owner.trace.multiplications.map fun multiplication =>
          (DotFactorTriple.ofMultiplication multiplication).productSum) := by
  cases chains with
  | nil => simp [LargeContributionStreams] at streams
  | cons qChain chains =>
      cases chains with
      | nil => simp [LargeContributionStreams] at streams
      | cons c0Chain chains =>
          cases chains with
          | nil => simp [LargeContributionStreams] at streams
          | cons sumChain chains =>
              cases chains with
              | cons extra rest => simp [LargeContributionStreams] at streams
              | nil => exact ⟨qChain, c0Chain, sumChain, rfl, streams⟩
def qFactorStream (assignment : Nat → Nat)
    (owner : DotOwnerKey) : List ProjectionProgram.F :=
  owner.trace.multiplications.map fun multiplication =>
    residue (lcEval assignment multiplication.left.c1) *
      residue (lcEval assignment multiplication.right.c1)
def productC0FactorStream (assignment : Nat → Nat)
    (owner : DotOwnerKey) : List ProjectionProgram.F :=
  owner.trace.multiplications.map fun multiplication =>
    residue (lcEval assignment multiplication.left.c0) *
      residue (lcEval assignment multiplication.right.c0)
def productSumFactorStream (assignment : Nat → Nat)
    (owner : DotOwnerKey) : List ProjectionProgram.F :=
  owner.trace.multiplications.map fun multiplication =>
    residue (lcEval assignment multiplication.sumLeft) *
      residue (lcEval assignment multiplication.sumRight)

private theorem qTerminalOutput_value
    {owner : DotOwnerKey}
    {output : DecodedLinearCombination Metadata.sourceRelationColumns}
    (decodes : decodeLinearCombination Metadata.sourceRelationColumns
      owner.qTerminalOutput = some output)
    (assignment : Nat → Nat) :
    linearCombinationValue output assignment =
      baseAt assignment owner.trace.qSumColumn := by
  rw [SelectiveArtifactPairs.linearCombinationValue_eq_raw decodes]
  apply Fin.ext
  have modulusEq : goldilocksP = goldilocksModulus := rfl
  simp [DotOwnerKey.qTerminalOutput,
    SourceAssignment.RawLinearCombination.programTerms,
    SourceAssignment.RawTerm.asNatTerm, fieldResidue, baseAt, residue,
    lcEval, modulusEq, Nat.mod_mod]

private theorem productC0TerminalOutput_value
    {owner : DotOwnerKey}
    {output : DecodedLinearCombination Metadata.sourceRelationColumns}
    (decodes : decodeLinearCombination Metadata.sourceRelationColumns
      owner.productC0TerminalOutput = some output)
    (assignment : Nat → Nat) :
    linearCombinationValue output assignment =
      baseAt assignment owner.trace.output.c0 +
        residue (goldilocksP - 7) *
          baseAt assignment owner.trace.qSumColumn := by
  rw [SelectiveArtifactPairs.linearCombinationValue_eq_raw decodes]
  apply Fin.ext
  have modulusEq : goldilocksP = goldilocksModulus := rfl
  simp [DotOwnerKey.productC0TerminalOutput,
    SourceAssignment.RawLinearCombination.programTerms,
    SourceAssignment.RawTerm.asNatTerm, fieldResidue, baseAt, residue,
    lcEval, Fin.val_add, Fin.val_mul, Nat.add_mod, Nat.mul_mod,
    modulusEq, Nat.mod_mod]

private theorem productSumTerminalOutput_value
    {owner : DotOwnerKey}
    {output : DecodedLinearCombination Metadata.sourceRelationColumns}
    (decodes : decodeLinearCombination Metadata.sourceRelationColumns
      owner.productSumTerminalOutput = some output)
    (assignment : Nat → Nat) :
    linearCombinationValue output assignment =
      baseAt assignment owner.trace.output.c1 +
        baseAt assignment owner.trace.output.c0 +
          residue (goldilocksP - 6) *
            baseAt assignment owner.trace.qSumColumn := by
  rw [SelectiveArtifactPairs.linearCombinationValue_eq_raw decodes]
  apply Fin.ext
  have modulusEq : goldilocksP = goldilocksModulus := rfl
  simp [DotOwnerKey.productSumTerminalOutput,
    SourceAssignment.RawLinearCombination.programTerms,
    SourceAssignment.RawTerm.asNatTerm, fieldResidue, baseAt, residue,
    lcEval, Fin.val_add, Fin.val_mul, Nat.add_mod, Nat.mul_mod,
    modulusEq, Nat.mod_mod]

structure LargeFactorFoldData where
  qChain : List RawRewriteStep
  c0Chain : List RawRewriteStep
  sumChain : List RawRewriteStep
  qDecoded : List (DecodedRewriteStep Metadata.sourceRelationColumns)
  c0Decoded : List (DecodedRewriteStep Metadata.sourceRelationColumns)
  sumDecoded : List (DecodedRewriteStep Metadata.sourceRelationColumns)
  qOutput : DecodedLinearCombination Metadata.sourceRelationColumns
  c0Output : DecodedLinearCombination Metadata.sourceRelationColumns
  sumOutput : DecodedLinearCombination Metadata.sourceRelationColumns

structure LargeFactorFoldWitness (owner : DotOwnerKey)
    (raws : List RawRewriteStep) (assignment : Nat → Nat)
    (data : LargeFactorFoldData) : Prop where
  partitionExact : partitionChains raws = [data.qChain, data.c0Chain, data.sumChain]
  qDecodes : DecodedSteps data.qChain data.qDecoded
  c0Decodes : DecodedSteps data.c0Chain data.c0Decoded
  sumDecodes : DecodedSteps data.sumChain data.sumDecoded
  qSourceChain : SourceChain none data.qDecoded data.qOutput
  c0SourceChain : SourceChain none data.c0Decoded data.c0Output
  sumSourceChain : SourceChain none data.sumDecoded data.sumOutput
  qOutputDecodes : decodeLinearCombination Metadata.sourceRelationColumns
    owner.qTerminalOutput = some data.qOutput
  c0OutputDecodes : decodeLinearCombination Metadata.sourceRelationColumns
    owner.productC0TerminalOutput = some data.c0Output
  sumOutputDecodes : decodeLinearCombination Metadata.sourceRelationColumns
    owner.productSumTerminalOutput = some data.sumOutput
  qOutputValue : linearCombinationValue data.qOutput assignment =
    baseAt assignment owner.trace.qSumColumn
  c0OutputValue : linearCombinationValue data.c0Output assignment =
    baseAt assignment owner.trace.output.c0 + residue (goldilocksP - 7) *
      baseAt assignment owner.trace.qSumColumn
  sumOutputValue : linearCombinationValue data.sumOutput assignment =
    baseAt assignment owner.trace.output.c1 +
      baseAt assignment owner.trace.output.c0 + residue (goldilocksP - 6) *
        baseAt assignment owner.trace.qSumColumn
  qFold : contributionSum assignment data.qDecoded =
    (qFactorStream assignment owner).foldr
      (fun value suffix => value + suffix) (0 : ProjectionProgram.F)
  productC0Fold : contributionSum assignment data.c0Decoded =
    (productC0FactorStream assignment owner).foldr
      (fun value suffix => value + suffix) (0 : ProjectionProgram.F)
  productSumFold : contributionSum assignment data.sumDecoded =
    (productSumFactorStream assignment owner).foldr
      (fun value suffix => value + suffix) (0 : ProjectionProgram.F)
/-- Source-terminal equations; the middle one retains raw `productC0`. -/
structure LargeSourceTerminalEquations (owner : DotOwnerKey)
    (assignment : Nat → Nat) : Prop where
  qSum : baseAt assignment owner.trace.qSumColumn =
    (qFactorStream assignment owner).foldr
      (fun value suffix => value + suffix) 0
  productC0 :
    baseAt assignment owner.trace.output.c0 +
        residue (goldilocksP - 7) *
          baseAt assignment owner.trace.qSumColumn =
      (productC0FactorStream assignment owner).foldr
        (fun value suffix => value + suffix) 0
  productSum :
    baseAt assignment owner.trace.output.c1 +
        baseAt assignment owner.trace.output.c0 +
          residue (goldilocksP - 6) *
            baseAt assignment owner.trace.qSumColumn =
      (productSumFactorStream assignment owner).foldr
        (fun value suffix => value + suffix) 0

private structure DirectMultiplicationFolds
    (multiplications : List ProjectionProgram.KMulTrace)
    (assignment : Nat → Nat) : Prop where
  productC1 :
    (multiplications.map fun multiplication =>
        baseAt assignment multiplication.productC1).foldr
          (fun value suffix => value + suffix) 0 =
      (multiplications.map fun multiplication =>
        residue (lcEval assignment multiplication.left.c1) *
          residue (lcEval assignment multiplication.right.c1)).foldr
            (fun value suffix => value + suffix) 0
  outputC0 :
    (multiplications.map fun multiplication =>
        baseAt assignment multiplication.output.c0).foldr
          (fun value suffix => value + suffix) 0 =
      (multiplications.map fun multiplication =>
          residue (lcEval assignment multiplication.left.c0) *
            residue (lcEval assignment multiplication.right.c0)).foldr
            (fun value suffix => value + suffix) 0 +
        7 * (multiplications.map fun multiplication =>
          residue (lcEval assignment multiplication.left.c1) *
            residue (lcEval assignment multiplication.right.c1)).foldr
              (fun value suffix => value + suffix) 0
  productSum :
    (multiplications.map fun multiplication =>
        baseAt assignment multiplication.productSum).foldr
          (fun value suffix => value + suffix) 0 =
      (multiplications.map fun multiplication =>
        residue (lcEval assignment multiplication.sumLeft) *
          residue (lcEval assignment multiplication.sumRight)).foldr
            (fun value suffix => value + suffix) 0

private theorem pairMultiplication_directFolds (assignment : Nat → Nat) :
    ∀ (base : Nat) (left right : List ProjectionProgram.KColumns),
      DefinitionsHold assignment (TerminalProgram.tracesDefinitions
        (TerminalProgram.pairMulTracesFrom base left right)) →
      DirectMultiplicationFolds
        (TerminalProgram.pairMulTracesFrom base left right) assignment := by
  have zeroOutputC0 : (0 : ProjectionProgram.F) = 0 + 7 * 0 := by
    rw [Fin.mul_zero, Fin.zero_add]
  intro base left
  induction left generalizing base with
  | nil =>
      intro right _holds
      exact ⟨rfl, zeroOutputC0, rfl⟩
  | cons leftHead leftTail inductionHypothesis =>
      intro right holds
      cases right with
      | nil => exact ⟨rfl, zeroOutputC0, rfl⟩
      | cons rightHead rightTail =>
          let multiplication := TerminalProgram.mulColumnsAt base leftHead rightHead
          have headHolds : DefinitionsHold assignment multiplication.definitions := by
            intro definition member
            apply holds definition
            change definition ∈ multiplication.definitions ++
              TerminalProgram.tracesDefinitions
                (TerminalProgram.pairMulTracesFrom (base + 5) leftTail rightTail)
            exact List.mem_append_left _ member
          have tailHolds : DefinitionsHold assignment
              (TerminalProgram.tracesDefinitions
                (TerminalProgram.pairMulTracesFrom
                  (base + 5) leftTail rightTail)) := by
            intro definition member
            apply holds definition
            change definition ∈ multiplication.definitions ++
              TerminalProgram.tracesDefinitions
                (TerminalProgram.pairMulTracesFrom (base + 5) leftTail rightTail)
            exact List.mem_append_right multiplication.definitions member
          have headComponents := TerminalProgram.mulColumnsAt_components
            base leftHead rightHead assignment headHolds
          have headOutput := TerminalProgram.mulColumnsAt_sound
            base leftHead rightHead assignment headHolds
          have tail := inductionHypothesis (base + 5) rightTail tailHolds
          have headQ : baseAt assignment multiplication.productC1 =
              residue (lcEval assignment multiplication.left.c1) *
                residue (lcEval assignment multiplication.right.c1) := by
            simpa [multiplication, TerminalProgram.mulColumnsAt,
              TerminalProgram.mulAt, TerminalProgram.columnsTerms,
              ProjectionProgram.KTerms.ofColumns, baseAt, residue, lcEval]
              using headComponents.productC1
          have headC0 : baseAt assignment multiplication.output.c0 =
              residue (lcEval assignment multiplication.left.c0) *
                  residue (lcEval assignment multiplication.right.c0) +
                7 * (residue (lcEval assignment multiplication.left.c1) *
                  residue (lcEval assignment multiplication.right.c1)) := by
            simpa [multiplication, TerminalProgram.mulColumnsAt,
              TerminalProgram.mulAt, TerminalProgram.columnsTerms,
              ProjectionProgram.KTerms.ofColumns, ProjectionProgram.KColumns.value,
              ProjectionProgram.K.mul, baseAt, residue, lcEval] using
                congrArg ProjectionProgram.K.c0 headOutput
          have headSum : baseAt assignment multiplication.productSum =
              residue (lcEval assignment multiplication.sumLeft) *
                residue (lcEval assignment multiplication.sumRight) := by
            have leftSum : residue (lcEval assignment multiplication.sumLeft) =
                baseAt assignment leftHead.c0 + baseAt assignment leftHead.c1 := by
              apply Fin.ext
              simp [multiplication, TerminalProgram.mulColumnsAt,
                TerminalProgram.mulAt, TerminalProgram.columnsTerms,
                ProjectionProgram.KTerms.ofColumns, baseAt, residue, lcEval,
                Fin.val_add, Nat.add_mod, Nat.mod_mod]
            have rightSum : residue (lcEval assignment multiplication.sumRight) =
                baseAt assignment rightHead.c0 + baseAt assignment rightHead.c1 := by
              apply Fin.ext
              simp [multiplication, TerminalProgram.mulColumnsAt,
                TerminalProgram.mulAt, TerminalProgram.columnsTerms,
                ProjectionProgram.KTerms.ofColumns, baseAt, residue, lcEval,
                Fin.val_add, Nat.add_mod, Nat.mod_mod]
            rw [leftSum, rightSum]
            exact headComponents.productSum
          constructor
          · simp only [TerminalProgram.pairMulTracesFrom, List.map_cons,
              List.foldr_cons]
            rw [headQ, tail.productC1]
          · simp only [TerminalProgram.pairMulTracesFrom, List.map_cons,
              List.foldr_cons]
            rw [headC0, tail.outputC0, fmul_add]
            simp only [multiplication]
            exact fadd_shuffle _ _ _ _
          · simp only [TerminalProgram.pairMulTracesFrom, List.map_cons,
              List.foldr_cons]
            rw [headSum, tail.productSum]

private theorem dotTrace_sourceTerminalEquations (owner : DotOwnerKey)
    (assignment : Nat → Nat)
    (definitionsHold : DefinitionsHold assignment owner.trace.definitions) :
    LargeSourceTerminalEquations owner assignment := by
  have multiplicationHolds : DefinitionsHold assignment
      (TerminalProgram.tracesDefinitions owner.trace.multiplications) := by
    intro definition member
    apply definitionsHold definition
    simp [TerminalProgram.DotTrace.definitions, member]
  have direct := pairMultiplication_directFolds assignment owner.trace.base
    owner.trace.left owner.trace.right multiplicationHolds
  change DirectMultiplicationFolds owner.trace.multiplications assignment at direct
  have components := owner.trace.components_sound assignment definitionsHold
  have negativeSeven : residue (goldilocksP - 7) =
      -(7 : ProjectionProgram.F) := by decide
  have cancel (value : ProjectionProgram.F) : value + -value = 0 := by
    rw [ProjectionProgram.fadd_comm]
    exact Lean.Grind.Fin.neg_add_cancel value
  constructor
  · exact components.qSum.trans direct.productC1
  · rw [components.outputC0, components.qSum, direct.outputC0,
      direct.productC1, negativeSeven, Lean.Grind.Fin.neg_mul,
      Lean.Grind.Fin.add_assoc, cancel, Fin.add_zero]
    rfl
  · exact components.productSumTerminal.trans direct.productSum

/-- Literal dot definitions identify decoded terminals with factor folds. -/
theorem LargeFactorFoldWitness.sourceContributionEquations
    {owner : DotOwnerKey} {raws : List RawRewriteStep}
    {assignment : Nat → Nat} {data : LargeFactorFoldData}
    (witness : LargeFactorFoldWitness owner raws assignment data)
    (definitionsHold : DefinitionsHold assignment owner.trace.definitions) :
    linearCombinationValue data.qOutput assignment =
        contributionSum assignment data.qDecoded ∧
      linearCombinationValue data.c0Output assignment =
        contributionSum assignment data.c0Decoded ∧
      linearCombinationValue data.sumOutput assignment =
        contributionSum assignment data.sumDecoded := by
  have source := dotTrace_sourceTerminalEquations owner assignment definitionsHold
  exact ⟨witness.qOutputValue.trans (source.qSum.trans witness.qFold.symm),
    witness.c0OutputValue.trans
      (source.productC0.trans witness.productC0Fold.symm),
    witness.sumOutputValue.trans
      (source.productSum.trans witness.productSumFold.symm)⟩

/-- Actual rewrite-row recurrences telescope to the exact terminal formulas. -/
theorem LargeFactorFoldWitness.sourceTerminalEquations
    {owner : DotOwnerKey} {raws : List RawRewriteStep}
    {assignment : Nat → Nat}
    {derivedValue : Nat → ProjectionProgram.F} {data : LargeFactorFoldData}
    (witness : LargeFactorFoldWitness owner raws assignment data)
    (holds : ∀ step,
      step ∈ data.qDecoded ++ data.c0Decoded ++ data.sumDecoded →
        SelectiveCompilerBridge.RewriteStepHolds assignment derivedValue step) :
    LargeSourceTerminalEquations owner assignment := by
  have qChainValue := RewriteChain.sourceValue_eq_previous_add_contributions
    witness.qSourceChain (fun step member => holds step (by simp [member]))
  have c0ChainValue := RewriteChain.sourceValue_eq_previous_add_contributions
    witness.c0SourceChain (fun step member => holds step (by simp [member]))
  have sumChainValue := RewriteChain.sourceValue_eq_previous_add_contributions
    witness.sumSourceChain (fun step member => holds step (by simp [member]))
  simp only [SelectiveCompilerBridge.rewritePreviousValue, Fin.zero_add] at qChainValue
  simp only [SelectiveCompilerBridge.rewritePreviousValue, Fin.zero_add] at c0ChainValue
  simp only [SelectiveCompilerBridge.rewritePreviousValue, Fin.zero_add] at sumChainValue
  exact
    { qSum := witness.qOutputValue.symm.trans
        (qChainValue.trans witness.qFold)
      productC0 := witness.c0OutputValue.symm.trans
        (c0ChainValue.trans witness.productC0Fold)
      productSum := witness.sumOutputValue.symm.trans
        (sumChainValue.trans witness.productSumFold) }

theorem LargeBatchWitness.factorFolds {batch : Batch}
    (witness : LargeBatchWitness batch) (assignment : Nat → Nat) :
    ∃ (owner : DotOwnerKey) (raws : List RawRewriteStep)
      (data : LargeFactorFoldData),
      owner ∈ dotOwners ∧
      owner.sourceRange = batch.descriptor.sourceRange ∧
      rawStepsFor? batch = some raws ∧
      LargeFactorFoldWitness owner raws assignment data := by
  rcases witness with
    ⟨(owner : DotOwnerKey), raws, _ownerExact, ownerMember,
      sourceRangeExact, rawsExact, _compactExact, _sourceRangesExact,
      _chainCount, _chainLengths, _targetColumns, chainsValid,
      _chainsTriangular, contributionsExact⟩
  have streams := largeContributionsCheck_sound contributionsExact
  rcases largeStreams_cases streams with
    ⟨qChain, c0Chain, sumChain, partitionExact, qBases, c0Bases,
      sumBases, qTerminal, c0Terminal, sumTerminal, qFactors, c0Factors,
      sumFactors⟩
  have generated : ∀ chain ∈ partitionChains raws,
      ∀ raw ∈ chain, raw ∈ Provenance.RewriteSteps.values := by
    intro chain chainMember raw rawMember
    apply generatedRawMember_of_rawStepsFor rawsExact raw
    rw [← partitionChains_flatten raws]
    exact List.mem_flatten.mpr ⟨chain, chainMember, rawMember⟩
  have valid : ∀ chain ∈ partitionChains raws,
      ∀ raw ∈ chain, RawRewriteStepValid raw := by
    intro chain chainMember raw rawMember
    exact generatedRaw_valid (generated chain chainMember raw rawMember)
  have qMember : qChain ∈ partitionChains raws := by
    rw [partitionExact]
    simp
  have c0Member : c0Chain ∈ partitionChains raws := by
    rw [partitionExact]
    simp
  have sumMember : sumChain ∈ partitionChains raws := by
    rw [partitionExact]
    simp
  rcases typedSourceChain_of_generated (generated qChain qMember)
      (chainsValid qChain qMember) with
    ⟨⟨qDecoded, qOutput, qRawOutput⟩, qDecodes, qSourceChain,
      qRawOutputExact, qOutputDecodes⟩
  rcases typedSourceChain_of_generated (generated c0Chain c0Member)
      (chainsValid c0Chain c0Member) with
    ⟨⟨c0Decoded, c0Output, c0RawOutput⟩, c0Decodes, c0SourceChain,
      c0RawOutputExact, c0OutputDecodes⟩
  rcases typedSourceChain_of_generated (generated sumChain sumMember)
      (chainsValid sumChain sumMember) with
    ⟨⟨sumDecoded, sumOutput, sumRawOutput⟩, sumDecodes, sumSourceChain,
      sumRawOutputExact, sumOutputDecodes⟩
  have qRawOutputEq : qRawOutput = owner.qTerminalOutput :=
    Option.some.inj (qRawOutputExact.symm.trans qTerminal)
  have c0RawOutputEq : c0RawOutput = owner.productC0TerminalOutput :=
    Option.some.inj (c0RawOutputExact.symm.trans c0Terminal)
  have sumRawOutputEq : sumRawOutput = owner.productSumTerminalOutput :=
    Option.some.inj (sumRawOutputExact.symm.trans sumTerminal)
  subst qRawOutput
  subst c0RawOutput
  subst sumRawOutput
  have qFold := contributionSum_eq_rawFactorFold qDecodes assignment qBases
    (valid qChain qMember)
  have c0Fold := contributionSum_eq_rawFactorFold c0Decodes assignment c0Bases
    (valid c0Chain c0Member)
  have sumFold := contributionSum_eq_rawFactorFold sumDecodes assignment sumBases
    (valid sumChain sumMember)
  rw [qFactors] at qFold
  rw [c0Factors] at c0Fold
  rw [sumFactors] at sumFold
  let data : LargeFactorFoldData :=
    { qChain := qChain
      c0Chain := c0Chain
      sumChain := sumChain
      qDecoded := qDecoded
      c0Decoded := c0Decoded
      sumDecoded := sumDecoded
      qOutput := qOutput
      c0Output := c0Output
      sumOutput := sumOutput
    }
  refine ⟨owner, raws, data, ownerMember, sourceRangeExact, rawsExact, ?_⟩
  exact
    { partitionExact := partitionExact
      qDecodes := qDecodes
      c0Decodes := c0Decodes
      sumDecodes := sumDecodes
      qSourceChain := qSourceChain
      c0SourceChain := c0SourceChain
      sumSourceChain := sumSourceChain
      qOutputDecodes := qOutputDecodes
      c0OutputDecodes := c0OutputDecodes
      sumOutputDecodes := sumOutputDecodes
      qOutputValue := qTerminalOutput_value qOutputDecodes assignment
      c0OutputValue := productC0TerminalOutput_value c0OutputDecodes assignment
      sumOutputValue :=
        productSumTerminalOutput_value sumOutputDecodes assignment
      qFold := by simpa [data, qFactorStream, List.foldr_map,
        dotFactorTriple_qSum_value] using qFold
      productC0Fold := by simpa [data, productC0FactorStream, List.foldr_map,
        dotFactorTriple_productC0_value] using c0Fold
      productSumFold := by simpa [data, productSumFactorStream, List.foldr_map,
        dotFactorTriple_productSum_value] using sumFold }

/-- Literal large source definitions imply the checked scalar equations. -/
theorem LargeBatchWitness.components {batch : Batch}
    (witness : LargeBatchWitness batch) (assignment : Nat → Nat)
    : ∃ (owner : DotOwnerKey) (raws : List RawRewriteStep),
      owner.sourceRange = batch.descriptor.sourceRange ∧
      rawStepsFor? batch = some raws ∧
      (DefinitionsHold assignment owner.trace.definitions →
        TerminalProgram.DotTrace.Components owner.trace assignment) := by
  rcases witness with
    ⟨(owner : DotOwnerKey), raws, _ownerExact, _ownerMember,
      sourceRangeExact, rawsExact, _compactExact, _sourceRangesExact,
      _chainCount, _chainLengths, _targetColumns, _chainsValid,
      _chainsTriangular, _contributionsExact⟩
  exact ⟨owner, raws, sourceRangeExact, rawsExact,
    TerminalProgram.DotTrace.components_sound owner.trace assignment⟩

theorem LargeBatchWitness.componentEquations {batch : Batch}
    (witness : LargeBatchWitness batch) (assignment : Nat → Nat) :
    ∃ (owner : DotOwnerKey) (raws : List RawRewriteStep),
      owner.sourceRange = batch.descriptor.sourceRange ∧
      rawStepsFor? batch = some raws ∧
      (DefinitionsHold assignment owner.trace.definitions →
        baseAt assignment owner.trace.qSumColumn =
            (owner.trace.multiplications.map fun multiplication =>
              baseAt assignment multiplication.productC1).foldr
                (fun left right => left + right) 0 ∧
          baseAt assignment owner.trace.output.c0 =
            (owner.trace.multiplications.map fun multiplication =>
              baseAt assignment multiplication.output.c0).foldr
                (fun left right => left + right) 0 ∧
          baseAt assignment owner.trace.output.c1 +
              baseAt assignment owner.trace.output.c0 +
              residue (goldilocksP - 6) *
                baseAt assignment owner.trace.qSumColumn =
            (owner.trace.multiplications.map fun multiplication =>
              baseAt assignment multiplication.productSum).foldr
                (fun left right => left + right) 0) := by
  rcases witness with
    ⟨(owner : DotOwnerKey), raws, _ownerExact, _ownerMember,
      sourceRangeExact, rawsExact, _compactExact, _sourceRangesExact,
      _chainCount, _chainLengths, _targetColumns, _chainsValid,
      _chainsTriangular, _contributionsExact⟩
  refine ⟨owner, raws, sourceRangeExact, rawsExact, ?_⟩
  intro definitionsHold
  have components := TerminalProgram.DotTrace.components_sound owner.trace
    assignment definitionsHold
  exact ⟨components.qSum, components.outputC0,
    components.productSumTerminal⟩

/-! ## Exhaustive batch eliminator -/

inductive BatchWitness (batch : Batch) : Prop where
  | small (witness : SmallBatchWitness batch) : BatchWitness batch
  | large (witness : LargeBatchWitness batch) : BatchWitness batch

theorem batchWitness_of_check {batch : Batch}
    (checked : BatchCheck batch = true) : BatchWitness batch := by
  unfold BatchCheck at checked
  split at checked
  next small => exact .small (smallBatchWitness_of_check checked)
  next large => exact .large (largeBatchWitness_of_check checked)

/-- Every one of the 462 exact generated owners has one semantic witness.
This theorem adds no acceptance premise; it only eliminates the bounded
artifact certificate proved by `RewriteBlockSemantics`. -/
theorem generatedBatchWitness {batch : Batch}
    (member : batch ∈ RewriteBatchIndex.allBatches) :
    BatchWitness batch :=
  batchWitness_of_check (generatedBatchCheck member)

/-- Public exact contract for joining the batch-local target order to a
separately proved direct pivot schedule without importing its owner here. -/
def TargetScheduleMatches (directPivotColumns : List Nat) : Prop :=
  generatedChainTargetColumns = directPivotColumns

/-! ## Exact direct-pivot schedule -/

private def compactTerminalPivotColumn? (step : CompactStep) : Option Nat :=
  match step.output with
  | .derivedProductSum _ => none
  | .source linear => linear.terms.head?.map fun term => term.column

private theorem compactTerminalPivotColumn?_ofRaw
    (raw : RawRewriteStep) :
    compactTerminalPivotColumn? (CompactStep.ofRaw raw) =
      SourceDisposition.terminalPivotColumn? raw := by
  cases raw.output <;> rfl

private theorem terminalPivot_of_rawTarget {output : RawLinearCombination}
    {target : Nat} (targetExact : rawTargetColumn? output = some target) :
    output.terms.head?.map (fun term => term.column) = some target := by
  unfold rawTargetColumn? at targetExact
  split at targetExact
  next constantZero =>
    cases termsExact : output.terms with
    | nil => simp [termsExact] at targetExact
    | cons head tail =>
        rcases head with ⟨column, coefficient⟩
        cases coefficient with
        | zero => simp [termsExact] at targetExact
        | succ coefficient =>
            cases coefficient with
            | zero =>
                simp [termsExact] at targetExact ⊢
                exact targetExact
            | succ coefficient => simp [termsExact] at targetExact
  next constantNonzero => simp at targetExact

private theorem chainTerminalPivots_eq_singleton :
    ∀ {first : RawRewriteStep} {rest : List RawRewriteStep} {target : Nat},
      RawChainLinked first.output rest →
      rawChainTarget? (first :: rest) = some target →
      (first :: rest).filterMap SourceDisposition.terminalPivotColumn? =
        [target] := by
  intro first rest
  induction rest generalizing first with
  | nil =>
      intro target linked targetExact
      cases outputExact : first.output with
      | derivedProductSum compilerIndex =>
          simp [RawChainLinked, outputExact] at linked
      | source output =>
          have rawTarget : rawTargetColumn? output = some target := by
            simpa [rawChainTarget?, rawSourceOutput?, outputExact] using
              targetExact
          have directTarget := terminalPivot_of_rawTarget rawTarget
          simp [SourceDisposition.terminalPivotColumn?, outputExact,
            directTarget]
  | cons next remaining inductionHypothesis =>
      intro target linked targetExact
      cases outputExact : first.output with
      | source output =>
          simp [RawChainLinked, outputExact] at linked
      | derivedProductSum compilerIndex =>
          simp only [RawChainLinked, outputExact] at linked
          rcases linked with ⟨_nextPrevious, remainingLinked⟩
          have remainingTarget :
              rawChainTarget? (next :: remaining) = some target := by
            simpa [rawChainTarget?, rawSourceOutput?] using targetExact
          have tailExact := inductionHypothesis remainingLinked remainingTarget
          simpa [SourceDisposition.terminalPivotColumn?, outputExact] using
            tailExact

private theorem validChainTerminalPivots_eq_singleton
    {chain : List RawRewriteStep} {target : Nat}
    (valid : RawChainValid chain)
    (targetExact : rawChainTarget? chain = some target) :
    chain.filterMap SourceDisposition.terminalPivotColumn? = [target] := by
  cases chain with
  | nil => contradiction
  | cons first rest =>
      exact chainTerminalPivots_eq_singleton valid.2 targetExact

private theorem targets_exist_of_triangular :
    ∀ {known : List Nat} {chains : List (List RawRewriteStep)},
      RawChainsTriangularFrom known chains →
      ∃ targets, chains.mapM rawChainTarget? = some targets := by
  intro known chains
  induction chains generalizing known with
  | nil =>
      intro _triangular
      exact ⟨[], rfl⟩
  | cons chain chains inductionHypothesis =>
      intro triangular
      rcases triangular with
        ⟨_output, target, _outputExact, targetExact, _shape, tail⟩
      rcases inductionHypothesis tail with ⟨targets, targetsExact⟩
      exact ⟨target :: targets, by simp [targetExact, targetsExact]⟩

private theorem chainTargets_eq_terminalPivots :
    ∀ {chains : List (List RawRewriteStep)} {targets : List Nat},
      chains.mapM rawChainTarget? = some targets →
      (∀ chain ∈ chains, RawChainValid chain) →
      targets = chains.flatMap
        (List.filterMap SourceDisposition.terminalPivotColumn?) := by
  intro chains
  induction chains with
  | nil =>
      intro targets targetExact _valid
      simp at targetExact
      subst targets
      rfl
  | cons chain chains inductionHypothesis =>
      intro targets targetExact valid
      cases chainTarget : rawChainTarget? chain with
      | none => simp [chainTarget] at targetExact
      | some target =>
          cases remainingTargets : chains.mapM rawChainTarget? with
          | none => simp [chainTarget, remainingTargets] at targetExact
          | some tailTargets =>
              simp [chainTarget, remainingTargets] at targetExact
              subst targets
              have headExact := validChainTerminalPivots_eq_singleton
                (valid chain (by simp)) chainTarget
              have tailExact := inductionHypothesis remainingTargets (by
                intro candidate member
                exact valid candidate (by simp [member]))
              simp [headExact, tailExact]

private theorem flatMap_filterMap_eq_filterMap_flatten
    {Alpha Beta : Type} (decode : Alpha → Option Beta) :
    ∀ (values : List (List Alpha)),
      values.flatMap (List.filterMap decode) =
        values.flatten.filterMap decode := by
  intro values
  induction values with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      simp [inductionHypothesis, List.filterMap_append]

private theorem filterMap_compact_ofRaw (raws : List RawRewriteStep) :
    (raws.map CompactStep.ofRaw).filterMap compactTerminalPivotColumn? =
      raws.filterMap SourceDisposition.terminalPivotColumn? := by
  induction raws with
  | nil => rfl
  | cons raw raws inductionHypothesis =>
      simp only [List.map_cons, List.filterMap_cons]
      rw [compactTerminalPivotColumn?_ofRaw raw, inductionHypothesis]

private theorem batchTargets_eq_compact {batch : Batch}
    (witness : BatchWitness batch) :
    (batchTargetColumns? batch).getD [] =
      batch.steps.filterMap compactTerminalPivotColumn? := by
  cases witness with
  | small smallWitness =>
      rcases smallWitness with
        ⟨definitions, raws, _definitionsExact, rawsExact, _sourceLength,
          _definitionCount, compactExact, _sourceRangesExact, _rawCount,
          _decodedMatches, chainsValid, chainsTriangular⟩
      rcases targets_exist_of_triangular chainsTriangular with
        ⟨targets, targetExact⟩
      have targetsAreDirect := chainTargets_eq_terminalPivots targetExact
        chainsValid
      have flattened := flatMap_filterMap_eq_filterMap_flatten
        SourceDisposition.terminalPivotColumn? (partitionChains raws)
      have batchLookup : batchTargetColumns? batch = some targets := by
        simp [batchTargetColumns?, batchChains?, rawsExact, targetExact]
      rw [batchLookup]
      simp only [Option.getD_some]
      rw [targetsAreDirect, flattened, partitionChains_flatten,
        ← filterMap_compact_ofRaw raws, compactExact]
  | large largeWitness =>
      rcases largeWitness with
        ⟨owner, raws, _ownerExact, _ownerMember, _sourceRangeExact,
          rawsExact, compactExact, _sourceRangesExact, _chainCount,
          _chainLengths, targetExact, chainsValid, _chainsTriangular,
          _contributionsExact⟩
      have targetsAreDirect := chainTargets_eq_terminalPivots targetExact
        chainsValid
      have flattened := flatMap_filterMap_eq_filterMap_flatten
        SourceDisposition.terminalPivotColumn? (partitionChains raws)
      have batchLookup :
          batchTargetColumns? batch = some owner.targetColumns := by
        simp [batchTargetColumns?, batchChains?, rawsExact, targetExact]
      rw [batchLookup]
      simp only [Option.getD_some]
      rw [targetsAreDirect, flattened, partitionChains_flatten,
        ← filterMap_compact_ofRaw raws, compactExact]

private theorem flatMap_stepTargets_eq_flatten
    (batches : List Batch) :
    batches.flatMap
        (fun batch =>
          batch.steps.filterMap compactTerminalPivotColumn?) =
      (batches.flatMap Batch.steps).filterMap
        compactTerminalPivotColumn? := by
  induction batches with
  | nil => rfl
  | cons batch batches inductionHypothesis =>
      simp [inductionHypothesis, List.filterMap_append]

private theorem targetScheduleForBatches
    (batches : List Batch)
    (generated : ∀ batch ∈ batches,
      batch ∈ RewriteBatchIndex.allBatches) :
    batches.flatMap (fun batch => (batchTargetColumns? batch).getD []) =
      batches.flatMap
        (fun batch =>
          batch.steps.filterMap compactTerminalPivotColumn?) := by
  induction batches with
  | nil => rfl
  | cons batch batches inductionHypothesis =>
      simp only [List.flatMap_cons]
      rw [batchTargets_eq_compact
          (generatedBatchWitness (generated batch (by simp))),
        inductionHypothesis (by
          intro candidate member
          exact generated candidate (by simp [member]))]

private theorem generatedTargets_eq_directStream :
    generatedChainTargetColumns =
      Provenance.RewriteSteps.values.filterMap
        SourceDisposition.terminalPivotColumn? := by
  unfold generatedChainTargetColumns
  rw [targetScheduleForBatches RewriteBatchIndex.allBatches
    (by intro batch member; exact member)]
  rw [flatMap_stepTargets_eq_flatten]
  change
    (RewriteBatchIndex.batches.flatMap Batch.steps).filterMap
        compactTerminalPivotColumn? =
      Provenance.RewriteSteps.values.filterMap
        SourceDisposition.terminalPivotColumn?
  rw [RewriteBatchIndex.batches_cover_provenance]
  exact filterMap_compact_ofRaw Provenance.RewriteSteps.values

/-- The 462 checked owner-local targets are exactly the direct 941-pivot
production schedule owned by `SourceDisposition`; no stage label or digest
is used in the equality. -/
theorem generatedTargetSchedule_exact :
    TargetScheduleMatches SourceDisposition.terminalPivotColumns := by
  unfold TargetScheduleMatches
  rw [generatedTargets_eq_directStream,
    SourceDisposition.terminalPivotColumns_exact]

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.RewriteBlockSoundness
