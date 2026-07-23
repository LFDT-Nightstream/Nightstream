import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.AssignmentAgreement
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.RewriteBatchIndex
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.RewriteSourceSemantics
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.SourceAssignment
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.StageProgram

/-!
Exact source-block vocabulary for the production combined-NC rewrite batches.

Owns: recovery of the full generated provenance behind each compact batch,
absolute-source-row lookup across the initial, 25 relabeled round, and
terminal programs, closed-chain partitioning, triangular source-output
targets, and the raw/typed dependency seam used by the ordered reconstruction.

Does not own: the global prior-target schedule, selected-row satisfaction,
source-program reconstruction, transcript order, parent or raw-child
authority, commitment binding, costs, or permission to remove rows.

The 445 five-row owners are the only blocks eligible for symbolic
`ExactChainMatch`.  The 17 larger owners are represented by the existing
`TerminalProgram.DotTrace.Components`; this module never symbolically
executes or natively decides their 73-, 78-, or 323-definition blocks.

Assurance tier: artifact-checked for the fixed generated profile after the
bounded certificates at the end of this file validate.
-/

/-!
Emits constraints: none; this module interprets an existing rewrite block.

| Stable stage path | Obligation | Authority class |
|---|---|---|
| `f_prime.pi_ccs_nc.delayed.combined.rewrite_block_semantics` | State the source-expression meaning of polynomial, product-sum, and linear-definition rewrites. | derived |

-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.RewriteBlockSemantics

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.CheckedProgram
open Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc
open Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized
open Decoder
open Semantics
open SelectiveCompilerBridge
open RewriteBatchIndex
open RewriteChain
open RewriteSourceSemantics.ChainAgreement

private theorem boolAnd_left {left right : Bool}
    (both : (left && right) = true) : left = true := by
  cases left <;> cases right <;> simp_all

private theorem boolAnd_right {left right : Bool}
    (both : (left && right) = true) : right = true := by
  cases left <;> cases right <;> simp_all

/-! ## Direct bounded lookup -/

private theorem decodedField_val {raw : Nat} {decoded : F}
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

/-- Read one generated rewrite record without normalizing the 1,493-record
facade.  Every branch reads one direct shard of at most 64 records; shard 23
is the actual 21-record remainder. -/
def rawStepAt? (index : Nat) : Option RawRewriteStep :=
  let offset := index % 64
  match index / 64 with
  | 0 => Provenance.RewriteSteps.Chunk0.values[offset]?
  | 1 => Provenance.RewriteSteps.Chunk1.values[offset]?
  | 2 => Provenance.RewriteSteps.Chunk2.values[offset]?
  | 3 => Provenance.RewriteSteps.Chunk3.values[offset]?
  | 4 => Provenance.RewriteSteps.Chunk4.values[offset]?
  | 5 => Provenance.RewriteSteps.Chunk5.values[offset]?
  | 6 => Provenance.RewriteSteps.Chunk6.values[offset]?
  | 7 => Provenance.RewriteSteps.Chunk7.values[offset]?
  | 8 => Provenance.RewriteSteps.Chunk8.values[offset]?
  | 9 => Provenance.RewriteSteps.Chunk9.values[offset]?
  | 10 => Provenance.RewriteSteps.Chunk10.values[offset]?
  | 11 => Provenance.RewriteSteps.Chunk11.values[offset]?
  | 12 => Provenance.RewriteSteps.Chunk12.values[offset]?
  | 13 => Provenance.RewriteSteps.Chunk13.values[offset]?
  | 14 => Provenance.RewriteSteps.Chunk14.values[offset]?
  | 15 => Provenance.RewriteSteps.Chunk15.values[offset]?
  | 16 => Provenance.RewriteSteps.Chunk16.values[offset]?
  | 17 => Provenance.RewriteSteps.Chunk17.values[offset]?
  | 18 => Provenance.RewriteSteps.Chunk18.values[offset]?
  | 19 => Provenance.RewriteSteps.Chunk19.values[offset]?
  | 20 => Provenance.RewriteSteps.Chunk20.values[offset]?
  | 21 => Provenance.RewriteSteps.Chunk21.values[offset]?
  | 22 => Provenance.RewriteSteps.Chunk22.values[offset]?
  | 23 => Provenance.RewriteSteps.Chunk23.values[offset]?
  | _ => none

def rawStepsAt? (offset count : Nat) : Option (List RawRewriteStep) :=
  (List.range count).mapM fun index => rawStepAt? (offset + index)

def rawStepsFor? (batch : Batch) : Option (List RawRewriteStep) :=
  rawStepsAt? batch.descriptor.stepOffset batch.descriptor.stepCount

private def arraySlice? {Alpha : Type} (values : Array Alpha)
    (offset count : Nat) : Option (List Alpha) :=
  (List.range count).mapM fun index => values[offset + index]?

def rangeContained (outer inner : RawRowRange) : Bool :=
  decide (outer.start ≤ inner.start ∧ inner.stop ≤ outer.stop)

private def instructionDefinition? : Instruction → Option Definition
  | .define definition => some definition
  | .check _ => none

private def initialDefinitionsFor? (sourceRange : RawRowRange) :
    Option (List Definition) :=
  if rangeContained Metadata.boundary.claimedInitialRows sourceRange then
    arraySlice? InitialProgram.definitions.toArray
      (sourceRange.start - Metadata.boundary.claimedInitialRows.start)
      (sourceRange.stop - sourceRange.start)
  else
    none

private def roundDefinitionsFor? (sourceRange : RawRowRange) :
    Option (List Definition) := do
  let round ← RoundMaps.values.find? fun round =>
    rangeContained round.rowRange sourceRange
  let instructions ← arraySlice?
    (StageProgram.roundInstructionsAt round.roundIndex).toArray
    (sourceRange.start - round.rowRange.start)
    (sourceRange.stop - sourceRange.start)
  instructions.mapM instructionDefinition?

private def terminalDefinitionsFor? (sourceRange : RawRowRange) :
    Option (List Definition) :=
  if rangeContained Metadata.boundary.terminalIdentityRows sourceRange then
    arraySlice? TerminalProgram.definitions.toArray
      (sourceRange.start - Metadata.boundary.terminalIdentityRows.start)
      (sourceRange.stop - sourceRange.start)
  else
    none

/-- Exact typed definition block at one absolute source interval.  Rewrite
ranges occur in the claimed-initial program, the 25 relabeled round programs,
or the terminal identity program.  Padding and terminal equality checks are
intentionally not accepted by this lookup. -/
def sourceDefinitionsFor? (sourceRange : RawRowRange) :
    Option (List Definition) :=
  match initialDefinitionsFor? sourceRange with
  | some definitions => some definitions
  | none =>
      match roundDefinitionsFor? sourceRange with
      | some definitions => some definitions
      | none => terminalDefinitionsFor? sourceRange

def sourceDefinitionsForBatch? (batch : Batch) : Option (List Definition) :=
  sourceDefinitionsFor? batch.descriptor.sourceRange

private theorem mapM_some_member_input {Alpha Beta : Type}
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
      intro outputs decoded output member
      cases headResult : decode head with
      | none => simp [headResult] at decoded
      | some decodedHead =>
          cases tailResult : tail.mapM decode with
          | none => simp [headResult, tailResult] at decoded
          | some decodedTail =>
              simp [headResult, tailResult] at decoded
              subst outputs
              simp only [List.mem_cons] at member
              rcases member with rfl | tailMember
              · exact ⟨head, by simp, headResult⟩
              · rcases inductionHypothesis tailResult output tailMember with
                  ⟨input, inputMember, inputDecodes⟩
                exact ⟨input, by simp [inputMember], inputDecodes⟩

private theorem arraySlice?_member {Alpha : Type} {values : Array Alpha}
    {offset count : Nat} {slice : List Alpha}
    (lookup : arraySlice? values offset count = some slice) :
    ∀ value ∈ slice, value ∈ values.toList := by
  intro value member
  unfold arraySlice? at lookup
  rcases mapM_some_member_input
      (fun index => values[offset + index]?) lookup value member with
    ⟨index, _indexMember, valueLookup⟩
  rcases getElem?_eq_some_iff.mp valueLookup with ⟨bound, valueExact⟩
  have inArray : values[offset + index]'bound ∈ values.toList :=
    Array.getElem_mem_toList bound
  rwa [valueExact] at inArray

private theorem definition_mem_of_define_mem
    {definition : Definition} {instructions : List Instruction}
    (member : Instruction.define definition ∈ instructions) :
    definition ∈ CheckedProgram.definitions instructions := by
  apply List.mem_filterMap.mpr
  exact ⟨.define definition, member, rfl⟩

private theorem initialDefinitionsFor?_member
    {sourceRange : RawRowRange} {definitions : List Definition}
    (lookup : initialDefinitionsFor? sourceRange = some definitions) :
    ∀ definition ∈ definitions,
      definition ∈ CheckedProgram.definitions StageProgram.instructions := by
  intro definition member
  unfold initialDefinitionsFor? at lookup
  split at lookup
  next _contained =>
    have initialMember : definition ∈ InitialProgram.definitions := by
      simpa using arraySlice?_member lookup definition member
    apply definition_mem_of_define_mem
    simp [StageProgram.instructions, StageProgram.initialInstructions,
      initialMember]
  next _notContained => simp at lookup

private theorem terminalDefinitionsFor?_member
    {sourceRange : RawRowRange} {definitions : List Definition}
    (lookup : terminalDefinitionsFor? sourceRange = some definitions) :
    ∀ definition ∈ definitions,
      definition ∈ CheckedProgram.definitions StageProgram.instructions := by
  intro definition member
  unfold terminalDefinitionsFor? at lookup
  split at lookup
  next _contained =>
    have terminalMember : definition ∈ TerminalProgram.definitions := by
      simpa using arraySlice?_member lookup definition member
    apply definition_mem_of_define_mem
    simp [StageProgram.instructions, StageProgram.terminalInstructions,
      terminalMember]
  next _notContained => simp at lookup

private theorem roundDefinitionsFor?_member
    {sourceRange : RawRowRange} {definitions : List Definition}
    (lookup : roundDefinitionsFor? sourceRange = some definitions) :
    ∀ definition ∈ definitions,
      definition ∈ CheckedProgram.definitions StageProgram.instructions := by
  intro definition member
  unfold roundDefinitionsFor? at lookup
  cases roundResult : RoundMaps.values.find? (fun round =>
      rangeContained round.rowRange sourceRange) with
  | none => simp [roundResult] at lookup
  | some round =>
      cases instructionsResult : arraySlice?
          (StageProgram.roundInstructionsAt round.roundIndex).toArray
          (sourceRange.start - round.rowRange.start)
          (sourceRange.stop - sourceRange.start) with
      | none => simp [roundResult, instructionsResult] at lookup
      | some instructions =>
          simp [roundResult, instructionsResult] at lookup
          rcases mapM_some_member_input instructionDefinition? lookup
              definition member with
            ⟨instruction, instructionMember, instructionDecodes⟩
          have instructionInRound : instruction ∈
              StageProgram.roundInstructionsAt round.roundIndex := by
            simpa using arraySlice?_member instructionsResult instruction
              instructionMember
          cases instruction with
          | check _ => simp [instructionDefinition?] at instructionDecodes
          | define current =>
              simp [instructionDefinition?] at instructionDecodes
              subst current
              have roundMember : round ∈ RoundMaps.values :=
                List.mem_of_find?_eq_some roundResult
              have roundBound : round.roundIndex < sumcheckRoundCount :=
                (RoundArtifact.generatedRoundMapsValid.2.2 round roundMember).2.2.2.2.1
              have stageMember :
                  StageProgram.roundInstructionsAt round.roundIndex ∈
                    StageProgram.roundInstructionStages := by
                unfold StageProgram.roundInstructionStages
                exact List.mem_ofFn.mpr
                  ⟨⟨round.roundIndex, roundBound⟩, rfl⟩
              have roundInstructionMember :
                  Instruction.define definition ∈
                    StageProgram.roundInstructions := by
                unfold StageProgram.roundInstructions
                exact List.mem_flatten.mpr
                  ⟨StageProgram.roundInstructionsAt round.roundIndex,
                    stageMember, instructionInRound⟩
              apply definition_mem_of_define_mem
              simp [StageProgram.instructions, roundInstructionMember]

/-- A successful absolute source-definition lookup is a literal slice of the
production checked program. No digest, stage label, or source satisfaction
premise participates in this subset theorem. -/
theorem generatedDefinitionMember_of_sourceDefinitionsForBatch
    {batch : Batch} {definitions : List Definition}
    (lookup : sourceDefinitionsForBatch? batch = some definitions) :
    ∀ definition ∈ definitions,
      definition ∈ CheckedProgram.definitions StageProgram.instructions := by
  intro definition member
  unfold sourceDefinitionsForBatch? at lookup
  unfold sourceDefinitionsFor? at lookup
  cases initialResult : initialDefinitionsFor? batch.descriptor.sourceRange with
  | some initialDefinitions =>
      simp [initialResult] at lookup
      subst definitions
      exact initialDefinitionsFor?_member initialResult definition member
  | none =>
      cases roundResult : roundDefinitionsFor? batch.descriptor.sourceRange with
      | some roundDefinitions =>
          simp [initialResult, roundResult] at lookup
          subst definitions
          exact roundDefinitionsFor?_member roundResult definition member
      | none =>
          simp [initialResult, roundResult] at lookup
          exact terminalDefinitionsFor?_member lookup definition member

/-! ## Raw dependency vocabulary -/

def rawLinearReferences (value : RawLinearCombination) : List Nat :=
  0 :: value.terms.map fun term => term.column

def rawFactorReferences (factor : RawProductFactor) : List Nat :=
  rawLinearReferences factor.left ++ rawLinearReferences factor.right

def rawContributionReferences (step : RawRewriteStep) : List Nat :=
  rawLinearReferences step.base ++
    step.factors.flatMap rawFactorReferences

def RawContributionReferencesOnly (known : List Nat)
    (step : RawRewriteStep) : Prop :=
  ∀ column ∈ rawContributionReferences step, column ∈ known

instance (known : List Nat) (step : RawRewriteStep) :
    Decidable (RawContributionReferencesOnly known step) := by
  unfold RawContributionReferencesOnly
  infer_instance

private theorem decodedTerms_reference_of_decode
    {columns : Nat} {raw : List RawTerm}
    {decoded : List (DecodedTerm columns)}
    (decodes : decodeTerms columns raw = some decoded) :
    ∀ term ∈ decoded, term.column.val ∈ raw.map (fun item => item.column) := by
  induction raw generalizing decoded with
  | nil =>
      simp [decodeTerms] at decodes
      subst decoded
      simp
  | cons head tail inductionHypothesis =>
      cases headResult : decodeTerm columns head with
      | none => simp [decodeTerms, headResult] at decodes
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
              intro term member
              simp only [List.mem_cons] at member
              rcases member with rfl | tailMember
              · have headWords :=
                  SourceDecodeBridge.termAsNatTerm_eq_of_decodeTerm headResult
                have headColumn : term.column.val = head.column :=
                  congrArg Prod.fst headWords
                simp [headColumn]
              · exact List.mem_cons_of_mem _
                  (inductionHypothesis tailDecodes term tailMember)

private theorem decodedLinear_references_of_decode
    {columns : Nat} {raw : RawLinearCombination}
    {decoded : DecodedLinearCombination columns}
    (decodes : decodeLinearCombination columns raw = some decoded) :
    ∀ term ∈ linearCombinationTerms decoded,
      term.1 ∈ rawLinearReferences raw := by
  unfold decodeLinearCombination at decodes
  cases constantResult : decodeField raw.constant with
  | none => simp [constantResult] at decodes
  | some constant =>
      cases termsResult : decodeTerms columns raw.terms with
      | none => simp [constantResult, termsResult] at decodes
      | some terms =>
          simp [constantResult, termsResult] at decodes
          subst decoded
          intro term member
          simp only [linearCombinationTerms, List.mem_cons] at member
          rcases member with rfl | member
          · simp [rawLinearReferences]
          · rcases List.mem_map.mp member with
              ⟨decodedTerm, decodedMember, rfl⟩
            simp only [termAsNatTerm]
            exact List.mem_cons_of_mem _
              (decodedTerms_reference_of_decode termsResult decodedTerm
                decodedMember)

private theorem decodedFactors_reference_of_decode
    {columns : Nat} {raw : List RawProductFactor}
    {decoded : List (DecodedProductFactor columns)}
    (decodes : raw.mapM (decodeProductFactor columns) = some decoded) :
    ∀ factor ∈ decoded,
      ∃ rawFactor, rawFactor ∈ raw ∧
        decodeProductFactor columns rawFactor = some factor := by
  induction raw generalizing decoded with
  | nil => simp at decodes; subst decoded; simp
  | cons head tail inductionHypothesis =>
      cases headResult : decodeProductFactor columns head with
      | none => simp [headResult] at decodes
      | some decodedHead =>
          cases tailResult : tail.mapM (decodeProductFactor columns) with
          | none => simp [headResult, tailResult] at decodes
          | some decodedTail =>
              simp [headResult, tailResult] at decodes
              subst decoded
              intro factor member
              simp only [List.mem_cons] at member
              rcases member with rfl | member
              · exact ⟨head, by simp, headResult⟩
              · rcases inductionHypothesis tailResult factor member with
                  ⟨rawFactor, rawMember, decodedEq⟩
                exact ⟨rawFactor, by simp [rawMember], decodedEq⟩

private theorem decodedFactor_references_of_decode
    {columns : Nat} {raw : RawProductFactor}
    {decoded : DecodedProductFactor columns}
    (decodes : decodeProductFactor columns raw = some decoded) :
    AssignmentAgreement.ProductFactorReferencesOnly
      (rawFactorReferences raw) decoded := by
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
              · intro term member
                exact List.mem_append_left _
                  (decodedLinear_references_of_decode leftResult term member)
              · intro term member
                exact List.mem_append_right _
                  (decodedLinear_references_of_decode rightResult term member)

/-- Lossless decoding turns the proof-free raw reference set into the typed
`ContributionReferencesOnly` premise consumed by `AssignmentAgreement`. -/
theorem contributionReferencesOnly_of_raw
    {known : List Nat} {raw : RawRewriteStep}
    {decoded : DecodedRewriteStep Metadata.sourceRelationColumns}
    (decodes : decodeRewriteStep Metadata.sourceRelationRows
      Metadata.sourceRelationColumns Metadata.finalRelationRows raw =
        some decoded)
    (references : RawContributionReferencesOnly known raw) :
    AssignmentAgreement.ContributionReferencesOnly known decoded := by
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
              | none => simp [outputResult, baseResult, factorsResult] at decodes
              | some factors =>
                  simp [outputResult, baseResult, factorsResult] at decodes
                  subst decoded
                  constructor
                  · intro term member
                    apply references term.1
                    apply List.mem_append_left
                    exact decodedLinear_references_of_decode baseResult term member
                  · intro factor member
                    rcases decodedFactors_reference_of_decode factorsResult
                        factor member with
                      ⟨rawFactor, rawMember, factorDecodes⟩
                    have localReferences :=
                      decodedFactor_references_of_decode factorDecodes
                    constructor
                    · intro term termMember
                      apply references term.1
                      apply List.mem_append_right
                      apply List.mem_flatMap.mpr
                      exact ⟨rawFactor, rawMember,
                        localReferences.1 term termMember⟩
                    · intro term termMember
                      apply references term.1
                      apply List.mem_append_right
                      apply List.mem_flatMap.mpr
                      exact ⟨rawFactor, rawMember,
                        localReferences.2 term termMember⟩
    next sourceRangesFalse => simp at decodes
  next emittedBoundFalse => simp at decodes

/-! ## Closed-chain partition and triangular outputs -/

def closesChain (step : RawRewriteStep) : Bool :=
  match step.output with
  | .source _ => true
  | .derivedProductSum _ => false

private def partitionChainsAux (current : List RawRewriteStep) :
    List RawRewriteStep -> List (List RawRewriteStep)
  | [] => if current.isEmpty then [] else [current]
  | step :: rest =>
      let next := current ++ [step]
      if closesChain step then
        next :: partitionChainsAux [] rest
      else
        partitionChainsAux next rest

def partitionChains (steps : List RawRewriteStep) :
    List (List RawRewriteStep) :=
  partitionChainsAux [] steps

private theorem partitionChainsAux_flatten
    (current steps : List RawRewriteStep) :
    (partitionChainsAux current steps).flatten = current ++ steps := by
  induction steps generalizing current with
  | nil =>
      cases current <;> simp [partitionChainsAux]
  | cons step rest inductionHypothesis =>
      simp only [partitionChainsAux]
      split
      · rw [List.flatten_cons, inductionHypothesis]
        simp [List.append_assoc]
      · rw [inductionHypothesis]
        simp [List.append_assoc]

/-- Chain partitioning neither drops nor duplicates a generated rewrite
record.  This kernel fact lets downstream semantic witnesses inherit the
direct-shard validity certificate one raw record at a time. -/
theorem partitionChains_flatten (steps : List RawRewriteStep) :
    (partitionChains steps).flatten = steps := by
  simpa [partitionChains] using partitionChainsAux_flatten [] steps

def rawSourceOutput? (chain : List RawRewriteStep) :
    Option RawLinearCombination := do
  let terminal ← chain.getLast?
  match terminal.output with
  | .source output => some output
  | .derivedProductSum _ => none

def rawTargetColumn? (output : RawLinearCombination) : Option Nat :=
  if output.constant = 0 then
    match output.terms with
    | { column, coefficient := 1 } :: _ => some column
    | _ => none
  else
    none

def rawChainTarget? (chain : List RawRewriteStep) : Option Nat := do
  let output ← rawSourceOutput? chain
  rawTargetColumn? output

def batchChains? (batch : Batch) : Option (List (List RawRewriteStep)) := do
  pure (partitionChains (← rawStepsFor? batch))

def batchTargetColumns? (batch : Batch) : Option (List Nat) := do
  (← batchChains? batch).mapM rawChainTarget?

/-- The exact generated chain-target order: within every large dot owner this
is `qSum`, then `c0`, then `c1`; five-row owners contribute their two limb
targets in generated recurrence order. -/
def generatedChainTargetColumns : List Nat :=
  RewriteBatchIndex.allBatches.flatMap fun batch =>
    (batchTargetColumns? batch).getD []

def RawTriangularAt (known : List Nat)
    (output : RawLinearCombination) (target : Nat) : Prop :=
  output.constant = 0 ∧
    ∃ tail,
      output.terms = { column := target, coefficient := 1 } :: tail ∧
      ∀ term ∈ tail, term.column ∈ known

instance (known : List Nat) (output : RawLinearCombination) (target : Nat) :
    Decidable (RawTriangularAt known output target) := by
  by_cases constantZero : output.constant = 0
  · cases termsExact : output.terms with
    | nil =>
        apply isFalse
        intro triangular
        rcases triangular.2 with ⟨tail, impossible, _⟩
        simp [termsExact] at impossible
    | cons head tail =>
        let targetTerm : RawTerm :=
          { column := target, coefficient := 1 }
        by_cases headExact : head = targetTerm
        · by_cases tailReferences :
              ∀ term ∈ tail, term.column ∈ known
          · apply isTrue
            refine ⟨constantZero, tail, ?_, tailReferences⟩
            rw [termsExact, headExact]
          · apply isFalse
            intro triangular
            rcases triangular.2 with ⟨otherTail, outputExact, references⟩
            rw [termsExact] at outputExact
            have tailExact : tail = otherTail :=
              (List.cons.inj outputExact).2
            apply tailReferences
            simpa [tailExact] using references
        · apply isFalse
          intro triangular
          rcases triangular.2 with ⟨otherTail, outputExact, _⟩
          rw [termsExact] at outputExact
          exact headExact (List.cons.inj outputExact).1
  · exact isFalse fun triangular => constantZero triangular.1

def triangularCheck : List Nat → List (List RawRewriteStep) → Bool
  | _, [] => true
  | known, chain :: chains =>
      match rawSourceOutput? chain, rawChainTarget? chain with
      | some output, some target =>
          decide (RawTriangularAt known output target) &&
            triangularCheck (known ++ [target]) chains
      | _, _ => false

def RawChainLinked : RawRewriteOutput → List RawRewriteStep → Prop
  | .source _, [] => True
  | .source _, _ :: _ => False
  | .derivedProductSum previous, next :: tail =>
      next.previous = some previous ∧ RawChainLinked next.output tail
  | .derivedProductSum _, [] => False

def RawChainValid : List RawRewriteStep → Prop
  | [] => False
  | first :: rest =>
      first.previous = none ∧ RawChainLinked first.output rest

private def rawChainLinkedDecidable :
    (output : RawRewriteOutput) → (steps : List RawRewriteStep) →
      Decidable (RawChainLinked output steps)
  | .source _, [] => isTrue trivial
  | .source _, _ :: _ => isFalse id
  | .derivedProductSum _, [] => isFalse id
  | .derivedProductSum previous, next :: remaining => by
      letI := rawChainLinkedDecidable next.output remaining
      unfold RawChainLinked
      infer_instance

instance (chain : List RawRewriteStep) : Decidable (RawChainValid chain) := by
  cases chain with
  | nil => simp only [RawChainValid]; infer_instance
  | cons head tail =>
      simp only [RawChainValid]
      letI := rawChainLinkedDecidable head.output tail
      infer_instance

def chainShapeCheck (chains : List (List RawRewriteStep)) : Bool :=
  chains.all fun chain => decide (RawChainValid chain)

private def decodeRewrite? (raw : RawRewriteStep) :=
  decodeRewriteStep Metadata.sourceRelationRows Metadata.sourceRelationColumns
    Metadata.finalRelationRows raw

/-- This checker is used only when `definitions.length = 5` and on one
singleton chain.  Its input consists of five proof-free definitions and one
proof-free raw recurrence; it never receives a list of decoded records. -/
def exactSingletonCheck (definitions : List Definition)
    (raw : RawRewriteStep) : Bool :=
  match decodeRewrite? raw with
  | none => false
  | some decoded =>
      match decoded.output with
      | .derivedProductSum _ => false
      | .source output =>
          decide (ExactChainMatch definitions [decoded] output)

def rawRangeExact (sourceRange : RawRowRange)
    (raw : RawRewriteStep) : Bool :=
  decide (raw.sourceRows = [sourceRange])

def SmallBatchCheck (batch : Batch) : Bool :=
  match sourceDefinitionsForBatch? batch, rawStepsFor? batch with
  | some definitions, some raws =>
      let chains := partitionChains raws
      decide (batch.descriptor.sourceRange.stop -
          batch.descriptor.sourceRange.start = 5) &&
        decide (definitions.length = 5) &&
        decide (raws.map CompactStep.ofRaw = batch.steps) &&
        raws.all (rawRangeExact batch.descriptor.sourceRange) &&
        decide (raws.length = 2) &&
        raws.all (exactSingletonCheck definitions) &&
        chainShapeCheck chains && triangularCheck [] chains
  | _, _ => false

/-! ## Structural dot-block owners -/

structure DotOwnerKey where
  definitionOffset : Nat
  trace : TerminalProgram.DotTrace
deriving DecidableEq, Repr

def outputDotOwnersFrom : Nat → List TerminalProgram.OutputTrace →
    List DotOwnerKey
  | _, [] => []
  | offset, trace :: traces =>
      { definitionOffset := offset, trace := trace.evaluation } ::
        outputDotOwnersFrom (offset + trace.definitions.length) traces

def outputDotOwners : List DotOwnerKey :=
  outputDotOwnersFrom TerminalProgram.chiDefinitions.length
    TerminalProgram.outputTraces

def ordinaryDotOwner : DotOwnerKey :=
  { definitionOffset := TerminalProgram.chiDefinitions.length +
      TerminalProgram.outputDefinitions.length +
      TerminalProgram.gammaPowers.definitions.length
    trace := TerminalProgram.ordinarySum }

def runningDotOwner : DotOwnerKey :=
  { definitionOffset := TerminalProgram.laneComputationDefinitions.length +
      TerminalProgram.ordinaryDefinitions.length +
      TerminalProgram.radixConstant.definitions.length +
      TerminalProgram.radixPowers.definitions.length
    trace := TerminalProgram.runningSum }

def dotOwners : List DotOwnerKey :=
  outputDotOwners ++ [ordinaryDotOwner, runningDotOwner]

def DotOwnerKey.sourceRange (owner : DotOwnerKey) : RawRowRange :=
  { start := Metadata.boundary.terminalIdentityRows.start +
      owner.definitionOffset
    stop := Metadata.boundary.terminalIdentityRows.start +
      owner.definitionOffset + owner.trace.definitions.length }

def DotOwnerKey.targetColumns (owner : DotOwnerKey) : List Nat :=
  [owner.trace.qSumColumn, owner.trace.output.c0, owner.trace.output.c1]

def DotOwnerKey.chainLength (owner : DotOwnerKey) : Nat :=
  (owner.trace.multiplications.length + 4) / 5

/-- Exact terminal source form for the `productC1` contribution chain. -/
def DotOwnerKey.qTerminalOutput (owner : DotOwnerKey) :
    RawLinearCombination :=
  { constant := 0
    terms := [{ column := owner.trace.qSumColumn, coefficient := 1 }] }

/-- Exact terminal source form for the raw `productC0` contribution chain.
The already visible `qSum` pivot supplies the quadratic-extension `7` term. -/
def DotOwnerKey.productC0TerminalOutput (owner : DotOwnerKey) :
    RawLinearCombination :=
  { constant := 0
    terms :=
      [{ column := owner.trace.output.c0, coefficient := 1 },
       { column := owner.trace.qSumColumn,
         coefficient := goldilocksP - 7 }] }

/-- Exact terminal source form for the `productSum` contribution chain. -/
def DotOwnerKey.productSumTerminalOutput (owner : DotOwnerKey) :
    RawLinearCombination :=
  { constant := 0
    terms :=
      [{ column := owner.trace.output.c1, coefficient := 1 },
       { column := owner.trace.output.c0, coefficient := 1 },
       { column := owner.trace.qSumColumn,
         coefficient := goldilocksP - 6 }] }

/-! ## Compact large-block contribution observation -/

private def rawTerm (term : Nat × Nat) : RawTerm :=
  { column := term.1, coefficient := term.2 }

private def rawLinear (terms : List (Nat × Nat)) : RawLinearCombination :=
  { constant := 0, terms := terms.map rawTerm }

private def rawFactor (left right : List (Nat × Nat)) : RawProductFactor :=
  { left := rawLinear left
    right := rawLinear right
    coefficient := 1 }

private theorem rawLinear_programTerms (terms : List (Nat × Nat)) :
    SourceAssignment.RawLinearCombination.programTerms (rawLinear terms) =
      terms := by
  simp [rawLinear, rawTerm,
    SourceAssignment.RawLinearCombination.programTerms,
    SourceAssignment.RawTerm.asNatTerm, Function.comp_def]

/-- One proof-free observation covers the three compiler contribution streams
allocated by one literal five-row `KMulTrace`: `productC1`, `productC0`, and
`productSum`, in that order. -/
structure DotFactorTriple where
  qSum : RawProductFactor
  outputC0 : RawProductFactor
  productSum : RawProductFactor
deriving DecidableEq, Repr

def DotFactorTriple.ofMultiplication
    (multiplication : ProjectionProgram.KMulTrace) : DotFactorTriple :=
  { qSum := rawFactor multiplication.left.c1 multiplication.right.c1
    outputC0 := rawFactor multiplication.left.c0 multiplication.right.c0
    productSum := rawFactor multiplication.sumLeft multiplication.sumRight }

def rawFactorValue (assignment : Nat → Nat)
    (factor : RawProductFactor) : F :=
  fieldResidue factor.coefficient *
    fieldResidue (lcEval assignment
      (SourceAssignment.RawLinearCombination.programTerms factor.left)) *
    fieldResidue (lcEval assignment
      (SourceAssignment.RawLinearCombination.programTerms factor.right))

private theorem fieldResidue_eq_projectionResidue (value : Nat) :
    fieldResidue value = ProjectionProgram.residue value := by
  have modulusEq : goldilocksP = goldilocksModulus := rfl
  apply Fin.ext
  simp [fieldResidue, ProjectionProgram.residue, modulusEq]

private theorem oneMulThen (left right : ProjectionProgram.F) :
    (1 * left) * right = left * right :=
  congrArg (fun value : ProjectionProgram.F => value * right)
    (Fin.one_mul left)

theorem dotFactorTriple_qSum_value (assignment : Nat → Nat)
    (multiplication : ProjectionProgram.KMulTrace) :
    rawFactorValue assignment
        (DotFactorTriple.ofMultiplication multiplication).qSum =
      ProjectionProgram.residue (lcEval assignment multiplication.left.c1) *
        ProjectionProgram.residue
          (lcEval assignment multiplication.right.c1) := by
  simp only [rawFactorValue, DotFactorTriple.ofMultiplication, rawFactor,
    rawLinear_programTerms, fieldResidue_eq_projectionResidue,
    ProjectionProgram.residue_one]
  exact oneMulThen _ _

theorem dotFactorTriple_productC0_value (assignment : Nat → Nat)
    (multiplication : ProjectionProgram.KMulTrace) :
    rawFactorValue assignment
        (DotFactorTriple.ofMultiplication multiplication).outputC0 =
      ProjectionProgram.residue (lcEval assignment multiplication.left.c0) *
        ProjectionProgram.residue
          (lcEval assignment multiplication.right.c0) := by
  simp only [rawFactorValue, DotFactorTriple.ofMultiplication, rawFactor,
    rawLinear_programTerms, fieldResidue_eq_projectionResidue,
    ProjectionProgram.residue_one]
  exact oneMulThen _ _

theorem dotFactorTriple_productSum_value (assignment : Nat → Nat)
    (multiplication : ProjectionProgram.KMulTrace) :
    rawFactorValue assignment
        (DotFactorTriple.ofMultiplication multiplication).productSum =
      ProjectionProgram.residue (lcEval assignment multiplication.sumLeft) *
        ProjectionProgram.residue
          (lcEval assignment multiplication.sumRight) := by
  simp only [rawFactorValue, DotFactorTriple.ofMultiplication, rawFactor,
    rawLinear_programTerms, fieldResidue_eq_projectionResidue,
    ProjectionProgram.residue_one]
  exact oneMulThen _ _

private def zipFactorTriples :
    List RawProductFactor → List RawProductFactor → List RawProductFactor →
      Option (List DotFactorTriple)
  | [], [], [] => some []
  | q :: qs, c0 :: c0s, sum :: sums => do
      pure ({ qSum := q, outputC0 := c0, productSum := sum } ::
        (← zipFactorTriples qs c0s sums))
  | _, _, _ => none

def chainFactors (chain : List RawRewriteStep) :
    List RawProductFactor :=
  chain.flatMap RawRewriteStep.factors

private def chainBasesZero (chain : List RawRewriteStep) : Bool :=
  chain.all fun step => decide (step.base = rawLinear [])

/-- Kernel meaning of the compact three-stream observation used for a large
dot owner.  The middle stream is deliberately named `productC0`: the
triangular source terminal later combines it with the already visible
`qSum` target to reconstruct the aggregate output `c0`. -/
def LargeContributionStreams (owner : DotOwnerKey) :
    List (List RawRewriteStep) → Prop
  | [qChain, c0Chain, sumChain] =>
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
          (DotFactorTriple.ofMultiplication multiplication).productSum)
  | _ => False

private theorem zipFactorTriples_sound :
    ∀ {qFactors c0Factors sumFactors triples},
      zipFactorTriples qFactors c0Factors sumFactors = some triples →
      qFactors = triples.map DotFactorTriple.qSum ∧
      c0Factors = triples.map DotFactorTriple.outputC0 ∧
      sumFactors = triples.map DotFactorTriple.productSum := by
  intro qFactors
  induction qFactors with
  | nil =>
      intro c0Factors sumFactors triples decoded
      cases c0Factors with
      | nil =>
          cases sumFactors with
          | nil =>
              simp [zipFactorTriples] at decoded
              subst triples
              exact ⟨rfl, rfl, rfl⟩
          | cons _ _ => simp [zipFactorTriples] at decoded
      | cons _ _ => simp [zipFactorTriples] at decoded
  | cons qFactor qFactors inductionHypothesis =>
      intro c0Factors sumFactors triples decoded
      cases c0Factors with
      | nil => simp [zipFactorTriples] at decoded
      | cons c0Factor c0Factors =>
          cases sumFactors with
          | nil => simp [zipFactorTriples] at decoded
          | cons sumFactor sumFactors =>
              cases tailResult : zipFactorTriples qFactors c0Factors
                  sumFactors with
              | none => simp [zipFactorTriples, tailResult] at decoded
              | some tailTriples =>
                  simp [zipFactorTriples, tailResult] at decoded
                  subst triples
                  rcases inductionHypothesis tailResult with
                    ⟨qExact, c0Exact, sumExact⟩
                  simp [qExact, c0Exact, sumExact]

/-- Exact coefficient observation for one large dot owner. It compares only
three compact factor streams with one `DotFactorTriple` per multiplication;
it does not inspect the owner's 73-, 78-, or 323-definition list. -/
def largeContributionsCheck (owner : DotOwnerKey)
    (chains : List (List RawRewriteStep)) : Bool :=
  match chains with
  | [qChain, c0Chain, sumChain] =>
      chainBasesZero qChain && chainBasesZero c0Chain &&
        chainBasesZero sumChain &&
        decide (rawSourceOutput? qChain = some owner.qTerminalOutput ∧
          rawSourceOutput? c0Chain =
            some owner.productC0TerminalOutput ∧
          rawSourceOutput? sumChain =
            some owner.productSumTerminalOutput) &&
        decide (zipFactorTriples (chainFactors qChain)
            (chainFactors c0Chain) (chainFactors sumChain) =
          some (owner.trace.multiplications.map
            DotFactorTriple.ofMultiplication))
  | _ => false

theorem largeContributionsCheck_sound {owner : DotOwnerKey}
    {chains : List (List RawRewriteStep)}
    (checked : largeContributionsCheck owner chains = true) :
    LargeContributionStreams owner chains := by
  cases chains with
  | nil => simp [largeContributionsCheck, LargeContributionStreams] at checked
  | cons qChain chains =>
      cases chains with
      | nil => simp [largeContributionsCheck, LargeContributionStreams] at checked
      | cons c0Chain chains =>
          cases chains with
          | nil =>
              simp [largeContributionsCheck, LargeContributionStreams] at checked
          | cons sumChain chains =>
              cases chains with
              | cons extra rest =>
                  simp [largeContributionsCheck, LargeContributionStreams]
                    at checked
              | nil =>
                  simp only [largeContributionsCheck] at checked
                  have factorCheck := boolAnd_right checked
                  have prefix4 := boolAnd_left checked
                  have terminalCheck := boolAnd_right prefix4
                  have prefix3 := boolAnd_left prefix4
                  have sumBaseCheck := boolAnd_right prefix3
                  have prefix2 := boolAnd_left prefix3
                  have c0BaseCheck := boolAnd_right prefix2
                  have qBaseCheck := boolAnd_left prefix2
                  have factorsExact := of_decide_eq_true factorCheck
                  have terminalsExact := of_decide_eq_true terminalCheck
                  rcases zipFactorTriples_sound factorsExact with
                    ⟨qExact, c0Exact, sumExact⟩
                  have qStream :
                      chainFactors qChain =
                        (owner.trace.multiplications.map fun multiplication =>
                          (DotFactorTriple.ofMultiplication multiplication).qSum) := by
                    simpa only [List.map_map, Function.comp_apply] using qExact
                  have c0Stream :
                      chainFactors c0Chain =
                        (owner.trace.multiplications.map fun multiplication =>
                          (DotFactorTriple.ofMultiplication multiplication).outputC0) := by
                    simpa only [List.map_map, Function.comp_apply] using c0Exact
                  have sumStream :
                      chainFactors sumChain =
                        (owner.trace.multiplications.map fun multiplication =>
                          (DotFactorTriple.ofMultiplication multiplication).productSum) := by
                    simpa only [List.map_map, Function.comp_apply] using sumExact
                  simp only [LargeContributionStreams]
                  refine ⟨?_, ?_, ?_, terminalsExact.1,
                    terminalsExact.2.1, terminalsExact.2.2,
                    qStream, c0Stream, sumStream⟩
                  · intro step member
                    exact of_decide_eq_true
                      ((List.all_eq_true.mp qBaseCheck) step member)
                  · intro step member
                    exact of_decide_eq_true
                      ((List.all_eq_true.mp c0BaseCheck) step member)
                  · intro step member
                    exact of_decide_eq_true
                      ((List.all_eq_true.mp sumBaseCheck) step member)

def LargeBatchCheck (batch : Batch) : Bool :=
  match dotOwners.find? fun owner => decide (owner.sourceRange =
      batch.descriptor.sourceRange), rawStepsFor? batch with
  | some owner, some raws =>
      let chains := partitionChains raws
      decide (raws.map CompactStep.ofRaw = batch.steps) &&
        raws.all (rawRangeExact batch.descriptor.sourceRange) &&
        decide (chains.length = 3) &&
        decide (chains.map List.length =
          List.replicate 3 owner.chainLength) &&
        decide (chains.mapM rawChainTarget? = some owner.targetColumns) &&
        chainShapeCheck chains && triangularCheck [] chains &&
        largeContributionsCheck owner chains
  | _, _ => false

def BatchCheck (batch : Batch) : Bool :=
  if batch.descriptor.sourceRange.stop -
      batch.descriptor.sourceRange.start = 5 then
    SmallBatchCheck batch
  else
    LargeBatchCheck batch

/-- Record-count budget of one executable batch group. Five-row owners charge
their two raws and five source definitions. Dot owners charge their at-most-39
raws, one key, and one `DotFactorTriple` per multiplication. Their large
definition lists are never part of a native subject. -/
def batchCertificateRecords (batch : Batch) : Nat :=
  if batch.descriptor.sourceRange.stop -
      batch.descriptor.sourceRange.start = 5 then
    batch.descriptor.stepCount + 5
  else
    match dotOwners.find? fun owner => decide (owner.sourceRange =
        batch.descriptor.sourceRange) with
    | some owner => batch.descriptor.stepCount +
        owner.trace.multiplications.length + 1
    | none => 257

def certificateRecords (batches : List Batch) : Nat :=
  (batches.map batchCertificateRecords).sum

def BatchChunkCheck (batches : List Batch) : Bool :=
  batches.all BatchCheck && decide (certificateRecords batches ≤ 256)

/-! ## Bounded generated certificates

Each theorem below examines one existing closed-batch chunk.  The exact
proof-free subject sizes are, respectively,
`217, 224, 224, 224, 224, 224, 224, 224, 175, 118, 236, 118, 236, 118,
236, 118, 236, 118, 104/167, 214, 224, 220, 224, 70` records. Chunk 18 is
the sole source chunk split into two certificates: its first 64-multiplication
dot owner is 104 records and the remaining owner plus nine five-row owners is
167 records. The final carry is seven records. No decoded value is stored in
a subject; `exactSingletonCheck` immediately projects its one-record decode
and five-definition match to a Boolean.
-/

set_option maxRecDepth 100000 in
private theorem batchChunk0Certificate :
    BatchChunkCheck RewriteBatchIndex.result0.closed = true := by
  native_decide

set_option maxRecDepth 100000 in
private theorem batchChunk1Certificate :
    BatchChunkCheck RewriteBatchIndex.result1.closed = true := by
  native_decide

set_option maxRecDepth 100000 in
private theorem batchChunk2Certificate :
    BatchChunkCheck RewriteBatchIndex.result2.closed = true := by
  native_decide

set_option maxRecDepth 100000 in
private theorem batchChunk3Certificate :
    BatchChunkCheck RewriteBatchIndex.result3.closed = true := by
  native_decide

set_option maxRecDepth 100000 in
private theorem batchChunk4Certificate :
    BatchChunkCheck RewriteBatchIndex.result4.closed = true := by
  native_decide

set_option maxRecDepth 100000 in
private theorem batchChunk5Certificate :
    BatchChunkCheck RewriteBatchIndex.result5.closed = true := by
  native_decide

set_option maxRecDepth 100000 in
private theorem batchChunk6Certificate :
    BatchChunkCheck RewriteBatchIndex.result6.closed = true := by
  native_decide

set_option maxRecDepth 100000 in
private theorem batchChunk7Certificate :
    BatchChunkCheck RewriteBatchIndex.result7.closed = true := by
  native_decide

set_option maxRecDepth 100000 in
private theorem batchChunk8Certificate :
    BatchChunkCheck RewriteBatchIndex.result8.closed = true := by
  native_decide

set_option maxRecDepth 100000 in
private theorem batchChunk9Certificate :
    BatchChunkCheck RewriteBatchIndex.result9.closed = true := by
  native_decide

set_option maxRecDepth 100000 in
private theorem batchChunk10Certificate :
    BatchChunkCheck RewriteBatchIndex.result10.closed = true := by
  native_decide

set_option maxRecDepth 100000 in
private theorem batchChunk11Certificate :
    BatchChunkCheck RewriteBatchIndex.result11.closed = true := by
  native_decide

set_option maxRecDepth 100000 in
private theorem batchChunk12Certificate :
    BatchChunkCheck RewriteBatchIndex.result12.closed = true := by
  native_decide

set_option maxRecDepth 100000 in
private theorem batchChunk13Certificate :
    BatchChunkCheck RewriteBatchIndex.result13.closed = true := by
  native_decide

set_option maxRecDepth 100000 in
private theorem batchChunk14Certificate :
    BatchChunkCheck RewriteBatchIndex.result14.closed = true := by
  native_decide

set_option maxRecDepth 100000 in
private theorem batchChunk15Certificate :
    BatchChunkCheck RewriteBatchIndex.result15.closed = true := by
  native_decide

set_option maxRecDepth 100000 in
private theorem batchChunk16Certificate :
    BatchChunkCheck RewriteBatchIndex.result16.closed = true := by
  native_decide

set_option maxRecDepth 100000 in
private theorem batchChunk17Certificate :
    BatchChunkCheck RewriteBatchIndex.result17.closed = true := by
  native_decide

set_option maxRecDepth 100000 in
private theorem batchChunk18HeadCertificate :
    BatchChunkCheck (RewriteBatchIndex.result18.closed.take 1) = true := by
  native_decide

set_option maxRecDepth 100000 in
private theorem batchChunk18TailCertificate :
    BatchChunkCheck (RewriteBatchIndex.result18.closed.drop 1) = true := by
  native_decide

set_option maxRecDepth 100000 in
private theorem batchChunk19Certificate :
    BatchChunkCheck RewriteBatchIndex.result19.closed = true := by
  native_decide

set_option maxRecDepth 100000 in
private theorem batchChunk20Certificate :
    BatchChunkCheck RewriteBatchIndex.result20.closed = true := by
  native_decide

set_option maxRecDepth 100000 in
private theorem batchChunk21Certificate :
    BatchChunkCheck RewriteBatchIndex.result21.closed = true := by
  native_decide

set_option maxRecDepth 100000 in
private theorem batchChunk22Certificate :
    BatchChunkCheck RewriteBatchIndex.result22.closed = true := by
  native_decide

set_option maxRecDepth 100000 in
private theorem batchChunk23Certificate :
    BatchChunkCheck RewriteBatchIndex.result23.closed = true := by
  native_decide

set_option maxRecDepth 100000 in
private theorem finalCarryCertificate :
    BatchChunkCheck [RewriteBatchIndex.carry23] = true := by
  native_decide

private theorem batchCheck_of_chunkCheck {batches : List Batch}
    (checked : BatchChunkCheck batches = true)
    {batch : Batch} (member : batch ∈ batches) :
    BatchCheck batch = true := by
  unfold BatchChunkCheck at checked
  cases allResult : batches.all BatchCheck with
  | false => simp [allResult] at checked
  | true =>
      exact (List.all_eq_true.mp allResult) batch member

theorem generatedBatchCheck {batch : Batch}
    (member : batch ∈ RewriteBatchIndex.allBatches) :
    BatchCheck batch = true := by
  have split : batch ∈ RewriteBatchIndex.batchChunks.flatten ∨
      batch ∈ [RewriteBatchIndex.carry23] := by
    simpa [RewriteBatchIndex.allBatches, RewriteBatchIndex.batches] using member
  rcases split with inChunks | inCarry
  · rcases List.mem_flatten.mp inChunks with ⟨chunk, chunkMember, batchMember⟩
    simp only [RewriteBatchIndex.batchChunks, List.mem_cons,
      List.not_mem_nil, or_false] at chunkMember
    rcases chunkMember with
      rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl |
      rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl |
      rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
    · exact batchCheck_of_chunkCheck batchChunk0Certificate batchMember
    · exact batchCheck_of_chunkCheck batchChunk1Certificate batchMember
    · exact batchCheck_of_chunkCheck batchChunk2Certificate batchMember
    · exact batchCheck_of_chunkCheck batchChunk3Certificate batchMember
    · exact batchCheck_of_chunkCheck batchChunk4Certificate batchMember
    · exact batchCheck_of_chunkCheck batchChunk5Certificate batchMember
    · exact batchCheck_of_chunkCheck batchChunk6Certificate batchMember
    · exact batchCheck_of_chunkCheck batchChunk7Certificate batchMember
    · exact batchCheck_of_chunkCheck batchChunk8Certificate batchMember
    · exact batchCheck_of_chunkCheck batchChunk9Certificate batchMember
    · exact batchCheck_of_chunkCheck batchChunk10Certificate batchMember
    · exact batchCheck_of_chunkCheck batchChunk11Certificate batchMember
    · exact batchCheck_of_chunkCheck batchChunk12Certificate batchMember
    · exact batchCheck_of_chunkCheck batchChunk13Certificate batchMember
    · exact batchCheck_of_chunkCheck batchChunk14Certificate batchMember
    · exact batchCheck_of_chunkCheck batchChunk15Certificate batchMember
    · exact batchCheck_of_chunkCheck batchChunk16Certificate batchMember
    · exact batchCheck_of_chunkCheck batchChunk17Certificate batchMember
    · rw [← List.take_append_drop 1 RewriteBatchIndex.result18.closed] at batchMember
      rcases List.mem_append.mp batchMember with headMember | tailMember
      · exact batchCheck_of_chunkCheck batchChunk18HeadCertificate headMember
      · exact batchCheck_of_chunkCheck batchChunk18TailCertificate tailMember
    · exact batchCheck_of_chunkCheck batchChunk19Certificate batchMember
    · exact batchCheck_of_chunkCheck batchChunk20Certificate batchMember
    · exact batchCheck_of_chunkCheck batchChunk21Certificate batchMember
    · exact batchCheck_of_chunkCheck batchChunk22Certificate batchMember
    · exact batchCheck_of_chunkCheck batchChunk23Certificate batchMember
  · exact batchCheck_of_chunkCheck finalCarryCertificate inCarry

def RawChainsTriangularFrom : List Nat → List (List RawRewriteStep) → Prop
  | _, [] => True
  | known, chain :: chains =>
      ∃ output target,
        rawSourceOutput? chain = some output ∧
        rawChainTarget? chain = some target ∧
        RawTriangularAt known output target ∧
        RawChainsTriangularFrom (known ++ [target]) chains

theorem triangularCheck_sound {known : List Nat}
    {chains : List (List RawRewriteStep)}
    (checked : triangularCheck known chains = true) :
    RawChainsTriangularFrom known chains := by
  induction chains generalizing known with
  | nil => trivial
  | cons chain chains inductionHypothesis =>
      unfold triangularCheck at checked
      cases outputResult : rawSourceOutput? chain with
      | none => simp [outputResult] at checked
      | some output =>
          cases targetResult : rawChainTarget? chain with
          | none => simp [outputResult, targetResult] at checked
          | some target =>
              cases localResult : decide
                  (RawTriangularAt known output target) with
              | false =>
                  simp [outputResult, targetResult, localResult] at checked
              | true =>
                  have tailChecked :
                      triangularCheck (known ++ [target]) chains = true := by
                    simpa [outputResult, targetResult, localResult] using checked
                  exact ⟨output, target, outputResult, targetResult,
                    of_decide_eq_true localResult,
                    inductionHypothesis tailChecked⟩

theorem chainShapeCheck_sound {chains : List (List RawRewriteStep)}
    (checked : chainShapeCheck chains = true) :
    ∀ chain ∈ chains, RawChainValid chain := by
  intro chain member
  exact of_decide_eq_true
    ((List.all_eq_true.mp checked) chain member)

theorem exactSingletonCheck_sound {definitions : List Definition}
    {raw : RawRewriteStep}
    (checked : exactSingletonCheck definitions raw = true) :
    ∃ decoded output,
      decodeRewrite? raw = some decoded ∧
      decoded.output = .source output ∧
      ExactChainMatch definitions [decoded] output := by
  unfold exactSingletonCheck at checked
  cases decodeResult : decodeRewrite? raw with
  | none => simp [decodeResult] at checked
  | some decoded =>
      cases decodedOutput : decoded.output with
      | derivedProductSum compilerIndex =>
          simp [decodeResult, decodedOutput] at checked
      | source output =>
          refine ⟨decoded, output, rfl, decodedOutput, ?_⟩
          apply of_decide_eq_true
          simpa only [decodeResult, decodedOutput] using checked

/-! ## Typed triangular target transport -/

def TypedTriangularAt {columns : Nat} (known : List Nat)
    (output : DecodedLinearCombination columns) (target : Nat) : Prop :=
  output.constant = 0 ∧
    ∃ head tail,
      output.terms = head :: tail ∧
      head.column.val = target ∧ head.coefficient = 1 ∧
      ∀ term ∈ tail, term.column.val ∈ known

theorem typedTriangular_of_decode
    {columns : Nat} {known : List Nat} {target : Nat}
    {raw : RawLinearCombination}
    {decoded : DecodedLinearCombination columns}
    (decodes : decodeLinearCombination columns raw = some decoded)
    (triangular : RawTriangularAt known raw target) :
    TypedTriangularAt known decoded target := by
  rcases triangular with ⟨constantZero, tail, termsExact, tailReferences⟩
  unfold decodeLinearCombination at decodes
  cases constantResult : decodeField raw.constant with
  | none => simp [constantResult] at decodes
  | some constant =>
      cases termsResult : decodeTerms columns raw.terms with
      | none => simp [constantResult, termsResult] at decodes
      | some terms =>
          simp [constantResult, termsResult] at decodes
          subst decoded
          rw [termsExact] at termsResult
          cases headResult : decodeTerm columns
              { column := target, coefficient := 1 } with
          | none => simp [decodeTerms, headResult] at termsResult
          | some head =>
              cases tailResult : tail.mapM (decodeTerm columns) with
              | none =>
                  simp [decodeTerms, headResult, tailResult] at termsResult
              | some decodedTail =>
                  simp [decodeTerms, headResult, tailResult] at termsResult
                  subst terms
                  have headWords :=
                    SourceDecodeBridge.termAsNatTerm_eq_of_decodeTerm headResult
                  have headColumn : head.column.val = target := by
                    exact congrArg Prod.fst headWords
                  have headCoefficientVal : head.coefficient.val = 1 := by
                    exact congrArg Prod.snd headWords
                  have headCoefficient : head.coefficient = 1 := by
                    apply Fin.ext
                    exact headCoefficientVal
                  refine ⟨?_, head, decodedTail, rfl, headColumn,
                    headCoefficient, ?_⟩
                  · apply Fin.ext
                    have constantVal := decodedField_val constantResult
                    simpa [constantZero] using constantVal
                  · intro term member
                    have rawMember := decodedTerms_reference_of_decode
                      (by simpa [decodeTerms] using tailResult) term member
                    rcases List.mem_map.mp rawMember with
                      ⟨rawTerm, rawTermMember, columnExact⟩
                    rw [← columnExact]
                    exact tailReferences rawTerm rawTermMember

private theorem linearCombinationValue_of_typedTriangularComponents
    {columns : Nat} {output : DecodedLinearCombination columns}
    {target : Nat} {head : DecodedTerm columns}
    {tail : List (DecodedTerm columns)}
    {assignment : Nat → Nat}
    (constantZero : output.constant = 0)
    (termsExact : output.terms = head :: tail)
    (headColumn : head.column.val = target)
    (headCoefficient : head.coefficient = 1) :
    linearCombinationValue output assignment =
      fieldResidue (assignment target) +
        fieldResidue (lcEval assignment (termsAsNatTerms tail)) := by
  apply Fin.ext
  have modulusEq : goldilocksP = goldilocksModulus := rfl
  have oneVal : (1 : F).val = (1 : Nat) := rfl
  simp only [linearCombinationValue, linearCombinationTerms, termsExact,
    termsAsNatTerms, List.map_cons, termAsNatTerm, constantZero,
    headColumn, headCoefficient, Fin.val_zero, oneVal, fieldResidue,
    Fin.val_add]
  rw [Program.lcEval_eq_raw_mod, Program.lcEval_eq_raw_mod]
  simp only [Program.rawLcEval, Nat.zero_mul, Nat.zero_add, Nat.one_mul,
    Nat.add_zero, Nat.mod_mod, modulusEq]
  change
    (assignment target +
        Program.rawLcEval assignment (termsAsNatTerms tail)) %
      goldilocksModulus =
    (assignment target % goldilocksModulus +
        Program.rawLcEval assignment (termsAsNatTerms tail) %
          goldilocksModulus) % goldilocksModulus
  exact Nat.add_mod (assignment target)
    (Program.rawLcEval assignment (termsAsNatTerms tail))
    goldilocksModulus

private theorem field_add_right_cancel {left right suffix : F}
    (equal : left + suffix = right + suffix) : left = right := by
  calc
    left = (left + suffix) + -suffix := by
      rw [Lean.Grind.Fin.add_assoc,
        Lean.Grind.Fin.add_comm suffix (-suffix),
        Lean.Grind.Fin.neg_add_cancel, Fin.add_zero]
    _ = (right + suffix) + -suffix := congrArg (fun value => value + -suffix) equal
    _ = right := by
      rw [Lean.Grind.Fin.add_assoc,
        Lean.Grind.Fin.add_comm suffix (-suffix),
        Lean.Grind.Fin.neg_add_cancel, Fin.add_zero]

theorem typedTriangular_target_eq_of_value_eq
    {columns : Nat} {known : List Nat}
    {output : DecodedLinearCombination columns} {target : Nat}
    {left right : Nat -> Nat}
    (triangular : TypedTriangularAt known output target)
    (agreement : AgreeOn left right known)
    (leftCanonical : left target < goldilocksP)
    (rightCanonical : right target < goldilocksP)
    (valueEqual : linearCombinationValue output left =
      linearCombinationValue output right) :
    left target = right target := by
  rcases triangular with
    ⟨constantZero, head, tail, termsExact, headColumn, headCoefficient,
      tailReferences⟩
  have leftShape := linearCombinationValue_of_typedTriangularComponents
    (assignment := left) constantZero termsExact headColumn headCoefficient
  have rightShape := linearCombinationValue_of_typedTriangularComponents
    (assignment := right) constantZero termsExact headColumn headCoefficient
  have tailAgreement :
      lcEval left (termsAsNatTerms tail) =
        lcEval right (termsAsNatTerms tail) := by
    apply AssignmentAgreement.lcEval_eq_of_agreeOn agreement
    intro term member
    rcases List.mem_map.mp member with ⟨decodedTerm, decodedMember, rfl⟩
    exact tailReferences decodedTerm decodedMember
  have residueEqual : fieldResidue (left target) = fieldResidue (right target) := by
    rw [leftShape, rightShape] at valueEqual
    have tailResidues :
        fieldResidue (lcEval left (termsAsNatTerms tail)) =
          fieldResidue (lcEval right (termsAsNatTerms tail)) := by
      rw [tailAgreement]
    rw [tailResidues] at valueEqual
    exact field_add_right_cancel valueEqual
  exact AssignmentAgreement.fieldResidue_injective_of_canonical
    leftCanonical rightCanonical residueEqual

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.RewriteBlockSemantics
