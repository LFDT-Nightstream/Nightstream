import Lean.Data.Json
import NightstreamFPrime.Export.Stage1.Poseidon2HashChainV1Package
import NightstreamFPrime.Export.Stage1.PerApplicationAssignmentTransport

/-!
Exports ownership and relocation metadata for the unchanged shared verifier.
The reference package remains the source of rows and witness recipes. The
manifest identifies its children and the existing application insertion rules.
It does not construct an application, change a layout, or prove Rust assembly.
-/

namespace NightstreamFPrime.Export.SharedVerifier

open NightstreamFPrime.Export
open NightstreamFPrime.Export.Stage1
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.MatrixProgram
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Spec

private abbrev referenceApplication := Poseidon2HashChainV1Package.application
private abbrev Kind := PerApplicationProductionPlan.BlockKind

/-- Coefficients of `constant + witness*w + local*l + rows*r` in that order. -/
private def dimension (constant : Nat) (witness localWords rows : Nat := 0) : Lean.Json :=
  Lean.toJson [constant, witness, localWords, rows]

private def jsonNat (value : Nat) : Lean.Json := Lean.toJson value
private def strings (values : List String) : Lean.Json := Lean.toJson values

private def referenceCounts (_ : Unit) : List Nat :=
  [referenceApplication.witnessWordCount,
    ApplicationRetainedBlocks.localCount referenceApplication,
    PerApplicationPackage.directApplicationRowCount referenceApplication]

private def referencePrivate (_ : Unit) : Nat :=
  PerApplicationPackage.directAddedPrivateColumnCount referenceApplication

private structure SourceContext where
  privateDelta : Nat
  constant : Nat
  total : Nat

@[noinline] private def sourceContext (_ : Unit) : SourceContext :=
  let privateDelta := referencePrivate ()
  { privateDelta
    constant := Data.physicalLayout.constantColumn + privateDelta
    total := Data.physicalLayout.totalColumnCount + privateDelta }

private def sharedRetainedWidth (_ : Unit) : Nat :=
  ApplicationRetainedGeometry.witnessStart referenceApplication

private def kindName : Kind → String
  | .pilotPoseidon => "pilot_poseidon"
  | .piCcsPoseidon => "pi_ccs_poseidon"
  | .piCcsOrdinary => "pi_ccs_ordinary"
  | .pilotOrdinary => "pilot_ordinary"
  | .pilotDigestBinding => "pilot_digest_binding"
  | .piCcsEndpoint => "pi_ccs_endpoint"
  | .samplerPoseidon => "pi_rlc_sampler_poseidon"
  | .samplerOrdinary => "pi_rlc_sampler_ordinary"
  | .piRlc => "pi_rlc_combination"
  | .piDec => "pi_dec"
  | .runningTransition => "running_transition"
  | .application => "application"
  | .nextPreimage => "next_preimage"
  | .recursivePublicOutput => "recursive_public_output"

private def kindOpcode (kind : Kind) : Nat :=
  match PerApplicationProductionPlan.BlockKind.format.encode kind with
  | .atom value => value
  | _ => 0

private def assignmentOpcode (kind : PerApplicationAssignmentPlan.BlockKind) : Nat :=
  match PerApplicationAssignmentPlan.BlockKind.format.encode kind with
  | .atom value => value
  | _ => 0

private def slotOpcode : LowNormSlot.Kind → Nat
  | .bit => 0
  | .centered => 1
  | .field => 2

private def sourceDomainOpcode : PerApplicationAssignmentBlocks.SourceDomain → Nat
  | .retained => 0
  | .physicalBase => 1

private def sharedRows (_ : Unit) : Nat :=
  (PerApplicationProductionPlan.canonicalKinds.filter (· != .application)).foldl
    (fun total kind => total + kind.rowCount referenceApplication) 0

private def port (name role : String) (count sourceStart retainedStart : Lean.Json)
    (slotKind : Nat := 2) :
    Lean.Json :=
  Lean.Json.mkObj [("name", .str name), ("role", .str role), ("count", count),
    ("source_start", sourceStart), ("retained_start", retainedStart),
    ("slot_kind", jsonNat slotKind)]

private def ports (_ : Unit) : Lean.Json := Lean.toJson [
  port "state_input" "private_input"
    (dimension Lifecycle.Stage1.Application.stateWordCount)
    (dimension (Layout.Stage1.ApplicationInputs.inputColumn ⟨0, by decide⟩))
    (dimension (ApplicationRetainedGeometry.inputStart referenceApplication)),
  port "state_output" "private_output"
    (dimension Lifecycle.Stage1.Application.stateWordCount)
    (dimension (Layout.Stage1.ApplicationInputs.outputColumn ⟨0, by decide⟩))
    (dimension (ApplicationRetainedGeometry.outputStart referenceApplication)),
  port "application_witness" "private_input" (dimension 0 1)
    (dimension Layout.Stage1.ApplicationInputs.witnessStart)
    (dimension (sharedRetainedWidth ())),
  port "application_local" "private_witness" (dimension 0 0 1)
    (dimension Layout.Stage1.ApplicationInputs.witnessStart 1)
    (dimension (sharedRetainedWidth ()) LowNormSlot.Kind.field.width),
  port "one" "constant" (dimension 1)
    (dimension Data.physicalLayout.constantColumn 1 1)
    (dimension encHashMarkerIndex.val) 0,
  port "prior_public_input" "public_input"
    (dimension Layout.PilotValues.priorPublicInputWords)
    (dimension (Data.physicalLayout.constantColumn + 1) 1 1)
    (dimension (PiCCSOrdinaryRetainedGeometry.freshPublicInputStart referenceApplication)),
  port "output_digest" "public_output" (dimension Layout.PilotValues.digestWords)
    (dimension (Data.physicalLayout.constantColumn + 1 +
      Layout.PilotValues.priorPublicInputWords) 1 1)
    (dimension (PilotOrdinaryRetainedGeometry.outputDigestStart referenceApplication)),
  port "verifier_context" "public_input"
    (dimension Layout.Stage1.Spartan.expectedContextColumnCount)
    (dimension Layout.Stage1.Spartan.expectedContextPublicStart 1 1)
    (dimension (PiCCSOrdinaryRetainedGeometry.expectedContextStart referenceApplication))]

private def recursivePublic : Lean.Json :=
  let firstBit := (digestBitIndexNat
    (logicalWidth := PerApplicationFixedPoint.logicalWidth referenceApplication) 0 0).val
  let wordStride := (digestBitIndexNat
    (logicalWidth := PerApplicationFixedPoint.logicalWidth referenceApplication) 1 0).val - firstBit
  Lean.Json.mkObj [
    ("role", .str "public_output"), ("start", jsonNat 0),
    ("count", jsonNat PerApplicationCanonicalPackage.logicalPublicInputCount),
    ("digest_port", .str "output_digest"), ("marker_index", jsonNat encHashMarkerIndex.val),
    ("digest_words", jsonNat Layout.PilotValues.digestWords),
    ("first_bit", jsonNat firstBit), ("word_bits", jsonNat wordStride),
    ("zero_tail_start", jsonNat (firstBit + Layout.PilotValues.digestWords * wordStride)),
    ("bit_order", .str "little_endian")]

private def geometry (_ : Unit) : Lean.Json := Lean.Json.mkObj [
  ("source_rows", dimension (Data.physicalLayout.rowCount + 5) 0 0 1),
  ("source_private", dimension Data.physicalLayout.privateColumnCount 1 1),
  ("source_constant", dimension Data.physicalLayout.constantColumn 1 1),
  ("source_public", jsonNat Data.physicalLayout.publicColumnCount),
  ("source_total", dimension Data.physicalLayout.totalColumnCount 1 1),
  ("logical_rows", dimension (sharedRows ()) 0 0 1),
  ("logical_width", dimension (sharedRetainedWidth ())
    LowNormSlot.Kind.field.width LowNormSlot.Kind.field.width),
  ("logical_public", jsonNat PerApplicationCanonicalPackage.logicalPublicInputCount),
  ("field_slot_width", jsonNat LowNormSlot.Kind.field.width),
  ("ring_degree", jsonNat ringDegree),
  ("domain", jsonNat (2 ^ Lifecycle.cubeVariables)),
  ("one_column", jsonNat encHashMarkerIndex.val)]

private def relocation (path : List Nat) (value : Lean.Json) : Lean.Json :=
  Lean.Json.mkObj [("path", Lean.toJson path), ("value", value)]

private def valueJson : Codec.Value → Lean.Json
  | .atom value => jsonNat value
  | .array values => Lean.Json.arr (values.map valueJson).toArray

/-- The generic application source connector contains no application rows.
These paths follow the existing Program/Ordinary/SourceRange codecs. -/
private def applicationMatrixRelocations (_ : Unit) : List Lean.Json := [
  relocation [0, 1, 0, 1, 0, 1] (dimension 0 0 0 1),
  relocation [0, 1, 2, 0, 1, 1] (dimension 0 1),
  relocation [0, 1, 2, 0, 1, 2, 1] (dimension 0 1),
  relocation [0, 1, 2, 0, 3, 0]
    (dimension Layout.Stage1.ApplicationInputs.witnessStart 1),
  relocation [0, 1, 2, 0, 3, 1] (dimension 0 0 1),
  relocation [0, 1, 2, 0, 3, 2, 1] (dimension 0 0 1),
  relocation [0, 1, 2, 0, 3, 2, 2]
    (dimension (sharedRetainedWidth ()) LowNormSlot.Kind.field.width)]

/-- Paths select typed source-projection fields, never coefficients or tags.
The export rejects a mapped range that straddles the private insertion. -/
private def blockRelocations (context : SourceContext) (kind : Kind) (blockIndex : Nat)
    (block : MatrixProgram.Block) : Except String (List Lean.Json) := do
  if kind = .application then return []
  let mut result := []
  match block with
  | .ordinary ordinary =>
    match ordinary.projection with
    | .identity => pure ()
    | .mapped ranges =>
      for (rangeIndex, range) in ranges.zipIdx |>.map (fun pair => (pair.2, pair.1)) do
        if range.packageStart ≥ context.constant then
          unless range.packageStart + range.count ≤ context.total do
            throw "projection range exceeds the reference public suffix"
          result := result ++ [relocation [blockIndex, 1, 3, 1, rangeIndex, 0]
            (dimension (range.packageStart - context.privateDelta) 1 1)]
        else
          unless range.packageStart + range.count ≤ Data.physicalLayout.constantColumn do
            throw "projection range crosses the application-private insertion"
    if kind = .nextPreimage then
      match ordinary.rows with
      | .indexTable _ => throw "next-preimage source rows are not a range list"
      | .rangeList ranges =>
        for (rangeIndex, range) in ranges.zipIdx |>.map (fun pair => (pair.2, pair.1)) do
          let applicationRows := PerApplicationPackage.directApplicationRowCount referenceApplication
          unless applicationRows ≤ range.start do
            throw "next-preimage row starts before the application suffix"
          result := result ++ [relocation [blockIndex, 1, 0, 1, rangeIndex, 0]
            (dimension (range.start - applicationRows) 0 0 1)]
  | _ => pure ()
  pure result

private def childMetadata (context : SourceContext) : Except String (List Lean.Json × List Lean.Json) := do
  let mut blockStart := 0
  let mut fixedRowStart := 0
  let mut afterApplication := false
  let mut children := []
  let mut relocations := []
  for kind in PerApplicationProductionPlan.canonicalKinds do
    let program := PerApplicationMatrixProgram.blockProgram referenceApplication kind
    let application := kind == .application
    let rowCount := if application then dimension 0 0 0 1
      else dimension (kind.rowCount referenceApplication)
    children := children ++ [Lean.Json.mkObj [
      ("id", .str (kindName kind)), ("opcode", jsonNat (kindOpcode kind)),
      ("block_start", jsonNat blockStart), ("block_count", jsonNat program.blocks.length),
      ("row_start", dimension fixedRowStart 0 0 (if afterApplication then 1 else 0)),
      ("row_count", rowCount), ("replaceable", Lean.toJson application)]]
    for (localIndex, block) in program.blocks.zipIdx |>.map (fun pair => (pair.2, pair.1)) do
      relocations := relocations ++ (← blockRelocations context kind (blockStart + localIndex) block)
    blockStart := blockStart + program.blocks.length
    if application then afterApplication := true
    else fixedRowStart := fixedRowStart + kind.rowCount referenceApplication
  pure (children, relocations)

private def runJson (first count : Lean.Json) (step : Nat) : Lean.Json :=
  Lean.Json.mkObj [("first", first), ("step", jsonNat step), ("count", count)]

/-- Split one monotone affine run at the declared source insertion. Derived
product sources after the package move with the same private-column delta. -/
private def sourceRun (context : SourceContext) (run : AffineRuns.Run) : List Lean.Json :=
  let before := if run.first ≥ context.constant then 0
    else if run.step = 0 then run.count
    else min run.count ((context.constant - run.first + run.step - 1) / run.step)
  let unchanged := if before = 0 then [] else
    [runJson (dimension run.first) (dimension before) run.step]
  let shifted := if before = run.count then [] else
    [runJson (dimension (run.first + before * run.step - context.privateDelta) 1 1)
      (dimension (run.count - before)) run.step]
  unchanged ++ shifted

private def assignmentBlock (context : SourceContext)
    (kind : PerApplicationAssignmentPlan.BlockKind) : Lean.Json :=
  let block := PerApplicationAssignmentBlocks.BlockPlan.ofKind referenceApplication kind
  let count := match kind with
    | .applicationWitness => dimension 0 1
    | .applicationLocal => dimension 0 0 1
    | _ => dimension block.slotCount
  let runs := match kind with
    | .applicationWitness => [runJson
        (dimension Layout.Stage1.ApplicationInputs.witnessStart) (dimension 0 1) 1]
    | .applicationLocal => [runJson
        (dimension Layout.Stage1.ApplicationInputs.witnessStart 1) (dimension 0 0 1) 1]
    | _ => block.sourceRuns.flatMap (sourceRun context)
  Lean.Json.mkObj [
    ("opcode", jsonNat (assignmentOpcode kind)),
    ("slot_kind", jsonNat (slotOpcode block.slotKind)),
    ("source_domain", jsonNat (sourceDomainOpcode block.sourceDomain)),
    ("slot_count", count), ("source_runs", Lean.toJson runs)]

private def sourceColumnFields : List String := [
  "hash_chains.input_start", "hash_chains.witness_start", "hash_chains.digest_start",
  "permutation_invocations.witness_start", "permutation_invocations.inputs.terms.column",
  "compact_row_invocations.local_start", "compact_row_invocations.input_ranges.column_start",
  "witness_batches.start", "witness_batches.recipes.var", "witness_batches.hints.source.var",
  "witness_instructions.target", "witness_instructions.a.terms.column",
  "witness_instructions.b.terms.column", "assertion_rows.a.terms.column",
  "assertion_rows.b.terms.column", "assertion_rows.c.terms.column",
  "layout.public_segments.start", "assignment.output_digest_expressions.var"]

private def sourceRelocation (context : SourceContext) : Lean.Json := Lean.Json.mkObj [
  ("source_private_start", jsonNat Data.physicalLayout.constantColumn),
  ("reference_constant", jsonNat context.constant),
  ("reference_total", jsonNat context.total),
  ("source_column_fields", strings sourceColumnFields),
  ("prefix_row_start", jsonNat 0),
  ("prefix_row_count", jsonNat Data.physicalLayout.rowCount),
  ("application_row_start", jsonNat Data.physicalLayout.rowCount),
  ("next_preimage_row_start", dimension Data.physicalLayout.rowCount 0 0 1),
  ("next_preimage_row_count", jsonNat 5),
  ("prefix_witness_end", jsonNat Data.physicalLayout.constantColumn)]

private def recipeContract : Lean.Json := Lean.Json.mkObj [
  ("row_form", .str "r1cs_a_times_b_equals_c"),
  ("expression_tags", strings ["var", "const", "add", "mul"]),
  ("hint_tags", strings ["bit", "inverse_or_zero", "quotient_five", "remainder_five"]),
  ("witness_instruction", .str "target_equals_a_times_b"),
  ("field_encoding", .str "balanced_ternary_41"),
  ("source_support", strings ["state_input", "application_witness", "state_output", "application_local"]),
  ("native_outputs_are_hints", Lean.toJson true)]

private def contracts : List String := [
  "NightstreamFPrime.Export.Stage1.PerApplicationPackage.shiftSparseRow_holds",
  "NightstreamFPrime.Export.Stage1.PerApplicationPackage.directFinalLayout_eq_finalLayout",
  "NightstreamFPrime.Export.Stage1.PerApplicationSourceProjection.base_column",
  "NightstreamFPrime.Export.Stage1.PerApplicationSourceProjection.pilot_column",
  "NightstreamFPrime.Export.Stage1.PerApplicationMatrixProgram.matrixProgram_blocks",
  "NightstreamFPrime.Export.Stage1.PerApplicationCanonicalPackage.matrixProgram_exact",
  "NightstreamFPrime.Export.Stage1.ApplicationRetainedGeometry.completeLogicalWidth_eq_applicationCounts",
  "NightstreamFPrime.Export.Stage1.ApplicationRetainedGeometry.carrierWidth_le_twoPow28_iff",
  "NightstreamFPrime.Export.Stage1.PerApplicationFixedPoint.plan_fixedPoint",
  "NightstreamFPrime.Export.Stage1.PerApplicationAssignmentBlocks.sourceRuns_expand"]

def value (_ : Unit) : Except String Lean.Json := do
  let context := sourceContext ()
  unless referenceApplication.witnessWordCount +
      ApplicationRetainedBlocks.localCount referenceApplication = context.privateDelta do
    throw "reference application private counts disagree"
  let metadata ← childMetadata context
  let profile := [goldilocksModulus, productionGlobalParams.b, productionGlobalParams.k,
    productionGlobalParams.bigB, ringDegree, Lifecycle.cubeVariables,
    Spec.ProductionRelation.matrixCount, Spec.ProductionRelation.meaningfulPortCount]
  pure (Lean.Json.mkObj [
    ("format", .str "nightstream.shared-verifier"), ("version", jsonNat 1),
    ("id", .str "shared-recursive-verifier-v1"), ("profile", Lean.toJson profile),
    ("dependencies", strings ["poseidon2-permutation-v1", "poseidon2-external-v1", "phi81-product-v1"]),
    ("parameters", strings ["witness_words", "local_words", "application_rows"]),
    ("reference", Lean.toJson (referenceCounts ())),
    ("geometry", geometry ()), ("ports", ports ()), ("recursive_public", recursivePublic),
    ("children", Lean.toJson metadata.1),
    ("matrix_relocations", Lean.toJson metadata.2),
    ("application_matrix_template", valueJson (Program.format.encode
      (PerApplicationMatrixProgram.applicationProgram referenceApplication))),
    ("application_matrix_relocations", Lean.toJson (applicationMatrixRelocations ())),
    ("assignment_blocks", Lean.toJson
      (PerApplicationAssignmentPlan.canonicalKinds.map (assignmentBlock context))),
    ("phi81_value_sources", Lean.toJson
      ((PerApplicationAssignmentTransport.phi81ValueSources referenceApplication).flatMap (sourceRun context))),
    ("source_relocation", sourceRelocation context),
    ("recipe_contract", recipeContract),
    ("required_dimension_checks", strings
      ["source_rows_le_domain", "source_total_le_domain", "ring_padded_logical_width_le_domain"]),
    ("phase_order", strings ["prior_state_hash", "pi_ccs_sumcheck", "pi_ccs_final",
      "pi_rlc", "pi_dec", "application", "output_hash"]),
    ("terminal", Lean.Json.mkObj [("running_claims", jsonNat productionShape.runningCount),
      ("fresh_claims", jsonNat productionShape.freshCount), ("all_final_rows", Lean.toJson true)]),
    ("contracts", strings contracts),
    ("proof_scope", .str "Existing theorems require a Lean Application.Program, FitsTwoPow28, and their encoding hypotheses. This manifest does not prove Rust application semantics, relocation, or assembly.")])

def write (path : System.FilePath) : IO Unit := do
  match value () with
  | .error error => throw (IO.userError error)
  | .ok manifest => IO.FS.writeFile path (manifest.compress ++ "\n")

end NightstreamFPrime.Export.SharedVerifier
