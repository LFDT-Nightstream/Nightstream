import Lean.Data.Json
import NightstreamFPrime.Export.Stage1.Wide.PhysicalMatrixSource
import NightstreamFPrime.Export.Stage1.Wide.AssignmentTransport
import NightstreamFPrime.Export.Stage1.Poseidon2HashChainV1Package

/-! Export typed ownership and relocation fields for the wide shared verifier.
The pinned package supplies the rows and recipes. The manifest preserves its
shared children when an application replaces the reference. -/

namespace NightstreamFPrime.Export.SharedVerifier

open NightstreamFPrime.Export
open NightstreamFPrime.Export.Stage1
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.MatrixProgram
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Spec

private abbrev referenceApplication := Poseidon2HashChainV1Package.application
private abbrev Kind := PerApplicationProductionPlan.BlockKind

private def dimension (constant : Nat) (witness localWords rows : Nat := 0) : Lean.Json :=
  Lean.toJson [constant, witness, localWords, rows]
private def jsonNat (value : Nat) : Lean.Json := Lean.toJson value
private def strings (values : List String) : Lean.Json := Lean.toJson values

private def referenceCounts (_ : Unit) : List Nat :=
  [referenceApplication.witnessWordCount,
    ApplicationRetainedBlocks.localCount referenceApplication,
    PerApplicationPackage.directApplicationRowCount referenceApplication]

private def sharedRetainedWidth (_ : Unit) : Nat :=
  Wide.RetainedLayout.hashEnd referenceApplication +
    (Wide.RetainedLayout.sharedEnd referenceApplication - Wide.RetainedLayout.sharedStart referenceApplication)

private structure SourceContext where
  privateDelta : Nat
  retainedDelta : Nat
  commonStart : Nat
  referenceWidth : Nat
  referenceProjection : SourceProjection
  outputStart : Nat
  quotientStart : Nat
  constant : Nat
  total : Nat
  prefixRows : Nat

@[noinline] private def sourceContext (_ : Unit) : Except String SourceContext := do
  let privateDelta := PerApplicationPackage.directAddedPrivateColumnCount referenceApplication
  let retainedDelta := Wide.RetainedLayout.applicationCount referenceApplication
  unless referenceApplication.witnessWordCount + ApplicationRetainedBlocks.localCount referenceApplication = privateDelta do
    throw "reference application private counts disagree"
  return {
    privateDelta
    retainedDelta
    commonStart := sharedRetainedWidth () + retainedDelta
    referenceWidth := PerApplicationFixedPoint.logicalWidth referenceApplication
    referenceProjection := Wide.MatrixProjection.projection referenceApplication
    outputStart := Wide.RetainedLayout.outputStart referenceApplication
    quotientStart := Wide.RetainedLayout.quotientStart referenceApplication
    constant := Layout.Stage1.Wide.SourceOrder.constantColumn + privateDelta
    total := Layout.Stage1.Wide.SourceOrder.totalColumns + privateDelta
    prefixRows := ← Wide.PhysicalRelabel.row Data.physicalLayout.rowCount }

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

private def slotOpcode : LowNormSlot.Kind → Nat
  | .bit => 0
  | .centered => 1
  | .field => 2

private def childProgram (compiled : PiRlcWideSampler.RangePlan.Compiled)
    (application : Lifecycle.Stage1.Application.Program) (kind : Kind) : Except String Program := do
  let sampler := Wide.PiRLCGeometry.sampler (Wide.Stage1Plan.piRlcInterface application)
  let program := match kind with
    | .samplerPoseidon => PiRlcWideSampler.PoseidonMatrix.program sampler
    | .samplerOrdinary => PiRlcWideSampler.BatchMatrix.ranges compiled sampler
    | .piRlc => Wide.ProductMatrix.matrixProgram application
    | _ => Wide.MatrixProjection.program application (PerApplicationMatrixProgram.blockProgram application kind)
  Wide.PhysicalMatrixSource.relocateProgram
    (PerApplicationPackage.directAddedPrivateColumnCount application) program

private structure ChildPrograms where
  kind : Kind
  program : Program

private def children (compiled : PiRlcWideSampler.RangePlan.Compiled) : Except String (List ChildPrograms) :=
  PerApplicationProductionPlan.canonicalKinds.mapM fun kind => do
    return ⟨kind, ← childProgram compiled referenceApplication kind⟩

private def port (name role : String) (count sourceStart retainedStart : Lean.Json)
    (slotKind : Nat := 2) : Lean.Json :=
  Lean.Json.mkObj [("name", .str name), ("role", .str role), ("count", count),
    ("source_start", sourceStart), ("retained_start", retainedStart), ("slot_kind", jsonNat slotKind)]

private def retained (column : Nat) : Except String Nat :=
  match Wide.RetainedLayout.column? referenceApplication column with
  | some value => .ok value
  | none => .error s!"shared verifier port has no wide coordinate: {column}"

private def ports (_ : Unit) : Except String Lean.Json := do
  let privateStart := Layout.Stage1.Wide.SourceOrder.privateColumns
  return Lean.toJson [
    port "state_input" "private_input" (dimension Lifecycle.Stage1.Application.stateWordCount)
      (dimension (Layout.Stage1.ApplicationInputs.inputColumn ⟨0, by decide⟩))
      (dimension (← retained (ApplicationOrdinaryGeometry.inputStart referenceApplication))),
    port "state_output" "private_output" (dimension Lifecycle.Stage1.Application.stateWordCount)
      (dimension (Layout.Stage1.ApplicationInputs.outputColumn ⟨0, by decide⟩))
      (dimension (← retained (ApplicationOrdinaryGeometry.outputStart referenceApplication))),
    port "application_witness" "private_input" (dimension 0 1)
      (dimension privateStart) (dimension (sharedRetainedWidth ())),
    port "application_local" "private_witness" (dimension 0 0 1)
      (dimension privateStart 1) (dimension (sharedRetainedWidth ()) LowNormSlot.Kind.field.width),
    port "one" "constant" (dimension 1) (dimension privateStart 1 1)
      (dimension encHashMarkerIndex.val) 0,
    port "prior_public_input" "public_input" (dimension Layout.PilotValues.priorPublicInputWords)
      (dimension (privateStart + 1) 1 1)
      (dimension (← retained (PiCCSOrdinaryRetainedGeometry.freshPublicInputStart referenceApplication))),
    port "output_digest" "public_output" (dimension Layout.PilotValues.digestWords)
      (dimension (privateStart + 1 + Layout.PilotValues.priorPublicInputWords) 1 1)
      (dimension (← retained (PilotOrdinaryRetainedGeometry.outputDigestStart referenceApplication))),
    port "verifier_context" "public_input" (dimension Layout.Stage1.Spartan.expectedContextColumnCount)
      (dimension (privateStart + 1 + Layout.PilotSpartan.publicColumnCount) 1 1)
      (dimension (← retained (PiCCSOrdinaryRetainedGeometry.expectedContextStart referenceApplication)))]

private def recursivePublic : Lean.Json :=
  let firstBit := (digestBitIndexNat
    (logicalWidth := Wide.RetainedLayout.logicalWidth referenceApplication) 0 0).val
  let wordStride := (digestBitIndexNat
    (logicalWidth := Wide.RetainedLayout.logicalWidth referenceApplication) 1 0).val - firstBit
  Lean.Json.mkObj [
    ("role", .str "public_output"), ("start", jsonNat 0),
    ("count", jsonNat PerApplicationCanonicalPackage.logicalPublicInputCount),
    ("digest_port", .str "output_digest"), ("marker_index", jsonNat encHashMarkerIndex.val),
    ("digest_words", jsonNat Layout.PilotValues.digestWords),
    ("first_bit", jsonNat firstBit), ("word_bits", jsonNat wordStride),
    ("zero_tail_start", jsonNat (firstBit + Layout.PilotValues.digestWords * wordStride)),
    ("bit_order", .str "little_endian")]

private def geometry (context : SourceContext) (programs : List ChildPrograms) : Lean.Json :=
  let sharedRows := (programs.filter (·.kind != .application)).foldl
    (fun total child => total + child.program.rowCount) 0
  Lean.Json.mkObj [
    ("source_rows", dimension (context.prefixRows + 5) 0 0 1),
    ("source_private", dimension Layout.Stage1.Wide.SourceOrder.privateColumns 1 1),
    ("source_constant", dimension Layout.Stage1.Wide.SourceOrder.constantColumn 1 1),
    ("source_public", jsonNat Data.physicalLayout.publicColumnCount),
    ("source_total", dimension Layout.Stage1.Wide.SourceOrder.totalColumns 1 1),
    ("logical_rows", dimension sharedRows 0 0 1),
    ("logical_width", dimension (sharedRetainedWidth () + Wide.PiRLCGeometry.coordinateCount)
      LowNormSlot.Kind.field.width LowNormSlot.Kind.field.width),
    ("logical_public", jsonNat PerApplicationCanonicalPackage.logicalPublicInputCount),
    ("field_slot_width", jsonNat LowNormSlot.Kind.field.width), ("ring_degree", jsonNat ringDegree),
    ("domain", jsonNat (2 ^ Lifecycle.cubeVariables)), ("one_column", jsonNat encHashMarkerIndex.val)]

private def valueJson : Codec.Value → Lean.Json
  | .atom value => jsonNat value
  | .array values => Lean.Json.arr (values.map valueJson).toArray

private structure Relocation where
  path : List Nat
  value : Lean.Json

private def Relocation.json (relocation : Relocation) (blockStart : Nat := 0) : Lean.Json :=
  let path := match relocation.path with
    | block :: rest => (blockStart + block) :: rest
    | [] => []
  Lean.Json.mkObj [("path", Lean.toJson path), ("value", relocation.value)]

/-- Only columns in the wide sampler tail move with the application slots. -/
private def tailColumn (context : SourceContext) (path : List Nat) (column : Nat) : List Relocation :=
  if context.commonStart ≤ column then
    [⟨path, dimension (column - context.retainedDelta) 41 41⟩]
  else []

private def retainedRelocations (context : SourceContext) (path : List Nat)
    (block : RetainedBlock) : List Relocation := tailColumn context (path ++ [2]) block.start

private def substitutionRelocations (context : SourceContext) (path : List Nat)
    (substitution : SourceSubstitution) : List Relocation :=
  (substitution.ranges.zipIdx).flatMap (fun (range, index) =>
    retainedRelocations context (path ++ [0, index, 2]) range.retained) ++
  (substitution.grids.zipIdx).flatMap (fun (grid, index) =>
    retainedRelocations context (path ++ [1, index, 6]) grid.retained)

private def formRelocations (context : SourceContext) (path : List Nat) (form : WireForm) : List Relocation :=
  (form.entries.zipIdx).flatMap fun (entry, index) => tailColumn context (path ++ [index, 0]) entry.column

private def inputRelocations (context : SourceContext) (path : List Nat)
    (program : PoseidonInput.Program) : List Relocation :=
  (program.rules.zipIdx).flatMap fun (rule, index) =>
    let path := path ++ [index, 1]
    match rule.term with
    | .retained block .. | .external block .. | .taggedRetained block .. =>
        retainedRelocations context (path ++ [1]) block
    | .sparse forms _ => (forms.toList.zipIdx).flatMap fun (form, index) =>
        formRelocations context (path ++ [1, index]) form
    | .taggedAffine _ substitution .. => substitutionRelocations context (path ++ [2]) substitution
    | .constant _ | .optionalConstant .. => []

/-- Physical projections may contain empty intersections. Their endpoints
still carry the source insertion, so their exact wire values are relocated. -/
private def sourceRelocations (context : SourceContext) (path : List Nat)
    (projection : SourceProjection) : Except String (List Relocation) := do
  let .mapped ranges := projection | return []
  let mut result := []
  for (range, index) in ranges.zipIdx do
    if context.constant ≤ range.packageStart then
      unless range.count = 0 ∨ range.packageStart + range.count ≤ context.total do
        throw "projection exceeds the reference public suffix"
      result := result ++ [⟨path ++ [1, index, 0], dimension (range.packageStart - context.privateDelta) 1 1⟩]
    else if range.packageStart + range.count = context.total then
      result := result ++ [⟨path ++ [1, index, 2], dimension (range.count - context.privateDelta) 1 1⟩]
    else
      unless range.count = 0 ∨ range.packageStart + range.count ≤ Layout.Stage1.Wide.SourceOrder.privateColumns do
        throw "projection crosses an unsupported application insertion"
  return result

private def ordinaryRelocations (context : SourceContext) (kind : Kind) (path : List Nat)
    (block : Ordinary.Block) : Except String (List Relocation) := do
  let mut result ← sourceRelocations context (path ++ [3]) block.projection
  if kind = .nextPreimage then
    let .rangeList ranges := block.rows | throw "next-preimage rows are not ranges"
    for (range, index) in ranges.zipIdx do
      let rows := PerApplicationPackage.directApplicationRowCount referenceApplication
      unless rows ≤ range.start do throw "next-preimage row precedes application"
      result := result ++ [⟨path ++ [0, 1, index, 0], dimension (range.start - rows) 0 0 1⟩]
  if kind = .application then
    result := result ++ [
      ⟨path ++ [0, 1, 0, 1], dimension 0 0 0 1⟩,
      ⟨path ++ [2, 0, 1, 1], dimension 0 1⟩,
      ⟨path ++ [2, 0, 1, 2, 1], dimension 0 1⟩,
      ⟨path ++ [2, 0, 3, 0], dimension Layout.Stage1.ApplicationInputs.witnessStart 1⟩,
      ⟨path ++ [2, 0, 3, 1], dimension 0 0 1⟩,
      ⟨path ++ [2, 0, 3, 2, 1], dimension 0 0 1⟩,
      ⟨path ++ [2, 0, 3, 2, 2], dimension (ApplicationOrdinaryGeometry.witnessStart referenceApplication) 41⟩]
  return result

private def blockRelocations (context : SourceContext) (kind : Kind) (path : List Nat)
    (block : MatrixProgram.Block) : Except String (List Relocation) := do
  match block with
  | .mapped width projection inner =>
    unless width = context.referenceWidth ∧ projection = context.referenceProjection do
      throw "unexpected shared retained projection"
    let wrapper := [
      Relocation.mk (path ++ [1]) (dimension (width - context.retainedDelta) 41 41),
      ⟨path ++ [2, 1, 2, 2], dimension 0 41 41⟩,
      ⟨path ++ [2, 1, 3, 1], dimension (context.outputStart - context.retainedDelta) 41 41⟩,
      ⟨path ++ [2, 1, 4, 1], dimension (context.quotientStart - context.retainedDelta) 41 41⟩]
    match inner with
    | .ordinary ordinary => return wrapper ++ (← ordinaryRelocations context kind (path ++ [3, 1]) ordinary)
    | _ => return wrapper
  | .ordinaryTemplate ordinary _ =>
    return substitutionRelocations context (path ++ [1, 2]) ordinary.substitution
  | .poseidon poseidon =>
    return retainedRelocations context (path ++ [1, 2]) poseidon.retained ++
      inputRelocations context (path ++ [1, 3]) poseidon.input
  | .phi81Product product =>
    let .direct forms _ := product.challenge | throw "wide product uses retained scalar words"
    return (forms.toList.zipIdx).flatMap (fun (form, index) =>
        formRelocations context (path ++ [1, 2, index]) form) ++
      substitutionRelocations context (path ++ [1, 4]) product.input ++
      retainedRelocations context (path ++ [1, 5]) product.output ++
      retainedRelocations context (path ++ [1, 6]) product.group
  | _ => throw "unexpected unprojected shared matrix block"

private structure Metadata where
  children : List Lean.Json
  shared : List Lean.Json
  application : List Lean.Json

private def childMetadata (context : SourceContext) (programs : List ChildPrograms) : Except String Metadata := do
  let mut blockStart := 0
  let mut fixedRowStart := 0
  let mut afterApplication := false
  let mut result : Metadata := ⟨[], [], []⟩
  for child in programs do
    let application := child.kind == .application
    result := { result with children := Lean.Json.mkObj [
      ("id", .str (kindName child.kind)), ("opcode", jsonNat (kindOpcode child.kind)),
      ("block_start", jsonNat blockStart), ("block_count", jsonNat child.program.blocks.length),
      ("row_start", dimension fixedRowStart 0 0 (if afterApplication then 1 else 0)),
      ("row_count", if application then dimension 0 0 0 1 else dimension child.program.rowCount),
      ("replaceable", Lean.toJson application)] :: result.children }
    for (block, index) in child.program.blocks.zipIdx do
      let relocations ← blockRelocations context child.kind [index] block
      if application then
        result := { result with application :=
          (relocations.foldl (fun accumulated relocation => relocation.json 0 :: accumulated) result.application) }
      else
        result := { result with shared :=
          (relocations.foldl (fun accumulated relocation => relocation.json blockStart :: accumulated) result.shared) }
    blockStart := blockStart + child.program.blocks.length
    if application then afterApplication := true
    else fixedRowStart := fixedRowStart + child.program.rowCount
  return ⟨result.children.reverse, result.shared.reverse,
    result.application.reverse⟩

private def runJson (first count : Lean.Json) (step : Nat) : Lean.Json :=
  Lean.Json.mkObj [("first", first), ("step", jsonNat step), ("count", count)]

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

private def applicationWitnessIndex : Nat := Wide.AssignmentTransport.commonKinds.idxOf .applicationWitness
private def applicationLocalIndex : Nat := Wide.AssignmentTransport.commonKinds.idxOf .applicationLocal

private def assignmentBlock (context : SourceContext) (index : Nat)
    (block : Wide.AssignmentTransport.Values) : Lean.Json :=
  let count := if index = applicationWitnessIndex then dimension 0 1
    else if index = applicationLocalIndex then dimension 0 0 1 else dimension block.count
  let runs := if index = applicationWitnessIndex then
      [runJson (dimension Layout.Stage1.Wide.SourceOrder.privateColumns) (dimension 0 1) 1]
    else if index = applicationLocalIndex then
      [runJson (dimension Layout.Stage1.Wide.SourceOrder.privateColumns 1) (dimension 0 0 1) 1]
    else block.sources.flatMap (sourceRun context)
  Lean.Json.mkObj [("opcode", jsonNat index), ("slot_kind", jsonNat (slotOpcode block.kind)),
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
  ("source_private_start", jsonNat Layout.Stage1.Wide.SourceOrder.privateColumns),
  ("reference_constant", jsonNat context.constant), ("reference_total", jsonNat context.total),
  ("source_column_fields", strings sourceColumnFields), ("prefix_row_start", jsonNat 0),
  ("prefix_row_count", jsonNat context.prefixRows), ("application_row_start", jsonNat context.prefixRows),
  ("next_preimage_row_start", dimension context.prefixRows 0 0 1),
  ("next_preimage_row_count", jsonNat 5),
  ("prefix_witness_end", jsonNat Layout.Stage1.Wide.SourceOrder.privateColumns)]

private def recipeContract : Lean.Json := Lean.Json.mkObj [
  ("row_form", .str "r1cs_a_times_b_equals_c"),
  ("expression_tags", strings ["var", "const", "add", "mul"]),
  ("hint_tags", strings ["bit", "inverse_or_zero", "quotient_five", "remainder_five"]),
  ("witness_instruction", .str "target_equals_a_times_b"),
  ("field_encoding", .str "balanced_ternary_41"),
  ("source_support", strings ["state_input", "application_witness", "state_output", "application_local"]),
  ("native_outputs_are_hints", Lean.toJson true)]

private def contracts : List String := [
  "NightstreamFPrime.Export.Stage1.Wide.PackageAuthority.matrix_exact",
  "NightstreamFPrime.Export.Stage1.Wide.PackageAuthority.transport_emitted",
  "NightstreamFPrime.Export.Stage1.Wide.PhysicalMatrixSemantics.program_correct",
  "NightstreamFPrime.Export.Stage1.Wide.MatrixProjection.column_eq",
  "NightstreamFPrime.Export.Stage1.Wide.FixedPoint.plan_fixedPoint",
  "NightstreamFPrime.Export.Stage1.Wide.AssignmentTransportCorrectness.canonical_execute_eq_assignment",
  "NightstreamFPrime.Export.Stage1.Wide.PackageCompleteness.complete"]

private def manifestValue (context : SourceContext) (programs : List ChildPrograms)
    (metadata : Metadata) (transport : Wide.AssignmentTransport.Plan) : Except String Lean.Json := do
  let some application := programs.find? (·.kind == .application) | throw "missing application child"
  let profile := [goldilocksModulus, productionGlobalParams.b, productionGlobalParams.k,
    productionGlobalParams.bigB, ringDegree, Lifecycle.cubeVariables,
    Spec.ProductionRelation.matrixCount, Spec.ProductionRelation.meaningfulPortCount]
  return Lean.Json.mkObj [
    ("format", .str "nightstream.shared-verifier"), ("version", jsonNat 3),
    ("id", .str "shared-recursive-verifier-v1"), ("profile", Lean.toJson profile),
    ("dependencies", strings ["poseidon2-permutation-v1", "poseidon2-external-v1", "phi81-product-v1"]),
    ("parameters", strings ["witness_words", "local_words", "application_rows"]),
    ("reference", Lean.toJson (referenceCounts ())),
    ("application_local_index", jsonNat applicationLocalIndex),
    ("geometry", geometry context programs), ("ports", ← ports ()), ("recursive_public", recursivePublic),
    ("children", Lean.toJson metadata.children), ("matrix_relocations", Lean.toJson metadata.shared),
    ("application_matrix_template", valueJson (Program.format.encode application.program)),
    ("application_matrix_relocations", Lean.toJson metadata.application),
    ("assignment_blocks", Lean.toJson ((transport.blocks.zipIdx).map fun (block, index) => assignmentBlock context index block)),
    ("phi81_value_sources", Lean.toJson (transport.valueSources.flatMap (sourceRun context))),
    ("phi81_challenge_sources", Lean.toJson (transport.challengeSources.flatMap (sourceRun context))),
    ("source_relocation", sourceRelocation context), ("recipe_contract", recipeContract),
    ("required_dimension_checks", strings ["source_rows_le_domain", "source_total_le_domain", "ring_padded_logical_width_le_domain"]),
    ("phase_order", strings ["prior_state_hash", "pi_ccs_sumcheck", "pi_ccs_final", "pi_rlc", "pi_dec", "application", "output_hash"]),
    ("terminal", Lean.Json.mkObj [("running_claims", jsonNat productionShape.runningCount),
      ("fresh_claims", jsonNat productionShape.freshCount), ("all_final_rows", Lean.toJson true)]),
    ("contracts", strings contracts),
    ("proof_scope", .str "The wide owners prove the selected Lean application and its exact prepared archive. The application connector uses Lean Application.Program and the stated encoding hypotheses. This manifest does not prove Rust application semantics, relocation, or assembly.")]

def value (_ : Unit) : Except String Lean.Json := do
  let some compiled := PiRlcWideSampler.RangePlan.compile? | throw "wide range compilation failed"
  let context ← sourceContext ()
  let programs ← children compiled
  let metadata ← childMetadata context programs
  let transport ← Wide.AssignmentTransport.plan referenceApplication context.total
  manifestValue context programs metadata transport

private def progress (message : String) : IO Unit := do
  IO.println s!"wide shared verifier: {message}"
  (← IO.getStdout).flush

@[noinline] private def timed {α : Type} (phase : String)
    (action : Unit → Except String α) : IO α := do
  progress s!"start {phase}"
  let started ← IO.monoNanosNow
  let value ← match action () with
    | .ok value => pure value
    | .error error => throw (IO.userError error)
  progress s!"done {phase} elapsed_ns={(← IO.monoNanosNow) - started}"
  return value

private def prepareChildren (compiled : PiRlcWideSampler.RangePlan.Compiled) : IO (List ChildPrograms) := do
  let mut tasks : Array (Task (Except IO.Error ChildPrograms)) := #[]
  for kind in PerApplicationProductionPlan.canonicalKinds do
    tasks := tasks.push (← IO.asTask do
      let program ← timed s!"child {kindName kind}" fun _ =>
        childProgram compiled referenceApplication kind
      return ⟨kind, program⟩)
  let mut programs := []
  for task in tasks do
    match ← IO.wait task with
    | .ok child => programs := child :: programs
    | .error error => throw error
  return programs.reverse

/-- Use the runtime task pool for independent children, then collect them in
the same canonical order as `value`. Both paths use the same typed builders. -/
def prepare : IO Lean.Json := do
  let compiled ← timed "range compiler" fun _ =>
    match PiRlcWideSampler.RangePlan.compile? with
    | some compiled => .ok compiled
    | none => .error "wide range compilation failed"
  let context ← timed "source context" fun _ => sourceContext ()
  progress "start children"
  let programs ← prepareChildren compiled
  progress "done children"
  let metadata ← timed "child metadata" fun _ => childMetadata context programs
  let transport ← timed "assignment transport" fun _ =>
    Wide.AssignmentTransport.plan referenceApplication context.total
  timed "manifest assembly" fun _ => manifestValue context programs metadata transport

def write (path : System.FilePath) : IO Unit := do
  let manifest ← prepare
  IO.FS.writeFile path (manifest.compress ++ "\n")

end NightstreamFPrime.Export.SharedVerifier
