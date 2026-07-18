import Nightstream.Implementation.R1CS.Artifacts.FPrimeRecursive.PiRlcChallenge.Generated.TranscriptLayoutData

/-!
Stable facade for the active fixed-recursive PiRLC transcript layout.

Owns: the exact physical source-row partition, constant pins, compact
Poseidon2-call locations, emission order, state-column continuity, boundary
state columns/cursors, 240 field-output aliases, and four external bind-input
columns exported by the Rust trace drift gate.

Does not own: row satisfaction, message or cursor semantics, Poseidon2
correctness, transcript replay, Fiat-Shamir authority, sampler correctness, or
permission to remove rows.

Assurance tier: artifact-checked physical layout. Stage labels are extraction
provenance only; digests remain non-authoritative until replayed by a verifier.

| Surface | Fixed profile | Structural boundary |
|---|---:|---|
| source partition | 76 ranges / 47,091 rows | 291 pins plus 78 compact 600-row calls |
| ordered emissions | 369 | every pin and call occurs exactly once |
| state continuity | 77 adjacent call edges | exact same-lane column aliases only |
| field outputs | 15 x 4 x 4 = 240 | compact-call output to canonical-u64 input aliases |
| external bind inputs | 4 columns | physical locations only |
-/

namespace Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcChallenge.TranscriptLayout

open Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcChallengeTranscriptLayoutSchema

namespace Generated

open Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcChallengeTranscriptLayoutData

abbrev artifact : TranscriptLayout :=
  FPrimeRecursivePiRlcChallengeTranscriptLayoutData.layout

end Generated

abbrev artifact : TranscriptLayout := Generated.artifact

abbrev ownedRanges : List OwnedRange := artifact.ownedRanges
abbrev constantPins : List ConstantPin := artifact.constantPins
abbrev calls : List CompactCall := artifact.calls
abbrev emissionOrder : List EmissionRef := artifact.emissionOrder
abbrev stateContinuity : List StateContinuity := artifact.stateContinuity
abbrev fieldOutputAliases : List FieldOutputAlias := artifact.fieldOutputAliases

def groupCount : Nat := 15
def digestBlockCount : Nat := 4
def lanesPerBlock : Nat := 4

def constantPinAt (index : Fin constantPins.length) : ConstantPin :=
  constantPins.get index

def compactCallAt (index : Fin calls.length) : CompactCall :=
  calls.get index

def emissionAt (index : Fin emissionOrder.length) : EmissionRef :=
  emissionOrder.get index

def fieldOutputAliasAt
    (group : Fin groupCount) (block : Fin digestBlockCount)
    (lane : Fin lanesPerBlock) : FieldOutputAlias :=
  fieldOutputAliases.getD
    (group.val * (digestBlockCount * lanesPerBlock) +
      block.val * lanesPerBlock + lane.val)
    default

abbrev initialStateColumns : List Nat := artifact.entryBoundary.stateColumns
abbrev initialCursor : Nat := artifact.entryBoundary.cursor
abbrev postBindStateColumns : List Nat := artifact.postBindBoundary.stateColumns
abbrev postBindCursor : Nat := artifact.postBindBoundary.cursor
abbrev finalStateColumns : List Nat := artifact.finalBoundary.stateColumns
abbrev finalCursor : Nat := artifact.finalBoundary.cursor
abbrev piCcsOutputDigestInputColumns : List Nat := artifact.bindInputColumns

private def emissionSpan : EmissionRef → Nat × Nat
  | .pin index =>
      let pin := constantPins.getD index default
      (pin.row, pin.row + 1)
  | .call index =>
      let call := calls.getD index default
      (call.rowStart, call.rowEnd)

private def spansCover (cursor finish : Nat) : List (Nat × Nat) → Bool
  | [] => decide (cursor = finish)
  | (start, stop) :: rest =>
      decide (start = cursor) && decide (start < stop) &&
        spansCover stop finish rest

private def rangeCovered (owned : OwnedRange) : Bool :=
  let scheduled :=
    (emissionOrder.drop owned.emissionStart).take
      (owned.emissionEnd - owned.emissionStart)
  decide (owned.emissionStart < owned.emissionEnd) &&
    decide (owned.emissionEnd ≤ emissionOrder.length) &&
    spansCover owned.rowStart owned.rowEnd (scheduled.map emissionSpan)

private def rangesCoverEmissionsFrom : Nat → List OwnedRange → Bool
  | cursor, [] => decide (cursor = emissionOrder.length)
  | cursor, owned :: rest =>
      decide (owned.emissionStart = cursor) && rangeCovered owned &&
        rangesCoverEmissionsFrom owned.emissionEnd rest

private def sourceRangesOrdered : List OwnedRange → Bool
  | [] => true
  | [_] => true
  | first :: second :: rest =>
      decide (first.rowEnd ≤ second.rowStart) &&
        sourceRangesOrdered (second :: rest)

private def ownedRowTotal : Nat :=
  ownedRanges.foldl
    (fun total owned => total + (owned.rowEnd - owned.rowStart)) 0

private def emissionRefValid : EmissionRef → Bool
  | .pin index => decide (index < constantPins.length)
  | .call index => decide (index < calls.length)

private def emissionSpansStrictlyOrdered : List (Nat × Nat) → Bool
  | [] => true
  | [_] => true
  | first :: second :: rest =>
      decide (first.2 ≤ second.1) &&
        emissionSpansStrictlyOrdered (second :: rest)

private def emissionMultiplicityValid : Bool :=
  emissionOrder.all emissionRefValid &&
    emissionSpansStrictlyOrdered (emissionOrder.map emissionSpan) &&
    decide (emissionOrder.length = constantPins.length + calls.length)

private def pinValid (pin : ConstantPin) : Bool :=
  decide (pin.row < artifact.sourceRows) &&
    decide (pin.column < artifact.sourceColumns) &&
    decide (pin.value < 18446744069414584321)

private def callValidAt (index : Nat) : Bool :=
  let call := calls.getD index default
  decide (call.traceIndex = 6537 + index) &&
    decide (call.rowStart < call.rowEnd) &&
    decide (call.rowEnd - call.rowStart = 600) &&
    decide (call.rowEnd ≤ artifact.sourceRows) &&
    decide (call.inputColumns.length = 8) &&
    call.inputColumns.all (fun column => decide (column < artifact.sourceColumns)) &&
    decide (call.firstAllocatedColumn + 600 ≤ artifact.sourceColumns)

private def matchingLanes (fromCall toCall : CompactCall) : List Nat :=
  (List.range 8).filter fun lane =>
    decide (fromCall.outputColumn lane = toCall.inputColumns.getD lane 0)

private def continuityValidAt (index : Nat) : Bool :=
  let continuity := stateContinuity.getD index default
  let fromCall := calls.getD index default
  let toCall := calls.getD (index + 1) default
  decide (continuity.fromCall = index) &&
    decide (continuity.toCall = index + 1) &&
    decide (continuity.lanes = matchingLanes fromCall toCall)

private def matchingBoundaryLanes
    (boundary : Boundary) (call : CompactCall) : List Nat :=
  (List.range 8).filter fun lane =>
    decide (boundary.stateColumns.getD lane 0 =
      call.inputColumns.getD lane 0)

private def boundaryValid : Bool :=
  let firstCall := calls.getD 0 default
  let firstRhoCall := calls.getD artifact.firstRhoCallIndex default
  let lastCall := calls.getD (calls.length - 1) default
  decide (artifact.entryProducerTraceIndex = 6519) &&
    decide (initialStateColumns.length = 8) &&
    decide (initialCursor = 0) &&
    decide (postBindStateColumns.length = 8) &&
    decide (postBindCursor = 2) &&
    decide (finalStateColumns.length = 8) &&
    decide (finalCursor = 0) &&
    decide (artifact.entryToFirstCallLanes =
      matchingBoundaryLanes artifact.entryBoundary firstCall) &&
    decide (artifact.entryToFirstCallLanes = [4, 5, 6, 7]) &&
    decide (artifact.postBindToFirstRhoCallLanes =
      matchingBoundaryLanes artifact.postBindBoundary firstRhoCall) &&
    decide (artifact.postBindToFirstRhoCallLanes = [0, 1, 4, 5, 6, 7]) &&
    decide (finalStateColumns = lastCall.outputColumns)

private def fieldOutputAliasValidAt (index : Nat) : Bool :=
  let alias := fieldOutputAliases.getD index default
  let call := calls.getD alias.callIndex default
  decide (alias.ordinal = index) &&
    decide (alias.groupIndex = index / 16) &&
    decide (alias.blockIndex = (index / 4) % 4) &&
    decide (alias.laneIndex = index % 4) &&
    decide (alias.groupIndex < groupCount) &&
    decide (alias.blockIndex < digestBlockCount) &&
    decide (alias.laneIndex < lanesPerBlock) &&
    decide (alias.callIndex = 4 + 5 * alias.groupIndex + alias.blockIndex) &&
    decide (alias.outputLane = alias.laneIndex) &&
    decide (alias.fieldColumn = call.outputColumn alias.outputLane) &&
    decide (alias.canonicalRowEnd - alias.canonicalRowStart = 69) &&
    decide (call.rowEnd ≤ alias.canonicalRowStart) &&
    decide (alias.canonicalRowEnd ≤ artifact.sourceRows)

def StructureValid : Prop :=
  artifact.sourceRows = 9503595 ∧
    artifact.sourceColumns = 9316338 ∧
    artifact.ownedRowCount = 47091 ∧
    ownedRanges.length = 76 ∧
    constantPins.length = 291 ∧
    calls.length = 78 ∧
    emissionOrder.length = 369 ∧
    stateContinuity.length = 77 ∧
    fieldOutputAliases.length = 240 ∧
    artifact.bindCallIndices = [0, 1] ∧
    artifact.firstRhoCallIndex = 2 ∧
    piCcsOutputDigestInputColumns.length = 4 ∧
    ownedRowTotal = artifact.ownedRowCount ∧
    rangesCoverEmissionsFrom 0 ownedRanges = true ∧
    sourceRangesOrdered ownedRanges = true ∧
    emissionMultiplicityValid = true ∧
    constantPins.all pinValid = true ∧
    (List.range calls.length).all callValidAt = true ∧
    (List.range stateContinuity.length).all continuityValidAt = true ∧
    boundaryValid = true ∧
    (List.range fieldOutputAliases.length).all fieldOutputAliasValidAt = true

instance : Decidable StructureValid := by
  unfold StructureValid
  infer_instance

theorem structure_check : StructureValid := by
  set_option maxRecDepth 100000 in
    decide

theorem exact_pin_count : constantPins.length = 291 :=
  structure_check.2.2.2.2.1

theorem exact_call_count : calls.length = 78 :=
  structure_check.2.2.2.2.2.1

theorem exact_emission_count : emissionOrder.length = 369 := by
  set_option maxRecDepth 100000 in
    decide

theorem exact_field_output_alias_count : fieldOutputAliases.length = 240 :=
  structure_check.2.2.2.2.2.2.2.2.1

theorem exact_external_bind_input_count :
    piCcsOutputDigestInputColumns.length = 4 :=
  structure_check.2.2.2.2.2.2.2.2.2.2.2.1

theorem ordered_emissions_cover_owned_ranges :
    rangesCoverEmissionsFrom 0 ownedRanges = true := by
  set_option maxRecDepth 100000 in
    decide

theorem emissions_are_unique_and_in_bounds :
    emissionMultiplicityValid = true := by
  set_option maxRecDepth 100000 in
    decide

theorem pins_are_canonical_and_in_bounds :
    constantPins.all pinValid = true := by
  set_option maxRecDepth 100000 in
    decide

theorem calls_have_exact_compact_abi :
    (List.range calls.length).all callValidAt = true := by
  set_option maxRecDepth 100000 in
    decide

theorem adjacent_call_state_columns_match :
    (List.range stateContinuity.length).all continuityValidAt = true := by
  set_option maxRecDepth 100000 in
    decide

theorem boundary_columns_and_cursors_match : boundaryValid = true := by
  set_option maxRecDepth 100000 in
    decide

theorem field_output_aliases_match_calls :
    (List.range fieldOutputAliases.length).all fieldOutputAliasValidAt = true := by
  set_option maxRecDepth 100000 in
    decide

theorem initial_state_columns_eq :
    initialStateColumns =
      [4198711, 4198712, 4198713, 4198714,
        4198715, 4198716, 4198717, 4198718] := by
  rfl

theorem initial_cursor_eq : initialCursor = 0 := by
  rfl

theorem post_bind_state_columns_eq :
    postBindStateColumns =
      [5037860, 5037861, 5039070, 5039071,
        5039072, 5039073, 5039074, 5039075] := by
  rfl

theorem post_bind_cursor_eq : postBindCursor = 2 := by
  rfl

theorem final_state_columns_eq :
    finalStateColumns =
      [5156360, 5156361, 5156362, 5156363,
        5156364, 5156365, 5156366, 5156367] := by
  rfl

theorem final_cursor_eq : finalCursor = 0 := by
  rfl

theorem pi_ccs_output_digest_input_columns_eq :
    piCcsOutputDigestInputColumns = [5037858, 5037859, 5037860, 5037861] := by
  rfl

theorem constant_pin_profile :
    ∀ index : Fin constantPins.length,
      let pin := constantPinAt index
      pin.row < artifact.sourceRows ∧
        pin.column < artifact.sourceColumns ∧
        pin.value < 18446744069414584321 := by
  set_option maxRecDepth 100000 in
    decide

theorem compact_call_profile :
    ∀ index : Fin calls.length,
      let call := compactCallAt index
      call.traceIndex = 6537 + index.val ∧
        call.rowStart < call.rowEnd ∧
        call.rowEnd - call.rowStart = 600 ∧
        call.inputColumns.length = 8 ∧
        call.outputColumns.length = 8 ∧
        call.rowEnd ≤ artifact.sourceRows ∧
        call.firstAllocatedColumn + 600 ≤ artifact.sourceColumns := by
  set_option maxRecDepth 100000 in
    decide

theorem field_output_alias_at_formula :
    ∀ group : Fin groupCount,
      ∀ block : Fin digestBlockCount,
        ∀ lane : Fin lanesPerBlock,
          let alias := fieldOutputAliasAt group block lane
          let call := calls.getD alias.callIndex default
          alias.ordinal = group.val * 16 + block.val * 4 + lane.val ∧
            alias.groupIndex = group.val ∧
            alias.blockIndex = block.val ∧
            alias.laneIndex = lane.val ∧
            alias.callIndex = 4 + 5 * group.val + block.val ∧
            alias.outputLane = lane.val ∧
            alias.fieldColumn = call.outputColumn lane.val ∧
            alias.canonicalRowEnd - alias.canonicalRowStart = 69 := by
  set_option maxRecDepth 100000 in
    decide

end Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcChallenge.TranscriptLayout
