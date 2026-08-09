import Nightstream.Implementation.R1CS.Artifacts.FPrimeRecursive.PiRlcChallenge.Generated.TranscriptLayoutData

/-!
Stable facade for the active fixed-recursive PiRLC transcript layout.

Owns: the exact physical source-row partition, constant pins, compact
Poseidon2-call locations, emission order, state-column continuity, boundary
state columns/cursors, 480 field-output aliases, and four external bind-input
columns exported by the Rust trace drift gate.

Does not own: row satisfaction, message or cursor semantics, Poseidon2
correctness, transcript replay, Fiat-Shamir authority, sampler correctness, or
permission to remove rows.

Assurance tier: artifact-checked physical layout. Each proof certificate stays
inside one protocol prelude or sampler-group phase. Stage labels are extraction
provenance only; digests remain non-authoritative until replayed by a verifier.

| Surface | Fixed profile | Structural boundary |
|---|---:|---|
| source partition | 136 ranges / 154,972 rows | 772 pins plus 257 compact 600-row calls |
| ordered emissions | 1,029 | every pin and call occurs exactly once |
| state continuity | 256 adjacent call edges | exact same-lane column aliases only |
| field outputs | 15 x 8 x 4 = 480 | compact-call output to canonical-u64 input aliases |
| external bind inputs | 4 columns | physical locations only |
-/

namespace Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcChallenge.TranscriptLayout

open Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcChallengeTranscriptLayoutSchema

namespace Generated

open Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcChallengeTranscriptLayoutData

abbrev artifact : TranscriptLayout := layout
abbrev phases : List Phase := phaseSequence

abbrev preludePhase : Phase := Prelude.phase
abbrev group00Phase : Phase := Group00.phase
abbrev group01Phase : Phase := Group01.phase
abbrev group02Phase : Phase := Group02.phase
abbrev group03Phase : Phase := Group03.phase
abbrev group04Phase : Phase := Group04.phase
abbrev group05Phase : Phase := Group05.phase
abbrev group06Phase : Phase := Group06.phase
abbrev group07Phase : Phase := Group07.phase
abbrev group08Phase : Phase := Group08.phase
abbrev group09Phase : Phase := Group09.phase
abbrev group10Phase : Phase := Group10.phase
abbrev group11Phase : Phase := Group11.phase
abbrev group12Phase : Phase := Group12.phase
abbrev group13Phase : Phase := Group13.phase
abbrev group14Phase : Phase := Group14.phase

/-- Kernel-checked evidence at every protocol-owned phase boundary. -/
structure Certificates : Prop where
  prelude : Phase.ValidAfter artifact.sourceRows artifact.sourceColumns none
    preludePhase
  group00 : Phase.ValidAfter artifact.sourceRows artifact.sourceColumns
    (some preludePhase) group00Phase
  group01 : Phase.ValidAfter artifact.sourceRows artifact.sourceColumns
    (some group00Phase) group01Phase
  group02 : Phase.ValidAfter artifact.sourceRows artifact.sourceColumns
    (some group01Phase) group02Phase
  group03 : Phase.ValidAfter artifact.sourceRows artifact.sourceColumns
    (some group02Phase) group03Phase
  group04 : Phase.ValidAfter artifact.sourceRows artifact.sourceColumns
    (some group03Phase) group04Phase
  group05 : Phase.ValidAfter artifact.sourceRows artifact.sourceColumns
    (some group04Phase) group05Phase
  group06 : Phase.ValidAfter artifact.sourceRows artifact.sourceColumns
    (some group05Phase) group06Phase
  group07 : Phase.ValidAfter artifact.sourceRows artifact.sourceColumns
    (some group06Phase) group07Phase
  group08 : Phase.ValidAfter artifact.sourceRows artifact.sourceColumns
    (some group07Phase) group08Phase
  group09 : Phase.ValidAfter artifact.sourceRows artifact.sourceColumns
    (some group08Phase) group09Phase
  group10 : Phase.ValidAfter artifact.sourceRows artifact.sourceColumns
    (some group09Phase) group10Phase
  group11 : Phase.ValidAfter artifact.sourceRows artifact.sourceColumns
    (some group10Phase) group11Phase
  group12 : Phase.ValidAfter artifact.sourceRows artifact.sourceColumns
    (some group11Phase) group12Phase
  group13 : Phase.ValidAfter artifact.sourceRows artifact.sourceColumns
    (some group12Phase) group13Phase
  group14 : Phase.ValidAfter artifact.sourceRows artifact.sourceColumns
    (some group13Phase) group14Phase

theorem certificates : Certificates where
  prelude := Prelude.valid
  group00 := Group00.valid
  group01 := Group01.valid
  group02 := Group02.valid
  group03 := Group03.valid
  group04 := Group04.valid
  group05 := Group05.valid
  group06 := Group06.valid
  group07 := Group07.valid
  group08 := Group08.valid
  group09 := Group09.valid
  group10 := Group10.valid
  group11 := Group11.valid
  group12 := Group12.valid
  group13 := Group13.valid
  group14 := Group14.valid

theorem pinCount : artifact.constantPins.length = 772 :=
  constantPins_length

theorem callCount : artifact.calls.length = 257 :=
  calls_length

theorem emissionCount : artifact.emissionOrder.length = 1029 :=
  emissionOrder_length

theorem ownedRangeCount : artifact.ownedRanges.length = 136 :=
  ownedRanges_length

theorem continuityCount : artifact.stateContinuity.length = 256 :=
  stateContinuity_length

theorem aliasCount : artifact.fieldOutputAliases.length = 480 :=
  fieldOutputAliases_length

theorem pinIndices :
    pinEmissionIndices artifact.emissionOrder = List.range 772 :=
  pinEmissionIndices_eq

theorem callIndices :
    callEmissionIndices artifact.emissionOrder = List.range 257 :=
  callEmissionIndices_eq

theorem pinValuesCanonical :
    artifact.constantPins.all
      (fun pin => decide (pin.value < 18446744069414584321)) = true :=
  constantPinValuesCanonical

end Generated

abbrev artifact : TranscriptLayout := Generated.artifact

abbrev ownedRanges : List OwnedRange := artifact.ownedRanges
abbrev constantPins : List ConstantPin := artifact.constantPins
abbrev calls : List CompactCall := artifact.calls
abbrev emissionOrder : List EmissionRef := artifact.emissionOrder
abbrev stateContinuity : List StateContinuity := artifact.stateContinuity
abbrev fieldOutputAliases : List FieldOutputAlias := artifact.fieldOutputAliases

def groupCount : Nat := 15
def digestBlockCount : Nat := 8
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

/-- One property stated uniformly for every generated protocol phase. -/
structure PhaseChecks (property : Phase → Prop) : Prop where
  prelude : property Generated.preludePhase
  group00 : property Generated.group00Phase
  group01 : property Generated.group01Phase
  group02 : property Generated.group02Phase
  group03 : property Generated.group03Phase
  group04 : property Generated.group04Phase
  group05 : property Generated.group05Phase
  group06 : property Generated.group06Phase
  group07 : property Generated.group07Phase
  group08 : property Generated.group08Phase
  group09 : property Generated.group09Phase
  group10 : property Generated.group10Phase
  group11 : property Generated.group11Phase
  group12 : property Generated.group12Phase
  group13 : property Generated.group13Phase
  group14 : property Generated.group14Phase

def Generated.Certificates.map
    {property : Phase → Prop} (certificate : Generated.Certificates)
    (ofValid : ∀ {previous phase},
      Phase.ValidAfter artifact.sourceRows artifact.sourceColumns previous phase →
        property phase) : PhaseChecks property where
  prelude := ofValid certificate.prelude
  group00 := ofValid certificate.group00
  group01 := ofValid certificate.group01
  group02 := ofValid certificate.group02
  group03 := ofValid certificate.group03
  group04 := ofValid certificate.group04
  group05 := ofValid certificate.group05
  group06 := ofValid certificate.group06
  group07 := ofValid certificate.group07
  group08 := ofValid certificate.group08
  group09 := ofValid certificate.group09
  group10 := ofValid certificate.group10
  group11 := ofValid certificate.group11
  group12 := ofValid certificate.group12
  group13 := ofValid certificate.group13
  group14 := ofValid certificate.group14

private def matchingBoundaryLanes
    (boundary : Boundary) (call : CompactCall) : List Nat :=
  (List.range 8).filter fun lane =>
    decide (boundary.stateColumns.getD lane 0 =
      call.inputColumns.getD lane 0)

private def boundaryValid : Bool :=
  let firstCall := Generated.preludePhase.calls.getD 0 default
  let firstRhoCall := Generated.group00Phase.calls.getD 0 default
  let lastCalls := Generated.group14Phase.calls
  let lastCall := lastCalls.getD (lastCalls.length - 1) default
  decide (artifact.entryProducerTraceIndex = 156) &&
    decide (initialStateColumns.length = 8) &&
    decide (initialCursor = 0) &&
    decide (postBindStateColumns.length = 8) &&
    decide (postBindCursor = 1) &&
    decide (finalStateColumns.length = 8) &&
    decide (finalCursor = 0) &&
    decide (artifact.entryToFirstCallLanes =
      matchingBoundaryLanes artifact.entryBoundary firstCall) &&
    decide (artifact.entryToFirstCallLanes = [4, 5, 6, 7]) &&
    decide (artifact.postBindToFirstRhoCallLanes =
      matchingBoundaryLanes artifact.postBindBoundary firstRhoCall) &&
    decide (artifact.postBindToFirstRhoCallLanes = [0, 4, 5, 6, 7]) &&
    decide (finalStateColumns = lastCall.outputColumns)

/-- Complete physical-layout contract. Large data is proved only through the
kernel-sized phase certificates. -/
structure StructureValid : Prop where
  sourceRows : artifact.sourceRows = 7169252
  sourceColumns : artifact.sourceColumns = 7100181
  ownedRows : artifact.ownedRowCount = 154972
  ownedRangeCount : ownedRanges.length = 136
  pinCount : constantPins.length = 772
  callCount : calls.length = 257
  emissionCount : emissionOrder.length = 1029
  continuityCount : stateContinuity.length = 256
  aliasCount : fieldOutputAliases.length = 480
  bindCallIndices : artifact.bindCallIndices = [0, 1, 2]
  firstRhoCall : artifact.firstRhoCallIndex = 3
  bindInputCount : piCcsOutputDigestInputColumns.length = 4
  finalOwnedRowCount :
    Generated.group14Phase.ownedRowEnd = artifact.ownedRowCount
  pinIndicesExact :
    pinEmissionIndices emissionOrder = List.range constantPins.length
  callIndicesExact :
    callEmissionIndices emissionOrder = List.range calls.length
  phasesValid : Generated.Certificates
  boundaryValid : boundaryValid = true

theorem exact_pin_count : constantPins.length = 772 := Generated.pinCount
theorem exact_call_count : calls.length = 257 := Generated.callCount
theorem exact_emission_count : emissionOrder.length = 1029 :=
  Generated.emissionCount
theorem exact_field_output_alias_count : fieldOutputAliases.length = 480 :=
  Generated.aliasCount

theorem exact_external_bind_input_count :
    piCcsOutputDigestInputColumns.length = 4 := by
  rfl

private theorem pin_indices_exact :
    pinEmissionIndices emissionOrder = List.range constantPins.length := by
  rw [exact_pin_count]
  exact Generated.pinIndices

private theorem call_indices_exact :
    callEmissionIndices emissionOrder = List.range calls.length := by
  rw [exact_call_count]
  exact Generated.callIndices

theorem structure_check : StructureValid where
  sourceRows := rfl
  sourceColumns := rfl
  ownedRows := rfl
  ownedRangeCount := Generated.ownedRangeCount
  pinCount := exact_pin_count
  callCount := exact_call_count
  emissionCount := exact_emission_count
  continuityCount := Generated.continuityCount
  aliasCount := exact_field_output_alias_count
  bindCallIndices := rfl
  firstRhoCall := rfl
  bindInputCount := exact_external_bind_input_count
  finalOwnedRowCount := by decide
  pinIndicesExact := pin_indices_exact
  callIndicesExact := call_indices_exact
  phasesValid := Generated.certificates
  boundaryValid := by decide

theorem ordered_emissions_cover_owned_ranges :
    PhaseChecks (fun phase => phase.ownedRangesCovered = true) :=
  Generated.certificates.map (fun valid => valid.rangesCoverEmissions)

theorem emissions_are_unique_and_in_bounds :
    PhaseChecks (fun phase =>
      phase.emissionOrder.length =
          phase.constantPins.length + phase.calls.length ∧
        pinEmissionIndices phase.emissionOrder =
          List.range' phase.pinStart phase.constantPins.length ∧
        callEmissionIndices phase.emissionOrder =
          List.range' phase.callStart phase.calls.length) :=
  Generated.certificates.map fun valid =>
    ⟨valid.emissionCount, valid.pinIndicesExact, valid.callIndicesExact⟩

theorem pins_are_canonical_and_in_bounds :
    PhaseChecks (fun phase =>
      phase.constantPinsValid artifact.sourceRows artifact.sourceColumns = true) :=
  Generated.certificates.map (fun valid => valid.pinsValid)

theorem calls_have_exact_compact_abi :
    PhaseChecks (fun phase =>
      phase.compactCallsValid artifact.sourceRows artifact.sourceColumns = true) :=
  Generated.certificates.map (fun valid => valid.callsValid)

theorem adjacent_call_state_columns_match : Generated.Certificates :=
  Generated.certificates

theorem boundary_columns_and_cursors_match : boundaryValid = true :=
  structure_check.boundaryValid

theorem field_output_aliases_match_calls :
    PhaseChecks (fun phase =>
      phase.fieldOutputAliasesMatch artifact.sourceRows = true) :=
  Generated.certificates.map (fun valid => valid.aliasesValid)

private theorem pin_mem_indices_iff (index : Nat) (refs : List EmissionRef) :
    .pin index ∈ refs ↔ index ∈ pinEmissionIndices refs := by
  induction refs with
  | nil => simp [pinEmissionIndices]
  | cons head tail ih =>
      cases head <;> simp [pinEmissionIndices, ih]

private theorem call_mem_indices_iff (index : Nat) (refs : List EmissionRef) :
    .call index ∈ refs ↔ index ∈ callEmissionIndices refs := by
  induction refs with
  | nil => simp [callEmissionIndices]
  | cons head tail ih =>
      cases head <;> simp [callEmissionIndices, ih]

theorem pin_mem_emissionOrder_iff (index : Nat) :
    .pin index ∈ emissionOrder ↔ index < constantPins.length := by
  rw [pin_mem_indices_iff, pin_indices_exact, List.mem_range]

theorem call_mem_emissionOrder_iff (index : Nat) :
    .call index ∈ emissionOrder ↔ index < calls.length := by
  rw [call_mem_indices_iff, call_indices_exact, List.mem_range]

theorem constant_pin_value_canonical
    (pin : ConstantPin) (member : pin ∈ constantPins) :
    pin.value < 18446744069414584321 := by
  exact of_decide_eq_true
    ((List.all_eq_true.mp Generated.pinValuesCanonical) pin member)

theorem initial_state_columns_eq :
    initialStateColumns =
      [3015069, 3015070, 3015071, 3015072,
        3015073, 3015074, 3015075, 3015076] := by
  rfl

theorem initial_cursor_eq : initialCursor = 0 := by
  rfl

theorem post_bind_state_columns_eq :
    postBindStateColumns =
      [3854221, 3856570, 3856571, 3856572,
        3856573, 3856574, 3856575, 3856576] := by
  rfl

theorem post_bind_cursor_eq : postBindCursor = 1 := by
  rfl

theorem final_state_columns_eq :
    finalStateColumns =
      [4097163, 4097164, 4097165, 4097166,
        4097167, 4097168, 4097169, 4097170] := by
  rfl

theorem final_cursor_eq : finalCursor = 0 := by
  rfl

theorem pi_ccs_output_digest_input_columns_eq :
    piCcsOutputDigestInputColumns = [3854218, 3854219, 3854220, 3854221] := by
  rfl

theorem constant_pin_profile :
    PhaseChecks (fun phase =>
      phase.constantPinsValid artifact.sourceRows artifact.sourceColumns = true) :=
  pins_are_canonical_and_in_bounds

theorem compact_call_profile :
    PhaseChecks (fun phase =>
      phase.compactCallsValid artifact.sourceRows artifact.sourceColumns = true) :=
  calls_have_exact_compact_abi

theorem field_output_alias_at_formula :
    PhaseChecks (fun phase =>
      phase.fieldOutputAliasesMatch artifact.sourceRows = true) :=
  field_output_aliases_match_calls

end Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcChallenge.TranscriptLayout
