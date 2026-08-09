/-!
Physical schema for the active fixed-recursive PiRLC transcript artifact.

Owns locations and ordering only. It assigns no protocol meaning to pins,
cursor values, stage ordinals, bind inputs, or digest outputs.
-/

namespace Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcChallengeTranscriptLayoutSchema

structure OwnedRange where
  checkpointIndex : Nat
  rowStart : Nat
  rowEnd : Nat
  emissionStart : Nat
  emissionEnd : Nat
deriving DecidableEq, Repr, Inhabited

structure ConstantPin where
  row : Nat
  column : Nat
  value : Nat
deriving DecidableEq, Repr, Inhabited

structure CompactCall where
  traceIndex : Nat
  rowStart : Nat
  rowEnd : Nat
  inputColumns : List Nat
  firstAllocatedColumn : Nat
deriving DecidableEq, Repr, Inhabited

def CompactCall.outputColumn (call : CompactCall) (lane : Nat) : Nat :=
  call.firstAllocatedColumn + 592 + lane

def CompactCall.outputColumns (call : CompactCall) : List Nat :=
  (List.range 8).map call.outputColumn

inductive EmissionRef where
  | pin (index : Nat)
  | call (index : Nat)
deriving DecidableEq, Repr, Inhabited

structure Boundary where
  stateColumns : List Nat
  cursor : Nat
deriving DecidableEq, Repr, Inhabited

structure StateContinuity where
  fromCall : Nat
  toCall : Nat
  lanes : List Nat
deriving DecidableEq, Repr, Inhabited

structure FieldOutputAlias where
  ordinal : Nat
  groupIndex : Nat
  blockIndex : Nat
  laneIndex : Nat
  callIndex : Nat
  outputLane : Nat
  fieldColumn : Nat
  canonicalRowStart : Nat
  canonicalRowEnd : Nat
deriving DecidableEq, Repr, Inhabited

structure TranscriptLayout where
  sourceRows : Nat
  sourceColumns : Nat
  ownedRowCount : Nat
  ownedRanges : List OwnedRange
  constantPins : List ConstantPin
  calls : List CompactCall
  emissionOrder : List EmissionRef
  entryProducerTraceIndex : Nat
  entryBoundary : Boundary
  postBindBoundary : Boundary
  finalBoundary : Boundary
  entryToFirstCallLanes : List Nat
  postBindToFirstRhoCallLanes : List Nat
  stateContinuity : List StateContinuity
  fieldOutputAliases : List FieldOutputAlias
  bindCallIndices : List Nat
  firstRhoCallIndex : Nat
  bindInputColumns : List Nat
deriving DecidableEq, Repr, Inhabited

/-- One protocol-owned transcript phase. Indices remain absolute so the phase
can be checked independently and then recomposed without renumbering data. -/
structure Phase where
  pinStart : Nat
  callStart : Nat
  emissionStart : Nat
  continuityStart : Nat
  aliasStart : Nat
  ownedRowStart : Nat
  ownedRanges : List OwnedRange
  constantPins : List ConstantPin
  calls : List CompactCall
  emissionOrder : List EmissionRef
  stateContinuity : List StateContinuity
  fieldOutputAliases : List FieldOutputAlias
deriving DecidableEq, Repr, Inhabited

def Phase.pinEnd (phase : Phase) : Nat :=
  phase.pinStart + phase.constantPins.length

def Phase.callEnd (phase : Phase) : Nat :=
  phase.callStart + phase.calls.length

def Phase.emissionEnd (phase : Phase) : Nat :=
  phase.emissionStart + phase.emissionOrder.length

def Phase.continuityEnd (phase : Phase) : Nat :=
  phase.continuityStart + phase.stateContinuity.length

def Phase.aliasEnd (phase : Phase) : Nat :=
  phase.aliasStart + phase.fieldOutputAliases.length

def Phase.ownedRowCount (phase : Phase) : Nat :=
  phase.ownedRanges.foldl
    (fun total owned => total + (owned.rowEnd - owned.rowStart)) 0

def Phase.ownedRowEnd (phase : Phase) : Nat :=
  phase.ownedRowStart + phase.ownedRowCount

def pinEmissionIndices : List EmissionRef → List Nat
  | [] => []
  | .pin index :: rest => index :: pinEmissionIndices rest
  | .call _ :: rest => pinEmissionIndices rest

def callEmissionIndices : List EmissionRef → List Nat
  | [] => []
  | .pin _ :: rest => callEmissionIndices rest
  | .call index :: rest => index :: callEmissionIndices rest

@[simp] theorem pinEmissionIndices_append
    (left right : List EmissionRef) :
    pinEmissionIndices (left ++ right) =
      pinEmissionIndices left ++ pinEmissionIndices right := by
  induction left with
  | nil => rfl
  | cons head tail ih =>
      cases head <;> simp [pinEmissionIndices, ih]

@[simp] theorem callEmissionIndices_append
    (left right : List EmissionRef) :
    callEmissionIndices (left ++ right) =
      callEmissionIndices left ++ callEmissionIndices right := by
  induction left with
  | nil => rfl
  | cons head tail ih =>
      cases head <;> simp [callEmissionIndices, ih]

private def Phase.pinAt (phase : Phase) (index : Nat) : ConstantPin :=
  phase.constantPins.getD (index - phase.pinStart) default

private def Phase.callAt (phase : Phase) (index : Nat) : CompactCall :=
  phase.calls.getD (index - phase.callStart) default

private def Phase.emissionSpan
    (phase : Phase) : EmissionRef → Nat × Nat
  | .pin index =>
      let pin := phase.pinAt index
      (pin.row, pin.row + 1)
  | .call index =>
      let call := phase.callAt index
      (call.rowStart, call.rowEnd)

private def spansCover (cursor finish : Nat) : List (Nat × Nat) → Bool
  | [] => decide (cursor = finish)
  | (start, stop) :: rest =>
      decide (start = cursor) && decide (start < stop) &&
        spansCover stop finish rest

private def Phase.rangeCovered
    (phase : Phase) (owned : OwnedRange) : Bool :=
  let localStart := owned.emissionStart - phase.emissionStart
  let scheduled :=
    (phase.emissionOrder.drop localStart).take
      (owned.emissionEnd - owned.emissionStart)
  decide (phase.emissionStart ≤ owned.emissionStart) &&
    decide (owned.emissionStart < owned.emissionEnd) &&
    decide (owned.emissionEnd ≤ phase.emissionEnd) &&
    spansCover owned.rowStart owned.rowEnd
      (scheduled.map phase.emissionSpan)

private def Phase.rangesCoverFrom
    (phase : Phase) : Nat → List OwnedRange → Bool
  | cursor, [] => decide (cursor = phase.emissionEnd)
  | cursor, owned :: rest =>
      decide (owned.emissionStart = cursor) && phase.rangeCovered owned &&
        phase.rangesCoverFrom owned.emissionEnd rest

private def sourceRangesOrdered : List OwnedRange → Bool
  | [] => true
  | [_] => true
  | first :: second :: rest =>
      decide (first.rowEnd ≤ second.rowStart) &&
        sourceRangesOrdered (second :: rest)

private def emissionSpansStrictlyOrdered : List (Nat × Nat) → Bool
  | [] => true
  | [_] => true
  | first :: second :: rest =>
      decide (first.2 ≤ second.1) &&
        emissionSpansStrictlyOrdered (second :: rest)

private def pinValid
    (sourceRows sourceColumns : Nat) (pin : ConstantPin) : Bool :=
  decide (pin.row < sourceRows) &&
    decide (pin.column < sourceColumns) &&
    decide (pin.value < 18446744069414584321)

private def callValid
    (sourceRows sourceColumns index : Nat) (call : CompactCall) : Bool :=
  decide (call.traceIndex = 174 + index) &&
    decide (call.rowStart < call.rowEnd) &&
    decide (call.rowEnd - call.rowStart = 600) &&
    decide (call.rowEnd ≤ sourceRows) &&
    decide (call.inputColumns.length = 8) &&
    call.inputColumns.all (fun column => decide (column < sourceColumns)) &&
    decide (call.firstAllocatedColumn + 600 ≤ sourceColumns)

private def callsValidFrom
    (sourceRows sourceColumns : Nat) : Nat → List CompactCall → Bool
  | _, [] => true
  | index, call :: rest =>
      callValid sourceRows sourceColumns index call &&
        callsValidFrom sourceRows sourceColumns (index + 1) rest

private def matchingLanes
    (fromCall toCall : CompactCall) : List Nat :=
  (List.range 8).filter fun lane =>
    decide (fromCall.outputColumn lane =
      toCall.inputColumns.getD lane 0)

private def continuityValidFrom :
    Nat → List CompactCall → List StateContinuity → Bool
  | _, [], [] => true
  | _, [_], [] => true
  | index, fromCall :: toCall :: calls, continuity :: rest =>
      decide (continuity.fromCall = index) &&
        decide (continuity.toCall = index + 1) &&
        decide (continuity.lanes = matchingLanes fromCall toCall) &&
        continuityValidFrom (index + 1) (toCall :: calls) rest
  | _, _, _ => false

private def fieldOutputAliasValid
    (sourceRows : Nat) (phase : Phase) (index : Nat)
    (alias : FieldOutputAlias) : Bool :=
  let call := phase.callAt alias.callIndex
  decide (alias.ordinal = index) &&
    decide (alias.groupIndex = index / 32) &&
    decide (alias.blockIndex = (index / 4) % 8) &&
    decide (alias.laneIndex = index % 4) &&
    decide (alias.groupIndex < 15) &&
    decide (alias.blockIndex < 8) &&
    decide (alias.laneIndex < 4) &&
    decide (phase.callStart ≤ alias.callIndex) &&
    decide (alias.callIndex < phase.callEnd) &&
    decide (alias.callIndex = 4 + 17 * alias.groupIndex +
      2 * alias.blockIndex) &&
    decide (alias.outputLane = alias.laneIndex) &&
    decide (alias.fieldColumn = call.outputColumn alias.outputLane) &&
    decide (alias.canonicalRowEnd - alias.canonicalRowStart = 69) &&
    decide (call.rowEnd ≤ alias.canonicalRowStart) &&
    decide (alias.canonicalRowEnd ≤ sourceRows)

private def fieldOutputAliasesValidFrom
    (sourceRows : Nat) (phase : Phase) : Nat →
      List FieldOutputAlias → Bool
  | _, [] => true
  | index, alias :: rest =>
      fieldOutputAliasValid sourceRows phase index alias &&
        fieldOutputAliasesValidFrom sourceRows phase (index + 1) rest

private def Phase.startsAfter
    (phase : Phase) (previous : Option Phase) : Bool :=
  decide (phase.pinStart = (previous.map Phase.pinEnd).getD 0) &&
    decide (phase.callStart = (previous.map Phase.callEnd).getD 0) &&
    decide (phase.emissionStart =
      (previous.map Phase.emissionEnd).getD 0) &&
    decide (phase.continuityStart =
      (previous.map Phase.continuityEnd).getD 0) &&
    decide (phase.aliasStart = (previous.map Phase.aliasEnd).getD 0) &&
    decide (phase.ownedRowStart =
      (previous.map Phase.ownedRowEnd).getD 0)

private def Phase.rowsFollow
    (phase : Phase) (previous : Option Phase) : Bool :=
  match previous.bind (fun prior => prior.ownedRanges.getLast?) with
  | none => true
  | some priorLast =>
      match phase.ownedRanges.head? with
      | none => false
      | some first => decide (priorLast.rowEnd ≤ first.rowStart)

private def Phase.continuityValidAfter
    (phase : Phase) (previous : Option Phase) : Bool :=
  match previous.bind (fun prior => prior.calls.getLast?) with
  | none => continuityValidFrom phase.callStart phase.calls
      phase.stateContinuity
  | some previousCall =>
      decide (0 < phase.callStart) &&
        continuityValidFrom (phase.callStart - 1)
          (previousCall :: phase.calls) phase.stateContinuity

def Phase.ownedRangesCovered (phase : Phase) : Bool :=
  phase.rangesCoverFrom phase.emissionStart phase.ownedRanges

def Phase.constantPinsValid
    (sourceRows sourceColumns : Nat) (phase : Phase) : Bool :=
  phase.constantPins.all (pinValid sourceRows sourceColumns)

theorem Phase.pinValueCanonical
    {sourceRows sourceColumns : Nat} {phase : Phase}
    (valid : phase.constantPinsValid sourceRows sourceColumns = true)
    {pin : ConstantPin} (member : pin ∈ phase.constantPins) :
    pin.value < 18446744069414584321 := by
  have checked := (List.all_eq_true.mp valid) pin member
  simp only [pinValid, Bool.and_eq_true, decide_eq_true_eq] at checked
  exact checked.2

def Phase.compactCallsValid
    (sourceRows sourceColumns : Nat) (phase : Phase) : Bool :=
  callsValidFrom sourceRows sourceColumns phase.callStart phase.calls

def Phase.fieldOutputAliasesMatch
    (sourceRows : Nat) (phase : Phase) : Bool :=
  fieldOutputAliasesValidFrom sourceRows phase phase.aliasStart
    phase.fieldOutputAliases

/-- Exact local physical checks for one phase and its protocol predecessor.
Each generated proof stays within one prelude or sampler-group boundary. -/
structure Phase.ValidAfter
    (sourceRows sourceColumns : Nat) (previous : Option Phase)
    (phase : Phase) : Prop where
  startsMatch : phase.startsAfter previous = true
  ownedRangesNonempty : phase.ownedRanges ≠ []
  callsNonempty : phase.calls ≠ []
  rowsFollowPrevious : phase.rowsFollow previous = true
  rangesCoverEmissions :
    phase.ownedRangesCovered = true
  rangesOrdered : sourceRangesOrdered phase.ownedRanges = true
  emissionCount :
    phase.emissionOrder.length =
      phase.constantPins.length + phase.calls.length
  pinIndicesExact :
    pinEmissionIndices phase.emissionOrder =
      List.range' phase.pinStart phase.constantPins.length
  callIndicesExact :
    callEmissionIndices phase.emissionOrder =
      List.range' phase.callStart phase.calls.length
  emissionSpansOrdered :
    emissionSpansStrictlyOrdered
      (phase.emissionOrder.map phase.emissionSpan) = true
  pinsValid :
    phase.constantPinsValid sourceRows sourceColumns = true
  callsValid :
    phase.compactCallsValid sourceRows sourceColumns = true
  continuityValid : phase.continuityValidAfter previous = true
  aliasesValid :
    phase.fieldOutputAliasesMatch sourceRows = true

end Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcChallengeTranscriptLayoutSchema
