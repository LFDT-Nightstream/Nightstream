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

end Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcChallengeTranscriptLayoutSchema
