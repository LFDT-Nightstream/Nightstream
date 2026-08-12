import Nightstream.Implementation.NebulaV2.Memory.Operation.PrefixRows
import Nightstream.Protocol.NebulaV2.Memory
import Nightstream.Protocol.NebulaV2.Ports

/-!
Contract: exact 3-by-21 application-port refinement for one V2 checked step.

Assurance tier: implementation-to-protocol bridge.

Owns the normalized application rows decoded from the same 63 physical slot
columns as the operation relation. It proves exact physical-position and
ordered-list equality, including active ports after inactive holes. It also
proves that the RS and WS product records are exactly the read and write
tuples of those application accesses.

Does not own WASM state-transition rows, row-kind selection, cross-step trace
chaining, the generated artifact, or terminal completion.

Emits constraints: no. It gives application-port meaning to existing
operation-source rows.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.NebulaV2.ApplicationPortRefinement

open Nightstream.Implementation.NebulaV2.MemoryProductClaimBridge
open Nightstream.Implementation.NebulaV2.MemoryProductSemanticBridge
open Nightstream.Implementation.NebulaV2.MemoryProductUpdateRows
open Nightstream.Implementation.NebulaV2.OperationPrefixRows
open Nightstream.Protocol.NebulaV2
open Nightstream.Protocol.NebulaV2.ConcreteLaneGeometry
open Nightstream.Protocol.NebulaV2.Fingerprint
open Nightstream.Protocol.NebulaV2.Memory
open Nightstream.Protocol.NebulaV2.OperationSlot
open Nightstream.Protocol.NebulaV2.Ports

/-- The exact typed slot value derived by the operation-source rows. -/
def slotValue
    {layout : OperationPrefixRows.Layout} {assignment : Nat → Nat}
    {claim : MemoryClaimCodec.Claim}
    (_sound : OperationPrefixRows.Sound layout assignment claim)
    (slot : Fin operationSlots) : OperationSlot.Value :=
  OperationSlotRows.decoded (layout.operationSlot slot) assignment
    (assignment (layout.countColumn (beforeIndex slot)))
    (assignment (layout.countColumn (afterIndex slot)))

theorem slotValue_validAt
    {layout : OperationPrefixRows.Layout} {assignment : Nat → Nat}
    {claim : MemoryClaimCodec.Claim}
    (sound : OperationPrefixRows.Sound layout assignment claim)
    (slot : Fin operationSlots) :
    OperationSlot.ValidAt (slotValue sound slot) claim.timestampIn := by
  exact sound.slotValid slot

/-- Canonical inactive padding becomes `none`; every active slot becomes its
one exact semantic access. -/
def slotPort
    {layout : OperationPrefixRows.Layout} {assignment : Nat → Nat}
    {claim : MemoryClaimCodec.Claim}
    (sound : OperationPrefixRows.Sound layout assignment claim)
    (slot : Fin operationSlots) : Option Access :=
  if (slotValue sound slot).pad = 0 then
    some (slotValue sound slot).access
  else none

@[simp]
theorem slotPort_of_active
    {layout : OperationPrefixRows.Layout} {assignment : Nat → Nat}
    {claim : MemoryClaimCodec.Claim}
    (sound : OperationPrefixRows.Sound layout assignment claim)
    (slot : Fin operationSlots)
    (active : (slotValue sound slot).pad = 0) :
    slotPort sound slot = some (slotValue sound slot).access := by
  simp [slotPort, active]

@[simp]
theorem slotPort_of_padded
    {layout : OperationPrefixRows.Layout} {assignment : Nat → Nat}
    {claim : MemoryClaimCodec.Claim}
    (sound : OperationPrefixRows.Sound layout assignment claim)
    (slot : Fin operationSlots)
    (padded : (slotValue sound slot).pad = 1) :
    slotPort sound slot = none := by
  simp [slotPort, padded]

/-- Row kinds come from the application-control relation. Memory-port values
do not depend on them and are decoded only from the shared slot columns. -/
def checkedStep
    {layout : OperationPrefixRows.Layout} {assignment : Nat → Nat}
    {claim : MemoryClaimCodec.Claim}
    (sound : OperationPrefixRows.Sound layout assignment claim)
    (kinds : ApplicationRowIndex → NormalizedRowKind) : CheckedStep :=
  { rows := fun row =>
      { kind := kinds row
        memoryPorts := fun port => slotPort sound (route row port) } }

@[simp]
theorem checkedStep_row_port
    {layout : OperationPrefixRows.Layout} {assignment : Nat → Nat}
    {claim : MemoryClaimCodec.Claim}
    (sound : OperationPrefixRows.Sound layout assignment claim)
    (kinds : ApplicationRowIndex → NormalizedRowKind)
    (row : ApplicationRowIndex) (port : RowPortIndex) :
    ((checkedStep sound kinds).rows row).memoryPorts port =
      slotPort sound (route row port) := rfl

@[simp]
theorem checkedStep_physicalPort
    {layout : OperationPrefixRows.Layout} {assignment : Nat → Nat}
    {claim : MemoryClaimCodec.Claim}
    (sound : OperationPrefixRows.Sound layout assignment claim)
    (kinds : ApplicationRowIndex → NormalizedRowKind)
    (position : Fin slotsPerStep) :
    (checkedStep sound kinds).physicalPorts position =
      slotPort sound position := by
  simp [CheckedStep.physicalPorts, checkedStep, route_unroute]

def accesses
    {layout : OperationPrefixRows.Layout} {assignment : Nat → Nat}
    {claim : MemoryClaimCodec.Claim}
    (sound : OperationPrefixRows.Sound layout assignment claim) :
    List Access :=
  compactPayloads (slotPort sound)

theorem checkedStep_accesses
    {layout : OperationPrefixRows.Layout} {assignment : Nat → Nat}
    {claim : MemoryClaimCodec.Claim}
    (sound : OperationPrefixRows.Sound layout assignment claim)
    (kinds : ApplicationRowIndex → NormalizedRowKind) :
    (checkedStep sound kinds).accesses = accesses sound := by
  unfold CheckedStep.accesses accesses
  apply congrArg compactPayloads
  funext position
  exact checkedStep_physicalPort sound kinds position

def rows
    {layout : OperationPrefixRows.Layout} {assignment : Nat → Nat}
    {claim : MemoryClaimCodec.Claim}
    (sound : OperationPrefixRows.Sound layout assignment claim)
    (kinds : ApplicationRowIndex → NormalizedRowKind) :
    List NormalizedRow :=
  (checkedStep sound kinds).rowList

@[simp]
theorem rows_length
    {layout : OperationPrefixRows.Layout} {assignment : Nat → Nat}
    {claim : MemoryClaimCodec.Claim}
    (sound : OperationPrefixRows.Sound layout assignment claim)
    (kinds : ApplicationRowIndex → NormalizedRowKind) :
    (rows sound kinds).length = applicationRowsPerStep := by
  simp [rows]

/-- The application row order and the operation-lane order are identical. -/
theorem rows_flatMap_accesses
    {layout : OperationPrefixRows.Layout} {assignment : Nat → Nat}
    {claim : MemoryClaimCodec.Claim}
    (sound : OperationPrefixRows.Sound layout assignment claim)
    (kinds : ApplicationRowIndex → NormalizedRowKind) :
    (rows sound kinds).flatMap NormalizedRow.accesses = accesses sound := by
  rw [rows, CheckedStep.rowList_flatMap_accesses,
    checkedStep_accesses sound kinds]

/-- One integer indicator per fixed physical slot. -/
def activeIndicators
    {layout : OperationPrefixRows.Layout} {assignment : Nat → Nat}
    {claim : MemoryClaimCodec.Claim}
    (sound : OperationPrefixRows.Sound layout assignment claim) : List Nat :=
  List.ofFn fun slot : Fin operationSlots =>
    1 - (slotValue sound slot).pad

private theorem accesses_length_eq_indicator_sum
    {layout : OperationPrefixRows.Layout} {assignment : Nat → Nat}
    {claim : MemoryClaimCodec.Claim}
    (sound : OperationPrefixRows.Sound layout assignment claim) :
    (accesses sound).length = (activeIndicators sound).sum := by
  have each : ∀ slots : List (Fin operationSlots),
      (slots.filterMap (slotPort sound)).length =
        (slots.map fun slot => 1 - (slotValue sound slot).pad).sum := by
    intro slots
    induction slots with
    | nil => rfl
    | cons slot rest inductionHypothesis =>
        rcases (sound.slotValid slot).padBinary with active | padded
        · have active' : (slotValue sound slot).pad = 0 := by
            simpa [slotValue] using active
          rw [List.filterMap_cons, slotPort_of_active sound slot active']
          simp [inductionHypothesis, active']
          omega
        · have padded' : (slotValue sound slot).pad = 1 := by
            simpa [slotValue] using padded
          rw [List.filterMap_cons, slotPort_of_padded sound slot padded']
          simp [inductionHypothesis, padded']
  unfold accesses compactPayloads activeIndicators
  change
    ((List.finRange operationSlots).filterMap (slotPort sound)).length =
      (List.ofFn fun slot : Fin operationSlots =>
        1 - (slotValue sound slot).pad).sum
  rw [List.ofFn_eq_map]
  exact each (List.finRange operationSlots)

private theorem activeIndicator_sum_eq_finalCount
    {layout : OperationPrefixRows.Layout} {assignment : Nat → Nat}
    {claim : MemoryClaimCodec.Claim}
    (sound : OperationPrefixRows.Sound layout assignment claim) :
    (activeIndicators sound).sum = claim.activeAccessCount := by
  have countAt : ∀ n (bound : n ≤ operationSlots),
      assignment (layout.countColumn
        (⟨n, by omega⟩ : CountIndex)) =
        ((activeIndicators sound).take n).sum := by
    intro n
    induction n with
    | zero =>
        intro _bound
        change assignment (layout.countColumn firstCount) = 0
        exact sound.countZero
    | succ n inductionHypothesis =>
        intro bound
        have slotBound : n < operationSlots := by omega
        let slot : Fin operationSlots := ⟨n, slotBound⟩
        have prior := inductionHypothesis (by omega)
        have step := sound.countStep slot
        calc
          assignment (layout.countColumn
              (⟨n + 1, by omega⟩ : CountIndex)) =
              assignment (layout.countColumn
                (⟨n, by omega⟩ : CountIndex)) +
                (1 - (slotValue sound slot).pad) := by
            simpa [slot, slotValue, beforeIndex, afterIndex] using step
          _ = ((activeIndicators sound).take n).sum +
                (1 - (slotValue sound slot).pad) := by rw [prior]
          _ = ((activeIndicators sound).take (n + 1)).sum := by
            symm
            rw [List.sum_take_succ]
            · simp [activeIndicators, slot]
            · simpa [activeIndicators] using slotBound
  have atEnd := countAt operationSlots (Nat.le_refl _)
  have lengthExact : (activeIndicators sound).length = operationSlots := by
    simp [activeIndicators]
  have takeExact :
      (activeIndicators sound).take operationSlots =
        activeIndicators sound := by
    rw [← lengthExact, List.take_length]
  rw [takeExact] at atEnd
  change assignment (layout.countColumn lastCount) =
    (activeIndicators sound).sum at atEnd
  exact atEnd.symm.trans sound.finalCount

/-- The claim active count is not trusted: the 63 pad bits and prefix rows
derive it exactly. -/
theorem accesses_length_eq_claimActiveCount
    {layout : OperationPrefixRows.Layout} {assignment : Nat → Nat}
    {claim : MemoryClaimCodec.Claim}
    (sound : OperationPrefixRows.Sound layout assignment claim) :
    (accesses sound).length = claim.activeAccessCount :=
  (accesses_length_eq_indicator_sum sound).trans
    (activeIndicator_sum_eq_finalCount sound)

def physicalPortList
    {layout : OperationPrefixRows.Layout} {assignment : Nat → Nat}
    {claim : MemoryClaimCodec.Claim}
    (sound : OperationPrefixRows.Sound layout assignment claim) :
    List (Option Access) :=
  List.ofFn (slotPort sound)

theorem accesses_eq_filterMap_physicalPortList
    {layout : OperationPrefixRows.Layout} {assignment : Nat → Nat}
    {claim : MemoryClaimCodec.Claim}
    (sound : OperationPrefixRows.Sound layout assignment claim) :
    accesses sound = (physicalPortList sound).filterMap id := by
  unfold accesses physicalPortList
  exact compactPayloads_eq_filterMap_ofFn (slotPort sound)

private theorem ordered_prefix
    {layout : OperationPrefixRows.Layout} {assignment : Nat → Nat}
    {claim : MemoryClaimCodec.Claim}
    (sound : OperationPrefixRows.Sound layout assignment claim) :
    ∀ n (bound : n ≤ operationSlots),
      Ordered claim.timestampIn
        (((physicalPortList sound).take n).filterMap id)
        (claim.timestampIn + assignment (layout.countColumn
          (⟨n, by omega⟩ : CountIndex))) := by
  intro n
  induction n with
  | zero =>
      intro _bound
      have zero := sound.countZero
      change Ordered claim.timestampIn []
        (claim.timestampIn + assignment (layout.countColumn firstCount))
      rw [zero, Nat.add_zero]
      exact .nil _
  | succ n inductionHypothesis =>
      intro bound
      have slotBound : n < operationSlots := by omega
      let slot : Fin operationSlots := ⟨n, slotBound⟩
      have prior := inductionHypothesis (by omega)
      have listBound : n < (physicalPortList sound).length := by
        simpa [physicalPortList] using slotBound
      have takeNext :=
        List.take_succ_eq_append_getElem (l := physicalPortList sound)
          listBound
      have currentPort :
          (physicalPortList sound)[n] = slotPort sound slot := by
        simp [physicalPortList, slot]
      have prefixExact :
          ((physicalPortList sound).take (n + 1)).filterMap id =
            ((physicalPortList sound).take n).filterMap id ++
              [slotPort sound slot].filterMap id := by
        rw [takeNext, List.filterMap_append, currentPort]
      rw [prefixExact]
      have step := sound.countStep slot
      rcases (sound.slotValid slot).padBinary with active | padded
      · have active' : (slotValue sound slot).pad = 0 := by
          simpa [slotValue] using active
        rw [slotPort_of_active sound slot active']
        simp only [List.filterMap_cons, List.filterMap_nil, id_eq]
        have accessValid :
            (slotValue sound slot).access.ValidAt
              (claim.timestampIn +
                assignment (layout.countColumn (beforeIndex slot))) := by
          simpa [slotValue] using
            (sound.slotValid slot).access_validAt active
        have activeAssignment :
            assignment (layout.operationSlot slot).padColumn = 0 := by
          simpa [slotValue, OperationSlotRows.decoded] using active
        have singleton :
            Ordered
              (claim.timestampIn +
                assignment (layout.countColumn (beforeIndex slot)))
              [(slotValue sound slot).access]
              (claim.timestampIn +
                assignment (layout.countColumn (afterIndex slot))) := by
          have outputExact :
              claim.timestampIn +
                  assignment (layout.countColumn (afterIndex slot)) =
                (claim.timestampIn +
                    assignment (layout.countColumn (beforeIndex slot))) + 1 := by
            rw [step]
            rw [activeAssignment]
            omega
          rw [outputExact]
          exact .cons accessValid (.nil _)
        simpa [slot, beforeIndex, afterIndex] using
          prior.append singleton
      · have padded' : (slotValue sound slot).pad = 1 := by
          simpa [slotValue] using padded
        rw [slotPort_of_padded sound slot padded']
        simp only [List.filterMap_cons, List.filterMap_nil, id_eq,
          List.append_nil]
        have countUnchanged :
            assignment (layout.countColumn (afterIndex slot)) =
              assignment (layout.countColumn (beforeIndex slot)) := by
          have paddedAssignment :
              assignment (layout.operationSlot slot).padColumn = 1 := by
            simpa [slotValue, OperationSlotRows.decoded] using padded
          rw [step]
          rw [paddedAssignment]
          omega
        have outputExact :
            claim.timestampIn + assignment (layout.countColumn
                (⟨n + 1, by omega⟩ : CountIndex)) =
              claim.timestampIn + assignment (layout.countColumn
                (⟨n, by omega⟩ : CountIndex)) := by
          simpa [slot, beforeIndex, afterIndex] using
            congrArg (claim.timestampIn + ·) countUnchanged
        rw [outputExact]
        exact prior

/-- The exact physical-port list satisfies the global integer timestamp
schedule. The output equation is supplied by the separately checked memory
transition rows. -/
theorem ordered
    {layout : OperationPrefixRows.Layout} {assignment : Nat → Nat}
    {claim : MemoryClaimCodec.Claim}
    (sound : OperationPrefixRows.Sound layout assignment claim)
    (timestampOutExact :
      claim.timestampOut = claim.timestampIn + claim.activeAccessCount) :
    Ordered claim.timestampIn (accesses sound) claim.timestampOut := by
  have complete := ordered_prefix sound operationSlots (Nat.le_refl _)
  have portLength : (physicalPortList sound).length = operationSlots := by
    simp [physicalPortList]
  have takeExact :
      (physicalPortList sound).take operationSlots =
        physicalPortList sound := by
    rw [← portLength, List.take_length]
  rw [takeExact] at complete
  have finalCount :
      assignment (layout.countColumn
          (⟨operationSlots, by omega⟩ : CountIndex)) =
        claim.activeAccessCount := by
    simpa [lastCount] using sound.finalCount
  rw [finalCount] at complete
  rw [accesses_eq_filterMap_physicalPortList]
  rw [timestampOutExact]
  exact complete

def tupleOfRole : OperationRole → Access → MemTuple
  | .reads => Access.read
  | .writes => Access.write

/-- One product-record option is the matching tuple of the same physical
application port. This statement retains the physical position. -/
theorem recordOption_eq_portTuple
    {layout : OperationPrefixRows.Layout} {assignment : Nat → Nat}
    {claim : MemoryClaimCodec.Claim}
    (sound : OperationPrefixRows.Sound layout assignment claim)
    (role : OperationRole) (slot : Fin operationSlots) :
    (sound.records role slot).map Subtype.val =
      (slotPort sound slot).map (tupleOfRole role) := by
  have valid := sound.slotValid slot
  rcases valid.padBinary with active | padded
  · cases role <;>
      simp [OperationPrefixRows.Sound.records,
        OperationSlotProductBridge.representedRecord, slotPort, slotValue,
        tupleOfRole, active, OperationSlot.ValidAt.readBounded,
        OperationSlot.ValidAt.writeBounded, OperationSlot.Value.access]
  · cases role <;>
      simp [OperationPrefixRows.Sound.records,
        OperationSlotProductBridge.representedRecord, slotPort, slotValue,
        tupleOfRole, padded]

/-- Hole removal on the product sources and hole removal on the application
ports produce the same ordered tuple list. -/
theorem recordTupleList_eq_accessTupleList
    {layout : OperationPrefixRows.Layout} {assignment : Nat → Nat}
    {claim : MemoryClaimCodec.Claim}
    (sound : OperationPrefixRows.Sound layout assignment claim)
    (role : OperationRole) :
    (activeRecords
      (operationRecords fun slot => sound.records role slot)).map Subtype.val =
      (accesses sound).map (tupleOfRole role) := by
  unfold activeRecords operationRecords accesses compactPayloads
  rw [List.map_filterMap, List.map_filterMap]
  rw [List.ofFn_eq_map, List.filterMap_map]
  apply List.filterMap_congr
  intro slot _member
  simpa only [Function.comp_apply, id_eq] using
    recordOption_eq_portTuple sound role slot

/-- Exact RS multiset equality derived from the shared physical ports. -/
theorem readRecordMultiset_eq
    {layout : OperationPrefixRows.Layout} {assignment : Nat → Nat}
    {claim : MemoryClaimCodec.Claim}
    (sound : OperationPrefixRows.Sound layout assignment claim) :
    activeRecordMultiset
        (operationRecords fun slot => sound.records .reads slot) =
      (Memory.readTuples (accesses sound) : Multiset MemTuple) := by
  exact congrArg (fun values : List MemTuple => (values : Multiset MemTuple))
    (recordTupleList_eq_accessTupleList sound .reads)

/-- Exact WS multiset equality derived from the shared physical ports. -/
theorem writeRecordMultiset_eq
    {layout : OperationPrefixRows.Layout} {assignment : Nat → Nat}
    {claim : MemoryClaimCodec.Claim}
    (sound : OperationPrefixRows.Sound layout assignment claim) :
    activeRecordMultiset
        (operationRecords fun slot => sound.records .writes slot) =
      (Memory.writeTuples (accesses sound) : Multiset MemTuple) := by
  exact congrArg (fun values : List MemTuple => (values : Multiset MemTuple))
    (recordTupleList_eq_accessTupleList sound .writes)

end Nightstream.Implementation.NebulaV2.ApplicationPortRefinement
