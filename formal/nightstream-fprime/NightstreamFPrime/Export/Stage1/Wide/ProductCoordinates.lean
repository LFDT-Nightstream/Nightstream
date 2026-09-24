import NightstreamFPrime.Export.Stage1.PiRLCProductRingSchedule

/-! Preserve the extension-field cell/lane order when moving old product
coordinates into the new ring-major retained allocation. -/

namespace NightstreamFPrime.Export.Stage1.Wide.ProductCoordinates

open NightstreamFPrime.Spec
open PiRLCProductRingSchedule

def ringLane : Fin PiRLCProductSchedule.invocationCount ≃ Fin invocationCount × Fin ringDegree where
  toFun := fun index => (ringInvocation index, (PiRLCProductSchedule.descriptor index).lane)
  invFun := fun index => laneInvocation index.1 index.2
  left_inv := laneInvocation_ringInvocation
  right_inv := by
    rintro ⟨ring, lane⟩
    simp only [ringInvocation_laneInvocation, descriptor_laneInvocation, Descriptor.withLane]

def slot (index : Fin PiRLCProductSchedule.invocationCount) : Fin 52326 :=
  Fin.encodeProd (ringLane index)

theorem slot_injective : Function.Injective slot := by
  intro left right same
  apply ringLane.injective
  calc
    ringLane left = Fin.decodeProd (slot left) := (Fin.decodeProd_encodeProd _).symm
    _ = Fin.decodeProd (slot right) := congrArg Fin.decodeProd same
    _ = ringLane right := Fin.decodeProd_encodeProd _

def coordinate (index : Fin 2145366) : Fin 2145366 :=
  let pair : Fin PiRLCProductSchedule.invocationCount × Fin 41 := Fin.decodeProd index
  Fin.encodeProd (slot pair.1, pair.2)

theorem coordinate_injective : Function.Injective coordinate := by
  intro left right same
  have pairs := congrArg (Fin.decodeProd (m := 52326) (n := 41)) same
  simp only [coordinate, Fin.decodeProd_encodeProd] at pairs
  have slots := congrArg Prod.fst pairs
  have digits := congrArg Prod.snd pairs
  have originals := slot_injective slots
  have decoded : Fin.decodeProd (m := PiRLCProductSchedule.invocationCount) (n := 41) left =
      Fin.decodeProd right := Prod.ext originals digits
  have encoded := congrArg Fin.encodeProd decoded
  simpa only [Fin.encodeProd_decodeProd] using encoded

def inverseSlot (index : Fin 52326) : Fin PiRLCProductSchedule.invocationCount :=
  ringLane.symm (Fin.decodeProd index)

def inverseCoordinate (index : Fin 2145366) : Fin 2145366 :=
  let pair : Fin 52326 × Fin 41 := Fin.decodeProd index
  Fin.encodeProd (inverseSlot pair.1, pair.2)

theorem inverseSlot_slot (index : Fin PiRLCProductSchedule.invocationCount) :
    inverseSlot (slot index) = index := by
  simp only [inverseSlot, slot, Fin.decodeProd_encodeProd, Equiv.symm_apply_apply]

theorem slot_inverseSlot (index : Fin 52326) : slot (inverseSlot index) = index := by
  simp only [inverseSlot, slot, Equiv.apply_symm_apply, Fin.encodeProd_decodeProd]

theorem inverseCoordinate_coordinate (index : Fin 2145366) :
    inverseCoordinate (coordinate index) = index := by
  simp only [inverseCoordinate, coordinate, Fin.decodeProd_encodeProd, inverseSlot_slot]
  exact Fin.encodeProd_decodeProd (m := 52326) (n := 41) index

theorem coordinate_inverseCoordinate (index : Fin 2145366) :
    coordinate (inverseCoordinate index) = index := by
  simp only [inverseCoordinate, coordinate, Fin.decodeProd_encodeProd, slot_inverseSlot]
  exact Fin.encodeProd_decodeProd (m := 52326) (n := 41) index

theorem coordinate_lane (ring : Fin invocationCount) (lane : Fin ringDegree) (digit : Fin 41) :
    coordinate (Fin.encodeProd (laneInvocation ring lane, digit)) =
      Fin.encodeProd (Fin.encodeProd (ring, lane), digit) := by
  simp only [coordinate, Fin.decodeProd_encodeProd, slot, ringLane,
    Equiv.coe_fn_mk, ringInvocation_laneInvocation, descriptor_laneInvocation, Descriptor.withLane]
  rfl

end NightstreamFPrime.Export.Stage1.Wide.ProductCoordinates
