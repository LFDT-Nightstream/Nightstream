import Nightstream.Chip8.Stage2.WitnessMemoryBinding
import SuperNeo.Primitives.PolynomialBridge

/-!
Owns the concrete CHIP-8 Twist instantiation formulas for the register and RAM
time tables. This file fixes the intended Stage-2 concrete address/cycle
surfaces and the exact bit-point identities they satisfy; it does not connect
generic authenticated claim witnesses to those concrete surfaces.
-/

namespace Nightstream.Chip8.TwistConcreteBinding

open Nightstream.Chip8
open Nightstream.Chip8.WitnessMemoryBinding
open TwistShout
open SuperNeo

abbrev F := SuperNeo.Fq
abbrev InitialState := WitnessMemoryBinding.InitialState
abbrev RegAddress := WitnessMemoryBinding.RegAddress
abbrev RamAddress := WitnessMemoryBinding.RamAddress

abbrev RegisterTimeTable (t : Nat) := TwistShout.TimeTable (K := F) 1 5 t
abbrev RamTimeTable (t : Nat) := TwistShout.TimeTable (K := F) 1 13 t
abbrev RegisterAddressColumns (t : Nat) := TwistShout.AddressColumns (K := F) 1 5 t
abbrev RamAddressColumns (t : Nat) := TwistShout.AddressColumns (K := F) 1 13 t
abbrev RegisterCycleValues (t : Nat) := TwistShout.CycleCube t → F
abbrev RamCycleValues (t : Nat) := TwistShout.CycleCube t → F

theorem registerReadCheckAtBitCycle_of_relation
  {t : Nat}
  {val : RegisterTimeTable t}
  {addr : TwistShout.CycleCube t → RegAddress}
  {rv : RegisterCycleValues t}
  {ra : RegisterAddressColumns t}
  (hRel : TwistShout.ReadWriteMemoryRelation (K := F) val addr rv)
  (hValid : TwistShout.ValidAddressColumns (K := F) ra addr)
  (j : TwistShout.CycleCube t) :
  rv j = TwistShout.rwReadCheckExpression (K := F) ra val
    (TwistShout.bitVec (K := F) j) := by
  exact hRel.readCheckAtBitCycle hValid j

theorem registerWriteCheckAtBitPoint_of_incrementRelation
  {t : Nat}
  {val inc : RegisterTimeTable t}
  {wa : RegisterAddressColumns t}
  {wv : RegisterCycleValues t}
  (hRel : TwistShout.IncrementRelation (K := F) val wa wv inc)
  (a : RegAddress)
  (j : TwistShout.CycleCube t) :
  inc a j =
    TwistShout.writeCheckExpression (K := F)
      (TwistShout.bitAddress (K := F) a)
      (TwistShout.bitVec (K := F) j)
      wa wv val := by
  exact hRel.writeCheckAtBitPoint a j

theorem registerShiftedValEvaluationExpression_at_bitPoint
  {t : Nat}
  (init : InitialState)
  (inc : RegisterTimeTable t)
  (a : RegAddress)
  (j : TwistShout.CycleCube t) :
  RegisterShiftedValEvaluationExpression init inc
      (TwistShout.bitAddress (K := F) a)
      (TwistShout.bitVec (K := F) j) =
    RegisterShiftedTimeTable init inc a j := by
  calc
    RegisterShiftedValEvaluationExpression init inc
        (TwistShout.bitAddress (K := F) a)
        (TwistShout.bitVec (K := F) j)
      =
        TwistShout.timeTableMLE (K := F) (RegisterShiftedTimeTable init inc)
          (TwistShout.bitAddress (K := F) a)
          (TwistShout.bitVec (K := F) j) := by
            symm
            exact registerShiftedValEvaluationExpression_eq_timeTableMLE
              init inc (TwistShout.bitAddress (K := F) a)
              (TwistShout.bitVec (K := F) j)
    _ = RegisterShiftedVirtualValue init inc a
          (TwistShout.bitVec (K := F) j) := by
            exact registerTimeTableMLE_shifted_at_bitAddress init inc a
              (TwistShout.bitVec (K := F) j)
    _ = RegisterShiftedTimeTable init inc a j := by
            exact registerShiftedVirtualValue_at_bitCycle init inc a j

theorem registerShiftedReadCheckAtBitCycle_of_relation
  {t : Nat}
  {init : InitialState}
  {addr : TwistShout.CycleCube t → RegAddress}
  {rv : RegisterCycleValues t}
  {ra : RegisterAddressColumns t}
  {inc : RegisterTimeTable t}
  (hRel :
    TwistShout.ReadWriteMemoryRelation (K := F)
      (RegisterShiftedTimeTable init inc) addr rv)
  (hValid : TwistShout.ValidAddressColumns (K := F) ra addr)
  (j : TwistShout.CycleCube t) :
  rv j = TwistShout.rwReadCheckExpression (K := F) ra
    (RegisterShiftedTimeTable init inc)
    (TwistShout.bitVec (K := F) j) := by
  exact registerReadCheckAtBitCycle_of_relation hRel hValid j

theorem registerShiftedWriteCheckAtBitPoint_of_incrementRelation
  {t : Nat}
  {init : InitialState}
  {val : RegisterTimeTable t}
  {wa : RegisterAddressColumns t}
  {wv : RegisterCycleValues t}
  {inc : RegisterTimeTable t}
  (hRel :
    TwistShout.IncrementRelation (K := F)
      (RegisterShiftedTimeTable init val) wa wv inc)
  (a : RegAddress)
  (j : TwistShout.CycleCube t) :
  inc a j =
    TwistShout.writeCheckExpression (K := F)
      (TwistShout.bitAddress (K := F) a)
      (TwistShout.bitVec (K := F) j)
      wa wv (RegisterShiftedTimeTable init val) := by
  exact registerWriteCheckAtBitPoint_of_incrementRelation hRel a j

theorem ramReadCheckAtBitCycle_of_relation
  {t : Nat}
  {val : RamTimeTable t}
  {addr : TwistShout.CycleCube t → RamAddress}
  {rv : RamCycleValues t}
  {ra : RamAddressColumns t}
  (hRel : TwistShout.ReadWriteMemoryRelation (K := F) val addr rv)
  (hValid : TwistShout.ValidAddressColumns (K := F) ra addr)
  (j : TwistShout.CycleCube t) :
  rv j = TwistShout.rwReadCheckExpression (K := F) ra val
    (TwistShout.bitVec (K := F) j) := by
  exact hRel.readCheckAtBitCycle hValid j

theorem ramWriteCheckAtBitPoint_of_incrementRelation
  {t : Nat}
  {val inc : RamTimeTable t}
  {wa : RamAddressColumns t}
  {wv : RamCycleValues t}
  (hRel : TwistShout.IncrementRelation (K := F) val wa wv inc)
  (a : RamAddress)
  (j : TwistShout.CycleCube t) :
  inc a j =
    TwistShout.writeCheckExpression (K := F)
      (TwistShout.bitAddress (K := F) a)
      (TwistShout.bitVec (K := F) j)
      wa wv val := by
  exact hRel.writeCheckAtBitPoint a j

theorem ramShiftedValEvaluationExpression_at_bitPoint
  {t : Nat}
  (init : InitialState)
  (inc : RamTimeTable t)
  (a : RamAddress)
  (j : TwistShout.CycleCube t) :
  RamShiftedValEvaluationExpression init inc
      (TwistShout.bitAddress (K := F) a)
      (TwistShout.bitVec (K := F) j) =
    RamShiftedTimeTable init inc a j := by
  calc
    RamShiftedValEvaluationExpression init inc
        (TwistShout.bitAddress (K := F) a)
        (TwistShout.bitVec (K := F) j)
      =
        TwistShout.timeTableMLE (K := F) (RamShiftedTimeTable init inc)
          (TwistShout.bitAddress (K := F) a)
          (TwistShout.bitVec (K := F) j) := by
            symm
            exact ramShiftedValEvaluationExpression_eq_timeTableMLE
              init inc (TwistShout.bitAddress (K := F) a)
              (TwistShout.bitVec (K := F) j)
    _ = RamShiftedVirtualValue init inc a
          (TwistShout.bitVec (K := F) j) := by
            exact ramTimeTableMLE_shifted_at_bitAddress init inc a
              (TwistShout.bitVec (K := F) j)
    _ = RamShiftedTimeTable init inc a j := by
            exact ramShiftedVirtualValue_at_bitCycle init inc a j

theorem ramShiftedReadCheckAtBitCycle_of_relation
  {t : Nat}
  {init : InitialState}
  {addr : TwistShout.CycleCube t → RamAddress}
  {rv : RamCycleValues t}
  {ra : RamAddressColumns t}
  {inc : RamTimeTable t}
  (hRel :
    TwistShout.ReadWriteMemoryRelation (K := F)
      (RamShiftedTimeTable init inc) addr rv)
  (hValid : TwistShout.ValidAddressColumns (K := F) ra addr)
  (j : TwistShout.CycleCube t) :
  rv j = TwistShout.rwReadCheckExpression (K := F) ra
    (RamShiftedTimeTable init inc)
    (TwistShout.bitVec (K := F) j) := by
  exact ramReadCheckAtBitCycle_of_relation hRel hValid j

theorem ramShiftedWriteCheckAtBitPoint_of_incrementRelation
  {t : Nat}
  {init : InitialState}
  {val : RamTimeTable t}
  {wa : RamAddressColumns t}
  {wv : RamCycleValues t}
  {inc : RamTimeTable t}
  (hRel :
    TwistShout.IncrementRelation (K := F)
      (RamShiftedTimeTable init val) wa wv inc)
  (a : RamAddress)
  (j : TwistShout.CycleCube t) :
  inc a j =
    TwistShout.writeCheckExpression (K := F)
      (TwistShout.bitAddress (K := F) a)
      (TwistShout.bitVec (K := F) j)
      wa wv (RamShiftedTimeTable init val) := by
  exact ramWriteCheckAtBitPoint_of_incrementRelation hRel a j

end Nightstream.Chip8.TwistConcreteBinding
