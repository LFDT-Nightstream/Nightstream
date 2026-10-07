import Mathlib.Data.Multiset.AddSub
import Mathlib.Data.Nat.Bitwise
import NightstreamFPrime.Spec.Nebula.Plan

/-! Owns the typed memory records of spec §6: port accesses, operation and scan
slots, the tuples they define, the per-step multisets, and the bit lanes of
§6.3. The widths of `Fits` and `Shaped` are the typed form of rows O1 and S1.
It does not own packing, hashing, fingerprints, or row checks. -/

namespace NightstreamFPrime.Spec.Nebula

/-- A memory tuple `(t, g, v)`: timestamp, global index, value (spec §5). -/
abbrev Tuple := ℕ × ℕ × ℕ

/-- One active port access (spec §10). For a read, `vw = vr`. -/
structure PortAccess where
  isWrite : Bool
  isRam : Bool
  addr : ℕ
  vr : ℕ
  vw : ℕ
deriving DecidableEq

namespace PortAccess

/-- `g = addr + is_ram · R` (spec §6.1). -/
def globalIndex (p : Plan) (a : PortAccess) : ℕ :=
  if a.isRam then p.romSize + a.addr else a.addr

/-- A well-formed access: address in its namespace, 32-bit values, a read
keeps the value, no write to ROM (spec §5, rows O3, O5, O6). -/
def Valid (p : Plan) (a : PortAccess) : Prop :=
  (if a.isRam then a.addr < p.ramSize else a.addr < p.romSize) ∧
    a.vr < 2 ^ 32 ∧ a.vw < 2 ^ 32 ∧
    (a.isWrite = false → a.vw = a.vr) ∧ (a.isWrite = true → a.isRam = true)

instance (p : Plan) (a : PortAccess) : Decidable (a.Valid p) := by
  unfold Valid; infer_instance

end PortAccess

/-- Spec §6.1 operation slot. An inactive slot is `padSlot`. -/
structure OpSlot where
  pad : Bool
  isWrite : Bool
  isRam : Bool
  addr : ℕ
  vr : ℕ
  vw : ℕ
  rt : ℕ
deriving DecidableEq

namespace OpSlot

/-- The canonical inactive slot (row O7). -/
def padSlot : OpSlot := ⟨true, false, false, 0, 0, 0, 0⟩

/-- The port access that an active slot carries. -/
def port (s : OpSlot) : Option PortAccess :=
  if s.pad then none else some ⟨s.isWrite, s.isRam, s.addr, s.vr, s.vw⟩

/-- Field widths of row O1. -/
def Fits (p : Plan) (s : OpSlot) : Prop :=
  s.addr < 2 ^ p.μ ∧ s.vr < 2 ^ 32 ∧ s.vw < 2 ^ 32 ∧ s.rt < 2 ^ p.wTs

end OpSlot

/-- Spec §6.2 scan slot. -/
structure ScanSlot where
  value : ℕ
  stamp : ℕ
deriving DecidableEq

/-- Field widths of row S1. -/
def ScanSlot.Fits (p : Plan) (c : ScanSlot) : Prop :=
  c.value < 2 ^ 32 ∧ c.stamp < 2 ^ p.wTs

/-- The records of one step: the ops lane and the IS and FS chunks. -/
structure StepRecords where
  ops : List OpSlot
  initialScan : List ScanSlot
  finalScan : List ScanSlot

/-- Rows O1 and S1 at the typed level: lane lengths and field widths. -/
structure StepRecords.Shaped (p : Plan) (z : StepRecords) : Prop where
  opsLength : z.ops.length = p.bOps
  initialLength : z.initialScan.length = p.bScan
  finalLength : z.finalScan.length = p.bScan
  opsFit : ∀ s ∈ z.ops, s.Fits p
  initialFit : ∀ c ∈ z.initialScan, c.Fits p
  finalFit : ∀ c ∈ z.finalScan, c.Fits p

/-- One active operation with its claimed previous stamp and its write stamp. -/
structure MemOp where
  access : PortAccess
  rt : ℕ
  wt : ℕ

namespace MemOp

/-- `RS = (rt, g, v_r)`. -/
def read (p : Plan) (o : MemOp) : Tuple := (o.rt, o.access.globalIndex p, o.access.vr)

/-- `WS = (wt, g, v_w)`. -/
def write (p : Plan) (o : MemOp) : Tuple := (o.wt, o.access.globalIndex p, o.access.vw)

end MemOp

/-- Active operations of a step entered with timestamp `ts`: the `k`-th active
slot (from 0) writes at `ts + k + 1` (row O2 and spec §6.1). -/
def activeOps (ts : ℕ) : List OpSlot → List MemOp
  | [] => []
  | s :: rest =>
    match s.port with
    | none => activeOps ts rest
    | some a => ⟨a, s.rt, ts + 1⟩ :: activeOps (ts + 1) rest

/-- Number of active slots of a step. -/
def activeCount (z : StepRecords) : ℕ := (z.ops.filterMap OpSlot.port).length

/-- Scan tuples with the structural global index `base + j` (spec §6.2). -/
def scanTuplesFrom (base : ℕ) : List ScanSlot → List Tuple
  | [] => []
  | c :: rest => (c.stamp, base, c.value) :: scanTuplesFrom (base + 1) rest

/-- IS or FS tuples of step `idx`: cells `idx · B_scan + j`. -/
def scanTuples (p : Plan) (idx : ℕ) (scan : List ScanSlot) : List Tuple :=
  scanTuplesFrom (idx * p.bScan) scan

/-- The four multisets of a check (spec §5). -/
structure Multisets where
  read : Multiset Tuple
  write : Multiset Tuple
  initial : Multiset Tuple
  final : Multiset Tuple

namespace Multisets

instance : Zero Multisets := ⟨⟨0, 0, 0, 0⟩⟩

instance : Add Multisets :=
  ⟨fun a b => ⟨a.read + b.read, a.write + b.write, a.initial + b.initial, a.final + b.final⟩⟩

/-- `IS ∪ WS = RS ∪ FS`. -/
def Balanced (m : Multisets) : Prop := m.initial + m.write = m.read + m.final

end Multisets

/-- The multisets of one step entered with timestamp `ts` at segment index `idx`. -/
def stepMultisets (p : Plan) (ts idx : ℕ) (z : StepRecords) : Multisets :=
  ⟨((activeOps ts z.ops).map (MemOp.read p) : List Tuple),
   ((activeOps ts z.ops).map (MemOp.write p) : List Tuple),
   (scanTuples p idx z.initialScan : List Tuple),
   (scanTuples p idx z.finalScan : List Tuple)⟩

/-- Little-endian bits of `x` at width `w`. -/
def bitsLE (w x : ℕ) : List Bool := (List.range w).map x.testBit

/-- Spec §6.3 encoding of one operation slot: field order, little-endian bits. -/
def OpSlot.bits (p : Plan) (s : OpSlot) : List Bool :=
  [s.pad, s.isWrite, s.isRam] ++ bitsLE p.μ s.addr ++ bitsLE 32 s.vr ++
    bitsLE 32 s.vw ++ bitsLE p.wTs s.rt

/-- Spec §6.3 encoding of one scan slot. IS and FS share it. -/
def ScanSlot.bits (p : Plan) (c : ScanSlot) : List Bool :=
  bitsLE 32 c.value ++ bitsLE p.wTs c.stamp

/-- The ops lane of a step: slot-major. -/
def opsLane (p : Plan) (ops : List OpSlot) : List Bool := ops.flatMap (OpSlot.bits p)

/-- An IS or FS lane of a step: slot-major. -/
def scanLane (p : Plan) (scan : List ScanSlot) : List Bool := scan.flatMap (ScanSlot.bits p)

end NightstreamFPrime.Spec.Nebula
