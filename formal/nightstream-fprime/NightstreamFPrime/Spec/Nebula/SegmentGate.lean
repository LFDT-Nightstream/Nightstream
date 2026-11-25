import Mathlib.Data.Nat.Log
import Mathlib.Data.Nat.Size
import NightstreamFPrime.Spec.Folding.Schedule
import NightstreamFPrime.Spec.Nebula.SetupCheck

/-! Owns the memory gate of the Rust verifier (SPEC.md section 7.1): the bound of rule 6, the
geometry that Rust `Geometry::new` builds from its counts, the bits that the gate certifies for
all `S_max` segments of a proof, and the four fixed WASM sizes with their segment counts, whose
runs never reach a refresh fold (`WasmSize.steps_le_light`). Rust
states none of these numbers on its own authority: `Export.NebulaGateParity` emits them, and
Rust tests compare them. -/

namespace NightstreamFPrime.Spec.Nebula

namespace Geometry

/-- The bound of rule 6 on `m_mem`. -/
def secureTuples : Nat := 139509

theorem secure_iff_secureTuples (p : Geometry) : p.Secure ↔ p.maxTuples ≤ secureTuples :=
  Iff.rfl

instance (p : Geometry) : Decidable p.Valid :=
  decidable_of_iff
    (p.n * p.bScan = p.cells ∧ p.sMax * p.n * p.bOps < 2 ^ p.wTs ∧ p.wTs ≤ 31 ∧
      (p.bOps * p.opWidth < fieldModulus ∧ p.bScan * p.scanWidth < fieldModulus) ∧
      p.cells < fieldModulus ∧ (0 < p.bOps ∧ 0 < p.bScan ∧ 0 < p.n ∧ 0 < p.sMax))
    ⟨fun ⟨cover, stamps, encoding, lanes, below, positive⟩ =>
      ⟨cover, stamps, encoding, lanes, below, positive⟩,
     fun ⟨cover, stamps, encoding, lanes, below, positive⟩ =>
      ⟨cover, stamps, encoding, lanes, below, positive⟩⟩

instance (p : Geometry) : Decidable p.Secure := Nat.decLe _ _

/-- Rust `Geometry::new`: `w_ts` is the fewest bits that hold `S_max · N · B_ops`, and `B_scan`
is the cells over `N`. -/
def ofCounts (r μ bOps n sMax : ℕ) : Geometry where
  r := r
  μ := μ
  wTs := (sMax * n * bOps).size
  bOps := bOps
  bScan := (2 ^ r + 2 ^ μ) / n
  n := n
  sMax := sMax

/-- The fingerprint error of all `S_max` segments of a proof is `segmentError / q²`. -/
def segmentError (p : Geometry) : ℕ := 2 * p.maxTuples * p.sMax

/-- Rust `segments_security_bits`: the largest `k` with `S_max · 2 · m_mem / q² ≤ 2^−k`. -/
def segmentBits (p : Geometry) : ℕ := (fieldModulus ^ 2 / p.segmentError).log2

/-- The gate's bits are exactly the largest `k` with `segmentError · 2^k ≤ q²`. -/
theorem segmentBits_eq_iff (p : Geometry) (bits : ℕ) (positive : 0 < p.segmentError)
    (below : p.segmentError ≤ fieldModulus ^ 2) :
    p.segmentBits = bits ↔
      p.segmentError * 2 ^ bits ≤ fieldModulus ^ 2 ∧
        fieldModulus ^ 2 < p.segmentError * 2 ^ (bits + 1) := by
  have nonzero : fieldModulus ^ 2 / p.segmentError ≠ 0 :=
    (Nat.div_pos below positive).ne'
  rw [segmentBits, Nat.log2_eq_log_two, Nat.log_eq_iff (Or.inr ⟨Nat.one_lt_two, nonzero⟩),
    Nat.le_div_iff_mul_le positive, Nat.div_lt_iff_lt_mul positive, Nat.mul_comm (2 ^ bits),
    Nat.mul_comm (2 ^ (bits + 1))]

/-- The gate is sound: the error of all segments is at most `2^−segmentBits`. -/
theorem segmentError_le (p : Geometry) (positive : 0 < p.segmentError)
    (below : p.segmentError ≤ fieldModulus ^ 2) :
    p.segmentError * 2 ^ p.segmentBits ≤ fieldModulus ^ 2 :=
  ((p.segmentBits_eq_iff _ positive below).mp rfl).1

end Geometry

/-- The four fixed WASM sizes (`decisions/wasm-fixed-sizes.md`). -/
inductive WasmSize where
  | micro
  | small
  | medium
  | large
  deriving DecidableEq, Repr

/-- The counts of a WASM size: `log2` of the ROM cells and of the RAM cells, the trace rows of
one step, the steps of one segment and the most segments. -/
structure WasmShape where
  r : ℕ
  μ : ℕ
  rows : ℕ
  n : ℕ
  sMax : ℕ

namespace WasmSize

/-- The bits of security that each WASM size certifies for the memory fingerprints of all its
segments. -/
def gateBits : Nat := 109

/-- The slots of one trace row. -/
def rowSlots : ℕ := 8

def shape : WasmSize → WasmShape
  | micro => ⟨11, 8, 800, 1, 30⟩
  | small => ⟨13, 10, 256, 1, 23⟩
  | medium => ⟨13, 11, 664, 2, 12⟩
  | large => ⟨14, 12, 664, 4, 6⟩

/-- The geometry with `sMax` segments. -/
def geometryWith (size : WasmSize) (sMax : ℕ) : Geometry :=
  Geometry.ofCounts size.shape.r size.shape.μ (size.shape.rows * rowSlots) size.shape.n sMax

/-- The geometry of a size. -/
def geometry (size : WasmSize) : Geometry := size.geometryWith size.shape.sMax

theorem valid (size : WasmSize) : size.geometry.Valid := by
  cases size <;> exact
    { exactCover := by decide +kernel
      timestampRange := by decide +kernel
      fieldEncoding := by decide +kernel
      laneBits := by decide +kernel
      belowModulus := by decide +kernel
      positive := by decide +kernel }

theorem secure (size : WasmSize) : size.geometry.Secure := by
  cases size <;> (unfold Geometry.Secure; decide +kernel)

/-- Each size certifies exactly `gateBits`. -/
theorem segmentBits_eq (size : WasmSize) : size.geometry.segmentBits = gateBits := by
  cases size <;>
    exact (Geometry.segmentBits_eq_iff _ _ (by decide +kernel) (by decide +kernel)).mpr
      ⟨by decide +kernel, by decide +kernel⟩

/-- Each size has the most segments at `gateBits`: one more segment certifies fewer bits. -/
theorem segmentBits_succ_lt (size : WasmSize) :
    (size.geometryWith (size.shape.sMax + 1)).segmentBits < gateBits := by
  cases size <;>
    exact Nat.lt_of_le_of_lt
      (Nat.le_of_eq ((Geometry.segmentBits_eq_iff _ 108 (by decide +kernel)
        (by decide +kernel)).mpr ⟨by decide +kernel, by decide +kernel⟩)) (by decide)

/-- A proof of a size has at most `S_max · N` steps (Algorithm 8, step 3), and that is at most
`cap + 1`, so no run of a size reaches a refresh fold: its first step is the base step and every
later step is a light fold into slot 0 below the cap (SPEC.md, Algorithm 9). -/
theorem steps_le_light (size : WasmSize) :
    size.geometry.sMax * size.geometry.n ≤ Folding.Schedule.productionLightFoldCap + 1 := by
  cases size <;> decide

end WasmSize

end NightstreamFPrime.Spec.Nebula
