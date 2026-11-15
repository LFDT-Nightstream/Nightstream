import NightstreamFPrime.Export.Codec
import NightstreamFPrime.Spec.Nebula.SegmentGate

/-! The memory gate (`Spec.Nebula.SegmentGate`): the bound of rule 6, the bits of the WASM
sizes, and for each WASM size its geometry, `m_mem` and the bits of its segments. Rust must match
them (`crates/nightstream/tests/nebula/memory_gate.rs`, `crates/neo-wasm/tests/wasm_sizes_lean.rs`). -/

namespace NightstreamFPrime.Export.NebulaGateParity

open NightstreamFPrime.Export.Codec
open NightstreamFPrime.Spec.Nebula

def schema : Nat := 1

private def size (size : WasmSize) : Value :=
  let p := size.geometry
  .array [.atom p.r, .atom p.μ, .atom p.wTs, .atom p.bOps, .atom p.bScan, .atom p.n,
    .atom p.sMax, .atom p.maxTuples, .atom p.segmentBits, .atom size.shape.2.1]

/-- Geometries for the comparison of Rust `is_valid` and `is_secure` with `Geometry.Valid` and
`Geometry.Secure`: the first test geometry, one geometry that breaks each of rules 1 to 5 or has a
count of zero, and the two geometries at the bound of rule 6. -/
def cases : List Geometry :=
  let first : Geometry := ⟨2, 2, 5, 2, 4, 2, 4⟩
  [first, { first with bScan := 5 }, { first with wTs := 4 }, { first with wTs := 32 },
    { first with sMax := 0 }, { first with r := 64 }, { first with bOps := 2 ^ 62 },
    ⟨17, 13, 8, Geometry.secureTuples - (2 ^ 17 + 2 ^ 13), 2 ^ 17 + 2 ^ 13, 1, 1⟩,
    ⟨17, 13, 8, Geometry.secureTuples + 1 - (2 ^ 17 + 2 ^ 13), 2 ^ 17 + 2 ^ 13, 1, 1⟩]

private def decided (p : Geometry) : Value :=
  .array [.atom p.r, .atom p.μ, .atom p.wTs, .atom p.bOps, .atom p.bScan, .atom p.n,
    .atom p.sMax, .atom (if p.Valid then 1 else 0), .atom (if p.Secure then 1 else 0)]

def parityValue : Value :=
  .array [
    .atom schema,
    .atom Geometry.secureTuples,
    .atom WasmSize.gateBits,
    .array [size .small, size .medium, size .large],
    .array (cases.map decided)]

end NightstreamFPrime.Export.NebulaGateParity
