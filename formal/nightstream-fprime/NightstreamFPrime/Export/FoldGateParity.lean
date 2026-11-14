import Mathlib.Data.Nat.Digits.Defs
import NightstreamFPrime.Export.Codec
import NightstreamFPrime.Lifecycle.BoundedDepth.RandomOracle.FoldGate

/-! The fold gate at the production counts: the counts, the census numerator, `bits`, and `count`
and `denominator` as little-endian 64-bit words. Rust computes the same values from the shape of
each step relation and must match them
(`crates/nightstream/tests/bounded_depth_lifecycle/fold_gate.rs`). -/

namespace NightstreamFPrime.Export.FoldGateParity

open NightstreamFPrime.Export.Codec
open NightstreamFPrime.Lifecycle.BoundedDepth.RandomOracle.FoldGate

def schema : Nat := 1

private def words (value : Nat) : Value :=
  .array ((Nat.digits (2 ^ 64) value).map .atom)

def parityValue : Value :=
  .array [
    .atom schema,
    .atom production.cubeVariables,
    .atom production.width,
    .atom production.matrices,
    .atom production.numerator,
    .atom (bits production),
    words (count production),
    words denominator]

end NightstreamFPrime.Export.FoldGateParity
