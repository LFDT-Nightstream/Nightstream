import CompressionLean.Spec.Checks

/-! Readers of the files that the Rust tests `setup_runs.rs` (`nightstream`) and `setup_columns.rs` (`neo-spartan`) write. -/

namespace CompressionLeanExe.Runs

open CompressionLean.Setup
open NightstreamFPrime.Spec (goldilocksModulus)

def littleEndian (bytes : ByteArray) (base n : ℕ) : ℕ := Id.run do
  let mut value : UInt64 := 0
  for k in [0:n] do
    value := value ||| bytes[base + k]!.toUInt64 <<< (8 * k).toUInt64
  return value.toNat

def readRuns (bytes : ByteArray) : Except String (List Push) := do
  let count := littleEndian bytes 24 8
  unless bytes.size = 32 + 26 * count do throw s!"{bytes.size} bytes do not hold {count} runs"
  let mut runs := []
  for i in [0:count] do
    let k := count - 1 - i
    let base := 32 + 26 * k
    let initial := littleEndian bytes (base + 10) 8
    let ratio := littleEndian bytes (base + 18) 8
    unless initial < goldilocksModulus ∧ ratio < goldilocksModulus do
      throw s!"run {k} has a word that is not canonical"
    runs := { row := littleEndian bytes base 4, start := littleEndian bytes (base + 4) 4,
              matrix := bytes[base + 8]!.toNat, len := bytes[base + 9]!.toNat, initial := initial,
              ratio := ratio } :: runs
  return runs

def readColumns (bytes : ByteArray) : Except String CompressionLean.Spec.SetupColumns := do
  let n := littleEndian bytes 0 8
  let s := littleEndian bytes 8 8
  unless bytes.size = 16 + 16 * n + 4 * s do throw s!"{bytes.size} bytes do not hold {n} runs and {s} slots"
  let column (base width count : ℕ) : Except String (Array CompressionLean.Gl) := do
    let mut out : Array CompressionLean.Gl := Array.mkEmpty count
    for k in [0:count] do
      let word := littleEndian bytes (base + width * k) width
      unless word < goldilocksModulus do throw s!"word {k} at {base} is not canonical"
      out := out.push (word : CompressionLean.Gl)
    return out
  return { row := ← column 16 4 n, slot := ← column (16 + 4 * n) 4 n, coef := ← column (16 + 8 * n) 8 n,
           block := ← column (16 + 16 * n) 4 s }

end CompressionLeanExe.Runs
