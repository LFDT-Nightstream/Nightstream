import CompressionLean.Spec.Checks

/-!
Owns the readers of the setup files that the Rust tests write: the runs that
`nightstream`'s `setup_runs.rs` writes (`production_runs.bin`), which
`setupCheck` and `runsCheck` read, and the run oracles that `neo-spartan`'s
`setup_columns.rs` writes (`production_setup_columns.bin`), which `setupCheck`
reads.
-/

namespace CompressionLeanExe.Runs

open CompressionLean.Setup
open NightstreamFPrime.Spec (goldilocksModulus)

/-- The little-endian word of the `n` bytes at `base`. -/
def littleEndian (bytes : ByteArray) (base n : ℕ) : ℕ := Id.run do
  let mut value : UInt64 := 0
  for k in [0:n] do
    value := value ||| bytes[base + k]!.toUInt64 <<< (8 * k).toUInt64
  return value.toNat

/-- The runs of the file in file order (`setup_runs.rs`): after four `u64`
words (rows, columns, matrices, run count), 26 bytes per run: the row and the
start (`u32`), the matrix and the length (`u8`), and the initial value and the
ratio (canonical `u64`). -/
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

/-- The run oracles of the file (`setup_columns.rs`): after the run count `n`
and the slot count `s` (`u64`), `n` rows and `n` slots (`u32`), `n`
coefficients (canonical `u64`) and `s` blocks (`u32`). -/
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
