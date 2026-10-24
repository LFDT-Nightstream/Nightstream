import CompressionLean.Checks.SetupFacts
import CompressionLeanExe.Parse

/-!
The `setupCheck` executable. It reads the layer-1 key that `neo-spartan`'s
`lowering.rs` writes (`production_layer1.json`) and the runs that
`nightstream`'s `setup_runs.rs` writes (`production_runs.bin`), and runs the
setup check (`setupFailure`) on them. It prints `setup facts hold`, or the
first fact that fails and exits with 1.
-/

namespace CompressionLeanExe.SetupCheck

open CompressionLean.Checks
open CompressionLean.Game
open CompressionLean CompressionLean.Final
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

end CompressionLeanExe.SetupCheck

open CompressionLeanExe.SetupCheck in
def main (args : List String) : IO UInt32 := do
  let [layer1, runs] := args | IO.eprintln "usage: setupCheck <layer1.json> <runs.bin>"; return 2
  let key ← IO.ofExcept (← CompressionLeanExe.Parse.readJson CompressionLean.Circuit.Layer1Input layer1)
  let pushes ← IO.ofExcept (readRuns (← IO.FS.readBinFile runs))
  match CompressionLean.Checks.setupFailure key pushes with
  | none => IO.println "setup facts hold"; return 0
  | some fact => IO.println s!"setup fact fails: {fact}"; return 1
