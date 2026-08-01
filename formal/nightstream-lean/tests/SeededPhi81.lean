import Nightstream.Implementation.R1CS.Artifacts.SeededPhi81.Generated.SeededPhi81Artifact

namespace NightstreamTests.SeededPhi81

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.SeededPhi81
open Nightstream.Implementation.R1CS.SeededPhi81Artifact
open Nightstream.Implementation.R1CS.ChaCha8Fast

set_option maxRecDepth 8192

/-- Exact `rand_chacha::ChaCha8Rng` stream parity for the Rust fixture. -/
example : ChaCha8.words seed 0 64 = expectedWords := by native_decide

/-- The native-efficient evaluator is extensionally pinned to the same Rust
fixture before it is used for large production coefficient schedules. -/
example : Nightstream.Implementation.R1CS.ChaCha8Fast.words seed 0 64 =
    expectedWords := by native_decide

/-- A distant counter slice pins random access and the 64-bit block counter,
which production schedules exercise far beyond the first four blocks. -/
example : Nightstream.Implementation.R1CS.ChaCha8Fast.words seed highWordStart 64 =
    expectedHighWords := by native_decide

/-- The verifier-owned master seed expands into the exact row/chunk seeds
used by Rust `setup_par`, including the first multi-chunk boundary. -/
example :
    (Nightstream.Implementation.R1CS.SeededAjtai.schedule
      seed setupRows setupMessageCols 4).chunkSize =
        expectedSetupChunkSize := by
  native_decide

example :
    (Nightstream.Implementation.R1CS.SeededAjtai.schedule
      seed setupRows setupMessageCols 4).seedsByOutput =
        expectedSetupSeedsByOutput := by
  native_decide

/-- The compact seed/geometry certificate is checked, not assumed. -/
example : block.Valid := by native_decide

/-- Expanding the compact Lean compiler yields every exact Rust fixture row,
including Phi81 rotation and zero-elision behavior. -/
example : block.rows = expectedRows := by native_decide

def inputs : List Nat := [0, 1, 2, 3, 4]

def initial : Nat → Nat := fun column =>
  match column with
  | 0 => 1
  | 1 => 1
  | 2 => 0
  | 3 => 1
  | 4 => 1
  | _ => 0

example : WellFormed inputs block.definitions := by native_decide

theorem initial_canonical : ∀ column, initial column < goldilocksP := by
  intro column
  have bounded : initial column ≤ 1 := by
    simp only [initial]
    split <;> omega
  unfold goldilocksP
  omega

/-- The compact compiler constructs a satisfying witness from its linear
equations, exercising the generic CIR-COMPLETE theorem on exact fixture rows. -/
example : Satisfies block.rows (run initial block.definitions) := by
  apply SeededPhi81.complete (block := block)
  · exact run_canonical initial_canonical
  · have preserved := run_preserves_known
      (by native_decide : WellFormed inputs block.definitions) initial
    exact preserved 0 (by decide)
  · exact run_definitions_hold
      (by native_decide : WellFormed inputs block.definitions) initial

end NightstreamTests.SeededPhi81
