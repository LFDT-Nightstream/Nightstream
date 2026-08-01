import Nightstream.Implementation.R1CS.Artifacts.FPrime.Generated.FPrimeChunkDigestDefinitions0
import Nightstream.Implementation.R1CS.Artifacts.FPrime.Generated.FPrimeChunkDigestDefinitions1
import Nightstream.Implementation.R1CS.Artifacts.FPrime.Generated.FPrimeChunkDigestDefinitions2
import Nightstream.Implementation.R1CS.Artifacts.FPrime.Generated.FPrimeChunkDigestDefinitions3
import Nightstream.Implementation.R1CS.Artifacts.FPrime.Generated.FPrimeChunkDigestDefinitions4
import Nightstream.Implementation.R1CS.Artifacts.FPrime.Generated.FPrimeChunkDigestDefinitions5
import Nightstream.Implementation.R1CS.Artifacts.FPrime.Generated.FPrimeChunkDigestDefinitions6

/-! Exact sharded SSA artifact for the production F' chunk-shape digest. -/

namespace Nightstream.Implementation.R1CS.FPrimeChunkDigest

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program

set_option maxRecDepth 262144

def schemaVersion : Nat := 1
def artifactKind : String := "r1cs/f-prime-chunk-shape-digest"
def sourceAnchor : String := "enforce_f_prime_chunk_public_digest_circuit"
def artifactSha256 : String := "ba805f614665abd53dbc25416b067db62ee1efecc2691a773c2b57446a9da3da"
def witnessSha256 : String := "0a823e9eaaf616899b7f376c3cf9122fb1f39fb76411d4c8da61fec5f17e28e4"

def inputColumns : List Nat := [0, 1]
def claimedColumns : List Nat := [2, 3, 4, 5]
def computedColumns : List Nat := [7261, 7262, 7263, 7264]
def bindingRowStart : Nat := 7263
def fullRowCount : Nat := 7267
def fullColCount : Nat := 7269

def definitions : List Definition :=
    Generated.definitions0 ++
    Generated.definitions1 ++
    Generated.definitions2 ++
    Generated.definitions3 ++
    Generated.definitions4 ++
    Generated.definitions5 ++
    Generated.definitions6

def rows : List Row := definitions.map Definition.builderRow

def columnPairs : List (Nat × Nat) := claimedColumns.zip computedColumns
def equalityRow (columns : Nat × Nat) : Row :=
⟨[(columns.1, 1), (columns.2, goldilocksP - 1)], [(0, 1)], []⟩
def bindingRows : List Row := columnPairs.map equalityRow

theorem definitions_length : definitions.length = fullRowCount := by native_decide
theorem rows_length : rows.length = fullRowCount := by native_decide
theorem bindingRows_length : bindingRows.length = 4 := by native_decide
theorem definitions_canonical :
∀ definition ∈ definitions, definition.Canonical := by native_decide
theorem definitions_wellFormed : WellFormed inputColumns definitions := by native_decide

end Nightstream.Implementation.R1CS.FPrimeChunkDigest
