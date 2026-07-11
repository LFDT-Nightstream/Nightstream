import Nightstream.Implementation.R1CS.FPrimeChunkDigestDefinitions0
import Nightstream.Implementation.R1CS.FPrimeChunkDigestDefinitions1
import Nightstream.Implementation.R1CS.FPrimeChunkDigestDefinitions2
import Nightstream.Implementation.R1CS.FPrimeChunkDigestDefinitions3
import Nightstream.Implementation.R1CS.FPrimeChunkDigestDefinitions4
import Nightstream.Implementation.R1CS.FPrimeChunkDigestDefinitions5

/-! Exact sharded SSA artifact for the production F' chunk-shape digest. -/

namespace Nightstream.Implementation.R1CS.FPrimeChunkDigest

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program

set_option maxRecDepth 262144

def schemaVersion : Nat := 1
def artifactKind : String := "r1cs/f-prime-chunk-shape-digest"
def sourceAnchor : String := "enforce_f_prime_chunk_public_digest_circuit"
def artifactSha256 : String := "cb9587540e95fe4cef093a9bfbe9957281a37144978cc4625ba69bd512a17b7b"
def witnessSha256 : String := "f273609f24aedadfc97ced41aa4d4c47e29ba7d52f0bb273d850508dad6b9a8f"

def inputColumns : List Nat := [0, 1]
def claimedColumns : List Nat := [2, 3, 4, 5]
def computedColumns : List Nat := [6655, 6656, 6657, 6658]
def bindingRowStart : Nat := 6657
def fullRowCount : Nat := 6661
def fullColCount : Nat := 6663

def definitions : List Definition :=
    Generated.definitions0 ++
    Generated.definitions1 ++
    Generated.definitions2 ++
    Generated.definitions3 ++
    Generated.definitions4 ++
    Generated.definitions5

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
