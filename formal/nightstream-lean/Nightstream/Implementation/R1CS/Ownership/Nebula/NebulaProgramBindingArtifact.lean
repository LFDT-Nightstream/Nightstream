import Nightstream.Implementation.R1CS.Artifacts.NebulaProgramBinding.Generated.NebulaProgramBindingInstructions0
import Nightstream.Implementation.R1CS.Artifacts.NebulaProgramBinding.Generated.NebulaProgramBindingInstructions1
import Nightstream.Implementation.R1CS.Artifacts.NebulaProgramBinding.Generated.NebulaProgramBindingInstructions2
import Nightstream.Implementation.R1CS.Artifacts.NebulaProgramBinding.Generated.NebulaProgramBindingInstructions3

/-! Exact checked-program artifact for the production Nebula base binding. -/

namespace Nightstream.Implementation.R1CS.NebulaProgramBinding

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.CheckedProgram

set_option maxRecDepth 524288
set_option maxHeartbeats 5000000

def schemaVersion : Nat := 1
def artifactKind : String := "r1cs/nebula-program-binding"
def sourceAnchor : String := "enforce_nebula_lane_base_circuit"
def domainTag : String := "neo.fold.clean/nebula/program_binding/v1"
def artifactSha256 : String := "76bf1da9bd91a088e8500e304bf3244707cfeae5081ab2c43e8e924b0ae37c66"
def witnessSha256 : String := "93977a8e7b5bace2b1626b6acf9868e9da67c819d94bf15c1d3b0ff4bd2d0851"

def inputColumns : List Nat := [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66]
def laneColumns : List Nat := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50]
def semanticStateColumns : List Nat := [51, 52, 53, 54]
def initialSemanticStateColumns : List Nat := [55, 56, 57, 58]
def planDigestColumns : List Nat := [59, 60, 61, 62]
def initialMemoryDigestColumns : List Nat := [63, 64, 65, 66]
def tagColumns : List Nat := [67, 68, 69, 70, 71, 72, 73]
def computedBindingColumns : List Nat := [3687, 3688, 3689, 3690]
def carriedBindingColumns : List Nat := [1, 2, 3, 4]
def carriedMemoryColumns : List Nat := [47, 48, 49, 50]
def bindingLinkRowStart : Nat := 3628
def semanticLinkRowStart : Nat := 3632
def memoryLinkRowStart : Nat := 3678
def rowCount : Nat := 3682
def colCount : Nat := 3695
def definitionCount : Nat := 3628
def checkCount : Nat := 54

def instructions : List Instruction :=
    Generated.instructions0 ++
    Generated.instructions1 ++
    Generated.instructions2 ++
    Generated.instructions3

def rows : List Row := CheckedProgram.rows instructions

def bindingLinkRows : List Row :=
(List.range 4).map fun lane =>
builderLinearRow (computedBindingColumns.getD lane 0)
[(carriedBindingColumns.getD lane 0, 1)]

def semanticLinkRows : List Row :=
(List.range 4).map fun lane =>
builderLinearRow (semanticStateColumns.getD lane 0)
[(initialSemanticStateColumns.getD lane 0, 1)]

def memoryLinkRows : List Row :=
(List.range 4).map fun lane =>
builderLinearRow (carriedMemoryColumns.getD lane 0)
[(initialMemoryDigestColumns.getD lane 0, 1)]

theorem digest_widths :
semanticStateColumns.length = 4 ∧
initialSemanticStateColumns.length = 4 ∧
planDigestColumns.length = 4 ∧
initialMemoryDigestColumns.length = 4 ∧
computedBindingColumns.length = 4 ∧
carriedBindingColumns.length = 4 ∧
carriedMemoryColumns.length = 4 := by decide
theorem binding_link_rows_exact :
(rows.drop bindingLinkRowStart).take 4 = bindingLinkRows := by decide
theorem semantic_link_rows_exact :
(rows.drop semanticLinkRowStart).take 4 = semanticLinkRows := by decide
theorem memory_link_rows_exact :
(rows.drop memoryLinkRowStart).take 4 = memoryLinkRows := by decide

end Nightstream.Implementation.R1CS.NebulaProgramBinding
