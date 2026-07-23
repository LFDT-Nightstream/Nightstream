import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.ProductionDomain.WidthCensus.Schema

/-! Generated file: compact fixed-point selective-width census.

Owns: three proof-free arm records and the exact prefix/max/round-up scalars
read from the stabilized selective compiler audit.

Does not own: emitted full matrices, semantic authority, exclusive row costs,
resource-ceiling changes, or row-removal permission. Do not hand-edit.

Emits constraints: no.

| Stable stage path | Obligation | Authority class |
|---|---|---|
| `f_prime.fixed_point.width.generated` | Pin exact prefix, arm widths, maximum, and Phi81 round-up | checked artifact data |
-/

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.ProductionDomain.WidthCensus.Generated

def schemaVersion : Nat := 1
def ringDegree : Nat := 54
def relationRows : Nat := 14944219
def unpaddedCoordinates : Nat := 11437010
def physicalCoordinates : Nat := 11437038
def ringPaddingCoordinates : Nat := 28
def constantCoordinates : Nat := 1
def logicalPublicCoordinates : Nat := 256
def publicCarrierPadding : Nat := 13
def selectorCoordinates : Nat := 3
def alignmentPadding : Nat := 38
def sharedPrivateCoordinates : Nat := 0
def branchStart : Nat := 311
def maxArmIndex : Nat := 2
def maxArmTotal : Nat := 11436699
def arms : List RawArm := [
  { sourceColumns := 13049, eliminatedColumns := 10591, unitColumns := 647, balancedColumns := 1804, binaryColumns := 7, retainedCoordinatesBeforeAliases := 75059, decompositionAliases := 448, equalityAliases := 0, equalityAliasCoordinateSavings := 0, branchCoordinates := 74611, derivedProductSums := 0, derivedCoordinates := 0, totalBranchCoordinates := 74611, poseidonPermutations := 20, poseidonCoordinates := 70776 }
, { sourceColumns := 4419258, eliminatedColumns := 1622873, unitColumns := 2720149, balancedColumns := 76207, binaryColumns := 29, retainedCoordinatesBeforeAliases := 5846492, decompositionAliases := 1366049, equalityAliases := 384, equalityAliasCoordinateSavings := 5504, branchCoordinates := 4474939, derivedProductSums := 2276, derivedCoordinates := 93316, totalBranchCoordinates := 4568255, poseidonPermutations := 436, poseidonCoordinates := 1540430 }
, { sourceColumns := 10997106, eliminatedColumns := 3963194, unitColumns := 6863364, balancedColumns := 170295, binaryColumns := 253, retainedCoordinatesBeforeAliases := 13861651, decompositionAliases := 3459864, equalityAliases := 1760, equalityAliasCoordinateSavings := 61920, branchCoordinates := 10339867, derivedProductSums := 26752, derivedCoordinates := 1096832, totalBranchCoordinates := 11436699, poseidonPermutations := 541, poseidonCoordinates := 1924996 }
]

end Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.ProductionDomain.WidthCensus.Generated
