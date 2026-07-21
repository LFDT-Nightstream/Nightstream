/-!
Schema for the compact stabilized fixed-point committed-width certificate.

Owns: the proof-free field layout of one compiler arm record.

Does not own: concrete counts, semantic authority, costs outside this profile,
or row-removal permission.

Emits constraints: no.

| Stable stage path | Obligation | Authority class |
|---|---|---|
| `f_prime.fixed_point.width.schema.arm` | Type every source, alias, derived, and transcript width component | direct data layout |
-/

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.ProductionDomain.WidthCensus

/-- Exact committed-width accounting for one selector-disjoint source arm. -/
structure RawArm where
  sourceColumns : Nat
  eliminatedColumns : Nat
  unitColumns : Nat
  balancedColumns : Nat
  binaryColumns : Nat
  retainedCoordinatesBeforeAliases : Nat
  decompositionAliases : Nat
  equalityAliases : Nat
  equalityAliasCoordinateSavings : Nat
  branchCoordinates : Nat
  derivedProductSums : Nat
  derivedCoordinates : Nat
  totalBranchCoordinates : Nat
  poseidonPermutations : Nat
  poseidonCoordinates : Nat
deriving DecidableEq, Repr

end Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.ProductionDomain.WidthCensus
