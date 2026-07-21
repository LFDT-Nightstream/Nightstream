import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.ProductionDomain.WidthCensus.Generated.Layout

/-!
Exact committed-width accounting for the stabilized fixed-point candidate.

Assurance tier: artifact-checked compiler accounting only.

Owns: the three generated selector-disjoint arm records; the exact public,
selector, and alignment prefix; alias and derived-coordinate accounting; the
unique maximum-width arm; and the final 54-coordinate alignment calculation.

Does not own: a materialized production relation, exclusive constraint-row
costs, semantic acceptance, authority to raise the memory ceiling, or
permission to remove constraints.

Emits constraints: none.

The complete closed certificate contains exactly three proof-free `RawArm`
records. These theorems use kernel reduction (`decide`), not `native_decide`.

| Stable stage path | Obligation | Authority class |
|---|---|---|
| `f_prime.fixed_point.width.prefix` | prefix is exactly 311 coordinates | computed artifact |
| `f_prime.fixed_point.width.arms` | every arm has exact alias/derived accounting | computed artifact |
| `f_prime.fixed_point.width.maximum` | steady arm uniquely owns the 11,725,143-coordinate suffix | computed artifact |
| `f_prime.fixed_point.width.alignment` | unpadded 11,725,454 rounds to 11,725,506 | computed artifact |
-/

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.ProductionDomain.WidthCensus

namespace G

export Generated
  (schemaVersion ringDegree relationRows unpaddedCoordinates
    physicalCoordinates ringPaddingCoordinates constantCoordinates
    logicalPublicCoordinates publicCarrierPadding selectorCoordinates
    alignmentPadding sharedPrivateCoordinates branchStart maxArmIndex
    maxArmTotal arms)

end G

/-- Exact scalar header exported by the stabilized compiler audit. -/
theorem header_exact :
    G.schemaVersion = 1 /\
      G.ringDegree = 54 /\
      G.relationRows = 14946911 /\
      G.unpaddedCoordinates = 11725454 /\
      G.physicalCoordinates = 11725506 /\
      G.ringPaddingCoordinates = 52 /\
      G.branchStart = 311 /\
      G.maxArmIndex = 2 /\
      G.maxArmTotal = 11725143 := by
  decide

/-- The verifier-owned prefix is constant, logical public carrier, public
padding, three branch selectors, private alignment padding, and no shared
private coordinates—in that order. -/
theorem prefix_accounting :
    G.constantCoordinates + G.logicalPublicCoordinates +
        G.publicCarrierPadding + G.selectorCoordinates +
        G.alignmentPadding + G.sharedPrivateCoordinates =
      G.branchStart := by
  decide

/-- The generated certificate contains exactly base, bootstrap-recursive, and
steady-recursive arm records. -/
theorem arms_length : G.arms.length = 3 := by
  decide

/-- Every generated arm partitions its pre-alias coordinates into
decomposition reuse, weighted equality reuse, and owned branch coordinates;
its derived region is exactly 41 coordinates per grouped product sum. -/
theorem arm_accounting
    {arm : RawArm}
    (member : arm ∈ G.arms) :
    arm.retainedCoordinatesBeforeAliases =
        arm.decompositionAliases + arm.equalityAliasCoordinateSavings +
          arm.branchCoordinates /\
      arm.derivedCoordinates = 41 * arm.derivedProductSums /\
      arm.totalBranchCoordinates =
        arm.branchCoordinates + arm.derivedCoordinates := by
  simp only [G.arms, List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with equality | equality | equality <;>
    subst arm <;> decide

/-- The steady-recursive arm is the unique maximum-width selector-disjoint
suffix. -/
theorem steady_unique_maximum :
    G.arms.map RawArm.totalBranchCoordinates =
        [74611, 4853189, G.maxArmTotal] /\
      74611 < G.maxArmTotal /\
      4853189 < G.maxArmTotal := by
  decide

/-- Selective branch arenas overlap under disjoint selectors, so the unpadded
width is the 311-coordinate prefix plus the maximum arm—not the sum of arms. -/
theorem unpadded_accounting :
    G.unpaddedCoordinates = G.branchStart + G.maxArmTotal := by
  decide

/-- Exact Phi81 alignment equation used by the projected emitter. -/
theorem physical_round_up :
    G.physicalCoordinates =
      ((G.unpaddedCoordinates + G.ringDegree - 1) / G.ringDegree) *
        G.ringDegree := by
  decide

theorem physical_eq_unpadded_add_padding :
    G.physicalCoordinates =
      G.unpaddedCoordinates + G.ringPaddingCoordinates := by
  decide

/-- The candidate fits the current constructor guard with 4,274,494 physical
coordinates of headroom. This is compiler accounting, not evidence that the
complete relation was materialized. -/
theorem fits_current_constructor_guard :
    G.physicalCoordinates <= 16000000 /\
      16000000 - G.physicalCoordinates = 4274494 := by
  decide

end Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.ProductionDomain.WidthCensus
