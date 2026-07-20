//! Compact committed-width census for the stabilized fixed-point candidate.
//!
//! Owns: lossless export of the three-arm selective width audit and the exact
//! unpadded-to-Phi81-aligned width calculation used by fixed-point discovery.
//!
//! Does not own: relation materialization, semantic authority, row ownership,
//! permission to raise resource ceilings, or permission to remove constraints.

use std::fmt::Write as _;

use neo_fold_clean::frontends::r1cs_f_prime::ivc::R1csIvcFixedPointShapeAudit;
use neo_fold_clean::frontends::r1cs_f_prime::SelectiveProjectedRowsAudit;
use neo_math::D;

use super::GeneratedLeanFile;

const GENERATED_PATH: &str =
    "formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeSelectiveFixedPoint/PiCcsNc/ProductionDomain/WidthCensus/Generated/Layout.lean";
const IMPORT_ROOT: &str =
    "Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.ProductionDomain.WidthCensus.Schema";
const NAMESPACE: &str =
    "Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.ProductionDomain.WidthCensus.Generated";
const SCHEMA_VERSION: usize = 1;

pub(super) fn render(
    fixed_point: &R1csIvcFixedPointShapeAudit,
    projected: &SelectiveProjectedRowsAudit,
) -> GeneratedLeanFile {
    let width = fixed_point.width();
    assert_eq!(width.arms.len(), 3, "base/bootstrap/steady width census");
    assert_eq!(width.constant_coordinate, 1, "conventional constant coordinate");
    assert_eq!(
        width.branch_start,
        width.constant_coordinate
            + width.logical_public_coordinates
            + width.public_carrier_padding
            + width.selector_coordinates
            + width.alignment_padding
            + width.shared_private_coordinates,
        "exact selective prefix partition",
    );

    let arm_totals = width
        .arms
        .iter()
        .map(|arm| arm.total_branch_coordinates)
        .collect::<Vec<_>>();
    let max_arm_total = arm_totals.iter().copied().max().expect("three arms");
    let max_arm_indices = arm_totals
        .iter()
        .enumerate()
        .filter_map(|(index, &total)| (total == max_arm_total).then_some(index))
        .collect::<Vec<_>>();
    assert_eq!(max_arm_indices.len(), 1, "unique maximum-width arm");
    assert_eq!(
        width.total_coordinates,
        width.branch_start + max_arm_total,
        "unpadded width is prefix plus maximum selector-disjoint arm",
    );
    assert_eq!(
        projected.columns(),
        width.total_coordinates.next_multiple_of(D),
        "projected emitter uses exact Phi81-aligned width",
    );

    let mut contents = String::new();
    writeln!(
        contents,
        "import {IMPORT_ROOT}\n\n\
/-! Generated file: compact fixed-point selective-width census.\n\n\
Owns: three proof-free arm records and the exact prefix/max/round-up scalars\n\
read from the stabilized selective compiler audit.\n\n\
Does not own: emitted full matrices, semantic authority, exclusive row costs,\n\
resource-ceiling changes, or row-removal permission. Do not hand-edit.\n\n\
Emits constraints: no.\n\n\
| Stable stage path | Obligation | Authority class |\n\
|---|---|---|\n\
| `f_prime.fixed_point.width.generated` | Pin exact prefix, arm widths, maximum, and Phi81 round-up | checked artifact data |\n-/\n\n\
namespace {NAMESPACE}\n\n\
def schemaVersion : Nat := {SCHEMA_VERSION}\n\
def ringDegree : Nat := {D}\n\
def relationRows : Nat := {}\n\
def unpaddedCoordinates : Nat := {}\n\
def physicalCoordinates : Nat := {}\n\
def ringPaddingCoordinates : Nat := {}\n\
def constantCoordinates : Nat := {}\n\
def logicalPublicCoordinates : Nat := {}\n\
def publicCarrierPadding : Nat := {}\n\
def selectorCoordinates : Nat := {}\n\
def alignmentPadding : Nat := {}\n\
def sharedPrivateCoordinates : Nat := {}\n\
def branchStart : Nat := {}\n\
def maxArmIndex : Nat := {}\n\
def maxArmTotal : Nat := {}\n\
def arms : List RawArm := [",
        projected.rows(),
        width.total_coordinates,
        projected.columns(),
        projected.columns() - width.total_coordinates,
        width.constant_coordinate,
        width.logical_public_coordinates,
        width.public_carrier_padding,
        width.selector_coordinates,
        width.alignment_padding,
        width.shared_private_coordinates,
        width.branch_start,
        max_arm_indices[0],
        max_arm_total,
    )
    .expect("render width-census header");

    for (index, arm) in width.arms.iter().enumerate() {
        let equality_alias_coordinate_savings = arm
            .retained_coordinates_before_aliases
            .checked_sub(arm.decomposition_aliases)
            .and_then(|after_decomposition| after_decomposition.checked_sub(arm.branch_coordinates))
            .expect("aliases cannot increase branch width");
        assert_eq!(
            arm.retained_coordinates_before_aliases,
            arm.decomposition_aliases + equality_alias_coordinate_savings + arm.branch_coordinates,
            "exact alias savings partition",
        );
        assert_eq!(
            arm.derived_coordinates,
            41 * arm.derived_product_sums,
            "balanced derived-product encoding width",
        );
        assert_eq!(
            arm.total_branch_coordinates,
            arm.branch_coordinates + arm.derived_coordinates,
            "exact arm total",
        );
        let separator = if index == 0 { "  " } else { ", " };
        writeln!(
            contents,
            "{separator}{{ sourceColumns := {}, eliminatedColumns := {}, unitColumns := {}, balancedColumns := {}, binaryColumns := {}, retainedCoordinatesBeforeAliases := {}, decompositionAliases := {}, equalityAliases := {}, equalityAliasCoordinateSavings := {}, branchCoordinates := {}, derivedProductSums := {}, derivedCoordinates := {}, totalBranchCoordinates := {}, poseidonPermutations := {}, poseidonCoordinates := {} }}",
            arm.branch_source_columns,
            arm.eliminated_columns,
            arm.unit_columns,
            arm.balanced_columns,
            arm.binary_columns,
            arm.retained_coordinates_before_aliases,
            arm.decomposition_aliases,
            arm.equality_aliases,
            equality_alias_coordinate_savings,
            arm.branch_coordinates,
            arm.derived_product_sums,
            arm.derived_coordinates,
            arm.total_branch_coordinates,
            arm.traces.poseidon2_permutations,
            arm.traces.poseidon2_coordinates,
        )
        .expect("render width-census arm");
    }
    writeln!(contents, "]\n\nend {NAMESPACE}").expect("render width-census footer");
    assert!(
        contents.lines().count() < 1_500,
        "generated width census exceeds line limit"
    );
    GeneratedLeanFile {
        relative_path: GENERATED_PATH.to_owned(),
        contents,
    }
}
