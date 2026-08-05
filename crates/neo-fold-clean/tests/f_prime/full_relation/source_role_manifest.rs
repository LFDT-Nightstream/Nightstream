//! Fixed F-prime source-role artifact generation and adversarial checks.
//!
//! Owns: exact base/recursive source audit invocation, event-column role and
//! source-loop placement assertions, deterministic Lean rendering, and
//! `.lean.expected` output.
//!
//! Does not own: deferred encoded coordinates, CE coordinates, selector
//! composition, source semantics, or permission to promote generated data.
//!
//! Emits constraints: no.
//!
//! | Artifact branch | Mathematical obligation | Mutation evidence |
//! |---|---|---|
//! | source partition | every column occurs once in positive abutting runs | start/length/drop mutations |
//! | exclusive roles | every run has one estimator-reconciled role | run/total mutations |
//! | sampler projections | retained bits stay private; decoded projections stay derived | exact event-column assertions |
//! | canonical u64 | every traced field is direct or linearly derived | overlap-census mutation |
//! | ordinary placement | source-order cursor fixes every 41-coordinate word start | pointwise base-plan check and placement mutations |
//! | source phase | compact end/final-width metadata matches the exact estimator | prefix/end/bound mutations |

use std::collections::BTreeSet;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

use neo_fold_clean::frontends::f_prime::gadget_native::{
    audit_r1cs_gadget_native_ordinary_placement, audit_r1cs_gadget_native_source_manifest,
    GadgetNativeOrdinaryPlacementManifest, GadgetNativeOrdinaryPlacementManifestTestMutation, GadgetNativePlan,
    GadgetNativeSourceManifest, GadgetNativeSourceManifestTestMutation, GadgetNativeSourceRole,
    ORDINARY_PRIVATE_DIGITS,
};
use neo_fold_clean::frontends::r1cs_f_prime::FullFPrimeBranchExecution;

const ARTIFACT_PATH: &str = "formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeRecursive/Generated/FPrimeBranchSourceRoleManifestData.lean";
const PLACEMENT_ARTIFACT_PATH: &str = "formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeRecursive/Generated/FPrimeBranchOrdinaryPlacementData.lean";
const PACKED_CHUNK_TARGET_BYTES: usize = 10 * 1024;
const PACKED_CHUNK_MAX_BYTES: usize = 12 * 1024;

pub(super) fn check_source_role_manifest(base: &FullFPrimeBranchExecution, recursive: &FullFPrimeBranchExecution) {
    let base_manifest =
        audit_r1cs_gadget_native_source_manifest(base.snapshot(), base.encoding_trace(), base.public_bit_columns())
            .expect("exact base source-role manifest");
    let recursive_manifest = audit_r1cs_gadget_native_source_manifest(
        recursive.snapshot(),
        recursive.encoding_trace(),
        recursive.public_bit_columns(),
    )
    .expect("exact recursive source-role manifest");
    let base_placement =
        audit_r1cs_gadget_native_ordinary_placement(base.snapshot(), base.encoding_trace(), base.public_bit_columns())
            .expect("exact base ordinary-placement manifest");
    let recursive_placement = audit_r1cs_gadget_native_ordinary_placement(
        recursive.snapshot(),
        recursive.encoding_trace(),
        recursive.public_bit_columns(),
    )
    .expect("exact recursive ordinary-placement manifest");

    assert_event_roles(&base_manifest, base);
    assert_event_roles(&recursive_manifest, recursive);
    assert_mutations_rejected(&base_manifest);
    assert_mutations_rejected(&recursive_manifest);
    let materialized_base = base
        .encode_gadget_native()
        .expect("bounded base branch materializes from the shared source schedule");
    assert_eq!(
        materialized_base
            .decode_source()
            .expect("materialized gadget-native base branch is satisfied and decodes"),
        base.snapshot().witness(),
        "materialized gadget-native base branch must decode to the source witness"
    );
    assert_materializer_pointwise(&base_manifest, &materialized_base.plan);
    assert_ordinary_placement_pointwise(&base_placement, &materialized_base.plan);
    assert_equal_aggregate_swap_detected(&base_manifest, &materialized_base.plan);
    assert_placement_branch(
        "base",
        &base_manifest,
        &base_placement,
        3_226,
        132_911,
        132_911,
        (1, 257),
        (23_550, 132_870),
    );
    assert_placement_branch(
        "recursive",
        &recursive_manifest,
        &recursive_placement,
        93_896,
        12_108_509,
        12_330_019,
        (1, 257),
        (8_975_795, 12_108_468),
    );
    assert_placement_mutations_rejected(&base_placement);
    assert_placement_mutations_rejected(&recursive_placement);
    assert_canonical_source_loop_width(&recursive_manifest, &recursive_placement);

    let rendered = render(&base_manifest, &recursive_manifest);
    assert_eq!(rendered, render(&base_manifest, &recursive_manifest));
    write_expected_only(&repo_root().join(ARTIFACT_PATH), &rendered, "source-role");
    let placement_rendered = render_ordinary_placement(&base_placement, &recursive_placement);
    assert_eq!(
        placement_rendered,
        render_ordinary_placement(&base_placement, &recursive_placement)
    );
    write_expected_only(
        &repo_root().join(PLACEMENT_ARTIFACT_PATH),
        &placement_rendered,
        "ordinary-placement",
    );

    print_census("base", &base_manifest);
    print_census("recursive", &recursive_manifest);
}

fn assert_ordinary_placement_pointwise(manifest: &GadgetNativeOrdinaryPlacementManifest, plan: &GadgetNativePlan) {
    assert_eq!(manifest.public_input_len(), plan.public_input_len());
    assert_eq!(manifest.encoded_columns(), plan.encoded_cols());
    for index in 0..manifest.placement_count() {
        let placement = manifest.placement(index).expect("ordinary placement");
        assert_eq!(
            plan.source_role_for_column(placement.source_column()),
            Some(GadgetNativeSourceRole::OrdinaryPrivateField)
        );
        assert_eq!(
            plan.encoded_range_for_source_column(placement.source_column()),
            Some(placement.encoded_range()),
            "production materializer/ordinary cursor mismatch at source column {}",
            placement.source_column()
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn assert_placement_branch(
    branch: &str,
    source: &GadgetNativeSourceManifest,
    placement: &GadgetNativeOrdinaryPlacementManifest,
    expected_fields: usize,
    expected_source_phase_end: usize,
    expected_encoded_columns: usize,
    expected_first: (usize, usize),
    expected_last: (usize, usize),
) {
    assert_eq!(placement.source_columns(), source.source_columns());
    assert_eq!(
        placement.placement_count(),
        source.role_count(GadgetNativeSourceRole::OrdinaryPrivateField)
    );
    assert_eq!(placement.placement_count(), expected_fields);
    assert_eq!(
        placement.placement_count() * ORDINARY_PRIVATE_DIGITS,
        expected_fields * 41
    );
    assert_eq!(placement.source_phase_end(), expected_source_phase_end);
    assert_eq!(placement.encoded_columns(), expected_encoded_columns);
    assert!(placement.source_phase_end() <= placement.encoded_columns());
    let first = placement.placement(0).expect("first ordinary placement");
    let last = placement
        .placement(placement.placement_count() - 1)
        .expect("last ordinary placement");
    assert_eq!(
        (first.source_column(), first.encoded_range().start),
        expected_first,
        "{branch} first ordinary placement"
    );
    assert_eq!(
        (last.source_column(), last.encoded_range().start),
        expected_last,
        "{branch} last ordinary placement"
    );
    assert_eq!(first.encoded_range().len(), 41);
    assert_eq!(last.encoded_range().len(), 41);
}

fn assert_placement_mutations_rejected(manifest: &GadgetNativeOrdinaryPlacementManifest) {
    let first = manifest.placement(0).expect("first ordinary placement");
    let mut cases = Vec::new();

    let mut changed = manifest.clone();
    changed.apply_test_mutation(GadgetNativeOrdinaryPlacementManifestTestMutation::PlacementSource {
        placement: 0,
        source_column: first.source_column() + 1,
    });
    cases.push(changed);

    let mut changed = manifest.clone();
    changed.apply_test_mutation(GadgetNativeOrdinaryPlacementManifestTestMutation::PlacementStart {
        placement: 0,
        encoded_start: first.encoded_range().start + 1,
    });
    cases.push(changed);

    let mut changed = manifest.clone();
    changed.apply_test_mutation(GadgetNativeOrdinaryPlacementManifestTestMutation::DropPlacement { placement: 0 });
    cases.push(changed);

    let mut changed = manifest.clone();
    changed.apply_test_mutation(GadgetNativeOrdinaryPlacementManifestTestMutation::DuplicatePlacement { placement: 0 });
    cases.push(changed);

    let mut changed = manifest.clone();
    changed.apply_test_mutation(GadgetNativeOrdinaryPlacementManifestTestMutation::PublicInputLen {
        value: manifest.public_input_len() + 1,
    });
    cases.push(changed);

    let mut changed = manifest.clone();
    changed.apply_test_mutation(GadgetNativeOrdinaryPlacementManifestTestMutation::SourcePhaseEnd {
        value: manifest.source_phase_end() + 1,
    });
    cases.push(changed);

    let mut changed = manifest.clone();
    changed.apply_test_mutation(GadgetNativeOrdinaryPlacementManifestTestMutation::EncodedColumns {
        value: manifest.source_phase_end() - 1,
    });
    cases.push(changed);

    let mut changed = manifest.clone();
    changed.apply_test_mutation(GadgetNativeOrdinaryPlacementManifestTestMutation::SourceRole {
        column: first.source_column(),
        role: GadgetNativeSourceRole::PrivateBoolean,
    });
    cases.push(changed);

    for changed in cases {
        assert!(changed.validate().is_err(), "ordinary-placement mutation must fail");
    }
}

fn assert_canonical_source_loop_width(
    source: &GadgetNativeSourceManifest,
    placement: &GadgetNativeOrdinaryPlacementManifest,
) {
    let canonical = (0..source.run_count())
        .filter_map(|run| source.run(run))
        .filter(|(_, role, _)| *role == GadgetNativeSourceRole::CanonicalU64)
        .flat_map(|(_, _, columns)| columns)
        .collect::<Vec<_>>();
    assert_eq!(canonical.len(), 2, "fixed recursive branch direct canonical-u64 census");
    for column in canonical {
        let range = placement
            .source_loop_allocation_range_for_column(column)
            .expect("direct canonical-u64 source-loop allocation");
        assert_eq!(range.len(), 95, "64 value bits plus 31 prefix auxiliaries");
    }
}

fn assert_materializer_pointwise(manifest: &GadgetNativeSourceManifest, plan: &GadgetNativePlan) {
    for column in 0..manifest.source_columns() {
        assert_eq!(
            plan.source_role_for_column(column),
            manifest.role_for_column(column),
            "shared source schedule/materializer role mismatch at column {column}"
        );
    }
    assert_eq!(plan.source_role_for_column(manifest.source_columns()), None);
}

fn assert_equal_aggregate_swap_detected(manifest: &GadgetNativeSourceManifest, plan: &GadgetNativePlan) {
    let mut by_length = std::collections::BTreeMap::<usize, Vec<usize>>::new();
    for run in 1..manifest.run_count() {
        let (_, _, columns) = manifest.run(run).expect("manifest run");
        by_length.entry(columns.len()).or_default().push(run);
    }
    let mut swapped = None;
    'lengths: for runs in by_length.values() {
        for (offset, &left) in runs.iter().enumerate() {
            let left_role = manifest.run(left).expect("left run").1;
            for &right in &runs[offset + 1..] {
                if manifest.run(right).expect("right run").1 == left_role {
                    continue;
                }
                let mut candidate = manifest.clone();
                candidate.apply_test_mutation(GadgetNativeSourceManifestTestMutation::SwapRunRoles { left, right });
                if candidate.validate().is_ok() {
                    swapped = Some(candidate);
                    break 'lengths;
                }
            }
        }
    }
    let swapped = swapped.expect("two equal-size role runs must admit an aggregate-preserving swap");
    for role in GadgetNativeSourceRole::ALL {
        assert_eq!(
            swapped.role_count(role),
            manifest.role_count(role),
            "equal-size role swap must preserve every aggregate"
        );
    }
    assert!(
        (0..manifest.source_columns())
            .any(|column| plan.source_role_for_column(column) != swapped.role_for_column(column)),
        "pointwise materializer comparison must reject an aggregate-preserving role swap"
    );
}

fn assert_event_roles(manifest: &GadgetNativeSourceManifest, branch: &FullFPrimeBranchExecution) {
    let role = |column| {
        manifest
            .role_for_column(column)
            .unwrap_or_else(|| panic!("missing source role for column {column}"))
    };
    for event in branch.encoding_trace().acceptance_chunks() {
        assert_eq!(role(event.accept.col()), GadgetNativeSourceRole::PrivateBoolean);
        assert_eq!(role(event.inverse.col()), GadgetNativeSourceRole::GadgetTemporary);
    }
    for event in branch.encoding_trace().mod5_chunks() {
        let (high, low) = event
            .quotient_bits
            .split_last()
            .expect("production Mod-5 quotient has a high bit");
        for bit in low {
            assert_eq!(role(bit.col()), GadgetNativeSourceRole::PrivateBoolean);
        }
        for column in [event.index.col(), event.quotient.col(), high.col()] {
            assert_eq!(role(column), GadgetNativeSourceRole::LinearlyDerived);
        }
        for product in event.index_products {
            assert_eq!(role(product.col()), GadgetNativeSourceRole::ProductDerived);
        }
    }
    for event in branch.encoding_trace().canonical_u64_decompositions() {
        assert!(matches!(
            role(event.field.col()),
            GadgetNativeSourceRole::CanonicalU64 | GadgetNativeSourceRole::LinearlyDerived
        ));
    }
    assert_eq!(
        manifest.canonical_u64_overlap().0,
        branch.encoding_trace().canonical_u64_decompositions().len()
    );
}

fn assert_mutations_rejected(manifest: &GadgetNativeSourceManifest) {
    assert!(manifest.run_count() > 1);
    let mut cases = Vec::new();

    let mut changed = manifest.clone();
    changed.apply_test_mutation(GadgetNativeSourceManifestTestMutation::RunStart { run: 1, start: 0 });
    cases.push(changed);

    let mut changed = manifest.clone();
    changed.apply_test_mutation(GadgetNativeSourceManifestTestMutation::RunLength { run: 1, length: 0 });
    cases.push(changed);

    let mut changed = manifest.clone();
    changed.apply_test_mutation(GadgetNativeSourceManifestTestMutation::RunRole {
        run: 1,
        role: GadgetNativeSourceRole::ConstantOne,
    });
    cases.push(changed);

    let mut changed = manifest.clone();
    changed.apply_test_mutation(GadgetNativeSourceManifestTestMutation::DropRun { run: 1 });
    cases.push(changed);

    let mut changed = manifest.clone();
    changed.apply_test_mutation(GadgetNativeSourceManifestTestMutation::RoleTotal {
        role: GadgetNativeSourceRole::OrdinaryPrivateField,
        count: manifest.role_count(GadgetNativeSourceRole::OrdinaryPrivateField) + 1,
    });
    cases.push(changed);

    let mut changed = manifest.clone();
    changed.apply_test_mutation(GadgetNativeSourceManifestTestMutation::CanonicalTraced {
        count: manifest.canonical_u64_overlap().0 + 1,
    });
    cases.push(changed);

    for changed in cases {
        assert!(changed.validate().is_err(), "source-role mutation must fail");
    }
}

fn render(base: &GadgetNativeSourceManifest, recursive: &GadgetNativeSourceManifest) -> String {
    let mut out = String::new();
    out.push_str(
        "import Nightstream.Implementation.R1CS.Correspondence.FieldEncoding.PackedSourceCensus\n\n\
/-! Generated by `f_prime_full_relation`; do not hand-edit.\n\n\
Owns: exact source-only role runs for the fixed F-prime base and recursive\n\
gadget-native branches.\n\n\
Does not own: encoded-coordinate or CE-coordinate runs, selector composition,\n\
row removal, or authority for an encoding change.\n\n\
Emits constraints: no.\n\n\
Authority boundary: the source R1CS and validated production trace are\n\
authoritative. These data are diagnostic source-role census evidence only.\n\n",
    );
    writeln!(
        out,
        "| Branch | Source columns | Runs | Ordinary eligible | Canonical-u64 traced/linear/direct |"
    )
    .expect("render table");
    out.push_str("|---|---:|---:|---:|---:|\n");
    for (name, manifest) in [("base", base), ("recursive", recursive)] {
        let canonical = manifest.canonical_u64_overlap();
        writeln!(
            out,
            "| {name} | {} | {} | {} | {}/{}/{} |",
            manifest.source_columns(),
            manifest.run_count(),
            manifest.role_count(GadgetNativeSourceRole::OrdinaryPrivateField),
            canonical.0,
            canonical.1,
            canonical.2,
        )
        .expect("render table row");
    }
    out.push_str(
        "\
| Packed field | Exact decoder |\n\
|---|---|\n\
| stage | `packed % stagePaths.size` |\n\
| role | `(packed / stagePaths.size) % PackedSourceCensus.slotRoleCount` |\n\
| length | `(packed / stagePaths.size) / PackedSourceCensus.slotRoleCount` |\n\
| chunks | comma-decimal runs; non-final chunks target 10 KiB and never exceed 12 KiB |\n\
-/\n\nnamespace Nightstream.Implementation.R1CS.FPrimeBranchSourceRoleManifestData\n\n",
    );
    out.push_str("open Nightstream.Implementation.R1CS.FPrimeFieldLayout\n\n");
    append_branch(&mut out, "base", base);
    append_branch(&mut out, "recursive", recursive);
    out.push_str("end Nightstream.Implementation.R1CS.FPrimeBranchSourceRoleManifestData\n");
    out
}

fn render_ordinary_placement(
    base: &GadgetNativeOrdinaryPlacementManifest,
    recursive: &GadgetNativeOrdinaryPlacementManifest,
) -> String {
    let mut out = String::new();
    out.push_str(
        "import Nightstream.Implementation.R1CS.Correspondence.FieldEncoding.OrdinaryPlacement\n\n\
/-! Generated by `f_prime_full_relation`; do not hand-edit.\n\n\
Owns: compact source-allocation phase and final encoded-column metadata for\n\
the fixed F-prime base and recursive gadget-native branches.\n\n\
Does not own: source roles, per-field starts, deferred allocation details, CE\n\
coordinates, chosen centered words, row removal, or lifecycle authority.\n\n\
Emits constraints: no.\n\n\
Authority boundary: Lean derives every ordinary-private start from the\n\
separately checked source-role runs and fixed allocation widths. These two\n\
branch summaries are non-authoritative Rust drift evidence.\n\n\
| Branch | Source phase end | Final encoded columns |\n\
|---|---:|---:|\n",
    );
    for (name, manifest) in [("base", base), ("recursive", recursive)] {
        writeln!(
            out,
            "| {name} | {} | {} |",
            manifest.source_phase_end(),
            manifest.encoded_columns()
        )
        .expect("render ordinary placement table");
    }
    out.push_str(
        "-/\n\nnamespace Nightstream.Implementation.R1CS.FPrimeBranchOrdinaryPlacementData\n\n\
open Nightstream.Implementation.R1CS.FPrimeFieldLayout.OrdinaryPlacement\n\n",
    );
    append_ordinary_placement_branch(&mut out, "base", base);
    append_ordinary_placement_branch(&mut out, "recursive", recursive);
    out.push_str("end Nightstream.Implementation.R1CS.FPrimeBranchOrdinaryPlacementData\n");
    out
}

fn append_ordinary_placement_branch(out: &mut String, name: &str, manifest: &GadgetNativeOrdinaryPlacementManifest) {
    writeln!(out, "def {name}Data : Metadata where").expect("render ordinary placement metadata");
    out.push_str("  formatVersion := 1\n");
    writeln!(out, "  sourcePhaseEnd := {}", manifest.source_phase_end()).expect("render source phase end");
    writeln!(out, "  encodedColumnCount := {}\n", manifest.encoded_columns()).expect("render encoded column count");
}

fn append_branch(out: &mut String, name: &str, manifest: &GadgetNativeSourceManifest) {
    let canonical = manifest.canonical_u64_overlap();
    writeln!(out, "def {name}CanonicalU64Traced : Nat := {}", canonical.0).expect("render count");
    writeln!(out, "def {name}CanonicalU64Linear : Nat := {}", canonical.1).expect("render count");
    writeln!(out, "def {name}CanonicalU64Direct : Nat := {}", canonical.2).expect("render count");
    let balanced = manifest.balanced_source_census();
    writeln!(out, "def {name}BalancedOpeningFields : Nat := {}", balanced.0).expect("render count");
    writeln!(out, "def {name}BalancedDigitAliases : Nat := {}", balanced.1).expect("render count");
    writeln!(out, "def {name}BalancedBinaryColumns : Nat := {}", balanced.2).expect("render count");
    writeln!(out, "def {name}RoleCounts : PackedSourceCensus.RoleCounts where").expect("render role counts");
    for role in GadgetNativeSourceRole::ALL {
        writeln!(
            out,
            "  {} := {}",
            role.lean_role_count_field(),
            manifest.role_count(role)
        )
        .expect("render role count");
    }
    let stage_paths = (0..manifest.run_count())
        .map(|index| manifest.run(index).expect("manifest run").0)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let mut packed_runs = Vec::with_capacity(manifest.run_count());
    for index in 0..manifest.run_count() {
        let (stage, role, columns) = manifest.run(index).expect("manifest run");
        let stage_index = stage_paths
            .binary_search(&stage)
            .expect("sorted stage path must contain every run stage");
        let role_index = GadgetNativeSourceRole::ALL
            .iter()
            .position(|candidate| *candidate == role)
            .expect("role order must be exhaustive");
        let packed = columns
            .len()
            .checked_mul(GadgetNativeSourceRole::ALL.len())
            .and_then(|value| value.checked_add(role_index))
            .and_then(|value| value.checked_mul(stage_paths.len()))
            .and_then(|value| value.checked_add(stage_index))
            .expect("packed source run");
        packed_runs.push(packed);
        assert_eq!(
            unpack_run(packed, &stage_paths),
            (stage, role, columns.len()),
            "packed source run must round trip"
        );
    }
    let chunks = packed_run_chunks(&packed_runs);
    writeln!(out, "def {name}Data : PackedSourceCensus.Data where").expect("render data");
    // Pin the producer format literally. A Lean decoder-version bump must
    // reject this artifact until Rust deliberately regenerates new bytes.
    out.push_str("  formatVersion := 1\n");
    writeln!(out, "  sourceColumnCount := {}", manifest.source_columns()).expect("render source count");
    writeln!(out, "  runCount := {}", manifest.run_count()).expect("render run count");
    writeln!(out, "  declaredRoleCounts := {name}RoleCounts").expect("render role counts reference");
    out.push_str("  stagePaths := #[");
    for (index, stage) in stage_paths.iter().enumerate() {
        if index != 0 {
            out.push_str(", ");
        }
        write!(out, "{stage:?}").expect("render stage");
    }
    out.push_str("]\n");
    out.push_str("  packedChunks :=\n    [\n");
    for (index, chunk) in chunks.iter().enumerate() {
        write!(out, "      {chunk:?}").expect("render packed chunk");
        if index + 1 != chunks.len() {
            out.push(',');
        }
        out.push('\n');
    }
    out.push_str("    ]\n\n");
}

fn packed_run_chunks(packed_runs: &[usize]) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut current = String::new();
    for packed in packed_runs {
        let value = packed.to_string();
        let separator = usize::from(!current.is_empty());
        if !current.is_empty() && current.len() + separator + value.len() > PACKED_CHUNK_TARGET_BYTES {
            assert!(current.len() <= PACKED_CHUNK_MAX_BYTES);
            chunks.push(std::mem::take(&mut current));
        }
        if !current.is_empty() {
            current.push(',');
        }
        current.push_str(&value);
        assert!(current.len() <= PACKED_CHUNK_MAX_BYTES);
    }
    if !current.is_empty() {
        chunks.push(current);
    }
    assert!(!chunks.is_empty());
    assert!(chunks
        .iter()
        .all(|chunk| chunk.len() <= PACKED_CHUNK_MAX_BYTES));
    chunks
}

fn unpack_run(packed: usize, stage_paths: &[&'static str]) -> (&'static str, GadgetNativeSourceRole, usize) {
    let stage = packed % stage_paths.len();
    let role_and_length = packed / stage_paths.len();
    let role = role_and_length % GadgetNativeSourceRole::ALL.len();
    let length = role_and_length / GadgetNativeSourceRole::ALL.len();
    (stage_paths[stage], GadgetNativeSourceRole::ALL[role], length)
}

fn write_expected_only(path: &Path, rendered: &str, family: &str) {
    if fs::read_to_string(path).ok().as_deref() == Some(rendered) {
        return;
    }
    let expected = PathBuf::from(format!("{}.expected", path.display()));
    fs::write(&expected, rendered).unwrap_or_else(|error| panic!("write {}: {error}", expected.display()));
    if path.exists() {
        panic!(
            "{family} Lean artifact drifted; inspect {} and promote intentionally",
            expected.display()
        );
    }
    panic!(
        "{family} Lean artifact is missing; inspect {} and promote intentionally",
        expected.display()
    );
}

fn print_census(branch: &str, manifest: &GadgetNativeSourceManifest) {
    eprintln!(
        "SOURCE_ROLE|{branch}|columns={}|runs={}|ordinary={}|canonical_u64={:?}|balanced={:?}",
        manifest.source_columns(),
        manifest.run_count(),
        manifest.role_count(GadgetNativeSourceRole::OrdinaryPrivateField),
        manifest.canonical_u64_overlap(),
        manifest.balanced_source_census(),
    );
    for role in GadgetNativeSourceRole::ALL {
        eprintln!("SOURCE_ROLE_COUNT|{branch}|{:?}|{}", role, manifest.role_count(role));
    }
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("workspace root")
        .to_path_buf()
}
