//! Drift gate for the bounded production combined-NC selective-row artifact.
//!
//! Owns construction of the stabilized delayed block-by-lane NC audit,
//! coefficient-level comparison of all 25 production SumCheck rounds with
//! the isolated five-coefficient `enforce_sumcheck_round` relation, and
//! fail-closed rendering
//! into the dedicated CombinedNc generated tree.
//!
//! Does not own transcript authority, commitment binding, delayed-state
//! continuity, Lean row semantics, security bounds, or row-removal authority.

#[path = "../support/mod.rs"]
mod support;

#[path = "block_lane_nc_lean_artifact/mod.rs"]
mod artifact;

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use neo_fold_clean::engine::r1cs_circuit::builder::SumcheckRoundAudit;
use neo_fold_clean::engine::r1cs_circuit::{enforce_sumcheck_round, KVar, R1csBuilder};
use neo_fold_clean::frontends::r1cs_f_prime::ivc::{R1csIvcBlockLaneNcSelectiveRowsAudit, R1csIvcRelation};
use neo_math::F;
use p3_field::PrimeCharacteristicRing;
use support::r1cs_compiler_fixtures::{make_tiny_lifecycle_plan, one_product_r1cs, tiny_params};

use artifact::{GeneratedLeanFile, TinyFixtureScope};

const GENERATED_DIRECTORY: &str =
    "formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeSelectiveFixedPoint/PiCcsNc/DelayedProjection/CombinedNc/Generated";
const ROUND_COEFFICIENTS: usize = 5;
const ISOLATED_ROUND_COLUMNS: usize = 43;
const ISOLATED_ROUND_ROWS: usize = 30;
const ISOLATED_ROUND_ALLOCATED_COLUMNS: usize = 28;
const ROUND_COUNT: usize = 25;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("workspace root")
        .to_path_buf()
}

fn compare_or_write_expected(root: &Path, file: GeneratedLeanFile, drifted: &mut Vec<String>) {
    let path = root.join(file.relative_path);
    let committed = fs::read_to_string(&path).unwrap_or_default();
    if committed == file.contents {
        return;
    }

    let expected = path.with_extension("lean.expected");
    fs::create_dir_all(expected.parent().expect("generated artifact parent"))
        .expect("create generated artifact directory");
    fs::write(&expected, file.contents).expect("write generated Lean candidate");
    drifted.push(
        expected
            .strip_prefix(root)
            .unwrap_or(&expected)
            .display()
            .to_string(),
    );
}

fn committed_lean_files(directory: &Path, root: &Path, files: &mut BTreeSet<String>) {
    if !directory.exists() {
        return;
    }
    for entry in fs::read_dir(directory).expect("read generated artifact directory") {
        let path = entry.expect("read generated artifact entry").path();
        if path.is_dir() {
            committed_lean_files(&path, root, files);
        } else if path
            .extension()
            .is_some_and(|extension| extension == "lean")
        {
            files.insert(
                path.strip_prefix(root)
                    .unwrap_or(&path)
                    .display()
                    .to_string(),
            );
        }
    }
}

fn expected_round_column_map(round: &SumcheckRoundAudit) -> Result<[usize; ISOLATED_ROUND_COLUMNS], String> {
    if round.coefficient_cols.len() != ROUND_COEFFICIENTS
        || round.allocated_cols.len() != ISOLATED_ROUND_ALLOCATED_COLUMNS
    {
        return Err(format!(
            "round does not have {ROUND_COEFFICIENTS} K coefficients and {ISOLATED_ROUND_ALLOCATED_COLUMNS} allocated columns"
        ));
    }
    let mut map = [usize::MAX; ISOLATED_ROUND_COLUMNS];
    map[0] = 0;
    map[1] = round.claim_in_cols[0];
    map[ROUND_COEFFICIENTS + 2] = round.claim_in_cols[1];
    for (index, pair) in round.coefficient_cols.iter().enumerate() {
        map[2 + index] = pair[0];
        map[ROUND_COEFFICIENTS + 3 + index] = pair[1];
    }
    map[2 * ROUND_COEFFICIENTS + 3] = round.challenge_cols[0];
    map[2 * ROUND_COEFFICIENTS + 5] = round.challenge_cols[1];
    map[2 * ROUND_COEFFICIENTS + 4] = round.allocated_cols[0];
    for (local, &source) in ((2 * ROUND_COEFFICIENTS + 6)..ISOLATED_ROUND_COLUMNS).zip(&round.allocated_cols[1..]) {
        map[local] = source;
    }
    Ok(map)
}

fn validate_round_column_map(round: &SumcheckRoundAudit, map: &[usize; ISOLATED_ROUND_COLUMNS]) -> Result<(), String> {
    if &expected_round_column_map(round)? != map {
        return Err("round map differs from the exact named wire schedule".to_owned());
    }
    if round.claim_out_cols != [map[ISOLATED_ROUND_COLUMNS - 2], map[ISOLATED_ROUND_COLUMNS - 1]] {
        return Err(format!(
            "round output does not occupy isolated columns {} and {}",
            ISOLATED_ROUND_COLUMNS - 2,
            ISOLATED_ROUND_COLUMNS - 1
        ));
    }
    if map.iter().copied().collect::<BTreeSet<_>>().len() != map.len() {
        return Err("round map is not injective".to_owned());
    }
    Ok(())
}

fn isolated_round() -> (
    neo_fold_clean::engine::r1cs_circuit::R1csSnapshot,
    [usize; ISOLATED_ROUND_COLUMNS],
) {
    let mut builder = R1csBuilder::new();
    let coefficients = (0..ROUND_COEFFICIENTS)
        .map(|_| KVar::alloc(&mut builder, F::ZERO, F::ZERO))
        .collect::<Vec<_>>();
    let challenge = KVar::alloc(&mut builder, F::ZERO, F::ZERO);
    let claim_in = KVar::alloc(&mut builder, F::ZERO, F::ZERO);
    let row_start = builder.rows();
    let _claim_out = enforce_sumcheck_round(&mut builder, &coefficients, challenge, claim_in);
    let [round] = builder.sumcheck_round_audits() else {
        panic!("isolated builder must record exactly one SumCheck round")
    };
    assert_eq!(row_start, 0, "isolated round starts at row zero");
    assert_eq!(round.row_end - round.row_start, ISOLATED_ROUND_ROWS);
    let local_to_isolated = expected_round_column_map(round).expect("isolated round wire map");
    assert_eq!(builder.cols(), ISOLATED_ROUND_COLUMNS, "isolated round column count");
    (builder.snapshot(), local_to_isolated)
}

fn remap_isolated_terms(
    terms: &[(usize, F)],
    isolated_to_local: &BTreeMap<usize, usize>,
    local_to_source: &[usize; ISOLATED_ROUND_COLUMNS],
) -> Vec<(usize, F)> {
    let mut remapped = terms
        .iter()
        .map(|&(isolated, coefficient)| {
            let local = isolated_to_local
                .get(&isolated)
                .copied()
                .expect("isolated row references a mapped column");
            (local_to_source[local], coefficient)
        })
        .collect::<Vec<_>>();
    remapped.sort_unstable_by_key(|term| term.0);
    remapped
}

fn assert_round_rows_match_isolated(audit: &R1csIvcBlockLaneNcSelectiveRowsAudit) {
    assert_eq!(audit.rounds().len(), ROUND_COUNT, "production round count");
    assert_eq!(audit.round_column_maps().len(), ROUND_COUNT, "round-map count");

    let (isolated, local_to_isolated) = isolated_round();
    let isolated_to_local = local_to_isolated
        .iter()
        .copied()
        .enumerate()
        .map(|(local, isolated)| (isolated, local))
        .collect::<BTreeMap<_, _>>();
    assert_eq!(
        isolated_to_local.len(),
        ISOLATED_ROUND_COLUMNS,
        "isolated map injectivity"
    );
    let source_rows = audit
        .source_row_artifacts()
        .iter()
        .map(|row| (row.index(), row))
        .collect::<BTreeMap<_, _>>();

    for (round_index, (round, map)) in audit
        .rounds()
        .iter()
        .zip(audit.round_column_maps())
        .enumerate()
    {
        validate_round_column_map(round, map).unwrap_or_else(|error| panic!("round {round_index} column map: {error}"));
        assert_eq!(round.row_end - round.row_start, isolated.rows());
        for offset in 0..isolated.rows() {
            let source = source_rows
                .get(&(round.row_start + offset))
                .unwrap_or_else(|| panic!("round {round_index} source row {offset} is exported"));
            assert_eq!(
                source.a(),
                remap_isolated_terms(isolated.a_row(offset), &isolated_to_local, map),
                "round {round_index} A row {offset}"
            );
            assert_eq!(
                source.b(),
                remap_isolated_terms(isolated.b_row(offset), &isolated_to_local, map),
                "round {round_index} B row {offset}"
            );
            assert_eq!(
                source.c(),
                remap_isolated_terms(isolated.c_row(offset), &isolated_to_local, map),
                "round {round_index} C row {offset}"
            );
        }
    }

    let mut mutated = audit.round_column_maps()[0];
    mutated.swap(1, 2);
    assert!(
        validate_round_column_map(&audit.rounds()[0], &mutated).is_err(),
        "one changed local/source association must fail closed"
    );
}

fn render_generated_files() -> Vec<GeneratedLeanFile> {
    let params = tiny_params();
    let app = one_product_r1cs();
    let fixture = TinyFixtureScope {
        parameter_constraint_count: params.m() as usize,
        commitment_width: params.kappa() as usize,
        security_bits: params.lambda() as usize,
        application_row_count: app.n(),
        application_column_count: app.m(),
        application_public_input_count: app.m_in,
    };
    let plan = make_tiny_lifecycle_plan(app.m(), app.m_in);
    let audit = R1csIvcRelation::audit_fixed_point_block_lane_nc_rows(&params, &app.into(), &plan)
        .expect("materialize bounded production combined-NC rows");
    assert_round_rows_match_isolated(&audit);
    artifact::generated_files(&audit, fixture)
}

#[test]
fn production_block_lane_nc_artifact_matches_generated_certificate() {
    let files = render_generated_files();
    assert!(!files.is_empty(), "combined-NC renderer emits a nonempty certificate");

    let root = repo_root();
    let expected_paths = files
        .iter()
        .map(|file| file.relative_path.clone())
        .collect::<BTreeSet<_>>();
    assert_eq!(expected_paths.len(), files.len(), "generated paths are unique");
    let mut drifted = Vec::new();
    for file in files {
        compare_or_write_expected(&root, file, &mut drifted);
    }
    let mut committed_paths = BTreeSet::new();
    committed_lean_files(&root.join(GENERATED_DIRECTORY), &root, &mut committed_paths);
    for stale in committed_paths.difference(&expected_paths) {
        drifted.push(format!("stale generated module: {stale}"));
    }
    if !drifted.is_empty() {
        let preview = drifted.iter().take(8).collect::<Vec<_>>();
        panic!(
            "production combined-NC Lean artifacts drifted: {} candidates; inspect and deliberately promote every `.lean.expected` file (first paths: {preview:?})",
            drifted.len()
        );
    }
}

#[test]
#[ignore = "deliberately rewrites reviewed generated Lean artifacts; the ordinary drift test remains fail-closed"]
fn regenerate_production_block_lane_nc_artifacts() {
    let files = render_generated_files();
    let root = repo_root();
    let expected_paths = files
        .iter()
        .map(|file| file.relative_path.clone())
        .collect::<BTreeSet<_>>();
    let mut committed_paths = BTreeSet::new();
    committed_lean_files(&root.join(GENERATED_DIRECTORY), &root, &mut committed_paths);
    for stale in committed_paths.difference(&expected_paths) {
        fs::remove_file(root.join(stale)).expect("remove stale generated Lean artifact");
    }
    for file in files {
        let path = root.join(file.relative_path);
        fs::create_dir_all(path.parent().expect("generated artifact parent"))
            .expect("create generated artifact directory");
        fs::write(&path, file.contents).expect("write generated Lean artifact");
        let expected = path.with_extension("lean.expected");
        if expected.exists() {
            fs::remove_file(expected).expect("remove promoted generated candidate");
        }
    }
}
