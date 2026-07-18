//! Drift gate for the bounded tiny-lifecycle 15-source/13-matrix PiRLC
//! `y_zcol` artifact.
//!
//! Owns: construction of the stabilized selective fixed-point audit, exact
//! serialization of its retained projection certificate, and fail-closed
//! comparison against every generated Lean payload and sparse-row shard.
//!
//! Does not own: the Lean interpretation of those bytes, PiCCS source truth,
//! transcript authority, source-to-selective lowering, security bounds, cost
//! claims, or permission to remove constraints.
//!
//! Emits constraints: no.
//!
//! | Branch | Exported evidence | Trust boundary |
//! |---|---|---|
//! | metadata | exact tiny-fixture scope, trace coordinates, raw producer indices | artifact-checked only |
//! | stage paths | exact 14-leaf Rust vocabulary | source ownership only |
//! | selective rows | 139 source-to-emitted interval fragments | compiler ownership only |
//! | 12 row shards | 5,724 indexed normalized A/B/C rows | selected source-R1CS rows |

#[path = "../support/mod.rs"]
mod support;

#[path = "selective_fixed_point_projection_lean_artifact/mod.rs"]
mod artifact;

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

use neo_fold_clean::frontends::r1cs_f_prime::ivc::R1csIvcRelation;
use support::r1cs_compiler_fixtures::{make_tiny_lifecycle_plan, one_product_r1cs, tiny_params};

use artifact::{GeneratedLeanFile, TinyFixtureScope};

const GENERATED_DIRECTORY: &str = "formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeSelectiveFixedPoint/PiRlcProjection/YZcol/Generated";

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

#[test]
fn active_selective_fixed_point_projection_artifact_matches_retained_certificate() {
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
    let fixed_point = R1csIvcRelation::audit_fixed_point_shape(&params, &app.into(), &plan)
        .expect("audit active selective fixed point");
    let projection = fixed_point.pi_ccs_output_digest().y_zcol_projection();

    let files = artifact::generated_files(projection, fixture);
    assert_eq!(
        files.len(),
        15,
        "metadata, stage paths, selective intervals, and 12 exact row shards"
    );

    let root = repo_root();
    let expected_paths = files
        .iter()
        .map(|file| file.relative_path.clone())
        .collect::<BTreeSet<_>>();
    let mut drifted = Vec::new();
    for file in files {
        compare_or_write_expected(&root, file, &mut drifted);
    }
    let mut committed_paths = BTreeSet::new();
    committed_lean_files(&root.join(GENERATED_DIRECTORY), &root, &mut committed_paths);
    for stale in committed_paths.difference(&expected_paths) {
        drifted.push(format!("stale generated module: {stale}"));
    }
    assert!(
        drifted.is_empty(),
        "active selective fixed-point projection artifacts drifted; inspect and deliberately promote every generated `.lean.expected` candidate: {drifted:?}"
    );
}
