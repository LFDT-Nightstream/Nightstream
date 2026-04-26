use std::fs;
use std::path::{Path, PathBuf};

fn workspace_root() -> PathBuf {
    // CARGO_MANIFEST_DIR = .../crates/neo-reductions
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest_dir
        .parent()
        .and_then(Path::parent)
        .expect("workspace root")
        .to_path_buf()
}

fn collect_rs_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let entries = fs::read_dir(dir).unwrap_or_else(|e| panic!("read_dir failed for {}: {e}", dir.display()));
    for entry in entries {
        let entry = entry.unwrap_or_else(|e| panic!("read_dir entry failed under {}: {e}", dir.display()));
        let path = entry.path();
        if path.is_dir() {
            collect_rs_files(&path, out);
            continue;
        }
        if path.extension().and_then(|s| s.to_str()) == Some("rs") {
            out.push(path);
        }
    }
}

fn assert_absent_tokens_in_src(crate_rel_path: &str, banned: &[&str]) {
    let root = workspace_root();
    let src_dir = root.join(crate_rel_path).join("src");
    let mut files = Vec::new();
    collect_rs_files(&src_dir, &mut files);

    let mut hits: Vec<String> = Vec::new();
    for file in files {
        let text = fs::read_to_string(&file).unwrap_or_else(|e| panic!("failed to read {}: {e}", file.display()));
        for token in banned {
            if text.contains(token) {
                hits.push(format!("{} contains `{token}`", file.display()));
            }
        }
    }

    assert!(
        hits.is_empty(),
        "legacy identifiers found in {} src:\n{}",
        crate_rel_path,
        hits.join("\n")
    );
}

#[test]
fn legacy_me_mcs_identifiers_do_not_reappear_in_core_src() {
    let banned = [
        "MeInstance",
        "McsInstance",
        "MeWitness",
        "McsWitness",
        "DenseUnpacked",
        "WitnessMatLayout",
        "witness_mat_layout",
        "project_x_from_public_inputs",
    ];
    assert_absent_tokens_in_src("crates/neo-ccs", &banned);
    assert_absent_tokens_in_src("crates/neo-reductions", &banned);
    assert_absent_tokens_in_src("crates/neo-fold-next", &banned);
}
