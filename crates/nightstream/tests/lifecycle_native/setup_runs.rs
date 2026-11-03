//! Writes the runs of the production layer-1 key for the Lean `setupCheck` and `runsCheck`
//! (`runs_file`).

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use neo_spartan::Key;

use super::staged::prepare;
use crate::lifecycle::runs_file::{self, RunsFile};

fn workspace() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// The layer-1 key of the committed production key, read as `neo-spartan`'s
/// `production_key` reads it.
fn committed_layer1() -> Key {
    let bytes = std::fs::read(workspace().join("crates/nightstream/tests/fixtures/compression/key.bin")).unwrap();
    let (key, ..): (Key, [u64; 4], u32, f64, neo_spartan::shrink::Shape) = bincode::deserialize(&bytes).unwrap();
    key
}

#[test]
#[ignore = "Writes about 1 GB under target/lean-lowering; run this test separately under the 300-second cap."]
fn production_runs_rebuild_the_layer1_key() {
    let dir = workspace().join("target/lean-lowering");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("production_runs.bin");
    {
        let package = prepare();
        let started = Instant::now();
        let count = runs_file::write(&path, &package.matrix_rows());
        eprintln!("{count} runs written in {:?}", started.elapsed());
    }
    let started = Instant::now();
    let key = Key::derive(&RunsFile::read(&path)).unwrap();
    eprintln!("key derived from the file in {:?}", started.elapsed());
    assert!(
        key == committed_layer1(),
        "the runs in the file rebuild the committed production layer-1 key"
    );
}

#[test]
#[ignore = "needs production_runs.bin of production_runs_rebuild_the_layer1_key, the production_layer1.json of \
            neo-spartan's production_layer1_matches_lean, the production_setup_columns.bin of neo-spartan's \
            production_setup_columns, and lake build setupCheck in formal/compression-lean"]
fn production_setup_matches_lean() {
    let dir = workspace().join("target/lean-lowering");
    let layer1 = dir.join("production_layer1.json");
    let runs = dir.join("production_runs.bin");
    let columns = dir.join("production_setup_columns.bin");
    assert!(
        layer1.exists(),
        "run neo-spartan's production_layer1_matches_lean first"
    );
    assert!(runs.exists(), "run production_runs_rebuild_the_layer1_key first");
    assert!(columns.exists(), "run neo-spartan's production_setup_columns first");
    let check = workspace().join("formal/compression-lean/.lake/build/bin/setupCheck");
    assert!(
        check.exists(),
        "build the check with `lake build setupCheck` in formal/compression-lean"
    );
    let started = Instant::now();
    let output = Command::new(&check)
        .arg(&layer1)
        .arg(&runs)
        .arg(&columns)
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    eprintln!("setupCheck: {} in {:?}", stdout.trim_end(), started.elapsed());
    assert!(
        output.status.success() && stdout.contains("setup facts hold") && stdout.contains("setup columns match"),
        "{stdout}{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
#[ignore = "needs production_runs.bin of production_runs_rebuild_the_layer1_key, and lake build runsCheck in \
            formal/compression-lean"]
fn production_runs_match_lean() {
    let runs = workspace().join("target/lean-lowering/production_runs.bin");
    assert!(runs.exists(), "run production_runs_rebuild_the_layer1_key first");
    let check = workspace().join("formal/compression-lean/.lake/build/bin/runsCheck");
    assert!(
        check.exists(),
        "build the check with `lake build runsCheck` in formal/compression-lean"
    );
    let started = Instant::now();
    let output = Command::new(&check).arg(&runs).output().unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    eprintln!("runsCheck: {} in {:?}", stdout.trim_end(), started.elapsed());
    assert!(
        output.status.success() && stdout.contains("runs give the matrices"),
        "{stdout}{}",
        String::from_utf8_lossy(&output.stderr)
    );
}
