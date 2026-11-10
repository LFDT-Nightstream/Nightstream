//! Both bounded-depth layer-1 relations' runs and setups against Lean (`formal/compression-lean`).
//!
//! `bounded_depth_runs_rebuild_the_layer1_keys` writes each relation's runs (`runs_file`) and requires
//! `Key::derive` of the file to be the slot's layer-1 key. `bounded_depth_runs_match_lean` requires
//! `runsCheck` to find the runs to be the matrices of the slot's circuit key, read from
//! its step package (`Spec.BoundedDepth.keyRunsFailure`). `neo-spartan`'s `bounded_depth_setup_columns`
//! writes the setup columns from the runs, and `bounded_depth_setup_matches_lean` requires
//! `setupCheck` to accept each slot over its key's own columns (`keyColumns`).

use std::process::Command;
use std::time::Instant;

use neo_spartan::Key;

use super::final_ir::{lowering, workspace};
use super::prover;
use crate::lifecycle::runs_file::{self, RunsFile};

#[test]
#[ignore = "writes both relations' runs under target/lean-lowering; run it alone under the 300-second cap"]
fn bounded_depth_runs_rebuild_the_layer1_keys() {
    let prover = prover(2);
    let keys = prover.compression_keys().expect("compression keys");
    for slot in 0..2 {
        let path = lowering(&format!("bounded_depth_runs_{slot}.bin"));
        let started = Instant::now();
        let count = runs_file::write(&path, &prover.relations[slot]);
        eprintln!("slot {slot}: {count} runs written in {:?}", started.elapsed());
        let key = Key::derive(&RunsFile::read(&path)).unwrap();
        assert!(key == keys.layer1[slot], "slot {slot}'s runs rebuild its layer-1 key");
    }
}

#[test]
#[ignore = "needs bounded_depth_runs_{0,1}.bin of bounded_depth_runs_rebuild_the_layer1_keys, and lake build runsCheck in \
            formal/compression-lean"]
fn bounded_depth_runs_match_lean() {
    let check = workspace().join("formal/compression-lean/.lake/build/bin/runsCheck");
    assert!(
        check.exists(),
        "build the check with `lake build runsCheck` in formal/compression-lean"
    );
    for (slot, circuit) in ["light", "refresh"].into_iter().enumerate() {
        let package = workspace()
            .join("formal/nightstream-fprime/artifacts")
            .join(format!("nightstream-fprime-bounded-depth-{circuit}-v1.json"));
        let runs = lowering(&format!("bounded_depth_runs_{slot}.bin"));
        assert!(runs.exists(), "run bounded_depth_runs_rebuild_the_layer1_keys first");
        let started = Instant::now();
        let output = Command::new(&check)
            .arg(&package)
            .arg(&runs)
            .output()
            .unwrap();
        let stdout = String::from_utf8_lossy(&output.stdout);
        eprintln!(
            "slot {slot}: runsCheck: {} in {:?}",
            stdout.trim_end(),
            started.elapsed()
        );
        assert!(
            output.status.success() && stdout.contains("runs give the matrices"),
            "{stdout}{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

#[test]
#[ignore = "needs bounded_depth_runs_{0,1}.bin of bounded_depth_runs_rebuild_the_layer1_keys, bounded_depth_layer1_{0,1}.json \
            of neo-spartan's bounded_depth_layer1_matches_lean, bounded_depth_setup_columns_{0,1}.bin of neo-spartan's \
            bounded_depth_setup_columns, and lake build setupCheck in formal/compression-lean"]
fn bounded_depth_setup_matches_lean() {
    let check = workspace().join("formal/compression-lean/.lake/build/bin/setupCheck");
    assert!(
        check.exists(),
        "build the check with `lake build setupCheck` in formal/compression-lean"
    );
    for slot in 0..2 {
        let files = [
            format!("bounded_depth_layer1_{slot}.json"),
            format!("bounded_depth_runs_{slot}.bin"),
            format!("bounded_depth_setup_columns_{slot}.bin"),
        ]
        .map(|name| lowering(&name));
        for file in &files {
            assert!(file.exists(), "write {} first", file.display());
        }
        let started = Instant::now();
        let output = Command::new(&check).args(&files).output().unwrap();
        let stdout = String::from_utf8_lossy(&output.stdout);
        eprintln!(
            "slot {slot}: setupCheck: {} in {:?}",
            stdout.trim_end(),
            started.elapsed()
        );
        assert!(
            output.status.success() && stdout.contains("setup facts hold") && stdout.contains("setup columns match"),
            "{stdout}{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}
