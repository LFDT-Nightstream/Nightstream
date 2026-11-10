//! The bounded-depth scheme's final program against Lean's copy (`formal/compression-lean`,
//! `Ports/BoundedDepth/Program.lean`): the shape run of `FinalProgram::run` on the
//! finish package, the state keys and both slots' layer-1 relations, one line
//! per `Backend` call (`ir_log`).
//!
//! `bounded_depth_layer1_keys_are_written` writes both layer-1 keys, from which
//! `neo-spartan`'s `bounded_depth_layer1_matches_lean` writes the JSON that Lean
//! reads and checks their key facts. `bounded_depth_final_program_matches_lean`
//! then requires `boundedDepthCheck` to accept the log. The two pins record the
//! run digests that the check prints, so a change to the program fails
//! without Lean: `bounded_depth_prefix_digest_is_pinned` (in CI) the run up to
//! layer 1 on `toy_relation`, `bounded_depth_program_digest_is_pinned` the whole
//! run, which derives both layer-1 keys.

use std::path::{Path, PathBuf};
use std::process::Command;

use neo_spartan::shrink::{run_digest, Program};
use neo_spartan::{Backend, Error};
use p3_field::PrimeField64;

use super::super::compression::Keys;
use super::super::final_program::{FinalProgram, Finish};
use super::super::BoundedDepth;
use super::{prover, support, words};

use crate::lifecycle::ir_log::{toy_key, toy_relation, Log};

const FINISH: &str = "nightstream-fprime-bounded-depth-finish-v1.json";

/// The run digests of the final program's shape run and of its run up to
/// layer 1, as `boundedDepthCheck` computes them from Lean's `BoundedDepth.finalProgram`
/// (`digest` and `prefix digest` on its `match` line). Update them only after
/// `bounded_depth_final_program_matches_lean` passes.
const BOUNDED_DEPTH_PROGRAM_DIGEST: [u64; 4] = [
    15784576640903226257,
    18033717694290237781,
    10391104671215674562,
    8238677054024326082,
];

const BOUNDED_DEPTH_PREFIX_DIGEST: [u64; 4] = [
    1127690888971602825,
    8298036602455951323,
    11644551335968869541,
    5423398277105987646,
];

pub(super) fn workspace() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

pub(super) fn lowering(name: &str) -> PathBuf {
    let dir = workspace().join("target/lean-lowering");
    std::fs::create_dir_all(&dir).unwrap();
    dir.join(name)
}

fn finish() -> Finish {
    Finish::new(
        nightstream_fprime::load_bounded_depth_finish(
            &support::artifact(FINISH),
            nightstream_fprime::BOUNDED_DEPTH_FINISH_IDENTITY,
        )
        .expect("finish package"),
    )
    .expect("finish rows")
}

/// `FinalProgram` without a witness, on any statement: its instructions
/// depend only on the package, the state keys and the layer-1 keys.
fn with_program<T>(
    prover: &BoundedDepth,
    keys: &Keys,
    finish: &Finish,
    use_it: impl FnOnce(FinalProgram<'_>) -> T,
) -> T {
    let relations = [
        BoundedDepth::layer1_relation(keys, 0).expect("light relation"),
        BoundedDepth::layer1_relation(keys, 1).expect("refresh relation"),
    ];
    use_it(prover.final_program(finish, [&relations[0], &relations[1]], statement(), None))
}

fn statement() -> (u64, [neo_math::F; 4], [neo_math::F; 4]) {
    (3, words([1, 2, 3, 4]), words([5, 6, 7, 8]))
}

/// `FinalProgram::run` on `toy_relation`, which stops at layer 1's shape
/// check after the first slot's seed: the run up to layer 1.
struct Prefix<'a>(FinalProgram<'a>);

impl Program for Prefix<'_> {
    fn statement(&self) -> Vec<u64> {
        self.0.statement()
    }

    fn key(&self) -> Vec<u64> {
        self.0.key()
    }

    fn run<B: Backend>(&self, b: &mut B) -> Result<(), Error> {
        match self.0.run(b) {
            Err(Error::Shape("claim words")) => Ok(()),
            other => other,
        }
    }
}

#[test]
#[ignore = "writes the layer-1 keys that neo-spartan's bounded_depth_layer1_matches_lean reads"]
fn bounded_depth_layer1_keys_are_written() {
    let keys = prover(2).compression_keys().expect("compression keys");
    for (slot, key) in keys.layer1.iter().enumerate() {
        let path = lowering(&format!("bounded_depth_layer1_{slot}.bin"));
        std::fs::write(&path, bincode::serialize(&(key, keys.bound, keys.layer_bits)).unwrap()).unwrap();
        eprintln!("wrote {}", path.display());
    }
}

#[test]
fn bounded_depth_prefix_digest_is_pinned() {
    let key = toy_key();
    let relation = toy_relation(&key);
    let prover = prover(2);
    let finish = finish();
    let program = prover.final_program(&finish, [&relation, &relation], statement(), None);
    assert_eq!(run_digest(&Prefix(program)).unwrap(), BOUNDED_DEPTH_PREFIX_DIGEST);
}

#[test]
#[ignore = "derives both layer-1 keys, about two minutes; scripts/crosschecks.sh runs it"]
fn bounded_depth_program_digest_is_pinned() {
    let prover = prover(2);
    let keys = prover.compression_keys().expect("compression keys");
    let finish = finish();
    let digest = with_program(&prover, &keys, &finish, |program| run_digest(&program).unwrap());
    assert_eq!(digest, BOUNDED_DEPTH_PROGRAM_DIGEST);
}

#[test]
#[ignore = "needs bounded_depth_layer1_{0,1}.json from neo-spartan's bounded_depth_layer1_matches_lean, \
            and lake build boundedDepthCheck in formal/compression-lean"]
fn bounded_depth_final_program_matches_lean() {
    let prover = prover(2);
    let keys = prover.compression_keys().expect("compression keys");
    let finish = finish();
    let out = with_program(&prover, &keys, &finish, |program| {
        let mut run = Log::default();
        program.run(&mut run).expect("the shape run");
        let state_keys: Vec<String> = prover
            .keys
            .iter()
            .flatten()
            .map(|word| word.as_canonical_u64().to_string())
            .collect();
        eprintln!("bounded-depth final program: {} values", run.values);
        format!("keys {} {}\n{}end\n", state_keys.len(), state_keys.join(" "), run.ir)
    });
    let path = lowering("bounded_depth_final.txt");
    std::fs::write(&path, out).unwrap();
    let layer1 = [0, 1].map(|slot| lowering(&format!("bounded_depth_layer1_{slot}.json")));
    for key in &layer1 {
        assert!(
            key.exists(),
            "run neo-spartan's bounded_depth_layer1_matches_lean first: {}",
            key.display()
        );
    }
    let check = workspace().join("formal/compression-lean/.lake/build/bin/boundedDepthCheck");
    assert!(
        check.exists(),
        "wrote {}; build the check with `lake build boundedDepthCheck` in formal/compression-lean",
        path.display()
    );
    let output = Command::new(&check)
        .arg(&path)
        .arg(
            workspace()
                .join("formal/nightstream-fprime/artifacts")
                .join(FINISH),
        )
        .args(&layer1)
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    eprintln!("boundedDepthCheck: {}", stdout.trim_end());
    assert!(
        output.status.success(),
        "{stdout}{}",
        String::from_utf8_lossy(&output.stderr)
    );
}
