//! One Poseidon2 proof chain through the public lifecycle. Emits phase timings
//! as JSON lines; an external process timer measures peak resident memory.

use std::{error::Error, fs, io::Write, path::Path, process::ExitCode, time::Instant};

use nightstream::{
    application::{poseidon2_hash_chain_step, poseidon2_hash_chain_v1},
    Circuit, Engine, State,
};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::Goldilocks as F;
use serde_json::{json, Value};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

// The retained Poseidon2 base fixture supplies these inputs. Every later step
// uses the same message, so separate engine runs have identical workloads.
const INITIAL: [u64; 4] = [202, 203, 204, 205];
const MESSAGE: [u64; 4] = [7, 11, 13, 17];

struct Options {
    engine: Engine,
    steps: u64,
}

fn options() -> Result<Option<Options>> {
    let mut args = std::env::args().skip(1);
    let mut engine = None;
    let mut steps = None;
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--help" | "-h" => {
                println!("Usage: nightstream-poseidon2-bench --engine optimized|metal|cuda --steps COUNT");
                println!("COUNT includes the base step. Counts of 2 or more execute active folds.");
                println!("Build with --release; run under the AGENTS.md 300-second test cap.");
                return Ok(None);
            }
            "--engine" if engine.is_none() => {
                engine = Some(match args.next().as_deref() {
                    Some("optimized") => Engine::Optimized,
                    Some("metal") => Engine::Metal,
                    Some("cuda") => Engine::Cuda,
                    _ => return Err("--engine requires optimized, metal, or cuda".into()),
                });
            }
            "--steps" if steps.is_none() => {
                let count: u64 = args
                    .next()
                    .ok_or("--steps requires a positive count")?
                    .parse()?;
                if count == 0 || count >= F::ORDER_U64 {
                    return Err("--steps must be positive and below the Goldilocks modulus".into());
                }
                steps = Some(count);
            }
            _ => return Err(format!("unknown or repeated argument: {argument}").into()),
        }
    }
    Ok(Some(Options {
        engine: engine.ok_or("--engine is required")?,
        steps: steps.ok_or("--steps is required")?,
    }))
}

fn emit(record: Value) -> Result<()> {
    let mut output = std::io::stdout().lock();
    serde_json::to_writer(&mut output, &record)?;
    writeln!(output)?;
    output.flush()?;
    Ok(())
}

fn begin(phase: &str, step: Option<u64>) -> Result<Instant> {
    emit(json!({"event":"phase_started", "phase":phase, "step":step}))?;
    Ok(Instant::now())
}

fn end(phase: &str, step: Option<u64>, started: Instant) -> Result<()> {
    let seconds = started.elapsed().as_secs_f64();
    emit(json!({"event":"phase_finished", "phase":phase, "step":step, "seconds":seconds}))
}

fn run() -> Result<()> {
    let Some(options) = options()? else {
        return Ok(());
    };
    if cfg!(debug_assertions) {
        return Err("build this benchmark with cargo build --release".into());
    }
    let profile = neo_params::NeoParams::nightstream_goldilocks_k16();
    emit(json!({
        "event":"benchmark_started", "schema":1, "benchmark":"poseidon2_hash_chain_v1",
        "engine":format!("{:?}", options.engine).to_lowercase(),
        "steps":options.steps, "active_extends":options.steps - 1,
        "initial_state":INITIAL, "message":MESSAGE,
        "profile":{"b":profile.b, "k_rho":profile.k_rho, "B":profile.B},
    }))?;
    let total = Instant::now();
    let started = begin("prepare", None)?;
    let reference = fs::read(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("artifacts/nightstream-fprime-stage1-poseidon2-hash-chain-v1.json"),
    )?;
    let circuit = Circuit::prepare_with_engine(&reference, poseidon2_hash_chain_v1()?, options.engine)?;
    drop(reference);
    end("prepare", None, started)?;

    let initial = INITIAL.map(F::from_u64);
    let message = MESSAGE.map(F::from_u64);
    let mut expected = State::new(1, initial, poseidon2_hash_chain_step(initial, message));
    let started = begin("prove", Some(1))?;
    let mut proof = circuit.prove(initial, &message)?;
    end("prove", Some(1), started)?;
    if proof.state() != &expected {
        return Err("base proof state differs from native Poseidon2".into());
    }

    for step in 2..=options.steps {
        expected = State::new(step, initial, poseidon2_hash_chain_step(expected.current(), message));
        let started = begin("extend", Some(step))?;
        proof = circuit.extend(proof, &message)?;
        end("extend", Some(step), started)?;
        if proof.state() != &expected {
            return Err(format!("step {step} differs from native Poseidon2").into());
        }
    }

    let started = begin("verify", Some(options.steps))?;
    circuit.verify(&expected, &proof)?;
    end("verify", Some(options.steps), started)?;
    emit(json!({
        "event":"benchmark_finished", "verified":true,
        "steps":options.steps, "seconds":total.elapsed().as_secs_f64(),
        "circuit_identity":circuit.identity(),
        "final_state":expected.current().map(|value| value.as_canonical_u64()),
    }))
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            let _ = emit(json!({"event":"benchmark_failed", "error":error.to_string()}));
            eprintln!("benchmark failed: {error}");
            ExitCode::FAILURE
        }
    }
}
