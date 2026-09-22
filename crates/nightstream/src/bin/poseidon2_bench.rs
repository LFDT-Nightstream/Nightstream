//! Separate package compilation from a loaded Poseidon2 proof lifecycle.
//! JSON lines report phase timings; an external process measures peak memory.

use std::{
    error::Error,
    fs,
    io::Write,
    path::{Path, PathBuf},
    process::ExitCode,
    time::Instant,
};

use nightstream::{
    application::{poseidon2_hash_chain_step, poseidon2_hash_chain_v1},
    Circuit, Engine, State, Verifier,
};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::Goldilocks as F;
use serde_json::{json, Value};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

// The retained Poseidon2 base fixture supplies these inputs. Every later step
// uses the same message, so separate engine runs have identical workloads.
const INITIAL: [u64; 4] = [202, 203, 204, 205];
const MESSAGE: [u64; 4] = [7, 11, 13, 17];

enum Command {
    Compile {
        output: PathBuf,
    },
    Run {
        package: PathBuf,
        engine: Engine,
        steps: u64,
    },
}

fn usage() {
    println!("Usage: nightstream-poseidon2-bench compile --output PATH");
    println!("       nightstream-poseidon2-bench run --package PATH --engine optimized|metal|cuda --steps COUNT");
    println!("Run uses the caller-selected package for proving and verification.");
    println!("COUNT includes the base step. Counts of 2 or more execute active folds.");
    println!("Build with --release. AGENTS.md caps: 300 seconds normally; 1800 seconds for Instruments.");
}

fn options() -> Result<Option<Command>> {
    let mut args = std::env::args().skip(1);
    let command = args.next().ok_or("compile or run is required")?;
    match command.as_str() {
        "--help" | "-h" => {
            usage();
            return Ok(None);
        }
        "compile" | "run" => {}
        _ => return Err(format!("unknown command: {command}").into()),
    }
    let mut output = None;
    let mut package = None;
    let mut engine = None;
    let mut steps = None;
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--help" | "-h" => {
                usage();
                return Ok(None);
            }
            "--output" if command == "compile" && output.is_none() => {
                output = Some(PathBuf::from(args.next().ok_or("--output requires a path")?));
            }
            "--package" if command == "run" && package.is_none() => {
                package = Some(PathBuf::from(args.next().ok_or("--package requires a path")?));
            }
            "--engine" if command == "run" && engine.is_none() => {
                engine = Some(match args.next().as_deref() {
                    Some("optimized") => Engine::Optimized,
                    Some("metal") => Engine::Metal,
                    Some("cuda") => Engine::Cuda,
                    _ => return Err("--engine requires optimized, metal, or cuda".into()),
                });
            }
            "--steps" if command == "run" && steps.is_none() => {
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
    Ok(Some(if command == "compile" {
        Command::Compile {
            output: output.ok_or("--output is required")?,
        }
    } else {
        Command::Run {
            package: package.ok_or("--package is required")?,
            engine: engine.ok_or("--engine is required")?,
            steps: steps.ok_or("--steps is required")?,
        }
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

fn compile(output: &Path) -> Result<()> {
    let profile = neo_params::NeoParams::nightstream_goldilocks_k16();
    emit(json!({
        "event":"package_compile_started", "schema":2, "benchmark":"poseidon2_hash_chain_v1",
        "output":output,
        "profile":{"b":profile.b, "k_rho":profile.k_rho, "B":profile.B},
    }))?;
    let total = Instant::now();
    let started = begin("compile", None)?;
    let reference = fs::read(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("artifacts/nightstream-fprime-stage1-poseidon2-hash-chain-v1.json"),
    )?;
    let circuit = Circuit::compile(&reference, poseidon2_hash_chain_v1()?)?;
    drop(reference);
    end("compile", None, started)?;
    let started = begin("save", None)?;
    circuit.write(output)?;
    end("save", None, started)?;
    emit(json!({
        "event":"package_compile_finished", "schema":2, "saved":true,
        "output":output, "seconds":total.elapsed().as_secs_f64(),
        "circuit_identity":circuit.identity(),
    }))
}

fn benchmark(package: &Path, engine: Engine, steps: u64) -> Result<()> {
    let profile = neo_params::NeoParams::nightstream_goldilocks_k16();
    emit(json!({
        "event":"benchmark_started", "schema":2, "benchmark":"poseidon2_hash_chain_v1",
        "timing_scope":"prepared_package_load_prove_verify", "package":package,
        "engine":format!("{engine:?}").to_lowercase(),
        "steps":steps, "active_extends":steps - 1,
        "initial_state":INITIAL, "message":MESSAGE,
        "profile":{"b":profile.b, "k_rho":profile.k_rho, "B":profile.B},
    }))?;
    let total = Instant::now();
    let started = begin("load", None)?;
    // The caller selects the package. Loading and both selected-engine
    // capabilities are part of the measured lifecycle.
    let circuit = Circuit::load(package)?;
    let prover = circuit.prover(engine)?;
    let verifier = Verifier::from_package(&circuit, engine)?;
    end("load", None, started)?;

    let initial = INITIAL.map(F::from_u64);
    let message = MESSAGE.map(F::from_u64);
    let mut expected = State::new(1, initial, poseidon2_hash_chain_step(initial, message));
    let started = begin("prove", Some(1))?;
    let mut proof = prover.prove(initial, &message)?;
    end("prove", Some(1), started)?;
    if proof.state() != &expected {
        return Err("base proof state differs from native Poseidon2".into());
    }

    for step in 2..=steps {
        expected = State::new(step, initial, poseidon2_hash_chain_step(expected.current(), message));
        let started = begin("extend", Some(step))?;
        proof = prover.extend(proof, &message)?;
        end("extend", Some(step), started)?;
        if proof.state() != &expected {
            return Err(format!("step {step} differs from native Poseidon2").into());
        }
    }

    let started = begin("verify", Some(steps))?;
    verifier.verify(&expected, &proof)?;
    end("verify", Some(steps), started)?;
    emit(json!({
        "event":"benchmark_finished", "schema":2, "verified":true,
        "timing_scope":"prepared_package_load_prove_verify",
        "steps":steps, "seconds":total.elapsed().as_secs_f64(),
        "circuit_identity":circuit.identity(),
        "final_state":expected.current().map(|value| value.as_canonical_u64()),
    }))
}

fn run() -> Result<()> {
    let Some(command) = options()? else {
        return Ok(());
    };
    if cfg!(debug_assertions) {
        return Err("build this benchmark with cargo build --release".into());
    }
    match command {
        Command::Compile { output } => compile(&output),
        Command::Run { package, engine, steps } => benchmark(&package, engine, steps),
    }
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
