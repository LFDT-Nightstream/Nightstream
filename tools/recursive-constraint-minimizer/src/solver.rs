//! Bounded cvc5 process execution and fail-closed result classification.

use std::error::Error;
use std::fmt;
use std::io::{Read, Write};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use serde::Serialize;

use crate::Query;

pub const MAX_TIMEOUT_MS: u64 = 300_000;
const SOLVER_OUTPUT_MARGIN_MS: u64 = 1_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SolverMode {
    Gb,
    Split,
}

impl SolverMode {
    pub fn parse(value: &str) -> Result<Self, SolverError> {
        match value {
            "gb" => Ok(Self::Gb),
            "split" => Ok(Self::Split),
            _ => Err(SolverError::new(format!(
                "invalid finite-field solver mode {value:?}; expected gb or split"
            ))),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Gb => "gb",
            Self::Split => "split",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SolverConfig {
    pub executable: PathBuf,
    pub mode: SolverMode,
    pub timeout_ms: u64,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            executable: PathBuf::from("cvc5"),
            mode: SolverMode::Gb,
            timeout_ms: 60_000,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SolverStatus {
    Sat,
    Unsat,
    Unknown,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Conclusion {
    CounterexampleCandidate,
    RedundancyCandidate,
    Inconclusive,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SolverRun {
    pub status: SolverStatus,
    pub conclusion: Conclusion,
    pub stdout: String,
    pub stderr: String,
    pub exit_code: Option<i32>,
    pub elapsed_ms: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SolverError(String);

impl SolverError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for SolverError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for SolverError {}

pub fn run_cvc5(query: &Query, config: &SolverConfig) -> Result<SolverRun, SolverError> {
    if config.timeout_ms == 0 || config.timeout_ms > MAX_TIMEOUT_MS {
        return Err(SolverError::new(format!("timeout_ms must be in 1..={MAX_TIMEOUT_MS}")));
    }
    let solver_limit_ms = config
        .timeout_ms
        .saturating_sub(SOLVER_OUTPUT_MARGIN_MS)
        .max(1);

    let start = Instant::now();
    let mut child = Command::new(&config.executable)
        .arg("--lang=smt2")
        .arg(format!("--ff-solver={}", config.mode.as_str()))
        .arg(format!("--tlimit-per={solver_limit_ms}"))
        .arg("--produce-models")
        .arg("--dump-models")
        .arg("--produce-unsat-cores")
        .arg("--dump-unsat-cores")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| {
            SolverError::new(format!(
                "failed to start cvc5 executable {:?}: {error}",
                config.executable
            ))
        })?;
    let stdin = child
        .stdin
        .take()
        .ok_or_else(|| SolverError::new("failed to open cvc5 stdin"))?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| SolverError::new("failed to open cvc5 stdout"))?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| SolverError::new("failed to open cvc5 stderr"))?;
    let stdout_reader = thread::spawn(move || read_output(stdout));
    let stderr_reader = thread::spawn(move || read_output(stderr));
    let input = query.smt2.clone();
    let stdin_writer = thread::spawn(move || write_input(stdin, input));

    let wall_limit = Duration::from_millis(config.timeout_ms);
    let exit_status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) if start.elapsed() < wall_limit => thread::sleep(Duration::from_millis(10)),
            Ok(None) => {
                let _ = child.kill();
                let _ = child.wait();
                let _ = stdin_writer.join();
                let _ = stdout_reader.join();
                let _ = stderr_reader.join();
                return Err(SolverError::new(format!(
                    "cvc5 exceeded the {} ms wall-clock limit",
                    config.timeout_ms
                )));
            }
            Err(error) => {
                let _ = child.kill();
                let _ = child.wait();
                let _ = stdin_writer.join();
                let _ = stdout_reader.join();
                let _ = stderr_reader.join();
                return Err(SolverError::new(format!("failed to wait for cvc5: {error}")));
            }
        }
    };
    join_input(stdin_writer)?;
    let stdout = join_output(stdout_reader, "stdout")?;
    let stderr = join_output(stderr_reader, "stderr")?;
    let elapsed_ms = start.elapsed().as_millis().min(u128::from(u64::MAX)) as u64;
    let stdout = String::from_utf8_lossy(&stdout).into_owned();
    let stderr = String::from_utf8_lossy(&stderr).into_owned();
    if !exit_status.success() {
        return Err(SolverError::new(format!(
            "cvc5 exited with status {:?}: {}",
            exit_status.code(),
            stderr.trim()
        )));
    }
    let status = parse_status(&stdout)?;
    let conclusion = match status {
        SolverStatus::Sat => Conclusion::CounterexampleCandidate,
        SolverStatus::Unsat => Conclusion::RedundancyCandidate,
        SolverStatus::Unknown => Conclusion::Inconclusive,
    };
    Ok(SolverRun {
        status,
        conclusion,
        stdout,
        stderr,
        exit_code: exit_status.code(),
        elapsed_ms,
    })
}

fn write_input(mut pipe: impl Write, input: String) -> std::io::Result<()> {
    pipe.write_all(input.as_bytes())
}

fn join_input(writer: thread::JoinHandle<std::io::Result<()>>) -> Result<(), SolverError> {
    writer
        .join()
        .map_err(|_| SolverError::new("cvc5 stdin writer panicked"))?
        .map_err(|error| SolverError::new(format!("failed to write the cvc5 query: {error}")))
}

fn read_output(mut pipe: impl Read) -> std::io::Result<Vec<u8>> {
    let mut bytes = Vec::new();
    pipe.read_to_end(&mut bytes)?;
    Ok(bytes)
}

fn join_output(reader: thread::JoinHandle<std::io::Result<Vec<u8>>>, stream: &str) -> Result<Vec<u8>, SolverError> {
    reader
        .join()
        .map_err(|_| SolverError::new(format!("cvc5 {stream} reader panicked")))?
        .map_err(|error| SolverError::new(format!("failed to read cvc5 {stream}: {error}")))
}

fn parse_status(stdout: &str) -> Result<SolverStatus, SolverError> {
    stdout
        .lines()
        .find_map(|line| match line.trim() {
            "sat" => Some(SolverStatus::Sat),
            "unsat" => Some(SolverStatus::Unsat),
            "unknown" => Some(SolverStatus::Unknown),
            _ => None,
        })
        .ok_or_else(|| SolverError::new("cvc5 output did not contain sat, unsat, or unknown"))
}
