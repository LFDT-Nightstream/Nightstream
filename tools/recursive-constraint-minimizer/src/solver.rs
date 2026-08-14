//! Bounded cvc5 process execution and fail-closed result classification.

use std::error::Error;
use std::fmt;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::Instant;

use serde::Serialize;

use crate::Query;

pub const MAX_TIMEOUT_MS: u64 = 300_000;

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

    let start = Instant::now();
    let mut child = Command::new(&config.executable)
        .arg("--lang=smt2")
        .arg(format!("--ff-solver={}", config.mode.as_str()))
        .arg(format!("--tlimit-per={}", config.timeout_ms))
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
    child
        .stdin
        .take()
        .ok_or_else(|| SolverError::new("failed to open cvc5 stdin"))?
        .write_all(query.smt2.as_bytes())
        .map_err(|error| SolverError::new(format!("failed to write the cvc5 query: {error}")))?;
    let output = child
        .wait_with_output()
        .map_err(|error| SolverError::new(format!("failed to wait for cvc5: {error}")))?;
    let elapsed_ms = start.elapsed().as_millis().min(u128::from(u64::MAX)) as u64;
    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    if !output.status.success() {
        return Err(SolverError::new(format!(
            "cvc5 exited with status {:?}: {}",
            output.status.code(),
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
        exit_code: output.status.code(),
        elapsed_ms,
    })
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
