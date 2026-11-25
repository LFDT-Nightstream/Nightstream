//! The boundary to the Akita commitment scheme (ePrint 2026/1983).

pub(crate) mod prover;
pub(crate) mod verifier;

use akita_config::proof_optimized::fp64;
use jolt_field::Ring;
use neo_math::{KExtensions, F, K};
use p3_field::PrimeField64;
use rayon::{ThreadPool, ThreadPoolBuilder};
use std::sync::OnceLock;

use crate::key::ROW_VARS;
use crate::layout::LANE_VARS;
use crate::Error;

/// Akita's `F_q`, with the modulus of `neo_math::F`.
pub(crate) type Base = fp64::Field;
/// Akita's `F_q[u]/(u² − 2)`, which is `K` with the same basis.
pub(crate) type Extension = fp64::ExtensionField;

/// Akita checks each row against the policy of `Config` when it decodes the file.
pub(crate) struct Schedules {
    pub(crate) key_vars: usize,
    pub(crate) rows: &'static [(usize, usize)],
    bytes: &'static [u8],
    pub(crate) key: &'static [u8],
}

pub(crate) const SCHEDULES: [Schedules; 2] = [
    Schedules {
        key_vars: 20,
        rows: &[(16, 16)],
        bytes: include_bytes!("../artifacts/key20.aks"),
        key: include_bytes!("../artifacts/key20.vk"),
    },
    Schedules {
        key_vars: 30,
        rows: &[(26, 26), (27, 26)],
        bytes: include_bytes!("../artifacts/key30.aks"),
        key: include_bytes!("../artifacts/key30.vk"),
    },
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Shape {
    pub(crate) key: usize,
    pub(crate) witness: usize,
    pub(crate) inverse: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Polynomial {
    Key,
    Witness,
    Inverse,
}

impl Schedules {
    pub(crate) fn of_blocks(blocks: usize) -> Option<&'static Self> {
        SCHEDULES
            .iter()
            .find(|schedules| schedules.blocks() == blocks)
    }

    pub(crate) fn blocks(&self) -> usize {
        1 << (self.key_vars - LANE_VARS - ROW_VARS)
    }

    pub(crate) fn first(&self) -> Shape {
        let (witness, inverse) = self.rows[0];
        Shape {
            key: self.key_vars,
            witness,
            inverse,
        }
    }
}

pub(crate) fn base(value: F) -> Base {
    Base::from_u64(value.as_canonical_u64())
}

pub(crate) fn extension(value: K) -> Extension {
    let [real, imaginary] = value.as_coeffs();
    Extension::new(base(real), base(imaginary))
}

pub(crate) fn failed(error: impl std::fmt::Display) -> Error {
    Error::Akita(error.to_string())
}

thread_local! {
    static PROVER_WORKER: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

pub fn in_cpu_prover_pool() -> bool {
    PROVER_WORKER.get()
}

pub struct CpuProverPool(ThreadPool);

impl CpuProverPool {
    /// Create CPU prover workers with stacks that can run Akita under fat LTO.
    pub fn new(threads: std::num::NonZeroUsize) -> Result<Self, rayon::ThreadPoolBuildError> {
        ThreadPoolBuilder::new()
            .num_threads(threads.get())
            .stack_size(64 << 20)
            .start_handler(|_| PROVER_WORKER.set(true))
            .build()
            .map(Self)
    }

    /// Run the closure and its nested Rayon work on these workers.
    pub fn install<R: Send>(&self, work: impl FnOnce() -> R + Send) -> R {
        self.0.install(work)
    }
}

/// Under fat LTO, Akita's NTT loops need more than rayon's default 2 MB stack.
pub(crate) fn large_stack_pool() -> Option<&'static ThreadPool> {
    static POOL: OnceLock<Option<ThreadPool>> = OnceLock::new();
    POOL.get_or_init(|| ThreadPoolBuilder::new().stack_size(64 << 20).build().ok())
        .as_ref()
}

pub(crate) fn on_large_stacks<R: Send>(work: impl FnOnce() -> R + Send) -> R {
    if in_cpu_prover_pool() {
        return work();
    }
    match large_stack_pool() {
        Some(pool) => pool.install(work),
        None => work(),
    }
}

#[cfg(test)]
#[path = "../tests/unit/akita.rs"]
mod tests;
