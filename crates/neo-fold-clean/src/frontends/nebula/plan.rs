//! `NebulaPlan` — everything a chain is bound to before proving starts.
//!
//! Owns: the validated plan constants, the `S_mem` structure built from
//! them, the lane-commitment scheme (seeded from the plan), the public
//! ROM/RAM image, the γ-independent `D_init` initial-memory handle, and the
//! plan digest that every segment's γ transcript absorbs.
//!
//! Does not own: chain state (`NebulaLane`), proving flow
//! ([`super::prove`]), or memory semantics ([`super::trace`]).

use std::sync::Arc;

use neo_math::{D, F};

use crate::frontends::nebula::circuit::SMemCircuit;
use crate::frontends::nebula::layout::{CellRecord, LayoutError, NebulaParams};
use crate::paper::construction2::NebulaConfig;
use crate::paper::digest;
use crate::paper::relations::{LaneScheme, LaneSchemeError};
use p3_field::PrimeCharacteristicRing;
use thiserror::Error;

/// Version string bound into the plan digest.
pub const PLAN_VERSION: &[u8] = b"nebula-superneo/v3.1";
/// Domain label for deriving the `A_ops` seed from the plan seed.
const A_OPS_LABEL: &[u8] = b"nebula/A_ops/v3";
/// Domain label for deriving the `A_mem` seed from the plan seed.
const A_MEM_LABEL: &[u8] = b"nebula/A_mem/v3";

#[derive(Debug, Error)]
pub enum PlanError {
    #[error("plan: ROM image must have exactly R = {want} cells, got {got}")]
    RomImageLength { want: usize, got: usize },
    #[error("plan: RAM image must have exactly M = {want} cells, got {got}")]
    RamImageLength { want: usize, got: usize },
    #[error("plan: {0}")]
    Layout(#[from] LayoutError),
    #[error("plan: {0}")]
    Lanes(#[from] LaneSchemeError),
}

/// The compiled plan: constants, structure, scheme, and public bindings.
///
/// Build once per program; hand [`Self::config`] to
/// `Preprocessing::with_nebula` and drive segments with
/// [`super::prove::prove_segment`].
pub struct NebulaPlan {
    params: NebulaParams,
    circuit: Arc<SMemCircuit>,
    scheme: LaneScheme,
    rom_image: Vec<u32>,
    ram_image: Vec<u32>,
    plan_seed: [u8; 32],
    kappa: usize,
    plan_digest: [F; 4],
    d_init: [F; 4],
}

impl NebulaPlan {
    /// Compile a plan with zero-initialized RAM.
    pub fn new(
        params: NebulaParams,
        rom_image: Vec<u32>,
        plan_seed: [u8; 32],
        kappa: usize,
    ) -> Result<Self, PlanError> {
        let ram_image = vec![0; params.ram_cells() as usize];
        Self::new_with_initial_ram(params, rom_image, ram_image, plan_seed, kappa)
    }

    /// Compile a plan: build the `S_mem` structure, derive the lane
    /// matrices from the plan seed, lay the verifier-owned initial ROM and
    /// RAM images into per-step scan lanes, and chain their `A_mem`
    /// commitments into `D_init`. Anyone can recompute it from this
    /// public data, and it does not contain γ.
    pub fn new_with_initial_ram(
        params: NebulaParams,
        rom_image: Vec<u32>,
        ram_image: Vec<u32>,
        plan_seed: [u8; 32],
        kappa: usize,
    ) -> Result<Self, PlanError> {
        validate_initial_memory_shape(&params, &rom_image, &ram_image)?;
        let circuit = Arc::new(SMemCircuit::new(params));
        let scheme = LaneScheme::from_seeds(
            kappa,
            circuit.lane_ranges(),
            derive_seed(plan_seed, A_OPS_LABEL),
            derive_seed(plan_seed, A_MEM_LABEL),
        )?;
        let d_init = compute_d_init(&params, &scheme, &rom_image, &ram_image)?;
        let plan_digest = plan_digest(&params, &rom_image, &ram_image, plan_seed, kappa, d_init);
        Ok(Self {
            params,
            circuit,
            scheme,
            rom_image,
            ram_image,
            plan_seed,
            kappa,
            plan_digest,
            d_init,
        })
    }

    /// Bind new public ROM/RAM values while reusing the immutable circuit and
    /// lane commitment matrices of this profile.
    pub fn bind_initial_memory(&self, rom_image: Vec<u32>, ram_image: Vec<u32>) -> Result<Self, PlanError> {
        validate_initial_memory_shape(&self.params, &rom_image, &ram_image)?;
        #[cfg(feature = "perf-timers")]
        let d_init_started = std::time::Instant::now();
        let d_init = compute_d_init(&self.params, &self.scheme, &rom_image, &ram_image)?;
        #[cfg(feature = "perf-timers")]
        let d_init_elapsed = d_init_started.elapsed();
        #[cfg(feature = "perf-timers")]
        let plan_digest_started = std::time::Instant::now();
        let plan_digest = plan_digest(&self.params, &rom_image, &ram_image, self.plan_seed, self.kappa, d_init);
        #[cfg(feature = "perf-timers")]
        eprintln!(
            "[nebula-plan-bind] d_init={:.3}s plan_digest={:.3}s",
            d_init_elapsed.as_secs_f64(),
            plan_digest_started.elapsed().as_secs_f64(),
        );
        Ok(Self {
            params: self.params,
            circuit: Arc::clone(&self.circuit),
            scheme: self.scheme.clone(),
            rom_image,
            ram_image,
            plan_seed: self.plan_seed,
            kappa: self.kappa,
            plan_digest,
            d_init,
        })
    }

    /// The lifecycle-facing plan constants.
    pub fn config(&self) -> NebulaConfig {
        NebulaConfig {
            scheme: self.scheme.clone(),
            steps_per_segment: self.params.steps_per_segment() as u64,
            seg_max: self.params.seg_max,
            stacks: self.params.stack_shape(),
            initial_semantic_state_digest: digest::AccumulatorHandle::empty().digest_fields(),
            plan_digest: self.plan_digest,
            d_init: self.d_init,
        }
    }

    pub fn params(&self) -> &NebulaParams {
        &self.params
    }

    pub fn circuit(&self) -> &SMemCircuit {
        self.circuit.as_ref()
    }

    pub fn scheme(&self) -> &LaneScheme {
        &self.scheme
    }

    pub fn rom_image(&self) -> &[u32] {
        &self.rom_image
    }

    pub fn ram_image(&self) -> &[u32] {
        &self.ram_image
    }

    /// The verifier's gamma-independent initial-memory handle. It is
    /// recomputable from the public ROM/RAM images and plan parameters.
    pub fn d_init(&self) -> [F; 4] {
        self.d_init
    }

    pub fn plan_digest(&self) -> [F; 4] {
        self.plan_digest
    }

    /// The evaluated per-attempt fingerprint error bound:
    /// per Fiat–Shamir attempt, a false-but-balancing segment survives
    /// with probability `m_seg / |K|`, `m_seg = |IS|+|WS|+|RS|+|FS|`.
    /// The geometry is derived from plan-bound constants. This reports
    /// evidence and does not manufacture a global target or query cap.
    pub fn error_budget(&self) -> ErrorBudget {
        let p = &self.params;
        let m_seg = 2 * (p.steps_per_segment() as u64 * p.b_ops as u64 + p.scanned_cells());
        ErrorBudget {
            m_seg,
            // |K| = q² ≈ 2^128 for Goldilocks².
            log2_bound_per_attempt: (m_seg as f64).log2() - 128.0,
        }
    }
}

fn validate_initial_memory_shape(params: &NebulaParams, rom_image: &[u32], ram_image: &[u32]) -> Result<(), PlanError> {
    if rom_image.len() != params.rom_cells() as usize {
        return Err(PlanError::RomImageLength {
            want: params.rom_cells() as usize,
            got: rom_image.len(),
        });
    }
    if ram_image.len() != params.ram_cells() as usize {
        return Err(PlanError::RamImageLength {
            want: params.ram_cells() as usize,
            got: ram_image.len(),
        });
    }
    Ok(())
}

/// The fingerprint budget recorded by the plan.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ErrorBudget {
    /// Total multiset size per segment: `2·(N·B_ops + R + M)`.
    pub m_seg: u64,
    /// `log2(m_seg / |K|)` — the Lemma-3 term per Fiat–Shamir attempt.
    pub log2_bound_per_attempt: f64,
}

/// Chain the initial memory's per-step IS-lane commitments with the
/// same mem-domain leaf/link formula and header as the live IS/FS chains:
/// `D_init = fold_{j ∈ [0,N)} link("mem", leaf_mem(c_j))`.
fn compute_d_init(
    params: &NebulaParams,
    scheme: &LaneScheme,
    rom_image: &[u32],
    ram_image: &[u32],
) -> Result<[F; 4], PlanError> {
    #[cfg(feature = "perf-timers")]
    let total_started = std::time::Instant::now();
    #[cfg(feature = "perf-timers")]
    let cells_started = std::time::Instant::now();
    let cells: Vec<CellRecord> = rom_image
        .iter()
        .chain(ram_image)
        .map(|&v| CellRecord { v, t: 0 })
        .collect();
    #[cfg(feature = "perf-timers")]
    let cells_elapsed = cells_started.elapsed();
    debug_assert_eq!(cells.len(), params.scanned_cells() as usize);
    let mut chain = digest::nebula_chain_mem_header();
    #[cfg(feature = "perf-timers")]
    let mut encode_elapsed = std::time::Duration::ZERO;
    #[cfg(feature = "perf-timers")]
    let mut commit_elapsed = std::time::Duration::ZERO;
    #[cfg(feature = "perf-timers")]
    let mut leaf_elapsed = std::time::Duration::ZERO;
    #[cfg(feature = "perf-timers")]
    let mut link_elapsed = std::time::Duration::ZERO;
    #[cfg(feature = "perf-timers")]
    let mut zero_lanes = 0usize;
    let mut zero_leaf = None;
    for step in 0..params.steps_per_segment() {
        let chunk = &cells[step * params.b_scan..(step + 1) * params.b_scan];
        #[cfg(feature = "perf-timers")]
        let phase_started = std::time::Instant::now();
        let bits = params.encode_scan_lane(chunk)?;
        let zero_lane = bits.iter().all(|bit| *bit == F::ZERO);
        #[cfg(feature = "perf-timers")]
        {
            encode_elapsed += phase_started.elapsed();
            zero_lanes += usize::from(zero_lane);
        }
        #[cfg(feature = "perf-timers")]
        let phase_started = std::time::Instant::now();
        let commitment = scheme.commit_mem_lane_bits(&bits)?;
        #[cfg(feature = "perf-timers")]
        {
            commit_elapsed += phase_started.elapsed();
        }
        #[cfg(feature = "perf-timers")]
        let phase_started = std::time::Instant::now();
        let leaf = if zero_lane {
            *zero_leaf.get_or_insert_with(|| digest::nebula_mem_leaf(&commitment))
        } else {
            digest::nebula_mem_leaf(&commitment)
        };
        #[cfg(feature = "perf-timers")]
        {
            leaf_elapsed += phase_started.elapsed();
        }
        #[cfg(feature = "perf-timers")]
        let phase_started = std::time::Instant::now();
        chain = digest::nebula_chain_link(&chain, digest::NEBULA_CHAIN_MEM_TAG, &leaf);
        #[cfg(feature = "perf-timers")]
        {
            link_elapsed += phase_started.elapsed();
        }
    }
    #[cfg(feature = "perf-timers")]
    eprintln!(
        "[nebula-d-init] steps={} zero_lanes={} nonzero_lanes={} cells={:.3}s encode={:.3}s commit={:.3}s leaf={:.3}s link={:.3}s total={:.3}s",
        params.steps_per_segment(),
        zero_lanes,
        params.steps_per_segment() - zero_lanes,
        cells_elapsed.as_secs_f64(),
        encode_elapsed.as_secs_f64(),
        commit_elapsed.as_secs_f64(),
        leaf_elapsed.as_secs_f64(),
        link_elapsed.as_secs_f64(),
        total_started.elapsed().as_secs_f64(),
    );
    Ok(chain)
}

/// Independent, domain-separated per-matrix seeds from one plan seed.
fn derive_seed(plan_seed: [u8; 32], label: &[u8]) -> [u8; 32] {
    let mut preimage = digest::digest32_as_fields(plan_seed).to_vec();
    preimage.extend(label.iter().map(|&b| F::from_u64(b as u64)));
    digest::digest_fields_as_digest32(neo_ccs::crypto::poseidon2_goldilocks::poseidon2_hash(&preimage))
}

/// `plan_digest = Poseidon2(canonical serialization)` over the version,
/// every plan constant, the ROM/RAM image, the scheme seed, and `D_init`.
/// Changing anything changes the digest — and the digest is absorbed at
/// every segment open, so it changes every γ.
fn plan_digest(
    params: &NebulaParams,
    rom_image: &[u32],
    ram_image: &[u32],
    plan_seed: [u8; 32],
    kappa: usize,
    d_init: [F; 4],
) -> [F; 4] {
    let mut preimage: Vec<F> = Vec::new();
    preimage.push(F::from_u64(PLAN_VERSION.len() as u64));
    preimage.extend(PLAN_VERSION.iter().map(|&b| F::from_u64(b as u64)));
    preimage.push(F::from_u64(params.r as u64));
    preimage.push(F::from_u64(params.mu as u64));
    preimage.push(F::from_u64(params.b_ops as u64));
    preimage.push(F::from_u64(params.b_scan as u64));
    preimage.push(F::from_u64(params.seg_max));
    preimage.push(F::from_u64(params.num_stacks as u64));
    preimage.push(F::from_u64(params.sigma as u64));
    preimage.push(F::from_u64(kappa as u64));
    preimage.push(F::from_u64(D as u64));
    preimage.extend(digest::digest32_as_fields(plan_seed));
    preimage.push(F::from_u64(rom_image.len() as u64));
    preimage.extend(rom_image.iter().map(|&v| F::from_u64(v as u64)));
    preimage.push(F::from_u64(ram_image.len() as u64));
    preimage.extend(ram_image.iter().map(|&v| F::from_u64(v as u64)));
    preimage.extend_from_slice(&d_init);
    neo_ccs::crypto::poseidon2_goldilocks::poseidon2_hash(&preimage)
}
