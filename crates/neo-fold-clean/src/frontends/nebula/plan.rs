//! `NebulaPlan` — the spec §11 plan artifact: everything a chain is bound
//! to before any proving starts.
//!
//! Owns: the validated plan constants, the `S_mem` structure built from
//! them, the lane-commitment scheme (seeded from the plan), the public
//! ROM/RAM image, the γ-independent `D_init` (the verifier's initial-memory handle,
//! spec §7), and the plan digest that every segment's γ transcript
//! absorbs (spec §6.2).
//!
//! Does not own: chain state (`NebulaLane`), proving flow
//! ([`super::prove`]), or memory semantics ([`super::trace`]).

use neo_math::{D, F};

use crate::frontends::nebula::circuit::SMemCircuit;
use crate::frontends::nebula::layout::{CellRecord, LayoutError, NebulaParams};
use crate::paper::construction2::NebulaConfig;
use crate::paper::digest;
use crate::paper::relations::{LaneScheme, LaneSchemeError};
use p3_field::PrimeCharacteristicRing;
use thiserror::Error;

use crate::config::{NEBULA_END_TO_END_SECURITY_BITS, NEBULA_MAX_FS_QUERY_BITS};

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
    circuit: SMemCircuit,
    scheme: LaneScheme,
    rom_image: Vec<u32>,
    ram_image: Vec<u32>,
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
    /// commitments into `D_init` — recomputable by anyone from this
    /// public data, with no γ anywhere (spec §7).
    pub fn new_with_initial_ram(
        params: NebulaParams,
        rom_image: Vec<u32>,
        ram_image: Vec<u32>,
        plan_seed: [u8; 32],
        kappa: usize,
    ) -> Result<Self, PlanError> {
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
        let circuit = SMemCircuit::new(params);
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
            plan_digest,
            d_init,
        })
    }

    /// The lifecycle-facing view (spec §6.1's plan constants).
    pub fn config(&self) -> NebulaConfig {
        NebulaConfig {
            scheme: self.scheme.clone(),
            steps_per_segment: self.params.steps_per_segment() as u64,
            seg_max: self.params.seg_max,
            stacks: self.params.stack_shape(),
            plan_digest: self.plan_digest,
            d_init: self.d_init,
        }
    }

    pub fn params(&self) -> &NebulaParams {
        &self.params
    }

    pub fn circuit(&self) -> &SMemCircuit {
        &self.circuit
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

    /// The verifier's initial-memory handle (spec §7): gamma-independent and
    /// recomputable from the public ROM/RAM images and plan parameters.
    pub fn d_init(&self) -> [F; 4] {
        self.d_init
    }

    pub fn plan_digest(&self) -> [F; 4] {
        self.plan_digest
    }

    /// The evaluated §9 fingerprint error budget (security-note Cor. 4.1):
    /// per Fiat–Shamir attempt, a false-but-balancing segment survives
    /// with probability `m_seg / |K|`, `m_seg = |IS|+|WS|+|RS|+|FS|`.
    /// The geometry is derived from plan-bound constants; the global target
    /// and query cap are verifier-owned production-profile constants.
    pub fn error_budget(&self) -> ErrorBudget {
        let p = &self.params;
        let m_seg = 2 * (p.steps_per_segment() as u64 * p.b_ops as u64 + p.scanned_cells());
        ErrorBudget {
            m_seg,
            // |K| = q² ≈ 2^128 for Goldilocks².
            log2_bound_per_attempt: (m_seg as f64).log2() - 128.0,
            end_to_end_target_bits: NEBULA_END_TO_END_SECURITY_BITS,
            max_fs_query_bits: NEBULA_MAX_FS_QUERY_BITS,
        }
    }
}

/// The §9 budget line the plan records (spec §11 `error_budget`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ErrorBudget {
    /// Total multiset size per segment: `2·(N·B_ops + R + M)`.
    pub m_seg: u64,
    /// `log2(m_seg / |K|)` — the Lemma-3 term per Fiat–Shamir attempt.
    pub log2_bound_per_attempt: f64,
    /// Declared maximum-chain security target for the production profile.
    pub end_to_end_target_bits: u32,
    /// Allowed `log2(q_H)` adversarial random-oracle queries in the global bound.
    pub max_fs_query_bits: u32,
}

/// Chain the initial memory's per-step IS-lane commitments with the
/// identical mem-domain leaf/link formula and header as the live IS/FS
/// chains (spec §6.1/§7): `D_init = fold_{j ∈ [0,N)} link("mem", leaf_mem(c_j))`.
fn compute_d_init(
    params: &NebulaParams,
    scheme: &LaneScheme,
    rom_image: &[u32],
    ram_image: &[u32],
) -> Result<[F; 4], PlanError> {
    let cells: Vec<CellRecord> = rom_image
        .iter()
        .chain(ram_image)
        .map(|&v| CellRecord { v, t: 0 })
        .collect();
    debug_assert_eq!(cells.len(), params.scanned_cells() as usize);
    let mut chain = digest::nebula_chain_mem_header();
    for step in 0..params.steps_per_segment() {
        let chunk = &cells[step * params.b_scan..(step + 1) * params.b_scan];
        let bits = params.encode_scan_lane(chunk)?;
        let commitment = scheme.commit_mem_lane_bits(&bits)?;
        chain = digest::nebula_chain_link(
            &chain,
            digest::NEBULA_CHAIN_MEM_TAG,
            &digest::nebula_mem_leaf(&commitment),
        );
    }
    Ok(chain)
}

/// Independent per-matrix seeds from one plan seed (domain-separated;
/// security-note A2 assumes matrix independence).
fn derive_seed(plan_seed: [u8; 32], label: &[u8]) -> [u8; 32] {
    let mut preimage = digest::digest32_as_fields(plan_seed).to_vec();
    preimage.extend(label.iter().map(|&b| F::from_u64(b as u64)));
    digest::digest_fields_as_digest32(neo_ccs::crypto::poseidon2_goldilocks::poseidon2_hash(&preimage))
}

/// `plan_digest = Poseidon2(canonical serialization)` (spec §11): version,
/// every §2 constant, the ROM/RAM image, the scheme seed, and `D_init`.
/// Changing anything changes the digest — and the digest is absorbed at
/// every segment open (spec §6.2), so it changes every γ.
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
