//! The parity gates, one module per protocol piece. Each gate builds one
//! workload, runs the canonical CPU prover and the device implementation on
//! it, and asserts field equality before printing a one-line timing summary.
//!
//! Shared imports, fixture shapes, and cross-gate helpers live here; the
//! submodules pull them in with `use super::*;`.

use std::sync::Arc;

use cuda_core::DeviceBuffer;
use neo_ajtai::AjtaiSModule;
use neo_ccs::traits::SModuleHomomorphism as _;
use neo_fold_clean::engine::transcript::Transcript;
use neo_fold_clean::paper::nifs::{
    prove as nifs_cpu_prove, NifsFreshInstancesRequest, NifsProof, NifsProverAdapter, NifsProverRequest,
};
use neo_fold_clean::paper::pi_rlc;
use neo_fold_clean::paper::relations::{ajtai_dec_mixer, ajtai_rlc_mixer, LaneRanges, LaneScheme};
use neo_fold_clean::{CcsInstance, RunningInstance};
use neo_math::{KExtensions, D, F, K};
use neo_prover_cuda::commit::DeviceAjtai;
use neo_prover_cuda::device::Device;
use neo_prover_cuda::graph::CapturedGraph;
use neo_prover_cuda::ingest::upload_witness_planes;
use neo_prover_cuda::kernels::goldilocks::Kx;
use neo_prover_cuda::kernels::probe::{launch_cooperative_grid_sync_probe, launch_k_mul_add, load_probe_kernels};
use neo_prover_cuda::reduce::ccs::{
    DeviceFeBackend, DeviceFeOracle, DeviceFeTailRound, DeviceNcBackend, DeviceNcOracle, DevicePiCcsKSurfaces,
    DevicePiCcsOutputsDigest, DevicePiCcsPhaseBackend, PiCcsOutputDigestShell, SumcheckKernels,
};
use neo_prover_cuda::reduce::dec::{DecOutputMode, DecParentWitness, DecRecompositionMode, DeviceDec};
use neo_prover_cuda::reduce::rlc as device_rlc;
use neo_prover_cuda::ring_forms::DeviceBarMatrices;
use neo_prover_cuda::CudaNifsProver;
use neo_reductions::optimized_engine::legacy_split_nc::{BackendTranscriptMode, FeSumcheckBackend as _};
use neo_reductions::sumcheck::RoundOracle as _;
use p3_field::PrimeCharacteristicRing;
use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::fixtures::{install_seeded_global_pp, pack_ring_matrix, rand_bounded, rand_f, rand_k, timed, Fixture};

/// Fixture shape shared by the `fresh` and `dec` gates: small enough to be
/// quick, with the R1CS matrix count so multi-row y_ring paths run.
const FIXTURE_N: usize = 20_000;
const FIXTURE_T: usize = 3;
const FIXTURE_M_IN: usize = 8;

/// Real sha256-workload scale for the bench gates: m = 8377 ring columns,
/// t = 3 matrices (n is m-tied because the fixture matrices are square).
const BENCH_N: usize = 8377 * D;

mod ccs;
mod commit;
mod dec;
mod e2e;
mod nebula;
mod nifs;
mod rlc;
mod transcript;

pub use ccs::{
    ccs_bench, ccs_fe, ccs_graph_replay, ccs_graph_replay_bench, ccs_nc, ccs_output_digest, ccs_phase_bench,
    ccs_phase_summary, ccs_prove,
};
pub use commit::{ajtai, fresh, fresh_bench, smoke};
pub use dec::{dec, dec_bench};
pub use e2e::{
    e2e_bench, e2e_gpu_fast_bench, e2e_graph_bench, e2e_graph_once_bench, e2e_graph_three_bench,
    e2e_graph_three_recapture_bench, e2e_graph_two_bench, e2e_multichain16_fast_bench, e2e_multichain8_bench,
    e2e_multichain8_fast_bench, e2e_multichain_bench, e2e_whole_fe_bench, e2e_whole_fe_fast_bench,
};
pub use nebula::nebula_lifecycle;
pub use nifs::{nifs, nifs_bench, nifs_nebula, nifs_whole_phase};
pub use rlc::{rlc, rlc_bench};
pub use transcript::transcript;
