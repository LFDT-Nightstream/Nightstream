//! Per-phase measurements and routing decisions for one Metal NIFS fold.
//!
//! These values are execution evidence for profiling, not proof or verifier
//! authority. Path booleans are set only after the corresponding work succeeds.

use std::time::Duration;

use crate::sumcheck::{FeSumcheckProfile, NcSumcheckProfile};
use crate::MetalActivity;

#[derive(Clone, Copy, Debug, Default)]
pub struct MetalNifsProfile {
    pub total: Duration,
    pub fresh: MetalFreshProfile,
    pub pi_ccs: MetalPiCcsProfile,
    pub pi_rlc: MetalPiRlcProfile,
    pub pi_dec: MetalPiDecProfile,
    pub residency: MetalResidencyProfile,
    pub activity: MetalActivity,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct MetalFreshProfile {
    pub commit_total: Duration,
    pub commit_gpu: Duration,
    pub commit_count: usize,
    pub masks_reused: bool,
    pub lane_commit_gpu: Duration,
    pub lane_commit_count: usize,
    pub lanes_from_resident_masks: bool,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct MetalPiCcsProfile {
    pub elapsed: Duration,
    pub fe_sumcheck: Duration,
    pub nc_sumcheck: Duration,
    pub activity: MetalActivity,
    pub fe: MetalFeProfile,
    pub nc: MetalNcProfile,
    pub ajtai: MetalAjtaiProfile,
    pub witness_masks_shared: bool,
    pub folded_tables: usize,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct MetalFeProfile {
    pub rounds: usize,
    pub mcs_tables: usize,
    pub mcs_table_bytes: usize,
    pub seeded_build: Duration,
    pub seeded_patch_entries: usize,
    pub seeded_patch_bytes: usize,
    pub explicit_coefficients: usize,
    pub explicit_row_list_histogram: [usize; 8],
    pub max_explicit_row_entries: usize,
    pub carried_eval_on_metal: bool,
    pub on_metal: bool,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct MetalNcProfile {
    pub rounds: usize,
    pub input_witnesses: usize,
    pub active_witnesses: usize,
    pub on_metal: bool,
    pub mask_native_on_metal: bool,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct MetalAjtaiProfile {
    pub y_eval: Duration,
    pub seeded_build: Duration,
    pub device_eval: Duration,
    pub tensor_gpu: Duration,
    pub form_gpu: Duration,
    pub tail_gpu: Duration,
    pub seeded_patch_entries: usize,
    pub seeded_patch_bytes: usize,
    pub form_blocks: usize,
    pub form_bytes: usize,
    pub explicit_coefficients: usize,
    pub signed_unit_coefficients: usize,
    /// Explicit-list bins: 0, 1, 2-3, 4-7, 8-15, 16-31, 32-63, and 64+ entries.
    pub explicit_form_list_histogram: [usize; 8],
    pub max_explicit_form_list_entries: usize,
    pub parallel_form_lists: usize,
    pub parallel_form_entries: usize,
    pub y_eval_on_metal: bool,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct MetalPiRlcProfile {
    pub elapsed: Duration,
    pub activity: MetalActivity,
    pub witness_on_metal: bool,
    pub witness_resident_only: bool,
    pub witness_masks_reused: bool,
    pub rho_small_coefficients: bool,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct MetalPiDecProfile {
    pub elapsed: Duration,
    pub activity: MetalActivity,
    pub form_build: Duration,
    pub projection: Duration,
    pub lane_commit_gpu: Duration,
    pub y_zcol_gpu: Duration,
    pub host_materialization: Duration,
    pub split_on_metal: bool,
    pub recomposition_on_metal: bool,
    pub forms_on_metal: bool,
    pub y_on_metal: bool,
    pub commit_on_metal: bool,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct MetalResidencyProfile {
    pub running_input: bool,
    pub running_output: bool,
    pub proof_deferred: bool,
    pub running_deferred: bool,
    pub recursive_compile_reverify_required: bool,
}

impl MetalPiCcsProfile {
    pub(crate) fn from_sumchecks(
        elapsed: Duration,
        fe_sumcheck: Duration,
        nc_sumcheck: Duration,
        activity: MetalActivity,
        fe: FeSumcheckProfile,
        nc: NcSumcheckProfile,
        witness_masks_shared: bool,
    ) -> Self {
        Self {
            elapsed,
            fe_sumcheck,
            nc_sumcheck,
            activity,
            fe: MetalFeProfile {
                rounds: fe.fe_rounds,
                mcs_tables: fe.fe_mcs_tables,
                mcs_table_bytes: fe.fe_mcs_table_bytes,
                seeded_build: fe.fe_seeded_build,
                seeded_patch_entries: fe.fe_seeded_patch_entries,
                seeded_patch_bytes: fe.fe_seeded_patch_bytes,
                explicit_coefficients: fe.fe_explicit_coefficients,
                explicit_row_list_histogram: fe.fe_explicit_row_list_histogram,
                max_explicit_row_entries: fe.fe_max_explicit_row_entries,
                carried_eval_on_metal: fe.fe_carried_eval_on_metal,
                on_metal: fe.fe_rounds > 0,
            },
            nc: MetalNcProfile {
                rounds: nc.nc_rounds,
                input_witnesses: nc.nc_input_witnesses,
                active_witnesses: nc.nc_active_witnesses,
                on_metal: nc.nc_rounds > 0,
                mask_native_on_metal: nc.nc_mask_native_on_metal,
            },
            ajtai: MetalAjtaiProfile {
                y_eval: fe.ajtai_y_eval,
                seeded_build: fe.ajtai_seeded_build,
                device_eval: fe.ajtai_device_eval,
                tensor_gpu: fe.ajtai_tensor_gpu,
                form_gpu: fe.ajtai_form_gpu,
                tail_gpu: fe.ajtai_tail_gpu,
                seeded_patch_entries: fe.ajtai_seeded_patch_entries,
                seeded_patch_bytes: fe.ajtai_seeded_patch_bytes,
                form_blocks: fe.ajtai_form_blocks,
                form_bytes: fe.ajtai_form_bytes,
                explicit_coefficients: fe.ajtai_explicit_coefficients,
                signed_unit_coefficients: fe.ajtai_signed_unit_coefficients,
                explicit_form_list_histogram: fe.ajtai_explicit_form_list_histogram,
                max_explicit_form_list_entries: fe.ajtai_max_explicit_form_list_entries,
                parallel_form_lists: fe.ajtai_parallel_form_lists,
                parallel_form_entries: fe.ajtai_parallel_form_entries,
                y_eval_on_metal: fe.ajtai_y_eval_on_metal,
            },
            witness_masks_shared,
            folded_tables: fe.folded_tables + nc.folded_tables,
        }
    }
}
