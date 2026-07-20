//! Typed public binding selected before Π_CCS transcript replay.

use neo_math::F;

use crate::engines::utils::{Dims, PiCcsTranscriptVariant};

use super::oracle::BlockLaneNcPending;
use super::PiCcsProofVariant;

#[derive(Clone, Debug)]
pub(super) enum NcReplayMode {
    LegacyFlat,
    BlockLaneDelayed(Option<BlockLaneNcPending>),
}

#[derive(Clone, Debug)]
pub(super) struct ReplayBinding {
    pub public_instance_digest: Option<[F; 4]>,
    pub me_input_accumulator_handle: Option<[F; 4]>,
    pub nc_mode: NcReplayMode,
}

impl ReplayBinding {
    pub fn block_pending(&self) -> Option<Option<BlockLaneNcPending>> {
        match &self.nc_mode {
            NcReplayMode::LegacyFlat => None,
            NcReplayMode::BlockLaneDelayed(pending) => Some(pending.clone()),
        }
    }

    pub fn transcript_variant(&self) -> PiCcsTranscriptVariant {
        match &self.nc_mode {
            NcReplayMode::LegacyFlat => PiCcsTranscriptVariant::SplitNcV1,
            NcReplayMode::BlockLaneDelayed(_) => PiCcsTranscriptVariant::BlockLaneNcDelayedV1,
        }
    }

    pub fn proof_variant(&self) -> PiCcsProofVariant {
        match &self.nc_mode {
            NcReplayMode::LegacyFlat => PiCcsProofVariant::SplitNcV1,
            NcReplayMode::BlockLaneDelayed(_) => PiCcsProofVariant::BlockLaneNcDelayedV1,
        }
    }

    pub fn nc_point_variables(&self, dims: Dims) -> usize {
        match &self.nc_mode {
            NcReplayMode::LegacyFlat => dims.ell_m,
            NcReplayMode::BlockLaneDelayed(_) => super::oracle::BLOCK_LANE_NC_BLOCK_VARIABLES,
        }
    }

    pub fn claims() -> Self {
        Self {
            public_instance_digest: None,
            me_input_accumulator_handle: None,
            nc_mode: NcReplayMode::LegacyFlat,
        }
    }

    pub fn instance_digest(public_instance_digest: [F; 4]) -> Self {
        Self {
            public_instance_digest: Some(public_instance_digest),
            me_input_accumulator_handle: None,
            nc_mode: NcReplayMode::LegacyFlat,
        }
    }

    pub fn legacy_handle(public_instance_digest: [F; 4], handle: [F; 4]) -> Self {
        Self {
            public_instance_digest: Some(public_instance_digest),
            me_input_accumulator_handle: Some(handle),
            nc_mode: NcReplayMode::LegacyFlat,
        }
    }

    pub fn block_lane_handle(
        public_instance_digest: [F; 4],
        handle: [F; 4],
        pending: Option<BlockLaneNcPending>,
    ) -> Self {
        Self {
            public_instance_digest: Some(public_instance_digest),
            me_input_accumulator_handle: Some(handle),
            nc_mode: NcReplayMode::BlockLaneDelayed(pending),
        }
    }
}
