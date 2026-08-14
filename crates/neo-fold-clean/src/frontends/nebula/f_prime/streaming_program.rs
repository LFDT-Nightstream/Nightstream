//! Verifier-owned work schedule for the bounded-width F-prime relation.
//!
//! This module owns only phase order and exact phase counts. It does not own
//! phase-local constraints, relation synthesis, or recursive proof state.

pub(super) const CLAIM_FRAME_FIELDS: usize = 88_023;
pub(super) const CLAIM_CHUNK_FIELDS: usize = 1_024;
const STATE_CHUNK_FIELDS: usize = 1_024;
const PRIOR_STATE_FRAME_FIELDS: usize = 83_874;
const SUCCESSOR_PREFIX_FRAME_FIELDS: usize = 83_756;
const PI_CCS_ROUNDS: usize = 26;
const PI_RLC_FAMILIES: usize = 110;
const FIXED_WORK_ITEMS: usize = 14;

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NebulaFPrimeStreamingPhase {
    Prelude = 0,
    ClaimReplay = 1,
    PiCcsStart = 2,
    PiCcsRound = 3,
    PiCcsFinish = 4,
    RunningParentPiDec = 5,
    PiRlcStart = 6,
    PiRlcFamily = 7,
    PiRlcFinish = 8,
    PiDec = 9,
    PointBinding = 10,
    PriorStateReplay = 11,
    Nebula = 12,
    Accumulator = 13,
    Counters = 14,
    Output = 15,
    Application = 16,
    SemanticLinks = 17,
    SuccessorPrefixReplay = 18,
}

impl NebulaFPrimeStreamingPhase {
    pub const fn code(self) -> u8 {
        self as u8
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NebulaFPrimeStreamingWorkItem {
    phase: NebulaFPrimeStreamingPhase,
    index: usize,
}

impl NebulaFPrimeStreamingWorkItem {
    const fn new(phase: NebulaFPrimeStreamingPhase, index: usize) -> Self {
        Self { phase, index }
    }

    pub const fn phase(self) -> NebulaFPrimeStreamingPhase {
        self.phase
    }

    pub const fn index(self) -> usize {
        self.index
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NebulaFPrimeStreamingRun {
    phase: NebulaFPrimeStreamingPhase,
    first_index: usize,
    count: usize,
}

impl NebulaFPrimeStreamingRun {
    pub const fn phase(self) -> NebulaFPrimeStreamingPhase {
        self.phase
    }

    pub const fn first_index(self) -> usize {
        self.first_index
    }

    pub const fn count(self) -> usize {
        self.count
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NebulaFPrimeStreamingProgramAudit {
    work_items: Vec<NebulaFPrimeStreamingWorkItem>,
}

impl NebulaFPrimeStreamingProgramAudit {
    pub fn production() -> Self {
        use NebulaFPrimeStreamingPhase as Phase;

        let prior_state_chunks = PRIOR_STATE_FRAME_FIELDS.div_ceil(STATE_CHUNK_FIELDS);
        let claim_chunks = CLAIM_FRAME_FIELDS.div_ceil(CLAIM_CHUNK_FIELDS);
        let successor_prefix_chunks = SUCCESSOR_PREFIX_FRAME_FIELDS.div_ceil(STATE_CHUNK_FIELDS);
        let mut work_items = Vec::with_capacity(
            prior_state_chunks
                + claim_chunks
                + PI_CCS_ROUNDS
                + PI_RLC_FAMILIES
                + successor_prefix_chunks
                + FIXED_WORK_ITEMS,
        );
        push_single(&mut work_items, Phase::Prelude);
        push_indexed(&mut work_items, Phase::PriorStateReplay, prior_state_chunks);
        push_indexed(&mut work_items, Phase::ClaimReplay, claim_chunks);
        push_single(&mut work_items, Phase::PiCcsStart);
        push_indexed(&mut work_items, Phase::PiCcsRound, PI_CCS_ROUNDS);
        push_single(&mut work_items, Phase::PiCcsFinish);
        push_single(&mut work_items, Phase::RunningParentPiDec);
        push_single(&mut work_items, Phase::PiRlcStart);
        push_indexed(&mut work_items, Phase::PiRlcFamily, PI_RLC_FAMILIES);
        push_single(&mut work_items, Phase::PiRlcFinish);
        push_single(&mut work_items, Phase::PiDec);
        push_single(&mut work_items, Phase::PointBinding);
        push_single(&mut work_items, Phase::Application);
        push_single(&mut work_items, Phase::Counters);
        push_indexed(&mut work_items, Phase::SuccessorPrefixReplay, successor_prefix_chunks);
        push_single(&mut work_items, Phase::Nebula);
        push_single(&mut work_items, Phase::Accumulator);
        push_single(&mut work_items, Phase::Output);
        push_single(&mut work_items, Phase::SemanticLinks);
        Self { work_items }
    }

    pub const fn state_chunk_fields(&self) -> usize {
        STATE_CHUNK_FIELDS
    }

    pub const fn prior_state_frame_fields(&self) -> usize {
        PRIOR_STATE_FRAME_FIELDS
    }

    pub const fn prior_state_chunks(&self) -> usize {
        PRIOR_STATE_FRAME_FIELDS.div_ceil(STATE_CHUNK_FIELDS)
    }

    pub const fn claim_frame_fields(&self) -> usize {
        CLAIM_FRAME_FIELDS
    }

    pub const fn claim_chunk_fields(&self) -> usize {
        CLAIM_CHUNK_FIELDS
    }

    pub const fn claim_chunks(&self) -> usize {
        CLAIM_FRAME_FIELDS.div_ceil(CLAIM_CHUNK_FIELDS)
    }

    pub const fn pi_ccs_rounds(&self) -> usize {
        PI_CCS_ROUNDS
    }

    pub const fn pi_rlc_families(&self) -> usize {
        PI_RLC_FAMILIES
    }

    pub const fn successor_prefix_frame_fields(&self) -> usize {
        SUCCESSOR_PREFIX_FRAME_FIELDS
    }

    pub const fn successor_prefix_chunks(&self) -> usize {
        SUCCESSOR_PREFIX_FRAME_FIELDS.div_ceil(STATE_CHUNK_FIELDS)
    }

    pub fn work_items(&self) -> &[NebulaFPrimeStreamingWorkItem] {
        &self.work_items
    }

    pub fn runs(&self) -> Vec<NebulaFPrimeStreamingRun> {
        let mut runs = Vec::<NebulaFPrimeStreamingRun>::new();
        for item in &self.work_items {
            match runs.last_mut() {
                Some(run) if run.phase == item.phase && run.first_index + run.count == item.index => {
                    run.count += 1;
                }
                _ => runs.push(NebulaFPrimeStreamingRun {
                    phase: item.phase,
                    first_index: item.index,
                    count: 1,
                }),
            }
        }
        runs
    }
}

fn push_single(items: &mut Vec<NebulaFPrimeStreamingWorkItem>, phase: NebulaFPrimeStreamingPhase) {
    items.push(NebulaFPrimeStreamingWorkItem::new(phase, 0));
}

fn push_indexed(items: &mut Vec<NebulaFPrimeStreamingWorkItem>, phase: NebulaFPrimeStreamingPhase, count: usize) {
    items.extend((0..count).map(|index| NebulaFPrimeStreamingWorkItem::new(phase, index)));
}
