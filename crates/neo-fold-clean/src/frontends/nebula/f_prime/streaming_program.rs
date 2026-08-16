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
const LIFECYCLE_GROUPS: usize = 2;
const CIRCUIT_KINDS: usize = 23;
pub(super) const FIRST_CLAIM_PROGRAM_CURSOR: usize = 1 + PRIOR_STATE_FRAME_FIELDS.div_ceil(STATE_CHUNK_FIELDS);
pub(super) const FIRST_PI_CCS_ROUND_PROGRAM_CURSOR: usize =
    FIRST_CLAIM_PROGRAM_CURSOR + CLAIM_FRAME_FIELDS.div_ceil(CLAIM_CHUNK_FIELDS) + 1;
pub(super) const FIRST_PI_RLC_FAMILY_PROGRAM_CURSOR: usize =
    FIRST_CLAIM_PROGRAM_CURSOR + CLAIM_FRAME_FIELDS.div_ceil(CLAIM_CHUNK_FIELDS) + 1 + PI_CCS_ROUNDS + 3;

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

/// Distinct field-R1CS shapes stored once by the phased relation.
///
/// Full and final replay chunks are separate kinds because their row and
/// assignment shapes differ. Indexed PiCCS rounds share one kind. PiRLC
/// families use separate even-cursor and odd-cursor replay shapes.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NebulaFPrimeStreamingCircuitKind {
    Prelude = 0,
    PriorStateReplayFull = 1,
    PriorStateReplayFinal = 2,
    ClaimReplayFull = 3,
    ClaimReplayFinal = 4,
    PiCcsStart = 5,
    PiCcsRound = 6,
    PiCcsFinish = 7,
    RunningParentPiDec = 8,
    PiRlcStart = 9,
    PiRlcFamilyEven = 10,
    PiRlcFamilyOdd = 11,
    PiRlcFinish = 12,
    PiDec = 13,
    PointBinding = 14,
    Application = 15,
    Counters = 16,
    SuccessorPrefixReplayFull = 17,
    SuccessorPrefixReplayFinal = 18,
    Nebula = 19,
    Accumulator = 20,
    Output = 21,
    SemanticLinks = 22,
}

impl NebulaFPrimeStreamingCircuitKind {
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

    pub const fn first_claim_program_cursor(&self) -> usize {
        FIRST_CLAIM_PROGRAM_CURSOR
    }

    pub const fn first_pi_rlc_family_program_cursor(&self) -> usize {
        FIRST_PI_RLC_FAMILY_PROGRAM_CURSOR
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

    pub const fn lifecycle_group_count(&self) -> usize {
        LIFECYCLE_GROUPS
    }

    pub const fn circuit_kind_count(&self) -> usize {
        CIRCUIT_KINDS
    }

    pub fn work_items(&self) -> &[NebulaFPrimeStreamingWorkItem] {
        &self.work_items
    }

    /// Exact base or shared-recursive circuit for each schedule arm.
    pub fn lifecycle_group_map(&self) -> Vec<usize> {
        (0..self.work_items.len())
            .map(|arm| usize::from(arm != 0))
            .collect()
    }

    /// Exact shared circuit kind for each schedule arm.
    pub fn circuit_kind_map(&self) -> Vec<usize> {
        self.work_items
            .iter()
            .map(|&item| self.circuit_kind(item).code() as usize)
            .collect()
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

    fn circuit_kind(&self, item: NebulaFPrimeStreamingWorkItem) -> NebulaFPrimeStreamingCircuitKind {
        use NebulaFPrimeStreamingCircuitKind as Kind;
        use NebulaFPrimeStreamingPhase as Phase;

        match item.phase {
            Phase::Prelude => Kind::Prelude,
            Phase::PriorStateReplay if item.index + 1 == self.prior_state_chunks() => Kind::PriorStateReplayFinal,
            Phase::PriorStateReplay => Kind::PriorStateReplayFull,
            Phase::ClaimReplay if item.index + 1 == self.claim_chunks() => Kind::ClaimReplayFinal,
            Phase::ClaimReplay => Kind::ClaimReplayFull,
            Phase::PiCcsStart => Kind::PiCcsStart,
            Phase::PiCcsRound => Kind::PiCcsRound,
            Phase::PiCcsFinish => Kind::PiCcsFinish,
            Phase::RunningParentPiDec => Kind::RunningParentPiDec,
            Phase::PiRlcStart => Kind::PiRlcStart,
            Phase::PiRlcFamily if item.index % 2 == 0 => Kind::PiRlcFamilyEven,
            Phase::PiRlcFamily => Kind::PiRlcFamilyOdd,
            Phase::PiRlcFinish => Kind::PiRlcFinish,
            Phase::PiDec => Kind::PiDec,
            Phase::PointBinding => Kind::PointBinding,
            Phase::Application => Kind::Application,
            Phase::Counters => Kind::Counters,
            Phase::SuccessorPrefixReplay if item.index + 1 == self.successor_prefix_chunks() => {
                Kind::SuccessorPrefixReplayFinal
            }
            Phase::SuccessorPrefixReplay => Kind::SuccessorPrefixReplayFull,
            Phase::Nebula => Kind::Nebula,
            Phase::Accumulator => Kind::Accumulator,
            Phase::Output => Kind::Output,
            Phase::SemanticLinks => Kind::SemanticLinks,
        }
    }
}

fn push_single(items: &mut Vec<NebulaFPrimeStreamingWorkItem>, phase: NebulaFPrimeStreamingPhase) {
    items.push(NebulaFPrimeStreamingWorkItem::new(phase, 0));
}

fn push_indexed(items: &mut Vec<NebulaFPrimeStreamingWorkItem>, phase: NebulaFPrimeStreamingPhase, count: usize) {
    items.extend((0..count).map(|index| NebulaFPrimeStreamingWorkItem::new(phase, index)));
}
