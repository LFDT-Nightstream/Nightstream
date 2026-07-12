//! `ProofState` — the soundness-relevant fold pair, tagged.
//!
//! `Initial` and `Active` are structurally distinct so the prover cannot
//! accidentally fold a missing `latest`: the compiler refuses to let
//! `nifs::prove` be called on `Initial` because there's no running/latest
//! pair to destructure. This is the type-level expression of Construction
//! 2's "i = 0 is special" rule.
//!
//! Naming: we use `Initial` (not `Base`) because `Base` is overloaded in
//! cryptography (base field, base case, base commitment, base proof). The
//! variant means "before the first fold, no `latest` exists yet" —
//! `Initial` says that directly.

use crate::paper::construction2::latest::LatestInstance;
use crate::paper::construction2::running::RunningInstance;
use crate::paper::nifs::{Error as NifsError, NifsRunningCarrier};

/// The fold-relevant pair `(U_i, u_i)`.
#[derive(Clone, Debug)]
pub enum ProofState {
    /// i = 0: no NIFS.P has run yet, no `latest` exists.
    Initial,
    /// i ≥ 1: a previous `extend` populated `latest` for this step's fold.
    Active {
        running: NifsRunningCarrier,
        latest: LatestInstance,
    },
}

impl ProofState {
    pub fn initial() -> Self {
        Self::Initial
    }

    pub fn active(running: RunningInstance, latest: LatestInstance) -> Self {
        Self::active_carrier(NifsRunningCarrier::materialized(running), latest)
    }

    pub fn active_carrier(running: NifsRunningCarrier, latest: LatestInstance) -> Self {
        Self::Active { running, latest }
    }

    pub fn is_initial(&self) -> bool {
        matches!(self, Self::Initial)
    }

    /// Borrow the materialized running accumulator if the state is active and
    /// already materialized.
    pub fn running(&self) -> Option<&RunningInstance> {
        match self {
            Self::Initial => None,
            Self::Active { running, .. } => running.as_materialized(),
        }
    }

    /// Materialize the running accumulator if the state is active.
    pub fn materialized_running(&self) -> Result<Option<RunningInstance>, NifsError> {
        match self {
            Self::Initial => Ok(None),
            Self::Active { running, .. } => running.materialize().map(Some),
        }
    }

    /// Borrow the running carrier if the state is active.
    pub fn running_carrier(&self) -> Option<&NifsRunningCarrier> {
        match self {
            Self::Initial => None,
            Self::Active { running, .. } => Some(running),
        }
    }

    /// Borrow the latest instance if the state is active.
    pub fn latest(&self) -> Option<&LatestInstance> {
        match self {
            Self::Initial => None,
            Self::Active { latest, .. } => Some(latest),
        }
    }
}
