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

/// The fold-relevant pair `(U_i, u_i)`.
#[derive(Clone, Debug)]
pub enum ProofState {
    /// i = 0: no NIFS.P has run yet, no `latest` exists.
    Initial,
    /// i ≥ 1: a previous `extend` populated `latest` for this step's fold.
    Active {
        running: RunningInstance,
        latest: LatestInstance,
    },
}

impl ProofState {
    pub fn initial() -> Self {
        Self::Initial
    }

    pub fn is_initial(&self) -> bool {
        matches!(self, Self::Initial)
    }

    /// Borrow the running accumulator if the state is active.
    pub fn running(&self) -> Option<&RunningInstance> {
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
