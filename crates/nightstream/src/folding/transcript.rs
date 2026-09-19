//! The selected native transcript position, with the existing session label.
#[cfg(test)]
use neo_math::F;
use neo_transcript::{Poseidon2Transcript, Transcript as _};

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct Poseidon2TranscriptSnapshot {
    state: [F; 8],
    absorbed: usize,
}
#[cfg(test)]
impl Poseidon2TranscriptSnapshot {
    pub(crate) fn state(&self) -> [F; 8] {
        self.state
    }
    pub(crate) fn absorbed(&self) -> usize {
        self.absorbed
    }
}
#[derive(Clone)]
pub(crate) struct Transcript {
    inner: Poseidon2Transcript,
}
impl Transcript {
    pub(crate) fn session() -> Self {
        Self {
            inner: Poseidon2Transcript::new(b"neo.fold.clean/session/v1"),
        }
    }
    pub(crate) fn inner_mut(&mut self) -> &mut Poseidon2Transcript {
        &mut self.inner
    }
    #[cfg(test)]
    pub(crate) fn snapshot(&self) -> Poseidon2TranscriptSnapshot {
        Poseidon2TranscriptSnapshot {
            state: self.inner.state(),
            absorbed: self.inner.absorbed(),
        }
    }
}
