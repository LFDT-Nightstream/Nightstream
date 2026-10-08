//! The selected native transcript position. A session starts at the Lean v1_2
//! zero state; no label is absorbed.
use neo_ccs::crypto::poseidon2_goldilocks::WIDTH;
use neo_math::F;
use neo_transcript::Poseidon2Transcript;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct Poseidon2TranscriptSnapshot {
    state: [F; WIDTH],
    absorbed: usize,
}
#[cfg(test)]
impl Poseidon2TranscriptSnapshot {
    pub(crate) fn state(&self) -> [F; WIDTH] {
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
            inner: Poseidon2Transcript::new_v1_2(),
        }
    }
    pub(crate) fn inner_mut(&mut self) -> &mut Poseidon2Transcript {
        &mut self.inner
    }
    pub(crate) fn snapshot(&self) -> Poseidon2TranscriptSnapshot {
        Poseidon2TranscriptSnapshot {
            state: self.inner.state(),
            absorbed: self.inner.absorbed(),
        }
    }
}
