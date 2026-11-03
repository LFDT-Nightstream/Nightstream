//! The selected native transcript position. A session starts at the Lean v1_2
//! zero state; no label is absorbed.
use neo_transcript::Poseidon2Transcript;

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
}
