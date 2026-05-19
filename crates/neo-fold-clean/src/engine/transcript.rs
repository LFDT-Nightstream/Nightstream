//! Poseidon2 transcript wrapper for the paper layer.
//!
//! Owns: a single name-spaced wrapper around `neo_transcript::Poseidon2Transcript`
//! so paper-layer code uses one type and one set of label conventions.
//!
//! Does not own: the Poseidon2 hash itself (lives in `neo-transcript`), nor
//! the choice of absorb labels (those are *paper*-driven; see usage sites).
//!
//! Hash discipline: Poseidon2 only on protocol-binding paths.

use neo_transcript::{Poseidon2Transcript, Transcript as NeoTranscript};

/// The single transcript type used by the paper layer.
///
/// Construction is via `Transcript::session` (top-level) or `Transcript::fork`
/// (sub-transcript with a scope label). All challenges are derived through
/// this handle; nobody else calls `Poseidon2Transcript::*` directly in paper
/// code.
pub struct Transcript {
    inner: Poseidon2Transcript,
}

impl Transcript {
    /// Top-of-session transcript. The label distinguishes the protocol so
    /// transcripts from different protocols cannot be confused.
    pub fn session() -> Self {
        Self::with_label(b"neo.fold.clean/session/v1")
    }

    /// Top-of-session transcript with a caller-supplied init label.
    ///
    /// Used by callers that need their transcript chain to match the
    /// in-circuit [`TranscriptGadget::new`] label (F' R1CS, tests that
    /// mirror a recursive step's pre-NIFS.V absorbs).
    pub fn with_label(label: &'static [u8]) -> Self {
        Self {
            inner: Poseidon2Transcript::new(label),
        }
    }

    /// Sub-transcript with a scope label. Used at the boundary of each
    /// reduction so an auditor can see which absorbs belong to which §.
    pub fn fork(&self, scope: &'static [u8]) -> Self {
        Self {
            inner: self.inner.fork(scope),
        }
    }

    pub fn append_message(&mut self, label: &'static [u8], msg: &[u8]) {
        self.inner.append_message(label, msg);
    }

    pub fn append_fields(&mut self, label: &'static [u8], fs: &[neo_math::F]) {
        self.inner.append_fields(label, fs);
    }

    pub fn challenge_field(&mut self, label: &'static [u8]) -> neo_math::F {
        self.inner.challenge_field(label)
    }

    pub fn challenge_fields(&mut self, label: &'static [u8], n: usize) -> Vec<neo_math::F> {
        self.inner.challenge_fields(label, n)
    }

    pub fn digest32(&mut self) -> [u8; 32] {
        self.inner.digest32()
    }

    /// Borrow the underlying Poseidon2 transcript.
    ///
    /// **Auditor**: the only legitimate use is wiring engine calls that
    /// already speak `neo_transcript::Transcript`. Paper-layer code reaches
    /// the named methods above, never this.
    pub(crate) fn inner_mut(&mut self) -> &mut Poseidon2Transcript {
        &mut self.inner
    }
}
