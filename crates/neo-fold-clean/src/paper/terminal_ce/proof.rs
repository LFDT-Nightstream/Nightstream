use neo_math::F;

/// Opaque proof material for the future compact terminal CE verifier.
///
/// This type intentionally has no successful verifier today. It exists so any
/// future integration has to thread explicit proof material into the circuit
/// verifier instead of treating terminal-child digests as authority.
///
/// `public_digest` is the digest the proof claims to verify against. The
/// circuit must recompute that digest from verifier-owned preprocessing and
/// NIFS-output child wires, constrain equality, and then verify `bytes`.
/// Equality of `public_digest` alone is never a proof of terminal CE.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TerminalCeProof {
    public_digest: [F; 4],
    bytes: Vec<u8>,
}

impl TerminalCeProof {
    /// Construct opaque, unverified proof material for tests and future
    /// verifier plumbing.
    ///
    /// There is intentionally no `verify` method here: successful verification
    /// must happen inside the decider circuit after recomputing the terminal CE
    /// public statement from authoritative wires.
    pub fn new_unchecked(public_digest: [F; 4], bytes: Vec<u8>) -> Self {
        Self { public_digest, bytes }
    }

    pub fn public_digest(&self) -> [F; 4] {
        self.public_digest
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum TerminalCeVerifyError {
    #[error("compact terminal CE public statement is malformed: {0}")]
    PublicStatement(String),
    #[error("compact terminal CE proof verification is not implemented; keep using direct terminal CE rows")]
    Unsupported,
}
