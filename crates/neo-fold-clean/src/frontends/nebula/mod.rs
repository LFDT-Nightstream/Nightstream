//! Nebula memory-checking frontend — offline memory checking (RAM + public
//! ROM) for the SuperNeo folding chain.
//!
//! Normative spec: `specs/nebula-superneo-implementation.md` (v3); security
//! argument: `specs/nebula-superneo-security-note.md`. Section references in
//! this module (`spec §N`) point there.
//!
//! Owns the frontend-side building blocks, one concern per file:
//!
//! - [`layout`] — plan parameters and every bit-level encoding (lanes, step
//!   public input). Spec §2, §3, §4.4.
//! - [`fingerprint`] — the packed public-coin fingerprint and running
//!   products over `K`. Spec §4.3.
//! - [`trace`] — the native memory machine: sequential-consistency
//!   semantics, RS/WS tuple emission, IS/FS snapshots. This is the first
//!   pass of the two-pass prover (spec §1) and the test oracle; it is
//!   never verifier authority.
//! - [`circuit`] — the uniform `S_mem` step circuit (spec §4): one CCS
//!   structure per plan, plus the step witness builder.
//!
//! Does not own (later spec §13 steps): lane commitments on the claim, the
//! F′ `NebulaLane` carry, or the segment prover.

pub mod circuit;
pub mod fingerprint;
pub mod layout;
pub mod trace;
