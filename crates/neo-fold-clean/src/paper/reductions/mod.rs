//! The three SuperNeo NIFS sub-protocols, in §7.3 → §7.4 → §7.5 chain order.
//!
//! ```text
//! Π_CCS (§7.3)   K fresh CCS + k carried CE   →   K+k CE at point r'
//! Π_RLC (§7.4)   K+k CE                       →   1 CE of norm B
//! Π_DEC (§7.5)   1 CE of norm B               →   k CE of norm b
//! ```
//!
//! Each file owns one sub-protocol's `prove` / `verify` and its opaque
//! `Proof` wire-format type. The composition into NIFS lives in
//! `paper::nifs`; the IVC framing around NIFS lives in `paper::f_prime`
//! and `paper::construction2`.
//!
//! ## Soundness pairing
//!
//! - Π_CCS is **strong** wrt φ projecting commitments (Lemma 3, §D.4).
//! - Π_RLC is **weak** wrt the same φ (Lemma 4, §D.5).
//! - Π_DEC is a reduction of knowledge (Theorem 7, §D.6).
//!
//! Theorem 6 then composes Π_CCS (strong) with Π_RLC (weak) to a sound
//! NIFS; Π_DEC closes the loop by restoring norm bounds for the next round.

pub mod accumulator_sis_circuit;
pub mod pi_ccs;
pub mod pi_ccs_split_nc_circuit;
pub mod pi_dec;
pub mod pi_dec_circuit;
pub mod pi_rlc;
pub mod pi_rlc_circuit;
