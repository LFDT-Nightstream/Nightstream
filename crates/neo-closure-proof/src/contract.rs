//! Closure contract helpers shared across backends.
//!
//! This module centralizes:
//! - canonical digest computation used to bind Phase-2 closure proofs to Phase-1 obligations, and
//! - verification-time checks that the global seeded Ajtai public parameters match the statement.

#![forbid(unsafe_code)]

use neo_ajtai::Commitment as Cmt;
use neo_fold::shard::ShardObligations;
use neo_math::{F as NeoF, K as NeoK};
use neo_params::NeoParams;

/// Compute the canonical obligations digest bound by the Phase‑1 Spartan statement.
pub fn expected_obligations_digest(
    params: &NeoParams,
    obligations: &ShardObligations<Cmt, NeoF, NeoK>,
    pp_id_digest: [u8; 32],
) -> [u8; 32] {
    let acc_main =
        neo_fold::bridge_digests::compute_accumulator_digest_v2(params.b, obligations.main.as_slice());
    let acc_val =
        neo_fold::bridge_digests::compute_accumulator_digest_v2(params.b, obligations.val.as_slice());
    neo_fold::bridge_digests::compute_obligations_digest_v1(acc_main, acc_val, pp_id_digest)
}

/// Require that the globally loaded seeded Ajtai PP matches a statement’s `pp_id_digest`.
///
/// This enforces that the verifier is not accidentally checking against a different seeded PP
/// than the one bound by the proof statement.
pub fn require_global_pp_matches_statement(
    stmt_pp_id: [u8; 32],
    params: &NeoParams,
    d: usize,
    m: usize,
) -> Result<(usize, [u8; 32]), String> {
    let (kappa, seed) = neo_ajtai::get_global_pp_seeded_params_for_dims(d, m)
        .map_err(|e| format!("missing seeded PP for (d={d}, m={m}): {e:?}"))?;
    if kappa != params.kappa as usize {
        return Err(format!(
            "Ajtai κ mismatch: global κ={kappa}, params.kappa={}",
            params.kappa
        ));
    }
    let expected = neo_ajtai::compute_pp_id_digest_v1(d, m, kappa, seed);
    if expected != stmt_pp_id {
        return Err("pp_id_digest mismatch (statement not bound to the loaded seeded PP)".into());
    }
    Ok((kappa, seed))
}
