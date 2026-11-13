//! The verifier of the argument.

use std::time::{Duration, Instant};

use neo_math::K;
use neo_transcript::Poseidon2Transcript;
use p3_field::PrimeCharacteristicRing;

use crate::akita::verifier::Opening;
use crate::key::{self, VerificationKey};
use crate::layout::{LANE_VARS, LOOKUPS, LOOKUP_VARS};
use crate::transcript::{absorb_bytes, absorb_extension, challenges as draw, session};
use crate::weights::{self, Uses};
use crate::{
    absorb_statements, challenges, claimed_sum, inverse_point, layout, lookup_challenges, sumcheck, Error, Proof,
    Statement, SESSION,
};

/// Where the verifier's time goes.
#[derive(Clone, Copy, Debug, Default)]
pub struct VerifierTimes {
    /// The part of the public weights that the matrices give: one pass over
    /// the rows of each claim.
    pub matrices: Duration,
    /// Each other part of the public weights and both sum-checks.
    pub rest: Duration,
    /// The prefix of Akita's setup, from Akita's seed: once for each key
    /// and schedule row.
    pub akita_setup: Duration,
    pub akita: Duration,
}

/// Accept `proof` exactly when it proves each of `statements` under `key`.
/// The transcript must hold the same messages as the prover's when
/// `Committed::prove` began. The verifier reads no element of the Ajtai key.
pub fn verify(
    transcript: &mut Poseidon2Transcript,
    key: &VerificationKey,
    statements: &[Statement<'_>],
    proof: &Proof,
) -> Result<VerifierTimes, Error> {
    let start = Instant::now();
    let layout = layout(statements, key.blocks())?;
    if proof.rounds.len() != layout.witness_vars || proof.key_rounds.len() != LANE_VARS + layout.key_block_vars() {
        return Err(Error::Rejected("a sum-check has another number of rounds"));
    }
    absorb_statements(transcript, key, statements);
    absorb_bytes(transcript, &proof.witness_commitment);
    let taus = lookup_challenges(transcript);
    absorb_bytes(transcript, &proof.inverse_commitment);
    let challenges = challenges(transcript, &layout, taus);
    let (row_point, scale) = key::row_point(challenges.omega)?;
    let (rho, value) = sumcheck::verify(transcript, &proof.rounds, claimed_sum(statements, &challenges))?;
    absorb_extension(transcript, &[proof.witness_value]);
    absorb_extension(transcript, &proof.inverse_values);
    absorb_extension(transcript, &[proof.key_weight]);
    let column = draw(transcript, LOOKUP_VARS);
    let (mut key_point, key_claim) = sumcheck::verify(transcript, &proof.key_rounds, proof.key_weight)?;
    absorb_extension(transcript, &[proof.key_value]);
    let session = session(transcript, SESSION);

    let uses = Uses::new(&layout, &challenges, &rho);
    if key_claim != scale * proof.key_value * key::weight_at(&uses, &key_point) {
        return Err(Error::Rejected("the key sum-check ends in another value"));
    }
    key_point.extend(row_point);

    // Each lookup table of the sum-check is zero from cell 2^span_vars.
    let low: K = rho[layout.span_vars..]
        .iter()
        .map(|&coordinate| K::ONE - coordinate)
        .product();
    let combine = |weights: [K; LOOKUPS]| -> K {
        low * weights
            .iter()
            .zip(&proof.inverse_values)
            .map(|(&weight, &value)| weight * value)
            .sum::<K>()
    };
    let inverse = combine(challenges.psi);
    let scaled = combine(std::array::from_fn(|k| challenges.psi[k] * challenges.taus[k]));
    let (mut check, mut sum) = (K::ZERO, K::ZERO);
    for (table, (_, slice, _, _)) in layout.tables().enumerate() {
        let selector = slice.selector(&rho);
        let equality: K = challenges.beta[..slice.vars]
            .iter()
            .zip(&rho)
            .map(|(&beta, &rho)| beta * rho + (K::ONE - beta) * (K::ONE - rho))
            .product();
        check += challenges.xi[table] * equality * selector;
        sum += challenges.lambda[table] * selector;
    }
    let closed = weights::closed_part(statements, &layout, &challenges, &uses, &rho)?;
    let rest = start.elapsed();

    let start = Instant::now();
    let matrices = weights::matrix_part(statements, &challenges, &uses)?;
    let weights = proof.key_weight + matrices + closed;
    if value != proof.witness_value * (weights - check * inverse) + check * scaled + sum * inverse {
        return Err(Error::Rejected("the sum-check ends in another value"));
    }
    let matrices = start.elapsed();

    let start = Instant::now();
    let akita = key.akita(layout.shape())?;
    let akita_setup = start.elapsed();
    let start = Instant::now();
    let inverse_value = weights::equality(&column, proof.inverse_values.len())
        .iter()
        .zip(&proof.inverse_values)
        .map(|(&weight, &value)| weight * value)
        .sum();
    akita.verify(
        [
            Opening {
                commitment: key.commitment(),
                point: &key_point,
                value: proof.key_value,
            },
            Opening {
                commitment: &proof.witness_commitment,
                point: &rho,
                value: proof.witness_value,
            },
            Opening {
                commitment: &proof.inverse_commitment,
                point: &inverse_point(&layout, &rho, &column),
                value: inverse_value,
            },
        ],
        &proof.opening,
        &session,
    )?;
    Ok(VerifierTimes {
        matrices,
        rest,
        akita_setup,
        akita: start.elapsed(),
    })
}
