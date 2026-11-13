//! The verifier of the argument.

use std::time::{Duration, Instant};

use neo_math::K;
use neo_transcript::Poseidon2Transcript;
use p3_field::PrimeCharacteristicRing;

use crate::akita::verifier::{Opening, Verifier};
use crate::layout::{LOOKUPS, LOOKUP_VARS};
use crate::transcript::{absorb_bytes, absorb_extension, challenges as draw, session};
use crate::weights;
use crate::{
    absorb_statements, challenges, claimed_sum, inverse_point, layout, lookup_challenges, sumcheck, Error, Proof,
    Statement, INVERSE_SESSION, WITNESS_SESSION,
};

/// Where the verifier's time goes.
#[derive(Clone, Copy, Debug, Default)]
pub struct VerifierTimes {
    /// The public weights at the point of the sum-check.
    pub weights: Duration,
    /// The Akita setup and both openings.
    pub akita: Duration,
}

/// Accept `proof` exactly when it proves each of `statements`. The transcript
/// must hold the same messages as the prover's when `Committed::prove` began.
pub fn verify(
    transcript: &mut Poseidon2Transcript,
    statements: &[Statement<'_>],
    proof: &Proof,
) -> Result<VerifierTimes, Error> {
    let layout = layout(statements)?;
    if proof.rounds.len() != layout.witness_vars {
        return Err(Error::Rejected("the sum-check has another number of rounds"));
    }
    absorb_statements(transcript, statements);
    absorb_bytes(transcript, &proof.witness_commitment);
    let taus = lookup_challenges(transcript);
    absorb_bytes(transcript, &proof.inverse_commitment);
    let challenges = challenges(transcript, &layout, taus);
    let (rho, value) = sumcheck::verify(transcript, &proof.rounds, claimed_sum(statements, &challenges))?;
    absorb_extension(transcript, &[proof.witness_value]);
    absorb_extension(transcript, &proof.inverse_values);
    let column = draw(transcript, LOOKUP_VARS);

    let start = Instant::now();
    let inverse_value = weights::equality(&column, proof.inverse_values.len())
        .iter()
        .zip(&proof.inverse_values)
        .map(|(&weight, &value)| weight * value)
        .sum();
    let akita = Verifier::new(&[layout.witness_vars, layout.inverse_vars])?;
    akita.verify(&Opening {
        commitment: &proof.witness_commitment,
        point: &rho,
        value: proof.witness_value,
        proof: &proof.witness_opening,
        session: &session(transcript, WITNESS_SESSION),
    })?;
    akita.verify(&Opening {
        commitment: &proof.inverse_commitment,
        point: &inverse_point(&layout, &rho, &column),
        value: inverse_value,
        proof: &proof.inverse_opening,
        session: &session(transcript, INVERSE_SESSION),
    })?;
    let akita = start.elapsed();

    let start = Instant::now();
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
    let weights = weights::evaluate(statements, &layout, &challenges, &rho)?;
    if value != proof.witness_value * (weights - check * inverse) + check * scaled + sum * inverse {
        return Err(Error::Rejected("the sum-check ends in another value"));
    }
    Ok(VerifierTimes {
        weights: start.elapsed(),
        akita,
    })
}
