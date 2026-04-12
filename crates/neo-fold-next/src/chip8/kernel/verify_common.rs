//! Owns shared verifier-side equality checks, sumcheck replay, and opening-manifest validation.

use neo_math::{from_complex, KExtensions, K};
use neo_reductions::sumcheck::{verify_sumcheck_rounds, SUMCHECK_CHALLENGE_LABEL, SUMCHECK_ROUND_COEFF_LABEL};
use neo_transcript::Transcript;
use p3_field::PrimeCharacteristicRing;

use crate::chip8::spec::CommitmentId;

use super::{
    is_kernel_commitment_id, is_root_commitment_id, kernel_opening_claim_cmp, normalize_polynomial_ids,
    KernelOpeningClaim, KernelOpeningManifest, KernelOpeningSource, RootOpeningManifest, SimpleKernelError,
};

pub(crate) fn expect_digest32(actual: [u8; 32], expected: [u8; 32], label: &str) -> Result<(), SimpleKernelError> {
    if actual == expected {
        Ok(())
    } else {
        Err(SimpleKernelError::OpeningFailed(format!("{label} mismatch")))
    }
}

pub(crate) fn sample_sumcheck_challenge<Tr: Transcript>(transcript: &mut Tr) -> K {
    let pair = transcript.challenge_fields(SUMCHECK_CHALLENGE_LABEL, 2);
    from_complex(pair[0], pair[1])
}

pub(crate) fn replay_sumcheck_unchecked<Tr: Transcript>(
    transcript: &mut Tr,
    degree_bound: usize,
    rounds: &[Vec<K>],
    label: &str,
) -> Result<Vec<K>, SimpleKernelError> {
    let mut challenges = Vec::with_capacity(rounds.len());
    for (round_idx, round) in rounds.iter().enumerate() {
        if round.len() > degree_bound + 1 {
            return Err(SimpleKernelError::SumcheckFailed(format!(
                "{label} round {round_idx} exceeds degree bound {degree_bound}"
            )));
        }
        let mut packed = Vec::with_capacity(round.len() * 2);
        for coeff in round {
            packed.extend(coeff.as_coeffs());
        }
        transcript.append_fields(SUMCHECK_ROUND_COEFF_LABEL, &packed);
        challenges.push(sample_sumcheck_challenge(transcript));
    }
    Ok(challenges)
}

pub(crate) fn verify_sumcheck_known<Tr: Transcript>(
    transcript: &mut Tr,
    degree_bound: usize,
    initial_sum: K,
    rounds: &[Vec<K>],
    label: &str,
) -> Result<Vec<K>, SimpleKernelError> {
    let (challenges, _, ok) = verify_sumcheck_rounds(transcript, degree_bound, initial_sum, rounds);
    if ok {
        Ok(challenges)
    } else {
        Err(SimpleKernelError::SumcheckFailed(format!(
            "{label} sumcheck verification failed"
        )))
    }
}

pub(crate) fn verify_sumcheck_known_with_terminal<Tr: Transcript>(
    transcript: &mut Tr,
    degree_bound: usize,
    initial_sum: K,
    rounds: &[Vec<K>],
    label: &str,
) -> Result<(Vec<K>, K), SimpleKernelError> {
    let (challenges, final_value, ok) = verify_sumcheck_rounds(transcript, degree_bound, initial_sum, rounds);
    if ok {
        Ok((challenges, final_value))
    } else {
        Err(SimpleKernelError::SumcheckFailed(format!(
            "{label} sumcheck verification failed"
        )))
    }
}

pub(crate) fn expect_equal_k_slice(actual: &[K], expected: &[K], label: &str) -> Result<(), SimpleKernelError> {
    if actual == expected {
        Ok(())
    } else {
        Err(SimpleKernelError::OpeningFailed(format!("{label} mismatch")))
    }
}

pub(crate) fn expect_equal_k(actual: K, expected: K, label: &str) -> Result<(), SimpleKernelError> {
    if actual == expected {
        Ok(())
    } else {
        Err(SimpleKernelError::OpeningFailed(format!("{label} mismatch")))
    }
}

pub(crate) fn batch_values(values: &[K], gamma: K) -> K {
    let mut acc = K::ZERO;
    let mut gamma_power = K::ONE;
    for &value in values {
        acc += gamma_power * value;
        gamma_power *= gamma;
    }
    acc
}

pub(crate) fn split_round_groups<'a>(
    rounds: &'a [Vec<K>],
    first_len: usize,
    second_len: usize,
    third_len: usize,
    label: &str,
) -> Result<(&'a [Vec<K>], &'a [Vec<K>], &'a [Vec<K>]), SimpleKernelError> {
    let expected = first_len + second_len + third_len;
    if rounds.len() != expected {
        return Err(SimpleKernelError::SumcheckFailed(format!(
            "{label} round count {} != expected {expected}",
            rounds.len()
        )));
    }
    let (first, rest) = rounds.split_at(first_len);
    let (second, third) = rest.split_at(second_len);
    Ok((first, second, third))
}

fn assert_opening_manifest_canonical(
    claims: &[KernelOpeningClaim],
    digest: [u8; 32],
    expected_digest: [u8; 32],
    expected_source: KernelOpeningSource,
    manifest_label: &str,
) -> Result<(), SimpleKernelError> {
    expect_digest32(digest, expected_digest, &format!("{manifest_label} digest"))?;

    for claim in claims {
        let claim_allowed = match expected_source {
            KernelOpeningSource::Kernel => {
                claim.source == KernelOpeningSource::Kernel && is_kernel_commitment_id(claim.commitment_id)
            }
            KernelOpeningSource::Root => {
                claim.source == KernelOpeningSource::Root && is_root_commitment_id(claim.commitment_id)
            }
        };
        if !claim_allowed {
            return Err(SimpleKernelError::OpeningFailed(format!(
                "{manifest_label} contains mis-owned claim"
            )));
        }
        if claim.polynomial_ids.is_empty() {
            return Err(SimpleKernelError::OpeningFailed(format!(
                "{manifest_label} contains claim with empty polynomial_ids"
            )));
        }
        if !claim
            .polynomial_ids
            .windows(2)
            .all(|pair| pair[0] < pair[1])
        {
            return Err(SimpleKernelError::OpeningFailed(format!(
                "{manifest_label} contains claim with unsorted or duplicate polynomial_ids"
            )));
        }
        expect_digest32(claim.digest, claim.expected_digest(), "opening-claim digest")?;
    }

    for window in claims.windows(2) {
        let lhs = &window[0];
        let rhs = &window[1];
        if kernel_opening_claim_cmp(lhs, rhs).is_gt() {
            return Err(SimpleKernelError::OpeningFailed(format!(
                "{manifest_label} is not in canonical order"
            )));
        }
        if lhs.source == rhs.source
            && lhs.commitment_id == rhs.commitment_id
            && lhs.point == rhs.point
            && lhs.polynomial_ids == rhs.polynomial_ids
        {
            return Err(SimpleKernelError::OpeningFailed(format!(
                "{manifest_label} contains duplicate claims"
            )));
        }
    }
    Ok(())
}

pub(crate) fn assert_manifest_canonical(manifest: &KernelOpeningManifest) -> Result<(), SimpleKernelError> {
    assert_opening_manifest_canonical(
        &manifest.claims,
        manifest.digest,
        manifest.expected_digest(),
        KernelOpeningSource::Kernel,
        "kernel opening manifest",
    )
}

pub(crate) fn assert_root_manifest_canonical(manifest: &RootOpeningManifest) -> Result<(), SimpleKernelError> {
    assert_opening_manifest_canonical(
        &manifest.claims,
        manifest.digest,
        manifest.expected_digest(),
        KernelOpeningSource::Root,
        "root opening manifest",
    )
}

pub(crate) fn find_manifest_claim<'a>(
    manifest: &'a KernelOpeningManifest,
    commitment_id: CommitmentId,
    point: &[K],
    polynomial_ids: &[usize],
    label: &str,
) -> Result<&'a KernelOpeningClaim, SimpleKernelError> {
    let normalized_polynomial_ids = normalize_polynomial_ids(polynomial_ids);
    let mut matches = manifest.claims.iter().filter(|claim| {
        claim.commitment_id == commitment_id
            && claim.point == point
            && claim.polynomial_ids == normalized_polynomial_ids
    });
    let claim = matches
        .next()
        .ok_or_else(|| SimpleKernelError::OpeningFailed(format!("{label} missing from kernel opening manifest")))?;
    if matches.next().is_some() {
        return Err(SimpleKernelError::OpeningFailed(format!(
            "{label} appears multiple times in kernel opening manifest"
        )));
    }
    Ok(claim)
}
