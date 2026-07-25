//! Exact native contract regressions for the delayed F' public-input link.

use neo_fold_clean::paper::construction2::EncInst;
use neo_fold_clean::paper::digest::digest_fields_as_digest32;
use neo_fold_clean::paper::f_prime::r1cs::{
    encode_f_prime_public_input, encode_f_prime_superneo_public_input, f_prime_public_input_link_matches,
    f_prime_public_input_link_program, FPrimePublicInputLayout, FPrimePublicInputLinkInstruction,
    F_PRIME_ENC_INST_OFFSET, F_PRIME_PUBLIC_INPUT_LEN, F_PRIME_PUBLIC_ONE_OFFSET,
};
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

fn fixture() -> ([F; 4], EncInst) {
    let digest = [
        F::from_u64(0),
        F::from_u64(1),
        F::from_u64(0x0123_4567_89ab_cdef),
        F::from_u64(0xffff_ffff_0000_0000),
    ];
    let encoded = EncInst::from_digest(digest_fields_as_digest32(digest));
    (digest, encoded)
}

fn matches(layout: FPrimePublicInputLayout, expected: &EncInst, m_in: usize, x: &[F]) -> bool {
    f_prime_public_input_link_matches(layout, expected, layout.total_len(), m_in, x)
}

#[test]
fn plain_link_program_has_the_exact_ordered_owner_schedule() {
    let layout = FPrimePublicInputLayout::plain();
    assert_eq!(
        f_prime_public_input_link_program(layout),
        [
            FPrimePublicInputLinkInstruction::ExpectedPublicInputLen { expected: 270 },
            FPrimePublicInputLinkInstruction::ClaimMIn { expected: 270 },
            FPrimePublicInputLinkInstruction::ClaimXLen { expected: 270 },
            FPrimePublicInputLinkInstruction::AffineOne { claim_index: 0 },
            FPrimePublicInputLinkInstruction::BodyRange {
                expected_offset: 0,
                claim_offset: 1,
                len: 256,
            },
            FPrimePublicInputLinkInstruction::PaddingZeroRange {
                claim_offset: 257,
                len: 13,
            },
        ]
    );
}

#[test]
fn plain_link_accepts_exact_canonical_carrier_and_rejects_every_coordinate_mutation() {
    let (digest, expected) = fixture();
    let layout = FPrimePublicInputLayout::plain();
    let canonical = encode_f_prime_superneo_public_input(digest);

    assert_eq!(canonical.len(), layout.total_len());
    assert!(matches(layout, &expected, layout.total_len(), &canonical));

    let mut wrong_m_in = layout.total_len() - 1;
    assert!(!matches(layout, &expected, wrong_m_in, &canonical));
    wrong_m_in += 2;
    assert!(!matches(layout, &expected, wrong_m_in, &canonical));

    let mut short = canonical.clone();
    short.pop();
    assert!(!matches(layout, &expected, layout.total_len(), &short));
    let mut long = canonical.clone();
    long.push(F::ZERO);
    assert!(!matches(layout, &expected, layout.total_len(), &long));

    let mut wrong_one = canonical.clone();
    wrong_one[F_PRIME_PUBLIC_ONE_OFFSET] = F::ZERO;
    assert!(!matches(layout, &expected, layout.total_len(), &wrong_one));

    for coordinate in F_PRIME_ENC_INST_OFFSET..F_PRIME_PUBLIC_INPUT_LEN {
        let mut mutated = canonical.clone();
        mutated[coordinate] = if mutated[coordinate] == F::ZERO {
            F::ONE
        } else {
            F::ZERO
        };
        assert!(
            !matches(layout, &expected, layout.total_len(), &mutated),
            "body coordinate {coordinate} was not checked"
        );
    }

    for coordinate in layout.carrier_padding_offset()..layout.total_len() {
        let mut mutated = canonical.clone();
        mutated[coordinate] = F::ONE;
        assert!(
            !matches(layout, &expected, layout.total_len(), &mutated),
            "padding coordinate {coordinate} was not checked"
        );
    }

    assert!(!f_prime_public_input_link_matches(
        layout,
        &expected,
        layout.total_len() - 1,
        layout.total_len(),
        &canonical,
    ));
}

#[test]
fn composed_suffix_is_separate_but_post_suffix_padding_is_checked() {
    let (digest, expected) = fixture();
    let layout = FPrimePublicInputLayout::with_suffix(3);
    let mut claim = encode_f_prime_public_input(digest);
    claim.extend([F::from_u64(7), F::from_u64(11), F::from_u64(13)]);
    claim.resize(layout.total_len(), F::ZERO);

    assert_eq!(layout.suffix_offset(), F_PRIME_PUBLIC_INPUT_LEN);
    assert_eq!(layout.suffix_end(), layout.carrier_padding_offset());
    assert_eq!(
        f_prime_public_input_link_program(layout)[5],
        FPrimePublicInputLinkInstruction::PaddingZeroRange {
            claim_offset: 260,
            len: 10,
        }
    );
    assert!(matches(layout, &expected, layout.total_len(), &claim));

    for coordinate in layout.suffix_offset()..layout.suffix_end() {
        let mut changed_suffix = claim.clone();
        changed_suffix[coordinate] += F::ONE;
        assert!(
            matches(layout, &expected, layout.total_len(), &changed_suffix),
            "link predicate improperly claimed suffix coordinate {coordinate}"
        );
    }

    for coordinate in layout.carrier_padding_offset()..layout.total_len() {
        let mut changed_padding = claim.clone();
        changed_padding[coordinate] = F::ONE;
        assert!(
            !matches(layout, &expected, layout.total_len(), &changed_padding),
            "post-suffix padding coordinate {coordinate} was not checked"
        );
    }
}
