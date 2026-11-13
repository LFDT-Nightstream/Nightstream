//! A domain label packs eight bytes into each word. `q = 2^64 − 59`, so the 59 words from `q` to
//! `2^64 − 1` would alias the words 0 to 58. A label with such a word is rejected.

use neo_math::F;
use neo_transcript::{domain_chunk_v1_2, fold_domain_chunk_v1_2, prior_digest_v1_2};
use p3_field::{PrimeCharacteristicRing, PrimeField64};

fn unpack(chunk: &[F]) -> Vec<u8> {
    chunk
        .iter()
        .flat_map(|word| word.as_canonical_u64().to_le_bytes())
        .collect()
}

#[test]
fn the_fold_label_survives_the_packing() {
    let label = b"Nightstream/SuperNeo/fold/v2";
    let bytes = unpack(&fold_domain_chunk_v1_2());
    assert_eq!(&bytes[..label.len()], label);
    assert!(bytes[label.len()..].iter().all(|&byte| byte == 0));
}

#[test]
fn each_ascii_label_survives_the_packing() {
    let label: Vec<u8> = (0..96).map(|index| 0x7f - (index % 5) as u8).collect();
    assert_eq!(unpack(&domain_chunk_v1_2(&label)), label);
}

#[test]
fn the_largest_canonical_word_is_accepted() {
    let label = (F::ORDER_U64 - 1).to_le_bytes();
    assert_eq!(domain_chunk_v1_2(&label)[0], -F::ONE);
}

#[test]
#[should_panic(expected = "domain label word is not below q")]
fn a_word_that_aliases_zero_is_rejected() {
    domain_chunk_v1_2(&F::ORDER_U64.to_le_bytes());
}

#[test]
#[should_panic(expected = "domain label word is not below q")]
fn the_largest_word_is_rejected() {
    domain_chunk_v1_2(&[0xff; 16]);
}

#[test]
fn the_prior_digest_is_the_word_value_modulo_q() {
    let bits_of = |words: [u64; 4]| {
        let mut input = vec![F::ONE];
        for word in words {
            input.extend((0..64).map(|bit| F::from_u64(word >> bit & 1)));
        }
        input
    };
    let words = [0, 58, F::ORDER_U64 - 1, 0x0123_4567_89ab_cdef];
    assert_eq!(prior_digest_v1_2(&bits_of(words)), words.map(F::from_u64));
    assert_eq!(
        prior_digest_v1_2(&bits_of([F::ORDER_U64, u64::MAX, 0, 0])),
        prior_digest_v1_2(&bits_of([0, 58, 0, 0]))
    );
}
