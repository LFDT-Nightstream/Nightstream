//! Check the actual R witness split and full optimized PiDEC execution.
//! Independent selected-key and matrix checks own the supplied openings.
//! This phase checks every native private digit and the complete public result.

use std::{fs, path::Path, time::Instant};

use neo_ajtai::Commitment;
use neo_ccs::{CcsStructure, CeClaim, Mat, V1_1Evaluations};
use neo_fold_clean::paper::{params::Params, pi_dec, relations::ajtai_dec_mixer};
use neo_math::{D, F, K};
use neo_reductions::common::split_b_matrix_k_with_nonzero_flags;
use p3_field::PrimeCharacteristicRing;
use rayon::prelude::*;
use serde_json::{json, Value};

use super::native_driver::{canonical, fields, public_words, running_claims, words};

#[path = "../../../tests/nifs/pi_dec_actual_mutations.rs"]
mod mutations;

type Claim = CeClaim<Commitment, F, K>;
const CHILDREN: usize = 16;
const MATRICES: usize = 14;
const PUBLIC: usize = 270;
const MODULUS: u64 = 0xffff_ffff_0000_0001;
const BOUND: u64 = 1 << CHILDREN;

fn claim_value(claim: &Claim, combined: bool) -> Value {
    assert_eq!((claim.c.d, claim.c.kappa, claim.m_in), (D, 22, PUBLIC));
    assert_eq!(claim.c.data.len(), 22 * D);
    assert_eq!(claim.r.len(), 28);
    assert_eq!(claim.eval_k.len(), D.next_power_of_two());
    assert_eq!(claim.eval_a.len(), MATRICES);
    assert!(claim.eval_k[D..].iter().all(|&value| value == K::ZERO));
    for family in &claim.eval_a {
        assert_eq!(family.len(), D.next_power_of_two());
        assert!(family[D..].iter().all(|&value| value == K::ZERO));
    }
    json!([
        fields(&claim.c.data),
        public_words(&claim.X),
        words(&claim.r),
        words(&claim.eval_k[..D]),
        claim
            .eval_a
            .iter()
            .map(|family| words(&family[..D]))
            .collect::<Vec<_>>(),
        u64::from(combined)
    ])
}

fn running_value(children: &[Claim]) -> Value {
    assert_eq!(children.len(), CHILDREN);
    json!([
        words(&children[0].r),
        children
            .iter()
            .map(|child| fields(&child.c.data))
            .collect::<Vec<_>>(),
        children
            .iter()
            .map(|child| public_words(&child.X))
            .collect::<Vec<_>>(),
        children
            .iter()
            .map(|child| words(&child.eval_k[..D]))
            .collect::<Vec<_>>(),
        children
            .iter()
            .map(|child| child
                .eval_a
                .iter()
                .map(|family| words(&family[..D]))
                .collect::<Vec<_>>())
            .collect::<Vec<_>>()
    ])
}

fn magnitude(word: u64) -> u64 {
    word.min(MODULUS - word)
}

fn result_value(
    params: &Params,
    structure: &CcsStructure<F>,
    parent: &Claim,
    children: &[Claim],
    state: [F; 8],
) -> Value {
    let proof = pi_dec::Proof {
        children: children.to_vec(),
    };
    let accepted = pi_dec::verify(params, structure, ajtai_dec_mixer, parent, &proof).is_ok();
    let (digits, _) =
        split_b_matrix_k_with_nonzero_flags(&parent.X, CHILDREN, 2).expect("bounded actual public parent");
    let public_digits = digits.iter().map(public_words).collect::<Vec<_>>();
    let parent_bounds = public_words(&parent.X)
        .into_iter()
        .map(|word| u64::from(magnitude(word) < BOUND))
        .collect::<Vec<_>>();
    let ranges = public_digits
        .iter()
        .map(|digit| {
            digit
                .iter()
                .map(|&word| u64::from(magnitude(word) < 2))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let recomposed_c = ajtai_dec_mixer(
        &children
            .iter()
            .map(|child| child.c.clone())
            .collect::<Vec<_>>(),
        2,
    );
    let mut recomposed_x = vec![F::ZERO; PUBLIC];
    let mut recomposed_k = vec![K::ZERO; D];
    let mut recomposed_a = vec![vec![K::ZERO; D]; MATRICES];
    for (index, child) in children.iter().enumerate() {
        let weight = F::from_u64(1 << index);
        let extension_weight = K::from(weight);
        for (coordinate, value) in recomposed_x.iter_mut().enumerate() {
            *value += digits[index][(coordinate % D, coordinate / D)] * weight;
        }
        for coordinate in 0..D {
            recomposed_k[coordinate] += child.eval_k[coordinate] * extension_weight;
            for matrix in 0..MATRICES {
                recomposed_a[matrix][coordinate] += child.eval_a[matrix][coordinate] * extension_weight;
            }
        }
    }
    // Public outputs belong to the verifier's canonical split and shared point.
    let computed_children = children
        .iter()
        .zip(digits)
        .map(|(child, public)| {
            let mut computed = child.clone();
            computed.X = public;
            computed.r = parent.r.clone();
            computed
        })
        .collect::<Vec<_>>();
    let mut unbounded = parent.clone();
    unbounded.X[(0, 0)] = F::from_u64(BOUND);
    let unbounded_rejected = split_b_matrix_k_with_nonzero_flags(&unbounded.X, CHILDREN, 2).is_err()
        && pi_dec::verify(params, structure, ajtai_dec_mixer, &unbounded, &proof).is_err();
    json!([
        u64::from(accepted),
        u64::from(parent_bounds.iter().all(|&check| check == 1)),
        public_digits,
        parent_bounds,
        ranges,
        fields(&recomposed_c.data),
        u64::from(recomposed_c == parent.c),
        fields(&recomposed_x),
        u64::from(fields(&recomposed_x) == public_words(&parent.X)),
        words(&recomposed_k),
        u64::from(recomposed_k == parent.eval_k[..D]),
        recomposed_a
            .iter()
            .map(|family| words(family))
            .collect::<Vec<_>>(),
        u64::from(
            recomposed_a
                .iter()
                .zip(&parent.eval_a)
                .all(|(left, right)| left == &right[..D])
        ),
        computed_children
            .iter()
            .map(|child| claim_value(child, false))
            .collect::<Vec<_>>(),
        fields(&state),
        u64::from(unbounded_rejected),
        if accepted {
            json!([1, running_value(&computed_children)])
        } else {
            json!([0])
        }
    ])
}

fn check_private_digits(witness: &Mat<F>, raw: &[u8]) -> (Vec<Mat<F>>, Vec<bool>) {
    let started = Instant::now();
    assert_eq!(raw.len(), 2 * witness.rows() * witness.cols());
    assert_eq!(witness.rows(), D);
    let (digits, flags) =
        split_b_matrix_k_with_nonzero_flags(witness, CHILDREN, 2).expect("actual optimized native R witness split");
    let mask = raw
        .par_chunks_exact(2)
        .map(|word| i16::from_le_bytes(word.try_into().unwrap()).unsigned_abs())
        .reduce(|| 0, |a, b| a | b);
    assert_eq!(flags.len(), CHILDREN);
    assert_eq!(digits.len(), CHILDREN);
    digits.par_iter().enumerate().for_each(|(child, digit)| {
        assert_eq!((digit.rows(), digit.cols()), (witness.rows(), witness.cols()));
        let expected_nonzero = mask & (1 << child) != 0;
        assert_eq!(flags[child], expected_nonzero);
        if !expected_nonzero {
            assert_eq!(digit.virtual_constant_value(), Some(&F::ZERO));
            return;
        }
        (0..witness.rows() * witness.cols())
            .into_par_iter()
            .for_each(|index| {
                let row = index / witness.cols();
                let column = index % witness.cols();
                let offset = 2 * (column * D + row);
                let value = i16::from_le_bytes(raw[offset..offset + 2].try_into().unwrap());
                let bit = F::from_u64(u64::from((value.unsigned_abs() >> child) & 1));
                assert_eq!(
                    digit[(row, column)],
                    if value < 0 { -bit } else { bit },
                    "actual PiDEC private digit {child} ({row}, {column})"
                );
            });
    });
    println!(
        "actual_native_pi_dec_split=passed planes={CHILDREN} private_coordinates={} nonzero_planes={} elapsed={:?}",
        witness.rows() * witness.cols(),
        flags.iter().filter(|&&flag| flag).count(),
        started.elapsed()
    );
    (digits, flags)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn prove(
    params: &Params,
    structure: &CcsStructure<F>,
    parent: &Claim,
    witness: Mat<F>,
    raw: &[u8],
    state: [F; 8],
    messages_path: &Path,
    reference_input: &Value,
    reference_result: &Value,
    output: &Path,
) {
    let started = Instant::now();
    assert!(!output.exists(), "fresh native PiDEC output");
    assert_eq!((params.b(), params.k_rho()), (2, 16));
    let messages: Value = serde_json::from_slice(&fs::read(messages_path).expect("actual PiDEC child messages"))
        .expect("numeric PiDEC running output");
    canonical(&messages);
    let digest = std::array::from_fn(|lane| {
        u64::from_le_bytes(
            parent.fold_digest[lane * 8..lane * 8 + 8]
                .try_into()
                .unwrap(),
        )
    });
    let supplied = running_claims(&messages, digest);
    let input = json!([
        claim_value(parent, true),
        messages[1],
        messages[3],
        messages[4],
        messages[2],
        messages[0],
        fields(&state)
    ]);
    assert!(
        input == *reference_input,
        "complete actual R-to-D handoff and supplied messages"
    );
    pi_dec::verify(
        params,
        structure,
        ajtai_dec_mixer,
        parent,
        &pi_dec::Proof {
            children: supplied.clone(),
        },
    )
    .expect("actual supplied PiDEC messages accepted");
    let (digits, flags) = check_private_digits(&witness, raw);
    drop(witness);
    let commitments = supplied.iter().map(|child| child.c.clone()).collect();
    let openings = supplied
        .iter()
        .map(|child| V1_1Evaluations {
            eval_k: child.eval_k[..D].to_vec(),
            eval_a: child
                .eval_a
                .iter()
                .map(|family| family[..D].to_vec())
                .collect(),
        })
        .collect();
    let (children, proof) = pi_dec::prove_from_split_material(
        params,
        structure,
        None,
        None,
        ajtai_dec_mixer,
        parent,
        digits,
        flags,
        commitments,
        openings,
    )
    .expect("actual native PiDEC prover from checked split and openings");
    assert_eq!(children.claims, supplied, "all actual native PiDEC prover children");
    let verified = pi_dec::verify(params, structure, ajtai_dec_mixer, parent, &proof)
        .expect("verify the actual native PiDEC proof");
    assert_eq!(verified, children.claims);
    let result = result_value(params, structure, parent, &verified, state);
    let expected = reference_result
        .as_array()
        .expect("complete Lean PiDEC result");
    assert_eq!(expected.len(), 17);
    for (index, (actual, expected)) in result.as_array().unwrap().iter().zip(expected).enumerate() {
        assert!(actual == expected, "complete PiDEC result field {index}");
    }
    assert_eq!(result[0], 1);
    assert_eq!(result[15], 1);
    let generated = running_value(&verified);
    assert!(generated == messages, "complete actual native final running output");
    mutations::check(params, structure, parent, &verified);
    let mut encoded = serde_json::to_vec(&generated).expect("complete native PiDEC output");
    encoded.push(b'\n');
    fs::write(output, encoded).expect("native PiDEC sink");
    println!(
        "complete_native_pi_dec=passed children={} result_fields={} proof={} elapsed={:?}",
        verified.len(),
        expected.len(),
        output.display(),
        started.elapsed()
    );
}
