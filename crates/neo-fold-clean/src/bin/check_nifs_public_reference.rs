//! Compare public PaperExact R/D results with one retained Lean C/R/D envelope.
//! The caller selects the package identity and supplies the previously checked C
//! result. This tool does not run a prover, load witnesses, or execute a backend.

use std::{env, fs, path::Path, time::Instant};

use neo_ajtai::{scale_commitment_add_inplace, Commitment};
use neo_ccs::{CcsStructure, CeClaim, Mat, SparsePoly, Term};
use neo_fold_clean::{engine::paper_exact, paper::params::Params};
use neo_math::{from_complex, KExtensions, D, F, K};
use neo_reductions::engines::paper_exact_engine::rlc_claim_paper_exact_with_commit_mix;
use neo_transcript::Poseidon2Transcript;
use nightstream_fprime::{load_per_application_package, PackageCcsRelation};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use serde_json::{json, Value};

const MODULUS: u64 = 0xffff_ffff_0000_0001;
const SOURCES: usize = 17;
const CHILDREN: usize = 16;
const MATRICES: usize = 14;
const ROUNDS: usize = 28;
const PUBLIC: usize = 270;
const RANK: usize = 22;
const BOUND: u64 = 65_536;
type Claim = CeClaim<Commitment, F, K>;

fn array(value: &Value, expected: usize) -> &[Value] {
    let values = value.as_array().expect("numeric array");
    assert_eq!(values.len(), expected, "exact selected-profile array length");
    values
}

fn canonical(value: &Value) {
    match value {
        Value::Array(values) => values.iter().for_each(canonical),
        Value::Number(number) => {
            assert!(number.as_u64().is_some_and(|word| word < MODULUS));
        }
        _ => panic!("only canonical field words and arrays are permitted"),
    }
}

fn read_envelope(path: &Path) -> Value {
    let bytes = fs::read(path).expect("retained Lean C/R/D envelope");
    let value: Value = serde_json::from_slice(&bytes).expect("C/R/D JSON");
    canonical(&value);
    let encoded = serde_json::to_vec(&value).expect("canonical C/R/D encoding");
    assert!(
        bytes == encoded || bytes.strip_suffix(b"\n") == Some(encoded.as_slice()),
        "canonical numeric JSON with optional final newline"
    );
    array(&value, 10);
    assert_eq!(value[0], 1);
    array(&value[1], 7);
    assert_eq!(value[1][0], 2);
    array(&value[5], 15);
    assert_eq!(value[5][0], 1, "the supplied C record must be accepted");
    array(&value[6], 7);
    array(&value[7], 11);
    array(&value[8], 7);
    array(&value[9], 17);
    value
}

fn fields(value: &Value, count: usize) -> Vec<F> {
    array(value, count)
        .iter()
        .map(|word| F::from_u64(word.as_u64().expect("field word")))
        .collect()
}

fn extensions(value: &Value, count: usize) -> Vec<K> {
    array(value, count)
        .iter()
        .map(|pair| {
            let words = fields(pair, 2);
            from_complex(words[0], words[1])
        })
        .collect()
}

fn words(values: &[F]) -> Vec<u64> {
    values
        .iter()
        .map(|value| value.as_canonical_u64())
        .collect()
}

fn extension_words(values: &[K]) -> Vec<[u64; 2]> {
    values
        .iter()
        .map(|value| value.to_limbs_u64().into())
        .collect()
}

fn public_matrix(value: &Value) -> Mat<F> {
    let values = fields(value, PUBLIC);
    let mut output = Mat::zero(D, PUBLIC / D, F::ZERO);
    for (index, value) in values.into_iter().enumerate() {
        output[(index % D, index / D)] = value;
    }
    output
}

fn public_words(value: &Mat<F>) -> Vec<u64> {
    assert_eq!((value.rows(), value.cols()), (D, PUBLIC / D));
    (0..PUBLIC)
        .map(|index| value[(index % D, index / D)].as_canonical_u64())
        .collect()
}

fn evaluation(value: &Value) -> Vec<K> {
    let mut result = extensions(value, D);
    result.resize(D.next_power_of_two(), K::ZERO);
    result
}

fn decode_claim(value: &Value, combined: bool, digest: [u8; 32]) -> Claim {
    let value = array(value, 6);
    assert_eq!(value[5], u64::from(combined), "norm stage");
    Claim {
        c: Commitment {
            d: D,
            kappa: RANK,
            data: fields(&value[0], RANK * D),
        },
        X: public_matrix(&value[1]),
        r: extensions(&value[2], ROUNDS),
        eval_k: evaluation(&value[3]),
        eval_a: array(&value[4], MATRICES).iter().map(evaluation).collect(),
        m_in: PUBLIC,
        fold_digest: digest,
        adv: None,
    }
}

fn claim_value(claim: &Claim, combined: bool) -> Value {
    assert_eq!((claim.c.d, claim.c.kappa, claim.m_in), (D, RANK, PUBLIC));
    assert_eq!(claim.c.data.len(), RANK * D);
    assert_eq!(claim.r.len(), ROUNDS);
    assert_eq!(claim.eval_a.len(), MATRICES);
    for family in std::iter::once(&claim.eval_k).chain(&claim.eval_a) {
        assert_eq!(family.len(), D.next_power_of_two());
        assert!(family[D..].iter().all(|value| *value == K::ZERO));
    }
    assert!(claim.adv.is_none());
    json!([
        words(&claim.c.data),
        public_words(&claim.X),
        extension_words(&claim.r),
        extension_words(&claim.eval_k[..D]),
        claim
            .eval_a
            .iter()
            .map(|family| extension_words(&family[..D]))
            .collect::<Vec<_>>(),
        u64::from(combined)
    ])
}

fn running_value(children: &[Claim]) -> Value {
    assert_eq!(children.len(), CHILDREN);
    let values = children
        .iter()
        .map(|child| claim_value(child, false))
        .collect::<Vec<_>>();
    assert!(children.iter().all(|child| child.r == children[0].r));
    json!([
        values[0][2],
        values.iter().map(|value| &value[0]).collect::<Vec<_>>(),
        values.iter().map(|value| &value[1]).collect::<Vec<_>>(),
        values.iter().map(|value| &value[3]).collect::<Vec<_>>(),
        values.iter().map(|value| &value[4]).collect::<Vec<_>>()
    ])
}

fn relation(header: &PackageCcsRelation) -> CcsStructure<F> {
    assert_eq!(header.cube_variables(), ROUNDS);
    assert_eq!(header.matrix_sources().len(), MATRICES);
    assert_eq!(header.degree_bound(), 9);
    let polynomial = SparsePoly::new(
        MATRICES,
        header
            .terms()
            .iter()
            .map(|term| Term {
                coeff: F::from_u64(term.coefficient()),
                exps: term
                    .exponents()
                    .iter()
                    .map(|&n| u32::try_from(n).expect("CCS exponent"))
                    .collect(),
            })
            .collect(),
    );
    CcsStructure::new_verifier_artifact_header(1 << ROUNDS, header.column_count(), MATRICES, polynomial)
        .expect("the selected Lean relation header")
}

// Apply the sampled reference ring-action matrices directly to each public
// commitment. No native RLC commitment mixer computes this comparison result.
fn combine_commitments(rhos: &[Mat<F>], commitments: &[Commitment]) -> Commitment {
    assert_eq!(rhos.len(), commitments.len());
    let mut output = Commitment::zeros(D, RANK);
    for (rho, commitment) in rhos.iter().zip(commitments) {
        assert_eq!((rho.rows(), rho.cols()), (D, D));
        assert_eq!(
            (commitment.d, commitment.kappa, commitment.data.len()),
            (D, RANK, D * RANK)
        );
        for block in 0..RANK {
            for row in 0..D {
                for column in 0..D {
                    output.data[block * D + row] += rho[(row, column)] * commitment.data[block * D + column];
                }
            }
        }
    }
    output
}

fn nonzero(claim: &Claim) -> [u64; 3] {
    [
        u64::from(claim.c.data.iter().any(|value| *value != F::ZERO)),
        u64::from(public_words(&claim.X).iter().any(|value| *value != 0)),
        u64::from(
            claim
                .eval_k
                .iter()
                .chain(claim.eval_a.iter().flatten())
                .any(|value| *value != K::ZERO),
        ),
    ]
}

fn compare_result(label: &str, actual: &Value, expected: &Value, count: usize) {
    for (index, (actual, expected)) in array(actual, count)
        .iter()
        .zip(array(expected, count))
        .enumerate()
    {
        assert!(actual == expected, "{label} result field {index} differs");
    }
    assert_eq!(
        serde_json::to_vec(actual).expect("computed result bytes"),
        serde_json::to_vec(expected).expect("retained result bytes"),
        "{label} complete canonical result bytes"
    );
}

fn check_rlc(params: &Params, structure: &CcsStructure<F>, envelope: &Value, identity: [u64; 4]) -> (Claim, [F; 8]) {
    let ccs = &envelope[5];
    let input = &envelope[6];
    let expected = &envelope[7];
    assert_eq!(
        input,
        &json!([ccs[14], ccs[6], ccs[10], ccs[11], ccs[12], ccs[13], identity]),
        "R receives the complete checked C output and selected relation"
    );
    for family in [2, 3, 4, 5] {
        array(&input[family], SOURCES);
    }
    let state: [F; 8] = fields(&input[0], 8)
        .try_into()
        .expect("complete C endpoint");
    let mut digest = [0u8; 32];
    for (lane, word) in words(&state)[..4].iter().enumerate() {
        digest[lane * 8..lane * 8 + 8].copy_from_slice(&word.to_le_bytes());
    }
    let claims = (0..SOURCES)
        .map(|source| {
            decode_claim(
                &json!([
                    input[2][source],
                    input[3][source],
                    input[1],
                    input[4][source],
                    input[5][source],
                    0
                ]),
                false,
                digest,
            )
        })
        .collect::<Vec<_>>();
    let supplied = decode_claim(
        &json!([expected[3], expected[4], expected[5], expected[6], expected[7], 1]),
        true,
        digest,
    );
    let mut transcript = Poseidon2Transcript::from_state_and_absorbed(state, 0);
    let rhos = paper_exact::sample_rho_n(&mut transcript, params, SOURCES).expect("public reference sampler");
    let rho_matrices = rhos
        .iter()
        .map(|rho| rho.as_mat().clone())
        .collect::<Vec<_>>();
    let challenges = rho_matrices
        .iter()
        .map(|rho| {
            (0..D)
                .map(|row| rho[(row, 0)].as_canonical_u64())
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let membership = challenges
        .iter()
        .map(|rho| {
            u64::from(
                rho.iter()
                    .all(|word| matches!(*word, 0 | 1 | 2) || *word == MODULUS - 1 || *word == MODULUS - 2),
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(membership, vec![1; SOURCES]);
    let partials = (1..=SOURCES)
        .map(|count| {
            rlc_claim_paper_exact_with_commit_mix(
                structure,
                params.inner(),
                &rho_matrices[..count],
                &claims[..count],
                D.next_power_of_two().trailing_zeros() as usize,
                combine_commitments,
            )
        })
        .collect::<Vec<_>>();
    let accepted = paper_exact::verify_pi_rlc(params, structure, &rhos, &claims, &supplied, combine_commitments);
    assert!(accepted, "public reference R rejects the supplied parent");
    let parent = partials.last().expect("17 public partial claims").clone();
    assert_eq!(claim_value(&parent, true), claim_value(&supplied, true));
    let parent_value = claim_value(&parent, true);
    let assurance = nonzero(&parent);
    let computed = json!([
        u64::from(accepted),
        challenges,
        membership,
        parent_value[0],
        parent_value[1],
        parent_value[2],
        parent_value[3],
        parent_value[4],
        partials
            .iter()
            .map(|claim| {
                let value = claim_value(claim, true);
                json!([value[0], value[1], value[3], value[4]])
            })
            .collect::<Vec<_>>(),
        words(&transcript.state()),
        [
            u64::from(claims.iter().all(|claim| nonzero(claim) == [1; 3])),
            assurance[0],
            assurance[1],
            assurance[2]
        ]
    ]);
    assert_eq!(transcript.absorbed(), 0);
    compare_result("PiRLC", &computed, expected, 11);
    println!("paper_public_pi_rlc=passed complete_fields=11 indexed_sources=17");
    (parent, transcript.state())
}

fn magnitude(value: u64) -> u64 {
    value.min(MODULUS - value)
}

fn public_digits(parent: &Mat<F>) -> Option<Vec<Vec<u64>>> {
    let mut digits = vec![vec![0; PUBLIC]; CHILDREN];
    for (coordinate, value) in public_words(parent).into_iter().enumerate() {
        let mut remaining = magnitude(value);
        if remaining >= BOUND {
            return None;
        }
        let negative = value > (MODULUS - 1) / 2;
        for digit in &mut digits {
            let bit = remaining % 2;
            digit[coordinate] = if negative && bit != 0 { MODULUS - bit } else { bit };
            remaining /= 2;
        }
        assert_eq!(remaining, 0, "the fixed 16 digits represent every bounded parent");
    }
    Some(digits)
}

fn recompose_commitments(children: &[Commitment], base: u32) -> Commitment {
    assert_eq!(children.len(), CHILDREN);
    assert_eq!(base, 2);
    let mut output = Commitment::zeros(D, RANK);
    let mut weight = F::ONE;
    for child in children {
        scale_commitment_add_inplace(&mut output, weight, child);
        weight *= F::from_u64(2);
    }
    output
}

fn check_dec(params: &Params, parent: &Claim, state: [F; 8], envelope: &Value) {
    let input = &envelope[8];
    let expected = &envelope[9];
    assert_eq!(
        input[0],
        claim_value(parent, true),
        "D receives the exact checked R parent"
    );
    assert_eq!(input[6], json!(words(&state)), "D receives the full R endpoint");
    for family in [1, 2, 3, 4] {
        array(&input[family], CHILDREN);
    }
    let children = (0..CHILDREN)
        .map(|child| {
            decode_claim(
                &json!([
                    input[1][child],
                    input[4][child],
                    input[5],
                    input[2][child],
                    input[3][child],
                    0
                ]),
                false,
                parent.fold_digest,
            )
        })
        .collect::<Vec<_>>();
    let accepted = paper_exact::verify_pi_dec(params, parent, &children, recompose_commitments);
    assert!(accepted, "public reference D rejects the supplied children");
    let digits = public_digits(&parent.X).expect("strictly bounded public parent");
    let mut computed_children = children.clone();
    for (child, digit) in computed_children.iter_mut().zip(&digits) {
        child.X = public_matrix(&json!(digit));
        child.r = parent.r.clone();
    }
    assert_eq!(
        running_value(&computed_children),
        running_value(&children),
        "verifier-computed public children"
    );
    let parent_bounds = public_words(&parent.X)
        .into_iter()
        .map(|value| u64::from(magnitude(value) < BOUND))
        .collect::<Vec<_>>();
    let ranges = digits
        .iter()
        .map(|digit| {
            digit
                .iter()
                .map(|value| u64::from(magnitude(*value) < 2))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let recomposed_c = recompose_commitments(
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
        for coordinate in 0..PUBLIC {
            recomposed_x[coordinate] += F::from_u64(digits[index][coordinate]) * weight;
        }
        for coordinate in 0..D {
            recomposed_k[coordinate] += child.eval_k[coordinate] * K::from(weight);
            for matrix in 0..MATRICES {
                recomposed_a[matrix][coordinate] += child.eval_a[matrix][coordinate] * K::from(weight);
            }
        }
    }
    let mut unbounded = parent.clone();
    unbounded.X[(0, 0)] = F::from_u64(BOUND);
    let unbounded_rejected = public_digits(&unbounded.X).is_none()
        && !paper_exact::verify_pi_dec(params, &unbounded, &children, recompose_commitments);
    let computed = json!([
        u64::from(accepted),
        u64::from(parent_bounds.iter().all(|check| *check == 1)),
        digits,
        parent_bounds,
        ranges,
        words(&recomposed_c.data),
        u64::from(recomposed_c == parent.c),
        words(&recomposed_x),
        u64::from(words(&recomposed_x) == public_words(&parent.X)),
        extension_words(&recomposed_k),
        u64::from(recomposed_k == parent.eval_k[..D]),
        recomposed_a
            .iter()
            .map(|family| extension_words(family))
            .collect::<Vec<_>>(),
        u64::from(
            recomposed_a
                .iter()
                .zip(&parent.eval_a)
                .all(|(actual, expected)| actual == &expected[..D])
        ),
        computed_children
            .iter()
            .map(|child| claim_value(child, false))
            .collect::<Vec<_>>(),
        words(&state),
        u64::from(unbounded_rejected),
        [1, running_value(&computed_children)]
    ]);
    compare_result("PiDEC", &computed, expected, 17);
    println!("paper_public_pi_dec=passed complete_fields=17 children=16 exact_final_running=true");
}

fn main() {
    let arguments = env::args().skip(1).collect::<Vec<_>>();
    assert_eq!(
        arguments.len(),
        6,
        "usage: check_nifs_public_reference CANDIDATE C_R_D_ENVELOPE ID0 ID1 ID2 ID3"
    );
    let identity: [u64; 4] = std::array::from_fn(|index| {
        let word = arguments[index + 2]
            .parse::<u64>()
            .expect("caller-selected identity word");
        assert!(word < MODULUS);
        word
    });
    let started = Instant::now();
    let bytes = fs::read(&arguments[0]).expect("selected canonical candidate");
    let package = load_per_application_package(&bytes, identity).expect("caller-selected package identity");
    let binding = package
        .production_verifier_binding()
        .expect("selected profile and key binding");
    assert_eq!(binding.structural_identifier(), identity);
    let structure = relation(package.ccs_relation());
    let params = Params::for_ccs_shape(structure.n, structure.m, structure.t(), structure.max_degree())
        .expect("selected Nightstream parameters");
    assert_eq!(params.b(), 2);
    assert_eq!(params.k_rho(), CHILDREN as u32);
    drop(package);
    drop(bytes);
    let envelope = read_envelope(Path::new(&arguments[1]));
    let (parent, state) = check_rlc(&params, &structure, &envelope, identity);
    check_dec(&params, &parent, state, &envelope);
    println!("paper_public_nifs_reference=passed elapsed={:?}", started.elapsed());
}
