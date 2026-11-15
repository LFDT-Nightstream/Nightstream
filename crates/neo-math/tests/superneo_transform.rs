use neo_math::{cf_inv, ct, superneo_bar_block, superneo_bar_matrix, superneo_bar_vec, Fq, Rq, D};
use p3_field::PrimeCharacteristicRing;

fn deterministic_block(seed: u64) -> [Fq; D] {
    let mut out = [Fq::ZERO; D];
    let mut x = seed;
    for oi in &mut out {
        x = x
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *oi = Fq::from_u64(x);
    }
    out
}

fn dot(a: &[Fq; D], b: &[Fq; D]) -> Fq {
    let mut acc = Fq::ZERO;
    for i in 0..D {
        acc += a[i] * b[i];
    }
    acc
}

/// SuperNeo v1.2 Section 5: `ct(a · bar(b)) = ⟨a, b⟩`.
#[test]
fn superneo_transform_matches_one_sided_inner_product_identity() {
    let extreme = [-Fq::ONE; D];
    assert_eq!(
        ct(&cf_inv(extreme).mul(&cf_inv(superneo_bar_block(extreme)))),
        dot(&extreme, &extreme)
    );

    for round in 0..16u64 {
        let a = deterministic_block(0x1234_5678_9abc_def0 ^ round);
        let b = deterministic_block(0xfedc_ba98_7654_3210 ^ (round.wrapping_mul(17)));

        assert_eq!(
            ct(&cf_inv(a).mul(&cf_inv(superneo_bar_block(b)))),
            dot(&a, &b),
            "round={round}"
        );
        assert_eq!(
            ct(&cf_inv(superneo_bar_block(a)).mul(&cf_inv(b))),
            dot(&a, &b),
            "round={round}"
        );
    }
}

/// The identity fixes the transform: its matrix is the inverse of the Gram
/// matrix `G[i][j] = ct(X^i · X^j)`.
#[test]
fn superneo_transform_matrix_inverts_the_constant_term_gram_matrix() {
    let monomial = |degree: usize| {
        let mut coefficients = [Fq::ZERO; D];
        coefficients[degree] = Fq::ONE;
        Rq(coefficients)
    };
    let gram: Vec<[Fq; D]> = (0..D)
        .map(|i| core::array::from_fn(|j| ct(&monomial(i).mul(&monomial(j)))))
        .collect();
    let bar = superneo_bar_matrix();
    for i in 0..D {
        for j in 0..D {
            let entry = (0..D).fold(Fq::ZERO, |acc, r| acc + gram[i][r] * bar[r][j]);
            assert_eq!(entry, Fq::from_bool(i == j), "(G · bar)[{i}][{j}]");
        }
    }
}

#[test]
fn superneo_transform_is_b0_minus_the_reversed_tail() {
    let b = deterministic_block(0x0bad_cafe);
    let bar = superneo_bar_block(b);
    assert_eq!(bar[0], b[0]);
    for j in 1..D {
        assert_eq!(bar[j], -b[D - j], "coefficient {j}");
    }
    assert_eq!(superneo_bar_block(bar), b);
}

#[test]
fn superneo_transform_is_linear_on_blocks() {
    for round in 0..16u64 {
        let a = deterministic_block(0x1111_1111_1111_1111 ^ round);
        let b = deterministic_block(0x2222_2222_2222_2222 ^ (round.wrapping_mul(9)));

        let mut a_plus_b = [Fq::ZERO; D];
        for i in 0..D {
            a_plus_b[i] = a[i] + b[i];
        }

        let bar_sum = superneo_bar_block(a_plus_b);
        let bar_a = superneo_bar_block(a);
        let bar_b = superneo_bar_block(b);

        for i in 0..D {
            assert_eq!(bar_sum[i], bar_a[i] + bar_b[i], "round={round}, idx={i}");
        }
    }
}

#[test]
fn superneo_transform_commutes_with_base_scalars() {
    let block = deterministic_block(0x5555_5555_5555_5555);
    let scalar = Fq::from_u64(37);
    let scaled = std::array::from_fn(|i| scalar * block[i]);
    let transformed = superneo_bar_block(block);

    assert_eq!(
        superneo_bar_block(scaled),
        std::array::from_fn(|i| scalar * transformed[i])
    );
}

#[test]
fn superneo_transform_vector_lifts_blockwise() {
    let mut v = vec![Fq::ZERO; 2 * D];
    let b0 = deterministic_block(0x3333_3333_3333_3333);
    let b1 = deterministic_block(0x4444_4444_4444_4444);
    v[..D].copy_from_slice(&b0);
    v[D..2 * D].copy_from_slice(&b1);

    let out = superneo_bar_vec(&v);
    let want0 = superneo_bar_block(b0);
    let want1 = superneo_bar_block(b1);

    assert_eq!(&out[..D], &want0);
    assert_eq!(&out[D..2 * D], &want1);
}

#[test]
fn superneo_transform_block_matches_dense_matrix() {
    let matrix = superneo_bar_matrix();
    let dense = |block: &[Fq; D]| -> [Fq; D] { std::array::from_fn(|row| dot(&matrix[row], block)) };
    for column in 0..D {
        let mut basis = [Fq::ZERO; D];
        basis[column] = Fq::ONE;
        assert_eq!(superneo_bar_block(basis), dense(&basis), "basis {column}");
    }
    for round in 0..64u64 {
        let block = deterministic_block(0x5555_5555_5555_5555 ^ round);
        assert_eq!(superneo_bar_block(block), dense(&block), "round={round}");
    }
}

#[test]
#[should_panic(expected = "superneo_bar_vec expects length multiple of D")]
fn superneo_transform_rejects_partial_blocks() {
    superneo_bar_vec(&vec![Fq::ZERO; D + 1]);
}
