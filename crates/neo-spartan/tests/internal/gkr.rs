//! The fraction-sum GKR and the histogram side of logUp.

use neo_transcript::Poseidon2Transcript;
use p3_field_v08::{BasedVectorSpace, Field, PrimeCharacteristicRing};

use super::word;
use crate::field::{eq_table, signed, Ext, Gl};
use crate::gkr::{prove, verify};
use crate::hash::challenger;
use crate::norm::{histogram, table_sum};

const BOUND: u32 = 5;

fn witness(variables: usize) -> Vec<Gl> {
    (0..1u64 << variables)
        .map(|i| signed((word(11, i) % (2 * BOUND as u64 + 1)) as i64 - BOUND as i64))
        .collect()
}

fn beta() -> Ext {
    Ext::from_basis_coefficients_fn(|i| Gl::from_u64(word(12, i as u64)))
}

#[test]
fn leaf_claim_matches_the_witness_and_the_root_matches_the_histogram() {
    for variables in [1, 2, 7] {
        let z = witness(variables);
        let beta = beta();
        let (proof, prover_point) = prove(&z, beta, &mut challenger(Poseidon2Transcript::new_v1_1()));
        let total = table_sum(&histogram(&z, BOUND).unwrap(), BOUND, beta).unwrap();
        let direct: Ext = z.iter().map(|&value| (beta - value).inverse()).sum();
        assert_eq!(total, direct);

        let leaf = verify(
            &proof,
            variables,
            total,
            &mut challenger(Poseidon2Transcript::new_v1_1()),
        )
        .unwrap();
        assert_eq!(leaf.point, prover_point);
        let z_at_point: Ext = eq_table(&leaf.point)
            .iter()
            .zip(&z)
            .map(|(&w, &v)| w * v)
            .sum();
        assert_eq!(leaf.q, beta - z_at_point, "{variables} variables");
    }
}

#[test]
fn tampered_trees_and_forged_histograms_reject() {
    let variables = 7;
    let z = witness(variables);
    let beta = beta();
    let (proof, _) = prove(&z, beta, &mut challenger(Poseidon2Transcript::new_v1_1()));
    let counts = histogram(&z, BOUND).unwrap();
    let check = |proof: &crate::gkr::GkrProof, counts: &[u32]| {
        let total = table_sum(counts, BOUND, beta).unwrap();
        verify(
            proof,
            variables,
            total,
            &mut challenger(Poseidon2Transcript::new_v1_1()),
        )
        .is_ok()
    };
    assert!(check(&proof, &counts));

    // One witness value claimed as another table value.
    let mut forged = counts.clone();
    let from = forged.iter().position(|&count| count > 0).unwrap();
    forged[from] -= 1;
    let next = (from + 1) % forged.len();
    forged[next] += 1;
    assert!(!check(&proof, &forged));

    let mut root = proof.clone();
    root.root[1] += Ext::ONE;
    assert!(!check(&root, &counts));
    for layer in [0, 3, variables - 1] {
        let mut children = proof.clone();
        children.layers[layer].children[0] += Ext::ONE;
        assert!(!check(&children, &counts), "children of layer {layer}");
    }
    let mut round = proof.clone();
    round.layers[4].rounds[2][1] += Ext::ONE;
    assert!(!check(&round, &counts));
}

#[test]
fn the_prover_refuses_a_witness_beyond_the_bound() {
    let mut z = witness(4);
    z[3] = signed(BOUND as i64 + 1);
    assert!(histogram(&z, BOUND).is_err());
}
