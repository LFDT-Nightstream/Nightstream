//! The bounded-depth scheme's light PiCCS (`NormSources::Fresh`, Lean `BoundedDepth.lightPiCcsOracle`)
//! checks the fresh source's norm only: a running witness outside the digit
//! range folds under the light profile and fails under the production one.

use neo_ajtai::Commitment;
use neo_ccs::{CcsClaim, CcsStructure, CcsWitness, CeClaim, Mat, SparsePoly};
use neo_math::{superneo_bar_block, KExtensions, Rq, D, F, K};
use neo_params::NeoParams;
use neo_reductions::optimized_engine::{optimized_prove_with_profile, optimized_verify_with_profile};
use neo_reductions::superneo_eval::{
    build_superneo_eval_cache, eval_all_mats_ring_cached, CachedMatrixRows, EqualityWeights, MatrixRows, MatrixWindow,
};
use neo_reductions::{FoldProfile, NormSources, PiCcsProof};
use neo_transcript::{domain_chunk_v1_2, Poseidon2Transcript};
use p3_field::PrimeCharacteristicRing;

const ROWS: usize = 5;
const LOGICAL: usize = D + 3;
const WIDTH: usize = LOGICAL.div_ceil(D) * D;

fn transcript() -> Poseidon2Transcript {
    Poseidon2Transcript::from_state_and_absorbed([F::ZERO; 16], 0)
}

fn light() -> FoldProfile<'static> {
    FoldProfile {
        domain: domain_chunk_v1_2(b"Nightstream/SuperNeo/fold/light/v1"),
        children: &[],
        norm: NormSources::Fresh,
    }
}

fn pad(values: &[F], point: &[K]) -> Vec<K> {
    let weights = EqualityWeights::new(point);
    let mut result = [K::ZERO; D];
    for row in 0..values.len() {
        let mut unit = [F::ZERO; D];
        unit[row % D] = F::ONE;
        let block = Rq(std::array::from_fn(|lane| values[row / D * D + lane]));
        let product = Rq(superneo_bar_block(unit)).mul(&block);
        for lane in 0..D {
            result[lane] += weights.at(row) * K::from(product.0[lane]);
        }
    }
    let mut result = result.to_vec();
    result.resize(D.next_power_of_two(), K::ZERO);
    result
}

fn packed(values: &[F]) -> Mat<F> {
    Mat::from_row_major(
        D,
        WIDTH / D,
        (0..D)
            .flat_map(|lane| (0..WIDTH / D).map(move |block| values[block * D + lane]))
            .collect(),
    )
}

struct Fold {
    params: NeoParams,
    structure: CcsStructure<F>,
    fresh_claim: CcsClaim<Commitment, F>,
    fresh: CcsWitness<F>,
    running_claim: CeClaim<Commitment, F, K>,
    running: Mat<F>,
}

impl Fold {
    /// One fresh instance with digits in range, and a running claim whose
    /// witness has the entry `running_entry` (out of range when it is 5).
    fn new(running_entry: F) -> Self {
        let mut matrix = Mat::zero(ROWS, LOGICAL, F::ZERO);
        matrix[(0, 0)] = F::ONE;
        matrix[(4, LOGICAL - 1)] = F::from_u64(7);
        let structure = CcsStructure::new(
            vec![matrix, Mat::zero(ROWS, LOGICAL, F::ZERO)],
            SparsePoly::new(2, vec![]),
        )
        .unwrap();
        let cache = build_superneo_eval_cache(&structure).unwrap();
        let base = NeoParams::nightstream_goldilocks_k16();
        let bits = base
            .padded_row_security_summary_for_shape(
                ROWS,
                LOGICAL,
                structure.t(),
                structure.max_degree(),
                neo_params::goldilocks_paper_b2::CHALLENGE_ALPHABET.len() as u32,
            )
            .unwrap()
            .security_bits;
        let params = base.with_lambda(base.lambda().min(bits)).unwrap();
        let mut fresh_values = vec![F::ZERO; WIDTH];
        fresh_values[0] = F::ONE;
        fresh_values[LOGICAL - 1] = -F::ONE;
        let mut running_values = vec![F::ZERO; WIDTH];
        running_values[D + 1] = running_entry;
        running_values[WIDTH - 1] = F::ONE;
        let point = vec![K::from_coeffs([F::from_u64(3), F::ONE]); WIDTH.next_power_of_two().trailing_zeros() as usize];
        let weights = EqualityWeights::new(&point);
        let row_weights = (0..ROWS).map(|row| weights.at(row)).collect::<Vec<_>>();
        let running_k = running_values
            .iter()
            .copied()
            .map(K::from)
            .collect::<Vec<_>>();
        let eval_a = eval_all_mats_ring_cached(&cache, &running_k, &row_weights, ROWS)
            .into_iter()
            .map(|values| {
                let mut values = values.to_vec();
                values.resize(D.next_power_of_two(), K::ZERO);
                values
            })
            .collect();
        Self {
            fresh_claim: CcsClaim {
                c: Commitment::zeros(params.kappa() as usize),
                x: fresh_values[..D].to_vec(),
                m_in: D,
                adv: None,
            },
            fresh: CcsWitness {
                Z: packed(&fresh_values),
            },
            running_claim: CeClaim {
                c: Commitment::zeros(params.kappa() as usize),
                X: Mat::zero(D, 1, F::ZERO),
                eval_k: pad(&running_values, &point),
                r: point,
                eval_a,
                m_in: D,
                fold_digest: [0; 32],
                adv: None,
            },
            running: packed(&running_values),
            params,
            structure,
        }
    }

    fn prove(&self, profile: &FoldProfile<'_>) -> Result<(Vec<CeClaim<Commitment, F, K>>, PiCcsProof), String> {
        let cache = build_superneo_eval_cache(&self.structure).unwrap();
        let rows = CachedMatrixRows::new(&cache).unwrap();
        let payload = 2 * self.structure.t() * size_of::<K>() + size_of::<F>() + size_of::<K>();
        let workspace = MatrixWindow::required_workspace(&rows, 0..rows.shape().rows, payload).unwrap() * 2;
        optimized_prove_with_profile(
            &mut transcript(),
            &self.params,
            &self.structure,
            std::slice::from_ref(&self.fresh_claim),
            std::slice::from_ref(&self.fresh),
            std::slice::from_ref(&self.running_claim),
            std::slice::from_ref(&self.running),
            &rows,
            workspace,
            None,
            profile,
        )
        .map(|(outputs, proof, _)| (outputs, proof))
        .map_err(|error| error.to_string())
    }

    fn verify(&self, outputs: &[CeClaim<Commitment, F, K>], proof: &PiCcsProof, profile: &FoldProfile<'_>) -> bool {
        optimized_verify_with_profile(
            &mut transcript(),
            &self.params,
            &self.structure,
            std::slice::from_ref(&self.fresh_claim),
            std::slice::from_ref(&self.running_claim),
            outputs,
            proof,
            profile,
        )
        .unwrap_or(false)
    }
}

#[test]
fn light_profile_folds_a_running_witness_outside_the_digit_range() {
    let fold = Fold::new(F::from_u64(5));
    let (outputs, proof) = fold.prove(&light()).expect("light proof");
    assert!(fold.verify(&outputs, &proof, &light()));
    assert!(!fold.verify(&outputs, &proof, &FoldProfile::production()));
}

#[test]
fn production_profile_rejects_a_running_witness_outside_the_digit_range() {
    let fold = Fold::new(F::from_u64(5));
    let accepted = fold
        .prove(&FoldProfile::production())
        .is_ok_and(|(outputs, proof)| fold.verify(&outputs, &proof, &FoldProfile::production()));
    assert!(!accepted);
}

#[test]
fn light_profile_still_checks_the_fresh_norm() {
    let mut fold = Fold::new(F::ONE);
    fold.fresh.Z[(1, 1)] = F::from_u64(5);
    let accepted = fold
        .prove(&light())
        .is_ok_and(|(outputs, proof)| fold.verify(&outputs, &proof, &light()));
    assert!(!accepted);
}

/// Two profiles that differ only in the domain do not accept each other's proofs.
#[test]
fn the_domain_separates_light_proofs() {
    let fold = Fold::new(F::ONE);
    let (outputs, proof) = fold.prove(&light()).expect("light proof");
    assert!(fold.verify(&outputs, &proof, &light()));
    let production_domain = FoldProfile {
        domain: FoldProfile::production().domain,
        ..light()
    };
    assert!(!fold.verify(&outputs, &proof, &production_domain));
}

#[test]
fn refresh_profile_binds_the_children_words() {
    let fold = Fold::new(F::ONE);
    let children = [F::from_u64(11), F::from_u64(12)];
    let other = [F::from_u64(11), F::from_u64(13)];
    let refresh = |children| FoldProfile {
        domain: domain_chunk_v1_2(b"Nightstream/SuperNeo/fold/refresh/v1"),
        children,
        norm: NormSources::All,
    };
    let (outputs, proof) = fold.prove(&refresh(&children)).expect("refresh proof");
    assert!(fold.verify(&outputs, &proof, &refresh(&children)));
    assert!(!fold.verify(&outputs, &proof, &refresh(&other)));
}
