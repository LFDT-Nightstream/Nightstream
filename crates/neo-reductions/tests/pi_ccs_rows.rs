use neo_ajtai::Commitment;
use neo_ccs::{CcsClaim, CcsStructure, CcsWitness, CeClaim, Mat, SparsePoly};
use neo_math::{superneo_bar_block, KExtensions, Rq, D, F, K};
use neo_params::NeoParams;
use neo_reductions::optimized_engine::{optimized_prove_with_row_cache, optimized_verify_with_trace};
use neo_reductions::superneo_eval::{build_superneo_eval_cache, eval_all_mats_ring_cached, EqualityWeights};
use neo_transcript::Poseidon2Transcript;
use p3_field::PrimeCharacteristicRing;

fn transcript() -> Poseidon2Transcript {
    Poseidon2Transcript::from_state_and_absorbed([F::ZERO; 8], 0)
}

fn direct_pad(values: &[F], point: &[K]) -> Vec<K> {
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
    result.to_vec()
}

#[test]
fn row_prover_preserves_odd_prefixes_singletons_and_complete_pad_blocks() {
    let rows = 5;
    let logical = D + 3;
    let width = logical.div_ceil(D) * D;
    let mut matrix = Mat::zero(rows, logical, F::ZERO);
    matrix[(0, 0)] = F::ONE;
    matrix[(4, logical - 1)] = F::from_u64(7);
    let structure = CcsStructure::new(
        vec![matrix, Mat::zero(rows, logical, F::ZERO)],
        SparsePoly::new(2, vec![]),
    )
    .unwrap();
    let cache = build_superneo_eval_cache(&structure).unwrap();
    let mut params = NeoParams::nightstream_goldilocks_k16();
    params.lambda = params.lambda.min(
        params
            .padded_row_security_summary_for_shape(
                rows,
                logical,
                structure.t(),
                structure.max_degree(),
                neo_params::goldilocks_paper_b2::CHALLENGE_ALPHABET.len() as u32,
            )
            .unwrap()
            .security_bits,
    );
    let mut fresh_values = vec![F::ZERO; width];
    fresh_values[0] = F::ONE;
    fresh_values[logical - 1] = -F::ONE;
    let mut running_values = vec![F::ZERO; width];
    running_values[logical - 1] = -F::ONE;
    running_values[width - 1] = F::ONE; // A permitted nonzero running carrier tail.
    let packed = |values: &[F]| {
        Mat::from_row_major(
            D,
            width / D,
            (0..D)
                .flat_map(|lane| (0..width / D).map(move |block| values[block * D + lane]))
                .collect(),
        )
    };
    let fresh = CcsWitness {
        w: Vec::new(),
        Z: packed(&fresh_values),
    };
    let running_witness = packed(&running_values);
    let fresh_claim = CcsClaim {
        c: Commitment::zeros(D, params.kappa as usize),
        x: fresh_values[..D].to_vec(),
        m_in: D,
        adv: None,
    };
    let point = vec![K::from_coeffs([F::from_u64(3), F::ONE]); width.next_power_of_two().trailing_zeros() as usize];
    let weights = EqualityWeights::new(&point);
    let row_weights = (0..rows).map(|row| weights.at(row)).collect::<Vec<_>>();
    let running_k = running_values
        .iter()
        .copied()
        .map(K::from)
        .collect::<Vec<_>>();
    let mut eval_k = direct_pad(&running_values, &point);
    eval_k.resize(D.next_power_of_two(), K::ZERO);
    let eval_a = eval_all_mats_ring_cached(&cache, &running_k, &row_weights, rows)
        .into_iter()
        .map(|values| {
            let mut values = values.to_vec();
            values.resize(D.next_power_of_two(), K::ZERO);
            values
        })
        .collect();
    let running_claim = CeClaim {
        c: Commitment::zeros(D, params.kappa as usize),
        X: Mat::zero(D, 1, F::ZERO),
        r: point,
        eval_k,
        eval_a,
        m_in: D,
        fold_digest: [0; 32],
        adv: None,
    };
    let (outputs, proof, _, prover_trace) = optimized_prove_with_row_cache(
        &mut transcript(),
        &params,
        &structure,
        std::slice::from_ref(&fresh_claim),
        std::slice::from_ref(&fresh),
        std::slice::from_ref(&running_claim),
        std::slice::from_ref(&running_witness),
        &cache,
    )
    .unwrap();
    let (valid, verifier_trace) = optimized_verify_with_trace(
        &mut transcript(),
        &params,
        &structure,
        std::slice::from_ref(&fresh_claim),
        std::slice::from_ref(&running_claim),
        &outputs,
        &proof,
    )
    .unwrap();
    assert!(valid);
    assert_eq!(prover_trace.events, verifier_trace.events);
    assert_eq!(prover_trace.round_states, verifier_trace.round_states);
    assert_eq!(prover_trace.round_claims, verifier_trace.round_claims);
    assert_eq!(prover_trace.terminal_claim, verifier_trace.terminal_claim);
    assert_eq!(prover_trace.outgoing_state, verifier_trace.outgoing_state);
    for (output, values) in outputs.iter().zip([&fresh_values, &running_values]) {
        assert_eq!(&output.eval_k[..D], direct_pad(values, &output.r));
        assert_eq!(output.eval_a[1], vec![K::ZERO; D.next_power_of_two()]);
    }
    let full_pad = direct_pad(&fresh_values, &outputs[0].r);
    // Dropping the51 zero scalar lanes would lose higher Pad coefficients.
    let weights = EqualityWeights::new(&outputs[0].r);
    let mut omitted = [K::ZERO; D];
    for row in logical..width {
        let mut unit = [F::ZERO; D];
        unit[row % D] = F::ONE;
        let product =
            Rq(superneo_bar_block(unit)).mul(&Rq(std::array::from_fn(|lane| fresh_values[row / D * D + lane])));
        for lane in 0..D {
            omitted[lane] += weights.at(row) * K::from(product.0[lane]);
        }
    }
    assert!(omitted.iter().skip(1).any(|value| *value != K::ZERO));
    assert_eq!(outputs[0].eval_k[..D], full_pad);

    let mut invalid_fresh = fresh.clone();
    invalid_fresh.Z[(D - 1, width / D - 1)] = F::ONE;
    assert!(optimized_prove_with_row_cache(
        &mut transcript(),
        &params,
        &structure,
        &[fresh_claim],
        &[invalid_fresh],
        &[running_claim],
        &[running_witness],
        &cache
    )
    .is_err());
}
