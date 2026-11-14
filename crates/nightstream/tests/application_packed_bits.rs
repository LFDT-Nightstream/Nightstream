use nightstream::application::{Affine, ApplicationBuilder};
use nightstream::{Circuit, Engine, Fq as F, PrimeCharacteristicRing, State};

fn circuit() -> Circuit {
    let mut builder = ApplicationBuilder::new(32).unwrap();
    let state = builder.input_state();
    let mut word = Affine::constant(F::ZERO);
    for (index, bit) in builder.private_inputs().to_vec().into_iter().enumerate() {
        builder
            .assert_product(bit.into(), bit.into(), bit.into())
            .unwrap();
        word = word + Affine::from(bit) * F::from_u64(1 << index);
    }
    Circuit::compile(
        builder
            .finish([
                Affine::from(state[0]) + word,
                state[1].into(),
                state[2].into(),
                state[3].into(),
            ])
            .unwrap(),
    )
    .unwrap()
}

fn bits(word: u32) -> Vec<F> {
    (0..32)
        .map(|index| F::from_u64(u64::from((word >> index) & 1)))
        .collect()
}

#[test]
fn packed_bits_keep_public_multistep_statements_and_reject_false_inputs() {
    let circuit = circuit();
    let prover = circuit.prover(Engine::Optimized, 109).unwrap();
    let verifier = circuit.verifier(Engine::Optimized, 109).unwrap();
    let initial = [F::ZERO; 4];
    let words = [0x8000_0001u32, 0xffff_ffff, 0x5555_5555];
    let mut proof = prover.prove(initial, &bits(words[0])).unwrap();
    let mut current = [F::from_u64(u64::from(words[0])), F::ZERO, F::ZERO, F::ZERO];
    verifier
        .verify(&State::new(1, initial, current), &proof)
        .unwrap();
    for (index, word) in words.into_iter().enumerate().skip(1) {
        proof = prover.extend(&proof, &bits(word)).unwrap();
        current[0] += F::from_u64(u64::from(word));
        verifier
            .verify(&State::new(index as u64 + 1, initial, current), &proof)
            .unwrap();
    }
    let decoded = verifier
        .decode_proof(&prover.encode_proof(&proof).unwrap())
        .unwrap();
    verifier
        .verify(&State::new(3, initial, current), &decoded)
        .unwrap();
    let other = prover.prove(initial, &bits(7)).unwrap();
    verifier
        .verify(
            &State::new(1, initial, [F::from_u64(7), F::ZERO, F::ZERO, F::ZERO]),
            &other,
        )
        .unwrap();
    assert!(verifier
        .verify(
            &State::new(
                1,
                initial,
                [F::from_u64(u64::from(words[0])), F::ZERO, F::ZERO, F::ZERO]
            ),
            &other
        )
        .is_err());
    for value in [F::NEG_ONE, F::from_u64(2)] {
        let mut wrong = bits(0);
        wrong[31] = value;
        assert!(prover.prove(initial, &wrong).is_err());
        assert!(prover.extend(&proof, &wrong).is_err());
    }
}
