use super::*;

fn scalar(i: usize) -> K {
    K::from_coeffs([F::from_usize(31 * i + 7), F::from_usize(19 * i + 11)])
}

fn bind(a: K, b: K, x: K) -> K {
    a + x * (b - a)
}

fn value(coefficients: &[K], x: K) -> K {
    coefficients.iter().rev().fold(K::ZERO, |v, c| v * x + *c)
}

#[test]
fn packed_gate_keeps_both_rows_of_a_signed_pair() {
    let session = MetalSession::new().unwrap();
    let rows = vec![[F::ZERO; 4], [-F::ONE, -F::TWO, F::TWO, F::ZERO]];
    let mut packed = Packed::new(&session, rows, &[0], &[0], 2).unwrap();
    packed.submit(0, K::ZERO, (&[K::ONE], &[K::ONE])).unwrap();
    let (gate, _) = packed.coefficients(0).unwrap();
    let mut expected = [K::ZERO; 8];
    expected[1] = -K::TWO;
    expected[2] = K::TWO;
    assert_eq!(gate, expected);
}

#[test]
fn packed_rounds_and_handoff_match_direct_evaluation_with_partial_tiles() {
    let session = MetalSession::new().unwrap();
    let before = session.activity();
    let checked: Vec<[F; 4]> = (0..1027)
        .map(|i| std::array::from_fn(|j| F::from_usize((i * (j + 1) + j + 3) % 97)))
        .collect();
    let mut gates: Vec<[K; 4]> = checked.iter().map(|row| row.map(K::from)).collect();
    let width = 163 * 64;
    let (mut positive, mut negative) = (vec![0u64; width / 64], vec![0u64; width / 64]);
    let mut digits: Vec<_> = (0..width)
        .map(|i| match (i * 73 + i / 17) % 5 {
            0 => {
                positive[i / 64] |= 1 << (i % 64);
                K::ONE
            }
            1 => {
                negative[i / 64] |= 1 << (i % 64);
                -K::ONE
            }
            _ => K::ZERO,
        })
        .collect();
    let mut packed = Packed::new(&session, checked, &positive, &negative, width).unwrap();
    let mut point = Vec::new();
    for round in 0..=7 {
        let low = (0..8).map(scalar).collect::<Vec<_>>();
        let high = (0..digits.len().div_ceil(16))
            .map(|i| scalar(i + 13))
            .collect::<Vec<_>>();
        let weight = |i: usize| low[i % low.len()] * high[i / low.len()];
        packed
            .submit(round, point.last().copied().unwrap_or(K::ZERO), (&low, &high))
            .unwrap();
        let (gate, norm) = packed.coefficients(round).unwrap();
        let after_round = session.activity();
        assert_eq!(packed.coefficients(round).unwrap(), (gate, norm));
        assert_eq!(session.activity(), after_round);
        for sample in 0..8 {
            let x = scalar(sample + 101);
            let expected: K = gates
                .chunks(2)
                .enumerate()
                .map(|(i, pair)| {
                    let [a, b, c, s]: [K; 4] =
                        std::array::from_fn(|j| bind(pair[0][j], pair.get(1).map_or(K::ZERO, |right| right[j]), x));
                    weight(i) * (a * b - c + s.exp_u64(7))
                })
                .sum();
            assert_eq!(value(&gate, x), expected, "gate round {round}, sample {sample}");
            if sample < 4 {
                let expected: K = digits
                    .chunks(2)
                    .enumerate()
                    .map(|(i, pair)| {
                        let v = bind(pair[0], pair.get(1).copied().unwrap_or(K::ZERO), x);
                        weight(i) * (v * v * v - v)
                    })
                    .sum();
                assert_eq!(value(&norm, x), expected, "norm round {round}, sample {sample}");
            }
        }
        if round == 7 {
            let (columns, signed) = packed.take_tables(round).unwrap();
            let expected: Vec<_> = (0..4)
                .flat_map(|j| gates.iter().map(move |row| row[j]))
                .collect();
            assert_eq!(columns, expected);
            assert_eq!(signed, digits);
            break;
        }
        let coin = scalar(round + 41);
        gates = gates
            .chunks(2)
            .map(|pair| std::array::from_fn(|j| bind(pair[0][j], pair.get(1).map_or(K::ZERO, |right| right[j]), coin)))
            .collect();
        digits = digits
            .chunks(2)
            .map(|pair| bind(pair[0], pair.get(1).copied().unwrap_or(K::ZERO), coin))
            .collect();
        point.push(coin);
    }
    let after = session.activity();
    assert_eq!(after.command_buffers - before.command_buffers, 8);
    assert_eq!(after.dispatches - before.dispatches, 32);
    assert_eq!(after.host_waits - before.host_waits, 8);
    assert_eq!(after.allocated_bytes - before.allocated_bytes, 606_864);
    assert_eq!(after.uploaded_bytes - before.uploaded_bytes, 592_368);
    assert_eq!(after.downloaded_bytes - before.downloaded_bytes, 3_424);
}

#[test]
fn packed_startup_uses_cpu_only_on_buffer_allocation_failure() {
    let session = MetalSession::new().unwrap();
    let before = session.activity();
    let limit = session
        .buffer(usize::MAX)
        .expect_err("allocation exceeds the device budget");
    for error in [limit, MetalError::Buffer { bytes: 4096 }] {
        assert!(optional_packed(Err(error)).unwrap().is_none());
    }
    assert!(matches!(
        optional_packed(Err(MetalError::Shape("row count"))),
        Err(MetalError::Shape("row count"))
    ));
    assert_eq!(session.activity(), before);
}

#[test]
fn fresh_only_packed_round_uses_checked_rows_of_the_proving_shape() {
    use neo_ccs::{poly::Term, CcsStructure, CcsWitness, Mat, SparsePoly};
    use neo_math::D;
    use neo_reductions::{
        engines::pi_ccs_joint::build_joint_dims,
        superneo_eval::{MatrixRowSink, MatrixRows, MatrixShape},
        Challenges,
    };
    use std::sync::atomic::AtomicBool;

    struct Rows {
        shape: MatrixShape,
        checked: bool,
        requested: AtomicBool,
    }
    impl MatrixRows for Rows {
        fn shape(&self) -> MatrixShape {
            self.shape
        }
        fn visit_rows(&self, _: std::ops::Range<usize>, _: &mut dyn MatrixRowSink) -> Result<(), PiCcsError> {
            panic!("the packed oracle consumes previously checked rows");
        }
        fn take_checked_fresh_rows(&self) -> Option<Vec<[F; 4]>> {
            self.requested.store(true, Ordering::Relaxed);
            self.checked.then(|| {
                let mut rows = vec![[F::ZERO; 4]; self.shape.rows];
                rows[self.shape.rows - 1] = [F::ONE, F::TWO, F::TWO, F::ZERO];
                rows
            })
        }
    }

    let metal = MetalSession::new().unwrap();
    for (rows, width, checked) in [(175_752, 7_010_304, true), (1_200_265, 31_785_216, false)] {
        let source = Rows {
            shape: MatrixShape {
                rows,
                columns: width,
                matrices: 4,
            },
            checked,
            requested: AtomicBool::new(false),
        };
        let terms =
            [([1, 1, 0, 0], F::ONE), ([0, 0, 1, 0], -F::ONE), ([0, 0, 0, 7], F::ONE)].map(|(exps, coeff)| Term {
                coeff,
                exps: exps.to_vec(),
            });
        let structure = CcsStructure::new_verifier_artifact_header(rows, width, 4, SparsePoly::new(4, terms.to_vec()))
            .unwrap()
            .with_domain_variables(28)
            .unwrap();
        let params = neo_params::NeoParams::nightstream_p64_k14();
        let security = params
            .padded_row_security_summary_for_shape(
                structure.domain_rows(),
                width,
                4,
                structure.max_degree(),
                neo_params::nightstream_p64_k14::CHALLENGE_ALPHABET.len() as u32,
            )
            .unwrap();
        let params = params
            .with_certified_lambda(security.security_bits)
            .unwrap();
        let dims = build_joint_dims(&params, &structure, 1, 0).unwrap();
        let blocks = width / D;
        let mut positive = vec![0; blocks];
        positive[blocks - 1] = 1 << 63;
        let fresh = [CcsWitness {
            Z: Mat::from_owned_column_masks(D, blocks, positive, vec![0; blocks]).unwrap(),
        }];
        let input = PaperJointOracleInput {
            structure: &structure,
            params: &params,
            fresh_witnesses: &fresh,
            running_witnesses: &[],
            challenges: Challenges::new(vec![K::ONE; dims.variables], K::ONE),
            prior_point: None,
            dims,
            rows: &source,
            workspace_bytes: usize::MAX,
            norm_sources: 1,
        };
        let device = metal.packed_rounds(&input).unwrap();
        assert!(source.requested.load(Ordering::Relaxed));
        assert_eq!(device.is_some(), checked);
        if let Some(mut device) = device {
            let low = vec![K::ONE; 1 << ((dims.variables - 1) / 2)];
            let high = vec![K::ONE; 1 << ((dims.variables - 1).div_ceil(2))];
            device.submit(0, K::ZERO, (&low, &high)).unwrap();
            let (gate, norm) = device.coefficients(0).unwrap();
            for sample in 0..8 {
                let x = scalar(sample);
                assert_eq!(value(&gate, x), (x.square() - x).double());
                assert_eq!(value(&norm, x), x.cube() - x);
            }
        }
    }
}
