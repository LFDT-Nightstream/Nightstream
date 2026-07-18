//! Exact Rust/R1CS correspondence for the active `Pi_CCS` output serializer.
//!
//! | Test family | Mathematical obligation | Failure detected |
//! |---|---|---|
//! | profile | active `15 × 13` layout has 23,033 fields | legacy three-matrix or count drift |
//! | decoder | every field index has one typed source/matrix/lane/limb path | omission, duplication, or reordering |
//! | parity | native fields equal the values at the recorded R1CS columns | native/circuit protocol drift |
//! | rejection | verifier-owned source and matrix counts are exact | prover-shaped profile inference |

use std::collections::HashSet;

use neo_ajtai::Commitment;
use neo_ccs::{CeClaim, Mat};
use neo_fold_clean::engine::r1cs_circuit::field_ext::KVar;
use neo_fold_clean::engine::r1cs_circuit::R1csBuilder;
use neo_fold_clean::paper::digest::{pi_ccs_outputs_digest, pi_ccs_outputs_preimage};
use neo_fold_clean::paper::reductions::pi_ccs_output_message::{
    FieldPath, KLimb, Profile, R1csInputOwner, ACTIVE_F_PRIME_FIELD_COUNT, LEGACY_THREE_MATRIX_FIELD_COUNT,
};
use neo_fold_clean::paper::reductions::pi_ccs_split_nc_circuit::{
    encode_pi_ccs_outputs_preimage, enforce_pi_ccs_outputs_digest, PiCcsOutputMessageDigestInputs,
};
use neo_math::ring::D;
use neo_math::{KExtensions, F, K};
use p3_field::PrimeCharacteristicRing;

type Claim = CeClaim<Commitment, F, K>;

struct NativeOutput {
    y_ring: Vec<Vec<K>>,
    y_zcol: Vec<K>,
}

struct WireOutput {
    y_ring: Vec<Vec<KVar>>,
    y_zcol: Vec<KVar>,
}

fn value(source: usize, vector: usize, lane: usize) -> K {
    let base = 1 + ((source * 20 + vector) * D + lane) as u64 * 2;
    K::from_coeffs([F::from_u64(base), F::from_u64(base + 1)])
}

fn native_outputs(profile: Profile) -> Vec<NativeOutput> {
    (0..profile.source_count())
        .map(|source| NativeOutput {
            y_ring: (0..profile.matrix_count())
                .map(|matrix| (0..D + 2).map(|lane| value(source, matrix, lane)).collect())
                .collect(),
            y_zcol: (0..D + 2)
                .map(|lane| value(source, profile.matrix_count(), lane))
                .collect(),
        })
        .collect()
}

fn claim(output: &NativeOutput) -> Claim {
    Claim {
        c: Commitment {
            d: D,
            kappa: 1,
            data: vec![F::ZERO; D],
        },
        X: Mat::zero(D, 1, F::ZERO),
        r: Vec::new(),
        s_col: Vec::new(),
        y_ring: output.y_ring.clone(),
        ct: vec![K::ZERO; output.y_ring.len()],
        aux_openings: Vec::new(),
        y_zcol: output.y_zcol.clone(),
        m_in: 1,
        fold_digest: [0; 32],
        c_step_coords: Vec::new(),
        u_offset: 0,
        u_len: 0,
        adv: None,
    }
}

fn alloc_k(builder: &mut R1csBuilder, value: K) -> KVar {
    let [c0, c1] = value.as_coeffs();
    KVar::alloc(builder, c0, c1)
}

fn wire_output(builder: &mut R1csBuilder, output: &NativeOutput) -> WireOutput {
    WireOutput {
        y_ring: output
            .y_ring
            .iter()
            .map(|row| row.iter().map(|value| alloc_k(builder, *value)).collect())
            .collect(),
        y_zcol: output
            .y_zcol
            .iter()
            .map(|value| alloc_k(builder, *value))
            .collect(),
    }
}

#[test]
fn active_profile_decodes_every_native_and_r1cs_field_exactly_once() {
    let profile = Profile::active_f_prime();
    assert_eq!(profile.field_count(), ACTIVE_F_PRIME_FIELD_COUNT);
    assert_eq!(Profile::new(15, 3).field_count(), LEGACY_THREE_MATRIX_FIELD_COUNT);
    assert_ne!(profile.field_count(), LEGACY_THREE_MATRIX_FIELD_COUNT);

    let native = native_outputs(profile);
    let claims: Vec<_> = native.iter().map(claim).collect();
    let expected = pi_ccs_outputs_preimage(&claims);
    assert_eq!(expected.len(), ACTIVE_F_PRIME_FIELD_COUNT);

    let mut builder = R1csBuilder::new();
    let wires: Vec<_> = native
        .iter()
        .map(|output| wire_output(&mut builder, output))
        .collect();
    let inputs: Vec<_> = wires
        .iter()
        .map(|output| PiCcsOutputMessageDigestInputs {
            y_ring: &output.y_ring,
            y_zcol: &output.y_zcol,
        })
        .collect();
    let decoded = encode_pi_ccs_outputs_preimage(&mut builder, profile, &inputs).expect("active output preimage");

    assert_eq!(decoded.profile(), profile);
    assert_eq!(decoded.fields().len(), ACTIVE_F_PRIME_FIELD_COUNT);
    assert_eq!(profile.decode(ACTIVE_F_PRIME_FIELD_COUNT), None);

    let mut dynamic_columns = HashSet::new();
    let mut paths = HashSet::new();
    let mut owner_counts = [0usize; 3];
    for (index, (binding, expected_value)) in decoded.fields().iter().zip(&expected).enumerate() {
        assert_eq!(binding.index(), index);
        assert_eq!(Some(binding.path()), profile.decode(index));
        let expected_wire = match binding.path() {
            FieldPath::YRingLimb {
                source,
                matrix,
                lane,
                limb,
            } => Some(match limb {
                KLimb::C0 => wires[source].y_ring[matrix][lane].c0,
                KLimb::C1 => wires[source].y_ring[matrix][lane].c1,
            }),
            FieldPath::YZcolLimb { source, lane, limb } => Some(match limb {
                KLimb::C0 => wires[source].y_zcol[lane].c0,
                KLimb::C1 => wires[source].y_zcol[lane].c1,
            }),
            _ => None,
        };
        if let Some(expected_wire) = expected_wire {
            assert_eq!(binding.wire(), expected_wire, "wrong source wire at field {index}");
            assert!(
                dynamic_columns.insert(binding.source_column()),
                "dynamic field {index} reused a source column"
            );
        }
        assert_eq!(builder.witness()[binding.source_column()], *expected_value);
        assert!(paths.insert(binding.path()), "field {index} reused a typed path");
        owner_counts[match binding.r1cs_input_owner() {
            R1csInputOwner::VerifierShape => 0,
            R1csInputOwner::YRingOutput => 1,
            R1csInputOwner::YZcolOutput => 2,
        }] += 1;
    }
    assert_eq!(paths.len(), ACTIVE_F_PRIME_FIELD_COUNT);
    assert_eq!(owner_counts, [353, 21_060, 1_620]);
    assert!(builder.is_satisfied(), "verifier-owned constants must be pinned");

    assert_eq!(profile.decode(7), Some(FieldPath::SourceCount));
    assert_eq!(
        profile.decode(18),
        Some(FieldPath::YRingLimb {
            source: 0,
            matrix: 0,
            lane: 0,
            limb: KLimb::C0,
        })
    );
    assert_eq!(
        profile.decode(ACTIVE_F_PRIME_FIELD_COUNT - 1),
        Some(FieldPath::YZcolLimb {
            source: 14,
            lane: D - 1,
            limb: KLimb::C1,
        })
    );
}

#[test]
fn profile_pinned_decoder_rejects_source_and_matrix_shape_drift() {
    let profile = Profile::new(2, 3);
    let native = native_outputs(profile);
    let mut builder = R1csBuilder::new();
    let wires: Vec<_> = native
        .iter()
        .map(|output| wire_output(&mut builder, output))
        .collect();
    let inputs: Vec<_> = wires
        .iter()
        .map(|output| PiCcsOutputMessageDigestInputs {
            y_ring: &output.y_ring,
            y_zcol: &output.y_zcol,
        })
        .collect();

    let source_error = encode_pi_ccs_outputs_preimage(&mut builder, Profile::new(3, 3), &inputs)
        .err()
        .expect("wrong source count must fail");
    assert!(source_error.to_string().contains("source count 2"));

    let matrix_error = encode_pi_ccs_outputs_preimage(&mut builder, Profile::new(2, 13), &inputs)
        .err()
        .expect("legacy matrix count must not inhabit active matrix profile");
    assert!(matrix_error
        .to_string()
        .contains("3 y_ring matrices, expected 13"));
}

#[test]
fn sis_digest_matches_the_native_active_lane_message() {
    let profile = Profile::new(1, 3);
    let native = native_outputs(profile);
    let claims: Vec<_> = native.iter().map(claim).collect();
    let expected = pi_ccs_outputs_digest(&claims);

    let mut builder = R1csBuilder::new();
    let wires: Vec<_> = native
        .iter()
        .map(|output| wire_output(&mut builder, output))
        .collect();
    let inputs: Vec<_> = wires
        .iter()
        .map(|output| PiCcsOutputMessageDigestInputs {
            y_ring: &output.y_ring,
            y_zcol: &output.y_zcol,
        })
        .collect();
    let digest = enforce_pi_ccs_outputs_digest(&mut builder, profile, &inputs)
        .expect("Pi_CCS output digest")
        .digest;

    assert_eq!(digest.map(|wire| builder.witness()[wire.col()]), expected);
    assert!(builder.is_satisfied());

    let mut changed = claims;
    changed[0].y_zcol[0] += K::ONE;
    assert_ne!(
        pi_ccs_outputs_digest(&changed),
        expected,
        "the pre-rho digest must bind the active y_zcol payload"
    );
}
