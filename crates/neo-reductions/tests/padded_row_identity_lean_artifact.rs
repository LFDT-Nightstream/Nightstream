//! Rust-to-Lean drift gate for the padded-row PiCCS layout and proof codec.

use neo_ajtai::Commitment;
use neo_ccs::{CeClaim, Mat};
use neo_math::{D, F};
use neo_reductions::engines::pi_ccs_joint::{carried_gamma_exponent, JointDims};
use neo_reductions::engines::pi_ccs_joint_protocol::output_message_fields;
use neo_reductions::optimized_engine::PiCcsProof;
use p3_field::{PrimeCharacteristicRing, PrimeField64};

const FRESH_COUNT: usize = 2;
const RUNNING_COUNT: usize = 2;
const MATRIX_COUNT: usize = 3;
const PRODUCTION_FRESH_COUNT: usize = 1;
const PRODUCTION_APPLICATION_MATRIX_COUNT: usize = 13;
const ARTIFACT_REL_PATH: &str = "/../../formal/nightstream-lean/Nightstream/Implementation/Rust/CanonicalConformance/PiCcsPaddedRowIdentity/Generated/Layout.lean";

fn sample_proof_words() -> Vec<u64> {
    let rounds = vec![
        vec![
            neo_math::from_complex(F::from_u64(1), F::from_u64(101)),
            neo_math::from_complex(F::from_u64(2), F::from_u64(102)),
            neo_math::from_complex(F::from_u64(3), F::from_u64(103)),
        ],
        vec![
            neo_math::from_complex(F::from_u64(4), F::from_u64(104)),
            neo_math::from_complex(F::from_u64(5), F::from_u64(105)),
            neo_math::from_complex(F::from_u64(6), F::from_u64(106)),
        ],
    ];
    PiCcsProof::new(rounds)
        .canonical_bytes()
        .chunks_exact(8)
        .map(|chunk| u64::from_le_bytes(chunk.try_into().expect("eight-byte word")))
        .collect()
}

fn sample_output_words() -> Vec<u64> {
    let source_count = FRESH_COUNT + RUNNING_COUNT;
    let outputs = (0..source_count)
        .map(|source| {
            let y_ring = (0..MATRIX_COUNT)
                .map(|matrix| {
                    (0..D)
                        .map(|coefficient| {
                            let ordinal = source * MATRIX_COUNT * D + matrix * D + coefficient + 1;
                            neo_math::from_complex(F::from_u64(ordinal as u64), F::from_u64((10_000 + ordinal) as u64))
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            CeClaim {
                c: Commitment::zeros(D, 1),
                X: Mat::zero(D, 0, F::ZERO),
                r: Vec::new(),
                ct: y_ring.iter().map(|row| row[0]).collect(),
                y_ring,
                m_in: 0,
                fold_digest: [0; 32],
                adv: None,
            }
        })
        .collect::<Vec<_>>();
    output_message_fields(
        &outputs,
        JointDims {
            assignment_width: D,
            row_count: D.next_power_of_two(),
            variables: D.next_power_of_two().trailing_zeros() as usize,
            matrix_count: MATRIX_COUNT,
            degree: 1,
        },
    )
    .expect("selected output codec")
    .iter()
    .map(PrimeField64::as_canonical_u64)
    .collect()
}

fn construction3_output_frame(compact: &[u64]) -> Vec<u64> {
    let (&message_type, payload) = compact
        .split_first()
        .expect("PiCCS output has its message-type tag");
    assert_eq!(message_type, 47, "selected PiCCS output message type");

    let label = b"prover-message";
    let mut fields = Vec::with_capacity(2 + label.len() + 4 + payload.len());
    fields.extend([32, label.len() as u64]);
    fields.extend(label.iter().map(|byte| u64::from(*byte)));
    fields.extend([51, 25, message_type, payload.len() as u64]);
    fields.extend_from_slice(payload);
    fields
}

fn render() -> String {
    let mut carried = Vec::with_capacity(RUNNING_COUNT * MATRIX_COUNT * D);
    for coefficient in 0..D {
        for matrix in 0..MATRIX_COUNT {
            for running in 0..RUNNING_COUNT {
                carried.push(carried_gamma_exponent(
                    FRESH_COUNT,
                    RUNNING_COUNT,
                    MATRIX_COUNT,
                    running,
                    matrix,
                    coefficient,
                ));
            }
        }
    }
    let carried = carried.iter().map(usize::to_string).collect::<Vec<_>>();
    let proof_words = sample_proof_words()
        .iter()
        .map(u64::to_string)
        .collect::<Vec<_>>();
    let compact_output_words = sample_output_words();
    let output_words = construction3_output_frame(&compact_output_words);
    let output_coordinate_count = (FRESH_COUNT + RUNNING_COUNT) * MATRIX_COUNT * D;
    let mut expected_output_words = Vec::with_capacity(20 + 2 * output_coordinate_count);
    expected_output_words.extend([
        32,
        14,
        112,
        114,
        111,
        118,
        101,
        114,
        45,
        109,
        101,
        115,
        115,
        97,
        103,
        101,
        51,
        25,
        47,
        (2 * output_coordinate_count) as u64,
    ]);
    for ordinal in 1..=output_coordinate_count as u64 {
        expected_output_words.extend([ordinal, 10_000 + ordinal]);
    }
    assert_eq!(
        output_words, expected_output_words,
        "Rust output message is not source/matrix/coefficient and low/high limb ordered"
    );
    let sample_output_field_count = output_words.len();
    let production_running_count = neo_params::goldilocks_paper_b2::K_RHO as usize;
    let production_matrix_count = PRODUCTION_APPLICATION_MATRIX_COUNT + 1;
    let production_output_payload_count =
        (PRODUCTION_FRESH_COUNT + production_running_count) * production_matrix_count * D * 2;
    let production_output_field_count = 20 + production_output_payload_count;

    format!(
        "import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Coefficients\n\n\
/-!\n\
GENERATED FILE - do not edit by hand.\n\n\
Exact Rust output for the selected gamma slots and PiCCS proof codec.\n\
Regenerated and drift-checked by\n\
`cargo test -p neo-reductions --release --test padded_row_identity_lean_artifact`.\n\
-/\n\n\
namespace Nightstream.Implementation.Rust.CanonicalConformance.PiCcsPaddedRowIdentity.Generated\n\n\
def freshCount : Nat := {FRESH_COUNT}\n\
def runningCount : Nat := {RUNNING_COUNT}\n\
def matrixCount : Nat := {MATRIX_COUNT}\n\
def coefficientCount : Nat := {D}\n\
def carriedCount : Nat := {}\n\
def productionFreshCount : Nat := {PRODUCTION_FRESH_COUNT}\n\
def productionRunningCount : Nat := {production_running_count}\n\
def productionMatrixCount : Nat := {production_matrix_count}\n\
def productionOutputFieldCount : Nat := {production_output_field_count}\n\
def rowVariablesWhenRowsLtAssignment : Nat := 6\n\
def rowVariablesWhenRowsGtAssignment : Nat := 7\n\
def transcriptTags : List Nat := [40, 41, 42, 43, 45, 46, 47]\n\n\
def freshGammaExponents : List Nat := [0, 1]\n\
def normGammaExponents : List Nat := [2, 3, 4, 5]\n\
def carriedGammaExponents : List Nat :=\n  [{}]\n\n\
def sampleProofWords : List Nat :=\n  [{}]\n\n\
def sampleOutputFieldCount : Nat := {sample_output_field_count}\n\
def sampleOutputOrderMatches : Bool := true\n\n\
end Nightstream.Implementation.Rust.CanonicalConformance.PiCcsPaddedRowIdentity.Generated\n",
        carried.len(),
        carried.join(", "),
        proof_words.join(", ")
    )
}

#[test]
fn padded_row_identity_lean_artifact_matches_rust() {
    assert_eq!(D, 54, "the bounded artifact is pinned to Phi81");
    let emitted = render();
    let path = format!("{}{}", env!("CARGO_MANIFEST_DIR"), ARTIFACT_REL_PATH);
    let committed = std::fs::read_to_string(&path).unwrap_or_default();
    if committed != emitted {
        let expected_path = format!("{path}.expected");
        let parent = std::path::Path::new(&expected_path)
            .parent()
            .expect("artifact path has a parent");
        std::fs::create_dir_all(parent).expect("create generated artifact directory");
        std::fs::write(&expected_path, emitted).expect("write expected artifact");
        panic!("padded-row Lean artifact drifted; inspect and promote {expected_path}");
    }
}
