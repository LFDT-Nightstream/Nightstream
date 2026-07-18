//! Compact drift manifest for the exact steady-state plain F' recursive step.
//!
//! Owns: production-row regeneration, top-level/NIFS partition checks,
//! projection census, source provenance, and Lean-data drift output.
//!
//! Does not own: semantic soundness, cryptographic reductions, or permission
//! to trust a manifest instead of the production relation.
//!
//! Emits constraints: no; it executes the production builder and audits its
//! emitted rows.
//!
//! Authority boundary: generated JSON and Lean data are review artifacts only;
//! every run recomputes rows and hashes from the current source.
//!
//! | Artifact branch | Guarantee | Evidence tier | Permits row removal? |
//! |---|---|---|---|
//! | Top-level families | Recursive rows form one gap-free partition | artifact-checked | no |
//! | NIFS families | PiCCS, PiRLC, PiDEC, and point binding partition NIFS | artifact-checked | no |
//! | Projection census | Every identity shares one rho-evaluation phase | artifact-checked | no |
//! | Source hashes | Reviewed implementation surface is explicit | drift sentinel | no |

use std::collections::HashMap;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

use neo_ccs::Mat;
use neo_fold_clean::engine::r1cs_circuit::builder::RowFamilyRange;
use neo_fold_clean::engine::r1cs_circuit::projection_identity_trace::validate_projection_identity_traces;
use neo_fold_clean::engine::r1cs_circuit::ring_action::PROJECTION_QUOTIENT_LEN;
use neo_fold_clean::engine::r1cs_circuit::{
    PolynomialEvaluationTraceTestMutation, ProjectionIdentityRole, ProjectionIdentityTraceTestMutation, R1csBuilder,
};
use neo_fold_clean::engine::transcript::Transcript;
use neo_fold_clean::frontends::direct_ccs::{self, R1cs};
use neo_fold_clean::paper::construction2::RunningInstance;
use neo_fold_clean::paper::digest::{
    digest32_as_fields, digest_fields_as_digest32, f_prime_chunk_public_digest, state_x_out_digest_with_mode,
    AccumulatorHandle, StateXOutDigestMode,
};
use neo_fold_clean::paper::f_prime::r1cs::{
    encode_f_prime_superneo_public_input, enforce_f_prime_recursive_step_circuit, FPrimePublicInputLayout,
    FPrimeRecursiveInputs, FPrimeStateIn, FPrimeStepConfig, F_PRIME_ENC_INST_BITS, F_PRIME_PUBLIC_INPUT_LEN,
    F_PRIME_SUPERNEO_PUBLIC_INPUT_LEN,
};
use neo_fold_clean::paper::f_prime::source_image::{BitRange, FPrimeSourceImage, Word64Image};
use neo_fold_clean::paper::nifs::circuit::{NifsVCircuitConfig, NifsVCircuitMessages};
use neo_fold_clean::paper::nifs::NifsProof;
use neo_fold_clean::paper::reductions::pi_ccs_split_nc_circuit::SplitNcPiCcsVConfig;
use neo_fold_clean::paper::relations::{CcsClaim, CeClaim};
use neo_math::ring::D;
use neo_math::F;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

const MANIFEST_PATH: &str = "formal/nightstream-lean/assurance/fprime-recursive-program-manifest.json";
const LEAN_DATA_PATH: &str = "formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeRecursive/Generated/FPrimeRecursiveManifestData.lean";
const TRANSCRIPT_LABEL: &[u8] = b"neo.formal.f_prime/recursive-manifest/v1";
const TOP_LEVEL_FAMILIES: &[&str] = &[
    "fprime.recursive.prelude",
    "fprime.recursive.transcript",
    "fprime.recursive.nifs",
    "fprime.recursive.prior_link",
    "fprime.recursive.nebula",
    "fprime.recursive.accumulator",
    "fprime.recursive.counter",
    "fprime.recursive.output",
];
const NIFS_FAMILIES: &[&str] = &["nifs.pi_ccs", "nifs.pi_rlc", "nifs.pi_dec", "nifs.point_binding"];
const PROJECTION_SHARED_FAMILY: &str = "nifs.pi_rlc.projection_shared";
const PROJECTION_IDENTITY_FAMILY: &str = "nifs.pi_rlc.projection_identity";
const K_MUL_ROWS: usize = 5;

/// Direct row-emission, phase-composition, and Lean-consumer owners covered by
/// the manifest drift sentinel. Keep this grouped by ownership layer so a
/// protocol-phase change cannot update row hashes while escaping provenance.
const SOURCE_PATHS: &[&str] = &[
    // Shared R1CS emitters used by every child verifier.
    "crates/neo-fold-clean/src/engine/r1cs_circuit/builder.rs",
    "crates/neo-fold-clean/src/engine/r1cs_circuit/field_ext.rs",
    "crates/neo-fold-clean/src/engine/r1cs_circuit/poseidon2.rs",
    "crates/neo-fold-clean/src/engine/r1cs_circuit/projection_identity_trace.rs",
    "crates/neo-fold-clean/src/engine/r1cs_circuit/ring_action.rs",
    "crates/neo-fold-clean/src/engine/r1cs_circuit/sumcheck.rs",
    "crates/neo-fold-clean/src/engine/r1cs_circuit/transcript.rs",
    "crates/neo-fold-clean/src/engine/r1cs_circuit/u64.rs",
    // PiRLC transcript challenge and rejection sampler.
    "crates/neo-fold-clean/src/engine/r1cs_circuit/alphabet_sampling/mod.rs",
    "crates/neo-fold-clean/src/engine/r1cs_circuit/alphabet_sampling/acceptance.rs",
    "crates/neo-fold-clean/src/engine/r1cs_circuit/alphabet_sampling/chunk.rs",
    "crates/neo-fold-clean/src/engine/r1cs_circuit/alphabet_sampling/digest_rounds.rs",
    "crates/neo-fold-clean/src/engine/r1cs_circuit/alphabet_sampling/selection.rs",
    // F-prime and NIFS protocol composition.
    "crates/neo-fold-clean/src/paper/f_prime/r1cs.rs",
    "crates/neo-fold-clean/src/paper/nifs/circuit/mod.rs",
    "crates/neo-fold-clean/src/paper/nifs/circuit/stage.rs",
    "crates/neo-fold-clean/src/paper/nifs/circuit/pi_rlc/mod.rs",
    "crates/neo-fold-clean/src/paper/nifs/circuit/pi_rlc/consistency.rs",
    "crates/neo-fold-clean/src/paper/nifs/circuit/pi_rlc/fold_wires.rs",
    "crates/neo-fold-clean/src/paper/nifs/circuit/pi_rlc/padding.rs",
    "crates/neo-fold-clean/src/paper/nifs/circuit/pi_rlc/projection/mod.rs",
    "crates/neo-fold-clean/src/paper/nifs/circuit/pi_rlc/projection/binding.rs",
    "crates/neo-fold-clean/src/paper/nifs/circuit/pi_rlc/projection/identities.rs",
    "crates/neo-fold-clean/src/paper/nifs/circuit/pi_rlc/projection/shared.rs",
    // PiCCS constraint families.
    "crates/neo-fold-clean/src/paper/reductions/pi_ccs_split_nc_circuit/mod.rs",
    "crates/neo-fold-clean/src/paper/reductions/pi_ccs_split_nc_circuit/stage.rs",
    "crates/neo-fold-clean/src/paper/reductions/pi_ccs_split_nc_circuit/digests.rs",
    "crates/neo-fold-clean/src/paper/reductions/pi_ccs_split_nc_circuit/fe.rs",
    "crates/neo-fold-clean/src/paper/reductions/pi_ccs_split_nc_circuit/nc.rs",
    "crates/neo-fold-clean/src/paper/reductions/pi_ccs_split_nc_circuit/transcript.rs",
    "crates/neo-fold-clean/src/paper/reductions/pi_ccs_split_nc_circuit/verifier.rs",
    // Shared SIS, PiRLC algebra, and PiDEC constraint families.
    "crates/neo-fold-clean/src/paper/reductions/accumulator_sis_circuit.rs",
    "crates/neo-fold-clean/src/paper/reductions/pi_rlc_circuit/mod.rs",
    "crates/neo-fold-clean/src/paper/reductions/pi_rlc_circuit/stage.rs",
    "crates/neo-fold-clean/src/paper/reductions/pi_rlc_circuit/commitment.rs",
    "crates/neo-fold-clean/src/paper/reductions/pi_rlc_circuit/consistency.rs",
    "crates/neo-fold-clean/src/paper/reductions/pi_rlc_circuit/padded_k.rs",
    "crates/neo-fold-clean/src/paper/reductions/pi_rlc_circuit/x.rs",
    "crates/neo-fold-clean/src/paper/reductions/pi_rlc_circuit/y_ring.rs",
    "crates/neo-fold-clean/src/paper/reductions/pi_dec_circuit.rs",
    // Drift-gate implementation and Lean consumers.
    "crates/neo-fold-clean/tests/gadgets/f_prime_recursive_manifest.rs",
    "formal/nightstream-lean/Nightstream/Implementation/R1CS/Ownership/FPrimeRecursive/FPrimeRecursiveManifestSchema.lean",
    "formal/nightstream-lean/Nightstream/Implementation/R1CS/Ownership/FPrimeRecursive/FPrimeRecursiveManifest.lean",
    "formal/nightstream-lean/Nightstream/SuperNeo/ProjectionCheck.lean",
    "formal/nightstream-lean/Nightstream/Assurance/FPrimeRecursiveCircuit.lean",
];

#[path = "f_prime_recursive_manifest/aggregate_acceptance_outer_image.rs"]
mod aggregate_acceptance_outer_image;
#[path = "f_prime_recursive_manifest/output_authority_poseidon2_sbox.rs"]
mod output_authority_poseidon2_sbox;
#[path = "f_prime_recursive_manifest/output_authority_sbox_lean.rs"]
mod output_authority_sbox_lean;
#[path = "f_prime_recursive_manifest/pi_rlc_transcript_schedule.rs"]
mod pi_rlc_transcript_schedule;
#[path = "f_prime_recursive_manifest/projection_binding_shape.rs"]
mod projection_binding_shape;
#[path = "f_prime_recursive_manifest/projection_certificate.rs"]
mod projection_certificate;

struct Fixture {
    prep: neo_fold_clean::Preprocessing,
    fresh_claims: Vec<CcsClaim>,
    running: RunningInstance,
    proof: NifsProof,
    combined: CeClaim,
    children: Vec<CeClaim>,
    state: FPrimeStateIn,
    chunk_digest: [F; 4],
}

struct RecursiveSource {
    image: FPrimeSourceImage,
    chunk_count: Word64Image,
    step_count: Word64Image,
    pc: Word64Image,
    prior_x_out: BitRange,
    public_x_out: BitRange,
}

fn bit_carrier_r1cs() -> R1cs {
    let padding = F_PRIME_SUPERNEO_PUBLIC_INPUT_LEN - F_PRIME_PUBLIC_INPUT_LEN;
    let mut a = Mat::zero(padding, F_PRIME_SUPERNEO_PUBLIC_INPUT_LEN, F::ZERO);
    let mut b = Mat::zero(padding, F_PRIME_SUPERNEO_PUBLIC_INPUT_LEN, F::ZERO);
    for row in 0..padding {
        a[(row, F_PRIME_PUBLIC_INPUT_LEN + row)] = F::ONE;
        b[(row, 0)] = F::ONE;
    }
    R1cs {
        a,
        b,
        c: Mat::zero(padding, F_PRIME_SUPERNEO_PUBLIC_INPUT_LEN, F::ZERO),
        m_in: F_PRIME_SUPERNEO_PUBLIC_INPUT_LEN,
    }
}

fn rand_digest(seed: u64) -> [F; 4] {
    std::array::from_fn(|lane| F::from_u64(seed.wrapping_mul(31).wrapping_add(lane as u64 + 1)))
}

fn append_step_context(transcript: &mut Transcript, state: &FPrimeStateIn, chunk_digest: [F; 4]) {
    transcript.append_fields(b"f_prime/vk_fs", &state.vk_fs_digest);
    transcript.append_fields(b"f_prime/pi_ccs_header", &state.pi_ccs_header_bundle);
    transcript.append_fields(b"f_prime/chunk_count_in", &[F::from_u64(state.chunk_count_in)]);
    transcript.append_fields(b"f_prime/step_count_in", &[F::from_u64(state.step_count_in)]);
    transcript.append_fields(b"f_prime/z_0", &state.z_0);
    transcript.append_fields(b"f_prime/z_i_in", &state.z_i_in);
    transcript.append_fields(b"f_prime/pc", &[F::from_u64(state.pc)]);
    transcript.append_fields(b"f_prime/semantic_state_in", &state.semantic_state_digest_in);
    transcript.append_fields(b"f_prime/acc_digest_in", &state.acc_digest_in);
    transcript.append_fields(b"f_prime/public_trace_in", &state.public_trace_in);
    transcript.append_fields(b"f_prime/chunk_digest", &chunk_digest);
}

fn running_digest(running: &RunningInstance) -> [F; 4] {
    AccumulatorHandle::from_running_parts(&running.claims, running.parent_authority.as_ref()).digest_fields()
}

fn prior_x_out(state: &FPrimeStateIn) -> [F; 4] {
    digest32_as_fields(state_x_out_digest_with_mode(
        StateXOutDigestMode::Stateless,
        digest_fields_as_digest32(state.vk_fs_digest),
        state.pi_ccs_header_bundle,
        &state.pi_ccs_header_bundle,
        state.chunk_count_in,
        state.step_count_in,
        digest_fields_as_digest32(state.z_0),
        digest_fields_as_digest32(state.z_i_in),
        state.pc,
        digest_fields_as_digest32(state.acc_digest_in),
        digest_fields_as_digest32(state.acc_digest_in),
        digest_fields_as_digest32(state.public_trace_in),
        None,
    ))
}

fn split_nc_config(prep: &neo_fold_clean::Preprocessing) -> SplitNcPiCcsVConfig<'_> {
    let raw_params = neo_params::NeoParams::goldilocks_auto_r1cs_ccs_with(
        prep.structure().n.max(prep.structure().m),
        neo_fold_clean::config::MIN_EFFECTIVE_LAMBDA,
        neo_fold_clean::config::EXTENSION_SAFETY_MARGIN_BITS,
    )
    .expect("raw params");
    let dims =
        neo_reductions::engines::utils::build_dims_and_policy(&raw_params, prep.structure()).expect("engine dims");
    let matrix_digest = neo_reductions::engines::utils::digest_ccs_matrices_with_sparse_cache(prep.structure(), None);
    let header_bundle = neo_reductions::engines::utils::pi_ccs_header_bundle_digest_fields(
        &raw_params,
        prep.structure(),
        dims,
        &matrix_digest,
    )
    .expect("header bundle");
    SplitNcPiCcsVConfig {
        params: &prep.params,
        structure: prep.structure().into(),
        header_bundle,
        ell_d: dims.ell_d,
        ell_n: dims.ell_n,
        ell_m: dims.ell_m,
        d_sc: dims.d_sc,
    }
}

fn step_config(prep: &neo_fold_clean::Preprocessing) -> FPrimeStepConfig<'_> {
    FPrimeStepConfig {
        nifs: NifsVCircuitConfig {
            pi_ccs: split_nc_config(prep),
        },
        b: prep.params.b(),
        transcript_label: TRANSCRIPT_LABEL,
        public_input_layout: FPrimePublicInputLayout::plain(),
        nebula: None,
        state_x_out_digest_mode: StateXOutDigestMode::Stateless,
    }
}

fn build_fixture() -> Fixture {
    let r1cs = bit_carrier_r1cs();
    let prep = direct_ccs::preprocess_seeded(&r1cs, 42).expect("preprocess bit carrier");
    let zero_assignment = vec![F::ZERO; prep.structure().m];
    let first = direct_ccs::build_instance(&prep, &r1cs, &zero_assignment).expect("first instance");
    let mut first_transcript = Transcript::session();
    let (running, _) = neo_fold_clean::paper::nifs::prove(
        &mut first_transcript,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        &prep.log,
        None,
        prep.mix_rhos_commits(),
        prep.combine_b_pows(),
        vec![first],
        &RunningInstance::default(),
    )
    .expect("seed running accumulator");

    let acc_digest = running_digest(&running);
    let state = FPrimeStateIn {
        vk_fs_digest: rand_digest(0x10),
        pi_ccs_header_bundle: prep.pi_ccs_header_bundle(),
        chunk_count_in: 1,
        step_count_in: 1,
        z_0: rand_digest(0x100),
        z_i_in: rand_digest(0x101),
        pc: 1,
        semantic_state_digest_in: acc_digest,
        acc_digest_in: acc_digest,
        public_trace_in: rand_digest(0x40),
        nebula: None,
    };
    let mut assignment = encode_f_prime_superneo_public_input(prior_x_out(&state));
    assignment.resize(prep.structure().m, F::ZERO);
    let fresh = direct_ccs::build_instance(&prep, &r1cs, &assignment).expect("linked fresh instance");
    let fresh_claims = vec![fresh.claim.clone()];
    let chunk_digest = f_prime_chunk_public_digest(state.step_count_in, &fresh_claims);
    let mut transcript = Transcript::with_label(TRANSCRIPT_LABEL);
    append_step_context(&mut transcript, &state, chunk_digest);
    let (next_running, proof) = neo_fold_clean::paper::nifs::prove(
        &mut transcript,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        &prep.log,
        None,
        prep.mix_rhos_commits(),
        prep.combine_b_pows(),
        vec![fresh],
        &running,
    )
    .expect("recursive NIFS proof");
    let combined = proof.pi_rlc.combined.clone();
    Fixture {
        prep,
        fresh_claims,
        running,
        children: next_running.claims,
        proof,
        combined,
        state,
        chunk_digest,
    }
}

fn recursive_acc_digest(fixture: &Fixture) -> [F; 4] {
    AccumulatorHandle::from_running_parts(&fixture.children, Some(&fixture.combined)).digest_fields()
}

fn recursive_x_out(fixture: &Fixture) -> [F; 4] {
    let new_acc = recursive_acc_digest(fixture);
    let boundary = digest_fields_as_digest32(fixture.chunk_digest);
    digest32_as_fields(state_x_out_digest_with_mode(
        StateXOutDigestMode::Stateless,
        digest_fields_as_digest32(fixture.state.vk_fs_digest),
        fixture.state.pi_ccs_header_bundle,
        &fixture.state.pi_ccs_header_bundle,
        fixture.state.chunk_count_in + 1,
        fixture.state.step_count_in + 1,
        digest_fields_as_digest32(fixture.state.z_0),
        boundary,
        fixture.state.pc,
        digest_fields_as_digest32(new_acc),
        digest_fields_as_digest32(new_acc),
        boundary,
        None,
    ))
}

fn source_image(fixture: &Fixture) -> RecursiveSource {
    let mut image = FPrimeSourceImage::new();
    let chunk_count = image.push_u64_le(fixture.state.chunk_count_in);
    let step_count = image.push_u64_le(fixture.state.step_count_in);
    let pc = image.push_u64_le(fixture.state.pc);
    let prior = image.push_f_prime_public_input(prior_x_out(&fixture.state));
    let prior_x_out = BitRange::new(prior.start() + 1, F_PRIME_ENC_INST_BITS);
    let public_x_out = image.push_enc_inst(recursive_x_out(fixture));
    RecursiveSource {
        image,
        chunk_count,
        step_count,
        pc,
        prior_x_out,
        public_x_out,
    }
}

fn build_recursive_program() -> R1csBuilder {
    let fixture = build_fixture();
    let config = step_config(&fixture.prep);
    let source = source_image(&fixture);
    let output_acc = recursive_acc_digest(&fixture);
    let inputs = FPrimeRecursiveInputs {
        state: fixture.state,
        chunk_digest: fixture.chunk_digest,
        semantic_state_digest_out: output_acc,
        acc_digest_out: output_acc,
        nifs_msg: NifsVCircuitMessages {
            fresh: &fixture.fresh_claims,
            running: &fixture.running.claims,
            running_parent_authority: fixture.running.parent_authority.as_ref(),
            pi_ccs: &fixture.proof.pi_ccs,
            combined: &fixture.combined,
            children: &fixture.children,
        },
        rows_in_chunk: 1,
        source_image: &source.image,
        chunk_count_in_word: source.chunk_count,
        step_count_in_word: source.step_count,
        pc_word: source.pc,
        prior_x_out_bits: source.prior_x_out,
        public_x_out_bits: source.public_x_out,
    };
    let mut builder = R1csBuilder::new();
    builder.enable_encoding_trace();
    enforce_f_prime_recursive_step_circuit(&mut builder, &fixture.prep.params, &config, &inputs)
        .expect("emit recursive program");
    builder.begin_encoding_stage("complete");
    builder
}

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("repository root")
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn source_hash(relative: &str) -> Value {
    let bytes = fs::read(repo_root().join(relative)).unwrap_or_else(|error| panic!("read {relative}: {error}"));
    json!({ "path": relative, "sha256": sha256_hex(&bytes) })
}

fn range_hash(builder: &R1csBuilder, range: &RowFamilyRange) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"nightstream/fprime-recursive-row-range/v1");
    hasher.update((range.row_end - range.row_start).to_le_bytes());
    let (a, b, c) = builder.sparse_triplets();
    for (tag, trips) in [(b'A', a), (b'B', b), (b'C', c)] {
        for &(row, column, coefficient) in trips {
            if row < range.row_start || row >= range.row_end {
                continue;
            }
            hasher.update([tag]);
            hasher.update((row - range.row_start).to_le_bytes());
            hasher.update(column.to_le_bytes());
            hasher.update(coefficient.as_canonical_u64().to_le_bytes());
        }
    }
    format!("{:x}", hasher.finalize())
}

/// Hash the exact sparse equality pattern while renaming every non-constant
/// column by first occurrence. Equal hashes mean two ranges are the same row
/// program up to wire allocation names; the exact per-range hash remains the
/// drift authority for concrete columns.
fn range_shape_hash(builder: &R1csBuilder, range: &RowFamilyRange) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"nightstream/fprime-recursive-row-shape/v1");
    hasher.update((range.row_end - range.row_start).to_le_bytes());
    let mut renaming = HashMap::from([(0usize, 0usize)]);
    let mut next_column = 1usize;
    let (a, b, c) = builder.sparse_triplets();
    for (tag, trips) in [(b'A', a), (b'B', b), (b'C', c)] {
        for &(row, column, coefficient) in trips {
            if row < range.row_start || row >= range.row_end {
                continue;
            }
            let normalized_column = *renaming.entry(column).or_insert_with(|| {
                let current = next_column;
                next_column += 1;
                current
            });
            hasher.update([tag]);
            hasher.update((row - range.row_start).to_le_bytes());
            hasher.update(normalized_column.to_le_bytes());
            hasher.update(coefficient.as_canonical_u64().to_le_bytes());
        }
    }
    format!("{:x}", hasher.finalize())
}

fn range_nonzeros(builder: &R1csBuilder, range: &RowFamilyRange) -> usize {
    let (a, b, c) = builder.sparse_triplets();
    a.iter()
        .chain(b)
        .chain(c)
        .filter(|&&(row, _, _)| row >= range.row_start && row < range.row_end)
        .count()
}

fn range_json(builder: &R1csBuilder, range: &RowFamilyRange) -> Value {
    json!({
        "name": range.name,
        "row_start": range.row_start,
        "row_end": range.row_end,
        "row_count": range.row_end - range.row_start,
        "nonzero_entries": range_nonzeros(builder, range),
        "sha256": range_hash(builder, range),
    })
}

fn one_range<'a>(builder: &'a R1csBuilder, name: &str) -> &'a RowFamilyRange {
    let matches = builder
        .row_family_ranges()
        .iter()
        .filter(|range| range.name == name)
        .collect::<Vec<_>>();
    assert_eq!(matches.len(), 1, "expected one {name} range, got {}", matches.len());
    matches[0]
}

fn ranges_inside<'a>(builder: &'a R1csBuilder, owner: &RowFamilyRange, names: &[&str]) -> Vec<&'a RowFamilyRange> {
    let mut ranges = names
        .iter()
        .map(|name| {
            let matches = builder
                .row_family_ranges()
                .iter()
                .filter(|range| {
                    range.name == *name && range.row_start >= owner.row_start && range.row_end <= owner.row_end
                })
                .collect::<Vec<_>>();
            assert_eq!(matches.len(), 1, "expected one {name} inside {}", owner.name);
            matches[0]
        })
        .collect::<Vec<_>>();
    ranges.sort_by_key(|range| range.row_start);
    ranges
}

fn all_named_ranges_inside<'a>(
    builder: &'a R1csBuilder,
    owner: &RowFamilyRange,
    name: &str,
) -> Vec<&'a RowFamilyRange> {
    let mut ranges = builder
        .row_family_ranges()
        .iter()
        .filter(|range| range.name == name && range.row_start >= owner.row_start && range.row_end <= owner.row_end)
        .collect::<Vec<_>>();
    ranges.sort_by_key(|range| range.row_start);
    ranges
}

fn projection_eval_rows(width: usize) -> usize {
    assert!(width > 0, "projection polynomial must be nonempty");
    2 * (width - 1) + 2
}

fn projection_identity_pair_count(range: &RowFamilyRange) -> usize {
    let fixed_rows = projection_eval_rows(D) + projection_eval_rows(PROJECTION_QUOTIENT_LEN) + K_MUL_ROWS + 2;
    let per_pair_rows = projection_eval_rows(D) + K_MUL_ROWS;
    let row_count = range.row_end - range.row_start;
    assert!(row_count >= fixed_rows, "projection identity shorter than fixed suffix");
    assert_eq!(
        (row_count - fixed_rows) % per_pair_rows,
        0,
        "projection identity is not composed from exact production row blocks",
    );
    let pair_count = (row_count - fixed_rows) / per_pair_rows;
    assert!(pair_count > 0, "projection identity must consume at least one pair");
    pair_count
}

fn assert_partition(owner: &RowFamilyRange, ranges: &[&RowFamilyRange]) {
    let mut cursor = owner.row_start;
    for range in ranges {
        assert_eq!(range.row_start, cursor, "gap or overlap before {}", range.name);
        assert!(range.row_end >= range.row_start, "reversed range {}", range.name);
        cursor = range.row_end;
    }
    assert_eq!(cursor, owner.row_end, "row families do not cover {}", owner.name);
}

fn build_manifest() -> Value {
    let builder = build_recursive_program();
    assert!(builder.is_satisfied(), "honest recursive program must satisfy");
    let builder = &builder;

    let recursive = one_range(builder, "fprime.recursive.total");
    assert_eq!(recursive.row_start, 0, "recursive program must start at row zero");
    let top_level = ranges_inside(builder, recursive, TOP_LEVEL_FAMILIES);
    assert_partition(recursive, &top_level);

    let nifs = ranges_inside(builder, recursive, &["nifs.total"])[0];
    let nifs_families = ranges_inside(builder, nifs, NIFS_FAMILIES);
    assert_partition(nifs, &nifs_families);
    assert_eq!(
        one_range(builder, "fprime.recursive.nifs").row_start,
        nifs.row_start,
        "F' and NIFS ownership must start on the same row",
    );
    assert_eq!(
        one_range(builder, "fprime.recursive.nifs").row_end,
        nifs.row_end,
        "F' and NIFS ownership must end on the same row",
    );

    let pi_rlc = one_range(builder, "nifs.pi_rlc");
    let projection_shared = all_named_ranges_inside(builder, pi_rlc, PROJECTION_SHARED_FAMILY);
    assert_eq!(projection_shared.len(), 1, "expected one shared projection range");
    let projection_identities = all_named_ranges_inside(builder, pi_rlc, PROJECTION_IDENTITY_FAMILY);
    assert!(
        !projection_identities.is_empty(),
        "production PiRLC must emit projection identities"
    );
    let projection_pair_counts = projection_identities
        .iter()
        .map(|range| projection_identity_pair_count(range))
        .collect::<Vec<_>>();
    assert!(
        projection_pair_counts
            .windows(2)
            .all(|window| window[0] == window[1]),
        "every identity in one PiRLC fold must consume the shared rho census",
    );
    let projection_shape_hashes = projection_identities
        .iter()
        .map(|range| range_shape_hash(builder, range))
        .collect::<Vec<_>>();
    assert!(
        projection_shape_hashes
            .windows(2)
            .all(|window| window[0] == window[1]),
        "every production identity must be the same exact sparse program up to wire renaming",
    );
    let shared_row_count = projection_shared[0].row_end - projection_shared[0].row_start;
    let expected_shared_rows = 2 + D * K_MUL_ROWS + projection_pair_counts[0] * projection_eval_rows(D);
    assert_eq!(shared_row_count, expected_shared_rows, "shared ladder/rho row census");
    let projection_identity_rows = projection_identities
        .iter()
        .map(|range| range.row_end - range.row_start)
        .sum::<usize>();

    json!({
        "schema": 2,
        "artifact_kind": "r1cs/f-prime-recursive-program-manifest",
        "profile": {
            "layout": "plain",
            "semantic_mode": "stateless",
            "carrier_relation": "minimal-supported-bit-carrier",
            "carrier_rows": F_PRIME_SUPERNEO_PUBLIC_INPUT_LEN - F_PRIME_PUBLIC_INPUT_LEN,
            "public_input_fields": F_PRIME_SUPERNEO_PUBLIC_INPUT_LEN,
            "current_batch_size": 1,
            "running_shape": "nonempty-steady-recursive",
        },
        "source_hashes": SOURCE_PATHS
            .iter()
            .map(|path| source_hash(path))
            .collect::<Vec<_>>(),
        "recursive_total": range_json(builder, recursive),
        "top_level_families": top_level
            .iter()
            .map(|range| range_json(builder, range))
            .collect::<Vec<_>>(),
        "nifs_families": nifs_families
            .iter()
            .map(|range| range_json(builder, range))
            .collect::<Vec<_>>(),
        "projection_census": {
            "shared": range_json(builder, projection_shared[0]),
            "identity_count": projection_identities.len(),
            "identity_rows": projection_identity_rows,
            "identity_shape_sha256": projection_shape_hashes[0],
            "pair_count_per_identity": projection_pair_counts,
            "identities": projection_identities
                .iter()
                .enumerate()
                .map(|(index, range)| {
                    let mut value = range_json(builder, range);
                    value["index"] = json!(index);
                    value["pair_count"] = json!(projection_identity_pair_count(range));
                    value["shape_sha256"] = json!(range_shape_hash(builder, range));
                    value
                })
                .collect::<Vec<_>>(),
        },
        "full_builder_rows": builder.rows(),
        "full_builder_columns": builder.cols(),
    })
}

fn json_nat(value: &Value, field: &str) -> u64 {
    value[field]
        .as_u64()
        .unwrap_or_else(|| panic!("manifest field {field} must be a natural number"))
}

fn json_string<'a>(value: &'a Value, field: &str) -> &'a str {
    value[field]
        .as_str()
        .unwrap_or_else(|| panic!("manifest field {field} must be a string"))
}

fn lean_string(value: &str) -> String {
    serde_json::to_string(value).expect("Lean manifest strings are JSON-compatible ASCII")
}

fn append_lean_ranges(rendered: &mut String, definition: &str, ranges: &[Value]) {
    writeln!(rendered, "def {definition} : List RowRange :=").expect("write string");
    for (index, range) in ranges.iter().enumerate() {
        let prefix = if index == 0 { "  [" } else { "  ," };
        writeln!(
            rendered,
            "{prefix} {{ name := {}, rowStart := {}, rowEnd := {}, nonzeroEntries := {}, sha256 := {} }}",
            lean_string(json_string(range, "name")),
            json_nat(range, "row_start"),
            json_nat(range, "row_end"),
            json_nat(range, "nonzero_entries"),
            lean_string(json_string(range, "sha256")),
        )
        .expect("write string");
    }
    rendered.push_str("  ]\n");
}

fn render_lean_data(manifest: &Value) -> String {
    let recursive_total = &manifest["recursive_total"];
    let top_level = manifest["top_level_families"]
        .as_array()
        .expect("top-level row families");
    let nifs = manifest["nifs_families"]
        .as_array()
        .expect("NIFS row families");
    let projection = &manifest["projection_census"];
    let projection_shared = &projection["shared"];
    let projection_identities = projection["identities"]
        .as_array()
        .expect("projection identity ranges");
    let projection_pair_counts = projection["pair_count_per_identity"]
        .as_array()
        .expect("projection pair counts");
    let mut rendered = String::new();
    rendered
        .push_str("import Nightstream.Implementation.R1CS.Ownership.FPrimeRecursive.FPrimeRecursiveManifestSchema\n\n");
    rendered.push_str("/-! Generated by `gadgets_f_prime_recursive_manifest`; do not hand-edit. -/\n\n");
    rendered.push_str("namespace Nightstream.Implementation.R1CS.FPrimeRecursiveManifest\n\n");
    writeln!(rendered, "def schemaVersion : Nat := {}", json_nat(manifest, "schema")).expect("write string");
    writeln!(
        rendered,
        "def artifactKind : String := {}",
        lean_string(json_string(manifest, "artifact_kind"))
    )
    .expect("write string");
    rendered.push_str("def profile : String := \"plain/stateless/minimal-supported-bit-carrier/steady-recursive\"\n");
    writeln!(
        rendered,
        "def totalRows : Nat := {}",
        json_nat(manifest, "full_builder_rows")
    )
    .expect("write string");
    writeln!(
        rendered,
        "def totalColumns : Nat := {}",
        json_nat(manifest, "full_builder_columns")
    )
    .expect("write string");
    let nifs_owner = &top_level[2];
    writeln!(
        rendered,
        "def nifsRowStart : Nat := {}",
        json_nat(nifs_owner, "row_start")
    )
    .expect("write string");
    writeln!(rendered, "def nifsRowEnd : Nat := {}", json_nat(nifs_owner, "row_end")).expect("write string");
    writeln!(
        rendered,
        "def nifsRowCount : Nat := {}",
        json_nat(nifs_owner, "row_count")
    )
    .expect("write string");
    writeln!(
        rendered,
        "def totalNonzeroEntries : Nat := {}",
        json_nat(recursive_total, "nonzero_entries")
    )
    .expect("write string");
    writeln!(
        rendered,
        "def totalSha256 : String := {}\n",
        lean_string(json_string(recursive_total, "sha256"))
    )
    .expect("write string");
    append_lean_ranges(&mut rendered, "topLevelFamilies", top_level);
    rendered.push('\n');
    append_lean_ranges(&mut rendered, "nifsFamilies", nifs);
    rendered.push('\n');
    writeln!(
        rendered,
        "def projectionShared : RowRange := {{ name := {}, rowStart := {}, rowEnd := {}, nonzeroEntries := {}, sha256 := {} }}",
        lean_string(json_string(projection_shared, "name")),
        json_nat(projection_shared, "row_start"),
        json_nat(projection_shared, "row_end"),
        json_nat(projection_shared, "nonzero_entries"),
        lean_string(json_string(projection_shared, "sha256")),
    )
    .expect("write string");
    writeln!(
        rendered,
        "def projectionIdentityCount : Nat := {}",
        json_nat(projection, "identity_count"),
    )
    .expect("write string");
    writeln!(
        rendered,
        "def projectionIdentityRows : Nat := {}",
        json_nat(projection, "identity_rows"),
    )
    .expect("write string");
    writeln!(
        rendered,
        "def projectionPairCounts : List Nat := [{}]",
        projection_pair_counts
            .iter()
            .map(|value| value.as_u64().expect("projection pair count").to_string())
            .collect::<Vec<_>>()
            .join(", "),
    )
    .expect("write string");
    append_lean_ranges(&mut rendered, "projectionIdentityRanges", projection_identities);
    rendered.push_str("\nend Nightstream.Implementation.R1CS.FPrimeRecursiveManifest\n");
    rendered
}

#[test]
fn recursive_program_manifest_matches_production_rows() {
    let manifest = build_manifest();
    let rendered = format!(
        "{}\n",
        serde_json::to_string_pretty(&manifest).expect("render manifest")
    );
    let path = repo_root().join(MANIFEST_PATH);
    let committed = fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("read {}: {error}\nexpected manifest:\n{rendered}", path.display()));
    if committed != rendered {
        let expected = path.with_extension("json.expected");
        fs::write(&expected, &rendered).expect("write expected recursive manifest");
    }
    assert_eq!(
        committed, rendered,
        "recursive program manifest drifted; reviewed output:\n{rendered}"
    );

    let lean_rendered = render_lean_data(&manifest);
    let lean_path = repo_root().join(LEAN_DATA_PATH);
    let lean_committed = fs::read_to_string(&lean_path).unwrap_or_else(|error| {
        panic!(
            "read {}: {error}\nexpected Lean data:\n{lean_rendered}",
            lean_path.display()
        )
    });
    if lean_committed != lean_rendered {
        let expected = lean_path.with_extension("lean.expected");
        fs::write(&expected, &lean_rendered).expect("write expected Lean recursive manifest");
    }
    assert_eq!(
        lean_committed, lean_rendered,
        "recursive Lean manifest data drifted; reviewed output:\n{lean_rendered}"
    );
}

#[test]
fn projection_identity_trace_exactly_replays_production_rows_and_rejects_corruption() {
    let builder = build_recursive_program();
    let source = builder.snapshot();
    let trace = builder.encoding_trace();
    let validated = validate_projection_identity_traces(&source, trace).expect("exact production projection trace");

    assert_eq!(validated.census.identities, 31);
    assert_eq!(validated.census.pairs, 31 * 15);
    assert_eq!(validated.census.polynomial_evaluations, 31 * 17);
    assert_eq!(validated.census.k_products, 31 * 16);
    assert_eq!(validated.census.source_rows, 59_396);
    assert_eq!(validated.census.source_columns, 59_334);
    assert_eq!(
        validated
            .roles
            .iter()
            .filter(|role| matches!(role, ProjectionIdentityRole::CommitmentLane { .. }))
            .count(),
        18
    );
    assert_eq!(
        validated
            .roles
            .iter()
            .filter(|role| matches!(role, ProjectionIdentityRole::ActiveXColumn { .. }))
            .count(),
        5
    );
    assert_eq!(
        validated
            .roles
            .iter()
            .filter(|role| matches!(role, ProjectionIdentityRole::YRingLimb { .. }))
            .count(),
        6
    );
    assert_eq!(
        validated
            .roles
            .iter()
            .filter(|role| matches!(role, ProjectionIdentityRole::YZColLimb { .. }))
            .count(),
        2
    );
    assert!(!validated.roles.iter().any(|role| matches!(
        role,
        ProjectionIdentityRole::Standalone | ProjectionIdentityRole::NebulaCommitmentLane { .. }
    )));

    let first = &trace.projection_identities()[0];
    let first_evaluation = first.input_evaluations.start;
    let corruptions = [
        {
            let mut corrupted = trace.clone();
            corrupted.apply_projection_identity_trace_test_mutation(
                0,
                ProjectionIdentityTraceTestMutation::SourceRowEnd {
                    row_end: first.source_rows.end - 1,
                },
            );
            corrupted
        },
        {
            let mut corrupted = trace.clone();
            corrupted.apply_projection_identity_trace_test_mutation(
                0,
                ProjectionIdentityTraceTestMutation::FinalLimbRowEnd {
                    row_end: first.final_limb_rows.end - 1,
                },
            );
            corrupted
        },
        {
            let mut corrupted = trace.clone();
            corrupted.apply_projection_identity_trace_test_mutation(
                0,
                ProjectionIdentityTraceTestMutation::InputColumn {
                    pair: 0,
                    coefficient: 0,
                    column: first.input_columns[0][0] + 1,
                },
            );
            corrupted
        },
        {
            let mut corrupted = trace.clone();
            corrupted.apply_polynomial_evaluation_trace_test_mutation(
                first_evaluation,
                PolynomialEvaluationTraceTestMutation::CoefficientColumn {
                    offset: 0,
                    column: trace.polynomial_evaluations()[first_evaluation].coefficient_cols[0] + 1,
                },
            );
            corrupted
        },
    ];
    for corrupted in corruptions {
        assert!(
            validate_projection_identity_traces(&source, &corrupted).is_err(),
            "corrupted provenance must fail closed"
        );
    }
}
