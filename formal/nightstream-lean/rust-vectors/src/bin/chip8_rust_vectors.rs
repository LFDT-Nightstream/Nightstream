#[path = "../render_support.rs"]
mod render_support;

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use neo_fold_prototype::chip8::kernel::{
    absorb_root0, build_kernel_exact_frames, build_kernel_stage3_digest_surfaces,
    build_kernel_external_release_artifact, build_kernel_staged_execution_digest_bundle, new_simple_kernel_transcript,
    prepared_step_digest, prove_simple_kernel, KernelBridgeBindingClaim, KernelBridgeBindingSummary, KernelCommitments,
    KernelExactFrame, KernelExactOpeningTranscriptEntry, KernelExternalReleaseArtifact,
    KernelFrameDecodeView, KernelJointOpeningTranscriptUnification, KernelMetaPub, KernelOpeningClaim,
    KernelOpeningSource, KernelOpeningTranscriptSource, KernelOpeningTranscriptSurface,
    KernelRoot0CommitmentBinding, KernelRowProjection, KernelRowProjectionSummary, KernelStage3CurrentRow,
    KernelStage3DigestSurface, KernelStage3LaneColumn, KernelStage3RowClaim, KernelStage3ShiftClaim,
    KernelStage3ShiftWitness, KernelStage3ShiftedColumn, KernelStepAux, KernelStagedExecutionDigestBundle,
    KernelTimeOpeningTranscriptGroup, KernelTimeOpeningTranscriptUnification, KernelTraceDigestSource,
    KernelTranscriptEvent, SimpleKernelProverInput, SimpleKernelPublicInput, SimpleKernelWitness,
    Stage1ShoutChannel, AddressFamily, TwistReadFamily, TwistMemoryFamily, KernelErrorTerm,
};
use neo_fold_prototype::chip8::stage1;
use neo_fold_prototype::chip8::stage2;
use neo_fold_prototype::chip8::stage3;
use neo_fold_prototype::chip8::spec::{build_pad_row, COL_I_NEXT, COL_I_REG, COL_REG_X, COL_REG_X_NEXT, WITNESS_WIDTH};
use neo_fold_prototype::chip8::tables::{
    build_alu_table, build_decode_table, build_eq4_table, build_rom_table, flatten_alu_key, flatten_eq4_key,
    LookupKind, ADDR_REG_BITS, RAM_SINK_ADDR, REG_SINK_ADDR,
};
use neo_fold_prototype::chip8::trace::Chip8TraceBuilder;
use neo_fold_prototype::chip8::{Chip8Program, Chip8State, CHIP8_PROGRAM_START};
use neo_math::{F, KExtensions, K};
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use render_support::{
    lean_ident_fragment, render_bool, render_bool_list, render_commitment_id, render_opcode_id, render_u8_list,
    render_u64_list,
};

#[derive(Clone, Copy)]
struct CursorSnapshot {
    state_words: [u64; 8],
    absorbed: usize,
}

#[derive(Clone)]
struct LoggedChallengeGroup {
    cursor: CursorSnapshot,
    outputs: Vec<u64>,
}

#[derive(Clone)]
struct LoggingTranscript {
    inner: Poseidon2Transcript,
    groups: BTreeMap<&'static str, LoggedChallengeGroup>,
}

#[derive(Clone)]
struct TranscriptVectorCase {
    name: &'static str,
    transcript_seed: Vec<u8>,
    commitment_bindings: Vec<(&'static str, Vec<u8>)>,
    meta_pub: KernelMetaPub,
    root0_transcript_cursor: CursorSnapshot,
    root0_digest_cursor: CursorSnapshot,
    root0_digest_words: [u64; 4],
    root0_digest_bytes: [u8; 32],
    stage1_lookup_point: Vec<(u64, u64)>,
    stage1_gamma_lookup_link_cursor: CursorSnapshot,
    stage1_gamma_lookup_link: (u64, u64),
    stage2_twist_cycle_cursor: CursorSnapshot,
    stage2_twist_cycle_point: Vec<(u64, u64)>,
    stage2_gamma_reg_cursor: CursorSnapshot,
    stage2_gamma_reg: (u64, u64),
    stage2_reg_addr_cursor: CursorSnapshot,
    stage2_reg_addr_point: Vec<(u64, u64)>,
    stage2_gamma_ram_cursor: CursorSnapshot,
    stage2_gamma_ram: (u64, u64),
    stage2_ram_addr_cursor: CursorSnapshot,
    stage2_ram_addr_point: Vec<(u64, u64)>,
    stage2_gamma_twist_link_cursor: CursorSnapshot,
    stage2_gamma_twist_link: (u64, u64),
    stage3_beta1_cursor: CursorSnapshot,
    stage3_beta1: (u64, u64),
    stage3_beta2_cursor: CursorSnapshot,
    stage3_beta2: (u64, u64),
    stage3_shift_cursor: CursorSnapshot,
    stage3_shift_point: Vec<(u64, u64)>,
    stage3_gamma_shift_cursor: CursorSnapshot,
    stage3_gamma_shift: (u64, u64),
}

#[derive(Clone)]
struct BundleVectorCase {
    name: &'static str,
    public: SimpleKernelPublicInput,
    meta_pub: KernelMetaPub,
    frames: Vec<KernelExactFrame>,
    stage3s: Vec<KernelStage3DigestSurface>,
    bundle: KernelStagedExecutionDigestBundle,
}

#[derive(Clone)]
struct ReleaseArtifactCase {
    name: &'static str,
    imported_artifact: KernelExternalReleaseArtifact,
}

#[derive(Clone)]
struct KernelFixture {
    name: &'static str,
    program: Chip8Program,
    initial_state: Chip8State,
    semantic_steps: usize,
    transcript_seed: Vec<u8>,
}

impl LoggingTranscript {
    fn from_inner(inner: Poseidon2Transcript) -> Self {
        Self {
            inner,
            groups: BTreeMap::new(),
        }
    }

    fn snapshot(&self) -> CursorSnapshot {
        CursorSnapshot {
            state_words: self.inner.state().map(|x| x.as_canonical_u64()),
            absorbed: self.inner.absorbed(),
        }
    }

    fn tracked_label_name(label: &[u8]) -> Option<&'static str> {
        match label {
            b"stage1/gamma_lookup_link" => Some("stage1/gamma_lookup_link"),
            b"stage2/r_cycle" => Some("stage2/r_cycle"),
            b"stage2/gamma_reg" => Some("stage2/gamma_reg"),
            b"stage2/r_addr_reg" => Some("stage2/r_addr_reg"),
            b"stage2/gamma_ram" => Some("stage2/gamma_ram"),
            b"stage2/r_addr_ram" => Some("stage2/r_addr_ram"),
            b"stage2/gamma_twist_link" => Some("stage2/gamma_twist_link"),
            b"stage3/beta1" => Some("stage3/beta1"),
            b"stage3/beta2" => Some("stage3/beta2"),
            b"stage3/r_shift" => Some("stage3/r_shift"),
            b"stage3/gamma_shift" => Some("stage3/gamma_shift"),
            _ => None,
        }
    }

    fn logged_group(&self, label: &'static str) -> &LoggedChallengeGroup {
        self.groups
            .get(label)
            .unwrap_or_else(|| panic!("missing logged challenge group for {label}"))
    }
}

impl Transcript for LoggingTranscript {
    fn new(app_label: &'static [u8]) -> Self {
        Self::from_inner(Poseidon2Transcript::new(app_label))
    }

    fn append_message(&mut self, label: &'static [u8], msg: &[u8]) {
        self.inner.append_message(label, msg);
    }

    fn append_fields(&mut self, label: &'static [u8], fs: &[F]) {
        self.inner.append_fields(label, fs);
    }

    fn challenge_bytes(&mut self, label: &'static [u8], out: &mut [u8]) {
        self.inner.challenge_bytes(label, out);
    }

    fn challenge_field(&mut self, label: &'static [u8]) -> F {
        if let Some(name) = Self::tracked_label_name(label) {
            let snapshot = self.snapshot();
            self.groups.entry(name).or_insert_with(|| LoggedChallengeGroup {
                cursor: snapshot,
                outputs: Vec::new(),
            });
            let out = self.inner.challenge_field(label);
            self.groups
                .get_mut(name)
                .expect("tracked group present")
                .outputs
                .push(out.as_canonical_u64());
            out
        } else {
            self.inner.challenge_field(label)
        }
    }

    fn fork(&self, scope: &'static [u8]) -> Self {
        Self {
            inner: self.inner.fork(scope),
            groups: self.groups.clone(),
        }
    }

    fn digest32(&mut self) -> [u8; 32] {
        self.inner.digest32()
    }
}

fn build_fixture_input(fixture: &KernelFixture) -> SimpleKernelProverInput {
    let execution =
        Chip8TraceBuilder::<()>::execute_program(&fixture.program, &fixture.initial_state, fixture.semantic_steps)
            .expect("fixture trace");

    let mut semantic_trace_rows = Vec::new();
    let mut semantic_aux_data = Vec::new();
    for step in execution {
        for row_trace in step.row_traces {
            semantic_trace_rows.push(row_trace.row);
            semantic_aux_data.push(row_trace.kernel_aux);
        }
    }

    SimpleKernelProverInput {
        public: SimpleKernelPublicInput {
            program_image: fixture.program.bytes.clone(),
            initial_pc_word: fixture.initial_state.pc / 2,
            initial_registers: fixture.initial_state.v,
            initial_i: fixture.initial_state.i,
            initial_ram: fixture.initial_state.memory.to_vec(),
            transcript_seed: fixture.transcript_seed.clone(),
        },
        witness: SimpleKernelWitness {
            semantic_trace_rows,
            semantic_aux_data,
        },
    }
}

fn make_fixture(
    name: &'static str,
    opcodes: &[u16],
    semantic_steps: usize,
    transcript_seed: &[u8],
    patch: impl FnOnce(&mut Chip8State),
) -> KernelFixture {
    let program = Chip8Program::from_opcodes(opcodes);
    let mut initial_state = Chip8State::with_program(&program).expect("fixture initial state");
    patch(&mut initial_state);
    KernelFixture {
        name,
        program,
        initial_state,
        semantic_steps,
        transcript_seed: transcript_seed.to_vec(),
    }
}

fn chip8_subset_fixtures() -> Vec<KernelFixture> {
    vec![
        make_fixture("jump_rows_2_seed_empty", &[0x1200], 2, b"", |_| {}),
        make_fixture(
            "jump_rows_3_seed_nonempty",
            &[0x1200],
            3,
            b"chip8-transcript-seed-v1",
            |_| {},
        ),
        make_fixture(
            "ldimm_chain_4",
            &[
                0x6001, // LD V0, 0x01
                0x6102, // LD V1, 0x02
                0x6203, // LD V2, 0x03
                0x6304, // LD V3, 0x04
            ],
            4,
            b"chip8-ldimm-chain-v1",
            |_| {},
        ),
        make_fixture(
            "addimm_chain_4",
            &[
                0x7001, // ADD V0, 0x01
                0x7102, // ADD V1, 0x02
                0x7203, // ADD V2, 0x03
                0x7304, // ADD V3, 0x04
            ],
            4,
            b"chip8-addimm-chain-v1",
            |state| {
                state.v[0] = 10;
                state.v[1] = 20;
                state.v[2] = 30;
                state.v[3] = 40;
            },
        ),
        make_fixture(
            "ldimm_addimm_mix_2",
            &[
                0x6001, // LD V0, 0x01
                0x7002, // ADD V0, 0x02
            ],
            2,
            b"chip8-ldimm-addimm-mix-v1",
            |_| {},
        ),
        make_fixture(
            "ldimm_skipeq_mix_2",
            &[
                0x6001, // LD V0, 0x01
                0x3001, // SE V0, 0x01
            ],
            2,
            b"chip8-ldimm-skipeq-mix-v1",
            |_| {},
        ),
        make_fixture(
            "mixed_ldimm_addimm_10",
            &[
                0x6001, // LD V0, 0x01
                0x6102, // LD V1, 0x02
                0x6203, // LD V2, 0x03
                0x6304, // LD V3, 0x04
                0x7001, // ADD V0, 0x01
                0x7102, // ADD V1, 0x02
                0x7203, // ADD V2, 0x03
                0x7304, // ADD V3, 0x04
                0x6405, // LD V4, 0x05
                0x7406, // ADD V4, 0x06
            ],
            10,
            b"chip8-mixed-ldimm-addimm-10-v1",
            |_| {},
        ),
    ]
}

fn commitment_bindings(commitments: &KernelCommitments) -> Vec<(&'static str, Vec<u8>)> {
    vec![
        ("lane", commitments.c_lane.to_vec()),
        ("fetchRa", commitments.c_fetch_ra.to_vec()),
        ("decodeRa", commitments.c_decode_ra.to_vec()),
        ("aluRa", commitments.c_alu_ra.to_vec()),
        ("eq4Ra", commitments.c_eq4_ra.to_vec()),
        ("decodeHandoff", commitments.c_decode_handoff.to_vec()),
        ("regTwist", commitments.c_reg.to_vec()),
        ("ramTwist", commitments.c_ram.to_vec()),
        ("romTable", commitments.c_rom_table.to_vec()),
        ("decodeTable", commitments.c_decode_table.to_vec()),
        ("aluTable", commitments.c_alu_table.to_vec()),
        ("eq4Table", commitments.c_eq4_table.to_vec()),
    ]
}

fn digest_words(bytes: [u8; 32]) -> [u64; 4] {
    let mut out = [0u64; 4];
    for (idx, chunk) in bytes.chunks_exact(8).enumerate() {
        let mut word = [0u8; 8];
        word.copy_from_slice(chunk);
        out[idx] = u64::from_le_bytes(word);
    }
    out
}

fn k_pair(value: K) -> (u64, u64) {
    let [re, im] = value.as_coeffs();
    (re.as_canonical_u64(), im.as_canonical_u64())
}

fn cursor_snapshot(transcript: &Poseidon2Transcript) -> CursorSnapshot {
    CursorSnapshot {
        state_words: transcript.state().map(|x| x.as_canonical_u64()),
        absorbed: transcript.absorbed(),
    }
}

fn logged_pair(log: &LoggingTranscript, label: &'static str) -> (CursorSnapshot, (u64, u64)) {
    let group = log.logged_group(label);
    assert_eq!(group.outputs.len(), 2, "{label} must emit exactly two base-field outputs");
    (
        group.cursor,
        (group.outputs[0], group.outputs[1]),
    )
}

fn logged_point(log: &LoggingTranscript, label: &'static str) -> (CursorSnapshot, Vec<(u64, u64)>) {
    let group = log.logged_group(label);
    assert_eq!(
        group.outputs.len() % 2,
        0,
        "{label} must emit an even number of base-field outputs"
    );
    let point = group
        .outputs
        .chunks_exact(2)
        .map(|chunk| (chunk[0], chunk[1]))
        .collect();
    (group.cursor, point)
}

fn build_pad_aux(pad_pc_word: u16) -> KernelStepAux {
    KernelStepAux {
        fetch_addr: pad_pc_word as usize,
        decode_addr: 0x1000 | (2 * pad_pc_word),
        alu_key: flatten_alu_key(LookupKind::NoLookup, 0, 0),
        eq4_key: flatten_eq4_key(0, 0),
        reg_ra_x_addr: 0,
        reg_ra_y_addr: REG_SINK_ADDR,
        reg_ra_i_addr: 16,
        reg_wa_addr: REG_SINK_ADDR,
        ram_ra_addr: RAM_SINK_ADDR,
        ram_wa_addr: RAM_SINK_ADDR,
        reg_inc: F::ZERO,
        ram_inc: F::ZERO,
        uses_y: false,
        reads_ram: false,
        writes_ram: false,
    }
}

fn final_register_domain(
    initial_registers: &[u8; 16],
    initial_i: u16,
    aux: &[KernelStepAux],
) -> Vec<F> {
    let mut state = vec![F::ZERO; 1usize << ADDR_REG_BITS];
    for (idx, &value) in initial_registers.iter().enumerate() {
        state[idx] = F::from_u64(value as u64);
    }
    state[16] = F::from_u64(initial_i as u64);
    for step in aux {
        assert!(
            step.reg_wa_addr < state.len(),
            "reg_wa_addr {} escapes register domain {}",
            step.reg_wa_addr,
            state.len()
        );
        state[step.reg_wa_addr] += step.reg_inc;
    }
    state
}

fn build_kernel_pad_row(pad_pc_word: u16, final_reg_state: &[F]) -> [F; WITNESS_WIDTH] {
    let mut row = build_pad_row(pad_pc_word);
    row[COL_REG_X] = final_reg_state[0];
    row[COL_REG_X_NEXT] = final_reg_state[0];
    row[COL_I_REG] = final_reg_state[16];
    row[COL_I_NEXT] = final_reg_state[16];
    row
}

fn pad_semantic_witness(
    semantic_trace_rows: &[[F; WITNESS_WIDTH]],
    semantic_aux_data: &[KernelStepAux],
    initial_registers: &[u8; 16],
    initial_i: u16,
    pad_pc_word: u16,
    padded_len: usize,
) -> (Vec<[F; WITNESS_WIDTH]>, Vec<KernelStepAux>) {
    assert_eq!(semantic_trace_rows.len(), semantic_aux_data.len(), "trace/aux mismatch");
    assert!(!semantic_trace_rows.is_empty(), "semantic trace must be non-empty");
    assert!(padded_len.is_power_of_two(), "padded length must be a power of two");
    assert!(
        semantic_trace_rows.len() <= padded_len,
        "semantic trace length {} exceeds padded length {}",
        semantic_trace_rows.len(),
        padded_len
    );

    let final_reg_state = final_register_domain(initial_registers, initial_i, semantic_aux_data);
    let pad_row = build_kernel_pad_row(pad_pc_word, &final_reg_state);
    let pad_aux = build_pad_aux(pad_pc_word);
    let mut trace_rows = semantic_trace_rows.to_vec();
    let mut aux_data = semantic_aux_data.to_vec();
    while trace_rows.len() < padded_len {
        trace_rows.push(pad_row);
        aux_data.push(pad_aux.clone());
    }
    (trace_rows, aux_data)
}

fn build_case(fixture: &KernelFixture) -> TranscriptVectorCase {
    let input = build_fixture_input(fixture);
    let (_output, proof) = prove_simple_kernel(&input).expect("simple kernel proof");

    let mut root0_transcript = new_simple_kernel_transcript(&input.public.transcript_seed);
    absorb_root0(&mut root0_transcript, &proof.commitments, &proof.meta_pub);
    let root0_transcript_cursor = cursor_snapshot(&root0_transcript);
    let mut root0_digest_transcript = root0_transcript.clone();
    let root0_digest_bytes = root0_digest_transcript.digest32();
    let root0_digest_words = digest_words(root0_digest_bytes);
    let root0_digest_cursor = cursor_snapshot(&root0_digest_transcript);

    let program = Chip8Program {
        bytes: input.public.program_image.clone(),
        start_pc: CHIP8_PROGRAM_START,
    };
    let pad_pc_word = proof.meta_pub.pad_pc_word;
    let (trace_rows, aux_data) = pad_semantic_witness(
        &input.witness.semantic_trace_rows,
        &input.witness.semantic_aux_data,
        &input.public.initial_registers,
        input.public.initial_i,
        pad_pc_word,
        proof.meta_pub.padded_trace_length,
    );
    let rom_table = build_rom_table(&program, pad_pc_word);
    let decode_table = build_decode_table();
    let alu_table = build_alu_table();
    let eq4_table = build_eq4_table();

    let mut stage_inner = new_simple_kernel_transcript(&input.public.transcript_seed);
    absorb_root0(&mut stage_inner, &proof.commitments, &proof.meta_pub);
    let mut transcript = LoggingTranscript::from_inner(stage_inner);
    let stage1_proof = stage1::prove_stage1(
        &trace_rows,
        &aux_data,
        &rom_table,
        &decode_table,
        &alu_table,
        &eq4_table,
        proof.meta_pub.cycle_bits,
        &mut transcript,
    )
    .expect("stage1 proof");
    let _stage2_proof = stage2::prove_stage2(
        &trace_rows,
        &aux_data,
        &input.public.initial_registers,
        input.public.initial_i,
        &input.public.initial_ram,
        proof.meta_pub.cycle_bits,
        &mut transcript,
    )
    .expect("stage2 proof");
    let _stage3_proof = stage3::prove_stage3(
        &trace_rows,
        input.witness.semantic_trace_rows.len(),
        proof.meta_pub.cycle_bits,
        &mut transcript,
    )
    .expect("stage3 proof");

    let (stage1_gamma_lookup_link_cursor, stage1_gamma_lookup_link) =
        logged_pair(&transcript, "stage1/gamma_lookup_link");
    let (stage2_twist_cycle_cursor, stage2_twist_cycle_point) =
        logged_point(&transcript, "stage2/r_cycle");
    let (stage2_gamma_reg_cursor, stage2_gamma_reg) =
        logged_pair(&transcript, "stage2/gamma_reg");
    let (stage2_reg_addr_cursor, stage2_reg_addr_point) =
        logged_point(&transcript, "stage2/r_addr_reg");
    let (stage2_gamma_ram_cursor, stage2_gamma_ram) =
        logged_pair(&transcript, "stage2/gamma_ram");
    let (stage2_ram_addr_cursor, stage2_ram_addr_point) =
        logged_point(&transcript, "stage2/r_addr_ram");
    let (stage2_gamma_twist_link_cursor, stage2_gamma_twist_link) =
        logged_pair(&transcript, "stage2/gamma_twist_link");
    let (stage3_beta1_cursor, stage3_beta1) =
        logged_pair(&transcript, "stage3/beta1");
    let (stage3_beta2_cursor, stage3_beta2) =
        logged_pair(&transcript, "stage3/beta2");
    let (stage3_shift_cursor, stage3_shift_point) =
        logged_point(&transcript, "stage3/r_shift");
    let (stage3_gamma_shift_cursor, stage3_gamma_shift) =
        logged_pair(&transcript, "stage3/gamma_shift");

    TranscriptVectorCase {
        name: fixture.name,
        transcript_seed: fixture.transcript_seed.clone(),
        commitment_bindings: commitment_bindings(&proof.commitments),
        meta_pub: proof.meta_pub,
        root0_transcript_cursor,
        root0_digest_cursor,
        root0_digest_words,
        root0_digest_bytes,
        stage1_lookup_point: stage1_proof.cycle_point.iter().copied().map(k_pair).collect(),
        stage1_gamma_lookup_link_cursor,
        stage1_gamma_lookup_link,
        stage2_twist_cycle_cursor,
        stage2_twist_cycle_point,
        stage2_gamma_reg_cursor,
        stage2_gamma_reg,
        stage2_reg_addr_cursor,
        stage2_reg_addr_point,
        stage2_gamma_ram_cursor,
        stage2_gamma_ram,
        stage2_ram_addr_cursor,
        stage2_ram_addr_point,
        stage2_gamma_twist_link_cursor,
        stage2_gamma_twist_link,
        stage3_beta1_cursor,
        stage3_beta1,
        stage3_beta2_cursor,
        stage3_beta2,
        stage3_shift_cursor,
        stage3_shift_point,
        stage3_gamma_shift_cursor,
        stage3_gamma_shift,
    }
}

fn build_bundle_case(fixture: &KernelFixture) -> BundleVectorCase {
    let input = build_fixture_input(fixture);
    let (output, proof) = prove_simple_kernel(&input).expect("simple kernel proof");
    let frames = build_kernel_exact_frames(&input.public, &proof).expect("kernel exact frames");
    let stage3s =
        build_kernel_stage3_digest_surfaces(&input.public, &proof, &output).expect("kernel stage3 digest surfaces");
    let bundle = build_kernel_staged_execution_digest_bundle(&input.public, &proof, &output)
        .expect("kernel staged execution digest bundle");
    BundleVectorCase {
        name: fixture.name,
        public: input.public,
        meta_pub: proof.meta_pub,
        frames,
        stage3s,
        bundle,
    }
}

fn build_release_artifact_case(fixture: &KernelFixture) -> ReleaseArtifactCase {
    let input = build_fixture_input(fixture);
    let (output, proof) = prove_simple_kernel(&input).expect("simple kernel proof");
    let imported_artifact = build_kernel_external_release_artifact(&input.public, &proof, &output)
        .expect("kernel external release artifact");
    ReleaseArtifactCase {
        name: fixture.name,
        imported_artifact,
    }
}

fn render_meta_pub(meta: &KernelMetaPub) -> String {
    format!(
        "mkMetaPub\n    {program_image_digest}\n    {initial_state_digest}\n    {rom_table_digest}\n    {decode_table_digest}\n    {alu_table_digest}\n    {eq4_table_digest}\n    {transcript_seed_digest}\n    {protocol_version_id}\n    {field_id}\n    {extension_field_id}\n    {root_params_id}\n    {variable_order_id}\n    {domain_shape_id}\n    {sink_convention_id}\n    {init_mode_id}\n    {lowering_convention_id}\n    {padding_convention_id}\n    {table_auth_mode_id}\n    {opening_reduction_mode_id}\n    {program_word_count}\n    {semantic_rows}\n    {padded_trace_length}\n    {pad_pc_word}\n    {program_base_addr}\n    {cycle_bits}",
        program_image_digest = render_u8_list(&meta.program_image_digest),
        initial_state_digest = render_u8_list(&meta.initial_state_digest),
        rom_table_digest = render_u8_list(&meta.rom_table_digest),
        decode_table_digest = render_u8_list(&meta.decode_table_digest),
        alu_table_digest = render_u8_list(&meta.alu_table_digest),
        eq4_table_digest = render_u8_list(&meta.eq4_table_digest),
        transcript_seed_digest = render_u8_list(&meta.transcript_seed_digest),
        protocol_version_id = meta.protocol_version_id,
        field_id = meta.field_id,
        extension_field_id = meta.extension_field_id,
        root_params_id = render_u8_list(&meta.root_params_id),
        variable_order_id = meta.variable_order_id,
        domain_shape_id = meta.domain_shape_id,
        sink_convention_id = meta.sink_convention_id,
        init_mode_id = meta.init_mode_id,
        lowering_convention_id = meta.lowering_convention_id,
        padding_convention_id = meta.padding_convention_id,
        table_auth_mode_id = meta.table_auth_mode_id,
        opening_reduction_mode_id = meta.opening_reduction_mode_id,
        program_word_count = meta.program_word_count,
        semantic_rows = meta.semantic_rows,
        padded_trace_length = meta.padded_trace_length,
        pad_pc_word = meta.pad_pc_word,
        program_base_addr = meta.program_base_addr,
        cycle_bits = meta.cycle_bits,
    )
}

fn render_binding(id: &str, digest: &[u8]) -> String {
    format!("binding .{id} ({})", render_u8_list(digest))
}

fn render_pair(value: (u64, u64)) -> String {
    format!("pair {} {}", value.0, value.1)
}

fn render_point(values: &[(u64, u64)]) -> String {
    let mut out = String::from("[");
    for (idx, value) in values.iter().enumerate() {
        if idx > 0 {
            out.push_str(", ");
        }
        out.push_str(&render_pair(*value));
    }
    out.push(']');
    out
}

fn render_cursor_snapshot(snapshot: CursorSnapshot) -> String {
    format!(
        "cursorSnapshot {} {}",
        render_u64_list(&snapshot.state_words),
        snapshot.absorbed
    )
}

fn meta_pub_def_name(case: &TranscriptVectorCase) -> String {
    transcript_meta_pub_def_name_from_name(case.name)
}

fn render_meta_pub_def(case: &TranscriptVectorCase) -> String {
    render_named_meta_pub_def(&transcript_meta_pub_def_name_from_name(case.name), &case.meta_pub)
}

fn render_named_meta_pub_def(def_name: &str, meta_pub: &KernelMetaPub) -> String {
    format!(
        "def {name} : MetaPub :=\n  {value}\n",
        name = def_name,
        value = render_meta_pub(meta_pub),
    )
}

fn transcript_meta_pub_def_name_from_name(name: &str) -> String {
    format!("transcriptMetaPub_{}", lean_ident_fragment(name))
}

fn bundle_meta_pub_def_name_from_name(name: &str) -> String {
    format!("bundleMetaPub_{}", lean_ident_fragment(name))
}

fn release_artifact_meta_pub_def_name_from_name(name: &str) -> String {
    format!("releaseArtifactMetaPub_{}", lean_ident_fragment(name))
}

fn imported_release_artifact_meta_pub_def_name_from_name(name: &str) -> String {
    format!("importedReleaseArtifactMetaPub_{}", lean_ident_fragment(name))
}

fn bundle_case_def_name_from_name(name: &str) -> String {
    format!("bundleCase_{}", lean_ident_fragment(name))
}

fn release_artifact_case_def_name_from_name(name: &str) -> String {
    format!("releaseArtifactCase_{}", lean_ident_fragment(name))
}

fn render_case(case: &TranscriptVectorCase) -> String {
    let mut bindings = String::from("[");
    for (idx, (id, digest)) in case.commitment_bindings.iter().enumerate() {
        if idx > 0 {
            bindings.push_str(", ");
        }
        bindings.push_str(&render_binding(id, digest));
    }
    bindings.push(']');

    format!(
        "mkTranscriptVectorCase\n      \"{name}\"\n      {transcript_seed}\n      {commitment_bindings}\n      {meta_pub_name}\n      ({root0_transcript_cursor})\n      ({root0_digest_cursor})\n      ({root0_digest_words})\n      {root0_digest_bytes}\n      ({stage1_lookup_point})\n      ({stage1_gamma_lookup_link_cursor})\n      ({stage1_gamma_lookup_link})\n      ({stage2_twist_cycle_cursor})\n      ({stage2_twist_cycle_point})\n      ({stage2_gamma_reg_cursor})\n      ({stage2_gamma_reg})\n      ({stage2_reg_addr_cursor})\n      ({stage2_reg_addr_point})\n      ({stage2_gamma_ram_cursor})\n      ({stage2_gamma_ram})\n      ({stage2_ram_addr_cursor})\n      ({stage2_ram_addr_point})\n      ({stage2_gamma_twist_link_cursor})\n      ({stage2_gamma_twist_link})\n      ({stage3_beta1_cursor})\n      ({stage3_beta1})\n      ({stage3_beta2_cursor})\n      ({stage3_beta2})\n      ({stage3_shift_cursor})\n      ({stage3_shift_point})\n      ({stage3_gamma_shift_cursor})\n      ({stage3_gamma_shift})",
        name = case.name,
        transcript_seed = render_u8_list(&case.transcript_seed),
        commitment_bindings = bindings,
        meta_pub_name = meta_pub_def_name(case),
        root0_transcript_cursor = render_cursor_snapshot(case.root0_transcript_cursor),
        root0_digest_cursor = render_cursor_snapshot(case.root0_digest_cursor),
        root0_digest_words = render_u64_list(&case.root0_digest_words),
        root0_digest_bytes = render_u8_list(&case.root0_digest_bytes),
        stage1_lookup_point = render_point(&case.stage1_lookup_point),
        stage1_gamma_lookup_link_cursor = render_cursor_snapshot(case.stage1_gamma_lookup_link_cursor),
        stage1_gamma_lookup_link = render_pair(case.stage1_gamma_lookup_link),
        stage2_twist_cycle_cursor = render_cursor_snapshot(case.stage2_twist_cycle_cursor),
        stage2_twist_cycle_point = render_point(&case.stage2_twist_cycle_point),
        stage2_gamma_reg_cursor = render_cursor_snapshot(case.stage2_gamma_reg_cursor),
        stage2_gamma_reg = render_pair(case.stage2_gamma_reg),
        stage2_reg_addr_cursor = render_cursor_snapshot(case.stage2_reg_addr_cursor),
        stage2_reg_addr_point = render_point(&case.stage2_reg_addr_point),
        stage2_gamma_ram_cursor = render_cursor_snapshot(case.stage2_gamma_ram_cursor),
        stage2_gamma_ram = render_pair(case.stage2_gamma_ram),
        stage2_ram_addr_cursor = render_cursor_snapshot(case.stage2_ram_addr_cursor),
        stage2_ram_addr_point = render_point(&case.stage2_ram_addr_point),
        stage2_gamma_twist_link_cursor = render_cursor_snapshot(case.stage2_gamma_twist_link_cursor),
        stage2_gamma_twist_link = render_pair(case.stage2_gamma_twist_link),
        stage3_beta1_cursor = render_cursor_snapshot(case.stage3_beta1_cursor),
        stage3_beta1 = render_pair(case.stage3_beta1),
        stage3_beta2_cursor = render_cursor_snapshot(case.stage3_beta2_cursor),
        stage3_beta2 = render_pair(case.stage3_beta2),
        stage3_shift_cursor = render_cursor_snapshot(case.stage3_shift_cursor),
        stage3_shift_point = render_point(&case.stage3_shift_point),
        stage3_gamma_shift_cursor = render_cursor_snapshot(case.stage3_gamma_shift_cursor),
        stage3_gamma_shift = render_pair(case.stage3_gamma_shift),
    )
}

fn render_lane_column(column: KernelStage3LaneColumn) -> &'static str {
    match column {
        KernelStage3LaneColumn::Pc => ".pc",
        KernelStage3LaneColumn::XIdx => ".xIdx",
        KernelStage3LaneColumn::IsMemOp => ".isMemOp",
    }
}

fn render_shifted_column(column: KernelStage3ShiftedColumn) -> &'static str {
    match column {
        KernelStage3ShiftedColumn::ShiftPc => ".shiftPc",
        KernelStage3ShiftedColumn::ShiftXIdx => ".shiftXIdx",
        KernelStage3ShiftedColumn::ShiftIsMemOp => ".shiftIsMemOp",
    }
}

fn render_lane_columns(values: &[KernelStage3LaneColumn]) -> String {
    let mut out = String::from("[");
    for (idx, value) in values.iter().enumerate() {
        if idx > 0 {
            out.push_str(", ");
        }
        out.push_str(render_lane_column(*value));
    }
    out.push(']');
    out
}

fn render_shifted_columns(values: &[KernelStage3ShiftedColumn]) -> String {
    let mut out = String::from("[");
    for (idx, value) in values.iter().enumerate() {
        if idx > 0 {
            out.push_str(", ");
        }
        out.push_str(render_shifted_column(*value));
    }
    out.push(']');
    out
}

fn render_f_value(value: F) -> u64 {
    value.as_canonical_u64()
}

fn render_f_row(values: &[F]) -> String {
    let words: Vec<u64> = values.iter().map(|value| render_f_value(*value)).collect();
    render_u64_list(&words)
}

fn render_k_list(values: &[K]) -> String {
    let pairs: Vec<(u64, u64)> = values.iter().copied().map(k_pair).collect();
    render_point(&pairs)
}

fn render_k_value(value: K) -> String {
    format!("({})", render_pair(k_pair(value)))
}

fn render_k_rounds(rounds: &[Vec<K>]) -> String {
    let mut out = String::from("[");
    for (idx, round) in rounds.iter().enumerate() {
        if idx > 0 {
            out.push_str(", ");
        }
        out.push_str(&render_k_list(round));
    }
    out.push(']');
    out
}

fn render_machine_state_view(state: &Chip8State) -> String {
    render_machine_state_view_from_parts(state.pc, state.i, &state.v, &render_u8_list(&state.memory))
}

fn render_machine_state_view_from_parts(
    pc: u16,
    i: u16,
    registers: &[u8],
    ram_expr: &str,
) -> String {
    format!(
        "mkMachineStateView {} {} {} {}",
        pc,
        i,
        render_u8_list(registers),
        ram_expr,
    )
}

fn render_frame_decode_view(dec: &KernelFrameDecodeView) -> String {
    format!(
        "mkFrameDecodeView {} {} {} {} {} {} {} {} {} {} {} {}",
        render_opcode_id(dec.core.opcode_id),
        dec.core.x,
        dec.core.y,
        dec.core.kk,
        dec.core.nnn,
        dec.opcode_word,
        dec.pc_word,
        dec.row_x_idx,
        dec.row_y_idx,
        render_bool(dec.is_memop),
        render_bool(dec.burst_last),
        dec.ram_addr,
    )
}

fn render_frame_source_view(frame: &KernelExactFrame) -> String {
    render_frame_source_view_from_parts(
        frame.step_idx,
        &frame.dec,
        &render_machine_state_view(&frame.pre),
        &render_machine_state_view(&frame.post),
        &render_f_row(&frame.row),
    )
}

fn render_frame_source_view_from_parts(
    step_idx: usize,
    dec: &KernelFrameDecodeView,
    pre_expr: &str,
    post_expr: &str,
    row_expr: &str,
) -> String {
    format!(
        "mkFrameSourceView {} ({}) ({}) ({}) ({})",
        step_idx,
        render_frame_decode_view(dec),
        pre_expr,
        post_expr,
        row_expr,
    )
}

fn render_public_input_view(public: &SimpleKernelPublicInput) -> String {
    render_public_input_view_from_parts(
        &public.program_image,
        public.initial_pc_word,
        &public.initial_registers,
        public.initial_i,
        &render_u8_list(&public.initial_ram),
        &public.transcript_seed,
    )
}

fn render_public_input_view_from_parts(
    program_image: &[u8],
    initial_pc_word: u16,
    initial_registers: &[u8],
    initial_i: u16,
    initial_ram_expr: &str,
    transcript_seed: &[u8],
) -> String {
    format!(
        "mkPublicInputView\n      {}\n      {}\n      {}\n      {}\n      {}\n      {}",
        render_u8_list(program_image),
        initial_pc_word,
        render_u8_list(initial_registers),
        initial_i,
        initial_ram_expr,
        render_u8_list(transcript_seed),
    )
}

fn render_digest_public_view_with_meta_expr(public: &SimpleKernelPublicInput, meta_pub_expr: &str) -> String {
    format!(
        "mkDigestPublicView\n      ({})\n      {}",
        render_public_input_view(public),
        meta_pub_expr,
    )
}

fn render_stage3_shift_claim_view(claim: &KernelStage3ShiftClaim) -> String {
    let claimed_shift_values: Vec<(u64, u64)> =
        claim.claimed_shift_values.iter().copied().map(k_pair).collect();
    format!(
        "mkStage3ShiftClaimView\n        {}\n        ({})\n        ({})\n        ({})\n        ({})",
        render_commitment_id(claim.source_commitment),
        render_k_list(&claim.source_point),
        render_lane_columns(&claim.source_columns),
        render_shifted_columns(&claim.shifted_columns),
        render_point(&claimed_shift_values),
    )
}

fn render_stage3_shift_witness_view(witness: &KernelStage3ShiftWitness) -> String {
    format!(
        "mkStage3ShiftWitnessView\n        ({})\n        ({})\n        ({})\n        ({})",
        render_pair(k_pair(witness.shift_pc)),
        render_pair(k_pair(witness.shift_x_idx)),
        render_pair(k_pair(witness.shift_is_memop)),
        render_k_rounds(&witness.reduction_rounds),
    )
}

fn render_stage3_current_row_view(row: &KernelStage3CurrentRow) -> String {
    format!(
        "mkStage3CurrentRowView {} {} {} {} {} {}",
        row.row_index,
        render_f_value(row.pair_mask),
        render_f_value(row.pc_next),
        render_f_value(row.x_idx),
        render_f_value(row.is_memop),
        render_f_value(row.burst_last),
    )
}

fn render_stage3_row_claim_view(claim: &KernelStage3RowClaim) -> String {
    format!(
        "mkStage3RowClaimView {} ({}) ({})",
        claim.row_index,
        render_bool_list(&claim.row_bits),
        render_k_list(&claim.opened_values),
    )
}

fn render_stage3_view(stage3: &KernelStage3DigestSurface) -> String {
    format!(
        "mkStage3View\n        {}\n        {}\n        ({})\n        ({})\n        ({})\n        ({})\n        ({})\n        ({})\n        {}",
        stage3.step_idx,
        stage3.n,
        render_pair(k_pair(stage3.beta1)),
        render_pair(k_pair(stage3.beta2)),
        render_stage3_shift_claim_view(&stage3.shift_claim),
        render_stage3_shift_witness_view(&stage3.shift_proof),
        render_stage3_current_row_view(&stage3.current_row),
        render_stage3_row_claim_view(&stage3.row_claim),
        render_u8_list(&prepared_step_digest(&stage3.prepared_step)),
    )
}

fn render_staged_execution_digest_view_from_parts(
    stage1_expr: &str,
    stage2_expr: &str,
    stage3_expr: &str,
    result_expr: &str,
) -> String {
    format!(
        "mkStagedExecutionDigestView\n      ({})\n      ({})\n      ({})\n      ({})",
        stage1_expr,
        stage2_expr,
        stage3_expr,
        result_expr,
    )
}

fn render_bundle_view_from_parts(public_expr: &str, digests_expr: &str) -> String {
    format!(
        "mkStagedExecutionDigestBundleView\n      ({})\n      {}",
        public_expr,
        digests_expr,
    )
}

fn render_trace_digest_source_view(source: &KernelTraceDigestSource) -> String {
    format!(
        "mkTraceDigestSourceView\n      {}\n      {}\n      {}\n      {}",
        render_u8_list(&source.stage1_digest),
        render_u8_list(&source.stage2_digest),
        render_u8_list(&source.stage3_digest),
        render_u8_list(&source.semantic_evidence_summary_digest),
    )
}

fn render_root0_binding_view(binding: &KernelRoot0CommitmentBinding) -> String {
    format!(
        "binding {} ({})",
        render_commitment_id(binding.id),
        render_u8_list(&binding.digest),
    )
}

fn render_usize_list(values: &[usize]) -> String {
    let words: Vec<u64> = values.iter().map(|&value| value as u64).collect();
    render_u64_list(&words)
}

fn render_opening_source(source: KernelOpeningSource) -> &'static str {
    match source {
        KernelOpeningSource::Kernel => ".kernel",
        KernelOpeningSource::Root => ".root",
    }
}

fn render_opening_claim_view(claim: &KernelOpeningClaim) -> String {
    format!(
        "mkKernelOpeningClaimView\n        {}\n        {}\n        ({})\n        ({})\n        ({})\n        {}",
        render_opening_source(claim.source),
        render_commitment_id(claim.commitment_id),
        render_k_list(&claim.point),
        render_usize_list(&claim.polynomial_ids),
        render_k_list(&claim.claimed_values),
        render_u8_list(&claim.digest),
    )
}

fn render_opening_manifest_view_from_claims_expr(claims_expr: &str, digest: &[u8; 32]) -> String {
    format!(
        "mkKernelOpeningManifestView\n        {}\n        {}",
        claims_expr,
        render_u8_list(digest),
    )
}

fn render_row_projection_view(projection: &KernelRowProjection) -> String {
    format!(
        "mkKernelRowProjectionView\n          {}\n          {}\n          {}\n          {}\n          {}\n          {}",
        projection.row_index,
        render_u8_list(&projection.row_binding_claim_digest),
        render_u8_list(&projection.row_binding_refinement_digest),
        render_u8_list(&projection.semantic_row_digest),
        render_u8_list(&projection.semantic_view_digest),
        render_u8_list(&projection.digest),
    )
}

fn render_row_projection_summary_view(summary: &KernelRowProjectionSummary) -> String {
    let mut projections = String::from("[");
    for (idx, projection) in summary.projections.iter().enumerate() {
        if idx > 0 {
            projections.push_str(", ");
        }
        projections.push_str(&format!("({})", render_row_projection_view(projection)));
    }
    projections.push(']');
    format!(
        "mkKernelRowProjectionSummaryView\n        {}\n        {}",
        projections,
        render_u8_list(&summary.digest),
    )
}

fn render_bridge_binding_claim_view(claim: &KernelBridgeBindingClaim) -> String {
    format!(
        "mkKernelBridgeBindingClaimView\n          {}\n          {}\n          {}\n          {}\n          {}",
        claim.row_index,
        render_u8_list(&claim.row_binding_claim_digest),
        render_u8_list(&claim.row_binding_refinement_digest),
        render_u8_list(&claim.prepared_step_digest),
        render_u8_list(&claim.digest),
    )
}

fn render_bridge_binding_summary_view(summary: &KernelBridgeBindingSummary) -> String {
    let mut claims = String::from("[");
    for (idx, claim) in summary.claims.iter().enumerate() {
        if idx > 0 {
            claims.push_str(", ");
        }
        claims.push_str(&format!("({})", render_bridge_binding_claim_view(claim)));
    }
    claims.push(']');
    format!(
        "mkKernelBridgeBindingSummaryView\n        {}\n        {}",
        claims,
        render_u8_list(&summary.digest),
    )
}

fn render_kernel_trace_surface_view_from_frames_expr(
    trace: &neo_fold_prototype::chip8::kernel::KernelTraceSurface,
    frames_expr: &str,
) -> String {
    format!(
        "mkKernelTraceSurfaceView\n      {}\n      {}\n      {}\n      {}\n      {}",
        frames_expr,
        render_u8_list(&trace.stage1_digest),
        render_u8_list(&trace.stage2_digest),
        render_u8_list(&trace.stage3_digest),
        render_u8_list(&trace.semantic_evidence_summary_digest),
    )
}

fn render_kernel_export_surface_view(
    export: &neo_fold_prototype::chip8::kernel::KernelExportSurface,
) -> String {
    let prepared_step_digests: Vec<[u8; 32]> =
        export.prepared_steps.iter().map(prepared_step_digest).collect();
    let mut digests = String::from("[");
    for (idx, digest) in prepared_step_digests.iter().enumerate() {
        if idx > 0 {
            digests.push_str(", ");
        }
        digests.push_str(&render_u8_list(digest));
    }
    digests.push(']');
    format!(
        "mkKernelExportSurfaceView\n      {}\n      {}",
        export.semantic_rows,
        digests,
    )
}

fn render_kernel_audit_surface_view(
    audit: &neo_fold_prototype::chip8::kernel::KernelAuditSurface,
) -> String {
    format!(
        "mkKernelAuditSurfaceView\n      ({})\n      ({})",
        render_row_projection_summary_view(&audit.row_projection_summary),
        render_bridge_binding_summary_view(&audit.bridge_binding_summary),
    )
}

fn render_kernel_transcript_event(event: &KernelTranscriptEvent) -> String {
    match event {
        KernelTranscriptEvent::AbsorbCommitment(id) => format!("(.absorbCommitment {})", render_commitment_id(*id)),
        KernelTranscriptEvent::AbsorbMetaPub => ".absorbMetaPub".into(),
        KernelTranscriptEvent::SampleStage1Cycle => ".sampleStage1Cycle".into(),
        KernelTranscriptEvent::Stage1FetchSumcheck => ".stage1FetchSumcheck".into(),
        KernelTranscriptEvent::Stage1DecodeSumcheck => ".stage1DecodeSumcheck".into(),
        KernelTranscriptEvent::Stage1AluSumcheck => ".stage1AluSumcheck".into(),
        KernelTranscriptEvent::Stage1Eq4Sumcheck => ".stage1Eq4Sumcheck".into(),
        KernelTranscriptEvent::Stage1AddrCheckFetch => ".stage1AddrCheckFetch".into(),
        KernelTranscriptEvent::Stage1AddrCheckDecode => ".stage1AddrCheckDecode".into(),
        KernelTranscriptEvent::Stage1AddrCheckAlu => ".stage1AddrCheckAlu".into(),
        KernelTranscriptEvent::Stage1AddrCheckEq4 => ".stage1AddrCheckEq4".into(),
        KernelTranscriptEvent::RecordFetchAddr => ".recordFetchAddr".into(),
        KernelTranscriptEvent::RecordDecodeAddr => ".recordDecodeAddr".into(),
        KernelTranscriptEvent::RecordAluAddr => ".recordAluAddr".into(),
        KernelTranscriptEvent::DeriveAdd8LoAddr => ".deriveAdd8LoAddr".into(),
        KernelTranscriptEvent::RecordEq4Addr => ".recordEq4Addr".into(),
        KernelTranscriptEvent::SampleGammaLookupLink => ".sampleGammaLookupLink".into(),
        KernelTranscriptEvent::Stage1LinkageBatch => ".stage1LinkageBatch".into(),
        KernelTranscriptEvent::SampleStage2Cycle => ".sampleStage2Cycle".into(),
        KernelTranscriptEvent::SampleGammaReg => ".sampleGammaReg".into(),
        KernelTranscriptEvent::Stage2RegRwBatched => ".stage2RegRwBatched".into(),
        KernelTranscriptEvent::Stage2RegValFromInc => ".stage2RegValFromInc".into(),
        KernelTranscriptEvent::SampleGammaRam => ".sampleGammaRam".into(),
        KernelTranscriptEvent::Stage2RamRwBatched => ".stage2RamRwBatched".into(),
        KernelTranscriptEvent::Stage2RamValFromInc => ".stage2RamValFromInc".into(),
        KernelTranscriptEvent::Stage2RamRafRead => ".stage2RamRafRead".into(),
        KernelTranscriptEvent::Stage2RamRafWrite => ".stage2RamRafWrite".into(),
        KernelTranscriptEvent::Stage2AddrCheckRegRaX => ".stage2AddrCheckRegRaX".into(),
        KernelTranscriptEvent::Stage2AddrCheckRegRaY => ".stage2AddrCheckRegRaY".into(),
        KernelTranscriptEvent::Stage2AddrCheckRegRaI => ".stage2AddrCheckRegRaI".into(),
        KernelTranscriptEvent::Stage2AddrCheckRegWa => ".stage2AddrCheckRegWa".into(),
        KernelTranscriptEvent::Stage2AddrCheckRamRa => ".stage2AddrCheckRamRa".into(),
        KernelTranscriptEvent::Stage2AddrCheckRamWa => ".stage2AddrCheckRamWa".into(),
        KernelTranscriptEvent::RecordRegAddr => ".recordRegAddr".into(),
        KernelTranscriptEvent::RecordRamAddr => ".recordRamAddr".into(),
        KernelTranscriptEvent::SampleGammaTwistLink => ".sampleGammaTwistLink".into(),
        KernelTranscriptEvent::Stage2LinkageBatch => ".stage2LinkageBatch".into(),
        KernelTranscriptEvent::SampleBeta1 => ".sampleBeta1".into(),
        KernelTranscriptEvent::SampleBeta2 => ".sampleBeta2".into(),
        KernelTranscriptEvent::SampleStage3Cycle => ".sampleStage3Cycle".into(),
        KernelTranscriptEvent::LaneShiftReduction => ".laneShiftReduction".into(),
        KernelTranscriptEvent::Stage3Continuity => ".stage3Continuity".into(),
        KernelTranscriptEvent::Stage3StartBoundaryOpening => ".stage3StartBoundaryOpening".into(),
        KernelTranscriptEvent::Stage3FinalBoundaryOpening => ".stage3FinalBoundaryOpening".into(),
        KernelTranscriptEvent::RowBinding(j) => format!("(.rowBinding {j})"),
        KernelTranscriptEvent::EmitKernelOpeningClaims => ".emitKernelOpeningClaims".into(),
    }
}

fn render_kernel_transcript_surface_view(
    transcript: &neo_fold_prototype::chip8::kernel::KernelTranscriptSurface,
) -> String {
    let mut events = String::from("[");
    for (idx, event) in transcript.events.iter().enumerate() {
        if idx > 0 {
            events.push_str(", ");
        }
        events.push_str(&render_kernel_transcript_event(event));
    }
    events.push(']');
    format!("mkKernelTranscriptSurfaceView\n      {}", events)
}

fn render_digest_list(values: &[[u8; 32]]) -> String {
    let mut out = String::from("[");
    for (idx, value) in values.iter().enumerate() {
        if idx > 0 {
            out.push_str(", ");
        }
        out.push_str(&render_u8_list(value));
    }
    out.push(']');
    out
}

fn render_kernel_exact_opening_transcript_entry_view(entry: &KernelExactOpeningTranscriptEntry) -> String {
    format!(
        "mkKernelExactOpeningTranscriptEntryView\n      {}\n      {}",
        render_u8_list(&entry.claim_digest),
        render_u8_list(&entry.witness_digest),
    )
}

fn render_kernel_time_opening_transcript_group_view(group: &KernelTimeOpeningTranscriptGroup) -> String {
    format!(
        "mkKernelTimeOpeningTranscriptGroupView\n      {}\n      {}",
        render_u8_list(&group.group_digest),
        render_u8_list(&group.reduced_digest),
    )
}

fn render_kernel_time_opening_transcript_unification_view(
    unification: &KernelTimeOpeningTranscriptUnification,
) -> String {
    format!(
        "mkKernelTimeOpeningTranscriptUnificationView\n      {}\n      {}\n      {}\n      {}\n      {}\n      {}",
        render_k_value(unification.claimed_sum),
        render_k_rounds(&unification.round_polys),
        render_k_list(&unification.r_unify),
        render_k_list(&unification.unified_point),
        render_bool(unification.can_unify),
        render_u8_list(&unification.unified_digest),
    )
}

fn render_optional_digest(value: Option<[u8; 32]>) -> String {
    match value {
        Some(digest) => format!("(some {})", render_u8_list(&digest)),
        None => "none".into(),
    }
}

fn render_kernel_joint_opening_transcript_unification_view(
    unification: &KernelJointOpeningTranscriptUnification,
) -> String {
    format!(
        "mkKernelJointOpeningTranscriptUnificationView\n      {}\n      {}\n      {}\n      {}",
        render_k_value(unification.claimed_sum),
        render_k_rounds(&unification.round_polys),
        render_k_list(&unification.r_unify),
        render_optional_digest(unification.unified_fold_digest),
    )
}

fn render_kernel_opening_transcript_source_view(source: &KernelOpeningTranscriptSource) -> String {
    let mut exact_openings = String::from("[");
    for (idx, entry) in source.exact_openings.iter().enumerate() {
        if idx > 0 {
            exact_openings.push_str(", ");
        }
        exact_openings.push_str(&format!("({})", render_kernel_exact_opening_transcript_entry_view(entry)));
    }
    exact_openings.push(']');

    let mut time_opening_groups = String::from("[");
    for (idx, group) in source.time_opening_groups.iter().enumerate() {
        if idx > 0 {
            time_opening_groups.push_str(", ");
        }
        time_opening_groups.push_str(&format!("({})", render_kernel_time_opening_transcript_group_view(group)));
    }
    time_opening_groups.push(']');

    format!(
        "mkKernelOpeningTranscriptSourceView\n      {}\n      {}\n      {}\n      {}\n      {}\n      ({})\n      {}\n      {}\n      ({})\n      {}",
        exact_openings,
        render_digest_list(&source.refinement_digests),
        render_u8_list(&source.time_opening_manifest_digest),
        render_u8_list(&source.time_opening_proof_digest),
        time_opening_groups,
        render_kernel_time_opening_transcript_unification_view(&source.time_opening_unification),
        render_digest_list(&source.joint_claim_digests),
        render_digest_list(&source.joint_group_digests),
        render_kernel_joint_opening_transcript_unification_view(&source.joint_opening_unification),
        render_digest_list(&source.fold_bucket_digests),
    )
}

fn render_kernel_opening_transcript_surface_view(surface: &KernelOpeningTranscriptSurface) -> String {
    format!(
        "mkKernelOpeningTranscriptSurfaceView\n      {}\n      {}\n      ({})",
        render_u8_list(&surface.kernel_manifest_digest),
        render_u8_list(&surface.root_manifest_digest),
        render_kernel_opening_transcript_source_view(&surface.source),
    )
}

fn render_stage1_channel(channel: Stage1ShoutChannel) -> &'static str {
    match channel {
        Stage1ShoutChannel::Fetch => ".fetch",
        Stage1ShoutChannel::Decode => ".decode",
        Stage1ShoutChannel::Alu => ".alu",
        Stage1ShoutChannel::Eq4 => ".eq4",
    }
}

fn render_address_family(family: AddressFamily) -> &'static str {
    match family {
        AddressFamily::Fetch => ".fetch",
        AddressFamily::Decode => ".decode",
        AddressFamily::Alu => ".alu",
        AddressFamily::Eq4 => ".eq4",
        AddressFamily::RegRaX => ".regRaX",
        AddressFamily::RegRaY => ".regRaY",
        AddressFamily::RegRaI => ".regRaI",
        AddressFamily::RegWa => ".regWa",
        AddressFamily::RamRa => ".ramRa",
        AddressFamily::RamWa => ".ramWa",
    }
}

fn render_twist_read_family(family: TwistReadFamily) -> &'static str {
    match family {
        TwistReadFamily::RegX => ".regX",
        TwistReadFamily::RegY => ".regY",
        TwistReadFamily::RegI => ".regI",
        TwistReadFamily::Ram => ".ram",
    }
}

fn render_twist_memory_family(family: TwistMemoryFamily) -> &'static str {
    match family {
        TwistMemoryFamily::Reg => ".reg",
        TwistMemoryFamily::Ram => ".ram",
    }
}

fn render_stage1_channel_list(values: &[Stage1ShoutChannel]) -> String {
    let mut out = String::from("[");
    for (idx, value) in values.iter().enumerate() {
        if idx > 0 {
            out.push_str(", ");
        }
        out.push_str(render_stage1_channel(*value));
    }
    out.push(']');
    out
}

fn render_address_family_list(values: &[AddressFamily]) -> String {
    let mut out = String::from("[");
    for (idx, value) in values.iter().enumerate() {
        if idx > 0 {
            out.push_str(", ");
        }
        out.push_str(render_address_family(*value));
    }
    out.push(']');
    out
}

fn render_twist_read_family_list(values: &[TwistReadFamily]) -> String {
    let mut out = String::from("[");
    for (idx, value) in values.iter().enumerate() {
        if idx > 0 {
            out.push_str(", ");
        }
        out.push_str(render_twist_read_family(*value));
    }
    out.push(']');
    out
}

fn render_twist_memory_family_list(values: &[TwistMemoryFamily]) -> String {
    let mut out = String::from("[");
    for (idx, value) in values.iter().enumerate() {
        if idx > 0 {
            out.push_str(", ");
        }
        out.push_str(render_twist_memory_family(*value));
    }
    out.push(']');
    out
}

fn render_error_term(term: KernelErrorTerm) -> String {
    match term {
        KernelErrorTerm::ShoutCore(channel) => format!("(.shoutCore {})", render_stage1_channel(channel)),
        KernelErrorTerm::Addr(family) => format!("(.addr {})", render_address_family(family)),
        KernelErrorTerm::TwistRead(family) => format!("(.twistRead {})", render_twist_read_family(family)),
        KernelErrorTerm::TwistWrite(family) => format!("(.twistWrite {})", render_twist_memory_family(family)),
        KernelErrorTerm::TwistVal(family) => format!("(.twistVal {})", render_twist_memory_family(family)),
        KernelErrorTerm::RamRafRead => ".ramRafRead".into(),
        KernelErrorTerm::RamRafWrite => ".ramRafWrite".into(),
        KernelErrorTerm::ShiftReduce => ".shiftReduce".into(),
        KernelErrorTerm::Continuity => ".continuity".into(),
        KernelErrorTerm::RegRwBatch => ".regRwBatch".into(),
        KernelErrorTerm::RamRwBatch => ".ramRwBatch".into(),
        KernelErrorTerm::LookupLink => ".lookupLink".into(),
        KernelErrorTerm::TwistLink => ".twistLink".into(),
        KernelErrorTerm::Pcs => ".pcs".into(),
        KernelErrorTerm::Fs => ".fs".into(),
        KernelErrorTerm::Outer => ".outer".into(),
    }
}

fn render_error_term_list(values: &[KernelErrorTerm]) -> String {
    let mut out = String::from("[");
    for (idx, value) in values.iter().enumerate() {
        if idx > 0 {
            out.push_str(", ");
        }
        out.push_str(&render_error_term(*value));
    }
    out.push(']');
    out
}

fn render_kernel_error_surface_view(
    error: &neo_fold_prototype::chip8::kernel::KernelErrorSurface,
) -> String {
    format!(
        "mkKernelErrorSurfaceView\n      {}\n      {}\n      {}\n      {}\n      {}\n      {}\n      {}\n      {}\n      {}\n      {}\n      {}\n      {}\n      {}",
        render_stage1_channel_list(&error.stage1_channels),
        render_address_family_list(&error.stage1_address_families),
        render_twist_read_family_list(&error.reg_read_families),
        render_address_family_list(&error.reg_address_families),
        render_address_family_list(&error.ram_address_families),
        render_twist_memory_family_list(&error.twist_memory_families),
        render_error_term_list(&error.stage1_terms),
        render_error_term_list(&error.stage2_terms),
        render_error_term_list(&error.stage3_terms),
        render_error_term_list(&error.batch_terms),
        render_error_term_list(&error.tail_terms),
        render_u8_list(&error.total_upper_digest),
        render_u8_list(&error.digest),
    )
}

fn case_module_fragment(name: &str) -> String {
    format!("Case_{}", lean_ident_fragment(name))
}

fn release_artifact_legacy_output_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../Nightstream/Chip8/Generated/ReleaseArtifactVectors.lean")
}

fn imported_opening_transcript_output_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../Nightstream/Chip8/Generated/ImportedOpeningTranscriptCases.lean")
}

fn imported_release_artifact_legacy_output_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../Nightstream/Chip8/Generated/ImportedReleaseArtifact.lean")
}

fn transcript_output_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../Nightstream/Chip8/Generated/TranscriptVectors.lean")
}

fn bundle_legacy_output_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../Nightstream/Chip8/Generated/StagedExecutionDigestBundleVectors.lean")
}

fn release_artifact_output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../Nightstream/Chip8/Generated/ReleaseArtifactVectors")
}

fn release_artifact_corpus_output_path() -> PathBuf {
    release_artifact_output_dir().join("Corpus.lean")
}

fn release_artifact_case_output_path(name: &str) -> PathBuf {
    release_artifact_output_dir().join(format!("{}.lean", case_module_fragment(name)))
}

fn imported_release_artifact_output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../Nightstream/Chip8/Generated/ImportedReleaseArtifact")
}

fn imported_release_artifact_corpus_output_path() -> PathBuf {
    imported_release_artifact_output_dir().join("Corpus.lean")
}

fn imported_release_artifact_case_output_path(name: &str) -> PathBuf {
    imported_release_artifact_output_dir().join(format!("{}.lean", case_module_fragment(name)))
}

fn bundle_output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../Nightstream/Chip8/Generated/StagedExecutionDigestBundleVectors")
}

fn bundle_corpus_output_path() -> PathBuf {
    bundle_output_dir().join("Corpus.lean")
}

fn bundle_case_output_path(name: &str) -> PathBuf {
    bundle_output_dir().join(format!("{}.lean", case_module_fragment(name)))
}

fn reset_generated_module_dir(path: &Path) {
    if path.exists() {
        fs::remove_dir_all(path).expect("remove stale generated module dir");
    }
    fs::create_dir_all(path).expect("create generated module dir");
}

fn remove_generated_path_if_exists(path: &Path) {
    if !path.exists() {
        return;
    }
    let metadata = fs::metadata(path).expect("read generated path metadata");
    if metadata.is_dir() {
        fs::remove_dir_all(path).expect("remove stale generated dir");
    } else {
        fs::remove_file(path).expect("remove stale generated file");
    }
}

fn fixture_named<'a>(fixtures: &'a [KernelFixture], name: &str) -> &'a KernelFixture {
    fixtures
        .iter()
        .find(|fixture| fixture.name == name)
        .unwrap_or_else(|| panic!("unknown CHIP-8 fixture '{name}'"))
}

fn is_release_artifact_fixture(name: &str) -> bool {
    matches!(
        name,
        "jump_rows_2_seed_empty" | "jump_rows_3_seed_nonempty" | "mixed_ldimm_addimm_10"
    )
}

fn imported_release_artifact_def_name(name: &str) -> String {
    format!("importedReleaseArtifact_{}", lean_ident_fragment(name))
}

fn imported_root0_bindings_def_name(name: &str) -> String {
    format!("importedRoot0Bindings_{}", lean_ident_fragment(name))
}

fn imported_trace_digests_def_name(name: &str) -> String {
    format!("importedTraceDigests_{}", lean_ident_fragment(name))
}

fn imported_frames_def_name(name: &str) -> String {
    format!("importedFrames_{}", lean_ident_fragment(name))
}

fn imported_stage3s_def_name(name: &str) -> String {
    format!("importedStage3s_{}", lean_ident_fragment(name))
}

fn imported_artifact_view_def_name(name: &str) -> String {
    format!("importedArtifactView_{}", lean_ident_fragment(name))
}

fn imported_opening_transcript_source_def_name(name: &str) -> String {
    format!("importedOpeningTranscriptSource_{}", lean_ident_fragment(name))
}

fn imported_opening_transcript_surface_def_name(name: &str) -> String {
    format!("importedOpeningTranscriptSurface_{}", lean_ident_fragment(name))
}

fn imported_opening_transcript_case_def_name(name: &str) -> String {
    format!("importedOpeningTranscriptCase_{}", lean_ident_fragment(name))
}

fn imported_kernel_manifest_view_def_name(name: &str) -> String {
    format!("importedKernelManifestView_{}", lean_ident_fragment(name))
}

fn imported_root_manifest_view_def_name(name: &str) -> String {
    format!("importedRootManifestView_{}", lean_ident_fragment(name))
}

fn imported_manifest_surface_view_def_name(name: &str) -> String {
    format!("importedManifestSurfaceView_{}", lean_ident_fragment(name))
}

fn imported_trace_surface_view_def_name(name: &str) -> String {
    format!("importedTraceSurfaceView_{}", lean_ident_fragment(name))
}

fn imported_export_surface_view_def_name(name: &str) -> String {
    format!("importedExportSurfaceView_{}", lean_ident_fragment(name))
}

fn imported_audit_surface_view_def_name(name: &str) -> String {
    format!("importedAuditSurfaceView_{}", lean_ident_fragment(name))
}

fn imported_transcript_surface_view_def_name(name: &str) -> String {
    format!("importedTranscriptSurfaceView_{}", lean_ident_fragment(name))
}

fn imported_error_surface_view_def_name(name: &str) -> String {
    format!("importedErrorSurfaceView_{}", lean_ident_fragment(name))
}

fn imported_kernel_digest_view_def_name(name: &str) -> String {
    format!("importedKernelDigestView_{}", lean_ident_fragment(name))
}

fn imported_staged_bundle_view_def_name(name: &str) -> String {
    format!("importedStagedBundleView_{}", lean_ident_fragment(name))
}

fn release_artifact_bindings_def_name(name: &str) -> String {
    format!("releaseArtifactBindings_{}", lean_ident_fragment(name))
}

fn release_artifact_trace_digests_def_name(name: &str) -> String {
    format!("releaseArtifactTraceDigests_{}", lean_ident_fragment(name))
}

fn release_artifact_frames_def_name(name: &str) -> String {
    format!("releaseArtifactFrames_{}", lean_ident_fragment(name))
}

fn release_artifact_stage3s_def_name(name: &str) -> String {
    format!("releaseArtifactStage3s_{}", lean_ident_fragment(name))
}

fn release_artifact_kernel_manifest_view_def_name(name: &str) -> String {
    format!("releaseArtifactKernelManifestView_{}", lean_ident_fragment(name))
}

fn release_artifact_root_manifest_view_def_name(name: &str) -> String {
    format!("releaseArtifactRootManifestView_{}", lean_ident_fragment(name))
}

fn release_artifact_manifest_surface_view_def_name(name: &str) -> String {
    format!("releaseArtifactManifestSurfaceView_{}", lean_ident_fragment(name))
}

fn release_artifact_trace_surface_view_def_name(name: &str) -> String {
    format!("releaseArtifactTraceSurfaceView_{}", lean_ident_fragment(name))
}

fn release_artifact_export_surface_view_def_name(name: &str) -> String {
    format!("releaseArtifactExportSurfaceView_{}", lean_ident_fragment(name))
}

fn release_artifact_audit_surface_view_def_name(name: &str) -> String {
    format!("releaseArtifactAuditSurfaceView_{}", lean_ident_fragment(name))
}

fn release_artifact_transcript_surface_view_def_name(name: &str) -> String {
    format!("releaseArtifactTranscriptSurfaceView_{}", lean_ident_fragment(name))
}

fn release_artifact_error_surface_view_def_name(name: &str) -> String {
    format!("releaseArtifactErrorSurfaceView_{}", lean_ident_fragment(name))
}

fn release_artifact_kernel_digest_view_def_name(name: &str) -> String {
    format!("releaseArtifactKernelDigestView_{}", lean_ident_fragment(name))
}

fn release_artifact_bundle_view_def_name(name: &str) -> String {
    format!("releaseArtifactBundleView_{}", lean_ident_fragment(name))
}

fn release_artifact_view_def_name(name: &str) -> String {
    format!("releaseArtifactView_{}", lean_ident_fragment(name))
}

fn indexed_def_name(prefix: &str, idx: usize) -> String {
    format!("{prefix}_{idx}")
}

fn render_named_reference_list(def_names: &[String]) -> String {
    let mut out = String::from("[");
    for (idx, def_name) in def_names.iter().enumerate() {
        if idx > 0 {
            out.push_str(", ");
        }
        out.push_str(def_name);
    }
    out.push(']');
    out
}

fn render_named_value_defs(type_name: &str, def_prefix: &str, values: &[String]) -> (String, Vec<String>) {
    let mut out = String::new();
    let mut names = Vec::with_capacity(values.len());
    for (idx, value) in values.iter().enumerate() {
        let def_name = indexed_def_name(def_prefix, idx);
        out.push_str(&format!(
            "def {def_name} : {type_name} :=\n  ({value})\n\n",
            def_name = def_name,
            type_name = type_name,
            value = value,
        ));
        names.push(def_name);
    }
    (out, names)
}

fn render_named_list_def(def_name: &str, item_type: &str, item_names: &[String]) -> String {
    format!(
        "def {def_name} : List {item_type} :=\n  {items}\n\n",
        def_name = def_name,
        item_type = item_type,
        items = render_named_reference_list(item_names),
    )
}

fn render_staged_execution_digest_view_from_refs(frame_def: &str, stage3_def: &str) -> String {
    render_staged_execution_digest_view_from_parts(
        &format!(
            "mkStage1View\n        {frame_def}.pre\n        {frame_def}.dec\n        {frame_def}.row",
            frame_def = frame_def,
        ),
        &format!(
            "mkStage2View\n        {frame_def}.pre\n        {frame_def}.post\n        {frame_def}.dec\n        {frame_def}.row",
            frame_def = frame_def,
        ),
        stage3_def,
        &format!(
            "mkExecutionResultView\n        {frame_def}.stepIdx\n        {frame_def}.pre\n        {frame_def}.post\n        {frame_def}.dec",
            frame_def = frame_def,
        ),
    )
}

fn render_named_staged_digest_defs_from_refs(
    def_prefix: &str,
    frame_names: &[String],
    stage3_names: &[String],
) -> (String, Vec<String>) {
    assert_eq!(
        frame_names.len(),
        stage3_names.len(),
        "frame/stage3 count mismatch when rendering staged digest defs",
    );
    let values: Vec<String> = frame_names
        .iter()
        .zip(stage3_names.iter())
        .map(|(frame_name, stage3_name)| render_staged_execution_digest_view_from_refs(frame_name, stage3_name))
        .collect();
    render_named_value_defs("StagedExecutionDigestView", def_prefix, &values)
}

fn render_imported_root0_bindings_list(case: &ReleaseArtifactCase) -> String {
    let mut bindings = String::from("[");
    for (idx, binding) in case.imported_artifact.root0_bindings.iter().enumerate() {
        if idx > 0 {
            bindings.push_str(", ");
        }
        bindings.push_str(&render_root0_binding_view(binding));
    }
    bindings.push(']');
    bindings
}

fn render_imported_artifact_value(
    root0_bindings_def: &str,
    trace_digests_def: &str,
    frames_def: &str,
    stage3s_def: &str,
    source_def: &str,
    surface_def: &str,
    artifact_view_def: &str,
) -> String {
    format!(
        "Nightstream.Chip8.ExternalReleaseArtifact.ImportedArtifact.mk\n\
      {root0_bindings_def}\n\
      {trace_digests_def}\n\
      {frames_def}\n\
      {stage3s_def}\n\
      (some {source_def})\n\
      (some {surface_def})\n\
      {artifact_view_def}",
        root0_bindings_def = root0_bindings_def,
        trace_digests_def = trace_digests_def,
        frames_def = frames_def,
        stage3s_def = stage3s_def,
        source_def = source_def,
        surface_def = surface_def,
        artifact_view_def = artifact_view_def,
    )
}

fn render_imported_release_artifact_case_module(case: &ReleaseArtifactCase) -> String {
    assert_eq!(
        case.imported_artifact.frames.len(),
        case.imported_artifact.stage3s.len(),
        "imported frame/stage3 count mismatch for {}",
        case.name,
    );
    assert_eq!(
        case.imported_artifact.frames.len(),
        case.imported_artifact.artifact.staged_bundle.digests.len(),
        "imported frame/bundle digest count mismatch for {}",
        case.name,
    );
    let source_def = imported_opening_transcript_source_def_name(case.name);
    let surface_def = imported_opening_transcript_surface_def_name(case.name);
    let artifact_def = imported_release_artifact_def_name(case.name);
    let meta_pub_def = imported_release_artifact_meta_pub_def_name_from_name(case.name);
    let root0_bindings_def = imported_root0_bindings_def_name(case.name);
    let trace_digests_def = imported_trace_digests_def_name(case.name);
    let frames_def = imported_frames_def_name(case.name);
    let stage3s_def = imported_stage3s_def_name(case.name);
    let artifact_view_def = imported_artifact_view_def_name(case.name);
    let kernel_manifest_view_def = imported_kernel_manifest_view_def_name(case.name);
    let root_manifest_view_def = imported_root_manifest_view_def_name(case.name);
    let manifest_surface_view_def = imported_manifest_surface_view_def_name(case.name);
    let trace_surface_view_def = imported_trace_surface_view_def_name(case.name);
    let export_surface_view_def = imported_export_surface_view_def_name(case.name);
    let audit_surface_view_def = imported_audit_surface_view_def_name(case.name);
    let transcript_surface_view_def = imported_transcript_surface_view_def_name(case.name);
    let error_surface_view_def = imported_error_surface_view_def_name(case.name);
    let kernel_digest_view_def = imported_kernel_digest_view_def_name(case.name);
    let staged_bundle_view_def = imported_staged_bundle_view_def_name(case.name);
    let bundle_public_surface_def = format!("importedBundlePublicSurface_{}", lean_ident_fragment(case.name));
    let bundle_digests_def = format!("importedBundleDigests_{}", lean_ident_fragment(case.name));
    let bundle_digest_def_prefix = format!("importedBundleDigest_{}", lean_ident_fragment(case.name));
    let frame_def_prefix = format!("importedFrame_{}", lean_ident_fragment(case.name));
    let stage3_def_prefix = format!("importedStage3_{}", lean_ident_fragment(case.name));
    let kernel_manifest_claims_def = format!("{}_claims", kernel_manifest_view_def);
    let root_manifest_claims_def = format!("{}_claims", root_manifest_view_def);
    let kernel_manifest_claim_def_prefix = format!("{}_claim", kernel_manifest_view_def);
    let root_manifest_claim_def_prefix = format!("{}_claim", root_manifest_view_def);
    let mut root0_ids = String::from("[");
    for (idx, id) in case
        .imported_artifact
        .artifact
        .kernel_digest
        .manifest_surface
        .root0_commitment_ids
        .iter()
        .enumerate()
    {
        if idx > 0 {
            root0_ids.push_str(", ");
        }
        root0_ids.push_str(&render_commitment_id(*id));
    }
    root0_ids.push(']');
    let frame_values: Vec<String> = case
        .imported_artifact
        .frames
        .iter()
        .map(render_frame_source_view)
        .collect();
    let stage3_values: Vec<String> = case
        .imported_artifact
        .stage3s
        .iter()
        .map(render_stage3_view)
        .collect();
    let (frame_defs, frame_names) = render_named_value_defs("FrameSourceView", &frame_def_prefix, &frame_values);
    let (stage3_defs, stage3_names) = render_named_value_defs("Stage3View", &stage3_def_prefix, &stage3_values);
    let (bundle_digest_defs, bundle_digest_names) =
        render_named_staged_digest_defs_from_refs(&bundle_digest_def_prefix, &frame_names, &stage3_names);
    let kernel_manifest_claim_values: Vec<String> = case
        .imported_artifact
        .artifact
        .kernel_digest
        .manifest_surface
        .kernel_manifest
        .claims
        .iter()
        .map(render_opening_claim_view)
        .collect();
    let (kernel_manifest_claim_defs, kernel_manifest_claim_names) = render_named_value_defs(
        "KernelOpeningClaimView",
        &kernel_manifest_claim_def_prefix,
        &kernel_manifest_claim_values,
    );
    let root_manifest_claim_values: Vec<String> = case
        .imported_artifact
        .artifact
        .kernel_digest
        .manifest_surface
        .root_manifest
        .claims
        .iter()
        .map(render_opening_claim_view)
        .collect();
    let (root_manifest_claim_defs, root_manifest_claim_names) = render_named_value_defs(
        "KernelOpeningClaimView",
        &root_manifest_claim_def_prefix,
        &root_manifest_claim_values,
    );

    let mut out = String::new();
    out.push_str("import Nightstream.Chip8.Kernel.ExternalReleaseArtifact\n\n");
    out.push_str("set_option maxHeartbeats 0\n\n");
    out.push_str("namespace Nightstream.Chip8.Generated.ImportedReleaseArtifact\n\n");
    out.push_str("open Nightstream.Chip8.ExternalReleaseArtifact\n\n");
    out.push_str(&render_named_meta_pub_def(
        &meta_pub_def,
        &case.imported_artifact.artifact.staged_bundle.public.meta_pub,
    ));
    out.push('\n');
    out.push_str(&frame_defs);
    out.push_str(&render_named_list_def(&frames_def, "FrameSourceView", &frame_names));
    out.push_str(&stage3_defs);
    out.push_str(&render_named_list_def(&stage3s_def, "Stage3View", &stage3_names));
    out.push_str(&bundle_digest_defs);
    out.push_str(&render_named_list_def(
        &bundle_digests_def,
        "StagedExecutionDigestView",
        &bundle_digest_names,
    ));
    out.push_str(&kernel_manifest_claim_defs);
    out.push_str(&render_named_list_def(
        &kernel_manifest_claims_def,
        "KernelOpeningClaimView",
        &kernel_manifest_claim_names,
    ));
    out.push_str(&root_manifest_claim_defs);
    out.push_str(&render_named_list_def(
        &root_manifest_claims_def,
        "KernelOpeningClaimView",
        &root_manifest_claim_names,
    ));
    out.push_str(&format!(
        "def {root0_bindings_def} : List CommitmentBinding :=\n  {root0_bindings}\n\n\
def {trace_digests_def} : TraceDigestSourceView :=\n  ({trace_digests})\n\n\
def {kernel_manifest_view_def} : KernelOpeningManifestView :=\n  ({kernel_manifest_view})\n\n\
def {root_manifest_view_def} : KernelOpeningManifestView :=\n  ({root_manifest_view})\n\n\
def {manifest_surface_view_def} : KernelManifestSurfaceView :=\n  mkKernelManifestSurfaceView\n      {root0_ids}\n      {kernel_manifest_view_def}\n      {root_manifest_view_def}\n\n\
def {trace_surface_view_def} : KernelTraceSurfaceView :=\n  ({trace_surface_view})\n\n\
def {export_surface_view_def} : KernelExportSurfaceView :=\n  ({export_surface_view})\n\n\
def {audit_surface_view_def} : KernelAuditSurfaceView :=\n  ({audit_surface_view})\n\n\
def {transcript_surface_view_def} : KernelTranscriptSurfaceView :=\n  ({transcript_surface_view})\n\n\
def {error_surface_view_def} : KernelErrorSurfaceView :=\n  ({error_surface_view})\n\n\
def {kernel_digest_view_def} : KernelExecutionDigestView :=\n  mkKernelExecutionDigestView\n    {trace_surface_view_def}\n    {export_surface_view_def}\n    {audit_surface_view_def}\n    {manifest_surface_view_def}\n    {transcript_surface_view_def}\n    {error_surface_view_def}\n\n\
def {bundle_public_surface_def} : DigestPublicView :=\n  ({bundle_public_surface})\n\n\
def {staged_bundle_view_def} : StagedExecutionDigestBundleView :=\n  ({staged_bundle_view})\n\n\
def {artifact_view_def} : KernelReleaseArtifactView :=\n  mkKernelReleaseArtifactView\n      {kernel_digest_view_def}\n      {staged_bundle_view_def}\n\n\
def {source_def} : KernelOpeningTranscriptSourceView :=\n  {opening_transcript_source}\n\n\
def {surface_def} : KernelOpeningTranscriptSurfaceView :=\n  {opening_transcript_surface}\n\n\
def {artifact_def} : Nightstream.Chip8.ExternalReleaseArtifact.ImportedArtifact :=\n  {artifact_value}\n\n\
",
        root0_bindings_def = root0_bindings_def,
        root0_bindings = render_imported_root0_bindings_list(case),
        trace_digests_def = trace_digests_def,
        trace_digests = render_trace_digest_source_view(&case.imported_artifact.trace_digests),
        kernel_manifest_view_def = kernel_manifest_view_def,
        kernel_manifest_view =
            render_opening_manifest_view_from_claims_expr(
                &kernel_manifest_claims_def,
                &case.imported_artifact.artifact.kernel_digest.manifest_surface.kernel_manifest.digest,
            ),
        root_manifest_view_def = root_manifest_view_def,
        root_manifest_view =
            render_opening_manifest_view_from_claims_expr(
                &root_manifest_claims_def,
                &case.imported_artifact.artifact.kernel_digest.manifest_surface.root_manifest.digest,
            ),
        manifest_surface_view_def = manifest_surface_view_def,
        root0_ids = root0_ids,
        trace_surface_view_def = trace_surface_view_def,
        trace_surface_view =
            render_kernel_trace_surface_view_from_frames_expr(
                &case.imported_artifact.artifact.kernel_digest.trace_surface,
                &frames_def,
            ),
        export_surface_view_def = export_surface_view_def,
        export_surface_view =
            render_kernel_export_surface_view(&case.imported_artifact.artifact.kernel_digest.export_surface),
        audit_surface_view_def = audit_surface_view_def,
        audit_surface_view =
            render_kernel_audit_surface_view(&case.imported_artifact.artifact.kernel_digest.audit_surface),
        transcript_surface_view_def = transcript_surface_view_def,
        transcript_surface_view =
            render_kernel_transcript_surface_view(&case.imported_artifact.artifact.kernel_digest.transcript_surface),
        error_surface_view_def = error_surface_view_def,
        error_surface_view =
            render_kernel_error_surface_view(&case.imported_artifact.artifact.kernel_digest.error_surface),
        kernel_digest_view_def = kernel_digest_view_def,
        bundle_public_surface_def = bundle_public_surface_def,
        bundle_public_surface = render_digest_public_view_with_meta_expr(
            &case.imported_artifact.artifact.staged_bundle.public.public,
            &meta_pub_def,
        ),
        staged_bundle_view_def = staged_bundle_view_def,
        staged_bundle_view = render_bundle_view_from_parts(&bundle_public_surface_def, &bundle_digests_def),
        artifact_view_def = artifact_view_def,
        source_def = source_def,
        surface_def = surface_def,
        artifact_def = artifact_def,
        artifact_value = render_imported_artifact_value(
            &root0_bindings_def,
            &trace_digests_def,
            &frames_def,
            &stage3s_def,
            &source_def,
            &surface_def,
            &artifact_view_def,
        ),
        opening_transcript_source =
            render_kernel_opening_transcript_source_view(&case.imported_artifact.opening_transcript_source),
        opening_transcript_surface =
            render_kernel_opening_transcript_surface_view(&case.imported_artifact.opening_transcript_surface),
    ));
    out.push_str("end Nightstream.Chip8.Generated.ImportedReleaseArtifact\n");
    out
}

fn render_imported_opening_transcript_case(case: &ReleaseArtifactCase) -> String {
    let mut manifest_claim_digests = String::from("[");
    for (idx, claim) in case
        .imported_artifact
        .artifact
        .kernel_digest
        .manifest_surface
        .kernel_manifest
        .claims
        .iter()
        .enumerate()
    {
        if idx > 0 {
            manifest_claim_digests.push_str(", ");
        }
        manifest_claim_digests.push_str(&render_u8_list(&claim.digest));
    }
    manifest_claim_digests.push(']');

    format!(
        "Nightstream.Chip8.ExternalReleaseArtifact.mkImportedOpeningTranscriptCase\n      \"{name}\"\n      {manifest_claim_digests}\n      {kernel_manifest_digest}\n      {root_manifest_digest}\n      ({source})\n      ({surface})",
        name = case.name,
        manifest_claim_digests = manifest_claim_digests,
        kernel_manifest_digest =
            render_u8_list(&case.imported_artifact.artifact.kernel_digest.manifest_surface.kernel_manifest.digest),
        root_manifest_digest =
            render_u8_list(&case.imported_artifact.artifact.kernel_digest.manifest_surface.root_manifest.digest),
        source =
            render_kernel_opening_transcript_source_view(&case.imported_artifact.opening_transcript_source),
        surface =
            render_kernel_opening_transcript_surface_view(&case.imported_artifact.opening_transcript_surface),
    )
}

fn render_imported_opening_transcript_corpus_module(cases: &[ReleaseArtifactCase]) -> String {
    let mut out = String::new();
    out.push_str("import Nightstream.Chip8.Kernel.ExternalReleaseArtifact\n\n");
    out.push_str("set_option maxHeartbeats 0\n\n");
    out.push_str("namespace Nightstream.Chip8.Generated\n\n");
    out.push_str("open Nightstream.Chip8.ExternalReleaseArtifact\n\n");
    for case in cases {
        let case_def = imported_opening_transcript_case_def_name(case.name);
        out.push_str(&format!(
            "def {case_def} : ImportedOpeningTranscriptCase :=\n  {case_value}\n\n",
            case_def = case_def,
            case_value = render_imported_opening_transcript_case(case),
        ));
    }
    out.push_str("def importedOpeningTranscriptCases : List ImportedOpeningTranscriptCase :=\n");
    out.push_str("  [\n");
    for (idx, case) in cases.iter().enumerate() {
        let case_def = imported_opening_transcript_case_def_name(case.name);
        out.push_str(&format!("    {}", case_def));
        if idx + 1 < cases.len() {
            out.push(',');
        }
        out.push('\n');
    }
    out.push_str("  ]\n\n");
    out.push_str("end Nightstream.Chip8.Generated\n");
    out
}

fn render_imported_release_artifact_corpus_module(cases: &[ReleaseArtifactCase]) -> String {
    let mut out = String::new();
    for case in cases {
        out.push_str(&format!(
            "import Nightstream.Chip8.Generated.ImportedReleaseArtifact.{}\n",
            case_module_fragment(case.name),
        ));
    }
    out.push('\n');
    out.push_str("namespace Nightstream.Chip8.Generated\n\n");
    out.push_str("open Nightstream.Chip8.Generated.ImportedReleaseArtifact\n\n");
    out.push_str(
        "def importedReleaseArtifacts : List (String × Nightstream.Chip8.ExternalReleaseArtifact.ImportedArtifact) :=\n",
    );
    out.push_str("  [\n");
    for (idx, case) in cases.iter().enumerate() {
        let artifact_def = imported_release_artifact_def_name(case.name);
        out.push_str(&format!(
            "    (\"{name}\", {artifact_def})",
            name = case.name,
            artifact_def = artifact_def,
        ));
        if idx + 1 < cases.len() {
            out.push(',');
        }
        out.push('\n');
    }
    out.push_str("  ]\n\n");
    out.push_str("def importedReleaseArtifactNames : List String :=\n");
    out.push_str("  importedReleaseArtifacts.map Prod.fst\n\n");
    out.push_str(
        "def importedReleaseArtifactValues : List Nightstream.Chip8.ExternalReleaseArtifact.ImportedArtifact :=\n",
    );
    out.push_str("  importedReleaseArtifacts.map Prod.snd\n\n");
    out.push_str("end Nightstream.Chip8.Generated\n");
    out
}

fn render_bundle_case_module(case: &BundleVectorCase) -> String {
    assert_eq!(
        case.frames.len(),
        case.bundle.digests.len(),
        "frame/bundle digest count mismatch for {}",
        case.name,
    );
    let frame_def_prefix = format!("bundleFrame_{}", lean_ident_fragment(case.name));
    let stage3_def_prefix = format!("bundleStage3_{}", lean_ident_fragment(case.name));
    let digest_def_prefix = format!("bundleDigest_{}", lean_ident_fragment(case.name));
    let frames_def = format!("bundleFrames_{}", lean_ident_fragment(case.name));
    let stage3s_def = format!("bundleStage3s_{}", lean_ident_fragment(case.name));
    let digests_def = format!("bundleDigests_{}", lean_ident_fragment(case.name));
    let public_surface_def = format!("bundlePublicSurface_{}", lean_ident_fragment(case.name));
    let expected_bundle_def = format!("bundleExpected_{}", lean_ident_fragment(case.name));
    let case_def = bundle_case_def_name_from_name(case.name);
    let frame_values: Vec<String> = case.frames.iter().map(render_frame_source_view).collect();
    let stage3_values: Vec<String> = case.stage3s.iter().map(render_stage3_view).collect();
    let (frame_defs, frame_names) = render_named_value_defs("FrameSourceView", &frame_def_prefix, &frame_values);
    let (stage3_defs, stage3_names) = render_named_value_defs("Stage3View", &stage3_def_prefix, &stage3_values);
    let (digest_defs, digest_names) =
        render_named_staged_digest_defs_from_refs(&digest_def_prefix, &frame_names, &stage3_names);
    let mut out = String::new();
    out.push_str("import Nightstream.Chip8.Generated.StagedExecutionDigestBundleVectorTypes\n\n");
    out.push_str("set_option maxHeartbeats 0\n\n");
    out.push_str("namespace Nightstream.Chip8.Generated.StagedExecutionDigestBundleVectors\n\n");
    out.push_str(&render_named_meta_pub_def(
        &bundle_meta_pub_def_name_from_name(case.name),
        &case.meta_pub,
    ));
    out.push('\n');
    out.push_str(&frame_defs);
    out.push_str(&render_named_list_def(&frames_def, "FrameSourceView", &frame_names));
    out.push_str(&stage3_defs);
    out.push_str(&render_named_list_def(&stage3s_def, "Stage3View", &stage3_names));
    out.push_str(&digest_defs);
    out.push_str(&render_named_list_def(&digests_def, "StagedExecutionDigestView", &digest_names));
    out.push_str(&format!(
        "def {public_surface_def} : DigestPublicView :=\n  ({public_surface})\n\n\
def {expected_bundle_def} : StagedExecutionDigestBundleView :=\n  ({expected_bundle})\n\n\
def {case_def} : StagedExecutionDigestBundleVectorCase :=\n  mkStagedExecutionDigestBundleVectorCase\n      \"{name}\"\n      {public_surface_def}\n      {frames_def}\n      {stage3s_def}\n      {expected_bundle_def}\n",
        public_surface_def = public_surface_def,
        public_surface = render_digest_public_view_with_meta_expr(
            &case.public,
            &bundle_meta_pub_def_name_from_name(case.name),
        ),
        expected_bundle_def = expected_bundle_def,
        expected_bundle = render_bundle_view_from_parts(&public_surface_def, &digests_def),
        case_def = case_def,
        name = case.name,
        frames_def = frames_def,
        stage3s_def = stage3s_def,
    ));
    out.push('\n');
    out.push_str("end Nightstream.Chip8.Generated.StagedExecutionDigestBundleVectors\n");
    out
}

fn render_bundle_corpus_module(cases: &[BundleVectorCase]) -> String {
    let mut out = String::new();
    for case in cases {
        out.push_str(&format!(
            "import Nightstream.Chip8.Generated.StagedExecutionDigestBundleVectors.{}\n",
            case_module_fragment(case.name),
        ));
    }
    out.push('\n');
    out.push_str("namespace Nightstream.Chip8.Generated\n\n");
    out.push_str("open Nightstream.Chip8.Generated.StagedExecutionDigestBundleVectors\n\n");
    out.push_str("def stagedExecutionDigestBundleVectorCases : List StagedExecutionDigestBundleVectorCase :=\n");
    out.push_str("  [\n");
    for (idx, case) in cases.iter().enumerate() {
        out.push_str("    ");
        out.push_str(&bundle_case_def_name_from_name(case.name));
        if idx + 1 < cases.len() {
            out.push(',');
        }
        out.push('\n');
    }
    out.push_str("  ]\n\n");
    out.push_str("end Nightstream.Chip8.Generated\n");
    out
}

fn render_release_artifact_case_module(case: &ReleaseArtifactCase) -> String {
    assert_eq!(
        case.imported_artifact.frames.len(),
        case.imported_artifact.stage3s.len(),
        "release-artifact frame/stage3 count mismatch for {}",
        case.name,
    );
    assert_eq!(
        case.imported_artifact.frames.len(),
        case.imported_artifact.artifact.staged_bundle.digests.len(),
        "release-artifact frame/bundle digest count mismatch for {}",
        case.name,
    );
    let bindings_def = release_artifact_bindings_def_name(case.name);
    let trace_digests_def = release_artifact_trace_digests_def_name(case.name);
    let frames_def = release_artifact_frames_def_name(case.name);
    let stage3s_def = release_artifact_stage3s_def_name(case.name);
    let kernel_manifest_view_def = release_artifact_kernel_manifest_view_def_name(case.name);
    let root_manifest_view_def = release_artifact_root_manifest_view_def_name(case.name);
    let manifest_surface_view_def = release_artifact_manifest_surface_view_def_name(case.name);
    let trace_surface_view_def = release_artifact_trace_surface_view_def_name(case.name);
    let export_surface_view_def = release_artifact_export_surface_view_def_name(case.name);
    let audit_surface_view_def = release_artifact_audit_surface_view_def_name(case.name);
    let transcript_surface_view_def = release_artifact_transcript_surface_view_def_name(case.name);
    let error_surface_view_def = release_artifact_error_surface_view_def_name(case.name);
    let kernel_digest_view_def = release_artifact_kernel_digest_view_def_name(case.name);
    let bundle_public_surface_def = format!("releaseArtifactBundlePublicSurface_{}", lean_ident_fragment(case.name));
    let bundle_digests_def = format!("releaseArtifactBundleDigests_{}", lean_ident_fragment(case.name));
    let bundle_digest_def_prefix = format!("releaseArtifactBundleDigest_{}", lean_ident_fragment(case.name));
    let frame_def_prefix = format!("releaseArtifactFrame_{}", lean_ident_fragment(case.name));
    let stage3_def_prefix = format!("releaseArtifactStage3_{}", lean_ident_fragment(case.name));
    let kernel_manifest_claims_def = format!("{}_claims", kernel_manifest_view_def);
    let root_manifest_claims_def = format!("{}_claims", root_manifest_view_def);
    let kernel_manifest_claim_def_prefix = format!("{}_claim", kernel_manifest_view_def);
    let root_manifest_claim_def_prefix = format!("{}_claim", root_manifest_view_def);
    let bundle_view_def = release_artifact_bundle_view_def_name(case.name);
    let artifact_view_def = release_artifact_view_def_name(case.name);
    let mut root0_ids = String::from("[");
    for (idx, id) in case
        .imported_artifact
        .artifact
        .kernel_digest
        .manifest_surface
        .root0_commitment_ids
        .iter()
        .enumerate()
    {
        if idx > 0 {
            root0_ids.push_str(", ");
        }
        root0_ids.push_str(&render_commitment_id(*id));
    }
    root0_ids.push(']');
    let frame_values: Vec<String> = case
        .imported_artifact
        .frames
        .iter()
        .map(render_frame_source_view)
        .collect();
    let stage3_values: Vec<String> = case
        .imported_artifact
        .stage3s
        .iter()
        .map(render_stage3_view)
        .collect();
    let (frame_defs, frame_names) = render_named_value_defs("FrameSourceView", &frame_def_prefix, &frame_values);
    let (stage3_defs, stage3_names) = render_named_value_defs("Stage3View", &stage3_def_prefix, &stage3_values);
    let (bundle_digest_defs, bundle_digest_names) =
        render_named_staged_digest_defs_from_refs(&bundle_digest_def_prefix, &frame_names, &stage3_names);
    let kernel_manifest_claim_values: Vec<String> = case
        .imported_artifact
        .artifact
        .kernel_digest
        .manifest_surface
        .kernel_manifest
        .claims
        .iter()
        .map(render_opening_claim_view)
        .collect();
    let (kernel_manifest_claim_defs, kernel_manifest_claim_names) = render_named_value_defs(
        "KernelOpeningClaimView",
        &kernel_manifest_claim_def_prefix,
        &kernel_manifest_claim_values,
    );
    let root_manifest_claim_values: Vec<String> = case
        .imported_artifact
        .artifact
        .kernel_digest
        .manifest_surface
        .root_manifest
        .claims
        .iter()
        .map(render_opening_claim_view)
        .collect();
    let (root_manifest_claim_defs, root_manifest_claim_names) = render_named_value_defs(
        "KernelOpeningClaimView",
        &root_manifest_claim_def_prefix,
        &root_manifest_claim_values,
    );
    let mut out = String::new();
    out.push_str("import Nightstream.Chip8.Generated.ReleaseArtifactVectorTypes\n\n");
    out.push_str("set_option maxHeartbeats 0\n\n");
    out.push_str("namespace Nightstream.Chip8.Generated.ReleaseArtifactVectors\n\n");
    out.push_str(&render_named_meta_pub_def(
        &release_artifact_meta_pub_def_name_from_name(case.name),
        &case.imported_artifact.artifact.staged_bundle.public.meta_pub,
    ));
    out.push('\n');
    out.push_str(&frame_defs);
    out.push_str(&render_named_list_def(&frames_def, "FrameSourceView", &frame_names));
    out.push_str(&stage3_defs);
    out.push_str(&render_named_list_def(&stage3s_def, "Stage3View", &stage3_names));
    out.push_str(&bundle_digest_defs);
    out.push_str(&render_named_list_def(
        &bundle_digests_def,
        "StagedExecutionDigestView",
        &bundle_digest_names,
    ));
    out.push_str(&kernel_manifest_claim_defs);
    out.push_str(&render_named_list_def(
        &kernel_manifest_claims_def,
        "KernelOpeningClaimView",
        &kernel_manifest_claim_names,
    ));
    out.push_str(&root_manifest_claim_defs);
    out.push_str(&render_named_list_def(
        &root_manifest_claims_def,
        "KernelOpeningClaimView",
        &root_manifest_claim_names,
    ));
    out.push_str(&format!(
        "def {bindings_def} : List CommitmentBinding :=\n  {bindings}\n\n\
def {trace_digests_def} : TraceDigestSourceView :=\n  ({trace_digests})\n\n\
def {kernel_manifest_view_def} : KernelOpeningManifestView :=\n  ({kernel_manifest_view})\n\n\
def {root_manifest_view_def} : KernelOpeningManifestView :=\n  ({root_manifest_view})\n\n\
def {manifest_surface_view_def} : KernelManifestSurfaceView :=\n  mkKernelManifestSurfaceView\n      {root0_ids}\n      {kernel_manifest_view_def}\n      {root_manifest_view_def}\n\n\
def {trace_surface_view_def} : KernelTraceSurfaceView :=\n  ({trace_surface_view})\n\n\
def {export_surface_view_def} : KernelExportSurfaceView :=\n  ({export_surface_view})\n\n\
def {audit_surface_view_def} : KernelAuditSurfaceView :=\n  ({audit_surface_view})\n\n\
def {transcript_surface_view_def} : KernelTranscriptSurfaceView :=\n  ({transcript_surface_view})\n\n\
def {error_surface_view_def} : KernelErrorSurfaceView :=\n  ({error_surface_view})\n\n\
def {kernel_digest_view_def} : KernelExecutionDigestView :=\n  mkKernelExecutionDigestView\n    {trace_surface_view_def}\n    {export_surface_view_def}\n    {audit_surface_view_def}\n    {manifest_surface_view_def}\n    {transcript_surface_view_def}\n    {error_surface_view_def}\n\n\
def {bundle_public_surface_def} : DigestPublicView :=\n  ({bundle_public_surface})\n\n\
def {bundle_view_def} : StagedExecutionDigestBundleView :=\n  ({bundle_view})\n\n\
def {artifact_view_def} : KernelReleaseArtifactView :=\n  mkKernelReleaseArtifactView\n      {kernel_digest_view_def}\n      {bundle_view_def}\n\n\
def {case_def} : KernelReleaseArtifactVectorCase :=\n  mkKernelReleaseArtifactVectorCase\n      \"{name}\"\n      {bindings_def}\n      ({trace_digests_def})\n      {frames_def}\n      {stage3s_def}\n      ({artifact_view_def})\n",
        bindings_def = bindings_def,
        bindings = render_imported_root0_bindings_list(case),
        trace_digests_def = trace_digests_def,
        trace_digests = render_trace_digest_source_view(&case.imported_artifact.trace_digests),
        kernel_manifest_view_def = kernel_manifest_view_def,
        kernel_manifest_view =
            render_opening_manifest_view_from_claims_expr(
                &kernel_manifest_claims_def,
                &case.imported_artifact.artifact.kernel_digest.manifest_surface.kernel_manifest.digest,
            ),
        root_manifest_view_def = root_manifest_view_def,
        root_manifest_view =
            render_opening_manifest_view_from_claims_expr(
                &root_manifest_claims_def,
                &case.imported_artifact.artifact.kernel_digest.manifest_surface.root_manifest.digest,
            ),
        manifest_surface_view_def = manifest_surface_view_def,
        root0_ids = root0_ids,
        trace_surface_view_def = trace_surface_view_def,
        trace_surface_view =
            render_kernel_trace_surface_view_from_frames_expr(
                &case.imported_artifact.artifact.kernel_digest.trace_surface,
                &frames_def,
            ),
        export_surface_view_def = export_surface_view_def,
        export_surface_view =
            render_kernel_export_surface_view(&case.imported_artifact.artifact.kernel_digest.export_surface),
        audit_surface_view_def = audit_surface_view_def,
        audit_surface_view =
            render_kernel_audit_surface_view(&case.imported_artifact.artifact.kernel_digest.audit_surface),
        transcript_surface_view_def = transcript_surface_view_def,
        transcript_surface_view =
            render_kernel_transcript_surface_view(&case.imported_artifact.artifact.kernel_digest.transcript_surface),
        error_surface_view_def = error_surface_view_def,
        error_surface_view =
            render_kernel_error_surface_view(&case.imported_artifact.artifact.kernel_digest.error_surface),
        kernel_digest_view_def = kernel_digest_view_def,
        bundle_public_surface_def = bundle_public_surface_def,
        bundle_public_surface = render_digest_public_view_with_meta_expr(
            &case.imported_artifact.artifact.staged_bundle.public.public,
            &release_artifact_meta_pub_def_name_from_name(case.name),
        ),
        bundle_view_def = bundle_view_def,
        bundle_view = render_bundle_view_from_parts(&bundle_public_surface_def, &bundle_digests_def),
        artifact_view_def = artifact_view_def,
        case_def = release_artifact_case_def_name_from_name(case.name),
        name = case.name,
    ));
    out.push('\n');
    out.push_str("end Nightstream.Chip8.Generated.ReleaseArtifactVectors\n");
    out
}

fn render_release_artifact_corpus_module(cases: &[ReleaseArtifactCase]) -> String {
    let mut out = String::new();
    for case in cases {
        out.push_str(&format!(
            "import Nightstream.Chip8.Generated.ReleaseArtifactVectors.{}\n",
            case_module_fragment(case.name),
        ));
    }
    out.push('\n');
    out.push_str("namespace Nightstream.Chip8.Generated\n\n");
    out.push_str("open Nightstream.Chip8.Generated.ReleaseArtifactVectors\n\n");
    out.push_str("def releaseArtifactVectorCases : List KernelReleaseArtifactVectorCase :=\n");
    out.push_str("  [\n");
    for (idx, case) in cases.iter().enumerate() {
        out.push_str("    ");
        out.push_str(&release_artifact_case_def_name_from_name(case.name));
        if idx + 1 < cases.len() {
            out.push(',');
        }
        out.push('\n');
    }
    out.push_str("  ]\n\n");
    out.push_str("end Nightstream.Chip8.Generated\n");
    out
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let fixtures = chip8_subset_fixtures();

    if !args.is_empty() {
        if args.first().map(String::as_str) == Some("--export-imported-release-artifact") {
            if args.len() > 3 {
                panic!("usage: chip8_rust_vectors --export-imported-release-artifact <fixture> [output-path]");
            }
            let fixture_name = args
                .get(1)
                .map(String::as_str)
                .unwrap_or("jump_rows_2_seed_empty");
            if !is_release_artifact_fixture(fixture_name) {
                panic!(
                    "fixture '{fixture_name}' is not in the audited release-artifact corpus; supported fixtures: jump_rows_2_seed_empty, jump_rows_3_seed_nonempty, mixed_ldimm_addimm_10"
                );
            }
            let fixture = fixture_named(&fixtures, fixture_name);
            let case = build_release_artifact_case(fixture);
            let out = render_imported_release_artifact_case_module(&case);
            let output_path = args
                .get(2)
                .map(PathBuf::from)
                .unwrap_or_else(|| imported_release_artifact_case_output_path(fixture_name));
            fs::create_dir_all(output_path.parent().expect("generated dir"))
                .expect("create generated dir");
            fs::write(&output_path, out).expect("write imported release artifact");
            println!("{}", output_path.display());
            return;
        }
        panic!(
            "unknown arguments: {:?}\nusage:\n  chip8_rust_vectors\n  chip8_rust_vectors --export-imported-release-artifact <fixture> [output-path]",
            args
        );
    }

    let transcript_cases: Vec<_> = fixtures.iter().map(build_case).collect();
    let artifact_fixtures: Vec<_> = fixtures
        .iter()
        .filter(|fixture| is_release_artifact_fixture(fixture.name))
        .collect();
    let bundle_cases: Vec<_> = artifact_fixtures.iter().map(|fixture| build_bundle_case(fixture)).collect();
    let release_artifact_cases: Vec<_> = artifact_fixtures
        .iter()
        .map(|fixture| build_release_artifact_case(fixture))
        .collect();

    let mut transcript_out = String::new();
    transcript_out.push_str("import Nightstream.Chip8.Generated.TranscriptVectorTypes\n\n");
    transcript_out.push_str("namespace Nightstream.Chip8.Generated\n\n");
    for case in &transcript_cases {
        transcript_out.push_str(&render_meta_pub_def(case));
        transcript_out.push('\n');
    }
    transcript_out.push_str("def transcriptVectorCases : List TranscriptVectorCase :=\n");
    transcript_out.push_str("  [\n");
    for (idx, case) in transcript_cases.iter().enumerate() {
        transcript_out.push_str("    ");
        transcript_out.push_str(&render_case(case));
        if idx + 1 < transcript_cases.len() {
            transcript_out.push(',');
        }
        transcript_out.push('\n');
    }
    transcript_out.push_str("  ]\n\n");
    transcript_out.push_str("end Nightstream.Chip8.Generated\n");

    let bundle_out = render_bundle_corpus_module(&bundle_cases);
    let release_artifact_out = render_release_artifact_corpus_module(&release_artifact_cases);
    let imported_opening_transcript_out =
        render_imported_opening_transcript_corpus_module(&release_artifact_cases);
    let imported_release_artifact_out = render_imported_release_artifact_corpus_module(&release_artifact_cases);

    let transcript_path = transcript_output_path();
    fs::create_dir_all(transcript_path.parent().expect("generated dir")).expect("create generated dir");
    fs::write(&transcript_path, transcript_out).expect("write transcript vectors");
    remove_generated_path_if_exists(&bundle_legacy_output_path());
    remove_generated_path_if_exists(&release_artifact_legacy_output_path());
    remove_generated_path_if_exists(&imported_release_artifact_legacy_output_path());

    let bundle_dir = bundle_output_dir();
    reset_generated_module_dir(&bundle_dir);
    for case in &bundle_cases {
        fs::write(bundle_case_output_path(case.name), render_bundle_case_module(case))
            .expect("write bundle case module");
    }
    let bundle_path = bundle_corpus_output_path();
    fs::write(&bundle_path, bundle_out).expect("write bundle corpus");

    let release_artifact_dir = release_artifact_output_dir();
    reset_generated_module_dir(&release_artifact_dir);
    for case in &release_artifact_cases {
        fs::write(
            release_artifact_case_output_path(case.name),
            render_release_artifact_case_module(case),
        )
        .expect("write release artifact case module");
    }
    let release_artifact_path = release_artifact_corpus_output_path();
    fs::write(&release_artifact_path, release_artifact_out).expect("write release artifact corpus");

    let imported_opening_transcript_path = imported_opening_transcript_output_path();
    fs::create_dir_all(imported_opening_transcript_path.parent().expect("generated dir"))
        .expect("create generated dir");
    fs::write(&imported_opening_transcript_path, imported_opening_transcript_out)
        .expect("write imported opening transcript cases");
    let imported_release_artifact_dir = imported_release_artifact_output_dir();
    reset_generated_module_dir(&imported_release_artifact_dir);
    for case in &release_artifact_cases {
        fs::write(
            imported_release_artifact_case_output_path(case.name),
            render_imported_release_artifact_case_module(case),
        )
        .expect("write imported release artifact case module");
    }
    let imported_release_artifact_path = imported_release_artifact_corpus_output_path();
    fs::write(&imported_release_artifact_path, imported_release_artifact_out)
        .expect("write imported release artifact corpus");
    println!("{}", transcript_path.display());
    println!("{}", bundle_path.display());
    println!("{}", release_artifact_path.display());
    println!("{}", imported_opening_transcript_path.display());
    println!("{}", imported_release_artifact_path.display());
}
