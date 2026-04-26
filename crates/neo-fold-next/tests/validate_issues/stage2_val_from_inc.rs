use neo_ajtai::Commitment;
use neo_ccs::traits::SModuleHomomorphism;
use neo_ccs::Mat;
use neo_fold_next::chip8::kernel::KernelStepAux;
use neo_fold_next::chip8::spec::{build_pad_row, Chip8Program, Chip8State, Chip8VmSpec, WITNESS_WIDTH};
use neo_fold_next::chip8::stage2::{prove_stage2, verify_stage2};
use neo_fold_next::chip8::tables::{flatten_alu_key, flatten_eq4_key, LookupKind, RAM_SINK_ADDR, REG_SINK_ADDR};
use neo_fold_next::chip8::trace::Chip8TraceBuilder;
use neo_fold_next::step_build::StepBuild;
use neo_math::{F, K};
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::PrimeCharacteristicRing;

struct ToyModule;

impl SModuleHomomorphism<F, Commitment> for ToyModule {
    fn commit(&self, z: &Mat<F>) -> Commitment {
        let mut out = Commitment::zeros(z.rows(), 1);
        for r in 0..z.rows() {
            let mut acc = F::ZERO;
            for c in 0..z.cols() {
                acc += z[(r, c)];
            }
            out.data[r] = acc;
        }
        out
    }
}

fn build_steps(program: &Chip8Program, initial_state: &Chip8State, step_count: usize) -> Vec<StepBuild> {
    let vm = Chip8VmSpec::new().expect("vm");
    Chip8TraceBuilder::new(&ToyModule)
        .build_program(&vm, program, initial_state, step_count)
        .expect("build steps")
}

fn pad_opcode(pad_pc_word: u16) -> u16 {
    0x1000 | (2 * pad_pc_word)
}

fn build_pad_aux(pad_pc_word: u16) -> KernelStepAux {
    KernelStepAux {
        fetch_addr: pad_pc_word as usize,
        decode_addr: pad_opcode(pad_pc_word),
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

fn build_padded_stage2_witness(
    program: &Chip8Program,
    initial_state: &Chip8State,
    step_count: usize,
) -> (Vec<[F; WITNESS_WIDTH]>, Vec<KernelStepAux>, usize) {
    let steps = build_steps(program, initial_state, step_count);
    let mut trace_rows = Vec::with_capacity(steps.len());
    let mut aux_data = Vec::with_capacity(steps.len());
    for step in steps {
        let mut row = step.prepared.mcs.x.clone();
        row.extend_from_slice(&step.prepared.witness.w);
        assert_eq!(row.len(), WITNESS_WIDTH);
        let mut arr = [F::ZERO; WITNESS_WIDTH];
        arr.copy_from_slice(&row);
        trace_rows.push(arr);
        aux_data.push(step.kernel_aux.expect("kernel aux"));
    }

    let padded_len = trace_rows.len().next_power_of_two();
    let pad_pc_word = program.start_pc / 2 + (program.bytes.len() / 2) as u16;
    let pad_row = build_pad_row(pad_pc_word);
    let pad_aux = build_pad_aux(pad_pc_word);
    while trace_rows.len() < padded_len {
        trace_rows.push(pad_row);
        aux_data.push(pad_aux.clone());
    }

    (trace_rows, aux_data, padded_len.trailing_zeros() as usize)
}

fn eq_table(point_le: &[K]) -> Vec<K> {
    let ell = point_le.len();
    let n = 1usize << ell;
    let mut out = vec![K::ONE; n];
    for (i, &ri) in point_le.iter().enumerate() {
        let stride = 1usize << i;
        let block = 1usize << (ell - i - 1);
        let one_minus = K::ONE - ri;
        let mut idx = 0usize;
        for _ in 0..block {
            for j in 0..stride {
                out[idx + j] *= one_minus;
            }
            for j in 0..stride {
                out[idx + stride + j] *= ri;
            }
            idx += 2 * stride;
        }
    }
    out
}

fn mle_eval_f_be(values: &[F], point_be: &[K], domain_size: usize) -> K {
    let point_le: Vec<K> = point_be.iter().rev().copied().collect();
    let eq = eq_table(&point_le);
    assert_eq!(eq.len(), domain_size);
    values
        .iter()
        .copied()
        .chain(std::iter::repeat(F::ZERO))
        .take(domain_size)
        .zip(eq.iter())
        .fold(K::ZERO, |acc, (value, &weight)| acc + K::from(value) * weight)
}

fn build_register_write_program() -> (Chip8Program, Chip8State) {
    let program = Chip8Program::from_opcodes(&[0x637b]);
    let mut initial_state = Chip8State::with_program(&program).expect("initial state");
    initial_state.i = 0x0240;
    initial_state.v[0] = 1;
    initial_state.v[1] = 2;
    initial_state.v[2] = 3;
    initial_state.v[3] = 4;
    (program, initial_state)
}

fn build_quiescent_ram_program() -> (Chip8Program, Chip8State) {
    let program = Chip8Program::from_opcodes(&[0x1200]);
    let mut initial_state = Chip8State::with_program(&program).expect("initial state");
    initial_state.i = 0x0240;
    initial_state.v[0] = 1;
    initial_state.memory[0x300] = 0x05;
    initial_state.memory[0x301] = 0x06;
    initial_state.memory[0x302] = 0x07;
    (program, initial_state)
}

fn build_store_burst_program() -> (Chip8Program, Chip8State) {
    let program = Chip8Program::from_opcodes(&[
        0x600a, // LD V0, 10
        0xA300, // LD I, 0x300
        0xF055, // StoreRegs V0
        0x1200, // Jump 0x200
    ]);
    let mut initial_state = Chip8State::with_program(&program).expect("initial state");
    initial_state.i = 0x0240;
    initial_state.v[0] = 1;
    initial_state.memory[0x300] = 0x05;
    (program, initial_state)
}

#[test]
fn stage2_val_from_inc_claims_are_anchored_to_selected_point_and_initial_state() {
    let (reg_program, reg_initial_state) = build_register_write_program();
    let (reg_trace_rows, reg_aux_data, reg_cycle_bits) =
        build_padded_stage2_witness(&reg_program, &reg_initial_state, 1);

    let mut transcript = Poseidon2Transcript::new(b"neo.fold.next/tests/stage2_val_from_inc");
    let reg_proof = prove_stage2(
        &reg_trace_rows,
        &reg_aux_data,
        &reg_initial_state.v,
        reg_initial_state.i,
        &reg_initial_state.memory,
        reg_cycle_bits,
        &mut transcript,
    )
    .expect("stage2 proof");

    let mut initial_reg_domain = vec![F::ZERO; 32];
    for idx in 0..16 {
        initial_reg_domain[idx] = F::from_u64(reg_initial_state.v[idx] as u64);
    }
    initial_reg_domain[16] = F::from_u64(reg_initial_state.i as u64);
    let reg_init_at_point = mle_eval_f_be(&initial_reg_domain, &reg_proof.reg_addr_point, 32);
    assert_eq!(
        reg_proof.reg_val_at_point - reg_init_at_point,
        reg_proof.reg_val_from_inc_claim
    );
    assert_ne!(reg_init_at_point, K::ZERO);
    assert_ne!(reg_proof.reg_val_at_point, reg_proof.reg_val_from_inc_claim);

    let (ram_program, ram_initial_state) = build_quiescent_ram_program();
    let (ram_trace_rows, ram_aux_data, ram_cycle_bits) =
        build_padded_stage2_witness(&ram_program, &ram_initial_state, 1);
    let mut transcript = Poseidon2Transcript::new(b"neo.fold.next/tests/stage2_ram_val_from_inc");
    let ram_proof = prove_stage2(
        &ram_trace_rows,
        &ram_aux_data,
        &ram_initial_state.v,
        ram_initial_state.i,
        &ram_initial_state.memory,
        ram_cycle_bits,
        &mut transcript,
    )
    .expect("stage2 proof");

    let initial_ram_domain: Vec<F> = ram_initial_state
        .memory
        .iter()
        .map(|&value| F::from_u64(value as u64))
        .collect();
    let ram_init_at_point = mle_eval_f_be(&initial_ram_domain, &ram_proof.ram_addr_point, 1usize << 13);
    assert_eq!(
        ram_proof.ram_val_at_point - ram_init_at_point,
        ram_proof.ram_val_from_inc_claim
    );
    assert_ne!(ram_init_at_point, K::ZERO);
    assert_eq!(ram_proof.ram_val_from_inc_claim, K::ZERO);
    assert_eq!(ram_proof.ram_val_at_point, ram_init_at_point);
}

#[test]
fn stage2_verifier_rejects_tampered_val_from_inc_anchor_claims() {
    let (reg_program, reg_initial_state) = build_register_write_program();
    let (reg_trace_rows, reg_aux_data, reg_cycle_bits) =
        build_padded_stage2_witness(&reg_program, &reg_initial_state, 1);

    let mut prove_transcript = Poseidon2Transcript::new(b"neo.fold.next/tests/stage2_val_tamper");
    let reg_proof = prove_stage2(
        &reg_trace_rows,
        &reg_aux_data,
        &reg_initial_state.v,
        reg_initial_state.i,
        &reg_initial_state.memory,
        reg_cycle_bits,
        &mut prove_transcript,
    )
    .expect("stage2 proof");

    let mut reg_tampered = reg_proof;
    reg_tampered.reg_val_at_point += K::ONE;
    let mut verify_transcript = Poseidon2Transcript::new(b"neo.fold.next/tests/stage2_val_tamper");
    let err = verify_stage2(
        &reg_tampered,
        &reg_initial_state.v,
        reg_initial_state.i,
        &reg_initial_state.memory,
        reg_cycle_bits,
        &mut verify_transcript,
    )
    .err()
    .expect("tampered register anchor must fail");
    assert!(format!("{err}").contains("stage2 register val-from-inc anchor"));

    let (ram_program, ram_initial_state) = build_quiescent_ram_program();
    let (ram_trace_rows, ram_aux_data, ram_cycle_bits) =
        build_padded_stage2_witness(&ram_program, &ram_initial_state, 1);
    let mut prove_transcript = Poseidon2Transcript::new(b"neo.fold.next/tests/stage2_ram_val_tamper");
    let ram_proof = prove_stage2(
        &ram_trace_rows,
        &ram_aux_data,
        &ram_initial_state.v,
        ram_initial_state.i,
        &ram_initial_state.memory,
        ram_cycle_bits,
        &mut prove_transcript,
    )
    .expect("stage2 proof");

    let mut ram_tampered = ram_proof;
    ram_tampered.ram_val_at_point += K::ONE;
    let mut verify_transcript = Poseidon2Transcript::new(b"neo.fold.next/tests/stage2_ram_val_tamper");
    let err = verify_stage2(
        &ram_tampered,
        &ram_initial_state.v,
        ram_initial_state.i,
        &ram_initial_state.memory,
        ram_cycle_bits,
        &mut verify_transcript,
    )
    .err()
    .expect("tampered RAM anchor must fail");
    assert!(format!("{err}").contains("stage2 RAM val-from-inc anchor"));
}

#[test]
fn stage2_verifier_rejects_tampered_raw_sink_address_proofs() {
    let (program, initial_state) = build_quiescent_ram_program();
    let (trace_rows, aux_data, cycle_bits) = build_padded_stage2_witness(&program, &initial_state, 1);

    let mut prove_transcript = Poseidon2Transcript::new(b"neo.fold.next/tests/stage2_raw_reg_sink");
    let proof = prove_stage2(
        &trace_rows,
        &aux_data,
        &initial_state.v,
        initial_state.i,
        &initial_state.memory,
        cycle_bits,
        &mut prove_transcript,
    )
    .expect("stage2 proof");

    let mut reg_tampered = proof;
    reg_tampered.reg_addr_correctness[3].raw_address_rounds[0][0] += K::ONE;
    let mut verify_transcript = Poseidon2Transcript::new(b"neo.fold.next/tests/stage2_raw_reg_sink");
    let err = verify_stage2(
        &reg_tampered,
        &initial_state.v,
        initial_state.i,
        &initial_state.memory,
        cycle_bits,
        &mut verify_transcript,
    )
    .err()
    .expect("tampered raw register sink proof must fail");
    assert!(format!("{err}").contains("raw address"));

    let mut prove_transcript = Poseidon2Transcript::new(b"neo.fold.next/tests/stage2_raw_ram_sink");
    let proof = prove_stage2(
        &trace_rows,
        &aux_data,
        &initial_state.v,
        initial_state.i,
        &initial_state.memory,
        cycle_bits,
        &mut prove_transcript,
    )
    .expect("stage2 proof");

    let mut ram_tampered = proof;
    ram_tampered.ram_addr_correctness[0].raw_address_rounds[0][0] += K::ONE;
    let mut verify_transcript = Poseidon2Transcript::new(b"neo.fold.next/tests/stage2_raw_ram_sink");
    let err = verify_stage2(
        &ram_tampered,
        &initial_state.v,
        initial_state.i,
        &initial_state.memory,
        cycle_bits,
        &mut verify_transcript,
    )
    .err()
    .expect("tampered raw RAM sink proof must fail");
    assert!(format!("{err}").contains("raw address"));
}

#[test]
fn stage2_accepts_mixed_store_burst_trace() {
    let (program, initial_state) = build_store_burst_program();
    let (trace_rows, aux_data, cycle_bits) = build_padded_stage2_witness(&program, &initial_state, 4);

    let mut prove_transcript = Poseidon2Transcript::new(b"neo.fold.next/tests/stage2_store_burst");
    let proof = prove_stage2(
        &trace_rows,
        &aux_data,
        &initial_state.v,
        initial_state.i,
        &initial_state.memory,
        cycle_bits,
        &mut prove_transcript,
    )
    .expect("mixed store-burst stage2 proof");

    let mut verify_transcript = Poseidon2Transcript::new(b"neo.fold.next/tests/stage2_store_burst");
    verify_stage2(
        &proof,
        &initial_state.v,
        initial_state.i,
        &initial_state.memory,
        cycle_bits,
        &mut verify_transcript,
    )
    .expect("mixed store-burst stage2 verify");
}
