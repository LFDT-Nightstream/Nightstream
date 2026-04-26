use neo_ajtai::Commitment;
use neo_ccs::traits::SModuleHomomorphism;
use neo_ccs::Mat;
use neo_fold_next::chip8::spec::{
    Chip8Program, Chip8State, Chip8VmSpec, COL_BURST_LAST, COL_NNN_ADDR, COL_NNN_WORD, COL_RAM_ADDR, COL_REG_X,
    COL_REG_X_NEXT, COL_X_IDX,
};
use neo_fold_next::chip8::tables::{RAM_SINK_ADDR, REG_SINK_ADDR};
use neo_fold_next::chip8::trace::{build_row_extension_trace, Chip8TraceBuilder};
use neo_fold_next::step_build::StepBuild;
use neo_math::F;
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

fn row(step: &StepBuild) -> Vec<F> {
    let mut row = step.prepared.mcs.x.clone();
    row.extend_from_slice(&step.prepared.witness.w);
    row
}

#[test]
fn jump_rows_use_dummy_x_lane_and_attach_kernel_aux() {
    let program = Chip8Program::from_opcodes(&[
        0x6007, // LD V0, 0x07
        0x1200, // Jump 0x200
    ]);
    let initial_state = Chip8State::with_program(&program).expect("initial state");
    let steps = build_steps(&program, &initial_state, 2);
    let jump = &steps[1];
    let jump_row = row(jump);
    let aux = jump
        .kernel_aux
        .as_ref()
        .expect("kernel aux should be attached");

    assert_eq!(jump_row[COL_X_IDX], F::ZERO);
    assert_eq!(jump_row[COL_REG_X], F::from_u64(7));
    assert_eq!(jump_row[COL_REG_X_NEXT], F::from_u64(7));
    assert_eq!(aux.reg_ra_x_addr, 0);
    assert_eq!(aux.reg_wa_addr, REG_SINK_ADDR);
    assert_eq!(aux.reg_inc, F::ZERO);
}

#[test]
fn ld_i_routes_through_i_slot_and_uses_delta_inc() {
    let program = Chip8Program::from_opcodes(&[0xA345]);
    let mut initial_state = Chip8State::with_program(&program).expect("initial state");
    initial_state.i = 0x0123;
    initial_state.v[0] = 0x22;
    initial_state.v[3] = 0x99;

    let steps = build_steps(&program, &initial_state, 1);
    let ld_i = &steps[0];
    let ld_i_row = row(ld_i);
    let aux = ld_i
        .kernel_aux
        .as_ref()
        .expect("kernel aux should be attached");

    assert_eq!(ld_i_row[COL_X_IDX], F::ZERO);
    assert_eq!(ld_i_row[COL_REG_X], F::from_u64(0x22));
    assert_eq!(ld_i_row[COL_REG_X_NEXT], F::from_u64(0x22));
    assert_eq!(aux.reg_ra_x_addr, 0);
    assert_eq!(aux.reg_wa_addr, 16);
    assert_eq!(aux.reg_inc, F::from_u64((0x0345 - 0x0123) as u64));
}

#[test]
fn lookup_writes_record_full_register_delta_not_activity_flag() {
    let program = Chip8Program::from_opcodes(&[0x637b]);
    let steps = build_steps(&program, &Chip8State::with_program(&program).expect("initial state"), 1);
    let aux = steps[0]
        .kernel_aux
        .as_ref()
        .expect("kernel aux should be attached");

    assert_eq!(aux.reg_wa_addr, 3);
    assert_eq!(aux.reg_inc, F::from_u64(0x7b));
}

#[test]
fn non_nnn_rows_zero_nnn_lane_columns() {
    let program = Chip8Program::from_opcodes(&[0x637b]);
    let steps = build_steps(&program, &Chip8State::with_program(&program).expect("initial state"), 1);
    let witness_row = row(&steps[0]);

    assert_eq!(witness_row[COL_NNN_ADDR], F::ZERO);
    assert_eq!(witness_row[COL_NNN_WORD], F::ZERO);
}

#[test]
fn store_regs_burst_rows_write_ram_with_value_deltas() {
    let program = Chip8Program::from_opcodes(&[
        0x600a, // LD V0, 10
        0x610b, // LD V1, 11
        0xA300, // LD I, 0x300
        0xF155, // StoreRegs V0..V1
    ]);
    let steps = build_steps(&program, &Chip8State::with_program(&program).expect("initial state"), 4);

    let first_store = &steps[3];
    let second_store = &steps[4];
    let first_aux = first_store
        .kernel_aux
        .as_ref()
        .expect("kernel aux should be attached");
    let second_aux = second_store
        .kernel_aux
        .as_ref()
        .expect("kernel aux should be attached");
    let first_row = row(first_store);
    let second_row = row(second_store);

    assert_eq!(first_aux.reg_wa_addr, REG_SINK_ADDR);
    assert_eq!(first_aux.reg_inc, F::ZERO);
    assert_eq!(first_aux.ram_ra_addr, RAM_SINK_ADDR);
    assert_eq!(first_aux.ram_wa_addr, 0x300);
    assert_eq!(first_aux.ram_inc, F::from_u64(10));
    assert_eq!(first_row[COL_RAM_ADDR], F::from_u64(0x300));
    assert_eq!(first_row[COL_BURST_LAST], F::ZERO);

    assert_eq!(second_aux.reg_wa_addr, REG_SINK_ADDR);
    assert_eq!(second_aux.reg_inc, F::ZERO);
    assert_eq!(second_aux.ram_ra_addr, RAM_SINK_ADDR);
    assert_eq!(second_aux.ram_wa_addr, 0x301);
    assert_eq!(second_aux.ram_inc, F::from_u64(11));
    assert_eq!(second_row[COL_RAM_ADDR], F::from_u64(0x301));
    assert_eq!(second_row[COL_BURST_LAST], F::ONE);
}

#[test]
fn load_regs_burst_rows_write_registers_with_value_deltas() {
    let program = Chip8Program::from_opcodes(&[
        0xA300, // LD I, 0x300
        0xF165, // LoadRegs V0..V1
    ]);
    let mut initial_state = Chip8State::with_program(&program).expect("initial state");
    initial_state.memory[0x300] = 0xaa;
    initial_state.memory[0x301] = 0xbb;
    initial_state.v[0] = 1;
    initial_state.v[1] = 2;

    let steps = build_steps(&program, &initial_state, 2);
    let first_load = &steps[1];
    let second_load = &steps[2];
    let first_aux = first_load
        .kernel_aux
        .as_ref()
        .expect("kernel aux should be attached");
    let second_aux = second_load
        .kernel_aux
        .as_ref()
        .expect("kernel aux should be attached");

    assert_eq!(first_aux.ram_ra_addr, 0x300);
    assert_eq!(first_aux.ram_wa_addr, RAM_SINK_ADDR);
    assert_eq!(first_aux.ram_inc, F::ZERO);
    assert_eq!(first_aux.reg_wa_addr, 0);
    assert_eq!(first_aux.reg_inc, F::from_u64(0xaa - 1));

    assert_eq!(second_aux.ram_ra_addr, 0x301);
    assert_eq!(second_aux.ram_wa_addr, RAM_SINK_ADDR);
    assert_eq!(second_aux.ram_inc, F::ZERO);
    assert_eq!(second_aux.reg_wa_addr, 1);
    assert_eq!(second_aux.reg_inc, F::from_u64(0xbb - 2));
}

#[test]
fn execute_program_captures_row_traces_used_by_extension_projection() {
    let program = Chip8Program::from_opcodes(&[
        0x600a, // LD V0, 10
        0x610b, // LD V1, 11
        0xA300, // LD I, 0x300
        0xF155, // StoreRegs V0..V1
    ]);
    let initial_state = Chip8State::with_program(&program).expect("initial state");
    let steps = Chip8TraceBuilder::<()>::execute_program(&program, &initial_state, 4).expect("execution");
    let store_step = &steps[3];

    assert_eq!(store_step.row_traces.len(), 2);
    assert_eq!(store_step.row_traces[0].kernel_aux.ram_wa_addr, 0x300);
    assert_eq!(store_step.row_traces[1].kernel_aux.ram_wa_addr, 0x301);
    assert_eq!(store_step.row_traces[0].row[COL_RAM_ADDR], F::from_u64(0x300));
    assert_eq!(store_step.row_traces[1].row[COL_RAM_ADDR], F::from_u64(0x301));

    let extension_rows = build_row_extension_trace(store_step);
    assert_eq!(extension_rows.len(), 2);
    assert_eq!(extension_rows[0].ram_writes[0].addr, 0x300);
    assert_eq!(extension_rows[0].ram_writes[0].value, 10);
    assert_eq!(extension_rows[1].ram_writes[0].addr, 0x301);
    assert_eq!(extension_rows[1].ram_writes[0].value, 11);
}
