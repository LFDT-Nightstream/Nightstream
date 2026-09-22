use super::*;

const ROM_ADDRESS: usize = 0;
const ROM_VALUE: usize = 1;
const RAM_ADDRESS: usize = 2;
const RAM_VALUE_AFTER: usize = 3;
const ROM_ACTIVE: usize = 4;
const RAM_ACTIVE: usize = 5;
const RAM_VALUE_BEFORE: usize = 6;
const APPLICATION_COLUMNS: usize = 7;

#[derive(Clone, Copy)]
struct ApplicationRow {
    rom_address: u64,
    rom_value: u64,
    ram_address: u64,
    ram_value_before: u64,
    ram_value_after: u64,
    rom_active: bool,
    ram_active: bool,
}

impl ApplicationRow {
    fn encode(self) -> [F; APPLICATION_COLUMNS] {
        let mut assignment = [F::ZERO; APPLICATION_COLUMNS];
        assignment[ROM_ADDRESS] = F::from_u64(self.rom_address);
        assignment[ROM_VALUE] = F::from_u64(self.rom_value);
        assignment[RAM_ADDRESS] = F::from_u64(self.ram_address);
        assignment[RAM_VALUE_BEFORE] = F::from_u64(self.ram_value_before);
        assignment[RAM_VALUE_AFTER] = F::from_u64(self.ram_value_after);
        assignment[ROM_ACTIVE] = F::from_u64(u64::from(self.rom_active));
        assignment[RAM_ACTIVE] = F::from_u64(u64::from(self.ram_active));
        assignment
    }
}

#[derive(Clone, Copy)]
struct PhysicalSlot {
    is_padding: bool,
    is_write: bool,
    is_ram: bool,
    address: u64,
    value_read: u64,
    value_write: u64,
}

impl PhysicalSlot {
    fn encode(self, circuit: &SMemCircuit) -> Vec<F> {
        let mut assignment = vec![F::ZERO; circuit.cols()];
        assignment[0] = F::ONE;
        let start = circuit.op_slot_column(0);
        assignment[start] = F::from_u64(u64::from(self.is_padding));
        assignment[start + 1] = F::from_u64(u64::from(self.is_write));
        assignment[start + 2] = F::from_u64(u64::from(self.is_ram));
        let address_start = start + 3 + circuit.params().num_stacks;
        write_bits(
            &mut assignment,
            address_start,
            circuit.params().addr_bits(),
            self.address,
        );
        let value_read_start = address_start + circuit.params().addr_bits();
        write_bits(&mut assignment, value_read_start, VAL_BITS, self.value_read);
        write_bits(&mut assignment, value_read_start + VAL_BITS, VAL_BITS, self.value_write);
        assignment
    }
}

fn test_layout() -> MemoryPortLayout {
    MemoryPortLayout::new(
        vec![
            MemoryRegion::new("rom", MemoryRegionKind::Rom, 0, vec![2]).expect("ROM region"),
            MemoryRegion::new("ram", MemoryRegionKind::Ram, 0, vec![2]).expect("RAM region"),
        ],
        vec![MemoryOpSlot::new(vec![
            MemoryPort::new(
                0,
                vec![ROM_ADDRESS],
                ROM_VALUE,
                MemoryPortKind::Read,
                MemoryPortActivation::Column(ROM_ACTIVE),
            ),
            MemoryPort::new(
                1,
                vec![RAM_ADDRESS],
                RAM_VALUE_AFTER,
                MemoryPortKind::Write {
                    value_before_column: Some(RAM_VALUE_BEFORE),
                },
                MemoryPortActivation::Column(RAM_ACTIVE),
            ),
        ])],
    )
    .expect("multiplexed layout")
}

fn write_bits(assignment: &mut [F], start: usize, width: usize, value: u64) {
    for bit in 0..width {
        assignment[start + bit] = F::from_u64((value >> bit) & 1);
    }
}

fn binding_is_satisfied(
    circuit: &SMemCircuit,
    layout: &MemoryPortLayout,
    application_row: ApplicationRow,
    physical_slot: PhysicalSlot,
) -> bool {
    let app_assignment = application_row.encode();
    let slot_assignment = physical_slot.encode(circuit);
    let mut builder = R1csBuilder::new();
    let s_mem = builder.alloc_vec(&slot_assignment);
    let app_vars = builder.alloc_vec(&app_assignment);
    enforce_memory_ports(&mut builder, circuit, &s_mem, &app_assignment, &app_vars, layout)
        .expect("memory-port constraints");
    builder.is_satisfied()
}

#[test]
fn multiplexed_slot_constraints_bind_the_selected_candidate() {
    let layout = test_layout();
    let circuit = SMemCircuit::new(NebulaParams::new(2, 2, 1, 1, 4).expect("test geometry"));

    let inactive = ApplicationRow {
        rom_address: 1,
        rom_value: 7,
        ram_address: 2,
        ram_value_before: 3,
        ram_value_after: 9,
        rom_active: false,
        ram_active: false,
    };
    let rom_read = ApplicationRow {
        rom_active: true,
        ..inactive
    };
    let rom_slot = PhysicalSlot {
        is_padding: false,
        is_write: false,
        is_ram: false,
        address: 1,
        value_read: 7,
        value_write: 7,
    };
    assert!(binding_is_satisfied(&circuit, &layout, rom_read, rom_slot));

    let ram_write = ApplicationRow {
        ram_active: true,
        ..inactive
    };
    let ram_write_slot = PhysicalSlot {
        is_padding: false,
        is_write: true,
        is_ram: true,
        address: 2,
        value_read: 3,
        value_write: 9,
    };
    assert!(binding_is_satisfied(&circuit, &layout, ram_write, ram_write_slot));
    assert!(!binding_is_satisfied(
        &circuit,
        &layout,
        ram_write,
        PhysicalSlot {
            is_write: false,
            ..ram_write_slot
        },
    ));
    assert!(!binding_is_satisfied(
        &circuit,
        &layout,
        ram_write,
        PhysicalSlot {
            is_ram: false,
            ..ram_write_slot
        },
    ));

    let padding = PhysicalSlot {
        is_padding: true,
        is_write: false,
        is_ram: false,
        address: 0,
        value_read: 0,
        value_write: 0,
    };
    assert!(binding_is_satisfied(&circuit, &layout, inactive, padding));

    let collision = ApplicationRow {
        rom_active: true,
        ram_active: true,
        ..inactive
    };
    assert!(!binding_is_satisfied(&circuit, &layout, collision, rom_slot));
}

fn profile_application(anchor: [u8; 32]) -> NebulaApplication {
    use crate::frontends::direct_ccs::R1cs;
    use crate::frontends::f_prime::recursive_plan::{RecursiveStepImagePlan, StateXOutPlanOptions};
    use crate::paper::f_prime::ring_action_trace::{LowNormEncoding, RingActionTraceLayout};
    use neo_ccs::Mat;
    use neo_math::D;

    let shape = R1cs {
        a: Mat::zero(1, D, F::ZERO),
        b: Mat::zero(1, D, F::ZERO),
        c: Mat::zero(1, D, F::ZERO),
        m_in: 1,
    };
    let plan = RecursiveStepImagePlan {
        limbs: D * 64 + 1,
        app_private_var_widths: Vec::new(),
        boundary_bits: 4 * 64,
        kmul_count: 0,
        ring_action_pair_count: 0,
        projection_batches: Vec::new(),
        ring_action_pair_layout: RingActionTraceLayout::new(
            LowNormEncoding::U64,
            LowNormEncoding::U64,
            LowNormEncoding::U64,
            LowNormEncoding::U64,
        ),
        sponge_transcript_permutes: 0,
        nifs_payload_shapes: Vec::new(),
        accumulator: None,
        state_x_out: Some(StateXOutPlanOptions {
            pc: 1,
            public_x_out_lane_bit_starts: [0, 64, 128, 192],
            app_public_input_var_indices: vec![0],
            app_public_input_bit_var_indices: Vec::new(),
            semantic_state_in_var_indices: vec![0],
            semantic_state_out_var_indices: vec![0],
            initial_semantic_state_digest_anchor: Some(anchor),
        }),
    };
    NebulaApplication::new(shape, plan, test_layout()).expect("profile application")
}

#[test]
fn prepared_application_profile_ignores_only_the_program_anchor() {
    let reference = profile_application([1; 32]);
    let rebound = reference
        .bind_program_profile([2; 32], test_layout())
        .expect("program-bound application");
    assert!(reference.same_relation_profile_as(&rebound));

    let mut changed_plan = rebound.clone();
    changed_plan
        .recursive_plan
        .state_x_out
        .as_mut()
        .expect("state binding")
        .pc += 1;
    assert!(!reference.same_relation_profile_as(&changed_plan));

    let mut changed_relation = rebound;
    let R1csShape::Dense(shape) = &mut changed_relation.shape else {
        panic!("test application uses a dense relation");
    };
    shape.a[(0, 0)] = F::ONE;
    assert!(!reference.same_relation_profile_as(&changed_relation));
}
