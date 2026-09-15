use super::{build_memory_backend, WasmNebulaProfile};
use crate::{extract_wasm_program_artifacts, WasmMemoryId, RANGE_CHECKED_WITNESS_WIDTH};
use neo_fold_clean::frontends::nebula::layout::NebulaParams;

#[test]
fn dense_pc_program_fits_128_rom_cells() {
    let wasm = wat::parse_str(r#"(module (func (export "main") (result i32) i32.const 7))"#).unwrap();
    let artifacts = extract_wasm_program_artifacts(&wasm).unwrap();
    let rom_address_bits = 7; // 128 cells
    let ram_address_bits = 10; // 1,024 cells
    let memory_op_slots_per_step = 64;
    let scan_cells_per_step = 64;
    let max_segments = 16;
    let geometry = NebulaParams::new(
        rom_address_bits,
        ram_address_bits,
        memory_op_slots_per_step,
        scan_cells_per_step,
        max_segments,
    )
    .unwrap();
    let profile = WasmNebulaProfile::test_profile_with_geometry(geometry);
    let backend = build_memory_backend(&artifacts, None, &profile, RANGE_CHECKED_WITNESS_WIDTH)
        .expect("two-operator program must fit a 128-cell ROM");
    let opcode_region = backend
        .layout
        .regions()
        .iter()
        .find(|region| region.name() == WasmMemoryId::ProgramOpcode.name())
        .unwrap();
    assert_eq!(opcode_region.cells(), 2);
    assert_eq!(backend.rom_image.len(), 128);
}
