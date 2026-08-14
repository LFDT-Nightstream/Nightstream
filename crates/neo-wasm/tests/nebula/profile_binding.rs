//! Profile-level ROM layout and program-binding checks.

use super::{build_memory_backend, rom_component_bits, WasmNebulaProfile, WasmNebulaRomLimits};
use neo_fold_clean::frontends::nebula::plan::NebulaPlan;

const SHORT_PROGRAM: &str = r#"
(module
  (func (export "main") (result i32)
    i32.const 7))
"#;

const LONG_PROGRAM: &str = r#"
(module
  (func (export "main") (result i32)
    i32.const 9))
"#;

fn backend(source: &str, profile: &WasmNebulaProfile) -> super::MemoryBackend {
    let wasm = wat::parse_str(source).expect("valid test WASM");
    let artifacts = crate::extract_wasm_program_artifacts(&wasm).expect("program artifacts");
    build_memory_backend(&artifacts, &[], None, profile, crate::RANGE_CHECKED_WITNESS_WIDTH)
        .expect("profile memory backend")
}

#[test]
fn one_profile_fixes_rom_layout_but_not_program_contents() {
    let profile = WasmNebulaProfile::test_profile();
    let short_addresses = vec![(vec![1], 0)];
    let long_addresses = vec![(vec![1], 0), (vec![31], 0)];
    assert_eq!(
        rom_component_bits(
            "program_opcodes",
            1,
            WasmNebulaRomLimits::test_profile(),
            false,
            Some(&short_addresses),
        )
        .expect("short ROM geometry"),
        rom_component_bits(
            "program_opcodes",
            1,
            WasmNebulaRomLimits::test_profile(),
            false,
            Some(&long_addresses),
        )
        .expect("long ROM geometry"),
    );
    let short = backend(SHORT_PROGRAM, &profile);
    let long = backend(LONG_PROGRAM, &profile);

    assert_eq!(
        short.layout, long.layout,
        "program length must not change profile geometry"
    );
    assert_ne!(
        short.rom_image, long.rom_image,
        "the profile must retain exact ROM contents"
    );
    assert_eq!(short.layout.logical_port_count(), 76 * profile.batch_size());
    assert_eq!(short.layout.slot_count(), 21 * profile.batch_size());

    let short_plan = NebulaPlan::new_with_initial_ram(
        *profile.memory(),
        short.rom_image,
        short.ram_image,
        super::WASM_NEBULA_PLAN_SEED,
        1,
    )
    .expect("short program plan");
    let long_plan = NebulaPlan::new_with_initial_ram(
        *profile.memory(),
        long.rom_image,
        long.ram_image,
        super::WASM_NEBULA_PLAN_SEED,
        1,
    )
    .expect("long program plan");
    assert_ne!(short_plan.d_init(), long_plan.d_init());
    assert_ne!(short_plan.plan_digest(), long_plan.plan_digest());
}
