mod common;

use neo_math::F;
use neo_wasm::layout::{COL_HOST_CONTEXT_EXPORT, COL_HOST_CONTEXT_IMPORT, COL_HOST_CONTEXT_REFS};
use p3_field::PrimeCharacteristicRing;

#[test]
fn witness_builder_rejects_oversized_saved_references() {
    let run = common::checked_main("(module (func (export \"main\") (result i32) i32.const 0))");
    let mut row = run
        .trace
        .iter()
        .find(|row| row.row_kind.is_program())
        .unwrap()
        .clone();
    let mut context = neo_wasm::WasmHostContext {
        continuation_pc: 0,
        locals_fbp: 0,
        stack_frame_base: 0,
        export_fref: 65535,
        import_fref: 65535,
        args_base: 0,
        parent_host_frame_depth: 0,
        retained_sp: 0,
    };
    row.host_context = Some(context);
    assert!(neo_wasm::build_witness_vector(&row).is_ok());
    for (kind, export, import) in [("export", 65536, 1), ("import", 1, 65536)] {
        context.export_fref = export;
        context.import_fref = import;
        row.host_context = Some(context);
        let error = neo_wasm::build_witness_vector(&row)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains(&format!(
                "saved host-context {kind} reference 65536 exceeds the 16-bit limit 65535"
            )),
            "{error}"
        );
    }
}

#[test]
fn context_references_have_unique_sixteen_bit_halves() {
    let run = common::checked_main("(module (func (export \"main\") (result i32) i32.const 0))");
    let i = run
        .trace
        .iter()
        .position(|row| row.row_kind.is_program())
        .unwrap();
    for (export, import) in [(0, 0), (65535, 0), (0, 65535), (65535, 65535)] {
        let mut w = run.witnesses[i].clone();
        w[COL_HOST_CONTEXT_EXPORT] = F::from_u64(export);
        w[COL_HOST_CONTEXT_IMPORT] = F::from_u64(import);
        w[COL_HOST_CONTEXT_REFS] = F::from_u64(export + (import << 16));
        neo_wasm::write_range_check_bits(&mut w);
        common::assert_satisfied(&w, "both context reference halves include their endpoints");
        w[COL_HOST_CONTEXT_REFS] += F::ONE;
        common::assert_rejected(&w, "the packed context word binds its decoded references");
    }
    // Rocq `ProgramBinding.full_host_context_refs`: the RAM word alone cannot
    // distinguish these aliases.
    for (export, import, packed) in [(65536, 0, 65536), (0, 65536, 1_u64 << 32)] {
        let mut w = run.witnesses[i].clone();
        w[COL_HOST_CONTEXT_EXPORT] = F::from_u64(export);
        w[COL_HOST_CONTEXT_IMPORT] = F::from_u64(import);
        w[COL_HOST_CONTEXT_REFS] = F::from_u64(packed);
        neo_wasm::write_range_check_bits(&mut w);
        common::assert_rejected(&w, "each decoded reference must fit sixteen bits");
    }
}
