mod common;

use neo_math::F;
use neo_wasm::layout::{
    COL_HOST_CONTEXT_ARGS_BASE, COL_HOST_CONTEXT_ARGS_DEPTH, COL_HOST_CONTEXT_EXPORT, COL_HOST_CONTEXT_IMPORT,
    COL_HOST_CONTEXT_PARENT_DEPTH, COL_HOST_CONTEXT_REFS,
};
use p3_field::PrimeCharacteristicRing;

#[test]
fn witness_builder_rejects_oversized_saved_fields() {
    let run = common::checked_main("(module (func (export \"main\") (result i32) i32.const 0))");
    let mut row = run
        .trace
        .iter()
        .find(|row| row.row_kind.is_program())
        .unwrap()
        .clone();
    let context = neo_wasm::WasmHostContext {
        continuation_pc: 0,
        locals_fbp: 0,
        stack_frame_base: 0,
        export_fref: 65535,
        import_fref: 65535,
        args_base: 65535,
        parent_host_frame_depth: 65535,
        retained_sp: 0,
    };
    row.host_context = Some(context);
    assert!(neo_wasm::build_witness_vector(&row).is_ok());
    for (field, value) in [
        ("export reference", 65536),
        ("import reference", 65536),
        ("argument base", 65536),
        ("parent host depth", 65536),
        ("argument base", 1_u64 << 32),
        ("parent host depth", 1_u64 << 32),
    ] {
        let mut invalid = context;
        match field {
            "export reference" => invalid.export_fref = value as u32,
            "import reference" => invalid.import_fref = value as u32,
            "argument base" => invalid.args_base = value,
            "parent host depth" => invalid.parent_host_frame_depth = value,
            _ => unreachable!(),
        }
        row.host_context = Some(invalid);
        let error = neo_wasm::build_witness_vector(&row)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains(&format!(
                "saved host-context {field} {value} exceeds the 16-bit limit 65535"
            )),
            "{error}"
        );
    }
}

#[test]
fn context_words_have_unique_sixteen_bit_halves() {
    let run = common::checked_main("(module (func (export \"main\") (result i32) i32.const 0))");
    let i = run
        .trace
        .iter()
        .position(|row| row.row_kind.is_program())
        .unwrap();
    for (lo, hi, word) in [
        (COL_HOST_CONTEXT_EXPORT, COL_HOST_CONTEXT_IMPORT, COL_HOST_CONTEXT_REFS),
        (
            COL_HOST_CONTEXT_ARGS_BASE,
            COL_HOST_CONTEXT_PARENT_DEPTH,
            COL_HOST_CONTEXT_ARGS_DEPTH,
        ),
    ] {
        for (low, high) in [(0, 0), (65535, 0), (0, 65535), (65535, 65535)] {
            let mut w = run.witnesses[i].clone();
            w[lo] = F::from_u64(low);
            w[hi] = F::from_u64(high);
            w[word] = F::from_u64(low + (high << 16));
            neo_wasm::write_range_check_bits(&mut w);
            common::assert_satisfied(&w, "both context halves include their endpoints");
            w[word] += F::ONE;
            common::assert_rejected(&w, "the packed context word binds its decoded fields");
        }
        // Rocq `ProgramBinding.full_host_context_refs`: the RAM word alone cannot distinguish these aliases.
        for (low, high, packed) in [(65536, 0, 65536), (0, 65536, 1_u64 << 32)] {
            let mut w = run.witnesses[i].clone();
            w[lo] = F::from_u64(low);
            w[hi] = F::from_u64(high);
            w[word] = F::from_u64(packed);
            neo_wasm::write_range_check_bits(&mut w);
            common::assert_rejected(&w, "each decoded context field must fit sixteen bits");
        }
    }
}
