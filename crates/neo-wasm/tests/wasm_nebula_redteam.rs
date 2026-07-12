//! End-to-end attacks that the authoritative WASM + Nebula proof must reject.

mod common;

use std::collections::HashSet;
use std::sync::OnceLock;

use neo_fold_clean::paper::params::Params;
use neo_wasm::layout::{COL_OP_TABLE_ENABLED, COL_STACK_WRITE0_VALUE_LO};
use neo_wasm::{WasmOpTable, WasmOpcode};
use p3_field::{PrimeCharacteristicRing, PrimeField64};

struct Fixture {
    checked: common::CheckedWasmRun,
    prep: neo_wasm::nebula::WasmNebulaPreprocessing,
}

fn fixture() -> &'static Fixture {
    static FIXTURE: OnceLock<Fixture> = OnceLock::new();
    FIXTURE.get_or_init(|| {
        let checked = common::checked_main(
            r#"(module
                (memory 1 1)
                (func (export "main") (result i32)
                    i32.const 0
                    i32.const 255
                    i32.store8
                    i32.const 6
                    i32.const 7
                    i32.mul))"#,
        );
        let entry_pc = common::single_function_entry_pc(&checked.artifacts);
        let prep = neo_wasm::nebula::preprocess_seeded_reduced_memory_test_only(
            nebula_test_params(),
            neo_wasm::nebula::WasmNebulaProfile::test_profile(),
            &checked.artifacts,
            &checked.run.initial_locals,
            entry_pc,
            0x57a5_0001,
        )
        .expect("WASM Nebula preprocessing");
        Fixture { checked, prep }
    })
}

fn proof() -> &'static neo_wasm::nebula::WasmNebulaProof {
    static PROOF: OnceLock<neo_wasm::nebula::WasmNebulaProof> = OnceLock::new();
    PROOF.get_or_init(|| {
        let fixture = fixture();
        neo_wasm::prove(&fixture.prep, &fixture.checked.trace).expect("WASM Nebula proof")
    })
}

fn nebula_test_params() -> Params {
    let raw = neo_params::NeoParams::new(
        neo_params::goldilocks_paper_b2::Q,
        neo_params::goldilocks_paper_b2::ETA as u32,
        neo_params::goldilocks_paper_b2::D as u32,
        1,
        1 << 14,
        neo_params::goldilocks_paper_b2::B_BASE,
        neo_params::goldilocks_paper_b2::K_RHO,
        neo_params::goldilocks_paper_b2::T,
        neo_params::goldilocks_paper_b2::EXTENSION_DEGREE,
        20,
    )
    .expect("test SuperNeo parameters");
    Params::test_only_from_neo_params(raw)
}

#[test]
fn wasm_nebula_proves_program_memory_and_terminal_induction() {
    let fixture = fixture();
    let batch_size = fixture.prep.profile().batch_size();
    let logical_batches = fixture.checked.trace.len().div_ceil(batch_size);
    assert!(
        logical_batches > fixture.prep.profile().memory().steps_per_segment(),
        "fixture must cross a Nebula segment boundary"
    );
    assert!(
        proof().inner().final_fold.is_some(),
        "Nebula must consume its trailing delayed claim"
    );
    neo_wasm::verify(&fixture.prep, proof(), common::final_state(&fixture.checked.trace))
        .expect("terminal-only WASM Nebula verification");
}

#[test]
fn wasm_nebula_adapter_covers_every_declared_memory_port_exactly() {
    let fixture = fixture();
    let declared = neo_wasm::build_wasm_relation_layout();
    let application = fixture
        .prep
        .inner()
        .relation()
        .application()
        .expect("WASM application");
    let memory = application.memory();
    let batch_size = fixture.prep.profile().batch_size();
    let single_step_columns = application.shape().m() / batch_size;
    let declared_ports = declared
        .auxiliary
        .memories
        .iter()
        .map(|memory| memory.columns.len())
        .sum::<usize>();
    assert_eq!(memory.regions().len(), declared.auxiliary.memories.len());
    assert_eq!(memory.port_count(), declared_ports * batch_size);

    let mut slot = 0;
    for block in 0..batch_size {
        let offset = block * single_step_columns;
        for (region_index, declared_memory) in declared.auxiliary.memories.iter().enumerate() {
            let region = &memory.regions()[region_index];
            assert_eq!(region.name(), declared_memory.name);
            assert_eq!(
                region.kind(),
                if declared_memory.is_rom {
                    neo_fold_clean::frontends::nebula::application::MemoryRegionKind::Rom
                } else {
                    neo_fold_clean::frontends::nebula::application::MemoryRegionKind::Ram
                }
            );
            for declared_port in &declared_memory.columns {
                let port = &memory.ports()[slot];
                assert_eq!(port.region(), region_index, "slot {slot}");
                assert_eq!(
                    port.address_columns(),
                    declared_port
                        .address_columns
                        .iter()
                        .map(|column| offset + column.0)
                        .collect::<Vec<_>>(),
                    "slot {slot}"
                );
                assert_eq!(
                    port.value_column(),
                    offset + declared_port.value_column.0,
                    "slot {slot}"
                );
                let expected_kind = match declared_port.kind {
                    neo_wasm::WasmMemoryColumnKind::Read => {
                        neo_fold_clean::frontends::nebula::application::MemoryPortKind::Read
                    }
                    neo_wasm::WasmMemoryColumnKind::Write { value_before_column } => {
                        neo_fold_clean::frontends::nebula::application::MemoryPortKind::Write {
                            value_before_column: value_before_column.map(|column| offset + column.0),
                        }
                    }
                };
                assert_eq!(port.kind(), expected_kind, "slot {slot}");
                let expected_activation = match declared_port.activation {
                    neo_wasm::WasmMemoryActivation::Always => {
                        neo_fold_clean::frontends::nebula::application::MemoryPortActivation::UnlessColumn(
                            offset + neo_wasm::layout::COL_PADDING_ACTIVE,
                        )
                    }
                    neo_wasm::WasmMemoryActivation::BooleanGate(column) => {
                        neo_fold_clean::frontends::nebula::application::MemoryPortActivation::Column(offset + column.0)
                    }
                };
                assert_eq!(port.activation(), expected_activation, "slot {slot}");
                slot += 1;
            }
        }
    }
    assert_eq!(declared_ports, 60, "Enzo's current layout declares 60 ports per step");
    assert_eq!(slot, declared_ports * batch_size);
}

#[test]
fn wasm_proof_rejects_forged_i32_mul_lookup() {
    let fixture = fixture();
    let mut forged = fixture.checked.trace.clone();
    let mul = forged
        .iter()
        .position(|row| row.opcode == WasmOpcode::I32Mul)
        .expect("i32.mul row");
    let honest = forged[mul].stack_write0.expect("i32.mul output").value_lo;
    let forged_value = honest + 1;
    forged[mul]
        .stack_write0
        .as_mut()
        .expect("i32.mul output")
        .value_lo = forged_value;

    for row in &mut forged[mul + 1..] {
        for read in [&mut row.stack_read0, &mut row.stack_read1, &mut row.stack_read2]
            .into_iter()
            .flatten()
        {
            if read.value_lo == honest {
                read.value_lo = forged_value;
            }
        }
        if row.output_captured {
            row.state_after.output.value_lo = forged_value;
        }
    }
    common::ccs_check_trace(&forged);

    assert!(
        neo_wasm::prove(&fixture.prep, &forged).is_err(),
        "the proof accepted a forged i32.mul result because lookup semantics were not authoritative",
    );
}

#[test]
fn wasm_proof_rejects_false_initial_linear_memory_value() {
    let fixture = fixture();
    let mut forged = fixture.checked.trace.clone();
    let store = forged
        .iter()
        .position(|row| row.opcode == WasmOpcode::I32Store8)
        .expect("i32.store8 row");
    let fake_before = 0xdead_beefu32;
    let fake_after = (fake_before & !0xff) | 0xff;
    let access = forged[store]
        .linear_memory
        .as_mut()
        .expect("store memory access");
    access.lane0.value_before = fake_before;
    access.lane0.value_after = fake_after;
    common::ccs_check_trace(&forged);

    assert!(
        neo_wasm::prove(&fixture.prep, &forged).is_err(),
        "the proof accepted a self-consistent history starting from false linear memory",
    );
}

#[test]
fn wasm_proof_rejects_trace_from_a_different_program() {
    let fixture = fixture();
    let substituted = common::checked_main(
        r#"(module
            (memory 1 1)
            (func (export "main") (result i32)
                i32.const 0
                i32.const 255
                i32.store8
                i32.const 6
                i32.const 8
                i32.mul))"#,
    );

    assert!(
        neo_wasm::prove(&fixture.prep, &substituted.trace).is_err(),
        "the verifier accepted a trace from a different WASM program",
    );
}

#[test]
fn wasm_proof_rejects_false_terminal_claim_for_a_prefix() {
    let fixture = fixture();
    let prefix = &fixture.checked.trace[..1];
    assert!(!prefix[0].state_after.halted);
    let mut false_terminal = prefix[0].state_after;
    false_terminal.halted = true;

    assert!(
        neo_wasm::verify(&fixture.prep, proof(), false_terminal).is_err(),
        "the verifier accepted a nonterminal prefix after changing only the unbound halted flag",
    );
}

#[test]
fn wasm_nebula_terminal_only_rejects_earlier_fold_tamper() {
    let fixture = fixture();
    let mut tampered = proof().inner().clone();
    tampered
        .final_fold
        .as_mut()
        .expect("terminal Nebula fold")
        .terminal_inputs
        .pre_final_running
        .claims[0]
        .c
        .data[0] += neo_math::F::ONE;

    neo_fold_clean::verify_uncompressed(&fixture.prep.inner().prep, &tampered)
        .expect_err("terminal verifier accepted a changed earlier-history accumulator");
}

#[test]
fn wasm_nebula_rejects_unbound_host_imports() {
    let wasm = wat::parse_str(
        r#"(module
            (import "host" "value" (func $value (result i32)))
            (func (export "main") (result i32)
                call $value))"#,
    )
    .expect("host-import WAT");
    let artifacts = neo_wasm::extract_wasm_program_artifacts(&wasm).expect("host-import artifacts");
    let entry_pc = common::single_function_entry_pc(&artifacts);
    assert!(matches!(
        neo_wasm::nebula::preprocess_seeded(
            nebula_test_params(),
            neo_wasm::nebula::WasmNebulaProfile::test_profile(),
            &artifacts,
            &[],
            entry_pc,
            0x57a5_00ff,
        ),
        Err(neo_wasm::nebula::WasmNebulaError::HostImportsUnsupported)
    ));
}

#[test]
fn wasm_nebula_sound_preprocess_rejects_imported_memory_and_globals() {
    for (label, wat) in [
        (
            "memory",
            r#"(module
                (import "host" "memory" (memory 0 0))
                (func (export "main") (result i32)
                    i32.const 7))"#,
        ),
        (
            "global",
            r#"(module
                (import "host" "global" (global (mut i32)))
                (func (export "main") (result i32)
                    i32.const 7))"#,
        ),
    ] {
        let wasm = wat::parse_str(wat).expect("valid imported-state WAT");
        let artifacts = neo_wasm::extract_wasm_program_artifacts(&wasm).expect("imported-state artifacts");
        let entry_pc = common::single_function_entry_pc(&artifacts);
        assert!(
            matches!(
                neo_wasm::nebula::preprocess_seeded(
                    nebula_test_params(),
                    neo_wasm::nebula::WasmNebulaProfile::test_profile(),
                    &artifacts,
                    &[],
                    entry_pc,
                    0x57a5_0100,
                ),
                Err(neo_wasm::nebula::WasmNebulaError::HostImportsUnsupported)
            ),
            "sound preprocessing accepted imported {label} state",
        );
    }
}

#[test]
fn wasm_nebula_sound_preprocess_rejects_declared_memory_outside_dense_domain() {
    let wasm = wat::parse_str(
        r#"(module
            (memory 3 3)
            (func (export "main") (result i32)
                i32.const 7))"#,
    )
    .expect("valid oversized-memory WAT");
    let artifacts = neo_wasm::extract_wasm_program_artifacts(&wasm).expect("oversized-memory artifacts");
    let entry_pc = common::single_function_entry_pc(&artifacts);
    assert!(
        matches!(
            neo_wasm::nebula::preprocess_seeded(
                nebula_test_params(),
                neo_wasm::nebula::WasmNebulaProfile::production(),
                &artifacts,
                &[],
                entry_pc,
                0x57a5_0101,
            ),
            Err(neo_wasm::nebula::WasmNebulaError::DeclaredLinearMemoryTooLarge {
                initial_pages: 3,
                max_pages: 3,
                capacity_pages: 2,
            })
        ),
        "sound preprocessing accepted a declared memory larger than its dense domain",
    );
}

#[test]
fn wasm_nebula_final_claim_authenticates_memory_presence() {
    let checked = common::checked_main(
        r#"(module
            (func (export "main") (result i32)
                i32.const 7))"#,
    );
    let entry_pc = common::single_function_entry_pc(&checked.artifacts);
    let prep = neo_wasm::nebula::preprocess_seeded(
        nebula_test_params(),
        neo_wasm::nebula::WasmNebulaProfile::test_profile(),
        &checked.artifacts,
        &checked.run.initial_locals,
        entry_pc,
        0x57a5_0102,
    )
    .expect("no-memory preprocessing");
    let proof = neo_wasm::prove(&prep, &checked.trace).expect("no-memory proof");
    let mut forged = common::final_state(&checked.trace);
    assert_eq!(forged.memory_pages, None);
    assert_eq!(forged.max_memory_pages, None);
    forged.memory_pages = Some(0);
    forged.max_memory_pages = Some(0);
    assert!(
        matches!(
            neo_wasm::verify(&prep, &proof, forged),
            Err(neo_wasm::nebula::WasmNebulaError::MemoryPresenceMismatch { .. })
        ),
        "terminal verification accepted Some(0) in place of an authenticated absent memory",
    );
}

#[test]
fn wasm_nebula_proves_a_division_trap_with_lookup_disabled() {
    let checked = common::checked_main(
        r#"(module
            (func (export "main") (result i32)
                i32.const 7
                i32.const 0
                i32.div_u))"#,
    );
    let division = checked
        .trace
        .iter()
        .find(|row| row.opcode == WasmOpcode::I32DivU)
        .expect("division row");
    assert!(division.state_after.trapped);
    let witness = neo_wasm::build_witness_vector(division);
    assert_eq!(witness[COL_OP_TABLE_ENABLED], neo_math::F::ZERO);
    neo_wasm::audit_compact_lookup_witness(&witness).expect("trapping division must deactivate lookup arithmetic");

    let entry_pc = common::single_function_entry_pc(&checked.artifacts);
    let prep = neo_wasm::nebula::preprocess_seeded(
        nebula_test_params(),
        neo_wasm::nebula::WasmNebulaProfile::test_profile(),
        &checked.artifacts,
        &checked.run.initial_locals,
        entry_pc,
        0x57a5_00d1,
    )
    .expect("division-trap preprocessing");
    let proof = neo_wasm::prove(&prep, &checked.trace).expect("division-trap proof");
    neo_wasm::verify(&prep, &proof, common::final_state(&checked.trace)).expect("division-trap terminal verification");
}

#[test]
fn wasm_nebula_compact_lookup_relation_covers_and_rejects_all_families() {
    let fixture = fixture();
    eprintln!(
        "compact lookup relation: auxiliary_columns={}",
        fixture.prep.lookup_auxiliary_columns_per_instruction()
    );

    let checked = common::checked_main(
        r#"(module
            (func (export "main") (result i32)
                i32.const 305419896 i32.clz drop
                i32.const 305419896 i32.ctz drop
                i32.const -123 i32.const 7 i32.lt_s drop
                i32.const 123 i32.const 7 i32.lt_u drop
                i32.const -123 i32.const 7 i32.gt_s drop
                i32.const 123 i32.const 7 i32.gt_u drop
                i32.const -123 i32.const 7 i32.le_s drop
                i32.const 123 i32.const 7 i32.le_u drop
                i32.const -123 i32.const 7 i32.ge_s drop
                i32.const 123 i32.const 7 i32.ge_u drop
                i32.const 305419896 i32.const 252645135 i32.and drop
                i32.const 305419896 i32.const 252645135 i32.or drop
                i32.const 305419896 i32.const 252645135 i32.xor drop
                i32.const 305419896 i32.const 7 i32.mul drop
                i64.const 81985529216486895 i64.const 1085102592571150095 i64.and drop
                i64.const 81985529216486895 i64.const 1085102592571150095 i64.or drop
                i64.const 81985529216486895 i64.const 1085102592571150095 i64.xor drop
                i64.const 81985529216486895 i64.const 7 i64.mul drop
                i32.const 305419896 i32.const 5 i32.shl drop
                i32.const 305419896 i32.const 5 i32.shr_u drop
                i32.const -305419896 i32.const 5 i32.shr_s drop
                i32.const 305419896 i32.const 5 i32.rotl drop
                i32.const 305419896 i32.const 5 i32.rotr drop
                i32.const 305419896 i32.const 7 i32.div_u drop
                i32.const -305419896 i32.const 7 i32.div_s drop
                i32.const 305419896 i32.const 7 i32.rem_u drop
                i32.const -305419896 i32.const 7 i32.rem_s drop
                i32.const 305419896 i32.popcnt drop
                i64.const -81985529216486895 i64.const 7 i64.lt_s drop
                i64.const 81985529216486895 i64.const 7 i64.lt_u drop
                i64.const -81985529216486895 i64.const 7 i64.gt_s drop
                i64.const 81985529216486895 i64.const 7 i64.gt_u drop
                i64.const -81985529216486895 i64.const 7 i64.le_s drop
                i64.const 81985529216486895 i64.const 7 i64.le_u drop
                i64.const -81985529216486895 i64.const 7 i64.ge_s drop
                i64.const 81985529216486895 i64.const 7 i64.ge_u drop
                i64.const 81985529216486895 i64.const 13 i64.shl drop
                i64.const -81985529216486895 i64.const 13 i64.shr_s drop
                i64.const 81985529216486895 i64.const 13 i64.shr_u drop
                i64.const 81985529216486895 i64.const 13 i64.rotl drop
                i64.const 81985529216486895 i64.const 13 i64.rotr drop
                i64.const -81985529216486895 i64.const 7 i64.div_s drop
                i64.const 81985529216486895 i64.const 7 i64.div_u drop
                i64.const -81985529216486895 i64.const 7 i64.rem_s drop
                i64.const 81985529216486895 i64.const 7 i64.rem_u drop
                i64.const 81985529216486895 i64.clz drop
                i64.const 81985529216486895 i64.ctz drop
                i64.const 81985529216486895 i64.popcnt drop
                i32.const 0))"#,
    );
    let mut seen = HashSet::new();
    for row in checked.trace.iter().filter(|row| row.info.uses_op_table) {
        seen.insert(row.opcode);
        let witness = neo_wasm::build_witness_vector(row);
        let auxiliary_columns = neo_wasm::audit_compact_lookup_witness(&witness)
            .unwrap_or_else(|error| panic!("compact lookup relation rejected honest {:?}: {error}", row.opcode));
        assert_eq!(
            auxiliary_columns,
            fixture.prep.lookup_auxiliary_columns_per_instruction()
        );

        let mut tampered = witness;
        let value = tampered[COL_STACK_WRITE0_VALUE_LO].as_canonical_u64() ^ 1;
        tampered[COL_STACK_WRITE0_VALUE_LO] = neo_math::F::from_u64(value);
        neo_wasm::write_range_check_bits(&mut tampered);
        assert!(
            neo_wasm::audit_compact_lookup_witness(&tampered).is_err(),
            "compact lookup relation accepted a forged {:?} output",
            row.opcode
        );
    }
    let expected = WasmOpTable::all()
        .into_iter()
        .map(WasmOpTable::opcode)
        .collect::<HashSet<_>>();
    assert_eq!(
        seen, expected,
        "fixture must execute every lookup family exactly at least once"
    );
}

#[test]
fn wasm_nebula_compact_lookup_signed_division_edges_and_advice_are_bound() {
    let checked = common::checked_main(
        r#"(module
            (func (export "main") (result i32)
                i32.const -1 i32.const 7 i32.div_s drop
                i32.const 1 i32.const -7 i32.div_s drop
                i32.const -13 i32.const -5 i32.div_s drop
                i32.const -13 i32.const -5 i32.rem_s drop
                i64.const -1 i64.const 7 i64.div_s drop
                i64.const 1 i64.const -7 i64.div_s drop
                i64.const -13 i64.const -5 i64.div_s drop
                i64.const -13 i64.const -5 i64.rem_s drop
                i32.const 0))"#,
    );
    let rows = checked
        .trace
        .iter()
        .filter(|row| {
            matches!(
                row.opcode,
                WasmOpcode::I32DivS | WasmOpcode::I32RemS | WasmOpcode::I64DivS | WasmOpcode::I64RemS
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(rows.len(), 8);
    for row in &rows {
        let witness = neo_wasm::build_witness_vector(row);
        neo_wasm::audit_compact_lookup_witness(&witness)
            .unwrap_or_else(|error| panic!("signed division edge {:?} failed: {error}", row.opcode));
        let mut tampered = witness;
        let value = tampered[COL_STACK_WRITE0_VALUE_LO].as_canonical_u64() ^ 1;
        tampered[COL_STACK_WRITE0_VALUE_LO] = neo_math::F::from_u64(value);
        neo_wasm::write_range_check_bits(&mut tampered);
        assert!(
            neo_wasm::audit_compact_lookup_witness(&tampered).is_err(),
            "signed division edge accepted a forged {:?} result",
            row.opcode
        );
    }

    let representative = neo_wasm::build_witness_vector(rows.last().expect("signed division row"));
    assert_eq!(
        neo_wasm::audit_compact_lookup_auxiliary_load_bearing(&representative)
            .expect("every compact lookup auxiliary must be load-bearing"),
        fixture().prep.lookup_auxiliary_columns_per_instruction(),
    );
}

#[test]
fn wasm_nebula_compact_lookup_accepts_nontrapping_signed_remainder_overflow() {
    let checked = common::checked_main(
        r#"(module
            (func (export "main") (result i32)
                i32.const -2147483648 i32.const -1 i32.rem_s drop
                i64.const -9223372036854775808 i64.const -1 i64.rem_s drop
                i32.const 0))"#,
    );
    let rows = checked
        .trace
        .iter()
        .filter(|row| matches!(row.opcode, WasmOpcode::I32RemS | WasmOpcode::I64RemS))
        .collect::<Vec<_>>();
    assert_eq!(rows.len(), 2);
    for row in rows {
        assert!(!row.state_after.trapped, "signed remainder overflow does not trap");
        let mut witness = neo_wasm::build_witness_vector(row);
        neo_wasm::audit_compact_lookup_witness(&witness)
            .unwrap_or_else(|error| panic!("compact lookup rejected honest {:?}: {error}", row.opcode));
        witness[COL_STACK_WRITE0_VALUE_LO] = neo_math::F::ONE;
        neo_wasm::write_range_check_bits(&mut witness);
        assert!(
            neo_wasm::audit_compact_lookup_witness(&witness).is_err(),
            "signed remainder overflow accepted a nonzero {:?} result",
            row.opcode
        );
    }
}
