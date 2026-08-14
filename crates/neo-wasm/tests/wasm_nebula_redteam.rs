//! End-to-end attacks that the authoritative WASM + Nebula proof must reject.

mod common;

use std::collections::HashSet;
use std::sync::OnceLock;

use neo_fold_clean::frontends::nebula::application::{MemoryPort, MemoryPortActivation, MemoryPortKind};
use neo_fold_clean::paper::params::Params;
#[cfg(all(feature = "metal", target_vendor = "apple"))]
use neo_prover_metal::MetalNifsProver;
use neo_wasm::layout::{COL_OP_TABLE_ENABLED, COL_STACK_WRITE0_VALUE_LO};
use neo_wasm::{WasmMemoryActivation, WasmMemoryColumnKind, WasmMemoryColumnSpec, WasmOpTable, WasmOpcode};
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

#[cfg(all(feature = "metal", target_vendor = "apple"))]
fn proof() -> &'static neo_wasm::nebula::WasmNebulaProof {
    static PROOF: OnceLock<neo_wasm::nebula::WasmNebulaProof> = OnceLock::new();
    PROOF.get_or_init(|| {
        let fixture = fixture();
        let mut prover = neo_wasm::WasmProver::metal().expect("Metal WASM prover");
        prover
            .prove(&fixture.prep, &fixture.checked.trace)
            .expect("Metal WASM Nebula proof")
    })
}

#[cfg(all(feature = "metal", target_vendor = "apple"))]
fn verify_with_metal(
    proof: &neo_wasm::nebula::WasmNebulaProof,
    final_state: neo_wasm::WasmStepState,
) -> Result<(), neo_wasm::nebula::WasmNebulaError> {
    let mut backend = MetalNifsProver::new().expect("Metal opening verifier");
    neo_wasm::nebula::verify_with_witness_opening_backend(&fixture().prep, proof, final_state, &mut backend)
}

fn nebula_test_params() -> Params {
    let raw = neo_params::NeoParams::new(
        neo_params::goldilocks_paper_b2::Q,
        neo_params::goldilocks_paper_b2::ETA as u32,
        neo_params::goldilocks_paper_b2::D as u32,
        1,
        neo_params::goldilocks_paper_b2::M,
        neo_params::goldilocks_paper_b2::B_BASE,
        neo_params::goldilocks_paper_b2::K_RHO,
        neo_params::goldilocks_paper_b2::T,
        neo_params::goldilocks_paper_b2::EXTENSION_DEGREE,
        20,
    )
    .expect("test SuperNeo parameters");
    Params::test_only_from_neo_params(raw)
}

fn two_page_profile() -> neo_wasm::nebula::WasmNebulaProfile {
    const WASM32_PAGE_WORDS: u64 = 65_536 / 4;

    let limits = neo_wasm::nebula::WasmNebulaLimits::new(
        2,
        2,
        2,
        2,
        2 * WASM32_PAGE_WORDS,
        2,
        2,
        2,
        neo_wasm::WasmNebulaRomLimits::test_profile(),
    )
    .expect("two-page WASM test limits");
    neo_wasm::nebula::WasmNebulaProfile::production(limits, 3).expect("two-page WASM test profile")
}

#[test]
#[cfg(all(feature = "metal", target_vendor = "apple"))]
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
    verify_with_metal(proof(), common::final_state(&fixture.checked.trace))
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
    let physical_slots_per_step = memory.slot_count() / batch_size;
    assert_eq!(memory.regions().len(), declared.auxiliary.memories.len());
    assert_eq!(physical_slots_per_step, 21);
    assert_eq!(memory.slot_count(), physical_slots_per_step * batch_size);
    assert_eq!(memory.logical_port_count(), declared_ports * batch_size);
    assert_eq!(
        fixture.prep.profile().memory().b_ops,
        physical_slots_per_step * batch_size,
        "geometry must be sized by physical slots rather than logical ports"
    );

    for block in 0..batch_size {
        let offset = block * single_step_columns;
        let mut expected = declared
            .auxiliary
            .memories
            .iter()
            .enumerate()
            .flat_map(|(region, memory)| memory.columns.iter().map(move |port| (region, port)))
            .collect::<Vec<_>>();

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
        }

        let block_slots = &memory.slots()[block * physical_slots_per_step..(block + 1) * physical_slots_per_step];
        for (slot, physical) in block_slots.iter().enumerate() {
            for candidate in physical.candidates() {
                let position = expected
                    .iter()
                    .position(|(region, declared)| port_matches(candidate, *region, declared, offset))
                    .unwrap_or_else(|| panic!("block {block} slot {slot} candidate is not a declared logical port"));
                expected.remove(position);
            }
        }
        assert!(
            expected.is_empty(),
            "block {block} must route every logical port exactly once"
        );
    }
    assert_eq!(declared_ports, 76, "Current layout declares 76 ports per step");
}

fn port_matches(routed: &MemoryPort, region: usize, declared: &WasmMemoryColumnSpec, offset: usize) -> bool {
    routed.region() == region
        && routed.address_columns().iter().copied().eq(declared
            .address_columns
            .iter()
            .map(|column| offset + column.0))
        && routed.value_column() == offset + declared.value_column.0
        && routed.kind()
            == match declared.kind {
                WasmMemoryColumnKind::Read => MemoryPortKind::Read,
                WasmMemoryColumnKind::Write { value_before_column } => MemoryPortKind::Write {
                    value_before_column: value_before_column.map(|column| offset + column.0),
                },
            }
        && routed.activation()
            == match declared.activation {
                WasmMemoryActivation::Always => {
                    MemoryPortActivation::UnlessColumn(offset + neo_wasm::layout::COL_PADDING_ACTIVE)
                }
                WasmMemoryActivation::BooleanGate(column) => MemoryPortActivation::Column(offset + column.0),
            }
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
#[cfg(all(feature = "metal", target_vendor = "apple"))]
fn wasm_proof_rejects_false_terminal_claim_for_a_prefix() {
    let fixture = fixture();
    let prefix = &fixture.checked.trace[..1];
    assert!(!prefix[0].state_after.halted);
    let mut false_terminal = prefix[0].state_after;
    false_terminal.halted = true;

    assert!(
        verify_with_metal(proof(), false_terminal).is_err(),
        "the verifier accepted a nonterminal prefix after changing only the unbound halted flag",
    );
}

#[test]
#[cfg(all(feature = "metal", target_vendor = "apple"))]
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

    let mut backend = MetalNifsProver::new().expect("Metal opening verifier");
    neo_fold_clean::lifecycle::verify_uncompressed_with_opening_backend(
        &fixture.prep.inner().prep,
        &tampered,
        &mut backend,
    )
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
                Err(neo_wasm::nebula::WasmNebulaError::ImportedStateUnsupported)
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
                two_page_profile(),
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

/// The grammar entry point waives ONLY the host-function-import rejection
/// (those calls are chain-bound by templates); imported memories/globals
/// stay verifier-unbound state, and the linear-memory limits still apply.
#[test]
fn wasm_nebula_grammar_preprocess_keeps_imported_state_and_memory_checks() {
    let mut grammar = neo_wasm::event_grammar::HostEventGrammar::default();
    grammar
        .exports
        .insert(0, neo_wasm::event_grammar::ExportTemplate::default());
    let grammar_preprocess = |wat: &str, seed: u64| {
        let wasm = wat::parse_str(wat).expect("valid WAT");
        let artifacts = neo_wasm::extract_wasm_program_artifacts(&wasm).expect("artifacts");
        let entry_pc = common::single_function_entry_pc(&artifacts);
        neo_wasm::nebula::preprocess_seeded_grammar_test_only(
            nebula_test_params(),
            neo_wasm::nebula::WasmNebulaProfile::test_profile(),
            &artifacts,
            &[],
            entry_pc,
            &grammar,
            0,
            seed,
            Default::default(),
        )
    };
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
        assert!(
            matches!(
                grammar_preprocess(wat, 0x57a5_0102),
                Err(neo_wasm::nebula::WasmNebulaError::ImportedStateUnsupported)
            ),
            "grammar preprocessing accepted imported {label} state",
        );
    }
    assert!(
        matches!(
            grammar_preprocess(
                r#"(module
                    (memory 20000 20000)
                    (func (export "main") (result i32)
                        i32.const 7))"#,
                0x57a5_0103,
            ),
            Err(neo_wasm::nebula::WasmNebulaError::DeclaredLinearMemoryTooLarge { .. })
        ),
        "grammar preprocessing accepted a declared memory larger than its capacity",
    );
}

#[test]
#[cfg(all(feature = "metal", target_vendor = "apple"))]
fn wasm_nebula_final_claim_authenticates_memory_presence() {
    let fixture = fixture();
    let mut forged = common::final_state(&fixture.checked.trace);
    assert!(forged.memory_pages.is_some());
    assert!(forged.max_memory_pages.is_some());
    forged.memory_pages = None;
    forged.max_memory_pages = None;
    assert!(
        matches!(
            verify_with_metal(proof(), forged),
            Err(neo_wasm::nebula::WasmNebulaError::MemoryPresenceMismatch { .. })
        ),
        "terminal verification accepted absent memory in place of authenticated memory",
    );
}

#[test]
fn wasm_nebula_division_trap_trace_disables_the_lookup() {
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
