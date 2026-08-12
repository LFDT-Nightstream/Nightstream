//! Grammar-driven linear-memory integration and negative constraint tests.

mod common;

use neo_wasm::comm_chain::COMM_CHAIN_EVENT_ARGS;
use neo_wasm::event_grammar::{ExportTemplate, GrammarEvent, HostEventGrammar, ImportTemplate, MemoryBase, SlotSource};
use neo_wasm::witness_builder::build_witness_vector;
use neo_wasm::{WasmGrammarSlotKind, WasmOpcode, WasmVmStep};
use p3_field::PrimeCharacteristicRing;

const ZERO: SlotSource = SlotSource::Const(0);

fn slots(entries: &[(usize, SlotSource)]) -> [SlotSource; COMM_CHAIN_EVENT_ARGS] {
    let mut out = [ZERO; COMM_CHAIN_EVENT_ARGS];
    for &(idx, source) in entries {
        out[idx] = source;
    }
    out
}

fn component_wat() -> &'static str {
    r#"
    (component
      (type $host-touch (func (param "ptr" s32)))
      (type $run-type (func))
      (import "host-touch" (func $host-touch (type $host-touch)))
      (core module $m
        (import "" "0" (func $touch (param i32)))
        (memory 1)
        (data (i32.const 24) "\7b\00\00\00")
        (func (export "run")
          i32.const 16
          i32.const 99
          i32.store
          i32.const 19
          i32.const 0x1234
          i32.store16
          i32.const 16
          call $touch))
      (core func $lowered (canon lower (func $host-touch)))
      (core instance $host (export "0" (func $lowered)))
      (core instance $i (instantiate $m (with "" (instance $host))))
      (alias core export $i "run" (core func $run))
      (func (export "run") (type $run-type)
        (canon lift (core func $run))))
    "#
}

fn run_frefs(run: &neo_wasm::WasmtimeTraceRun) -> (u32, u32) {
    let host = run
        .steps
        .iter()
        .find(|row| matches!(row.opcode_decoded, Some(WasmOpcode::Call)) && !row.target_function_is_guest)
        .and_then(|row| row.function_ref)
        .expect("host function ref");
    let export = run
        .steps
        .iter()
        .find_map(|row| row.current_function_ref)
        .expect("export function ref");
    (host, export)
}

fn memory_grammar(host_fref: u32, export_fref: u32) -> HostEventGrammar {
    let arg = MemoryBase::Arg(0);
    let mut grammar = HostEventGrammar::default();
    grammar.imports.insert(
        host_fref,
        ImportTemplate {
            events: vec![
                GrammarEvent::op(
                    40,
                    slots(&[
                        (
                            0,
                            SlotSource::MemoryRead32 {
                                base: arg,
                                byte_offset: 0,
                            },
                        ),
                        (
                            1,
                            SlotSource::MemoryRead32 {
                                base: arg,
                                byte_offset: 4,
                            },
                        ),
                        (
                            2,
                            SlotSource::MemoryWrite32 {
                                claim: 0,
                                base: arg,
                                byte_offset: 4,
                            },
                        ),
                        (
                            3,
                            SlotSource::MemoryRead32 {
                                base: arg,
                                byte_offset: 4,
                            },
                        ),
                        (
                            4,
                            SlotSource::MemoryRead32 {
                                base: arg,
                                byte_offset: 8,
                            },
                        ),
                    ]),
                ),
                GrammarEvent::op(
                    41,
                    slots(&[
                        (
                            0,
                            SlotSource::MemoryRead8 {
                                base: arg,
                                byte_offset: 0,
                            },
                        ),
                        (
                            1,
                            SlotSource::MemoryRead8 {
                                base: arg,
                                byte_offset: 1,
                            },
                        ),
                        (
                            2,
                            SlotSource::MemoryRead8 {
                                base: arg,
                                byte_offset: 2,
                            },
                        ),
                        (
                            3,
                            SlotSource::MemoryRead8 {
                                base: arg,
                                byte_offset: 3,
                            },
                        ),
                        (
                            4,
                            SlotSource::MemoryWrite8 {
                                claim: 0,
                                base: arg,
                                byte_offset: 2,
                            },
                        ),
                        (
                            5,
                            SlotSource::MemoryRead8 {
                                base: arg,
                                byte_offset: 2,
                            },
                        ),
                    ]),
                ),
                GrammarEvent::op(
                    42,
                    slots(&[
                        (
                            0,
                            SlotSource::MemoryRead16 {
                                base: arg,
                                byte_offset: 0,
                            },
                        ),
                        (
                            1,
                            SlotSource::MemoryRead16 {
                                base: arg,
                                byte_offset: 2,
                            },
                        ),
                        (
                            2,
                            SlotSource::MemoryWrite16 {
                                claim: 0,
                                base: arg,
                                byte_offset: 2,
                            },
                        ),
                        (
                            3,
                            SlotSource::MemoryRead16 {
                                base: arg,
                                byte_offset: 2,
                            },
                        ),
                        (
                            4,
                            SlotSource::MemoryRead16 {
                                base: arg,
                                byte_offset: 10,
                            },
                        ),
                    ]),
                ),
            ],
            claim_count: 1,
        },
    );
    grammar
        .exports
        .insert(export_fref, ExportTemplate::default());
    grammar
}

struct ImportMemoryFixture {
    component_bytes: Vec<u8>,
    run: neo_wasm::WasmtimeTraceRun,
    host_fref: u32,
    grammar: HostEventGrammar,
    trace: Vec<WasmVmStep>,
}

fn import_memory_fixture() -> ImportMemoryFixture {
    let component_bytes = wat::parse_str(component_wat()).expect("component wat");
    let run = neo_wasm::collect_wasmtime_component_run_with_linker(&component_bytes, "run", |linker| {
        linker
            .root()
            .func_wrap("host-touch", |mut store, (_ptr,): (i32,)| {
                store.data_mut().record_call_claims(&[77])?;
                Ok(())
            })
            .map_err(|err| neo_wasm::WasmBuildError::Trace(format!("failed to define host-touch: {err}")))
    })
    .expect("component run");
    let (host_fref, export_fref) = run_frefs(&run);
    let grammar = memory_grammar(host_fref, export_fref);
    let trace = neo_wasm::traces_from_wasmtime_steps_with_grammar(
        &run.steps,
        &run.program_tables,
        &grammar,
        &[Default::default()],
        Default::default(),
    )
    .expect("grammar trace");

    neo_wasm::comm_chain::sanity_check_comm_chain(&trace).expect("chain checker");
    common::ccs_check_trace(&trace);
    check_memory_rows(&component_bytes, &run, &grammar, &trace);

    ImportMemoryFixture {
        component_bytes,
        run,
        host_fref,
        grammar,
        trace,
    }
}

fn check_memory_rows(
    component_bytes: &[u8],
    run: &neo_wasm::WasmtimeTraceRun,
    grammar: &HostEventGrammar,
    trace: &[WasmVmStep],
) {
    let artifacts = neo_wasm::extract_first_component_core_program_artifacts(component_bytes).expect("artifacts");
    let mut preload = neo_wasm::memory_semantics::preload_from_program_artifacts(&artifacts, &run.initial_locals);
    neo_wasm::memory_semantics::preload_grammar_tables(&mut preload, grammar);
    let witnesses: Vec<Vec<neo_math::F>> = trace.iter().map(build_witness_vector).collect();
    let layout = neo_wasm::relation_layout::build_wasm_relation_layout();
    neo_wasm::memory_semantics::sanity_check_memory_rows(layout, &witnesses, &preload)
        .expect("grammar argument base and linear-memory accesses match");
}

#[test]
fn import_memory_accesses_use_argument_based_addresses() {
    let fixture = import_memory_fixture();

    let observed_word_reads: Vec<u32> = fixture
        .trace
        .iter()
        .filter_map(|row| {
            (row.grammar_rom_slot?.kind == WasmGrammarSlotKind::MemoryRead && row.linear_memory?.width_bytes == 4)
                .then_some(row.linear_memory?.lane0.value_before)
        })
        .collect();
    assert_eq!(observed_word_reads, [0x3400_0063, 0x12, 77, 123]);

    let observed_byte_reads: Vec<u8> = fixture
        .trace
        .iter()
        .filter_map(|row| {
            let access = row.linear_memory?;
            (row.grammar_rom_slot?.kind == WasmGrammarSlotKind::MemoryRead && access.width_bytes == 1)
                .then_some(access.lane0.value_before.to_le_bytes()[usize::from(access.byte_offset)])
        })
        .collect();
    assert_eq!(observed_byte_reads, [99, 0, 0, 0x34, 77]);

    let observed_half_reads: Vec<u16> = fixture
        .trace
        .iter()
        .filter_map(|row| {
            let access = row.linear_memory?;
            if row.grammar_rom_slot?.kind != WasmGrammarSlotKind::MemoryRead || access.width_bytes != 2 {
                return None;
            }
            let bytes = access.lane0.value_before.to_le_bytes();
            let offset = usize::from(access.byte_offset);
            Some(u16::from_le_bytes([bytes[offset], bytes[offset + 1]]))
        })
        .collect();
    assert_eq!(observed_half_reads, [99, 0x344d, 77, 0]);
}

#[test]
fn import_memory_normalization_rejects_invalid_addresses() {
    let fixture = import_memory_fixture();

    let mut misaligned_grammar = fixture.grammar.clone();
    let half_read = misaligned_grammar
        .imports
        .get_mut(&fixture.host_fref)
        .expect("host template")
        .events
        .iter_mut()
        .flat_map(|event| &mut event.block)
        .find_map(|slot| match slot {
            SlotSource::MemoryRead16 { byte_offset, .. } => Some(byte_offset),
            _ => None,
        })
        .expect("half-word read");
    *half_read = 1;
    let err = neo_wasm::traces_from_wasmtime_steps_with_grammar(
        &fixture.run.steps,
        &fixture.run.program_tables,
        &misaligned_grammar,
        &[Default::default()],
        Default::default(),
    )
    .expect_err("misaligned grammar half-word access must be rejected");
    assert!(err.to_string().contains("is not naturally aligned"));

    let mut high_pointer_steps = fixture.run.steps.clone();
    let host_call = high_pointer_steps
        .iter_mut()
        .find(|step| step.opcode_decoded == Some(WasmOpcode::Call) && !step.target_function_is_guest)
        .expect("host call step");
    host_call
        .operand_stack_words_hi
        .resize(host_call.operand_stack_words.len(), 0);
    *host_call
        .operand_stack_words_hi
        .last_mut()
        .expect("pointer argument high limb") = 1;
    let err = neo_wasm::traces_from_wasmtime_steps_with_grammar(
        &high_pointer_steps,
        &fixture.run.program_tables,
        &fixture.grammar,
        &[Default::default()],
        Default::default(),
    )
    .expect_err("wasm32 grammar pointer with a high limb must be rejected");
    assert!(err.to_string().contains("not a wasm32 pointer"));
}

#[test]
fn grammar_memory_reads_reject_forged_addresses() {
    let fixture = import_memory_fixture();
    let read = fixture
        .trace
        .iter()
        .find(|row| {
            row.grammar_rom_slot
                .is_some_and(|rom| rom.kind == WasmGrammarSlotKind::MemoryRead)
        })
        .expect("memory read gather");
    let baseline = build_witness_vector(read);
    common::assert_satisfied(&baseline, "untampered argument-base memory read");

    let mut forged = baseline.clone();
    forged[neo_wasm::layout::COL_LINEAR_MEM_LANE_ADDR[0]] += neo_math::F::ONE;
    common::assert_rejected(&forged, "grammar memory read redirected to another word");

    let mut high_pointer = read.clone();
    high_pointer.wide_values_enabled = true;
    high_pointer
        .stack_read0
        .as_mut()
        .expect("pointer argument read")
        .value_hi = Some(1);
    common::assert_rejected(
        &build_witness_vector(&high_pointer),
        "grammar memory read with a nonzero pointer high limb",
    );

    let mut oob_read = read.clone();
    let first_oob_word = u64::from(oob_read.state_before.memory_pages.expect("memory pages")) * 16384;
    oob_read
        .stack_read0
        .as_mut()
        .expect("pointer argument read")
        .value_lo = u32::try_from(first_oob_word * 4).expect("wasm32 byte address");
    oob_read
        .linear_memory
        .as_mut()
        .expect("grammar memory access")
        .lane0
        .word_addr = first_oob_word;
    let forged = build_witness_vector(&oob_read);
    assert_eq!(forged[neo_wasm::layout::COL_CMP_GE], neo_math::F::ONE);
    assert_eq!(forged[neo_wasm::layout::COL_MEM_OOB], neo_math::F::ONE);
    common::assert_rejected(&forged, "aligned OOB grammar memory read");
}

#[test]
fn grammar_subword_routing_rejects_forged_offsets() {
    let fixture = import_memory_fixture();

    let byte_read = fixture
        .trace
        .iter()
        .find(|row| {
            row.grammar_rom_slot
                .is_some_and(|rom| rom.kind == WasmGrammarSlotKind::MemoryRead)
                && row
                    .linear_memory
                    .is_some_and(|access| access.width_bytes == 1 && access.byte_offset == 3)
        })
        .expect("byte memory read gather");
    common::assert_satisfied(&build_witness_vector(byte_read), "untampered grammar byte read");
    let mut redirected = byte_read.clone();
    redirected
        .linear_memory
        .as_mut()
        .expect("byte access")
        .byte_offset = 2;
    common::assert_rejected(
        &build_witness_vector(&redirected),
        "grammar byte read with a forged intra-word offset",
    );

    let equal_neighbor_read = fixture
        .trace
        .iter()
        .find(|row| {
            row.grammar_rom_slot
                .is_some_and(|rom| rom.kind == WasmGrammarSlotKind::MemoryRead)
                && row
                    .linear_memory
                    .is_some_and(|access| access.width_bytes == 1 && access.byte_offset == 1)
        })
        .expect("byte read beside an equal-valued byte");
    let mut forged = build_witness_vector(equal_neighbor_read);
    common::assert_satisfied(&forged, "untampered grammar byte offset selector");
    forged[neo_wasm::layout::COL_LINEAR_MEM_OFFSET_IS_1] = neo_math::F::ZERO;
    forged[neo_wasm::layout::COL_LINEAR_MEM_OFFSET_IS_2] = neo_math::F::ONE;
    forged[neo_wasm::layout::COL_LINEAR_MEM_BYTE_WIDTH_OFFSET_IS_1] = neo_math::F::ZERO;
    forged[neo_wasm::layout::COL_LINEAR_MEM_BYTE_WIDTH_OFFSET_IS_2] = neo_math::F::ONE;
    common::assert_rejected(
        &forged,
        "grammar byte routing selector diverging from the effective address",
    );

    let half_read = fixture
        .trace
        .iter()
        .find(|row| {
            row.grammar_rom_slot
                .is_some_and(|rom| rom.kind == WasmGrammarSlotKind::MemoryRead)
                && row.linear_memory.is_some_and(|access| {
                    access.width_bytes == 2 && access.byte_offset == 2 && access.lane0.word_addr == 6
                })
        })
        .expect("zero-valued aligned half-word read");
    common::assert_satisfied(&build_witness_vector(half_read), "untampered grammar half-word read");
    let mut misaligned = half_read.clone();
    misaligned
        .stack_read0
        .as_mut()
        .expect("pointer argument read")
        .value_lo = 15;
    misaligned
        .linear_memory
        .as_mut()
        .expect("half-word access")
        .byte_offset = 1;
    common::assert_rejected(
        &build_witness_vector(&misaligned),
        "grammar half-word read with an odd effective address",
    );
}

#[test]
fn grammar_memory_writes_bind_values_and_preserve_unselected_bytes() {
    let fixture = import_memory_fixture();

    let word_write = fixture
        .trace
        .iter()
        .find(|row| {
            row.grammar_rom_slot
                .is_some_and(|rom| rom.kind == WasmGrammarSlotKind::MemoryWrite)
                && row
                    .linear_memory
                    .is_some_and(|access| access.width_bytes == 4)
        })
        .expect("word memory write gather");
    let mut forged = build_witness_vector(word_write);
    common::assert_satisfied(&forged, "untampered grammar word write");
    forged[neo_wasm::layout::COL_LINEAR_MEM_LANE_VALUE[0]] += neo_math::F::ONE;
    common::assert_rejected(&forged, "grammar memory write diverging from the staged claim");

    let byte_write = fixture
        .trace
        .iter()
        .find(|row| {
            row.grammar_rom_slot
                .is_some_and(|rom| rom.kind == WasmGrammarSlotKind::MemoryWrite)
                && row
                    .linear_memory
                    .is_some_and(|access| access.width_bytes == 1)
        })
        .expect("byte memory write gather");
    let mut forged = build_witness_vector(byte_write);
    common::assert_satisfied(&forged, "untampered grammar byte write");
    forged[neo_wasm::layout::COL_LINEAR_MEM_LANE_VALUE[0]] += neo_math::F::ONE;
    forged[neo_wasm::layout::COL_LINEAR_MEM_LANE0_BYTE0] += neo_math::F::ONE;
    common::assert_rejected(&forged, "grammar byte write changing an unselected byte");

    let half_write = fixture
        .trace
        .iter()
        .find(|row| {
            row.grammar_rom_slot
                .is_some_and(|rom| rom.kind == WasmGrammarSlotKind::MemoryWrite)
                && row
                    .linear_memory
                    .is_some_and(|access| access.width_bytes == 2)
        })
        .expect("half-word memory write gather");
    let mut forged = build_witness_vector(half_write);
    common::assert_satisfied(&forged, "untampered grammar half-word write");
    forged[neo_wasm::layout::COL_LINEAR_MEM_LANE_VALUE[0]] += neo_math::F::ONE;
    forged[neo_wasm::layout::COL_LINEAR_MEM_LANE0_BYTE0] += neo_math::F::ONE;
    common::assert_rejected(&forged, "grammar half-word write changing an unselected byte");
}

#[test]
fn grammar_memory_replay_authenticates_prior_values() {
    let fixture = import_memory_fixture();
    let byte_write_index = fixture
        .trace
        .iter()
        .position(|row| {
            row.grammar_rom_slot
                .is_some_and(|rom| rom.kind == WasmGrammarSlotKind::MemoryWrite)
                && row
                    .linear_memory
                    .is_some_and(|access| access.width_bytes == 1)
        })
        .expect("byte memory write gather");
    let mut forged_rows: Vec<Vec<neo_math::F>> = fixture.trace.iter().map(build_witness_vector).collect();
    forged_rows[byte_write_index][neo_wasm::layout::COL_LINEAR_MEM_LANE_VALUE_BEFORE[0]] +=
        neo_math::F::from_u64(1 << 16);
    forged_rows[byte_write_index][neo_wasm::layout::COL_LINEAR_MEM_LANE0_BYTE2_BEFORE] += neo_math::F::ONE;

    let artifacts =
        neo_wasm::extract_first_component_core_program_artifacts(&fixture.component_bytes).expect("artifacts");
    let mut preload =
        neo_wasm::memory_semantics::preload_from_program_artifacts(&artifacts, &fixture.run.initial_locals);
    neo_wasm::memory_semantics::preload_grammar_tables(&mut preload, &fixture.grammar);
    let layout = neo_wasm::relation_layout::build_wasm_relation_layout();
    neo_wasm::memory_semantics::sanity_check_memory_rows(layout, &forged_rows, &preload)
        .expect_err("grammar byte write must authenticate its prior word");
}
