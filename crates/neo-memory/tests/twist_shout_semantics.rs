//! Integration tests for Twist and Shout semantic checkers.
//!
//! These tests verify that the full pipeline from VM trace → plain trace → Ajtai encoding
//! produces witnesses that pass the semantic checks.
//!
//! Note: With index-bit addressing, we now commit bit-decomposed addresses instead of one-hot vectors.

use neo_ccs::matrix::Mat;
use neo_memory::ajtai::decode_vector as ajtai_decode_vector;
use neo_memory::encode::{encode_lut_for_shout, encode_mem_for_twist};
use neo_memory::plain::{build_plain_lut_traces, build_plain_mem_traces, LutTable, PlainMemLayout};
use neo_memory::shout::check_shout_semantics;
use neo_memory::twist::check_twist_semantics;
use neo_params::NeoParams;
use neo_vm_trace::{ShoutEvent, ShoutId, StepTrace, TwistEvent, TwistId, TwistOpKind, VmTrace};
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks;
use std::collections::HashMap;

/// Create a simple NeoParams for testing.
///
/// Note: The dimension `d` in NeoParams must match the `d` in PlainMemLayout
/// for the encoding to work correctly.
fn test_params() -> NeoParams {
    NeoParams {
        q: 0xFFFFFFFF00000001, // Goldilocks modulus
        eta: 8,
        d: 1, // 1 Ajtai digit (must match layout.d)
        kappa: 1,
        m: 16,
        b: 256, // base 256 (single digit can represent 0-255)
        k_rho: 2,
        B: 65536, // b^k_rho = 256^2
        T: 4,
        s: 2,
        lambda: 128,
    }
}

/// Dummy commit function that returns unit.
fn dummy_commit(_mat: &Mat<Goldilocks>) -> () {
    ()
}

/// Build a simple VM trace with reads and writes to a single memory.
fn build_simple_memory_trace() -> VmTrace<u64, u64> {
    // Trace:
    // Step 0: Write 42 to addr 0
    // Step 1: Read from addr 0 (should get 42)
    // Step 2: Write 100 to addr 1
    // Step 3: Read from addr 1 (should get 100)
    // Step 4: Write 50 to addr 0 (overwrite)
    // Step 5: Read from addr 0 (should get 50)
    // Step 6: Read from addr 1 (should get 100)
    // Step 7: No-op (no memory access)

    let mem_id = TwistId(0);

    let steps = vec![
        // Step 0: Write 42 to addr 0
        StepTrace {
            cycle: 0,
            pc_before: 0,
            pc_after: 1,
            opcode: 1, // store
            regs_before: vec![0],
            regs_after: vec![0],
            twist_events: vec![TwistEvent {
                twist_id: mem_id,
                kind: TwistOpKind::Write,
                addr: 0,
                value: 42,
            }],
            shout_events: vec![],
            halted: false,
        },
        // Step 1: Read from addr 0
        StepTrace {
            cycle: 1,
            pc_before: 1,
            pc_after: 2,
            opcode: 2, // load
            regs_before: vec![0],
            regs_after: vec![42],
            twist_events: vec![TwistEvent {
                twist_id: mem_id,
                kind: TwistOpKind::Read,
                addr: 0,
                value: 42, // What the VM claims to have read
            }],
            shout_events: vec![],
            halted: false,
        },
        // Step 2: Write 100 to addr 1
        StepTrace {
            cycle: 2,
            pc_before: 2,
            pc_after: 3,
            opcode: 1,
            regs_before: vec![42],
            regs_after: vec![42],
            twist_events: vec![TwistEvent {
                twist_id: mem_id,
                kind: TwistOpKind::Write,
                addr: 1,
                value: 100,
            }],
            shout_events: vec![],
            halted: false,
        },
        // Step 3: Read from addr 1
        StepTrace {
            cycle: 3,
            pc_before: 3,
            pc_after: 4,
            opcode: 2,
            regs_before: vec![42],
            regs_after: vec![100],
            twist_events: vec![TwistEvent {
                twist_id: mem_id,
                kind: TwistOpKind::Read,
                addr: 1,
                value: 100,
            }],
            shout_events: vec![],
            halted: false,
        },
        // Step 4: Write 50 to addr 0 (overwrite)
        StepTrace {
            cycle: 4,
            pc_before: 4,
            pc_after: 5,
            opcode: 1,
            regs_before: vec![100],
            regs_after: vec![100],
            twist_events: vec![TwistEvent {
                twist_id: mem_id,
                kind: TwistOpKind::Write,
                addr: 0,
                value: 50,
            }],
            shout_events: vec![],
            halted: false,
        },
        // Step 5: Read from addr 0 (should get 50)
        StepTrace {
            cycle: 5,
            pc_before: 5,
            pc_after: 6,
            opcode: 2,
            regs_before: vec![100],
            regs_after: vec![50],
            twist_events: vec![TwistEvent {
                twist_id: mem_id,
                kind: TwistOpKind::Read,
                addr: 0,
                value: 50,
            }],
            shout_events: vec![],
            halted: false,
        },
        // Step 6: Read from addr 1 (should get 100)
        StepTrace {
            cycle: 6,
            pc_before: 6,
            pc_after: 7,
            opcode: 2,
            regs_before: vec![50],
            regs_after: vec![100],
            twist_events: vec![TwistEvent {
                twist_id: mem_id,
                kind: TwistOpKind::Read,
                addr: 1,
                value: 100,
            }],
            shout_events: vec![],
            halted: false,
        },
        // Step 7: No-op
        StepTrace {
            cycle: 7,
            pc_before: 7,
            pc_after: 8,
            opcode: 0,
            regs_before: vec![100],
            regs_after: vec![100],
            twist_events: vec![],
            shout_events: vec![],
            halted: true,
        },
    ];

    VmTrace { steps }
}

#[test]
fn test_twist_semantic_check_passes_for_valid_trace() {
    let params = test_params();
    let trace = build_simple_memory_trace();

    // Memory layout: 4 cells, d=1 dimension, n_side=4
    let mut layouts = HashMap::new();
    layouts.insert(0, PlainMemLayout { k: 4, d: 1, n_side: 4 });

    // No initial memory values (all zero)
    let initial_mem: HashMap<(u32, u64), Goldilocks> = HashMap::new();

    // Build plain memory trace
    let plain_traces = build_plain_mem_traces::<Goldilocks>(&trace, &layouts, &initial_mem);
    let plain_trace = plain_traces.get(&0).expect("Missing memory trace");

    // Encode to Ajtai (now with index-bit addressing)
    // Use legacy mode (None for ccs_m) since we don't have a CCS structure in this test
    let mem_init = neo_memory::MemInit::Zero;
    let (inst, wit) = encode_mem_for_twist(&params, &layouts[&0], &mem_init, plain_trace, &dummy_commit, None, 0);

    // Check semantics using the new API
    let result = check_twist_semantics(&params, &inst, &wit);

    assert!(result.is_ok(), "Semantic check failed: {:?}", result.err());
}

#[test]
fn test_twist_detects_bad_read_value() {
    let params = test_params();

    // Create a trace where the read value is wrong
    let mem_id = TwistId(0);
    let trace = VmTrace {
        steps: vec![
            // Write 42 to addr 0
            StepTrace {
                cycle: 0,
                pc_before: 0,
                pc_after: 1,
                opcode: 1,
                regs_before: vec![],
                regs_after: vec![],
                twist_events: vec![TwistEvent {
                    twist_id: mem_id,
                    kind: TwistOpKind::Write,
                    addr: 0,
                    value: 42,
                }],
                shout_events: vec![],
                halted: false,
            },
            // Read from addr 0 but claim wrong value (99 instead of 42)
            StepTrace {
                cycle: 1,
                pc_before: 1,
                pc_after: 2,
                opcode: 2,
                regs_before: vec![],
                regs_after: vec![],
                twist_events: vec![TwistEvent {
                    twist_id: mem_id,
                    kind: TwistOpKind::Read,
                    addr: 0,
                    value: 99, // WRONG! Should be 42
                }],
                shout_events: vec![],
                halted: true,
            },
        ],
    };

    let mut layouts = HashMap::new();
    layouts.insert(0, PlainMemLayout { k: 4, d: 1, n_side: 4 });
    let initial_mem: HashMap<(u32, u64), Goldilocks> = HashMap::new();

    let plain_traces = build_plain_mem_traces::<Goldilocks>(&trace, &layouts, &initial_mem);
    let plain_trace = plain_traces.get(&0).unwrap();

    // Use legacy mode (None for ccs_m) since we don't have a CCS structure in this test.
    // Keep init consistent with `initial_mem` so the failure is specifically the bad read value.
    let mem_init = neo_memory::MemInit::Zero;
    let (inst, wit) = encode_mem_for_twist(&params, &layouts[&0], &mem_init, plain_trace, &dummy_commit, None, 0);

    let result = check_twist_semantics(&params, &inst, &wit);

    assert!(result.is_err(), "Should detect bad read value");
    let err_msg = format!("{:?}", result.err().unwrap());
    assert!(
        err_msg.contains("read mismatch"),
        "Error should mention read mismatch: {}",
        err_msg
    );
}

#[test]
fn test_twist_with_initial_memory() {
    let params = test_params();

    // Create a trace that reads from pre-initialized memory
    let mem_id = TwistId(0);
    let trace = VmTrace {
        steps: vec![
            // Read from addr 0 (should get initial value 123)
            StepTrace {
                cycle: 0,
                pc_before: 0,
                pc_after: 1,
                opcode: 2,
                regs_before: vec![],
                regs_after: vec![],
                twist_events: vec![TwistEvent {
                    twist_id: mem_id,
                    kind: TwistOpKind::Read,
                    addr: 0,
                    value: 123,
                }],
                shout_events: vec![],
                halted: true,
            },
        ],
    };

    let mut layouts = HashMap::new();
    layouts.insert(0, PlainMemLayout { k: 4, d: 1, n_side: 4 });

    // Initialize memory: addr 0 = 123
    let mut initial_mem = HashMap::new();
    initial_mem.insert((0, 0), Goldilocks::from_u64(123));

    let plain_traces = build_plain_mem_traces::<Goldilocks>(&trace, &layouts, &initial_mem);
    let plain_trace = plain_traces.get(&0).unwrap();

    // Use legacy mode (None for ccs_m) since we don't have a CCS structure in this test
    let mem_init = neo_memory::MemInit::Sparse(vec![(0, Goldilocks::from_u64(123))]);
    let (inst, wit) = encode_mem_for_twist(&params, &layouts[&0], &mem_init, plain_trace, &dummy_commit, None, 0);

    let result = check_twist_semantics(&params, &inst, &wit);

    // With MemInit properly propagated, this should now pass!
    assert!(
        result.is_ok(),
        "Twist semantic check should pass with initial memory: {:?}",
        result
    );
}

#[test]
fn test_ajtai_encode_decode_roundtrip() {
    let params = test_params();

    // Create a simple vector with values in the range [0, b/2) to avoid balanced decomposition issues
    // With b=256 and balanced decomposition, values > 128 get negative digits
    // For d=1, we need values that fit in a single digit without wrapping
    let original: Vec<Goldilocks> = vec![
        Goldilocks::from_u64(0),
        Goldilocks::from_u64(1),
        Goldilocks::from_u64(42),
        Goldilocks::from_u64(100),
        Goldilocks::from_u64(127), // Max safe value for balanced base-256
    ];

    // Encode
    let mat = neo_memory::encode::ajtai_encode_vector(&params, &original);

    // Decode
    let decoded = ajtai_decode_vector(&params, &mat);

    // Check roundtrip
    assert_eq!(original.len(), decoded.len());
    for (i, (orig, dec)) in original.iter().zip(decoded.iter()).enumerate() {
        assert_eq!(*orig, *dec, "Mismatch at index {}: {:?} vs {:?}", i, orig, dec);
    }
}

#[test]
fn test_shout_semantic_check_passes_for_valid_lookups() {
    let params = test_params();

    // Create a lookup table: Table[0] = 10, Table[1] = 20, Table[2] = 30, Table[3] = 40
    let table = LutTable {
        table_id: 0,
        k: 4,
        d: 1,
        n_side: 4,
        content: vec![
            Goldilocks::from_u64(10),
            Goldilocks::from_u64(20),
            Goldilocks::from_u64(30),
            Goldilocks::from_u64(40),
        ],
    };

    // Create a trace with lookups
    let shout_id = ShoutId(0);
    let trace = VmTrace {
        steps: vec![
            // Lookup Table[0] = 10
            StepTrace {
                cycle: 0,
                pc_before: 0,
                pc_after: 1,
                opcode: 3, // lookup
                regs_before: vec![],
                regs_after: vec![],
                twist_events: vec![],
                shout_events: vec![ShoutEvent {
                    shout_id,
                    key: 0,
                    value: 10,
                }],
                halted: false,
            },
            // Lookup Table[2] = 30
            StepTrace {
                cycle: 1,
                pc_before: 1,
                pc_after: 2,
                opcode: 3,
                regs_before: vec![],
                regs_after: vec![],
                twist_events: vec![],
                shout_events: vec![ShoutEvent {
                    shout_id,
                    key: 2,
                    value: 30,
                }],
                halted: false,
            },
            // No lookup
            StepTrace {
                cycle: 2,
                pc_before: 2,
                pc_after: 3,
                opcode: 0,
                regs_before: vec![],
                regs_after: vec![],
                twist_events: vec![],
                shout_events: vec![],
                halted: false,
            },
            // Lookup Table[1] = 20
            StepTrace {
                cycle: 3,
                pc_before: 3,
                pc_after: 4,
                opcode: 3,
                regs_before: vec![],
                regs_after: vec![],
                twist_events: vec![],
                shout_events: vec![ShoutEvent {
                    shout_id,
                    key: 1,
                    value: 20,
                }],
                halted: true,
            },
        ],
    };

    // Build plain lookup trace
    let mut table_sizes = HashMap::new();
    table_sizes.insert(0, (4, 1)); // k=4, d=1

    let plain_traces = build_plain_lut_traces::<Goldilocks>(&trace, &table_sizes);
    let plain_trace = plain_traces.get(&0).unwrap();

    // Encode to Ajtai (now with index-bit addressing)
    // Use legacy mode (None for ccs_m) since we don't have a CCS structure in this test
    let (inst, wit) = encode_lut_for_shout(&params, &table, plain_trace, &dummy_commit, None, 0);

    // Check semantics using the new API
    let result = check_shout_semantics(&params, &inst, &wit, &plain_trace.val);

    assert!(result.is_ok(), "Semantic check failed: {:?}", result.err());
}

#[test]
#[cfg_attr(debug_assertions, should_panic(expected = "lookup mismatch"))]
#[allow(unreachable_code)]
fn test_shout_detects_bad_lookup_value() {
    // This test only panics in debug mode because the semantic check
    // is under #[cfg(debug_assertions)] in encode_lut_for_shout.
    // In release mode, we skip the test entirely.
    #[cfg(not(debug_assertions))]
    return;
    let params = test_params();

    // Create a lookup table
    let table = LutTable {
        table_id: 0,
        k: 4,
        d: 1,
        n_side: 4,
        content: vec![
            Goldilocks::from_u64(10),
            Goldilocks::from_u64(20),
            Goldilocks::from_u64(30),
            Goldilocks::from_u64(40),
        ],
    };

    // Create a trace with a wrong lookup value
    // Note: Using key=1 instead of key=0 because with index-bit addressing,
    // address 0 (all bits zero) is indistinguishable from "no lookup"
    let shout_id = ShoutId(0);
    let trace = VmTrace {
        steps: vec![StepTrace {
            cycle: 0,
            pc_before: 0,
            pc_after: 1,
            opcode: 3,
            regs_before: vec![],
            regs_after: vec![],
            twist_events: vec![],
            shout_events: vec![ShoutEvent {
                shout_id,
                key: 1,     // Lookup at address 1
                value: 999, // WRONG! Table[1] = 20, not 999
            }],
            halted: true,
        }],
    };

    let mut table_sizes = HashMap::new();
    table_sizes.insert(0, (4, 1));

    let plain_traces = build_plain_lut_traces::<Goldilocks>(&trace, &table_sizes);
    let plain_trace = plain_traces.get(&0).unwrap();

    // In debug mode, encoding now runs the semantic checker and will panic on bad values.
    // This is correct behavior - fail fast during encoding rather than later.
    // Use legacy mode (None for ccs_m) since we don't have a CCS structure in this test
    let (_inst, _wit) = encode_lut_for_shout(&params, &table, plain_trace, &dummy_commit, None, 0);
}
