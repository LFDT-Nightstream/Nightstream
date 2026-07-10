//! Native expansion of host-event grammar templates: the worked resource
//! method from `docs/host-event-grammar-tables.md` §3.2, pinned against
//! hand-built absorb blocks and `commit_event` folds.
//!
//! The discriminants and slot indices below are EXAMPLE embedder data
//! (mirroring `starstream-interleaving-spec`'s `EffectDiscriminant` and
//! `ArgName::idx`); neo-wasm itself never interprets them.

use neo_wasm::comm_chain::{commit_event, COMM_CHAIN_EVENT_ARGS};
use neo_wasm::event_grammar::{expand_import_events, GrammarEvent, ImportTemplate, Limb, SlotSource};
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks;

const ZERO: SlotSource = SlotSource::Const(0);

fn slots(entries: &[(usize, SlotSource)]) -> [SlotSource; COMM_CHAIN_EVENT_ARGS] {
    let mut out = [ZERO; COMM_CHAIN_EVENT_ARGS];
    for &(idx, source) in entries {
        out[idx] = source;
    }
    out
}

/// `foo(handle, a: u32, b: u64) -> u32` from the design note: payload tuple
/// `(a, b)` encodes to one ref word `[a, b.lo, b.hi, 0]`.
fn method_template() -> ImportTemplate {
    let arg = |arg, limb| SlotSource::ArgElem { arg, limb };
    let oracle = |idx| SlotSource::Oracle { idx };
    ImportTemplate {
        pre_result: vec![
            // NewRef(size=1) -> payload ref (oracle 0). Ret=slot2, Size=slot3.
            GrammarEvent {
                discriminant: 10,
                slots: slots(&[(2, oracle(0)), (3, SlotSource::Const(1))]),
            },
            // RefPush([a, b.lo, b.hi, 0]). PackedRef0..3 = slots 0..3.
            GrammarEvent {
                discriminant: 11,
                slots: slots(&[(0, arg(1, Limb::Lo)), (1, arg(2, Limb::Lo)), (2, arg(2, Limb::Hi))]),
            },
            // Resume(target, f_id, val_ref) -> (ret_ref, caller).
            // Target=slot0, Val=slot1, Ret=slot2, Caller=slot3, FunctionId1=slot4.
            GrammarEvent {
                discriminant: 0,
                slots: slots(&[
                    (0, oracle(3)),
                    (1, oracle(0)),
                    (2, oracle(1)),
                    (3, oracle(2)),
                    (4, SlotSource::Const(77)), // dummy f_id
                ]),
            },
        ],
        post_result: vec![
            // RefGet(ret_ref, 0) -> [r, 0, 0, 0].
            // Val(=ref)=slot1, Offset=slot3, PackedRef0/2/4/5 = slots 0/2/4/5.
            GrammarEvent {
                discriminant: 12,
                slots: slots(&[(1, oracle(1)), (0, SlotSource::ResultElem { limb: Limb::Lo })]),
            },
        ],
        oracle_count: 4,
    }
}

#[test]
fn method_template_expands_to_pinned_blocks_and_chain() {
    let template = method_template();
    template
        .validate(3, 1)
        .expect("template validates against arity (3, 1)");

    // handle = 0xAA, a = 5, b = 3·2^32 + 7, result = 42.
    let args = [(0xAA, 0), (5, 0), (7, 3)];
    let result = Some((42, 0));
    let oracles = [100u64, 101, 102, 103]; // payload ref, ret ref, caller, target

    let expanded = expand_import_events(&template, &args, result, &oracles).expect("expansion");

    assert_eq!(
        expanded.pre_result_blocks,
        vec![
            [10, 0, 0, 100, 1, 0, 0, 0],       // NewRef
            [11, 5, 7, 3, 0, 0, 0, 0],         // RefPush
            [0, 103, 100, 101, 102, 77, 0, 0], // Resume
        ],
    );
    assert_eq!(expanded.post_result_blocks, vec![[12, 42, 101, 0, 0, 0, 0, 0]]);

    // The chain fold over the blocks equals the manual commit_event sequence.
    let f = Goldilocks::from_u64;
    let mut chain = [Goldilocks::ZERO; 4];
    for block in expanded
        .pre_result_blocks
        .iter()
        .chain(&expanded.post_result_blocks)
    {
        let words: [Goldilocks; COMM_CHAIN_EVENT_ARGS] = core::array::from_fn(|i| f(block[1 + i]));
        chain = commit_event(chain, f(block[0]), words);
    }
    let expected = {
        let mut c = [Goldilocks::ZERO; 4];
        c = commit_event(c, f(10), [f(0), f(0), f(100), f(1), f(0), f(0), f(0)]);
        c = commit_event(c, f(11), [f(5), f(7), f(3), f(0), f(0), f(0), f(0)]);
        c = commit_event(c, f(0), [f(103), f(100), f(101), f(102), f(77), f(0), f(0)]);
        commit_event(c, f(12), [f(42), f(101), f(0), f(0), f(0), f(0), f(0)])
    };
    assert_eq!(chain, expected);
}

#[test]
fn zero_arg_import_expands_to_single_const_event() {
    // `burn()`: one event, all slots constant.
    let template = ImportTemplate {
        pre_result: vec![GrammarEvent {
            discriminant: 7,
            slots: [ZERO; COMM_CHAIN_EVENT_ARGS],
        }],
        post_result: vec![],
        oracle_count: 0,
    };
    template.validate(0, 0).expect("burn validates");
    let expanded = expand_import_events(&template, &[], None, &[]).expect("expansion");
    assert_eq!(expanded.pre_result_blocks, vec![[7, 0, 0, 0, 0, 0, 0, 0]]);
    assert!(expanded.post_result_blocks.is_empty());
}

#[test]
fn validation_rejects_unresolvable_templates() {
    let event = |slot: SlotSource| GrammarEvent {
        discriminant: 0,
        slots: slots(&[(0, slot)]),
    };

    // Arg index beyond the import's arity.
    let template = ImportTemplate {
        pre_result: vec![event(SlotSource::ArgElem { arg: 2, limb: Limb::Lo })],
        ..Default::default()
    };
    assert!(template.validate(2, 0).is_err());

    // Result reference on a resultless import.
    let template = ImportTemplate {
        post_result: vec![event(SlotSource::ResultElem { limb: Limb::Lo })],
        ..Default::default()
    };
    assert!(template.validate(0, 0).is_err());

    // Result reference before the result exists.
    let template = ImportTemplate {
        pre_result: vec![event(SlotSource::ResultElem { limb: Limb::Lo })],
        ..Default::default()
    };
    assert!(template.validate(0, 1).is_err());

    // Oracle index beyond the declared count.
    let template = ImportTemplate {
        pre_result: vec![event(SlotSource::Oracle { idx: 1 })],
        oracle_count: 1,
        ..Default::default()
    };
    assert!(template.validate(0, 0).is_err());

    // Non-canonical constant.
    let template = ImportTemplate {
        pre_result: vec![event(SlotSource::Const(u64::MAX))],
        ..Default::default()
    };
    assert!(template.validate(0, 0).is_err());
}

#[test]
fn expansion_rejects_wrong_oracle_count() {
    let template = method_template();
    let args = [(0, 0), (0, 0), (0, 0)];
    assert!(expand_import_events(&template, &args, Some((0, 0)), &[1, 2, 3]).is_err());
}

#[test]
fn expansion_rejects_non_canonical_oracle() {
    let template = method_template();
    let args = [(0, 0), (0, 0), (0, 0)];
    assert!(expand_import_events(&template, &args, Some((0, 0)), &[1, 2, 3, u64::MAX]).is_err());
}
