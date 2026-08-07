//! Native expansion of host-event grammar templates: the worked resource
//! method from `docs/host-event-grammar-tables.md` §3.2, pinned against
//! hand-built absorb blocks and `commit_event` folds.
//!
//! The discriminants and slot indices below are EXAMPLE embedder data
//! (mirroring `starstream-interleaving-spec`'s `EffectDiscriminant` and
//! `ArgName::idx`); neo-wasm itself never interprets them.

use neo_wasm::comm_chain::{commit_event, COMM_CHAIN_EVENT_ARGS};
use neo_wasm::event_grammar::{
    expand_export_entry, expand_import_events, ExportTemplate, GrammarEvent, ImportTemplate, Limb, MemoryBase,
    SlotSource,
};
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
    let oracle = |idx| SlotSource::Claim { idx };
    ImportTemplate {
        events: vec![
            // NewRef(size=1) -> payload ref (oracle 0). Ret=slot2, Size=slot3.
            GrammarEvent::op(10, slots(&[(2, oracle(0)), (3, SlotSource::Const(1))])),
            // RefPush([a, b.lo, b.hi, 0]). PackedRef0..3 = slots 0..3.
            GrammarEvent::op(
                11,
                slots(&[(0, arg(1, Limb::Lo)), (1, arg(2, Limb::Lo)), (2, arg(2, Limb::Hi))]),
            ),
            // Resume(target, f_id, val_ref) -> (ret_ref, caller).
            // Target=slot0, Val=slot1, Ret=slot2, Caller=slot3, FunctionId1=slot4.
            GrammarEvent::op(
                0,
                slots(&[
                    (0, oracle(3)),
                    (1, oracle(0)),
                    (2, oracle(1)),
                    (3, oracle(2)),
                    (4, SlotSource::Const(77)), // dummy f_id
                ]),
            ),
            // RefGet(ret_ref, 0) -> [r, 0, 0, 0]: the ResultElem Lo slot is
            // also the row that pushes the host result onto the stack.
            // Val(=ref)=slot1, Offset=slot3, PackedRef0/2/4/5 = slots 0/2/4/5.
            GrammarEvent::op(
                12,
                slots(&[
                    (1, oracle(1)),
                    (0, SlotSource::ResultElem { limb: Limb::Lo }),
                    (2, SlotSource::ResultElem { limb: Limb::Hi }),
                ]),
            ),
        ],
        claim_count: 4,
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

    let blocks = expand_import_events(&template, &args, result, &oracles, &[]).expect("expansion");

    assert_eq!(
        blocks,
        vec![
            [10, 0, 0, 100, 1, 0, 0, 0],       // NewRef
            [11, 5, 7, 3, 0, 0, 0, 0],         // RefPush
            [0, 103, 100, 101, 102, 77, 0, 0], // Resume
            [12, 42, 101, 0, 0, 0, 0, 0],      // RefGet (result push)
        ],
    );

    // The chain fold over the blocks equals the manual commit_event sequence.
    let f = Goldilocks::from_u64;
    let mut chain = [Goldilocks::ZERO; 4];
    for block in &blocks {
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
        events: vec![GrammarEvent::op(7, [ZERO; COMM_CHAIN_EVENT_ARGS])],
        claim_count: 0,
    };
    template.validate(0, 0).expect("burn validates");
    let blocks = expand_import_events(&template, &[], None, &[], &[]).expect("expansion");
    assert_eq!(blocks, vec![[7, 0, 0, 0, 0, 0, 0, 0]]);
}

#[test]
fn validation_rejects_unresolvable_templates() {
    let event = |slot: SlotSource| GrammarEvent::op(0, slots(&[(0, slot)]));

    let result_lo = SlotSource::ResultElem { limb: Limb::Lo };
    let result_hi = SlotSource::ResultElem { limb: Limb::Hi };

    // Arg index beyond the import's arity.
    let template = ImportTemplate {
        events: vec![event(SlotSource::ArgElem { arg: 2, limb: Limb::Lo })],
        ..Default::default()
    };
    assert!(template.validate(2, 0).is_err());

    // Result reference on a resultless import.
    let template = ImportTemplate {
        events: vec![event(result_lo)],
        ..Default::default()
    };
    assert!(template.validate(0, 0).is_err());

    // A returning import MUST push: the ResultElem Lo slot is the push.
    let template = ImportTemplate {
        events: vec![event(SlotSource::Const(1))],
        ..Default::default()
    };
    assert!(template.validate(0, 1).is_err());

    // ... and must push exactly once.
    let template = ImportTemplate {
        events: vec![event(result_lo), event(result_lo)],
        ..Default::default()
    };
    assert!(template.validate(0, 1).is_err());

    // The Hi slot writes the pushed cell's hi lane, so it must follow the
    // Lo slot.
    let template = ImportTemplate {
        events: vec![event(result_hi), event(result_lo)],
        ..Default::default()
    };
    assert!(template.validate(0, 1).is_err());
    let template = ImportTemplate {
        events: vec![event(result_lo), event(result_hi)],
        ..Default::default()
    };
    assert!(template.validate(0, 1).is_ok());

    // ... and is REQUIRED: a Lo-only template leaves the pushed hi lane as
    // unbound advice (an i32 result absorbs 0).
    let template = ImportTemplate {
        events: vec![event(result_lo)],
        ..Default::default()
    };
    assert!(template.validate(0, 1).is_err());

    // Claim index beyond the declared count.
    let template = ImportTemplate {
        events: vec![event(SlotSource::Claim { idx: 1 })],
        claim_count: 1,
    };
    assert!(template.validate(0, 0).is_err());

    // Non-canonical constant.
    let template = ImportTemplate {
        events: vec![event(SlotSource::Const(u64::MAX))],
        ..Default::default()
    };
    assert!(template.validate(0, 0).is_err());

    // Advice events allow only VM effects and padding.
    let advice = |slot: SlotSource| {
        let mut block = [ZERO; 8];
        block[0] = slot;
        GrammarEvent::advice(block)
    };
    let template = ImportTemplate {
        events: vec![advice(result_lo), advice(result_hi)],
        ..Default::default()
    };
    assert!(template.validate(0, 1).is_ok());
    let template = ImportTemplate {
        events: vec![advice(SlotSource::ArgElem { arg: 0, limb: Limb::Lo })],
        ..Default::default()
    };
    assert!(template.validate(1, 0).is_err());
    let template = ImportTemplate {
        events: vec![advice(SlotSource::Claim { idx: 0 })],
        claim_count: 1,
    };
    assert!(template.validate(0, 0).is_err());
    let template = ImportTemplate {
        events: vec![advice(result_lo), advice(result_hi)],
        claim_count: 1,
    };
    assert!(template.validate(0, 1).is_err(), "claims need an absorbing event");
    let template = ExportTemplate {
        entry: vec![GrammarEvent::advice([ZERO; 8])],
        ..Default::default()
    };
    assert!(template.validate(1).is_err(), "export events must absorb");

    // Argument 0 after the result push (its stack slot holds the result).
    let template = ImportTemplate {
        events: vec![event(result_lo), event(SlotSource::ArgElem { arg: 0, limb: Limb::Lo })],
        ..Default::default()
    };
    assert!(template.validate(1, 1).is_err());
    // Later arguments stay addressable after the push.
    let template = ImportTemplate {
        events: vec![
            event(result_lo),
            event(result_hi),
            event(SlotSource::ArgElem { arg: 1, limb: Limb::Lo }),
        ],
        ..Default::default()
    };
    assert!(template.validate(2, 1).is_ok());
}

/// Export entry-phase rules: each local lane written at most once, lo
/// before hi, indices inside the declared claim counts, and every
/// `ClaimLocal` word must fit the 32-bit locals lane.
#[test]
fn export_entry_validation_and_expansion_rules() {
    let event = |slot: SlotSource| GrammarEvent::op(0, slots(&[(0, slot)]));

    // Claim index beyond the phase's declared count.
    let template = ExportTemplate {
        entry: vec![event(SlotSource::Claim { idx: 1 })],
        entry_claim_count: 1,
        ..Default::default()
    };
    assert!(template.validate(1).is_err());

    // Locals bootstrap is entry-phase only.
    let template = ExportTemplate {
        exit: vec![event(SlotSource::ClaimLocal {
            idx: 0,
            local: 0,
            limb: Limb::Lo,
        })],
        exit_claim_count: 1,
        ..Default::default()
    };
    assert!(template.validate(1).is_err());

    // A local lane written twice is rejected.
    let lo = |local| SlotSource::ClaimLocal {
        idx: 0,
        local,
        limb: Limb::Lo,
    };
    let hi = |local| SlotSource::ClaimLocal {
        idx: 1,
        local,
        limb: Limb::Hi,
    };
    let template = ExportTemplate {
        entry: vec![event(lo(0)), event(lo(0))],
        entry_claim_count: 2,
        ..Default::default()
    };
    assert!(template.validate(1).is_err());

    // Local index out of range.
    let template = ExportTemplate {
        entry: vec![event(lo(1))],
        entry_claim_count: 2,
        ..Default::default()
    };
    assert!(template.validate(1).is_err());

    // A hi-lane write requires (and must follow) its local's lo-lane write,
    // because the lo write zeroes the hi lane.
    let template = ExportTemplate {
        entry: vec![event(hi(0))],
        entry_claim_count: 2,
        ..Default::default()
    };
    assert!(template.validate(1).is_err());
    let template = ExportTemplate {
        entry: vec![event(hi(0)), event(lo(0))],
        entry_claim_count: 2,
        ..Default::default()
    };
    assert!(template.validate(1).is_err());
    let template = ExportTemplate {
        entry: vec![event(lo(0)), event(hi(0))],
        entry_claim_count: 2,
        ..Default::default()
    };
    template.validate(1).expect("lo-then-hi validates");

    // Entry expansion resolves indexed claim words — the same index may
    // feed several slots — and rejects a wrong array length or a
    // locals-bound word that does not fit the lane.
    let template = ExportTemplate {
        entry: vec![GrammarEvent::op(
            9,
            slots(&[
                (0, SlotSource::Claim { idx: 1 }),
                (1, lo(0)),
                (2, SlotSource::Claim { idx: 0 }),
            ]),
        )],
        entry_claim_count: 2,
        ..Default::default()
    };
    template.validate(1).expect("entry template validates");
    let blocks = expand_export_entry(&template, &[7, 500]).expect("entry expansion");
    assert_eq!(blocks, vec![[9, 500, 7, 7, 0, 0, 0, 0]]);
    assert!(expand_export_entry(&template, &[7]).is_err());
    assert!(expand_export_entry(&template, &[1 << 32, 500]).is_err());
}

#[test]
fn expansion_rejects_wrong_claim_count() {
    let template = method_template();
    let args = [(0, 0), (0, 0), (0, 0)];
    assert!(expand_import_events(&template, &args, Some((0, 0)), &[1, 2, 3], &[]).is_err());
}

#[test]
fn expansion_rejects_non_canonical_claim() {
    let template = method_template();
    let args = [(0, 0), (0, 0), (0, 0)];
    assert!(expand_import_events(&template, &args, Some((0, 0)), &[1, 2, 3, u64::MAX], &[]).is_err());
}

#[test]
fn memory_slots_validate_phase_base_and_claim_source() {
    let event = |source| GrammarEvent::op(1, slots(&[(0, source)]));
    let import = ImportTemplate {
        events: vec![event(SlotSource::MemoryRead32 {
            base: MemoryBase::Local(0),
            byte_offset: 0,
        })],
        claim_count: 0,
    };
    assert!(import.validate(1, 0).is_err());

    let import = ImportTemplate {
        events: vec![event(SlotSource::MemoryWrite32 {
            claim: 0,
            base: MemoryBase::Arg(0),
            byte_offset: 0,
        })],
        claim_count: 0,
    };
    assert!(import.validate(1, 0).is_err());

    let import = ImportTemplate {
        events: vec![GrammarEvent::op(
            1,
            slots(&[
                (0, SlotSource::ResultElem { limb: Limb::Lo }),
                (1, SlotSource::ResultElem { limb: Limb::Hi }),
                (
                    2,
                    SlotSource::MemoryRead32 {
                        base: MemoryBase::Arg(0),
                        byte_offset: 0,
                    },
                ),
            ]),
        )],
        claim_count: 0,
    };
    assert!(import.validate(1, 1).is_err());

    let export = ExportTemplate {
        entry: vec![event(SlotSource::MemoryRead32 {
            base: MemoryBase::Local(0),
            byte_offset: 0,
        })],
        ..Default::default()
    };
    assert!(export.validate(1).is_err());

    let export = ExportTemplate {
        exit: vec![event(SlotSource::MemoryWrite32 {
            claim: 0,
            base: MemoryBase::Local(0),
            byte_offset: 0,
        })],
        exit_claim_count: 1,
        ..Default::default()
    };
    assert!(export.validate(1).is_err());

    let pointer = SlotSource::ClaimLocal {
        idx: 0,
        local: 0,
        limb: Limb::Lo,
    };
    let write = SlotSource::MemoryWrite32 {
        claim: 1,
        base: MemoryBase::Local(0),
        byte_offset: 0,
    };
    let missing_pointer = ExportTemplate {
        entry: vec![event(write)],
        entry_claim_count: 2,
        ..Default::default()
    };
    assert!(missing_pointer.validate(1).is_err());

    let late_pointer = ExportTemplate {
        entry: vec![event(write), event(pointer)],
        entry_claim_count: 2,
        ..Default::default()
    };
    assert!(late_pointer.validate(1).is_err());

    let ordered = ExportTemplate {
        entry: vec![event(pointer), event(write)],
        entry_claim_count: 2,
        ..Default::default()
    };
    ordered
        .validate(1)
        .expect("pointer bootstrap precedes memory write");

    let byte_write = ImportTemplate {
        events: vec![event(SlotSource::MemoryWrite8 {
            claim: 0,
            base: MemoryBase::Arg(0),
            byte_offset: 0,
        })],
        claim_count: 1,
    };
    assert!(expand_import_events(&byte_write, &[(0, 0)], None, &[256], &[]).is_err());

    let half_write = ImportTemplate {
        events: vec![event(SlotSource::MemoryWrite16 {
            claim: 0,
            base: MemoryBase::Arg(0),
            byte_offset: 0,
        })],
        claim_count: 1,
    };
    assert!(expand_import_events(&half_write, &[(0, 0)], None, &[1 << 16], &[]).is_err());
}
