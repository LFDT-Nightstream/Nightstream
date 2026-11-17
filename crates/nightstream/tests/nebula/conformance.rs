//! SPEC.md section 9.

use std::ops::Range;

use super::*;
use crate::lifecycle::bounded_depth::{BoundedDepthError, CHECK_ROWS, FAILED_ROW};
use crate::nebula::{Circuit, Machine, MachineState, Package, Step};
use crate::{Engine, Error};
use crate::{Proof, Prover, Verifier};

const EXECUTE: [Step; 2] = [Step::Execute, Step::Execute];

struct Fixture {
    package: Package,
    rows: &'static [u8],
}

const FIRST: Fixture = Fixture {
    package: Package::V1,
    rows: include_bytes!("../fixtures/nebula-memory-v1-rows.json"),
};

const N1: Fixture = Fixture {
    package: Package::N1,
    rows: include_bytes!("../fixtures/nebula-memory-n1-rows.json"),
};

struct Harness {
    context: Context<Machine>,
    prover: Prover,
    verifier: Verifier,
    checks: Vec<(String, Range<usize>)>,
    kept: Vec<usize>,
}

impl Harness {
    fn new(fixture: &Fixture) -> Self {
        let circuit = Circuit::compile(fixture.package.geometry(), Machine).unwrap();
        let named: Vec<(String, usize, usize)> = serde_json::from_slice(fixture.rows).unwrap();
        let application = circuit.circuit.application().unwrap();
        assert_eq!(
            named.last().map(|check| check.2),
            Some(application.row_count()),
            "names cover the rows"
        );
        let records = application.records();
        let first = application.generated_range().start;
        let mut blocked = vec![false; records.row_count()];
        for block in &application.shape().blocks {
            for local in block.first_local..block.first_local + 1096 {
                blocked[records.recipe_row(local - first).unwrap()] = true;
            }
        }
        let kept = (0..records.row_count())
            .filter(|&row| !blocked[row])
            .collect();
        Self {
            context: Context::new(Machine, &fixture.package.program()),
            prover: circuit.circuit.prover(Engine::Optimized, 109).unwrap(),
            verifier: circuit
                .circuit
                .stage1_verifier(Engine::Optimized, 109)
                .unwrap(),
            checks: named
                .into_iter()
                .map(|(name, first, end)| (name, first..end))
                .collect(),
            kept,
        }
    }

    fn step(&self, proof: Option<&Proof>, invocation: &Invocation) -> Result<Proof, Error> {
        match proof {
            None => {
                let z0 = self
                    .context
                    .state(&MachineState::default(), &self.context.start_carry());
                self.prover
                    .prove_with_output(z0, &invocation.words, invocation.output)
            }
            Some(proof) => self
                .prover
                .extend_with_output(proof, &invocation.words, invocation.output),
        }
    }

    fn reject(&self, proof: Option<&Proof>, invocation: &Invocation, check: &str) {
        CHECK_ROWS.set(true);
        FAILED_ROW.set(None);
        let proved = self.step(proof, invocation);
        CHECK_ROWS.set(false);
        let bad = proved.unwrap();
        let row = FAILED_ROW.take().expect("the faulty witness fails a row");
        let named = self.kept.get(row).and_then(|row| {
            self.checks
                .iter()
                .find(|(_, rows)| rows.contains(row))
                .map(|(name, _)| name.as_str())
        });
        assert_eq!(named, Some(check), "first failing row {row}");
        eprintln!("rejected at row {row} ({check})");
        assert!(
            matches!(
                self.verifier.verify(&bad.state(), &bad),
                Err(Error::BoundedDepth(BoundedDepthError::Fresh(_)))
            ),
            "{check}"
        );
    }

    fn first_segment(&self, fault: impl Fn(&mut FirstSegment)) -> Vec<Invocation> {
        let (geometry, program) = (self.context.geometry(), self.context.program());
        let mut segment = FirstSegment {
            execution: initial_memory(&program.rom, &program.ram),
            start: initial_memory(&program.rom, &program.ram),
            executed: Vec::new(),
            proposal_records: None,
            final_: Vec::new(),
        };
        fault(&mut segment);
        let steps = vec![Step::Execute; geometry.n];
        let (executed, final_memory, _) = self
            .context
            .execute_steps(&MachineState::default(), &segment.execution, 0, &steps)
            .unwrap();
        segment.executed = executed;
        segment.final_ = final_memory;
        fault(&mut segment);
        let proposal = self.context.proposal(
            segment
                .proposal_records
                .as_deref()
                .unwrap_or(&segment.executed),
            &segment.final_,
        );
        self.context.invocations(
            self.context.start_carry(),
            proposal,
            &segment.executed,
            &segment.start,
            &segment.final_,
        )
    }
}

struct FirstSegment {
    execution: Vec<Cell>,
    start: Vec<Cell>,
    executed: Vec<Executed>,
    proposal_records: Option<Vec<Executed>>,
    final_: Vec<Cell>,
}

fn conformance() {
    let h = Harness::new(&FIRST);
    let (geometry, program) = (h.context.geometry(), h.context.program());
    let ram = geometry.rom_size();
    let memory = initial_memory(&program.rom, &program.ram);
    let honest = h
        .context
        .run_segment(h.context.start_carry(), &MachineState::default(), &memory, &EXECUTE)
        .unwrap();
    let base = h.step(None, &honest.invocations[0]).unwrap();

    for word in [words::CARRY_IN + 2, h.context.layout.app_in(1)] {
        let mut changed = honest.invocations[0].clone();
        changed.words[word] += F::ONE;
        h.reject(None, &changed, "state_in");
    }

    // SPEC.md section 7.3; Lean `AuxHonest`.
    let layout = &h.context.layout;
    for (word, check) in [
        (layout.start(0, 0), "arm start products"),
        (layout.cnt(0), "O2 slot 0"),
        (layout.ops_factor(0, 1, 2), "O9 slot 0"),
        (layout.scan_factor(0, 0, 2), "S2 slot 0"),
    ] {
        let mut changed = honest.invocations[0].clone();
        changed.words[word] += F::ONE;
        h.reject(None, &changed, check);
    }

    let mut changed = honest.invocations[0].clone();
    changed.words[words::SEEN_PREV] += F::ONE;
    h.reject(None, &changed, "chain_ops");

    let invocations = h.first_segment(|s| {
        if let Some(step) = s.executed.first_mut() {
            step.ops[0].rt = step.write_stamps[0];
        }
    });
    h.reject(None, &invocations[0], "O4 slot 0");

    let invocations = h.first_segment(|s| {
        if let Some(step) = s.executed.first_mut() {
            step.ops[1].vr = 1;
            step.ops[1].vw = 1;
        }
    });
    h.reject(None, &invocations[0], "O7 slot 1");

    let mut changed = honest.invocations[1].clone();
    let (bit0, bit1) = (layout.op(0, layout.op_diff(0)), layout.op(0, layout.op_diff(1)));
    assert_eq!((changed.words[bit0], changed.words[bit1]), (F::ONE, F::ZERO));
    changed.words[bit0] = -F::ONE;
    changed.words[bit1] = F::ONE;
    h.reject(Some(&base), &changed, "O1 diff bits");

    let second = |fault: &dyn Fn(&mut FirstSegment), check: &str| {
        let invocations = h.first_segment(fault);
        let opened = h.step(None, &invocations[0]).unwrap();
        h.reject(Some(&opened), &invocations[1], check);
    };
    second(
        &|s| {
            if let Some(step) = s.executed.get_mut(1) {
                step.ops[1].is_ram = false;
            }
        },
        "O5 slot 1",
    );
    second(
        &|s| {
            if s.executed.len() == 2 && s.proposal_records.is_none() {
                s.proposal_records = Some(s.executed.clone());
                s.executed[1].ops[1].vw = 7;
            }
        },
        "close D_seen.ops = D_pre.ops",
    );
    second(
        &|s| {
            if s.executed.is_empty() {
                s.start[ram + 1].value = 9;
                s.execution[ram + 1].value = 9;
            }
        },
        "close D_seen.is = D_mem",
    );
    second(
        &|s| {
            if s.executed.len() == 2 {
                let cells = ram..ram + geometry.b_scan;
                let start = s.start[cells.clone()].to_vec();
                s.start[cells.clone()].copy_from_slice(&s.final_[cells.clone()]);
                s.final_[cells].copy_from_slice(&start);
            }
        },
        "close D_seen.is = D_mem",
    );
    second(
        &|s| {
            if s.executed.is_empty() {
                s.execution[1].value = 6;
            }
        },
        "close product equation",
    );

    let opened = h.step(Some(&base), &honest.invocations[1]).unwrap();
    let fresh = initial_memory(&program.rom, &program.ram);
    let (executed, final_memory, _) = h
        .context
        .execute_steps(&honest.state, &fresh, honest.carry.ts, &EXECUTE)
        .unwrap();
    let invocations = h.context.invocations(
        honest.carry,
        h.context.proposal(&executed, &final_memory),
        &executed,
        &fresh,
        &final_memory,
    );
    let opened = h.step(Some(&opened), &invocations[0]).unwrap();
    h.reject(Some(&opened), &invocations[1], "close D_seen.is = D_mem");
}

#[test]
#[ignore = "Production-profile memory proofs; run separately under the 300-second cap."]
fn memory_rejections_on_cpu() {
    conformance();
}

/// SPEC.md section 7.4.
#[test]
fn packing_puts_at_most_63_bits_in_one_element() {
    use p3_field::PrimeField64;
    for length in [1, 62, 63, 64, 126, 127] {
        let packed = pack(&vec![true; length]);
        assert_eq!(packed.len(), length.div_ceil(63));
        assert!(packed.iter().all(|word| word.as_canonical_u64() < 1 << 63));
    }
}

fn n1_conformance() {
    let h = Harness::new(&N1);
    let invocations = h.first_segment(|s| {
        if let Some(step) = s.executed.first_mut() {
            step.ops[0].addr = 1 << h.context.geometry().r;
        }
    });
    h.reject(None, &invocations[0], "O6 slot 0");
}

#[test]
#[ignore = "Production-profile memory proofs; run separately under the 300-second cap."]
fn n1_rejections_on_cpu() {
    n1_conformance();
}
