use std::convert::Infallible;

use p3_field::PrimeField64;

use super::*;
use crate::nebula::{
    circuit::memory_circuit, words, Application, Circuit, Geometry, Memory, PortAccess, Program, Run, Transition,
};

#[derive(Clone, Copy, Debug, Default)]
enum Fault {
    #[default]
    None,
    Coefficient,
    RowStart,
    AuxStart,
    RowOverlap,
    AuxOverlap,
    InputLane,
    OutputLane,
    Carry,
    Eta,
    Cycle,
    DeadRead,
    PriorDeadInput,
}

#[derive(Clone, Copy)]
struct PortHash {
    optimized: bool,
    fault: Fault,
}

impl PortHash {
    fn honest(optimized: bool) -> Self {
        Self {
            optimized,
            fault: Fault::None,
        }
    }

    fn source(&self, view: &View) -> (Vec<Expr>, Vec<Poseidon2Hint>) {
        let reference = reference().unwrap();
        let mut rows = Vec::new();
        let mut hints = Vec::new();
        let mut lanes: [Expr; WIDTH] = std::array::from_fn(|_| Expr::constant(0));
        lanes[0] = Expr::constant(0x4e53504f52544831);
        lanes[1] = view.vr(0) + Expr::constant(2) * view.vw(0);
        for block in 0..2 {
            if block == 1 && matches!(self.fault, Fault::PriorDeadInput) {
                lanes[1] = view.aux(0);
            }
            let aux: Vec<_> = (0..WORDS)
                .map(|i| variable(view.aux(block * WORDS + i)).unwrap())
                .collect();
            let input = std::array::from_fn(|i| lower_affine(&lanes[i]).unwrap());
            hints.push(Poseidon2Hint {
                row_start: rows.len(),
                aux_start: block * WORDS,
                lanes: lanes.clone(),
            });
            let expr = |value: &Affine| {
                value
                    .terms()
                    .iter()
                    .fold(Expr::Const(value.constant_term()), |sum, (v, c)| {
                        sum + Expr::Const(*c) * Expr::Var(*v)
                    })
            };
            for row in 0..WORDS {
                let [a, b, c] = expected_row(&reference, row, &input, &aux).unwrap();
                rows.push(expr(&c) - expr(&a) * expr(&b));
            }
            lanes = std::array::from_fn(|i| view.aux(block * WORDS + 1080 + i));
            if block == 0 {
                lanes[0] = lanes[0].clone() + Expr::constant(1);
            }
        }
        for (i, lane) in lanes.iter().take(4).enumerate() {
            rows.push(view.output(i) - lane.clone());
        }
        rows.extend([
            view.pad(0),
            view.is_write(0),
            view.is_ram(0) - Expr::constant(1),
            view.addr(0),
            view.vw(0) - view.vr(0),
        ]);
        match self.fault {
            Fault::Coefficient => rows[19] = rows[19].clone() + Expr::constant(1),
            Fault::RowStart => hints[0].row_start += 1,
            Fault::AuxStart => hints[0].aux_start += 1,
            Fault::RowOverlap => hints[1].row_start -= 1,
            Fault::AuxOverlap => hints[1].aux_start -= 1,
            Fault::InputLane => hints[0].lanes.swap(0, 1),
            Fault::OutputLane => rows[WORDS - 1] = view.aux(WORDS - 2) - Expr::constant(1) * Expr::constant(1),
            Fault::Carry => hints[0].lanes[0] = Expr::Var(Variable(4 + words::CARRY_IN)),
            Fault::Eta => hints[0].lanes[0] = Expr::Var(Variable(4 + words::ETA_FRESH)),
            Fault::Cycle => hints[0].lanes[0] = view.aux(WORDS + 1080),
            Fault::DeadRead => rows.push(view.aux(0)),
            Fault::None | Fault::PriorDeadInput => (),
        }
        (rows, hints)
    }
}

impl Application for PortHash {
    type Input = ();
    type State = [u64; 4];
    type Error = Infallible;
    fn width(&self) -> usize {
        4
    }
    fn aux(&self) -> usize {
        2 * WORDS
    }
    fn port_count(&self) -> usize {
        1
    }
    fn words(&self, state: &Self::State) -> Vec<u64> {
        state.to_vec()
    }
    fn rows(&self, view: &View) -> Result<Vec<Expr>, crate::Error> {
        Ok(self.source(view).0)
    }
    fn poseidon2_hints(&self, view: &View) -> Result<Vec<Poseidon2Hint>, crate::Error> {
        Ok(if self.optimized {
            self.source(view).1
        } else {
            Vec::new()
        })
    }
    fn step(&self, _: &Self::State, _: &(), memory: &Memory) -> Result<Transition<Self::State>, Infallible> {
        let value = memory.ram(0).unwrap();
        let reference = reference().unwrap();
        let mut lanes = [F::ZERO; WIDTH];
        lanes[0] = F::from_u64(0x4e53504f52544831);
        lanes[1] = F::from_u64(value) * F::from_u64(3);
        let mut aux = Vec::new();
        for block in 0..2 {
            let witness = reference.execute([F::ZERO; 4], &lanes).unwrap();
            let trace = &witness.values()[reference.generated_range()];
            aux.extend_from_slice(trace);
            lanes.copy_from_slice(&trace[1080..]);
            if block == 0 {
                lanes[0] += F::ONE;
            }
        }
        Ok(Transition {
            next: std::array::from_fn(|i| lanes[i].as_canonical_u64()),
            ports: vec![Some(PortAccess {
                is_write: false,
                is_ram: true,
                addr: 0,
                vr: value,
                vw: value,
            })],
            aux,
        })
    }
}

fn geometry() -> Geometry {
    Geometry::new(2, 2, 1, 2, 2).unwrap()
}

#[test]
fn checked_port_hash_reconstructs_source_witness() {
    let geometry = geometry();
    let original = memory_circuit(&geometry, &PortHash::honest(false)).unwrap();
    let optimized = memory_circuit(&geometry, &PortHash::honest(true)).unwrap();
    crate::application::check_blocks(&optimized).unwrap();
    assert_eq!(optimized.row_count(), original.row_count());
    assert_eq!(optimized.shape().blocks.len(), original.shape().blocks.len() + 2);
    let image = Program::new(geometry, vec![0; 4], vec![12345, 0, 0, 0]).unwrap();
    let mut run = Run::new(PortHash::honest(true), &image, [0; 4]);
    let mut prior = run.initial_state();
    let first_app_block = original.shape().blocks.len();
    let layout = words::Layout::new(&geometry, 4, 2 * WORDS);
    for invocation in run.segment(&[(), ()]).unwrap() {
        original.execute(prior, &invocation.words).unwrap();
        let mut erased = invocation.words.clone();
        for word in &mut erased[layout.aux(0)..layout.aux(2 * WORDS)] {
            *word = F::ZERO;
        }
        assert!(original.execute(prior, &erased).is_err());
        let accepted = optimized.execute(prior, &erased).unwrap();
        assert_eq!(accepted.output_state(), invocation.output);
        for (index, block) in optimized.shape().blocks[first_app_block..]
            .iter()
            .enumerate()
        {
            let values = &accepted.values()[block.first_local..block.first_local + WORDS];
            erased[layout.aux(index * WORDS)..layout.aux((index + 1) * WORDS)].copy_from_slice(values);
        }
        assert_eq!(erased, invocation.words);
        original.execute(prior, &erased).unwrap();
        prior = invocation.output;
    }
    let expected = neo_ccs::crypto::poseidon2_goldilocks::poseidon2_hash(&[
        F::from_u64(0x4e53504f52544831),
        F::from_u64(3 * 12345),
    ])
    .map(|value| value.as_canonical_u64());
    assert_eq!(*run.state(), expected);
}

#[test]
fn checked_port_hash_rejects_malformed_hints() {
    for fault in [
        Fault::Coefficient,
        Fault::RowStart,
        Fault::AuxStart,
        Fault::RowOverlap,
        Fault::AuxOverlap,
        Fault::InputLane,
        Fault::OutputLane,
        Fault::Carry,
        Fault::Eta,
        Fault::Cycle,
    ] {
        assert!(
            memory_circuit(&geometry(), &PortHash { optimized: true, fault }).is_err(),
            "{fault:?}"
        );
    }
    for fault in [Fault::DeadRead, Fault::PriorDeadInput] {
        assert!(
            Circuit::compile(geometry(), PortHash { optimized: true, fault }).is_err(),
            "{fault:?}"
        );
    }
}

/// The two hints make two native blocks (`SPEC.md` section 7.7). The final
/// program then checks each block as one permutation in place of its 1,096
/// source rows (`CompressionLean.BoundedDepth.finishRows`). The plan rows do
/// not change: section 7.11 lowers each unhinted trace to one seventh-power
/// row for each of its 150 S-boxes (`Export.Generic.NativePower.Chain.sound`)
/// and substitutes its affine rows, and a block has 150 S-box rows.
#[test]
fn checked_port_hash_compiles_to_two_native_blocks() {
    let compile = |optimized| {
        let application = memory_circuit(&geometry(), &PortHash::honest(optimized)).unwrap();
        crate::application::check_blocks(&application).unwrap();
        let bytes = crate::nebula::test_packages::bounded_depth_packages("");
        nightstream_fprime::load_bounded_depth_packages(
            &bytes[0],
            &bytes[1],
            nightstream_fprime::BOUNDED_DEPTH_PACKAGES_IDENTITY,
        )
        .unwrap()
        .with_application(
            application.records().clone(),
            application.private_input_count(),
            application.shape(),
        )
        .unwrap()
    };
    let old = compile(false);
    let new = compile(true);
    for (name, old, new) in [
        ("light", old.light(), new.light()),
        ("refresh", old.refresh(), new.refresh()),
    ] {
        let old_rows = old.plan_rows().unwrap();
        let new_rows = new.plan_rows().unwrap();
        assert_eq!(new.block_count() - old.block_count(), 2);
        assert_eq!(
            old.source_rows().unwrap().ordinary_row_count() - new.source_rows().unwrap().ordinary_row_count(),
            2 * 1096
        );
        assert_eq!(old_rows.row_count(), new_rows.row_count());
        eprintln!("port_hash {name} before_rows={} after_rows={} before_entries={} after_entries={} before_coordinates={} after_coordinates={}",
            old_rows.row_count(), new_rows.row_count(), old_rows.entry_count(), new_rows.entry_count(),
            old.layout().unwrap().width(), new.layout().unwrap().width());
    }
}
