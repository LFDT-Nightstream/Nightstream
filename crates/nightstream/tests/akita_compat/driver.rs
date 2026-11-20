use std::{convert::Infallible, env, error::Error, fs, path::Path};

use nightstream::nebula::{
    Application, Circuit, FinalProof, Geometry, Memory, PortAccess, Program, Row, Transition, View,
};
use nightstream::Engine;

#[derive(Clone)]
struct ReadRom;

impl Application for ReadRom {
    type Input = ();
    type State = u64;
    type Error = Infallible;

    fn width(&self) -> usize {
        1
    }
    fn aux(&self) -> usize {
        0
    }
    fn port_count(&self) -> usize {
        1
    }

    fn words(&self, state: &u64) -> Vec<u64> {
        vec![*state]
    }

    fn rows(&self, view: &View) -> Result<Vec<Row>, nightstream::Error> {
        Ok(vec![
            view.pad(0).into(),
            view.is_write(0).into(),
            view.is_ram(0).into(),
            view.addr(0).into(),
            (view.vw(0) - view.vr(0)).into(),
            (view.output(0) - (view.input(0) + view.vr(0))).into(),
        ])
    }

    fn step(&self, state: &u64, _: &(), memory: &Memory<'_>) -> Result<Transition<u64>, Infallible> {
        let value = memory.rom(0).unwrap();
        Ok(Transition {
            next: state + value,
            ports: vec![Some(PortAccess {
                is_write: false,
                is_ram: false,
                addr: 0,
                vr: value,
                vw: value,
            })],
            aux: Vec::new(),
        })
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let [_, action, directory]: [String; 3] = env::args()
        .collect::<Vec<_>>()
        .try_into()
        .map_err(|_| "usage: akita_compat cpu|metal|verify DIRECTORY")?;
    let geometry = Geometry::new(1, 1, 1, 1, 2).unwrap();
    let circuit = Circuit::compile(geometry, ReadRom)?;
    let fixtures = (0..6u32).map(|i| {
        let value = 5 + 7 * i;
        let initial = u64::from(3 * i);
        let steps = 1 + u64::from(i % 2);
        let last = initial + steps * u64::from(value);
        let program = Program::new(geometry, vec![value, 0], vec![0; 2]).unwrap();
        (
            Path::new(&directory).join(format!("{i}.proof")),
            program,
            initial,
            steps,
            last,
        )
    });
    match action.as_str() {
        "cpu" | "metal" => {
            let engine = if action == "metal" {
                Engine::Metal
            } else {
                Engine::Optimized
            };
            let prover = circuit.prover(engine, 109)?;
            let setup = prover.compression_setup()?;
            for (path, program, initial, steps, _) in fixtures {
                let mut run = prover.start(&program, initial)?;
                for _ in 0..steps {
                    run.segment(&[()])?;
                }
                let proof = prover.compress(&run.finish()?, &setup)?;
                fs::write(path, proof.to_bytes()?)?;
            }
        }
        "verify" => {
            let verifier = circuit.verifier(Engine::Optimized, 109)?;
            let key = verifier.compression_key()?;
            for (path, program, initial, steps, last) in fixtures {
                let mut bytes = fs::read(&path)?;
                let proof = FinalProof::from_bytes(&bytes)?;
                let verify =
                    |last: u64, proof: &FinalProof| verifier.verify_final(&program, &initial, &last, &key, proof);
                assert_eq!(verify(last, &proof)?.steps, steps);
                assert!(
                    verify(last + 1, &proof).is_err(),
                    "accepted changed statement for {path:?}"
                );
                let at = bytes.len() - 8;
                bytes[at] ^= 1;
                let changed = FinalProof::from_bytes(&bytes)?;
                assert!(verify(last, &changed).is_err(), "accepted changed proof for {path:?}");
            }
        }
        _ => return Err("expected cpu, metal, or verify".into()),
    }
    Ok(())
}
