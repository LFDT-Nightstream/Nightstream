//! The per-step cost of the Rust-built memory relation for the free
//! application (no state word, no private word, no row) with `B_ops = 21`, a
//! ROM of `2^12` cells, and a RAM of `2^16` cells, at each feasible `N` with
//! `S_max = 1` and the smallest `W_ts` that spec §4.2 rule 2 allows. A step's
//! Stage 1 words are its witness words (the memory words of
//! `WitnessWords.lean`) and its local words (the variables of the Poseidon2
//! sponges and of the product rows). The F′ carrier of the shared verifier
//! must fit the approved key capacity (`MAX_CARRIER_WIDTH`) and the `2^28`
//! domain checks of the assembler.

use neo_ajtai::nightstream_fprime_setup::MAX_CARRIER_WIDTH;

use super::memory_circuit;
use crate::nebula::{Application, Expr, Geometry, Memory, Transition, View};

#[derive(Clone, Copy, Debug)]
struct Free;

impl Application for Free {
    type Input = ();
    type State = ();
    type Error = std::convert::Infallible;

    fn width(&self) -> usize {
        0
    }

    fn aux(&self) -> usize {
        0
    }

    fn words(&self, _: &()) -> Vec<u64> {
        Vec::new()
    }

    fn rows(&self, _: &View) -> Vec<Expr> {
        Vec::new()
    }

    fn step(&self, _: &(), _: &(), _: &Memory) -> Result<Transition<()>, Self::Error> {
        unreachable!("the cost measurement runs no step")
    }
}

/// Build the relation of `N = n` and `S_max = s_max` with the smallest
/// `W_ts`, print its counts, and return whether it fits; `None` for a
/// geometry that breaks spec §4.2.
fn measure(n: usize, s_max: usize) -> Option<bool> {
    let (r, mu, b_ops) = (12, 16, 21);
    let cells = (1usize << r) + (1 << mu);
    let geometry = Geometry {
        r,
        mu,
        w_ts: (usize::BITS - (s_max * n * b_ops).leading_zeros()) as usize,
        b_ops,
        b_scan: cells / n,
        n,
        s_max,
    };
    if !geometry.is_valid() || !geometry.is_secure() {
        return None;
    }
    let started = std::time::Instant::now();
    let circuit = memory_circuit(&geometry, &Free).unwrap();
    let (witness, local) = (circuit.private_input_count(), circuit.generated_range().len());
    let carrier = crate::assembly::logical_width(&circuit);
    let fits = carrier
        .as_ref()
        .is_ok_and(|width| *width <= MAX_CARRIER_WIDTH);
    eprintln!(
        "{n} {s_max} {} {} {witness} {local} {} {} {} {fits} {:.1}",
        geometry.b_scan,
        geometry.w_ts,
        witness + local,
        circuit.row_count(),
        carrier.map_or_else(|error| error.to_string(), |width| width.to_string()),
        started.elapsed().as_secs_f64()
    );
    Some(fits)
}

#[test]
#[ignore = "Builds memory relations of up to millions of words; run separately under the 300-second cap."]
fn free_application_words_per_step() {
    let cells = (1usize << 12) + (1 << 16);
    eprintln!("N S_max B_scan W_ts witness local words rows carrier fits seconds");
    let mut smallest = None;
    for n in (1..=cells).rev() {
        if measure(n, 1) == Some(true) {
            smallest = Some(n);
        }
    }
    let smallest = smallest.expect("some feasible N fits");
    eprintln!("budget MAX_CARRIER_WIDTH={MAX_CARRIER_WIDTH} smallest fitting N={smallest}");
    // One more timestamp bit at the smallest N: S_max = 2.
    measure(smallest, 2);
}
