//! The per-step cost of the Rust-built memory relation for the free
//! application (no state word, no private word, no row) with `B_ops = 21`, a
//! ROM of `2^12` cells, and a RAM of `2^16` cells, at each feasible `N` with
//! `S_max = 1` and the smallest `W_ts` that spec §4.2 rule 2 allows; and for
//! the geometry, state words, and private words of the WASM test of
//! `neo-wasm` (`tests/wasm_nebula.rs`), without its rows. A step's
//! Stage 1 words are its witness words (the memory words of
//! `WitnessWords.lean`) and its local words (the variables of the Poseidon2
//! sponges and of the product rows). The F′ carrier of the shared verifier
//! must fit the approved key capacity (`MAX_CARRIER_WIDTH`) and the `2^28`
//! domain checks of the assembler.

use neo_ajtai::nightstream_fprime_setup::MAX_CARRIER_WIDTH;

use super::memory_circuit;
use crate::nebula::{Application, Expr, Geometry, Memory, Transition, View};

/// An application with `width` state words, `aux` private words, and no row.
#[derive(Clone, Copy, Debug)]
struct Free {
    width: usize,
    aux: usize,
}

impl Application for Free {
    type Input = ();
    type State = ();
    type Error = std::convert::Infallible;

    fn width(&self) -> usize {
        self.width
    }

    fn aux(&self) -> usize {
        self.aux
    }

    fn words(&self, _: &()) -> Vec<u64> {
        vec![0; self.width]
    }

    fn rows(&self, _: &View) -> Vec<Expr> {
        Vec::new()
    }

    fn step(&self, _: &(), _: &(), _: &Memory) -> Result<Transition<()>, Self::Error> {
        unreachable!("the cost measurement runs no step")
    }
}

/// Build the free relation of `N = n` and `S_max = s_max` with the smallest
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
    count(&geometry, Free { width: 0, aux: 0 })
}

/// Print the counts of the relation of `geometry` and `app`, and return
/// whether it fits; `None` for a geometry that breaks spec §4.2.
fn count(geometry: &Geometry, app: Free) -> Option<bool> {
    if !geometry.is_valid() || !geometry.is_secure() {
        return None;
    }
    let n = geometry.n;
    let s_max = geometry.s_max;
    let started = std::time::Instant::now();
    let circuit = memory_circuit(geometry, &app).unwrap();
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

/// The WASM test geometry with its 61 state words and 8,549 private words.
#[test]
#[ignore = "Prints the per-step words of the WASM test geometry."]
fn wasm_geometry_words_per_step() {
    let geometry = Geometry {
        r: 8,
        mu: 8,
        w_ts: 8,
        b_ops: 21,
        b_scan: 64,
        n: 8,
        s_max: 1,
    };
    eprintln!("N S_max B_scan W_ts witness local words rows carrier fits seconds");
    assert_eq!(count(&geometry, Free { width: 61, aux: 8_549 }), Some(true));
    eprintln!("budget MAX_CARRIER_WIDTH={MAX_CARRIER_WIDTH}");
}

/// The parts of `size_parts` sum to the counts of the memory circuit, and
/// each phase and region has its own name.
#[test]
fn size_parts_sum_to_the_circuit_counts() {
    let geometry = Geometry {
        r: 2,
        mu: 3,
        w_ts: 4,
        b_ops: 3,
        b_scan: 3,
        n: 4,
        s_max: 1,
    };
    let app = Free { width: 2, aux: 3 };
    let parts = crate::nebula::size_parts(&geometry, &app).unwrap();
    let circuit = memory_circuit(&geometry, &app).unwrap();
    let total = parts
        .iter()
        .fold((0, 0, 0), |(witness, local, rows), part| {
            (
                witness + part.size.witness_words,
                local + part.size.local_words,
                rows + part.size.rows,
            )
        });
    assert_eq!(
        total,
        (
            circuit.private_input_count(),
            circuit.generated_range().len(),
            circuit.row_count()
        )
    );
    let mut names: Vec<&str> = parts.iter().map(|part| part.name).collect();
    names.sort_unstable();
    names.dedup();
    assert_eq!(names.len(), parts.len());
}
