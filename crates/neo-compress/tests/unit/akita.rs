//! The Akita boundary on the three polynomials of a proof: the pinned
//! schedule files, the order of the variables of a point, and the commitment
//! profiles and the order of the polynomials that the verifier accepts.

use akita_types::CommittedGroup;
use neo_math::{from_complex, F, K};
use p3_field::PrimeCharacteristicRing;
use sha2::{Digest, Sha256};

use super::prover::Prover;
use super::verifier::{Opening, Verifier};
use super::{base, commitment, commitment_bytes, profile, row, Polynomial, Schedules, Shape, SCHEDULES};
use crate::Error;

/// The SHA-256 of each file of `SCHEDULES`, as `scripts/akita_schedules.sh`
/// prints it.
const DIGESTS: [&str; 2] = [
    "8ee08715b2120cfb576227f195fde041e5e1dd828511ca5e75d5df73d4c097b5",
    "b71861d25a4cb23b8eb68b20365bc5b76336224dcacdcada86242c9d9c084f94",
];
const SESSION: &[u8] = b"neo-compress/tests/akita";
const ORDER: [Polynomial; 3] = [Polynomial::Key, Polynomial::Witness, Polynomial::Inverse];

const SMALL: &Schedules = &SCHEDULES[0];
const LARGE: &Schedules = &SCHEDULES[1];

fn small() -> Shape {
    SMALL.first()
}

fn vars(shape: Shape) -> [usize; 3] {
    [shape.key, shape.witness, shape.inverse]
}

/// Cells 0 to 4 of polynomial `index`. The other cells are zero, so no two
/// variables have the same role.
fn cells(index: usize) -> Vec<F> {
    (1..=5)
        .map(|cell| F::from_u64(10 * index as u64 + cell))
        .collect()
}

fn point(index: usize, vars: usize) -> Vec<K> {
    (0..vars as u64)
        .map(|variable| {
            from_complex(
                F::from_u64(3 + variable + 50 * index as u64),
                F::from_u64(100 + 7 * variable),
            )
        })
        .collect()
}

/// The multilinear polynomial with the evaluation table `cells` at `point`,
/// where coordinate `variable` binds bit `bit(variable)` of the cell index.
fn value(cells: &[F], point: &[K], bit: impl Fn(usize) -> usize) -> K {
    cells
        .iter()
        .enumerate()
        .map(|(index, &cell)| {
            let weight: K = point
                .iter()
                .enumerate()
                .map(|(variable, &coordinate)| {
                    if index >> bit(variable) & 1 == 1 {
                        coordinate
                    } else {
                        K::ONE - coordinate
                    }
                })
                .product();
            weight * cell
        })
        .sum()
}

/// The three polynomials of the first row of `schedules` with their points
/// and the values with bit 0 first.
struct Polynomials {
    schedules: &'static Schedules,
    shape: Shape,
    points: [Vec<K>; 3],
    values: [K; 3],
}

impl Polynomials {
    fn new(schedules: &'static Schedules) -> Self {
        let shape = schedules.first();
        let points: [Vec<K>; 3] = std::array::from_fn(|index| point(index, vars(shape)[index]));
        let values = std::array::from_fn(|index| value(&cells(index), &points[index], |variable| variable));
        Self {
            schedules,
            shape,
            points,
            values,
        }
    }

    /// The commitments of the polynomials, the setup prefixes of the prover
    /// and the proof that they have `values` at the points.
    fn opened(&self, values: [K; 3]) -> Result<([Vec<u8>; 3], Vec<u8>, Vec<u8>), Error> {
        let prover = Prover::new(self.schedules)?;
        let committed: [_; 3] = std::array::from_fn(|index| {
            prover
                .commit(self.shape, ORDER[index], cells(index).into_iter().map(base).collect())
                .expect("commitment")
        });
        let commitments = std::array::from_fn(|index| committed[index].bytes.clone());
        let [key, witness, inverse] = committed;
        let proof = prover.open(
            [
                (key, &self.points[0], values[0]),
                (witness, &self.points[1], values[1]),
                (inverse, &self.points[2], values[2]),
            ],
            SESSION,
        )?;
        Ok((commitments, prover.setup_prefixes()?, proof))
    }

    fn verify(&self, prefixes: &[u8], commitments: [&[u8]; 3], values: [K; 3], proof: &[u8]) -> Result<(), Error> {
        let openings = std::array::from_fn(|index| Opening {
            commitment: commitments[index],
            point: &self.points[index],
            value: values[index],
        });
        Verifier::new(self.shape, prefixes)?.verify(openings, proof, SESSION)
    }
}

fn each(commitments: &[Vec<u8>; 3]) -> [&[u8]; 3] {
    std::array::from_fn(|index| &commitments[index][..])
}

/// Each file is the one that Akita's planner made, its rows are the rows of
/// `SCHEDULES` in increasing order, and one commitment of the key polynomial
/// serves each row of its file.
#[test]
fn each_schedule_file_has_its_pinned_digest_and_its_rows() {
    for (schedules, pinned) in SCHEDULES.iter().zip(DIGESTS) {
        let digest: String = Sha256::digest(schedules.bytes)
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect();
        assert_eq!(digest, pinned, "key of {} variables", schedules.key_vars);
        let catalog = schedules.catalog().expect("catalog");
        assert_eq!(catalog.rows().count(), schedules.rows.len());
        assert!(schedules.rows.is_sorted());
        let key = *profile(row(&catalog, schedules.first()).expect("row"), Polynomial::Key);
        for &(witness, inverse) in schedules.rows {
            let shape = Shape {
                key: schedules.key_vars,
                witness,
                inverse,
            };
            let row = row(&catalog, shape).expect("row");
            assert_eq!(*profile(row, Polynomial::Key), key, "{shape:?}");
        }
    }
}

#[test]
fn coordinate_b_of_a_point_binds_bit_b_of_the_cell_index() {
    let polynomials = Polynomials::new(SMALL);
    let low_first = polynomials.values;
    let (commitments, prefixes, proof) = polynomials.opened(low_first).expect("opening");
    polynomials
        .verify(&prefixes, each(&commitments), low_first, &proof)
        .expect("the values with bit 0 first verify");
    for index in 0..3 {
        let count = polynomials.points[index].len();
        let mut high_first = low_first;
        high_first[index] = value(&cells(index), &polynomials.points[index], |variable| {
            count - 1 - variable
        });
        assert_ne!(high_first[index], low_first[index]);
        assert!(
            polynomials
                .verify(&prefixes, each(&commitments), high_first, &proof)
                .is_err(),
            "polynomial {index}"
        );
        let other_order = polynomials
            .opened(high_first)
            .and_then(|(_, _, proof)| polynomials.verify(&prefixes, each(&commitments), high_first, &proof));
        assert!(
            other_order.is_err(),
            "polynomial {index}: the value with bit 0 last has no opening"
        );
    }
}

/// The proof names its polynomials in order: the commitment of other cells
/// in place of the key polynomial, and `W` and `V` in the other order with
/// their points and values, do not verify.
#[test]
fn the_proof_is_bound_to_each_commitment_and_to_their_order() {
    let polynomials = Polynomials::new(SMALL);
    let values = polynomials.values;
    let (commitments, prefixes, proof) = polynomials.opened(values).expect("opening");
    let prover = Prover::new(SMALL).expect("prover");
    let other_key = prover
        .commit(small(), Polynomial::Key, cells(7).into_iter().map(base).collect())
        .expect("commitment")
        .bytes;
    assert_ne!(other_key, commitments[0]);
    let result = polynomials.verify(
        &prefixes,
        [&other_key, &commitments[1], &commitments[2]],
        values,
        &proof,
    );
    assert!(matches!(result, Err(Error::Rejected(_))), "{result:?}");

    assert_eq!(small().witness, small().inverse);
    let swapped = Polynomials {
        schedules: SMALL,
        shape: small(),
        points: [
            polynomials.points[0].clone(),
            polynomials.points[2].clone(),
            polynomials.points[1].clone(),
        ],
        values: [values[0], values[2], values[1]],
    };
    let result = swapped.verify(
        &prefixes,
        [&commitments[0], &commitments[2], &commitments[1]],
        swapped.values,
        &proof,
    );
    assert!(matches!(result, Err(Error::Rejected(_))), "{result:?}");
}

/// A commitment of the cells of `W` as a polynomial of 26 variables, and
/// its rows under the profile of 16 variables, do not verify as `W`. Akita
/// rejects a change of one profile field when it decodes.
#[test]
fn a_commitment_under_another_profile_does_not_verify() {
    let polynomials = Polynomials::new(SMALL);
    let values = polynomials.values;
    let (commitments, prefixes, proof) = polynomials.opened(values).expect("opening");

    let larger = Prover::new(LARGE)
        .expect("prover")
        .commit(
            LARGE.first(),
            Polynomial::Witness,
            cells(1).into_iter().map(base).collect(),
        )
        .expect("commitment")
        .bytes;
    let catalog = SMALL.catalog().expect("catalog");
    let profile = *profile(row(&catalog, small()).expect("row"), Polynomial::Witness);
    let rows = commitment(&larger)
        .expect("commitment")
        .commitment()
        .clone();
    let mut others = vec![larger];
    if let Ok(relabeled) = CommittedGroup::new(profile, rows) {
        others.push(commitment_bytes(&relabeled).expect("bytes"));
    }
    for other in others {
        let result = polynomials.verify(&prefixes, [&commitments[0], &other, &commitments[2]], values, &proof);
        assert!(matches!(result, Err(Error::Rejected(_))), "{result:?}");
    }
}

/// The rows of the large key offload a prefix of Akita's setup, and the
/// verifier takes the commitment of that prefix from its key. A proof of the
/// three polynomials with few cells verifies with the prefixes of the
/// prover, and bytes with one changed byte are other prefixes or none.
#[test]
fn a_proof_does_not_verify_with_other_setup_prefixes() {
    let polynomials = Polynomials::new(LARGE);
    let values = polynomials.values;
    let (commitments, prefixes, proof) = polynomials.opened(values).expect("opening");
    assert!(prefixes.len() > 500, "{} bytes hold no commitment", prefixes.len());
    polynomials
        .verify(&prefixes, each(&commitments), values, &proof)
        .expect("the proof verifies with the prefixes of the prover");
    for at in [0, 40, prefixes.len() / 2, prefixes.len() - 1] {
        let mut changed = prefixes.clone();
        changed[at] ^= 1;
        let result = polynomials.verify(&changed, each(&commitments), values, &proof);
        assert!(result.is_err(), "byte {at}");
    }
    let mut longer = prefixes.clone();
    longer.push(0);
    assert!(matches!(
        polynomials.verify(&longer, each(&commitments), values, &proof),
        Err(Error::Rejected(_))
    ));
    let none = Prover::new(SMALL)
        .expect("prover")
        .setup_prefixes()
        .expect("prefixes");
    assert!(polynomials
        .verify(&none, each(&commitments), values, &proof)
        .is_err());
}
