//! Honest setup oracles at toy size: the streaming root against a direct
//! encoding, the store round trip, and the honest fold with tampering.

use p3_field_v08::{BasedVectorSpace, Field, PrimeCharacteristicRing, TwoAdicField};

use super::{word, Scratch};
use crate::circuit::algebra::eq_eval;
use crate::circuit::Native;
use crate::field::{eq_table, Ext, Gl};
use crate::hash::{compress, hash_leaf};
use crate::setup::{build, check, fold, partials, query_point, Leaf, LeafView, Store, FOLD};
use crate::sumcheck::evaluate;
use crate::Error;

const VARIABLES: usize = 6;
const COUNT: usize = 3;

#[test]
fn rebuilding_a_live_setup_is_rejected() {
    let scratch = Scratch::new("setup-generation");
    let root = build(VARIABLES, COUNT, table, Some(&scratch.0)).unwrap();
    let store = Store::open(&scratch.0, VARIABLES, COUNT, &root).unwrap();
    let original = store.read(0).unwrap();
    let replacement = |k| {
        let mut words = table(k);
        words[0] += Gl::ONE;
        words
    };
    assert!(build(VARIABLES, COUNT, replacement, Some(&scratch.0)).is_err());
    assert_eq!(store.read(0).unwrap(), original);
}

fn table(k: usize) -> Vec<Gl> {
    (0..1 << VARIABLES)
        .map(|x| Gl::from_u64(word(70 + k as u64, x)))
        .collect()
}

fn ext(seed: u64) -> Ext {
    Ext::from_basis_coefficients_fn(|i| Gl::from_u64(word(seed, i as u64)))
}

fn bits(index: usize) -> Vec<Gl> {
    (0..VARIABLES + 1 - FOLD)
        .map(|t| Gl::from_usize((index >> t) & 1))
        .collect()
}

fn point_of(index: usize) -> Vec<Ext> {
    query_point(&mut Native, VARIABLES, &bits(index))
}

/// The leaf check on plain values.
fn run_check(
    root: &[Gl; 4],
    index: usize,
    leaf: &Leaf,
    groups: &[Vec<Ext>],
    alpha: &[Ext; FOLD],
) -> Result<Vec<Ext>, Error> {
    let b = &mut Native;
    let view = LeafView::read(b, Some(leaf), COUNT, VARIABLES)?;
    check(b, *root, VARIABLES, &bits(index), &view, groups, alpha)
}

fn lift(values: &[Gl]) -> Vec<Ext> {
    values.iter().map(|&value| Ext::from(value)).collect()
}

/// Leaf j, oracle k, value v: Õ_k at the power point of `g·ω^{j + v·2^{n-2}}`.
fn direct_root(count: usize) -> [Gl; 4] {
    let n = VARIABLES;
    let leaves = 1 << (n + 1 - FOLD);
    let omega = Gl::two_adic_generator(n + 1);
    let tables: Vec<Vec<Ext>> = (0..count).map(|k| lift(&table(k))).collect();
    let mut layer: Vec<[Gl; 4]> = (0..leaves)
        .map(|j| {
            let mut values = Vec::new();
            for oracle in &tables {
                for v in 0..1 << FOLD {
                    let mut y = Gl::GENERATOR * omega.exp_u64((j + v * leaves) as u64);
                    let point: Vec<Ext> = (0..n)
                        .map(|_| {
                            let value = Ext::from(y);
                            y = y.square();
                            value
                        })
                        .collect();
                    let value = evaluate(oracle, &point);
                    let base = <Ext as BasedVectorSpace<Gl>>::as_basis_coefficients_slice(&value)[0];
                    assert_eq!(value, Ext::from(base));
                    values.push(base);
                }
            }
            hash_leaf(&values)
        })
        .collect();
    while layer.len() > 1 {
        layer = layer
            .chunks(2)
            .map(|pair| compress(pair[0], pair[1]))
            .collect();
    }
    layer[0]
}

#[test]
fn streaming_root_matches_a_direct_encoding() {
    // 8, 16 and 24 values per leaf: a partial block, a full block and a
    // partial block, two full blocks (rate 12).
    for count in 1..=COUNT {
        assert_eq!(
            build(VARIABLES, count, table, None).unwrap(),
            direct_root(count),
            "{count} oracles"
        );
    }
    // Control: a different oracle order gives a different root.
    assert_ne!(
        build(VARIABLES, COUNT, |k| table(COUNT - 1 - k), None).unwrap(),
        direct_root(COUNT)
    );
}

#[test]
fn honest_fold_accepts_and_tampering_rejects() {
    let n = VARIABLES;
    let scratch = Scratch::new("setup-fold");
    let root = build(n, COUNT, table, Some(&scratch.0)).unwrap();
    assert!(matches!(
        Store::open(&scratch.0, n, COUNT, &[Gl::ONE; 4]),
        Err(Error::Setup(_))
    ));
    let store = Store::open(&scratch.0, n, COUNT, &root).unwrap();

    // Two groups: oracles 0 and 1, then oracle 2.
    let groups = vec![vec![ext(1), ext(2)], vec![ext(5)]];
    let combined: Vec<Vec<Ext>> = [0..2, 2..COUNT]
        .into_iter()
        .zip(&groups)
        .map(|(oracles, weights)| {
            (0..1 << n)
                .map(|x| {
                    oracles
                        .clone()
                        .zip(weights)
                        .map(|(k, &w)| w * table(k)[x])
                        .sum()
                })
                .collect()
        })
        .collect();

    // Prover: partials at the claim point, then the fold by α.
    let point: Vec<Ext> = (0..n).map(|t| ext(10 + t as u64)).collect();
    let alpha: [Ext; FOLD] = std::array::from_fn(|t| ext(20 + t as u64));
    for oracle in &combined {
        let w = partials(oracle, &point[FOLD..]);
        let low = eq_table(&point[..FOLD]);
        let claim: Ext = low.iter().zip(&w).map(|(&e, &w)| e * w).sum();
        assert_eq!(claim, evaluate(oracle, &point));
        let folded = fold(oracle, &alpha);
        let expected: Ext = eq_table(&alpha).iter().zip(&w).map(|(&e, &w)| e * w).sum();
        assert_eq!(evaluate(&folded, &point[FOLD..]), expected);
        // Control: a wrong partial moves the folded claim.
        let mut wrong = w;
        wrong[5] += Ext::ONE;
        let moved: Ext = eq_table(&alpha)
            .iter()
            .zip(&wrong)
            .map(|(&e, &w)| e * w)
            .sum();
        assert_eq!(
            moved - expected,
            eq_eval(&mut Native, &alpha, &[Ext::ONE, Ext::ZERO, Ext::ONE])
        );
    }

    // Verifier: every leaf gives the fold's value at its query point.
    let folds: Vec<Vec<Ext>> = combined.iter().map(|oracle| fold(oracle, &alpha)).collect();
    let leaves = 1 << (n + 1 - FOLD);
    for index in 0..leaves {
        let leaf = store.read(index).unwrap();
        let values = run_check(&root, index, &leaf, &groups, &alpha).unwrap();
        let q = point_of(index);
        for (value, folded) in values.iter().zip(&folds) {
            assert_eq!(*value, evaluate(folded, &q), "leaf {index}");
        }
    }

    // A fold that differs in one entry fails at most leaves.
    let mut forged = folds[0].clone();
    forged[3] += Ext::ONE;
    let caught = (0..leaves)
        .filter(|&index| {
            let leaf = store.read(index).unwrap();
            let values = run_check(&root, index, &leaf, &groups, &alpha).unwrap();
            values[0] != evaluate(&forged, &point_of(index))
        })
        .count();
    assert!(2 * caught > leaves, "caught {caught} of {leaves}");

    // Tampered leaves, paths, roots and shapes are rejected.
    let index = 5;
    let leaf = store.read(index).unwrap();
    let reject = |leaf: &Leaf, root: &[Gl; 4], index: usize| run_check(root, index, leaf, &groups, &alpha).unwrap_err();
    for position in [0, 9, leaf.values.len() - 1] {
        let mut tampered = leaf.clone();
        tampered.values[position] += Gl::ONE;
        assert!(matches!(
            reject(&tampered, &root, index),
            Error::Rejected("setup leaf path")
        ));
    }
    for level in 0..leaf.path.len() {
        let mut tampered = leaf.clone();
        tampered.path[level][2] += Gl::ONE;
        assert!(matches!(
            reject(&tampered, &root, index),
            Error::Rejected("setup leaf path")
        ));
    }
    assert!(matches!(
        reject(&leaf, &root, index + 1),
        Error::Rejected("setup leaf path")
    ));
    let mut wrong_root = root;
    wrong_root[0] += Gl::ONE;
    assert!(matches!(
        reject(&leaf, &wrong_root, index),
        Error::Rejected("setup leaf path")
    ));
    let mut short = leaf.clone();
    short.path.pop();
    assert!(matches!(
        reject(&short, &root, index),
        Error::Rejected("setup leaf shape")
    ));
    let mut short = leaf.clone();
    short.values.truncate(2 * (1 << FOLD));
    assert!(matches!(
        reject(&short, &root, index),
        Error::Rejected("setup leaf shape")
    ));
}

#[test]
fn scratches_with_one_name_do_not_share_a_directory() {
    let (first, second) = (Scratch::new("shared"), Scratch::new("shared"));
    assert_ne!(first.0, second.0);
    drop(first);
    assert!(second.0.is_dir());
}

#[test]
fn rebuilding_setup_preserves_an_open_store() {
    let scratch = Scratch::new("setup-rebuild");
    let root = build(VARIABLES, COUNT, table, Some(&scratch.0)).unwrap();
    let store = Store::open(&scratch.0, VARIABLES, COUNT, &root).unwrap();
    let original = store.read(0).unwrap();

    let replacement = build(VARIABLES, COUNT, |k| table(k + COUNT), Some(&scratch.0));

    assert_eq!(
        store.read(0).unwrap(),
        original,
        "rebuilding changed an open setup store"
    );
    assert!(replacement.is_err(), "an existing setup must not be replaced");
}

#[test]
#[cfg(unix)]
fn rebuilding_setup_preserves_a_read_only_open_store() {
    use std::os::unix::fs::PermissionsExt;

    let scratch = Scratch::new("setup-failed-rebuild");
    let root = build(VARIABLES, COUNT, table, Some(&scratch.0)).unwrap();
    let store = Store::open(&scratch.0, VARIABLES, COUNT, &root).unwrap();
    let original = store.read(0).unwrap();
    let blocked = scratch.0.join("setup/oracle-01.bin");
    let permissions = std::fs::metadata(&blocked).unwrap().permissions();
    std::fs::set_permissions(&blocked, std::fs::Permissions::from_mode(0o400)).unwrap();
    let replacement = build(VARIABLES, COUNT, |k| table(k + COUNT), Some(&scratch.0));
    std::fs::set_permissions(&blocked, permissions).unwrap();

    assert!(replacement.is_err(), "a ready setup must not be replaced");
    assert!(Store::open(&scratch.0, VARIABLES, COUNT, &root).is_ok());
    assert_eq!(
        store.read(0).unwrap(),
        original,
        "rebuilding changed a read-only open setup store"
    );
}

#[test]
fn failed_setup_write_removes_its_candidate_and_allows_retry() {
    let scratch = Scratch::new("setup-write-failure");
    let failed = build(
        VARIABLES,
        COUNT,
        |k| {
            if k == 1 {
                let paths: Vec<_> = std::fs::read_dir(&scratch.0)
                    .unwrap()
                    .map(|entry| entry.unwrap().path())
                    .collect();
                assert_eq!(paths.len(), 1, "one private candidate belongs to this build");
                std::fs::create_dir(paths[0].join("oracle-01.bin")).unwrap();
            }
            table(k)
        },
        Some(&scratch.0),
    );
    assert!(matches!(failed, Err(Error::Io(_))));
    assert_eq!(std::fs::read_dir(&scratch.0).unwrap().count(), 0);

    let expected = build(VARIABLES, COUNT, table, None).unwrap();
    assert!(Store::open(&scratch.0, VARIABLES, COUNT, &expected).is_err());
    let root = build(VARIABLES, COUNT, table, Some(&scratch.0)).unwrap();
    assert_eq!(root, expected);
    let store = Store::open(&scratch.0, VARIABLES, COUNT, &root).unwrap();
    for leaf in 0..1 << (VARIABLES + 1 - FOLD) {
        assert!(store.read(leaf).is_ok());
    }
}

#[test]
fn concurrent_setup_builds_do_not_publish_mixed_oracles() {
    use std::sync::mpsc;

    let scratch = Scratch::new("setup-concurrent-build");
    let references = [
        Scratch::new("setup-concurrent-first"),
        Scratch::new("setup-concurrent-second"),
    ];
    let roots: [_; 2] = std::array::from_fn(|generation| {
        build(
            VARIABLES,
            COUNT,
            |k| table(k + generation * COUNT),
            Some(&references[generation].0),
        )
        .unwrap()
    });
    let (ready_tx, ready_rx) = mpsc::channel();
    let (resume_tx, resume_rx) = mpsc::channel();

    let (first, second, partial_rejected) = std::thread::scope(|scope| {
        let directory = &scratch.0;
        let first = scope.spawn(move || {
            build(
                VARIABLES,
                COUNT,
                |k| {
                    if k == 1 {
                        ready_tx.send(()).unwrap();
                        resume_rx.recv().unwrap();
                    }
                    table(k)
                },
                Some(directory),
            )
        });
        ready_rx.recv().unwrap();
        let partial_rejected = Store::open(&scratch.0, VARIABLES, COUNT, &roots[0]).is_err();
        let second = build(VARIABLES, COUNT, |k| table(k + COUNT), Some(&scratch.0));
        resume_tx.send(()).unwrap();
        (first.join().unwrap(), second, partial_rejected)
    });

    assert!(partial_rejected, "an incomplete setup must not be available to open");
    assert_ne!(first.is_ok(), second.is_ok(), "exactly one setup build must publish");
    let winner = usize::from(second.is_ok());
    let root = first.or(second).unwrap();
    assert_eq!(root, roots[winner]);
    let store = Store::open(&scratch.0, VARIABLES, COUNT, &root).unwrap();
    let expected = Store::open(&references[winner].0, VARIABLES, COUNT, &root).unwrap();
    for leaf in 0..1 << (VARIABLES + 1 - FOLD) {
        assert_eq!(
            store.read(leaf).unwrap(),
            expected.read(leaf).unwrap(),
            "concurrent builds published mixed oracle files at leaf {leaf}"
        );
    }
}
