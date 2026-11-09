//! Honest setup oracles at toy size: the streaming root against a direct
//! encoding, the store round trip with fewer stored levels, and the honest
//! fold with tampering.

use std::path::Path;

use p3_field_v08::{BasedVectorSpace, Field, PrimeCharacteristicRing, TwoAdicField};

use super::{word, Scratch};
use crate::circuit::algebra::eq_eval;
use crate::circuit::Native;
use crate::field::{eq_table, Ext, Gl};
use crate::hash::{compress, hash_leaf};
use crate::setup::{
    build, build_stored, check, encoded_leaves, fold, partials, query_point, Leaf, LeafView, Store, FOLD,
};
use crate::sumcheck::evaluate;
use crate::Error;

const VARIABLES: usize = 6;
const COUNT: usize = 3;

fn table(k: usize) -> Vec<Gl> {
    (0..1 << VARIABLES)
        .map(|x| Gl::from_u64(word(70 + k as u64, x)))
        .collect()
}

/// `Store::leaves`' encoder for the oracles `tables(k)` on the CPU.
fn encoder(
    tables: impl Fn(usize) -> Vec<Gl>,
) -> impl FnOnce(&[usize], &[usize]) -> Result<Vec<Vec<[Gl; 1 << FOLD]>>, Error> {
    move |oracles, leaves| {
        oracles
            .iter()
            .map(|&k| Ok(encoded_leaves(tables(k), VARIABLES, leaves)))
            .collect()
    }
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

/// Leaf `j` of an oracle on `2^n` points, value `v`: its extension at the
/// power point of `g·ω^{j + v·2^{n-2}}`.
fn direct_leaf(oracle: &[Ext], n: usize, j: usize) -> [Gl; 1 << FOLD] {
    let leaves = 1 << (n + 1 - FOLD);
    let omega = Gl::two_adic_generator(n + 1);
    std::array::from_fn(|v| {
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
        base
    })
}

/// Leaf j, oracle k, value v: Õ_k at the power point of `g·ω^{j + v·2^{n-2}}`.
fn direct_root(count: usize) -> [Gl; 4] {
    let n = VARIABLES;
    let leaves = 1 << (n + 1 - FOLD);
    let tables: Vec<Vec<Ext>> = (0..count).map(|k| lift(&table(k))).collect();
    let mut layer: Vec<[Gl; 4]> = (0..leaves)
        .map(|j| {
            let values: Vec<Gl> = tables
                .iter()
                .flat_map(|oracle| direct_leaf(oracle, n, j))
                .collect();
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

/// The pruned FFT gives each leaf's values from the definition, for one,
/// some, repeated and all leaves, from the smallest shape up.
#[test]
fn pruned_leaves_match_the_codeword_definition() {
    for n in [3, 4, 6, 9] {
        let values: Vec<Gl> = (0..1u64 << n)
            .map(|x| Gl::from_u64(word(90 + n as u64, x)))
            .collect();
        let oracle = lift(&values);
        let leaves = 1usize << (n + 1 - FOLD);
        let sets = [
            vec![0],
            vec![leaves - 1],
            vec![leaves / 2, 0, leaves / 2],
            (0..leaves).rev().collect(),
        ];
        for set in sets {
            let pruned = encoded_leaves(values.clone(), n, &set);
            for (got, &j) in pruned.iter().zip(&set) {
                assert_eq!(*got, direct_leaf(&oracle, n, j), "n {n}, leaf {j}");
            }
        }
    }
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
    let read = store
        .leaves(&(0..leaves).collect::<Vec<_>>(), encoder(table))
        .unwrap();
    for (index, leaf) in read.iter().enumerate() {
        let values = run_check(&root, index, leaf, &groups, &alpha).unwrap();
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
            let values = run_check(&root, index, &read[index], &groups, &alpha).unwrap();
            values[0] != evaluate(&forged, &point_of(index))
        })
        .count();
    assert!(2 * caught > leaves, "caught {caught} of {leaves}");

    // Tampered leaves, paths, roots and shapes are rejected.
    let index = 5;
    let leaf = read[index].clone();
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

/// A store with fewer tree levels and some stored codewords gives the same
/// leaves as one with the whole tree and no codeword. It catches wrong
/// oracles, a changed node, a changed codeword and a cut file, and a build
/// with a smaller budget removes the codeword files past it.
#[test]
fn a_store_with_fewer_levels_gives_the_same_leaves() {
    let n = VARIABLES;
    let codeword = (2 << n) * 8;
    // 16 leaves: all five levels and no codeword, or only the levels of 2
    // and 1 nodes and the codewords of oracles 0 and 1.
    let (full, top) = (Scratch::new("setup-full"), Scratch::new("setup-top"));
    let root = build_stored(n, COUNT, table, Some(&full.0), 4, 0).unwrap();
    assert_eq!(
        build_stored(n, COUNT, table, Some(&top.0), 1, 2 * codeword).unwrap(),
        root
    );
    let bytes = |dir: &Path, name: &str| std::fs::metadata(dir.join(name)).unwrap().len() as usize;
    let files = |dir: &Path| std::fs::read_dir(dir).unwrap().count();
    assert_eq!(bytes(&full.0, "tree.bin"), 31 * 32);
    assert_eq!(files(&full.0), 1, "only the tree is stored");
    assert_eq!(bytes(&top.0, "tree.bin"), 3 * 32);
    assert_eq!(bytes(&top.0, "oracle-01.bin"), codeword);
    assert_eq!(files(&top.0), 3, "the tree and two codewords");

    let indices = [0, 5, 6, 15, 5];
    let expected = Store::open(&full.0, n, COUNT, &root)
        .unwrap()
        .leaves(&indices, encoder(table))
        .unwrap();
    let store = Store::open(&top.0, n, COUNT, &root).unwrap();
    assert_eq!(store.leaves(&indices, encoder(table)).unwrap(), expected);
    assert!(matches!(
        store.leaves(&indices, encoder(|k| table(k + 1))),
        Err(Error::Setup("setup files do not match the oracles"))
    ));
    assert!(matches!(
        store.leaves(&[16], encoder(table)),
        Err(Error::Setup("setup leaf index"))
    ));

    let flip = |name: &str, length: Option<usize>| {
        let path = top.0.join(name);
        let mut file = std::fs::read(&path).unwrap();
        file[0] ^= 1;
        std::fs::write(&path, &file[..length.unwrap_or(file.len())]).unwrap();
    };
    flip("oracle-00.bin", None);
    let changed = Store::open(&top.0, n, COUNT, &root).unwrap();
    assert!(matches!(
        changed.leaves(&indices, encoder(table)),
        Err(Error::Setup("setup files do not match the oracles"))
    ));
    flip("oracle-00.bin", None);
    flip("tree.bin", None);
    let changed = Store::open(&top.0, n, COUNT, &root).unwrap();
    assert!(matches!(
        changed.leaves(&indices, encoder(table)),
        Err(Error::Setup("setup files do not match the oracles"))
    ));
    flip("tree.bin", Some(64));
    assert!(matches!(
        Store::open(&top.0, n, COUNT, &root),
        Err(Error::Setup("setup files do not match the key shape"))
    ));

    assert_eq!(build_stored(n, COUNT, table, Some(&top.0), 1, codeword).unwrap(), root);
    assert_eq!(files(&top.0), 2, "the second codeword is removed");
    assert_eq!(
        Store::open(&top.0, n, COUNT, &root)
            .unwrap()
            .leaves(&indices, encoder(table))
            .unwrap(),
        expected
    );
}

#[test]
fn scratches_with_one_name_do_not_share_a_directory() {
    let (first, second) = (Scratch::new("shared"), Scratch::new("shared"));
    assert_ne!(first.0, second.0);
    drop(first);
    assert!(second.0.is_dir());
}
