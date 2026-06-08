use neo_fold_clean::engine::r1cs_circuit::{Lc, R1csBuilder, Var};
use neo_fold_clean::paper::terminal_ce::merkle::{
    enforce_terminal_ce_merkle_root_from_leaf, terminal_ce_merkle_root_from_leaf, TerminalCeMerkleError,
};
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

fn f(value: u64) -> F {
    F::from_u64(value)
}

fn digest(seed: u64) -> [F; 4] {
    [f(seed), f(seed + 1), f(seed + 2), f(seed + 3)]
}

fn alloc_digest(builder: &mut R1csBuilder, digest: [F; 4]) -> [Var; 4] {
    [
        builder.alloc(digest[0]),
        builder.alloc(digest[1]),
        builder.alloc(digest[2]),
        builder.alloc(digest[3]),
    ]
}

fn pin_digest(builder: &mut R1csBuilder, digest: [Var; 4], expected: [F; 4]) {
    for (wire, value) in digest.into_iter().zip(expected) {
        builder.enforce_eq(&Lc::from_var(wire), &Lc::from_const(value));
    }
}

#[test]
fn terminal_ce_merkle_native_root_binds_same_shape_leaf_and_siblings() {
    let leaf = digest(10);
    let path = [digest(20), digest(30), digest(40)];
    let root = terminal_ce_merkle_root_from_leaf(leaf, &path, 5).expect("valid Merkle path");

    let mut changed_leaf = leaf;
    changed_leaf[2] += F::ONE;
    assert_ne!(
        terminal_ce_merkle_root_from_leaf(changed_leaf, &path, 5).expect("same-shape changed leaf"),
        root,
        "native terminal CE Merkle root did not bind a same-shape leaf field change"
    );

    let mut changed_path = path;
    changed_path[1][3] += F::ONE;
    assert_ne!(
        terminal_ce_merkle_root_from_leaf(leaf, &changed_path, 5).expect("same-shape changed sibling"),
        root,
        "native terminal CE Merkle root did not bind a same-shape sibling field change"
    );

    assert_ne!(
        terminal_ce_merkle_root_from_leaf(leaf, &path, 1).expect("same-depth changed index"),
        root,
        "native terminal CE Merkle root did not bind the verifier-derived leaf index"
    );
}

#[test]
fn terminal_ce_merkle_rejects_index_outside_path_depth() {
    let err = terminal_ce_merkle_root_from_leaf(digest(10), &[], 1).expect_err("depth-0 path must only admit index 0");
    assert_eq!(err, TerminalCeMerkleError::IndexTooLarge { index: 1, depth: 0 });
}

#[test]
fn terminal_ce_merkle_circuit_matches_native_and_rejects_tamper() {
    let leaf = digest(100);
    let path = [digest(200), digest(300), digest(400)];
    let expected = terminal_ce_merkle_root_from_leaf(leaf, &path, 6).expect("valid native Merkle path");

    let mut builder = R1csBuilder::new();
    let leaf_vars = alloc_digest(&mut builder, leaf);
    let path_vars = path
        .iter()
        .copied()
        .map(|node| alloc_digest(&mut builder, node))
        .collect::<Vec<_>>();
    let root_vars =
        enforce_terminal_ce_merkle_root_from_leaf(&mut builder, leaf_vars, &path_vars, 6).expect("circuit Merkle path");
    pin_digest(&mut builder, root_vars, expected);

    assert!(
        builder.is_satisfied(),
        "honest terminal CE Merkle path should satisfy (first bad row: {:?})",
        builder.first_unsatisfied_row()
    );

    let leaf_col = leaf_vars[1].col();
    let original_leaf = builder.witness()[leaf_col];
    builder.tamper_witness(leaf_col, original_leaf + F::ONE);
    assert!(
        !builder.is_satisfied(),
        "terminal CE Merkle circuit accepted a same-shape leaf witness tamper"
    );
    builder.tamper_witness(leaf_col, original_leaf);
    assert!(builder.is_satisfied(), "restoring leaf should restore satisfiability");

    let sibling_col = path_vars[1][2].col();
    let original_sibling = builder.witness()[sibling_col];
    builder.tamper_witness(sibling_col, original_sibling + F::ONE);
    assert!(
        !builder.is_satisfied(),
        "terminal CE Merkle circuit accepted a same-shape sibling witness tamper"
    );
}
