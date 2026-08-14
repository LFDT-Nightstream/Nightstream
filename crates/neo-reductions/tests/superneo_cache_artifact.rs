use std::io::Cursor;
use std::sync::Arc;

use neo_ccs::{CcsMatrix, CcsStructure, CscMat, GeometricRowRun, SeededPhi81LinearBlock, SparsePoly};
use neo_math::{KExtensions, D, F, K};
use neo_reductions::optimized_engine::OptimizedStructureCache;
use neo_reductions::superneo_eval::{
    build_superneo_eval_cache, eval_all_mats_cached, SuperneoCacheArtifactLimits, SuperneoCacheArtifactReceipt,
};
use p3_field::PrimeCharacteristicRing;

fn artifact_fixture() -> CcsStructure<F> {
    let rows = 2 * D;
    let cols = rows;
    let word_width = 41;
    let word_starts = vec![1, 42];
    let message_cols = (word_starts.len() * word_width).div_ceil(D);
    let (chunk_size, chunk_seeds) = neo_ajtai::seeded_pp_chunk_seeds([0x5c; 32], 1, message_cols);
    let seeded = SeededPhi81LinearBlock::new_with_word_width(
        D,
        word_starts,
        word_width,
        1,
        message_cols,
        chunk_size,
        chunk_seeds,
    )
    .expect("valid seeded block")
    .with_superneo_transformed_columns();
    let geometric = GeometricRowRun::new(7, D - 9, 41, F::from_u64(7), F::from_u64(3));
    let explicit = CscMat::from_triplets(
        vec![
            (0, 0, F::ONE),
            (3, 2, F::from_u64(5)),
            (3, 4, F::from_u64(9)),
            (D + 2, D + 1, -F::ONE),
        ],
        rows,
        cols,
    );
    let compact =
        CcsMatrix::csc_with_compact_rows(explicit, vec![seeded], vec![geometric]).expect("valid compact matrix");
    CcsStructure::new_sparse(
        vec![CcsMatrix::Identity { n: rows }, compact],
        SparsePoly::new(2, vec![]),
    )
    .expect("valid artifact fixture")
}

fn limits(bytes: u64) -> SuperneoCacheArtifactLimits {
    SuperneoCacheArtifactLimits::new(bytes, 2 * D, 2 * D, 2)
}

#[test]
fn compact_cache_artifact_roundtrip_preserves_every_evaluator_result() {
    let structure = artifact_fixture();
    let cache = build_superneo_eval_cache(&structure).expect("SuperNeo cache");
    let matrix_digest = [F::from_u64(11), F::from_u64(13), F::from_u64(17), F::from_u64(19)];
    let mut bytes = Vec::new();
    let receipt = cache
        .write_artifact(&mut bytes, matrix_digest)
        .expect("write compact cache artifact");
    assert_eq!(receipt.artifact_bytes(), bytes.len() as u64);
    assert_eq!(receipt.matrix_count(), 2);

    let trusted_receipt = SuperneoCacheArtifactReceipt::from_parts(
        receipt.artifact_bytes(),
        receipt.matrix_count(),
        receipt.matrix_digest(),
        receipt.cache_digest(),
    )
    .expect("trusted receipt fields");
    let loaded = neo_reductions::superneo_eval::SuperneoEvalCache::read_artifact(
        Cursor::new(&bytes),
        &trusted_receipt,
        limits(receipt.artifact_bytes()),
    )
    .expect("load compact cache artifact");

    let z = (0..structure.m)
        .map(|column| {
            K::from_coeffs([
                F::from_u64((column * 7 + 1) as u64),
                F::from_u64((column * 11 + 3) as u64),
            ])
        })
        .collect::<Vec<_>>();
    let chi = (0..structure.n)
        .map(|row| K::from_coeffs([F::from_u64((row * 13 + 5) as u64), F::from_u64((row * 17 + 7) as u64)]))
        .collect::<Vec<_>>();
    assert_eq!(
        eval_all_mats_cached(&loaded, &z, &chi, structure.n),
        eval_all_mats_cached(&cache, &z, &chi, structure.n),
    );

    let mut rewritten = Vec::new();
    let rewritten_receipt = loaded
        .write_artifact(&mut rewritten, matrix_digest)
        .expect("rewrite compact cache artifact");
    assert_eq!(rewritten_receipt, receipt);
    assert_eq!(rewritten, bytes);
}

#[test]
fn compact_cache_artifact_rejects_tampered_content() {
    let cache = build_superneo_eval_cache(&artifact_fixture()).expect("SuperNeo cache");
    let matrix_digest = [F::from_u64(23), F::from_u64(29), F::from_u64(31), F::from_u64(37)];
    let mut bytes = Vec::new();
    let receipt = cache
        .write_artifact(&mut bytes, matrix_digest)
        .expect("write compact cache artifact");
    let index = bytes.len() / 2;
    bytes[index] ^= 1;

    assert!(
        neo_reductions::superneo_eval::SuperneoEvalCache::read_artifact(
            Cursor::new(bytes),
            &receipt,
            limits(receipt.artifact_bytes()),
        )
        .is_err(),
        "a changed cache byte must not load under the verifier receipt",
    );
}

#[test]
fn compact_cache_artifact_checks_size_before_body_allocation() {
    let cache = build_superneo_eval_cache(&artifact_fixture()).expect("SuperNeo cache");
    let mut bytes = Vec::new();
    let receipt = cache
        .write_artifact(&mut bytes, [F::ONE, F::from_u64(2), F::from_u64(3), F::from_u64(4)])
        .expect("write compact cache artifact");
    let too_small = SuperneoCacheArtifactLimits::new(receipt.artifact_bytes() - 1, 2 * D, 2 * D, 2);

    assert!(
        neo_reductions::superneo_eval::SuperneoEvalCache::read_artifact(Cursor::new(bytes), &receipt, too_small,)
            .is_err(),
        "the loader must reject the receipt before it reads an oversized body",
    );
}

#[test]
fn verified_cache_artifact_rejects_a_different_ccs_shape() {
    let structure = artifact_fixture();
    let cache = build_superneo_eval_cache(&structure).expect("SuperNeo cache");
    let mut bytes = Vec::new();
    let receipt = cache
        .write_artifact(&mut bytes, [F::ONE, F::from_u64(2), F::from_u64(3), F::from_u64(4)])
        .expect("write compact cache artifact");
    let verified = neo_reductions::superneo_eval::SuperneoEvalCache::read_verified_artifact(
        Cursor::new(bytes),
        &receipt,
        limits(receipt.artifact_bytes()),
    )
    .expect("load verified compact cache artifact");
    let wrong_rows = 3 * D;
    let wrong_structure = CcsStructure::new_sparse(
        vec![
            CcsMatrix::Identity { n: wrong_rows },
            CcsMatrix::Identity { n: wrong_rows },
        ],
        SparsePoly::new(2, vec![]),
    )
    .expect("valid different CCS shape");

    assert!(
        OptimizedStructureCache::from_verified_artifact(Arc::new(wrong_structure), verified).is_err(),
        "a verified cache artifact must not install under a different CCS header",
    );
}
