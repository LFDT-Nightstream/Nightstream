//! Diagnostic split of the existing full-window opening algorithm, checked against production.
use super::*;
use neo_ccs::V1_2Evaluations;
use std::time::Instant;

pub(crate) fn measured(
    cache: &SuperneoEvalCache,
    point: &[K],
    witnesses: &[SuperneoZBlocks],
) -> (Vec<V1_2Evaluations<K>>, serde_json::Value) {
    let (rows, width, matrices) = cache.relation_shape().unwrap();
    let start = Instant::now();
    let mut result: Vec<_> = openings::pad_openings(witnesses, point)
        .into_iter()
        .map(|eval_k| V1_2Evaluations {
            eval_k: eval_k.to_vec(),
            eval_a: vec![vec![K::ZERO; D]; matrices],
        })
        .collect();
    let pad = start.elapsed().as_secs_f64();
    let start = Instant::now();
    let weights = EqualityWeights::new(point);
    let row_weights: Vec<_> = (0..rows).map(|r| weights.at(r)).collect();
    let equality = start.elapsed().as_secs_f64();
    let start = Instant::now();
    let mut scratch = RingEvalScratch::reuse(Vec::new(), width / D);
    let allocation = start.elapsed().as_secs_f64();
    let active: Vec<_> = witnesses
        .iter()
        .enumerate()
        .filter(|(_, w)| !w.real_is_zero())
        .collect();
    let sources: Vec<_> = active.iter().map(|(_, w)| *w).collect();
    let mut accumulation = Vec::new();
    let mut contraction = Vec::new();
    let mut clearing = Vec::new();
    for (matrix, cache) in cache.matrix_caches().iter().enumerate() {
        let start = Instant::now();
        cache.accumulate_barred_ring_form_parallel(rows, &mut scratch, |r| row_weights[r]);
        accumulation.push(start.elapsed().as_secs_f64());
        let start = Instant::now();
        let values = parallel::eval_active_blocks_many(&scratch, &sources);
        contraction.push(start.elapsed().as_secs_f64());
        for ((index, _), value) in active.iter().zip(values) {
            result[*index].eval_a[matrix] = value.to_vec();
        }
        let start = Instant::now();
        scratch.clear_active();
        clearing.push(start.elapsed().as_secs_f64());
    }
    (
        result,
        serde_json::json!({"pad_seconds":pad,"row_weights_seconds":equality,"scratch_allocation_seconds":allocation,"accumulation_seconds":accumulation,"contraction_seconds":contraction,"clearing_seconds":clearing}),
    )
}
