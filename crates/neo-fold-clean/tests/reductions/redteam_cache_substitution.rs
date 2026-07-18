#[path = "../support/mod.rs"]
mod support;

use neo_ccs::{check_ccs_rowwise_zero, CcsStructure, Mat, SparsePoly, Term};
use neo_fold_clean::engine::transcript::Transcript;
use neo_fold_clean::paper::construction2::RunningInstance;
use neo_fold_clean::paper::reductions::pi_ccs;
use neo_fold_clean::{config, CcsInstance, Structure};
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

const LABEL: &[u8] = b"neo.fold.clean/redteam/cache-substitution/v1";

fn linear_zero_structure(matrix_entry: F) -> Structure {
    let mut matrix = Mat::zero(1, 1, F::ZERO);
    matrix[(0, 0)] = matrix_entry;
    let f = SparsePoly::new(
        1,
        vec![Term {
            coeff: F::ONE,
            exps: vec![1],
        }],
    );
    CcsStructure::new(vec![matrix], f).expect("well-shaped one-row CCS")
}

#[test]
fn pi_ccs_rejects_same_shape_cache_from_different_structure() {
    let claimed = linear_zero_structure(F::ONE);
    let cached = linear_zero_structure(F::ZERO);
    let z = vec![F::ONE];
    assert!(check_ccs_rowwise_zero(&claimed, &[], &z).is_err());
    assert!(check_ccs_rowwise_zero(&cached, &[], &z).is_ok());
    assert_eq!((claimed.n, claimed.m, claimed.t()), (cached.n, cached.m, cached.t()));

    let params = config::ccs_params(claimed.n, claimed.m, claimed.t(), claimed.max_degree()).expect("shape params");
    support::install_ajtai_module(&params, &claimed);
    let cols = claimed.m.div_ceil(neo_math::D);
    let log = neo_ajtai::AjtaiSModule::from_global_for_dims(neo_math::D, cols).expect("Ajtai module");
    let instance = CcsInstance::from_low_norm_assignment(&params, &log, &claimed, &z, 0)
        .expect("low-norm false assignment can still be committed");
    let wrong_cache =
        neo_reductions::optimized_engine::OptimizedStructureCache::build(&cached).expect("same-shape cache");
    let running = RunningInstance::default();

    let mut prover_transcript = Transcript::with_label(LABEL);
    let accepted = match pi_ccs::prove(
        &mut prover_transcript,
        &params,
        &claimed,
        &wrong_cache,
        &log,
        vec![instance.clone()],
        &running,
    ) {
        Err(_) => false,
        Ok(proof) => {
            let mut verifier_transcript = Transcript::with_label(LABEL);
            pi_ccs::verify(
                &mut verifier_transcript,
                &params,
                &claimed,
                &wrong_cache,
                &[instance.claim],
                &running,
                &proof,
            )
            .is_ok()
        }
    };

    assert!(
        !accepted,
        "Pi_CCS proved and verified a different matrix relation supplied only through a stale cache"
    );
}
