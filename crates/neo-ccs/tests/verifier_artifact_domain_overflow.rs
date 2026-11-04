//! A verifier-artifact header must not round its carrier width past `usize`.

use neo_ccs::{poly::SparsePoly, CcsStructure};
use p3_goldilocks::Goldilocks as F;

#[cfg(target_pointer_width = "64")]
#[test]
fn verifier_artifact_header_rejects_wrapped_carrier_width() {
    let header = CcsStructure::new_verifier_artifact_header(1, usize::MAX, 1, SparsePoly::<F>::new(1, Vec::new()));
    assert!(
        header
            .and_then(|header| header.with_domain_variables(1))
            .is_err(),
        "a two-row domain cannot hold a verifier-artifact carrier with usize::MAX logical columns"
    );
}
