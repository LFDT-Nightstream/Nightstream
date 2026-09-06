//! Run one exact conformance gate on an external Lean-emitted candidate.
//! Expected identities and metadata are separate caller-selected Lean outputs;
//! no published production identity pin is changed by this executable.

#[allow(dead_code, unused_imports)]
#[path = "../../tests/base_step_assignment.rs"]
mod base_checks;
#[path = "check_package_conformance/candidate.rs"]
mod candidate;
#[allow(dead_code, unused_imports)]
#[path = "../../tests/per_application_logical_matrix_conformance.rs"]
mod logical_checks;
#[path = "../../tests/support/recursive_step.rs"]
mod recursive_checks;
#[allow(dead_code, unused_imports)]
#[path = "check_package_conformance/support.rs"]
mod support;

use std::{env, path::PathBuf};

fn main() {
    let mut arguments = env::args_os().skip(1);
    let mode = arguments
        .next()
        .expect("mode: physical, logical, mutations, base, recursive, recursive-mutations, commitment, detached, or primitive")
        .into_string()
        .expect("mode text");
    if mode == "primitive" {
        let path = PathBuf::from(
            arguments
                .next()
                .expect("Lean sparse-commitment parity path"),
        );
        assert!(arguments.next().is_none(), "primitive mode accepts one parity path");
        candidate::check_sparse_commitment(&path);
        return;
    }
    let candidate_path = PathBuf::from(arguments.next().expect("candidate sealed-package path"));
    let binding_path = PathBuf::from(arguments.next().expect("canonical Lean binding path"));
    let setup_path = PathBuf::from(arguments.next().expect("Lean setup-parity path"));
    let mut identity = [0u64; 4];
    for word in &mut identity {
        *word = arguments
            .next()
            .expect("four expected Lean structural identity words")
            .into_string()
            .expect("identity word text")
            .parse()
            .expect("identity word u64");
    }
    let inputs = arguments.map(PathBuf::from).collect::<Vec<_>>();
    candidate::run(&mode, &candidate_path, &binding_path, &setup_path, identity, &inputs);
}
