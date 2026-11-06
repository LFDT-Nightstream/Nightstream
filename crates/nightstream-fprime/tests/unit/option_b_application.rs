//! A native application replaces Lean's only inside the region that Lean
//! writes: a region that claims a later gadget's piece, or fewer locals than
//! the application's pieces read, is refused. The identity binds the
//! application's rows.

use std::path::PathBuf;
use std::sync::Arc;

use crate::application_records::{
    ApplicationForm, ApplicationRecords, ApplicationRecordsWriter, ApplicationRowHeader, ApplicationTerm,
};
use crate::option_b::{load_option_b_packages, OPTION_B_PACKAGES_IDENTITY};
use crate::package::PackageError;

fn artifact(name: &str) -> Vec<u8> {
    std::fs::read(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../formal/nightstream-fprime/artifacts")
            .join(name),
    )
    .unwrap_or_else(|error| panic!("read {name}: {error}"))
}

/// `(current_k + scale · witness_k) · 1 = next_k`.
fn addition(scale: u64) -> Arc<ApplicationRecords> {
    let mut writer = ApplicationRecordsWriter::new().expect("writer");
    let term = |form, variable, coefficient| {
        Ok(ApplicationTerm {
            form,
            variable,
            coefficient,
        })
    };
    for lane in 0..4 {
        let header = ApplicationRowHeader {
            constants: [0, 1, 0],
            term_counts: [2, 0, 1],
        };
        let terms = [
            term(ApplicationForm::A, lane, 1),
            term(ApplicationForm::A, 4 + lane, scale),
            term(ApplicationForm::C, 8 + lane, 1),
        ];
        writer.append_row(header, terms).expect("row");
    }
    Arc::new(writer.finish().expect("records"))
}

#[test]
fn substitution_binds_the_application_and_refuses_a_wrong_region() {
    let packages = load_option_b_packages(
        &artifact("nightstream-fprime-option-b-light-v1.json"),
        &artifact("nightstream-fprime-option-b-refresh-v1.json"),
        OPTION_B_PACKAGES_IDENTITY,
    )
    .expect("option-B packages");
    let identity = |scale| {
        packages
            .with_application(addition(scale), 4)
            .expect("Lean's region holds the application")
            .identity()
    };
    assert_ne!(identity(1), packages.identity());
    assert_ne!(identity(1), identity(2));

    let mut later_piece = packages.clone();
    later_piece.light.application.pieces += 1;
    let mut fewer_locals = packages.clone();
    fewer_locals.refresh.application.locals -= 1;
    for wrong in [later_piece, fewer_locals] {
        assert!(matches!(
            wrong.with_application(addition(1), 4),
            Err(PackageError::Malformed(
                "option-B application piece reads past its region"
            ))
        ));
    }
}
