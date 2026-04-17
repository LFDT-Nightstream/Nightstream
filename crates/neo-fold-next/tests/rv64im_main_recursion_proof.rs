#![allow(dead_code)]

#[path = "support/rv64im_n2.rs"]
mod rv64im_n2_support;

use neo_fold_next::rv64im::audit::{
    rv64im_main_recursion_proof_first_step_snark_bytes_mut, rv64im_main_recursion_proof_pop_last_step_proof,
    rv64im_main_recursion_proof_x_last_mut,
};
use neo_fold_next::rv64im::{build_rv64im_main_proof, prove_rv64im_recursion_proof, verify_rv64im_main_proof};

#[test]
#[ignore = "long-running end-to-end recursion proof round-trip"]
fn rv64im_main_recursion_proof_round_trip() {
    let fixture = rv64im_n2_support::build_rv64im_n2_fixture().expect("build rv64im n=2 fixture");
    let recursion_proof =
        prove_rv64im_recursion_proof(&fixture.final_statement, &fixture.final_proof).expect("prove recursion proof");
    let mut published_main =
        build_rv64im_main_proof(&fixture.final_statement, &fixture.final_proof).expect("build published main proof");
    *published_main.recursion_proof_mut() = recursion_proof;
    verify_rv64im_main_proof(&published_main).expect("verify published main proof");
}

#[test]
#[ignore = "long-running end-to-end recursion proof round-trip"]
fn rv64im_main_recursion_proof_round_trip_from_main_proof() {
    let fixture = rv64im_n2_support::build_rv64im_n2_fixture().expect("build rv64im n=2 fixture");
    let published_main =
        build_rv64im_main_proof(&fixture.final_statement, &fixture.final_proof).expect("build published main proof");
    verify_rv64im_main_proof(&published_main).expect("verify published main proof");
}

#[test]
#[ignore = "long-running end-to-end recursion proof tamper regression"]
fn rv64im_main_proof_rejects_tampered_carried_recursion_proof_bytes() {
    let fixture = rv64im_n2_support::build_rv64im_n2_fixture().expect("build rv64im n=2 fixture");
    let mut published_main =
        build_rv64im_main_proof(&fixture.final_statement, &fixture.final_proof).expect("build published main proof");
    verify_rv64im_main_proof(&published_main).expect("baseline published main proof must verify");

    if let Some(first) =
        rv64im_main_recursion_proof_first_step_snark_bytes_mut(published_main.recursion_proof_mut()).first_mut()
    {
        *first ^= 1;
    } else {
        rv64im_main_recursion_proof_first_step_snark_bytes_mut(published_main.recursion_proof_mut()).push(1);
    }

    let err = verify_rv64im_main_proof(&published_main).expect_err("tampered recursion step-proof bytes must fail");
    assert!(
        err.to_string().contains("decode failed")
            || err.to_string().contains("compressed-chain")
            || err.to_string().contains("verify failed")
    );
}

#[test]
#[ignore = "long-running end-to-end recursion proof tamper regression"]
fn rv64im_main_proof_rejects_tampered_recursion_backend_statement() {
    let fixture = rv64im_n2_support::build_rv64im_n2_fixture().expect("build rv64im n=2 fixture");
    let mut published_main =
        build_rv64im_main_proof(&fixture.final_statement, &fixture.final_proof).expect("build published main proof");
    verify_rv64im_main_proof(&published_main).expect("baseline published main proof must verify");

    rv64im_main_recursion_proof_x_last_mut(published_main.recursion_proof_mut())[0] ^= 1;

    let err = verify_rv64im_main_proof(&published_main).expect_err("tampered recursion final public image must fail");
    assert!(
        err.to_string().contains("x_last")
            || err.to_string().contains("published statement")
            || err.to_string().contains("public image")
    );
}

#[test]
#[ignore = "long-running end-to-end recursion proof tamper regression"]
fn rv64im_main_proof_rejects_tampered_recursion_proof_step_count() {
    let fixture = rv64im_n2_support::build_rv64im_n2_fixture().expect("build rv64im n=2 fixture");
    let mut published_main =
        build_rv64im_main_proof(&fixture.final_statement, &fixture.final_proof).expect("build published main proof");
    verify_rv64im_main_proof(&published_main).expect("baseline published main proof must verify");

    rv64im_main_recursion_proof_pop_last_step_proof(published_main.recursion_proof_mut())
        .expect("pop one step proof from the recursion proof carrier");

    let err = verify_rv64im_main_proof(&published_main).expect_err("tampered recursion proof step count must fail");
    assert!(err.to_string().contains("step count"));
}
