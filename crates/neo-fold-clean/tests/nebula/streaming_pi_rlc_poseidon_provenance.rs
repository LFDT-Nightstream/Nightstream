//! Focused production PiRLC Poseidon2 source-to-final provenance check.

use neo_fold_clean::frontends::nebula::f_prime::{
    production_pi_rlc_family_body_compiler_audit, production_pi_rlc_family_body_projected_rows_with_source_provenance,
};
use neo_fold_clean::frontends::r1cs_f_prime::SelectiveRewriteKind;

const SOURCE_START: usize = 165_446;
const SOURCE_END: usize = 166_046;
const EMITTED_START: usize = 74_375;
const EMITTED_END: usize = 74_461;

#[test]
fn production_even_replay_poseidon2_exports_exact_sbox_provenance() {
    let compiler = production_pi_rlc_family_body_compiler_audit().expect("production PiRLC compiler audit");
    let rewrite = compiler
        .rows()
        .rewrites()
        .iter()
        .find(|rewrite| {
            rewrite.arm() == 0
                && rewrite.kind() == SelectiveRewriteKind::Poseidon2
                && rewrite.emitted_rows() == (EMITTED_START..EMITTED_END)
        })
        .expect("first even replay Poseidon2 rewrite");
    assert_eq!(rewrite.source_rows(), &[SOURCE_START..SOURCE_END]);

    let selected_rows = rewrite.emitted_rows().collect::<Vec<_>>();
    let projected = production_pi_rlc_family_body_projected_rows_with_source_provenance(&selected_rows, 0, &[], &[])
        .expect("exact production Poseidon2 provenance");
    let source = projected
        .source_provenance()
        .expect("complete production source provenance");
    let steps = source.poseidon2_sbox_steps();

    assert_eq!(steps.len(), EMITTED_END - EMITTED_START);
    for (offset, step) in steps.iter().enumerate() {
        assert_eq!(step.emitted_row(), EMITTED_START + offset);
        assert_eq!(step.rewrite_id(), rewrite.id().index());
        assert_eq!(step.source_rows(), &[(SOURCE_START, SOURCE_END)]);
        assert_eq!(step.output().terms().len(), 1);
    }
}
