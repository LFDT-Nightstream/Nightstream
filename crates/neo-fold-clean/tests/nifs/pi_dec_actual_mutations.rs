//! Reject changes to the actual nonzero PiDEC phase and final output.

use neo_ajtai::Commitment;
use neo_ccs::{CcsStructure, CeClaim};
use neo_fold_clean::paper::{params::Params, pi_dec, relations::ajtai_dec_mixer};
use neo_math::{D, F, K};
use p3_field::PrimeCharacteristicRing;

type Claim = CeClaim<Commitment, F, K>;

fn rejected(name: &str, params: &Params, structure: &CcsStructure<F>, parent: &Claim, children: &[Claim]) {
    let result = pi_dec::verify(
        params,
        structure,
        ajtai_dec_mixer,
        parent,
        &pi_dec::Proof {
            children: children.to_vec(),
        },
    );
    assert!(result.is_err(), "native PiDEC accepted mutation {name}");
    println!("pi_dec_mutation={name} rejected={:?}", result.unwrap_err());
}

pub(super) fn check(params: &Params, structure: &CcsStructure<F>, parent: &Claim, children: &[Claim]) {
    for child in 0..children.len() {
        let mut changed = children.to_vec();
        changed[child].c.data[0] += F::ONE;
        rejected(
            &format!("child_{child}_commitment"),
            params,
            structure,
            parent,
            &changed,
        );
    }
    for family in 0..structure.t() + 3 {
        let mut changed = children.to_vec();
        let name = match family {
            0 => {
                changed[0].X[(0, 0)] += F::ONE;
                "child_public".to_owned()
            }
            1 => {
                changed[0].r[0] += K::ONE;
                "child_point".to_owned()
            }
            2 => {
                changed[0].eval_k[0] += K::ONE;
                "child_eval_K".to_owned()
            }
            index => {
                changed[0].eval_a[index - 3][0] += K::ONE;
                format!("child_eval_A{}", index - 3)
            }
        };
        rejected(&name, params, structure, parent, &changed);
    }
    let mut changed = children.to_vec();
    changed.pop();
    rejected("missing_child", params, structure, parent, &changed);
    let mut changed = children.to_vec();
    changed[0].c.data.pop();
    rejected("short_commitment", params, structure, parent, &changed);
    let mut changed = children.to_vec();
    changed[0].r.pop();
    rejected("short_point", params, structure, parent, &changed);
    let mut changed = children.to_vec();
    changed[0].eval_k.pop();
    rejected("short_eval_K", params, structure, parent, &changed);
    let mut changed = children.to_vec();
    changed[0].eval_a.pop();
    rejected("missing_matrix", params, structure, parent, &changed);
    let mut changed = children.to_vec();
    changed[0].eval_k[D] = K::ONE;
    rejected("nonzero_eval_K_padding", params, structure, parent, &changed);
    for matrix in 0..structure.t() {
        let mut changed = children.to_vec();
        changed[0].eval_a[matrix][D] = K::ONE;
        rejected(
            &format!("nonzero_eval_A{matrix}_padding"),
            params,
            structure,
            parent,
            &changed,
        );
    }
    let mut changed = children.to_vec();
    changed[0].fold_digest[0] ^= 1;
    rejected("child_fold_digest", params, structure, parent, &changed);
    let mut changed = children.to_vec();
    changed[0].X[(0, 0)] = F::from_u64(2);
    rejected("child_digit_range", params, structure, parent, &changed);
}
