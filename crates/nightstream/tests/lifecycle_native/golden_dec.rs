//! Replay every public PiDEC field and rejection case through the maintained verifier.

use super::*;
use crate::folding::{ajtai_dec_mixer, pi_dec, CeClaim, Params};
use neo_ccs::CcsStructure;
use neo_reductions::common::split_b_matrix_k_with_nonzero_flags;

const CHILDREN: usize = 16;
const MATRICES: usize = 14;
const PUBLIC: usize = 270;
const MODULUS: u64 = F::ORDER_U64;
const BOUND: u64 = 1 << CHILDREN;

fn magnitude(word: u64) -> u64 {
    word.min(MODULUS - word)
}

fn result_value(
    params: &Params,
    structure: &CcsStructure<F>,
    parent: &CeClaim,
    children: &[CeClaim],
    state: [F; 8],
) -> Value {
    let proof = pi_dec::Proof {
        children: children.to_vec(),
    };
    let accepted = pi_dec::verify(params, structure, ajtai_dec_mixer, parent, &proof).is_ok();
    let (digits, _) =
        split_b_matrix_k_with_nonzero_flags(&parent.X, CHILDREN, 2).expect("bounded actual public parent");
    let public_digits = digits.iter().map(public_words).collect::<Vec<_>>();
    let parent_bounds = public_words(&parent.X)
        .into_iter()
        .map(|word| u64::from(magnitude(word) < BOUND))
        .collect::<Vec<_>>();
    let ranges = public_digits
        .iter()
        .map(|digit| {
            digit
                .iter()
                .map(|&word| u64::from(magnitude(word) < 2))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let recomposed_c = ajtai_dec_mixer(
        &children
            .iter()
            .map(|child| child.c.clone())
            .collect::<Vec<_>>(),
        2,
    );
    let mut recomposed_x = vec![F::ZERO; PUBLIC];
    let mut recomposed_k = vec![K::ZERO; D];
    let mut recomposed_a = vec![vec![K::ZERO; D]; MATRICES];
    for (index, child) in children.iter().enumerate() {
        let weight = F::from_u64(1 << index);
        let extension_weight = K::from(weight);
        for (coordinate, value) in recomposed_x.iter_mut().enumerate() {
            *value += digits[index][(coordinate % D, coordinate / D)] * weight;
        }
        for coordinate in 0..D {
            recomposed_k[coordinate] += child.eval_k[coordinate] * extension_weight;
            for matrix in 0..MATRICES {
                recomposed_a[matrix][coordinate] += child.eval_a[matrix][coordinate] * extension_weight;
            }
        }
    }
    // Public outputs belong to the verifier's canonical split and shared point.
    let computed_children = children
        .iter()
        .zip(digits)
        .map(|(child, public)| {
            let mut computed = child.clone();
            computed.X = public;
            computed.r = parent.r.clone();
            computed
        })
        .collect::<Vec<_>>();
    let mut unbounded = parent.clone();
    unbounded.X[(0, 0)] = F::from_u64(BOUND);
    let unbounded_rejected = split_b_matrix_k_with_nonzero_flags(&unbounded.X, CHILDREN, 2).is_err()
        && pi_dec::verify(params, structure, ajtai_dec_mixer, &unbounded, &proof).is_err();
    json!([
        u64::from(accepted),
        u64::from(parent_bounds.iter().all(|&check| check == 1)),
        public_digits,
        parent_bounds,
        ranges,
        fields(&recomposed_c.data),
        u64::from(recomposed_c == parent.c),
        fields(&recomposed_x),
        u64::from(fields(&recomposed_x) == public_words(&parent.X)),
        words(&recomposed_k),
        u64::from(recomposed_k == parent.eval_k[..D]),
        recomposed_a
            .iter()
            .map(|family| words(family))
            .collect::<Vec<_>>(),
        u64::from(
            recomposed_a
                .iter()
                .zip(&parent.eval_a)
                .all(|(left, right)| left == &right[..D])
        ),
        computed_children
            .iter()
            .map(|child| claim_value(child, false))
            .collect::<Vec<_>>(),
        fields(&state),
        u64::from(unbounded_rejected),
        if accepted {
            json!([1, running_value(&computed_children)])
        } else {
            json!([0])
        }
    ])
}

pub(super) fn check_saved(package_path: &Path, actual: &Value, reference: &Value) {
    let package_bytes = fs::read(package_path).expect("selected package");
    let package = nightstream_fprime::load_poseidon2_hash_chain_v1_package(&package_bytes)
        .expect("verifier-owned selected package");
    assert_eq!(json!(package.structural_identifier()), actual["structural_identifier"]);
    assert_eq!(
        json!(package
            .production_verifier_binding()
            .unwrap()
            .package_identity()),
        actual["package_identity"]
    );
    let structure = package.ccs_structure_header().unwrap();
    drop((package, package_bytes));
    let params = Params::production();
    let proof = crate::lifecycle::tests::proof(actual);
    let parent = &proof.pi_rlc.combined;
    let children = &proof.pi_dec.children;
    let state = crate::lifecycle::tests::fields(&actual["outgoing_state"])
        .try_into()
        .unwrap();
    let computed = result_value(&params, &structure, parent, children, state);
    let expected = reference[9].as_array().expect("complete Lean D result");
    assert_eq!(expected.len(), 17);
    for (index, (observed, expected)) in computed
        .as_array()
        .unwrap()
        .iter()
        .zip(expected)
        .enumerate()
    {
        assert_eq!(observed, expected, "saved actual PiDEC result field {index}");
    }
    mutations(&params, &structure, parent, children);
    println!("saved_actual_pi_dec=passed complete_fields=17 normal_wrapper=true");
}

fn rejected(name: &str, params: &Params, structure: &CcsStructure<F>, parent: &CeClaim, children: &[CeClaim]) {
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

fn mutations(params: &Params, structure: &CcsStructure<F>, parent: &CeClaim, children: &[CeClaim]) {
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
