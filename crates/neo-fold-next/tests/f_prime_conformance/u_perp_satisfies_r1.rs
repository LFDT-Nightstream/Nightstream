//! HyperNova §6.2(4): the default instance-witness pair must literally satisfy
//! the carried R1 relation, not just look structurally zero-shaped.

use neo_fold_next::rv64im::audit::audit_rv64im_main_recursion_canonical_default_slot_satisfies_r1_literally;

#[test]
fn f_prime_default_slot_satisfies_r1_literally() {
    audit_rv64im_main_recursion_canonical_default_slot_satisfies_r1_literally()
        .expect("the canonical default slot must satisfy R1 literally");
}
