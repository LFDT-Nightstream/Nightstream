//! HyperNova §6.2(4): the default instance-witness pair must literally satisfy
//! the carried R1 relation, not just look structurally zero-shaped.

use neo_fold_next::rv64im::audit::audit_rv64im_main_recursion_default_carry_satisfies_r1_literally;

use super::support::single_step_advices;

#[test]
fn f_prime_default_carry_satisfies_r1_literally() {
    let advices = single_step_advices();
    audit_rv64im_main_recursion_default_carry_satisfies_r1_literally(advices[0].fresh_state_out())
        .expect("the canonical default carry must satisfy R1 literally");
}
