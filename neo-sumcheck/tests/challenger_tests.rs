use neo_sumcheck::*;

#[test]
fn challenge_rotation_is_always_invertible() {
    let mut ch = NeoChallenger::new("test_rot");
    ch.observe_bytes("prefix", b"unit");
    for n in [2usize, 4, 8, 64] {
        for _ in 0..10 {
            let rho = ch.challenge_rotation("rlc_rho", n);
            assert!(rho.is_invertible(), "ρ must be invertible for n={}", n);
        }
    }
}
