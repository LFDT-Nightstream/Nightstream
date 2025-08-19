use neo_modint::*;

#[test]
fn test_ops() {
    let a = ModInt::from_u64(5);
    let b = ModInt::from_u64(3);
    assert_eq!(a + b, ModInt::from_u64(8));
    assert_eq!(a - b, ModInt::from_u64(2));
    assert_eq!(a * b, ModInt::from_u64(15));
    assert_eq!(-a, ModInt::from_u64(ModInt::Q - 5));
}

#[test]
fn test_overflow() {
    let max = ModInt::from_u64(ModInt::Q - 1);
    assert_eq!(max + ModInt::one(), ModInt::zero());
    assert_eq!(
        ModInt::zero() - ModInt::one(),
        ModInt::from_u64(ModInt::Q - 1)
    );
}

#[test]
fn test_inverse() {
    let a = ModInt::from_u64(5);
    let inv = a.inverse();
    assert_eq!(a * inv, ModInt::one());
}
