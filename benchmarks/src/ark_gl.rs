//! Arkworks Goldilocks field (p = 2^64 - 2^32 + 1) for ark-ff v0.4

use ark_ff::BigInt;
use ark_ff::fields::models::fp::{Fp64, MontBackend, MontConfig, MontFp};

/// Goldilocks field configuration for ark-ff 0.4
/// p = 18446744069414584321 = 2^64 - 2^32 + 1
pub struct GoldilocksConfig;

impl MontConfig<1> for GoldilocksConfig {
    /// The modulus: 2^64 - 2^32 + 1 = 18446744069414584321
    const MODULUS: BigInt<1> = BigInt::new([0xFFFFFFFF00000001]);
    
    /// A multiplicative generator (7)
    const GENERATOR: Fp64<MontBackend<Self, 1>> = MontFp!("7");
    
    /// 2^32-th root of unity = 7^((p-1)/2^32) mod p
    /// 0x1856_29DC_DA58_878C = 1763593322112749452
    const TWO_ADIC_ROOT_OF_UNITY: Fp64<MontBackend<Self, 1>> = MontFp!("1763593322112749452");
}

/// Arkworks Goldilocks field type for 0.4  
pub type ArkGL = Fp64<MontBackend<GoldilocksConfig, 1>>;
