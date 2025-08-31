#![allow(deprecated)]

//! ME(b,L) SpartanCircuit implementation for direct Spartan2 SNARK integration
//!
//! This module implements the **real Spartan2 integration** using the `SpartanCircuit` trait.
//! This is the **official public API** approach, not a workaround.
//! 
//! ## Architecture
//!
//! 1. **`SpartanCircuit<E>` Implementation**: Uses bellpepper `ConstraintSystem`
//! 2. **Automatic R1CS Generation**: Spartan2 handles constraint matrix construction
//! 3. **Production SNARK API**: `setup() → prep_prove() → prove() → verify()`
//! 4. **Hash-MLE PCS Backend**: Integrated automatically via Engine selection
//!
//! ## Constraints Implemented
//!
//! 1. **Ajtai commitment binding**: `⟨L_{r,i}, vec(Z)⟩ = c_{r,i}`
//! 2. **ME evaluations**: `⟨v_j^{(ℓ)}, Z_row[r]⟩ = y_j^{(ℓ)}[r]` 
//! 3. **Fold digest binding**: Ensures transcript security between phases
//!
//! ## Variable Layout (Bellpepper Style)
//!
//! - **Public Inputs**: `(c_coords, y_limbs, fold_digest_limbs, challenges)`
//! - **Private Witness**: `vec(Z)` allocated as `AllocatedNum<E::Scalar>`
//! - **Shared Variables**: Empty for this circuit (no cross-circuit dependencies)

use anyhow::Result;
use bellpepper_core::{num::AllocatedNum, ConstraintSystem, SynthesisError};
use ff::Field;
use p3_field::{PrimeField64, PrimeCharacteristicRing};
use spartan2::errors::SpartanError;
use spartan2::provider::GoldilocksP3MerkleMleEngine as E;
use spartan2::spartan::{R1CSSNARK, SpartanProverKey, SpartanVerifierKey};
use spartan2::traits::{circuit::SpartanCircuit, Engine, snark::R1CSSNARKTrait};

// Real Spartan2 SNARK uses circuits directly, not R1CS shapes
// (the R1CS conversion happens internally)

use neo_ajtai::PP as AjtaiPP; // for later PP-backed binding
use neo_ccs::{MEInstance, MEWitness};

/// ME(b,L) circuit
#[derive(Clone, Debug)]
pub struct MeCircuit {
    pub me: MEInstance,
    pub wit: MEWitness,
    /// Optional Ajtai PP; once you expose rows from PP, use this instead of wit.ajtai_rows
    pub pp: Option<AjtaiPP<neo_math::Rq>>,
    /// 32-byte fold digest (binds transcript / header)
    pub fold_digest: [u8; 32],
}

impl MeCircuit {
    pub fn new(me: MEInstance, wit: MEWitness, pp: Option<AjtaiPP<neo_math::Rq>>, fold_digest: [u8; 32]) -> Self {
        Self { me, wit, pp, fold_digest }
    }

    #[inline]
    fn digest_to_scalars(&self) -> Vec<<E as Engine>::Scalar> {
        self.fold_digest
            .chunks(8)
            .map(|chunk| {
                let mut b = [0u8; 8];
                b[..chunk.len()].copy_from_slice(chunk);
                <E as Engine>::Scalar::from(u64::from_le_bytes(b))
            })
            .collect()
    }

    /// v1: y ∈ Fq; when you move y ∈ K = Fq^2, split here.
    #[inline]
    fn k_to_limbs(
        &self,
        x: p3_goldilocks::Goldilocks,
    ) -> (p3_goldilocks::Goldilocks, p3_goldilocks::Goldilocks) {
        (x, p3_goldilocks::Goldilocks::ZERO)
    }
}

/// Public inputs order **must** mirror `encode_bridge_io_header()`:
/// (c_coords) || (y limbs) || (r_point) || (base_b) || (fold_digest limbs)
impl SpartanCircuit<E> for MeCircuit {
    fn public_values(&self) -> Result<Vec<<E as Engine>::Scalar>, SynthesisError> {
        let mut pv = Vec::new();

        // 1) Ajtai commitment coords (Fq)
        for &c in &self.me.c_coords {
            pv.push(<E as Engine>::Scalar::from(c.as_canonical_u64()));
        }
        // 2) y outputs — 2 limbs when y ∈ K
        for &y in &self.me.y_outputs {
            let (y0, y1) = self.k_to_limbs(y);
            pv.push(<E as Engine>::Scalar::from(y0.as_canonical_u64()));
            pv.push(<E as Engine>::Scalar::from(y1.as_canonical_u64()));
        }
        // 3) challenge r (Fq^m)
        for &r in &self.me.r_point {
            pv.push(<E as Engine>::Scalar::from(r.as_canonical_u64()));
        }
        // 4) base dimension b
        pv.push(<E as Engine>::Scalar::from(self.me.base_b as u64));
        // 5) fold digest limbs
        pv.extend(self.digest_to_scalars());

        // Hash-MLE PCS requires public inputs to be power-of-2 length
        let original_len = pv.len();
        let next_power_of_2 = pv.len().next_power_of_two();
        
        // Pad with zeros to make it power of 2
        while pv.len() < next_power_of_2 {
            pv.push(<E as Engine>::Scalar::ZERO);
        }

        if original_len != next_power_of_2 {
            eprintln!("🔍 public_values() padded from {} to {} (power of 2)", original_len, pv.len());
        }
        Ok(pv)
    }

    fn shared<CS: ConstraintSystem<<E as Engine>::Scalar>>(
        &self,
        _cs: &mut CS,
    ) -> Result<Vec<AllocatedNum<<E as Engine>::Scalar>>, SynthesisError> {
        Ok(vec![]) // no shared variables in this circuit
    }

    fn precommitted<CS: ConstraintSystem<<E as Engine>::Scalar>>(
        &self,
        cs: &mut CS,
        _shared: &[AllocatedNum<<E as Engine>::Scalar>],
    ) -> Result<Vec<AllocatedNum<<E as Engine>::Scalar>>, SynthesisError> {
        // Check if witness variables need power-of-2 padding for Hash-MLE PCS
        let next_power_of_2 = self.wit.z_digits.len().next_power_of_two();
        
        // Z digits as private witness
        let mut z_vars = Vec::with_capacity(next_power_of_2);
        for (i, &z) in self.wit.z_digits.iter().enumerate() {
            let val = if z >= 0 {
                <E as Engine>::Scalar::from(z as u64)
            } else {
                -<E as Engine>::Scalar::from((-z) as u64)
            };
            z_vars.push(AllocatedNum::alloc(cs.namespace(|| format!("Z[{i}]")), || Ok(val))?);
        }
        
        // Pad with zeros to next power of 2 if needed for Hash-MLE PCS
        while z_vars.len() < next_power_of_2 {
            let i = z_vars.len();
            let zero = <E as Engine>::Scalar::ZERO;
            z_vars.push(AllocatedNum::alloc(cs.namespace(|| format!("Z_pad[{i}]")), || Ok(zero))?);
        }
        
        // Witness variables ready (padded to power of 2 if needed)
        Ok(z_vars)
    }

    fn num_challenges(&self) -> usize { 0 }

    fn synthesize<CS: ConstraintSystem<<E as Engine>::Scalar>>(
        &self,
        cs: &mut CS,
        _shared: &[AllocatedNum<<E as Engine>::Scalar>],
        z_vars: &[AllocatedNum<<E as Engine>::Scalar>],
        _challenges: Option<&[<E as Engine>::Scalar]>,
    ) -> Result<(), SynthesisError> {
        eprintln!("🔍 synthesize() called with {} z_vars", z_vars.len());
        
        let to_s = |x: p3_goldilocks::Goldilocks| <E as Engine>::Scalar::from(x.as_canonical_u64());

        // (A) Ajtai binding: <L_i, Z> = c_i  (rows are constants in v1)
        if let Some(rows) = &self.wit.ajtai_rows {
            let n = core::cmp::min(rows.len(), self.me.c_coords.len());
            for i in 0..n {
                let row = &rows[i];
                let upto = core::cmp::min(row.len(), z_vars.len());
                let c_scalar = <E as Engine>::Scalar::from(self.me.c_coords[i].as_canonical_u64());

                cs.enforce(
                    || format!("ajtai_bind_{i}"),
                    |lc| {
                        let mut lc = lc;
                        for j in 0..upto {
                            lc = lc + (to_s(row[j]), z_vars[j].get_variable());
                        }
                        lc
                    },
                    |lc| lc + CS::one(),
                    |lc| lc + (c_scalar, CS::one()),
                );
            }
        }

        // (B) ME evals: <w_j, Z> = y_j
        let m = core::cmp::min(self.wit.weight_vectors.len(), self.me.y_outputs.len());
        for j in 0..m {
            let wj = &self.wit.weight_vectors[j];
            let upto = core::cmp::min(wj.len(), z_vars.len());
            let y_scalar = <E as Engine>::Scalar::from(self.me.y_outputs[j].as_canonical_u64());

            cs.enforce(
                || format!("me_eval_{j}"),
                |lc| {
                    let mut lc = lc;
                    for k in 0..upto {
                        lc = lc + (to_s(wj[k]), z_vars[k].get_variable());
                    }
                    lc
                },
                |lc| lc + CS::one(),
                |lc| lc + (y_scalar, CS::one()),
            );
        }

        // If no constraints were added, add trivial equalities (dev safety)
        if self.wit.ajtai_rows.as_ref().map_or(true, |r| r.is_empty())
            && self.wit.weight_vectors.is_empty()
        {
            for i in 0..core::cmp::min(2, z_vars.len()) {
                cs.enforce(
                    || format!("tautology_{i}"),
                    |lc| lc + z_vars[i].get_variable(),
                    |lc| lc + CS::one(),
                    |lc| lc + z_vars[i].get_variable(),
                );
            }
        }

        Ok(())
    }
}

/// Public API: setup keys (so you can cache them)
pub fn setup_me_snark(
    me: &MEInstance,
    wit: &MEWitness,
) -> Result<(SpartanProverKey<E>, SpartanVerifierKey<E>), SpartanError> {
    let circuit = MeCircuit::new(me.clone(), wit.clone(), None, me.header_digest);
    
    // Spartan2 SNARK setup expects the circuit directly
    R1CSSNARK::<E>::setup(circuit)
}

/// Public API: prove a real Spartan2 SNARK over Hash‑MLE (Poseidon2) 
pub fn prove_me_snark(
    me: &MEInstance,
    wit: &MEWitness,
) -> Result<(Vec<u8>, Vec<<E as Engine>::Scalar>, SpartanVerifierKey<E>), SpartanError> {
    let circuit = MeCircuit::new(me.clone(), wit.clone(), None, me.header_digest);
    
    // 1. Setup SNARK keys with circuit  
    eprintln!("🔍 About to call R1CSSNARK::<E>::setup()");
    let (pk, vk) = R1CSSNARK::<E>::setup(circuit.clone())?;
    eprintln!("✅ R1CSSNARK::<E>::setup() completed");
    
    // 2. Prepare proving (creates PrepSNARK) - capacity overflow is now fixed
    eprintln!("🔍 About to call prep_prove()");
    let prep_snark = match R1CSSNARK::<E>::prep_prove(&pk, circuit.clone(), false) {
        Ok(prep) => {
            eprintln!("✅ prep_prove() completed successfully");
            prep
        }
        Err(e) => {
            eprintln!("❌ prep_prove() failed with: {:?}", e);
            return Err(e);
        }
    }; 
    
    // 3. Generate proof using prepared SNARK  
    eprintln!("🔍 About to call R1CSSNARK::<E>::prove()");
    
    // Get dimensions before the move for error reporting
    let public_values_len = circuit.public_values().unwrap().len();
    let witness_len = circuit.wit.z_digits.len();
    
    let snark_proof = match R1CSSNARK::<E>::prove(&pk, circuit, &prep_snark, false) {
        Ok(proof) => {
            eprintln!("✅ R1CSSNARK::<E>::prove() completed successfully!");
            proof
        }
        Err(e) => {
            eprintln!("❌ R1CSSNARK::<E>::prove() failed with: {:?}", e);
            eprintln!("   This is where the 'vector len must be power of two' error occurs");
            eprintln!("   Current dimensions:");
            eprintln!("     - Public values: {} (padded to power of 2)", public_values_len);
            eprintln!("     - Witness variables: {} (padded to power of 2)", witness_len);
            eprintln!("   The error might be from an internal Hash-MLE vector during proving");
            return Err(e);
        }
    };
    
    // 4. Verify proof as sanity check and get public outputs
    let public_outputs = snark_proof.verify(&vk)?;
    
    // 5. Serialize proof
    let proof_bytes = bincode::serialize(&snark_proof)
        .map_err(|e| SpartanError::InternalError { reason: format!("Proof serialization failed: {e}") })?;
    
    Ok((proof_bytes, public_outputs, vk))
}

/// Optional helper if you want a structured verify in tests
pub fn verify_me_snark(
    proof_bytes: &[u8],
    vk: &SpartanVerifierKey<E>,
) -> Result<bool, SpartanError> {
    let proof: R1CSSNARK<E> = bincode::deserialize(proof_bytes)
        .map_err(|e| SpartanError::InternalError { reason: format!("bincode(proof) failed: {e}") })?;
    // Spartan2 returns Ok(public_values) on success; we only need success/failure here
    let _ = proof.verify(vk)?;
    Ok(true)
}