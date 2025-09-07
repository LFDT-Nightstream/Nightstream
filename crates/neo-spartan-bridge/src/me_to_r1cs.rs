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
use spartan2::provider::GoldilocksMerkleMleEngine as E;
use spartan2::spartan::{R1CSSNARK, SpartanProverKey, SpartanVerifierKey};
use spartan2::traits::{circuit::SpartanCircuit, Engine, pcs::PCSEngineTrait, snark::R1CSSNARKTrait};

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
    #[allow(dead_code)]
    fn k_to_limbs(
        &self,
        x: p3_goldilocks::Goldilocks,
    ) -> (p3_goldilocks::Goldilocks, p3_goldilocks::Goldilocks) {
        (x, p3_goldilocks::Goldilocks::ZERO)
    }

    #[inline]
    #[allow(dead_code)]
    fn is_pow2(n: usize) -> bool {
        n.is_power_of_two()
    }
}

/// Public inputs order **must** mirror `encode_bridge_io_header()`:
/// (c_coords) || (y limbs) || (r_point) || (base_b) || (fold_digest limbs)
impl SpartanCircuit<E> for MeCircuit {
    fn public_values(&self) -> Result<Vec<<E as Engine>::Scalar>, SynthesisError> {
        // TRANSCRIPT CONSISTENCY FIX: Return the SAME values that inputize() creates
        // Both prover (via this method) and verifier (via U.validate) must absorb identical data
        let mut pv = Vec::new();

        // 1) Ajtai commitment coords (Fq) - SAME ORDER as inputize() calls
        for &c in &self.me.c_coords {
            pv.push(<E as Engine>::Scalar::from(c.as_canonical_u64()));
        }
        // 2) y outputs — already flattened (K -> [F;2]) in adapter
        for &y_limb in &self.me.y_outputs {
            pv.push(<E as Engine>::Scalar::from(y_limb.as_canonical_u64()));
        }
        // 3) challenge r (Fq^m)
        for &r in &self.me.r_point {
            pv.push(<E as Engine>::Scalar::from(r.as_canonical_u64()));
        }
        // 4) base dimension b
        pv.push(<E as Engine>::Scalar::from(self.me.base_b as u64));
        
        // 5) Pad to power-of-2 BEFORE digest - CRITICAL: match encode_bridge_io_header() order
        let current_count_without_digest = self.me.c_coords.len() + self.me.y_outputs.len() + 
                                           self.me.r_point.len() + 1;
        let digest_scalars = self.digest_to_scalars();
        let total_needed = current_count_without_digest + digest_scalars.len();
        let target_count = total_needed.next_power_of_two();
        let padding_needed = target_count - total_needed;
        
        // Add padding zeros BEFORE digest to match encode_bridge_io_header() order
        for _ in 0..padding_needed {
            pv.push(<E as Engine>::Scalar::ZERO);
        }
        
        // 6) fold digest limbs LAST (after padding)
        pv.extend(digest_scalars);

        if padding_needed > 0 {
            eprintln!("🔧 TRANSCRIPT FIX: public_values() padded from {} to {} (matches encode_bridge_io_header)", total_needed, pv.len());
        }
        Ok(pv)
    }

    fn shared<CS: ConstraintSystem<<E as Engine>::Scalar>>(
        &self,
        _cs: &mut CS,
    ) -> Result<Vec<AllocatedNum<<E as Engine>::Scalar>>, SynthesisError> {
        eprintln!("✅ shared(): EMPTY (Hash-MLE single-blind strategy)");
        Ok(vec![])
    }

    fn precommitted<CS: ConstraintSystem<<E as Engine>::Scalar>>(
        &self,
        _cs: &mut CS,
        _shared: &[AllocatedNum<<E as Engine>::Scalar>],
    ) -> Result<Vec<AllocatedNum<<E as Engine>::Scalar>>, SynthesisError> {
        eprintln!("🔧 precommitted(): EMPTY (avoiding multiple blinds for Hash-MLE)");
        eprintln!("   Hash-MLE PCS requires exactly 1 blind, so we use only REST segment");
        Ok(vec![])
    }

    fn num_challenges(&self) -> usize { 0 }

    fn synthesize<CS: ConstraintSystem<<E as Engine>::Scalar>>(
        &self,
        cs: &mut CS,
        _shared: &[AllocatedNum<<E as Engine>::Scalar>],  // empty now
        _z_pre: &[AllocatedNum<<E as Engine>::Scalar>],  // empty
        _challenges: Option<&[<E as Engine>::Scalar]>,
    ) -> Result<(), SynthesisError> {
        // SECURITY: refuse to synthesize an unbound circuit
        if self.wit.ajtai_rows.as_ref().map_or(true, |r| r.is_empty()) {
            eprintln!("❌ SECURITY: Ajtai rows missing; refusing to synthesize unbound circuit.");
            return Err(SynthesisError::AssignmentMissing);
        }

        let mut z_vars = Vec::<AllocatedNum<<E as Engine>::Scalar>>::new();
        
        // 1) Allocate REST witness variables exactly in the z_digits order
        #[cfg(feature = "debug-logs")]
        // eprintln!("🔍 Witness allocation (REST segment - z_digits first):");
        for (i, &z) in self.wit.z_digits.iter().enumerate() {
            let val = if z >= 0 {
                <E as Engine>::Scalar::from(z as u64)
            } else {
                -<E as Engine>::Scalar::from((-z) as u64)
            };
            #[cfg(feature = "debug-logs")]
            // eprintln!("   z_vars[{}] = {} (from z_digits[{}] = {})", i, val.to_canonical_u64(), i, z);
            z_vars.push(AllocatedNum::alloc(cs.namespace(|| format!("REST_Z[{i}]")), || Ok(val))?);
        }
        
        // 2) Pad REST (only REST) to a power-of-two for Hash-MLE
        let orig_rest = z_vars.len();
        let target_rest = if orig_rest <= 1 { 1 } else { orig_rest.next_power_of_two() };
        for i in orig_rest..target_rest {
            z_vars.push(AllocatedNum::alloc(cs.namespace(|| format!("REST_Z_pad[{i}]")), 
                || Ok(<E as Engine>::Scalar::ZERO))?);
        }
        eprintln!("🔍 synthesize(): REST padded from {} to {} (power-of-two)", orig_rest, z_vars.len());
        debug_assert!(z_vars.len().is_power_of_two());
        
        let mut n_constraints = 0usize;
        let to_s = |x: p3_goldilocks::Goldilocks| <E as Engine>::Scalar::from(x.as_canonical_u64());

        // Enforce padded limbs are exactly zero (avoid unconstrained witness coords)
        if target_rest > orig_rest {
            let mut pad_ct = 0usize;
            for i in orig_rest..target_rest {
                cs.enforce(
                    || format!("pad_zero_{}", i),
                    |lc| lc + z_vars[i].get_variable(),
                    |lc| lc + CS::one(),
                    |lc| lc, // == 0
                );
                pad_ct += 1;
            }
            n_constraints += pad_ct;
            eprintln!("🔧 Enforced {} padded REST limbs to zero", pad_ct);
        }

        // (RANGE) Enforce |Z_i| < b  (digits ∈ {-(b-1), …, (b-1)})
        // For small b (e.g., b=2 => {-1,0,1}), constrain with a product polynomial.
        // For b=2: v*(v-1)*(v+1) == 0.
        for (i, z) in z_vars.iter().enumerate().take(orig_rest) { // Only apply to original witness, not padding
            if self.me.base_b == 2 {
                let one = <E as Engine>::Scalar::ONE;

                // (z)*(z-1)*(z+1) == 0
                // Implement as three constraints using a temp:
                let z_minus_one = AllocatedNum::alloc(cs.namespace(|| format!("z_minus_one_{}", i)), || {
                    Ok(z.get_value().ok_or(SynthesisError::AssignmentMissing)? - one)
                })?;
                cs.enforce(
                    || format!("link_z_minus_one_{}", i),
                    |lc| lc + z.get_variable(),
                    |lc| lc + CS::one(),
                    |lc| lc + z_minus_one.get_variable() + (one, CS::one()),
                );
                let z_plus_one = AllocatedNum::alloc(cs.namespace(|| format!("z_plus_one_{}", i)), || {
                    Ok(z.get_value().ok_or(SynthesisError::AssignmentMissing)? + one)
                })?;
                cs.enforce(
                    || format!("link_z_plus_one_{}", i),
                    |lc| lc + z.get_variable(),
                    |lc| lc + CS::one(),
                    |lc| lc + z_plus_one.get_variable() + (-one, CS::one()),
                );
                // w := z * (z-1)
                let w = AllocatedNum::alloc(cs.namespace(|| format!("w_{}", i)), || {
                    let zv = z.get_value().ok_or(SynthesisError::AssignmentMissing)?;
                    let zm1 = z_minus_one.get_value().ok_or(SynthesisError::AssignmentMissing)?;
                    Ok(zv * zm1)
                })?;
                cs.enforce(
                    || format!("w=z*(z-1)_{}", i),
                    |lc| lc + z.get_variable(),
                    |lc| lc + z_minus_one.get_variable(),
                    |lc| lc + w.get_variable(),
                );
                // w * (z+1) == 0
                cs.enforce(
                    || format!("range_b2_{}", i),
                    |lc| lc + w.get_variable(),
                    |lc| lc + z_plus_one.get_variable(),
                    |lc| lc,
                );
                n_constraints += 4;
            } else {
                // Generic small-b range: ∏_{t=-(b-1)}^{b-1} (z - t) == 0
                // Keep it simple for now (b is small).
                let mut acc = AllocatedNum::alloc(
                    cs.namespace(|| format!("range_acc_init_{}", i)),
                    || Ok(<E as Engine>::Scalar::ONE)
                )?;
                // CRITICAL: anchor the accumulator to ONE
                // Enforce acc * 1 = 1  ⇒  acc == 1
                cs.enforce(
                    || format!("range_acc_init_is_one_{}", i),
                    |lc| lc + acc.get_variable(),
                    |lc| lc + CS::one(),
                    |lc| lc + CS::one(),
                );
                n_constraints += 1;
                for t in (-(self.me.base_b as i64 - 1))..=(self.me.base_b as i64 - 1) {
                    let c = if t >= 0 { 
                        <E as Engine>::Scalar::from(t as u64) 
                    } else { 
                        -<E as Engine>::Scalar::from((-t) as u64) 
                    };
                    // lin := (z - t)
                    let lin = AllocatedNum::alloc(cs.namespace(|| format!("range_lin_{}_{}", i, t)), || {
                        let zv = z.get_value().ok_or(SynthesisError::AssignmentMissing)?;
                        Ok(zv - c)
                    })?;
                    // enforce lin == z - t
                    cs.enforce(
                        || format!("lin=z-{}_digit_{}", t, i),
                        |lc| lc + z.get_variable(),
                        |lc| lc + CS::one(),
                        |lc| lc + lin.get_variable() + (c, CS::one()),
                    );
                    // acc := acc * lin
                    let new_acc = AllocatedNum::alloc(cs.namespace(|| format!("range_acc_next_{}_{}", i, t)), || {
                        Ok(acc.get_value().ok_or(SynthesisError::AssignmentMissing)? *
                           lin.get_value().ok_or(SynthesisError::AssignmentMissing)?)
                    })?;
                    cs.enforce(
                        || format!("acc_mul_{}_{}", i, t),
                        |lc| lc + acc.get_variable(),
                        |lc| lc + lin.get_variable(),
                        |lc| lc + new_acc.get_variable(),
                    );
                    acc = new_acc;
                    n_constraints += 2;
                }
                // acc must be 0
                cs.enforce(
                    || format!("range_final_zero_{}", i),
                    |lc| lc + acc.get_variable(),
                    |lc| lc + CS::one(),
                    |lc| lc,
                );
                n_constraints += 1;
            }
        }

        // (A) Ajtai binding: <L_i, Z> = c_i  (rows are constants in v1)
        if let Some(rows) = &self.wit.ajtai_rows {
            let n = core::cmp::min(rows.len(), self.me.c_coords.len());
            for i in 0..n {
                let row = &rows[i];
                let upto = core::cmp::min(row.len(), z_vars.len());
                let _c_scalar = <E as Engine>::Scalar::from(self.me.c_coords[i].as_canonical_u64());

                // 🔧 DEBUG: Compute expected LHS value for constraint debugging
                let mut expected_lhs = <E as Engine>::Scalar::ZERO;
                for j in 0..upto {
                    let row_coeff = to_s(row[j]);
                    let z_val = if j < self.wit.z_digits.len() {
                        let z = self.wit.z_digits[j];
                        if z >= 0 { <E as Engine>::Scalar::from(z as u64) }
                        else { -<E as Engine>::Scalar::from((-z) as u64) }
                    } else {
                        <E as Engine>::Scalar::ZERO // padded variables
                    };
                    expected_lhs += row_coeff * z_val;
                }
                #[cfg(feature = "debug-logs")]
                // eprintln!("🔍 Ajtai constraint {}: computed LHS = {}, expected RHS = {}", 
                //     i, expected_lhs.to_canonical_u64(), _c_scalar.to_canonical_u64());

                cs.enforce(
                    || format!("ajtai_bind_{i}"),
                    |lc| {
                        let mut lc = lc;
                        for j in 0..upto {
                            lc = lc + (to_s(row[j]), z_vars[j].get_variable());
                        }
                        lc
                    },
                    |lc| lc + CS::one(),            // multiply by constant 1 (X side)
                    |lc| lc + (c_scalar, CS::one()),// right side also uses constant 1
                );
                n_constraints += 1;
            }
        }

        // (B) ME evals: <w_j, Z> = y_j  
        let m = core::cmp::min(self.wit.weight_vectors.len(), self.me.y_outputs.len());
        for j in 0..m {
            let wj = &self.wit.weight_vectors[j];
            let upto = core::cmp::min(wj.len(), z_vars.len());
            let _y_scalar = <E as Engine>::Scalar::from(self.me.y_outputs[j].as_canonical_u64());

            // 🔧 DEBUG: Compute expected LHS value for ME constraint debugging
            let mut expected_lhs = <E as Engine>::Scalar::ZERO;
            for k in 0..upto {
                let w_coeff = to_s(wj[k]);
                let z_val = if k < self.wit.z_digits.len() {
                    let z = self.wit.z_digits[k];
                    if z >= 0 { <E as Engine>::Scalar::from(z as u64) }
                    else { -<E as Engine>::Scalar::from((-z) as u64) }
                } else {
                    <E as Engine>::Scalar::ZERO // padded variables
                };
                expected_lhs += w_coeff * z_val;
            }
            #[cfg(feature = "debug-logs")]
            // eprintln!("🔍 ME constraint {}: computed LHS = {}, expected RHS = {}", 
            //     j, expected_lhs.to_canonical_u64(), _y_scalar.to_canonical_u64());

            cs.enforce(
                || format!("me_eval_{j}"),
                |lc| {
                    let mut lc = lc;
                    for k in 0..upto {
                        lc = lc + (to_s(wj[k]), z_vars[k].get_variable());
                    }
                    lc
                },
                |lc| lc + CS::one(),               // multiply by constant 1
                |lc| lc + (y_scalar, CS::one()),   // RHS uses constant 1
            );
            n_constraints += 1;
        }

        // CRITICAL SECURITY: Reject vacuous circuits that have no binding to the statement
        if self.wit.ajtai_rows.as_ref().map_or(true, |r| r.is_empty())
            && self.wit.weight_vectors.is_empty()
        {
            eprintln!("❌ SECURITY VIOLATION: Cannot synthesize circuit with no Ajtai binding AND no ME constraints.");
            eprintln!("   This would create a vacuous proof with no binding to the statement.");
            return Err(SynthesisError::AssignmentMissing);
        }

        // All witness variables are now allocated as REST with power-of-2 length

        // --- Ensure TOTAL number of constraints is a power of two (Hash‑MLE requirement)
        let total_constraints_so_far = n_constraints;
        let want_constraints = if total_constraints_so_far <= 1 { 1 } else { total_constraints_so_far.next_power_of_two() };
        let need_to_add = want_constraints - total_constraints_so_far;
        if need_to_add > 0 {
            // Add 1*1 = 1 tautologies using CS::one()
            for i in 0..need_to_add {
                cs.enforce(
                    || format!("pad_const_tautology_{i}"),
                    |lc| lc + CS::one(),
                    |lc| lc + CS::one(),
                    |lc| lc + CS::one(),
                );
            }
            eprintln!("🔧 synthesize(): padded constraints {} → {} (total with shared = {})", total_constraints_so_far, want_constraints, want_constraints);
        } else {
            eprintln!("ℹ️  synthesize(): constraints = {} (total with shared = {} - power-of-two OK)", n_constraints, total_constraints_so_far);
        }

        // CRITICAL FIX: Create public inputs via inputize() for transcript fork compatibility
        // This replaces the old public_values() method approach
        
        // 1) Ajtai commitment coords (Fq)
        for (i, &c) in self.me.c_coords.iter().enumerate() {
            let c_val = <E as Engine>::Scalar::from(c.as_canonical_u64());
            let c_alloc = AllocatedNum::alloc(cs.namespace(|| format!("c_coord_{}", i)), || Ok(c_val))?;
            let _ = c_alloc.inputize(cs.namespace(|| format!("public_c_{}", i)));
        }
        
        // 2) y outputs — already flattened (K -> [F;2]) in adapter
        for (i, &y_limb) in self.me.y_outputs.iter().enumerate() {
            let v = <E as Engine>::Scalar::from(y_limb.as_canonical_u64());
            let a = AllocatedNum::alloc(cs.namespace(|| format!("y_output_limb_{}", i)), || Ok(v))?;
            let _ = a.inputize(cs.namespace(|| format!("public_y_limb_{}", i)));
        }
        
        // 3) challenge r (Fq^m)
        for (i, &r) in self.me.r_point.iter().enumerate() {
            let r_val = <E as Engine>::Scalar::from(r.as_canonical_u64());
            let r_alloc = AllocatedNum::alloc(cs.namespace(|| format!("r_point_{}", i)), || Ok(r_val))?;
            let _ = r_alloc.inputize(cs.namespace(|| format!("public_r_{}", i)));
        }
        
        // 4) base dimension b
        let b_val = <E as Engine>::Scalar::from(self.me.base_b as u64);
        let b_alloc = AllocatedNum::alloc(cs.namespace(|| "base_b"), || Ok(b_val))?;
        let _ = b_alloc.inputize(cs.namespace(|| "public_base_b"));
        
        // 5) Pad to power-of-2 BEFORE digest - CRITICAL: match encode_bridge_io_header() order
        let current_count_without_digest = self.me.c_coords.len() + self.me.y_outputs.len() + 
                                           self.me.r_point.len() + 1;
        let digest_scalars = self.digest_to_scalars();
        let total_needed = current_count_without_digest + digest_scalars.len();
        let target_count = total_needed.next_power_of_two();
        let padding_needed = target_count - total_needed;
        
        // Add padding zeros BEFORE digest to match encode_bridge_io_header() order
        for i in 0..padding_needed {
            let zero_alloc = AllocatedNum::alloc(cs.namespace(|| format!("padding_{}", i)), || Ok(<E as Engine>::Scalar::ZERO))?;
            let _ = zero_alloc.inputize(cs.namespace(|| format!("public_padding_{}", i)));
        }
        
        // 6) fold digest limbs LAST (after padding)
        for (i, &digest_val) in digest_scalars.iter().enumerate() {
            let digest_alloc = AllocatedNum::alloc(cs.namespace(|| format!("digest_{}", i)), || Ok(digest_val))?;
            let _ = digest_alloc.inputize(cs.namespace(|| format!("public_digest_{}", i)));
        }
        
        if padding_needed > 0 {
            eprintln!("🔍 inputize() padded from {} to {} public inputs (matches encode_bridge_io_header)", total_needed, target_count);
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
    
    // Assert PCS width is correct for Hash-MLE (binary hypercube)
    let pcs_width = <E as Engine>::PCS::width();
    eprintln!("🔍 ENGINE PCS WIDTH = {}", pcs_width);
    assert_eq!(pcs_width, 2, "Hash-MLE PCS must report width=2 (binary hypercube arity), got {}", pcs_width);
    
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
    // eprintln!("🔍 About to call R1CSSNARK::<E>::prove()");
    // eprintln!("🔧 Hash-MLE debugging: About to call prove with:");
    
    // Get dimensions before the move for error reporting
    let public_values_len = circuit.public_values().unwrap().len();
    let witness_len = circuit.wit.z_digits.len();
    
    // NOTE: The massive "TABLE STRUCTURE DEBUG" logs with W_full arrays 
    // are coming from the Spartan2 library itself, not this code.
    // eprintln!("   📊 Pre-prove dimensions:");
    // eprintln!("     - Public values: {}", public_values_len);
    // eprintln!("     - Original witness len: {}", witness_len);
    // eprintln!("     - Shape summary: shared=0, precommitted=0, rest=8 (from logs)");
    // eprintln!("   🎯 All vectors should be power-of-2 for Hash-MLE");
    
    let snark_proof = match R1CSSNARK::<E>::prove(&pk, circuit, &prep_snark, false) {
        Ok(proof) => {
            // eprintln!("✅ R1CSSNARK::<E>::prove() completed successfully!");
            // eprintln!("🎉 Hash-MLE commitment structure worked correctly!");
            proof
        }
        Err(e) => {
            eprintln!("❌ R1CSSNARK::<E>::prove() failed with: {:?}", e);
            
            // Detailed error analysis
            let err_str = format!("{:?}", e);
            if err_str.contains("combine_blinds") {
                eprintln!("🔍 HASH-MLE COMMITMENT ERROR ANALYSIS:");
                eprintln!("   This is a 'combine_blinds expects exactly one' error");
                eprintln!("   Likely causes:");
                eprintln!("   1. Hash-MLE expecting single commitment but getting multiple");
                eprintln!("   2. Mismatch between number of polynomial segments and blind factors");
                eprintln!("   3. Issue with how precommitted/rest segments are handled in commitments");
                eprintln!("   🎯 Current structure: precommitted=8, rest=8 (both power-of-2)");
                eprintln!("   💡 Suggestion: Hash-MLE might expect single unified commitment");
            } else if err_str.contains("power of two") {
                eprintln!("🔍 POWER-OF-2 ERROR (unexpected - should be fixed):");
                eprintln!("   We thought this was fixed with precommitted=8, rest=8");
                eprintln!("   There might be another vector we're missing");
            } else {
                eprintln!("🔍 OTHER HASH-MLE ERROR:");
                eprintln!("   Unknown Hash-MLE error type: {}", err_str);
            }
            
            eprintln!("   📊 Context:");
            eprintln!("     - Public values: {} (padded to power of 2)", public_values_len);
            eprintln!("     - Original witness: {}", witness_len);
            eprintln!("     - prep_prove() succeeded, so shape constraints are OK");
            eprintln!("     - Error is in final proof generation step");
            
            return Err(e);
        }
    };
    
    // 4. Do NOT self-verify here.
    // Some tests intentionally create inconsistent instances/witnesses to check tamper detection
    // via public IO changes. They expect proving to succeed and verification to fail externally.
    // Instead, deterministically reconstruct the circuit's public values for binding/debugging.
    let public_outputs = {
        let circuit_for_publics = MeCircuit::new(me.clone(), wit.clone(), None, me.header_digest);
        circuit_for_publics
            .public_values()
            .map_err(|e| SpartanError::SynthesisError { reason: format!("public_values() failed: {e}") })?
    };
    
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