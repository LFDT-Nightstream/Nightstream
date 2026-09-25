//! Π_RLC fold-input views.
//!
//! **Owns:** zero-allocation views of Π_CCS outputs and the Π_DEC parent.
//! **Does not own:** transcript sampling, projection advice, or arithmetic
//! identity implementations. **Emits constraints:** no; this phase only
//! constructs typed views. **Authority boundary:** Π_CCS fixes the input
//! shapes; the checked Π_DEC parent supplies the combined-output wires.
//!
//! | Stage child | Local operation | Semantic class |
//! | --- | --- | --- |
//! | `fold_wires.{commitment,x,y_ring}` | Build typed views without allocating wires | paper-public packed carrier |
//! | `fold_wires.adv` | Build Nebula commitment-extension view | protocol extension |

use neo_ccs::LaneCommitments;
use neo_math::ring::D;

use crate::engine::r1cs_circuit::field_ext::KVar;
use crate::engine::r1cs_circuit::{R1csBuilder, Var};
use crate::paper::reductions::pi_ccs_circuit::PiCcsOutputWires;
use crate::paper::reductions::pi_dec_circuit::DecInputWires;
use crate::paper::reductions::pi_rlc_circuit::{
    stage, RlcCommitmentWires, RlcPaddedKVectorPairWires, RlcPaddedKVectorWires, RlcPairWires, RlcXPairWires, RlcXWires,
};
use crate::paper::relations::product_commitment_circuit::{validate_adv_shape, AdvCommitmentWires, CommitmentWires};

use super::super::Error;

pub(super) struct FoldWires {
    pub(super) commitment: RlcCommitmentWires,
    pub(super) adv: Option<LaneCommitments<RlcCommitmentWires>>,
    pub(super) x: RlcXWires,
    pub(super) y_ring: Vec<RlcPaddedKVectorWires>,
}

pub(super) fn prepare(
    builder: &mut R1csBuilder,
    rho_wires: &[[Var; D]],
    outputs: &[PiCcsOutputWires],
    dec_wires: &DecInputWires,
    kappa: usize,
    m_in: usize,
    t: usize,
    d_pad: usize,
) -> Result<FoldWires, Error> {
    builder.begin_encoding_stage(stage::FOLD_WIRES);
    builder.begin_encoding_stage(stage::FOLD_WIRES_COMMITMENT);
    let commitment = commitment_wires(rho_wires, outputs, dec_wires, kappa)?;

    builder.begin_encoding_stage(stage::FOLD_WIRES_ADV);
    let adv = adv_commitment_wires(rho_wires, outputs, dec_wires)?;

    builder.begin_encoding_stage(stage::FOLD_WIRES_X);
    let x = x_wires(rho_wires, outputs, dec_wires, m_in);

    builder.begin_encoding_stage(stage::FOLD_WIRES_Y_RING);
    let mut y_ring = Vec::with_capacity(t);
    for row in 0..t {
        y_ring.push(y_ring_row_wires(rho_wires, outputs, dec_wires, row, d_pad)?);
    }

    Ok(FoldWires {
        commitment,
        adv,
        x,
        y_ring,
    })
}

fn commitment_wires(
    rho_wires: &[[Var; D]],
    outputs: &[PiCcsOutputWires],
    dec_wires: &DecInputWires,
    kappa: usize,
) -> Result<RlcCommitmentWires, Error> {
    if rho_wires.len() != outputs.len() {
        return Err(Error::Inner(format!(
            "ρ count {} != outputs count {}",
            rho_wires.len(),
            outputs.len()
        )));
    }
    let inputs = rho_wires
        .iter()
        .zip(outputs)
        .map(|(rho, output)| RlcPairWires {
            rho_coeffs: *rho,
            c_data: output.c_data.clone(),
            kappa: output.c_kappa,
        })
        .collect();
    Ok(RlcCommitmentWires {
        inputs,
        // Π_CCS fixes κ. A wider parent is rejected by shape rows before this
        // authoritative prefix is viewed by the projection identities.
        combined_c_data: dec_wires.parent.c_data[..D * kappa].to_vec(),
        kappa,
    })
}

fn adv_commitment_wires(
    rho_wires: &[[Var; D]],
    outputs: &[PiCcsOutputWires],
    dec_wires: &DecInputWires,
) -> Result<Option<LaneCommitments<RlcCommitmentWires>>, Error> {
    let present = outputs.iter().filter(|output| output.adv.is_some()).count();
    match (dec_wires.parent.adv.as_ref(), present) {
        (None, 0) => Ok(None),
        (Some(_), 0) | (None, _) => Err(Error::Inner(
            "Pi_RLC product-commitment adv presence differs between inputs and parent".into(),
        )),
        (Some(parent), count) if count == outputs.len() => {
            validate_adv_shape(Some(parent), parent.ops.d, parent.ops.kappa, "Pi_RLC parent").map_err(Error::Inner)?;
            let output_advs = outputs
                .iter()
                .map(|output| output.adv.as_ref().expect("presence counted above"))
                .collect::<Vec<_>>();
            for (index, adv) in output_advs.iter().enumerate() {
                validate_adv_shape(
                    Some(adv),
                    parent.ops.d,
                    parent.ops.kappa,
                    &format!("Pi_RLC output[{index}]"),
                )
                .map_err(Error::Inner)?;
            }
            let coordinate = |select: fn(&AdvCommitmentWires) -> &CommitmentWires,
                              combined: &CommitmentWires|
             -> RlcCommitmentWires {
                let inputs = rho_wires
                    .iter()
                    .zip(&output_advs)
                    .map(|(rho, adv)| {
                        let commitment = select(adv);
                        RlcPairWires {
                            rho_coeffs: *rho,
                            c_data: commitment.data.clone(),
                            kappa: commitment.kappa,
                        }
                    })
                    .collect();
                RlcCommitmentWires {
                    inputs,
                    combined_c_data: combined.data.clone(),
                    kappa: combined.kappa,
                }
            };
            Ok(Some(LaneCommitments {
                ops: coordinate(|adv| &adv.ops, &parent.ops),
                is: coordinate(|adv| &adv.is, &parent.is),
                fs: coordinate(|adv| &adv.fs, &parent.fs),
            }))
        }
        (Some(_), count) => Err(Error::Inner(format!(
            "Pi_RLC product-commitment adv presence is mixed ({count}/{})",
            outputs.len()
        ))),
    }
}

fn x_wires(rho_wires: &[[Var; D]], outputs: &[PiCcsOutputWires], dec_wires: &DecInputWires, m_in: usize) -> RlcXWires {
    let x_cols = crate::paper::relations::superneo_public_x_cols(m_in);
    let inputs = rho_wires
        .iter()
        .zip(outputs)
        .map(|(rho, output)| RlcXPairWires {
            rho_coeffs: *rho,
            x_flat: output.x.clone(),
            x_cols: output.x_cols,
        })
        .collect();
    RlcXWires {
        inputs,
        // Π_CCS fixes m_in. Shape rows reject a wider parent before this
        // authoritative prefix reaches projection enforcement.
        combined_x_flat: dec_wires.parent.x[..D * x_cols].to_vec(),
        x_cols,
    }
}

fn y_ring_row_wires(
    rho_wires: &[[Var; D]],
    outputs: &[PiCcsOutputWires],
    dec_wires: &DecInputWires,
    row: usize,
    d_pad: usize,
) -> Result<RlcPaddedKVectorWires, Error> {
    let inputs = outputs
        .iter()
        .map(|output| output.y_ring[row].clone())
        .collect::<Vec<_>>();
    let combined = kvars_from_flat_dec(&dec_wires.parent.y_ring[row])?;
    padded_k_vector_wires(rho_wires, &inputs, &combined, d_pad)
}

fn kvars_from_flat_dec(flat: &[Var]) -> Result<Vec<KVar>, Error> {
    if flat.len() % 2 != 0 {
        return Err(Error::Inner(format!(
            "DEC flat K-vector has odd limb count {}",
            flat.len()
        )));
    }
    Ok(flat
        .chunks_exact(2)
        .map(|limbs| KVar {
            c0: limbs[0],
            c1: limbs[1],
        })
        .collect())
}

fn padded_k_vector_wires(
    rhos: &[[Var; D]],
    inputs: &[Vec<KVar>],
    combined: &[KVar],
    d_pad: usize,
) -> Result<RlcPaddedKVectorWires, Error> {
    if combined.len() != d_pad {
        return Err(Error::Inner(format!(
            "padded K-vector: combined.len ({}) != d_pad ({d_pad})",
            combined.len()
        )));
    }
    if rhos.len() != inputs.len() {
        return Err(Error::Inner(format!(
            "padded K-vector: ρ count {} != input count {}",
            rhos.len(),
            inputs.len()
        )));
    }
    let mut pairs = Vec::with_capacity(inputs.len());
    for (index, (rho, y)) in rhos.iter().zip(inputs).enumerate() {
        if y.len() != d_pad {
            return Err(Error::Inner(format!(
                "padded K-vector: inputs[{index}].len ({}) != d_pad ({d_pad})",
                y.len()
            )));
        }
        pairs.push(RlcPaddedKVectorPairWires {
            rho_coeffs: *rho,
            y_c0: y.iter().map(|value| value.c0).collect(),
            y_c1: y.iter().map(|value| value.c1).collect(),
        });
    }
    Ok(RlcPaddedKVectorWires {
        inputs: pairs,
        combined_c0: combined.iter().map(|value| value.c0).collect(),
        combined_c1: combined.iter().map(|value| value.c1).collect(),
        d_pad,
    })
}
