//! In-circuit view of the product commitment `L+(Z) = (c, adv)`.
//!
//! This module owns only commitment-coordinate allocation and the two
//! linear relations shared by the SuperNeo reductions: exact equality in
//! Pi_CCS and radix recomposition in Pi_DEC. Transcript projection checks
//! remain owned by Pi_RLC.

use neo_ajtai::Commitment;
use neo_ccs::LaneCommitments;
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

use crate::engine::r1cs_circuit::{Lc, R1csBuilder, Var};

/// Wires for one Ajtai commitment, including constrained shape metadata.
#[derive(Clone, Debug)]
pub struct CommitmentWires {
    pub d: usize,
    pub d_var: Var,
    pub kappa: usize,
    pub kappa_var: Var,
    pub data: Vec<Var>,
}

/// The three Nebula commitment coordinates of the product commitment.
pub type AdvCommitmentWires = LaneCommitments<CommitmentWires>;

pub fn alloc_commitment(builder: &mut R1csBuilder, commitment: &Commitment) -> CommitmentWires {
    CommitmentWires {
        d: commitment.d,
        d_var: alloc_usize(builder, commitment.d),
        kappa: commitment.kappa,
        kappa_var: alloc_usize(builder, commitment.kappa),
        data: builder.alloc_vec(&commitment.data),
    }
}

pub fn alloc_adv(builder: &mut R1csBuilder, adv: Option<&LaneCommitments<Commitment>>) -> Option<AdvCommitmentWires> {
    adv.map(|adv| LaneCommitments {
        ops: alloc_commitment(builder, &adv.ops),
        is: alloc_commitment(builder, &adv.is),
        fs: alloc_commitment(builder, &adv.fs),
    })
}

/// Validate allocated coordinates against the main commitment's shape.
pub fn validate_adv_shape(
    adv: Option<&AdvCommitmentWires>,
    expected_d: usize,
    expected_kappa: usize,
    label: &str,
) -> Result<(), String> {
    let Some(adv) = adv else {
        return Ok(());
    };
    for (lane, commitment) in [("ops", &adv.ops), ("is", &adv.is), ("fs", &adv.fs)] {
        if commitment.d != expected_d || commitment.kappa != expected_kappa {
            return Err(format!(
                "{label}.adv.{lane}: commitment shape differs from main coordinate"
            ));
        }
        let expected_len = expected_d * expected_kappa;
        if commitment.data.len() != expected_len {
            return Err(format!(
                "{label}.adv.{lane}.data.len ({}) != d*kappa ({expected_len})",
                commitment.data.len()
            ));
        }
    }
    Ok(())
}

/// Enforce exact equality of the Nebula coordinates of two product commitments.
pub fn enforce_adv_equality(
    builder: &mut R1csBuilder,
    lhs: Option<&AdvCommitmentWires>,
    rhs: Option<&AdvCommitmentWires>,
    label: &str,
) -> Result<(), String> {
    match (lhs, rhs) {
        (None, None) => Ok(()),
        (Some(_), None) | (None, Some(_)) => Err(format!(
            "{label}: product-commitment adv presence differs between equal claims"
        )),
        (Some(lhs), Some(rhs)) => {
            for (lane, lhs, rhs) in component_pairs(lhs, rhs) {
                enforce_commitment_equality(builder, lhs, rhs, &format!("{label}.{lane}"))?;
            }
            Ok(())
        }
    }
}

/// Enforce `parent.adv = sum_i b^i child_i.adv` component-wise.
pub fn enforce_adv_recomposition(
    builder: &mut R1csBuilder,
    parent: Option<&AdvCommitmentWires>,
    children: &[Option<AdvCommitmentWires>],
    b_pows: &[F],
) -> Result<(), String> {
    if children.len() != b_pows.len() {
        return Err(format!(
            "adv recomposition: {} children but {} radix powers",
            children.len(),
            b_pows.len()
        ));
    }
    match parent {
        None => {
            if children.iter().any(Option::is_some) {
                return Err("adv recomposition: children carry adv but parent does not".into());
            }
            Ok(())
        }
        Some(parent) => {
            if children.iter().any(Option::is_none) {
                return Err("adv recomposition: parent carries adv but a child does not".into());
            }
            let children: Vec<&AdvCommitmentWires> = children.iter().map(|adv| adv.as_ref().unwrap()).collect();
            for lane in 0..3 {
                let parent_component = component(parent, lane);
                let child_components: Vec<&CommitmentWires> = children.iter().map(|adv| component(adv, lane)).collect();
                enforce_commitment_recomposition(builder, parent_component, &child_components, b_pows)?;
            }
            Ok(())
        }
    }
}

fn enforce_commitment_equality(
    builder: &mut R1csBuilder,
    lhs: &CommitmentWires,
    rhs: &CommitmentWires,
    label: &str,
) -> Result<(), String> {
    if lhs.d != rhs.d || lhs.kappa != rhs.kappa || lhs.data.len() != rhs.data.len() {
        return Err(format!("{label}: commitment shapes differ"));
    }
    enforce_var_eq(builder, lhs.d_var, rhs.d_var);
    enforce_var_eq(builder, lhs.kappa_var, rhs.kappa_var);
    for (&lhs, &rhs) in lhs.data.iter().zip(&rhs.data) {
        enforce_var_eq(builder, lhs, rhs);
    }
    Ok(())
}

fn enforce_commitment_recomposition(
    builder: &mut R1csBuilder,
    parent: &CommitmentWires,
    children: &[&CommitmentWires],
    b_pows: &[F],
) -> Result<(), String> {
    for child in children {
        if child.d != parent.d || child.kappa != parent.kappa || child.data.len() != parent.data.len() {
            return Err("adv recomposition: commitment shapes differ".into());
        }
        enforce_var_eq(builder, parent.d_var, child.d_var);
        enforce_var_eq(builder, parent.kappa_var, child.kappa_var);
    }
    for lane in 0..parent.data.len() {
        let mut combination = Lc::zero();
        for (child, coefficient) in children.iter().zip(b_pows) {
            combination.add_term(child.data[lane], *coefficient);
        }
        builder.enforce_eq(&Lc::from_var(parent.data[lane]), &combination);
    }
    Ok(())
}

fn component(adv: &AdvCommitmentWires, lane: usize) -> &CommitmentWires {
    match lane {
        0 => &adv.ops,
        1 => &adv.is,
        2 => &adv.fs,
        _ => unreachable!("Nebula product commitment has exactly three coordinates"),
    }
}

fn component_pairs<'a>(
    lhs: &'a AdvCommitmentWires,
    rhs: &'a AdvCommitmentWires,
) -> [(&'static str, &'a CommitmentWires, &'a CommitmentWires); 3] {
    [
        ("ops", &lhs.ops, &rhs.ops),
        ("is", &lhs.is, &rhs.is),
        ("fs", &lhs.fs, &rhs.fs),
    ]
}

fn alloc_usize(builder: &mut R1csBuilder, value: usize) -> Var {
    let field = F::from_u64(value as u64);
    let var = builder.alloc(field);
    builder.enforce_eq(&Lc::from_var(var), &Lc::from_const(field));
    var
}

fn enforce_var_eq(builder: &mut R1csBuilder, lhs: Var, rhs: Var) {
    builder.enforce_eq(&Lc::from_var(lhs), &Lc::from_var(rhs));
}
