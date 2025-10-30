use neo_fold::{pi_ccs_prove, pi_ccs_verify};
use neo_transcript::{Poseidon2Transcript, Transcript};
use neo_ccs::{r1cs_to_ccs, Mat, McsInstance, McsWitness, MeInstance, SModuleHomomorphism};
use neo_math::{F, K, D};
use neo_params::NeoParams;
use neo_ajtai::Commitment;
use p3_field::{PrimeCharacteristicRing, PrimeField64, PackedValue};
use serde::{Serialize, Deserialize};
use std::fs::File;
use std::io::Write;

struct DummyS;

impl SModuleHomomorphism<F, Commitment> for DummyS {
    fn commit(&self, z: &Mat<F>) -> Commitment {
        let d = z.rows();
        Commitment::zeros(d, 4)
    }
    
    fn project_x(&self, z: &Mat<F>, m_in: usize) -> Mat<F> {
        let rows = z.rows();
        let cols = m_in.min(z.cols());
        let mut result = Mat::zero(rows, cols, F::ZERO);
        for r in 0..rows {
            for c in 0..cols {
                result[(r, c)] = z[(r, c)];
            }
        }
        result
    }
}

fn create_test_ccs() -> neo_ccs::CcsStructure<F> {
    let rows: usize = 4;
    let cols: usize = 5;
    
    let mut a = vec![F::ZERO; rows * cols];
    let mut b = vec![F::ZERO; rows * cols];
    let c = vec![F::ZERO; rows * cols];
    
    a[0 * cols + 4] = F::ONE;
    a[0 * cols + 1] = -F::ONE;
    a[0 * cols + 2] = -F::ONE;
    b[0 * cols + 0] = F::ONE;
    
    for row in 1..rows {
        b[row * cols + 0] = F::ONE;
    }
    
    r1cs_to_ccs(
        Mat::from_row_major(rows, cols, a),
        Mat::from_row_major(rows, cols, b),
        Mat::from_row_major(rows, cols, c),
    )
}

#[allow(non_snake_case)]
fn create_mcs_from_witness(
    params: &NeoParams,
    z_full: Vec<F>,
    m_in: usize,
    l: &DummyS,
) -> (McsInstance<Commitment, F>, McsWitness<F>) {
    let d = D;
    let m = z_full.len();
    
    let z_digits = neo_ajtai::decomp_b(&z_full, params.b, d, neo_ajtai::DecompStyle::Balanced);
    
    let mut row_major = vec![F::ZERO; d * m];
    for col in 0..m {
        for row in 0..d {
            row_major[row * m + col] = z_digits[col * d + row];
        }
    }
    let Z = Mat::from_row_major(d, m, row_major);
    
    let c_commit = l.commit(&Z);
    
    let mcs_instance = McsInstance {
        c: c_commit.clone(),
        x: z_full[..m_in].to_vec(),
        m_in,
    };
    
    let mcs_witness = McsWitness {
        w: z_full[m_in..].to_vec(),
        Z,
    };
    
    (mcs_instance, mcs_witness)
}

#[allow(non_snake_case)]
fn create_me_from_mcs(
    params: &NeoParams,
    ccs: &neo_ccs::CcsStructure<F>,
    mcs_instance: McsInstance<Commitment, F>,
    mcs_witness: McsWitness<F>,
    l: &DummyS,
) -> (MeInstance<Commitment, F, K>, Mat<F>) {
    let mut tr = Poseidon2Transcript::new(b"test/fixture/k1");
    let prove_result = neo_fold::pi_ccs_prove_simple(
        &mut tr,
        params,
        ccs,
        &[mcs_instance],
        &[mcs_witness.clone()],
        l,
    );
    
    assert!(prove_result.is_ok(), "k=1 fold for ME fixture generation failed: {:?}", prove_result.err());
    let (me_outputs, _proof) = prove_result.unwrap();
    
    assert_eq!(me_outputs.len(), 1, "k=1 should produce exactly 1 ME output");
    (me_outputs[0].clone(), mcs_witness.Z)
}

#[derive(Serialize, Deserialize)]
struct FieldInfo {
    q: String,
    name: String,
    ext_deg: u32,
}

#[derive(Serialize, Deserialize)]
struct NeoParamsJson {
    n: usize,
    kappa: usize,
    b: u32,
    d: usize,
    q: String,
}

#[derive(Serialize, Deserialize)]
struct MatrixJson {
    rows: usize,
    cols: usize,
    data: Vec<String>,
}

#[derive(Serialize, Deserialize)]
struct SparsePolyTermJson {
    monomial: Vec<usize>,
    coeff: String,
}

#[derive(Serialize, Deserialize)]
struct CcsStructureJson {
    matrices: Vec<MatrixJson>,
    f: Vec<SparsePolyTermJson>,
    n: usize,
    m: usize,
}

#[derive(Serialize, Deserialize)]
struct McsInstanceJson {
    c: MatrixJson,
    x: Vec<String>,
    m_in: usize,
}

#[derive(Serialize, Deserialize)]
struct McsWitnessJson {
    w: Vec<String>,
    #[serde(rename = "Z")]
    z: MatrixJson,
}

#[derive(Serialize, Deserialize)]
struct MeInstanceJson {
    c: MatrixJson,
    #[serde(rename = "X")]
    x_mat: MatrixJson,
    r: Vec<String>,
    y: Vec<Vec<String>>,
    y_at_r: Vec<String>,
    m_in: usize,
}

#[derive(Serialize, Deserialize)]
struct PiCcsInputJson {
    field: FieldInfo,
    params: NeoParamsJson,
    ccs: CcsStructureJson,
    mcs_instances: Vec<McsInstanceJson>,
    mcs_witnesses: Vec<McsWitnessJson>,
    me_instances: Vec<MeInstanceJson>,
    me_witnesses: Vec<MatrixJson>,
}

fn field_to_string(f: F) -> String {
    format!("{}", f.as_canonical_u64())
}

fn field_k_to_string(k: K) -> String {
    let base_slice = k.as_slice();
    if base_slice.len() == 1 {
        format!("{}", base_slice[0])
    } else {
        format!("{}+{}i", base_slice[0], base_slice[1])
    }
}

fn mat_to_json<T: Copy>(mat: &Mat<T>, conv: fn(T) -> String) -> MatrixJson {
    let rows = mat.rows();
    let cols = mat.cols();
    let mut data = Vec::new();
    for r in 0..rows {
        for c in 0..cols {
            data.push(conv(mat[(r, c)]));
        }
    }
    MatrixJson { rows, cols, data }
}

fn commitment_to_json(c: &Commitment) -> MatrixJson {
    let rows = c.d;
    let cols = c.kappa;
    let mut data = Vec::new();
    for r in 0..rows {
        for c_idx in 0..cols {
            data.push(field_to_string(c.data[c_idx * c.d + r]));
        }
    }
    MatrixJson { rows, cols, data }
}

fn ccs_to_json(ccs: &neo_ccs::CcsStructure<F>) -> CcsStructureJson {
    let matrices = ccs.matrices.iter().map(|m| mat_to_json(m, field_to_string)).collect();
    
    let mut f = Vec::new();
    for term in ccs.f.terms() {
        f.push(SparsePolyTermJson {
            monomial: term.exps.iter().map(|&x| x as usize).collect(),
            coeff: field_to_string(term.coeff),
        });
    }
    
    CcsStructureJson {
        matrices,
        f,
        n: ccs.n,
        m: ccs.m,
    }
}

fn mcs_instance_to_json(inst: &McsInstance<Commitment, F>) -> McsInstanceJson {
    McsInstanceJson {
        c: commitment_to_json(&inst.c),
        x: inst.x.iter().map(|f| field_to_string(*f)).collect(),
        m_in: inst.m_in,
    }
}

fn mcs_witness_to_json(wit: &McsWitness<F>) -> McsWitnessJson {
    McsWitnessJson {
        w: wit.w.iter().map(|f| field_to_string(*f)).collect(),
        z: mat_to_json(&wit.Z, field_to_string),
    }
}

fn me_instance_to_json(inst: &MeInstance<Commitment, F, K>) -> MeInstanceJson {
    MeInstanceJson {
        c: commitment_to_json(&inst.c),
        x_mat: mat_to_json(&inst.X, field_to_string),
        r: inst.r.iter().map(|k| field_k_to_string(*k)).collect(),
        y: inst.y.iter().map(|y_j| y_j.iter().map(|k| field_k_to_string(*k)).collect()).collect(),
        y_at_r: inst.y_scalars.iter().map(|k| field_k_to_string(*k)).collect(),
        m_in: inst.m_in,
    }
}

#[test]
#[ignore]
fn export_pi_ccs_k2_to_json() {
    let params = NeoParams::goldilocks_for_circuit(3, 2, 2);
    let ccs = create_test_ccs();
    let l = DummyS;
    
    let z1_full = vec![F::ONE, F::from_u64(2), F::from_u64(3), F::ZERO, F::from_u64(5)];
    let m_in = 1;
    let (mcs_inst1, mcs_wit1) = create_mcs_from_witness(&params, z1_full, m_in, &l);
    
    let z2_full = vec![F::ONE, F::from_u64(7), F::from_u64(11), F::ZERO, F::from_u64(18)];
    let (mcs_inst2, mcs_wit2) = create_mcs_from_witness(&params, z2_full, m_in, &l);
    let (me_input, me_witness_z) = create_me_from_mcs(&params, &ccs, mcs_inst2, mcs_wit2, &l);
    
    let json_data = PiCcsInputJson {
        field: FieldInfo {
            q: "18446744069414584321".to_string(),
            name: "Goldilocks".to_string(),
            ext_deg: 1,
        },
        params: NeoParamsJson {
            n: params.eta as usize,
            kappa: params.kappa as usize,
            b: params.b,
            d: params.d as usize,
            q: "18446744069414584321".to_string(),
        },
        ccs: ccs_to_json(&ccs),
        mcs_instances: vec![mcs_instance_to_json(&mcs_inst1)],
        mcs_witnesses: vec![mcs_witness_to_json(&mcs_wit1)],
        me_instances: vec![me_instance_to_json(&me_input)],
        me_witnesses: vec![mat_to_json(&me_witness_z, field_to_string)],
    };
    
    let output_path = "pi_ccs_k2_input.json";
    let mut file = File::create(output_path).expect("Failed to create JSON file");
    let json_string = serde_json::to_string_pretty(&json_data).expect("Failed to serialize JSON");
    file.write_all(json_string.as_bytes()).expect("Failed to write JSON");
    
    println!("Exported π-CCS k=2 input to {}", output_path);
    println!("  - 1 MCS instance (2 + 3 = 5)");
    println!("  - 1 ME instance (7 + 11 = 18, pre-folded)");
    println!("\nTo run Sage prover:");
    println!("  cd new-neo-sage");
    println!("  sage pi_ccs_prover.sage ../pi_ccs_k2_input.json");
}

#[test]
fn pi_ccs_k2_honest_fold() {
    let params = NeoParams::goldilocks_for_circuit(3, 2, 2);
    let ccs = create_test_ccs();
    let l = DummyS;
    
    let z1_full = vec![F::ONE, F::from_u64(2), F::from_u64(3), F::ZERO, F::from_u64(5)];
    let m_in = 1;
    let (mcs_inst1, mcs_wit1) = create_mcs_from_witness(&params, z1_full, m_in, &l);
    
    let z2_full = vec![F::ONE, F::from_u64(7), F::from_u64(11), F::ZERO, F::from_u64(18)];
    let (mcs_inst2, mcs_wit2) = create_mcs_from_witness(&params, z2_full, m_in, &l);
    let (me_input, me_witness_z) = create_me_from_mcs(&params, &ccs, mcs_inst2, mcs_wit2, &l);
    
    let mut tr_p = Poseidon2Transcript::new(b"test/pi-ccs/k2");
    let prove_result = pi_ccs_prove(
        &mut tr_p,
        &params,
        &ccs,
        &[mcs_inst1.clone()],
        &[mcs_wit1],
        &[me_input.clone()],
        &[me_witness_z],
        &l,
    );
    
    assert!(prove_result.is_ok(), "k=2 proving should succeed for valid witnesses: {:?}", prove_result.err());
    let (me_outputs, proof) = prove_result.unwrap();
    
    assert_eq!(me_outputs.len(), 2, "k=2 should produce exactly 2 ME outputs");
    
    let mut tr_v = Poseidon2Transcript::new(b"test/pi-ccs/k2");
    let verify_result = pi_ccs_verify(
        &mut tr_v,
        &params,
        &ccs,
        &[mcs_inst1],
        &[me_input],
        &me_outputs,
        &proof,
    );
    
    assert!(verify_result.is_ok(), "Verification should not error");
    let is_valid = verify_result.unwrap();
    assert!(is_valid, "Valid k=2 fold (1 MCS + 1 ME → 2 ME) should verify");
}

