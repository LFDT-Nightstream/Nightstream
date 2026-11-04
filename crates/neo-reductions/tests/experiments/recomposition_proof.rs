//! Exact complete PiCCS comparison with a factored linear-map backend.
use super::*;
use crate::optimized_engine::{
    optimized_prove_with_matrix_rows, optimized_verify_with_trace, PaperJointOracleBackend, PaperJointOracleInput,
};
use neo_ajtai::Commitment;
use neo_ccs::{CcsClaim, CeClaim};
use neo_transcript::Poseidon2Transcript;

struct Backend<'a>(&'a Plan);
impl PaperJointOracleBackend for Backend<'_> {
    fn create<'a>(
        &'a mut self,
        input: PaperJointOracleInput<'a>,
    ) -> Result<Box<dyn PaperJointRoundOracle + 'a>, PiCcsError> {
        assert_eq!(input.fresh_witnesses.len(), 1);
        let witnesses: Vec<_> = input
            .fresh_witnesses
            .iter()
            .map(|w| &w.Z)
            .chain(input.running_witnesses)
            .collect();
        let witness_blocks: Vec<_> = witnesses
            .iter()
            .map(|w| SuperneoZBlocks::from_witness_mat(w, input.structure.m()))
            .collect::<Result<_, _>>()?;
        let assignments = witnesses
            .iter()
            .map(|w| prefix::Assignment::new(w, input.dims.assignment_width))
            .collect();
        let mut fresh_tables = ApplicationTables::new(input.rows, &witness_blocks[..1], input.workspace_bytes)?;
        fresh_tables.resident = Some(Values {
            words: self.0.fresh(&input.fresh_witnesses[0].Z),
            rows: self.0.rows,
        });
        let running = &witness_blocks[1..];
        let gamma = input.challenges.gamma;
        let width = input.dims.assignment_width;
        let evaluation_table = if running.iter().all(SuperneoZBlocks::is_zero) {
            Vec::new()
        } else {
            let powers: Vec<_> = (0..running.len()).map(|i| gamma_power(gamma, i)).collect();
            let weights = std::array::from_fn(|i| gamma_power(gamma, running.len() * 4 * i));
            let coefficients: Vec<_> = (0..4)
                .map(|i| gamma_power(gamma, running.len() * i))
                .collect();
            let mut result = vec![K::ZERO; width.max(self.0.rows)];
            fill_combined_projection(running, &powers, &weights, &mut result[..width]);
            let matrix = self.0.weighted(&result[..width], &coefficients);
            let weights = std::array::from_fn(|i| gamma_power(gamma, running.len() * i));
            fill_combined_projection(running, &powers, &weights, &mut result[..width]);
            let shift = gamma_power(gamma, running.len() * D);
            for (out, value) in result.iter_mut().zip(matrix) {
                *out += shift * value;
            }
            while result.last() == Some(&K::ZERO) {
                result.pop();
            }
            result
        };
        Ok(Box::new(OptimizedPaperJointOracle {
            structure: input.structure,
            source: input.rows,
            workspace_bytes: input.workspace_bytes,
            base: input.params.b(),
            challenges: input.challenges.clone(),
            dims: input.dims,
            point: Vec::new(),
            fixed_equality: K::ONE,
            prior_point: input.prior_point.map(<[K]>::to_vec),
            fixed_prior_equality: K::ONE,
            fresh_tables,
            assignments,
            witness_blocks,
            evaluation_table,
            constraint_shift: gamma_power(gamma, input.running_witnesses.len() * D * (input.structure.t() + 1)),
        }))
    }
}
#[test]
#[ignore = "complete actual-lifecycle PiCCS comparisons; use run.py 300-second cap"]
fn recomposition_proofs() {
    let mut request = String::new();
    std::io::stdin().read_to_string(&mut request).unwrap();
    let request: Request = serde_json::from_str(&request).unwrap();
    rayon::ThreadPoolBuilder::new().num_threads(THREADS).build().unwrap().install(|| {
        let package=nightstream_fprime::load_prepared_package(fs::File::open(request.directory.join("package.bin")).unwrap()).unwrap();
        let witnesses=read_witnesses(request.directory.join(format!("witnesses-{}.bin",request.step)));
        let (claim,running):(CcsClaim<Commitment,F>,Vec<CeClaim<Commitment,F,K>>)=serde_json::from_slice(&fs::read(request.directory.join(format!("claims-{}.json",request.step))).unwrap()).unwrap();
        let source=Rows(&package);let structure=package.ccs_structure_header().unwrap();
        let params=neo_params::NeoParams::nightstream_goldilocks_k16().with_lambda(114).unwrap();
        let (plan,setup_seconds)=seconds(||Plan::new(&package));
        let fresh=vec![CcsWitness {Z:witnesses[0].clone()}];let claims=vec![claim];
        for trial in 0..4 {
            let mut reference=None;
            for factor in if trial%2==0 {[false,true]} else {[true,false]} {
                let mut backend=Backend(&plan);
                let (result,elapsed)=seconds(||optimized_prove_with_matrix_rows(&mut Poseidon2Transcript::new_v1_2(),&params,&structure,&claims,&fresh,&running,&witnesses[1..],&source,BUDGET,if factor {Some(&mut backend)} else {None}).unwrap());
                let (accepted,_)=optimized_verify_with_trace(&mut Poseidon2Transcript::new_v1_2(),&params,&structure,&claims,&running,&result.0,&result.1).unwrap();assert!(accepted);
                let serialized=serde_json::to_vec(&(&result.0,&result.1.sumcheck_rounds)).unwrap();
                if let Some(expected)=&reference {assert_eq!(&serialized,expected);} else {reference=Some(serialized);}
                if trial!=0 {println!("EXPERIMENT {}",json!({"event":"proof","step":request.step,"trial":trial,"variant":if factor {"factor"} else {"production"},"seconds":elapsed,"setup_seconds":setup_seconds,"verified":true,"exact_equal":true}));}
            }
        }
    });
}
