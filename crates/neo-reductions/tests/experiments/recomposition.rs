//! Factor repeated geometric reconstructions without changing encoded witnesses or ring projections.
use super::super::OptimizedPaperJointOracle;
use super::*;
use crate::engines::pi_ccs_protocol::Challenges;
use crate::engines::{pi_ccs_joint::build_joint_dims, pi_ccs_joint_protocol::PaperJointRoundOracle};
use crate::superneo_eval::MatrixRowSink;
use neo_ccs::{CcsWitness, GeometricRowRun, Mat};
use neo_math::KExtensions;
use nightstream_fprime::LoadedPerApplicationPackage;
use serde_json::json;
use std::{fs, hint::black_box, ops::ControlFlow, path::PathBuf, time::Instant};

const BUDGET: usize = 1024 * 1024 * 1024;
const THREADS: usize = 4;
const REPEATS: usize = 5;

struct Rows<'a>(&'a LoadedPerApplicationPackage);
impl MatrixRows for Rows<'_> {
    fn shape(&self) -> MatrixShape {
        MatrixShape {
            rows: self.0.row_count(),
            columns: self.0.logical_column_count().div_ceil(D) * D,
            matrices: 4,
        }
    }
    fn visit_rows(&self, range: Range<usize>, sink: &mut dyn MatrixRowSink) -> Result<(), PiCcsError> {
        let mut error = None;
        let _ = self
            .0
            .visit_matrix_runs_until(range, |row, forms| {
                for (matrix, terms) in forms.into_iter().enumerate() {
                    for term in terms {
                        if let Err(e) = sink.push_run(
                            row,
                            matrix,
                            GeometricRowRun::new(
                                row,
                                term.column(),
                                term.column_count(),
                                F::from_u64(term.coefficient()),
                                F::from_u64(term.ratio()),
                            ),
                        ) {
                            error = Some(e);
                            return Ok(ControlFlow::Break(()));
                        }
                    }
                    match sink.finish_matrix_row(row, matrix) {
                        Ok(ControlFlow::Continue(())) => (),
                        Ok(ControlFlow::Break(())) => return Ok(ControlFlow::Break(())),
                        Err(e) => {
                            error = Some(e);
                            return Ok(ControlFlow::Break(()));
                        }
                    }
                }
                Ok(ControlFlow::Continue(()))
            })
            .unwrap();
        error.map_or(Ok(()), Err)
    }
}

fn coin(round: usize) -> K {
    K::from_coeffs([F::from_usize(round * 19 + 7), F::from_usize(round * 13 + 11)])
}
fn check_satisfied(values: &Values) {
    for row in 0..values.rows {
        let s = values.words[3 * values.rows + row];
        let s2 = s * s;
        let s4 = s2 * s2;
        assert_eq!(
            values.words[row] * values.words[values.rows + row] - values.words[2 * values.rows + row] + s4 * s2 * s,
            K::ZERO,
            "row {row}"
        );
    }
}

use crate::superneo_eval::fill_combined_projection;
use std::{collections::HashMap, io::Read};

#[derive(serde::Deserialize)]
struct Request {
    directory: PathBuf,
    step: usize,
    pairs: Option<usize>,
}
impl Request {
    /// Measured production/factored proof pairs per process, after one warmup pair.
    fn pairs(&self) -> usize {
        self.pairs.unwrap_or(3)
    }
}
#[derive(Clone, Copy, Hash, PartialEq, Eq)]
struct Key {
    start: usize,
    len: usize,
    ratio: u64,
}
struct Plan {
    keys: Vec<Key>,
    offsets: Vec<usize>,
    terms: Vec<(u32, F)>,
    rows: usize,
}
impl Plan {
    fn new(package: &LoadedPerApplicationPackage) -> Self {
        let mut plan = Self {
            keys: Vec::new(),
            offsets: vec![0],
            terms: Vec::new(),
            rows: package.row_count(),
        };
        let mut ids = HashMap::new();
        let _ = package
            .visit_matrix_runs_until(0..plan.rows, |_, forms| {
                for terms in forms {
                    for term in terms {
                        let key = Key {
                            start: term.column(),
                            len: term.column_count(),
                            ratio: term.ratio(),
                        };
                        let id = *ids.entry(key).or_insert_with(|| {
                            let id = plan.keys.len() as u32;
                            plan.keys.push(key);
                            id
                        });
                        plan.terms.push((id, F::from_u64(term.coefficient())));
                    }
                    plan.offsets.push(plan.terms.len());
                }
                Ok(ControlFlow::Continue(()))
            })
            .unwrap();
        plan.keys.shrink_to_fit();
        plan.offsets.shrink_to_fit();
        plan.terms.shrink_to_fit();
        plan
    }
    fn fresh(&self, witness: &Mat<F>) -> Vec<K> {
        let masks = witness.packed_signed_unit_column_masks();
        assert!(
            masks.is_some()
                || witness
                    .virtual_constant_value()
                    .is_some_and(|&v| v == F::ZERO),
            "prototype accepts only packed signed-unit or virtual-zero witnesses"
        );
        let reconstructed: Vec<F> = self
            .keys
            .par_iter()
            .map(|key| {
                let Some((positive, negative)) = masks else {
                    return F::ZERO;
                };
                let ratio = F::from_u64(key.ratio);
                // Horner reconstructs the same geometric run with no per-digit power stream.
                (key.start..key.start + key.len)
                    .rev()
                    .fold(F::ZERO, |acc, column| {
                        let bit = 1u64 << (column % D);
                        acc * ratio + F::from_u64(u64::from(positive[column / D] & bit != 0))
                            - F::from_u64(u64::from(negative[column / D] & bit != 0))
                    })
            })
            .collect();
        (0..self.rows * 4)
            .into_par_iter()
            .map(|index| {
                let row = index % self.rows;
                let matrix = index / self.rows;
                let offset = row * 4 + matrix;
                K::from(
                    self.terms[self.offsets[offset]..self.offsets[offset + 1]]
                        .iter()
                        .map(|&(id, c)| reconstructed[id as usize] * c)
                        .sum::<F>(),
                )
            })
            .collect()
    }
    fn weighted(&self, projection: &[K], coefficients: &[K]) -> Vec<K> {
        let reconstructed: Vec<K> = self
            .keys
            .par_iter()
            .map(|key| {
                let ratio = F::from_u64(key.ratio);
                projection[key.start..key.start + key.len]
                    .iter()
                    .rev()
                    .fold(K::ZERO, |acc, &value| acc.scale_base(ratio) + value)
            })
            .collect();
        (0..self.rows)
            .into_par_iter()
            .map(|row| {
                (0..4)
                    .map(|matrix| {
                        let offset = row * 4 + matrix;
                        self.terms[self.offsets[offset]..self.offsets[offset + 1]]
                            .iter()
                            .map(|&(id, c)| reconstructed[id as usize].scale_base(c))
                            .sum::<K>()
                            * coefficients[matrix]
                    })
                    .sum()
            })
            .collect()
    }
    fn bytes(&self) -> usize {
        self.keys.capacity() * size_of::<Key>()
            + self.offsets.capacity() * size_of::<usize>()
            + self.terms.capacity() * size_of::<(u32, F)>()
    }
}
fn read_witnesses(path: PathBuf) -> Vec<Mat<F>> {
    let mut input = std::io::BufReader::new(fs::File::open(path).unwrap());
    let mut magic = [0; 9];
    input.read_exact(&mut magic).unwrap();
    assert_eq!(&magic, b"NSRECOMP1");
    fn word(input: &mut impl Read) -> u64 {
        let mut bytes = [0; 8];
        input.read_exact(&mut bytes).unwrap();
        u64::from_le_bytes(bytes)
    }
    (0..word(&mut input))
        .map(|_| {
            let cols = word(&mut input) as usize;
            let mut flag = [0];
            input.read_exact(&mut flag).unwrap();
            if flag[0] == 0 {
                Mat::virtual_constant(D, cols, F::ZERO)
            } else {
                let positive: Vec<_> = (0..cols).map(|_| word(&mut input)).collect();
                let negative: Vec<_> = (0..cols).map(|_| word(&mut input)).collect();
                Mat::compact_signed_unit_from_column_masks(D, cols, &positive, &negative).unwrap()
            }
        })
        .collect()
}
fn seconds<T>(f: impl FnOnce() -> T) -> (T, f64) {
    let t = Instant::now();
    let value = f();
    (value, t.elapsed().as_secs_f64())
}

#[test]
#[ignore = "actual lifecycle linear-map experiment; use run.py's 300-second cap"]
fn recomposition_measure() {
    let mut request = String::new();
    std::io::stdin().read_to_string(&mut request).unwrap();
    let request: Request = serde_json::from_str(&request).unwrap();
    rayon::ThreadPoolBuilder::new().num_threads(THREADS).build().unwrap().install(|| {
        let package=nightstream_fprime::load_prepared_package(fs::File::open(request.directory.join("package.bin")).unwrap()).unwrap();
        let witnesses=read_witnesses(request.directory.join(format!("witnesses-{}.bin",request.step)));
        let source=Rows(&package);let structure=package.ccs_structure_header().unwrap();let rows=source.shape().rows;
        let blocks:Vec<_>=witnesses.iter().map(|w|SuperneoZBlocks::from_witness_mat(w,structure.m()).unwrap()).collect();
        let (plan,plan_seconds)=seconds(||Plan::new(&package));
        let digits:usize=plan.terms.iter().map(|&(id,_)|plan.keys[id as usize].len).sum();
        let mut lengths=std::collections::BTreeMap::new();
        for key in &plan.keys {*lengths.entry(key.len).or_insert(0usize)+=1;}
        let unique_digits:usize=plan.keys.iter().map(|key|key.len).sum();
        let (window,cache_seconds)=seconds(||MatrixWindow::load_next(&source,0..rows,BUDGET).unwrap());
        assert_eq!(window.rows(),0..rows);
        println!("EXPERIMENT {}",json!({"event":"plan","step":request.step,"rows":rows,"columns":source.shape().columns,"sources":blocks.len(),"active_sources":blocks.iter().filter(|b|!b.is_zero()).count(),"occurrences":plan.terms.len(),"unique_runs":plan.keys.len(),"unique_run_lengths":lengths,"digit_visits":digits,"unique_digit_visits":unique_digits,"plan_seconds":plan_seconds,"plan_bytes":plan.bytes(),"cache_seconds":cache_seconds,"cache_bytes":window.storage_bytes()}));
        let cache=window.cache();
        for trial in 0..=REPEATS {
            let generic=|| {let mut out=vec![K::ZERO;rows*4];out.par_chunks_mut(rows).enumerate().for_each(|(matrix,chunk)| {chunk.par_chunks_mut(ROW_DOT_CHUNK).enumerate().for_each(|(index,values)|cache.matrix_caches()[matrix].fill_row_dots_real_from(index*ROW_DOT_CHUNK,values,&blocks[0]));});out};
            let (reference,generic_seconds);let (actual,factor_seconds);
            if trial%2==0 { (reference,generic_seconds)=seconds(generic);(actual,factor_seconds)=seconds(||plan.fresh(&witnesses[0])); }
            else { (actual,factor_seconds)=seconds(||plan.fresh(&witnesses[0]));(reference,generic_seconds)=seconds(generic); }
            assert_eq!(reference,actual);check_satisfied(&Values {words:actual,rows});black_box(reference);
            if trial!=0 {println!("EXPERIMENT {}",json!({"event":"fresh","step":request.step,"trial":trial,"generic_seconds":generic_seconds,"factor_seconds":factor_seconds,"equal":true}));}
        }
        let params=neo_params::NeoParams::nightstream_goldilocks_k16().with_lambda(114).unwrap();
        let dims=build_joint_dims(&params,&structure,1,witnesses.len()-1).unwrap();
        let gamma=coin(99);let running=&blocks[1..];
        let powers:Vec<_>=(0..running.len()).map(|i|gamma_power(gamma,i)).collect();
        let weights=std::array::from_fn(|i|gamma_power(gamma,running.len()*4*i));
        let coefficients:Vec<_>=(0..4).map(|i|gamma_power(gamma,running.len()*i)).collect();
        if running.iter().any(|b|!b.is_zero()) {
            let mut projection=vec![K::ZERO;dims.assignment_width];
            for trial in 0..=REPEATS {
                let (_,projection_seconds)=seconds(||fill_combined_projection(running,&powers,&weights,&mut projection));
                let generic=||cache.eval_weighted_rows_from_projection(&projection,&coefficients,rows,rows);
                let (reference,generic_seconds);let (actual,factor_seconds);
                if trial%2==0 {(reference,generic_seconds)=seconds(generic);(actual,factor_seconds)=seconds(||plan.weighted(&projection,&coefficients));}
                else {(actual,factor_seconds)=seconds(||plan.weighted(&projection,&coefficients));(reference,generic_seconds)=seconds(generic);}
                assert_eq!(reference,actual);black_box(actual);
                let pad=std::array::from_fn(|i|gamma_power(gamma,running.len()*i));
                let (_,pad_seconds)=seconds(||fill_combined_projection(running,&powers,&pad,&mut projection));
                if trial!=0 {println!("EXPERIMENT {}",json!({"event":"carried","step":request.step,"trial":trial,"matrix_projection_seconds":projection_seconds,"pad_projection_seconds":pad_seconds,"generic_seconds":generic_seconds,"factor_seconds":factor_seconds,"equal":true}));}
            }
        }
        drop(window);
        // Actual source witnesses, deterministic full-extension challenges; no transcript-derived claim assumed.
        let fresh=vec![CcsWitness {Z:witnesses[0].clone()}];
        let challenges=Challenges::new((0..dims.variables).map(|i|coin(i+101)).collect(),gamma);
        let prior:Vec<_>=(0..dims.variables).map(|i|coin(i+201)).collect();
        let (mut oracle,build_seconds)=seconds(||OptimizedPaperJointOracle::new(&structure,&params,&fresh,&witnesses[1..],challenges,Some(&prior),dims,&source,BUDGET).unwrap());
        let points:Vec<_>=(0..=dims.degree).map(|i|K::from(F::from_usize(i))).collect();
        let mut rounds=0.;let mut binding=0.;
        for round in 0..dims.variables {
            let (_,message_seconds)=seconds(||black_box(oracle.evals_at(&points).unwrap()));
            let (_,bind_seconds)=seconds(||oracle.fold(coin(round)).unwrap());
            rounds+=message_seconds;binding+=bind_seconds;
        }
        let pad_scalars:Vec<_>=oracle.assignments.iter().map(|a|a.pair(0).0).collect();
        let scalar=oracle.fresh_tables.resident.as_ref().unwrap().words.clone();
        let (openings,opening_seconds)=seconds(||oracle.output_openings(&(0..dims.variables).map(coin).collect::<Vec<_>>()).unwrap().unwrap());
        for (value,opening) in pad_scalars.iter().zip(&openings) {assert_eq!(*value,opening.eval_k[0]);}
        for (matrix,&value) in scalar.iter().enumerate() {assert_eq!(value,openings[0].eval_a[matrix][0]);}
        println!("EXPERIMENT {}",json!({"event":"oracle","step":request.step,"build_seconds":build_seconds,"round_seconds":rounds,"binding_seconds":binding,"opening_seconds":opening_seconds,"scalar_reuse_equal":true,"reusable_fresh_scalars":scalar.len(),"opening_sources":openings.len()}));
        let (window,cache_seconds)=seconds(||MatrixWindow::load_next(&source,0..rows,BUDGET).unwrap());
        let point:Vec<_>=(0..dims.variables).map(coin).collect();
        for trial in 0..4 {
            let (got,timings)=crate::superneo_eval::recomposition_openings::measured(window.cache(),&point,&blocks);
            assert_eq!(serde_json::to_value(&got).unwrap(),serde_json::to_value(&openings).unwrap());
            if trial!=0 {println!("EXPERIMENT {}",json!({"event":"openings_split","step":request.step,"trial":trial,"timings":timings,"cache_seconds":cache_seconds,"equal":true}));}
        }

    });
}

#[path = "recomposition_proof.rs"]
mod proofs;
