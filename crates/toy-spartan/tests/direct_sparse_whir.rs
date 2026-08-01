use toy_spartan::{
  SparseMatrix, SplitR1CSShape,
  provider::{GoldilocksWhirEngine, goldi::F},
  spartan::{DIRECT_R1CS_REPETITIONS, R1CSSNARK, RepeatedR1CSSNARK},
  traits::snark::R1CSSNARKTrait,
};

type Engine = GoldilocksWhirEngine;
type Snark = R1CSSNARK<Engine>;
type ProductionSnark = RepeatedR1CSSNARK<Engine>;

fn matrix(rows: usize, cols: usize, entries: &[(usize, usize, u64)]) -> SparseMatrix<F> {
  let mut data = Vec::with_capacity(entries.len());
  let mut indices = Vec::with_capacity(entries.len());
  let mut indptr = vec![0usize; rows + 1];
  let mut cursor = 0usize;
  for row in 0..rows {
    while cursor < entries.len() && entries[cursor].0 == row {
      indices.push(entries[cursor].1);
      data.push(F::new(entries[cursor].2));
      cursor += 1;
    }
    indptr[row + 1] = cursor;
  }
  assert_eq!(cursor, entries.len());
  SparseMatrix::from_csr(rows, cols, data, indices, indptr).unwrap()
}

fn multiplication_shape() -> SplitR1CSShape<Engine> {
  // Private columns: x=0, y=1. Constant: 2. Public: product=3, sum=4.
  let rows = 2;
  let cols = 5;
  let a = matrix(rows, cols, &[(0, 0, 1), (1, 0, 1), (1, 1, 1)]);
  let b = matrix(rows, cols, &[(0, 1, 1), (1, 2, 1)]);
  let c = matrix(rows, cols, &[(0, 3, 1), (1, 4, 1)]);
  SplitR1CSShape::new(2, rows, 0, 0, 2, 2, 0, a, b, c).unwrap()
}

#[test]
fn direct_sparse_r1cs_proves_and_verifies_with_whir() {
  let (pk, vk) = Snark::setup_direct(multiplication_shape()).unwrap();
  let proof = Snark::prove_direct(
    &pk,
    &[F::new(6), F::new(7)],
    &[F::new(42), F::new(13)],
    true,
  )
  .unwrap();

  assert_eq!(proof.verify(&vk).unwrap(), [F::new(42), F::new(13)]);
}

#[test]
fn repeated_direct_sparse_r1cs_proves_and_verifies_with_whir() {
  let (pk, vk) = Snark::setup_direct(multiplication_shape()).unwrap();
  let proof = ProductionSnark::prove_direct(
    &pk,
    &[F::new(6), F::new(7)],
    &[F::new(42), F::new(13)],
    true,
  )
  .unwrap();

  assert_eq!(proof.proofs().len(), DIRECT_R1CS_REPETITIONS);
  assert_eq!(proof.verify(&vk).unwrap(), [F::new(42), F::new(13)]);
  assert!(proof.proofs()[0].verify(&vk).is_err());
}

#[test]
fn repeated_direct_sparse_r1cs_rejects_a_valid_member_from_another_witness() {
  let (pk, vk) = Snark::setup_direct(multiplication_shape()).unwrap();
  let first = ProductionSnark::prove_direct(
    &pk,
    &[F::new(6), F::new(7)],
    &[F::new(42), F::new(13)],
    true,
  )
  .unwrap();
  let second = ProductionSnark::prove_direct(
    &pk,
    &[F::new(7), F::new(6)],
    &[F::new(42), F::new(13)],
    true,
  )
  .unwrap();

  let mut encoded = bincode::serialize(&first).unwrap();
  let old_member = bincode::serialize(&first.proofs()[1]).unwrap();
  let new_member = bincode::serialize(&second.proofs()[1]).unwrap();
  assert_eq!(old_member.len(), new_member.len());
  let offsets = encoded
    .windows(old_member.len())
    .enumerate()
    .filter_map(|(offset, bytes)| (bytes == old_member).then_some(offset))
    .collect::<Vec<_>>();
  assert_eq!(offsets.len(), 1, "member encoding must occur exactly once");
  encoded[offsets[0]..offsets[0] + old_member.len()].copy_from_slice(&new_member);

  let spliced: ProductionSnark = bincode::deserialize(&encoded).unwrap();
  assert!(spliced.verify(&vk).is_err());
}

#[test]
fn repeated_direct_sparse_r1cs_rejects_an_unsatisfied_public_value() {
  let (pk, _) = Snark::setup_direct(multiplication_shape()).unwrap();
  let error = ProductionSnark::prove_direct(
    &pk,
    &[F::new(6), F::new(7)],
    &[F::new(43), F::new(13)],
    true,
  )
  .err()
  .expect("invalid public value must fail");

  assert!(matches!(
    error,
    toy_spartan::errors::SpartanError::UnSat { .. }
  ));
}

#[test]
fn direct_sparse_r1cs_rejects_an_unsatisfied_public_value() {
  let (pk, _) = Snark::setup_direct(multiplication_shape()).unwrap();
  let error = Snark::prove_direct(
    &pk,
    &[F::new(6), F::new(7)],
    &[F::new(43), F::new(13)],
    true,
  )
  .unwrap_err();

  assert!(matches!(
    error,
    toy_spartan::errors::SpartanError::UnSat { .. }
  ));
}

#[test]
fn direct_sparse_r1cs_rejects_noncanonical_csr() {
  let error =
    SparseMatrix::from_csr(1, 2, vec![F::new(1), F::new(1)], vec![1, 1], vec![0, 2]).unwrap_err();

  assert_eq!(error, toy_spartan::errors::SpartanError::InvalidIndex);
}
