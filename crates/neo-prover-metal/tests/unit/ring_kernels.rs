use super::*;
use neo_ajtai::nightstream_fprime_setup::PRODUCTION_MESSAGE_COLUMNS;
use neo_math::{superneo_bar_block, KExtensions, Rq, D, F, K};
use objc2_metal::MTLCommandEncoder;
use p3_field::{PrimeCharacteristicRing, PrimeField64};

const Q: u64 = F::ORDER_U64;
const PRODUCT: usize = 2 * D - 1;
const CHUNK_COLUMNS: usize = 512;

fn xorshift(seed: u64) -> impl FnMut() -> u64 {
    let mut state = seed;
    move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    }
}

fn ring(words: &[u64]) -> Rq {
    Rq(std::array::from_fn(|lane| F::new(words[lane])))
}

/// The second operand of each ring product.
enum Operands<'a> {
    /// One element for each column, lane by lane.
    Dense(&'a [u64]),
    /// The positive and the negative unit mask of each column.
    Masks(&'a [u64]),
}

/// The real and the imaginary sum over all columns of `form · operand`, from
/// the opening kernels: partial products, chunk sums and the ring fold. The
/// partial products run two times on the same buffers. The seconds are from
/// the commit of the second run to its completion, so they hold no page
/// mapping of new buffers.
fn device_products(session: &MetalSession, forms: &[u64], operands: Operands<'_>) -> (Vec<u64>, f64) {
    let columns = forms.len() / (2 * D);
    let chunks = columns.div_ceil(CHUNK_COLUMNS);
    let forms = session.buffer_from_slice(forms).unwrap();
    let shape = session
        .buffer_from_slice(&[columns as u64, 1, 2, columns as u64, chunks as u64, 0, 1, 0, 0, 1])
        .unwrap();
    let first = session.buffer_from_slice(&[0u32]).unwrap();
    let blocks = session
        .buffer_from_slice(&(0..columns as u32).collect::<Vec<_>>())
        .unwrap();
    let chunk_bases = session
        .buffer_from_slice(
            &(0..chunks)
                .map(|chunk| (chunk * CHUNK_COLUMNS) as u32)
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let chunk_matrices = session.buffer_from_slice(&vec![0u32; chunks]).unwrap();
    let active_offsets = session.buffer_from_slice(&[0u32, columns as u32]).unwrap();
    let chunk_offsets = session.buffer_from_slice(&[0u32, chunks as u32]).unwrap();
    let words = |count: usize| session.buffer(count * size_of::<u64>()).unwrap();
    let (partials, sums, output) = (words(chunks * 2 * PRODUCT), words(2 * PRODUCT), words(2 * D));
    let (pipeline, operand, grid) = match operands {
        Operands::Dense(values) => (
            &session.dec_dense_ring_partials,
            session.buffer_from_slice(values).unwrap(),
            Grid::Threads(chunks * 2 * PRODUCT),
        ),
        Operands::Masks(masks) => (
            &session.dec_sparse_ring_partials,
            session.buffer_from_slice(masks).unwrap(),
            Grid::Threads(chunks * 2 * PRODUCT),
        ),
    };
    let mut seconds = 0.0;
    for _ in 0..2 {
        let command = session.command_buffer("ring products").unwrap();
        let encoder = command.computeCommandEncoder().unwrap();
        session.launch(
            &encoder,
            pipeline,
            &[
                Arg::Buffer(0, &forms),
                Arg::Buffer(1, &operand),
                Arg::Buffer(2, &shape),
                Arg::Buffer(3, &partials),
                Arg::Buffer(4, &first),
                Arg::Buffer(5, &blocks),
                Arg::Buffer(6, &chunk_bases),
                Arg::Buffer(7, &chunk_matrices),
                Arg::Buffer(8, &active_offsets),
                Arg::Buffer(9, &first),
            ],
            grid,
        );
        encoder.endEncoding();
        let started = std::time::Instant::now();
        session.finish(&command).unwrap();
        seconds = started.elapsed().as_secs_f64();
    }

    let command = session.command_buffer("ring fold").unwrap();
    let encoder = command.computeCommandEncoder().unwrap();
    session.launch(
        &encoder,
        &session.dec_sparse_ring_sum_chunks,
        &[
            Arg::Buffer(0, &partials),
            Arg::Buffer(1, &shape),
            Arg::Buffer(2, &sums),
            Arg::Buffer(3, &chunk_offsets),
        ],
        Grid::Threads(2 * PRODUCT),
    );
    session.launch(
        &encoder,
        &session.dec_ring_reduce,
        &[Arg::Buffer(0, &sums), Arg::Buffer(1, &shape), Arg::Buffer(2, &output)],
        Grid::Threads(2 * D),
    );
    encoder.endEncoding();
    session.finish(&command).unwrap();
    (session.read_buffer(&output, 2 * D), seconds)
}

fn host_products(forms: &[u64], operands: &[Rq]) -> Vec<u64> {
    let mut sums = [Rq::zero(); 2];
    for (pair, operand) in forms.chunks_exact(2 * D).zip(operands) {
        for (sum, form) in sums.iter_mut().zip(pair.chunks_exact(D)) {
            *sum = sum.add(&ring(form).mul(operand));
        }
    }
    sums.iter()
        .flat_map(|sum| sum.0.iter().map(F::as_canonical_u64))
        .collect()
}

/// More than one chunk of columns, so a lane of one chunk adds
/// `CHUNK_COLUMNS · D / 2` products without a reduction between them.
#[test]
fn dense_ring_products_match_cpu_for_random_and_maximal_elements() {
    let session = MetalSession::new().unwrap();
    let columns = CHUNK_COLUMNS + 1;
    let mut next = xorshift(0x9e37_79b9_7f4a_7c15);
    let random = |count: usize, next: &mut dyn FnMut() -> u64| (0..count).map(|_| next() % Q).collect::<Vec<_>>();
    for (forms, values) in [
        (random(columns * 2 * D, &mut next), random(columns * D, &mut next)),
        (vec![Q - 1; columns * 2 * D], vec![Q - 1; columns * D]),
        (vec![Q - 1; columns * 2 * D], vec![1; columns * D]),
    ] {
        let operands: Vec<_> = values.chunks_exact(D).map(ring).collect();
        assert_eq!(
            device_products(&session, &forms, Operands::Dense(&values)).0,
            host_products(&forms, &operands)
        );
    }
}

/// Every lane of every signed-unit operand is set, so each product lane adds
/// the most words that it can.
#[test]
fn signed_unit_ring_products_match_cpu_for_random_and_maximal_elements() {
    let session = MetalSession::new().unwrap();
    let columns = CHUNK_COLUMNS + 1;
    let mut next = xorshift(0x2545_f491_4f6c_dd1d);
    let random_forms: Vec<_> = (0..columns * 2 * D).map(|_| next() % Q).collect();
    let random_masks: Vec<_> = (0..columns)
        .flat_map(|_| {
            let (signs, nonzero) = (next(), next());
            [nonzero & signs, nonzero & !signs]
        })
        .collect();
    for (forms, masks) in [
        (random_forms, random_masks),
        (vec![Q - 1; columns * 2 * D], [u64::MAX, 0].repeat(columns)),
        (vec![Q - 1; columns * 2 * D], [0, u64::MAX].repeat(columns)),
    ] {
        let operands: Vec<_> = masks
            .chunks_exact(2)
            .map(|pair| {
                Rq(std::array::from_fn(|lane| {
                    F::from_u64(pair[0] >> lane & 1) - F::from_u64(pair[1] >> lane & 1)
                }))
            })
            .collect();
        assert_eq!(
            device_products(&session, &forms, Operands::Masks(&masks)).0,
            host_products(&forms, &operands)
        );
    }
}

#[test]
fn bar_transform_matches_cpu() {
    let session = MetalSession::new().unwrap();
    let mut next = xorshift(0x1234_5678_9abc_def1);
    let mut forms: Vec<u64> = (0..5 * D).map(|_| next() % Q).collect();
    forms[D..2 * D].fill(0);
    forms[2 * D..3 * D].fill(Q - 1);
    let expected: Vec<u64> = forms
        .chunks_exact(D)
        .flat_map(|element| superneo_bar_block(ring(element).0).map(|value| value.as_canonical_u64()))
        .collect();
    let buffer = session.buffer_from_slice(&forms).unwrap();
    let command = session.command_buffer("bar").unwrap();
    let encoder = command.computeCommandEncoder().unwrap();
    session.launch(
        &encoder,
        &session.dec_bar_ring_forms_in_place,
        &[Arg::Buffer(0, &buffer)],
        Grid::Threads(forms.len() / 2),
    );
    encoder.endEncoding();
    session.finish(&command).unwrap();
    assert_eq!(session.read_buffer::<u64>(&buffer, forms.len()), expected);
}

/// `left + challenge · (right − left)` in `K` over words at the edges of the
/// field: 0, 1, the fold constant, the 32-bit limb edge, `q − 1`, and words
/// that are not below `q`.
#[test]
fn field_arithmetic_matches_cpu_at_edge_words() {
    let session = MetalSession::new().unwrap();
    let fold = Q.wrapping_neg();
    let edges = [
        0,
        1,
        2,
        fold - 1,
        fold,
        fold + 1,
        (1 << 32) - 1,
        1 << 32,
        (1 << 32) + 1,
        Q / 2,
        Q / 2 + 1,
        Q - fold,
        Q - 2,
        Q - 1,
        Q,
        u64::MAX,
    ];
    let k = |words: [u64; 2]| K::from_coeffs(words.map(F::new));
    let pairs: Vec<[u64; 2]> = edges
        .iter()
        .flat_map(|&c0| edges.iter().map(move |&c1| [c0, c1]))
        .collect();
    let lefts = [[0, 0], [Q - 1, Q - 1], [1, Q - 1]];
    let tables: Vec<[[u64; 2]; 2]> = lefts
        .iter()
        .flat_map(|&left| pairs.iter().map(move |&right| [left, right]))
        .collect();
    let device_tables = session.buffer_from_slice(&tables).unwrap();
    let shape = session
        .buffer_from_slice(&[2u64, tables.len() as u64])
        .unwrap();
    let output = session.buffer(tables.len() * size_of::<K>()).unwrap();
    for &challenge in &pairs {
        let device_challenge = session.buffer_from_slice(&challenge).unwrap();
        let command = session.command_buffer("field edges").unwrap();
        let encoder = command.computeCommandEncoder().unwrap();
        session.launch(
            &encoder,
            &session.joint_fold_k_tables,
            &[
                Arg::Buffer(0, &device_tables),
                Arg::Buffer(1, &device_challenge),
                Arg::Buffer(2, &shape),
                Arg::Buffer(3, &output),
            ],
            Grid::Threads(tables.len()),
        );
        encoder.endEncoding();
        session.finish(&command).unwrap();
        let actual = session.read_buffer::<[u64; 2]>(&output, tables.len());
        for (table, actual) in tables.iter().zip(actual) {
            let (left, right) = (k(table[0]), k(table[1]));
            assert_eq!(
                k(actual),
                left + k(challenge) * (right - left),
                "table={table:?} challenge={challenge:?}"
            );
            assert!(actual[0] < Q && actual[1] < Q);
        }
    }
}

#[test]
#[ignore = "timing evidence at production width; run on its own under the 300 s cap"]
fn ring_product_cost_at_production_width() {
    let session = MetalSession::new().unwrap();
    let columns = PRODUCTION_MESSAGE_COLUMNS as usize;
    let mut next = xorshift(0x9e37_79b9_7f4a_7c15);
    let forms: Vec<_> = (0..columns * 2 * D).map(|_| next() % Q).collect();
    let values: Vec<_> = (0..columns * D).map(|_| next() % Q).collect();
    let masks: Vec<_> = (0..columns)
        .flat_map(|_| {
            let (signs, nonzero) = (next(), next());
            [nonzero & signs, nonzero & !signs]
        })
        .collect();
    for run in ["warm-up", "timed"] {
        let dense = device_products(&session, &forms, Operands::Dense(&values)).1;
        let signed = device_products(&session, &forms, Operands::Masks(&masks)).1;
        eprintln!("ring products run={run} columns={columns} dense_seconds={dense:.4} signed_unit_seconds={signed:.4}");
    }
}
