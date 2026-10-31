use neo_verified::circuit::{Kind, Wire};
use neo_verified::degrees::{
    early, early_terms, gkr, late, late_terms, layer1, linear, linear_terms, row_count, shrink, shrink_terms, Relation,
    FIXED,
};
use neo_verified::field::{
    canonical, ext_add, ext_mul, ext_neg, ext_of_gl, ext_pow, ext_scale, ext_sub, gl_add, gl_inv, gl_mul, gl_neg,
    gl_pow, gl_sub, gl_word, reduce, Ext, EXT_ONE, EXT_ZERO, GL_ONE, GL_ZERO, MODULUS,
};
use neo_verified::layout::{
    block_cell, block_row, cell_variables, column, entry_column, glue_row, one, place, row_variables, slot, split,
    variables, Part, Shape,
};
use neo_verified::poseidon2::{permute, DIAGONAL, HALF_FULL_ROUNDS, INITIAL, PARTIAL, PARTIAL_ROUNDS, TERMINAL};
use std::panic::{catch_unwind, UnwindSafe};
use std::path::Path;
use std::process::Command;

const EDGES: [usize; 16] = [
    0,
    1,
    2,
    3,
    4,
    5,
    7,
    8,
    9,
    (1 << 32) - 1,
    1 << 32,
    (1 << 32) + 1,
    (1 << 63) - 1,
    1 << 63,
    (1 << 63) + 1,
    usize::MAX,
];

const SHIFTS: [u32; 11] = [0, 1, 31, 32, 63, 64, 65, 127, 128, 1 << 31, u32::MAX];

/// Block counts at the edges of `place`: `192·count` is the end of both part kinds.
const COUNTS: [usize; 8] = [
    0,
    1,
    3,
    1000,
    usize::MAX / 192,
    usize::MAX / 192 + 1,
    usize::MAX / 128 + 1,
    usize::MAX,
];

/// Glue rows, glue cells and statement words; `2^63` makes `cell_variables` 64.
const GLUES: [[usize; 3]; 9] = [
    [0, 0, 0],
    [70, 90, 5],
    [1 << 32, 1 << 32, 1 << 32],
    [usize::MAX, 0, 0],
    [0, usize::MAX, 0],
    [0, 0, usize::MAX],
    [0, 0, (1 << 63) - 1],
    [0, 0, 1 << 63],
    [1 << 63, 1 << 63, 0],
];

fn random(state: &mut u64) -> usize {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    (*state >> (*state % 64)) as usize
}

fn inputs() -> Vec<usize> {
    let mut state: u64 = 0x9e37_79b9_7f4a_7c15;
    let mut values = EDGES.to_vec();
    for _ in 0..200 {
        values.push(random(&mut state));
    }
    values
}

fn shapes() -> Vec<Shape> {
    let mut shapes = Vec::new();
    for count in COUNTS {
        for [glue_rows, glue_cells, publics] in GLUES {
            shapes.push(Shape {
                blocks: [count],
                glue_rows,
                glue_cells,
                publics,
            });
        }
    }
    let mut state: u64 = 0x2545_f491_4f6c_dd1d;
    for _ in 0..60 {
        shapes.push(Shape {
            blocks: [random(&mut state) >> 8],
            glue_rows: random(&mut state),
            glue_cells: random(&mut state),
            publics: random(&mut state),
        });
    }
    shapes
}

fn shape_words(shape: &Shape) -> [u64; 4] {
    [shape.blocks[0], shape.glue_rows, shape.glue_cells, shape.publics].map(|word| word as u64)
}

fn wire_words(wire: Wire) -> [u64; 3] {
    match wire {
        Wire::Glue(cell) => [0, cell.into(), 0],
        Wire::Block(_, block, cell) => [1, block.into(), cell.into()],
        Wire::Public(index) => [2, index.into(), 0],
    }
}

const WIRES: [Wire; 6] = [
    Wire::Glue(0),
    Wire::Glue(u32::MAX),
    Wire::Block(Kind::Permutation, 1, 129),
    Wire::Block(Kind::Permutation, u32::MAX, u16::MAX),
    Wire::Public(0),
    Wire::Public(u32::MAX),
];

fn show_parts(([wide, narrow], end): ([Part; 2], usize)) -> String {
    format!(
        "{} {} {} {} {end}",
        wide.start, wide.stride, narrow.start, narrow.stride
    )
}

fn words(values: &[usize]) -> Vec<u64> {
    values.iter().map(|&value| value as u64).collect()
}

fn outcome<T>(run: impl FnOnce() -> T + UnwindSafe, show: impl Fn(T) -> String) -> String {
    match catch_unwind(run) {
        Ok(value) => format!("ok {}", show(value)),
        Err(_) => "fail".to_string(),
    }
}

fn unwrap(value: Option<usize>) -> usize {
    value.unwrap()
}

fn option(value: Option<impl std::fmt::Display>) -> String {
    value.map_or("none".to_string(), |v| format!("some {v}"))
}

fn cases() -> Vec<String> {
    let mut cases = Vec::new();
    let mut case = |name: &str, args: &[u64], output: String| {
        let args: Vec<String> = args.iter().map(u64::to_string).collect();
        cases.push(format!("{name} {} -> {output}", args.join(" ")).replace("  ->", " ->"));
    };
    for x in inputs() {
        let w = x as u64;
        case("split", &[w], outcome(|| split(x), |[a, b]| format!("{a} {b}")));
        case("variables", &[w], outcome(|| variables(x), |v| v.to_string()));
        case(
            "core.num.Usize.leading_zeros",
            &[w],
            format!("ok {}", x.leading_zeros()),
        );
        for y in EDGES {
            case(
                "Usize.checked_sub",
                &[w, y as u64],
                format!("ok {}", option(x.checked_sub(y))),
            );
        }
    }
    for k in SHIFTS {
        for x in [1usize, 3, usize::MAX] {
            case(
                "core.num.Usize.wrapping_shl",
                &[x as u64, k.into()],
                format!("ok {}", x.wrapping_shl(k)),
            );
        }
        for j in SHIFTS {
            case(
                "U32.checked_sub",
                &[k.into(), j.into()],
                format!("ok {}", option(k.checked_sub(j))),
            );
        }
        case("UScalar.cast", &[k.into()], format!("ok {}", k as usize));
    }
    case(
        "core.option.Option.unwrap.some",
        &[5],
        outcome(|| unwrap(Some(5)), |v| v.to_string()),
    );
    case(
        "core.option.Option.unwrap.none",
        &[],
        outcome(|| unwrap(None), |v| v.to_string()),
    );
    case("core.num.Usize.BITS", &[], format!("ok {}", usize::BITS));
    case("Array.make", &[0, u64::MAX], format!("ok 0 {}", u64::MAX));
    layout_cases(&mut case);
    model_cases(&mut case);
    degrees_cases(&mut case);
    field_cases(&mut case);
    poseidon2_cases(&mut case);
    cases
}

fn layout_cases(case: &mut impl FnMut(&str, &[u64], String)) {
    let kind = Kind::Permutation;
    case("Kind.index", &[], format!("ok {}", kind.index()));
    case("Kind.rows", &[], format!("ok {}", kind.rows()));
    case("Kind.cells", &[], format!("ok {}", kind.cells()));
    let sizes = [
        0,
        1,
        2,
        3,
        4,
        5,
        7,
        8,
        9,
        166,
        182,
        1 << 32,
        (1 << 63) - 1,
        1 << 63,
        (1 << 63) + 1,
        usize::MAX,
    ];
    let counts = [0, 1, 2, 1000, 1 << 32, 1 << 62, usize::MAX];
    for size in sizes {
        for count in counts {
            case(
                "place",
                &words(&[size, count]),
                outcome(|| place(size, count), show_parts),
            );
        }
    }
    let mut state: u64 = 0x6a09_e667_f3bc_c908;
    for _ in 0..100 {
        let (size, count) = (random(&mut state), random(&mut state));
        case(
            "place",
            &words(&[size, count]),
            outcome(|| place(size, count), show_parts),
        );
    }
    for size in [1, 2, 3, 7, 166, 182, 1 << 63, usize::MAX] {
        for count in [0, 1, 5, 1 << 57, usize::MAX] {
            for b in [0, 1, 4, 1 << 60] {
                for k in [0, 1, 6, 127, 128, 165, usize::MAX] {
                    case(
                        "slot",
                        &words(&[size, count, b, k]),
                        outcome(|| slot(size, count, b, k), |v| v.to_string()),
                    );
                }
            }
        }
    }
    for shape in shapes() {
        let s = shape_words(&shape);
        let with = |extra: &[u64]| [s.as_slice(), extra].concat();
        case(
            "row_variables",
            &s,
            outcome(|| row_variables(&shape), |v| v.to_string()),
        );
        case(
            "cell_variables",
            &s,
            outcome(|| cell_variables(&shape), |v| v.to_string()),
        );
        case("one", &s, outcome(|| one(&shape), |v| v.to_string()));
        for g in [0, 1, usize::MAX] {
            case(
                "glue_row",
                &with(&words(&[g])),
                outcome(|| glue_row(&shape, g), |v| v.to_string()),
            );
        }
        for (b, k) in [(0, 0), (2, 130), (usize::MAX, 0)] {
            case(
                "block_row",
                &with(&words(&[b, k])),
                outcome(|| block_row(&shape, kind, b, k), |v| v.to_string()),
            );
        }
        for (b, k) in [(0, 0), (2, 150), (1 << 20, 181)] {
            case(
                "block_cell",
                &with(&words(&[b, k])),
                outcome(|| block_cell(&shape, kind, b, k), |v| v.to_string()),
            );
        }
        for wire in WIRES {
            case(
                "column",
                &with(&wire_words(wire)),
                outcome(|| column(&shape, wire), |v| v.to_string()),
            );
        }
        for (b, cell) in [(0, None), (1, Some(3)), (usize::MAX, Some(usize::MAX))] {
            let cell_words = cell.map_or([0, 0], |cell| [1, cell as u64]);
            case(
                "entry_column",
                &with(&[b as u64, cell_words[0], cell_words[1]]),
                outcome(|| entry_column(&shape, kind, b, cell), |v| v.to_string()),
            );
        }
    }
}

/// The models that `place`, `cell_variables` and the derived `Clone` and `PartialEq` impls call.
/// A clone of a `Copy` value is the value, so the Rust side of a clone case is its input.
fn model_cases(case: &mut impl FnMut(&str, &[u64], String)) {
    for x in EDGES {
        for y in EDGES {
            let args = words(&[x, y]);
            case("Usize.checked_add", &args, format!("ok {}", option(x.checked_add(y))));
            case("Usize.checked_mul", &args, format!("ok {}", option(x.checked_mul(y))));
            case("core.cmp.impls.OrdUsize.max", &args, format!("ok {}", x.max(y)));
            case("core.cmp.PartialEqUsize.eq", &args, format!("ok {}", x == y));
            case(
                "core.array.equality.PartialEqArray.eq",
                &args,
                format!("ok {}", [x] == [y]),
            );
        }
    }
    for i in [0, 1, 2, usize::MAX] {
        let array = [7usize, 9];
        case(
            "Array.index_usize",
            &words(&[array[0], array[1], i]),
            outcome(|| array[i], |v| v.to_string()),
        );
    }
    for x in [0u32, 1, 1 << 31, u32::MAX] {
        for y in [0u32, 1, 1 << 31, u32::MAX] {
            case(
                "core.cmp.impls.PartialEqU32.eq",
                &[x.into(), y.into()],
                format!("ok {}", x == y),
            );
        }
    }
    for x in [0u16, 1, 1 << 15, u16::MAX] {
        for y in [0u16, 1, 1 << 15, u16::MAX] {
            case(
                "core.cmp.impls.PartialEqU16.eq",
                &[x.into(), y.into()],
                format!("ok {}", x == y),
            );
        }
    }
    for a in WIRES {
        let words_a = wire_words(a);
        case(
            "core.clone.Clone.clone.wire",
            &words_a,
            format!("ok {}", show(&words_a)),
        );
        case(
            "core.marker.Copy.cloneInst.wire",
            &words_a,
            format!("ok {}", show(&words_a)),
        );
        case("core.marker.StructuralPartialEq.wire", &words_a, "ok".to_string());
        for b in WIRES {
            let args = [words_a, wire_words(b)].concat();
            case("core.cmp.PartialEq.eq.wire", &args, format!("ok {}", a == b));
        }
    }
    let kind = Kind::Permutation;
    case("core.clone.Clone.clone.kind", &[], format!("ok {}", kind.index()));
    case(
        "core.cmp.PartialEq.eq.kind",
        &[],
        format!("ok {}", kind == Kind::Permutation),
    );
    let shapes = shapes();
    for a in &shapes[..12] {
        let words_a = shape_words(a);
        case(
            "core.clone.Clone.clone.shape",
            &words_a,
            format!("ok {}", show(&words_a)),
        );
        for b in &shapes[..12] {
            let args = [words_a, shape_words(b)].concat();
            case("core.cmp.PartialEq.eq.shape", &args, format!("ok {}", a == b));
        }
    }
    case("core.clone.Clone.clone.part", &[3, 4], "ok 3 4".to_string());
}

fn relation_words(r: &Relation) -> Vec<u64> {
    vec![
        r.d as u64,
        r.lane_variables as u64,
        r.block_variables as u64,
        r.kappa as u64,
        r.matrices as u64,
        r.public_blocks as u64,
        r.norm_bound.into(),
        r.run_variables as u64,
        r.row_variables as u64,
        r.slot_variables as u64,
        r.matrix_variables as u64,
    ]
}

/// A relation from its fields in declaration order; the norm bound is the seventh.
fn relation(w: [usize; 11]) -> Relation {
    Relation {
        d: w[0],
        lane_variables: w[1],
        block_variables: w[2],
        kappa: w[3],
        matrices: w[4],
        public_blocks: w[5],
        norm_bound: w[6] as u32,
        run_variables: w[7],
        row_variables: w[8],
        slot_variables: w[9],
        matrix_variables: w[10],
    }
}

/// Relations at the edges of the layer-1 charges: a production-sized one, zero counts,
/// `D = 0`, exponents of 63 and 64, a norm bound of `u32::MAX`, and sums that overflow.
fn relations() -> Vec<Relation> {
    let mut relations = vec![
        relation([54, 6, 5, 16, 3, 1, 1 << 15, 20, 20, 18, 2]),
        relation([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        relation([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        relation([54, 6, 57, 0, 0, 0, 1, 1, 1, 1, 1]),
        relation([54, 6, 58, 0, 0, 0, 1, 1, 1, 1, 1]),
        relation([54, 0, 0, 0, 0, 0, u32::MAX as usize, 0, 0, 0, 0]),
        relation([54, 0, 0, 0, 0, 0, 0, 62, 0, 0, 0]),
        relation([54, 0, 0, 0, 0, 0, 0, 0, 0, 62, 1]),
        relation([54, 0, 63, 0, 0, 0, 0, 0, 0, 62, 0]),
        relation([54, 0, 0, 0, 0, 0, 0, 0, 0, 63, 0]),
        relation([usize::MAX, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        relation([54, usize::MAX, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
        relation([54, 6, 5, usize::MAX - 2, 0, 0, 0, 0, 0, 0, 0]),
        relation([54, 6, 5, usize::MAX - 3, 0, 0, 0, 0, 0, 0, 0]),
        relation([54, 6, 5, 0, usize::MAX / 2, 0, 0, 0, 0, 0, 0]),
        relation([54, 6, 5, 0, 0, usize::MAX, 0, 0, 0, 0, 0]),
    ];
    let mut state: u64 = 0x510e_527f_ade6_82d1;
    for _ in 0..80 {
        let mut words: [usize; 11] = std::array::from_fn(|_| random(&mut state) % 70);
        words[6] = random(&mut state) >> 32;
        relations.push(relation(words));
    }
    relations
}

fn show_values(values: &[usize]) -> String {
    show(&words(values))
}

/// The degrees that the layer-1 and shrink charges read, and the models that only they call.
fn degrees_cases(case: &mut impl FnMut(&str, &[u64], String)) {
    let counts = [
        0,
        1,
        2,
        16,
        (1 << 32) - 1,
        usize::MAX / 2 - 1,
        usize::MAX / 2,
        usize::MAX - 2,
        usize::MAX,
    ];
    for kappa in counts {
        for matrices in counts {
            for publics in [0, 1, usize::MAX / 2, usize::MAX] {
                case(
                    "row_count",
                    &words(&[kappa, matrices, publics]),
                    outcome(|| row_count(kappa, matrices, publics), |v| v.to_string()),
                );
            }
        }
    }
    let mut depth_lists: Vec<Vec<usize>> = vec![
        vec![],
        vec![0],
        vec![1],
        vec![0, 0, 0],
        vec![3, 1, 2],
        vec![25, 20, 20, 20, 20],
        vec![19, 5],
        vec![1 << 31],
        vec![(1 << 31) - 1, 1 << 20],
        vec![1 << 32],
        vec![usize::MAX],
        vec![usize::MAX / 2, usize::MAX / 2 + 2],
    ];
    let mut state: u64 = 0x1f83_d9ab_fb41_bd6b;
    for _ in 0..60 {
        let length = random(&mut state) % 7;
        depth_lists.push((0..length).map(|_| random(&mut state) % 100).collect());
    }
    for depths in &depth_lists {
        case("gkr", &words(depths), outcome(|| gkr(depths), |v| v.to_string()));
        case("Slice.len", &words(depths), format!("ok {}", depths.len()));
        for i in [0, 1, 4, usize::MAX] {
            case(
                "Slice.index_usize",
                &words(&[[i].as_slice(), depths].concat()),
                outcome(|| depths[i], |v| v.to_string()),
            );
        }
    }
    for x in [0, 7, usize::MAX] {
        case(
            "Array.to_slice",
            &words(&[x, 1, 2]),
            format!("ok {}", show_values([x, 1, 2].as_slice())),
        );
    }
    for x in EDGES {
        case("UScalar.cast.u32", &[x as u64], format!("ok {}", x as u32));
        for y in [0, 1, 2, 3, 8, usize::MAX] {
            let args = words(&[x, y]);
            case("add", &args, outcome(|| x.checked_add(y).unwrap(), |v| v.to_string()));
            case("sub", &args, outcome(|| x.checked_sub(y).unwrap(), |v| v.to_string()));
            case("mul", &args, outcome(|| x.checked_mul(y).unwrap(), |v| v.to_string()));
        }
    }
    for v in (0..70).chain([1 << 32, usize::MAX]) {
        let rust = if v < 64 {
            format!("ok {}", 1usize << v)
        } else {
            "fail".to_string()
        };
        case("exp2", &[v as u64], rust);
    }
    case("FIXED", &[], format!("ok {}", show_values(&FIXED)));
    for r in relations() {
        let args = relation_words(&r);
        let cube = r.lane_variables.checked_add(r.block_variables);
        case(
            "cube_variables",
            &args,
            cube.map_or("fail".to_string(), |v| format!("ok {v}")),
        );
        case("early_terms", &args, outcome(|| early_terms(&r), |v| show_values(&v)));
        case("linear_terms", &args, outcome(|| linear_terms(&r), |v| show_values(&v)));
        case("late_terms", &args, outcome(|| late_terms(&r), |v| show_values(&v)));
        case("early", &args, outcome(|| early(&r), |v| v.to_string()));
        case("linear", &args, outcome(|| linear(&r), |v| v.to_string()));
        case("late", &args, outcome(|| late(&r), |v| v.to_string()));
        case("layer1", &args, outcome(|| layer1(&r), |v| v.to_string()));
    }
    for shape in shapes() {
        let s = shape_words(&shape);
        case(
            "shrink_terms",
            &s,
            outcome(|| shrink_terms(&shape), |v| show_values(&v)),
        );
        case("shrink", &s, outcome(|| shrink(&shape), |v| v.to_string()));
    }
}

/// Words at the edges of the Goldilocks reductions, canonical and not.
const WORDS: [u64; 16] = [
    0,
    1,
    2,
    0xffff_fffe,
    0xffff_ffff,
    1 << 32,
    (1 << 32) + 1,
    1 << 63,
    MODULUS - 0xffff_ffff,
    MODULUS - 2,
    MODULUS - 1,
    MODULUS,
    MODULUS + 1,
    MODULUS + (1 << 31),
    u64::MAX - 1,
    u64::MAX,
];

const EXPONENTS: [u64; 10] = [0, 1, 2, 3, 7, 63, 64, MODULUS - 2, MODULUS - 1, u64::MAX];

fn field_words() -> Vec<u64> {
    let mut state: u64 = 0x676f_6c64_696c_6f63;
    let mut words = WORDS.to_vec();
    for _ in 0..100 {
        words.push(random(&mut state) as u64);
    }
    words
}

fn ext_words(e: Ext) -> [u64; 3] {
    [e.c0, e.c1, e.c2].map(gl_word)
}

fn show_ext(e: Ext) -> String {
    format!("ok {}", show(&ext_words(e)))
}

fn exts(words: &[u64]) -> Vec<Ext> {
    (0..words.len())
        .map(|i| Ext {
            c0: reduce(words[i]),
            c1: reduce(words[(i + 5) % words.len()]),
            c2: reduce(words[(i + 11) % words.len()]),
        })
        .collect()
}

/// The private word functions of `field.rs`, as `NativeGoldilocks.lean` writes them.
fn add_words(a: u64, b: u64) -> u64 {
    let room = MODULUS.wrapping_sub(b);
    if a < room {
        a.wrapping_add(b)
    } else {
        a.wrapping_sub(room)
    }
}

fn sub_words(a: u64, b: u64) -> u64 {
    if b <= a {
        a.wrapping_sub(b)
    } else {
        MODULUS.wrapping_sub(b.wrapping_sub(a))
    }
}

fn fold_high(w: u64) -> u64 {
    let low = w as u32 as u64;
    sub_words((low << 32).wrapping_sub(low), w >> 32)
}

/// The field of `neo_verified::field`, its private word functions and the models they call.
fn field_cases(case: &mut impl FnMut(&str, &[u64], String)) {
    let words = field_words();
    let gl = |w: u64| format!("ok {}", gl_word(reduce(w)));
    case("MODULUS", &[], format!("ok {MODULUS}"));
    case("GL_ZERO", &[], format!("ok {}", gl_word(GL_ZERO)));
    case("GL_ONE", &[], format!("ok {}", gl_word(GL_ONE)));
    case("EXT_ZERO", &[], show_ext(EXT_ZERO));
    case("EXT_ONE", &[], show_ext(EXT_ONE));
    for &x in &words {
        let a = reduce(x);
        let w = gl_word(a);
        case("canonical", &[x], format!("ok {}", option(canonical(x).map(gl_word))));
        case("reduce", &[x], gl(x));
        case("reduce_word", &[x], format!("ok {}", x % MODULUS));
        case("gl_word", &[w], format!("ok {w}"));
        case("low_half", &[x], format!("ok {}", x as u32 as u64));
        case("high_half", &[x], format!("ok {}", x >> 32));
        case("fold_high", &[x], format!("ok {}", fold_high(x)));
        case("gl_neg", &[w], format!("ok {}", gl_word(gl_neg(a))));
        case("gl_inv", &[w], format!("ok {}", gl_word(gl_inv(a))));
        case("ext_of_gl", &[w], show_ext(ext_of_gl(a)));
        case("UScalar.cast.u32", &[x], format!("ok {}", x as u32));
        case(
            "UScalar.cast.u64",
            &[x as u32 as u64],
            format!("ok {}", x as u32 as u64),
        );
        case(
            "core.num.U64.wrapping_shr",
            &[x, 32],
            format!("ok {}", x.wrapping_shr(32)),
        );
        case(
            "core.num.U64.wrapping_shr",
            &[x, 1],
            format!("ok {}", x.wrapping_shr(1)),
        );
        case(
            "core.num.U64.wrapping_shl",
            &[x, 32],
            format!("ok {}", x.wrapping_shl(32)),
        );
        case(
            "core.num.U64.is_multiple_of",
            &[x, 2],
            format!("ok {}", x.is_multiple_of(2)),
        );
        for e in EXPONENTS {
            case("gl_pow", &[w, e], format!("ok {}", gl_word(gl_pow(a, e))));
        }
        for &y in &WORDS {
            let b = reduce(y);
            let v = gl_word(b);
            case("add_words", &[x, y], format!("ok {}", add_words(x, y)));
            case("sub_words", &[x, y], format!("ok {}", sub_words(x, y)));
            let product = (u128::from(x) * u128::from(y) % u128::from(MODULUS)) as u64;
            case("mul_words", &[x, y], format!("ok {product}"));
            case("gl_add", &[w, v], format!("ok {}", gl_word(gl_add(a, b))));
            case("gl_sub", &[w, v], format!("ok {}", gl_word(gl_sub(a, b))));
            case("gl_mul", &[w, v], format!("ok {}", gl_word(gl_mul(a, b))));
            case(
                "core.num.U64.wrapping_add",
                &[x, y],
                format!("ok {}", x.wrapping_add(y)),
            );
            case(
                "core.num.U64.wrapping_sub",
                &[x, y],
                format!("ok {}", x.wrapping_sub(y)),
            );
            case(
                "core.num.U64.wrapping_mul",
                &[x, y],
                format!("ok {}", x.wrapping_mul(y)),
            );
            case("core.cmp.PartialEq.eq.gl", &[w, v], format!("ok {}", a == b));
        }
        case("core.clone.Clone.clone.gl", &[w], format!("ok {w}"));
        case("core.cmp.Eq.gl", &[w], "ok unit".to_string());
    }
    let values = exts(&words);
    for (i, &a) in values.iter().enumerate() {
        let wa = ext_words(a);
        case("ext_neg", &wa, show_ext(ext_neg(a)));
        case("core.clone.Clone.clone.ext", &wa, show_ext(a));
        case("core.cmp.Eq.ext", &wa, "ok unit".to_string());
        for e in EXPONENTS {
            case("ext_pow", &[&wa[..], &[e]].concat(), show_ext(ext_pow(a, e)));
        }
        let g = reduce(words[(7 * i) % words.len()]);
        case(
            "ext_scale",
            &[&wa[..], &[gl_word(g)]].concat(),
            show_ext(ext_scale(a, g)),
        );
        for &b in values.iter().step_by(7) {
            let args = [wa, ext_words(b)].concat();
            case("ext_add", &args, show_ext(ext_add(a, b)));
            case("ext_sub", &args, show_ext(ext_sub(a, b)));
            case("ext_mul", &args, show_ext(ext_mul(a, b)));
            case("core.cmp.PartialEq.eq.ext", &args, format!("ok {}", a == b));
        }
        case("core.cmp.PartialEq.eq.ext", &[wa, wa].concat(), "ok true".to_string());
    }
}

/// The permutation on states of one edge word, on a state of all the edge words, and on random
/// states, with its tables and the models that only it calls. Lean reads each word as given,
/// canonical or not; Rust reduces it first, as `neo-spartan`'s `to_verified` does.
fn poseidon2_cases(case: &mut impl FnMut(&str, &[u64], String)) {
    let mut states: Vec<[u64; 16]> = WORDS.iter().map(|&word| [word; 16]).collect();
    states.push(WORDS);
    let mut state: u64 = 0x706f_7365_6964_6f6e;
    for _ in 0..40 {
        states.push(std::array::from_fn(|_| random(&mut state) as u64));
    }
    for words in states {
        let output = permute(words.map(reduce)).map(gl_word);
        case("permute", &words, format!("ok {}", show(&output)));
    }
    case("HALF_FULL_ROUNDS", &[], format!("ok {HALF_FULL_ROUNDS}"));
    case("PARTIAL_ROUNDS", &[], format!("ok {PARTIAL_ROUNDS}"));
    case("INITIAL", &[], format!("ok {}", show(&INITIAL.concat())));
    case("PARTIAL", &[], format!("ok {}", show(&PARTIAL)));
    case("TERMINAL", &[], format!("ok {}", show(&TERMINAL.concat())));
    case("DIAGONAL", &[], format!("ok {}", show(&DIAGONAL)));
    for x in EDGES {
        for y in EDGES {
            case(
                "core.num.Usize.wrapping_add",
                &words(&[x, y]),
                format!("ok {}", x.wrapping_add(y)),
            );
        }
    }
    for i in [0, 1, 2, usize::MAX] {
        let update = move || {
            let mut array = [7usize, 9];
            array[i] = 5;
            array
        };
        case(
            "Array.update",
            &words(&[7, 9, i, 5]),
            outcome(update, |a| show_values(&a)),
        );
    }
}

fn show(words: &[u64]) -> String {
    words
        .iter()
        .map(u64::to_string)
        .collect::<Vec<_>>()
        .join(" ")
}

#[test]
#[ignore = "needs Lean: the pinned toolchain and `lake build CompressionLeanRust` in formal/compression-lean"]
fn translation_runs_as_rust() {
    assert_eq!(usize::BITS, 64, "the cases assume a 64-bit usize");
    std::panic::set_hook(Box::new(|_| {}));
    let cases = cases();
    let models = include_str!("../../../formal/aeneas/std-models.txt");
    for model in models
        .lines()
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
    {
        assert!(
            cases
                .iter()
                .any(|case| case.starts_with(&format!("{model} ")) || case.starts_with(&format!("{model}."))),
            "formal/aeneas/std-models.txt lists {model}, which has no case"
        );
    }
    let _ = std::panic::take_hook();
    let file = Path::new(env!("CARGO_TARGET_TMPDIR")).join("neo_verified_run_check.txt");
    std::fs::write(&file, cases.join("\n") + "\n").unwrap();
    let lean = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../formal/compression-lean");
    let output = Command::new("lake")
        .args(["env", "lean", "--run", "CompressionLeanRust/RunCheck.lean"])
        .arg(&file)
        .current_dir(&lean)
        .output()
        .expect("run lake from formal/compression-lean");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "{stdout}{stderr}");
    assert_eq!(stdout.trim(), format!("rustCheck: {} cases agree", cases.len()));
}
