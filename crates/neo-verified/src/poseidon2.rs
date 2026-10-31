// Provenance: the shape follows dcSpark/ProtogaLattice at commit 96b57a1c801b,
// `crates/pgl-hash/src/poseidon2.rs` (width 16, `x^7`, 4 + 22 + 4 rounds, the internal diagonal),
// Apache-2.0. Its round constants are replaced by Nightstream's, its external layer by Plonky3's
// `MDSMat4` with additions only, and its iterators and S-box closure by index loops.

//! The workspace Poseidon2 permutation over `Gl`: width 16, S-box `x^7`, 4 full, 22 partial and
//! 4 full rounds, after one external layer.
//!
//! The constants are those of `neo_ccs::crypto::poseidon2_goldilocks::round_constants()`, which
//! Plonky3's `new_from_rng_128` draws from a fixed ChaCha8 seed, in the rows of
//! `formal/nightstream-fprime/NightstreamFPrime/Spec/Poseidon2.lean`. The external layer is
//! Plonky3's `circ(2 M_4, M_4, M_4, M_4)` with `M_4 = circ(2, 3, 1, 1)`, and the internal layer is
//! `J + diag(DIAGONAL)`. `formal/compression-lean/CompressionLeanRust/Poseidon2.lean` proves
//! `permute` equal to that file's `permute` and each table equal to its rows.

use crate::circuit::WIDTH;
use crate::field::{gl_add, gl_mul, reduce, Gl, GL_ZERO};

/// Full rounds before the partial rounds, and after them.
pub const HALF_FULL_ROUNDS: usize = 4;
pub const PARTIAL_ROUNDS: usize = 22;

/// The constants of the first full rounds, one row per round.
pub const INITIAL: [[u64; WIDTH]; HALF_FULL_ROUNDS] = [
    [
        15504881536434223753,
        2212164856944708396,
        1885257220781225929,
        17531637481572944510,
        16769640728293682348,
        445908668462176974,
        1308472042479836079,
        17465001500823438575,
        1922033642430128704,
        2657514617275794404,
        17238706657248448792,
        7348277157222259646,
        10777112892842897939,
        1771261721914735482,
        9409693344407549465,
        16619731096074499912,
    ],
    [
        1922036059108268922,
        2681686362645798986,
        12432722052283819565,
        2826979200512189741,
        5080805286413226676,
        16827966425431695029,
        9196241087337510154,
        2350771591198563053,
        2989012136977041732,
        4359939046747977080,
        16089932437481530267,
        6601984573273403484,
        13005272261058756234,
        17128237926164276121,
        8240789415616872849,
        8676316357341090631,
    ],
    [
        16452552554259143025,
        17874550554210084887,
        3031715677034868367,
        18215520516675091549,
        18186005068527139405,
        11138995707668647102,
        15098195648006184282,
        2025927025270509469,
        9957669227203243937,
        11554336633716867616,
        9729067570563846225,
        4239770196713589268,
        4390607796152185292,
        17647511975646925721,
        7671337049037340193,
        4209452938403606590,
    ],
    [
        6593973666654839090,
        8390781086037206386,
        7324343054784993307,
        17780748563735894140,
        15974082699116886783,
        13213371256836887512,
        7312926934405385057,
        10393853239698468203,
        2710107888698774842,
        2801523468128575786,
        15894340394120906162,
        13510783799941644149,
        7917164295139071913,
        13839801071899888959,
        6672989303670154677,
        4519956214037211385,
    ],
];

/// The constant of each partial round, added to lane 0.
pub const PARTIAL: [u64; PARTIAL_ROUNDS] = [
    3785078240232899765,
    13753534232687663059,
    15164579152346391244,
    7188840087678967607,
    14466302407008998516,
    13002203018489042439,
    6216879054540921071,
    8639253411382064186,
    4983986534873630152,
    2915380551046574730,
    7837744426713061492,
    10524728973359902812,
    9778451922853050075,
    13310295482970625397,
    10551647698332907197,
    8509293821221348719,
    4319821126274335035,
    1417436523646501240,
    447570613036226838,
    16267896682506996530,
    14710998805405468930,
    3269000544927978862,
];

/// The constants of the last full rounds, one row per round.
pub const TERMINAL: [[u64; WIDTH]; HALF_FULL_ROUNDS] = [
    [
        7482194551502142718,
        3471957803411196592,
        8846669050136897522,
        4431017908497072775,
        14382646627736292998,
        15636596632746594248,
        14521990061611210983,
        4351091752509404379,
        14119848206371842921,
        528205008764728916,
        15379406877060454284,
        13572057177474709483,
        780214424511389757,
        10591233664360718633,
        1849508423779478786,
        7345390174439848870,
    ],
    [
        14580881241235634775,
        8777273265976228774,
        1758781345554053863,
        9701442189086298420,
        15685565327448534444,
        5672331717709479627,
        7675233227955155107,
        8852669876726984824,
        1218164705289579190,
        13224810758441726241,
        557024023478380004,
        3923346290699247117,
        4196774554581694822,
        16262909137268628555,
        6531098975686849205,
        538070144030448988,
    ],
    [
        16157559630818414765,
        3330859574359708906,
        13312877616183741059,
        15699706004066187344,
        2181468677625794151,
        12293285838251430515,
        17109377825740910727,
        11746958123598489878,
        7654965179475073269,
        15178922343313110770,
        14240408894833620294,
        4224192993509995210,
        13093043512401634422,
        16636225261759530156,
        13489384640167770266,
        8105602514957866176,
    ],
    [
        13910460326211973254,
        13010277363854955001,
        8570865802160232388,
        14830753997593808291,
        16178721091989175194,
        10926358020058189153,
        8413180834413067310,
        1124528750616792490,
        16054199595598201491,
        729673474029476808,
        1545919217216143455,
        15484244716222357662,
        9149791094276206087,
        3342519128264714984,
        14246551315881547762,
        6145097356981870399,
    ],
];

/// Plonky3's `MATRIX_DIAG_16_GOLDILOCKS`: the internal layer is `J + diag(DIAGONAL)`.
pub const DIAGONAL: [u64; WIDTH] = [
    18446744069414584319,
    1,
    2,
    9223372034707292161,
    3,
    4,
    9223372034707292160,
    18446744069414584318,
    18446744069414584317,
    16140901060737761281,
    17293822565076172801,
    17870283317245378561,
    2305843008676823040,
    1152921504338411520,
    576460752169205760,
    18446744065119617026,
];

/// The permutation of `input`.
pub fn permute(input: [Gl; WIDTH]) -> [Gl; WIDTH] {
    let mut state = full_rounds(external_layer(input), &INITIAL);
    let mut round = 0;
    while round < PARTIAL_ROUNDS {
        state = partial_round(state, PARTIAL[round]);
        round = round.wrapping_add(1);
    }
    full_rounds(state, &TERMINAL)
}

/// One full round for each row of `rows`.
fn full_rounds(mut state: [Gl; WIDTH], rows: &[[u64; WIDTH]; HALF_FULL_ROUNDS]) -> [Gl; WIDTH] {
    let mut round = 0;
    while round < HALF_FULL_ROUNDS {
        state = full_round(state, rows[round]);
        round = round.wrapping_add(1);
    }
    state
}

/// `x^7` with four multiplications.
fn sbox(x: Gl) -> Gl {
    let x2 = gl_mul(x, x);
    let x4 = gl_mul(x2, x2);
    gl_mul(gl_mul(x4, x2), x)
}

/// Each lane plus its constant, through the S-box, then the external layer.
fn full_round(state: [Gl; WIDTH], constants: [u64; WIDTH]) -> [Gl; WIDTH] {
    let mut boxed = state;
    let mut lane = 0;
    while lane < WIDTH {
        boxed[lane] = sbox(gl_add(state[lane], reduce(constants[lane])));
        lane = lane.wrapping_add(1);
    }
    external_layer(boxed)
}

/// Lane 0 plus the constant, through the S-box, then the internal layer.
fn partial_round(state: [Gl; WIDTH], constant: u64) -> [Gl; WIDTH] {
    let mut boxed = state;
    boxed[0] = sbox(gl_add(state[0], reduce(constant)));
    internal_layer(boxed)
}

/// `M_4` on each block of four lanes, then each lane plus the sum of the four lanes at its
/// position in the blocks. `M_4` is Plonky3's `apply_mat4`: eight additions and two doublings.
fn external_layer(state: [Gl; WIDTH]) -> [Gl; WIDTH] {
    let mut mixed = state;
    let mut sum0 = GL_ZERO;
    let mut sum1 = GL_ZERO;
    let mut sum2 = GL_ZERO;
    let mut sum3 = GL_ZERO;
    let mut at = 0;
    while at < WIDTH {
        let x0 = state[at];
        let x1 = state[at.wrapping_add(1)];
        let x2 = state[at.wrapping_add(2)];
        let x3 = state[at.wrapping_add(3)];
        let t01 = gl_add(x0, x1);
        let t23 = gl_add(x2, x3);
        let t0123 = gl_add(t01, t23);
        let t01123 = gl_add(t0123, x1);
        let t01233 = gl_add(t0123, x3);
        let y0 = gl_add(t01123, t01);
        let y1 = gl_add(t01123, gl_add(x2, x2));
        let y2 = gl_add(t01233, t23);
        let y3 = gl_add(t01233, gl_add(x0, x0));
        mixed[at] = y0;
        mixed[at.wrapping_add(1)] = y1;
        mixed[at.wrapping_add(2)] = y2;
        mixed[at.wrapping_add(3)] = y3;
        sum0 = gl_add(sum0, y0);
        sum1 = gl_add(sum1, y1);
        sum2 = gl_add(sum2, y2);
        sum3 = gl_add(sum3, y3);
        at = at.wrapping_add(4);
    }
    let mut out = mixed;
    let mut at = 0;
    while at < WIDTH {
        out[at] = gl_add(mixed[at], sum0);
        out[at.wrapping_add(1)] = gl_add(mixed[at.wrapping_add(1)], sum1);
        out[at.wrapping_add(2)] = gl_add(mixed[at.wrapping_add(2)], sum2);
        out[at.wrapping_add(3)] = gl_add(mixed[at.wrapping_add(3)], sum3);
        at = at.wrapping_add(4);
    }
    out
}

/// `J + diag(DIAGONAL)`: each lane times its diagonal entry, plus the sum of the lanes.
fn internal_layer(state: [Gl; WIDTH]) -> [Gl; WIDTH] {
    let mut sum = GL_ZERO;
    let mut lane = 0;
    while lane < WIDTH {
        sum = gl_add(sum, state[lane]);
        lane = lane.wrapping_add(1);
    }
    let mut out = state;
    let mut lane = 0;
    while lane < WIDTH {
        out[lane] = gl_add(gl_mul(reduce(DIAGONAL[lane]), state[lane]), sum);
        lane = lane.wrapping_add(1);
    }
    out
}
