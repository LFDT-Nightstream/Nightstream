//! The circuit of the general register row (scratch prototype): one
//! synthesis that records the rank-one rows and fills the honest witness in
//! the same pass, so the two cannot drift. `RowInput` (from `cpu.rs`) gives
//! the values the row does not compute; every other column is a function
//! of earlier columns. The shared blocks are X and Y (64 bits each) and
//! their products P = X ⊙ Y: the adder writes its sum to X, the bitwise
//! families read A and B through X and Y, the multiplier writes A₀·M to X
//! and A₁·M to Y, and a memory row reads its word through Y.

use nightstream::{Goldilocks as F, PrimeCharacteristicRing};
use p3_field::{Field, PrimeField64};

use super::cpu::RowInput;
use super::lower::{Fam, FAMILIES, K};

pub type Terms = Vec<(usize, F)>;
pub const ONE: usize = 0;
/// Address bits of the linear-memory region (words).
pub const LIN_WIDTH: usize = 9;

#[derive(Clone, Debug, Default)]
pub struct Lc(pub Terms);

pub fn v(c: usize) -> Lc {
    Lc(vec![(c, F::ONE)])
}

pub fn k(x: u64) -> Lc {
    Lc(vec![(ONE, F::from_u64(x))])
}

impl std::ops::Add for Lc {
    type Output = Lc;
    fn add(mut self, o: Lc) -> Lc {
        self.0.extend(o.0);
        self
    }
}

impl std::ops::Sub for Lc {
    type Output = Lc;
    fn sub(mut self, o: Lc) -> Lc {
        self.0.extend(o.0.into_iter().map(|(c, x)| (c, -x)));
        self
    }
}

impl std::ops::Mul<F> for Lc {
    type Output = Lc;
    fn mul(mut self, s: F) -> Lc {
        for t in &mut self.0 {
            t.1 *= s;
        }
        self
    }
}

fn f(x: u64) -> F {
    F::from_u64(x)
}

fn two(e: usize) -> F {
    F::TWO.exp_u64(e as u64)
}

/// `Σ 2^i · bits[i]`.
fn chunk(bits: &[usize]) -> Lc {
    Lc(bits.iter().enumerate().map(|(i, &c)| (c, two(i))).collect())
}

/// The rows and the witness of one register row.
pub struct Cs {
    pub rows: Vec<[Terms; 3]>,
    pub z: Vec<F>,
    /// Check each row as it is added (honest synthesis).
    pub check: bool,
    pub tags: Vec<(&'static str, usize)>,
    /// The word bits are the code slot's `v_r` bits (`R1csOptions::slot_bits`):
    /// row O1 forces them and the binding row packs them to the word, so the
    /// relation has no bit row and no packing row for them.
    pub slot_word: bool,
}

impl Cs {
    pub fn new(check: bool) -> Self {
        Self {
            rows: Vec::new(),
            z: vec![F::ONE],
            check,
            tags: Vec::new(),
            slot_word: false,
        }
    }

    pub fn tag(&mut self, name: &'static str) {
        self.tags.push((name, self.rows.len()));
    }

    pub fn val(&self, lc: &Lc) -> F {
        lc.0.iter().fold(F::ZERO, |s, (c, x)| s + *x * self.z[*c])
    }

    pub fn col(&mut self, x: F) -> usize {
        self.z.push(x);
        self.z.len() - 1
    }

    pub fn row(&mut self, a: Lc, b: Lc, c: Lc) {
        if self.check {
            assert_eq!(
                self.val(&a) * self.val(&b),
                self.val(&c),
                "row {} of the honest witness ({:?})",
                self.rows.len(),
                self.tags.last()
            );
        }
        self.rows.push([a.0, b.0, c.0]);
    }

    pub fn bit(&mut self, b: bool) -> usize {
        let x = self.col(F::from_bool(b));
        self.row(v(x), v(x) - k(1), Lc::default());
        x
    }

    pub fn bits(&mut self, value: u64, n: usize) -> Vec<usize> {
        (0..n).map(|i| self.bit(value >> i & 1 == 1)).collect()
    }

    pub fn mul(&mut self, a: Lc, b: Lc) -> usize {
        let x = self.val(&a) * self.val(&b);
        let c = self.col(x);
        self.row(a, b, v(c));
        c
    }

    /// A column `x = base + a·b`, with the one row `a·b = x − base`.
    pub fn mul_add(&mut self, a: Lc, b: Lc, base: Lc) -> usize {
        let x = self.val(&base) + self.val(&a) * self.val(&b);
        let c = self.col(x);
        self.row(a, b, v(c) - base);
        c
    }

    pub fn zero(&mut self, a: Lc) {
        self.row(a, k(1), Lc::default());
    }

    pub fn gate(&mut self, g: Lc, e: Lc) {
        self.row(g, e, Lc::default());
    }

    /// A column equal to `a` (one linear row).
    pub fn lin(&mut self, a: Lc) -> usize {
        let x = self.val(&a);
        let c = self.col(x);
        self.zero(a - v(c));
        c
    }

    /// One-hot bits `s` of `count` values with `Σ s = 1` and `Σ i·s_i = index`.
    pub fn one_hot(&mut self, index: Lc, count: usize) -> Vec<usize> {
        let at = self.val(&index).as_canonical_u64() as usize;
        let s: Vec<usize> = (0..count).map(|i| self.bit(i == at)).collect();
        self.zero(Lc(s.iter().map(|&c| (c, F::ONE)).collect()) - k(1));
        self.zero(
            Lc(s.iter()
                .enumerate()
                .map(|(i, &c)| (c, f(i as u64)))
                .collect())
                - index,
        );
        s
    }
}

/// The columns that the catalog, the continuity links, and the tests name.
#[derive(Clone, Debug, Default)]
pub struct Layout {
    pub pc: usize,
    pub fp: usize,
    pub flag: usize,
    pub regs: Vec<[usize; 2]>,
    pub word: usize,
    pub wb: Vec<usize>,
    pub immw: usize,
    pub old: usize,
    pub new: usize,
    pub laddr: usize,
    pub faddr: usize,
    pub gaddr: usize,
    pub taddr: usize,
    pub memrow: usize,
    pub linact: usize,
    pub frameact: usize,
    pub globact: usize,
    pub tabact: usize,
    pub pc1: usize,
    pub fp1: usize,
    pub flag1: usize,
    pub regs1: Vec<[usize; 2]>,
    pub res: [usize; 2],
    pub x: Vec<usize>,
    pub y: Vec<usize>,
    pub carry: [usize; 2],
    pub av: [usize; 2],
    pub fam: Vec<usize>,
    pub width: usize,
}

fn lo(x: u64) -> F {
    f(x & 0xffff_ffff)
}

fn hi(x: u64) -> F {
    f(x >> 32)
}

/// The honest values of the X and Y blocks and a few other private
/// columns, as the row's semantics fix them.
pub struct Private {
    pub x: u64,
    pub y: u64,
    pub c: [bool; 2],
    pub s: u64,
}

pub fn private(input: &RowInput) -> Private {
    let word = input.word;
    let bit = |i: u32| word >> i & 1 == 1;
    let fam = Fam::from((word & 31) as u8);
    let (w64, uimm) = (bit(5), bit(6));
    let m = [bit(7), bit(8), bit(9), bit(10)];
    let (ai, bi) = ((word >> 11 & 15) as usize, (word >> 15 & 15) as usize);
    let simm = u64::from(word >> 23);
    let a = input.regs[ai];
    let breg = input.regs[bi];
    let imm = u64::from(input.immw) | if simm & 1 == 1 { 0xffff_ffff << 32 } else { 0 };
    let b = if uimm {
        if fam == Fam::Br {
            0
        } else {
            imm
        }
    } else {
        breg
    };
    let (a0, a1, b0) = (a & 0xffff_ffff, a >> 32, b & 0xffff_ffff);
    let mut p = Private {
        x: 0,
        y: 0,
        c: [false; 2],
        s: 0,
    };
    let lin = !m[0] && !m[1];
    match fam {
        Fam::Add | Fam::Cmp | Fam::Br | Fam::Assert => {
            let x = if fam == Fam::Add && m[1] { 0 } else { a };
            let y = if fam == Fam::Add && m[2] { b0 << 32 } else { b };
            let sub = fam != Fam::Add || m[0];
            let (x0, x1, y0, y1) = (x & 0xffff_ffff, x >> 32, y & 0xffff_ffff, y >> 32);
            let (l, c0) = if sub {
                (x0.wrapping_sub(y0) & 0xffff_ffff, x0 < y0)
            } else {
                ((x0 + y0) & 0xffff_ffff, x0 + y0 >> 32 == 1)
            };
            let (h, c1) = if sub {
                let t = x1 as i64 - y1 as i64 - c0 as i64;
                ((t as u64) & 0xffff_ffff, t < 0)
            } else {
                let t = x1 + y1 + u64::from(c0);
                (t & 0xffff_ffff, t >> 32 == 1)
            };
            p.x = l | h << 32;
            p.c = [c0, c1 && w64];
            if fam != Fam::Add {
                let top = |v: u64| if w64 { v >> 32 } else { v & 0xffff_ffff };
                p.y = top(a) | top(b) << 32;
            }
        }
        Fam::And | Fam::Or | Fam::Xor => {
            p.x = a;
            p.y = b;
        }
        Fam::Ext => p.x = a,
        Fam::Mul => {
            let kind = u8::from(m[0]) + 2 * u8::from(m[1]) + 4 * u8::from(m[2]);
            let s = b0 & 63;
            p.s = if kind >= 3 { b0 } else { 0 };
            let mm = match kind {
                3 | 5 => 1u64 << (s % 32),
                4 | 6 => 1u64 << (32 - s % 32),
                _ => b0,
            };
            p.x = (a0 as u128 * mm as u128) as u64;
            p.y = if w64 { (a1 as u128 * mm as u128) as u64 } else { 0 };
        }
        Fam::Load | Fam::Store if lin => {
            let ea = a0 + simm;
            p.x = ea & 0xffff_ffff;
            p.c = [ea >> 32 == 1, false];
            let sv = if m[2] { breg >> 32 } else { breg & 0xffff_ffff };
            p.y = u64::from(input.old) | if fam == Fam::Store { sv << 32 } else { 0 };
        }
        Fam::Ret => p.x = u64::from(input.old),
        _ => {}
    }
    p
}

/// Synthesize one register row on `input` (zero for the shape only).
pub fn synthesize(cs: &mut Cs, input: &RowInput) -> Layout {
    synthesize_tampered(cs, input, &|_| {})
}

/// `synthesize` with the shared-bit values that `tamper` picks: an
/// adversary's witness in which every other column is consistent.
pub fn synthesize_tampered(cs: &mut Cs, input: &RowInput, tamper: &dyn Fn(&mut Private)) -> Layout {
    let mut l = Layout::default();
    let mut pv = private(input);
    tamper(&mut pv);
    let m32 = two(32);
    cs.tag("state");
    l.pc = cs.col(f(u64::from(input.pc)));
    l.fp = cs.col(f(u64::from(input.fp)));
    l.flag = cs.col(F::from_bool(input.flag));
    l.regs = input
        .regs
        .iter()
        .map(|&r| [cs.col(lo(r)), cs.col(hi(r))])
        .collect();

    cs.tag("program word");
    l.word = cs.col(f(u64::from(input.word)));
    let wb = if cs.slot_word {
        (0..32)
            .map(|i| cs.col(F::from_bool(input.word >> i & 1 == 1)))
            .collect()
    } else {
        let wb = cs.bits(u64::from(input.word), 32);
        cs.zero(chunk(&wb) - v(l.word));
        wb
    };
    l.wb = wb.clone();
    let field = |from: usize, n: usize| chunk(&wb[from..from + n]);
    let (w64, uimm) = (wb[5], wb[6]);
    let m = [wb[7], wb[8], wb[9], wb[10]];
    let simm = field(23, 9);
    let neg = wb[23];
    cs.tag("family");
    let fam = cs.one_hot(field(0, 5), FAMILIES);
    l.fam = fam.clone();
    let fl = |x: Fam| fam[x as usize];
    cs.tag("register selectors");
    let sa = cs.one_hot(field(11, 4), K);
    let sb = cs.one_hot(field(15, 4), K);
    let sd = cs.one_hot(field(19, 4), K);

    cs.tag("operands");
    let ai = (input.word >> 11 & 15) as usize;
    let bi = (input.word >> 15 & 15) as usize;
    let av = [cs.col(lo(input.regs[ai])), cs.col(hi(input.regs[ai]))];
    let brv = [cs.col(lo(input.regs[bi])), cs.col(hi(input.regs[bi]))];
    l.av = av;
    for kk in 0..K {
        for i in 0..2 {
            cs.gate(v(sa[kk]), v(av[i]) - v(l.regs[kk][i]));
            cs.gate(v(sb[kk]), v(brv[i]) - v(l.regs[kk][i]));
        }
    }
    l.immw = cs.col(f(u64::from(input.immw)));
    let br_imm0 = cs.mul(v(fl(Fam::Br)), v(l.immw));
    let br_neg = cs.mul(v(fl(Fam::Br)), v(neg));
    let immb = [v(l.immw) - v(br_imm0), (v(neg) - v(br_neg)) * (m32 - F::ONE)];
    let bv: Vec<usize> = (0..2)
        .map(|i| cs.mul_add(v(uimm), immb[i].clone() - v(brv[i]), v(brv[i])))
        .collect();

    cs.tag("shared bits");
    l.x = cs.bits(pv.x, 64);
    l.y = cs.bits(pv.y, 64);
    let (x0, x1) = (chunk(&l.x[..32]), chunk(&l.x[32..]));
    let (y0, y1) = (chunk(&l.y[..32]), chunk(&l.y[32..]));
    let p: Vec<usize> = (0..64).map(|i| cs.mul(v(l.x[i]), v(l.y[i]))).collect();
    let (p0, p1) = (chunk(&p[..32]), chunk(&p[32..]));
    let xw1 = cs.mul(v(w64), x1.clone());
    let yw1 = cs.mul(v(w64), y1.clone());
    let pw1 = cs.mul(v(w64), p1.clone());

    cs.tag("adder");
    let fload_store = v(fl(Fam::Load)) + v(fl(Fam::Store));
    let lm = cs.mul(fload_store.clone(), k(1) - v(m[0]) - v(m[1]));
    let noa = cs.mul(v(fl(Fam::Add)), v(m[1]));
    let hiroute = cs.mul(v(fl(Fam::Add)), v(m[2]));
    let subadd = cs.mul(v(fl(Fam::Add)), v(m[0]));
    let cmpish = v(fl(Fam::Cmp)) + v(fl(Fam::Br)) + v(fl(Fam::Assert));
    let sub = v(subadd) + cmpish.clone();
    let sgn = k(1) - sub * F::TWO;
    let xa: Vec<Lc> = (0..2)
        .map(|i| {
            let t = cs.mul(v(noa), v(av[i]));
            v(av[i]) - v(t)
        })
        .collect();
    let hb0 = cs.mul(v(hiroute), v(bv[0]));
    let lmb0 = cs.mul(v(lm), simm.clone() - v(bv[0]));
    let ya0 = v(bv[0]) - v(hb0) + v(lmb0);
    let hb1 = cs.mul(v(hiroute), v(bv[0]) - v(bv[1]));
    let lmb1 = cs.mul(v(lm), v(bv[1]));
    let ya1 = v(bv[1]) + v(hb1) - v(lmb1);
    l.carry = [cs.bit(pv.c[0]), cs.bit(pv.c[1])];
    let sy0 = cs.mul(sgn.clone(), ya0);
    let sc0 = cs.mul(sgn.clone(), v(l.carry[0]));
    let sy1 = cs.mul(sgn.clone(), ya1);
    let sc1 = cs.mul(sgn.clone(), v(l.carry[1]));
    let adder = v(fl(Fam::Add)) + cmpish.clone() + v(lm);
    cs.gate(adder, xa[0].clone() + v(sy0) - x0.clone() - v(sc0) * m32);
    let high = cs.mul(v(w64), v(fl(Fam::Add)) + cmpish.clone());
    cs.gate(v(high), xa[1].clone() + v(sy1) + v(sc0) - x1.clone() - v(sc1) * m32);
    let hi_sum = cs.mul(v(w64), Lc(l.x[32..].iter().map(|&c| (c, F::ONE)).collect()));
    let zsum = Lc(l.x[..32].iter().map(|&c| (c, F::ONE)).collect()) + v(hi_sum);
    let wc = cs.mul(v(w64), v(l.carry[1]) - v(l.carry[0]));
    let ltu = v(l.carry[0]) + v(wc);
    cs.gate(
        v(lm),
        v(l.carry[0])
            + Lc(l.x[2 + LIN_WIDTH..32]
                .iter()
                .map(|&c| (c, F::ONE))
                .collect()),
    );

    cs.tag("compare");
    let ta = cs.mul(v(w64), v(av[1]) - v(av[0]));
    let tb = cs.mul(v(w64), v(bv[1]) - v(bv[0]));
    cs.gate(cmpish.clone(), y0.clone() - v(av[0]) - v(ta));
    cs.gate(cmpish.clone(), y1.clone() - v(bv[0]) - v(tb));
    let (sa_bit, sb_bit) = (l.y[31], l.y[63]);
    let sab = cs.mul(v(sa_bit), v(sb_bit));
    let xs = v(sa_bit) + v(sb_bit) - v(sab) * F::TWO;
    let ltx = cs.mul(ltu.clone(), xs.clone());
    let lts = ltu.clone() + xs - v(ltx) * F::TWO;
    let sl = cs.mul(v(m[0]), lts - ltu.clone());
    let lt = ltu + v(sl);
    let zv = cs.val(&zsum);
    let inv = cs.col(zv.try_inverse().unwrap_or(F::ZERO));
    let eq = cs.col(F::from_bool(zv == F::ZERO));
    cs.row(zsum.clone(), v(inv), k(1) - v(eq));
    cs.row(zsum.clone(), v(eq), Lc::default());
    let t1 = cs.mul(v(m[2]), lt);
    let t2 = cs.mul(v(m[1]), v(eq));
    let t = v(t1) + v(t2);
    let nt = cs.mul(v(m[3]), k(1) - t.clone() * F::TWO);
    let cond = t + v(nt);
    let condc = cs.lin(cond);
    l.flag1 = cs.mul_add(v(fl(Fam::Cmp)), v(condc) - v(l.flag), v(l.flag));
    let taken = cs.mul(v(fl(Fam::Br)), v(condc));
    cs.gate(v(fl(Fam::Assert)), k(1) - v(condc));

    cs.tag("bitwise");
    let bw = v(fl(Fam::And)) + v(fl(Fam::Or)) + v(fl(Fam::Xor));
    let xread = bw.clone() + v(fl(Fam::Ext));
    cs.gate(xread.clone(), x0.clone() - v(av[0]));
    cs.gate(xread, x1.clone() - v(av[1]));
    cs.gate(bw.clone(), y0.clone() - v(bv[0]));
    cs.gate(bw, y1.clone() - v(bv[1]));

    cs.tag("multiplier");
    let m01 = cs.mul(v(m[0]), v(m[1]));
    let m02 = cs.mul(v(m[0]), v(m[2]));
    let m12 = cs.mul(v(m[1]), v(m[2]));
    let m012 = cs.mul(v(m01), v(m[2]));
    let kinds: [Lc; 7] = [
        k(1) - v(m[0]) - v(m[1]) - v(m[2]) + v(m01) + v(m02) + v(m12) - v(m012),
        v(m[0]) - v(m01) - v(m02) + v(m012),
        v(m[1]) - v(m01) - v(m12) + v(m012),
        v(m01) - v(m012),
        v(m[2]) - v(m02) - v(m12) + v(m012),
        v(m02) - v(m012),
        v(m12) - v(m012),
    ];
    let gk: Vec<usize> = kinds
        .iter()
        .map(|kl| cs.mul(v(fl(Fam::Mul)), kl.clone()))
        .collect();
    let shifting = v(gk[3]) + v(gk[4]) + v(gk[5]) + v(gk[6]);
    let s = cs.bits(pv.s, 6);
    cs.gate(shifting, v(bv[0]) - chunk(&s));
    let mut pow = k(1);
    let mut ipow = k(1);
    for (j, &sj) in s.iter().take(5).enumerate() {
        let up = two(1 << j) - F::ONE;
        let down = two(1 << j).inverse() - F::ONE;
        pow = v(cs.mul(pow, k(1) + v(sj) * up));
        ipow = v(cs.mul(ipow, k(1) + v(sj) * down));
    }
    let ipow = ipow * m32;
    let up = kinds[3].clone() + kinds[5].clone();
    let down = kinds[4].clone() + kinds[6].clone();
    let mu = cs.mul(up, pow - v(bv[0]));
    let md = cs.mul(down, ipow - v(bv[0]));
    let mm = v(bv[0]) + v(mu) + v(md);
    let pa = cs.mul(v(av[0]), mm.clone());
    let pb = cs.mul(v(av[1]), mm);
    cs.gate(v(fl(Fam::Mul)), x0.clone() + x1.clone() * m32 - v(pa));
    let ya = cs.mul(v(fl(Fam::Mul)), v(w64));
    cs.gate(v(ya), y0.clone() + y1.clone() * m32 - v(pb));
    for (gate, (h, lo_bits)) in [(fl(Fam::Mul), (x1.clone(), &l.x[..32])), (ya, (y1.clone(), &l.y[..32]))] {
        let e = h - k((1 << 32) - 1);
        let ev = cs.val(&e);
        let ie = cs.col(ev.try_inverse().unwrap_or(F::ZERO));
        let ze = cs.col(F::from_bool(ev == F::ZERO));
        cs.row(e.clone(), v(ie), k(1) - v(ze));
        cs.row(e, v(ze), Lc::default());
        let g = cs.mul(v(gate), v(ze));
        cs.gate(v(g), chunk(lo_bits));
    }
    let big = s[5];
    let u = x1.clone() + y0.clone();
    let vv = x0.clone() + y1.clone();
    let bx0 = cs.mul(v(big), x0.clone());
    let bxu = cs.mul(v(big), x0.clone() - u.clone());
    let shl = (x0.clone() - v(bx0), u.clone() + v(bxu));
    let byu = cs.mul(v(big), y1.clone() - u.clone());
    let by1 = cs.mul(v(big), y1.clone());
    let shr = (u.clone() + v(byu), y1.clone() - v(by1));
    let buv = cs.mul(v(big), u.clone() - vv.clone());
    let rotl_lo = vv.clone() + v(buv);
    let rotl = (rotl_lo.clone(), u.clone() + vv.clone() - rotl_lo.clone());
    let rotr = (u.clone() + vv.clone() - rotl_lo.clone(), rotl_lo);
    let rot32 = x0.clone() + x1.clone();
    let mut mul_vals: Vec<(Lc, Lc)> = vec![(x0.clone(), k(0)), (x0.clone(), x1.clone()), (y0.clone(), k(0))];
    for (w32, w64v) in [
        (x0.clone(), shl),
        (x1.clone(), shr),
        (rot32.clone(), rotl),
        (rot32, rotr),
    ] {
        let dl = cs.mul(v(w64), w64v.0 - w32.clone());
        let dh = cs.mul(v(w64), w64v.1);
        mul_vals.push((w32 + v(dl), v(dh)));
    }

    cs.tag("extend");
    let sign = {
        let s8 = cs.mul(v(m[0]), v(l.x[7]));
        let s16 = cs.mul(v(m[1]), v(l.x[15]));
        let s32 = cs.mul(v(m[2]), v(l.x[31]));
        v(s8) + v(s16) + v(s32)
    };
    let low8 = chunk(&l.x[..8]);
    let low16 = chunk(&l.x[..16]);
    let e8 = cs.mul(v(m[0]), low8 + v(l.x[7]) * (m32 - two(8)));
    let e16 = cs.mul(v(m[1]), low16 + v(l.x[15]) * (m32 - two(16)));
    let e32 = cs.mul(v(m[2]), v(av[0]));
    let pop = cs.mul(v(m[3]), zsum.clone());
    let ext_hi = cs.mul(v(w64), sign * (m32 - F::ONE));

    cs.tag("select");
    let sel: Vec<usize> = (0..2)
        .map(|i| cs.mul(v(l.flag), v(av[i]) - v(bv[i])))
        .collect();

    cs.tag("memory");
    l.old = cs.col(f(u64::from(input.old)));
    let (lx0, lx1) = (l.x[0], l.x[1]);
    cs.gate(v(lm), y0.clone() - v(l.old));
    let sv = {
        let d = cs.mul(v(m[2]), v(brv[1]) - v(brv[0]));
        v(brv[0]) + v(d)
    };
    let lst = cs.mul(v(lm), v(fl(Fam::Store)));
    cs.gate(v(lst), y1.clone() - sv.clone());
    let um = cs.mul(v(uimm), v(m[3]));
    let w1 = v(uimm) - v(um);
    let w2 = v(m[3]) - v(um);
    let w4 = k(1) - v(uimm) - v(m[3]) + v(um);
    let a1 = cs.mul(w4.clone(), v(lx0));
    let a2 = cs.mul(w4.clone(), v(lx1));
    let a3 = cs.mul(w2.clone(), v(lx0));
    cs.gate(v(lm), v(a1) + v(a2) + v(a3));
    let xp = cs.mul(v(lx0), v(lx1));
    let byte = |j: usize| chunk(&l.y[8 * j..8 * j + 8]);
    let lo16 = chunk(&l.y[..16]);
    let hi16 = chunk(&l.y[16..32]);
    let v16 = {
        let d = cs.mul(v(lx1), hi16 - lo16.clone());
        lo16 + v(d)
    };
    let s16 = {
        let d = cs.mul(v(lx1), v(l.y[31]) - v(l.y[15]));
        v(l.y[15]) + v(d)
    };
    let hb_lo = {
        let d = cs.mul(v(lx1), byte(2) - byte(0));
        byte(0) + v(d)
    };
    let hb_hi = {
        let d = cs.mul(v(lx1), byte(3) - byte(1));
        byte(1) + v(d)
    };
    let v8 = {
        let d = cs.mul(v(lx0), hb_hi - hb_lo.clone());
        hb_lo + v(d)
    };
    let tlo = {
        let d = cs.mul(v(lx1), v(l.y[23]) - v(l.y[7]));
        v(l.y[7]) + v(d)
    };
    let s8 = {
        let d = cs.mul(v(lx0), s16.clone() - tlo.clone());
        tlo + v(d)
    };
    let loaded = {
        let d8 = cs.mul(w1.clone(), v8.clone() - v(l.old));
        let d16 = cs.mul(w2.clone(), v16.clone() - v(l.old));
        v(l.old) + v(d8) + v(d16)
    };
    let sbit = {
        let q1 = cs.mul(w1.clone(), s8);
        let q2 = cs.mul(w2.clone(), s16);
        let q4 = cs.mul(w4.clone(), v(l.y[31]));
        cs.mul(v(m[2]), v(q1) + v(q2) + v(q4))
    };
    let sw1 = cs.mul(v(sbit), w1.clone());
    let sw2 = cs.mul(v(sbit), w2.clone());
    let lin_lo = loaded + v(sw1) * (m32 - two(8)) + v(sw2) * (m32 - two(16));
    let lin_hi = cs.mul(v(w64), v(sbit) * (m32 - F::ONE));
    let fg_lo = {
        let d = cs.mul(v(uimm), v(av[0]) - v(l.old));
        v(l.old) + v(d)
    };
    let fg_hi = cs.mul(v(uimm), v(l.old));
    l.new = cs.col(F::ZERO);
    let sh8 = k(1) - v(lx0) - v(lx1) + v(xp) + (v(lx0) - v(xp)) * two(8) + (v(lx1) - v(xp)) * two(16) + v(xp) * two(24);
    let sh16 = k(1) + v(lx1) * (two(16) - F::ONE);
    let sv8 = chunk(&l.y[32..40]);
    let sv16 = chunk(&l.y[32..48]);
    let d8 = cs.mul(sv8 - v8, sh8);
    let d16 = cs.mul(sv16 - v16, sh16);
    let n1 = cs.mul(w1, v(d8));
    let n2 = cs.mul(w2, v(d16));
    let n4 = cs.mul(w4, sv.clone() - v(l.old));
    let lin_new = v(l.old) + v(n1) + v(n2) + v(n4);
    let gs_lin = cs.mul(v(fl(Fam::Store)), k(1) - v(m[0]) - v(m[1]));
    let gl_lin = cs.mul(v(fl(Fam::Load)), k(1) - v(m[0]) - v(m[1]));
    let header = v(l.pc) + k(1) + v(l.fp) * two(16);
    let new_value = {
        let fam_now = Fam::from((input.word & 31) as u8);
        match fam_now {
            Fam::Store if !(input.word >> 7 & 1 == 1 || input.word >> 8 & 1 == 1) => cs.val(&lin_new),
            Fam::Store => cs.val(&sv),
            Fam::Call => cs.val(&header),
            _ => cs.val(&v(l.old)),
        }
    };
    cs.z[l.new] = new_value;
    cs.gate(v(gs_lin), v(l.new) - lin_new);
    cs.gate(v(fl(Fam::Store)) - v(gs_lin), v(l.new) - sv);
    cs.gate(v(fl(Fam::Load)) + v(fl(Fam::Ret)), v(l.new) - v(l.old));
    cs.gate(v(fl(Fam::Call)), v(l.new) - header);
    l.laddr = cs.lin(Lc(l.x[2..2 + LIN_WIDTH]
        .iter()
        .enumerate()
        .map(|(i, &c)| (c, two(i)))
        .collect()));
    l.faddr =
        cs.lin(v(l.fp) * F::TWO + simm.clone() * F::TWO + v(uimm) - (v(fl(Fam::Call)) + v(fl(Fam::Ret))) * F::TWO);
    l.gaddr = cs.lin(simm.clone() * F::TWO + v(uimm));
    l.taddr = cs.lin(simm.clone() + v(av[0]));
    l.linact = lm;
    let fm = cs.mul(fload_store.clone(), v(m[0]));
    l.frameact = cs.lin(v(fm) + v(fl(Fam::Call)) + v(fl(Fam::Ret)));
    l.globact = cs.mul(fload_store, v(m[1]));
    l.tabact = fl(Fam::BrTable);
    l.memrow = cs.lin(v(l.linact) + v(l.frameact) + v(l.globact) + v(l.tabact));

    cs.tag("control");
    cs.gate(v(fl(Fam::Ret)), x0.clone() - v(l.old));
    let hpc = chunk(&l.x[..16]);
    let hfp = chunk(&l.x[16..32]);
    let next = v(l.pc) + k(1);
    let target = field(11, 8) + field(7, 4) * two(8);
    let j1 = cs.mul(v(taken), v(l.immw) - next.clone());
    let j2 = cs.mul(v(fl(Fam::Jump)), v(l.immw) - next.clone());
    let j3 = cs.mul(v(fl(Fam::Call)), target - next.clone());
    let j4 = cs.mul(v(fl(Fam::Ret)), hpc - next.clone());
    let j5 = cs.mul(v(fl(Fam::BrTable)), v(l.old) - next.clone());
    l.pc1 = cs.lin(next + v(j1) + v(j2) + v(j3) + v(j4) + v(j5) - v(fl(Fam::Halt)));
    let f1 = cs.mul(v(fl(Fam::Call)), simm);
    let f2 = cs.mul(v(fl(Fam::Ret)), hfp - v(l.fp));
    l.fp1 = cs.lin(v(l.fp) + v(f1) + v(f2));

    cs.tag("result");
    let fam_now = Fam::from((input.word & 31) as u8);
    let results: [Lc; 2] = {
        let mv = |i: u8| mul_vals[i as usize].clone();
        let kind = (input.word >> 7 & 7) as u8;
        match fam_now {
            Fam::Add | Fam::Advice => [x0.clone(), v(xw1)],
            Fam::Cmp => [v(condc), k(0)],
            Fam::And => [p0.clone(), v(pw1)],
            Fam::Or => [x0.clone() + y0.clone() - p0.clone(), v(xw1) + v(yw1) - v(pw1)],
            Fam::Xor => [
                x0.clone() + y0.clone() - p0.clone() * F::TWO,
                v(xw1) + v(yw1) - v(pw1) * F::TWO,
            ],
            Fam::Mul if kind < 7 => [mv(kind).0, mv(kind).1],
            Fam::Ext => [v(e8) + v(e16) + v(e32) + v(pop), v(ext_hi)],
            Fam::Select => [v(bv[0]) + v(sel[0]), v(bv[1]) + v(sel[1])],
            Fam::Load if input.word >> 7 & 3 == 0 => [lin_lo.clone(), v(lin_hi)],
            Fam::Load => [fg_lo.clone(), v(fg_hi)],
            _ => [k(0), k(0)],
        }
    };
    l.res = [cs.col(cs.val(&results[0])), cs.col(cs.val(&results[1]))];
    let r = |i: usize| v(l.res[i]);
    for (gate, vals) in [
        (v(fl(Fam::Add)) + v(fl(Fam::Advice)), [x0.clone(), v(xw1)]),
        (v(fl(Fam::Cmp)), [v(condc), k(0)]),
        (v(fl(Fam::And)), [p0.clone(), v(pw1)]),
        (
            v(fl(Fam::Or)),
            [x0.clone() + y0.clone() - p0.clone(), v(xw1) + v(yw1) - v(pw1)],
        ),
        (
            v(fl(Fam::Xor)),
            [x0.clone() + y0.clone() - p0 * F::TWO, v(xw1) + v(yw1) - v(pw1) * F::TWO],
        ),
        (v(fl(Fam::Ext)), [v(e8) + v(e16) + v(e32) + v(pop), v(ext_hi)]),
        (v(fl(Fam::Select)), [v(bv[0]) + v(sel[0]), v(bv[1]) + v(sel[1])]),
        (v(gl_lin), [lin_lo, v(lin_hi)]),
        (v(fl(Fam::Load)) - v(gl_lin), [fg_lo, v(fg_hi)]),
    ] {
        for i in 0..2 {
            cs.gate(gate.clone(), r(i) - vals[i].clone());
        }
    }
    for (g, vals) in gk.iter().zip(&mul_vals) {
        cs.gate(v(*g), r(0) - vals.0.clone());
        cs.gate(v(*g), r(1) - vals.1.clone());
    }
    let silent = [
        Fam::Halt,
        Fam::Br,
        Fam::Store,
        Fam::Jump,
        Fam::Call,
        Fam::Ret,
        Fam::BrTable,
        Fam::Host,
        Fam::Assert,
    ]
    .iter()
    .fold(Lc::default(), |acc, &x| acc + v(fl(x)));
    cs.gate(silent.clone(), r(0));
    cs.gate(silent, r(1));

    cs.tag("write");
    l.regs1 = (0..K)
        .map(|kk| {
            let mut out = [0; 2];
            for i in 0..2 {
                out[i] = cs.mul_add(v(sd[kk]), r(i) - v(l.regs[kk][i]), v(l.regs[kk][i]));
            }
            out
        })
        .collect();
    l.width = cs.z.len();
    l
}
