# ========================= Π-CCS (paper-faithful) from Rust dump =========================
# Reads target/round0_dump.json and evaluates Q exactly as in Section 4.4,
# with literal hypercube enumeration and explicit gates—no shortcuts.

import os, sys, json

# ------------------------- CLI / env --------------------------------------------
dump_path = sys.argv[1] if len(sys.argv) >= 2 else os.environ.get("NEO_ROUND0_DUMP", "target/round0_dump.json")
NONRES_ENV = os.environ.get("NEO_NONRES", "auto")
VERBOSE = int(os.environ.get("VERBOSE", "2"))  # 0=silent, 1=summary, 2=full per-X
# Comparison defaults: show side-by-side aggregates by default (first 16 rows)
COMPARE_VERBOSE = int(os.environ.get("COMPARE_VERBOSE", "1"))  # 0=off, 1=per-X aggregates
COMPARE_ROWS = int(os.environ.get("COMPARE_ROWS", "16"))  # limit rows printed when COMPARE_VERBOSE>0 (0 = all)
COMPARE_MASK_ENV = os.environ.get("COMPARE_MASK", None)  # print detailed side-by-side for one mask
def parse_mask(s):
    try:
        if s is None: return None
        ss = s.strip().lower()
        if ss.startswith('0b'):
            return int(ss, 2)
        if ss.startswith('0x'):
            return int(ss, 16)
        return int(ss)
    except Exception:
        return None
COMPARE_MASK = parse_mask(COMPARE_MASK_ENV)

# ------------------------- Load JSON --------------------------------------------
with open(dump_path, "r") as f:
    DUMP = json.load(f)

p_dec = int(DUMP["p"]) if isinstance(DUMP["p"], (int,)) else int(str(DUMP["p"]))
b     = int(DUMP["b"])
d     = int(DUMP["d"])
n     = int(DUMP["n"])
m     = int(DUMP["m"])
t     = int(DUMP["t"])
k_total = int(DUMP["k_total"])

# ------------------------- Fields ------------------------------------------------
p = Integer(p_dec)
F = GF(p)

def make_K_with_nonres(F, nonres):
    R.<X> = PolynomialRing(F)
    if nonres == "auto":
        # Find a small quadratic non-residue
        candidates = [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53]
        c = None
        for cand in candidates:
            if F(cand).is_square() is False:
                c = cand
                break
        if c is None:
            # Fallback: just build GF(p^2) abstractly
            K.<u> = GF(p^2)
            return K, u, None
        nonres = c
    else:
        nonres = int(nonres)
        # Ensure chosen nonres is actually non-square; if not, fall back to auto
        if F(nonres).is_square():
            return make_K_with_nonres(F, "auto")

    f = X^2 - F(nonres)
    if not f.is_irreducible():
        # If accidental residue, fall back to auto
        return make_K_with_nonres(F, "auto")
    K.<u> = F.extension(f)
    return K, u, nonres

# Optional quadratic rule from Rust dump: quad_rule.u2 = [a,b] meaning u^2 = a + b·u
quad = DUMP.get("quad_rule", None)
if quad is not None:
    a_u2 = F(Integer(quad["u2"][0]))
    b_u2 = F(Integer(quad["u2"][1]))
    R.<X> = PolynomialRing(F)
    poly = X^2 - b_u2*X - a_u2
    K.<u> = F.extension(poly)
    used_nonres = f"custom: u^2 = {a_u2} + {b_u2}·u"
else:
    K, u, used_nonres = make_K_with_nonres(F, NONRES_ENV)

def F_from_str(x): return F(Integer(x))
def K_from_pair(pair):  # pair = ["a","b"]
    a = F_from_str(pair[0]); b2 = F_from_str(pair[1])
    return K(a) + K(b2)*u

# ------------------------- Matrices & witnesses ---------------------------------
def mat_from_rows(rows_F_strings):
    rows = len(rows_F_strings); cols = len(rows_F_strings[0]) if rows else 0
    M_ = Matrix(F, rows, cols)
    for r in range(rows):
        for c in range(cols):
            M_[r,c] = F_from_str(rows_F_strings[r][c])
    return M_

Mats = [mat_from_rows(Mj) for Mj in DUMP["M"]]         # t items, n x m
Zs   = [mat_from_rows(Zi) for Zi in DUMP["Z"]]         # k_total items, d x m

# Challenges (K pairs)
alpha  = [K_from_pair(ab) for ab in DUMP["alpha"]]     # ℓ_d entries
beta_a = [K_from_pair(ab) for ab in DUMP["beta_a"]]    # ℓ_d entries
beta_r = [K_from_pair(ab) for ab in DUMP["beta_r"]]    # ℓ_n entries
gamma  = K_from_pair(DUMP["gamma"])
r_vec  = [K_from_pair(ab) for ab in DUMP["r"]]         # ℓ_n entries
beta   = beta_a + beta_r
ell_d, ell_n, ell = len(alpha), len(beta_r), len(beta_a) + len(beta_r)

# Optional f-spec (sparse polynomial) if you ever add it to the dump:
# { "f": {"arity": t, "terms":[{"coeff":"...", "exps":[e1,...,et]}, ...]} }
f_spec = DUMP.get("f", None)

# ------------------------- Recompose z_1 from Z_1 (Def. 11) ---------------------
def recompose_from_Z(Z, b):
    rows, cols = Z.nrows(), Z.ncols()
    # pow(b, rho) over F
    powb = [F(1)]
    for _ in range(1, rows):
        powb.append(powb[-1] * F(b))
    z = [F(0)]*cols
    for c in range(cols):
        s = F(0)
        for rho in range(rows):
            s += powb[rho] * Z[rho, c]
        z[c] = s
    return vector(F, z)

z1 = recompose_from_Z(Zs[0], b)  # only instance 1 enters F

# ------------------------- Paper helpers: eq, χ, MLEs ---------------------------
def eq_general(X_bits, y_vec):
    # eq(X,y) = Π_i ((1-Xi)(1-yi) + Xi*yi)
    acc = K(1)
    for Xi, yi in zip(X_bits, y_vec):
        acc *= (K(1) - Xi) * (K(1) - yi) + Xi * yi
    return acc

def chi_index(X_bits, idx, ell_bits):
    # Boolean character χ_idx(X_bits) = Π_j (X_j)^(bit) * (1-X_j)^(1-bit)
    acc = K(1)
    for j in range(ell_bits):
        bit = (idx >> j) & 1
        acc *= X_bits[j] if bit == 1 else (K(1) - X_bits[j])
    return acc

def mle_vector_at(w_vec_F, Xr_bits):
    # ˜w(X_r) = Σ_row w[row]*χ_row(X_r)
    s = K(0)
    for row in range(n):
        s += K(w_vec_F[row]) * chi_index(Xr_bits, row, ell_n)
    return s

def mle_matrix_at(Y_F, Xa_bits, Xr_bits):
    # ˜Y(X) = Σ_{rho,row} Y[rho,row] χ_rho(X_a) χ_row(X_r)
    s = K(0)
    for rho in range(d):
        chi_a = chi_index(Xa_bits, rho, ell_d)
        for row in range(n):
            s += K(Y_F[rho, row]) * chi_a * chi_index(Xr_bits, row, ell_n)
    return s

def range_product(value_K, b):
    acc = K(1)
    for t_ in range(-(b-1), (b-1)+1):
        acc *= (value_K - K(t_))
    return acc

# ------------------------- Constraint polynomial f ------------------------------
def f_eval_in_K(m_vals):
    # If f is present in the dump, honor it exactly (sparse poly).
    if f_spec:
        assert int(f_spec["arity"]) == len(m_vals), "f.arity mismatch"
        total = K(0)
        for term in f_spec["terms"]:
            coeff = K(F_from_str(term["coeff"]))
            exps  = term["exps"]
            monom = K(1)
            for x, e in zip(m_vals, exps):
                monom *= x**int(e)
            total += coeff * monom
        return total
    # Otherwise: simple non-linear default (keeps the algebra honest):
    # f(X1,X2,...) = X1 * X2  (ignores higher arity)
    if len(m_vals) < 2:
        return m_vals[0] if m_vals else K(0)
    return m_vals[0] * m_vals[1]

# ------------------------- Paper blocks, exactly --------------------------------
# F(X_r) = f( ˜(M_1 z_1)(X_r), ..., ˜(M_t z_1)(X_r) )
def F_of_Xr(Xr_bits):
    m_vals = []
    for j in range(t):
        wj = list(vector(F, Mats[j] * z1))   # in F^n
        m_vals.append(mle_vector_at(wj, Xr_bits))
    return f_eval_in_K(m_vals)

# NC_i(X) = Π_t ( ˜(Z_i M_1^T)(X) - t )   (general M1; matches Rust path)
def NC_i_of_X(i_inst, Xa_bits, Xr_bits):
    Zi = Zs[i_inst]               # d×m
    M1T = Mats[0].transpose()     # m×n
    Y = Zi * M1T                  # d×n
    y_mle = mle_matrix_at(Y, Xa_bits, Xr_bits)
    return range_product(y_mle, b)

# Eval_{(i,j)}(X) = eq(X,(α,r)) · ˜(Z_i M_j^T)(X), i∈{2..k}, j∈{1..t}
def Eval_ij_of_X(i_1based, j_1based, Xa_bits, Xr_bits, X_bits_concat):
    Zi = Zs[i_1based - 1]
    MjT = Mats[j_1based - 1].transpose()
    Y = Zi * MjT                  # d×n
    y_mle = mle_matrix_at(Y, Xa_bits, Xr_bits)
    return eq_general(X_bits_concat, alpha + r_vec) * y_mle

# Q(X) as written
def Q_of_X(X_bits_concat):
    Xa_bits = X_bits_concat[:ell_d]
    Xr_bits = X_bits_concat[ell_d:]
    eq_beta = eq_general(X_bits_concat, beta)

    F_val = F_of_Xr(Xr_bits)

    NC_sum = K(0)
    gpow = gamma                  # γ^1
    for i in range(1, k_total+1):
        NC_sum += gpow * NC_i_of_X(i-1, Xa_bits, Xr_bits)
        gpow *= gamma

    Eval_sum = K(0)
    gamma_k = K(1)
    for _ in range(k_total): gamma_k *= gamma
    for j in range(1, t+1):
        for i in range(2, k_total+1):
            exp_ = i + (j-1)*k_total - 1
            w = K(1)
            for _ in range(exp_): w *= gamma
            Eval_sum += w * Eval_ij_of_X(i, j, Xa_bits, Xr_bits, X_bits_concat)

    return eq_beta * (F_val + NC_sum) + gamma_k * Eval_sum

# ------------------------- Enumerate hypercube ----------------------------------
def all_assignments(ell):
    for mask in range(1 << ell):
        bits = [K(1) if ((mask >> i) & 1) else K(0) for i in range(ell)]
        yield mask, bits

# Total sum S = Σ_X Q(X)
S_total = K(0)
# For step-by-step tables:
rows_detail = []  # one per X: dict with all partials

for mask, X_bits in all_assignments(ell):
    Xa_bits = X_bits[:ell_d]
    Xr_bits = X_bits[ell_d:]
    eq_beta = eq_general(X_bits, beta)

    # F
    Fval = F_of_Xr(Xr_bits)
    F_contrib = eq_beta * Fval

    # NC
    NC_items = []
    gpow = gamma
    NC_weighted_sum = K(0)
    for i in range(1, k_total+1):
        Ni = NC_i_of_X(i-1, Xa_bits, Xr_bits)
        wNi = eq_beta * gpow * Ni
        NC_items.append({"i": i, "Ni": Ni, "wNi": wNi})
        NC_weighted_sum += wNi
        gpow *= gamma

    # Eval
    gamma_k = K(1)
    for _ in range(k_total): gamma_k *= gamma
    Eval_terms = []
    Eval_inner = K(0)
    for j in range(1, t+1):
        for i in range(2, k_total+1):
            exp_ = i + (j-1)*k_total - 1
            w = K(1)
            for _ in range(exp_): w *= gamma
            Eij = Eval_ij_of_X(i, j, Xa_bits, Xr_bits, X_bits)
            Eval_terms.append({"i": i, "j": j, "w": w, "Eij": Eij, "wEij": w*Eij})
            Eval_inner += w * Eij
    Eval_contrib = gamma_k * Eval_inner

    Qx = F_contrib + NC_weighted_sum + Eval_contrib
    S_total += Qx

    rows_detail.append({
        "mask": mask, "Xa": Xa_bits, "Xr": Xr_bits,
        "eq_beta": eq_beta,
        "Fval": Fval, "F_contrib": F_contrib,
        "NC_items": NC_items, "NC_sum": NC_weighted_sum,
        "Eval_terms": Eval_terms, "Eval_contrib": Eval_contrib,
        "Qx": Qx
    })

# Round-0 polynomial p(t): split on FIRST ROW BIT (to match row-phase)
first_row_bit_index = ell_d  # Ajtai bits first, then row bits; take the first row bit
def p_of(t_bit):
    acc = K(0)
    for rd in rows_detail:
        bit_at_pos = (rd["mask"] >> first_row_bit_index) & 1
        if bit_at_pos == int(t_bit):
            acc += rd["Qx"]
    return acc

p0, p1 = p_of(0), p_of(1)

# Decomposed initial-sum identity (for cross-checking against your prover)
# F(β_r) by definition:
def F_at_beta_r():
    # ˜(M_j z_1)(β_r) = Σ_row (M_j z_1)[row] · χ_row(β_r)
    m_vals = []
    for j in range(t):
        wj = list(vector(F, Mats[j] * z1))
        s = K(0)
        for row in range(n):
            # χ_row(β_r) = eq( row_bits, β_r )
            row_bits = [K(1) if ((row >> i) & 1) else K(0) for i in range(ell_n)]
            s += K(wj[row]) * eq_general(row_bits, beta_r)
        m_vals.append(s)
    return f_eval_in_K(m_vals)

F_beta_r = F_at_beta_r()

# NC hypercube sum (def): Σ_X eq(X,β) Σ_i γ^i NC_i(X)
def NC_hypercube_sum():
    acc = K(0)
    for _mask, X_bits in all_assignments(ell):
        eq_beta = eq_general(X_bits, beta)
        Xa_bits, Xr_bits = X_bits[:ell_d], X_bits[ell_d:]
        gpow = gamma
        inner = K(0)
        for i in range(1, k_total+1):
            inner += gpow * NC_i_of_X(i-1, Xa_bits, Xr_bits)
            gpow *= gamma
        acc += eq_beta * inner
    return acc

NC_sum_def = NC_hypercube_sum()

# Eval hypercube term (def): γ^k Σ_{j,i≥2} γ^{i+(j-1)k-1} ˜y_(i,j)(α)
def Eval_sum_def():
    gamma_k = K(1)
    for _ in range(k_total): gamma_k *= gamma
    acc = K(0)
    for j in range(1, t+1):
        for i in range(2, k_total+1):
            exp_ = i + (j-1)*k_total - 1
            w = K(1)
            for _ in range(exp_): w *= gamma
            Zi = Zs[i-1]; MjT = Mats[j-1].transpose()
            Y = Zi * MjT
            # Build y_alpha_row[row] = Σ_rho Y[rho,row] χ_rho(α)
            y_alpha_row = []
            for row in range(n):
                srow = K(0)
                for rho in range(d):
                    rho_bits = [K(1) if ((rho >> b_) & 1) else K(0) for b_ in range(ell_d)]
                    srow += K(Y[rho, row]) * eq_general(rho_bits, alpha)
                y_alpha_row.append(srow)
            # ˜y_(i,j)(α) = Σ_row y_alpha_row[row] · χ_row(r) with χ_row(r) = eq(row_bits, r)
            s_ij = K(0)
            for row in range(n):
                row_bits = [K(1) if ((row >> b_) & 1) else K(0) for b_ in range(ell_n)]
                s_ij += y_alpha_row[row] * eq_general(row_bits, r_vec)
            acc += w * s_ij
    return ( (gamma_k) * acc )

Eval_sum_total = Eval_sum_def()

# ------------------------- Reporting --------------------------------------------
def kfmt(x):  # compact formatter for K
    return str(x)

print("=== Π-CCS Paper-faithful Round-0 from dump ===")
print(f"dump      : {dump_path}")
print(f"p         : {p}  (Goldilocks)")
print(f"b         : {b}")
print(f"d,n,m,t,k : {d}, {n}, {m}, {t}, {k_total}")
print(f"ℓ_d,ℓ_n,ℓ : {ell_d}, {ell_n}, {ell}")
if used_nonres is None:
    print(f"K         : GF(p^2) (abstract); mapping pairs as a + b·u")
else:
    print(f"K         : {used_nonres}; mapping pairs as a + b·u")

if VERBOSE >= 2:
    print("\n-- Per-X contributions (Ajtai bits then Row bits) --")
    for rd in rows_detail:
        mask = rd["mask"]; Xa=rd["Xa"]; Xr=rd["Xr"]
        print(f"X #{mask:0{ell}b}  Xa={Xa}  Xr={Xr}")
        print(f"  eq_beta     : {kfmt(rd['eq_beta'])}")
        print(f"  F(X_r)      : {kfmt(rd['Fval'])}")
        print(f"  F contrib   : {kfmt(rd['F_contrib'])}")
        for item in rd["NC_items"]:
            print(f"  NC i={item['i']}   Ni={kfmt(item['Ni'])}   eq·γ^i·Ni={kfmt(item['wNi'])}")
        print(f"  NC contrib  : {kfmt(rd['NC_sum'])}")
        for e in rd["Eval_terms"]:
            print(f"  Eval(i={e['i']},j={e['j']})  w={kfmt(e['w'])}  Eij={kfmt(e['Eij'])}  wEij={kfmt(e['wEij'])}")
        print(f"  Eval contrib: {kfmt(rd['Eval_contrib'])}")
        print(f"  Q(X) total  : {kfmt(rd['Qx'])}\n")

print("-- Block totals --")
print(f"S_total = Σ_X Q(X)            : {kfmt(S_total)}")
print(f"p(0) split on first row bit    : {kfmt(p0)}")
print(f"p(1) split on first row bit    : {kfmt(p1)}")
print(f"Check p(0)+p(1) == S_total     : {kfmt(p0 + p1)}  ==  {kfmt(S_total)}")
assert p0 + p1 == S_total, "Round-0 invariant failed: p(0)+p(1) ≠ Σ_X Q(X)"

# Decomposed initial-sum identity:
print("\n-- Decomposed initial-sum identity --")
print(f"F(β_r)                         : {kfmt(F_beta_r)}")
print(f"NC hypercube sum (def)         : {kfmt(NC_sum_def)}")
print(f"Eval total (def)               : {kfmt(Eval_sum_total)}")
print(f"T (paper)                      : {kfmt(Eval_sum_total)}")
print(f"F(β_r) + NC + Eval             : {kfmt(F_beta_r + NC_sum_def + Eval_sum_total)}")
print(f"Compare against S_total        : {kfmt(S_total)}")
assert F_beta_r + NC_sum_def + Eval_sum_total == S_total, "Initial-sum decomposition mismatch"

print("\n[OK] Paper-faithful round-0 checks passed.")
# =========================================================================================

# Optional: compare against a Rust trace dump (per-X). Pass as argv[2] or NEO_ROUND0_TRACE.
trace_path = sys.argv[2] if len(sys.argv) >= 3 else os.environ.get("NEO_ROUND0_TRACE", None)
if trace_path:
    with open(trace_path, 'r') as tf:
        T = json.load(tf)

    def pair_to_K(pair):
        return K(F(Integer(pair[0]))) + K(F(Integer(pair[1]))) * u

    print("\n== Rust Trace Comparison ==")
    # Totals
    assert S_total == pair_to_K(T["S_total"]), "S_total mismatch"
    assert p0 == pair_to_K(T["p0"]), "p0 mismatch"
    assert p1 == pair_to_K(T["p1"]), "p1 mismatch"
    assert F_beta_r == pair_to_K(T["F_beta_r"]), "F(β_r) mismatch"
    assert NC_sum_def == pair_to_K(T["NC_hypercube_sum"]), "NC hypercube sum mismatch"
    assert Eval_sum_total == pair_to_K(T["Eval_total"]), "Eval total mismatch"
    print("[COMPARE] Totals: OK (S_total, p0, p1, F(β_r), NC_sum, Eval_total)")

    # Per-X
    rust_rows = { int(r["mask"]): r for r in T["rows"] }
    rows_checked = 0
    nc_terms_checked = 0
    eval_terms_checked = 0
    for rd in rows_detail:
        mask = rd["mask"]
        r = rust_rows.get(mask)
        assert r is not None, f"Missing mask {mask} in Rust trace"
        # Aggregates
        assert rd["eq_beta"] == pair_to_K(r["eq_beta"]), f"eq_beta mismatch at mask={mask}"
        assert rd["Fval"] == pair_to_K(r["F_val"]), f"F_val mismatch at mask={mask}"
        assert rd["F_contrib"] == pair_to_K(r["F_contrib"]), f"F_contrib mismatch at mask={mask}"
        assert rd["NC_sum"] == pair_to_K(r["nc_contrib"]), f"NC contrib mismatch at mask={mask}"
        assert rd["Eval_contrib"] == pair_to_K(r["eval_contrib"]), f"Eval contrib mismatch at mask={mask}"
        assert rd["Qx"] == pair_to_K(r["Qx"]), f"Q(X) mismatch at mask={mask}"

        # NC items (per i)
        rust_nc = r.get("nc_items", [])
        assert len(rd["NC_items"]) == len(rust_nc), f"NC items length mismatch at mask={mask}"
        for idx in range(len(rust_nc)):
            rni = rust_nc[idx]
            sni = rd["NC_items"][idx]
            assert int(sni["i"]) == int(rni["i"]), f"NC i index mismatch at mask={mask}"
            assert sni["Ni"] == pair_to_K(rni["Ni"]), f"NC Ni mismatch at mask={mask}, i={sni['i']}"
            assert sni["wNi"] == pair_to_K(rni["wNi"]), f"NC wNi mismatch at mask={mask}, i={sni['i']}"
            nc_terms_checked += 1

        # Eval terms (per (i,j))
        rust_eval = r.get("eval_terms", [])
        assert len(rd["Eval_terms"]) == len(rust_eval), f"Eval terms length mismatch at mask={mask}"
        for idx in range(len(rust_eval)):
            re = rust_eval[idx]
            se = rd["Eval_terms"][idx]
            assert int(se["i"]) == int(re["i"]), f"Eval i mismatch at mask={mask}, idx={idx}"
            assert int(se["j"]) == int(re["j"]), f"Eval j mismatch at mask={mask}, idx={idx}"
            assert se["w"] == pair_to_K(re["w"]), f"Eval weight mismatch at mask={mask}, i={se['i']}, j={se['j']}"
            assert se["Eij"] == pair_to_K(re["Eij"]), f"Eval Eij mismatch at mask={mask}, i={se['i']}, j={se['j']}"
            assert se["wEij"] == pair_to_K(re["wEij"]), f"Eval wEij mismatch at mask={mask}, i={se['i']}, j={se['j']}"
            eval_terms_checked += 1

        rows_checked += 1

    print(f"[COMPARE] Per-X aggregates: OK ({rows_checked} rows)")
    print(f"[COMPARE] NC items: OK ({nc_terms_checked} items)")
    print(f"[COMPARE] Eval terms: OK ({eval_terms_checked} terms)")
    print("[COMPARE] Sage == Rust (paper trace): all checks passed.")

    # Optional side-by-side printing
    def fmtK(x):
        return kfmt(x)

    def show_row_side_by_side(mask_int, rd_sage, rd_rust):
        Xa = rd_sage["Xa"]; Xr = rd_sage["Xr"]
        print(f"\n== Side-by-side mask #{mask_int:0{ell}b}  Xa={Xa}  Xr={Xr} ==")
        print(f"eq_beta (Sage): {fmtK(rd_sage['eq_beta'])}")
        print(f"eq_beta (Rust): {fmtK(pair_to_K(rd_rust['eq_beta']))}")
        print(f"F_val (Sage): {fmtK(rd_sage['Fval'])}")
        print(f"F_val (Rust): {fmtK(pair_to_K(rd_rust['F_val']))}")
        print(f"F_contrib (Sage): {fmtK(rd_sage['F_contrib'])}")
        print(f"F_contrib (Rust): {fmtK(pair_to_K(rd_rust['F_contrib']))}")
        print(f"NC_contrib (Sage): {fmtK(rd_sage['NC_sum'])}")
        print(f"NC_contrib (Rust): {fmtK(pair_to_K(rd_rust['nc_contrib']))}")
        print(f"Eval_contrib (Sage): {fmtK(rd_sage['Eval_contrib'])}")
        print(f"Eval_contrib (Rust): {fmtK(pair_to_K(rd_rust['eval_contrib']))}")
        print(f"Q(X) (Sage): {fmtK(rd_sage['Qx'])}")
        print(f"Q(X) (Rust): {fmtK(pair_to_K(rd_rust['Qx']))}")
        # NC items
        rn = rd_rust.get('nc_items', [])
        sn = rd_sage['NC_items']
        for idx in range(min(len(sn), len(rn))):
            si, ri = sn[idx], rn[idx]
            print(f"  NC i={si['i']}  Ni (Sage):  {fmtK(si['Ni'])}")
            print(f"                 Ni (Rust):  {fmtK(pair_to_K(ri['Ni']))}")
            print(f"                 wNi (Sage): {fmtK(si['wNi'])}")
            print(f"                 wNi (Rust): {fmtK(pair_to_K(ri['wNi']))}")
        # Eval terms
        re = rd_rust.get('eval_terms', [])
        se = rd_sage['Eval_terms']
        for idx in range(min(len(se), len(re))):
            ss, rr = se[idx], re[idx]
            print(f"  Eval(i={ss['i']},j={ss['j']}) w (Sage):    {fmtK(ss['w'])}")
            print(f"                           w (Rust):    {fmtK(pair_to_K(rr['w']))}")
            print(f"                           Eij (Sage):  {fmtK(ss['Eij'])}")
            print(f"                           Eij (Rust):  {fmtK(pair_to_K(rr['Eij']))}")
            print(f"                           wEij (Sage): {fmtK(ss['wEij'])}")
            print(f"                           wEij (Rust): {fmtK(pair_to_K(rr['wEij']))}")

    # Single-row detail if requested
    if COMPARE_MASK is not None:
        rr = rust_rows.get(COMPARE_MASK)
        ss = next((rd for rd in rows_detail if rd['mask'] == COMPARE_MASK), None)
        if rr is None or ss is None:
            print(f"[COMPARE] Requested COMPARE_MASK={COMPARE_MASK} not found in rows.")
        else:
            show_row_side_by_side(COMPARE_MASK, ss, rr)

    # Aggregated per-row summaries if requested
    if COMPARE_VERBOSE > 0 and COMPARE_MASK is None:
        limit = COMPARE_ROWS if COMPARE_ROWS > 0 else len(rows_detail)
        print(f"\n== Per-X side-by-side aggregates (first {limit} rows) ==")
        count = 0
        for rd in rows_detail:
            if count >= limit: break
            mask = rd['mask']
            rr = rust_rows.get(mask)
            if rr is None: continue
            print(f"mask #{mask:0{ell}b}")
            print(f"  eq_beta (Sage): {fmtK(rd['eq_beta'])}")
            print(f"  eq_beta (Rust): {fmtK(pair_to_K(rr['eq_beta']))}")
            print(f"  F_contrib (Sage): {fmtK(rd['F_contrib'])}")
            print(f"  F_contrib (Rust): {fmtK(pair_to_K(rr['F_contrib']))}")
            print(f"  NC_contrib (Sage): {fmtK(rd['NC_sum'])}")
            print(f"  NC_contrib (Rust): {fmtK(pair_to_K(rr['nc_contrib']))}")
            print(f"  Eval_contrib (Sage): {fmtK(rd['Eval_contrib'])}")
            print(f"  Eval_contrib (Rust): {fmtK(pair_to_K(rr['eval_contrib']))}")
            print(f"  Q(X) (Sage): {fmtK(rd['Qx'])}")
            print(f"  Q(X) (Rust): {fmtK(pair_to_K(rr['Qx']))}")
            count += 1
