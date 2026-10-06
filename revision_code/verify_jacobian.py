"""Verify pointwise shape derivatives dR/dtheta, the integrals
K_theta(p) = int_0^p dR/dtheta dt, K_theta(1), and the MAD gradient
dh/dtheta = (2p-1) dR/dtheta - 2 K_theta(p) + K_theta(1)
against finite differences / quadrature.  Also checks the log-beta and
log-gamma hypergeometric identities, the L-moment tables, the Gumbel
kurtosis components and the Normal MAD-kurtosis baseline.
"""
import mpmath as mp
from verify_tables import R_factory, num, Binc, nu, B

mp.mp.dps = 30
psi = mp.digamma
G = mp.gamma

def Ba(x, a, b):  # d/da B(x;a,b)
    return mp.quad(lambda t: mp.log(t) * t ** (a - 1) * (1 - t) ** (b - 1), [0, x])
def Bb(x, a, b):  # d/db B(x;a,b)
    return mp.quad(lambda t: mp.log(1 - t) * t ** (a - 1) * (1 - t) ** (b - 1), [0, x])
def Ba_hyp(z, a, b):
    return mp.log(z) * Binc(z, a, b) - z ** a / a ** 2 * mp.hyp3f2(a, a, 1 - b, a + 1, a + 1, z)
def Bb_hyp(z, a, b):
    return (B(a, b) * (psi(b) - psi(a + b))
            - (mp.log(1 - z) * Binc(1 - z, b, a)
               - (1 - z) ** b / b ** 2 * mp.hyp3f2(b, b, 1 - a, b + 1, b + 1, 1 - z)))
def nu_c(c, x):  # d/dc lower incomplete gamma
    return mp.quad(lambda t: mp.log(t) * t ** (c - 1) * mp.exp(-t), [0, x])
def nu_c_hyp(c, x):
    return nu(c, x) * mp.log(x) - x ** c / c ** 2 * mp.hyp2f2(c, c, c + 1, c + 1, -x)

def rel(a, b):
    return float(abs(a - b) / max(abs(b), mp.mpf("1e-25")))

print("== special-function identities ==")
for (z, a, b) in [(0.3, 0.5, 2.0), (0.7, 1.7, 0.4), (0.5, 2.5, 3.0)]:
    z, a, b = map(mp.mpf, (z, a, b))
    print(f"  Ba z={float(z)} a={float(a)} b={float(b)}: {rel(Ba_hyp(z,a,b), Ba(z,a,b)):.1e}   Bb: {rel(Bb_hyp(z,a,b), Bb(z,a,b)):.1e}")
for (c, x) in [(0.4, 0.7), (1.6, 1.3), (2.5, 0.2)]:
    c, x = mp.mpf(c), mp.mpf(x)
    print(f"  nu_c c={float(c)} x={float(x)}: {rel(nu_c_hyp(c,x), nu_c(c,x)):.1e}")

# --------------------------------------------------------------------------
# corrected closed forms  (dR, K(p), K(1))
def corrected(name, a, k=None):
    if name in ("ParetoI", "Lomax"):
        up = lambda p: (1 - p) ** (1 / a)
        K1 = B(a - 1, 1) * (psi(a - 1) - psi(a))
        Kp = lambda p: K1 - Ba(up(p), a - 1, 1)
        if name == "ParetoI":
            dR = lambda p: mp.log(1 - p) / a ** 2 * (1 - p) ** (-1 / a)
        else:
            dR = lambda p: mp.log(1 - p) / a ** 2 * (1 - p) ** (-1 / a)
        return {"alpha": (dR, Kp, K1)}
    if name == "LogLogistic":
        A, Bq = 1 + 1 / a, 1 - 1 / a
        dR = lambda p: -mp.log(p / (1 - p)) / a ** 2 * (p / (1 - p)) ** (1 / a)
        Kp = lambda p: -(Ba(p, A, Bq) - Bb(p, A, Bq)) / a ** 2
        K1 = -B(A, Bq) * (psi(A) - psi(Bq)) / a ** 2
        K1b = -B(A, Bq) * (a - mp.pi * mp.cot(mp.pi / a)) / a ** 2
        assert rel(K1, K1b) < 1e-20
        return {"alpha": (dR, Kp, K1)}
    if name == "Burr":
        up = lambda p: (1 - p) ** (1 / a)
        v = lambda p: (1 - p) ** (-1 / a)
        dRa = lambda p: k * mp.log(1 - p) / a ** 2 * v(p) / (v(p) - 1) * (v(p) - 1) ** k
        K1a = k * B(a - k, k) * (psi(a - k) - psi(a))
        Kpa = lambda p: k * (B(a - k, k) * (psi(a - k) - psi(a)) - Ba(up(p), a - k, k))
        dRk = lambda p: mp.log(v(p) - 1) * (v(p) - 1) ** k
        aa, bb = a - k, k + 1
        K1k = a * B(aa, bb) * (psi(bb) - psi(aa))
        Kpk = lambda p: a * ((B(aa, bb) * (psi(bb) - psi(aa + bb)) - Bb(up(p), aa, bb))
                             - (B(aa, bb) * (psi(aa) - psi(aa + bb)) - Ba(up(p), aa, bb)))
        return {"alpha": (dRa, Kpa, K1a), "k": (dRk, Kpk, K1k)}
    if name == "Frechet":
        c = 1 - 1 / a
        dR = lambda p: mp.log(-mp.log(p)) / a ** 2 * (-mp.log(p)) ** (-1 / a)
        K1 = G(c) * psi(c) / a ** 2
        Kp = lambda p: (G(c) * psi(c) - nu_c(c, -mp.log(p))) / a ** 2
        return {"alpha": (dR, Kp, K1)}
    if name == "Weibull":
        c = 1 + 1 / a
        dR = lambda p: -mp.log(-mp.log(1 - p)) / a ** 2 * (-mp.log(1 - p)) ** (1 / a)
        K1 = -G(c) * psi(c) / a ** 2
        Kp = lambda p: -nu_c(c, -mp.log(1 - p)) / a ** 2
        return {"alpha": (dR, Kp, K1)}

# as printed in the submission (Tables A4/A5) for comparison
def printed(name, a, k=None):
    if name == "ParetoI":
        return {"alpha": (lambda p: -Ba((1 - p) ** (1 / a), a - 1, 1) / a,
                          -B(a - 1, 1) * (psi(a - 1) - psi(a)) / a)}
    if name == "Lomax":
        return {"alpha": (None, (B(a - 1, 2) * (psi(a - 1) - psi(a + 1)) + 1) / a)}
    if name == "LogLogistic":
        A, Bq = 1 + 1 / a, 1 - 1 / a
        return {"alpha": (None, B(A, Bq) * (psi(A) - psi(Bq)) / a ** 2)}
    if name == "Burr":
        return {"alpha": (None, k * B(a - k, k) * (psi(a - k) - psi(a)) / a ** 2),
                "k": (None, B(a - k, k + 1) * (psi(a - k) - psi(k + 1)))}
    if name == "Frechet":
        c = 1 - 1 / a
        return {"alpha": (None, G(c) * psi(c) / a ** 2)}
    if name == "Weibull":
        c = 1 + 1 / a
        return {"alpha": (None, G(c) * psi(c) / a ** 2)}

def param_R(name, a, k, which, eps):
    if which == "alpha":
        return R_factory(name, a + eps, k)
    return R_factory(name, a, k + eps)

print("\n== Jacobian ingredients (rel. errors vs numerics) ==")
cases = [("ParetoI", 2.5, None), ("Lomax", 2.5, None), ("LogLogistic", 2.5, None),
         ("Burr", 3.0, 1.5), ("Frechet", 2.5, None), ("Weibull", 2.5, None)]
eps = mp.mpf("1e-10")
for name, a, k in cases:
    a = mp.mpf(a); k = mp.mpf(k) if k else None
    cor = corrected(name, a, k)
    pr = printed(name, a, k)
    for which, (dR, Kp, K1) in cor.items():
        Rp = param_R(name, a, k, which, eps)
        Rm = param_R(name, a, k, which, -eps)
        R0 = R_factory(name, a, k)
        dR_num = lambda p: (Rp(p) - Rm(p)) / (2 * eps)
        errs_dR = max(rel(dR(p), dR_num(p)) for p in (mp.mpf(0.25), mp.mpf(0.5), mp.mpf(0.75)))
        Kp_num = lambda p: mp.quad(dR_num, [0, p])
        errs_K = max(rel(Kp(p), Kp_num(p)) for p in (mp.mpf(0.25), mp.mpf(0.75)))
        K1_num = mp.quad(dR_num, [0, 0.5, 1])
        # gradient of h vs finite difference of h
        def h_of(Rf):
            mu = mp.quad(Rf, [0, 0.5, 1])
            return lambda p: mu - 2 * mp.quad(Rf, [0, p]) + (2 * p - 1) * Rf(p)
        hp, hm = h_of(Rp), h_of(Rm)
        grad_err = max(rel((2 * p - 1) * dR(p) - 2 * Kp(p) + K1, (hp(p) - hm(p)) / (2 * eps))
                       for p in (mp.mpf(0.25), mp.mpf(0.5), mp.mpf(0.75)))
        line = f"  {name:11s} d/{which:5s}: dR {errs_dR:.1e}  K(p) {errs_K:.1e}  K(1) {rel(K1, K1_num):.1e}  grad-h {grad_err:.1e}"
        if which in pr:
            line += f"   | printed K(1) rel.err {rel(pr[which][1], K1_num):.2f}"
            if pr[which][0] is not None:
                line += f", printed K(p=.25) rel.err {rel(pr[which][0](mp.mpf(0.25)), Kp_num(mp.mpf(0.25))):.2f}"
        print(line)

# --------------------------------------------------------------------------
print("\n== L-moment tables (lambda2, tau3, tau4) ==")
def lm_closed(name, a, k=None):
    if name in ("ParetoI", "Lomax"):
        f = lambda j: a * B(j * a - 1, 2)
        return (f(1) - 2 * f(2), 6 * f(3) - 6 * f(2) + f(1), f(1) - 12 * f(2) + 30 * f(3) - 20 * f(4))
    if name == "Burr":
        f = lambda j: a * B(j * a - k, k + 1)
        return (f(1) - 2 * f(2), 6 * f(3) - 6 * f(2) + f(1), f(1) - 12 * f(2) + 30 * f(3) - 20 * f(4))
    if name == "LogLogistic":
        f = lambda j: B(j - 1 / a, 1 + 1 / a)
        return (f(1) - 2 * f(2), 6 * f(3) - 6 * f(2) + f(1), f(1) - 12 * f(2) + 30 * f(3) - 20 * f(4))
    if name == "Frechet":
        c = G(1 - 1 / a); r = 1 / a
        return (c * (2 ** r - 1), c * (1 - 3 * 2 ** r + 2 * 3 ** r), c * (-1 + 6 * 2 ** r - 10 * 3 ** r + 5 * 4 ** r))
    if name == "Gumbel":
        return (mp.log(2), 2 * mp.log(3) - 3 * mp.log(2), 16 * mp.log(2) - 10 * mp.log(3))
    if name == "Weibull":
        c = G(1 + 1 / a); r = -1 / a
        return (c * (1 - 2 ** r), c * (1 - 3 * 2 ** r + 2 * 3 ** r), c * (1 - 6 * 2 ** r + 10 * 3 ** r - 5 * 4 ** r))
for name, a, k in [("ParetoI", 2.5, None), ("Lomax", 2.5, None), ("Burr", 3.0, 1.5), ("LogLogistic", 2.5, None),
                   ("Frechet", 2.5, None), ("Gumbel", 1, None), ("Weibull", 2.5, None)]:
    a = mp.mpf(a); k = mp.mpf(k) if k else None
    n = num(name, a, k)
    l2, l3, l4 = lm_closed(name, a, k)
    print(f"  {name:11s} l2 {rel(l2, n['l2']):.1e} l3 {rel(l3, n['l3']):.1e} l4 {rel(l4, n['l4']):.1e}")
# log-logistic lambda2 as printed in Table L-scale (with extra alpha)
a = mp.mpf(2.5); n = num("LogLogistic", a)
print("  LogLogistic lambda2 as printed in L-scale table (with alpha):",
      f"{rel(a * B(1 - 1/a, 1 + 1/a) - 2 * B(2 - 1/a, 1 + 1/a), n['l2']):.2f}")

# --------------------------------------------------------------------------
print("\n== Gumbel kurtosis components (half values I(1/2)-2I(1/4), mu+I(1/2)-2I(3/4)) ==")
n = num("Gumbel", 1)
hl = mp.log(2) / 2 + mp.ei(-mp.log(2)) - 2 * mp.ei(-mp.log(4))
hr = (mp.euler - mp.log(mp.log(2)) / 2 + mp.ei(-mp.log(2))
      + mp.mpf(3) / 2 * mp.log(-mp.log(mp.mpf(0.75))) - 2 * mp.ei(mp.log(mp.mpf(0.75))))
print(f"  HL/2 {rel(hl, n['HL']/2):.1e}  HR/2 {rel(hr, n['HR']/2):.1e}")
Hg = mp.euler + mp.log(mp.log(2)) - 2 * mp.ei(-mp.log(2))
print(f"  H(1/2) Gumbel corrected {rel(Hg, n['Hmed']):.1e}")

print("\n== Normal baselines ==")
# K for standard normal
from mpmath import npdf, ncdf
q1 = mp.sqrt(2) * mp.erfinv(-mp.mpf(1)/2)
H = mp.sqrt(2 / mp.pi)
HL = 2 * mp.quad(lambda x: abs(x - q1) * npdf(x), [-mp.inf, q1, 0])
K = (HL + HL) / (2 * H)
print(f"  Normal K=(HL+HR)/(2H) = {float(K):.4f};  -1+2*sqrt(2pi)*phi(q1) = {float(-1 + 2*mp.sqrt(2*mp.pi)*npdf(q1)):.4f}")
print(f"  Normal tau4 = {float(30/mp.pi*mp.atan(mp.sqrt(2)) - 9):.4f}")
