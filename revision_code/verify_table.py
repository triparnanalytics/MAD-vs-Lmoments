"""Numerical verification of the closed-form formulas in main_mdpi.tex.

Each closed form (as printed in the submitted manuscript, and as corrected
in the revision) is compared with direct numerical quadrature of the
quantile function.  m = 0, s = 1 throughout (MAD and L-moments are
location-free and linear in s).
"""
import mpmath as mp

mp.mp.dps = 30
B = mp.beta
def Binc(x, a, b):
    return mp.betainc(a, b, 0, x)
def nu(s, x):
    return mp.gammainc(s, 0, x)
Ei = mp.ei
g = mp.euler

def R_factory(name, a, k=None):
    if name == "ParetoI":
        return lambda p: (1 - p) ** (-1 / a)
    if name == "Lomax":
        return lambda p: (1 - p) ** (-1 / a) - 1
    if name == "LogLogistic":
        return lambda p: ((1 - p) ** (-1) - 1) ** (1 / a)
    if name == "Burr":
        return lambda p: ((1 - p) ** (-1 / a) - 1) ** k
    if name == "Frechet":
        return lambda p: (-mp.log(p)) ** (-1 / a)
    if name == "Gumbel":
        return lambda p: -mp.log(-mp.log(p))
    if name == "Weibull":
        return lambda p: (-mp.log(1 - p)) ** (1 / a)

def num(name, a, k=None):
    R = R_factory(name, a, k)
    I = lambda p: mp.quad(R, [0, p]) if p > 0 else mp.mpf(0)
    mu = mp.quad(R, [0, mp.mpf(1)/2, 1])
    H = lambda p: mu - 2 * I(p) + (2 * p - 1) * R(p)
    lam = lambda w: mp.quad(lambda p: R(p) * w(p), [0, mp.mpf(1)/2, 1])
    l2 = lam(lambda p: 2 * p - 1)
    l3 = lam(lambda p: 6 * p**2 - 6 * p + 1)
    l4 = lam(lambda p: 20 * p**3 - 30 * p**2 + 12 * p - 1)
    q = mp.mpf(1) / 4
    h = mp.mpf(1) / 2
    M = R(h)
    Hmed = H(h)
    # left / right MAD of half-truncated variables around their medians
    HL = 2 * ((I(h) - I(q)) - I(q))                # MAD of X_L about Q(1/4)
    HR = 2 * ((mu - I(3 * q)) - (I(3 * q) - I(h)))  # MAD of X_R about Q(3/4)
    G = (mu - M) / Hmed
    K = (HL + HR) / (2 * Hmed)
    return dict(R=R, I=I, mu=mu, H=H, l2=l2, l3=l3, l4=l4, M=M, Hmed=Hmed,
                HL=HL, HR=HR, G=G, K=K, t3=l3 / l2, t4=l4 / l2)

# ---------------------------------------------------------------- closed forms
def closed(name, a, k=None):
    c = {}
    if name == "ParetoI":
        u = lambda p: (1 - p) ** (1 / a)
        c["mu"] = a * B(a - 1, 1)
        c["I"] = lambda p: a * B(a - 1, 1) - a * Binc(u(p), a - 1, 1)
        c["h"] = lambda p: 2 * a * Binc(u(p), a - 1, 1) - a * B(a - 1, 1) + (2 * p - 1) * (1 - p) ** (-1 / a)
        c["l2"] = a * (B(a - 1, 2) - 2 * B(2 * a - 1, 2))  # as printed
        c["G"] = (B(a - 1, 1) - mp.mpf(0.5) ** (1 / a)) / (2 * Binc(mp.mpf(0.5) ** (1 / a), a - 1, 1) - B(a - 1, 1))
        # table MAD skew as printed omits factor alpha -> check both
        c["G_alpha"] = (a * B(a - 1, 1) - 2 ** (1 / a)) / (2 * a * Binc(mp.mpf(0.5) ** (1 / a), a - 1, 1) - a * B(a - 1, 1))
        c["HL_tab"] = 2 * a * Binc(mp.mpf(0.75) ** (1 / a), a - 1, 1) - a * Binc(mp.mpf(0.5) ** (1 / a), a - 1, 1) - a * B(a - 1, 1)
        c["HR_tab"] = 2 * a * Binc(mp.mpf(0.25) ** (1 / a), a - 1, 1) - a * Binc(mp.mpf(0.5) ** (1 / a), a - 1, 1)
    if name == "Lomax":
        u = lambda p: (1 - p) ** (1 / a)
        c["mu"] = a * B(a - 1, 2)
        c["I"] = lambda p: a * B(a - 1, 2) - a * Binc(u(p), a - 1, 2)
        c["h"] = lambda p: 2 * a * Binc(u(p), a - 1, 2) - a * B(a - 1, 2) + (2 * p - 1) * ((1 - p) ** (-1 / a) - 1)
        c["G"] = (a * B(a - 1, 2) - 2 ** (1 / a) + 1) / (2 * a * Binc(mp.mpf(0.5) ** (1 / a), a - 1, 2) - a * B(a - 1, 2))
        c["HL_tab"] = 2 * a * Binc(mp.mpf(0.75) ** (1 / a), a - 1, 2) - a * Binc(mp.mpf(0.5) ** (1 / a), a - 1, 2) - a * B(a - 1, 2)
        c["HR_tab"] = 2 * a * Binc(mp.mpf(0.25) ** (1 / a), a - 1, 2) - a * Binc(mp.mpf(0.5) ** (1 / a), a - 1, 2)
    if name == "LogLogistic":
        A, Bb = 1 - 1 / a, 1 + 1 / a
        c["mu"] = B(A, Bb)
        c["I"] = lambda p: B(A, Bb) - Binc(1 - p, A, Bb)
        c["h"] = lambda p: 2 * Binc(1 - p, A, Bb) - B(A, Bb) + (2 * p - 1) * ((1 - p) ** (-1) - 1) ** (1 / a)
        c["G"] = (B(A, Bb) - 1) / (2 * Binc(mp.mpf(0.5), A, Bb) - B(A, Bb))
        c["HL_tab"] = 2 * Binc(mp.mpf(0.75), A, Bb) - Binc(mp.mpf(0.5), A, Bb) - B(A, Bb)
        c["HR_tab"] = Binc(mp.mpf(0.25), A, Bb) - Binc(mp.mpf(0.5), A, Bb)
        c["HR_fix"] = 2 * Binc(mp.mpf(0.25), A, Bb) - Binc(mp.mpf(0.5), A, Bb)
    if name == "Burr":
        u = lambda p: (1 - p) ** (1 / a)
        c["mu"] = a * B(a - k, k + 1)
        c["I"] = lambda p: a * B(a - k, k + 1) - a * Binc(u(p), a - k, k + 1)
        c["h"] = lambda p: 2 * a * Binc(u(p), a - k, k + 1) - a * B(a - k, k + 1) + (2 * p - 1) * ((1 - p) ** (-1 / a) - 1) ** k
        c["G"] = (a * B(a - k, k + 1) - (2 ** (1 / a) - 1) ** k) / (2 * a * Binc(mp.mpf(0.5) ** (1 / a), a - k, k + 1) - a * B(a - k, k + 1))
        c["HL_tab"] = 2 * a * Binc(mp.mpf(0.75) ** (1 / a), a - k, k + 1) - a * Binc(mp.mpf(0.5) ** (1 / a), a - k, k + 1) - a * B(a - k, k + 1)
        c["HR_tab"] = 2 * a * Binc(mp.mpf(0.25) ** (1 / a), a - k, k + 1) - a * Binc(mp.mpf(0.5) ** (1 / a), a - k, k + 1)
    if name == "Frechet":
        s_ = 1 - 1 / a
        c["mu"] = mp.gamma(s_)
        c["I"] = lambda p: mp.gamma(s_) - nu(s_, -mp.log(p))
        c["h"] = lambda p: 2 * nu(s_, -mp.log(p)) - mp.gamma(s_) + (2 * p - 1) * (-mp.log(p)) ** (-1 / a)
        c["G"] = (mp.gamma(s_) - mp.log(2) ** (-1 / a)) / (2 * nu(s_, mp.log(2)) - mp.gamma(s_))
        c["HL_tab"] = 2 * (nu(s_, mp.log(4)) - nu(s_, mp.log(2)) - mp.gamma(s_))  # as printed
        c["HL_fix"] = 2 * nu(s_, mp.log(4)) - nu(s_, mp.log(2)) - mp.gamma(s_)
        c["HR_tab"] = 2 * nu(s_, -mp.log(mp.mpf(0.75))) - nu(s_, mp.log(2))
    if name == "Gumbel":
        c["mu"] = g
        c["I"] = lambda p: p * mp.log(-mp.log(p)) + Ei(mp.log(p))  # as printed (sign?)
        c["I_fix"] = lambda p: -p * mp.log(-mp.log(p)) + Ei(mp.log(p))
        c["h"] = lambda p: g - 2 * p * mp.log(-mp.log(p)) - 2 * Ei(mp.log(p)) - (2 * p - 1) * mp.log(-mp.log(p))
        c["h_fix"] = lambda p: g - 2 * Ei(mp.log(p)) - mp.log(-mp.log(p))
        c["G"] = (g + mp.log(mp.log(2))) / (g + 2 * mp.log(mp.log(2)) - 2 * Ei(-mp.log(2)))
        c["G_fix"] = (g + mp.log(mp.log(2))) / (g - mp.log(mp.log(2)) - 2 * Ei(-mp.log(2)))
    if name == "Weibull":
        s_ = 1 + 1 / a
        c["mu"] = mp.gamma(s_)
        c["I"] = lambda p: mp.gamma(s_) - nu(s_, -mp.log(p))   # as printed
        c["I_fix"] = lambda p: nu(s_, -mp.log(1 - p))
        c["h"] = lambda p: mp.gamma(s_) - 2 * nu(s_, -mp.log(1 - p)) + (2 * p - 1) * (-mp.log(1 - p)) ** (1 / a)
        c["G"] = (mp.gamma(s_) - mp.log(2) ** (1 / a)) / (mp.gamma(s_) - 2 * nu(s_, mp.log(2)))
        c["HL_tab"] = 2 * nu(s_, mp.log(4)) - nu(s_, mp.log(2)) - mp.gamma(s_)
        c["HL_fix"] = 2 * nu(s_, -mp.log(mp.mpf(0.75))) - nu(s_, mp.log(2))
        c["HR_tab"] = 2 * nu(s_, -mp.log(mp.mpf(0.75))) - nu(s_, mp.log(2))
        c["HR_fix"] = mp.gamma(s_) + nu(s_, mp.log(2)) - 2 * nu(s_, mp.log(4))
    return c

def rel(a, b):
    return float(abs(a - b) / max(abs(b), mp.mpf("1e-30")))

cases = [("ParetoI", 1.5, None), ("ParetoI", 3.0, None), ("Lomax", 1.5, None), ("Lomax", 2.7, None),
         ("LogLogistic", 1.5, None), ("LogLogistic", 3.0, None), ("Burr", 3.0, 1.5), ("Burr", 1.5, 1.0),
         ("Frechet", 1.5, None), ("Frechet", 4.0, None), ("Gumbel", None, None),
         ("Weibull", 1.5, None), ("Weibull", 2.5, None)]

if __name__ == "__main__":
    for name, a, k in cases:
        n = num(name, a if a else 1, k)
        c = closed(name, a if a else 1, k)
        out = [f"{name:12s} a={a} k={k}"]
        out.append(f"mu:{rel(c['mu'], n['mu']):.1e}")
        for key in ("I", "I_fix", "h", "h_fix"):
            if key in c:
                errs = []
                for p in (mp.mpf(1)/4, mp.mpf(1)/2, mp.mpf(3)/4):
                    ref = n["I"](p) if key.startswith("I") else n["H"](p)
                    errs.append(rel(c[key](p), ref))
                out.append(f"{key}:{max(errs):.1e}")
        for key in ("G", "G_alpha", "G_fix"):
            if key in c:
                out.append(f"{key}:{rel(c[key], n['G']):.1e}")
        for key in ("HL_tab", "HL_fix"):
            if key in c:
                out.append(f"{key}:{rel(c[key], n['HL']):.1e}")
        for key in ("HR_tab", "HR_fix"):
            if key in c:
                out.append(f"{key}:{rel(c[key], n['HR']):.1e}")
        out.append(f"G={float(n['G']):.4f} K={float(n['K']):.4f} 2K={2*float(n['K']):.4f} t3={float(n['t3']):.4f} t4={float(n['t4']):.4f}")
        print("  ".join(out))
