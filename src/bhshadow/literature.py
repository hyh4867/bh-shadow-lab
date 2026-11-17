"""Literature values copied from the papers (page/table in docs/PREREGISTRATION.md section 2)."""

# EHT Paper VI (ApJL 875, L6), p. 23: combined ring diameter
D_EHT, D_EHT_ERR = 42.0, 3.0
# Paper VI p.10 Table 3, dynesty xs-ring (closure-only) fits: (median, minus, plus) in muas
TABLE3_XSRING = {
    "2017-04-05 hi": (42.9, 0.54, 0.59), "2017-04-05 lo": (43.5, 0.41, 0.44),
    "2017-04-06 hi": (43.3, 0.43, 0.44), "2017-04-06 lo": (43.4, 0.26, 0.27),
    "2017-04-10 hi": (43.3, 0.65, 0.79), "2017-04-10 lo": (44.4, 0.87, 0.84),
    "2017-04-11 hi": (43.4, 0.48, 0.74), "2017-04-11 lo": (41.6, 0.46, 0.51),
}
TABLE3_OBS_FRAC = 0.017  # observational uncertainty +1.75% / -1.69%, rounded
# Paper VI p.10 Table 3 brightness PA (deg E of N), dynesty xs-ring, for reference only
TABLE3_XSRING_PA = {
    "2017-04-05 hi": 160.5, "2017-04-05 lo": 157.9, "2017-04-06 hi": 149.1, "2017-04-06 lo": 148.5,
    "2017-04-10 hi": 194.3, "2017-04-10 lo": 204.0, "2017-04-11 hi": 180.1, "2017-04-11 lo": 175.9,
}
# Paper VI p.13 Table 4, xs-ring MAD+SANE
ALPHA_GC, ALPHA_GC_FRAC = 11.56, 0.114
# Paper VI p.13: photon-ring emission alpha range 9.6-10.4
ALPHA_PR, ALPHA_PR_ERR = 10.0, 0.4
ALPHA_SCHWARZSCHILD = 2 * 27**0.5
# Paper VI p.21 Table 7 note: distance
DIST_MPC, DIST_ERR = 16.8, 0.8

# Masses in 1e9 Msun at D = 16.8 Mpc: (value, minus, plus, source)
MASSES = {
    "EHT Paper VI (2019)": (6.5, 0.7, 0.7, "Paper VI p.21 Table 7"),
    "Gebhardt+2011 stars": (6.6 * 16.8 / 17.9, 0.4 * 16.8 / 17.9, 0.4 * 16.8 / 17.9,
                            "ApJ 729, 119; 6.6±0.4 at 17.9 Mpc, scaled by D"),
    "Walsh+2013 gas": (3.5 * 16.8 / 17.9, 0.7 * 16.8 / 17.9, 0.9 * 16.8 / 17.9,
                       "ApJ 770, 86; 3.5 (+0.9 -0.7) at 17.9 Mpc, scaled by D"),
    "Liepold+2023 stars": (5.37, 0.25, 0.37, "ApJL 945, L35; D = 16.8 Mpc"),
    "Simon+2024 stars": (8.7, (1.2**2 + 1.3**2) ** 0.5, (1.2**2 + 1.3**2) ** 0.5,
                         "MNRAS 527, 2341; random and systematic added in quadrature; D = 16.8 Mpc"),
}
