"""bhshadow: ring-diameter measurement of M87* from EHT 2017 public data."""

MUAS = 4.84813681109536e-12  # radians per micro-arcsecond
# 2017 codes (AP = APEX, JC = JCMT, SM = SMA) and 2018 codes (AX = APEX, MM = JCMT, SW = SMA)
INTRA_SITE = {frozenset(("AA", "AP")), frozenset(("JC", "SM")), frozenset(("AA", "AX")), frozenset(("MM", "SW"))}
DAYS = {"095": "2017-04-05", "096": "2017-04-06", "100": "2017-04-10", "101": "2017-04-11"}
