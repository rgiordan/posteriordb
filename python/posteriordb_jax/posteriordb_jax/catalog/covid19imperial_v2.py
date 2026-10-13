"""covid19imperial_v2.stan; see _covid19imperial.py (v2 and v3 are the same
Stan program)."""

from ._covid19imperial import covid_model

STAN_MODEL = "covid19imperial_v2"
POSTERIORS = ["ecdc0401-covid19imperial_v2", "ecdc0501-covid19imperial_v2"]


def build(raw):
    return covid_model(name=STAN_MODEL, raw=raw)
