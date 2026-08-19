import json
import sys
from pathlib import Path

import numpy as np
import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

REPO_ROOT = PACKAGE_ROOT.parents[1]
POSTERIOR_DB_ROOT = REPO_ROOT / "posterior_database"
STAN_FIXTURES = Path(__file__).resolve().parent / "stan_fixtures"


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(0)


def bridgestan_model(stan_file, data: dict):
    import bridgestan

    return bridgestan.StanModel.from_stan_file(str(stan_file), json.dumps(data))
