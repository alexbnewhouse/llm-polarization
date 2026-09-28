"""analysis/power.py: the MDE calculator behind docs/pap/README.md. Tiny simulations only."""
import importlib.util
from pathlib import Path
import pytest

_PATH = Path(__file__).resolve().parents[2] / "analysis" / "power.py"
_spec = importlib.util.spec_from_file_location("analysis_power", _PATH)
power = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(power)

SIMS = 400


def test_t_crit_matches_tables():
    assert power.t_crit(0.05, 14) == pytest.approx(2.1448, abs=1e-3)
    assert power.t_crit(0.05, 10) == pytest.approx(2.2281, abs=1e-3)
    assert power.t_crit(0.05, 1e9) == pytest.approx(1.9600, abs=1e-3)


def test_design_counts_the_frozen_grid():
    dz = power.design(135, 5, arms=1, topics=2)
    assert dz["n"] == 2970                   # dialogues per arm, factorial.md
    assert dz["df_r"] == 10                  # 15 roles - 5 levels
    assert power.design(90, 3, arms=3, topics=1)["df_r"] == 6
    with pytest.raises(ValueError):
        power.design(100)                    # does not split across 3 role variants


@pytest.mark.parametrize("model", ["slope", "strong_vs_control"])
def test_power_rises_with_effect(model):
    r = power.run(30, sd=0.6, icc=0.02, sims=SIMS, seed=1)
    p = [power.power(r["draws"][model], power.truth(model, e)) for e in (0.0, 0.1, 0.2, 0.4)]
    assert p == sorted(p) and p[-1] > p[0] + 0.3


@pytest.mark.parametrize("model", ["slope", "strong_vs_control"])
def test_power_rises_with_n(model):
    small = power.run(15, sd=0.6, icc=0.0, sims=SIMS, seed=2)
    large = power.run(135, sd=0.6, icc=0.0, sims=SIMS, seed=2)
    e = 0.12
    assert power.power(large["draws"][model], power.truth(model, e)) > \
        power.power(small["draws"][model], power.truth(model, e)) + 0.1


def test_size_is_near_alpha_and_mde_shrinks_with_n():
    r = power.run(90, sd=0.6, icc=0.05, sims=2000, seed=3)
    for model in ("slope", "strong_vs_control"):
        assert 0.02 < power.power(r["draws"][model], 0.0) < 0.09
    m135 = power.mde(power.run(135, 0.6, 0.0, sims=SIMS, seed=4)["draws"]["strong_vs_control"], "strong_vs_control")
    m90 = power.mde(power.run(90, 0.6, 0.0, sims=SIMS, seed=4)["draws"]["strong_vs_control"], "strong_vs_control")
    assert m135 < m90


def test_sampled_pre_costs_power():
    greedy = power.run(135, 0.6, 0.0, sims=SIMS, seed=5)["draws"]["strong_vs_control"]
    sampled = power.run(135, 0.6, 0.0, sims=SIMS, seed=5, pre_rho=0.0)["draws"]["strong_vs_control"]
    assert power.mde(sampled, "strong_vs_control") > power.mde(greedy, "strong_vs_control")


def test_cli_prints_the_mde_table(capsys):
    assert power.main(["--sims", "50", "--n-per-cell", "90", "--mde-only"]) == 0
    out = capsys.readouterr().out
    assert "MDE at 80% power" in out and out.count("\n 0.60") == 2    # five and three levels
