"""Subgrid-scale diagnostics from mean dissipation rates.

With eps = <2 nu S:S> (DNS), eps_bar = <2 nu Sbar:Sbar> (filtered DNS) and the LES budget
eps = <2 (nu + nu_sgs) Sbar:Sbar>, the ideal subgrid viscosity is

    <nu_sgs_ideal> = (eps - eps_bar) / (2 Sbar:Sbar) = nu (eps / eps_bar - 1),   Sbar:Sbar = eps_bar / (2 nu).
"""
from __future__ import annotations

import numpy as np


def nu_sgs_ideal(eps, eps_filtered, nu: float, guard: float = 1e-30, clip=None):
    """Ideal SGS viscosity from the DNS and filtered-DNS mean dissipation rates."""
    eps = np.asarray(eps, dtype=float)
    ef = np.asarray(eps_filtered, dtype=float)
    ef_safe = np.where(np.abs(ef) > guard, ef, guard)
    out = nu * (eps / ef_safe - 1.0)
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(out, *clip) if clip is not None else out


def sgs_dissipation(eps, eps_filtered):
    """Exact subgrid dissipation eps - eps_bar (the 'production' term 2 nu_sgs Sbar:Sbar)."""
    return np.asarray(eps, dtype=float) - np.asarray(eps_filtered, dtype=float)


def production(nu_sgs, s_bar_sq):
    """Modelled production term 2 nu_sgs Sbar:Sbar."""
    return 2.0 * np.asarray(nu_sgs, dtype=float) * np.asarray(s_bar_sq, dtype=float)


def peak(t, y):
    """(time, value) of the maximum of y(t)."""
    t, y = np.asarray(t, dtype=float), np.asarray(y, dtype=float)
    i = int(np.nanargmax(y))
    return float(t[i]), float(y[i])


def relative_peak_error(t, y, t_ref, y_ref):
    """(peak(y) - peak(y_ref)) / peak(y_ref); negative means the peak is under-estimated."""
    return (peak(t, y)[1] - peak(t_ref, y_ref)[1]) / peak(t_ref, y_ref)[1]
