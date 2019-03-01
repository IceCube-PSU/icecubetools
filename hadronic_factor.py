#!/usr/bin/env python
# pylint: disable=line-too-long

# Translated from C++ file accessible via:
# from icecube.multinest_icetray import HadronicFactorEMHad

from __future__ import absolute_import, division, print_function

__all__ = ['hadronic_factor_func_of_em_energy', 'hadronic_factor_func_of_hadr_energy']

import numpy as np
from weave import inline


HADR2EM_C = """
double hadronic_factor_func_of_hadr_energy(double hadr_energy) {
    const double E0 = 0.18791678;
    const double m = 0.16267529;
    const double f0 = 0.30974123;
    const double energy_threshold = 2.71828183;

    double energy = ( hadr_energy < energy_threshold ? energy_threshold : hadr_energy );
    double hadronic_factor = 1 - pow(energy / E0, -m) * (1 - f0);
    return hadronic_factor;
}
"""

EM2HADR_C = """
bool finished = false;
const double precision=1.e-6;

// The hadronic_factor is always between 0.5 and 1.0
double hadr_energy_min = em_energy / 1.0;
double hadr_energy_max = em_energy / 0.5;

while ((hadr_energy_max - hadr_energy_min) / hadr_energy_min > precision) {
    double hadr_energy_cur = (hadr_energy_max + hadr_energy_min) / 2;
    double estimated_em_energy_cur = hadr_energy_cur * hadronic_factor_func_of_hadr_energy(hadr_energy_cur);
    if(estimated_em_energy_cur < em_energy) {
        hadr_energy_min = hadr_energy_cur;
    }
    else if(estimated_em_energy_cur > em_energy) {
        hadr_energy_max = hadr_energy_cur;
    }
    else /* if(estimated_em_energy_cur == em_energy) */ {
        hfactors.append(hadronic_factor_func_of_hadr_energy(hadr_energy_cur));
        finished = true;
        break;
    }
}
if (! finished) {
    hfactors.append(hadronic_factor_func_of_hadr_energy((hadr_energy_max + hadr_energy_min) / 2));
}
"""


def hadronic_factor_func_of_em_energy(em_energies):
    is_scalar = np.isscalar(em_energies)
    if is_scalar:
        em_energies = np.array([em_energies])

    hfactors = []
    for em_energy in em_energies:
        em_energy = float(em_energy)
        inline(EM2HADR_C, ['hfactors', 'em_energy'], support_code=HADR2EM_C)

    if is_scalar:
        hfactors = hfactors[0]
    else:
        hfactors = np.array(hfactors)

    return hfactors


def hadronic_factor_func_of_hadr_energy(hadr_energies):
    E0 = 0.18791678
    m = 0.16267529
    f0 = 0.30974123
    energy_threshold = 2.71828183

    is_scalar = np.isscalar(hadr_energies)
    if is_scalar:
        hadr_energies = np.array([hadr_energies])

    energy = np.where(hadr_energies < energy_threshold, energy_threshold, hadr_energies)
    hadronic_factor = 1 - (energy / E0)**(-m) * (1 - f0)

    if is_scalar:
        hadronic_factor = hadronic_factor[0]

    return hadronic_factor


def test():
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt

    energies = np.logspace(0, 3, 1000)
    hadronic_factors_vs_em = hadronic_factor_func_of_em_energy(energies)
    hadronic_factors_vs_hadr = hadronic_factor_func_of_hadr_energy(energies)

    fig, ax0 = plt.subplots(1, 1, figsize=(8, 5))
    #ax0, ax1 = axes
    ylabel = (
        r'$\bar F$'
        + ' : Average relative luminosity of\n(hadr cascade) / (equally energetic EM cascade)'
    )

    color = 'C0'
    ax0.plot(energies, hadronic_factors_vs_em, color=color)
    ax0.set_xlabel('Energy of EM cascade (GeV)', color=color)
    ax0.tick_params(axis='x', labelcolor=color)

    ax1 = ax0.twiny()
    color = 'C1'
    ax1.plot(energies, hadronic_factors_vs_hadr, color=color, linestyle='--')
    ax1.set_xlabel('Energy of hadronic cascade (GeV)', color=color)
    ax1.tick_params(axis='x', labelcolor=color)

    for ax in [ax0, ax1]:
        ax.set_xscale('log')
        ax.set_xlim(energies[0], energies[-1])
        ax.set_ylim(0.55, 0.85)
        ax.set_ylabel(ylabel)
        ax.grid(True, which='both')

    fig.tight_layout()
    fig.savefig('hadronic_factor_vs_em_and_hadr_energy.pdf')


if __name__ == "__main__":
    test()
