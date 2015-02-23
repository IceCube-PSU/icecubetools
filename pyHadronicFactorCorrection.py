#!/usr/bin/env python

# Translated from C++ file accessible via:
# from icecube.multinest_icetray import HadronicFactorEMHad

import numpy as np
from scipy.weave import inline, converters

##@profile
#def HadronicFactorHadEM(HadronicEnergy):
#    E0 = 0.18791678
#    m  = 0.16267529
#    f0 = 0.30974123
#    e  = 2.71828183
#
#    #-- Plain python implementation:
#    #en = e if (e > HadronicEnergy) else HadronicEnergy
#
#    #-- Numpy implementation:
#    isNPArray = True
#    if not hasattr(HadronicEnergy, '__iter__'):
#        isNPArray = False
#        HadronicEnergy = np.array([HadronicEnergy])
#    en = np.where(e > HadronicEnergy, e, HadronicEnergy)
#    HF = 1 - (en/E0)**(-m) * (1-f0)
#
#    if not isNPArray:
#        HF = HF[0]
#
#    return HF

HadEM_c = """
    double HadronicFactorHadEM(double HadronicEnergy){
        const double E0 = 0.18791678;
        const double  m = 0.16267529;
        const double f0 = 0.30974123;
        const double  e = 2.71828183;
        
        double en = ( e > HadronicEnergy ? e : HadronicEnergy );
        double HF = 1 - pow(en/E0,-m) * (1-f0);
        return HF;
    }
"""

EMHad_c = """
    //py::list hfactors;
    bool finished = false;

    const double precision=1.e-6;

    // The HadronicFactor is always between 0.5 and 1.0
    double HadronicEnergy_max = EMEnergy_n/0.5;
    double HadronicEnergy_min = EMEnergy_n/1.0;
    double EstimatedEnergyEM_max = HadronicEnergy_max*HadronicFactorHadEM(HadronicEnergy_max);
    double EstimatedEnergyEM_min = HadronicEnergy_min*HadronicFactorHadEM(HadronicEnergy_min);
    //if(EstimatedEnergyEM_max < EMEnergy_n || EstimatedEnergyEM_min > EMEnergy_n){
    //    log_warn("Problem with boundary definition for hadronic factor calculation."
    //            "Returning NAN.");
    //    return NAN;
    //}
    while((HadronicEnergy_max-HadronicEnergy_min)/HadronicEnergy_min > precision){
        double HadronicEnergy_cur = (HadronicEnergy_max+HadronicEnergy_min)/2;
        double EstimatedEnergyEM_cur = HadronicEnergy_cur*HadronicFactorHadEM(HadronicEnergy_cur);
        if(EstimatedEnergyEM_cur < EMEnergy_n){
            HadronicEnergy_min = HadronicEnergy_cur;
            EstimatedEnergyEM_min = EstimatedEnergyEM_cur;
        }
        else if(EstimatedEnergyEM_cur > EMEnergy_n){
            HadronicEnergy_max = HadronicEnergy_cur;
            EstimatedEnergyEM_max = EstimatedEnergyEM_cur;
        }
        else /* if(EstimatedEnergyEM_cur == EMEnergy_n) */{
            //return HadronicFactorHadEM(HadronicEnergy_cur);
            hfactors.append(HadronicFactorHadEM(HadronicEnergy_cur));
            finished = true;
            break;
        }
    }
    if (! finished){
        hfactors.append(HadronicFactorHadEM((HadronicEnergy_max+HadronicEnergy_min)/2));
    }
    //return HadronicFactorHadEM((HadronicEnergy_max+HadronicEnergy_min)/2);
"""


#@profile
def HadronicFactorEMHad(EMEnergy):
    isNPArray = True
    if not hasattr(EMEnergy, '__iter__'):
        isNPArray = False
        EMEnergy = np.array([EMEnergy])

    precision = 1.e-6

    #if(EstimatedEnergyEM_max < EMEnergy || EstimatedEnergyEM_min > EMEnergy){
    #    log_warn("Problem with boundary definition for hadronic factor calculation."
    #            "Returning NAN.");
    #    return NAN;
    #}
  
    hfactors = []
    for EMEnergy_n in EMEnergy:
        EMEnergy_n = float(EMEnergy_n)
        inline(EMHad_c, ['hfactors', 'EMEnergy_n'], support_code=HadEM_c)
        """
        #-- The HadronicFactor is always between 0.5 and 1.0
        HadronicEnergy_max = EMEnergy_n/0.5
        HadronicEnergy_min = EMEnergy_n/1.0

        EstimatedEnergyEM_max = HadronicEnergy_max*HadronicFactorHadEM(HadronicEnergy_max)
        EstimatedEnergyEM_min = HadronicEnergy_min*HadronicFactorHadEM(HadronicEnergy_min)

        finished = False
        while ((HadronicEnergy_max-HadronicEnergy_min)/HadronicEnergy_min > precision):
            HadronicEnergy_cur = (HadronicEnergy_max+HadronicEnergy_min)/2
            EstimatedEnergyEM_cur = HadronicEnergy_cur*HadronicFactorHadEM(HadronicEnergy_cur)
            if(EstimatedEnergyEM_cur < EMEnergy_n):
                HadronicEnergy_min = HadronicEnergy_cur
                EstimatedEnergyEM_min = EstimatedEnergyEM_cur

            elif (EstimatedEnergyEM_cur > EMEnergy_n):
                HadronicEnergy_max = HadronicEnergy_cur
                EstimatedEnergyEM_max = EstimatedEnergyEM_cur

            else: # if(EstimatedEnergyEM_cur == EMEnergy_n)
                hfactors.append(HadronicFactorHadEM(HadronicEnergy_cur))
                finished = True
                break
        if not finished: 
            hfactors.append(HadronicFactorHadEM((HadronicEnergy_max+HadronicEnergy_min)/2))
        """
    
    if isNPArray:
        hfactors = np.array(hfactors)
    else:
        hfactors = hfactors[0]

    return hfactors

if __name__ == "__main__":
    x = np.array([1.2, 2.2, 3.2])
    y = HadronicFactorEMHad(x)
    print x, y
