#!/usr/bin/env python


import numpy as np
import IPython

uncertaintiesPresent = True
try:
    import uncertainties
except ImportError:
    uncertaintiesPresent = False


__author__ = "J.L. Lanfranchi"
__email__ = "justinlanfranchi@yahoo.com"
__copyright__ = "Copyright 2014 J.L. Lanfranchi"
__credits__ = ["J.L. Lanfranchi"]
__license__ = """Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including without
limitation the rights to use, copy, modify, merge, publish, distribute,
sublicense, and/or sell copies of the Software, and to permit persons to whom
the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE."""


def smartFormat(x, sigFigs=7, sciThresh=[7,6], keepAllSigFigs=False,
                alignOnDec=False, threeSpacing=True, alwaysShowSign=False,
                forcedExp=None, demarc=r'$', alignChar=r"&",
                leftSep=r"{,}", rightSep=r"{\,}", decSep=r".",
                nanSub=r"---", inftyThresh=np.infty,
                expLeftFmt=r"{\times 10^{", expRightFmt=r"}}"):

    if hasattr(x, '__iter__'):
        is_iterable = True
        vals = x
    else:
        is_iterable = False
        vals = [x]

    return_vals = []

    for x in vals:
        effectivelyZero = False

        if uncertaintiesPresent:
            if isinstance(x, uncertainties.UFloat):
                x = x.nominal_value

        if np.isinf(x) or x >= inftyThresh:
            if alignOnDec:
                return_vals.append( demarc+r"\infty" + demarc + r" " + alignChar )
                continue
            else:
                return_vals.append( demarc+r"\infty"+demarc )
                continue

        if np.isneginf(x) or -x >= inftyThresh:
            if alignOnDec:
                return_vals.append( demarc+r"-\infty" + demarc + " " + alignChar )
                continue
            else:
                return_vals.append( demarc+r"-\infty"+demarc )
                continue

        elif np.isnan(x):
            if alignOnDec:
                return_vals.append( nanSub + r" " + alignChar )
                continue
            else:
                return_vals.append( nanSub )
                continue

        elif x == 0:
            effectivelyZero = True

        if not effectivelyZero:
            #-- Record sign of number
            sign = np.sign(x)
            
            #-- Create unsigned version of number
            x_unsigned = sign * x

            #-- Find exact (decimal) magnitude
            exact_mag = np.log10(x_unsigned)

            #-- Find floor of exact magnitude (integral magnitude)
            floor_mag = float(np.floor(exact_mag))

            #-- Where is the most significant digit?
            mostSigDig = floor_mag

            #-- Where is the least significant digit?
            leastSigDig = mostSigDig-sigFigs+1
            #print 'mostSigDig',mostSigDig
            #print 'leastSigDig',leastSigDig

            #-- Round number to have correct # of sig figs
            x_u_rnd = np.round(x_unsigned, -int(leastSigDig))

            #------------------------------------------------------------
            # Repeat process from above to find mag, etc.
            #
            #-- Find exact (decimal) magnitude
            exact_mag = np.log10(x_u_rnd)
            #-- Find floor of exact magnitude (integral magnitude)
            floor_mag = float(np.floor(exact_mag))
            #-- Where is the most significant digit?
            mostSigDig = floor_mag
            #-- Where is the least significant digit?
            leastSigDig = mostSigDig-sigFigs+1
            #------------------------------------------------------------

            #-- Mantissa (integral value)
            x_mantissa = np.int64(x_u_rnd * 10**(-leastSigDig))

            #-- Find position of least signficant non-zero digit
            for n in range(sigFigs+1):
                lsnzd = n
                if ((x_mantissa/(10**(n+1))) * 10**(n+1)) != x_mantissa:
                    break

            #print 'lsnzd',lsnzd

            #-- Is the number effectively zero to this many sig figs?
            if lsnzd == sigFigs+1:
                effectivelyZero = True

            #-- Is the number effectively zero given the forced exponent?
            if forcedExp != None:
                if mostSigDig < forcedExp:
                    effectivelyZero = True

            #-- Scale mantissa down to nonzero digits unless keepAllSigFigs is true
            if not keepAllSigFigs:
                x_mantissa = np.int64(x_mantissa * 10**(-lsnzd))
                leastSigDig = int(mostSigDig - len(x_mantissa.__str__()) + 1)
                #print lsnzd, x_mantissa

        #-- If effectively zero, return the formatted string
        if effectivelyZero:
            s = demarc+r"0"

            if alignOnDec:
                s += demarc+r" " + alignChar+" "+demarc
            elif keepAllSigFigs:
                s += decSep

            if keepAllSigFigs:
                z = sepThreeTens(r"0"*(sigFigs-1), dir='right',
                                 leftSep=leftSep, rightSep=rightSep)
                s += z
            elif alignOnDec:
                s += r"0"

            if forcedExp != None and forcedExp != 0:
                s += expLeftFmt + format(forcedExp) + expRightFmt

            s += demarc

            return_vals.append( s )
            continue

        ## TODO: Why does the # of digits trailing decimal matter?
        useSciFmt = False
        if (mostSigDig > sciThresh[0]) or (leastSigDig < -sciThresh[1]):
            useSciFmt = True

        #print 'useSciFmt',useSciFmt

        if useSciFmt:
            numStr = demarc

            #-- Extend the mantissa to include zeros if keepAllSigFigs is true
            if keepAllSigFigs:
                x_mantissa = np.int64(x_mantissa *
                                      10**(sigFigs - len(x_mantissa.__str__())))

            #-- Convert mantissa to a string
            ms = x_mantissa.__str__()

            #-- Sign
            if sign == -1:
                numStr += r"-"
            elif alwaysShowSign:
                numStr += r"+"

            #-- One's place
            numStr += ms[0]

            #-- Decimal point
            if alignOnDec:
                numStr += demarc+" "+alignChar+" "+demarc
            elif (keepAllSigFigs) or (len(ms) > 1):
                numStr += decSep

            #-- Mantissa
            if len(ms) > 1:
                z = sepThreeTens(ms[1:], dir='right', leftSep=leftSep,
                                 rightSep=rightSep)
                numStr += z

            #-- Exponent
            numStr += expLeftFmt + format(int(mostSigDig),'d') + expRightFmt

            numStr += demarc

        else:
            numStr = demarc

            #-- Extend (or contract) the mantissa to include zeros if
            #   keepAllSigFigs is true
            if keepAllSigFigs:
                x_mantissa = np.int64(x_mantissa *
                                      10**(sigFigs - len(x_mantissa.__str__())))

            #-- Where does least significant digit fall now?
            lsd = int(mostSigDig - len(x_mantissa.__str__())+1)

            #-- Extend the mantissa to include any zeros between least sig dig and
            #   the decimal point
            if lsd > 0:
                x_mantissa = np.int64(x_mantissa * 10**(lsd))

                #-- Least significant digit is now at the one's place
                lsd = int(0)

            #-- Convert mantissa to a string
            ms = x_mantissa.__str__()

            msd = int(mostSigDig)

            #-- Extend the mantissa to the left to include any zeros between
            #   the decimal point and the most significant digit (to the right)
            if msd < 0:
                ms = "0"*int(-mostSigDig+1) + ms

                #-- Most significant digit is now at the one's place
                msd = int(0)

            #-- Sign
            if sign == -1:
                numStr += r"-"
            elif alwaysShowSign:
                numStr += r"+"

            #-- Digits left of decimal (ALWAYS one or more)
            numStr += sepThreeTens("".join(ms[0:msd+1]), dir='left',
                                   leftSep=leftSep, rightSep=rightSep)

            #-- Decimal point (force if alignOnDec; otherwise, only include if
            #   the LSD is to the right of the decimal point)
            if alignOnDec:
                numStr += demarc+" "+alignChar+" "+demarc
            elif lsd < 0:
                numStr += decSep

            #-- Digits right of decimal
            if lsd < 0:
                numStr += sepThreeTens("".join(ms[len(ms)+lsd:]), dir='right',
                                       leftSep=leftSep, rightSep=rightSep)

            numStr += demarc

        return_vals.append( numStr )
        continue

        ### <DEBUG>  ##
        #sys.stdout.write('x: ' + format(x) + '  ')
        #sys.stdout.write('x_u: ' + format(x_unsigned) + '  ')
        #sys.stdout.write('log10(x_u): ' + format(exact_mag) + '  ')
        #sys.stdout.write('floor_mag: ' + format(floor_mag) + '  ')
        #sys.stdout.write('x_u_rnd: ' + format(x_u_rnd) + '  ')
        #sys.stdout.write('x_fmt: ' + x_fmt)
        #sys.stdout.write('\n')
        ### </DEBUG> ##

    if not is_iterable:
        return_vals = return_vals[0]
    
    return return_vals

def sepThreeTens(stringifiedNum, dir, leftSep=r"{,}", rightSep=r"{\,}"):
    sFmt = r""

    if dir == 'left':
        R = range(len(stringifiedNum)-1,-1,-1)
        delta = len(stringifiedNum)-1
        sep = leftSep
        for cNum in R:
            sFmt = stringifiedNum[cNum] + sFmt
            if (((delta-cNum)+1) % 3 == 0) and (cNum not in [R[0], R[-1]]):
                sFmt = sep + sFmt

    else:
        R = range(len(stringifiedNum))
        sep = rightSep
        for cNum in R:
            sFmt = sFmt + stringifiedNum[cNum]
            if ((cNum+1) % 3 == 0) and (cNum not in [R[0], R[-1]]):
                sFmt = sFmt + sep

    return sFmt

# TODO: make this work so format, e.g., money w/ 2 dec regardless of $ amount
#def decimalFormat(x, dec=2):
#    x = np.round(x, dec)
#    smartFormat(x, sigFigs=4, sciThresh=[3,3], keepAllSigFigs=False,
#                alignOnDec=False, threeSpacing=False, alwaysShowSign=False,
#                forcedExp=None, demarc='', alignChar="", leftSep="",
#                rightSep="", decSep=r".", nanSub=r"NaN",
#                inftyThresh=np.infty, expLeftFmt=r"e", expRightFmt=""):
 


def texTableFormat(x, sigFigs=7, sciThresh=[7,6], keepAllSigFigs=False,
                   alignOnDec=False, threeSpacing=True,
                   alwaysShowSign=False, forcedExp=None, demarc=r'$',
                   alignChar=r"&", leftSep=r"{,}", rightSep=r"{\,}",
                   decSep=r".", nanSub=r"---", inftyThresh=np.infty,
                   expLeftFmt=r"{\times 10^{", expRightFmt=r"}}"):
    return smartFormat(
        x, sigFigs=sigFigs, sciThresh=sciThresh, keepAllSigFigs=keepAllSigFigs,
        alignOnDec=alignOnDec, threeSpacing=threeSpacing,
        alwaysShowSign=alwaysShowSign, forcedExp=forcedExp, demarc=demarc,
        alignChar=alignChar, leftSep=leftSep, rightSep=rightSep, decSep=decSep,
        nanSub=nanSub, inftyThresh=inftyThresh, expLeftFmt=expLeftFmt,
        expRightFmt=expRightFmt
    )

## TODO: sciThresh ... shouldn't the number on right represent MSD right of dec?
def simpleFormat(x, sigFigs=4, sciThresh=[3,3], keepAllSigFigs=False,
                 alignOnDec=False, threeSpacing=False, alwaysShowSign=False,
                 forcedExp=None, demarc='', alignChar="", leftSep="",
                 rightSep="", decSep=r".", nanSub=r"NaN",
                 inftyThresh=np.infty, expLeftFmt=r"e", expRightFmt=""):
    return smartFormat(
        x, sigFigs=sigFigs, sciThresh=sciThresh, keepAllSigFigs=keepAllSigFigs,
        alignOnDec=alignOnDec, threeSpacing=threeSpacing,
        alwaysShowSign=alwaysShowSign, forcedExp=forcedExp, demarc=demarc,
        alignChar=alignChar, leftSep=leftSep, rightSep=rightSep, decSep=decSep,
        nanSub=nanSub, inftyThresh=inftyThresh, expLeftFmt=expLeftFmt,
        expRightFmt=expRightFmt
    )

def lowPrec(x, sigFigs=4, sciThresh=[4,4], keepAllSigFigs=False,
                 alignOnDec=False, threeSpacing=False, alwaysShowSign=False,
                 forcedExp=None, demarc='', alignChar="", leftSep="",
                 rightSep="", decSep=r".", nanSub=r"NaN",
                 inftyThresh=np.infty, expLeftFmt=r"e", expRightFmt=""):
    return smartFormat(
        x, sigFigs=sigFigs, sciThresh=sciThresh, keepAllSigFigs=keepAllSigFigs,
        alignOnDec=alignOnDec, threeSpacing=threeSpacing,
        alwaysShowSign=alwaysShowSign, forcedExp=forcedExp, demarc=demarc,
        alignChar=alignChar, leftSep=leftSep, rightSep=rightSep, decSep=decSep,
        nanSub=nanSub, inftyThresh=inftyThresh, expLeftFmt=expLeftFmt,
        expRightFmt=expRightFmt
    )

def texLP(x,
          sigFigs=4,
          sciThresh=[4,4],
          keepAllSigFigs=False,
          alignOnDec=False,
          threeSpacing=True,
          alwaysShowSign=False,
          forcedExp=None,
          demarc='',
          alignChar="",
          leftSep="\,",
          rightSep="\,",
          decSep=r".",
          nanSub=r"\mathrm{NaN}",
          inftyThresh=np.infty,
          expLeftFmt=r"e",
          expRightFmt=""):
    return smartFormat(
        x, sigFigs=sigFigs, sciThresh=sciThresh, keepAllSigFigs=keepAllSigFigs,
        alignOnDec=alignOnDec, threeSpacing=threeSpacing,
        alwaysShowSign=alwaysShowSign, forcedExp=forcedExp, demarc=demarc,
        alignChar=alignChar, leftSep=leftSep, rightSep=rightSep, decSep=decSep,
        nanSub=nanSub, inftyThresh=inftyThresh, expLeftFmt=expLeftFmt,
        expRightFmt=expRightFmt
    )


def isInt(x, nDec):
    return np.abs(((x + .5) % 1) - .5) < 10**(-nDec)

def dispSymMath(symexpr, name=None, delimit=r"=", pre=r"", app=r""):
    """Display in IPython a symbolic expression"""
    import sympy
    if name == None:
        IPython.display.display(IPython.display.Math(
            pre+sympy.latex(symexpr)+app))
    else:
        s = r"$"+pre+name+delimit+sympy.latex(symexpr)+app+r"$"
        IPython.display.display(IPython.display.Latex(s))

def symTex(symexpr, name=None, dollars=False):
    """Convert a symbolic (sympy) expression to text (TeX) form"""
    import sympy
    if name == None:
        s = sympy.latex(symexpr)
    else:
        s = name+"="+sympy.latex(symexpr)
    if dollars:
        return r"$" + s + r"$"
    else:
        return s


if __name__ == "__main__":
    print "Test..."
    print smartFormat(0,sigFigs=5,forcedExp=4,keepAllSigFigs=True,
                    alignOnDec=True)
    print smartFormat(0,sigFigs=5,forcedExp=4,keepAllSigFigs=True)
    print smartFormat(0,sigFigs=5,forcedExp=4,keepAllSigFigs=False)
    print smartFormat(0,sigFigs=5,forcedExp=None,keepAllSigFigs=True)
    print smartFormat(0,sigFigs=5,forcedExp=None,keepAllSigFigs=False)
    print smartFormat(0,sigFigs=5,forcedExp=None,keepAllSigFigs=False,
                    alignOnDec=True)
    print ''
    print smartFormat(0.00010001,sigFigs=6,forcedExp=4,keepAllSigFigs=True)
    print smartFormat(0.00010001,sigFigs=6,forcedExp=None,keepAllSigFigs=True)
    print smartFormat(0.00010001,sigFigs=6,forcedExp=None,keepAllSigFigs=False)
    print smartFormat(0.00010001,sigFigs=6,forcedExp=None,keepAllSigFigs=False,
                    sciThresh=[7,8])
    print smartFormat(0.00010001,sigFigs=6,forcedExp=None,keepAllSigFigs=True,
                    sciThresh=[7,20])
    print ''
    print smartFormat(1,sigFigs=5,forcedExp=None,keepAllSigFigs=True)
    print smartFormat(1,sigFigs=5,forcedExp=None,keepAllSigFigs=False)
    print smartFormat(16,sigFigs=5,forcedExp=None,keepAllSigFigs=False)
    print smartFormat(160000,sigFigs=5,forcedExp=None,keepAllSigFigs=False)
    print smartFormat(123456789,sigFigs=15,forcedExp=None,keepAllSigFigs=False,
                    sciThresh=[20,20])
    print smartFormat(1.6e6,sigFigs=5,forcedExp=None,keepAllSigFigs=False)
    print smartFormat(1e6,sigFigs=5,forcedExp=None,keepAllSigFigs=False)
    print smartFormat(0.00134,sigFigs=5,forcedExp=None,keepAllSigFigs=False)
    print ''
    print smartFormat(1)
    print smartFormat(12)
    print smartFormat(123)
    print smartFormat(1234)
    print smartFormat(12345)
    print smartFormat(123456)
    print smartFormat(1234567)
    print smartFormat(12345678)
    print smartFormat(123456789)
    #print smartFormat(0.1e-3)
    #print smartFormat(-0.1e-1,5)
    #print smartFormat(-0.1e-0,5)
    #print ''
    #print smartFormat(-0.1e-3,4)
    #print smartFormat(-0.1e-3,3)
    #print smartFormat(-0.1e-3,2)
    #print smartFormat(-0.1e-3,1)
    #print ''
    #print smartFormat(-0.1e-3,4)
    #print smartFormat(-0.12e-3,4)
    #print smartFormat(-0.123e-3,4)
    #print smartFormat(-0.1234e-3,4)
    #print smartFormat(-0.12345e-3,4)
    #print ''
    #print smartFormat(-1.2345678e0,5)
    #print smartFormat(-1.2345678e1,5)
    #print smartFormat(-1.2345678e2,5)
    #print smartFormat(-1.2345678e3,5)
    #print smartFormat(-1.2345678e4,5)
    #print smartFormat(-1.2345678e5,5)
    #print smartFormat(-1.2345678e6,5)

