#!/usr/bin/env python

import numpy as np
try:
    import matplotlib.pyplot as plt
    import histogramTools as HIST 
except:
    pass
import copy


DARK_BLUE = (0.0, 0.0, 0.7)
DARK_RED =  (0.7, 0.0, 0.0)

LIGHT_BLUE = (0.4, 0.4, 0.8)
LIGHT_RED =  (0.8, 0.4, 0.4)


def error(df, err_col, ref_col, left_bin_edges=None, right_bin_edges=None,
               ref_min=None, ref_max=None, ref_win_width=None, ref_step=None,
               res_type=None, store_data=False):
    '''
    Take a Pandas DataFrame and extract "resolution" (median of absolute value)
    of a column as a function of another column. Arbitrary binning (including
    overlapped binning) is possible, but weighting (i.e., windows other than
    boxcar) is not supported.

    res_type: Valid options are 'med', 'rms'
    '''
    # Sort the DataFrame by the values in the reference column
    sorted_df = df[[ref_col, err_col]].loc[df[ref_col].order().index]

    if (left_bin_edges is None) or (right_bin_edges is None):
        # Find values where left- and right-sides of bins will lie
        left_bin_edges  = np.arange(ref_min,               ref_max-ref_win_width+ref_step/1e3, ref_step)
        right_bin_edges = np.arange(ref_min+ref_win_width, ref_max+ref_step/1e3,               ref_step)
    
    # Locate the 'insert' indices where the left- and right-sides of the bins should lie. I.e., these indices 
    left_inds  = np.searchsorted(sorted_df[ref_col].values, left_bin_edges, side='left')
    right_inds = np.searchsorted(sorted_df[ref_col].values, right_bin_edges, side='right') - 1

    if len(left_inds) != len(right_inds):
        print 'left_inds:', left_inds
        print 'right_inds:', right_inds


def resolution(df, err_col, ref_col, left_bin_edges=None, right_bin_edges=None,
               ref_min=None, ref_max=None, ref_win_width=None, ref_step=None,
               res_type=None, store_data=False):
    '''
    Take a Pandas DataFrame and extract "resolution" (median of absolute value)
    of a column as a function of another column. Arbitrary binning (including
    overlapped binning) is possible, but weighting (i.e., windows other than
    boxcar) is not supported.

    res_type: Valid options are 'med', 'rms'
    '''
    # Sort the DataFrame by the values in the reference column
    sorted_df = df[[ref_col, err_col]].loc[df[ref_col].order().index]

    if (left_bin_edges is None) or (right_bin_edges is None):
        # Find values where left- and right-sides of bins will lie
        left_bin_edges  = np.arange(ref_min,               ref_max-ref_win_width+ref_step/1e3, ref_step)
        right_bin_edges = np.arange(ref_min+ref_win_width, ref_max+ref_step/1e3,               ref_step)
    
    # Locate the 'insert' indices where the left- and right-sides of the bins should lie. I.e., these indices 
    left_inds  = np.searchsorted(sorted_df[ref_col].values, left_bin_edges, side='left')
    right_inds = np.searchsorted(sorted_df[ref_col].values, right_bin_edges, side='right') - 1

    if len(left_inds) != len(right_inds):
        print 'left_inds:', left_inds
        print 'right_inds:', right_inds
   
    res = []
    N = []
    data = []
    if res_type is 'med':
        for n in range(len(left_inds)):
            inds = range(left_inds[n],right_inds[n]+1)
            N.append(len(inds))
            d = sorted_df[err_col].iloc[inds]
            res.append(d.abs().median())
            if store_data:
                data.append(d.values)
    elif res_type is 'rms':
        for n in range(len(left_inds)):
            inds = range(left_inds[n],right_inds[n]+1)
            N.append(len(inds))
            d = sorted_df[err_col].iloc[inds]
            res.append(np.sqrt(d.pow(2).mean()))
            if store_data:
                data.append(d.values)
    else:
        raise Exception('Resolution res_type \'' + str(res_type) + '\' not implemented.')

    return {'res'             : np.array(res),
            'bin_centers'     : (left_bin_edges+right_bin_edges)/2.0,
            'left_bin_edges'  : left_bin_edges,
            'right_bin_edges' : right_bin_edges,
            'N'               : np.array(N),
            'data'            : data}


def uncertainty(res, N, conf=0.683, data=None, kind='root_n'):
    '''
    Compute resolutions' uncertainties.

    Parameters
    ----------
    res:  Resolutions

    N:    Number of points that contributed to each resolution 

    conf: Confidence level of uncertainty (default=0.683, or one sigma)

    data: List of arrays of the points that contributed to each resolution.
          Only necessary if using Monte Carlo method.

    kind: How to compute the uncertainty. Can be one of:
          'root_n': Reciprocal of square root of the number of points in each
                    bin (default) 
          'mc'    : Monte Carlo (?)

    Returns
    -------
    Dictionary containing 'lower_unc' and 'upper_unc', the respective lower-
    and upper-bounds on uncertainties. Each is a Numpy array.
    '''
    if kind is 'root_n':
        raise Exception('Unrecognized kind: \'' + str(kind) + '\'')
        # TODO: Compute uncertainties!
        pass
        #upper_unc = res
        #if plot_type is 'line':
        #    res_ax.fill_between(centers, res-1/np.sqrt(N), res+1/np.sqrt(N), alpha=0.5)
    elif kind is 'mc':
        raise Exception('Unrecognized kind: \'' + str(kind) + '\'')
        pass
    elif kind is None:
        pass
    else:
        raise Exception('Unrecognized kind: \'' + str(kind) + '\'')
    return {'lower_unc':lower_unc, 'upper_unc':upper_unc}


class ResPlot:
    '''
    Makes useful plots of resolutions for one dataset (or more datasets, for comparison purposes):
    * Resolution vs. reference parameter bin
    * Efficiency and resolution as percentages of reference 
    * Number of events vs. reference parameter bin
    '''
    def __init__(self, plot_type='line',
                 x_lim=None, xparam_tex='', xparam_units_tex='',
                 err_tex='', err_units_tex='',
                 plt_res=True, res_fig=None, res_ax=None, res_lim=None,
                 plt_fract=True, fract_fig=None, fractres_ax=None, fractres_lim=None, fractnum_ax=None, fractnum_lim=None,
                 plt_num=True, num_fig=None, num_ax=None, num_lim=None, **kwargs):
        
        self.plot_type = plot_type
        self.plt_res = plt_res
        self.plt_fract = plt_fract
        self.plt_num = plt_num

        if plt_res and (res_ax is None):
            if res_fig is None:
                self.res_fig = plt.figure()
            else:
                self.res_fig = res_fig
            self.res_fig.clf()
            self.res_ax = self.res_fig.add_subplot(111)
        else:
            self.res_ax = res_ax
        
        if plt_fract and ((fractres_ax is None) or (fractnum_ax is None)):
            if fract_fig is None:
                self.fract_fig = plt.figure()
            else:
                self.fract_fig = fract_fig
            self.fract_fig.clf()
            self.fractres_ax = self.fract_fig.add_subplot(111)
            self.fractnum_ax = self.fractres_ax.twinx()
        else:
            self.fractres_ax = fractres_ax
            self.fractnum_ax = fractnum_ax
        
        if plt_num and (num_ax is None):
            if num_fig is None:
                self.num_fig = plt.figure()
            else:
                self.num_fig = num_fig
            self.num_fig.clf()
            self.num_ax = self.num_fig.add_subplot(111)
        else:
            self.num_ax = num_ax

        self.x_lim = x_lim
        self.res_lim = res_lim
        self.fractres_lim = fractres_lim
        self.fractnum_lim = fractnum_lim
        self.num_lim = num_lim

        self.ref = None
        self.comp = []

        self.xparam_tex = xparam_tex
        self.err_tex = err_tex
        self.xparam_units_tex = xparam_units_tex
        self.err_units_tex = err_units_tex
    
    def addRef(self, res, left_bin_edges=None, right_bin_edges=None, N=None,
               #lower_unc=None, upper_unc=None, unc_kwargs={},
               cut_label_tex='', plot_kwargs={'color':DARK_BLUE}, **kwargs):
        # NOTE: This *may* interfere with a second axis tied to first axis, as
        # I've used for res plots to plot both E_nu_fract and coszen
        # resolutions on one plot
        #self.res_ax.cla()
        #self.fractres_ax.cla()
        #self.fractnum_ax.cla()
        #self.num_ax.cla()
        
        self.ref = {
            'res':res,
            'left_bin_edges':left_bin_edges,
            'right_bin_edges':right_bin_edges,
            'N':N,
            #'lower_unc':lower_unc,
            #'upper_unc':upper_unc,
            #'unc_kwargs':unc_kwargs,
            'cut_label_tex':cut_label_tex,
            'plot_kwargs':plot_kwargs,
        }
       
        self.plotComp(self.ref)
        #self.ref['res_plot'] = plot(res, plot_type=self.plot_type, ax=self.res_ax, **self.ref)
        
        #if not (N is None):
        #self.ref['num_plot'] = plot(N, plot_type=self.plot_type, ax=self.num_ax, **self.ref)
        #else:
        #    self.ref['num_plot'] = None
         
        # New plots for any already-existing comparison datasets
        for comp in self.comp:
            self.plotComp(comp)
    
    def addComp(self, res, N, cut_label_tex='', plot_kwargs={'color':LIGHT_BLUE}, **kwargs):
        comp = {
            'res':res,
            'left_bin_edges':self.left_bin_edges,
            'right_bin_edges':self.right_bin_edges,
            'N':N,
            #'lower_unc':lower_unc,
            #'upper_unc':upper_unc,
            #'unc_kwargs':unc_kwargs,
            'cut_label_tex':cut_label_tex,
            'plot_kwargs':plot_kwargs
        }
        self.plotComp(comp)
        self.comp.append(comp)
    
    def plotComp(self, comp):
        augment_label_strs = []
        fractres_str = ''
        fractnum_str = ''
        
        comp['fractres'] = None
        if self.plt_fract and not (comp['res'] is None)  and (comp != self.ref):
            if self.ref['res']:
                # Compute fractional change in resolution
                comp['fractres'] = (comp['res']-self.ref['res'])/self.ref['res']
                avg_fractres = np.mean(comp['fractres'])

                fractres_str = r',\,\langle\Delta(\mathrm{MAE})/\mathrm{MAE}_0\rangle='+numFmt(avg_fractres)
                augment_label_strs.append(fractres_str)
        
        comp['fractnum'] = None
        if self.plt_fract and not (comp['N'] is None) and (comp != self.ref):
            comp['num_plot'] = plot(y=comp['N'], plot_type=self.plot_type, **comp)
            if self.ref['N']:
                comp['fractnum'] = (comp['N']-self.ref['N'])/self.ref['N']
                avg_fractnum = np.mean(comp['fractnum'])

                fractnum_str = r',\,\langle\Delta N/N_0\rangle='+numFmt(avg_fractnum)
                augment_label_strs.append(fractnum_str)

        # Update label to include avg. cut fract. eff. gain & avg cut fract. gain
        if len(augment_label_strs) > 0:
            augment_label_str = r''.join(augment_label_strs)
        else:
            augment_label_str = r''
       
        tex_str = comp['cut_label_tex'] + augment_label_str
        if len(tex_str.strip()) == 0: res_label = self.err_tex
        else: res_label = r'$' + self.err_tex + r',\,' + tex_str + r'$'
        
        tex_str = comp['cut_label_tex']
        if len(tex_str.strip()) == 0: num_label = None
        else: num_label = r'$' + tex_str + r'$'
         
        tex_str = self.err_tex + r',\,' + comp['cut_label_tex'] + fractres_str
        if len(tex_str.strip()) == 0: fractres_label = None
        else: fractres_label = r'$' + tex_str + r'$'
        
        tex_str = self.err_tex + r',\,' + comp['cut_label_tex'] + fractnum_str
        if len(tex_str.strip()) == 0: fractres_label = None
        else: fractres_label = r'$' + tex_str + r'$'
        
        #
        # Make the plots!
        #
        
        # Basic resolution plot
        if self.plt_res and not (comp['res'] is None):
            # Plot
            comp['res_plot'] = plot(y=comp['res'], plot_type=self.plot_type, ax=self.res_ax, **comp)

            # Force label
            print 'res_label', res_label
            comp['res_plot']['legend_line'].set_label(res_label)

        # Basic number plot
        if self.plt_num and not (comp['N'] is None):
            # Plot
            comp['num_plot'] = plot(y=comp['N'], plot_type=self.plot_type, ax=self.num_ax, **comp)
            
            # Force label
            print 'num_label', num_label
            comp['num_plot']['legend_line'].set_label(num_label)

        linestyles = ['-', '--']

        # Fractional delta-resolution plot
        if self.plt_fract and not (comp['fractres'] is None):
            # Plot
            comp['fractres_plot'] = plot(y=comp['fractres'], plot_type=self.plot_type, ax=self.fractres_ax, **d)
            
            # Force linestyle to first available linestyle
            ls = linestyles.pop(0)
            [ l.set_linestyle(ls) for l in comp['fractres_plot']['all_lines'] ]

            # Force label
            print 'fractres_label', fractres_label
            comp['fractres_plot']['legend_line'].set_label(fractres_label)


        # Fractional delta-number plot
        if self.plt_fract and not (comp['fractnum'] is None):
            # Plot
            comp['fractnum_plot'] = plot(y=comp['fractnum'], plot_type=self.plot_type, ax=self.fractnum_ax, **d)

            # Force linestyle to first available linestyle
            ls = linestyles.pop(0)
            [ l.set_linestyle(ls) for l in comp['fractnum_plot']['all_lines'] ]

            # Force label
            print 'fractnum_label', fractnum_label
            comp['fractnum_plot']['legend_line'].set_label(fractnum_label)

        self.finishPlots()

    def finishPlots(self):
        # Set axes limits
        if self.plt_res:
            if not (self.x_lim is None): self.res_ax.set_xlim(self.x_lim)
            if not (self.res_lim is None): self.res_ax.set_ylim(self.res_lim)
            self.res_leg = self.res_ax.legend(loc='best')
       
        if self.plt_num:
            if not (self.x_lim is None): self.num_ax.set_xlim(self.x_lim)
            if not (self.num_lim is None): self.num_ax.set_ylim(self.num_lim)
            self.num_leg = self.num_ax.legend(loc='best')
       
        if self.plt_fract:
            if not (self.x_lim is None): self.fractres_ax.set_xlim(self.x_lim)
            if not (self.fractres_lim is None): self.fractres_ax.set_ylim(self.fractres_lim)
            if not (self.x_lim is None): self.fractnum_ax.set_xlim(self.x_lim)
            if not (self.fractnum_lim is None): self.fractnum_ax.set_ylim(self.fractnum_lim)
            self.fractres_leg = self.fractres_ax.legend(loc='lower left')
            self.fractnum_leg = self.fractnum_ax.legend(loc='upper left')

#def deltaResEffFractPlot(x, ref_res, ref_N, comp_res, comp_N, comp_centers,
#                         marked_x=None, ax=None,
#                         res_plot_kwargs={'color':DARK_BLUE},
#                         num_plot_kwargs={'color':DARK_RED}):
#    if ax is None:
#        f = plt.figure()
#        ax = f.add_subplot(111)
#    delta_res = (comp_res - ref_res)/comp_res
#    delta_N = (comp_N - ref_N)/comp_N
#    res_ret = ax.plot(x, delta_res, **res_plot_kwargs)
#    num_line = ax.plot(x, delta_N, **num_plot_kwargs)
#    return {'ax':ax, 'res_line':res_line, 'num_line':num_line,
#            'delta_res':delta_res, 'delta_N':delta_N}


def plot(y, left_bin_edges, right_bin_edges,
         plot_type='line',
         fig=None, ax=None,
         plot_kwargs={'color':DARK_BLUE}, **kwargs):
    '''
    Parameters
    ----------
    y:  Y values to be plotted
    
    left_bin_edges, right_bin_edges:
        Left and right sides of the bins. Note that the bins can be
        overlapping.
    
    unc_type: How to compute uncertainty bounds
        None     :
        'root_n' :
        'mc'     :
    
    unc_kwargs:
        Passed on to the plot call(s) while plotting uncertaintites
    
    plot_type:
        'line' : 
        'bars' : 
    
    fig, ax:
        If both fig and ax are NOT specified, a new figure is created
        (and a new axis created on that figure).
    
        If fig IS specified but ax is NOT specified, the figure is
        cleared and a new axis is added. This allows for new plots to be
        created without spawning new figures.
    
        Regardless if fig is specified, if axis is specified, plots are
        drawn on the given axis without clearing first. This allows for
        multiple plots to be drawn together seamlessly.
    
    plot_kwargs:
        Passed on when plotting.
    
    Returns
    -------
    Dictionary containing fields 'fig', 'ax', 'legend_line', and
    'all_lines'. This is made to be passed into another call of this function.
    '''
    
    if ax is None:
        if fig is None:
            fig = plt.figure()
        fig.clf()
        ax = fig.add_subplot(111)

    n_bins = len(y)
    centers = (left_bin_edges+right_bin_edges)/2.0
   
    all_lines = []
    legend_line = None

    if plot_type is 'line':
        legend_line = ax.plot(centers, y, **plot_kwargs)[0]
        all_lines.append(legend_line)

    elif plot_type is 'bars':
        binScaleFactor = 0.9
        for bin_n in xrange(n_bins):
            left_edge, right_edge = HIST.scaleBin([left_bin_edges[bin_n], right_bin_edges[bin_n]], binScaleFactor)
            if bin_n == 0:
                legend_line = ax.plot([ left_edge, right_edge ], [ y[bin_n] ]*2, **plot_kwargs)[0]
                all_lines.append(legend_line)
            else:
                all_lines.extend(ax.plot([ left_edge, right_edge ], [ y[bin_n] ]*2, **plot_kwargs))

    else:
        raise Exception('Unrecognized plot_type: \'' + str(plot_type) + '\'')

    return {'fig':fig, 'ax':ax, 'legend_line':legend_line, 'all_lines':all_lines}


#def resPlot(res, left_bin_edges=None, right_bin_edges=None, N=None,
#         lower_unc=None, upper_unc=None, unc_kwargs={}, plot_type='line',
#         res_fig=None, res_ax=None,
#         eff_fig=None, eff_ax=None,
#         label=None, plot_kwargs={'color':'b'}, **kwargs):
#def resPlot(res, left_bin_edges=None, right_bin_edges=None,
#            lower_unc=None, upper_unc=None, unc_kwargs={},
#            plot_type='line',
#            fig=None, res_ax=None,
#            plot_kwargs={'color':'b'}, **kwargs):
#    '''
#    Parameters
#    ----------
#    res:
#        Resolutions computed for each bin.
#
#    left_bin_edges, right_bin_edges:
#        Left and right sides of the bins. Note that the bins can be
#        overlapping.
#
#    unc_type: How to compute uncertainty bounds
#        None     :
#        'root_n' :
#        'mc'     :
#
#    unc_kwargs:
#        Passed on to the plot call(s) while plotting uncertaintites
#
#    plot_type:
#        'line' : 
#        'bars' : 
#
#    fig, ax:
#        For each {res,eff}_fig NOT specified that also has NO corresponding
#        {res,eff}_axis specified, a new figure is created (and a new axis
#        created on that figure).
#    
#        For each {res,eff}_fig specified with NO corresponding {res,eff}_axis
#        specified, the figure is cleared and a new axis is added. This allows
#        for many new plots to be created without an unweildy number of figures.
#    
#        Regardless if one of {res,eff}_fig is specified, if a {res,eff}_axis
#        specified, plots are drawn on the given axis(es) without clearing
#        first. This allows for multiple plots to be drawn together seamlessly.
#
#    plot_kwargs:
#        Passed on to resolution and efficiency but *not* uncertainty plots.
#
#    Returns
#    -------
#    Dictionary containing fields 'fig', 'ax', 'legend_line', and 'all_lines'.
#    This is easily passed into another call of 
#    '''
#
#    if ax is None:
#        if fig is None:
#            fig = plt.figure()
#        else:
#            fig.clf()
#        ax = fig.add_subplot(111)
# 
#    plot_eff = False
#    if not (N is None):
#        plot_eff = True
#        #if N0:
#        #    eff = N/N0
#        #else:
#        eff = N
#    
#    if plot_eff:
#        if eff_ax is None:
#            if eff_fig is None:
#                eff_fig = plt.figure()
#            eff_fig.clf()
#            eff_ax = eff_fig.add_subplot(111)
#    else:
#        eff_ax = None
#        eff_fig = None
#
#    n_bins = len(res)
#    centers = (left_bin_edges+right_bin_edges)/2.0
#   
#    all_lines = []
#    legend_line = None
#
#    if plot_type is 'line':
#        legend_line = ax.plot(centers, res, label=label, **plot_kwargs)[0]
#        all_lines.append(legend_line)
#        if plot_eff:
#            eff_ax.plot(centers, eff, label=label, **plot_kwargs)
#
#    elif plot_type is 'bars':
#        binScaleFactor = 0.9
#        for bin_n in xrange(n_bins):
#            left_edge, right_edge = HIST.scaleBin([left_bin_edges[bin_n], right_bin_edges[bin_n]], binScaleFactor)
#            if bin_n == 0:
#                legend_line = ax.plot([ left_edge, right_edge ], [ res[bin_n] ]*2, label=lab, **plot_kwargs)[0]
#                all_lines.append(legend_line)
#            else:
#                all_lines.extend(ax.plot([ left_edge, right_edge ], [ res[bin_n] ]*2, **plot_kwargs))
#
#            if plot_eff:
#                eff_ax.plot([ left_edge, right_edge ], [ eff[bin_n] ]*2, label=lab, **plot_kwargs)
#
#    else:
#        raise Exception('Unrecognized plot_type: \'' + str(plot_type) + '\'')
#
#    return {'fig':fig, 'ax':ax, 'legend_line':legend_line, 'all_lines':all_lines} #'eff_fig':eff_fig, 'eff_ax':eff_ax}
