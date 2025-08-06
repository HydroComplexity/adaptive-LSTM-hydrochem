import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.multiprocessing import Process, current_process
import os
import datetime
import utils
from utils import PID
import os.path
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from math import sqrt
import scipy.special as kld
import scipy.stats as cf
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
import matplotlib.cm as cm
from scipy import interpolate
import seaborn as sns
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import kstest
import scipy.stats as stats
from scikit_posthocs import posthoc_dunn
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Rectangle
import statistics as stat




def ttest(obs, pr1, pr2) :
    e1, e2 = obs - pr1, obs - pr2
    t_statistic, p_value = stats.ttest_rel (e1, e2)
    return p_value


def bootstrap_sample(data) :
    sample_with_replacement = np.random.choice (data, size=len (data), replace=True)
    return sample_with_replacement


def bootstrap_analysis(y_true, y_pred1, y_pred2, n_iterations=1000) :
    differences = []

    for _ in range (n_iterations) :
        indices = bootstrap_sample (np.arange (len (y_true)))
        sample_y_true = y_true[indices]
        sample_y_pred1 = y_pred1[indices]
        sample_y_pred2 = y_pred2[indices]

        error1 = mean_squared_error (sample_y_true, sample_y_pred1)
        error2 = mean_squared_error (sample_y_true, sample_y_pred2)

        differences.append (error1 - error2)

    differences = np.array (differences)
    mean_diff = np.mean (differences)
    conf_interval = np.percentile (differences, [2.5, 97.5])

    return mean_diff, conf_interval

def colorcodecombgate(slp, slp2, eps, ep) :
    slpid = slp
    slp2id = slp2
    for i1 in range (len (slp)) :
        if slp[i1] <= 0 + eps[ep] and slp[i1] >= 0 - eps[ep] :
            slpid[i1] = 0
        if slp[i1] > 0 + eps[ep] :
            slpid[i1] = 1
        if slp[i1] < 0 - eps[ep] :
            slpid[i1] = -1
    for i1 in range (len (slp2)) :
        if slp2[i1] <= 0 + eps[ep] and slp2[i1] >= 0 - eps[ep] :
            slp2id[i1] = 0
        if slp2[i1] > 0 + eps[ep] :
            slp2id[i1] = 1
        if slp2[i1] < 0 - eps[ep] :
            slp2id[i1] = -1

    col = []  # colors=['darkorange','#1f77b4','limegreen','red','black']
    ct = np.empty ((len (slpid), 1))
    for s1 in range (len (slp)) :
        if slpid[s1] == 0 and slp2id[s1] == 0 :
            ct[s1] = 0  # std
            col.append ('darkorange')
        if slpid[s1] != 0 and slp2id[s1] == 0 :
            ct[s1] = 1  # flowgate
            col.append ('#1f77b4')
        if slpid[s1] == 0 and slp2id[s1] != 0 :
            ct[s1] = 2  # flux gate
            col.append ('darkorchid')
        if slpid[s1] != 0 and slp2id[s1] != 0 :
            ct[s1] = 3  # flowflux gate
            col.append ('limegreen')

    return ct, col


def zoomplot5modelscomb(obsfnw11, prfnw11, prid, prfnw10, prfnw01, prfnw00, Q, var_list, varid, plotadd, title, eps,
                        xtickn, qulim, fr, bllim, window, color, slp, save) :
    colors = ['darkorange', '#1f77b4', 'darkorchid', 'limegreen']
    t1 = np.arange (len (slp[51 :]))
    col = color[51 :]
    L, W = (int (len (obsfnw11) * fr) + bllim), window
    fig, ax1 = plt.subplots ()
    ax1.plot (t1[L :L + W], obsfnw11[L :L + W], color='black', linewidth=1)
    ax1.scatter (t1[L :L + W], prid[L :L + W], c=col[L :L + W], s=10, marker='*', linewidth=0)
    ax1.plot (t1[L :L + W], prfnw11[L :L + W], color=colors[0], linewidth=.5)
    ax1.plot (t1[L :L + W], prfnw10[L :L + W], color=colors[1], linewidth=.5)
    ax1.plot (t1[L :L + W], prfnw01[L :L + W], color=colors[2], linewidth=.5)
    ax1.plot (t1[L :L + W], prfnw00[L :L + W], color=colors[3], linewidth=.5)

    ax1.set_ylabel (var_list[varid] + '  $mg/l$', fontsize=12)
    ax1.set_xticks (
        [len (t1[:L]), int ((len (t1[:L + W]) + len (t1[:L])) / 2) - 160, len (t1[:L + W - 320])])  # orgeval
    ax1.set_xticklabels (xtickn)
    ax1.set_xlabel ('Date', fontsize=12)

    ax20 = ax1.twinx ()
    ax20.plot (t1[L :L + W], Q[len (Q) - len (obsfnw11) + L :len (Q) - len (obsfnw11) + L + W], color='#929591')
    # t = np.arange (0, len (Q[len(Q)-len(obsfnw11)+L:len(Q)-len(obsfnw11)+L+W]), 1)
    ax20.fill_between (t1[L :L + W], 0, np.squeeze (Q[len (Q) - len (obsfnw11) + L :len (Q) - len (obsfnw11) + L + W]),
                       color='gainsboro', alpha=0.7)

    ax20.set_ylim (0, qulim)
    ax20.set_ylabel ('$Q$ $m^3/s$', fontsize=12)
    yl, yu = min (min (obsfnw11[L :L + W]), min (prfnw11[L :L + W])), max (max (obsfnw11[L :L + W]),
                                                                           max (prfnw11[L :L + W]))
    ax1.set_ylim (yl - abs (yl) * 2, yu + yu * .1)  # cl=6 other =2

    if save == 'yes' :
        plt.savefig (plotadd + var_list[varid] + '_time_series_5models_zoom_gapdata' + title + '.png')
    plt.show ()


def cqqscatter5models(obs, pridreg, prid, prid3, prid4, prid5, var_list, varid, plotadd, color, slp, save) :
    col = color[51 :]
    obs, pridreg, prid, prid3, prid4, prid5 = obs.reshape ((len (obs),)), pridreg.reshape (
        (len (pridreg),)), prid.reshape ((len (prid),)), prid3.reshape ((len (prid3),)), prid4.reshape (
        (len (prid4),)), prid5.reshape ((len (prid5),))
    sprfnw11, sprfnw10, sobsfnw11, sprfnw3, sprfnw4, sprfnw5 = np.sort (pridreg), np.sort (prid), np.sort (
        obs), np.sort (prid3), np.sort (prid4), np.sort (prid5)
    col = [col for _, col in sorted (zip (prid, col))]

    colors = ['black', 'gray', 'gray', 'gray', 'blue']
    col1 = [None] * (len (prid) + 1)
    col1[1 :] = col
    col1[0] = 'blue'
    col = col1
    sobsfnw110, sprfnw110, sprfnw100, sprfnw30, sprfnw40, sprfnw50 = np.empty ((len (sobsfnw11) + 1,)), np.empty (
        (len (sprfnw11) + 1,)), np.empty ((len (sprfnw10) + 1,)), np.empty ((len (sprfnw3) + 1,)), np.empty (
        (len (sprfnw4) + 1,)), np.empty ((len (sprfnw5) + 1,))
    sobsfnw110[1 :], sprfnw110[1 :], sprfnw100[1 :], sprfnw30[1 :], sprfnw40[1 :], sprfnw50[1 :] = sobsfnw11, sprfnw11, sprfnw10, sprfnw3, sprfnw4, sprfnw5
    sobsfnw110[0], sprfnw110[0], sprfnw100[0], sprfnw30[0], sprfnw40[0], sprfnw50[0] = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    fig, ax1 = plt.subplots ()

    sc0 = ax1.scatter (sobsfnw110, sprfnw110, s=5, color=colors[0], edgecolor=colors[0])
    scfg = ax1.scatter (sobsfnw110, sprfnw30, s=5, color=colors[1], edgecolor=colors[1])
    scfx = ax1.scatter (sobsfnw110, sprfnw40, s=5, color=colors[2], edgecolor=colors[2])
    scF = ax1.scatter (sobsfnw110, sprfnw50, s=5, color=colors[3], edgecolor=colors[3])
    sc3 = ax1.scatter (sobsfnw110, sprfnw100, s=50, c=col, marker='*', facecolors='none')

    x = np.linspace (0, max (max (sprfnw11), max (sprfnw10), max (sobsfnw11)) + .05 * max (sobsfnw11), 100)
    plt.plot (x, x, ':k')

    coefficient_of_dermination = r2_score (obs, pridreg)
    coefficient_of_derminationfg = r2_score (obs, prid3)
    coefficient_of_derminationfx = r2_score (obs, prid4)
    coefficient_of_derminationF = r2_score (obs, prid5)
    coefficient_of_dermination3 = r2_score (obs, prid)

    ax1.legend ([sc0, scfg, scfx, scF, sc3], ['$LSTM_{std}$ $r^2$:' + str (round (coefficient_of_dermination, 3)),
                                              '$LSTM_{fg}$ $r^2$:' + str (round (coefficient_of_derminationfg, 3)),
                                              '$LSTM_{fx}$ $r^2$:' + str (round (coefficient_of_derminationfx, 3)),
                                              '$LSTM_{F}$ $r^2$:' + str (round (coefficient_of_derminationF, 3)),
                                              '$LSTM_{adpt}$ $r^2$:' + str (round (coefficient_of_dermination3, 3))],
                loc="lower right", fontsize=12)

    ax1.set_xlabel ('Observed $mg/l$', fontsize=12)
    ax1.set_ylabel ('Predicted $mg/l$', fontsize=12)
    ax1.set_title ('Observed Vs Predicted ' + var_list[varid], fontsize=14)
    ax1.set_xlim ([min (min (sprfnw11), min (sprfnw10), min (sobsfnw11)) - .2 * min (sobsfnw11),
                   max (max (sprfnw11), max (sprfnw10), max (sobsfnw11)) + .05 * max (sobsfnw11)])
    ax1.set_ylim ([min (min (sprfnw11), min (sprfnw10), min (sobsfnw11)) - .2 * min (sobsfnw11),
                   max (max (sprfnw11), max (sprfnw10), max (sobsfnw11)) + .05 * max (sobsfnw11)])
    if save == 'yes' :
        plt.savefig (plotadd + var_list[varid] + '_quantile_plot_5models' + '.png')
    plt.show ()


def hysteresisplot(slp, obs, prid, pridstd, Q, eps, ep, var_list, varno, col, ct, qulim, xtick, ofsl, ofsu, fr, plotadd,
                   save) :  # varn,plotadd,var_list,varid,title,eps,save
    Q1 = Q[len (Q) - len (obs) + ofsl :len (Q) - len (obs) + ofsu]
    t = np.arange (len (slp))
    t1 = t[len (Q) - len (obs) + ofsl :len (Q) - len (obs) + ofsu]
    prid1 = prid[ofsl :ofsu]
    pridst = pridstd[ofsl :ofsu]
    obs1 = obs[ofsl :ofsu]
    col1 = col[len (Q) - len (obs) + ofsl - 1 :len (Q) - len (obs) + ofsu - 1]

    fig, ax = plt.subplots (2)

    z1 = ax[0].scatter (Q1, obs1, c=t1, cmap='gray', s=10)
    ax[0].set_ylabel (var_list[varno] + ' $mg/l$', fontsize=12)
    ax[0].set_xlabel ('Q $m^3/s$', fontsize=12)
    ax[0].set_title ('Observed', fontsize=12)

    ax[0].scatter (Q1, prid1, c=col1, s=10, linewidth=0)
    # ax[0].scatter (Q1, prid1, c='darkorchid', s=10, linewidth=0)

    ax[0].set_title ('Hysteresis ' + var_list[varno], fontsize=14)
    ax[0].set_ylabel (var_list[varno] + ' $mg/l$', fontsize=12)

    z2 = ax[1].scatter (Q1, obs1, c=t1, cmap='gray', s=10)
    ax[1].set_ylabel (var_list[varno] + ' $mg/l$', fontsize=12)
    ax[1].set_xlabel ('Q $m^3/s$', fontsize=12)

    ax[1].scatter (Q1, pridst, c='darkorange', s=10, linewidth=0)
    ax[1].set_ylabel (var_list[varno] + ' $mg/l$', fontsize=12)
    ax[1].set_xlabel ('$Q$ $m^3/s$', fontsize=12)

    fig.subplots_adjust (right=0.8)
    cbar_ax = fig.add_axes ([0.85, 0.15, 0.025, 0.7])
    cbar = fig.colorbar (z2, cax=cbar_ax, ticks=[min (t1), max (t1)])
    tick_texts = cbar.ax.set_yticklabels (xtick, rotation=90)
    tick_texts[0].set_verticalalignment ('bottom')
    tick_texts[-1].set_verticalalignment ('top')
    cbar.set_label ('Date', fontsize=12, labelpad=-10)

    ax[0].set_ylim (
        min (min (obs1), min (prid1), min (pridst)) - .15 * abs (min (min (obs1), min (prid1), min (pridst))),
        max (max (obs1), max (prid1), max (pridst)) + .05 * max (max (obs1), max (prid1), max (pridst)))
    ax[1].set_ylim (
        min (min (obs1), min (prid1), min (pridst)) - .15 * abs (min (min (obs1), min (prid1), min (pridst))),
        max (max (obs1), max (prid1), max (pridst)) + .05 * max (max (obs1), max (prid1), max (pridst)))

    if save == 'yes' :
        plt.savefig (plotadd + var_list[varno] + 'hysteresis_eps=' + str (eps[ep]) + '_ofsl_' + str (ofsl) + '.png')
    plt.show ()


def sort_by_argsort(arr, indices) :
    return [arr[i] for i in indices]


def colorcodecombgate2eps(slp, slp2, ep1, ep2) :
    slpid = slp
    slp2id = slp2
    for i1 in range (len (slp)) :
        if slp[i1] <= 0 + ep1 and slp[i1] >= 0 - ep1 :
            slpid[i1] = 0
        if slp[i1] > 0 + ep1 :
            slpid[i1] = 1
        if slp[i1] < 0 - ep1 :
            slpid[i1] = -1
    for i1 in range (len (slp2)) :
        if slp2[i1] <= 0 + ep2 and slp2[i1] >= 0 - ep2 :
            slp2id[i1] = 0
        if slp2[i1] > 0 + ep2 :
            slp2id[i1] = 1
        if slp2[i1] < 0 - ep2 :
            slp2id[i1] = -1

    col = []  # colors=['darkorange','#1f77b4','limegreen','red','black']
    ct = np.empty ((len (slpid), 1))
    for s1 in range (len (slp)) :
        if slpid[s1] == 0 and slp2id[s1] == 0 :
            ct[s1] = 0  # std
            col.append ('darkorange')
        if slpid[s1] != 0 and slp2id[s1] == 0 :
            ct[s1] = 1  # flowgate
            col.append ('#1f77b4')
        if slpid[s1] == 0 and slp2id[s1] != 0 :
            ct[s1] = 2  # flux gate
            col.append ('darkorchid')
        if slpid[s1] != 0 and slp2id[s1] != 0 :
            ct[s1] = 3  # flowflux gate
            col.append ('limegreen')

    return ct, col


def heatmap(data, var, plotadd, ep, save) :
    ax = sns.heatmap (data, cmap="coolwarm", xticklabels=ep, yticklabels=ep, linewidth=0.5,
                      cbar_kws={'label' : 'RMSE $mg/l$'})
    ax.set_xlabel ('$\epsilon_2 $', fontsize=14)
    ax.set_ylabel ('$\epsilon_1$', fontsize=14)
    plt.title ('Model RMSE for $LSTM_{adpt}$ ' + var, fontsize=14)
    ax.figure.axes[-1].yaxis.label.set_size (13)
    if save == 'yes' :
        plt.savefig (plotadd + var + '_heatmap.png')
    plt.show ()


def getobservedcomb2eps(folder, fdidi, var_list, invar_list1, varid, fr, gi, seqid, fq, fqid, seq_length, eps,
                        addtest) :
    gmsetrainfnw = []
    Hloc = []
    invar_list0 = invar_list1[fdidi]
    for ep1 in range (len (eps)) :  # flow gradient
        for ep2 in range (len (eps)) :  # flux gradient
            path = addtest + folder[fdidi] + var_list[varid] + '_' + str (invar_list0[varid]) + '_fr_' + str (
                fr) + '_' + str (gi) + '_nostatic_seq_' + str (seq_length[seqid]) + '_combgate_yes_' + 'eps_' + str (
                eps[ep1]) + '_epsfx_' + str (eps[ep2]) + '__' + fq[fqid] + '_0_mse_traindata.csv'
            if os.path.exists (path) :
                msetrainfnw = pd.read_csv (
                    addtest + folder[fdidi] + var_list[varid] + '_' + str (invar_list0[varid]) + '_fr_' + str (
                        fr) + '_' + str (gi) + '_nostatic_seq_' + str (
                        seq_length[seqid]) + '_combgate_yes_' + 'eps_' + str (eps[ep1]) + '_epsfx_' + str (
                        eps[ep2]) + '__' + fq[fqid] + '_0_mse_traindata.csv')
                msetrainfnw = msetrainfnw.iloc[:, 1 :].values
                idfom = np.where (msetrainfnw == np.min (msetrainfnw))
                minmsetrain = np.min (msetrainfnw)
                gmsetrainfnw.append (minmsetrain)
                Hloc.append (idfom[1][0])
            else :
                print ('file does not exist  ' + folder[fdidi] + var_list[varid] + '_' + str (
                    invar_list0[varid]) + '_fr_' + str (fr) + '_' + str (gi) + '_nostatic_seq_' + str (
                    seq_length[seqid]) + '_combgate_yes_' + 'eps_' + str (eps[ep1]) + '_epsfx_' + str (
                    eps[ep2]) + '__' + fq[fqid] + '_0_mse_traindata.csv')
    gmsetrainfnw1 = np.array (gmsetrainfnw)
    gmsetrainfnw1 = gmsetrainfnw1.reshape (len (eps), len (eps))
    idfo = np.where (gmsetrainfnw == np.min (gmsetrainfnw))
    ep11 = int ((idfo[0][0]) / len (eps))
    ep22 = (idfo[0][0]) - len (eps) * ep11
    hl = Hloc[idfo[0][0]]

    prfnw = pd.read_csv (
        addtest + folder[fdidi] + var_list[varid] + '_' + str (invar_list0[varid]) + '_fr_' + str (fr) + '_' + str (
            gi) + '_nostatic_seq_' + str (seq_length[seqid]) + '_combgate_yes_' + 'eps_' + str (
            eps[ep11]) + '_epsfx_' + str (eps[ep22]) + '__' + fq[fqid] + '_0_nprediction.csv')
    obsfnw = pd.read_csv (
        addtest + folder[fdidi] + var_list[varid] + '_' + str (invar_list0[varid]) + '_fr_' + str (fr) + '_' + str (
            gi) + '_nostatic_seq_' + str (seq_length[seqid]) + '_combgate_yes_' + 'eps_' + str (
            eps[ep11]) + '_epsfx_' + str (eps[ep22]) + '__' + fq[fqid] + '_0_nobserved.csv')
    prfnw = prfnw.iloc[:, hl + 1].values
    obsfnw = obsfnw.iloc[:, hl + 1].values

    ntr_pr, nts_pr, ntr_obs, nts_obs = prfnw[50 :round (len (prfnw) * fr)], prfnw[round (len (prfnw) * fr) :], obsfnw[
                                                                                                               50 :round (
                                                                                                                   len (
                                                                                                                       obsfnw) * fr)], obsfnw[
                                                                                                                                       round (
                                                                                                                                           len (
                                                                                                                                               obsfnw) * fr) :]

    return ntr_pr.reshape (len (ntr_pr), 1), nts_pr.reshape (len (nts_pr), 1), ntr_obs.reshape (len (ntr_obs),
                                                                                                1), nts_obs.reshape (
        len (nts_obs), 1), prfnw[50 :].reshape (len (prfnw) - 50, 1), obsfnw[50 :].reshape (len (obsfnw) - 50,
                                                                                            1), ep11, ep22, gmsetrainfnw1

def removelinintp(target, base) :
    nan_indices = np.where (np.isnan (base))
    target[nan_indices] = np.nan
    return target




def getobservedallfoldfluxorg(folder, var_list, invar_list1, varid, fr, gi, seqid, method, methodid, cgate, cgateid,
                              weightid, fluxid, fq, fqid, seq_length, addtest) :
    gmsetrainfnw = []
    gprfnw = []
    gobsfnw = []
    for k in range (len (folder)) :
        folderid = k
        invar_list0 = invar_list1[folderid]
        path = addtest + folder[folderid] + var_list[varid] + '_' + str (invar_list0[varid]) + '_fr_' + str (
            fr) + '_' + str (gi) + '_nostatic_seq_' + str (seq_length[seqid]) + '_' + method[methodid] + '_cgate_' + \
               cgate[cgateid] + '_weighted_' + cgate[weightid] + '_fluxgate_' + cgate[fluxid] + '_' + fq[
                   fqid] + '_0_mse_traindata.csv'
        if os.path.exists (path) :
            msetrainfnw = pd.read_csv (
                addtest + folder[folderid] + var_list[varid] + '_' + str (invar_list0[varid]) + '_fr_' + str (
                    fr) + '_' + str (gi) + '_nostatic_seq_' + str (seq_length[seqid]) + '_' + method[
                    methodid] + '_cgate_' + cgate[cgateid] + '_weighted_' + cgate[weightid] + '_fluxgate_' + cgate[
                    fluxid] + '_' + fq[fqid] + '_0_mse_traindata.csv')
            msetrainfnw = msetrainfnw.iloc[:, 1 :].values
            gmsetrainfnw.append (msetrainfnw)
        else :
            print ('file does not exist  ' + folder[folderid] + var_list[varid] + '_' + str (
                invar_list0[varid]) + '_fr_' + str (fr) + '_' + str (gi) + '_nostatic_seq_' + str (
                seq_length[seqid]) + '_' + method[methodid] + '_cgate_' + cgate[cgateid] + '_weighted_' + cgate[
                       weightid] + '_fluxgate_' + cgate[fluxid] + '_' + fq[fqid] + '_0_mse_traindata.csv')

        idfo = np.where (msetrainfnw == np.min (msetrainfnw))

        prfnw = pd.read_csv (
            addtest + folder[k] + var_list[varid] + '_' + str (invar_list0[varid]) + '_fr_' + str (fr) + '_' + str (
                gi) + '_nostatic_seq_' + str (seq_length[seqid]) + '_' + method[methodid] + '_cgate_' + cgate[
                cgateid] + '_weighted_' + cgate[weightid] + '_fluxgate_' + cgate[fluxid] + '_' + fq[
                fqid] + '_0_nprediction.csv')
        obsfnw = pd.read_csv (
            addtest + folder[k] + var_list[varid] + '_' + str (invar_list0[varid]) + '_fr_' + str (fr) + '_' + str (
                gi) + '_nostatic_seq_' + str (seq_length[seqid]) + '_' + method[methodid] + '_cgate_' + cgate[
                cgateid] + '_weighted_' + cgate[weightid] + '_fluxgate_' + cgate[fluxid] + '_' + fq[
                fqid] + '_0_nobserved.csv')
        prfnw = prfnw.iloc[:, idfo[1][0] + 1].values
        obsfnw = obsfnw.iloc[:, idfo[1][0] + 1].values
        gprfnw.append (prfnw)
        gobsfnw.append (obsfnw)

    gprfnw = np.array (gprfnw)
    gobsfnw = np.array (gobsfnw)
    gprfnw = np.transpose (gprfnw)
    gobsfnw = np.transpose (gobsfnw)
    gntr_pr, gnts_pr, gntr_obs, gnts_obs = gprfnw[50 :round (len (prfnw) * fr), :], gprfnw[round (len (prfnw) * fr) :,
                                                                                    :], gobsfnw[
                                                                                        50 :round (len (obsfnw) * fr),
                                                                                        :], gobsfnw[
                                                                                            round (len (obsfnw) * fr) :,
                                                                                            :]
    return gntr_pr, gnts_pr, gntr_obs, gnts_obs, gprfnw[50 :, :], gobsfnw[50 :, :]


def plotcombinedgate(slp, obs, prid, Q, eps, ep, var_list, i, col, ct, qulim, xtick, ofsl, ofsu, fr, plotcombadd,
                     save) :
    t = np.arange (len (slp[51 :]))
    col = col[51 :]
    fig, ax = plt.subplots ()
    ax.plot (obs, c='black', linewidth=.5)
    ax.scatter (t, prid, c=col, s=15, linewidth=0)
    ax.set_xlabel ('Date', fontsize=12)
    ax.set_ylabel (var_list[i] + ' $mg/l$', fontsize=12)
    ax.set_title (var_list[i], fontsize=14)

    yllim, yulim = min (obs) - min (obs) * 1.7, max (obs) + max (obs) * .1  # (Cl 3,.05),(k 8,.05),(Na 10,.1)
    ax.set_ylim (yllim, yulim)  # k

    rmse = sqrt (mean_squared_error (obs[int (len (obs) * .7) :], prid[int (len (obs) * .7) :]))
    rmsef = sqrt (mean_squared_error (obs, prid))

    print (rmse, var_list[i])
    print (rmsef, var_list[i])

    b11, b1, h1, w1 = .21, .2, .21, .21  # ca mg cl
    # b11, b1, h1, w1 = .41, .2, .21, .21 #k
    ax3 = fig.add_axes ([b11, b1, w1, h1])
    N, bins, patches = ax3.hist (ct[:int (0.7 * len (ct))], edgecolor='white', linewidth=1)
    patches[0].set_facecolor ('darkorange')
    patches[4].set_facecolor ('#1f77b4')
    patches[6].set_facecolor ('darkorchid')
    patches[9].set_facecolor ('limegreen')
    y_vals = ax3.get_yticks ()
    ax3.set_yticklabels (['{:3.0f}k'.format (x / 1000) for x in y_vals])
    ax3.set_ylabel ('Frequency', labelpad=-8, fontsize=12)
    ax3.set_xticks ([0.18, 1.05, 1.95, 2.85])
    ax3.set_xticklabels (['$std$', '$fg$', '$fx$', '$F$'])
    ax3.set_title ('Gate histogram', fontsize=12)

    ax20 = ax.twinx ()
    ax20.plot (Q[len (Q) - len (obs) :], color='#929591')
    t = np.arange (0, len (Q[len (Q) - len (obs) :]), 1)
    ax20.fill_between (t, 0, np.squeeze (Q[len (Q) - len (obs) :]), color='#929591', alpha=0.7)

    ax20.set_ylim (0, qulim)
    ax20.set_ylabel ('Q $m^3/s$', fontsize=12)
    ax.plot (np.nan, color='#929591')

    ax.xaxis.set_ticks ([0, len (prid) / 4, 2 * len (prid) / 4, 3 * len (prid) / 4, 4 * len (prid) / 4])
    ax.xaxis.set_ticklabels (xtick)
    ax.add_patch (Rectangle ((ofsl, yllim), ofsu - ofsl, qulim, linewidth=2, edgecolor='#00FFFF',
                             facecolor='none', linestyle='-'))

    ax.axvline (x=len (ct[:int (0.7 * len (ct))]), ymin=0, color='red', linestyle='dotted', linewidth=2)
    if save == 'yes' :
        plt.savefig (plotcombadd + var_list[i] + 'combined_gated_model_eps=' + str (eps) + '_' + str (ep) + '.png')
    plt.show ()

def boxplot(n_bootstraps, obs11, prid11, prid01, prid00, prid10, pridfnwcme,var_list,var_listn,varno,fq,title,plotcombadd,save):
    O = obs11
    P1 = prid11
    P2 = prid01
    P3 = prid00
    P4 = prid10
    P5 = pridfnwcme
    n_data = len (O)
    P = [P1, P2, P3, P4, P5]

    # Initialize lists to store bootstrap MSEs
    mse_1, mse_2, mse_3, mse_4, mse_5 = [], [], [], [], []

    for _ in range (n_bootstraps) :
        indices = bootstrap_sample (np.arange (len (O)))
        sample_y_true = O[indices]
        sample_y_pred1 = P1[indices]
        sample_y_pred2 = P2[indices]
        sample_y_pred3 = P3[indices]
        sample_y_pred4 = P4[indices]
        sample_y_pred5 = P5[indices]

        mse_1.append (sqrt (mean_squared_error (sample_y_true, sample_y_pred1)))
        mse_2.append (sqrt (mean_squared_error (sample_y_true, sample_y_pred2)))
        mse_3.append (sqrt (mean_squared_error (sample_y_true, sample_y_pred3)))
        mse_4.append (sqrt (mean_squared_error (sample_y_true, sample_y_pred4)))
        mse_5.append (sqrt (mean_squared_error (sample_y_true, sample_y_pred5)))

    # Combine all MSEs for statistical tests
    mse_data = [mse_1, mse_2, mse_3, mse_4, mse_5]

    #### normal distribution test
    for i in range (5) :
        stat, p = kstest (mse_data[i], 'norm')
        print ('Kolmogorov-Smirnov Test: Statistics=%.3f, p=%.3f' % (stat, p))
        if p > 0.05 :
            print ('Sample looks Gaussian (fail to reject H0) i=', str (i))
        else :
            print ('Sample does not look Gaussian (reject H0) i=', str (i))
    # ANOVA to compare means
    # anova_result = f_oneway (mse_1, mse_2, mse_3, mse_4)
    anova_result = stats.kruskal (mse_1, mse_2, mse_3, mse_4, mse_5)
    print (f'ANOVA result: H={anova_result.statistic}, p={anova_result.pvalue}')

    # If ANOVA is significant, perform Tukey's HSD
    if anova_result.pvalue < 0.05 :
        # Flatten the data and create group labels
        all_mse = np.concatenate (mse_data)
        groups = ['$LSTM_{std}$'] * n_bootstraps + ['$LSTM_{fg}$'] * n_bootstraps + ['$LSTM_{FF}$'] * n_bootstraps + [
            '$LSTM_{fx}'] * n_bootstraps + ['$LSTM_{intg}'] * n_bootstraps
        # tukey_result = pairwise_tukeyhsd (all_mse, groups,alpha=.05)
        tukey_result = posthoc_dunn ([mse_1, mse_2, mse_3, mse_4, mse_5], p_adjust='bonferroni')
        print (var_list[varno])
        print (tukey_result)

    # Visualize the results

    mseavg = [np.mean (mse_data[0]), np.mean (mse_data[1]), np.mean (mse_data[2]), np.mean (mse_data[3]),
              np.mean (mse_data[4])]
    msemin = np.where (mseavg == np.min (mseavg))
    modelss = ['$LSTM_{std}$', '$LSTM_{fg}$', '$LSTM_{F}$', '$LSTM_{fx}$', '$LSTM_{intg}$']
    my_pal = {'$LSTM_{std}$' : 'darkorange', '$LSTM_{fg}$' : '#1f77b4', '$LSTM_{FF}$' : 'limegreen',
              '$LSTM_{fx}$' : "darkorchid", '$LSTM_{intg}$' : 'blue'}
    sns.boxplot (data=mse_data, palette=[my_pal[l] for l in my_pal.keys ()])
    plt.xticks ([0, 1, 2, 3, 4], ['$LSTM_{std}$', '$LSTM_{fg}$', '$LSTM_{F}$', '$LSTM_{fx}$', '$LSTM_{adpt}$'], fontsize=12)
    plt.ylabel ('RMSE', labelpad=-2, fontsize=12)
    plt.title ('Bootstrap RMSE Comparison ' + var_listn[varno], fontsize=14)
    plt.yscale ("log")
    if save=='yes':
        plt.savefig(plotcombadd+'monti_rmse_'+title+'_'+var_listn[varno] + '_res_'+ fq[0]+'.png')
    plt.show ()

###########################################################################################################
##### Solute and stream flow rate plots ############################################################################################
###########################################################################################################
from datetime import datetime, timedelta

datar=pd.read_csv('/home/.../orgeval_0.5hr_preprocess_gaps.csv')
datar=datar.iloc[:,2:].values
var=[2,6,0,5,1,3,4]
var_list=['$Ca$','$Cl$','$Mg$','$NO_3$','$K$','$Na$','$SO_4$']

t = np.arange(datetime(2015,6,12), datetime(2016,9,7), timedelta(hours=0.5)).astype(datetime)
t=t.reshape((len(t),1))
fig, ax = plt.subplots (7)
for i in range(len(var)):
    if i==6:
        ax[i].plot (t[:18741],datar[:18741,var[i]],'black')
        ax[i].set_xlabel ('Date', fontsize=12)
        ax[i].text(0.01, 0.15, var_list[i], horizontalalignment='left',verticalalignment='bottom', transform=ax[i].transAxes)
        plt.gcf ().autofmt_xdate ()
    if i==0:
        ax[i].plot (t[:18741],datar[:18741,var[i]],'black')
        ax[i].set_title ('Solute Concentration, Orgeval', fontsize=14)
        ax[i].text (0.01, 0.15, var_list[i], horizontalalignment='left', verticalalignment='bottom',
                    transform=ax[i].transAxes)
        plt.gcf ().autofmt_xdate ()
    else:
        ax[i].plot (t[:18741],datar[:18741,var[i]],'black')
        ax[i].text (0.01, 0.15, var_list[i], horizontalalignment='left', verticalalignment='bottom',
                    transform=ax[i].transAxes)
        plt.gcf().autofmt_xdate()
plt.savefig('/home/taruna2/fall2021/output_p2lstm2023/global_test_mt_comgate/dataorgeval_processed.png')
plt.show ()

plt.figure()
plt.plot(t[:18741],datar[:18741,7],'black')
plt.xlabel ('Date', fontsize=12)
plt.ylabel ('$Q$ $m^3/s$', fontsize=12)
plt.title ('Stream Flow Rate, Orgeval', fontsize=14)
plt.gcf().autofmt_xdate()
# plt.savefig('/home/.../Qdataorgeval_processed.png')
plt.show()

Q=datar[:18741,7].reshape((18741,1))
for i in range(len(var)):
    flux=np.zeros((18741,1))
    for j in range(18741):
        if Q[j]==np.nan or datar[j,var[i]]==np.nan:
            flux[j,0]=np.nan
            Q[j,0]=np.nan
        else:
            flux[j,0]=Q[j,0]*datar[j,var[i]]
    fig, ax20 = plt.subplots ()
    ax1 = ax20.twinx ()
    ax20.scatter(Q,datar[:18741,var[i]], color='g',s=10,facecolor='none')
    ax1.scatter(Q,flux,c='black',s=15)
    ax20.set_xlabel ('$Q$ $m^3/s$', fontsize=12)
    ax1.set_ylabel ('Flux $mg.m^3/s$', fontsize=12)
    ax20.set_ylabel('Concentration $mg/l$', fontsize=12)
    ax1.set_title (var_list[i]+' Flux Vs Stream Flow Vs Concentration, Orgeval', fontsize=14)
    # plt.savefig('/home/.../qqQfluxdataorgeval_'+var_list[i]+'processed.png')
    plt.show()


############################################################################################################
##### Orgeval Flux gate and STD gate#####################################################################
############################################################################################################

var_listn = ['$Mg^{2+}$', '$K^+$', '$Ca^{2+}$', '$Na^+$', '$SO_4^{2-}$', '$NO_3^-$', '$Cl^-$']
var = [1, 2, 3, 4, 5, 6, 7]
var_list = ['Mg', 'K', 'Ca', 'Na', 'SO4', 'NO3', 'Cl']

# 2var
inputvar12 = [[8, 1], [8, 2], [8, 3], [8, 4], [8, 5], [8, 6], [8, 7]]
invar_list12 = [['Q', 'Mg'], ['Q', 'K'], ['Q', 'Ca'], ['Q', 'Na'], ['Q', 'SO4'], ['Q', 'NO3'], ['Q', 'Cl']]  # test

inputvar1 = [inputvar12]
invar_list1 = [invar_list12]

fq = ['0.5hr']
method = ['mLSTM(tanh)', 'regLSTM']

seq_length = [5]
folder = ['2var/']

fr = 0.7
fr1 = 0.7
cgate = ['yes', 'no']


gi = 1
fqid = 0  # resolution
seqid = fqid
cgateid = 1
weightid = 1

addtest = '/home/.../saved_files/'  # please change based on which one are you plotting


eps = [0.01, 0.001, 0.0005, 0.0001, 0.00001, 0]
fdid = [0, 0, 0, 0, 0, 0, 0]  # folderid

varno = 0
signo = 1  # for the extreme values box plot

inputvar0 = [inputvar12]
invar_list0 = [invar_list12]
invar_l = invar_list0


fol = ['2var/']

folder0 = '/home/.../saved_files/'
plotcomadd = '/home/../plot/'
tau_max = 2
nk = 1

tau_s = [tau_max]


resfilename = ['.5hr']
res = 0
print ('res=' + resfilename[res])

address = '/home/.../orgeval_0' + resfilename[res] + '_processed.csv'
sheetname = 'orgeval'

resample = '_' + resfilename[res]

rkey = 3
stpt, gi, gf = 1, 1, 18700

Gi = [1]



for i in range (len (var)) :
    if i == varno :
        print ('**************************')
        print (var_list[i])
        print ('**************************')
        g = 0
        inputvar1 = inputvar0[0]
        invar_list1 = invar_list0[0]

        if len (inputvar1[i]) == 2 :
            l, m, n = utils.dataretrive (var[i], inputvar1[i], tau_max, address, sheetname, stpt, gi, gf)
            u, v = np.hstack ((l, m[:, 0].reshape ((len (m), 1)), n[:, 0].reshape ((len (m), 1)))), l

        training_set = u
        slp = utils.getgrad (training_set[:, 1])
        flux = training_set[:, 1] * training_set[:, 2]
        slp2 = utils.getgrad (flux)

        St = pd.read_csv ('/home/.../orgeval_0' + resfilename[res] + '_processed.csv',header=0)
        St1 = St.iloc[1 :-1, [var[i]]].values
        gi, gf = 1, 18700
        X = St1[gi :gf]
        sc = MinMaxScaler ()
        X_sc = sc.fit_transform (X)
        q = pd.read_csv ('/home/.../orgeval_0' + resfilename[res] + '_processed.csv',header=0)
        q1 = q.iloc[1 :-1, [8]].values
        gi, gf = 1, 18700
        Q = q1[gi :gf]

        gap_data_set = pd.read_csv ('//home/.../orgeval_0.5hr_preprocess_gaps.csv', header=0)
        gapdata_set1 = gap_data_set.iloc[1 :-1, [var[i] + 1]].values
        gi, gf = 1, 18700
        gapdataQ = gapdata_set1[gi :gf]
        gapdataQ = gapdataQ.reshape (-1, )
        gapdataX = gapdata_set1[gi + 52 :gf]
        gapdataX = gapdataX.reshape (-1, )
        methodid, fluxid = 1, 1  # reg LSTM

        ntr_pr11, nts_pr11, ntr_obs11, nts_obs11, prfnw11, obsfnw11 = getobservedallfoldfluxorg (folder, var_list,invar_l, varno, fr,gi, seqid, method,methodid, cgate,cgateid, weightid,fluxid, fq, fqid,seq_length,addtest)
        ntr_pr11, nts_pr11, ntr_obs11, nts_obs11, prfnw11, obsfnw11 = sc.inverse_transform (ntr_pr11), sc.inverse_transform (nts_pr11), sc.inverse_transform (ntr_obs11), sc.inverse_transform (nts_obs11), sc.inverse_transform (prfnw11), sc.inverse_transform (obsfnw11)

        ntr_prcme, nts_prcme, ntr_obscme, nts_obscme, prfnwcme, obsfnwcme, ep11, ep22, gmse = getobservedcomb2eps (folder, fdid[i], var_list, invar_l, varno, fr, gi, seqid, fq, fqid, seq_length, eps, folder0)
        ntr_prcme, nts_prcme, ntr_obscme, nts_obscme, prfnwcme, obsfnwcme = sc.inverse_transform (ntr_prcme), sc.inverse_transform (nts_prcme), sc.inverse_transform (ntr_obscme), sc.inverse_transform (nts_obscme), sc.inverse_transform (prfnwcme), sc.inverse_transform (obsfnwcme)

        methodid, fluxid = 0, 1  # mLSTM
        ntr_pr01, nts_pr01, ntr_obs01, nts_obs01, prfnw01, obsfnw01 = getobservedallfoldfluxorg (folder, var_list,invar_l, varno, fr,gi, seqid, method,methodid, cgate,cgateid, weightid,fluxid, fq, fqid,seq_length,addtest)
        ntr_pr01, nts_pr01, ntr_obs01, nts_obs01, prfnw01, obsfnw01 = sc.inverse_transform (ntr_pr01), sc.inverse_transform (nts_pr01), sc.inverse_transform (ntr_obs01), sc.inverse_transform (nts_obs01), sc.inverse_transform (prfnw01), sc.inverse_transform (obsfnw01)
        methodid, fluxid = 0, 0  # ffLSTM
        ntr_pr00, nts_pr00, ntr_obs00, nts_obs00, prfnw00, obsfnw00 = getobservedallfoldfluxorg (folder, var_list,invar_l, varno, fr,gi, seqid, method,methodid, cgate,cgateid, weightid,fluxid, fq, fqid,seq_length,addtest)
        ntr_pr00, nts_pr00, ntr_obs00, nts_obs00, prfnw00, obsfnw00 = sc.inverse_transform (ntr_pr00), sc.inverse_transform (nts_pr00), sc.inverse_transform (ntr_obs00), sc.inverse_transform (nts_obs00), sc.inverse_transform (prfnw00), sc.inverse_transform (obsfnw00)
        methodid, fluxid = 1, 0  # fLSTM
        ntr_pr10, nts_pr10, ntr_obs10, nts_obs10, prfnw10, obsfnw10 = getobservedallfoldfluxorg (folder, var_list,invar_l, varno, fr,gi, seqid, method,methodid, cgate,cgateid, weightid,fluxid, fq, fqid,seq_length,addtest)
        ntr_pr10, nts_pr10, ntr_obs10, nts_obs10, prfnw10, obsfnw10 = sc.inverse_transform (ntr_pr10), sc.inverse_transform (nts_pr10), sc.inverse_transform (ntr_obs10), sc.inverse_transform (nts_obs10), sc.inverse_transform (prfnw10), sc.inverse_transform (obsfnw10)


        prid, obs = prfnwcme, obsfnwcme
        prid2, obs2 = prfnw11[:, 0].reshape ((len (prid), 1)), obsfnw11[:,0].reshape ((len (prid), 1))
        prid3, obs3 = prfnw10[:, 0].reshape ((len (prid), 1)), obsfnw10[:, 0].reshape ((len (prid), 1))
        prid4, obs4 = prfnw01[:, 0].reshape ((len (prid), 1)), obsfnw01[:, 0].reshape ((len (prid), 1))
        prid5, obs5 = prfnw00[:, 0].reshape ((len (prid), 1)), obsfnw00[:, 0].reshape ((len (prid), 1))

        ct, col = colorcodecombgate2eps (slp, slp2, eps[ep11], eps[ep22])
        qlim = 40
        xtick = ['Jun-15', 'Oct-15', 'Feb-16', 'May-16', 'Aug-16']
        ofsl, ofsu = 14400, 15000

        plotcombinedgate (slp, obs,prid,Q, eps[ep11], eps[ep22], var_listn, varno, col, ct,qlim,xtick,ofsl, ofsu,fr,plotcomadd, save='yes')
        #
        xtick=['22-May-16','31-May-16']
        t = np.arange (len (slp))
        ofsl, ofsu = 14500, 15000
        hysteresisplot(slp,obs,prid,prid2,Q, eps, ep11, var_listn, varno, col, ct,qlim,xtick,ofsl, ofsu,fr, plotcomadd,save='yes')

        ofsl, ofsu =12100, 12500
        xtick=['27-Feb-16','07-Mar-16']
        hysteresisplot(slp,obs,prid,prid2,Q, eps, ep11, var_listn, varno, col, ct,qlim,xtick,ofsl, ofsu,fr, plotcomadd,save='yes')
        ofsl, ofsu =11100, 11500
        xtick=['22-Mar-16','06-Apr-16']
        hysteresisplot(slp,obs,prid,prid2,Q, eps, ep11, var_listn, varno, col, ct,qlim,xtick,ofsl, ofsu,fr, plotcomadd,save='yes')
        qulim,bllim,window=40,400,2000

        xtickn = ['1-May-16', '27-May-16', '20-Jun-16']

        obs22, prid22, prid20,prid30,prid40,prid50=removelinintp(obs2,gapdataX),removelinintp(prid2,gapdataX),removelinintp(prid,gapdataX),removelinintp(prid3,gapdataX),removelinintp(prid4,gapdataX),removelinintp(prid5,gapdataX)
        Q1=removelinintp(Q,gapdataQ)
        zoomplot5modelscomb(obs22, prid22, prid20,prid40,prid30,prid50, Q1, var_listn, varno, plotcomadd, ' entire data set fr: ' + str (fr1) + ' res:' + fq[fqid], eps,xtickn,qulim,fr,bllim,window, col,slp, save='yes')
        gmse1=np.sqrt(gmse)
        heatmap(gmse1,var_listn[varno],plotcomadd,eps,save='yes')
        cqqscatter5models(obs2, prid2, prid, prid3,prid4,prid5,var_listn, varno, plotcomadd,col,slp, save='yes')



        pridfnwcme,obsfnwcme=prfnwcme,obsfnwcme
        prid11,obs11=prid2,obs2
        prid10,obs10=prid3,obs3
        prid01,obs01=prid4,obs4
        prid00,obs00=prid5,obs5
#
        tp = 12352
        ofsl, ofsu =12100, 12500
        rladpt = abs (prid[ofsl :tp] - obs[ofsl :tp])
        fladpt = abs (prid[tp :ofsu] - obs[tp :ofsu])
        rlstd = abs (prid2[ofsl :tp] - obs2[ofsl :tp])
        flstd = abs (prid2[tp :ofsu] - obs2[tp :ofsu])
        HRadpt = (np.sum (rladpt) + np.sum (fladpt)) / np.sum (obs[ofsl :ofsu])
        HRstd = (np.sum (rlstd) + np.sum (flstd)) / np.sum (obs[ofsl :ofsu])
        RLadpt = (np.sum (rladpt)) / np.sum (obs[ofsl :tp])
        FLadpt = (np.sum (fladpt)) / np.sum (obs[tp :ofsu])
        RLstd = (np.sum (rlstd)) / np.sum (obs[ofsl :tp])
        FLstd = (np.sum (flstd)) / np.sum (obs[tp :ofsu])
        print ('HRadpt:', HRadpt)
        print ('HRstd:', HRstd)
        print ('RLadpt:', RLadpt)
        print ('FLadpt:', FLadpt)
        print ('RLstd:', RLstd)
        print ('FLstd:', FLstd)


        ###Number of bootstrap samples
        n_bootstraps = 1000
        boxplot(n_bootstraps, obs11, prid11, prid01, prid00, prid10, pridfnwcme,var_list,var_listn,varno,fq,'entire',plotcomadd,save='yes')


        ############# extreme values #######
        mu=np.mean(obsfnwcme)
        sigma=np.std(obsfnwcme)
        if var_list[i]=='NO3' or var_list[i]=='K':
            obs1sigma=obsfnwcme[obsfnwcme>mu+signo*sigma]
            idx1=np.where(obsfnwcme>mu+signo*sigma)[0]
        else:
            obs1sigma = obsfnwcme[obsfnwcme > mu - signo * sigma]
            idx1 = np.where (obsfnwcme > mu - signo * sigma)[0]

        pridadpt1=pridfnwcme[idx1]
        prid111=prid11[idx1]
        prid110=prid10[idx1]
        prid101=prid01[idx1]
        prid100=prid00[idx1]

        boxplot (n_bootstraps, obs11, prid111, prid101, prid100, prid110, pridadpt1, var_list, var_listn, varno, fq,'extreme_1sigma', plotcomadd, save='yes')



############ flux gate summary mt ##############################
from matplotlib.legend_handler import HandlerTuple
var_listn=['$Ca^{2+}$','$Mg^{2+}$','$K^+$','$Na^+$','$Cl^-$','$SO_4^{2-}$','$NO_3^-$']

############ org
sigma1fx=[.07,.009,.006,.026,.034,.026,.023]
sigma1std=[.066,.009,.008,.035,.04,.024,.024]
sigma2fx=[.242,.026,.010,.042,.063,.097,.126]
sigma2std=[.25,.025,.015,.052,.076,.098,.133]
tfx=[.066,.0074,.0037,.0172,.0173,.0191,.0146]
tstd=[.0569,.0049,.004,.0149,.0303,.0150,.0157]

#### adaptive gate

color=['blue','red','orange','purple','green','brown','black']

a=np.empty(7,dtype=object)
b=np.empty(7,dtype=object)
c=np.empty(7,dtype=object)

plt.figure()
for i in range(7):
    a[i]=plt.scatter(tstd[i],tfx[i],c=color[i],marker='*',s=200,label=var_listn[i])
    b[i]=plt.scatter(sigma1std[i],sigma1fx[i],c=color[i],marker='o',s=100,label=var_listn[i])
    c[i]=plt.scatter(sigma2std[i],sigma2fx[i],edgecolors=color[i],marker='o',facecolors='none',s=100,label=var_listn[i])

plt.ylabel('RMSE $LSTM_{fx}$',fontsize=12)
plt.xlabel('RMSE $LSTM_{std}$',fontsize=12)
plt.title('Orgeval, RMSE',fontsize=14)

x = np.linspace(0, 10, 100)
y = x
plt.plot(x, y, color='black', linestyle='--', label='1:1 Line')
plt.yscale('log')
plt.xscale('log')
plt.xlim(.003,.3)
plt.ylim(.003,.3)

plt.legend([(a[0],b[0],c[0]),(a[1],b[1],c[1]),(a[2],b[2],c[2]),(a[3],b[3],c[3]),(a[4],b[4],c[4]),(a[5],b[5],c[5]),(a[6],b[6],c[6])],[var_listn[0],var_listn[1],
            var_listn[2],var_listn[3],var_listn[4],var_listn[5],var_listn[6]],handler_map={tuple: HandlerTuple(ndivide=None,pad=1)},ncol=2,fontsize=12,title='$RMSE_T$   $RMSE_{1}$   $RMSE_{2}$')
plt.legend(['FH $LSTM_{adpt}$','FH $LSTM_{std}$','RL $LSTM_{adpt}$','RL $LSTM_{std}$','FL $LSTM_{adpt}$','FL $LSTM_{std}$'])
plt.show()


#### adaptive gate
from matplotlib.legend_handler import HandlerTuple
var_listn=['$Ca^{2+}$','$Cl^-$','$Mg^{2+}$','$NO_3^-$','$K^+$','$Na^+$','$SO_4^{2-}$']
tadptorg=[.0567,.0195,.0052,.0152,.0031,.0111,.0136]
tstdorg=[.0569,.0303,.0049,.0157,.004,.0149,.0150]
tfxorg=[.066,.0173,.0074,.0146,.0037,.0172,.0191]



color=['blue','green','red','black','orange','purple','brown'] ##adaptive one
a=np.empty(7,dtype=object)
b=np.empty(7,dtype=object)
c=np.empty(7,dtype=object)

plt.figure()
for i in range(7):
    a[i]=plt.scatter(tstdorg[i],tadptorg[i],c=color[i],marker=',',s=100,label=var_listn[i])   #adptive
    b[i]=plt.scatter(tfxorg[i],tadptorg[i],edgecolors=color[i],marker=',',s=100,facecolors='none',label=var_listn[i])  #adptive

plt.ylabel('RMSE $LSTM_{adpt}$',fontsize=12)
plt.xlabel('RMSE $LSTM_{std}$ and RMSE $LSTM_{fx}$',fontsize=12)
plt.title('Orgeval, RMSE',fontsize=14)

x = np.linspace(0, 10, 100)
y = x
plt.plot(x, y, color='black', linestyle='--', label='1:1 Line')
plt.yscale('log')
plt.xscale('log')

plt.xlim(.002,.1)
plt.ylim(.002,.1)

plt.legend([(a[0],b[0]),(a[1],b[1]),(a[2],b[2]),(a[3],b[3]),(a[4],b[4]),(a[5],b[5]),(a[6],b[6])],[var_listn[0],var_listn[1],
            var_listn[2],var_listn[3],var_listn[4],var_listn[5],var_listn[6]],handler_map={tuple: HandlerTuple(ndivide=None,pad=1)},ncol=2,fontsize=12)

plt.show()


################# adpt Monticello Hysteresis residue plot #############
#
from matplotlib.legend_handler import HandlerTuple


#orgeval
Tadpt=[.011,.019,.009,.048,.030,.023,.033]
Tstd=[.016,.034,.01,.051,.032,.036,.038]


color=['blue','green','red','black','orange','purple','brown']
a=np.empty(7,dtype=object)
b=np.empty(7,dtype=object)
c=np.empty(7,dtype=object)

plt.figure()
for i in range(7):
    a[i]=plt.scatter(Tstd[i],Tadpt[i],c=color[i],marker='*',s=200,label=var_listn[i])

plt.ylabel('HR $LSTM_{adpt}$',fontsize=12)
plt.xlabel('HR $LSTM_{std}$',fontsize=12)
plt.title('Orgeval, Hysteresis Residual HL',fontsize=14)

x = np.linspace(0, 10, 100)
y = x
plt.plot(x, y, color='black', linestyle='--', label='1:1 Line')
plt.yscale('log')
plt.xscale('log')
plt.xlim(.01,.1)  #orgeval
plt.ylim(.01,.1)  #orgeval


plt.legend([(a[0]),(a[1]),(a[2]),(a[3]),(a[4]),(a[5]),(a[6])],[var_listn[0],var_listn[1],
            var_listn[2],var_listn[3],var_listn[4],var_listn[5],var_listn[6]],handler_map={tuple: HandlerTuple(ndivide=None)},ncol=3,fontsize=12)
plt.legend(['FH $LSTM_{adpt}$','FH $LSTM_{std}$','RL $LSTM_{adpt}$','RL $LSTM_{std}$','FL $LSTM_{adpt}$','FL $LSTM_{std}$'])
plt.show()