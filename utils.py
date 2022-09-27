import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import combinations
from scipy.stats import mannwhitneyu, norm


def errors_idxs(e, n_groups):
    for c in combinations(range(e+n_groups-1), n_groups-1):
        yield [b-a-1 for a, b in zip((-1,)+c, c+(e+n_groups-1,))]
        
def top_k(x, k, min_value=False):
    if min_value:
        ind = np.argpartition(x, k)[:k]
    else:
        ind = np.argpartition(x, -1*k)[-1*k:]
    return ind[np.argsort(x[ind])]

def find_PU(data, class_col, rank_df, n_changes, W, pvalues_c, nx, ny, relevant_label='all', 
            alternative='less'):
    if alternative == 'two-sided':
        pvalue_original = np.where(pvalues_c < 0.5, pvalues_c, 1 - pvalues_c) * 2
    else:
        pvalue_original = pvalues_c
    PU = [pvalue_original]
    N = rank_df.shape[0]
    PU_idxs = [np.array(['original'] * rank_df.shape[1])]
    for changes in errors_idxs(n_changes, 2):
        W_c = W.copy()
        add_nx = 0
        add_ny = 0
        idxs_pos = []
        idxs_neg = []
        if changes[0] and relevant_label in ['all', True]:
            if alternative == 'less':
                idxs_pos = np.apply_along_axis(lambda x: top_k(x, changes[0]), 0, 
                                           rank_df[data[class_col]].values)
            elif alternative == 'greater':
                idxs_pos = np.apply_along_axis(lambda x: top_k(x, changes[0], min_value=True), 0, 
                                           rank_df[data[class_col]].values)
            else: # two-sided
                idxs_pos_less = np.apply_along_axis(lambda x: top_k(x, changes[0]), 0, 
                                           rank_df[data[class_col]].values)
                idxs_pos_greater = np.apply_along_axis(lambda x: top_k(x, changes[0], min_value=True), 0, 
                                           rank_df[data[class_col]].values)
                idxs_pos = np.where(pvalues_c < 0.5, idxs_pos_less, idxs_pos_greater)         
            add_nx += changes[0]
            add_ny -= changes[0]
            
        if changes[1] and relevant_label in ['all', False]:
            if alternative == 'less':
                idxs_neg = np.apply_along_axis(lambda x: top_k(x, changes[1], min_value=True), 0, 
                                           rank_df[~data[class_col]].values)
            elif alternative == 'greater':
                idxs_neg = np.apply_along_axis(lambda x: top_k(x, changes[1]), 0, 
                                           rank_df[~data[class_col]].values)
            else: # two-sided
                idxs_neg_less = np.apply_along_axis(lambda x: top_k(x, changes[1], min_value=True), 0, 
                                           rank_df[~data[class_col]].values)
                idxs_neg_greater = np.apply_along_axis(lambda x: top_k(x, changes[1]), 0, 
                                           rank_df[~data[class_col]].values)
                idxs_neg = np.where(pvalues_c < 0.5, idxs_neg_less, idxs_neg_greater)                     
            add_nx -= changes[1]
            add_ny += changes[1]
        for idxs in idxs_pos:
            W_c = W_c + rank_df[data[class_col]].reindex(rank_df[data[class_col]].columns, 
                        axis=1).to_numpy()[idxs, range(rank_df[data[class_col]].columns.shape[0])]
        for idxs in idxs_neg:
            W_c = W_c - rank_df[~data[class_col]].reindex(rank_df[~data[class_col]].columns, 
                        axis=1).to_numpy()[idxs, range(rank_df[~data[class_col]].columns.shape[0])]
        idxs = np.array(list(idxs_pos) + list(idxs_neg))
        if len(idxs):
            PU_idxs.append([tuple(idxs[:, i]) for i in range(idxs.shape[1])])
            if alternative == 'less':
                PU.append(norm.cdf((W_c - (nx+add_nx) * (N + 1) / 2) / np.sqrt((nx+add_nx)*(ny+add_ny) * (N + 1)/ 12)))
            elif alternative == 'greater':
                PU.append(1 - norm.cdf((W_c - (nx+add_nx) * (N + 1) / 2) / np.sqrt((nx+add_nx)*(ny+add_ny) * (N + 1)/ 12)))
            else: # two-sided
                PU_c = norm.cdf((W_c - (nx+add_nx) * (N + 1) / 2) / np.sqrt((nx+add_nx)*(ny+add_ny) * (N + 1)/ 12))
                PU_c = np.where(PU_c < 0.5, PU_c, 1 - PU_c) * 2
                PU.append(PU_c)
    PU_idxs = np.array(PU_idxs)[np.array(PU).argmax(axis=0), range(np.array(PU_idxs).shape[1])]
    return np.array(PU).max(axis=0), PU_idxs

def find_PL(data, class_col, rank_df, n_changes, W, pvalues_c, nx, ny, relevant_label='all', 
           alternative='less'):
    if alternative == 'two-sided':
        pvalue_original = np.where(pvalues_c < 0.5, pvalues_c, 1 - pvalues_c) * 2
    else:
        pvalue_original = pvalues_c
    PL = [pvalue_original]
    N = rank_df.shape[0]
    PL_idxs = [np.array(['original'] * rank_df.shape[1])]
    for changes in errors_idxs(n_changes, 2):
        W_c = W.copy()
        add_nx = 0
        add_ny = 0
        idxs_pos = []
        idxs_neg = []
        if changes[0] and relevant_label in ['all', True]:            
            if alternative == 'less':
                idxs_pos = np.apply_along_axis(lambda x: top_k(x, changes[0], min_value=True), 0, 
                                               rank_df[data[class_col]].values)
            elif alternative == 'greater':
                idxs_pos = np.apply_along_axis(lambda x: top_k(x, changes[0]), 0, 
                                           rank_df[data[class_col]].values)
            else: # two-sided
                idxs_pos_less = np.apply_along_axis(lambda x: top_k(x, changes[0], min_value=True), 0, 
                                               rank_df[data[class_col]].values)
                idxs_pos_greater = np.apply_along_axis(lambda x: top_k(x, changes[0]), 0, 
                                                       rank_df[data[class_col]].values)
                idxs_pos = np.where(pvalues_c < 0.5, idxs_pos_less, idxs_pos_greater)    
            add_nx += changes[0]
            add_ny -= changes[0]
            
        if changes[1] and relevant_label in ['all', False]:
            if alternative == 'less':
                idxs_neg = np.apply_along_axis(lambda x: top_k(x, changes[1]), 0, 
                                               rank_df[~data[class_col]].values)
            elif alternative == 'greater':
                idxs_neg = np.apply_along_axis(lambda x: top_k(x, changes[1], min_value=True), 0, 
                                               rank_df[~data[class_col]].values)
            else: # two-sided
                idxs_neg_less = np.apply_along_axis(lambda x: top_k(x, changes[1]), 0,
                                                    rank_df[~data[class_col]].values)
                idxs_neg_greater = np.apply_along_axis(lambda x: top_k(x, changes[1], min_value=True), 0, 
                                                       rank_df[~data[class_col]].values)
                idxs_neg = np.where(pvalues_c < 0.5, idxs_neg_less, idxs_neg_greater)  
            add_nx -= changes[1]
            add_ny += changes[1]
        for idxs in idxs_pos:
            W_c = W_c + rank_df[data[class_col]].reindex(rank_df[data[class_col]].columns, 
                        axis=1).to_numpy()[idxs, range(rank_df[data[class_col]].columns.shape[0])]
        for idxs in idxs_neg:
            W_c = W_c - rank_df[~data[class_col]].reindex(rank_df[~data[class_col]].columns, 
                        axis=1).to_numpy()[idxs, range(rank_df[~data[class_col]].columns.shape[0])]
        idxs = np.array(list(idxs_pos) + list(idxs_neg))
        if len(idxs):
            PL_idxs.append([tuple(idxs[:, i]) for i in range(idxs.shape[1])])
            if alternative == 'less':
                PL.append(norm.cdf((W_c - (nx+add_nx) * (N + 1) / 2) / np.sqrt((nx+add_nx)*(ny+add_ny) * (N + 1)/ 12)))
            elif alternative == 'greater':
                PL.append(1 - norm.cdf((W_c - (nx+add_nx) * (N + 1) / 2) / np.sqrt((nx+add_nx)*(ny+add_ny) * (N + 1)/ 12)))
            else: # two-sided
                PL_c = norm.cdf((W_c - (nx+add_nx) * (N + 1) / 2) / np.sqrt((nx+add_nx)*(ny+add_ny) * (N + 1)/ 12))
                PL_c = np.where(PL_c < 0.5, PL_c, 1 - PL_c) * 2
                PL.append(PL_c)
    PL_idxs = np.array(PL_idxs)[np.array(PL).argmin(axis=0), range(np.array(PL_idxs).shape[1])]        
    return np.array(PL).min(axis=0), PL_idxs

def FDR_WRS_efficient(data, class_col, n_changes, relevant_W, df_ranks, n_always_in_genes, idxs, n_test_for_FDR,
                      nx, ny, tau, alternative='less'):
    pvalues = []
    fdr = []
    N = df_ranks.shape[0]
    r = df_ranks.loc[data[class_col].iloc[idxs].index].values
    pos = data[class_col].iloc[idxs].values * 1
    neg = ~data[class_col].iloc[idxs].values * 1
    pos_reshape = (pos).reshape(n_changes, 1)
    neg_reshape = (neg).reshape(n_changes, 1)
    pos_sum = pos.sum()
    neg_sum = neg.sum()
    W_new = relevant_W + (r * pos_reshape).sum(axis=0) - (r * neg_reshape).sum(axis=0)
    nx_new = nx + pos_sum - neg_sum
    ny_new = ny - pos_sum + neg_sum
    if alternative == 'less':
        pvalues = norm.cdf((W_new - nx_new * (N + 1) / 2) / np.sqrt(nx_new*ny_new * (N + 1)/ 12))
    elif alternative == 'greater':
        pvalues = 1 - norm.cdf((W_new - nx_new * (N + 1) / 2) / np.sqrt(nx_new*ny_new * (N + 1)/ 12))
    else: # two-sided
        pvalues_c = norm.cdf((W_new - nx_new * (N + 1) / 2) / np.sqrt(nx_new*ny_new * (N + 1)/ 12))
        pvalues = np.where(pvalues_c < 0.5, pvalues_c, 1 - pvalues_c) * 2
    pvalues_sorted, sorted_genes = zip(*sorted(zip(pvalues, data.columns[:-1]), reverse=True))    
    n_tests = len(pvalues_sorted) + n_always_in_genes
    for j, p in enumerate(pvalues_sorted):
        i = n_tests - j
        if p * n_test_for_FDR / i <= tau:
            return i, sorted_genes[j:], pvalues_sorted[j:]
    return 0, [], pvalues_sorted

def calc_differentially_expressed_genes(data, tau, class_col, n_changes=1, relevant_label='all', alternative='two-sided'):
    ranks = data.drop(class_col, axis=1).rank()
    W = ranks[~data[class_col]].sum().values
    nx = data[~data[class_col]].shape[0]
    ny = data[data[class_col]].shape[0]
    N = ranks.shape[0]
    if alternative == 'less':
        pvalue_original = norm.cdf((W - nx * (N + 1) / 2) / np.sqrt(nx*ny * (N + 1)/ 12))
        PU, PU_idxs = find_PU(data, class_col, ranks, n_changes, W, pvalue_original, nx, ny, 
                              relevant_label, alternative)
        PL, PL_idxs = find_PL(data, class_col, ranks, n_changes, W, pvalue_original, nx, ny,
                              relevant_label, alternative)  
    elif alternative == 'greater':
        pvalue_original = 1 - norm.cdf((W - nx * (N + 1) / 2) / np.sqrt(nx*ny * (N + 1)/ 12))
        PU, PU_idxs = find_PU(data, class_col, ranks, n_changes, W, pvalue_original, nx, ny, 
                              relevant_label, alternative)
        PL, PL_idxs = find_PL(data, class_col, ranks, n_changes, W, pvalue_original, nx, ny, 
                              relevant_label, alternative)  
    else: # two-sided
        pvalues_c = norm.cdf((W - nx * (N + 1) / 2) / np.sqrt(nx*ny * (N + 1)/ 12))
        pvalue_original = np.where(pvalues_c < 0.5, pvalues_c, 1 - pvalues_c) * 2
        PU, PU_idxs = find_PU(data, class_col, ranks, n_changes, W, pvalues_c, nx, ny, 
                              relevant_label, alternative)
        PL, PL_idxs = find_PL(data, class_col, ranks, n_changes, W, pvalues_c, nx, ny, 
                              relevant_label, alternative)  
    # First round of not relevant genes PU
    thr_U = tau / PU.shape[0]
    PU_c = PU[np.where(PU > thr_U)[0]]
    c_U = 0
    while True:
        PU_n = PU_c[np.where(PU_c > tau * (PU.shape[0] - PU_c.shape[0]) / PU.shape[0])[0]]
        if PU_n.shape == PU_c.shape:
            break
        c_U += 1
        thr_U = tau * (PU.shape[0] - PU_c.shape[0]) / PU.shape[0]
        PU_c = PU_n              
    # First round of not relevant genes PL
    thr_L = tau
    PL_c = PL[np.where(PL <= thr_L)[0]]
    c_L = 0
    while True:
        PL_n = PL_c[np.where(PL_c <= tau * PL_c.shape[0] / PL.shape[0])[0]]
        if PL_n.shape == PL_c.shape:
            break
        c_L += 1
        thr_L = tau * PL_c.shape[0] / PL.shape[0]
        PL_c = PL_n
    n_always_in_genes = (PU <= thr_U).sum()
    always_in_genes = ranks.columns[PU <= thr_U].values
    relevant_ranks = ranks[data.columns[:-1][(PL < thr_L) & (PU > thr_U)]]
    relevant_W = relevant_ranks[~data[class_col]].sum().values
    report = {
        'n_genes': [],
        'gene_list': set()
    }
    
    if alternative =='less':
        pvalues = norm.cdf((relevant_W - nx * (N + 1) / 2) / np.sqrt(nx * ny * (N + 1)/ 12))
    elif alternative == 'greater':
        pvalues = 1 - norm.cdf((relevant_W - nx * (N + 1) / 2) / np.sqrt(nx * ny * (N + 1)/ 12))
    else: # two-sided
        pvalues_c = norm.cdf((relevant_W - nx * (N + 1) / 2) / np.sqrt(nx * ny * (N + 1)/ 12))
        pvalues = np.where(pvalues_c < 0.5, pvalues_c, 1 - pvalues_c) * 2
    pvalues_sorted, sorted_genes = zip(*sorted(zip(pvalues, data.columns[:-1][(PL < thr_L) & (PU > thr_U)]), 
                                               reverse=True))
    n_genes, gene_list, pvalues = 0, [], []
    n_tests = len(pvalues_sorted) + n_always_in_genes
    n_test_for_FDR = ranks.shape[1]
    for j, p in enumerate(pvalues_sorted):
        i = n_tests - j
        if p * n_test_for_FDR / i <= tau:
            n_genes, gene_list, pvalues_res = i, sorted_genes[j:], pvalues_sorted[j:]
            break

    pvalue_original_sorted, sorted_genes = zip(*sorted(zip(pvalue_original, ranks.columns)))
    report['pvalue_original_sorted'] = pvalue_original_sorted
    report['sorted_genes'] = sorted_genes
    report['n_genes'].append(n_genes)
    report['gene_list'] = set(gene_list)
    report['original_list'] = list(gene_list) + list(always_in_genes)
    if relevant_label == 'all':
        comb = [list(x) for x in list(combinations(range(N), n_changes))]
    else:
        idxs = np.array(range(N))
        relevant_idxs = idxs[data[class_col] == relevant_label]
        comb = [list(x) for x in list(combinations(relevant_idxs, n_changes))]

    for idxs in tqdm(comb):
        n_genes, gene_list, pvalues = FDR_WRS_efficient(data, class_col, n_changes, relevant_W, 
                                                        relevant_ranks, n_always_in_genes, idxs,
                                                        n_test_for_FDR, nx, ny, tau, alternative)
        report['n_genes'].append(n_genes)
        report['gene_list'] = report['gene_list'].intersection(set(gene_list))
    report['gene_list'] = list(report['gene_list']) + list(always_in_genes)
    
    return report

def calc_res(title, n_genes, P, predicted_P, TP):
    print(title)
    print('Predicted Positive:', predicted_P)
    N = n_genes - P
    predicted_N = n_genes - predicted_P
    TN = predicted_N - (P - TP)
    FP = N - TN
    FN = P - TP
    print('TP:', TP, 'FP:', FP, 'TN:', TN, 'FN:', FN)
    precision = TP / predicted_P
    print('Precision: {:.3f}'.format(precision))
    recall = TP / P
    print('Recall: {:.3f}'.format(recall))