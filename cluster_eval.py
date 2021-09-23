from scipy.optimize import linear_sum_assignment as lsa
import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as NMI
from sklearn.metrics import f1_score


def label_accuracy(pred: torch.Tensor, labels: torch.Tensor) -> float:
    assert pred.size() == labels.size()
    return len([1 for x, y in zip(pred, labels) if x == y]) / pred.size(0)


def to_numpy(arr):
    return arr.detach().cpu().numpy()


def assign_components(components, clusters, measure='count'):
    """
    :param measure: Measurement for similarity
    :param components: The array encoding which component each node belongs to
    :param clusters: The array encoding which cluster each node belongs to
    :return: The reassigned component label array
    """
    assert measure in ['count', 'rate'], "`measure` should be in ['count', 'rate']"

    def shared_nodes(comp, clu):
        """
        Return the number of common nodes between `comp` and `clu`.
        :param comp: The numpy.ndarray of type bool indicating whether each nodes belong to the component
        :param clu: The numpy.ndarray of type bool indicating whether each nodes belong to the cluster
        :return:
        """
        assert comp.shape == clu.shape, "The input shape of `comp` and `clu` should be equal."
        return np.sum(np.logical_and(comp, clu).astype(np.int))

    def shared_rate(comp, clu):
        """
        Return (the number of common nodes between `comp` and `clu`) / min { |comp| , |clu| }
        :param comp: The numpy.ndarray of type bool indicating whether each nodes belong to the component
        :param clu: The numpy.ndarray of type bool indicating whether each nodes belong to the cluster
        :return:
        """
        return shared_nodes(comp, clu) / np.max([np.sum(comp), np.sum(clu)])

    f = shared_nodes if measure == 'count' else shared_rate
    n_comp = components.max() + 1
    n_clu = clusters.max() + 1
    c = np.ndarray([n_clu, n_comp])
    for i in range(n_clu):
        for j in range(n_comp):
            c[i][j] = -f(components == j, clusters == i)
    _, col_idx = lsa(c)
    assign = np.ndarray(n_comp)
    assign.fill(-1.)
    for clu, comp in enumerate(col_idx):
        assign[comp] = clu
    assign = np.where(assign != -1, assign, c.argmin(axis=0))
    comp = components.copy()
    for i in range(components.shape[0]):
        comp[i] = assign[components[i]]
    return comp


def assigned_accuracy(components: torch.Tensor, clusters: torch.Tensor) -> float:
    components = to_numpy(components)
    clusters_ = to_numpy(clusters)
    assigned_comp = assign_components(components, clusters_)
    return label_accuracy(torch.from_numpy(assigned_comp).cuda(), clusters)


def kmeans(embeddings, labels):
    n_clusters = labels.max().item() + 1
    clusters_ = KMeans(n_clusters=n_clusters, n_jobs=10, init='k-means++', n_init=20
                       ).fit(embeddings.cpu().detach().numpy()).labels_
    clusters = torch.from_numpy(clusters_).to(torch.long).cuda()
    labels_ = labels.cpu().detach().numpy()
    accuracy = assigned_accuracy(clusters, labels)
    assigned_clusters = assign_components(clusters_, labels_)
    nmi = NMI(labels_, assigned_clusters, average_method='arithmetic')
    f1mi = f1_score(labels_, assigned_clusters, average='micro')
    f1ma = f1_score(labels_, assigned_clusters, average='macro')
    return {
        'acc': accuracy,
        'nmi': nmi,
        'micro_f1': f1mi,
        'macro_f1': f1ma
    }