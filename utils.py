import torch
import random
import numpy as np
import os
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from sklearn import metrics
from munkres import Munkres
import matplotlib.pyplot as plt
from kmeans_gpu import kmeans
import sklearn.preprocessing as preprocess
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score

from sklearn.neighbors import NearestNeighbors


### <--- [MODIFIED] ---------------------------------------
def _postprocess_adj(adj):
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    adj = adj + adj.T
    adj = adj.sign()
    return adj


def _read_edge_list_to_adj(edge_path, num_nodes):
    try:
        edges = np.loadtxt(edge_path, dtype=int)
    except OSError:
        raise FileNotFoundError(f"Cannot find graph file: {edge_path}")
    except ValueError:
        edges = np.empty((0, 2), dtype=int)

    if edges.size == 0:
        adj = sp.csr_matrix((num_nodes, num_nodes), dtype=np.float32)
        return _postprocess_adj(adj)

    if edges.ndim == 1:
        if edges.shape[0] != 2:
            raise ValueError(f"Invalid edge list format in {edge_path}; expected two columns.")
        edges = edges.reshape(1, 2)

    src = edges[:, 0]
    dst = edges[:, 1]
    if src.min() < 0 or dst.min() < 0 or src.max() >= num_nodes or dst.max() >= num_nodes:
        raise ValueError(f"Node index out of range in {edge_path}. Node count: {num_nodes}")

    values = np.ones(edges.shape[0], dtype=np.float32)
    adj = sp.csr_matrix((values, (src, dst)), shape=(num_nodes, num_nodes))
    return _postprocess_adj(adj)


def _discover_npy_triplet(dataset):
    search_dirs = [
        os.path.join('data', 'full_dataset', dataset),
        os.path.join('dataset', dataset),
    ]
    for dataset_dir in search_dirs:
        if not os.path.isdir(dataset_dir):
            continue

        buckets = {}
        for root, _, files in os.walk(dataset_dir):
            for name in files:
                for key in ('feat', 'label', 'adj'):
                    suffix = f'_{key}.npy'
                    if name.endswith(suffix):
                        prefix = name[:-len(suffix)]
                        buckets.setdefault(prefix, {})[key] = os.path.join(root, name)
                        break

        preferred_keys = [dataset]
        if dataset == 'cite':
            preferred_keys.append('citeseer')
        preferred_keys.extend(sorted(buckets.keys()))

        for key in preferred_keys:
            paths = buckets.get(key)
            if paths and {'feat', 'label', 'adj'}.issubset(paths.keys()):
                return paths['feat'], paths['label'], paths['adj']

    return None


def _has_npy_style_dataset(dataset):
    return _discover_npy_triplet(dataset) is not None


def _prepare_label_array(label_array):
    labels = np.asarray(label_array)
    if labels.ndim == 2:
        if labels.shape[1] == 1:
            labels = labels[:, 0]
        else:
            labels = np.argmax(labels, axis=1)
    return np.asarray(labels).reshape(-1).astype(np.int64)


def _load_npy_adj(adj_path):
    adj_obj = np.load(adj_path, allow_pickle=True)
    if isinstance(adj_obj, np.ndarray) and adj_obj.dtype == object and adj_obj.shape == ():
        adj_obj = adj_obj.item()
    if sp.issparse(adj_obj):
        adj = adj_obj.tocsr().astype(np.float32)
    else:
        adj = sp.csr_matrix(np.asarray(adj_obj, dtype=np.float32))
    adj.eliminate_zeros()
    return adj


def load_npy_data(dataset, graph_mode='raw', ae_graph_path=None, knn_k=5):
    triplet = _discover_npy_triplet(dataset)
    if triplet is None:
        raise FileNotFoundError(f"Cannot find npy triplet for dataset: {dataset}")

    feat_path, label_path, adj_path = triplet
    print(f"Loading NPY data: {dataset} ...")
    print(f"Loading features from: {feat_path}")
    features = np.asarray(np.load(feat_path, allow_pickle=True), dtype=np.float32)
    labels = _prepare_label_array(np.load(label_path, allow_pickle=True))

    if features.ndim != 2:
        raise ValueError(f"Invalid feature shape for {dataset}: {features.shape}")
    if labels.shape[0] != features.shape[0]:
        raise ValueError(
            f"Feature/label length mismatch for {dataset}: "
            f"{features.shape[0]} vs {labels.shape[0]}"
        )

    _, _, _, default_ae_graph_path = _resolve_sdcn_paths(dataset)
    if graph_mode == 'raw':
        print(f"Loading original npy graph from: {adj_path}")
        adj = _load_npy_adj(adj_path)
    elif graph_mode == 'knn':
        adj = _build_knn_adj(features, knn_k)
    elif graph_mode == 'ae':
        if ae_graph_path is None or ae_graph_path.strip() == "":
            ae_graph_path = default_ae_graph_path
        print(f"Loading Ae graph from: {ae_graph_path}")
        adj = _read_edge_list_to_adj(ae_graph_path, features.shape[0])
    else:
        raise ValueError(f"Unsupported graph_mode: {graph_mode}. Use 'raw', 'knn' or 'ae'.")

    if adj.shape[0] != features.shape[0] or adj.shape[1] != features.shape[0]:
        raise ValueError(
            f"Feature/adj shape mismatch for {dataset}: "
            f"features={features.shape}, adj={adj.shape}"
        )

    features = torch.FloatTensor(features)
    idx_train = range(len(features))
    idx_val = range(len(features))
    idx_test = range(len(features))

    return adj, features, labels, idx_train, idx_val, idx_test


def _resolve_sdcn_paths(dataset):
    feat_path = os.path.join('data', 'data', f'{dataset}.txt')
    label_path = os.path.join('data', 'data', f'{dataset}_label.txt')
    base_graph_path = os.path.join('data', 'graph', f'{dataset}_graph.txt')
    default_ae_graph_path = os.path.join('data', 'ae_graph', f'{dataset}_ae_graph.txt')
    return feat_path, label_path, base_graph_path, default_ae_graph_path


def _load_sdcn_features_and_labels(dataset):
    feat_path, label_path, _, _ = _resolve_sdcn_paths(dataset)
    try:
        features = np.loadtxt(feat_path, dtype=float)
        labels = np.loadtxt(label_path, dtype=int)
    except OSError as exc:
        raise FileNotFoundError(f"Cannot find {feat_path} or {label_path}") from exc
    return features, labels


def _build_knn_adj(features, knn_k):
    print(f"Constructing KNN graph with k={knn_k}...")
    nbrs = NearestNeighbors(n_neighbors=knn_k + 1, algorithm='ball_tree').fit(features)
    adj = nbrs.kneighbors_graph(features)
    return _postprocess_adj(adj)


### <--- [MODIFIED] ---------------------------------------
def _resolve_existing_base_graph(dataset, knn_k):
    """
    Prefer reproducible base-graph assets before constructing KNN online.
    This keeps non-graph datasets aligned with the AE-pretrain base graph.
    """
    _, _, base_graph_path, _ = _resolve_sdcn_paths(dataset)
    if os.path.exists(base_graph_path):
        return base_graph_path

    knn_graph_path = os.path.join('data', 'graph', f'{dataset}{knn_k}_graph.txt')
    if os.path.exists(knn_graph_path):
        return knn_graph_path

    return None
### ---------------------------------------


def load_sdcn_data(dataset, graph_mode='raw', ae_graph_path=None, knn_k=5):
    print(f"Loading SDCN data: {dataset} ...")
    features, labels = _load_sdcn_features_and_labels(dataset)
    _, _, base_graph_path, default_ae_graph_path = _resolve_sdcn_paths(dataset)

    if graph_mode == 'raw':
        ### <--- [MODIFIED] ---------------------------------------
        base_graph_path = _resolve_existing_base_graph(dataset, knn_k)
        if base_graph_path is not None:
            print(f"Loading base graph from: {base_graph_path}")
            adj = _read_edge_list_to_adj(base_graph_path, features.shape[0])
        else:
            print(f"Base graph not found for {dataset}; fallback to online KNN graph.")
            adj = _build_knn_adj(features, knn_k)
        ### ---------------------------------------
    elif graph_mode == 'knn':
        adj = _build_knn_adj(features, knn_k)
    elif graph_mode == 'ae':
        if ae_graph_path is None or ae_graph_path.strip() == "":
            ae_graph_path = default_ae_graph_path
        print(f"Loading Ae graph from: {ae_graph_path}")
        adj = _read_edge_list_to_adj(ae_graph_path, features.shape[0])
    else:
        raise ValueError(f"Unsupported graph_mode: {graph_mode}. Use 'raw', 'knn' or 'ae'.")

    features = torch.FloatTensor(features)
    idx_train = range(len(features))
    idx_val = range(len(features))
    idx_test = range(len(features))

    return adj, features, labels, idx_train, idx_val, idx_test
### ---------------------------------------

### <--- [MODIFIED] ---------------------------------------
def _has_txt_style_dataset(dataset):
    feat_path, label_path, _, _ = _resolve_sdcn_paths(dataset)
    return os.path.exists(feat_path) and os.path.exists(label_path)
### ---------------------------------------


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=bool)


def load_data(dataset, graph_mode='raw', ae_graph_path=None, knn_k=5):
    ### <--- [MODIFIED] ---------------------------------------
    # Unified project data entry:
    # 1) Prefer full_dataset npy triplets to preserve original adj information.
    # 2) Fall back to current SDCN-style txt / edge-list inputs.
    # 3) In raw mode, graph A is the original graph if available; otherwise KNN.
    # 4) The legacy ind.* citation loader is retired in this project.
    if _has_npy_style_dataset(dataset):
        return load_npy_data(
            dataset,
            graph_mode=graph_mode,
            ae_graph_path=ae_graph_path,
            knn_k=knn_k
        )

    if not _has_txt_style_dataset(dataset):
        feat_path, label_path, _, _ = _resolve_sdcn_paths(dataset)
        npy_hint = os.path.join('data', 'full_dataset', dataset, f'{dataset}_feat.npy')
        raise FileNotFoundError(
            f"Dataset '{dataset}' must provide npy triplet or txt inputs under the current project format. "
            f"Expected npy near {npy_hint}, or txt files: {feat_path}, {label_path}"
        )

    return load_sdcn_data(
        dataset,
        graph_mode=graph_mode,
        ae_graph_path=ae_graph_path,
        knn_k=knn_k
    )
    ### ---------------------------------------


def load_wiki():
    f = open('data/graph.txt', 'r')
    adj, xind, yind = [], [], []
    for line in f.readlines():
        line = line.split()

        xind.append(int(line[0]))
        yind.append(int(line[1]))
        adj.append([int(line[0]), int(line[1])])
    f.close()
    ##print(len(adj))

    f = open('data/group.txt', 'r')
    label = []
    for line in f.readlines():
        line = line.split()
        label.append(int(line[1]))
    f.close()

    f = open('data/tfidf.txt', 'r')
    fea_idx = []
    fea = []
    adj = np.array(adj)
    adj = np.vstack((adj, adj[:, [1, 0]]))
    adj = np.unique(adj, axis=0)

    labelset = np.unique(label)
    labeldict = dict(zip(labelset, range(len(labelset))))
    label = np.array([labeldict[x] for x in label])
    adj = sp.csr_matrix((np.ones(len(adj)), (adj[:, 0], adj[:, 1])), shape=(len(label), len(label)))

    for line in f.readlines():
        line = line.split()
        fea_idx.append([int(line[0]), int(line[1])])
        fea.append(float(line[2]))
    f.close()

    fea_idx = np.array(fea_idx)
    features = sp.csr_matrix((fea, (fea_idx[:, 0], fea_idx[:, 1])), shape=(len(label), 4973)).toarray()
    scaler = preprocess.MinMaxScaler()
    # features = preprocess.normalize(features, norm='l2')
    features = scaler.fit_transform(features)
    features = torch.FloatTensor(features)

    return adj, features, label


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def decompose(adj, dataset, norm='sym', renorm=True):
    adj = sp.coo_matrix(adj)
    ident = sp.eye(adj.shape[0])
    if renorm:
        adj_ = adj + ident
    else:
        adj_ = adj

    rowsum = np.array(adj_.sum(1))

    if norm == 'sym':
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
        adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
        laplacian = ident - adj_normalized
    evalue, evector = np.linalg.eig(laplacian.toarray())
    np.save(dataset + ".npy", evalue)
    print(max(evalue))
    exit(1)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    n, bins, patches = ax.hist(evalue, 50, facecolor='g')
    plt.xlabel('Eigenvalues')
    plt.ylabel('Frequncy')
    fig.savefig("eig_renorm_" + dataset + ".png")


def preprocess_graph(adj, layer, norm='sym', renorm=True):
    adj = sp.coo_matrix(adj)
    ident = sp.eye(adj.shape[0])
    if renorm:
        adj_ = adj + ident
    else:
        adj_ = adj

    rowsum = np.array(adj_.sum(1))

    if norm == 'sym':
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
        adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
        laplacian = ident - adj_normalized
    elif norm == 'left':
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -1.).flatten())
        adj_normalized = degree_mat_inv_sqrt.dot(adj_).tocoo()
        laplacian = ident - adj_normalized

    # reg = [2 / 3] * (layer)
    reg = [1] * (layer)

    adjs = []
    for i in range(len(reg)):
        adjs.append(ident - (reg[i] * laplacian))

    return adjs


def laplacian(adj):
    rowsum = np.array(adj.sum(1))
    degree_mat = sp.diags(rowsum.flatten())
    lap = degree_mat - adj
    return torch.FloatTensor(lap.toarray())


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def get_roc_score(emb, adj_orig, edges_pos, edges_neg):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Predict on test set of edges
    adj_rec = np.dot(emb, emb.T)
    preds = []
    pos = []
    for e in edges_pos:
        preds.append(sigmoid(adj_rec[e[0], e[1]]))
        pos.append(adj_orig[e[0], e[1]])

    preds_neg = []
    neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
        neg.append(adj_orig[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score


def cluster_acc(y_true, y_pred):
    """
    calculate clustering acc and f1-score
    Args:
        y_true: the ground truth
        y_pred: the clustering id

    Returns: acc and f1-score
    """
    y_true = y_true - np.min(y_true)
    l1 = list(set(y_true))
    num_class1 = len(l1)
    l2 = list(set(y_pred))
    num_class2 = len(l2)
    ind = 0
    if num_class1 != num_class2:
        for i in l1:
            if i in l2:
                pass
            else:
                y_pred[ind] = i
                ind += 1
    l2 = list(set(y_pred))
    numclass2 = len(l2)
    if num_class1 != numclass2:
        print('error')
        return
    cost = np.zeros((num_class1, numclass2), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if y_pred[i1] == c2]
            cost[i][j] = len(mps_d)
    m = Munkres()
    cost = cost.__neg__().tolist()
    indexes = m.compute(cost)
    new_predict = np.zeros(len(y_pred))
    for i, c in enumerate(l1):
        c2 = l2[indexes[i][1]]
        ai = [ind for ind, elm in enumerate(y_pred) if elm == c2]
        new_predict[ai] = c
    acc = metrics.accuracy_score(y_true, new_predict)
    f1_macro = metrics.f1_score(y_true, new_predict, average='macro')
    return acc, f1_macro


def eva(y_true, y_pred, show_details=True):
    """
    evaluate the clustering performance
    Args:
        y_true: the ground truth
        y_pred: the predicted label
        show_details: if print the details
    Returns: None
    """
    acc, f1 = cluster_acc(y_true, y_pred)
    nmi = nmi_score(y_true, y_pred, average_method='arithmetic')
    ari = ari_score(y_true, y_pred)
    if show_details:
        print(':acc {:.4f}'.format(acc), ', nmi {:.4f}'.format(nmi), ', ari {:.4f}'.format(ari),
              ', f1 {:.4f}'.format(f1))
    return acc, nmi, ari, f1


def load_graph_data(dataset_name, show_details=False):
    """
    load graph data
    :param dataset_name: the name of the dataset
    :param show_details: if show the details of dataset
    - dataset name
    - features' shape
    - labels' shape
    - adj shape
    - edge num
    - category num
    - category distribution
    :return: the features, labels and adj
    """
    load_path = "dataset/" + dataset_name + "/" + dataset_name
    feat = np.load(load_path+"_feat.npy", allow_pickle=True)
    label = np.load(load_path+"_label.npy", allow_pickle=True)
    adj = np.load(load_path+"_adj.npy", allow_pickle=True)
    if show_details:
        print("++++++++++++++++++++++++++++++")
        print("---details of graph dataset---")
        print("++++++++++++++++++++++++++++++")
        print("dataset name:   ", dataset_name)
        print("feature shape:  ", feat.shape)
        print("label shape:    ", label.shape)
        print("adj shape:      ", adj.shape)
        print("undirected edge num:   ", int(np.nonzero(adj)[0].shape[0]/2))
        print("category num:          ", max(label)-min(label)+1)
        print("category distribution: ")
        for i in range(max(label)+1):
            print("label", i, end=":")
            print(len(label[np.where(label == i)]))
        print("++++++++++++++++++++++++++++++")

    return feat, label, adj


def normalize_adj(adj, self_loop=True, symmetry=False):
    """
    normalize the adj matrix
    :param adj: input adj matrix
    :param self_loop: if add the self loop or not
    :param symmetry: symmetry normalize or not
    :return: the normalized adj matrix
    """
    # add the self_loop
    if self_loop:
        adj_tmp = adj + np.eye(adj.shape[0])
    else:
        adj_tmp = adj

    # calculate degree matrix and it's inverse matrix
    d = np.diag(adj_tmp.sum(0))
    d_inv = np.linalg.inv(d)

    # symmetry normalize: D^{-0.5} A D^{-0.5}
    if symmetry:
        sqrt_d_inv = np.sqrt(d_inv)
        norm_adj = np.matmul(np.matmul(sqrt_d_inv, adj_tmp), adj_tmp)

    # non-symmetry normalize: D^{-1} A
    else:
        norm_adj = np.matmul(d_inv, adj_tmp)

    return norm_adj


def setup_seed(seed):
    """
    setup random seed to fix the result
    Args:
        seed: random seed
    Returns: None
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def clustering(feature, true_labels, cluster_num):
    ### <--- [MODIFIED] ---------------------------------------
    if isinstance(feature, torch.Tensor):
        km_device = feature.device
    else:
        km_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    predict_labels, dis, initial = kmeans(X=feature, num_clusters=cluster_num, distance="euclidean", device=km_device)
    ### ---------------------------------------
    ### <--- [MODIFIED] ---------------------------------------
    predict_labels_np = predict_labels.detach().cpu().numpy()
    acc, nmi, ari, f1 = eva(true_labels, predict_labels_np, show_details=False)
    return 100 * acc, 100 * nmi, 100 * ari, 100 * f1, predict_labels_np, dis
    ### ---------------------------------------
