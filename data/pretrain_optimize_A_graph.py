import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from torch.nn import Linear
### <--- [MODIFIED] ---------------------------------------
try:
    from kmeans_gpu import kmeans
except ImportError:
    import sys
    from pathlib import Path
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from kmeans_gpu import kmeans
### ---------------------------------------

try:
    from evaluation import eva
except ImportError:
    from data.evaluation import eva


class AE(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z):
        super(AE, self).__init__()
        self.enc_1 = Linear(n_input, n_enc_1)
        self.enc_2 = Linear(n_enc_1, n_enc_2)
        self.enc_3 = Linear(n_enc_2, n_enc_3)
        self.z_layer = Linear(n_enc_3, n_z)

        self.dec_1 = Linear(n_z, n_dec_1)
        self.dec_2 = Linear(n_dec_1, n_dec_2)
        self.dec_3 = Linear(n_dec_2, n_dec_3)
        self.x_bar_layer = Linear(n_dec_3, n_input)

    def forward(self, x):
        enc_h1 = F.relu(self.enc_1(x))
        enc_h2 = F.relu(self.enc_2(enc_h1))
        enc_h3 = F.relu(self.enc_3(enc_h2))
        z = self.z_layer(enc_h3)

        dec_h1 = F.relu(self.dec_1(z))
        dec_h2 = F.relu(self.dec_2(dec_h1))
        dec_h3 = F.relu(self.dec_3(dec_h2))
        x_bar = self.x_bar_layer(dec_h3)

        return x_bar, z, enc_h3


class LoadDataset(Dataset):
    def __init__(self, data):
        self.x = data

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])).float(), \
               torch.from_numpy(np.array(idx))


### <--- [MODIFIED] ---------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='acm')
    parser.add_argument('--feature_path', type=str, default='')
    parser.add_argument('--label_path', type=str, default='')
    parser.add_argument('--base_graph_path', type=str, default='')
    parser.add_argument('--out_graph_path', type=str, default='')
    parser.add_argument('--cluster_num', type=int, default=3)
    parser.add_argument('--ae_k', type=int, default=15)
    parser.add_argument('--sim_method', type=str, default='cos', choices=['heat', 'cos', 'ncos'])
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--n_enc_1', type=int, default=500)
    parser.add_argument('--n_enc_2', type=int, default=500)
    parser.add_argument('--n_enc_3', type=int, default=2000)
    parser.add_argument('--n_dec_1', type=int, default=2000)
    parser.add_argument('--n_dec_2', type=int, default=500)
    parser.add_argument('--n_dec_3', type=int, default=500)
    parser.add_argument('--n_z', type=int, default=3)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--model_save_path', type=str, default='')
    parser.add_argument('--pretrain_seed', type=int, default=None,
                        help='optional seed for AE pretraining; default keeps original random behavior')
    parser.add_argument('--graph_seed', type=int, default=None,
                        help='optional seed for graph-generation kmeans; default keeps original random behavior')
    return parser.parse_args()


def _set_optional_seed(seed):
    if seed is None:
        return

    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _discover_npy_triplet(dataset):
    dataset_dir = os.path.join('data', 'full_dataset', dataset)
    if not os.path.isdir(dataset_dir):
        return None

    buckets = {}
    for root, _, files in os.walk(dataset_dir):
        for name in files:
            for key in ('feat', 'label', 'adj'):
                suffix = f'_{key}.npy'
                if name.endswith(suffix):
                    prefix = name[:-len(suffix)]
                    buckets.setdefault(prefix, {})[key] = os.path.join(root, name)
                    break

    preferred = [dataset]
    if dataset == 'cite':
        preferred.append('citeseer')
    preferred.extend(sorted(buckets.keys()))
    for key in preferred:
        triplet = buckets.get(key)
        if triplet and {'feat', 'label', 'adj'}.issubset(triplet.keys()):
            return triplet['feat'], triplet['label'], triplet['adj']
    return None


def _prepare_label_array(label_array):
    labels = np.asarray(label_array)
    if labels.ndim == 2:
        if labels.shape[1] == 1:
            labels = labels[:, 0]
        else:
            labels = np.argmax(labels, axis=1)
    return np.asarray(labels).reshape(-1).astype(np.int64)


def resolve_paths(args):
    args.npy_triplet = None
    if args.feature_path.strip() == '' and args.label_path.strip() == '':
        args.npy_triplet = _discover_npy_triplet(args.dataset)

    if args.feature_path.strip() == '' and args.npy_triplet is None:
        args.feature_path = os.path.join('data', 'data', f'{args.dataset}.txt')
    if args.label_path.strip() == '' and args.npy_triplet is None:
        args.label_path = os.path.join('data', 'data', f'{args.dataset}_label.txt')
    if args.base_graph_path.strip() == '':
        if args.npy_triplet is None:
            args.base_graph_path = os.path.join('data', 'graph', f'{args.dataset}_graph.txt')
    if args.out_graph_path.strip() == '':
        args.out_graph_path = os.path.join('data', 'ae_graph', f'{args.dataset}_ae_graph.txt')
    if args.model_save_path.strip() == '':
        args.model_save_path = os.path.join('pretrain_graph', f'{args.dataset}_ae_pretrain.pkl')
    return args


def load_base_edges(path):
    try:
        edges = np.loadtxt(path, dtype=int)
    except OSError:
        raise FileNotFoundError(f'Cannot find base graph file: {path}')
    except ValueError:
        edges = np.empty((0, 2), dtype=int)

    if edges.size == 0:
        return set()
    if edges.ndim == 1:
        if edges.shape[0] != 2:
            raise ValueError(f'Invalid edge format in {path}; expected two columns.')
        edges = edges.reshape(1, 2)

    return {(int(u), int(v)) for u, v in edges[:, :2]}


def load_base_edges_from_npy(adj_path):
    adj = np.load(adj_path, allow_pickle=True)
    adj = np.asarray(adj)
    rows, cols = np.nonzero(adj)
    edge_set = set()
    for u, v in zip(rows.tolist(), cols.tolist()):
        if u == v:
            continue
        edge_set.add((int(u), int(v)))
    return edge_set


def load_features_and_labels(args):
    if args.npy_triplet is not None:
        feat_path, label_path, _ = args.npy_triplet
        print(f'Loading NPY features from: {feat_path}')
        x = np.asarray(np.load(feat_path, allow_pickle=True), dtype=np.float32)
        y = _prepare_label_array(np.load(label_path, allow_pickle=True))
        return x, y

    x = np.loadtxt(args.feature_path, dtype=float)
    y = np.loadtxt(args.label_path, dtype=int)
    return x, y


### <--- [MODIFIED] ---------------------------------------
def _gpu_kmeans_labels(z, cluster_num, device):
    pred_labels, _, _ = kmeans(
        X=z.detach(),
        num_clusters=cluster_num,
        distance='euclidean',
        device=device
    )
    ### <--- [MODIFIED] ---------------------------------------
    return pred_labels.detach().cpu().numpy()
    ### ---------------------------------------


def construct_graph_multi(features, label1, label2, label3, base_edge_set, out_path, topk=15, method='cos', device=torch.device('cuda')):
    num = len(label1)
    work_features = torch.from_numpy(features).float().to(device)

    if method == 'heat':
        dist = torch.cdist(work_features, work_features, p=2)
        sim = torch.exp(-0.5 * dist.pow(2))
    elif method == 'cos':
        work_features = (work_features > 0).float()
        sim = work_features @ work_features.T
    elif method == 'ncos':
        work_features = (work_features > 0).float()
        work_features = work_features / work_features.sum(dim=1, keepdim=True).clamp_min(1e-12)
        sim = work_features @ work_features.T
    else:
        raise ValueError(f'Unsupported sim method: {method}')

    k = min(topk + 1, sim.shape[1])
    inds = torch.topk(sim, k=k, dim=1, largest=True).indices.cpu().numpy()
    label1 = np.asarray(label1)
    label2 = np.asarray(label2)
    label3 = np.asarray(label3)

    out_dir = os.path.dirname(out_path)
    if out_dir != '':
        os.makedirs(out_dir, exist_ok=True)
    counter = 0
    with open(out_path, 'w') as f:
        for i, neighbors in enumerate(inds):
            for vv in neighbors:
                if vv == i:
                    continue

                if label1[vv] != label1[i]:
                    counter += 1

                same_cluster_in_3_views = (
                    label1[vv] == label1[i]
                    and label2[i] == label2[vv]
                    and label3[i] == label3[vv]
                )
                in_base_graph = (i, vv) in base_edge_set or (vv, i) in base_edge_set

                if same_cluster_in_3_views or in_base_graph:
                    f.write(f'{i} {vv}\n')

    print(f'[{os.path.basename(out_path)}] m{topk}+: {counter / (num * topk):.6f}')
### ---------------------------------------


def pretrain_ae(model, dataset, y, cluster_num, raw_features, base_edge_set, args, device):
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    print(model)
    optimizer = Adam(model.parameters(), lr=args.lr)
    ### <--- [MODIFIED] ---------------------------------------
    full_x = torch.as_tensor(dataset.x, dtype=torch.float32, device=device)
    ### ---------------------------------------

    for epoch in range(args.epochs):
        for _, (x, _) in enumerate(train_loader):
            x = x.to(device)

            x_bar, _, _ = model(x)
            loss = F.mse_loss(x_bar, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            x = full_x
            x_bar, z, _ = model(x)
            loss = F.mse_loss(x_bar, x)
            print(f'{epoch} loss: {loss}')

            if epoch != args.epochs - 1:
                ### <--- [MODIFIED] ---------------------------------------
                labels = _gpu_kmeans_labels(z, cluster_num, device)
                eva(y, labels, epoch)
                ### ---------------------------------------
            else:
                print('Do kmeans and K-NN to optimize the neighbor graph!')
                ### <--- [MODIFIED] ---------------------------------------
                if args.graph_seed is not None:
                    print(f'Using fixed graph seed: {args.graph_seed}')
                graph_seed = None if args.graph_seed is None else int(args.graph_seed)
                _set_optional_seed(graph_seed)
                labels1 = _gpu_kmeans_labels(z, cluster_num, device)
                eva(y, labels1, epoch)
                _set_optional_seed(None if graph_seed is None else graph_seed + 1)
                labels2 = _gpu_kmeans_labels(z, cluster_num, device)
                eva(y, labels2, epoch)
                _set_optional_seed(None if graph_seed is None else graph_seed + 2)
                labels3 = _gpu_kmeans_labels(z, cluster_num, device)
                eva(y, labels3, epoch)
                ### ---------------------------------------

                construct_graph_multi(
                    raw_features,
                    labels1,
                    labels2,
                    labels3,
                    base_edge_set,
                    args.out_graph_path,
                    topk=args.ae_k,
                    method=args.sim_method,
                    device=device
                )

    ### <--- [MODIFIED] ---------------------------------------
    model_save_dir = os.path.dirname(args.model_save_path)
    if model_save_dir != '':
        os.makedirs(model_save_dir, exist_ok=True)
    torch.save(model.state_dict(), args.model_save_path)
    ### ---------------------------------------
    print(f'Saved AE model to: {args.model_save_path}')


### <--- [MODIFIED] ---------------------------------------
def main():
    args = resolve_paths(parse_args())

    x, y = load_features_and_labels(args)
    if args.base_graph_path.strip() != '':
        base_edge_set = load_base_edges(args.base_graph_path)
    elif args.npy_triplet is not None:
        base_edge_set = load_base_edges_from_npy(args.npy_triplet[2])
    else:
        raise FileNotFoundError(
            f"Cannot find base graph for dataset {args.dataset}. "
            f"Expected edge list under data/graph or an npy adj triplet."
        )

    if args.device == 'cuda' and not torch.cuda.is_available():
        print('CUDA is not available; fallback to CPU.')
        device = torch.device('cpu')
    else:
        device = torch.device(args.device)

    if args.pretrain_seed is not None:
        print(f'Using fixed pretrain seed: {args.pretrain_seed}')
    _set_optional_seed(args.pretrain_seed)

    model = AE(
        n_enc_1=args.n_enc_1,
        n_enc_2=args.n_enc_2,
        n_enc_3=args.n_enc_3,
        n_dec_1=args.n_dec_1,
        n_dec_2=args.n_dec_2,
        n_dec_3=args.n_dec_3,
        n_input=x.shape[1],
        n_z=args.n_z,
    ).to(device)

    dataset = LoadDataset(x)
    pretrain_ae(model, dataset, y, args.cluster_num, x, base_edge_set, args, device)


if __name__ == '__main__':
    main()
### ---------------------------------------
