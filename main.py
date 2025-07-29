import argparse
import copy
import json
import glob
import warnings
from loader import DatasetLoader
from alive_progress import alive_bar
import random
import numpy as np
import os
import torch.nn as nn
from ChebnetII_pro import get_hypergraph_laplacian
from model import LogReg, Model
from UFGConv import get_dlist
import time
from utils import get_dataset_specific_settings
warnings.filterwarnings("ignore")
from evaluation import (End2EndNN, train_node_classification_finetune,
                        train_node_classification_linear)



def load_state_dict_with_prefix(model, pretrained_state_dict, prefix='encoder.'):
    model_dict = model.state_dict()
    new_state_dict = {}
    for k, v in pretrained_state_dict.items():
        new_key = k
        if not k.startswith('classifier.') and not k.startswith('encoder.'):
            new_key = prefix + k
        if new_key in model_dict and model_dict[new_key].shape == v.shape:
            new_state_dict[new_key] = v
        else:
            print(f"Skipping key {k}: new_key={new_key} not in model or shape mismatch")
    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict, strict=False)
    print("Loaded pretrained state_dict with prefix adjustment")
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
def parse_args():
    parser = argparse.ArgumentParser(description="Hypergraph Model Training")
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--dataset', type=str, default='cora_cocitation', help='Dataset name')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.005, help='Learning rate')
    parser.add_argument('--lr1', type=float, default=0.005, help='Learning rate 1')
    parser.add_argument('--lr2', type=float, default=0.005, help='Learning rate 2')
    parser.add_argument('--wd', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--wd1', type=float, default=5e-4, help='Weight decay 1')
    parser.add_argument('--wd2', type=float, default=5e-4, help='Weight decay 2')
    parser.add_argument('--hid_dim', type=int, default=512, help='Hidden dimension')
    parser.add_argument('--K', type=int, default=10, help='Chebyshev order')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--dprate', type=float, default=0.4, help='Dropout rate for propagation')
    parser.add_argument('--is_bns', type=bool, default=True, help='Use batch normalization')
    parser.add_argument('--act_fn', type=str, default='relu', help='Activation function')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device')
    parser.add_argument('--Lev', type=int, default=5, help='Level')
    parser.add_argument('--s', type=int, default=2, help='S parameter')
    parser.add_argument('--n', type=int, default=2, help='NA parameter')
    parser.add_argument('--FrameType', type=str, default='Haar', help='Frame type')
    parser.add_argument('--HighpassFilter', type=str, default='DOS', help='High-pass filter type')
    parser.add_argument('--LowpassFilter', type=str, default='g_low_pass', help='Low-pass filter type')
    parser.add_argument('--tasks', nargs='+', type=str, default=['pretrain'],
                        choices=['pretrain', 'finetune', 'linear', 'edge'])
    parser.add_argument('--skip_pretrain', action='store_true',
                        help='Skip pretraining if model exists')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to pretrained model for evaluation tasks')
    parser.add_argument('--freeze_encoder', action='store_true')
    parser.add_argument('--ft_wd', type=float, default=1e-5)
    parser.add_argument('--ft_lr', type=float, default=0.01)
    parser.add_argument('--ft_epochs', type=int, default=200)
    return parser.parse_args()


if __name__ == "__main__":
    import torch

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.reset_accumulated_memory_stats()
    args = parse_args()
    print(args)
    if args.gpu != -1 and torch.cuda.is_available():
        args.device = "cuda:{}".format(args.gpu)
    else:
        args.device = "cpu"
    args = parse_args()
    print("Original args:", args)
    dataset_settings = get_dataset_specific_settings(args.dataset)
    print(f"Applying optimized settings for dataset: {args.dataset}")
    for key, value in dataset_settings.items():
        if hasattr(args, key):
            setattr(args, key, value)
            print(f"Setting {key} = {value}")
    if 'coauthorship' in args.dataset:
        args.HighpassFilter = 'Linear'
        print(f"Setting HighpassFilter = Linear for coauthorship network")
    print("Optimized args:", args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    # Step 1: Load data
    data = DatasetLoader().load(args.dataset).to(args.device)
    feat = data.features.float()
    print("feat.shape:", feat.shape)
    label = data.labels
    hyperedge_index = data.hyperedge_index
    device = args.device
    FrameType = args.FrameType
    n = args.n
    s = args.s
    n_feat = feat.shape[1]
    n_classes = np.unique(label.cpu().numpy()).shape[0]

    feat = feat.to(args.device)
    hyperedge_index = hyperedge_index.to(device)
    n_node = feat.shape[0]
    lbl1 = torch.ones(n_node * 2, dtype=torch.float32)
    lbl2 = torch.zeros(n_node * 2, dtype=torch.float32)
    lbl = torch.cat((lbl1, lbl2))
    hyperedge_index, norm = get_hypergraph_laplacian(hyperedge_index.cpu(), n_node)
    r, Lev, d_list = get_dlist(hyperedge_index, norm, args, feat, device)
    lambda_contrast = 0.1
    lambda_ortho = 0.1
    lambda_edge = 0.1
    model = Model(
        in_dim=n_feat,
        hidden_dim=args.hid_dim,
        K=args.K,
        dprate=args.dprate,
        dropout=args.dropout,
        is_bns=args.is_bns,
        act_fn=args.act_fn,
        r=r,
        Lev=Lev,
        num_nodes=n_node,
        device=device,
        highpass_filter=args.HighpassFilter,
        lowpass_filter=args.LowpassFilter
    )
    model = model.to(args.device)
    lbl = lbl.to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=args.epochs,
        pct_start=0.3,
        div_factor=25.0,
        final_div_factor=10000.0
    )
    loss_fn = nn.BCEWithLogitsLoss()
    early_stopping = EarlyStopping(patience=100, min_delta=1e-4)
    print("=== Training without Early Stopping ===")
    best = float("inf")
    cnt_wait = 0
    best_t = 0
    save_dir = 'pkl'
    args.patience = 100
    tag = str(int(time.time()))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, 'best_model_' + args.dataset + tag + '.pkl')
    edge_weight = None
    loss_fn_ce = nn.CrossEntropyLoss()
    with alive_bar(args.epochs) as bar:
        n_node, n_feat = feat.shape
        noise_scale = 0.1
        grad_clip = min(1.0, 10.0 / np.sqrt(n_feat))
        for epoch in range(args.epochs):
            model.train()
            optimizer.zero_grad()
            shuf_idx = np.random.permutation(n_node)
            shuf_feat = feat[shuf_idx, :]
            noise_scale = 0.1 * (1 - epoch / args.epochs)
            shuf_feat = shuf_feat + torch.randn_like(shuf_feat) * noise_scale
            if epoch % 3 == 0:
                mask = torch.FloatTensor(n_node, n_feat).uniform_() > 0.2
                mask = mask.to(args.device)
                shuf_feat = shuf_feat * mask
            logits, contrast_loss = model(hyperedge_index, feat, shuf_feat, edge_weight, d_list)
            current_lambda_contrast = min(0.1 + 0.3 * (epoch / args.epochs), 0.4)
            bce_loss = loss_fn(logits, lbl)
            loss = bce_loss + current_lambda_contrast * contrast_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            early_stopping(loss)
            if early_stopping.early_stop:
                print("Early stopping triggered at epoch", epoch)
                break
            if loss < best:
                best = loss
                best_t = epoch
                cnt_wait = 0
                torch.save(model.state_dict(), save_path)
                print(f"Model saved to {save_path}")
            else:
                cnt_wait += 1
            if cnt_wait == args.patience:
                print("Early stopping")
                break
            bar()
        model.load_state_dict(torch.load(save_path))
        model.eval()
    edge_weight = None
    embeds = model.get_embedding(hyperedge_index, feat, edge_weight, d_list)
    embeds = embeds.cpu().detach()
    embeds_np = embeds.numpy()
    if np.any(np.isnan(embeds_np)) or np.any(np.isinf(embeds_np)):
        print("Warning: embeds_np contains NaN or inf values. Replacing them with 0.")
        embeds_np = np.nan_to_num(embeds_np, nan=0.0, posinf=1.0, neginf=-1.0)

    results = {}
    tag = str(int(time.time()))
    save_dir = 'pkl'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    splits = data.load_all_splits()
    if len(splits) != 20:
        raise ValueError(f"Expected 20 splits, but found {len(splits)}")
    results_file = os.path.join(save_dir, f'results_{args.dataset}_{tag}.txt')
    if 'finetune' in args.tasks:
        print("\n=== Node Classification (Fine-tuning) ===")

        if args.model_path is None:
            model_files = glob.glob(os.path.join(save_dir, f'best_model_{args.dataset}*.pkl'))
            if model_files:
                args.model_path = max(model_files, key=os.path.getctime)
            else:
                if os.path.exists(save_path):
                    args.model_path = save_path
                else:
                    raise FileNotFoundError(f"No pretrained model found for dataset {args.dataset}.")

        if not os.path.exists(args.model_path):
            raise FileNotFoundError(f"Pretrained model not found at {args.model_path}.")

        pretrained_model = Model(
            in_dim=n_feat,
            hidden_dim=args.hid_dim,
            K=args.K,
            dprate=args.dprate,
            dropout=args.dropout,
            is_bns=args.is_bns,
            act_fn=args.act_fn,
            r=r,
            Lev=Lev,
            num_nodes=n_node,
            device=device,
            highpass_filter=args.HighpassFilter,
            lowpass_filter=args.LowpassFilter
        ).to(device)
        end2end_model = End2EndNN(
            encoder=pretrained_model,
            hidden_dim=args.hid_dim,
            num_classes=n_classes
        ).to(device)
        pretrained_state_dict = torch.load(args.model_path, map_location=device)
        load_state_dict_with_prefix(end2end_model, pretrained_state_dict, prefix='encoder.')

        initial_model_state = copy.deepcopy(end2end_model.state_dict())
        if getattr(args, 'freeze_encoder', False):
            print("Freezing encoder parameters, only fine-tuning the classification head")
            for param in end2end_model.encoder.parameters():
                param.requires_grad = False

        results_list = []
        for split_idx, (train_mask, val_mask, test_mask) in enumerate(splits, 1):
            seed = 42 + split_idx
            train_size = train_mask.sum().item()
            val_size = val_mask.sum().item()
            test_size = test_mask.sum().item()
            end2end_model.load_state_dict(initial_model_state)
            dataset_loader = DatasetLoader()
            dataset_name = args.dataset.lower()
            if dataset_name in dataset_loader.dataset_map:
                dataset_type, dataset_name = dataset_loader.dataset_map[dataset_name]
            else:
                raise ValueError(f"Unsupported dataset: {dataset_name}")
            val_acc_ft, test_acc_ft = train_node_classification_finetune(
                model=end2end_model,
                data=data,
                device=device,
                d_list=d_list,
                train_mask=train_mask,
                val_mask=val_mask,
                test_mask=test_mask,
                seed=seed,
                lr=args.ft_lr,
                weight_decay=args.ft_wd,
                epochs=args.ft_epochs,
                dataset_type=dataset_type,
                dataset_name=dataset_name
            )
            results_list.append(test_acc_ft)
            print(f"Split {split_idx}: Val Acc={val_acc_ft:.4f}, Test Acc={test_acc_ft:.4f}")
            torch.cuda.empty_cache()
        if results_list:
            print(f"\nResults for {len(results_list)} splits:")
            for i, test_acc in enumerate(results_list, 1):
                print(f"Split {i}: Test Acc={test_acc:.4f}")
            avg_test_acc = float(np.mean(results_list))
            std_test_acc = float(np.std(results_list))
            print(f"Average Test Acc: {avg_test_acc:.4f}, Std Dev: {std_test_acc:.6f}")
            results['node_classification_finetune'] = {
                'val_acc': float(max(results_list)),
                'avg_test_acc': avg_test_acc,
                'std_test_acc': std_test_acc,
                'test_acc_list': results_list
            }
            with open(results_file, 'w' if not os.path.exists(results_file) else 'a') as f:
                f.write(f"=== Node Classification (Fine-tuning) Results ===\n")
                f.write(f"Dataset: {args.dataset}\n")
                f.write(f"Timestamp: {tag}\n")
                f.write(f"Number of Splits: {len(results_list)}\n")
                for i, test_acc in enumerate(results_list, 1):
                    f.write(f"Split {i}: Test Acc={test_acc:.4f}\n")
                f.write(f"Average Test Acc: {avg_test_acc:.4f}\n")
                f.write(f"Std Dev: {std_test_acc:.6f}\n")
                f.write(f"Max Val Acc: {float(max(results_list)):.4f}\n")
                f.write("\n")
        else:
            print("No valid splits processed.")
            results['node_classification_finetune'] = {
                'val_acc': 0.0,
                'avg_test_acc': 0.0,
                'std_test_acc': 0.0,
                'test_acc_list': []
            }
    if 'linear' in args.tasks:
        print("\n=== Node Classification (Linear Evaluation) ===")
        model.eval()
        with torch.no_grad():
            embeddings = model.get_embedding(hyperedge_index, feat, edge_weight, d_list)
            print(f"Embedding dimension: {embeddings.shape[1]}")

        test_acc_list = []
        for split_idx, (train_mask, val_mask, test_mask) in enumerate(splits, 1):
            print(f"\nProcessing Split {split_idx}")
            train_mask, val_mask, test_mask = train_mask.to(device), val_mask.to(device), test_mask.to(device)
            val_acc_linear, test_acc_linear = train_node_classification_linear(
                embeddings,
                data.labels,
                train_mask,
                val_mask,
                test_mask,
                device
            )
            test_acc_list.append(test_acc_linear)
            print(f"Split {split_idx}: Val Acc={val_acc_linear:.4f}, Test Acc={test_acc_linear:.4f}")

        avg_test_acc = float(np.mean(test_acc_list))
        std_test_acc = float(np.std(test_acc_list))
        results['node_classification_linear'] = {
            'avg_test_acc': avg_test_acc,
            'std_test_acc': std_test_acc,
            'test_acc_list': test_acc_list
        }
        print(f"Linear Evaluation Results - Avg Test Acc: {avg_test_acc:.4f}, Std Dev: {std_test_acc:.6f}")
        with open(results_file, 'a') as f:
            f.write(f"=== Node Classification (Linear Evaluation) Results ===\n")
            f.write(f"Dataset: {args.dataset}\n")
            f.write(f"Timestamp: {tag}\n")
            f.write(f"Number of Splits: {len(test_acc_list)}\n")
            for i, test_acc in enumerate(test_acc_list, 1):
                f.write(f"Split {i}: Test Acc={test_acc:.4f}\n")
            f.write(f"Average Test Acc: {avg_test_acc:.4f}\n")
            f.write(f"Std Dev: {std_test_acc:.6f}\n")
            f.write("\n")
    print("=== Model Training and Evaluation Complete ===")
