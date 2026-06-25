# Models.py
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

from datasets import *
from utils import * 

device = torch.device("mps")

class SamePadConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, groups=1):
        super().__init__()
        self.receptive_field = (kernel_size - 1) * dilation + 1
        padding = self.receptive_field // 2
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding,
            dilation=dilation,
            groups=groups
        )
        self.remove = 1 if self.receptive_field % 2 == 0 else 0
        
    def forward(self, x):
        out = self.conv(x)
        if self.remove > 0:
            out = out[:, :, : -self.remove]
        return out
    
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, final=False):
        super().__init__()
        self.conv1 = SamePadConv(in_channels, out_channels, kernel_size, dilation=dilation)
        self.conv2 = SamePadConv(out_channels, out_channels, kernel_size, dilation=dilation)
        self.projector = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels or final else None
    
    def forward(self, x):
        residual = x if self.projector is None else self.projector(x)
        x = F.gelu(x)
        x = self.conv1(x)
        x = F.gelu(x)
        x = self.conv2(x)
        return x + residual

class DilatedConvEncoder(nn.Module):
    def __init__(self, in_channels, channels, kernel_size):
        super().__init__()
        self.net = nn.Sequential(*[
            ConvBlock(
                channels[i-1] if i > 0 else in_channels,
                channels[i],
                kernel_size=kernel_size,
                dilation=2**i,
                final=(i == len(channels)-1)
            )
            for i in range(len(channels))
        ])
        
    def forward(self, x):
        return self.net(x)

class TSEncoder(nn.Module):
    def __init__(self, input_dims, seq_len, latent_dims=128, hidden_dims=64, depth=10):
        super().__init__()
        self.seq_len = seq_len
        self.input_dims = input_dims
        self.latent_dims = latent_dims
        self.hidden_dims = hidden_dims
        self.input_fc = nn.Linear(input_dims, hidden_dims)
        self.feature_extractor = DilatedConvEncoder(
            hidden_dims,
            [hidden_dims] * depth + [latent_dims],
            kernel_size=3
        )
        self.repr_dropout = nn.Dropout(p=0.1)
        self.fc_mu = nn.Sequential(
                nn.Linear(latent_dims * seq_len, 2 * latent_dims),
                nn.LayerNorm(2*latent_dims),
                nn.GELU(),
                nn.Linear(2 * latent_dims, 2 * latent_dims)
        ) 
    def forward(self, x, mask=None):  # x: B x T x input_dims
        x = self.input_fc(x)  # B x T x Ch
        x = x.transpose(1, 2)  # B x Ch x T
        x = self.repr_dropout(self.feature_extractor(x))  # B x Co x T
        x = self.fc_mu(x.flatten(start_dim=1))  # B x Co
        mu, logvar = x[:, :self.latent_dims], x[:, self.latent_dims:]
        return mu, logvar
        

class DilatedConvDecoder(nn.Module):
    """
    Este módulo intenta invertir el bloque convolucional dilatado.
    Se basa en la misma estructura que en el encoder, pero se invierte el orden de los bloques y se ajustan las dilataciones.
    """
    def __init__(self, out_channels, channels, kernel_size):
        super().__init__()
        reversed_channels = list(reversed(channels))
        self.net = nn.Sequential(*[
            ConvBlock(
                reversed_channels[i-1] if i > 0 else out_channels,
                reversed_channels[i],
                kernel_size=kernel_size,
                dilation=2**(len(reversed_channels)-1 - i),
                final=(i == len(reversed_channels)-1)
            )
            for i in range(len(reversed_channels))
        ])
        
    def forward(self, x):
        return self.net(x)


class TSDecoder(nn.Module):
    def __init__(self, latent_dims, seq_len, input_dims, hidden_dims=64, depth=10):
        super().__init__()
        self.seq_len = seq_len
        channels = [hidden_dims] * depth + [latent_dims]
        self.latent_dims = latent_dims
        self.fc = nn.Sequential(
            nn.Linear(latent_dims, 2 * latent_dims),
            nn.GELU(),
            nn.Linear(2 * latent_dims, seq_len * latent_dims)
            )

        self.feature_restorer = DilatedConvDecoder(latent_dims, channels, kernel_size=3)
        self.output_fc = nn.Linear(hidden_dims, input_dims)
        
    def forward(self, x, y):
        # x: (B, T, output_dims)
        x = self.fc(x).view(-1, self.seq_len, self.latent_dims)
        x = x.transpose(1, 2)  # B x output_dims x T
        x = self.feature_restorer(x)  # B x hidden_dims x T
        x = x.transpose(1, 2)  # B x T x hidden_dims
        x = self.output_fc(x)   # B x T x input_dims
        return x

class Network(nn.Module):
    def __init__(self, input_dims, seq_len, num_prototypes_class, transformations, n_transforms, hidden_dims=64, latent_dims=320, depth=10):
        super().__init__()
        self.seq_len = seq_len
        self.latent_dims = latent_dims
        self.transformations = transformations
        self.encoder = TSEncoder(input_dims = input_dims, seq_len = seq_len, latent_dims=latent_dims, hidden_dims=hidden_dims, depth=depth)
        self.explainer = TSDecoder(input_dims = input_dims, seq_len = seq_len, latent_dims=latent_dims, hidden_dims=hidden_dims, depth=depth)
        self.recon = TSDecoder(latent_dims=latent_dims, seq_len = seq_len, input_dims=input_dims, hidden_dims=hidden_dims, depth=depth)
        self.n_transforms = n_transforms
        self.num_prototypes_per_class = num_prototypes_class
        self.num_prototypes = num_prototypes_class * self.n_transforms
        self.prototype_shape = (self.num_prototypes, latent_dims)
        self.num_classes = self.n_transforms
        self.epsilon = 1e-4

        self.prototype_class_identity = torch.zeros(self.num_prototypes,
                                                    self.num_classes).to(device)

        for j in range(self.num_prototypes):
            self.prototype_class_identity[j, j // self.num_prototypes_per_class] = 1

        
        self.prototype_vectors = nn.Parameter(torch.randn(self.prototype_shape),
                                              requires_grad=True)

        self.classifier = nn.Linear(self.num_prototypes, n_transforms, bias=False)

        self.set_last_layer_incorrect_connection()



    def distance_2_similarity(self, distances):
        return torch.log((distances + 1)/(distances + self.epsilon))

    def calc_sim_scores(self, z):
        d = torch.cdist(z, self.prototype_vectors, p=2)**2
        return d

    def reparameterize(self, mu, logVar):
        # Reparameterization takes in the input mu and logVar and sample the mu + std * eps
        std = torch.exp(logVar / 2)
        eps = torch.randn_like(std)
        return mu + std * eps
    
    def set_last_layer_incorrect_connection(self, incorrect_strength=-0.5):
        positive_one_weights_locations = torch.t(self.prototype_class_identity)
        negative_one_weights_locations = 1 - positive_one_weights_locations
    
        correct_class_connection = 1
        incorrect_class_connection = incorrect_strength
        self.classifier.weight.data.copy_(
            correct_class_connection * positive_one_weights_locations
            + incorrect_class_connection * negative_one_weights_locations
        )

    def kl_divergence_mixture(self, mu, logvar, distances, y, tau=1.0):
        """
        KL mixture adapted for distances (conditional by class).
        distances: (B, P) squared L2 distances
        y: (B,) class labels
        """

        masked_distances = distances.clone()

        for b in range(distances.size(0)):
            k = int(y[b].item())
            valid = self.prototype_class_identity[:, k].bool()
            masked_distances[b, ~valid] = float('inf')
            
        weights = F.softmax(-masked_distances / tau, dim=1)  # (B, P)

        std = torch.exp(0.5 * logvar)
        q = torch.distributions.Normal(mu, std)

        total_kl = 0.0

        for p in range(self.num_prototypes):
            proto = self.prototype_vectors[p]
            p_dist = torch.distributions.Normal(proto, torch.ones_like(proto))

            kl = torch.distributions.kl.kl_divergence(q, p_dist).mean(dim=1)  # (B)
            total_kl += weights[:, p] * kl

        return total_kl.mean()

    def forward(self, x, y, identity_only: bool = False):
        x, y = x.to(device), y.to(device)

        if x.dim() == 2:
            x = x.unsqueeze(0)

        if self.training: 
            augmented = self.transformations.generate(x, n_repeats=5).float()
        else:
            augmented = self.transformations.generate(x).float()

        if identity_only:
            # We only consider the identity during inference
            augmented = augmented[:1]
            y = torch.zeros(1, dtype=torch.long, device=augmented.device)
            
        # Reparameterization for VAE backbone
        mu, logvar = self.encoder(augmented)
        logvar = logvar.clamp(-8, 8)

        if self.training:
            repre = self.reparameterize(mu, logvar)
        else:
            repre = mu

        # Compute distances with respect to the latent prototypes
        distances = self.calc_sim_scores(repre)

        # Predictions of the classifier based on the similarities with respect to the latent prototypes
        classif = self.classifier(-distances)

        # Reconstruction Module forward and KL mixutre-based regularization
        recon = self.recon(repre, y)
        explain = self.explainer(repre, y)
        kl_mix = self.kl_divergence_mixture(mu, logvar, distances, y)
        
        # Prototype Module Losses
        proto_labels = self.prototype_class_identity.argmax(dim=1)
        same_mask = y.unsqueeze(1) == proto_labels.unsqueeze(0)
        masked_same = torch.where(same_mask, distances, torch.full_like(distances, float('inf')))
        min_same = masked_same.min(dim=1)[0]
        clst_loss = (min_same.mean())

        masked_by_proto = torch.where(same_mask, distances, torch.full_like(distances, float('inf')))
        min_proto_to_data = masked_by_proto.min(dim=0)[0]
        valid_proto_mask = torch.isfinite(min_proto_to_data)
        proto_to_data_term = (min_proto_to_data[valid_proto_mask].mean())

        return augmented, repre, explain, recon, classif, clst_loss, proto_to_data_term, kl_mix



class ProtoXAD(nn.Module):
    def __init__(self, seq_len, latent_dims, input_dims, train_data, val_data, test_data, n_transforms, transformations, num_prototypes_per_class):
        super().__init__()
        self._net = Network(input_dims, latent_dims = latent_dims, seq_len = seq_len, n_transforms = n_transforms, 
                            transformations = transformations, num_prototypes_class = num_prototypes_per_class)
        self.latent_dims = latent_dims
        self.seq_len = seq_len
        self.n_transforms = self._net.n_transforms
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.train_loader = DataLoader(train_data, batch_size=4, shuffle=True)
        self.val_loader = DataLoader(val_data, batch_size=1, shuffle=False)
        self.test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
        self.kmeans_init_prototypes()

    def forward(self, batch):
        batch = batch.to(device)
        out = self._net(batch)
        return out
        
    def training_step(self, batch, y, training=True, identity_only: bool = False):
        batch, y = batch.to(device), y.to(device)

        augmented, repre, explain, recon, classif, clst_loss, proto_loss, kl_mix = self._net(batch, y, identity_only=identity_only)
        class_loss = F.cross_entropy(classif, y)

        if training:
            if self._net.training:
                recon_loss = F.l1_loss(recon, batch.repeat(self.n_transforms, 1, 1).repeat(5, 1, 1))
            else: 
                recon_loss = F.l1_loss(recon, batch.repeat(self.n_transforms, 1, 1))
                
            explain_loss = F.l1_loss(explain, augmented, reduction='mean')
            
            return explain_loss + class_loss + clst_loss + proto_loss + recon_loss + kl_mix
            
        return class_loss
        
    def train_model(self, max_epochs=1000, verbose=False):
        self.train()
        optimizer = torch.optim.Adam(self._net.parameters(), lr=1e-3)
        train_loader = self.train_loader
        val_loader = self.val_loader
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, min_lr=1e-7)

        interrupted = False
        n_epoch_iters = 0
        
        for n_epoch_iters in range(max_epochs):
            cum_loss = 0
            num_batches = 0
            for idx, batch in enumerate(train_loader):
                optimizer.zero_grad()
                loss = self.training_step(batch.to(device), y = torch.repeat_interleave(torch.arange(self.n_transforms), 
                                                                                        batch.shape[0]).repeat(5), training = True)
                loss.backward()

                optimizer.step()
                cum_loss += loss.item()
                num_batches += 1
                
            self.eval()
            val_loss = 0
            with torch.no_grad():
                val_loss = np.mean(self.compute_scores(val_loader))
            self.train()
            scheduler.step(val_loss)
            if verbose and n_epoch_iters%5==0:
                print(f"Epoch #{n_epoch_iters}: loss={cum_loss}, val_loss={val_loss}")

            if scheduler.optimizer.param_groups[0]['lr'] <= 1e-5:
                print(f"Epoch #{n_epoch_iters} Early Stopping")
                break  

    def compute_scores(self, val_loader=None, identity_only_test: bool = True):
        if val_loader is not None:
            data_loader = self.val_loader
            training = True
            identity_only = False   # We maintain the original setting for validation
        else:
            data_loader = self.test_loader
            training = False
            identity_only = identity_only_test

        scores = []
        with torch.no_grad():
            for idx, sample in enumerate(data_loader):
                if identity_only:
                    # Identity: class 0
                    y = torch.zeros(1, dtype=torch.long, device=device)
                else:
                    y = torch.arange(self.n_transforms, device=device)

                score = self.training_step(
                    sample.to(device),
                    y=y,
                    training=training,
                    identity_only=identity_only
                )
                scores.append(score.item())
        return scores
        
    @torch.no_grad()
    def infer_prototypes_input(self, class_labels=None):
        net = self._net
        self.eval()
    
        device_ = next(net.parameters()).device
        protos = net.prototype_vectors
    
        outs = []
    
        for i in range(len(protos)):
            class_id = i // net.num_prototypes_per_class
            y = torch.tensor([class_id], device=device_, dtype=torch.long)
            z_vec = protos[i].unsqueeze(0)
    
            out = net.explainer(z_vec, y)
            outs.append(out.detach().cpu())
    
        recon = torch.cat(outs, dim=0)  # (P, T, F)
    
        cls_per_proto = (
            np.arange(len(protos)) // net.num_prototypes_per_class
        )
    
        if class_labels is None:
            labels = [f"class {int(c)}" for c in cls_per_proto]
        else:
            labels = [
                class_labels[int(c)] if int(c) < len(class_labels) else f"class {int(c)}"
                for c in cls_per_proto
            ]
    
        out = {
            "recon": recon,
            "labels": labels,
            "class_ids": cls_per_proto.tolist(),
            "proto_ids": list(range(len(net.prototype_vectors))),
        }
    
        return out


    def plot_prototypes_input(
        self,
        proto_ids=None,
        feature: int = 0,
        transform_names=None,
        ncols: int = None,
        sharey: bool = True,
        figsize_per_cell=(3.0, 1.7),
        ylim=None,
        show_zero_line: bool = True,
    ):
        net = self._net
        net.eval()
        self.eval()
    
        info = self.infer_prototypes_input(class_labels=transform_names)
        recon = info["recon"].numpy()
        all_proto_ids = np.asarray(info["proto_ids"])
    
        P, T, F = recon.shape
        K = net.n_transforms
        Ppc = net.num_prototypes_per_class
    
        if feature >= F:
            raise ValueError(f"feature={feature} is out of range. The reconstructed prototypes have {F} feature(s).")
    
        if transform_names is None:
            transform_names = [f"Class {i}" for i in range(K)]
        else:
            transform_names = list(transform_names)
            if len(transform_names) < K:
                transform_names += [f"Class {i}" for i in range(len(transform_names), K)]
    
        if proto_ids is not None:
            proto_ids = np.asarray(proto_ids)
            valid_mask = np.isin(all_proto_ids, proto_ids)
            recon_to_plot = recon[valid_mask]
            ids_to_plot = all_proto_ids[valid_mask]
            class_ids_to_plot = ids_to_plot // Ppc
            proto_pos_to_plot = ids_to_plot % Ppc
        else:
            recon_to_plot = recon
            ids_to_plot = all_proto_ids
            class_ids_to_plot = ids_to_plot // Ppc
            proto_pos_to_plot = ids_to_plot % Ppc
    
        if len(ids_to_plot) == 0:
            raise ValueError("No prototypes selected for plotting.")
    
        if ncols is None:
            ncols = Ppc
    
        # If plotting all prototypes, keep the structured K x Ppc grid.
        # If plotting a subset, use a compact grid.
        plotting_all = proto_ids is None
    
        if plotting_all:
            nrows = K
            ncols = Ppc
            fig_width = figsize_per_cell[0] * ncols
            fig_height = figsize_per_cell[1] * nrows
    
            fig, axes = plt.subplots(
                nrows,
                ncols,
                figsize=(fig_width, fig_height),
                sharex=True,
                sharey=sharey,
                squeeze=False,
            )
    
            colors = plt.cm.tab20(np.linspace(0, 1, max(K, 1)))
            t = np.arange(T)
    
            if ylim is None:
                y_values = recon[:, :, feature]
                y_min, y_max = np.nanmin(y_values), np.nanmax(y_values)
                margin = 0.1 * max(1e-6, y_max - y_min)
                ylim = (y_min - margin, y_max + margin)
    
            for c in range(K):
                for j in range(Ppc):
                    ax = axes[c, j]
                    pid = c * Ppc + j
    
                    if pid >= P:
                        ax.axis("off")
                        continue
    
                    y = recon[pid, :, feature]
    
                    ax.plot(t, y, linewidth=1.4, color=colors[c])
                    if show_zero_line:
                        ax.axhline(0, linewidth=0.8, alpha=0.35)
    
                    ax.grid(True, alpha=0.25)
                    ax.set_ylim(*ylim)
                    ax.tick_params(labelsize=8)
    
                    if c == 0:
                        ax.set_title(f"Prototype {j + 1}", fontsize=10)
    
                    if j == 0:
                        ax.set_ylabel(transform_names[c], fontsize=9)
    
        else:
            nrows = int(np.ceil(len(ids_to_plot) / ncols))
            fig_width = figsize_per_cell[0] * ncols
            fig_height = figsize_per_cell[1] * nrows
    
            fig, axes = plt.subplots(
                nrows,
                ncols,
                figsize=(fig_width, fig_height),
                sharex=True,
                sharey=sharey,
                squeeze=False,
            )
    
            colors = plt.cm.tab20(np.linspace(0, 1, max(K, 1)))
            t = np.arange(T)
    
            if ylim is None:
                y_values = recon_to_plot[:, :, feature]
                y_min, y_max = np.nanmin(y_values), np.nanmax(y_values)
                margin = 0.1 * max(1e-6, y_max - y_min)
                ylim = (y_min - margin, y_max + margin)
    
            for ax_idx, ax in enumerate(axes.ravel()):
                if ax_idx >= len(ids_to_plot):
                    ax.axis("off")
                    continue
    
                pid = ids_to_plot[ax_idx]
                c = class_ids_to_plot[ax_idx]
                j = proto_pos_to_plot[ax_idx]
                y = recon_to_plot[ax_idx, :, feature]
    
                ax.plot(t, y, linewidth=1.4, color=colors[c])
                if show_zero_line:
                    ax.axhline(0, linewidth=0.8, alpha=0.35)
    
                ax.grid(True, alpha=0.25)
                ax.set_ylim(*ylim)
                ax.tick_params(labelsize=8)
                ax.set_title(
                    f"Prototype {int(pid)} | {transform_names[int(c)]}",
                    fontsize=9,
                )
    
        fig.suptitle(
            "Input-space prototypes learned by ProtoX-AD",
            fontsize=14,
            y=1.005,
        )
        fig.supxlabel("Time index", fontsize=11)
        fig.supylabel("Amplitude", fontsize=11)
    
        fig.tight_layout()
        return fig, recon
    
    @torch.no_grad()
    def kmeans_init_prototypes(self, batch_size=64, max_batches=None):
        from sklearn.cluster import KMeans
    
        device = next(self._net.parameters()).device
        net = self._net
        net.eval()
    
        # N_transforms = classes
        K = net.n_transforms
        Ppc = net.num_prototypes_per_class
        latent_list = [[] for _ in range(K)]
    
        loader = self.train_loader
        for i, (batch) in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
    
            batch = batch.to(device)
            # Generate augmented views
            aug = net.transformations.generate(batch)
    
            # Labels: [0,1,2...K-1] repeated B times
            y = torch.repeat_interleave(
                torch.arange(K, device=device), batch.size(0)
            )
    
            repre, _ = net.encoder(aug)
    
            for k in range(K):
                mask = (y == k)
                if mask.any():
                    latent_list[k].append(repre[mask].cpu())
    
        latent_list = [
            torch.cat(latent_list[k], dim=0).numpy() if len(latent_list[k]) > 0 else None
            for k in range(K)
        ]

        # K-Means over the classes
        proto_mat = []
    
        for k in range(K):
            X = latent_list[k]
            kmeans = KMeans(
                n_clusters=Ppc,
                n_init=10,
                max_iter=500,
                random_state=0
            )
            kmeans.fit(X)
            centroids = kmeans.cluster_centers_

            proto_mat.append(torch.tensor(centroids, dtype=torch.float32))
    
        proto_mat = torch.cat(proto_mat, dim=0).to(device)
    
        # Copy information to prototype_vectors
        net.prototype_vectors.data.copy_(proto_mat)
    
