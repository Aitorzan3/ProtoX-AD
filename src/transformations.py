from models import *
from utils import *

class UMDTransformations():
    """
    9 views per series (fixed order):
      0) identity
      1) bump+ at the beginning  (normal base)
      2) bump+ at the end        (normal base)
      3) bump- at the beginning  (normal base)
      4) bump- at the end        (normal base)
      5) bump+ at the beginning  (base inverted outside the bump)
      6) bump+ at the end        (base inverted outside the bump)
      7) bump- at the beginning  (base inverted outside the bump)
      8) bump- at the end        (base inverted outside the bump)
    
    The bump shape is Gaussian and is preserved (same width/sigma).
    A random shift towards the center is added at each forward pass.
    """

    def __init__(
        self,
        bump_width: int = 6,
        bump_amplitude: float = 1.,
        bump_random_shift: int = 5,  # small jitter close to the edge
        bump_random_scale: float = 0, # relative variation in width/amplitude
        width_random_scale: float=0,
        inward_shift_range: tuple = (15, 20),  # shift towards the center (in timesteps) [min, max]
        seed: int = None,
        mask_power: float = 1.0,        # >1 hardens the mask, <1 softens it
    ):
        self.bump_width = bump_width
        self.bump_amplitude = bump_amplitude
        self.bump_random_shift = bump_random_shift
        self.bump_random_scale = bump_random_scale
        self.width_random_scale = width_random_scale
        self.inward_shift_range = inward_shift_range
        self.mask_power = mask_power

    def _sample_params(self, length, start=True):
        width = int(self.bump_width * np.random.uniform(1 - self.width_random_scale,
                                                        1 + self.width_random_scale))
        width = max(3, width)
        sigma = max(1, width // 3)

        amplitude = float(np.random.uniform(0.75, 1.25))
        edge_jitter = np.random.randint(0, self.bump_random_shift + 1) if self.bump_random_shift > 0 else 0
        max_possible = max(0, length - width - 1)
        lo, hi = self.inward_shift_range
        hi = min(int(hi), max_possible)
        lo = max(0, int(lo))
        inward = 0 if hi <= 0 else int(np.random.randint(lo, hi + 1))
    
        if start:
            center = width // 2 + edge_jitter
            center = min(center + inward, length - width // 2 - 1)
        else:
            center = length - width // 2 - 1 - edge_jitter
            center = max(center - inward, width // 2)
    
        return width, sigma, amplitude, center


    def _make_bump_and_mask(self, length, sign=1, start=True, device=None):
        width, sigma, amplitude, center = self._sample_params(length, start=start)
        xs = torch.arange(length, device=device, dtype=torch.float32)
        # Gaussian mask (without amplitude), raised to harden it if desired
        mask = torch.exp(-0.5 * ((xs - center) / sigma) ** 2)
        if self.mask_power != 1.0:
            mask = torch.clamp(mask, 0.0, 1.0) ** float(self.mask_power)
        bump_signal = sign * amplitude * mask
        return bump_signal, mask

    def _apply_bump(self, x: torch.Tensor, sign=1, start=True, invert_base: bool = False) -> torch.Tensor:
        B, T, F = x.shape
        device = x.device
        out = torch.empty_like(x)
        for i in range(B):
            bump_signal, bump_mask = self._make_bump_and_mask(T, sign=sign, start=start, device=device)  # (T,)
            bump_sig_expanded = bump_signal[:, None].expand(T, F)  # (T,F)
            if invert_base:
                # normal base inside the bump (mask≈1), inverted outside (1-mask)
                base_inside  = x[i] * bump_mask[:, None]
                base_outside = (-x[i]) * (1.0 - bump_mask)[:, None]
                out[i] = base_inside + base_outside + bump_sig_expanded
            else:
                # intact base + superimposed bump (as in the original code)
                out[i] = x[i] + bump_sig_expanded
        return out

    def _make_bump(self, length, sign=1, start=True, device=None):
        width, sigma, amplitude, center = self._sample_params(length, start=start)
        xs = torch.arange(length, device=device, dtype=torch.float32)
        bump = sign * amplitude * torch.exp(-0.5 * ((xs - center) / sigma) ** 2)
        return bump

    def generate(self, x: torch.Tensor, n_repeats: int = 1) -> torch.Tensor:
        all_views = []
        for _ in range(n_repeats):
            views = [
                x,  # identidad
                self._apply_bump(x, sign=+1, start=True,  invert_base=False),
                self._apply_bump(x, sign=+1, start=False, invert_base=False),
                self._apply_bump(x, sign=-1, start=True,  invert_base=False),
                self._apply_bump(x, sign=-1, start=False, invert_base=False),
                self._apply_bump(x, sign=+1, start=True,  invert_base=True),
                self._apply_bump(x, sign=+1, start=False, invert_base=True),
                self._apply_bump(x, sign=-1, start=True,  invert_base=True),
                self._apply_bump(x, sign=-1, start=False, invert_base=True),
            ]
            all_views.append(torch.cat(views, dim=0))
        return torch.cat(all_views, dim=0)



class TransformationsYorkshire:
    """
    Multiplicative transformations over non-overlapping bands.
    
    Scales:
        - 1.0 (identity)
        - Uniform(1.1, 1.7)
        - Uniform(1.8, 2.4)
        - Uniform(2.5, 3.1)
    
    Each call to generate() samples new scales.
    """
    def __init__(self):

        self.scale_ranges = [
                    (1.0, 1.0),   # identidad
                    (1.3, 1.70),
                    (2.0, 2.3),
                    (2.6, 2.9),
                ]


    def _sample_scales(self):
        scales = []
        for lo, hi in self.scale_ranges:
            if lo == hi:
                scales.append(lo)
            else:
                scales.append(random.uniform(lo, hi))
        return scales

    def generate(self, x: torch.Tensor, n_repeats=1) -> torch.Tensor:
        """
        Input:
            x: (B, T, F)
        Output:
            (B * (4 * n_repeats), T, F)
        """
        x = x.float()
        views = []

        for _ in range(n_repeats):
            scales = self._sample_scales()
            for s in scales:
                views.append(x * s)

        return torch.cat(views, dim=0)


class TransformationsTemperature:
    """
    Generates augmented views based on disjoint annual thermal regimes.
    
    Classes:
        0  -> normal
       -1  -> cold-light
       +1  -> warm-light
       -2  -> cold-heavy
       +2  -> warm-heavy
    
    Each transformation shifts the series so that its annual mean
    falls within a target interval (target_mean).
    """

    def __init__(
        self,
        n_repeats=None,
        eps=0.05,
        warm_upper=1.225,
        cold_lower=-1.225
    ):
        self.n_repeats = n_repeats
        self.eps = eps
        self.warm_upper = warm_upper
        self.cold_lower = cold_lower

        # Target intervals by class
        self.target_intervals = {
            0: None,  # identity
            1:  (0.25 + eps, 0.75 - eps),          # warm-light
           -1: (-0.75 + eps, -0.25 - eps),         # cold-light
            2:  (0.75 + eps, warm_upper),           # warm-heavy
           -2: (cold_lower, -0.75 - eps)            # cold-heavy
        }

        # Fixed class order per cycle
        self.classes = [0, -1, 1, -2, 2]

    def _sample_target_mean(self, cls):
        if cls == 0:
            return None
        lo, hi = self.target_intervals[cls]
        return random.uniform(lo, hi)

    def _apply_transformation(self, x: torch.Tensor, cls: int) -> torch.Tensor:
        if cls == 0:
            return x

        # Annual mean per sample (B, 1, 1)
        mean = x.mean(dim=1, keepdim=True)

        target_mean = self._sample_target_mean(cls)

        # delta depends on each sample
        delta = target_mean - mean

        return x + delta

    def generate(self, x: torch.Tensor, n_repeats=None):
        x = x.float()

        if n_repeats is None:
            n_repeats = self.n_repeats if self.n_repeats is not None else 1

        views = []

        B = x.shape[0]

        for _ in range(n_repeats):
            for cls in self.classes:
                x_t = self._apply_transformation(x, cls)
                views.append(x_t)

        return torch.cat(views, dim=0)

class res_trans1d_block(torch.nn.Module):

    def __init__(self, channel,bias=False):
        super(res_trans1d_block, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv1d(channel, channel, 3, 1, 1, bias=bias)
        self.in1 = nn.InstanceNorm1d(channel, affine=bias)
        self.conv2 = nn.Conv1d(channel, channel, 3, 1, 1, bias=bias)
        self.in2 = nn.InstanceNorm1d(channel, affine=bias)

    def forward(self, x):
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        #        out = self.pool(out)
        out = self.in2(self.conv2(out))
        out = out + residual
        out = self.relu(out)
        return out


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation=1,bias = False):
        super(ConvLayer, self).__init__()
        #        padding = kernel_size // 2
        padding = dilation * (kernel_size // 2)
        self.reflection_pad = nn.ReflectionPad1d(padding)
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, stride, bias=bias)  # , padding)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv1d(out)
        return out


class SeqTransformNet(nn.Module):
    def __init__(self, x_dim,hdim,num_layers):
        super(SeqTransformNet, self).__init__()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.conv1 = ConvLayer(x_dim, hdim, 3, 1,bias=False)
        #        self.conv1 = nn.Conv1d(args.x_dim,2*args.x_dim,3,1,0,dilation=2**i)
        self.in1 = nn.InstanceNorm1d(hdim, affine=False)
        res_blocks = []
        for _ in range(num_layers-2):
            res_blocks.append(res_trans1d_block(hdim,False))
        self.res = nn.Sequential(*res_blocks)
        #        self.conv2 = nn.ConvTranspose1d(args.x_dim,2*args.x_dim,3,1,0,dilation=2**i)
        self.conv2 = ConvLayer(hdim, x_dim, 3, 1,bias=False)

    def forward(self, x):
        out = self.relu(self.in1(self.conv1(x)))
        for block in self.res:
            out = block(out)
        out = self.conv2(out)
        return out

class Transformations(nn.Module):
    def __init__(self, x_dim, n_transforms=4):
        super().__init__()
        self.transform_list = nn.ModuleList(
            [SeqTransformNet(x_dim, x_dim, 5) for _ in range(1, n_transforms)]
        )

    def generate(self, x, identity_only=False):
        # x: (B, T, F)
        if identity_only:
            return x

        x = x.permute(0, 2, 1)   # (B, F, T)

        augmented_views = [x]
        for transform in self.transform_list:
            augmented_views.append(transform(x))

        augmented_views = torch.cat(augmented_views, dim=0)
        augmented_views = augmented_views.permute(0, 2, 1)  # (B*n_views, T, F)
        return augmented_views




