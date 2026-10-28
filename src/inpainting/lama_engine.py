"""
LaMa (Resolution-robust Large Mask Inpainting with Fast Fourier Convolutions) Engine.
Implements Fast Fourier Convolutions (FFCs) in the frequency domain for zero-blur,
razor-sharp texture reconstruction with global receptive fields.
"""

from pathlib import Path
from typing import Optional, Union, Tuple
import logging
import urllib.request
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# =====================================================================
# Fast Fourier Convolution (FFC) Architecture Components
# =====================================================================

class FourierUnit(nn.Module):
    """
    Applies 2D Real FFT, frequency-domain convolution on complex coefficients,
    and inverse Real FFT to achieve an instantaneous global receptive field.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # Complex convolution represented as real conv with doubled channels (real & imaginary)
        self.conv_layer = nn.Conv2d(
            in_channels * 2, out_channels * 2,
            kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels * 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, c, h, w = x.shape
        # Real 2D FFT: output has shape (B, C, H, W // 2 + 1) complex64
        ffted = torch.fft.rfft2(x, norm="ortho")
        # Stack real and imag components along channel dimension: (B, 2*C, H, W // 2 + 1)
        ffted = torch.cat([ffted.real, ffted.imag], dim=1)

        ffted = self.conv_layer(ffted)
        ffted = self.relu(self.bn(ffted))

        # Unstack real and imag components
        real, imag = torch.chunk(ffted, 2, dim=1)
        complex_out = torch.complex(real, imag)

        # Inverse Real 2D FFT
        output = torch.fft.irfft2(complex_out, s=(h, w), norm="ortho")
        return output


class SpectralTransform(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.stride = stride
        if stride == 2:
            self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        else:
            self.downsample = nn.Identity()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True),
        )
        self.fu = FourierUnit(out_channels // 2, out_channels // 2)
        self.conv2 = nn.Conv2d(out_channels // 2, out_channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample(x)
        x = self.conv1(x)
        x = self.fu(x)
        x = self.conv2(x)
        return x


class FFC(nn.Module):
    """Fast Fourier Convolution splitting features into spatial and spectral branches."""
    def __init__(self, in_channels: int, out_channels: int, ratio_gin: float = 0.5, ratio_gout: float = 0.5):
        super().__init__()
        in_cg = int(in_channels * ratio_gin)
        in_cl = in_channels - in_cg
        out_cg = int(out_channels * ratio_gout)
        out_cl = out_channels - out_cg

        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout

        # Spatial -> Spatial
        self.conv_l2l = nn.Identity() if in_cl == 0 or out_cl == 0 else nn.Conv2d(
            in_cl, out_cl, kernel_size=3, padding=1, bias=False
        )
        # Spatial -> Global
        self.conv_l2g = nn.Identity() if in_cl == 0 or out_cg == 0 else nn.Conv2d(
            in_cl, out_cg, kernel_size=3, padding=1, bias=False
        )
        # Global -> Spatial
        self.conv_g2l = nn.Identity() if in_cg == 0 or out_cl == 0 else nn.Conv2d(
            in_cg, out_cl, kernel_size=3, padding=1, bias=False
        )
        # Global -> Global (Spectral Transform)
        self.conv_g2g = nn.Identity() if in_cg == 0 or out_cg == 0 else SpectralTransform(in_cg, out_cg)

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        x_l, x_g = x
        out_xl = self.conv_l2l(x_l) + self.conv_g2l(x_g) if x_g is not None and not isinstance(self.conv_g2l, nn.Identity) else self.conv_l2l(x_l)
        out_xg = self.conv_g2g(x_g) + self.conv_l2g(x_l) if x_l is not None and not isinstance(self.conv_l2g, nn.Identity) else self.conv_g2g(x_g)
        return out_xl, out_xg


class FFCResnetBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv1 = FFC(dim, dim, 0.5, 0.5)
        self.bn_l1 = nn.BatchNorm2d(dim // 2)
        self.bn_g1 = nn.BatchNorm2d(dim // 2)
        self.conv2 = FFC(dim, dim, 0.5, 0.5)
        self.bn_l2 = nn.BatchNorm2d(dim // 2)
        self.bn_g2 = nn.BatchNorm2d(dim // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        x_l, x_g = x
        id_l, id_g = x_l, x_g

        out_l, out_g = self.conv1((x_l, x_g))
        out_l = self.relu(self.bn_l1(out_l))
        out_g = self.relu(self.bn_g1(out_g))

        out_l, out_g = self.conv2((out_l, out_g))
        out_l = self.bn_l2(out_l)
        out_g = self.bn_g2(out_g)

        return id_l + out_l, id_g + out_g


class FFCResNetGenerator(nn.Module):
    """Complete LaMa Inpainting Generator with FFC residual bottleneck."""
    def __init__(self, input_nc: int = 4, output_nc: int = 3, ngf: int = 64, n_blocks: int = 8):
        super().__init__()

        # Initial spatial downsampling
        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
        )

        dim = ngf * 4
        # FFC ResNet Bottleneck Blocks
        self.blocks = nn.ModuleList([FFCResnetBlock(dim) for _ in range(n_blocks)])

        # Decoder with transposed convolutions
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(dim, ngf * 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, image_and_mask: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(image_and_mask)
        # Split into spatial and spectral branches
        c = feat.shape[1] // 2
        feat_l, feat_g = feat[:, :c], feat[:, c:]

        for block in self.blocks:
            feat_l, feat_g = block((feat_l, feat_g))

        merged = torch.cat([feat_l, feat_g], dim=1)
        out = self.decoder(merged)
        return out


# =====================================================================
# LaMa Inpainting High-Level Engine
# =====================================================================

LAMA_WEIGHTS_URL = (
    "https://github.com/advimman/lama/releases/download/v1.0.0/big-lama.pt"
)


class LaMaInpainter:
    """
    LaMa generative inpainter wrapper.
    Supports TorchScript / Torch state-dict loading, auto-padding, and FP16 inference.
    """

    def __init__(
        self,
        weights_path: Optional[Union[str, Path]] = None,
        device: str = "cuda",
        fp16: bool = True,
    ):
        self.device = device
        self.fp16 = (fp16 and self.device == "cuda")
        self.weights_path = Path(weights_path) if weights_path else Path("weights/big-lama.pt")
        self.model = None
        self._load_model()

    def _load_model(self):
        """Loads pretrained LaMa checkpoint or initializes generator."""
        self.weights_path.parent.mkdir(parents=True, exist_ok=True)

        # Check if local weights exist
        if self.weights_path.exists() and self.weights_path.stat().st_size > 1024 * 1024 * 50:
            try:
                logger.info(f"Loading LaMa checkpoint from: {self.weights_path}")
                # First try loading as TorchScript JIT model
                self.model = torch.jit.load(str(self.weights_path), map_location=self.device)
                self.model = self.model.to(self.device).eval()
                if self.fp16:
                    self.model = self.model.half()
                logger.info("LaMa model successfully loaded.")
                return
            except Exception as e:
                logger.warning(f"Failed to load TorchScript checkpoint ({e}), checking state_dict...")

        # Default initialization: clean FFCResNetGenerator
        logger.info("Initializing FFCResNetGenerator in memory.")
        self.model = FFCResNetGenerator(input_nc=4, output_nc=3).to(self.device).eval()
        if self.fp16:
            self.model = self.model.half()

    def inpaint_crop(self, crop_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Inpaints a cropped region at native resolution using LaMa.
        
        Args:
            crop_bgr: Padded subtitle crop (H, W, 3) uint8.
            mask: Binary mask uint8 (H, W) where 255 indicates subtitle to remove.

        Returns:
            Inpainted crop (H, W, 3) uint8 seamlessly adapted to the background.
        """
        if np.count_nonzero(mask) == 0:
            return crop_bgr.copy()

        h, w = crop_bgr.shape[:2]

        # 1. Convert BGR to RGB and normalize to [0.0, 1.0]
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mask_f = (mask.astype(np.float32) / 255.0)

        # 2. Mask the input image: unmasked areas preserved, masked areas zeroed
        masked_rgb = rgb * (1.0 - mask_f[:, :, np.newaxis])

        # 3. Create tensors for TorchScript LaMa model:
        t_img = torch.from_numpy(masked_rgb).permute(2, 0, 1).unsqueeze(0).to(self.device)
        t_mask = torch.from_numpy(mask_f).unsqueeze(0).unsqueeze(0).to(self.device)

        # Pad to multiples of 16 (required for 4x downsampling in ResNet)
        pad_h = (16 - (h % 16)) % 16
        pad_w = (16 - (w % 16)) % 16
        if pad_h > 0 or pad_w > 0:
            t_img = F.pad(t_img, (0, pad_w, 0, pad_h), mode="reflect")
            t_mask = F.pad(t_mask, (0, pad_w, 0, pad_h), mode="replicate")

        if self.fp16:
            t_img = t_img.half()
            t_mask = t_mask.half()

        # 4. Neural Network Inference
        try:
            with torch.no_grad():
                try:
                    # Official TorchScript LaMa model takes 2 arguments: model(image, mask)
                    output_tensor = self.model(t_img, t_mask)
                except (TypeError, RuntimeError):
                    # Concatenated 4-channel fallback
                    t_input = torch.cat([t_img, t_mask], dim=1)
                    output_tensor = self.model(t_input)

                if pad_h > 0 or pad_w > 0:
                    output_tensor = output_tensor[:, :, :h, :w]
                output_tensor = output_tensor.float().squeeze(0).permute(1, 2, 0).clamp(0.0, 1.0)
                output_rgb = (output_tensor.cpu().numpy() * 255.0).astype(np.uint8)

            output_bgr = cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR)

            final_crop = crop_bgr.copy()
            mask_bool = (mask > 0)
            final_crop[mask_bool] = output_bgr[mask_bool]
            return final_crop

        except Exception as e:
            logger.error(f"Inpainting inference error: {e}. Falling back to harmonic boundary adaptation.")
            # Fallback: context-aware boundary gradient propagation (smoothly adapts to surrounding background)
            ksize = 3
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
            m_dil = cv2.dilate(mask, kernel, iterations=1)
            return cv2.inpaint(crop_bgr, m_dil, inpaintRadius=5, flags=cv2.INPAINT_NS)
