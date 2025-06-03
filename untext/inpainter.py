"""Deep Image Prior implementation for inpainting."""

from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm

  # LamaInpainter expects image: np.ndarray (H, W, 3, uint8), mask: np.ndarray (H, W, uint8, 255=hole)
  # Returns: np.ndarray (H, W, 3, uint8), same shape as input image.

class SkipConnection(nn.Module):
    """Skip connection block for the U-Net architecture."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)))


class DeepImagePriorInpainter:
    """Inpainter using Deep Image Prior approach."""

    def __init__(
        self,
        num_channels: int = 3,
        num_filters: int = 64,
        num_layers: int = 5,
        learning_rate: float = 0.01,
        num_iterations: int = 2000,
        device: Optional[str] = None,
    ):
        """Initialize the inpainter.
        
        Args:
            num_channels: Number of input channels (default: 3 for RGB)
            num_filters: Base number of filters in the network
            num_layers: Number of down/up sampling layers
            learning_rate: Learning rate for optimization
            num_iterations: Number of optimization iterations
            device: Device to run on ('cuda' or 'cpu')
        """
        self.num_channels = num_channels
        self.num_filters = num_filters
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize network
        self.net = self._build_network()
        self.net.to(self.device)
        
    def _build_network(self) -> nn.Module:
        """Build the U-Net architecture."""
        class UNet(nn.Module):
            def __init__(self):
                super().__init__()
                # Encoder
                self.enc_blocks = nn.ModuleList()
                in_ch = self.num_channels
                for _ in range(self.num_layers):
                    self.enc_blocks.append(SkipConnection(in_ch, self.num_filters))
                    in_ch = self.num_filters
                    self.num_filters *= 2
                
                # Decoder
                self.dec_blocks = nn.ModuleList()
                for _ in range(self.num_layers):
                    self.num_filters //= 2
                    self.dec_blocks.append(SkipConnection(self.num_filters * 2, self.num_filters))
                
                # Final layer
                self.final = nn.Conv2d(self.num_filters, self.num_channels, 1)
            
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # Encoder
                enc_outputs = []
                for enc in self.enc_blocks:
                    x = enc(x)
                    enc_outputs.append(x)
                    x = F.max_pool2d(x, 2)
                
                # Decoder
                for i, dec in enumerate(self.dec_blocks):
                    x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
                    x = torch.cat([x, enc_outputs[-(i+1)]], dim=1)
                    x = dec(x)
                
                return self.final(x)
        
        return UNet()

    def inpaint(
        self, image: np.ndarray, mask: np.ndarray
    ) -> Tuple[np.ndarray, list]:
        """Inpaint the masked regions of the image.
        
        Args:
            image: Input image (H, W, C)
            mask: Binary mask (H, W) where 1 indicates pixels to inpaint
            
        Returns:
            Tuple containing:
            - Inpainted image
            - List of loss values during optimization
        """
        # Convert to torch tensors
        image = torch.from_numpy(image).permute(2, 0, 1).float().to(self.device)
        mask = torch.from_numpy(mask).float().to(self.device)
        
        # Normalize image
        image = image / 255.0
        
        # Create random input
        z = torch.randn(1, self.num_channels, *image.shape[1:]).to(self.device)
        
        # Setup optimizer
        optimizer = Adam(self.net.parameters(), lr=self.learning_rate)
        
        # Optimization loop
        losses = []
        pbar = tqdm(range(self.num_iterations), desc="Inpainting")
        
        for _ in pbar:
            optimizer.zero_grad()
            
            # Forward pass
            output = self.net(z)
            
            # Compute loss only on masked regions
            loss = F.mse_loss(output * mask, image * mask)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            pbar.set_postfix({"loss": f"{loss.item():.6f}"})
        
        # Get final result
        with torch.no_grad():
            result = self.net(z)
        
        # Convert back to numpy
        result = (result * 255).clamp(0, 255).cpu().numpy().astype(np.uint8)
        result = result.transpose(1, 2, 0)
        
        return result, losses 