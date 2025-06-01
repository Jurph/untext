"""Deep Image Prior implementation for image inpainting.

This module implements the Deep Image Prior (DIP) approach for image inpainting.
Unlike traditional deep learning methods, DIP doesn't require pre-trained weights.
Instead, it leverages the structure of a randomly initialized neural network as a
prior for natural images. The network is optimized directly on the input image,
learning to reconstruct it while being constrained by the known (unmasked) pixels.

Reference:
    Ulyanov, D., Vedaldi, A., & Lempitsky, V. (2018). Deep image prior.
    In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as tv
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Tuple, Optional, Union, Callable
import logging

logger = logging.getLogger(__name__)

class Hourglass(nn.Module):
    """Hourglass network architecture for Deep Image Prior."""
    
    def __init__(self) -> None:
        """Initialize the Hourglass network."""
        super(Hourglass, self).__init__()

        self.leaky_relu = nn.LeakyReLU()

        # Downsampling path
        self.d_conv_1 = nn.Conv2d(2, 8, 5, stride=2, padding=2)
        self.d_bn_1 = nn.BatchNorm2d(8)

        self.d_conv_2 = nn.Conv2d(8, 16, 5, stride=2, padding=2)
        self.d_bn_2 = nn.BatchNorm2d(16)

        self.d_conv_3 = nn.Conv2d(16, 32, 5, stride=2, padding=2)
        self.d_bn_3 = nn.BatchNorm2d(32)
        self.s_conv_3 = nn.Conv2d(32, 4, 5, stride=1, padding=2)

        self.d_conv_4 = nn.Conv2d(32, 64, 5, stride=2, padding=2)
        self.d_bn_4 = nn.BatchNorm2d(64)
        self.s_conv_4 = nn.Conv2d(64, 4, 5, stride=1, padding=2)

        self.d_conv_5 = nn.Conv2d(64, 128, 5, stride=2, padding=2)
        self.d_bn_5 = nn.BatchNorm2d(128)
        self.s_conv_5 = nn.Conv2d(128, 4, 5, stride=1, padding=2)

        self.d_conv_6 = nn.Conv2d(128, 256, 5, stride=2, padding=2)
        self.d_bn_6 = nn.BatchNorm2d(256)

        # Upsampling path
        self.u_deconv_5 = nn.ConvTranspose2d(256, 124, 4, stride=2, padding=1)
        self.u_bn_5 = nn.BatchNorm2d(124)

        self.u_deconv_4 = nn.ConvTranspose2d(128, 60, 4, stride=2, padding=1)
        self.u_bn_4 = nn.BatchNorm2d(60)

        self.u_deconv_3 = nn.ConvTranspose2d(64, 28, 4, stride=2, padding=1)
        self.u_bn_3 = nn.BatchNorm2d(28)

        self.u_deconv_2 = nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1)
        self.u_bn_2 = nn.BatchNorm2d(16)

        self.u_deconv_1 = nn.ConvTranspose2d(16, 8, 4, stride=2, padding=1)
        self.u_bn_1 = nn.BatchNorm2d(8)

        self.out_deconv = nn.ConvTranspose2d(8, 3, 4, stride=2, padding=1)
        self.out_bn = nn.BatchNorm2d(3)

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.
        
        Args:
            noise: Input noise tensor of shape (batch_size, 2, height, width)
            
        Returns:
            Reconstructed image tensor of shape (batch_size, 3, height, width)
        """
        # Downsampling path
        down_1 = self.leaky_relu(self.d_bn_1(self.d_conv_1(noise)))
        down_2 = self.leaky_relu(self.d_bn_2(self.d_conv_2(down_1)))
        down_3 = self.leaky_relu(self.d_bn_3(self.d_conv_3(down_2)))
        skip_3 = self.s_conv_3(down_3)
        
        down_4 = self.leaky_relu(self.d_bn_4(self.d_conv_4(down_3)))
        skip_4 = self.s_conv_4(down_4)
        
        down_5 = self.leaky_relu(self.d_bn_5(self.d_conv_5(down_4)))
        skip_5 = self.s_conv_5(down_5)
        
        down_6 = self.leaky_relu(self.d_bn_6(self.d_conv_6(down_5)))

        # Upsampling path with skip connections
        up_5 = self.leaky_relu(self.u_bn_5(self.u_deconv_5(down_6)))
        up_5 = torch.cat([up_5, skip_5], 1)
        
        up_4 = self.leaky_relu(self.u_bn_4(self.u_deconv_4(up_5)))
        up_4 = torch.cat([up_4, skip_4], 1)
        
        up_3 = self.leaky_relu(self.u_bn_3(self.u_deconv_3(up_4)))
        up_3 = torch.cat([up_3, skip_3], 1)
        
        up_2 = self.leaky_relu(self.u_bn_2(self.u_deconv_2(up_3)))
        up_1 = self.leaky_relu(self.u_bn_1(self.u_deconv_1(up_2)))
        
        out = self.out_bn(self.out_deconv(up_1))
        return torch.sigmoid(out)


class DeepImagePrior:
    """Deep Image Prior implementation for image inpainting.
    
    This class implements the Deep Image Prior approach, which uses a randomly
    initialized neural network as a prior for natural images. The network is
    optimized during the inpainting process, learning to reconstruct the image
    while being constrained by the known (unmasked) pixels. No pre-trained weights
    are required.
    """
    
    def __init__(
        self,
        device: str = 'cuda',
        learning_rate: float = 1e-2,
        num_iterations: int = 500,
        noise_type: str = 'gaussian',
        known_region_weight: float = 0.01,
        lr_decay: bool = True
    ) -> None:
        """Initialize the Deep Image Prior model.
        
        Args:
            device: Device to run the model on ('cuda' or 'cpu')
            learning_rate: Learning rate for the optimizer
            num_iterations: Number of optimization iterations. More iterations
                           generally lead to better results but take longer.
            noise_type: Kind of input noise ('gaussian' | 'mesh').
            known_region_weight: Weight for the reconstruction loss on *known* pixels.
            lr_decay: Whether to apply cosine-annealing learning-rate decay.
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.noise_type = noise_type.lower()
        self.known_region_weight = known_region_weight
        self.lr_decay = lr_decay
        
        # Initialize a new network with random weights
        self.model = Hourglass().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        if self.lr_decay:
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.num_iterations, eta_min=learning_rate * 0.1
            )
        else:
            self.scheduler = None
        self.criterion = nn.MSELoss()
        
        logger.info("Initialized Deep Image Prior on %s", self.device)

    def inpaint(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        progress_callback: Optional[Callable[[int,int,float], None]] = None
    ) -> np.ndarray:
        """Inpaint masked regions in an image.
        
        Args:
            image: Input image as numpy array (H, W, C)
            mask: Binary mask as numpy array (H, W) where 1 indicates pixels to inpaint
            progress_callback: Optional callback function to report progress
            
        Returns:
            Inpainted image as numpy array (H, W, C)
        """
        # Ensure mask is binary 0/1
        if mask.max() > 1:
            mask = (mask > 0).astype(np.float32)
        else:
            mask = mask.astype(np.float32)

        # Convert inputs to tensors
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float()
        
        # Move to device
        image_tensor = image_tensor.to(self.device)
        mask_tensor = mask_tensor.to(self.device)
        
        # --------- Padding so H and W are multiples of 64 ---------
        h, w = image.shape[:2]
        pad_h = (64 - h % 64) % 64
        pad_w = (64 - w % 64) % 64
        if pad_h or pad_w:
            pad = (0, pad_w, 0, pad_h)  # pad right, pad bottom
            image_tensor = nn.functional.pad(image_tensor, pad, mode='reflect')
            mask_tensor = nn.functional.pad(mask_tensor, pad, mode='reflect')
            h += pad_h
            w += pad_w

        # Create input noise
        if self.noise_type == 'gaussian':
            z = torch.randn(1, 2, h, w, device=self.device)
        else:
            mesh_y, mesh_x = np.mgrid[:h, :w].astype(np.float32)
            mesh = np.stack([mesh_y / h, mesh_x / w], 0)
            z = torch.from_numpy(mesh).unsqueeze(0).to(self.device)
        
        # Optimization loop
        losses = []
        for i in range(self.num_iterations):
            self.optimizer.zero_grad()
            
            # Forward pass
            output = self.model(z)
            
            # Compute loss primarily on the *known* (unmasked) pixels so the network
            # learns to reproduce the context and only then hallucinate the masked area.
            loss_known = self.criterion(output * (1 - mask_tensor), image_tensor * (1 - mask_tensor))

            if self.known_region_weight > 0:
                # Optionally add a very small consistency term on the masked region
                loss_unknown = self.criterion(output * mask_tensor, image_tensor * mask_tensor)
                loss = loss_known + self.known_region_weight * loss_unknown
            else:
                loss = loss_known
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Record loss
            losses.append(loss.item())
            
            # Report progress
            if progress_callback and (i + 1) % 10 == 0:
                progress_callback(i + 1, self.num_iterations, loss.item())
            
            if self.scheduler is not None:
                self.scheduler.step()
        
        # Combine original and inpainted regions
        with torch.no_grad():
            # Remove padding if any
            if pad_h or pad_w:
                output = output[:, :, :image.shape[0], :image.shape[1]]
            result = image_tensor[:, :, :image.shape[0], :image.shape[1]] * (1 - mask_tensor[:, :, :image.shape[0], :image.shape[1]]) + \
                     output[:, :, :image.shape[0], :image.shape[1]] * mask_tensor[:, :, :image.shape[0], :image.shape[1]]
            result = result[0].cpu().permute(1, 2, 0).clamp(0, 1).numpy()
            result = (result * 255).astype(np.uint8)
        
        return result 