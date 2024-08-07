import torch
import torch.nn as nn
import torch.nn.functional as F

# NAT Components
class Channel_Layernorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)  # Change shape to [B, H, W, C]
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2)  # Change shape back to [B, C, H, W]
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class NeighborhoodAttention(nn.Module):
    def __init__(self, input_size, dim, num_heads, window_size=7, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=qkv_bias)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.pad_idx = nn.ReplicationPad2d(self.window_size // 2)

        # Create relative bias for attention
        self.relative_bias = nn.Parameter(torch.zeros(num_heads, (2 * (self.window_size // 2) + 1), (2 * (self.window_size // 2) + 1)))
        nn.init.trunc_normal_(self.relative_bias, std=.02)
        self.set_input_size(input_size)
        
    def forward(self, x):
        x = self.attention(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def attention(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(x).view(B, 3, self.num_heads, C // self.num_heads, H * W).permute(1, 0, 2, 4, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        
        # Compute attention scores
        attn = torch.einsum('bnhw,bnhw->bnhw', q, k)  # Dot product: B, num_heads, H*W, H*W
        
        # Reshape relative_bias to match attention dimensions
        relative_bias = self.relative_bias.unsqueeze(0).unsqueeze(2)  # Shape: [1, num_heads, 1, window_size^2]
        relative_bias = relative_bias.expand(1, self.num_heads, H * W, H * W)  # Expand to [1, num_heads, H*W, H*W]

        # Add bias to attention scores
        attn = attn + relative_bias
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Apply attention weights to values
        x = torch.einsum('bnhw,bnhw->bnhw', attn, v).view(B, C, H, W)
        return x

    def set_input_size(self, input_size):
        H, W = input_size
        self.H, self.W = H, W
        # No need for attn_idx and bias_idx; adjusted for correct relative_bias

class NATLayer(nn.Module):
    def __init__(self, input_size, dim, num_heads, window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=Channel_Layernorm, layer_scale=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = NeighborhoodAttention(input_size, dim, num_heads, window_size, qkv_bias, qk_scale, attn_drop, drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        
    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def set_input_size(self, input_size):
        self.attn.set_input_size(input_size)

# U-Net + NAT Integration
class UNetWithNAT(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNetWithNAT, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsample = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)

        self.nat_layer = NATLayer(input_size=(64, 64), dim=1024, num_heads=8, window_size=7)

        self.up_conv4 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.up_conv3 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.up_conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.up_conv1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        
    def forward(self, x):
        c1 = F.relu(self.conv1(x))
        p1 = self.pool(c1)
        
        c2 = F.relu(self.conv2(p1))
        p2 = self.pool(c2)
        
        c3 = F.relu(self.conv3(p2))
        p3 = self.pool(c3)
        
        c4 = F.relu(self.conv4(p3))
        p4 = self.pool(c4)
        
        c5 = F.relu(self.conv5(p4))
        
        # Apply NAT layer after the deepest U-Net layer
        nat_output = self.nat_layer(c5)
        
        up6 = self.upsample(nat_output)
        merge6 = torch.cat([up6, c4], dim=1)
        c6 = F.relu(self.up_conv4(merge6))
        
        up7 = F.interpolate(c6, scale_factor=2, mode='bilinear', align_corners=True)
        merge7 = torch.cat([up7, c3], dim=1)
        c7 = F.relu(self.up_conv3(merge7))
        
        up8 = F.interpolate(c7, scale_factor=2, mode='bilinear', align_corners=True)
        merge8 = torch.cat([up8, c2], dim=1)
        c8 = F.relu(self.up_conv2(merge8))
        
        up9 = F.interpolate(c8, scale_factor=2, mode='bilinear', align_corners=True)
        merge9 = torch.cat([up9, c1], dim=1)
        c9 = F.relu(self.up_conv1(merge9))
        
        output = torch.sigmoid(self.final_conv(c9))
        return output

# Define DropPath as a placeholder
class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob > 0.:
            keep_prob = 1 - self.drop_prob
            mask = torch.empty(x.size(0), 1, 1, 1, device=x.device).bernoulli_(keep_prob)
            mask = mask / keep_prob
            x = x * mask
        return x
