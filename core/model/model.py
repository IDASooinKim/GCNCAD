from math import sqrt
import torch
import torch.nn.functional as F
from torch import nn
from torchsummary import summary
from .HrNet import HrNet
from einops.layers.torch import Rearrange


class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features, use_bias=True):
        super(GCNLayer, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(torch.zeros(size=(in_features, out_features))))
        if use_bias:
            self.bias = nn.Parameter(torch.FloatTensor(torch.zeros(size=(out_features,))))
        else:
            self.register_parameter('bias', None)

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        x = x @ self.weight
        if self.bias is not None:
            x += self.bias

        return torch.bmm(adj, x)


class GCN(nn.Module):


    def __init__(self, node_features, hidden_dim, num_classes, dropout, use_bias=True):
        super(GCN, self).__init__()
        self.gcn_1 = GCNLayer(node_features, hidden_dim, use_bias)
        self.gcn_2 = GCNLayer(hidden_dim, num_classes, use_bias)
        self.dropout = nn.Dropout(p=dropout)
        self.norm = nn.LayerNorm([4096,hidden_dim])
        self.norm_out = nn.LayerNorm([4096,num_classes])
        self.activation = nn.LeakyReLU()


    def initialize_weights(self):

        self.gcn_1.initialize_weights()
        self.gcn_2.initialize_weights()


    def forward(self, x, adj):

        x = self.activation(self.gcn_1(x, adj))

        x = self.norm(x)
        
        x = self.gcn_2(x, adj)

        x = self.norm_out(x)

        return x


class AttentionModule(nn.Module):


    def __init__(self, input_dim:int=256, qkv_dim=64):

        super(AttentionModule, self).__init__()

        self.query = nn.Linear(input_dim, qkv_dim, bias=True)
        self.adj_score = nn.Sigmoid()

    def forward(self, x):
     
        query = torch.sigmoid(self.query(x))

        dots = torch.matmul(query, query.transpose(-1,-2)) # dot prod
        dots = dots/torch.sqrt(torch.square(dots)) # cos-sim

        adj = self.adj_score(dots)
 
        return query, adj


class GAA(nn.Module):


    def __init__(self, image_size:int=512, patch_size:int=8, num_classes:int=8, in_channels:int=1):

        super(GAA, self).__init__()
        img_height, img_width = (image_size, image_size)
        patch_height, patch_width = (patch_size, patch_size)
        patch_dim = in_channels * patch_height * patch_width
        assert img_height % patch_height == 0 or img_width % patch_width == 0, "Image size must be divisible by the patch size"
        num_patches = (img_height // patch_height) * (img_width // patch_width)

        # TODO: rebuild embedding model e.g. HrNet
        self.conv_embedding = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, stride=1,
                    padding=1, bias=True),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 24, kernel_size=3, stride=1,
                    padding=1, bias=True),
            nn.ReLU(),
            nn.BatchNorm2d(24),
            nn.Conv2d(24, 8, kernel_size=3, stride=1,
                    padding=1, bias=True),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 24, kernel_size=3, stride=1,
                    padding=1, bias=True),
            nn.ReLU(),
            nn.BatchNorm2d(24),
            nn.Conv2d(24, 8, kernel_size=3, stride=1,
                    padding=1, bias=True),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, num_classes, kernel_size=3, stride=1,
                    padding=1, bias=True),
                    )
     
        self.seg_layer_1 = nn.Conv2d(in_channels=num_classes, out_channels=num_classes, kernel_size=3, stride=1, 
                    padding=1, bias=True)
        
        self.patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, patch_dim),
            nn.LayerNorm(patch_dim)
        )
        
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, patch_dim))

        self.att = AttentionModule(input_dim=64, qkv_dim=128)

        self.gcn = GCN(
            node_features=128,
            hidden_dim=256,
            num_classes=num_classes,
            dropout=0.2
        )
        self.hr_emb = HrNet()

    def forward(self, x):
   
        seg_res, x = self.hr_emb(x)

        nodes = torch.sum(x, dim=1, keepdim=True)
        nodes = self.patch_embedding(nodes) 
      
        nodes += self.pos_embedding 
        
        nodes_emb, adj = self.att(nodes)

        out = self.gcn(nodes_emb, adj)

        return out, seg_res


if __name__ == '__main__':

    from torchsummary import summary
    model = GAA()
    summary(model, (1,512,512), 4, 'cpu')
