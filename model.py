import dgl
import torch
import torch.nn as nn
from dgl.nn.pytorch.conv import SAGEConv
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from config import Config

config = Config()


class GCN(nn.Module):

    def __init__(self, embedding_size: int, h_feats: int, dropout: float):
        super().__init__()
        self.gcn_out_dim = h_feats
        self.embedding = nn.Embedding(257, embedding_size)
        self.gcn1 = SAGEConv(embedding_size, h_feats, 'mean', feat_drop=dropout,
                             activation=nn.PReLU(h_feats), norm=nn.BatchNorm1d(h_feats))

    def forward(self, g: dgl.DGLGraph, byte_feat: torch.Tensor):
        byte_feat = byte_feat.long().view(-1)
        h = self.embedding(byte_feat)
        h = self.gcn1(g, h)
        g.ndata['h'] = h
        return dgl.mean_nodes(g, 'h')


class BiLSTMHeaderEncoder(nn.Module):

    def __init__(self, emb=64, hid=64, dropout=0.2):
        super().__init__()
        self.emb = nn.Embedding(257, emb)
        self.byte_aggr = nn.Sequential(nn.Linear(emb, hid), nn.ReLU(), nn.Dropout(dropout))
        self.lstm = nn.LSTM(emb, hid, num_layers=2, bidirectional=True, dropout=dropout, batch_first=True)
        self.dim_adjust = nn.Linear(hid * 2, 128)

    def forward(self, x):
        B, T, L = x.size()
        x = self.emb(x.long().view(B * T, L)).mean(dim=1)
        x = self.byte_aggr(x).view(B, T, -1)
        out, _ = self.lstm(x)
        return self.dim_adjust(out)


class CrossGatedFilter(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.f1 = nn.Sequential(nn.Linear(dim, dim), nn.PReLU(), nn.Linear(dim, dim))
        self.f2 = nn.Sequential(nn.Linear(dim, dim), nn.PReLU(), nn.Linear(dim, dim))

    def forward(self, hx, hy):
        return torch.cat([self.f1(hx).sigmoid() * hy, self.f2(hy).sigmoid() * hx], dim=-1)


class PFA(nn.Module):

    def __init__(self,
                 gpt_layers: int = 6,
                 n_freeze:   int = 4,
                 U:          int = 1):
        super().__init__()
        assert 0 <= n_freeze <= gpt_layers, \
            "`n_freeze` must be between 0 and `gpt_layers`"

        self.U = U
        self.gpt_layers = gpt_layers


        self.gpt2 = GPT2Model.from_pretrained(
            "gpt2",
            output_attentions=True,
            output_hidden_states=True,
        )


        self.gpt2.h = self.gpt2.h[:gpt_layers]


        for layer_index, layer in enumerate(self.gpt2.h):
            for name, param in layer.named_parameters():


                if layer_index < gpt_layers - U:
                    if "ln" in name or "wpe" in name:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False


                else:
                    if "mlp" in name:
                        param.requires_grad = False
                    else:
                        param.requires_grad = True


        self.gpt2.wte.requires_grad_(False)
        self.gpt2.wpe.requires_grad_(False)


    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return self.gpt2(inputs_embeds=x).last_hidden_state


class ST_LLM(nn.Module):
    def __init__(self, in_dim: int = 256, gpt_layers: int = 6, hdf_u: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_dim, 256, kernel_size=1)
        self.proj = nn.Linear(256, 768)
        self.gpt = PFA(gpt_layers=gpt_layers, U=hdf_u)

    def forward(self, x):
        x = self.conv(x.permute(0, 2, 1).unsqueeze(-1)).squeeze(-1).permute(0, 2, 1)
        x = self.proj(x)
        return self.gpt(x)


class MixTemporalGNN(nn.Module):

    def __init__(self, num_classes: int, embedding_size: int = 64, h_feats: int = 128, dropout: float = 0.2, classifier_dropout: float = 0.2, gpt_layers: int = 6, hdf_u: int = 1):
        super().__init__()
        self.header_enc = BiLSTMHeaderEncoder(emb=embedding_size, hid=64, dropout=dropout)
        self.payload_gcn = GCN(embedding_size, h_feats, dropout)

        self.fuse = CrossGatedFilter(dim=h_feats)
        self.st_llm = ST_LLM(in_dim=2 * h_feats, gpt_layers=gpt_layers, hdf_u=hdf_u)

        self.cls = nn.Sequential(
            nn.Linear(768, h_feats), nn.ReLU(), nn.Dropout(classifier_dropout),
            nn.Linear(h_feats, num_classes)
        )

    def forward(self, hdr, g, labels=None):
        B = hdr.size(0)
        h_hdr = self.header_enc(hdr)
        h_pay = self.payload_gcn(g, g.ndata['feat']).view(B, -1, 128)
        h = self.fuse(h_hdr, h_pay)
        h = self.st_llm(h)
        feat = h.mean(dim=1)
        return self.cls(feat)


if __name__ == '__main__':
    cfg = Config()
    model = MixTemporalGNN(num_classes=cfg.NUM_CLASSES)
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total/1e6:.2f}M")
