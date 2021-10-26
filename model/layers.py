import torch
from torch import nn
from torch.nn import init
from torch.autograd import Variable
import torch.nn.functional as F

# def batched_index_select(target: torch.Tensor,
#                          indices: torch.LongTensor,
#                          flattened_indices: Optional[torch.LongTensor] = None) -> torch.Tensor:
#
#     if flattened_indices is None:
#         # Shape: (batch_size * d_1 * ... * d_n)
#         flattened_indices = flatten_and_batch_shift_indices(indices, target.size(1))
#
#     # Shape: (batch_size * sequence_length, embedding_size)
#     flattened_target = target.view(-1, target.size(-1))
#
#     # Shape: (batch_size * d_1 * ... * d_n, embedding_size)
#     flattened_selected = flattened_target.index_select(0, flattened_indices)
#     selected_shape = list(indices.size()) + [target.size(-1)]
#     # Shape: (batch_size, d_1, ..., d_n, embedding_size)
#     selected_targets = flattened_selected.view(*selected_shape)
#     return selected_targets

class Classifier(nn.Module):
    def __init__(self, hidden_dim, num_class):
        super(Classifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_class = num_class
        self.linear = nn.Linear(self.hidden_dim, self.num_class)

        self.linear.bias.data.fill_(0)
        init.xavier_uniform(self.linear.weight, gain=1) # initialize linear layer

    def forward(self, inputs):
        logits = self.linear(inputs)
        return logits

class Discriminator(nn.Module):
    """Discriminator model for source domain."""

    def __init__(self, input_dims, hidden_dims, output_dims):
        """Init discriminator."""
        super(Discriminator, self).__init__()

        self.layer = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(input_dims, hidden_dims),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dims, hidden_dims),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dims, output_dims),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        """Forward the discriminator."""
        out = self.layer(inputs)
        return out

class SimpleAttLayer(nn.Module):

    def __init__(self, in_features,  attention_size, n_class):
        super(SimpleAttLayer, self).__init__()
        self.in_features = in_features
        self.attention_size = attention_size
        self.input_linear = nn.Linear(in_features,attention_size,bias=True)
        self.att = nn.Linear(attention_size,1,bias=False)
        self.classifier = nn.Linear(in_features,n_class,bias=False)


    def forward(self, inputs):
        #inputs N, adj_num, att_size
        v = torch.tanh(self.input_linear(inputs))
        vu = self.att(v).transpose(-1,-2)
        alphas = F.softmax(vu,dim=2)
        output = torch.bmm(alphas,inputs).squeeze()
        logits = self.classifier(output)
        return logits


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h = torch.mm(input, self.W)
        N = h.size()[0]

        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""

    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)


class SpGraphAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)

        self.a = nn.Parameter(torch.zeros(size=(1, 2 * out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()

    def forward(self, input, adj):
        dv = 'cuda' if input.is_cuda else 'cpu'

        N = input.size()[0]
        edge = adj.nonzero().t()

        h = torch.mm(input, self.W)
        # h: N x out

        assert not torch.isnan(h).any()

        # Self-attention on the nodes - Shared attention mechanism
        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
        # edge: 2*D x E

        edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze()))
        assert not torch.isnan(edge_e).any()
        # edge_e: E

        e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N, 1), device=dv))
        # e_rowsum: N x 1

        edge_e = self.dropout(edge_e)
        # edge_e: E

        h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
        assert not torch.isnan(h_prime).any()
        # h_prime: N x out

        h_prime = h_prime.div(e_rowsum)
        # h_prime: N x out
        assert not torch.isnan(h_prime).any()

        if self.concat:
            # if this layer is not last layer,
            return F.elu(h_prime)
        else:
            # if this layer is last layer,
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
