"""Native contrastive projector."""

from torch import nn
import torch.nn.init as init


class ProjectionHead(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        num_layers=2,
        activation='relu',
        dropout=0.1,
        debug=False,
        name="ProjectionHead",
    ):
        super(ProjectionHead, self).__init__()
        self.debug = debug
        self.name = name
        activation_str = activation.lower()
        if activation_str == 'relu':
            # Avoid in-place ReLU to prevent autograd versioning issues
            act = nn.ReLU()
        elif activation_str == 'gelu':
            act = nn.GELU()
        else:
            raise ValueError("Unsupported activation type. Choose 'relu' or 'gelu'.")

        layers = []
        for i in range(num_layers):
            is_last = i == num_layers - 1

            # Create and tag linear layer
            lin = nn.Linear(input_dim if i == 0 else hidden_dim, output_dim if is_last else hidden_dim)
            tag = f"{name}_linear_{i}"
            if is_last:  # logits-producing layer
                tag += "_last"  # marks it for small-σ init
            lin = tag_module(lin, tag)
            layers.append(lin)

            # Only add activation and dropout for non-final layers
            if not is_last:
                layers.append(act)
                layers.append(nn.Dropout(dropout))

        self.layers = nn.Sequential(*layers)

        # Add LayerNorm for contrastive stability
        self.output_norm = nn.LayerNorm(output_dim, elementwise_affine=True)

        self.apply(initialize_weights)  # Use the global initializer only

    def forward(self, x):
        """
        Forward pass through the ProjectionHead.

        Parameters:
        x (Tensor): Input tensor of shape [batch_size, d_model] or [batch_size, seq_len, d_model]

        Returns:
        Tensor: Output tensor after passing through the projection head layers.
        """
        if self.debug:
            print(f"[{self.name}] Input Shape: {x.shape}")
        x = self.layers(x)  # Linear layers will apply to the last dimension
        if self.debug:
            print(f"[{self.name}] Output Shape: {x.shape}")
            self.debug = False
        x = self.output_norm(x)
        return x


def tag_module(module, tag):
    module._layer_tag = tag
    return module


def get_module_tag(module):
    return getattr(module, '_layer_tag', '')


def initialize_weights(module):
    """
    Apply optimized weight initialization to each module in the PyTorch model.
    This function is tailored for SSL contrastive learning architectures.
    """
    if isinstance(module, nn.Conv1d):
        print(f"Initializing Conv1d: {module}")
        nn.init.kaiming_normal_(module.weight)
        if module.bias is not None:
            init.constant_(module.bias, 0)

    elif isinstance(module, nn.Linear):
        tag = get_module_tag(module)

        # Use smaller std for logits-producing layers
        if tag.endswith('_last') or 'mlp.3' in tag:
            # These are the logits-producing layers → small σ
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        else:
            # For other linear layers (e.g., in transformer blocks)
            init.xavier_uniform_(module.weight)
            if module.bias is not None:
                init.constant_(module.bias, 0)

    elif isinstance(module, (nn.BatchNorm1d, nn.SyncBatchNorm)):  # Handle both BN types
        init.constant_(module.weight, 1)
        init.constant_(module.bias, 0)

    elif isinstance(module, nn.LayerNorm):
        # Initialize LayerNorm weights and biases
        init.constant_(module.weight, 1)
        init.constant_(module.bias, 0)

    elif isinstance(module, (nn.TransformerEncoderLayer, nn.TransformerDecoderLayer)):
        # Initialize feedforward and attention submodules
        for sub_module in module.modules():
            if isinstance(sub_module, nn.Linear):
                init.xavier_uniform_(sub_module.weight)
                if sub_module.bias is not None:
                    init.zeros_(sub_module.bias)
            elif isinstance(sub_module, nn.MultiheadAttention):
                init.xavier_uniform_(sub_module.in_proj_weight)
                init.xavier_uniform_(sub_module.out_proj.weight)
                if sub_module.in_proj_bias is not None:
                    init.zeros_(sub_module.in_proj_bias)
                if sub_module.out_proj.bias is not None:
                    init.zeros_(sub_module.out_proj.bias)
