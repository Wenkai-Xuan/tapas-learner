import torch
import torch.nn as nn


class TapasFormer(nn.Module):
    def __init__(self, obs_dim=40,
                 hidden_dim=128,
                 num_heads=1,
                 feed_fwd_dim=128,
                 dropout_p=0.1,
                 num_encoder_layers=1,
                 pooling='mean',
                 device='cpu',
                 normalization_params=None):
        super().__init__()
        if normalization_params is None:
            normalization_params = {"target_min": 0.0,
                                    "target_max": 1100,
                                    "new_min": -1,
                                    "new_max": 1}
        self.normalization_params = normalization_params
        self.hidden_dim = hidden_dim
        self.obs_dim = obs_dim

        self.tokenizing_net = nn.Sequential(
            nn.Linear(self.obs_dim, self.hidden_dim).type(torch.float32),
            nn.LayerNorm(self.hidden_dim).type(torch.float32),
        )

        self.regression_net = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim).type(torch.float32),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim).type(torch.float32),
            nn.ReLU(),
            # nn.Linear(self.hidden_dim, self.hidden_dim).type(torch.float32),
            # nn.LayerNorm(self.hidden_dim).type(torch.float32),
            nn.Linear(self.hidden_dim, 1).type(torch.float32),
            nn.Tanh()
        )

        # transformer only encodes target features
        self.pooling = pooling
        transformer_enc_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=num_heads,
            dim_feedforward=feed_fwd_dim,
            dropout=dropout_p,
            activation="relu",
            device=device,
            batch_first=True,
        )

        self.transformer_enc = nn.TransformerEncoder(
            encoder_layer=transformer_enc_layer,
            num_layers=num_encoder_layers,
        )

    def forward(self, observation, src_key_padding_mask):
        # print("observation.shape", observation.shape)
        # print("src_key_padding_mask.shape", src_key_padding_mask.shape)
        encoding = self.tokenizing_net(observation)
        # print("encoding.shape", encoding.shape)
        encoded_features = self.transformer_enc(src=encoding, src_key_padding_mask=src_key_padding_mask)
        # if self.pooling == 'mean':
        #     encoded_features = torch.mean(encoded_features, dim=1)
        # elif self.pooling == 'max':
        #     encoded_features = torch.max(encoded_features, dim=1)[0]
        # else:
        #     raise NotImplementedError
        # print("encoded_features.shape", encoded_features.shape)
        hidden_state = encoded_features[:, 0, :] # aka pooled_output
        # print("hidden_state.shape", hidden_state.shape)                
        regressed_targets = self.regression_net(hidden_state)        
        return regressed_targets


def normalize_targets(targets, normalization_params):
    target_min = normalization_params["target_min"]
    target_max = normalization_params["target_max"]
    new_min = normalization_params["new_min"]
    new_max = normalization_params["new_max"]
    return (targets - target_min) / (target_max - target_min) * (new_max - new_min) + new_min


def denormalize_predicted_targets(predicted_targets, normalization_params):
    target_min = normalization_params["target_min"]
    target_max = normalization_params["target_max"]
    new_min = normalization_params["new_min"]
    new_max = normalization_params["new_max"]
    return (predicted_targets - new_min) / (new_max - new_min) * (target_max - target_min) + target_min


def eval_batch(targets, predicted_targets, normalization_params):
    normalized_targets_ = normalize_targets(targets, normalization_params)
    normalized_error = torch.abs(normalized_targets_ - predicted_targets)
    # print("normalized_error.shape", normalized_error.shape)
    normalized_batch_error = normalized_error.sum()

    scaled_output = denormalize_predicted_targets(predicted_targets, normalization_params)
    scaled_error = torch.abs(targets - scaled_output)
    # print("scaled_error.shape", scaled_error.shape)
    scaled_batch_error = scaled_error.sum()

    return torch.tensor([normalized_batch_error, scaled_batch_error, len(targets)])
