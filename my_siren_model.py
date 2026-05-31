import torch
import torch.nn as nn
import math

# ------------------ SIREN LAYERS ------------------
class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, is_first=False, omega0=30.0):
        super().__init__()
        self.in_features = in_features
        self.is_first = is_first
        self.omega0 = omega0
        self.linear = nn.Linear(in_features, out_features)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                bound = math.sqrt(6 / self.in_features) / self.omega0
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.omega0 * self.linear(x))


# ------------------ MODEL ------------------
class SirenDeepSDF2D(nn.Module):
    def __init__(self, latent_dim=16, hidden_dim=128, omega0_first=30.0, omega0_hidden=30.0):
        super().__init__()
        input_dim = latent_dim + 2

        self.l1 = SineLayer(input_dim, hidden_dim, is_first=True, omega0=omega0_first)
        self.l2 = SineLayer(hidden_dim, hidden_dim, omega0=omega0_hidden)
        self.l3 = SineLayer(hidden_dim, hidden_dim, omega0=omega0_hidden)
        self.l4 = SineLayer(hidden_dim, hidden_dim, omega0=omega0_hidden)
        self.l5 = SineLayer(hidden_dim, hidden_dim, omega0=omega0_hidden)

        self.fc_out = nn.Linear(hidden_dim, 1)

        with torch.no_grad():
            bound = math.sqrt(6 / hidden_dim) * 0.5
            self.fc_out.weight.uniform_(-bound, bound)

    def forward(self, z, coords):
        x = torch.cat([z, coords], dim=1)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        return self.fc_out(x)


# ------------------ LOAD FUNCTION ------------------
def load_trained_model(checkpoint_path, num_wings, latent_dim=16, device="cuda"):
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # recreate model
    model = SirenDeepSDF2D(
        latent_dim=latent_dim,
        hidden_dim=128,
        omega0_first=30.0,
        omega0_hidden=30.0
    ).to(device)

    # recreate latent codes
    latent_codes = nn.Embedding(num_wings, latent_dim).to(device)

    # load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(ckpt["model"])
    latent_codes.load_state_dict(ckpt["latent"])

    model.eval()

    return model, latent_codes, device

# CODE TO LOAD
#from siren_model import load_trained_model
#model, latent_codes, device = load_trained_model(checkpoint_path='model_files/model.pth",num_wings=200, latent_dim=16)
#D = np.load("distance_mat/D_l2_noprocustus.npy")