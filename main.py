from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from models import create_vgae_model
from models import create_vgaecd_model
from data import load_data
from train import run
import torch

torch.autograd.set_detect_anomaly(True)
data = load_data('cora')
model = create_vgaecd_model(data, latent_dim=8)
z = run(data, model, 0.001, 0, epochs=200)
embed = z.detach().numpy()
color = data.labels.detach().numpy()
X_reduced = TSNE(n_components=2, random_state=0).fit_transform(embed)
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=color)
plt.colorbar()
plt.show()
