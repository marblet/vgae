from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from models import create_vgae_model
from data import load_data
from train import run

data = load_data('karate')
model = create_vgae_model(data)
mu = run(data, model, 0.005, 0, epochs=200)
embed = mu.detach().numpy()
color = data.labels.detach().numpy()

X_reduced = TSNE(n_components=2, random_state=0).fit_transform(embed)
print(X_reduced)
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=color)
plt.show()
