from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from models import *
from utils import Data
from train import EmbeddingTrainer, ClusteringTrainer

data = Data('cora')
model = VGAE(data)
trainer = EmbeddingTrainer(model, data, 0.01, 200)
output = trainer.run()
embed = output['z'].detach().numpy()
color = data.labels.detach().numpy()
X_reduced = TSNE(n_components=2, random_state=0).fit_transform(embed)
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=color)
plt.colorbar()
plt.show()
