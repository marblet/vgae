from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from models import *
from utils import *
from train import *

for i in range(1):
    data = LinkPredData('cora')
    model = SDNE(data)
    trainer = LinkPredTrainer(model, data, 0.01, 5e-4, 200)
    output = trainer.run()
embed = output['z'].detach().numpy()
color = data.labels.detach().numpy()
X_reduced = TSNE(n_components=2, random_state=0).fit_transform(embed)
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=color)
plt.colorbar()
plt.show()
