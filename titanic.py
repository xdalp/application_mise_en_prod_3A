"""
Titanic ML project.
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sns
import argparse
from dotenv import load_dotenv
import os
from functions import pipe_func

load_dotenv()  # reads variables from a .env file and sets them in os.environ



#jeton API
jeton_api = os.environ.get("JETON_API", "")

if jeton_api.startswith("$"):
    print("API token has been configured properly")
else:
    print("API token has not been configured")


N_TREES=20
# def n trees
parser = argparse.ArgumentParser(description="Combien d'arbres ?")
parser.add_argument(
    "--n_trees", type=int, default=N_TREES, help="Nombre d'arbres à utiliser"
)
args = parser.parse_args()
print(args.n_trees)

MAX_DEPTH=None
MAX_FEATURES="sqrt"


#
TrainingData = pd.read_csv("data.csv")

TrainingData.head()


TrainingData["Ticket"].str.split("/").str.len()

TrainingData["Name"].str.split(",").str.len()


TrainingData.isnull().sum()


## Un peu d'exploration et de feature engineering

### Statut socioéconomique

fig, axes = plt.subplots(
    1, 2, figsize=(12, 6)
)  # layout matplotlib 1 ligne 2 colonnes taile 16*8
fig1_pclass = sns.countplot(data=TrainingData, x="Pclass", ax=axes[0]).set_title(
    "fréquence des Pclass"
)
fig2_pclass = sns.barplot(
    data=TrainingData, x="Pclass", y="Survived", ax=axes[1]
).set_title("survie des Pclass")


### Age

sns.histplot(data=TrainingData, x="Age", bins=15, kde=False).set_title(
    "Distribution de l'âge"
)
plt.show()

## Encoder les données imputées ou transformées.

pipe=pipe_func(N_TREES)

# splitting samples
y = TrainingData["Survived"]
X = TrainingData.drop("Survived", axis="columns")

# On _split_ notre _dataset_ d'apprentisage pour faire de la validation croisée
#  une partie pour apprendre une partie pour regarder le score.
# Prenons arbitrairement 10% du dataset en test et 90% pour l'apprentissage.
X_TRAIN, X_TEST, y_train, y_test = train_test_split(X, y, test_size=0.1)
pd.concat([X_TRAIN, y_train], axis=1).to_csv("train.csv")
pd.concat([X_TEST, y_test], axis=1).to_csv("test.csv")


# Random Forest


# Ici demandons d'avoir 20 arbres
pipe.fit(X_TRAIN, y_train)

# calculons le score sur le dataset d'apprentissage et sur le dataset de test
# (10% du dataset d'apprentissage mis de côté)
# le score étant le nombre de bonne prédiction
rdmf_score = pipe.score(X_TEST, y_test)
rdmf_score_tr = pipe.score(X_TRAIN, y_train)
print(f"{rdmf_score:.1%} de bonnes réponses sur les données de test pour validation")
print(20 * "-")
print("matrice de confusion")
print(confusion_matrix(y_test, pipe.predict(X_TEST)))
