# Proposition de Projet : GLO-7030

**Titre** : Exploration d'architectures neuronales profondes pour l'évaluation du risque ordinal.

Nous utiliserons le jeu de données Prudential Life Insurance Assessment (Kaggle) pour prédire une mesure de risque à 8 niveaux ordonnés. Ce dataset est représentatif des problèmes de sélection des risques en assurance-vie. Le dataset présente un défi classique de données tabulaires avec 128 variables. Notre objectif est d'évaluer les avancées récentes du Deep Learning sur ce type de données, traditionnellement dominé par les modèles de boosting.

Notre projet visera comparer des techniques d'apprentissage automatique classiques à certaines récentes avancées en apprentissage profond sur les données tabulaires.

**Modèles de référence** : Utilisation de modèles de Gradient Boosting (ex: XGBoost, LightGBM, CatBoost) pour établir un seuil de performance.

**Architectures Deep Learning** : Exploration de réseaux de neurones profonds adaptés aux données tabulaires tels que des modèles de fondation pour données tabulaires (TabPFN-2.5), des approches d'ensemble (TabM) ou des Transformer pour l'entraînement semi-supervisé (TabDPT).

**Gestion des variables** : Une attention particulière sera portée à l'encodage des variables catégorielles à haute cardinalité et ainsi que la présence de variables fortement corrélées.

Pour les approches d'apprentissage profond, nous envisageons tester plusieurs variantes pour observer les impacts sur la performance:

- Exploration de l'usage des données non-étiquetées du jeu de test pour enrichir l'apprentissage (ex: pré-entraînement par auto-encodage ou pseudo-étiquetage).
- Test de différentes techniques (Dropout, Batch Normalization, ou Weight Decay) pour contrôler le surapprentissage sur les données tabulaires.
- Une analyse sera menée pour isoler l'impact des composants clés de l'architecture choisie (ex: avec ou sans embeddings, avec ou sans pré-entraînement).

Nous pourrions explorer une stratégie d'augmentation tabulaire consistant à générer un grand volume de données synthétiques sans étiquettes à partir des données existantes. Un modèle préalablement entraîné sur les données réelles (approche de type teacher–student) servirait à produire des pseudo-étiquettes sur ces nouvelles observations. Le modèle de Deep Learning serait entraîné, possiblement avec une phase de pré-entraînement suivie d'un fine-tuning sur les données originales.

Nous tenterons de surpasser les modèles de boosting classiques en exploitant des techniques d'apprentissage profond dans un domaine (l'assurance-vie) où les modèles de boosting sont encore l'état de l'art. Si ce n'est pas concluant, nous testerons si l'utilisation des représentations issues du DL dans les approches de boosting améliorent significativement les performances des approches de boosting.

## Échéancier (Points de suivi les mardis)

- **Mardi 31 mars** : Nettoyage des données, lecture des articles et mise en place des différents éléments du projet.
- **Mardi 7 avril** : Pipelines de données fonctionnels et premiers résultats comparatifs sur W&B.
- **Mardi 14 avril** : Fin des tests sur le volet semi-supervisé et les données synthétiques.
- **Mardi 21 avril** : Revue technique intermédiaire et ajustement des hyperparamètres.
- **Mardi 28 avril** : Arrêt définitif des expérimentations et production des visuels finaux.
- **Mardi 28 avril au 3 mai** : Phase intensive de rédaction, revue par les pairs et dépôt final.
