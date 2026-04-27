# CTGAN

On utilise ici un modèle **CTGAN (Conditional Tabular GAN)** pour générer des données synthétiques à partir du dataset tabulaire.

CTGAN repose sur une architecture **GAN (Generative Adversarial Network)** composée de deux réseaux neuronaux entraînés conjointement :

- un **générateur**, qui produit de nouvelles lignes de données synthétiques ;
- un **discriminateur**, qui tente de distinguer les données réelles des données générées.

Le générateur apprend progressivement à produire des observations de plus en plus réalistes afin de tromper le discriminateur. Ce mécanisme adversarial permet d'approcher la distribution statistique du dataset d'origine.

CTGAN intègre également plusieurs mécanismes spécifiques aux données tabulaires :

- **conditionnement sur les variables discrètes**, permettant de contrôler certaines caractéristiques des données générées ;
- **traitement spécifique des variables continues et catégorielles** ;
- **échantillonnage conditionnel**, facilitant la génération de données pour certaines catégories particulières.

## Entraînement

Dans ce projet, on entraîne **un modèle CTGAN par classe de la variable cible `Response`**.  
Chaque modèle apprend donc la distribution des observations correspondant à une seule classe.

Le modèle est ainsi entraîné à générer des données synthétiques ressemblant aux observations réelles de cette classe.

## Génération de données

Une fois les modèles entraînés, on génère de nouvelles observations synthétiques selon deux stratégies :

### Stratégie A — génération ciblée des classes rares

On génère des données synthétiques uniquement pour les **classes sous-représentées** afin de réduire le déséquilibre du dataset sans modifier excessivement la distribution globale.

### Stratégie B — équilibrage complet des classes

On génère des données synthétiques pour toutes les classes minoritaires jusqu'à atteindre le **même nombre d'observations que la classe majoritaire**, ce qui permet d'obtenir un dataset entièrement équilibré.