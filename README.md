# Particle Rolling MCMC with Double Block Sampling

## Overview
Ce dépôt contient l'implémentation Python de l'algorithme "Particle Rolling MCMC with Double Block Sampling", conçu pour une estimation efficace des paramètres et une modélisation de l'état dans les séries temporelles économiques. L'algorithme est présenté dans le contexte des modèles espace-état et aborde des défis tels que les changements structurels et les paramètres non constants sur de longues périodes d'échantillonnage.

## Description de l'algorithme
L'algorithme fonctionne en deux étapes principales, permettant l'ajout de nouvelles observations et la suppression d'anciennes, tout en traitant les problèmes de dégénérescence des poids par le biais d'une approche de fenêtre roulante.

- Étape 1 : Échantillonnage en amont**
  - Ajout d'une nouvelle observation \( y_t \) et mise à jour des particules et des poids pour refléter cet ajout.
  - Implémente le lissage de la simulation des particules pour améliorer les propriétés de mélange.

- Étape 2 : Échantillonnage à rebours**
  - Supprime l'observation la plus ancienne \N( y_{s-1} \N) et met à jour les particules et les poids en conséquence.
  - Améliore l'homogénéité des particules grâce à des techniques de lissage.

## Caractéristiques principales
- Traite la dégénérescence des poids en variant la taille des fenêtres de roulement (K = 5, 10, 15 \N)).
- Gère efficacement les changements structurels dans les séries temporelles économiques.
- Incorpore des justifications théoriques démontrant la densité marginale des paramètres et des états.
