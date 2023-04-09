
# Projet RODM

Ce répertoire contient nos travaux sur le projet RODM 2023. Nous avons décidé de tester deux nouvelles approches de regroupement prenant en compte des données de différentes classes. Les 2 méthodes ainsi que les résultats sont présentés dans le rapport ci-joint.

## Structure du répertoire

### Code Julia
Le code permettant de tester les différents algorithmes se trouvent dans le répertoire src/. Plus précisemment, les 2 fichiers contentant nos travaux sont :
- src/main_test_functions.jl : fichier similaire à main_merge, mais permettant d'utiliser les deux méthodes développées. Pour la première méthode, utiliser la méthode ConvexHullMerge. Pour la seconde méthode, Kmeans_Based_Merge (il faut "comment" les autres lignes et "decomment" la méthode voulue)
- src/merge.jl : les différentes fonctions que nous avons développées se trouvent dans le fichier merge. Il s'agit des fonctions "compute_dataframe_classes", "detect_points_small_classes","trimming_cluster","update_clusters", "ConvexHullMerge" et "Kmeans_Based_Merge".

### Resultats
Les fichiers csv contenant nos résultats sont trouvables dans le répertoire Results_data/.