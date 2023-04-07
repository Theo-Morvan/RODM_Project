include("struct/distance.jl")
include("utilities.jl")
include("new_functions.jl")
using DataFrames, Random
using DataStructures
using StatsBase
using Statistics
using LinearAlgebra
using LazySets
using Polyhedra
using Clustering
using Debugger

"""
Essaie de regrouper des données en commençant par celles qui sont les plus proches.
Deux clusters de données peuvent être fusionnés en un cluster C 
s'il n'existe aucune données x_i pour aucune caractéristique j 
qui intersecte l'intervalle représenté par les bornes minimale et maximale 
de C pour j (x_i,j n'appartient pas à [min_{x_k dans C} x_k,j ; max_{k dans C} x_k,j]).

Entrées :
- x : caractéristiques des données d'entraînement
- y : classe des données d'entraînement
- percentage : le nombre de clusters obtenu sera égal à n * percentage
 
Sorties :
- un tableau de Cluster constituant une partition de x
"""
function exactMerge(x, y)

    n = length(y)
    m = length(x[1,:])
    
    # Liste de tous les clusters obtenus
    # (les données non regroupées seront seules dans un cluster)
    clusters = Vector{Cluster}([])

    # Initialement, chaque donnée est dans un cluster
    for dataId in 1:size(x, 1)
        push!(clusters, Cluster(dataId, x, y))
    end

    # Id du cluster de chaque donnée dans clusters
    # (clusters[clusterId[i]] est le cluster contenant i)
    # (clusterId[i] = i, initialement)
    clusterId = collect(1:n)

    # Distances entre des couples de données de même classe
    distances = Vector{Distance}([])

    # Pour chaque couple de données de même classe
    for id1 in 1:n-1
        for id2 in id1+1:n
            if y[id1] == y[id2]
                # Ajoute leur distance
                push!(distances, Distance(id1, id2, x))
            end
        end
    end

    # Trie des distances par ordre croissant
    sort!(distances, by = v -> v.distance)
    
    # Pour chaque distance
    for distance in distances

        # Si les deux données associées ne sont pas déjà dans le même cluster
        cId1 = clusterId[distance.ids[1]]
        cId2 = clusterId[distance.ids[2]]

        if cId1 != cId2
            c1 = clusters[cId1]
            c2 = clusters[cId2]

            # Si leurs clusters satisfont les conditions de fusion
            if canMerge(c1, c2, x, y)

                # Les fusionner
                merge!(c1, c2)
                for id in c2.dataIds
                    clusterId[id]= cId1
                end

                # Vider le second cluster
                empty!(clusters[cId2].dataIds)
            end 
        end 
    end

    # Retourner tous les clusters non vides
    return filter(x -> length(x.dataIds) > 0, clusters)
end

"""
Regroupe des données en commençant par celles qui sont les plus proches jusqu'à ce qu'un certain pourcentage de clusters soit atteint

Entrées :
- x : caractéristiques des données
- y : classe des données
- gamma : le regroupement se termine quand il reste un nombre de clusters < n * gamma ou que plus aucun regroupement n'est possible

Sorties :
- un tableau de Cluster constituant une partition de x
"""
function simpleMerge(x, y, gamma)

    n = length(y)
    m = length(x[1,:])
    
    # Liste de tous les clusters obtenus
    # (les données non regroupées seront seules dans un cluster)
    clusters = Vector{Cluster}([])

    # Initialement, chaque donnée est dans un cluster
    for dataId in 1:size(x, 1)
        push!(clusters, Cluster(dataId, x, y))
    end
    # Id du cluster de chaque donnée dans clusters
    # (clusters[clusterId[i]] est le cluster contenant i)
    # (clusterId[i] = i, initialement)
    clusterId = collect(1:n)

    # Distances entre des couples de données de même classe
    distances = Vector{Distance}([])

    # Pour chaque couple de données de même classe
    for id1 in 1:n-1
        for id2 in id1+1:n
            if y[id1] == y[id2]
                # Ajoute leur distance
                push!(distances, Distance(id1, id2, x))
            end
        end
    end
    # Trie des distances par ordre croissant
    sort!(distances, by = v -> v.distance)

    remainingClusters = n
    distanceId = 1

    # Pour chaque distance et tant que le nombre de cluster souhaité n'est pas atteint
    while distanceId <= length(distances) && remainingClusters > n * gamma

        distance = distances[distanceId]
        cId1 = clusterId[distance.ids[1]]
        cId2 = clusterId[distance.ids[2]]

        # Si les deux données associées ne sont pas déjà dans le même cluster
        if cId1 != cId2
            remainingClusters -= 1

            # Fusionner leurs clusters 
            c1 = clusters[cId1]
            c2 = clusters[cId2]
            merge!(c1, c2)
            for id in c2.dataIds
                clusterId[id]= cId1
            end

            # Vider le second cluster
            empty!(clusters[cId2].dataIds)
        end
        distanceId += 1
    end
    
    # Retourner tous les clusters non vides
    return filter(x -> length(x.dataIds) > 0, clusters)
end 

"""
Test si deux clusters peuvent être fusionnés tout en garantissant l'optimalité

Entrées :
- c1 : premier cluster
- c2 : second cluster
- x  : caractéristiques des données d'entraînement
- y  : classe des données d'entraînement

Sorties :
- vrai si la fusion est possible ; faux sinon.
"""
function canMerge(c1::Cluster, c2::Cluster, x::Matrix{Float64}, y::Vector{Int})

    # Calcul des bornes inférieures si c1 et c2 étaient fusionnés
    mergedLBounds = min.(c1.lBounds, c2.lBounds)
    
    # Calcul des bornes supérieures si c1 et c2 étaient fusionnés
    mergedUBounds = max.(c1.uBounds, c2.uBounds)

    n = size(x, 1)
    id = 1
    canMerge = true

    # Tant que l'ont a pas vérifié que toutes les données n'intersectent la fusion de c1 et c2 sur aucune feature
    while id <= n && canMerge

        data = x[id, :]

        # Si la donnée n'est pas dans c1 ou c2 mais intersecte la fusion de c1 et c2 sur au moins une feature
        if !(id in c1.dataIds) && !(id in c2.dataIds) && isInABound(data, mergedLBounds, mergedUBounds)
            canMerge = false
        end 
        
        id += 1
    end 

    return canMerge
end

"""
Test si une donnée intersecte des bornes pour au moins une caractéristique 

Entrées :
- v : les caractéristique de la donnée
- lowerBounds : bornes inférieures pour chaque caractéristique
- upperBounds : bornes supérieures pour chaque caractéristique

Sorties :
- vrai s'il y a intersection ; faux sinon.
"""
function isInABound(v::Vector{Float64}, lowerBounds::Vector{Float64}, upperBounds::Vector{Float64})
    isInBound = false

    featureId = 1

    # Tant que toutes les features n'ont pas été testées et qu'aucune intersection n'a été trouvée
    while !isInBound && featureId <= length(v)

        # S'il y a intersection
        if v[featureId] >= lowerBounds[featureId] && v[featureId] <= upperBounds[featureId]
            isInBound = true
        end 
        featureId += 1
    end 

    return isInBound
end

"""
Fusionne deux clusters

Entrées :
- c1 : premier cluster
- c2 : second cluster

Sorties :
- aucune, c'est le cluster en premier argument qui contiendra le second
"""
function merge!(c1::Cluster, c2::Cluster)

    append!(c1.dataIds, c2.dataIds)
    c1.x = vcat(c1.x, c2.x)
    c1.lBounds = min.(c1.lBounds, c2.lBounds)
    c1.uBounds = max.(c1.uBounds, c2.uBounds)    
end

function NoClassMerge(x::Matrix{Float64},y::Vector{Int}, n_clusters::Int=4)
    n = length(y)
    m = length(x[1,:])
    clusters = Vector{Cluster}([])
    for dataId in 1:size(x,1)
        push!(clusters, Cluster(dataId, x,y))
    end
    clusterId = collect(1:n)
    distances = Vector{Distance}([])
    for id1 in 1:n-1
        for id2 in id1+1:n
            push!(distances, Distance(id1, id2, x))
        end
    end
    sort!(distances, by = v ->v.distance)
    remainingClusters=n
    distanceId = 1
    while distanceId <= length(distances) &&remainingClusters>n_clusters
        distance = distances[distanceId]
        cId1 =clusterId[distance.ids[1]]
        cId2 = clusterId[distance.ids[2]]
        if cId1 != cId2
            remainingClusters -=1
            c1 = clusters[cId1]
            c2 = clusters[cId2]
            merge!(c1, c2)
            for id in c2.dataIds
                clusterId[id]= cId1
            end

        # Vider le second cluster
            empty!(clusters[cId2].dataIds)
        end
        distanceId += 1
    end
    # Retourner tous les clusters non vides
    return filter(x -> length(x.dataIds) > 0, clusters)
end

function compute_dataframe_classes(
    cluster_interest::Int64,
    clusterId::Vector{Int64},
    y::Vector{Int64}
)
    count_classes_in_clusters = StatsBase.countmap(y[findall(x->x==cluster_interest, clusterId)])
    y_cluster = y[findall(x->x==cluster_interest, clusterId)]
    df = DataFrame()
    df."classes" = collect(keys(count_classes_in_clusters))
    df."count_classes" = collect(values(count_classes_in_clusters))
    df = sort(df, [:count_classes], rev=true)
    return (df, y_cluster)
end

function detect_points_small_classes(
    cluster_interest::Int64,
    clusterId::Vector{Int64},
    y::Vector{Int64},
    main_class::Int64,
)
    points_in_cluster = findall(x->x==cluster_interest, clusterId)
    points_reckon_with = Vector{Int64}([])
    points_sideline = Vector{Int64}([])
    for (i, point) in enumerate(points_in_cluster)
        if y[point] != main_class
            push!(points_sideline,i )
        else
            push!(points_reckon_with, i)
        end
    end
    return (points_reckon_with, points_sideline)
end

function trimming_cluster(
    clusters::Vector{Cluster},
    cluster_interest::Int64,
    points_reckon_with::Vector{Int64},
    points_sideline::Vector{Int64},
    y_cluster::Vector{Int64},
)
    matrix_interest = clusters[cluster_interest].x
    size(matrix_interest, 1)
    vectors_cluster = [matrix_interest[i,:] for i in 1:size(matrix_interest,1)]
    vectors_cluster[points_reckon_with]
    hull = LazySets.convex_hull(vectors_cluster[points_reckon_with])
    for i in 1:length(points_sideline)
        # println(i, " ",element(Singleton(vectors_cluster[points_sideline[i]])) ∈ VPolytope(hull))
        if element(Singleton(vectors_cluster[points_sideline[i]])) ∈ VPolytope(hull)
            push!(points_reckon_with,  points_sideline[i])
        end
    end
    sort!(points_reckon_with)
    points_to_be_taken_out_cluster = points_sideline[points_sideline .∉ Ref(points_reckon_with)]
    new_clusters = Vector{Cluster}([])
    point_cluster  = points_reckon_with[1]
    trimmed_cluster = Cluster(point_cluster, matrix_interest, y_cluster)
    for point_cluster in points_reckon_with[2:end]
        new_cluster = Cluster(point_cluster, matrix_interest, y_cluster)
        merge!(trimmed_cluster, new_cluster)
    end

    for point_cluster in points_to_be_taken_out_cluster
        new_cluster = Cluster(point_cluster, matrix_interest, y_cluster)
        push!(clusters, new_cluster)
    end
    clusters[cluster_interest] = trimmed_cluster
    return clusters
end

function update_clusters(
    clusters::Vector{Cluster},
    cluster_interest::Int64,
    clusterId::Vector{Int64},
    y::Vector{Int64},
)
    df, y_cluster = compute_dataframe_classes(cluster_interest, clusterId, y)
    if size(df)[1]>1
        main_class = df[1,1]
        points_reckon_with, points_sideline =detect_points_small_classes(cluster_interest, clusterId, y,main_class)
        clusters = trimming_cluster(clusters, cluster_interest,points_reckon_with, points_sideline, y_cluster)
        return clusters
    else
        return clusters
    end
end

function ConvexHullMerge(
    x::Matrix{Float64}, 
    y::Vector{Int}, 
    max_elements_small_classes::Int64,
    num_clusters::Int64,
)
    n = length(y)
    m = length(x[1,:])
    clusters = Vector{Cluster}([])
    for dataId in 1:size(x,1)
        push!(clusters, Cluster(dataId, x,y))
    end
    clusterId = collect(1:n) #On obtient un vecteur 1,2..., qui correspond pour chaque cluster à son clusterId
    distances = Vector{Distance}([])
    for id1 in 1:n-1
        for id2 in id1+1:n
            push!(distances, Distance(id1, id2, x))
        end
    end
    sort!(distances, by = v ->v.distance)
    remainingClusters=n
    distanceId = 1
    n_epochs = 1
    c1_bis = Nothing
    c2_bis = Nothing
    i = 1
    a=i
    

    while remainingClusters>= num_clusters
        distance = distances[distanceId]
        cId1 =clusterId[distance.ids[1]]
        cId2 = clusterId[distance.ids[2]]
        if cId1 != cId2
            c1 = clusters[cId1]
            c2 = clusters[cId2]
            count_classes_in_clusters = StatsBase.countmap([y[c2.dataIds]; y[c1.dataIds]])
            df = DataFrame()
            df."classes" = collect(keys(count_classes_in_clusters))
            df."count_classes" = collect(values(count_classes_in_clusters))
            df = sort(df, [:count_classes], rev=true)
            # if sum(df[df."count_classes".!=maximum(df."count_classes"),:]."count_classes") <= max_elements_small_classes
            if sum(df[df."classes".!=df[1,1],"count_classes"])<= max_elements_small_classes
                remainingClusters -=1
                if remainingClusters < num_clusters
                    break
                end
                merge!(c1, c2) #On merge les 2 clusters
                for id in c2.dataIds 
                    clusterId[id]= cId1 #On modifie le clusterId dans la serie pour le cluster_2, on lui affecte le cluster_1
                end
                # Vider le second cluster
                empty!(clusters[cId2].dataIds)
            end 
        end
        distanceId += 1
        if distanceId>=length(distances)
            break
        end
    end
    print("we're moving forward in algorithm")
    df_clusters = DataFrame()
    df_clusters."cluster_id" = collect(keys(StatsBase.countmap(clusterId)))
    df_clusters."number_elements" = collect(values(StatsBase.countmap(clusterId)))
    higher_than_threshold(value::Int64) = value >= 1
    clusters_to_treat = filter(:"number_elements"=> higher_than_threshold, df_clusters)."cluster_id"
    for cluster_interest in clusters_to_treat
        clusters = update_clusters(
            clusters,
            cluster_interest,
            clusterId,
            y
        )   
    end
    return filter(x -> length(x.dataIds) > 0, clusters)
end 


function Kmeans_Based_Merge(x::Matrix{Float64},y::Vector{Int}, number_clusters::Int=4)
    result_kmeans = Clustering.kmeans(transpose(x), number_clusters)
    a = assignments(result_kmeans)
    centers = result_kmeans.centers
    clusters = Vector{Cluster}([])
    for cluster_interest in 1:number_clusters
        count_classes_in_clusters = StatsBase.countmap(y[findall(x->x==cluster_interest, a)])
        y_cluster = y[findall(x->x==2, a)]
        df = DataFrame()
        df."classes" = collect(keys(count_classes_in_clusters))
        df."count_classes" = collect(values(count_classes_in_clusters))
        df = sort(df, [:count_classes], rev=true)
        number_classes = size(df, 1)
        if number_classes > 1
            println(cluster_interest)
        end
        points_cluster = findall(x->x==cluster_interest, a)
        cluster_group = Cluster(points_cluster[1], x, y)
        for point_cluster in points_cluster[2:end]
            new_cluster = Cluster(point_cluster, x, y)
            merge!(cluster_group, new_cluster)
        end
        push!(clusters, cluster_group)
    end
    clusterId = a
    df_clusters = DataFrame()
    df_clusters."cluster_id" = collect(keys(StatsBase.countmap(clusterId)))
    df_clusters."number_elements" = collect(values(StatsBase.countmap(clusterId)))
    higher_than_threshold(value::Int64) = value >= 1
    clusters_to_treat = filter(:"number_elements"=> higher_than_threshold, df_clusters)."cluster_id"
    for cluster_interest in clusters_to_treat
        clusters = update_clusters(
            clusters,
            cluster_interest,
            clusterId,
            y
        )   
    end
    return filter(x -> length(x.dataIds) > 0, clusters)
end

function PercentageMerge(x::Matrix{Float64},y::Vector{Int}, n_clusters::Int=4)
    n = length(y)
    m = length(x[1,:])
    clusters = Vector{Cluster}([])
    for dataId in 1:size(x,1)
        push!(clusters, Cluster(dataId, x,y))
    end
    clusterId = collect(1:n)
    distances = Vector{Distance}([])
    for id1 in 1:n-1
        for id2 in id1+1:n
            push!(distances, Distance(id1, id2, x))
        end
    end
    sort!(distances, by = v ->v.distance)
    remainingClusters=n
    distanceId = 1
    while distanceId <= length(distances) &&remainingClusters>n_clusters
        distance = distances[distanceId]
        cId1 =clusterId[distance.ids[1]]
        cId2 = clusterId[distance.ids[2]]
        if cId1 != cId2
            remainingClusters -=1
            c1 = clusters[cId1]
            c2 = clusters[cId2]
            merge!(c1, c2)
            for id in c2.dataIds
                clusterId[id]= cId1
            end

        # Vider le second cluster
            empty!(clusters[cId2].dataIds)
        end
        distanceId += 1
    end
    # Retourner tous les clusters non vides
    return filter(x -> length(x.dataIds) > 0, clusters)
end

