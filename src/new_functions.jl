include("src\\building_tree.jl")
include("src\\utilities.jl")
include("src\\merge.jl")
using Debugger
using DataFrames, Random
using DataStructures
using StatsBase
using Statistics
using LinearAlgebra
using LazySets
using Polyhedra

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
    y::Vector{Int64}
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
    cluster_interest::int64,
    points_reckon_with::Vector{int64},
    points_sideline::Vector{int64},
)
    matrix_interest = clusters[cluster_interest].x
    size(matrix_interest, 1)
    vectors_cluster = [matrix_interest[i,:] for i in 1:size(matrix_interest,1)]
    vectors_cluster[points_reckon_with]
    hull = LazySets.convex_hull(vectors_cluster[points_reckon_with])
    for i in 1:length(points_sideline)
        println(i, " ",element(Singleton(vectors_cluster[points_sideline[i]])) ∈ VPolytope(hull))
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
    clusters:Vector{Cluster},
    cluster_interest::Int64,
    clusterId::Vector{Int64},
    y::Vector{Int64},
)
    df, y_cluster = compute_dataframe_classes(cluster_interest, clusterId, y)
    if size(df)[1]>1
        points_reckon_with, points_sideline =detect_points_small_classes(cluster_interest, clusterId, y)
        clusters = trimming_cluster(clusters, cluster_interest, )
    else
        return clusters
    end
end