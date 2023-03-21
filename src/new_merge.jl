include("struct/distance.jl")
include("merge.jl")

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

function FractionsClassMerge(x::Matrix{Float64},y::Vector{Int}, n_clusters::Int=4, maximal_amount::Int64=20)
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
            if y[id1] == y[id2]
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