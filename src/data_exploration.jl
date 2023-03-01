include("building_tree.jl")
include("utilities.jl")

function main()

    # Pour chaque jeu de données
    for dataSetName in ["iris"]
        
        print("=== Dataset ", dataSetName)

        # Préparation des données
        include("./data/" * dataSetName * ".txt")

        # Ramener chaque caractéristique sur [0, 1]
        reducedX = Matrix{Float64}(X)
        for j in 1:size(X, 2)
            reducedX[:, j] .-= minimum(X[:, j])
            reducedX[:, j] ./= maximum(X[:, j])
        end

        train, test = train_test_indexes(length(Y))
        X_train = reducedX[train, :]
        Y_train = Y[train]
        X_test = reducedX[test, :]
        Y_test = Y[test]
        classes = unique(Y)

        println(" (train size ", size(X_train, 1), ", test size ", size(X_test, 1), ", ", size(X_train, 2), ", features count: ", size(X_train, 2), ")")
        
        # Temps limite de la méthode de résolution en secondes
        println("Attention : le temps est fixé à 30s pour permettre de faire des tests rapides. N'hésitez pas à l'augmenter lors du calcul des résultats finaux que vous intégrerez à votre rapport.")
        time_limit = 30
        println(size(X_train))
        cluster = Cluster(20, X_train, Y_train)
        println(cluster, cluster.dataIds, "lowerBounds",cluster.lBounds, "upperBounds",cluster.uBounds)
        # Pour chaque profondeur considérée
        
    end
end
