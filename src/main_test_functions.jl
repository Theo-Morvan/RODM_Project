include("building_tree.jl")
include("utilities.jl")
include("merge.jl")
using DataFrames, Random
using DataStructures
using StatsBase
using Statistics
using LinearAlgebra
using LazySets
using Polyhedra
using Debugger
using CSV

function main_merge()
    print("we're starting ConvexHullMerge methode")
    println("\n")
    final_dataframe_univariate = DataFrame(
        dataset_name=Any[],
        D=Int64[], 
        number_clusters=Int64[],
        Errors_train=Int64[],
        Errors_test=Int64[],
        Resolution_time=Any[],
    )
    final_dataframe_multivariate = DataFrame(
        dataset_name=Any[],
        D=Int64[], 
        number_clusters=Int64[],
        Errors_train=Int64[],
        Errors_test=Int64[],
        Resolution_time=Any[],
    )
    for dataSetName in ["titanic","iris", "seeds", "wine", "penguins"]
        
        print("=== Dataset ", dataSetName)
        @bp
        # Préparation des données
        include("./data/" * dataSetName * ".txt")
        
        # Ramener chaque caractéristique sur [0, 1]
        reducedX = Matrix{Float64}(X)
        for j in 1:size(X, 2)
            reducedX[:, j] .-= minimum(X[:, j])
            reducedX[:, j] ./= maximum(X[:, j])
        end

        train, test = train_test_indexes(length(Y))
        X_train = reducedX[train,:]
        Y_train = Y[train]
        X_test = reducedX[test,:]
        Y_test = Y[test]
        classes = unique(Y)

        println(" (train size ", size(X_train, 1), ", test size ", size(X_test, 1), ", ", size(X_train, 2), ", features count: ", size(X_train, 2), ")")
        
        # Temps limite de la méthode de résolution en secondes        
        time_limit = 10

        for D in 2:4
            println("\tD = ", D)
            println("\t\tUnivarié")
            df_temporary = testMerge(dataSetName ,X_train, Y_train, X_test, Y_test, D, classes, time_limit = time_limit, isMultivariate = false)
            final_dataframe_univariate = vcat(final_dataframe_univariate, df_temporary)
            println("\t\tMultivarié")
            df_temporary = testMerge(dataSetName ,X_train, Y_train, X_test, Y_test, D, classes, time_limit = time_limit, isMultivariate = true)
            final_dataframe_multivariate = vcat(final_dataframe_multivariate)
        end
    end
    CSV.write("final_results_univariate_exactmerge.csv", final_dataframe_univariate)
    CSV.write("final_results_multivariate_exactmerge.csv", final_dataframe_multivariate)
    return (final_dataframe_univariate, final_dataframe_multivariate)
end 


function testMerge(dataset_name, X_train, Y_train, X_test, Y_test, 
    D, classes; time_limit::Int=-1, isMultivariate::Bool = false,

)
    df_dataset_D = DataFrame(
        dataset_name=Any[],
        D=Int64[], 
        number_clusters=Int64[],
        Errors_train=Int64[],
        Errors_test=Int64[],
        Resolution_time=Any[],
    )
    # Pour tout pourcentage de regroupement considéré
    println("\t\t\tGamma\t\t# clusters\tGap")
    for number_cluster in 5:3:20
        # n = Int64(gamma*length(Y_train))
        # print("\t\t\t", gamma * 100, "%\t\t")
        clusters = exactMerge(X_train, Y_train)
        print("\t exactMerge clusters done")
        # clusters_bis = simpleMerge(X_train, Y_train, gamma)
        # clusters_third = NoClassMerge(X_train, Y_train, n)  
        # clusters = ConvexHullMerge(X_train, Y_train, 5,number_cluster)
        # clusters = ConvexHullMerge(X_train, Y_train, 5,number_cluster)
        T, obj, resolution_time, gap = build_tree(clusters, D, classes, multivariate = isMultivariate, time_limit = time_limit)
        
        print(round(gap, digits = 1), "%\t") 
        print("Erreurs train/test : ", prediction_errors(T,X_train,Y_train, classes))
        print("/", prediction_errors(T,X_test,Y_test, classes), "\t")
        errors_train = prediction_errors(T, X_train, Y_train, classes)
        errors_test = prediction_errors(T, X_test, Y_test, classes)
        println(round(resolution_time, digits=1), "s")
        output_vector = [dataset_name, D, number_cluster, errors_train, errors_test, round(resolution_time, digits=1)] 
        push!(df_dataset_D, output_vector)
    end
    println()
    return df_dataset_D
end


