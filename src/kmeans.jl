function num_clustersmeans(x,num_clusters, maxiters = 100, tol = 1e-5)
    x = collect(eachrow(x))
    N = length(x)
    n = length(x[1])
    distances = zeros(N) # used to store the distance of each
    # point to the nearest representative.
    reps = [zeros(n) for j=1:num_clusters] # used to store representatives.

    # ’assignment’ is an array of N integers between 1 and num_clusters.
    # The initial assignment is chosen randomly.
    assignment = [ rand(1:num_clusters) for i in 1:N ]

    Jprevious = Inf # used in stopping condition

    for iter = 1:maxiters

        # Cluster j representative is average of points in cluster j.
        for j = 1:num_clusters
            group = [i for i=1:N if assignment[i] == j]
            reps[j] = sum(x[group]) / length(group);
        end;

        # For each x[i], find distance to the nearest reprepresentative
        # and its group index

        for i = 1:N
            (distances[i], assignment[i]) = findmin([norm(x[i] - reps[j]) for j = 1:num_clusters])
        end;

        # Compute clustering objective.
        J = norm(distances)^2 / N

        # Show progress and terminate if J stopped decreasing.
        println("Iteration ", iter, ": Jclust = ", J, ".")

        if iter > 1 && abs(J - Jprevious) < tol * J
            return assignment, reps
        end

        Jprevious = J
    end

end
