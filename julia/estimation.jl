import Optim
using Compat, LinearAlgebra, Statistics, Random, Dates

Threads.nthreads()

dimension = 2
nObs = 1

Random.seed!(321)
true_par = [1.0, 1.0, 0.5] # lambda, nu, p
coord = rand(dimension, 2)
dist = vcat(dist_fun(coord[:, 1]), dist_fun(coord[:, 2]))
cor_mat = cor_fun(reshape(sqrt.(dist[1, :] .^ 2 .+ dist[2, :] .^ 2), dimension, dimension), true_par)
dat = rC(nObs, dimension, cor_mat, true_par[3])
(n, D) = size(dat)

@time x = optimize(x -> nllik(x, dat, coord, n, D, 1), true_par, NelderMead(), 
                   Optim.Options(g_tol = 1e-3, # default 1e-8
                                 show_trace = true,
                                 show_every = 1,
                                 extended_trace = true)
                    )

#############################################
# uncensored powered exponential
#############################################
copula_nocens = function (dat::Matrix{Float64}, coord::Matrix{Float64}, init_val::Vector{Float64}, ncores::Integer)
    # 2023-01-11, 13:10 - assuming data is complete for now. add checks for missing values later
    (n, D) = size(dat)

    x = Optim.optimize(x -> nllik(x, dat, coord, n, D, ncores), init_val, Optim.NelderMead())
    # shoul we also calculate gradient/Hessian/SE?
    return [x.Minimizer, x.Minimum, x.Iterations]
end

cor_fun = function (h::Matrix{Float64}, param::Vector{Float64})
    return exp.(-(h ./ exp(param[1])).^param[2])
end

cond_cor = function (param::Vector{Float64})
    return param[2] > 0 && param[2] < 2 && param[3] > 0 && param[3] < 1 # 0.95
end

nllik = function (param::Vector{Float64}, dat::Matrix{Float64}, coord::Matrix{Float64}, n::Integer, D::Integer, ncores::Integer)
    # println("Parameters: ", param , ". Time: ", Dates.format(now(), "HH:MM") )
    if !cond_cor(param) # check conditions on parameters
        return 1e+10
    end

    # calculate the distance matrix
    dists = vcat(dist_fun(coord[:, 1]), dist_fun(coord[:, 2]))
    Sigmab = cor_fun(reshape(sqrt.(dists[1, :] .^ 2 .+ dists[2, :] .^ 2), D, D), param) # compute the matrix of correlations in W

    if !isposdef(Sigmab) # check if correlatoin matrix is positive definit
        return 1e+10
    end

    nllik_res = Vector{Float64}(undef, ncores)
    # Threads.@threads 
    for i in 1:ncores # ncores can be no larger than the number of observations
        nllik_res[i] = nllik_block(i, dat, param, Sigmab, n, ncores)
    end
    if any(isnan.(nllik_res))
        @show nllik_res
        println("NaN produced")
        return 1e+10
    else
        return sum(nllik_res)
    end
end

dist_fun = function (coord_vec::Vector{Float64})
    coord_stack(j) = coord_vec
    return permutedims(vec(stack(coord_stack(j) for j = eachindex(coord_vec)) - permutedims(stack(coord_stack(j) for j = eachindex(coord_vec)))))
end


nllik_block = function (block::Integer, dat::Matrix{Float64}, param::Vector{Float64}, Sigmab::Matrix{Float64}, n::Integer, ncores::Integer)
    if ncores > 1
        indmin = vcat(0.5, quantile(1:n, LinRange(1 / ncores, (ncores - 1) / ncores, ncores - 1)))[block]
        indmax = vcat(quantile(1:n, LinRange(1 / ncores, (ncores - 1) / ncores, ncores - 1)), n + 0.5)[block]
        ind_block = round.(Int, LinRange(1, n, n)[(1:n.>indmin).==(1:n.≤indmax)][1]) # indices of the specific block
    elseif ncores == 1
        ind_block = 1:n
    end
    contrib = dC(reshape(dat[ind_block, :], length(ind_block), D), Sigmab, param[3])
    return -sum(contrib)
end



#############################################
# censored powered exponential
#############################################
dimension = 5
nObs = 5
# Random.seed!(321)
true_par = [1.0, 1.0, 0.5] # lambda, nu, p
coord = rand(dimension, 2)
dist = vcat(dist_fun(coord[:, 1]), dist_fun(coord[:, 2]))
cor_mat = cor_fun(reshape(sqrt.(dist[1, :] .^ 2 .+ dist[2, :] .^ 2), dimension, dimension), true_par) # compute the matrix of correlations in W
dat = rC(nObs, dimension, cor_mat, true_par[3])
(n, D) = size(dat)

thres = median(dat)

copula_cens = function (dat::Matrix{Float64}, coord::Matrix{Float64}, thres::Real, init_val::Vector{Float64}, ncores::Integer)
    dat_cens = dat
    dat_cens[dat .< thres] .= thres
    (n, D) = size(dat_cens)
    # 2023-01-12 skip checks for missing values for now
    n_sites = Vector{Int64}(undef, n)
    for i in 1:n
        n_sites[i] = sum(.!ismissing.(dat_cens[1, :]))
    end

    I_exc = Vector{Vector{Int64}}(undef, n)
    I_nexc = Vector{Vector{Int64}}(undef, n)
    I1 = repeat([false], n)
    I2 = repeat([false], n)
    I3 = repeat([false], n)
    for i in 1:n
        I_exc[i] = findall(dat_cens[i, :] .> thres) # indices of exceedances
        I_nexc[i] = findall(dat_cens[i, :] .≤ thres) # indices of non-exceedances
        n_exc = length(I_exc[i])
        if n_exc == D # no censoring
            I1[i] = 1
        end
        if n_exc > 0 && n_exc < D # partial censoring
            I2[i] = 1
        end
        if n_exc == 0 # fully censored
            I3[i] = 1
        end
    end

    # if there are ny fully censored observations
    if any(I3 .== 1)
        I_nexc = unique(I_nexc[I3]) # unique compositions of fully censored obs
        I_nexc_len = Vector{Int64}(undef, length(I_nexc)) 
        for i in eachindex(I_nexc)
            I_nexc_len[i] = length(I_nexc[i])
        end
        I_nexc_nb = Vector{Int64}(undef, length(I_nexc_len))
        # Threads.@threads 
        for i in eachindex(I_nexc)
            I_nexc_nb[i] = compute_I_nexc_nb_i(i, I_nexc)
        end

        dims_c = sort(unique(n_sites[I3])) # dimensions of fully censored obs. only one number for a complete data set
        nb_dims_c = Vector{Int64}(undef, length(dims_c))
        for i in eachindex(dims_c)
          nb_dims_c[i] = sum(n_sites[I3][1] == dims_c[i])
        end
        nfc = sum(I3) # number of fully censored obs
    end
    inds = I1 | I2 # indices for parallel computing of exceedance contributions (non-exceedances are treated separately)...

    x = Optim.optimize(x -> nllik(x, dat, coord, n, D, ncores), init_val, Optim.NelderMead())
    # shoul we also calculate gradient/Hessian/SE?
    return [x.Minimizer, x.Minimum, x.Iterations]
end

nllik_cens = function (param::Vector{Float64}, dat::Matrix{Float64}, coord::Matrix{Float64}, n::Integer, D::Integer, ncores::Integer)
    println("Parameters: ", param , ". Time: ", Dates.format(now(), "HH:MM") )
    
    if !cond_cor(param) # check conditions on parameters
        return 1e+10
    end

    # calculate the distance matrix
    dists = vcat(dist_fun(coord[:, 1]), dist_fun(coord[:, 2]))
    Sigmab = cor_fun(reshape(sqrt.(dists[1, :] .^ 2 .+ dists[2, :] .^ 2), D, D), param) # compute the matrix of correlations in W

    if !isposdef(Sigmab) # check if correlatoin matrix is positive definit
        return 1e+10
    end

    nllik_res = Vector{Float64}(undef, ncores)
    Threads.@threads for i in 1:ncores # ncores can be no larger than the number of observations
        nllik_res[i] = nllik_block_cens(i, dat, param, Sigmab, n, ncores)
    end
    if any(isnan.(nllik_res))
        return 1e+10
    end

    # fully censored observations
    Threads.@threads for i in eachindex(I_nexc_nb)
        I_nexc_nb[i] * pC(reshape(repeat(thres, I_nexc_len[i]), I_nexc_len[i], 1), Sigmab[I_nexc[i], I_nexc[i]], param[3]) # not finished
    end
    
    return sum(nllik_res) - sum(I_nexc_nb)
end


nllik_block_cens = function (block::Integer, dat::Matrix{Float64}, I_exc::Vector{Vector{Int64}}, param::Vector{Float64}, Sigmab::Matrix{Float64}, n::Integer, ncores::Integer)
    if ncores > 1
        indmin = vcat(0.5, quantile(1:n, LinRange(1 / ncores, (ncores - 1) / ncores, ncores - 1)))[block]
        indmax = vcat(quantile(1:n, LinRange(1 / ncores, (ncores - 1) / ncores, ncores - 1)), n + 0.5)[block]
        ind_block = round.(Int, LinRange(1, n, n)[(1:n.>indmin).==(1:n.≤indmax)][1]) # indices of the specific block
    elseif ncores == 1
        ind_block = 1:n
    end
    contrib1 = 0
    contrib2 = 0
    if sum(I1[inds][ind_block]) > 0 # no censoring
        contrib1 = sum(dC(reshape(dat[inds, :][ind_block, :][I1[inds][ind_block], :], length(ind_block), D), Sigmab, param[3]))
    end
    if sum(I2[inds][ind_block]) > 0 # partial censoring
        contrib2 = sum(dCI(reshape(dat[inds, :][ind_block, :][I2[inds][ind_block], :], length(ind_block), D), I_exc[inds][ind_block][I2[inds][ind_block]], Sigmab, param[3]))
    end
    return -(contrib1 + contrib2)
end

same_vec = function (vec1::Vector{Int64}, vec2::Vector{Int64})
    if length(vec1) != length(vec2)
        return false
    else
        return !any(vec1 .∉ [vec2])
    end
end

compute_I_nexc_nb_i = function (i::Integer, I_nexc::Vector{Vector{Int64}})
    res = 0
    for j in eachindex(I_nexc)
        res += same_vec(I_nexc[j], I_nexc[i])
    end
    return res
end

