using Distributed, SharedArrays, JLD2

@everywhere using Optim, Compat, LinearAlgebra, Statistics, Random, Dates
@everywhere include("../utils.jl")
@everywhere include("../FFT.jl")
@everywhere using .MepdCopula, .Utils

dimension = 4
nObs = 5*nprocs()


#Random.seed!(321)
true_par = [log(1.0), 1.0, 0.5] # lambda, nu, p
coord = rand(dimension, 2)
dist = vcat(dist_fun(coord[:, 1]), dist_fun(coord[:, 2]))
cor_mat = cor_fun(reshape(sqrt.(dist[1, :] .^ 2 .+ dist[2, :] .^ 2), dimension, dimension), true_par)
dat = rC(nObs, dimension, cor_mat, true_par[3])
(n, D) = size(dat)

nllik = function (param::Vector{Float64}, dat::Matrix{Float64}, coord::Matrix{Float64}, n::Integer, D::Integer, ncores::Integer)
    if !cond_cor(param) # check conditions on parameters
        return 1e+10
    end

    # compute the matrix of correlations in W
    dists = vcat(dist_fun(coord[:, 1]), dist_fun(coord[:, 2]))
    Sigmab = cor_fun(reshape(sqrt.(dists[1, :] .^ 2 .+ dists[2, :] .^ 2), D, D), param)
    if !isposdef(Sigmab)
        return 1e+10
    end

    nllik_res = SharedVector{Float64}(ncores)
    @sync @distributed for i in 1:ncores # ncores can be no larger than the number of observations
        nllik_res[i] = nllik_block(i, dat, param, Sigmab, n, ncores)
    end
    if any(isnan.(nllik_res))
        return 1e+10
    else
        return sum(nllik_res)
    end
end

@everywhere nllik_block = function (block::Integer, dat::Matrix{Float64}, param::Vector{Float64}, Sigmab::Matrix{Float64}, n::Integer, ncores::Integer)
    if ncores > 1
        indmin = vcat(0.5, quantile(1:n, LinRange(1 / ncores, (ncores - 1) / ncores, ncores - 1)))[block]
        indmax = vcat(quantile(1:n, LinRange(1 / ncores, (ncores - 1) / ncores, ncores - 1)), n + 0.5)[block]
        ind_block = round.(Int, LinRange(1, n, n)[(1:n .> indmin) .&& (1:n .≤ indmax)]) # indices of the specific block
    elseif ncores == 1
        ind_block = 1:n
    end
    contrib = dC(reshape(dat[ind_block, :], length(ind_block), D), Sigmab, param[3])
    return -sum(contrib)
end

x = optimize(x -> nllik(x, dat, coord, n, D, nprocs()), true_par, NelderMead(), 
                   Optim.Options(g_tol = 1e-4, # default 1e-8
                                 show_trace = true,
                                 show_every = 1,
                                 extended_trace = true)
                    )


jldsave("test.jld2"; x)