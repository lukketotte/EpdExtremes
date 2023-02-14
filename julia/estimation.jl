using Distributed, SharedArrays

@everywhere using Optim, Compat, LinearAlgebra, Statistics, Random, Dates
@everywhere include("./utils.jl")
@everywhere include("./FFT.jl")
@everywhere using .MepdCopula, .Utils
dimension = 2
nObs = nprocs() * 12

#Random.seed!(32)
true_par = [log(1.0), 1., 0.6] # lambda, nu, p
coord = rand(dimension, 2)
dist = vcat(dist_fun(coord[:, 1]), dist_fun(coord[:, 2]))
cor_mat = cor_fun(reshape(sqrt.(dist[1, :] .^ 2 .+ dist[2, :] .^ 2), dimension, dimension), true_par)
dat = rC(nObs, cor_mat, true_par[3])
(n, D) = size(dat)

#############################################
# uncensored powered exponential
#############################################
function copula_nocens(dat::Matrix{Float64}, coord::Matrix{Float64}, init_val::Vector{Float64}, ncores::Integer)
    # 2023-01-11, 13:10 - assuming data is complete for now. add checks for missing values later
    (n, D) = size(dat)

    opt_res = optimize(x -> nllik(x, dat, coord, n, D, ncores), init_val, NelderMead())
    # shoul we also calculate gradient/Hessian/SE?
    return [opt_res.Minimizer, opt_res.Minimum, opt_res.Iterations]
end

function nllik(param::Vector{Float64}, dat::Matrix{Float64}, coord::Matrix{Float64}, n::Integer, D::Integer, ncores::Integer)
    if !cond_cor(param) # check conditions on parameters
        return 1e+10
    end

    # compute the matrix of correlations in W
    dists = vcat(dist_fun(coord[:, 1]), dist_fun(coord[:, 2]))
    Sigmab = cor_fun(reshape(sqrt.(dists[1, :] .^ 2 .+ dists[2, :] .^ 2), D, D), param)
    if !isposdef(Sigmab)
        return 1e+10
    end

    nllik_res = SharedArray{Float64}(ncores)
    @sync @distributed for i in 1:ncores # ncores can be no larger than the number of observations
        nllik_res[i] = nllik_block(i, dat, param, Sigmab, n, ncores)
    end
    if any(isnan.(nllik_res))
        return 1e+10
    else
        return sum(nllik_res)
    end
end

@everywhere function nllik_block(block::Integer, dat::Matrix{Float64}, param::Vector{Float64}, Sigmab::Matrix{Float64}, n::Integer, ncores::Integer)
    if ncores > 1
        indmin = vcat(0.5, quantile(1:n, LinRange(1 / ncores, (ncores - 1) / ncores, ncores - 1)))[block]
        indmax = vcat(quantile(1:n, LinRange(1 / ncores, (ncores - 1) / ncores, ncores - 1)), n + 0.5)[block]
        ind_block = round.(Int, LinRange(1, n, n)[(1:n .> indmin) .&& (1:n .≤ indmax)]) # indices of the specific block
    elseif ncores == 1
        ind_block = 1:n
    end
    contrib = dC(reshape(dat[ind_block, :], length(ind_block), size(Sigmab, 1)), Sigmab, param[3])
    return -sum(contrib)
end

@time x = optimize(x -> nllik(x, dat, coord, n, D, nprocs()), true_par, NelderMead(), 
                   Optim.Options(g_tol = 2e-3, # default 1e-8
                                 show_trace = true,
                                 show_every = 1,
                                 extended_trace = true)
                    )




#############################################
# censored powered exponential
#############################################
dimension = 4
nObs = 4 * 2
Random.seed!(3454)
true_par = [1.0, 1.0, 0.5] # lambda, nu, p
coord = rand(dimension, 2)
dist = vcat(dist_fun(coord[:, 1]), dist_fun(coord[:, 2]))
cor_mat = cor_fun(reshape(sqrt.(dist[1, :] .^ 2 .+ dist[2, :] .^ 2), dimension, dimension), true_par)
dat = rC(nObs, dimension, cor_mat, true_par[3])
(n, D) = size(dat)

round(quantile(vec(dat), 0.5), digits=2)
thres = 0.7

copula_cens = function (dat::Matrix{Float64}, coord::Matrix{Float64}, thres::Real, init_val::Vector{Float64}, ncores::Integer)
    inds, I_exc, I_nexc_nb, I_nexc_len = censoring(dat, thres)
    opt_res = optimize(x -> nllik_cens(x, dat_cens, coord, thres, inds, I_exc, I_nexc_nb, I_nexc_len, n, D, ncores), init_val, NelderMead())
    # shoul we also calculate gradient/Hessian/SE?
    return [opt_res.Minimizer, opt_res.Minimum, opt_res.Iterations]
end

Dates.format(now(), "HH:MM")
@time x = optimize(x -> nllik_cens(x, dat_cens, coord, thres, inds, I_exc, I_nexc_nb, I_nexc_len, n, D, 4), [0.1,0.1,0.75], NelderMead(),
    Optim.Options(g_tol = 2e-3, # default 1e-8
        show_trace = true,
        show_every = 1,
        extended_trace = true)
)
Dates.format(now(), "HH:MM")

Optim.minimizer(x)

# Dates.format(now(), "HH:MM")
# @time test = nllik_cens([1.0, 1.0, 0.9], dat_cens, coord, thres, inds, I_exc, I_nexc_nb, I_nexc_len, n, D, 1)

function nllik(param::Vector{Float64}, dat::Matrix{Float64}, coord::Matrix{Float64}, thres::Real, inds::BitVector, I_exc::Vector{Vector{Int64}}, I_nexc_nb::Vector{Int64}, I_nexc_len::Vector{Int64}, n::Integer, D::Integer, ncores::Integer)
    # check conditions on parameters
    if !cond_cor(param)
        return 1e+10
    end

    # matrix of correlations in W
    dists = vcat(dist_fun(coord[:, 1]), dist_fun(coord[:, 2]))
    Sigmab = cor_fun(reshape(sqrt.(dists[1, :] .^ 2 .+ dists[2, :] .^ 2), D, D), param)
    if !isposdef(Sigmab)
        return 1e+10
    end

    # compute likelihood for partial or full exceedances
    nllik_res = SharedArray{Float64}(ncores)
    @sync @distributed for i in 1:ncores # ncores can be no larger than the number of observations
        nllik_res[i] = nllik_block_cens(i, dat, I_exc, param, Sigmab, inds, n, ncores)
    end
    if any(isnan.(nllik_res))
        return 1e+10
    end

    # fully censored observations
    contrib3 = SharedArray{Float64}(length(I_nexc_nb))
    @sync @distributed for i in eachindex(I_nexc_nb)
        contrib3[i] = sum(I_nexc_nb[i] .* pC(reshape(repeat([thres], I_nexc_len[i]), I_nexc_len[i], 1), Sigmab[I_nexc[i], I_nexc[i]], param[3]))
    end
    
    return sum(nllik_res) - sum(contrib3)
end


nllik_block_cens = function (block::Integer, dat::Matrix{Float64}, I_exc::Vector{Vector{Int64}}, param::Vector{Float64}, Sigmab::Matrix{Float64}, inds::BitVector, n::Integer, ncores::Integer)
    if ncores > 1
        indmin = vcat(0.5, quantile(1:sum(inds), LinRange(1 / ncores, (ncores - 1) / ncores, ncores - 1)))[block]
        indmax = vcat(quantile(1:sum(inds), LinRange(1 / ncores, (ncores - 1) / ncores, ncores - 1)), n + 0.5)[block]
        ind_block = round.(Int, LinRange(1, sum(inds), sum(inds))[(1:sum(inds) .> indmin) .&& (1:sum(inds) .≤ indmax)]) # indices of the specific block
    elseif ncores == 1
        ind_block = 1:sum(inds)
    end
    contrib1 = 0
    contrib2 = 0
    if sum(I1[inds][ind_block]) > 0 # no censoring
        contrib1 = sum(dC(reshape(dat[inds, :][ind_block, :][I1[inds][ind_block], :], sum(I1[inds][ind_block]), D), Sigmab, param[3]))
    end
    if sum(I2[inds][ind_block]) > 0 # partial censoring
        contrib2 = sum(dCI(reshape(dat[inds, :][ind_block, :][I2[inds][ind_block], :], sum(I2[inds][ind_block]), D), I_exc[inds][ind_block][I2[inds][ind_block]], Sigmab, param[3]))
    end
    return -(contrib1 + contrib2)
end

