################################
### QQ-plots for sum exceedances
################################

using DelimitedFiles, Distributions, Plots

include("./utils.jl")
include("./FFT.jl")
using .Utils

# include("./Distributions/mepd.jl")
# using .MepdCopula, .MultivariateEpd

coord = convert(Matrix{Float64}, readdlm("./application/data/wind_gust_coordinates_km.csv", ',')[2:end,:]) # lon, lat
dimension = size(coord, 1)
dist = vcat(dist_fun(coord[:, 1]), dist_fun(coord[:, 2]));
data = convert(Matrix{Float64}, readdlm("./application/data/model_data_complete.csv", ',')[2:end,:])
data_U = mapslices(r -> invperm(sortperm(r, rev=false)), data; dims = 1) ./ (size(data, 1) + 1) # data transformed to (pseudo)uniform(0,1)
sum_data_exc = readdlm("./application/data/gustData_sum_exceed_u95_noHead.csv", ',', Float64)
## locate sum exceedances
sum_data = reshape(sum(data, dims = 2), size(data, 1))
sum_data_exc_inds = [i for i in eachindex(sum_data) if sum_data[i] .> quantile(sum_data, 0.95)]

## transform data to pseudo uniform scale
# c = 2*quadgk(x -> df(x, βhat, dimension), 0, Inf; atol = 2e-3)[1] # constant

### transform sum exceedance data from u(0,1) to mepd scale
# sum_data_exc = zeros(size(vec(data_U[sum_data_exc_inds,:]), 1))
# using Dates; Dates.now()
# @time Threads.@threads for i in 1:size(vec(data_U[sum_data_exc_inds,:]), 1)
#   println(i)
#   sum_data_exc[i] = qF(vec(data_U[sum_data_exc_inds,:])[i], βhat, size(coord, 1), 1/c; intval = 40) 
# end
# sum_data_exc = reshape(sum_data_exc, size(data_U[sum_data_exc_inds,:]))
# CSV.write("gustData_sum_exceed_u95.csv", Tables.table(sum_data_exc))

## calculate correlations matrix based on parameter estimates
# anisotropic distances
param_ests = readdlm("./application/data/application_results_anisotropic.csv")[1:5]
θ, βhat = param_ests[1:4], param_ests[5]
angle_mat = reshape([cos(θ[3]), sin(θ[3]), -sin(θ[3]), cos(θ[3])], 2, 2)
Ω = angle_mat * reshape([1, 0, 0, θ[4]^(-2)], 2, 2) * transpose(angle_mat)
dists_mahal = zeros(size(dist, 2))
for i in eachindex(dist[1,:])
    dists_mahal[i] = transpose(dist[:,i]) * inv(Ω) * dist[:,i]
end
dists = reshape(dists_mahal, size(data, 2), size(data, 2))
### correlation matrix
cor_mat = cor_fun(dists, θ)
sum_var = sum(cor_mat)
#

############# copy-pasted from mepd.jl to be able to easily manipulate the code
using Distributions, LinearAlgebra, SpecialFunctions, PDMats, Random
import Base: rand, convert, length
import Distributions: pdf, logpdf, @check_args, params, sqmahal

abstract type AbstractMvEpd <: ContinuousMultivariateDistribution end

struct GenericMvEpd{T<:Real, Cov<:AbstractPDMat, Mean<:AbstractVector} <: AbstractMvEpd
    p::T
    dim::Int
    μ::Mean
    Σ::Cov

    function GenericMvEpd{T, Cov, Mean}(p::T, dim::Int, μ::Mean, Σ::AbstractPDMat{T}) where {T, Cov, Mean}
        p > zero(p) || error("p must be positive")
        new{T, Cov, Mean}(p, dim, μ, Σ)
    end
end

function GenericMvEpd(p::T, μ::Mean, Σ::Cov) where {Cov<:AbstractPDMat, Mean<:AbstractVector,T<:Real}
    d = length(μ)
    dim(Σ) == d || throw(DimensionMismatch("The dimensions of μ and Σ are inconsistent"))
    R = Base.promote_eltype(T, μ, Σ)
    S = convert(AbstractArray{R}, Σ)
    m = convert(AbstractArray{R}, μ)
    GenericMvEpd{R, typeof(S), typeof(m)}(R(p), d, m, S)
end

function GenericMvEpd(p::Real, Σ::AbstractPDMat)
    R = Base.promote_eltype(p, Σ)
    GenericMvEpd(p, zeros(R, dim(Σ)), Σ)
end

function convert(::Type{GenericMvEpd{T}}, d::GenericMvEpd) where T <:Real
    S = convert(AbstractArray{T}, d.Σ)
    m = convert(AbstractArray{T}, d.μ)
    GenericMvEpd{T, typeof(S), typeof(m)}(T(d.p), d.dim, m, S)
end
Base.convert(::Type{GenericMvEpd{T}}, d::GenericMvEpd{T}) where {T<:Real} = d

function convert(::Type{GenericMvEpd{T}}, p, dim, μ::AbstractVector, Σ::AbstractPDMat) where T<:Real
    S = convert(AbstractArray{T}, Σ)
    m = convert(AbstractArray{T}, μ)
    GenericMvEpd{T, typeof(S), typeof(m)}(T(p), dim, m, S)
end

MvEpd(p::Real, μ::Vector{<:Real}, Σ::PDMat) = GenericMvEpd(p, μ, Σ)
MvEpd(p::Real, Σ::PDMat) = GenericMvEpd(p, Σ)
MvEpd(p::Real, μ::Vector{<:Real}, Σ::Matrix{<:Real}) = GenericMvEpd(p, μ, PDMat(Σ))
MvEpd(p::Real, Σ::Matrix{<:Real}) = GenericMvEpd(p, PDMat(Σ))

length(d::GenericMvEpd) = d.dim
params(d::GenericMvEpd) = (d.p, d.dim, d.μ, d.Σ)
sqmahal(d::GenericMvEpd, x::AbstractVector{<:Real}) = invquad(d.Σ, x - d.μ)

"""function mvepd_const(d::AbstractMvEpd)
    H = convert(eltype(d), pi^(-d.dim/2))
    H * d.dim*gamma(d.dim/2) / (gamma(1+d.dim/(2*d.p)) * 2^(1+d.dim/(2*d.p)))
end"""

function mvepd_const(d::AbstractMvEpd)
    H = convert(eltype(d), pi^(-d.dim/2))
    log(H) + log(d.dim) + loggamma(d.dim/2) - loggamma(1+d.dim/(2*d.p)) - (1+d.dim/(2*d.p))*log(2)
end

function logpdf(d::AbstractMvEpd, x::AbstractVector{T}) where T<:Real
    k = mvepd_const(d)
    mvepd_const(d) -0.5 * logdet(d.Σ) -0.5*sqmahal(d, x)^d.p
end

pdf(d::AbstractMvEpd, x::AbstractVector{<:Real}) = exp(logpdf(d, x))

function runifsphere(d::Int)
    mvnorm = rand(Normal(), d)
    mvnorm ./ sqrt.(sum(mvnorm.^2))
end

function repd(n::Int, d::GenericMvEpd)
    p, dim, μ, Σ = params(d)
    Σ = sqrt(Σ)
    res = zeros(n,dim)
    for i ∈ 1:n
        R = rand(Gamma(2,12/(2*p))).^(1/(2*p)) # changed dimension to 12
        res[i,:] = μ + R*Σ*runifsphere(dim)
    end
    res
end

# simulate MEPD data
d = MvEpd(βhat, reshape([sum_var],1,1))
sim_data = sum(repd(10^6, d), dims = 2)

# plot quantiles
quants = range(0.95, 0.9999, length = size(sum_data_exc, 1))
plot(sort(sum(sum_data_exc, dims = 2), dims = 1), quantile(sim_data, quants); seriestype=:scatter)






### gaussian model
# anisotropic
θ = readdlm("./application/data/application_results_gauss_anisotropic.csv")[1:4]
angle_mat = reshape([cos(θ[3]), sin(θ[3]), -sin(θ[3]), cos(θ[3])], 2, 2)
Ω = angle_mat * reshape([1, 0, 0, θ[4]^(-2)], 2, 2) * transpose(angle_mat)
dists_mahal = zeros(size(dist, 2))
for i in eachindex(dist[1,:])
    dists_mahal[i] = transpose(dist[:,i]) * inv(Ω) * dist[:,i]
end
dists = reshape(dists_mahal, size(data, 2), size(data, 2))
### correlation matrix
cor_mat_gauss = cor_fun(dists, θ)
#

data_gauss = quantile(Normal(), data_U)
sum_data_gauss = sum(data_gauss, dims = 2)
sum_var_gauss = sum(cor_mat_gauss)
gauss = MvNormal(reshape([sum_var_gauss], 1, 1))
quants = range(0.95, 0.9999, length = size(sum_data_exc, 1))
gauss_model_quants = quantile(rand(gauss, 10^6), quants)
gauss_sum_exc = quantile(sum_data_gauss, quants)

plot(gauss_sum_exc, gauss_model_quants; seriestype=:scatter)


