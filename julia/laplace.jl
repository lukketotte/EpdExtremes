using LinearAlgebra, SpecialFunctions, QuadGK, Distributions, Optim, Roots, RCall
include("./Distributions/mepd.jl"); using .MultivariateEpd
using Plots, StatsPlots

function h05(r::Real, n::Integer)
    C = 2^((3*n + 1)/2)*gamma((n+1)/2)
    r^n * exp(-r^2/8) / C
end

f(r::Real, x::Real, n::Integer) = pdf(Normal(0, r), x) * h05(r, n)
f(x::Real, n::Integer) = quadgk(r -> f(r, x, n), 0, Inf)[1]

F(r::Real, x::Real, n::Integer) = cdf(Normal(0, 1), x/r) * h05(r, n)
F(x::Real, n::Integer) = quadgk(r -> F(r, x, n), 0, Inf)[1]

qroot(x::Real, q::Real, n::Int) = F(x, n) - q
Q(q::Real, n::Int) = find_zero((x -> qroot(x, q, n)), quantile(Laplace(), q), Order16())

function loglik(u::AbstractVector{<:Real}, Σ)
    nom = similar(u)
    d = length(u)
    denom = 0
    for i ∈ 1:d
        nom[i] = Q(u[i], d)
        denom += log(f(nom[i], d))
    end
    logpdf(MvEpd(0.5, Σ), nom) - denom
end

function lik(U::AbstractMatrix{<:Real}, λ::Real, grid::AbstractMatrix{<:Real})
    Σ = buildCovMat(grid, λ)
    -sum([loglik(U[i,:], Σ) for i in 1:size(U, 1)])
end

function sampleRank(x::AbstractVector{<:Real})
    sorted = sort(x)
    for i ∈ 1:length(x)
        sorted[i] = findall(sorted[i] .== x) |> first
    end
    sorted / (length(x) + 1)
end

spatgrid = spatialGrid(3^2);
Σ = buildCovMat(spatgrid, 0.55);
#d = MvEpd(0.5, zeros(size(Σ, 1)), Σ);
#y = repd(100, d)
d = MvNormal(Σ)
n = 200
#y = reshape(rand(d, n), n, size(Σ, 1))
y = rand(d,n)'
U = rcopy(R"apply($y, 2, function(x) rank(x)/($n + 1))")
#U = hcat([sampleRank(y[:,i]) for i in 1:size(y, 2)]...)

λ = range(0.01, 1, length = 30)
pltλ = [lik(U, λ[i], spatgrid) for i ∈ 1:length(λ)]
plot(λ, pltλ)

λ[findmin(pltλ)[2]] |> println
