using LinearAlgebra, SpecialFunctions,  QuadGK, Cubature, Distributions, Plots, KernelDensity
include("./Distributions/alphaStable.jl")
include("./Distributions/mepd.jl")
using .AlphaStableDistribution
using .MultivariateEpd

# generate from multviaraite EPD example
repd(10, MvEpd(0.4, diagm([1.1, 0.2, 0.9])))

# Conditional MEPD
## Normal scale mixture
# can sample using the procedure of Gomez et al, where the density function is not evaluated
function rmix(n::Int, x::AbstractVector{<:Real}, Σ::AbstractMatrix{<:Real}, p::Real)
    burn, thin = 2000,5
    v = zeros(burn + n*thin)
    v[1] = rand(Uniform())
    γ = 2^(1-1/p)*cos(π*p/2)^(1/p)
    δ = γ *tan(π*p/2)
    generator = AlphaStable(p, 1., γ, δ)
    for i in 2:length(v)
        w = 2^(1/p-1)*v[i-1]^(-2)
        y = rand(generator)
        u = rand(Uniform())
        λ = 0.5 * x' * Σ * x
        if u < exp(λ * (w - y))
            v[i] = (2^(1-1/p) * y)^(-1/2)
        else
            v[i] = v[i-1]
        end
    end
    ids = (((burn+1):(burn+n*thin)) .% thin) .=== 0
    v[(burn+1):(burn+n*thin)][ids]
end

v = rmix(1000, [0 , 0], diagm([1, 1])^(-1), 0.99)
v = rmix(1, [4.913907], diagm([1]), 0.9)
mean(v)
plot(v)

## Radial Representation
g(x::Float64, p::Real, d::Int) =exp(-x^p/2) * d*gamma(d/2)/(π^(d/2) * gamma(1+d/(2*p)) * 2^(1+d/(2*p)))
c₀(d::Int, c₁::Real, p::Real) = 2*π^(d/2)/gamma(d/2)*quadgk(r -> g(r^2 + c₁, p, d)*r^(d-1), 0, Inf)[1]
function dr(x::Real, p::Real, d::Int, c₁::Real)
    k = c₀(d, c₁, p)
    2*π^(d/2)/gamma(d/2)*x^(d-1)*g(x^2 + c₁, p, d)/k
end

function rr(n::Int, dist::ContinuousDistribution, M::Real, p::Real, d::Int, c₁::Real)
    res = zeros(n)
    for i in 1:n
        keepSamp = true
        while keepSamp
            y = rand(dist)
            if dr(y, p, d, c₁)/(M*pdf(dist, y)) > rand(Uniform())
                res[i] = y
                keepSamp = false
            end
        end
    end
    res
end

function runifsphere(n::Int, d::Int)
    d >= 2 || throw(DomainError("dim must be >= 2"))
    mvnorm = reshape(rand(Normal(), n*d), n, d)
    rownorms = sqrt.(sum(mvnorm.^2, dims = 2))
    broadcast(/, mvnorm, rownorms)
end

function rcondmep(n::Int, μ::AbstractVector{<:Real}, x::AbstractVector{<:Real}, Σ::AbstractMatrix{<:Real}, p::Real)
    d = length(μ)
    d₂ = d-length(x)
    μ₁,μ₂ = μ[1:d₂], μ[(d₂+1):d]
    Σ₁, Σ₂, Σ₂₁ = Σ[1:d₂, 1:d₂]^(-1), Σ[(d₂+1):d, 1:d₂], Σ[(d₂+1):d, (d₂+1):d]
    σ = Σ₂ - Σ₂₁ * Σ₁ * Σ₂₁'
    l = Σ₂₁ * Σ₁ * x

    repeat(l,1,n)' + broadcast(*, rr(n, Gamma(3.2, 1.2), 10, p, d₂, x'*Σ₁*x), runifsphere(n, d₂))
end






rcondmep(n, μ, x, Σ, 1/4)

x = range(0.001, 20, length = 1000)
plot(x, dr.(x, 0.5, 2, 1))
plot!(x, pdf.(Gamma(3.2, 1.2), x))

rr(1000, Gamma(3.2, 1.2), 10, 0.5, 2, 1.)

k = kde(rr(5000, Gamma(3.2, 1.2), 10, 0.5, 2, 1.))

plot(x, pdf(k, x))
plot!(x, dr.(x, 0.5, 2, 1))
