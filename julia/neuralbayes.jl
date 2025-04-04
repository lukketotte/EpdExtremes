using Flux, NeuralEstimators, Folds, Distributions, LinearAlgebra

include("./Distributions/mepd.jl")
using .MultivariateEpd


MvEpd(p::Real, μ::Vector{<:Real}, Σ::Matrix{<:Real})

d = 2

struct Parameters{T} <: ParameterConfigurations
	θ::Matrix{T}
	L
end

## Bivariate example, diagonal Σ
function sample(K::Integer)
	# Sample parameters from the prior
    p = rand(Gamma(1,2), K)
    σ1 = rand(InverseGamma(3, 1), K)
    σ2 = rand(InverseGamma(3, 1), K)
	μ1 = randn(K)
    μ2 = randn(K)
    return vcat(p', μ1', μ2', σ1', σ2')
end

function simulate(θ, m)
    Folds.map(eachcol(θ)) do x
        reshape(repd(m, MvEpd(x[1], [x[2], x[3]], [x[4] 0 ; 0 x[5]])), 2, m)
    end
end


n = 2    # dimension of each data replicate (bivariate)
d = 5    # dimension of the parameter vector θ
w = 128  # width of each hidden layer 

# Final layer has output dimension d and enforces parameter constraints
final_layer = Parallel(
    vcat,
    Dense(w, 1, softplus),     # p ∈ ℝ+
    Dense(w, 2, identity),     # μ ∈ ℝ
    Dense(w, 2, softplus)      # σ > 0
)

typeof(final_layer[1])


# Inner and outer networks
ψ = Chain(Dense(n, w, relu), Dense(w, d, relu))    
ϕ = Chain(Dense(d, w, relu), final_layer)          

# Combine into a DeepSet
network = DeepSet(ψ, ϕ)
estimator = PointEstimator(network)

m = 100
estimator = train(estimator, sample, simulate, m = m)

θ_test = sample(1000)
Z_test = simulate(θ_test, m)
assessment = assess(estimator, θ_test, Z_test, probs = [0.025, 0.975])

θ_test[:,1]
Z_test[1]