using LinearAlgebra, Distributions, Plots, CSV, DataFrames
using Random
function spatialGrid(nloc::Int)
    s = range(0,1,length=Integer(√nloc))
    s = vec(collect(Iterators.product(s,s)))
    locs = zeros(nloc, 2)
    for i in 1:nloc
        locs[i,:] .= s[i]
    end
    locs
end

ρ(s₁::AbstractVector{<:Real}, s₂::AbstractVector{<:Real}, λ::Real = 0.1) = exp(-sqrt((s₁-s₂)'*(s₁-s₂))/λ)

function buildCovMat(grid::Array{<:Real, 2}, λ::Real = 0.1)
    n = size(grid,1)
    covMat = ones(n, n)
    for i ∈ 1:(n-1)
        for j ∈ ((i+1):n)
            covMat[i,j] = ρ(grid[i,:], grid[j,:], λ)
            covMat[j,i] = covMat[i,j]
        end
    end
    covMat
end


## Recreate the quantile plot?
# position closest to [0.5, 0.5]

locs = spatialGrid(100^2) # Q(0.9999) ≈ 4.913907
x₁ = 4.913907
mpos = mapslices(x -> sum((x-[0.5,0.5]).^2), locs, dims = 2) |> x -> findmin(x)[2][1]

# shuffle the grid
locs = vcat(locs[1:(mpos-1),:], locs[(mpos+1):size(locs, 1),:], locs[mpos,:]')
locs = vcat(locs, [0.5,0.5]')

Σ = buildCovMat(locs, 0.1)
k = size(Σ,1)
σ = Σ[1:(k-1), 1:(k-1)] - Σ[1:(k-1), k] * Σ[k,1:(k-1)]'
μ = Σ[1:(k-1), k] * x₁

v = rmix(1, [4.913907], diagm([1]), 0.75)[1]
v = 1

Random.seed!(123)
res = v.*rand(MvNormal(μ./v, σ))
#surface(locs[1:(k-1),1], locs[1:(k-1), 2], res)
#reshape(res[1:100], 10, 10)
#heatmap(reshape(res[1:100], 10, 10))

#DataFrame(s1 = locs[1:(k-1),1], s2 = locs[1:(k-1),2], y = res) |> x -> CSV.write("../R/p75.csv", x)
