using Distributed
@everywhere using JLD2, SpecialFunctions, LinearAlgebra, QuadGK, Roots, Distributions, StatsFuns, MvNormalCDF, Random, InvertedIndices

@everywhere ζ(α::Real) = -tan(π*α/2)
@everywhere θ₀(α::Real) = 1/α * atan(tan(π*α/2))
@everywhere function V(α::Real, θ::Real)
  ϑ = θ₀(α)
  a = cos(α*ϑ)^(1/(α-1))
  b = (cos(θ)/sin(α*(ϑ + θ)))^(α/(α-1))
  c = cos(α*ϑ + (α-1)*θ) / cos(θ)
  return a*b*c 
end

@everywhere h(θ::Real, x::Real, α::Real) = (x-ζ(α))^(α/(α-1))*V(α,θ)*exp(-(x-ζ(α))^(α/(α-1))*V(α,θ))
@everywhere f(x::Real, α::Real) = α/(π*(x-ζ(α))*abs(α-1)) * quadgk(θ -> h(θ, x, α), -θ₀(α), α <= 0.95 ? π/2 : 1.565)[1]
@everywhere dstable(x::Real, α::Real, γ::Real) = f((x-γ * tan(π*α/2))/γ, α)/γ

@everywhere dF = function(x::Real, p::Real, d::Int)
  p > 0 && p < 1 || throw(DomainError(p,"must be on (0,1)")) 
  γ = 2^(1-1/p) * cos(π*p/2)^(1/p)
  C = 2^(1+d/2*(1-1/p)) * gamma(1+d/2) / gamma(1+d/(2*p))
  x > 0 ? C*x^(d-3)*dstable(x^(-2), p, γ) : 0
end

@everywhere pF = function(x::Real, p::Real, d::Int)
  quadgk(x -> dF(x,p,d), 0, x)[1]
end

@everywhere u_95, u_9, u_8, u_7, u_6, u_5, u_4 = 1.45, 1.52, 2.12, 3.44, 5.92, 15, 51;

# recreate d1 as forloop
@everywhere shapes = [0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
@everywhere dims = [1,2,4,6,8,10]

@sync @distributed for i in eachindex(dims)
  dims[i] |> x -> println("d = $x")
  d = Array{Dict{String, Any}}([])
  for p in shapes
    println("p = $p")
    if p == 0.95
      x = range(0.01, u_95, length = 1500);
      y = pF.(x, 0.95, dims[i]);
      y2 = pF.(x, 0.99, dims[i]);
    elseif p == 0.9
      x = range(0.01, u_9, length = 1500);
      y = pF.(x, 0.9, dims[i]);
      y2 = pF.(x, 0.95, dims[i]);
    elseif p == 0.8
      x = range(0.01, u_8, length = 2000);
      y = pF.(x, 0.8, dims[i]);
      y2 = pF.(x, 0.9, dims[i]);
    elseif p == 0.7
      x = range(0.01, u_7, length = 2000);
      y = pF.(x, 0.7, dims[i]);
      y2 = pF.(x, 0.8, dims[i]);
    elseif p == 0.6
      x = range(0.01, u_6, length = 2000);
      y = pF.(x, 0.6, dims[i]);
      y2 = pF.(x, 0.7, dims[i]);
    elseif p == 0.5
      x = range(0.01, u_5, length = 2500);
      y = pF.(x, 0.5, dims[i]);
      y2 = pF.(x, 0.6, dims[i]);
    else
      x = range(0.01, u_4, length = 3500);
      y = pF.(x, 0.4, dims[i]);
      y2 = pF.(x, 0.5, dims[i]);
    end
    append!(d, [
      Dict(
        "p" => p,
        "prob" => [0.75, 0.5, 0.25, 0.],
        "interval" => [
          (x[findall(y2 .> 0.74)[1]], x[findall(y .> 0.74)[end]]),
          (x[findall(y2 .> 0.49)[1]], x[findall(y .<= 0.76)[end]]),
          (x[findall(y2 .> 0.24)[1]], x[findall(y .<= 0.51)[end]]),
          (x[findall(y2 .> 0.0)[1]], x[findall(y .<= 0.26)[end]])]
      )
    ])
  end
  dims[i] |> x -> jldsave("./julia/d$x.jld"; d)
end


d1 = load_object("./julia/Constants/d1.jld2");
d2 = load_object("./julia/Constants/d2.jld2");
d4 = load_object("./julia/Constants/d4.jld2");
d6 = load_object("./julia/Constants/d6.jld2");
d10 = load_object("./julia/Constants/d10.jld2");
d = [d1, d2, d4, d6, d10];

jldsave("./julia/qFinterval.jld2"; d)