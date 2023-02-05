using Distributed
@everywhere using JLD2, SpecialFunctions, LinearAlgebra, QuadGK, Roots, Distributions, StatsFuns, MvNormalCDF, Random, InvertedIndices
@everywhere include("../FFT.jl")
@everywhere using .MepdCopula

@everywhere u_95, u_9, u_8, u_7, u_6, u_5, u_4 = 1.8, 2., 2.9, 8., 9., 20., 65.;
@everywhere shapes = [0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
@everywhere dims = [1,2,4,6,8,10,15]

@sync @distributed for i in eachindex(dims)
  dims[i] |> x -> println("d = $x")
  d = Array{Dict{String, Any}}([])
  for p in shapes
    println("p = $p")
    if p == 0.95
      x = range(0.01, u_95, length = 2000);
      y = pF.(x, 0.95, dims[i]);
      y2 = pF.(x, 0.99, dims[i]);
    elseif p == 0.9
      x = range(0.01, u_9, length = 3000);
      y = pF.(x, 0.9, dims[i]);
      y2 = pF.(x, 0.95, dims[i]);
    elseif p == 0.8
      x = range(0.01, u_8, length = 3000);
      y = pF.(x, 0.8, dims[i]);
      y2 = pF.(x, 0.9, dims[i]);
    elseif p == 0.7
      x = range(0.01, u_7, length = 3000);
      y = pF.(x, 0.7, dims[i]);
      y2 = pF.(x, 0.8, dims[i]);
    elseif p == 0.6
      x = range(0.01, u_6, length = 3500);
      y = pF.(x, 0.6, dims[i]);
      y2 = pF.(x, 0.7, dims[i]);
    elseif p == 0.5
      x = range(0.01, u_5, length = 4500);
      y = pF.(x, 0.5, dims[i]);
      y2 = pF.(x, 0.6, dims[i]);
    else
      x = range(0.01, u_4, length = 5500);
      y = pF.(x, 0.4, dims[i]);
      y2 = pF.(x, 0.5, dims[i]);
    end
    append!(d, [
      Dict(
        "p" => p,
        "prob" => [0.75, 0.5, 0.25, 0.],
        "interval" => [
          (x[findall(y2 .>= 0.74)[1]], x[findall(y .> 0.74)[end]]),
          (x[findall(y2 .>= 0.49)[1]], x[findall(y .<= 0.76)[end]]),
          (x[findall(y2 .>= 0.24)[1]], x[findall(y .<= 0.51)[end]]),
          (x[findall(y2 .>= 0.0)[1]], x[findall(y .<= 0.26)[end]])]
      )
    ])
  end
  dims[i] |> x -> jldsave((@__DIR__) * "/d$x.jld2"; d)
end


d1 = load_object((@__DIR__) * "d1.jld2");
d2 = load_object((@__DIR__) * "d2.jld2");
d4 = load_object((@__DIR__) * "d4.jld2");
d6 = load_object((@__DIR__) * "d6.jld2");
d8 = load_object((@__DIR__) * "d8.jld2");
d10 = load_object((@__DIR__) * "d10.jld2");
d15 = load_object((@__DIR__) * "d15.jld2");

[d1, d2, d4, d6, d8, d10, d15] |> d -> jldsave((@__DIR__) * "/qFinterval.jld2"; d);