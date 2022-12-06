using JLD2

function getInterval(prob::Real, p::Real, d::Int)
  # [1, 2, 4, 6, 8, 10]
  if d === 1
    return (1, sortByShape(p), sortByProb(prob))
  elseif d === 2
    return (2, sortByShape(p), sortByProb(prob))
  elseif d < 4
    return (3, sortByShape(p), sortByProb(prob))
  elseif d < 6
    return (4, sortByShape(p), sortByProb(prob))
  elseif d < 10
    return (5, sortByShape(p), sortByProb(prob))
  else
    return (6, 1, 1)
  end
end

function sortByShape(p::Real)
  if p > 0.95
    return 1
  elseif p > 0.9
    return 2
  elseif p > 0.8
    return 3
  elseif p > 0.7
    return 4
  elseif p > 0.6
    return 5
  elseif p > 0.5
    return 6
  elseif p > 0.4
    return 7
  else
    return 8
  end
end


function sortByProb(prob::Real)
  if prob > 0.75
    return 1
  elseif prob > 0.5
    return 2
  elseif prob > 0.25
    return 3
  else
    return 4
  end
end

### d = 1
# p = 0.95
d1 = [ 
Dict(
  "p" => 0.95, 
  "prob" => [0.75, 0.5, 0.25, 0.],
  "interval" => [(1.05, 1.25), (1.041, 1.12), (1.01, 1.08), (0.01, 1.01)]),
Dict(
  "p" => 0.9, 
  "prob" => [0.75, 0.5, 0.25, 0.],
  "interval" => [(1.11, 1.4), (1.08, 1.21), (1.02, 1.13), (0.01, 1.01)]),
Dict(
  "p" => 0.8, 
  "prob" => [0.75, 0.5, 0.25, 0.],
  "interval" => [(1.22, 2.), (1.14, 1.44), (1.02, 1.27), (0.01, 1.05)]),
Dict(
  "p" => 0.7, 
  "prob" => [0.75, 0.5, 0.25, 0.],
  "interval" => [(1.44, 2.9), (1.26, 1.77), (1.04, 1.451), (0.01, 1.11)]),
Dict(
  "p" => 0.6, 
  "prob" => [0.75, 0.5, 0.25, 0.],
  "interval" => [(1.77, 4.5), (1.45, 2.29), (1.11, 1.78), (0.01, 1.12)]),
Dict(
  "p" => 0.5, 
  "prob" => [0.75, 0.5, 0.25, 0.],
  "interval" => [(2.3, 9), (1.77, 3.33), (1.25, 2.35), (0.01, 1.51)]),
Dict(
  "p" => [0.4, 0.4, 0.4, 0.4], 
  "prob" => [0.75, 0.5, 0.25, 0.],
  "interval" => [(3.34, 25), (2.36, 5.94), (1.52, 3.77), (0.01, 2.17)])
];

d1[1]["interval"]

d2 = [ 
Dict(
  "p" => 0.95,
  "prob" => [0.75, 0.5, 0.25, 0.],
  "interval" => [(1.05, 1.25), (1.041, 1.12), (1.01, 1.08), (0.01, 1.02)]),
Dict(
  "p" => 0.9,
  "prob" => [0.75, 0.5, 0.25, 0.],
  "interval" => [(1.11, 1.4), (1.08, 1.22), (1.02, 1.15), (0.01, 1.05)]),
Dict(
  "p" => 0.8,
  "prob" => [0.75, 0.5, 0.25, 0.],
  "interval" => [(1.22, 2.), (1.14, 1.49), (1.04, 1.33), (0.01, 1.14)]),
Dict(
  "p" => 0.7, 
  "prob" => [0.75, 0.5, 0.25, 0.],
  "interval" => [(1.44, 2.9), (1.26, 1.77), (1.04, 1.451), (0.01, 1.11)]),
Dict(
  "p" => 0.6,
  "prob" => [0.75, 0.5, 0.25, 0.],
  "interval" => [(1.77, 4.5), (1.45, 2.29), (1.11, 1.78), (0.01, 1.12)]),
Dict(
  "p" => 0.5, 
  "prob" => [0.75, 0.5, 0.25, 0.],
  "interval" => [(2.3, 9), (1.77, 3.33), (1.25, 2.35), (0.01, 1.51)]),
Dict(
  "p" => 0.4,
  "prob" => [0.75, 0.5, 0.25, 0.],
  "interval" => [(3.34, 25), (2.36, 5.94), (1.52, 3.77), (0.01, 2.17)])
];
###
using Plots

pF(3., 0.7, 2)

x = range(0.1, 3., length=500);
y = pF.(x, 0.7, 2);
y2 = pF.(x, 0.8, 2);

(x[findall(y2 .> 0.75)[1]], x[findall(y .> 0.75)[end]])
(x[findall(y2 .> 0.5)[1]], x[findall(y .<= 0.75)[end]])
(x[findall(y2 .> 0.25)[1]], x[findall(y .<= 0.5)[end]])
(x[findall(y2 .> 0.0)[1]], x[findall(y .<= 0.25)[end]])

plot(x, y, legend =:bottomright)
plot!(x, y2)

@time find_zero(x -> qF₁(x, 0.99, 0.6, 1), (1.77,4.5), xatol=1e-4)

pF(1.3, 0.95, 2)
pF(1.4, 0.9, 2)
pF(2.1, 0.8, 2)
pF(3.1, 0.7, 2)
pF(5.2, 0.6, 2)
pF(10, 0.5, 2)
pF(28, 0.4, 2)

# recreate d1 as forloop
shapes = [0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]

d2 = Array{Dict{String, Any}}([])

for p in shapes
  println(p)
  if p == 0.95
    x = range(0.1, 1.3, length = 500);
    y = pF.(x, 0.95, 1);
    y2 = pF.(x, 0.98, 1);
  elseif p == 0.9
    x = range(0.1, 1.4, length = 500);
    y = pF.(x, 0.9, 1);
    y2 = pF.(x, 0.95, 1);
  elseif p == 0.8
    x = range(0.1, 2.1, length = 700);
    y = pF.(x, 0.8, 1);
    y2 = pF.(x, 0.9, 1);
  elseif p == 0.7
    x = range(0.1, 3.1, length = 1000);
    y = pF.(x, 0.7, 1);
    y2 = pF.(x, 0.8, 1);
  elseif p == 0.6
    x = range(0.1, 5.2, length = 1200);
    y = pF.(x, 0.6, 1);
    y2 = pF.(x, 0.7, 1);
  elseif p == 0.5
    x = range(0.1, 10., length = 1500);
    y = pF.(x, 0.5, 1);
    y2 = pF.(x, 0.6, 1);
  else
    x = range(0.1, 28., length = 2000);
    y = pF.(x, 0.4, 1);
    y2 = pF.(x, 0.5, 1);
  end
  append!(d2, [
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

jldsave("./julia/d2.jld2"; d2)
#load_object("./julia/d1.jld2")
