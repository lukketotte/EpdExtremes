module QFinterval

export getInterval

using JLD2 # for serializing data

intervals = load_object(joinpath(@__DIR__,"qFinterval.jld2"))

function getInterval(prob::Real, p::Real, d::Int; 
  intervals::Vector{Vector{Dict{String, Any}}} = intervals)::Tuple{Float64, Float64}
  if p >= 0.4
    if d <= 2
      return intervals[1][sortByShape(p)]["interval"][sortByProb(prob)]
    elseif d <= 4
      return intervals[2][sortByShape(p)]["interval"][sortByProb(prob)]
    elseif d <= 6
      return intervals[3][sortByShape(p)]["interval"][sortByProb(prob)]
    elseif d <= 8
      return intervals[4][sortByShape(p)]["interval"][sortByProb(prob)]
    elseif d <= 10
      return intervals[5][sortByShape(p)]["interval"][sortByProb(prob)]
    elseif d <= 15
      return intervals[5][sortByShape(p)]["interval"][sortByProb(prob)]
    else
      return (0.01, 100.)
    end
  else
    return (0.01, 100.)
  end
end

function sortByShape(p::Real)::Int64
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
  elseif p >= 0.4
    return 7
  else
    return 8
  end
end

function sortByProb(prob::Real)::Int64
  if prob > 0.75
    return 1
  elseif prob > 0.5
    return 2
  elseif prob >= 0.25
    return 3
  else
    return 4
  end
end

end