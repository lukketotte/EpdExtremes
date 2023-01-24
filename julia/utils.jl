module Utils

using LinearAlgebra

cor_fun = function (h::Matrix{Float64}, param::Vector{Float64})
    return exp.(-(h ./ exp(param[1])).^param[2])
end

cond_cor = function (param::Vector{Float64})
    return param[2] > 0 && param[2] < 2 && param[3] > 0 && param[3] < 0.95
end

dist_fun = function (coord_vec::Vector{Float64})
    coord_stack(j) = coord_vec
    return permutedims(vec(stack(coord_stack(j) for j = eachindex(coord_vec)) - permutedims(stack(coord_stack(j) for j = eachindex(coord_vec)))))
end

same_vec = function (vec1::Vector{Int64}, vec2::Vector{Int64})
    if length(vec1) != length(vec2)
        return false
    else
        return !any(vec1 .∉ [vec2])
    end
end

compute_I_nexc_nb_i = function (i::Integer, I_nexc::Vector{Vector{Int64}})
    res = 0
    for j in eachindex(I_nexc)
        res += same_vec(I_nexc[j], I_nexc[i])
    end
    return res
end

end