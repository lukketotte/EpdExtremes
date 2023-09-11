module Utils

export cor_fun, cond_cor, cond_cor_huser, dist_fun, same_vec, compute_I_nexc_nb_i, censoring, block_boot_sample

using LinearAlgebra 

function cor_fun(h::AbstractMatrix{<:Real}, param::AbstractVector{<:Real})
    return exp.(-(h ./ exp(param[1])).^param[2])
end

# function cor_fun_anisotropic(h::AbstractMatrix{<:Real}, param::AbstractVector{<:Real})
    
# end

# block bootstrap sample
function block_boot_sample(data::AbstractMatrix{<:Real}, block_length::Real)
    n = size(data, 1)
    boot_sample = 1
    while size(boot_sample, 1) < (size(data, 1) + 1)
        b = rand(1:(n-block_length), 1)[1]
        block_inds = b:(b+block_length-1)
        boot_sample = vcat(boot_sample, data[block_inds, :])
    end
    return boot_sample[2:(size(data, 1) + 1), :]
end

function cond_cor(param::AbstractVector{<:Real})
    # if length(param) == 2
    #     return param[2] > 0 && param[2] < 2
    # elseif length(param) == 3
    #     return param[2] > 0 && param[2] < 2 && param[3] >= 0.1 && param[3] <= 1.
    # else
    #     return param[2] > 0 && param[2] < 2 && param[3] >= 0.01 && param[4] > 0.2
    # end
    if length(param) == 2
        return param[2] > 0 && param[2] < 2 # for mepd with isotropic correlation function
    elseif length(param) == 4
        return param[2] > 0 && param[2] < 2 && param[3] >= -pi/2 && param[3] <= pi/2 && param[4] > 0 # for mepd with ANisotropic correlation function
    end
end

function cond_cor_huser(param::AbstractVector{<:Real})
    return param[2] > 0 && param[2] < 2 && param[3] >= 0.01 && param[4] > 0.2
end

dist_fun(coord_vec::AbstractVector{<:Real}) = permutedims(vec(permutedims(mapreduce(permutedims, vcat, vcat([coord_vec for i in eachindex(coord_vec)])))-mapreduce(permutedims, vcat, vcat([coord_vec for i in eachindex(coord_vec)]))))

function same_vec(vec1::AbstractVector{<:Int}, vec2::AbstractVector{<:Int})
    if length(vec1) != length(vec2)
        return false
    else
        return !any(vec1 .∉ [vec2])
    end
end

function compute_I_nexc_nb_i(i::Integer, I_nexc::Vector{Vector{Int64}})
    res = 0
    for j in eachindex(I_nexc)
        res += same_vec(I_nexc[j], I_nexc[i])
    end
    return res
end

function censoring(data::AbstractMatrix{<:Real}, thresh::Real)
    dat_cens = deepcopy(data)
    dat_cens[data.<thresh] .= thresh
    (n, D) = size(dat_cens)
    # 2023-01-12 skip checks for missing values for now
    n_sites = zeros(Int64, n)
    for i in 1:n
        n_sites[i] = sum(.!ismissing.(dat_cens[i, :]))
    end
    
    I_exc = Vector{Vector{Int64}}(undef, n)
    I_nexc = Vector{Vector{Int64}}(undef, n)
    I1 = repeat([false], n)
    I2 = repeat([false], n)
    I3 = repeat([false], n)
    for i in 1:n
      I_exc[i] = findall(dat_cens[i, :] .> thresh) # indices of exceedances
      I_nexc[i] = findall(dat_cens[i, :] .≤ thresh) # indices of non-exceedances
      n_exc = length(I_exc[i]) # number of exceedances in observation i
      if n_exc == D # no censoring
          I1[i] = 1
      end
      if n_exc > 0 && n_exc < D # partial censoring
          I2[i] = 1
      end
      if n_exc == 0 # full censoring
          I3[i] = 1
      end
    end
    
    # if there are any fully censored observations
    if any(I3 .== 1)
        # the following only really matters of there are missing values in data
        I_nexc = unique(I_nexc[I3]) # unique compositions of fully censored obs. will be a single element for a complete data set
        I_nexc_len = Vector{Int64}(undef, length(I_nexc)) # length of each unique composition of fully censored obs.
        for i in eachindex(I_nexc)
            I_nexc_len[i] = length(I_nexc[i])
        end
        I_nexc_nb = Vector{Int64}(undef, length(I_nexc_len))
        for i in eachindex(I_nexc) # check if any vectors in I_nexc are duplicates
            I_nexc_nb[i] = compute_I_nexc_nb_i(i, I_nexc)
        end
        # dims_c = sort(unique(n_sites[I3])) # dimensions of fully censored obs. only one number for a complete data set
        # nb_dims_c = Vector{Int64}(undef, length(dims_c))
        # for i in eachindex(dims_c)
        #     nb_dims_c[i] = sum(n_sites[I3][1] == dims_c[i]) # count number of occurrances of each dimension of fully censored obs.
        # end
    end
    # nfc <- sum(I3) # number of fully censored obs
    inds = I1 .|| I2 # indices of obs with no or partial censoring
    return inds, I_exc, I_nexc, I_nexc_nb, I_nexc_len, I1, I2
  end
  
  censoring(data::AbstractVector{<:Real}, thresh::Real) = censoring(reshape(data, (1,length(data))), thresh)
  

end