using Knet, Test, Base.Iterators, Printf, LinearAlgebra, Random, CUDA, IterTools
Knet.atype() = KnetArray{Float32}
include("tree.jl")
import Pkg; Pkg.add("PyCall")
using PyCall
@pyimport pickle
using JLD2

import Pkg 
Pkg.add("PyCall")

@pyimport pickle

function mypickle(filename, obj)
    out = open(filename,"w")
    pickle.dump(obj, out)
    close(out)
 end

function myunpickle(filename)
    r = nothing
    @pywith pybuiltin("open")(filename,"rb") as f begin
        r = pickle.load(f)
    end
    return r
end

mutable struct SymbolsManager
    symbol2idx#::Dict{String,Int}
    idx2symbol#::Dict{Int,String}
    vocab_size#::Int
    whether_add_special_tags#::Bool
end

function SymbolsManager(whether_add_special_tags)
    symbol2idx = Dict()
    idx2symbol = Dict()
    vocab_size = 0
    #vocab_size = 1
    symbols_manager = SymbolsManager(symbol2idx, idx2symbol, vocab_size, whether_add_special_tags)
    if whether_add_special_tags
        add_symbol(symbols_manager,"<S>")
        add_symbol(symbols_manager, "<E>")
        add_symbol(symbols_manager, "<U>")
        # non-terminal symbol = 4
        add_symbol(symbols_manager, "<N>")
        
    end
    #return SymbolsManager(symbol2idx, idx2symbol, vocab_size, whether_add_special_tags)
    return symbols_manager
end

function add_symbol(m::SymbolsManager, s)
    if s ∉ keys(m.symbol2idx)
        m.vocab_size = m.vocab_size + 1
        #s_man.symbol2idx[s] = s_man.vocab_size
        merge!(m.symbol2idx, Dict(s => m.vocab_size))
        #s_man.idx2symbol[s_man.vocab_size] = s
        merge!(m.idx2symbol, Dict(m.vocab_size=>s))
        
    end
    return m.symbol2idx[s]
end

function get_symbol_idx(s_man::SymbolsManager, s)
    if s ∉ keys(s_man.symbol2idx)
        if s_man.whether_add_special_tags
            return s_man.symbol2idx["<U>"]
        else
            println("this should never be reached (always add <U>")
            return 0
        end
    end
    return s_man.symbol2idx[s]
end

function get_idx_symbol(s_man::SymbolsManager, idx)
    #println(idx)
    if idx ∉ keys(s_man.idx2symbol)
        return "<U>"
    end
    return s_man.idx2symbol[idx]
end

function init_from_file(s_man::SymbolsManager, fn, min_freq, max_vocab_size)
    #print("loading vocabulary file: {}\n".format(fn))
    #f = open(fn, "r")
    open(fn) do file
        for line in eachline(file)
	    #println("LINELINELINELINELINE")
	    #println(line)
            l_list = split(strip(line), "\t")
            #c = Int(l_list[2])
	    c = parse.(Int,l_list[2] )
	    #c = l_list[1]
            if c >= min_freq
                add_symbol(s_man, l_list[1])
            end
            if s_man.vocab_size >= max_vocab_size
                break
            end
        end
    end
end



function get_symbol_idx_for_list(s_man::SymbolsManager,l)
    r = []
    for i in 1:(length(l))
        push!(r, get_symbol_idx(s_man, l[i]))
    end
    return r
end


struct MinibatchLoader
    enc_batch_list
    enc_len_batch_list
    dec_batch_list
    num_batch
    
end

function MinibatchLoader(opt, mode, using_gpu)
    path_data = opt["data_dir"] * "/" * string(mode) * ".jld2"
    @load  ( opt["data_dir"] * "/" * string(mode) * ".jld2") data
    
    #println("UTİL UTİL UTİL")
    #println(size(data))
    #data = shuffle(data)
    if length(data) % opt["batch_size"] != 0
        n = length(data)
        for i in 1:(length(data)%opt["batch_size"]) 
            insert!(data, n-i+1, data[n-i+1])
        end
    end
    enc_batch_list = []
    enc_len_batch_list = []
    dec_batch_list = []
    p = 0
    while p + opt["batch_size"] <= length(data)
        max_len = length(data[p + opt["batch_size"] ][1])
        #max_len = maximum([length(data[p + x ][1]) for x in 1:opt["batch_size"]])
	m_text = zeros(Int64, (opt["batch_size"], max_len + 2))
        enc_len_list = []
        m_text[:,1] .= 1
        for i in 1:(opt["batch_size"] )
            w_list = data[p + i][1]
            for j in range(1, stop=length(w_list))
#		println(m_text)
#		println(size(m_text))
#		println(w_list)
#		println(m_text[i,j+1])
#		println(i)
#		println(j)
#		println(length(w_list) - j + 1)
                m_text[i,j+1] = w_list[length(w_list) - j + 1]
            end
            for j in range(length(w_list)+2,stop=max_len+2)
                m_text[i, j] = 2
            end
            push!(enc_len_list, length(w_list)+2)
        end
        push!(enc_batch_list, m_text)
        push!(enc_len_batch_list, enc_len_list)
        tree_batch = []
        for i in range(1, stop=opt["batch_size"])
            push!(tree_batch, data[p+i][3])
        end
        push!(dec_batch_list, tree_batch)
        p += opt["batch_size"]
    end
    num_batch = length(enc_batch_list)
    enc_batch_list = shuffle(MersenneTwister(13), enc_batch_list)
    enc_len_batch_list = shuffle(MersenneTwister(13), enc_len_batch_list)
    dec_batch_list = shuffle(MersenneTwister(13), dec_batch_list)
    return MinibatchLoader(enc_batch_list, enc_len_batch_list, dec_batch_list, num_batch)
end

Base.IteratorSize(::Type{MinibatchLoader}) = Base.SizeUnknown()
Base.IteratorEltype(::Type{MinibatchLoader}) = Base.HasEltype()
#Base.eltype(::Type{MinibatchLoader}) = NTuple{3, Any}

function Base.iterate(mbl::MinibatchLoader, state=nothing)
    if (state == nothing) & (mbl.enc_batch_list != nothing)
	next_enc_batch, state_a = iterate(mbl.enc_batch_list)
	next_enc_len_batch, state_b = iterate(mbl.enc_len_batch_list)
	next_sec_batch, state_c = iterate(mbl.dec_batch_list)
    elseif (state > length(mbl.enc_batch_list))
	return nothing
    elseif (state <= length(mbl.enc_batch_list)) 
    	next_enc_batch, state_a = iterate(mbl.enc_batch_list, state)
    	next_enc_len_batch, state_b = iterate(mbl.enc_len_batch_list, state)
    	next_sec_batch, state_c = iterate(mbl.dec_batch_list, state)
    elseif (state == nothing) & (mbl.enc_batch_list == nothing)
	return nothing
    end
    if next_enc_batch == nothing 
	return nothing
    else
        return (next_enc_batch, next_enc_len_batch, next_sec_batch) , state_a 
    end
end
Base.IteratorSize(::Type{MinibatchLoader}) = Base.SizeUnknown()
Base.IteratorEltype(::Type{MinibatchLoader}) = Base.HasEltype()
#Base.eltype(::Type{MinibatchLoader}) = NTuple{3,Any}

function random_batch(mbl::MinibatchLoader)
    p = rand(1:mbl.num_batch,  1)
    
    return mbl.enc_batch_list[p], mbl.enc_len_batch_list[p][1], mbl.dec_batch_list[p][1]
end

function all_batch(mbl::MinibatchLoader)
    r = []
    for p in 1:(mbl.num_batch)
        push!(r, [mbl.enc_batch_list[p], mbl.enc_len_batch_list[p], mbl.dec_batch_list[p]])
    end
    return r
end


function convert_to_tree(r_list, i_left, i_right, form_manager)
    t = Tree()
    level = 0
    left = -1
    for i in range(i_left,stop=i_right)
        if r_list[i] ==  get_symbol_idx(form_manager,"(")
	    
            if level == 0
                left = i
            end
            level = level + 1
        elseif r_list[i] == get_symbol_idx(form_manager ,")")
	    
            level = level - 1
            if level == 0
                #if i == left + 1
                #    c = r_list[i]
                #else
                #    c = convert_to_tree(r_list, left + 1, i - 1, form_manager)
                #end
		#
                #add_child(t, c)
		c = convert_to_tree(r_list, left + 1, i - 1, form_manager)
		add_child(t, c)
            end
        elseif level == 0
            add_child(t, r_list[i])
            
        end
    end
    #println(typeof(t))
    return t
end
        
                    
function norm_tree(r_list, form_manager)
    #println("ererererererererer")
    #println(typeof(r_list))
    q = [convert_to_tree(r_list, 1, length(r_list), form_manager)]
    head = 1
    while head <= length(q)
        t = q[head]
	#println(q[head])
	#println(t.children)
        if (t.children[0] == get_symbol_idx(form_manager, "and")) || (t.children[0] == get_symbol_idx(form_manager, "or"))
            # sort the following subchildren
            #k = []
	    k = Dict()
            for i in range(2, stop=length(t.children))
                if t.children[i] isa Tree
		    #println("ttttttttttttttttttttttttt")
                    #push!(k, (string.(t.children[i]), i))
		    #push!(k, (to_string(t.children[i]), i))
		    k[to_string(t.children[i])] = i
                else
                    #k.append((str(t.children[i]), i))
                    #push!(k, (string.(t.children[i]), i))
		    k[string.(t.children[i])] = i
                end
            end
            #sorted_t_dict = []
	    sorted_t_dict = Dict()
            #k.sort(key=itemgetter(0))
            j = 2
            for keyd in k
		#println(keyd[1])
		#println(keyd[2])
                #push!(sorted_t_dict, t.children[keyd[2]])
		sorted_t_dict[j] = t.children[keyd[2]]
		j=j+1
            end
            for i in range(2, stop=length(t.children))
                t.children[i] = sorted_t_dict[i]
            end
        end
        # add children to q
	#println("zzzzzzzzzzzzzzzzzzzzzzz")
	#println(typeof(t.children))
        for i in range(1, stop=length(t.children))
	#for i in range(1, stop=t.num_children)
            if t.children[i] isa Tree
                
                push!(q, t.children[i])
            end
        end
        head = head + 1
	#println(head)
	#println(length(q))
    end
    return q[1]
end


function is_all_same(c1, c2)
    if length(c1) == length(c2)
        all_same = true
        for j in 1:(length(c1))
            if c1[j] != c2[j]
                all_same = false
                break
            end
        end
        return all_same
    else
        return false
    end
end

function compute_accuracy(candidate_list, reference_list)
    if length(candidate_list) != length(reference_list)
        #print("candidate list has length {}, reference list has length {}\n".format(len(candidate_list), len(reference_list)))
    end
    len_min = min(length(candidate_list), length(reference_list))
    c = 0
    for i in 1:(len_min)
        if is_all_same(candidate_list[i], reference_list[i])
            #print("above was all same")
            c = c+1
        else
            #println("***************")
        end
    end
    return c/float.(len_min)
end

function compute_tree_accuracy(candidate_list_, reference_list_, form_manager)
    candidate_list = []
    for i in 1:(length(candidate_list_))
        #print("candidate\n\n")
        #push!(candidate_list, to_list(norm_tree(candidate_list_[i], form_manager), form_manager))
        push!(candidate_list, to_list(norm_tree(candidate_list_[i], form_manager), form_manager))  
    end
    reference_list = []
    for i in 1:(length(reference_list_))
        #reference_list.append(norm_tree(reference_list_[i], form_manager).to_list(form_manager))
        push!(reference_list, to_list(norm_tree(reference_list_[i], form_manager), form_manager))
    end
    return compute_accuracy(candidate_list, reference_list)
end


