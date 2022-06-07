using Knet, Test, Base.Iterators, Printf, LinearAlgebra, Random, CUDA, IterTools
Knet.atype() = KnetArray{Float32}
import Pkg
Pkg.add("ArgParse")
using ArgParse
include("tree.jl")
include("util.jl")
using JLD2

function process_train_data(opt)
    word_manager = SymbolsManager(true)
    #println("1111111111111111111111111111111111111111111111111111")
    init_from_file(word_manager, "$(opt["data_dir"])/vocab.q.txt", opt["min_freq"], opt["max_vocab_size"])
    form_manager = SymbolsManager(true)
    #println("22222222222222222222222222222222222222222222222222222")
    init_from_file(form_manager,"$(opt["data_dir"])/vocab.f.txt", 0, opt["max_vocab_size"])
    data = []
    open("$(opt["data_dir"])/$(opt["train"]).txt", "r") do file
        for line in eachline(file)
            l_list = split(line, "\t")
            w_list = get_symbol_idx_for_list(word_manager, split(strip(l_list[1]), " "))
            r_list = get_symbol_idx_for_list(form_manager, split(strip(l_list[2]), " "))
            #println(line)
            #println(w_list)
            #println(r_list)
            cur_tree = convert_to_tree(r_list, 1, length(r_list), form_manager)
            push!(data, (w_list, r_list, cur_tree))
        end
    end
    out_mapfile = string(opt["data_dir"]) * "/map.jld2"
    out_datafile = string(opt["data_dir"]) * "/train.jld2"
    managers = [word_manager, form_manager]
    #println("++++++++++++++++++++++++++++")
    #println(data[1:24])
    #data = shuffle(MersenneTwister(13), data)
    #println(data[1:24])
    @save out_mapfile managers
    @save out_datafile  data
end




function serialize_data(opt, name)
    data = []
    #managers = pkl.load( open("{}/map.jld2".format(opt["data_dir"]), "rb" ) )
    @load  (string(opt["data_dir"]) * "/map.jld2") managers
    word_manager, form_manager = managers
    #-------
    #println("$(opt["data_dir"])/$(opt[name]).txt")
    open("$(opt["data_dir"])/$(opt[name]).txt", "r") do file
        for line in eachline(file)
            l_list = split(line, "\t")
            w_list = get_symbol_idx_for_list(word_manager, split(strip(l_list[1]), " "))
            r_list = get_symbol_idx_for_list(form_manager, split(strip(l_list[2]), " "))
            cur_tree = convert_to_tree(r_list, 1, length(r_list), form_manager)
            #data.append((w_list, r_list, cur_tree))
            push!(data, (w_list, r_list, cur_tree))
        end
    end
    out_datafile = string(opt["data_dir"]) * "/$name.jld2"
    #data = shuffle(data)
    @save out_datafile  data
    #-------
end

s = ArgParseSettings()
@add_arg_table! s begin
    "--data_dir"
        arg_type = String
        help = "data dir"
        default = "../data/"
    "--train"
        help = "train dir"
        arg_type = String
        default = "train"
    "--test"
        help = "minimum word frequency"
        arg_type = String
        default = "test"
    "--dev"
        help = "dev dir"
        arg_type = String
        default = "dev"
    "--min_freq"
        help = "minimum word frequency"
        arg_type = Int
        default = 2
    "--max_vocab_size"
        help = "max vocab size"
        arg_type = Int
        default = 15000
    "--seed"
        help = "random number generator seed"
        arg_type = Int
        default = 123
end


parsed_args = parse_args(s)
Random.seed!(parsed_args["seed"])
process_train_data(parsed_args)
serialize_data(parsed_args, "test")
serialize_data(parsed_args, "dev")
