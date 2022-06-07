using Knet, Test, Base.Iterators, Printf, LinearAlgebra, Random,  IterTools, CUDA
Knet.atype() = KnetArray{Float32}
include("/scratch/users/bberkay14/workfolder/language_to_logic/seq2tree/seq2tree/atis/tree.jl")
include("main.jl")
include("/scratch/users/bberkay14/workfolder/language_to_logic/seq2tree/seq2tree/atis/util.jl")
#using CUDA
#Knet.atype() = KnetArray{Float32}
using JLD2

function convert_to_string(idx_list, form_manager)
    w_list = []
    for i in range(1, stop=length(idx_list))
        push!(w_list, get_idx_symbol(form_manager, Int(idx_list[i])))
    end
    return join(w_list, " ")
end


function do_generate( l::L2L,  enc_w_list, word_manager, form_manager, args,  checkpoint)
    # initialize the rnn state to all zeros
    
    #prev_c  = torch.zeros((1, encoder.hidden_size), requires_grad=False)
    #prev_h  = torch.zeros((1, encoder.hidden_size), requires_grad=False)
    #prev_c  = zeros(1, 2*checkpoint["rnn_size"])
    #prev_h  = zeros(1, 2*checkpoint["rnn_size"])
    prev_c  = 0
    prev_h  = 0

    # reversed order
    push!(enc_w_list, get_symbol_idx(word_manager,"<S>"))
    insert!(enc_w_list, 1, get_symbol_idx(word_manager, "<E>"))
    end_w = length(enc_w_list)
    #enc_outputs = torch.zeros((1, end_w, encoder.hidden_size), requires_grad=False)
    enc_outputs = zeros(Float32, 1, end_w, 2*checkpoint["rnn_size"])
    
    for i in range(end_w, 1, step=-1)
	#println("ECCCCCCCCCCEEEEEEE")
	#println(enc_w_list[i])
        cur_input = enc_w_list[i]
	#println(size(cur_input))
	#println(size(prev_c))

	
        prev_c, prev_h = encode(l, cur_input, prev_c, prev_h)
	println("OOOOOOOOOOOOOOOOOO")
	println(size(enc_outputs))
	println(size(enc_outputs[:, i, :]))
	println(size(prev_h))
	prev_h = reshape(prev_h, (size(prev_h,1), size(prev_h,2)))	
        enc_outputs[:, end_w - i + 1, :] .= prev_h
    end
    # decode
    queue_decode = []
    #queue_decode.append({"s": (prev_c, prev_h), "parent":0, "child_index":1, "t": Tree()})
    push!(queue_decode, Dict("s"=> (prev_c, prev_h), "parent"=>0, "child_index"=>1, "t"=> Tree()))    
   
     
    head = 1
    
    println(head <= length(queue_decode))
    println(head <= 100)
    while (head <= length(queue_decode)) & ( head <= 100)
	println("OOOOOOOOOOOOOOOO2222222222222222222222222222222")
	println(head)
	println(queue_decode[head])
        s = queue_decode[head]["s"]
        parent_h = s[2]
        t = queue_decode[head]["t"]
        if head == 1
            prev_word = get_symbol_idx(form_manager, "<S>")
            
        else
	    println("WWWWWWWWWWW")
            prev_word = get_symbol_idx(form_manager, "(")
        end

        i_child = 1
        while true
            curr_c, curr_h = decode(l, prev_word,  parent_h)
	    #println("OOOOOOOOOOOOOOOOOOOOOOOO")
	    #println(size(enc_outputs))
	    #println("OOOOOOOOOOOOOOOO2222222222222222222222222222222")
	    #println(size(curr_h))
            prediction = attention_decode(l, enc_outputs, curr_h)
            
            s = (curr_c, curr_h)
	    println("PRRRRRRRRRRRRRRRRRRRR")
	    #println(length(queue_decode))
	    println(size(prediction))
	    println(typeof(maximum(prediction) ))
	    println(argmax(prediction))
	    #findfirst(x -> x == maximum(prediction), prediction)
	    #println(findmax(prediction, dims=1)[2][1][1])
            _prev_word = argmax(prediction)[1]
            
            prev_word = _prev_word
	    #println(Int.(prev_word))
	    #println(get_symbol_idx(form_manager, "<E>"))
	    println("NNNNNNNNNNNNNNNN")
	    #println(get_symbol_idx(form_manager, "<N>"))
	    #println(t.num_children)
	    #println(checkpoint["dec_seq_length"])
    "--beam_size"
        help = "beam size"
        arg_type = Int
        default = 20
    "--display"
        help = "whether display on console"
        arg_type = Int
        default = 1
    "--data_dir"
        help = "data path"
        arg_type = String
        default = "../data/"
    "--input"
        help = "input data filename"
        arg_type = String
        default = "test.t7"
    "--output"
        help =  "input data filename"
        arg_type = String
        default = "/scratch/users/bberkay14/workfolder/language_to_logic/seq2tree/seq2tree/atis/attention/output/seq2seq_output.txt"
    "--model"
        help = "model checkpoint to use for sampling"
        arg_type = String
        default = "/scratch/users/bberkay14/workfolder/language_to_logic/seq2tree/seq2tree/atis/dump_attention/model_seq2seq.jld2"
    "--seed"
        help = "torch manual random number generator seed"
        arg_type = Int
        default = 123
end


args = parse_args(s)


# load the model checkpoint
checkpoint = Knet.load(args["model"])
best_model = Knet.load("/scratch/users/bberkay14/workfolder/language_to_logic/seq2tree/seq2tree/atis/attn-2651239791785874.jld2")
best_model =  Knet.load("/scratch/users/bberkay14/workfolder/language_to_logic/seq2tree/seq2tree/atis/attn-8193375629368503.jld2")
l_model = Knet.load("/scratch/users/bberkay14/workfolder/language_to_logic/seq2tree/seq2tree/atis/attention/checkpoint_dir/model_seq2seq.jld2")
#println(l_model["checkpoint"]["opt"])
#l_model = checkpoint["lang2logic_model"]



# initialize the vocabulary manager to display text
@load  (string(args["data_dir"]) * "/map.jld2") managers
word_manager, form_manager = managers
@load  (string(args["data_dir"]) * "/test.jld2") data
reference_list = []
candidate_list = []
open(args["output"], "w") do output
    for i in range(1, stop=length(data))
        x = data[i]
        reference = x[2]

	println("DATA3DATA3DATA3DATA3 DATA3")
	refere = x[3]
	println(reference)
	println(refere)
        candidate = do_generate(l_model["checkpoint"]["lang2logic_model"], x[1], word_manager, form_manager, args, l_model["checkpoint"]["opt"])
        #candidate = do_generate(best_model["model"], x[1], word_manager, form_manager, args, l_model["checkpoint"]["opt"])
	candidate = [Int(c) for c in candidate]

        ref_str = convert_to_string(reference, form_manager)
        cand_str = convert_to_string(candidate, form_manager)

        push!(reference_list, reference)
        push!(candidate_list, candidate)
        if args["display"] > 0
            println("results: ")
            println(ref_str)
            println(cand_str)
            
	end
        #output.write("{}\n".format(ref_str))
        write(output, string(ref_str) *"\n")                    
        #output.write("{}\n".format(cand_str))
        write(output, string(cand_str) *"\n") 
    end
    val_acc = compute_tree_accuracy(candidate_list, reference_list, form_manager)
    println("ACCURACY = " * string(val_acc) * "\n")
    #output.write("ACCURACY = {}\n".format(val_acc))
    write(output,"ACCURACY = " * string(val_acc) *"\n") 
    
end
using Knet, Test, Base.Iterators, Printf, LinearAlgebra, Random,  IterTools, CUDA
Knet.atype() = KnetArray{Float32}
include("/scratch/users/bberkay14/workfolder/language_to_logic/seq2tree/seq2tree/atis/tree.jl")
include("main.jl")
include("/scratch/users/bberkay14/workfolder/language_to_logic/seq2tree/seq2tree/atis/util.jl")
#using CUDA
#Knet.atype() = KnetArray{Float32}
using JLD2

function convert_to_string(idx_list, form_manager)
    w_list = []
    for i in range(1, stop=length(idx_list))
        push!(w_list, get_idx_symbol(form_manager, Int(idx_list[i])))
    end
    return join(w_list, " ")
end


function do_generate( l::L2L,  enc_w_list, word_manager, form_manager, args,  checkpoint)
# initialize the rnn state to all zeros
@load  (string(args["data_dir"]) * "/map.jld2") managers
word_manager, form_manager = managers
# load data
#data = pkl.load(open("{}/test.pkl".format(args.data_dir), "rb"))
@load  (string(args["data_dir"]) * "/test.jld2") data
reference_list = []
candidate_list = []
#with open(args.output, "w") as output:
open(args["output"], "w") do output
    # TODO change when running full -- this is to just reproduce the error
    #for i in range(30,50):
    #for i in range(278,280):
    for i in range(1, stop=length(data))
        #print("example {}\n".format(i))
        x = data[i]
	#println(data)
	#println(x)
	#println(x[1])
	#println(x[2])
	#println(x[3])
	#println(x[3][3])
        reference = x[2]
        candidate = do_generate(l_model["checkpoint"]["lang2logic_model"], x[1], word_manager, form_manager, args, l_model["checkpoint"]["opt"])
        candidate = [Int(c) for c in candidate]

	#println(candidate)
	#println(form_manager.idx2symbol[Int(c)] == "(")
        #num_left_paren = sum(1 for c in candidate if form_manager.idx2symbol[Int(c)] == "(")
            
        #num_right_paren = sum(1 for c in candidate if form_manager.idx2symbol[Int(c)]== ")")
        #diff = num_left_paren - num_right_paren
        #print(diff)
        #if diff > 0
        #    for i in range(1, stop= diff)
        #        push!(candidate, form_manager.symbol2idx[")"])
        #    end
        #elseif diff < 0
        #    candidate = candidate[1:diff]
        #end
	println(candidate)
	println(reference)
        ref_str = convert_to_string(reference, form_manager)
        cand_str = convert_to_string(candidate, form_manager)

        push!(reference_list, reference)
        push!(candidate_list, candidate)
        # print to console
        if args["display"] > 0
            println("results: ")
            println(ref_str)
            println(cand_str)
            
	end
        #output.write("{}\n".format(ref_str))
        write(output, string(ref_str) *"\n")                    
        #output.write("{}\n".format(cand_str))
        write(output, string(cand_str) *"\n") 
    end
    val_acc = compute_tree_accuracy(candidate_list, reference_list, form_manager)
    print("ACCURACY = " * string(val_acc) * "\n")
    #output.write("ACCURACY = {}\n".format(val_acc))
    write(output,"ACCURACY = " * string(val_acc) *"\n") 
    
end
