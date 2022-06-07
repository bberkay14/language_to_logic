using Knet, Test, Base.Iterators, Printf, LinearAlgebra, Random,  IterTools, CUDA
include("/scratch/users/bberkay14/workfolder/language_to_logic/seq2tree/seq2tree/atis/tree.jl")
include("/scratch/users/bberkay14/workfolder/language_to_logic/seq2tree/seq2tree/atis/util.jl")
Knet.atype() = KnetArray{Float32}
using JLD2
using ArgParse
using Distributions
using Statistics



function uniform(Flo::Type{Float32}, dim)
    return rand(Flo,dim)  * (0.08 + 0.08) .- 0.08
end

function uniform_attn(Flo::Type{Float32}, dim)
    return rand(Flo,dim)  * (0.08 + 0.08) .- 0.08
end

struct Memory; w; end

struct Attention; wquery; wattn; scale; end

struct Embed; w; end


function Embed(opt, vocabsize::Int, embedsize::Int)
    return Embed(param(uniform(Float32, (embedsize,vocabsize))))
end

function (l::Embed)(x)
    x_copy = copy(x)
    for i in 1:size(x_copy, 1)
        if x_copy[i] == 0
            if typeof(x_copy[i]) == Int64
                #x_copy[i] = 171
                x_copy[i] = 2
            else
                #x_copy[i] = 171
                x_copy[i] = 2
            end
        end
    end
    x_copy = Int.(x_copy)
    return l.w[:,x_copy]
end
 


struct Linear; w; b; end


function Linear(opt, inputsize::Int, outputsize::Int)
    #return Linear(uniform(Float32, (outputsize,inputsize)),param0(outputsize))
    return Linear(param(uniform(Float32, (outputsize,inputsize))),param0(outputsize))
end

function (l::Linear)(x)
    return (mmul(l.w , x) .+ l.b)
end


struct Linear_attn; w; b; end


function Linear_attn(opt, inputsize::Int, outputsize::Int)
    return Linear_attn(param(uniform_attn(Float32, (outputsize,inputsize))),param0(outputsize))
end

function (l::Linear_attn)(x)
    return (mmul(l.w , x) .+ l.b)
end

struct L2L
    enc_lstm
    dec_lstm
    memory
    attention
    srcembed
    tgtembed
    dropout
    projection_1
    projection_2
    opt
    #projection
    word_manager
    form_manager
end


function L2L(src_input_size, tgt_input_size, opt, form_manager, word_manager)
    srcembed = Embed(opt, src_input_size  , opt["rnn_size"])
    tgtembed = Embed(opt, tgt_input_size , opt["rnn_size"])
    enc_lstm = RNN(opt["rnn_size"], opt["rnn_size"];dropout=0.3, winit=uniform, rnnType=:lstm)
    #input feeding deneme için aşağıdaki kodu aç
    dec_lstm = RNN(2*opt["rnn_size"], opt["rnn_size"]; dropout=0.3,  winit=uniform, rnnType=:lstm)
    #memory = Memory(param(uniform_attn(Float32, (opt["rnn_size"],2*opt["rnn_size"]))))
    memory = Memory(1)
    #attention = Attention(param(uniform(Float32, (opt["rnn_size"],opt["rnn_size"]))), param(uniform(Float32, (opt["rnn_size"],2*opt["rnn_size"]))), param(1) )
    #attention = Attention(param(uniform_attn(Float32, (opt["rnn_size"],opt["rnn_size"]))), param(uniform(Float32, (opt["rnn_size"],2*opt["rnn_size"]))), param(1) )
    attention = Attention(1, uniform(Float32, (opt["rnn_size"],2*opt["rnn_size"])), 1 )
    dropout = opt["dropoutrec"]
    projection_1 = Linear(opt, 2*opt["rnn_size"],  opt["rnn_size"] )
    #projection_1 = Linear(opt, opt["rnn_size"],  opt["rnn_size"] )
    projection_2 = Linear(opt, opt["rnn_size"],  tgt_input_size  )
    opt = opt
    form_manager = form_manager
    word_manager = word_manager
    return L2L(enc_lstm, dec_lstm, memory, attention, srcembed, tgtembed, dropout, projection_1, projection_2, opt, word_manager,form_manager)
end


function (m::Memory)(x)
    #keys = mmul(m.w, KnetArray(x))
    keys = mmul(m.w, x)
    return keys, x
end
# You can use the following helper function for scaling and linear transformations of 3-D tensors:
mmul(w,x) = (w == 1 ? x : w == 0 ? 0 : reshape(w * reshape(x,size(x,1),:), (:, size(x)[2:end]...)))


function encode(l::L2L, src)
    src_embedding = l.srcembed(src)
    src_embedding = dropout(src_embedding,l.dropout)
    if length(size(src_embedding)) == 2
        src_embedding = reshape(src_embedding, size(src_embedding,1), size(src_embedding,2), 1)
    end
    l.enc_lstm.h = value(l.enc_lstm.h)
    src_encoded = l.enc_lstm(src_embedding)
    l.enc_lstm.h = dropout(l.enc_lstm.h,l.dropout)
    l.enc_lstm.c = dropout(l.enc_lstm.c,l.dropout)
    return  l.enc_lstm.c , l.enc_lstm.h
end

function encode_sample(l::L2L, src)
    src_embedding = l.srcembed(src)
    src_embedding = reshape(src_embedding, size(src_embedding,1), 1, 1)
    src_encoded = l.enc_lstm(src_embedding)
    return  l.enc_lstm.c , l.enc_lstm.h
end



function (a::Attention)(src_encoded, decoder_output)
    keys, src_encoded = src_encoded
    query = mmul(a.wquery, decoder_output)
    #attention_scores = bmm( permutedims(query , (3,1,2) ) , KnetArray(permutedims( keys, (1,3,2))))
    attention_scores = bmm( permutedims(query , (3,1,2) ) , permutedims( keys, (1,3,2)))
    attention_scores = permutedims(attention_scores, (3,2,1))
    attention = Knet.softmax(attention_scores; dims=2)
    #context_vector = bmm(KnetArray(permutedims(src_encoded, (1,3,2))), permutedims(attention, (2,3,1)))
    context_vector = bmm(permutedims(src_encoded, (1,3,2)), permutedims(attention, (2,3,1)))
    concatenated_vectors = vcat(decoder_output , permutedims(context_vector, (1,3,2))) 
    return concatenated_vectors
end
     

function decode(l::L2L, tgt, prev_c, prev_h, parent_h)

    if typeof(tgt) == Int64
        tgt_embedding = l.tgtembed(tgt)
        tgt_embedding = reshape(tgt_embedding, (size(tgt_embedding,1), 1, 1))
        parent_h = permutedims(parent_h, (2,1))
        parent_h = reshape(parent_h, size(parent_h,1), size(parent_h,2), 1)
        input = vcat(tgt_embedding, parent_h)
        input = dropout(input, l.dropout)
    else
        input = l.tgtembed(tgt)
        input = reshape(input, size(input,1), size(input,2), 1)
        parent_h = permutedims(parent_h, (2,1)) 
        parent_h = reshape(parent_h, size(parent_h,1), size(parent_h,2), 1)
        input = vcat(input, parent_h)
        input = dropout(input, l.dropout)
    end
    prev_h, prev_c = permutedims(prev_h, (2,1)), permutedims(prev_c, (2,1))
    prev_h, prev_c = reshape(prev_h, size(prev_h,1), size(prev_h,2), 1) , reshape(prev_c, size(prev_c,1), size(prev_c,2), 1)
    l.dec_lstm.h, l.dec_lstm.c = value(prev_h), prev_c
    decoder_output = l.dec_lstm(input)
    l.dec_lstm.h = dropout(l.dec_lstm.h,l.dropout)
    l.dec_lstm.c = dropout(l.dec_lstm.c,l.dropout)
    return   l.dec_lstm.c, l.dec_lstm.h
end




function attention_decode(l::L2L, src_encoded, decoder_output)
    
    if length(size(decoder_output)) == 2
        decoder_output = reshape(decoder_output, size(decoder_output,1), size(decoder_output,2), 1)
    end
    decoder_output = permutedims(decoder_output, (2,1,3))
    keys, src_encoded  = l.memory(src_encoded)
    attention_vector = l.attention((keys, src_encoded), decoder_output)
    projected_mid  = tanh.(l.projection_1(attention_vector))
    #projected_mid  = tanh.(l.projection_1(decoder_output))
    projected_mid = dropout(projected_mid, l.dropout)
    projected = l.projection_2(projected_mid)
    projected_mid = dropout(projected_mid, l.dropout)
    return projected
end


function attention_decode_sample(l::L2L, src_encoded, decoder_output)
    if length(size(decoder_output)) == 2
        decoder_output = reshape(decoder_output, size(decoder_output,1), size(decoder_output,2), 1)
    end
    decoder_output = permutedims(decoder_output, (2,1,3))
    keys, src_encoded  = l.memory(src_encoded)
    attention_vector = l.attention((keys, src_encoded), decoder_output)
    projected_mid  = tanh.(l.projection_1(attention_vector))
    #projected_mid  = tanh.(l.projection_1(decoder_output))
    projected = l.projection_2(projected_mid)
    return projected
end



function (l::L2L)(enc_batch, enc_len_batch, dec_tree_batch )
    enc_max_len =  size(enc_batch,2)
    dec_s = Dict()
    for i in range(0,stop=l.opt["dec_seq_length"] )
        dec_s[i] = Dict()
        for j in range(0,stop=l.opt["dec_seq_length"])
            dec_s[i][j] = Dict()
        end
    end

    l.enc_lstm.c , l.enc_lstm.h = 0, 0
    enc_outputs_h = []
    enc_outputs_c = []
    for i in range(1, stop=enc_max_len )
        prev_c, prev_h = encode(l, enc_batch[:,i])
        push!(enc_outputs_h, prev_h)
        push!(enc_outputs_c, prev_c)
    end
    enc_outputs = reshape(hcat(enc_outputs_h...), (l.opt["rnn_size"], size(enc_batch,1),enc_max_len))    

    # tree decode
    queue_tree = Dict()
    for i in range(1, stop=l.opt["batch_size"])
        queue_tree[i] = []
        push!(queue_tree[i],Dict("tree"=>dec_tree_batch[i], "parent"=>0, "child_index"=>1))
    end
    loss = 0
    cur_index, max_index = 1,1
    dec_batch = Dict()
    dec_batch_copy = Dict()
    count1 = 0
    count2 = 0
    benim_sonu = []
    while (cur_index <= max_index)
        count1 = count1 + 1
        # build dec_batch for cur_index
        max_w_len = -1
        batch_w_list = []
        for i in range(1, stop=l.opt["batch_size"])
            w_list = []
            if (cur_index <= length(queue_tree[i]))
                t = queue_tree[i][cur_index ]["tree"]
                for ic in range(1, stop=t.num_children)
                    if t.children[ic] isa Tree
                        push!(w_list, 4)
                        push!(queue_tree[i], Dict("tree"=>t.children[ic], "parent"=>cur_index, "child_index"=>ic))
                    else
                        push!(w_list, t.children[ic])
                    end
                end
                if length(queue_tree[i]) > max_index
                    max_index = length(queue_tree[i])
                end
            end
            if length(w_list) > max_w_len
                max_w_len = length(w_list)
            end
            push!(batch_w_list, w_list)
        end
        dec_batch[cur_index] = (zeros(Int64, l.opt["batch_size"], max_w_len + 2))
        dec_batch_copy[cur_index] = (zeros(Int64, l.opt["batch_size"], max_w_len + 2))
        dec_batch[cur_index] = dec_batch[cur_index] .+ 2 


        for i in range(1, stop=l.opt["batch_size"])
            w_list = batch_w_list[i]
            if length(w_list) > 0
                for j in range(1, stop=length(w_list))
                    dec_batch[cur_index][i,j+1] = w_list[j]
                end
                # add <S>, <E>
                if cur_index == 1
                    dec_batch[cur_index][i,1] = 1
                else
                    dec_batch[cur_index][i,1] = get_symbol_idx(l.form_manager, "(")
                end    
                dec_batch[cur_index][i,length(w_list) + 2] = 2
            end
        end


        for i in range(1, stop=l.opt["batch_size"])
            w_list = batch_w_list[i]
            if length(w_list) > 0
                for j in range(1, stop=length(w_list))
                    dec_batch_copy[cur_index][i,j+1] = w_list[j]
                end
                # add <S>, <E>
                if cur_index == 1
                    dec_batch_copy[cur_index][i,1] = 1
                else
                    dec_batch_copy[cur_index][i,1] = get_symbol_idx(l.form_manager, "(")
                end
                dec_batch_copy[cur_index][i,length(w_list) + 2] = 2
            end
        end


        dec_s_a = []
        dec_s_b = []
        if cur_index == 1
            for i in range(1, stop=l.opt["batch_size"])
                push!(dec_s_a, enc_outputs_h[enc_len_batch[i]][:,i,:])
                push!(dec_s_b, enc_outputs_c[enc_len_batch[i]][:,i,:])
            end
            dec_s[1][0][1]  = permutedims(reshape(hcat(dec_s_a...), (l.opt["rnn_size"], l.opt["batch_size"])), (2,1))
            dec_s[1][0][2]  = permutedims(reshape(hcat(dec_s_b...), (l.opt["rnn_size"], l.opt["batch_size"])), (2,1))
        else
            for i in range(1, stop=l.opt["batch_size"])
                if (cur_index <= length(queue_tree[i]))
                    par_index = queue_tree[i][cur_index ]["parent"]
                    child_index = queue_tree[i][cur_index ]["child_index"]
                    push!(dec_s_a, dec_s[par_index][child_index][1][i,:])
                    push!(dec_s_b, dec_s[par_index][child_index][2][i,:])
                else
                    push!(dec_s_a, KnetArray(zeros(Float32, l.opt["rnn_size"])))
                    push!(dec_s_b, KnetArray(zeros(Float32, l.opt["rnn_size"])))
                end
            end
            dec_s[cur_index][0][1] = permutedims(reshape(hcat(dec_s_a...), (l.opt["rnn_size"], l.opt["batch_size"])), (2,1))
            dec_s[cur_index][0][2] = permutedims(reshape(hcat(dec_s_b...), (l.opt["rnn_size"], l.opt["batch_size"])), (2,1))
        end
        gold_string = " "
        parent_h = dec_s[cur_index][0][1]

        benim_sonu_clar = []
        loss_cum = 0
        for i in range(1, stop=size(dec_batch[cur_index], 2) - 1)
            decd_c, decd_h = decode(l, dec_batch[cur_index][:,i], dec_s[cur_index][i - 1][1], dec_s[cur_index][i - 1][2],  parent_h)
            dec_s[cur_index][i][1], dec_s[cur_index][i][2] = permutedims(reshape(decd_c, l.opt["rnn_size"], l.opt["batch_size"]), (2,1)), permutedims(reshape(decd_h, l.opt["rnn_size"], l.opt["batch_size"]), (2,1))
            pred = attention_decode(l, enc_outputs, dec_s[cur_index][i][2])
            loss2, count3 =  Knet.nll(pred, dec_batch_copy[cur_index][:,i+1]; dims=1, average=false)
            count2 = count2 + count3
            loss = loss + loss2
            max_vals = argmax(value(pred), dims=1)
            push!(benim_sonu_clar, [x[1] for x in max_vals])
        end
        cur_index = cur_index + 1
        push!(benim_sonu, benim_sonu_clar)
    end

    loss = loss / l.opt["batch_size"]
    #println("LOSSLOSSLOSSLOSSLOSSLOSSLOSSLOSSLOSSLOSSLOSSLOSSLOSSLOSSLOSS")
    for i in range(1, stop = max_index)
        #println(dec_batch[i][3,:])
        #println([[y[3] for y in x] for x in benim_sonu])
    end
    #println(loss)
    return loss
end       

function do_generate( l::L2L,  enc_w_list, word_manager, form_manager,   checkpoint)
    
    
    # initialize the rnn state to all zeros
    l.enc_lstm.h, l.enc_lstm.c = 0, 0

    
    # reversed order encoding
    end_w = length(enc_w_list)
    enc_outputs = []
    push!(enc_w_list, get_symbol_idx(word_manager,"<S>"))
    insert!(enc_w_list, 1, get_symbol_idx(word_manager, "<E>"))
    for i in range(end_w+2, 1, step=-1)
    prev_c, prev_h = encode_sample(l, i)
        push!(enc_outputs, prev_h)
    end
    prev_h = permutedims(reshape(prev_h, l.opt["rnn_size"], 1), (2,1))
    prev_c = permutedims(reshape(prev_c, l.opt["rnn_size"], 1), (2,1))
    enc_outputs = reshape(hcat(enc_outputs...), l.opt["rnn_size"],1, end_w+2) 
    
    # decoding
    queue_decode = []
    push!(queue_decode, Dict("s"=>  (prev_c, prev_h), "parent"=>0, "child_index"=>1, "t"=> Tree()))    
    head = 1
    while (head <= length(queue_decode)) & ( head <= 100)
        s = queue_decode[head]["s"]
        parent_h = s[2]
        t = queue_decode[head]["t"]
        if head == 1
            prev_word = get_symbol_idx(form_manager, "<S>")
        else
            prev_word = get_symbol_idx(form_manager, "(")
        end
        i_child = 1
        while true
            prev_c, prev_h = decode(l, prev_word, s[1], s[2],  parent_h)
            prev_c = permutedims(reshape(prev_c, size(prev_c, 1), 1) , (2,1))
            prev_h = permutedims(reshape(prev_h, size(prev_h, 1), 1) , (2,1))
            prediction = attention_decode(l, enc_outputs, prev_h) 
            s = (prev_c, prev_h)
            _prev_word = argmax(value(prediction), dims = 1 )[1][1]
            prev_word = _prev_word
            if (Int.(prev_word) == get_symbol_idx(form_manager, "<E>")) || ( t.num_children >= 50)
                break
            elseif Int.(prev_word) == get_symbol_idx(form_manager, "<N>")
                #print("we predicted N")
                push!(queue_decode, Dict("s"=>  deepcopy(s), "parent"=> head, "child_index"=>i_child, "t"=> Tree()))
                add_child(t, Int.(prev_word))
            else
        add_child(t, Int.(prev_word))
            end
            i_child = i_child + 1
        end
        head = head + 1

    end
    for i in range(length(queue_decode), stop=2, step=-1)
        cur = queue_decode[i]
        queue_decode[cur["parent"]]["t"].children[cur["child_index"]] = cur["t"]
    end
    return to_list(queue_decode[1]["t"], form_manager)
end

function main(opt)
    Random.seed!(opt["seed"])
    
    ##-- load data
    @load  (string(opt["data_dir"]) * "/map.jld2") managers
    word_manager, form_manager = managers
    train_loader = MinibatchLoader(opt, "train", true)
    #dev_loader = MinibatchLoader(opt, "dev", true)
    @load  (string(opt["data_dir"]) * "/test.jld2") data
    valid_data = shuffle(MersenneTwister(13),data)[1:100]
    
    
    ##-- model 
    if !isdir(opt["checkpoint_dir"])
        mkdir(opt["checkpoint_dir"])
    end
    model = L2L(word_manager.vocab_size, form_manager.vocab_size, opt, form_manager, word_manager)

    ##-- start training -- rmsprop  adam
    step = 0
    epoch = 0
    optim_state = Dict("learningRate" => opt["learning_rate"], "alpha" =>  opt["decay_rate"])

    ctrain_loader = collect(train_loader)
    iterations = 60 * train_loader.num_batch
    traindata = shuffle(MersenneTwister(123),collect(take(cycle(ctrain_loader), iterations)))
    accr_visual = [0.0]
    accr_curr = 0
    best_model = model
    #progress!(sgd(model, traindata; lr=optim_state["learningRate"], gclip=5), steps=1) do y
    #progress!(rmsprop(model, traindata; lr=optim_state["learningRate"],rho=0.95,  gclip=5), steps=1) do y
    progress!(adam(model, traindata; lr=optim_state["learningRate"],  gclip=5), steps=1) do y
        #pred_list = [do_generate(model, x[1], word_manager, form_manager, opt)  for x in valid_data]
        #accr_list = [compute_accuracy(pred_list[i], valid_data[i][2])  for i in range(1, stop=length(valid_data))]
        #accr_curr = Statistics.mean(accr_list)
        println("ACCURACY")
        println(Knet.params(model)[1].opt.lr)
        epoch = y.curriter / train_loader.num_batch
        if y.curriter % train_loader.num_batch == 0 
	    pred_list = [do_generate(model, x[1], word_manager, form_manager, opt)  for x in valid_data]
	    accr_list = [compute_accuracy(pred_list[i], valid_data[i][2])  for i in range(1, stop=length(valid_data))]
	    accr_curr = Statistics.mean(accr_list)
	    if accr_curr > accr_visual[end]; best_model = model; end
	    push!(accr_visual, accr_curr) 
	    println("ACCURACY")
	    println(accr_curr)
            if epoch >= opt["learning_rate_decay_after"] 
		for paramtr in Knet.params(model); paramtr.opt.lr = paramtr.opt.lr*opt["learning_rate_decay"]; end
            end
        end  
    end
    println(accr_visual)
    checkpoint = Dict()
    checkpoint["lang2logic_model"] = best_model
    checkpoint["opt"] = opt
    checkpoint["i"] = iterations
    checkpoint["epoch"] = epoch
    Knet.save(opt["checkpoint_dir"] *"/model_seq2seq.jld2", "checkpoint", checkpoint)

    Knet.save("attn-$(Int(time_ns())).jld2", "model", best_model)
end


s = ArgParseSettings()
@add_arg_table s begin
    "--gpuid"
        help = "data path"
        arg_type = Integer
        default = 0
    "--data_dir"
        help = "data path"
        arg_type = String
        default = "../data/"
    "--seed"
        help = "torch manual random number generator seed"
        default = 123
    "--checkpoint_dir"
        help = "filename to autosave the checkpont to. Will be inside checkpoint_dir/"
        arg_type = String
        default = "checkpoint_dir"
    "--savefile"
        help = "max vocab size"
        arg_type = String
        default = "save"
    "--print_every"
        help = "how many steps/minibatches between printing out the loss"
        arg_type = Integer
        default = 2000
    "--rnn_size"
        help = "size of LSTM internal state"
        arg_type = Integer
        default = 200
    "--num_layers"
        help = "number of layers in the LSTM"
        arg_type = Integer
        default = 1
    "--dropout"
        help = "dropout for regularization, used after each RNN hidden layer. 0 = no dropout"
        arg_type = Float64
        default = 0.3
    "--dropoutrec"
        help = "dropout for regularization, used after each c_i. 0 = no dropout"
        arg_type = Float64
        default = 0.3
    "--enc_seq_length"
        help = "number of timesteps to unroll for"
        arg_type = Int64
        default = 60
    "--dec_seq_length"
        help = "number of timesteps to unroll for"
        arg_type = Integer
        default = 220
    "--batch_size"
        help = "number of sequences to train on in parallel"
        arg_type = Integer
        default = 20
    "--max_epochs"
        help = "number of full passes through the training data"
        arg_type = Integer
        default = 130
    "--opt_method"
        help = "optimization method"
        arg_type = Integer
        default = 0
    "--learning_rate"
        help = "learning rate"
        arg_type = Float64
        default = 0.003
    "--init_weight"
        help = "initialization weight"
        arg_type = Float64
        default = 0.08
    "--learning_rate_decay"
        help = "learning rate decay"
        arg_type = Float64
        default = 1.00
    "--learning_rate_decay_after"
        help = "in number of epochs, when to start decaying the learning rate"
        arg_type = Integer
        default = 5
    "--restart"
        help = "in number of epochs, when to restart the optimization"
        arg_type = Integer
        default = -1
    "--decay_rate"
        help = "decay rate for rmsprop"
        arg_type = Float64
        default = 0.95
    "--max_vocab_size"
        help = "max vocab size"
        arg_type = Integer
        default = 15000
    "--grad_clip"
        help = "max vocab size"
        arg_type = Integer
        default = 5
end

args = parse_args(s)
#main(args)
