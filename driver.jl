include("./FingerprintMatching.jl")
using .FingerprintMatching

using Images
using ArgParse
using ProgressBars
using Plots

function list_images(path)::Vector{String}
    println(path)

    if path === nothing
        println("You must specify a path to the image(s).")
        exit(1)
    end

    path_is_file = isfile(path)
    path_is_dir = isdir(path)

    if !path_is_file && !path_is_dir
        println("The path '$path' does not exist.")
        exit(1)
    end

    # If the path points to a file, return it.
    path_is_file && return [path]

    # Since the path is now known to be a directory, get a list of all
    # the files in the directory
    joinpath.(path, readdir(path))
end

function compute_cylinders(img)
    img = Gray.(img)
    terms, bifur = extract_features(Matrix{Bool}(img .< 0.5))
    cylinder_set(img, [terms; bifur])
end

# Common evaluation code
struct Probe
    score::Float32
    id1::Int
    id2::Int
    tid1::Int
    tid2::Int
end

"""
Function to get the list of probe's scores
given the image path
"""
function get_probes(img_path)
    gray_img = Gray.(load(img_path))
    matrix = Float32.(gray_img)
    rows, cols = size(matrix)

    @assert rows == cols

    println("There are $(rows * cols - rows) probes in the matrix.")

    probes = Probe[]
    
    # Iterate over all coordinates of matrix
    @inbounds for i in 1:rows
        for j in 1:cols
            # If the probes are the same, do not add to scores
            if i != j
                id1 = (i - 1) ÷ 8
                id2 = (j - 1) ÷ 8

                attempt = matrix[i, j]
                push!(probes, Probe(attempt, id1, id2, i, j))
            end
        end
    end

    probes
end


### Code for verification evaluation
"""
Function to get the scores for the genuine and impostor attempts
(with the correlated number of attempts)
"""
function get_probes_scores(img_path)
    probes = get_probes(img_path)

    genuine_attempts = [probe.score for probe in probes if probe.id1 == probe.id2]
    impostor_attempts = [probe.score for probe in probes if probe.id1 != probe.id2]

    genuine_attempts, impostor_attempts
end

""" False Rejection Rate (FRR) """
frr(genuine::Vector{Float32}, threshold)::Float32 = count(genuine .<= threshold) / length(genuine) 
""" False Acceptance Rate (FAR) """
far(impostor::Vector{Float32}, threshold)::Float32 = count(impostor .>= threshold) / length(impostor)
""" Genuine Acceptance Rate (GAR) """
gar(genuine::Vector{Float32}, threshold)::Float32 = count(genuine .> threshold) / length(genuine)
""" Genuine Recognition Rate (GRR) """
grr(impostor::Vector{Float32}, threshold)::Float32 = count(impostor .< threshold) / length(impostor)

# Equations
# FRR = 1 - GAR <=> FRR + GAR = 1
# FAR = 1 - GRR <=> FAR + GRR = 1

""" Impostor/genuine Distributions """
function score_distributions(genuine::Vector{Float32}, impostor::Vector{Float32})
    histogram(xlabel="Similarity Score", ylabel="Frequency", title="Score Distributions", legend=:topright)
    histogram!(impostor, bins=500, label="Impostor", normalize=:probability, color=:red, alpha=0.5)
    histogram!(genuine, bins=500, label="Genuine", normalize=:probability, color=:blue, alpha=0.3)
end

""" FAR vs FRR curve """
function far_vs_frr(fars::Vector{Float32}, frrs::Vector{Float32}, thresholds)
    # Find the equal error rate
    index = findmin(abs.(frrs .- fars))[2]
    eer = (frrs[index] + fars[index]) / 2

    plot(xlabel="Threshold", ylabel="Rate", title="FAR vs FRR", legend=:right)
    plot!(thresholds, 100frrs, label="FRR", color=:blue)
    eer, plot!(thresholds, 100fars, label="FAR", color=:red)
end

""" Receiver Operating Characteristic (ROC) curve """
function roc_curve(fars::Vector{Float32}, frrs::Vector{Float32}, thresholds)
    gars = 1 .- frrs
    # Compute the AUC
    auc = -sum(diff(fars) .* (gars[2:end] .+ gars[1:end-1])) / 2
    plot(xlabel="FAR", ylabel="GAR", legend=:right, xlim=(-0.04, 1), ylim=(0, 1), title="ROC Curve")
    plot!(fars, gars, fillrange=zero(fars), fillalpha=0.3, label="ROC", color=:blue, aspect_ratio=1)
    auc, plot!(0:0.5:1, 0:0.5:1, label="Random", color=:red, linestyle=:dash)
end


### Code for open set identification evaluation
"""
Divide the probes by the first template id
"""
function divide_probes_by_template_id(probes::Vector{Probe})
    probes_by_tid = Dict{Tuple{Int, Int}, Vector{Probe}}()

    for probe in probes
        key = (probe.id1, probe.tid1)
        if haskey(probes_by_tid, key)
            push!(probes_by_tid[key], probe)
        else
            probes_by_tid[key] = Probe[probe]
        end
    end

    # Sort them (highest score first)
    for key in keys(probes_by_tid)
        probes_by_tid[key] = sort(probes_by_tid[key], by=probe -> probe.score, rev=true)
    end

    probes_by_tid
end

"""
Compute all the results for a given threshold
"""
function openset_results_for_threshold(probes_by_tid, t)
    DI = zeros(Int64, 1024)
    FA = 0
    GR = 0
    
    for ((id1, tid), probes) in probes_by_tid
        passing_probes = Probe[probe for probe in probes if probe.score >= t]

        # If there are no passing probes
        if length(passing_probes) == 0
            # Imposter case: the subject is not in the gallery
            # and we correctly do not detect it
            GR += 1

            # Genuine case: the subject is in the gallery but we do not detect it
            # We do not count it because we can get it using DIR(t, 1)
            continue
        end

        # If there are passing probes
        # Genuine detect+identify
        if id1 == passing_probes[1].id2
            # Genuine case: the subject is in the gallery and we detect it correctly
            DI[1] += 1

            # Imposter case: we let them pass but with the wrong identity
            # There are two cases for this:
            if any(probe -> probe.id2 != id1, passing_probes[2:end])
                # 1. The subject is in the gallery but we detect it incorrectly
                FA += 1
            else
                # 2. The subject is not in the gallery and we don't detect them
                GR += 1
            end
        else
            # Find the first k such that passing_probes[k].id2 == id1
            k = findfirst(probe -> probe.id2 == id1, passing_probes)

            # Genuine case: the subject is in the gallery but we detect it with higher rank
            if k != nothing
                DI[k] += 1
            end
            
                # Imposter case: we let them pass but with the wrong identity
            FA += 1
        end
    end

    FA, GR, DI
end

"""
Get FAs, GAs, GRs, FRs, DIRs in the open set verification.
"""
function open_set_results(probes_by_tid, thresholds, img_width=1024)
    FAs = Float32[]
    GRs = Float32[]
    GAs = Float32[]
    DIRs = Vector{Float32}[]
    for t_el in thresholds
        FA, GR, DI = openset_results_for_threshold(probes_by_tid, t_el)
        push!(FAs, FA / Float32(img_width))
        push!(GRs, GR / Float32(img_width))

        # Compute the cumulative DI (DIR)
        DIR = cumsum(DI) ./ Float32(img_width)
        push!(DIRs, DIR)
        push!(GAs, DIR[1])
    end

    FRs = 1 .- map(x -> x[1], DIRs)
    FAs, GAs, GRs, FRs, DIRs
end


### Code for closed set identification evaluation
"""
Computes the Cumulative Match Characteristic (CMC) curve
"""
function cmc_curve(probes_by_tid, img_width=1024)
    cms = zeros(img_width)

    for ((id, tid), probe) in probes_by_tid
        # Find the first probe with id == probe.id2
        k = findfirst(probe -> probe.id2 == id, probe)
        if k != nothing
            cms[k] += 1
        end
    end

    cms = cumsum(cms) ./ Float32(img_width)
    rr = cms[1]
    rr, plot(cms, xlabel="Rank", ylabel="identification probability", title="Cumulative Match Characteristic", ylim=(0, 1), xlim=(1, 64), legend=:none)
end



parse_settings = ArgParseSettings()
@add_arg_table parse_settings begin
    "command"
        help = "command to run"
        arg_type = String
        required = true
    ["--input", "-i"]
        help = "path to input image(s)"
        arg_type = String
    ["--output", "-o"]
        help = "path to output images"
        arg_type = String
    ["--rotate", "-r"]
        help = "rotate the image by 90 degrees to the right"
        action = :store_true
    ["--method", "-m"]
        help = "method to use for matching"
        arg_type = String
        default = "lssr"
end

parsed_args = parse_args(parse_settings)
command = parsed_args["command"]


if command == "enhance"
    input_imgs_paths = list_images(parsed_args["input"])
    println("Enhancing $(length(input_imgs_paths)) image(s) from '$(parsed_args["input"])'")
    output_imgs_paths = parsed_args["output"]
    println("Saving them to '$(parsed_args["output"])'")
    rotate = parsed_args["rotate"]
    rotate && println("Rotating the images by 90 degrees to the right")

    # Read the images and enhance them one at a time
    for name in ProgressBar(input_imgs_paths)
        img = load(name)
        rotate && (img = rotr90(img))
        enhanced_img = enhance_fingerprints(img)
        save(joinpath(output_imgs_paths, basename(name)), enhanced_img)
    end
elseif command == "match"
    input_imgs_path = parsed_args["input"]
    println("Matching all image(s) from '$(parsed_args["input"])'")
    output_img_paths = parsed_args["output"]
    println("The similarity matrix will be saved to '$(parsed_args["output"])'")

    method = nothing
    if parsed_args["method"] == "lssr"
        method = lssr
    elseif parsed_args["method"] == "lss"
        method = lss
    elseif parsed_args["method"] == "max"
        method = (x, y) -> max(lss(x, y), lssr(x, y))
    else 
        println("Unknown method '$(parsed_args["method"])'")
        exit(1)
    end

    println("Loading the images...")
    # Sort the inputs by name
    input_imgs_dir = sort!(readdir(input_imgs_path); by = (k) -> (x = split(k, ".")[1]; parse.(Int, split(x, "_"))))
    # Load the input images
    input_imgs = [load(joinpath(input_imgs_path, name)) for name in ProgressBar(input_imgs_dir)]
    println("Loaded images")

    println("Computing the cylinders...")
    # Compute the cylinders of the set of images
    cylinders = [compute_cylinders(img) for img in ProgressBar(input_imgs)]
    println("Computed cylinders")

    # Compute the matches
    lc = length(cylinders)
    # Compute the indices
    indices = [(i, j) for i in 1:lc for j in 1:lc if i < j]
    # Loop over the unordered pairs of keys
    similarity_mat = fill(0.0, lc, lc)

    println("Computing the similarity matrix...")
    for (i, j) in ProgressBar(indices)
        # Sort the similarities in descending order
        # Compute the similarity
        similarity_mat[i, j] = method(cylinders[i], cylinders[j])
    end
    # Complete the bottom half of the matrix
    similarity_mat = similarity_mat + similarity_mat'

    # Save the similarity matrix as an image
    save(output_img_paths, similarity_mat)
elseif command == "verification"
    input_similarity_matrix = parsed_args["input"]
    println("Loading the similarity matrix from '$(parsed_args["input"])'")
    output_img_path = parsed_args["output"]
    println("Saving the curves to '$(parsed_args["output"])'")

    genuine, impostor = get_probes_scores(input_similarity_matrix)
    thresholds = collect(range(0.3, stop=0.8, length=100))
    frrs = frr.(Ref(genuine), thresholds)
    fars = far.(Ref(impostor), thresholds)

    eer, curve = far_vs_frr(fars, frrs, thresholds)
    println("Equal Error Rate: $(round(100eer, digits=2))%")
    savefig(curve, output_img_path * "far_vs_frr.png")

    auc, curve = roc_curve(fars, frrs, thresholds)
    println("Area Under Curve: $(round(100auc, digits=2))%")
    savefig(curve, output_img_path * "roc_curve.png")

    curve = score_distributions(genuine, impostor)
    savefig(curve, output_img_path * "score_distributions.png")
elseif command == "identification"
    input_similarity_matrix = parsed_args["input"]
    println("Loading the similarity matrix from '$(parsed_args["input"])'")
    output_img_path = parsed_args["output"]
    println("Saving the curves to '$(parsed_args["output"])'")

    # Create the output dir if it does not exist
    isdir(output_img_path) || mkdir(output_img_path)

    # Get the width of the image
    img_width = size(load(input_similarity_matrix))[1]

    probes = get_probes(input_similarity_matrix)
    probes_by_tid = divide_probes_by_template_id(probes)

    thresholds = collect(range(0.3, stop=0.8, length=150))
    FAs, GAs, GRs, FRs, DIRs = open_set_results(probes_by_tid, thresholds, img_width)

    eer, curve = far_vs_frr(FAs, FRs, thresholds)
    println("Equal Error Rate: $(round(100eer, digits=2))%")
    savefig(curve, output_img_path * "far_vs_frr.png")

    auc, curve = roc_curve(FAs, FRs, thresholds)
    println("Area Under Curve: $(round(100auc, digits=2))%")
    savefig(curve, output_img_path * "roc_curve.png")

    rr, curve = cmc_curve(probes_by_tid, img_width)
    println("Recognition Rate: $(round(100rr, digits=2))%")
    savefig(curve, output_img_path * "cmc_curve.png") 
else
    println("Unknown command '$command'")
    exit(1)
end

