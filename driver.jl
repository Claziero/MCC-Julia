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

# Code for verification evaluation
"""
Function to get the list of probe's scores
given the image path
"""
function get_probes_scores(img_path)
    gray_img = Gray.(load(img_path))
    matrix = Float32.(gray_img)
    rows, cols = size(matrix)

    @assert rows == cols

    println("There are $(rows * cols - rows) probes in the matrix.")

    genuine_attempts = Float32[]
    impostor_attempts = Float32[]
    
    # Iterate over all coordinates of matrix
    @inbounds for i in 1:rows
        for j in 1:cols
            # If the probes are the same, do not add to scores
            i == j && continue

            id1 = (i - 1) ÷ 8
            id2 = (j - 1) ÷ 8

            attempt = matrix[i, j]
            id1 == id2 ? push!(genuine_attempts, attempt) : push!(impostor_attempts, attempt)
        end
    end

    genuine_attempts, impostor_attempts
end

""" False Rejection Rate (FRR) """
frr(genuine::Vector{Float32}, threshold) = count(genuine .<= threshold) / length(genuine) 
""" False Acceptance Rate (FAR) """
far(impostor::Vector{Float32}, threshold) = count(impostor .>= threshold) / length(impostor)
""" Genuine Acceptance Rate (GAR) """
gar(genuine::Vector{Float32}, threshold) = count(genuine .> threshold) / length(genuine)
""" Genuine Recognition Rate (GRR) """
grr(impostor::Vector{Float32}, threshold) = count(impostor .< threshold) / length(impostor)

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
function far_vs_frr(genuine::Vector{Float32}, impostor::Vector{Float32})
    thresholds = range(0.3, stop=0.8, length=100)
    frrs = frr.(Ref(genuine), thresholds)
    fars = far.(Ref(impostor), thresholds)

    # Find the equal error rate
    index = findmin(abs.(frrs .- fars))[2]
    eer = (frrs[index] + fars[index]) / 2
    eer = round(100eer, digits=2)
    println("Equal Error Rate: $eer%")

    plot(xlabel="Threshold", ylabel="Rate", title="FAR vs FRR", legend=:right)
    plot!(thresholds, 100frrs, label="FRR", color=:blue)
    plot!(thresholds, 100fars, label="FAR", color=:red)
end

""" Receiver Operating Characteristic (ROC) curve """
function roc_curve(genuine::Vector{Float32}, impostor::Vector{Float32})
    thresholds = range(0.3, stop=0.8, length=100)
    fars = far.(Ref(impostor), thresholds)
    gars = gar.(Ref(genuine), thresholds)

    # Compute the AUC
    auc = sum(diff(fars) .* (gars[2:end] .+ gars[1:end-1])) / 2
    auc = -round(100auc, digits=2)
    println("Area Under Curve: $auc%")

    plot(xlabel="FAR", ylabel="GAR", legend=:right, xlim=(-0.04, 1), ylim=(0, 1), title="ROC Curve")
    plot!(fars, gars, fillrange=zero(fars), fillalpha=0.3, label="ROC", color=:blue, aspect_ratio=1)
    plot!(0:0.5:1, 0:0.5:1, label="Random", color=:red, linestyle=:dash)
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

    curve = far_vs_frr(genuine, impostor)
    savefig(curve, output_img_path * "far_vs_frr.png")

    curve = roc_curve(genuine, impostor)
    savefig(curve, output_img_path * "roc_curve.png")

    curve = score_distributions(genuine, impostor)
    savefig(curve, output_img_path * "score_distributions.png")
else
    println("Unknown command '$command'")
    exit(1)
end

