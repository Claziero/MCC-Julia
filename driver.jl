include("./FingerprintMatching.jl")
using .FingerprintMatching

using Images
using ArgParse
using ProgressBars

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

posangle(x) = x > 0 ? x : x + 2π


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
else
    println("Unknown command '$command'")
    exit(1)
end

