const SPURIOUS_MINUTIAE_THRESH = 20

"""Enum for minutiae types"""
@enum MinutiaeType begin
    BIFURCATION = 0
    TERMINATION = 1
end

"""
Struct for minutiae features containing the coordinates and the angles.
# Fields
- `x::Int`: X coordinate of the minutiae.
- `y::Int`: Y coordinate of the minutiae.
- `θ::Vector{Float64}`: Angles in degrees.
- `type::MinutiaeType`: Type of the minutiae.
"""
struct Minutia
    x::Int
    y::Int
    θ::Vector{Float64} # Angles in degrees
    type::MinutiaeType
end


"""
Load an image from the given path.
Returns a white image on black background.
# Arguments
- `path::String`: Path to the image.
"""
function load_fingerprint_image(path::String)::Matrix{Bool}
    # Load an image
    img = load(path)
    # Binarize it using Otsu
    img = binarize(img, Otsu())
    # Convert it to a bit Matrix
    img = Bool.(img)

    # Invert the image if the background is white
    if sum(img) > length(img) / 2
        img = .!img
    end

    # Return the image
    img
end

"""
Compute the convex hull of the image.
Returns a binary image with the convex hull of the image.
# Arguments
- `img::Matrix{Bool}`: Image to compute the convex hull of.
- `erode::Int`: Number of times to erode the convex hull.
"""
function convex_hull_image(img::Matrix{Bool}, erode::Int=0)::Matrix{Bool}
    hull = convexhull(img)
    # Create a mask image
    mask = Gray.(zeros(size(img)))
    # Draw the path on the mask
    draw!(mask, Polygon(hull))

    # Fill the path
    for i in 1:size(mask, 1)
        # Find the first one on the row
        first_one = findfirst(==(1), mask[i, :])
        # Find the last one on the row
        last_one = findlast(==(1), mask[i, :])

        if first_one === nothing || last_one === nothing
            continue
        end

        # Fill the row
        mask[i, first_one:last_one] .= 1
    end

    # Convert the mask to a bitmap
    mask .> 0.5
    # Erode it the given number of times
    for _ in 1:erode
        mask = erode(mask)
    end

    # Return the mask
    mask
end

"""
Denoise the image and compute its skeleton.
Returns a skeletonized copy of the image.
# Arguments
- `img::Matrix{Bool}`: Image to skeletonize.
"""
function skeletonize(img::Matrix{Bool})::Matrix{Bool}
    # Create the skeleton
    skeleton = thinning(img)

    # Return the skeleton
    skeleton
end

"""
Identify terminations and bifurcations in a skeletonized image.
Returns two images, one with the terminations and one with the bifurcations.
# Arguments
- `img::Matrix{Bool}`: Image to identify the terminations and bifurcations in.
- `skeleton::Matrix{Bool}`: Skeleton of the image.
"""
function get_termination_bifurcation(img::Matrix{Bool}, skeleton::Matrix{Bool})::Tuple{Matrix{Bool}, Matrix{Bool}}
    kernel = centered(fill(1, 3, 3))
    block_sum = imfilter(skeleton, kernel) .* skeleton

    hull = convex_hull_image(img)
    hull = hull |> erode |> erode |> erode |> erode |> erode |> erode |> erode |> erode |> erode
    hull = hull .== 1

    # Find the terminations
    terminations = (block_sum .== 2) .& hull
    # Find the bifurcations
    bifurcations = block_sum .== 4

    # Return the images
    terminations, bifurcations
end

"""
Clean the list of terminations by removing the ones that are too close to each other.
Returns the terminations matrix without the spurious ones.
# Arguments
- `terminations::Matrix{Bool}`: Terminations image.
"""
function clean_minutiae(minutiae::Matrix{Bool})::Matrix{Bool}
    # Label all components in the termination image
    labels = label_components(minutiae)

    # Get the centroid of each component 
    # (omit the first one containing the centroid for the background)
    centroids = component_centroids(labels)[2:end]

    # Round all coordinates to integers
    centroids = [(Int64.(round.((x, y), RoundNearest))) for (y, x) in centroids]

    # Delete all the minutiae that are too close to each other
    centroids_min = Set{Tuple{Int64, Int64}}()
    for (x, y) in centroids
        if !any([(x - x1)^2 + (y - y1)^2 < SPURIOUS_MINUTIAE_THRESH for (x1, y1) in centroids_min])
            push!(centroids_min, (x, y))
        end
    end

    # Convert the set of points to a new matrix of bool
    output_minutiae = zeros(Bool, size(minutiae))
    for (x, y) in centroids_min
        output_minutiae[y, x] = true
    end

    # Return the new matrix
    output_minutiae
end

"""
Function to compute the angle of the minutia feature.
Returns a vector of angles in degrees.
# Arguments
- `block::Matrix{Bool}`: Block of the image containing the minutia.
- `min_type::Minutia`: Type of the minutia.
"""
function compute_angle(block::Matrix{Bool}, min_type::MinutiaeType)::Union{Nothing, Vector{Float64}}
    rows, cols = size(block)
    @assert rows == cols

    # Get the coordinates of the center of the block
    center_x = center_y = rows ÷ 2 + 1

    # Look for all the pixels that are along the borders of the block and that are true
    # and get the angle of that point from the center point (which will always be true)

    # Filter all the positions in block that lie in the border of the block itself
    positions = [(x, y) for x in 1:cols, y in 1:rows
        if (x == 1 || x == cols || y == 1 || y == rows) && block[y, x] != 0]
    
    if min_type == TERMINATION
        # If there are more or less than one angle found, then the minutia is not a termination
        length(positions) != 1 && return nothing

        # Return the angle of the found point
        angle = -atand(positions[1][2] - center_y, positions[1][1] - center_x)
        [angle]
        
    elseif min_type == BIFURCATION
        # If there are more or less than three angles found, then the minutia is not a bifurcation
        length(positions) != 3 && return nothing

        # Return the angle of the found points
        angle1 = -atand(positions[1][2] - center_y, positions[1][1] - center_x)
        angle2 = -atand(positions[2][2] - center_y, positions[2][1] - center_x)
        angle3 = -atand(positions[3][2] - center_y, positions[3][1] - center_x)
        [angle1, angle2, angle3]
    end
end

"""
Function to perform the feature extraction.
Returns a vector of MinutiaeFeature.
# Arguments
- `terminations::Matrix{Bool}`: Terminations image.
- `bifurcations::Matrix{Bool}`: Bifurcations image.
- `skeleton::Matrix{Bool}`: Skeleton image.
"""
function perform_feature_extraction(
    terminations::Matrix{Bool}, 
    bifurcations::Matrix{Bool}, 
    skeleton::Matrix{Bool}
)::Tuple{Vector{Minutia}, Vector{Minutia}}
    ## TERMINATIONS
    feature_terminations = []
    block_size = 2

    # Label all components in the termination image and get the centroid of each component 
    labels = label_components(terminations)
    centroids = component_centroids(labels)[2:end]
    centroids = [(Int64.(round.((x, y), RoundNearest))) for (y, x) in centroids]

    # Compute the angle of each centroid
    for (x, y) in centroids
        # Get the block of pixels around the centroid
        block = skeleton[y - block_size:y + block_size, x - block_size:x + block_size]
        # Compute the angle of the block
        angle = compute_angle(block, TERMINATION)
        # Add the feature to the list if the angle is valid
        if angle !== nothing 
            push!(feature_terminations, Minutia(x, y, angle, TERMINATION)) 
        end
    end

    ## BIFURCATIONS
    feature_bifurcations = []
    block_size = 1

    # Label all components in the bifurcation image and get the centroid of each component
    labels = label_components(bifurcations)
    centroids = component_centroids(labels)[2:end]
    centroids = [(Int64.(round.((x, y), RoundNearest))) for (y, x) in centroids]
    
    # Compute the angles of each centroid
    for (x, y) in centroids
        # Get the block of pixels around the centroid
        block = skeleton[y - block_size:y + block_size, x - block_size:x + block_size]
        # Compute the angle of the block
        angle = compute_angle(block, BIFURCATION)
        # Add the feature to the list if the angle is valid
        if angle !== nothing 
            push!(feature_bifurcations, Minutia(x, y, angle, BIFURCATION))
        end
    end

    # Return the list of features
    feature_terminations, feature_bifurcations
end

"""
Function to show the result of the feature extraction.
Returns a new image in RGB space.
# Arguments
- `img::Matrix{Bool}`: Original image.
- `terminations::Vector{Minutia}`: Terminations features.
- `bifurcations::Vector{Minutia}`: Bifurcations features.
"""
function create_minutiae_image(
    img::Matrix{Bool}, 
    terminations::Vector{Minutia}, 
    bifurcations::Vector{Minutia}
)::Matrix{ColorTypes.RGB{Float64}}
    # Create a new image in RGB space
    img_rgb = zeros(RGB{Float64}, size(img, 1), size(img, 2))

    # Copy the original image to the new image
    for x in 1:size(img, 2), y in 1:size(img, 1)
        img_rgb[y, x] = RGB(img[y, x], img[y, x], img[y, x])
    end

    # Mark the termination features using a red box of size 1
    for t in terminations
        img_rgb[t.y-1:t.y+1, t.x-1:t.x+1] .= RGB(1, 0, 0)
    end

    # Mark the bifurcation features using a blue box of size 1
    for b in bifurcations
        img_rgb[b.y-1:b.y+1, b.x-1:b.x+1] .= RGB(0, 0, 1)
    end
    
    # Show the image
    img_rgb
end

"""
Extract fingerprint features from an image file given its path.
# Arguments
- `filename::String`: The path to the image file.
- `save_result::String`: The path to save the result image, if empty no image will be saved.
- `save_skel::String`: The path to save the skeleton image, if empty no image will be saved.
- `verbose::Bool`: If true, the extraction results will be shown.
"""
function extract_features(
    filename::String; 
    save_result::String="", 
    save_skel::String="", 
    verbose::Bool=false
)::Tuple{Vector{Minutia}, Vector{Minutia}}
    # Load an image
    img = load_fingerprint_image(filename)
    extract_features(img; save_result=save_result, save_skel=save_skel, verbose=verbose)
end

"""
Extract the fingerprint features given a boolean matrix representing a fingerprint image.
# Arguments
- `img::Matrix{Bool}`: The fingerprint image.
- `save_result::String`: The path to save the result image, if empty no image will be saved.
- `save_skel::String`: The path to save the skeleton image, if empty no image will be saved.
- `verbose::Bool`: If true, the extraction results will be shown.
"""
function extract_features(
    img::Matrix{Bool}; 
    save_result::String="", 
    save_skel::String="", 
    verbose::Bool=false
)::Tuple{Vector{Minutia}, Vector{Minutia}}
    # Skeletonize the image
    skeleton = skeletonize(img)
    save_skel != "" && save(save_skel, skeleton)

    # Get the termination and bifurcation images
    terminations, bifurcations = get_termination_bifurcation(img, skeleton)

    # Clean the minutiae
    clean_terminations = clean_minutiae(terminations)
    clean_bifurcations = clean_minutiae(bifurcations)

    # Perform the feature extraction
    terminations, bifurcations = perform_feature_extraction(clean_terminations, clean_bifurcations, skeleton)
    verbose && println("Extracted features -> T=$(length(terminations)), B=$(length(bifurcations))")

    if save_result != ""
        img_rgb = create_minutiae_image(img, terminations, bifurcations)
        save(save_result, img_rgb)
    end

    terminations, bifurcations
end
