module FingerprintFeatureExtraction

# Libraries for working with images
using Images
using ImageDraw
using ImageView
using ImageMorphology
using ImageBinarization

const SPURIOUS_MINUTIAE_THRESH = 20

# Enum for minutiae types
@enum MinutiaeType begin
    BIFURCATION = 0
    TERMINATION = 1
    INVALID = -1
end

# Struct for minutiae features containing the coordinates and the angles.
struct MinutiaeFeature
    x::Int
    y::Int
    θ::Vector{Float64} # Angles in degrees
    type::MinutiaeType
end


# Loads an image from the given path.
# Returns a white image on black background.
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

# Denoise an image using "iterations" many morphological operations (open).
# Returns a denoised copy of the image.
function denoise_fingerprint(img::Matrix{Bool}; iterations = 1)::Matrix{Bool}
    # Create a copy of the image
    img = copy(img)

    # Denoise the image
    for _ in 1:iterations
        img = opening(img)
    end

    # Return the image
    img
end

# Compute the convex hull of the image.
# Returns a binary image with the convex hull of the image.
function convex_hull_image(img::Matrix{Bool}, erode = 0)::Matrix{Bool}
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

# Denoise the image and compute its skeleton.
# Returns a skeletonized copy of the image.
function skeletonize(img::Matrix{Bool})::Matrix{Bool}
    # Copy and denoise the image
    img = denoise_fingerprint(img)

    # Create the skeleton
    skeleton = thinning(img)

    # Return the skeleton
    skeleton
end

# Identify terminations and bifurcations in a skeletonized image.
# Returns two images, one with the terminations and one with the bifurcations.
function get_termination_bifurcation(img::Matrix{Bool}, skeleton::Matrix{Bool})::Vector{Matrix{Bool}}
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
    [terminations, bifurcations]
end

# Clean the list of terminations by removing the ones that are too close to each other.
# Returns the terminations matrix without the spurious ones.
function clean_minutiae(terminations::Matrix{Bool})::Matrix{Bool}
    # Label all components in the termination image
    labels = label_components(terminations)

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
    terminations = zeros(Bool, size(terminations))
    for (x, y) in centroids_min
        terminations[y, x] = true
    end

    # Return the new matrix
    terminations
end

# Function to compute the angle of the minutia feature
# Returns a vector of angles in degrees
function compute_angle(block::Matrix{Bool}, min_type::MinutiaeType)
    # Get the coordinates of the center of the block
    center_x = center_y = size(block, 1) ÷ 2 + 1

    # Look for all the pixels that are along the borders of the block and that are true
    # and get the angle of that point from the center point (which will always be true)

    # Filter all the positions in block that lie in the border of the block itself
    positions = [(x, y) for x in 1:size(block, 2), y in 1:size(block, 1) 
        if (x == 1 || x == size(block, 2) || y == 1 || y == size(block, 1)) && block[y, x] != 0]
    
    if min_type == TERMINATION
        # If there are more or less than one angle found, then the minutia is not a termination
        if length(positions) != 1 
            return INVALID 
        end

        # Return the angle of the found point
        angle = -atand(positions[1][2] - center_y, positions[1][1] - center_x)
        [angle]
        
    elseif min_type == BIFURCATION
        # If there are more or less than three angles found, then the minutia is not a bifurcation
        if length(positions) != 3 
            return INVALID 
        end

        # Return the angle of the found points
        angle1 = -atand(positions[1][2] - center_y, positions[1][1] - center_x)
        angle2 = -atand(positions[2][2] - center_y, positions[2][1] - center_x)
        angle3 = -atand(positions[3][2] - center_y, positions[3][1] - center_x)
        [angle1, angle2, angle3]
    end
end

# Function to perform the feature extraction
# Returns a vector of MinutiaeFeature
function perform_feature_extraction(terminations::Matrix{Bool}, bifurcations::Matrix{Bool}, skeleton::Matrix{Bool})::Vector{Vector{MinutiaeFeature}}
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
        if angle != INVALID 
            push!(feature_terminations, MinutiaeFeature(x, y, angle, TERMINATION)) 
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
        if angle != INVALID 
            push!(feature_bifurcations, MinutiaeFeature(x, y, angle, BIFURCATION))
        end
    end

    # Return the list of features
    [feature_terminations, feature_bifurcations]
end

# Function to show the result of the feature extraction
# Returns a new image in RGB space
function show_results(img::Matrix{Bool}, features::Vector{Vector{MinutiaeFeature}})::Matrix{ColorTypes.RGB{Float64}}
    # Create a new image in RGB space
    img_rgb = zeros(RGB{Float64}, size(img, 1), size(img, 2))

    # Copy the original image to the new image
    for x in 1:size(img, 2), y in 1:size(img, 1)
        img_rgb[y, x] = RGB(img[y, x], img[y, x], img[y, x])
    end

    # Mark the termination features using a red box of size 1
    for termination in features[1]
        for x in termination.x - 1:termination.x + 1, y in termination.y - 1:termination.y + 1
            img_rgb[y, x] = RGB(1, 0, 0)
        end
    end

    # Mark the bifurcation features using a blue box of size 1
    for bifurcation in features[2]
        for x in bifurcation.x - 1:bifurcation.x + 1, y in bifurcation.y - 1:bifurcation.y + 1
            img_rgb[y, x] = RGB(0, 0, 1)
        end
    end
    
    # Show the image
    RGB.(img_rgb)
end

# Function to save the result of the feature extraction
# Returns nothing
function save_results(img::Matrix{Bool}, features::Vector{Vector{MinutiaeFeature}}, filename::String)
    # Get the image from the show_results function
    img_rgb = show_results(img, features)
    
    # Save the image
    save(filename, RGB.(img_rgb))
end

# Function to save the skeleton of the fingerprint
# Returns nothing
function save_skeleton(skeleton::Matrix{Bool}, filename::String)
    save(filename, Gray.(skeleton))
end

# Driver function
# Returns nothing
function driver(filename::String; save_result::String="", save_skel::String="", show_result::Bool=false)
    # Load an image
    img = load_fingerprint_image(filename);

    # Skeletonize the image
    skeleton = skeletonize(img)
    if save_skel != "" save_skeleton(skeleton, save_skel) end

    # Get the termination and bifurcation images
    terminations, bifurcations = get_termination_bifurcation(img, skeleton)

    # Clean the minutiae
    clean_terminations = clean_minutiae(terminations)

    # Perform the feature extraction
    features = perform_feature_extraction(clean_terminations, bifurcations, skeleton)

    println("Num of terminations: ", length(features[1]))
    println("Num of bifurcations: ", length(features[2]))

    if show_result == true imshow(show_results(skeleton, features)) end
    if save_result != "" save_results(skeleton, features, save_result) end
end

# Export the driver function only
export driver

end # module