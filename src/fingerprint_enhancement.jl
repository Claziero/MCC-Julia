"""
Options for the fingerprint enhancement algorithm:
+ `target_width`: the width of the enhanced image 
(the height is computed automatically to keep the aspect ratio)
+ `angle_increment`: the increment (in degrees) between each Gabor filter orientation 
(higher values require computing less filters, but the results are less accurate)
+ `kx`: the scaling factor for the Gabor filter in the x direction
+ `ky`: the scaling factor for the Gabor filter in the y direction
+ `gradient_sigma`: the standard deviation of the Gaussian filter used to compute the gradient images
+ `block_sigma`: the standard deviation of the Gaussian filter used to smooth the gradient images
+ `orient_smooth_sigma`: the standard deviation of the Gaussian filter used to smooth the orientation image
+ `freq_value`: the frequency value used to compute the Gabor filter (higher values mean higher ridge 
frequencies in the image, so for best results it correlates directly with the size of the image)
+ `add_border`: whether to add a border to the image after enhancing it
"""
struct FingerprintEnhancementOptions
    target_width::Int64
    angle_increment::Int64
    kx::Float32
    ky::Float32
    block_sigma::Float32
    gradient_sigma::Float32
    orient_smooth_sigma::Float32
    freq_value::Float32
    add_border::Bool
end

function FingerprintEnhancementOptions(;
    target_width        =300,
    angle_increment     =3,
    kx                  =0.75,
    ky                  =0.75,
    block_sigma         =5.0,
    gradient_sigma      =1.0,
    orient_smooth_sigma =5.0,
    freq_value          =0.13,
    add_border          =true)
    FingerprintEnhancementOptions(
        target_width,
        angle_increment,
        kx,
        ky,
        block_sigma,
        gradient_sigma,
        orient_smooth_sigma,
        freq_value,
        add_border,
    )
end

default_options = FingerprintEnhancementOptions()

"""
Creates a gaussian filter with the given standard deviation.

Sets its size to be 6σ, rounded to the nearest odd integer.
"""
function gaussian_filter(σ::Float32)::Matrix{Float32}
    # Set the size
    sze = 6 * round(Int, σ)
    sze % 2 == 0 && (sze += 1)

    # Create the filter
    x = range(-sze÷2, stop=sze÷2, length=sze)
    y = x'
    g = exp.(-(x.^2 .+ y.^2) / (2 * σ^2))
    g ./= sum(g)
    g
end

"""
Compute the gradient in the x and y directions of the image
"""
function gradient(image::Matrix{Float32})
    gradx = zeros(size(image))
    grady = zeros(size(image))

    rows, cols = size(image)

    for i in 2:rows-1
        for j in 2:cols-1
            # Get the pixels in a cross
            #            ypixel1
            # xpixel1 <current i,j>  xpixel2
            #            ypixel2
            xpixel1 = image[i, j-1]
            xpixel2 = image[i, j+1]

            ypixel1 = image[i-1, j]
            ypixel2 = image[i+1, j]

            gradx[i, j] = 0.5 * (xpixel2 - xpixel1)
            grady[i, j] = 0.5 * (ypixel2 - ypixel1)
        end
    end

    return gradx, grady
end

"""
First step of the enhancement algorithm: used to maximize the contrast between
ridge and valley regions in the image.
"""
function normalize_image(image::Matrix{Float32}, req_mean::Float32, req_var::Float32)::Matrix{Float32}
    image_mean = mean(image)
    normalized_image = image .- image_mean
    image_std = std(normalized_image)
    normalized_image = normalized_image ./ image_std
    normalized_image = req_mean .+ normalized_image .* sqrt(req_var)

    normalized_image
end

"""
Estimate orientation field of fingerprint ridges.

Returns an image with the same size as the input image, where each pixel
contains the orientation of the ridge at that point (in radians).
"""
function orient_ridge(image::Matrix{Float32}; opts::FingerprintEnhancementOptions = default_options)::Matrix{Float32}
    gauss_kernel = gaussian_filter(opts.gradient_sigma)

    # Sobel filter (with gaussian smoothing)
    kernelx = Float32.([0 0 0; 0.5 0 -0.5; 0 0 0])
    kernely = Float32.([0 0.5 0; 0 0 0; 0 -0.5 0])
    fx = imfilter(gauss_kernel, kernelx)
    fy = imfilter(gauss_kernel, kernely)

    # Gradient of the image
    gradx = imfilter(image, fx)
    grady = imfilter(image, fy)

    # Compute the multiplications
    gradxx = gradx .* gradx
    gradyy = grady .* grady
    gradxy = gradx .* grady

    gauss_kernel2 = gaussian_filter(opts.block_sigma)

    # Smooth the covariance data using the gaussian kernel
    gradxx = imfilter(gradxx, gauss_kernel2)
    gradyy = imfilter(gradyy, gauss_kernel2)
    gradxy = 2imfilter(gradxy, gauss_kernel2)

    # Analytic solution of principal direction
    G1 = gradxy .* gradxy
    G2 = gradxx - gradyy
    G2 = G2 .* G2
    denom = sqrt.(G1 + G2)

    sin2theta = gradxy ./ denom
    cos2theta = (gradxx - gradyy) ./ denom

    # STUB Not sure if this is needed
    replace!(sin2theta, NaN=>0)
    replace!(cos2theta, NaN=>0)

    # Smooth the sin and cos values using a gaussian kernel
    gauss_kernel3 = gaussian_filter(opts.orient_smooth_sigma)

    sin2theta = imfilter(sin2theta, gauss_kernel3)
    cos2theta = imfilter(cos2theta, gauss_kernel3)

    # Calculate the orientation image
    orientation_image = (π .+ atan.(sin2theta, cos2theta)) ./ 2

    orientation_image
end


FILTERS_CACHE = Dict{Tuple{Float32, Float32, Float32, Int64}, Vector{Matrix{Float32}}}()

"""
Computes the Gabor filter with all orientations for the given frequency.
"""
function compute_filters(; opts::FingerprintEnhancementOptions)::Vector{Matrix{Float32}}
    unfreq = opts.freq_value

    # Check if we already have the filters for this frequency
    if haskey(FILTERS_CACHE, (unfreq, opts.kx, opts.ky, opts.angle_increment))
        return FILTERS_CACHE[(unfreq, opts.kx, opts.ky, opts.angle_increment)]
    end

    filters = Vector{Matrix{Float32}}()

    # This frequency index looks like something that would be used if
    # we had a frequency image, but we don't, so we just use a single
    # value for the frequency.
    # freq_index = ones(100)
    sigmax = (1.0 / unfreq) * opts.kx
    sigmaxx = sigmax * sigmax
    sigmay = (1.0 / unfreq) * opts.ky
    sigmayy = sigmay * sigmay

    szek = round(Int, 3max(sigmax, sigmay))

    # Create the mesh filter for the Gabor filter
    meshx = [j for _ in -szek:szek, j in -szek:szek]
    meshy = [i for i in -szek:szek, _ in -szek:szek]

    pi_by_unfreq_by_2 = 2π * unfreq

    # Compute the reference filter
    pixval = @. exp(-0.5 * (meshx * meshx / sigmaxx + meshy * meshy / sigmayy))
    reff = pixval .* cos.(pi_by_unfreq_by_2 * meshx)

    for angle in 0:opts.angle_increment:180-opts.angle_increment
        θ = deg2rad(angle + 90)
        # Rotate the reference filter
        rot = imrotate(reff, θ, axes(reff), 0)

        # Add the filter to the list
        push!(filters, Float32.(rot))
    end

    # Cache the filters
    FILTERS_CACHE[(unfreq, opts.kx, opts.ky, opts.angle_increment)] = filters

    filters
end

"""
Performing Gabor filtering for enhancement using previously calculated orientation
image and frequency. The output is final enhanced image.

In this approach, a constant frequency is used instead of a variable frequency, thus The
frequency does not need to be a matrix (and keeping it like that is inefficient).
"""
function filter_ridge(normim::Matrix{Float32}, orientim::Matrix{Float32}; opts::FingerprintEnhancementOptions = default_options)
    # Create the output image
    rows, cols = size(normim)
    enhanced_image = zeros(rows, cols)

    # Create the filters
    filters = compute_filters(; opts=opts)
    # Get the half size of a filter
    szek = (size(filters[1])[1] - 1) ÷ 2

    # Find indices of matrix points greater than maxsze from the image boundary
    maxsze = szek

    # Convert orientation matrix values from radians to an index value that
    # corresponds to round(degrees / opts.angle_increment)
    orientindex = round.(Int, orientim / deg2rad(opts.angle_increment), RoundUp)
    for r in maxsze+1:rows-maxsze, c in maxsze+1:cols-maxsze
        subfilter = filters[orientindex[r, c]]
        subim = normim[r-szek:r+szek, c-szek:c+szek]
        mulres = subim .* subfilter

        if sum(mulres) > 0
            enhanced_image[r, c] = 1
        end
    end

    if opts.add_border
        enhanced_image[1:rows, 1:szek+1] .= 1
        enhanced_image[1:szek+1, 1:cols] .= 1
        enhanced_image[rows-szek:rows, 1:cols] .= 1
        enhanced_image[1:rows, cols-szek:cols] .= 1
    end

    enhanced_image
end

"""
Perform Gabor filter based image enhancement using orientation field and frequency.
"""
function enhance_fingerprints(input_image::Matrix{T}; opts::FingerprintEnhancementOptions = default_options) where T <: Colorant
    # Convert the image to grayscale.
    input_image = Gray.(input_image)

    # Resize the image so that its width is equal to the target width
    height, width = size(input_image)
    target_height = round(Int64, height * opts.target_width / width)
    input_image = imresize(input_image, (target_height, opts.target_width))

    input_image = Float32.(input_image)

    # Perform median blurring to smooth the image
    blurred_image = mapwindow(median, input_image, (3, 3))

    # Perform normalization using the method provided in the paper
    normalized_image = normalize_image(blurred_image, 0.0f0, 1.0f0)

    # Calculate ridge orientation field
    orientation_image = orient_ridge(normalized_image; opts=opts)

    # Get the final enhanced image and return it as result
    enhanced_image = filter_ridge(normalized_image, orientation_image; opts=opts)

    enhanced_image
end
