"""
Constants for the MCC algorithm:
- `Ω`: Number of pixels to expand the convex hull of
- `R`: Radius of the cylinder
- `NS`: Number of cells along one direction of the cuboid base enclosing the cylinder
- `ND`: Number of cells along the cuboid height enclosing the cylinder
- `σS`: Spatial standard deviation 
- `σD`: Directional standard deviation
- `ΔS`: Cell size along one direction of the cuboid base
- `ΔD`: Cell size along the cuboid height
"""
struct Parameters
    Ω::Int64
    R::Int64
    NS::Int64
    ND::Int64
    σS::Int64
    σD::Int64
    μψ::Float64
    τψ::Float64
    ΔS::Float64
    ΔD::Float64
end

function Parameters(;
    Ω =     20,
    R =     10,
    NS =    20,
    ND =    10,
    σS =    2,
    σD =    2,
    uψ =    2,
    tψ =    2)
    Parameters(
        Ω,
        R,
        NS,
        ND,
        σS,
        σD,
        uψ,
        tψ,
        2R / NS,
        2π / ND)
end

params = Parameters()

"""
Struct representing a two-dimensional point corresponding to
the center of a cell with indices `(i, j)` of the minutia `m`.
"""
struct Point_ij
    x::Float64
    y::Float64
end

"""
Calculate the angle associated with all cells at height `k`.
Returns the angle in radians dϕ/dk.
# Parameters:
- `k::Int64`: The height of the cell.
- `pms::Parameters`: The parameters of the algorithm.
"""
function get_angle_from_height(k::Int64; pms::Parameters=params)::Float64
    -π + (Float64(k) - 0.5) * pms.ΔD
end

"""
Calculate a two-dimensional point corresponding to the center of a cell
with indices `(i, j)` of the minutia `m`.
# Parameters:
- `i::Int64`: The index of the cell along the cuboid base.
- `j::Int64`: The index of the cell along the cuboid base.
- `m::Minutia`: The minutia feature.
- `pms::Parameters`: The parameters of the algorithm.
"""
function get_cell_center_point(i::Int64, j::Int64, m::Minutia; pms::Parameters=params)::Point_ij
    # TODO θ in the Minutia type should be a single Float64 value instead of a Vector{Float64}
    tf = [cos(m.θ[1]) sin(m.θ[1]); -sin(m.θ[1]) cos(m.θ[1])]
    tf = tf * [Float64(i)-(pms.NS+1)/2; Float64(j)-(pms.NS+1)/2] * pms.ΔS + [m.x; m.y]
    Point_ij(tf[1], tf[2])
end

"""
Calculate the euclidean distance between a minutia and a point.
# Parameters:
- `m::Minutia`: The minutia feature.
- `p::Point`: The point.
"""
function euclidean_distance(m::Minutia, p::Point_ij)::Float64
    sqrt((m.x - p.x)^2 + (m.y - p.y)^2)
end

"""
Calculate the spatial contribution that the minutia `m_t` gives to cell (i, j, k)
represented by the point `point`.
# Parameters:
- `m_t::Minutia`: The minutia feature.
- `point::Point`: The point representing the cell.
- `pms::Parameters`: The parameters of the algorithm.
"""
function spatial_contribution_minutia(m_t::Minutia, point::Point_ij; pms::Parameters=params)::Float64
    # Calculate the euclidean distance between the minutia and the point
    d = euclidean_distance(m_t, point)
    # Calculate the gaussian function of the euclidean distance
    exp(-d^2 / (2 * pms.σS^2)) / (sqrt(2π) * pms.σS)
end

"""
Compute the difference of two given angles.
# Parameters:
- `θ1::Float64`: The first angle.
- `θ2::Float64`: The second angle.
"""
function angles_difference(θ1::Float64, θ2::Float64)::Float64
    θ1 - θ2 >= -π && θ1 - θ2 < π && return θ1 - θ2
    θ1 - θ2 < -π && return θ1 - θ2 + 2π
    θ1 - θ2 >= π && return θ1 - θ2 - 2π
end

"""
Compute the directional contribution of the minutia `m_t` to the minutia `m`.
# Parameters:
- `m_t::Minutia`: The minutia feature.
- `m::Minutia`: The minutia feature.
- `angle::Float64`: The angle.
- `pms::Parameters`: The parameters of the algorithm.
"""
function directional_contribution_minutia(m_t::Minutia, m::Minutia, angle::Float64; pms::Parameters=params)::Float64
    # Calculate the directional difference between the two minutiae
    dθ = angles_difference(m.θ[1], m_t.θ[1])
    # Calculate the difference between the "angle" and the directional difference
    dΦ = angles_difference(angle, dθ)
    # Calculate the intergral: ∫exp(-t^2 / (2 * σD^2)) dt / (sqrt(2π) * σD) in the range [dΦ - ΔD/2, dΦ + ΔD/2]
    # TODO
end

"""
Check if a point is valid. This function implements ξ(m, p). Returns a boolean value.
# Parameters:
- `m::Minutia`: The minutia feature.
- `p::Point`: The point.
- `extended_hull::Matrix`: The convex hull extended by Ω pixels.
- `pms::Parameters`: The parameters of the algorithm.
"""
function isvalid(m::Minutia, p::Point_ij, extended_hull::Matrix; pms::Parameters=params)::Bool
    # If the euclidean distance between the minutia and the point is greater than R, 
    # then the point is invalid
    euclidean_distance(m, p) > pms.R && return false
    # If point p is within the extended convex hull, then the point is valid
    extended_hull[round(Int64, p.y), round(Int64, p.x)] && return true
    # Otherwise, the point is invalid
    false
end

"""
Calculate the neighborhood of a point p of the minutia m.
# Parameters:
- `m::Minutia`: The minutia feature.
- `p::Point`: The point.
- `T::Vector{Minutia}`: The set of minutiae of the image.
- `pms::Parameters`: The parameters of the algorithm.
"""
function calculate_neighborhood(m::Minutia, p::Point_ij, T::Vector{Minutia}; pms::Parameters=params)::Vector{Minutia}
    # Find all minutiae different from m and whose distance is less than 3σS
    filter(m_t -> m_t != m && euclidean_distance(m_t, p) < 3pms.σS, T)
end

"""
Calculate the entire contribution of the minutia m to the point p.
# Parameters:
- `m::Minutia`: The minutia feature.
- `p::Point`: The point.
- `N_p::Vector{Minutia}`: The neighborhood of the point p.
- `angle::Float64`: The angle.
- `extended_hull::Matrix`: The convex hull extended by Ω pixels.
- `pms::Parameters`: The parameters of the algorithm.
"""
function entire_contribution_minutia(
    m::Minutia, 
    p::Point_ij, 
    N_p::Vector{Minutia}, 
    angle::Float64,
    extended_hull::Matrix; 
    pms::Parameters=params)::Float64

    # Check if the point is valid
    isvalid(m, p, extended_hull) || return -1.0
    # Sum all the contributions of the minutiae in the neighborhood N_p of point p
    v = sum([spatial_contribution_minutia(m_t, p; pms) * directional_contribution_minutia(m_t, m, angle; pms) for m_t in N_p])
    # Return the sigmoid function of the sum
    return 1 / (1 + exp(-pms.τψ * (v - pms.μψ)))
end

"""
Compute the convex hull of the image, extends it by Ω pixels
perpendicular to each line segment, and fills the resulting
hull to create a boolean mask image.
# Arguments
- `image::Matrix`: The original image (just to get the size).
- `minutiae::Vector{Minutia}`: The minutiae of the image to 
    compute the convex hull of.
- `Ω::Int64`: The number of pixels to extend the convex hull by.
"""
function extended_convex_hull_image(image::Matrix, minutiae::Vector{Minutia}, Ω::Int64=20)::Matrix{Bool}
    minutiae_image = zeros(Bool, size(image))
    for minutia in minutiae
        minutiae_image[minutia.y, minutia.x] = true
    end

    c = convexhull(minutiae_image)
    push!(c, c[1])
    push!(c, c[2])

    new_convex_hull = []
    # Expand each line segment by Ω pixels
    for i in 1:length(c)-2
        A = c[i]
        B = c[i + 1]
        C = c[i + 2]
    
        u = A - B
        v = C - B
        u = [u[1], u[2]]
        v = [v[1], v[2]]
    
        # Normalize
        u = u / norm(u)
        v = v / norm(v)
        w = (u + v) / norm(u + v)

        # Expand the line segment
        newx, newy = B[2], B[1]
    
        # TODO Make it more efficient by considering
        # the intersection between the segment B,B+Δ
        # and the image boundaries
        height, width = size(image)
        mindist = min(newy, height - newy, newx, width - newx, Ω)
    
        for j in mindist:Ω
            Δx = round(Int, -j * w[2])
            Δy = round(Int, -j * w[1])
    
            newx2 = B[2] + Δx
            newy2 = B[1] + Δy
            
            (newx2 < 1) && break
            (newy2 < 1) && break
            (newx2 > size(image)[2]) && break
            (newy2 > size(image)[1]) && break
    
            newx, newy = newx2, newy2
        end
    
        push!(new_convex_hull, Point(newx, newy))
    end

    # Draw the convex hull image
    mask = zeros(Gray, size(image))
    # Draw the path on the mask
    draw!(mask, Polygon(new_convex_hull))

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

    Bool.(mask)
end
