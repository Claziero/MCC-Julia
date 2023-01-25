"""
Constants for the MCC algorithm:
- `Ω`: Number of pixels to expand the convex hull of
- `R`: Radius of the cylinder
- `NS`: Number of cells along one direction of the cuboid base enclosing the cylinder
- `ND`: Number of cells along the cuboid height enclosing the cylinder
- `σS`: Spatial standard deviation 
- `σD`: Directional standard deviation
- `μψ`: Mean of the sigmoid function
- `τψ`: Parameter of the sigmoid function
- `minVC`: Minimum number of cells in a minutia cylinder
- `minM`: Minimum number of minutiae contributing to the cylinder
- `minME`: Minimum number of matching elements between two cylinders
- `δθ`: Maximum directional difference between two minutiae
- `ΔS`: Cell size along one direction of the cuboid base
- `ΔD`: Cell size along the cuboid height
"""
struct Parameters
    Ω::Int64
    R::Int64
    NS::Int64
    ND::Int64
    σS::Float64
    σD::Float64
    μψ::Float64
    τψ::Float64
    minVC::Int64
    minM::Int64
    minME::Int64
    δθ::Float64
    ΔS::Float64
    ΔD::Float64
end

function Parameters(;
    Ω =     20,
    R =     80,
    NS =    16,
    ND =    6,
    σS =    12,
    σD =    2π/9,
    uψ =    0.01,
    tψ =    400,
    minVC = NS * NS * ND ÷ 3,
    minM =  5,
    minME = 5,
    δθ =    2π/3)
    Parameters(
        Ω,
        R,
        NS,
        ND,
        σS,
        σD,
        uψ,
        tψ,
        minVC,
        minM,
        minME,
        δθ,
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

    Point_ij(p::Vector{Float64}) = new(p[1], p[2])
end

""" Struct representing a cylinder with its associated minutia. """
struct Cylinder
    minutia::Minutia
    cuboid::Array{Float64, 3}

    Cylinder(m::Minutia, c::Array{Float64, 3}) = new(m, c)
end

""" Compute the angle associated with all cells at height `k`. """
@inline height2angle(k::Int64; pms::Parameters=params) = -π + (Float64(k) - 0.5) * pms.ΔD

"""
Compute a two-dimensional point corresponding to the center of a cell
with indices `(i, j)` of the minutia `m`.
# Parameters:
- `i::Int64`: The index of the cell along the cuboid base.
- `j::Int64`: The index of the cell along the cuboid base.
- `m::Minutia`: The minutia feature.
"""
function center(i::Int64, j::Int64, m::Minutia; pms::Parameters=params)::Point_ij
    # TODO θ in the Minutia type should be a single Float64 value instead of a Vector{Float64}
    θ = m.θ[1]
    rot = [cos(θ) sin(θ); -sin(θ) cos(θ)]
    tf = rot * [i-(pms.NS+1)/2; j-(pms.NS+1)/2] * pms.ΔS + [m.x; m.y]
    Point_ij(tf)
end

"""
Compute the euclidean distance between a minutia and a point
or between two minutiae. 
"""
@inline dist(a::Minutia, b::Point_ij) = sqrt((a.x - b.x)^2 + (a.y - b.y)^2)
@inline dist(a::Minutia, b::Minutia) = sqrt((a.x - b.x)^2 + (a.y - b.y)^2)

"""
Compute the spatial contribution that the minutia `m_t` gives to cell (i, j, k)
represented by the point `point`.
# Parameters:
- `m_t::Minutia`: The minutia feature.
- `point::Point_ij`: The point representing the cell.
"""
function spatial_contribution(m_t::Minutia, point::Point_ij; pms::Parameters=params)::Float64
    d = dist(m_t, point)
    # Weigh the distance with a gaussian function
    exp(-d^2 / (2 * pms.σS^2)) / (sqrt(2π) * pms.σS)
end

""" Compute the difference of two given angles `θ1` and `θ2`. """
function angles_difference(θ1::Float64, θ2::Float64)::Float64
    -π <= θ1 - θ2 < π && return θ1 - θ2
    θ1 - θ2 < -π && return θ1 - θ2 + 2π
    θ1 - θ2 >= π && return θ1 - θ2 - 2π
end

"""
Compute the directional contribution of the minutia `m_t` to the minutia `m`.
# Parameters:
- `m_t::Minutia`: The minutia feature.
- `m::Minutia`: The minutia feature.
- `angle::Float64`: The angle at the height of the cylinder.
"""
function directional_contribution(m_t::Minutia, m::Minutia, angle::Float64; pms::Parameters=params)::Float64
    dθ = angles_difference(m.θ[1], m_t.θ[1])
    dΦ = angles_difference(angle, dθ)
    # Compute the intergral: 
    # ∫exp(-t^2 / (2 * σD^2)) dt / (sqrt(2π) * σD) in the range [dΦ - ΔD/2, dΦ + ΔD/2]
    A = pms.ΔD / 2
    B = √2 * pms.σD

    0.5(erf((dΦ + A) / B) - erf((dΦ - A) / B))
end

"""
Check if a point is valid. This function implements ξ(m, p). Returns a boolean value.
# Parameters:
- `m::Minutia`: The minutia feature.
- `p::Point_ij`: The point.
- `extended_hull::Matrix`: The convex hull extended by Ω pixels.
"""
function isvalid(m::Minutia, p::Point_ij, extended_hull::Matrix; pms::Parameters=params)::Bool
    dist(m, p) > pms.R && return false

    # If the point is within the convex hull image, it is valid
    x = round(Int64, p.x)
    y = round(Int64, p.y)
    h, w = size(extended_hull)
    x >= 1 && x <= w && y >= 1 && y <= h && extended_hull[y, x]
end

"""
Compute the convex hull of the image, extends it by Ω pixels perpendicular to each
line segment, and fills the resulting hull to create a boolean mask image.
# Arguments
- `image::Matrix`: The original image (just to get the size).
- `minutiae::Vector{Minutia}`: The minutiae of the image to 
    compute the convex hull of.
"""
function extended_convex_hull_image(image::Matrix, minutiae::Vector{Minutia}; pms::Parameters=params)::Matrix{Bool}
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
        mindist = min(newy, height - newy, newx, width - newx, pms.Ω)
    
        for j in mindist:pms.Ω
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

"""
Create a cylinder for the minutia `m`.
# Parameters:
- `hull::Matrix`: The convex hull of the image.
- `m::Minutia`: The minutia feature.
- `minutiae::Vector{Minutia}`: The set of minutiae of the image.
"""
function cylinder(hull::Matrix, m::Minutia, minutiae::Vector{Minutia}; pms::Parameters=params)::Cylinder
    cyl = zeros(pms.NS, pms.NS, pms.ND)

    for i in 1:pms.NS, j in 1:pms.NS
        # Compute the point and its neighborhood N_p of minutiae
        p = center(i, j, m; pms)
        N_p = filter(m_t -> m_t != m && dist(m_t, p) < 3pms.σS, minutiae)

        for k in 1:pms.ND
            # The contribution for this cell is set to -1 (invalid) if the point is not valid
            if !isvalid(m, p, hull) 
                cyl[i, j, k] = -1.0
                continue
            end
            
            # Else compute the contributions of the minutiae in the neighborhood N_p of point p
            angle = height2angle(k; pms)
            v = sum([spatial_contribution(m_t, p; pms) * directional_contribution(m_t, m, angle; pms) for m_t in N_p])
            cyl[i, j, k] = 1 / (1 + exp(-pms.τψ * (v - pms.μψ)))
        end
    end

    # Return the cylinder
    Cylinder(m, cyl)
end

"""
Create the cylinder set for the image.
# Parameters:
- `image::Matrix`: The original image.
- `minutiae::Vector{Minutia}`: The set of minutiae of the image.
"""
function cylinder_set(image::Matrix, minutiae::Vector{Minutia}; pms::Parameters=params)::Vector{Cylinder}
    hull = extended_convex_hull_image(image, minutiae; pms)

    set = []
    for min in minutiae
        # Check if the cylinder is invalid
        count(m -> m != min && dist(m, min) <= pms.R + 3pms.σS, minutiae) < pms.minM && continue
        cyl = cylinder(hull, min, minutiae; pms)
        sum(cyl.cuboid .== -1) < pms.minVC && continue

        push!(set, cyl)
    end

    set
end

""" Linearize the cylinder cell indices. """
@inline linearize(i::Int64, j::Int64, k::Int64; pms::Parameters=params) = (k - 1)pms.NS^2 + (j - 1)pms.NS + i

"""
Vectorize the cylinder `cylinder`.
# Parameters:
- `cylinder::Array{Float64, 3}`: The cylinder to vectorize.
"""
function vectorize(cylinder::Array{Float64, 3}; pms::Parameters=params)::Vector{Float64}
    vec = zeros(pms.NS^2 * pms.ND)
    for i in 1:pms.NS, j in 1:pms.NS, k in 1:pms.ND
        vec[linearize(i, j, k; pms)] = cylinder[i, j, k]
    end
    vec
end

"""
Compute the vectors c_a|b and c_b|a necessary to compute similarity.
# Parameters:
- `c_a::Vector{Float64}`: The vectorized cylinder of minutia a.
- `c_b::Vector{Float64}`: The vectorized cylinder of minutia b.
"""
function aux_vectors(c_a::Vector{Float64}, c_b::Vector{Float64})::Tuple{Vector{Float64}, Vector{Float64}, Int64}
    c_ab = c_ba = zeros(length(c_a))
    count = 0

    for t in 1:length(c_a)
        if c_a[t] != -1 && c_b[t] != -1
            c_ab[t] = c_a[t]
            c_ba[t] = c_b[t]
            count += 1
        else
            c_ab[t] = c_ba[t] = 0
        end
    end

    (c_ab, c_ba, count)
end

"""
Compute the similarity between two cylinders. Returns a value in [0, 1].
# Parameters:
- `Cyl_a::Cylinder`: The first cylinder.
- `Cyl_b::Cylinder`: The second cylinder.
"""
function similarity(Cyl_a::Cylinder, Cyl_b::Cylinder; pms::Parameters=params)::Float64
    # Compute the vectorized cylinders
    vec_a = vectorize(Cyl_a.cuboid; pms)
    vec_b = vectorize(Cyl_b.cuboid; pms)

    # Check if the cylinders are matchable
    # TODO Remove the "[1]" when the Minutia struct is updated
    angles_difference(vec_a.minutia.θ[1], vec_b.minutia.θ[1]; pms) > pms.δθ && return 0
    
    c_ab, c_ba, count = aux_vectors(vec_a, vec_b)
    count < pms.minME && return 0

    norm_ab = norm(c_ab)
    norm_ba = norm(c_ba)
    norm_ab + norm_ba == 0 && return 0
    
    # Compute the similarity
    1 - (norm(c_ab - c_ba) / (norm_ab + norm_ba))
end
