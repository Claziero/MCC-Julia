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
