"""
Plots the cylinder of the minutia `cyl_num` in the list `cyl_list`.
Highlights the point (`hi`, `hj`) in the cylinder.
# Parameters:
- `cyl_num::Int64`: The index of the cylinder to plot.
- `cyl_list::Vector{Cylinder}`: The list of cylinders.
- `hi::Int64`: The i coordinate of the point to highlight.
- `hj::Int64`: The j coordinate of the point to highlight.
"""
function plot_cylinder(cyl_num::Int64, cyl_list::Vector{Cylinder}, hi=8, hj=8; pms::Parameters=params)::Matrix{RGB}
    @assert cyl_num in 1:length(cyl_list)
    @assert hi in 1:pms.NS
    @assert hj in 1:pms.NS

    # This method returns a diagram image
    diagram = ones(RGB, 900, 1800)
    # Diagram border
    BORDER = 50
    # Offsets around the center minutia to include in the slice
    # The offset refers to all directions
    OFFSET = 100
    # Size of the image slice drawn on the diagram
    SLICE_SIZE = 800
    # Radius of the circle drawn around the each minutia
    RADIUS = 12
    # Cylinder slice size
    CYLINDER_SIZE = 240

    # Get the minutiae list from the list of cylinders
    min_list = []
    for cyl in cyl_list
        push!(min_list, cyl.minutia)
    end

    # Center minutia to show in the image slice
    min = min_list[cyl_num]

    # Get the minutiae to display that are part of the slice
    inoffset = filter(m -> m.x in (min.x-OFFSET):(min.x+OFFSET) && m.y in (min.y-OFFSET):(min.y+OFFSET), min_list)
    inputrangex = min.x-OFFSET:min.x+OFFSET
    inputrangey = min.y-OFFSET:min.y+OFFSET
    outputrange = BORDER:SLICE_SIZE+BORDER

    # Draw the border of the slice
    draw!(diagram, RectanglePoints(Point(BORDER, BORDER), Point(SLICE_SIZE + BORDER, SLICE_SIZE + BORDER)), RGB(0))
    
    rmap(x, r1::UnitRange, r2::UnitRange) = r2[1] + (x - r1[1]) * (r2[end] - r2[1]) / (r1[end] - r1[1])
    function pmap(px, py)
        x = rmap(px, inputrangex, outputrange)
        y = rmap(py, inputrangey, outputrange)
        x = round(Int, x)
        y = round(Int, y)
        Point(x, y)
    end

    θ = min.θ[1]
    rot = [cos(θ) sin(θ); -sin(θ) cos(θ)]

    begin
        point = rot * [hi-(pms.NS+1)/2, hj-(pms.NS+1)/2] * pms.ΔS + [min.x, min.y]
        point = pmap(point[1], point[2])
        # Scale 3σS to the diagram
        scale = SLICE_SIZE / (inputrangex[end] - inputrangex[1])
        radius = 3pms.σS * scale

        # Draw the 3σS Radius
        draw!(diagram, CirclePointRadius(point, radius), RGB(0, 1, 1))

        # Draw a circle around it
        draw!(diagram, CirclePointRadius(point, 5), RGB(1, 0, 0))
        draw!(diagram, CirclePointRadius(point, 3), RGB(0))
    end


    # Draw the cylinder slices
    for i in 1:pms.NS, j in 1:pms.NS
        # Quit if you're not inside of the circle
        (i-(pms.NS+1)/2)^2 + (j-(pms.NS+1)/2)^2 > (pms.NS/2+1/2)^2 && continue

        point1 = rot * [i-(pms.NS+2)/2, j-(pms.NS+2)/2] * pms.ΔS + [min.x, min.y]
        point2 = point1 + rot * [pms.ΔS, 0]
        point3 = point1 + rot * [0, pms.ΔS]
        point4 = point1 + rot * [pms.ΔS, pms.ΔS]

        point1 = pmap(point1[1], point1[2])
        point2 = pmap(point2[1], point2[2])
        point3 = pmap(point3[1], point3[2])
        point4 = pmap(point4[1], point4[2])

        # Draw the cylinder slice
        draw!(diagram, LineSegment(point1, point2), RGB(0))
        draw!(diagram, LineSegment(point1, point3), RGB(0))
        draw!(diagram, LineSegment(point4, point2), RGB(0))
        draw!(diagram, LineSegment(point4, point3), RGB(0))
    end

    # Draw each minutia in the diagram
    for m in inoffset
        # Convert the coordinates to the coordinates in the diagram
        x = rmap(m.x, inputrangex, outputrange)
        y = rmap(m.y, inputrangey, outputrange)

        color = m === min ? RGB(1, 0, 0) : RGB(0, 0, 1)

        # Draw the direction around each minutia as a line
        angle = -m.θ[1]
        dx = round(Int, cos(angle) * 3RADIUS)
        dy = round(Int, sin(angle) * 3RADIUS)
        draw!(diagram, LineSegment(Point(x, y), Point(x + dx, y + dy)), color)
        
        # Draw the minutia as a circle
        draw!(diagram, CirclePointRadius(Point(x, y), RADIUS), RGB(0))
        draw!(diagram, CirclePointRadius(Point(x, y), RADIUS - 2), color)
    end

    cylinder = cyl_list[cyl_num].cuboid

    # Create an image for each cylinder
    cylimgs = []
    for k in 1:pms.ND
        img = ones(RGB, pms.NS, pms.NS)
        for j in 1:pms.NS, i in 1:pms.NS
            if cylinder[i, j, k] > 0
                img[i, j] = RGB(cylinder[i, j, k])
            else
                img[i, j] = RGB(1)
            end
        end

        img15times = zeros(RGB, 15pms.NS, 15pms.NS)
        for j in 1:15pms.NS, i in 1:15pms.NS
            img15times[i, j] = img[ceil(Int, i/15), ceil(Int, j/15)]
        end

        push!(cylimgs, img15times)
    end

    # Draw the cylinder images
    for k in 1:pms.ND
        # Draw them in this order
        # 1 2 3
        # 4 5 6
        vindex = ceil(Int, k / 3)
        hindex = (k - 1) % 3 + 1

        # Draw the cylinder image
        x = 2BORDER + SLICE_SIZE + 10 + (hindex - 1) * (CYLINDER_SIZE + 10)
        y = (SLICE_SIZE - 2CYLINDER_SIZE) ÷ 2 + (vindex - 1) * (CYLINDER_SIZE + 10)

        h, w = size(cylimgs[k])
        diagram[y:y+h-1, x:x+w-1] .= cylimgs[k]


        # Find hi and hj on the printed image
        hi2 = rmap(hj, 1:pms.NS, x:x+15pms.NS)
        hj2 = rmap(hi, 1:pms.NS, y:y+15pms.NS)

        # Draw a circle around the hi and hj
        draw!(diagram, CirclePointRadius(Point(hi2, hj2), 5), RGB(0.5, 1, 0))
    end

    diagram
end