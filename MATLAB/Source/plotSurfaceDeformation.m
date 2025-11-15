function plotSurfaceDeformation(surfaceRange, points, deformations, resolution)
    % plotSurfaceDeformationFromPoints: Generates contour and 3D plots of surface deformation.
    %
    % Inputs:
    %   - surfaceRange : [x_min, x_max, y_min, y_max], limits of the surface
    %   - points : Nx2 array of (x, y) coordinates defining the deformation points
    %   - deformations : Nx1 array of vertical deformations at the given points
    %   - resolution : [x_res, y_res], number of points in the x and y directions
    %
    % Outputs:
    %   - None (Plots are displayed)
    %
    % Example Usage:
    %   surfaceRange = [0, 5, 0, 2.015];
    %   points = [2.9425, 0; 3.6925, 2.015; 2.9425, 2.015; 2.1925, 2.015];
    %   deformations = [0.5, -0.3, 0.4, -0.2]; % Vertical deformations
    %   resolution = [100, 100];
    %   plotSurfaceDeformationFromPoints(surfaceRange, points, deformations, resolution);

    % Parse surface range and resolution
    x_min = surfaceRange(1); x_max = surfaceRange(2);
    y_min = surfaceRange(3); y_max = surfaceRange(4);
    x_res = resolution(1); y_res = resolution(2);

    % Create grid
    x = linspace(x_min, x_max, x_res);
    y = linspace(y_min, y_max, y_res);
    [X, Y] = meshgrid(x, y);

    % Interpolate the deformation across the grid
    F = scatteredInterpolant(points(:, 1), points(:, 2), deformations, 'natural', 'none');
    Z = F(X, Y); % Interpolated deformation values

    % Plot contour plot
    figure;
    contourf(X, Y, Z, 20, 'LineColor', 'none'); % Filled contour plot
    colorbar; % Add color bar
    title('Contour Plot of Surface Deformation');
    xlabel('x (m)');
    ylabel('y (m)');

    % Plot 3D surface plot
    figure;
    surf(X, Y, Z, 'EdgeColor', 'none'); % 3D surface plot
    colorbar; % Add color bar
    title('3D Plot of Surface Deformation');
    xlabel('x (m)');
    ylabel('y (m)');
    zlabel('Deflection (m)');
    view(3); % Set 3D view
end
