%%% This script is to Plot mode shapes
%The script was writen by Prof. Dr. Eleni Chatzi

clear all; close all;clc
pwd
addpath('Source');
% Define the surface range
surfaceRange = [0, 5, 0, 2.015];

% Define the point coordinates (Nx2 array) and use also the edges of the
% surface
BCs = [0,0;0,2.015;5,0;5,2.015];
points = [BCs;2.9425, 0; 3.6925, 2.015; 2.9425, 2.015; 2.1925, 2.015];

% Define the vertical deformations at these points
deformations = [zeros(size(BCs,1),1);
     -1; 
    0.007799045; 
    0.01553062; 
    -0.047086716]; % Example deformation values

% Define the resolution [x_resolution, y_resolution]
resolution = [100, 100];

% Call the function
plotSurfaceDeformation(surfaceRange, points, deformations, resolution);