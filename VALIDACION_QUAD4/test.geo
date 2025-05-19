SetFactory("OpenCASCADE");

Point(1) = {0, 0, 0, 1.0};
Point(2) = {1000, 0, 0, 1.0};
Point(3) = {1000, 100, 0, 1.0};
Point(4) = {0, 100, 0, 1.0};

Line(1) = {1,2};
Line(2) = {2,3};
Line(3) = {3,4};
Line(4) = {4,1};

Line Loop(1) = {1,2,3,4};
Plane Surface(1) = {1};
Physical Surface(1) = {1};

Physical Line('Restr XY') = {4};
Physical Line ('Fuerza') = {2};