SetFactory("OpenCASCADE");

// Creo puntos
Point(1) = {0.0, 0.0, 0.0, 1.0};
Point(2) = {800.0, 0.0, 0.0, 1.0};
Point(3) = {1000.0, 200.0, 0.0, 1.0};
Point(4) = {1200.0, 0.0, 0.0, 1.0};
Point(5) = {2000.0, 0.0, 0.0, 1.0};
Point(6) = {2000, 200, 0, 1};
Point(7) = {2000.0, 1000.0, 0.0, 1.0};
Point(8) = {1200, 1000.0, 0.0, 1.0};
Point(9) = {1000, 1000, 0, 1};
Point(10) = {800, 1000, 0, 1};
Point(11) = {0.0, 1000.0, 0.0, 1.0};
Point(12) = {0, 200, 0, 1};



// Ahora puedo unir los puntos con lineas
Line(1) = {1,2};
Line(2) = {2,3};
Line(3) = {3,4};
Line(4) = {4,5};
Line(5) = {5,6};
Line(6) = {6,7};
Line(7) = {7,8};
Line(8) = {8,9};
Line(9) = {9,10};
Line(10) = {10,11};
Line(11) = {11,12};
Line(12) = {12,1};

// Genera loop
Line Loop(1) = {1,2,3,4,5,6,7,8,9,10,11,12};

// Creo superficies
Plane Surface(1) = {1};

// Grupos fisicos
Physical Surface(1) = {1};

// Condiciones de borde
Physical Line("F1") = {7};
Physical Line("F2") = {5};
Physical Point(8) = {8};
Physical Point(3) = {3};

//+
Transfinite Curve {12, 2} = 5 Using Progression 1;
//+
Transfinite Curve {2, 9} = 5 Using Progression 1;
//+
Transfinite Curve {5, 3} = 5 Using Progression 1;
//+
Transfinite Curve {3, 8} = 5 Using Progression 1;
//+
Transfinite Curve {1, 10} = 5 Using Progression 1;
//+
Transfinite Curve {4, 7} = 5 Using Progression 1;
//+
Transfinite Curve {6, 11} = 5 Using Progression 1;
