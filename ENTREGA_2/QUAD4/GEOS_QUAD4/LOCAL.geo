SetFactory("OpenCASCADE");

// Creo puntos
Point(1) = {0.0, 0.0, 0.0, 1.0};
Point(2) = {800.0, 0.0, 0.0, 1.0};
Point(3) = {1000.0, 200.0, 0.0, 1.0};
Point(4) = {1000.0, 1000.0, 0.0, 1.0};
Point(5) = {0.0, 1000.0, 0.0, 1.0};
Point(6) = {0, 200, 0, 1};
Point(7) = {400, 200, 0, 1};
Point(8) = {400, 1000, 0, 1};
Point(9) = {400, 0, 0, 1};


// Ahora puedo unir los puntos con lineas externas
Line(1) = {1,9};
Line(2) = {9,2};
Line(3) = {2,3};
Line(4) = {3,4};
Line(5) = {4,8};
Line(6) = {8,5};
Line(7) = {5,6};
Line(8) = {6,1};

// Lineas internas
Line(9) = {6,7};
Line(10) = {7,3};
Line(11) = {9,7};
Line(12) = {7,8};

// Genera loop
Line Loop(1) = {1,11,8,9};
Line Loop(2) = {2,3,10,11};
Line Loop(3) = {10,4,5,12};
Line Loop(4) = {9,12,6,7};

// Creo superficies
Plane Surface(1) = {1};
Plane Surface(2) = {2};
Plane Surface(3) = {3};
Plane Surface(4) = {4};

// Hago transfinita cada superficie (Local refinment)
// Longitudes de curvas (mm)

L3  = Sqrt((1000-800)*(1000-800) + (200-0)*(200-0));
L_tot = 1000; //total
h = 2; //Altura elemento: 2 - 1 - 0.5 - 0.25
b = 20;
h_h = 1.5; //h para refinar linea 3

N = L_tot/h;
h1 = (N*800)/1000;
h2 = (N*200)/1000;

B = 1000/b;

b1 = (B*400)/1000;
b2 = (B*600)/1000;

m3  = L3/h_h;    // refinamiento solo curva 3

Transfinite Curve {1, 9, 6} = b1;
Transfinite Curve {2,10,5} =  b2; 
Transfinite Curve {7,12,4} =  h1; //Altura Superior, jugar con este
Transfinite Curve {8,11,3} =  m3; //Altura inferior, jugar con este

Transfinite Surface {1};
Transfinite Surface {2};
Transfinite Surface {3};
Transfinite Surface {4};

// Defino las superficies fisicas
Physical Surface(1) = {1};
Physical Surface(2) = {2};
Physical Surface(3) = {3};
Physical Surface(4) = {4};

// Restricciones

Physical Line("Fuerza_Y_1") = {7};
Physical Line("Fuerza_Y_2") = {8};
Physical Line("Restriccion") = {4};