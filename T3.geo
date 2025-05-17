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
Point(111) = {400, 200, 0, 1};
Point(112) = {1600, 200, 0, 1};
Point(113) = {1600, 1000, 0, 1};
Point(114) = {400, 1000, 0, 1};
Point(115) = {800, 200, 0, 1};
Point(116) = {1200, 200, 0, 1};
Point(117) = {400, 0, 0, 1};
Point(118) = {1600, 0, 0, 1};


// Ahora puedo unir los puntos con lineas
Line(1) = {1,117};
Line(2) = {117,2};
Line(3) = {2,3};
Line(4) = {3,4};
Line(5) = {4,118};
Line(6) = {118,5};
Line(7) = {5,6};
Line(8) = {6,7};
Line(9) = {7,113};
Line(10) = {113,8};
Line(11) = {8,9};
Line(12) = {9,10};
Line(13) = {10,114};
Line(14) = {114,11};
Line(15) = {11, 12};
Line(16) = {12, 1};

// Agrego lineas interiores Line(punto_ipunto_f)
Line(12111) = {12,111};
Line(111115) = {111,115};
Line(1153) = {115,3};
Line(3116) = {3,116};
Line(116112) = {116,112};
Line(1126) = {112,6};
Line(114111) = {114,111};
Line(10115) = {10,115};
Line(93) = {9,3};
Line(8116) = {8,116};
Line(113112) = {113,112};
Line(1152) = {115,2};
Line(1164) = {116,4};
Line(111117) = {111,117};
Line(112118) = {112,118};


// Genera loop
Line Loop(1) = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};

// Creo superficies
Plane Surface(1) = {1};

// Grupos fisicos
Physical Surface(1) = {1};

// Condiciones de borde
Physical Line("F1") = {7};
Physical Line("F2") = {5};
Physical Point(8) = {8};
Physical Point(3) = {3};

// Crear curvas transfinitas
l1 = 10;
l2 = 10;
l3 = 10;
l4 = 10;
l5 = 10;
l6 = 10;
l7 = 10;
l8 = 10;
l9 = 10;
l10 = 10;
l11 = 10;
l12 = 10;

//+
Transfinite Curve {16, 111117} = l1 Using Progression 1;
//+
Transfinite Curve {1, 12111} = l1 Using Progression 1; //De aqui para arriba cuadrado inf_iz_ext
//+
Transfinite Curve {111117, 1152} = l2 Using Progression 1;
//+
Transfinite Curve {2, 111115} = l2 Using Progression 1; //De aqui para arriba cuadrado inf_iz_medio
//+
Transfinite Curve {1152, 3} = l3 Using Progression 1;
//+
Transfinite Curve {3, 1153} = l3 Using Progression 1; //De aqui para arriba triangulo inf_iz_int
//+
Transfinite Curve {7, 112118} = l4 Using Progression 1; 
//+
Transfinite Curve {6, 1126} = l4 Using Progression 1; //De aqui para arriba cuadrado inf_der_ext
//+
Transfinite Curve {112118, 1164} = l5 Using Progression 1;
//+
Transfinite Curve {5, 116112} = l5 Using Progression 1; //De aqui para arriba cuadrado inf_der_medio
//+
Transfinite Curve {1164, 4} = l6 Using Progression 1;
//+
Transfinite Curve {4, 3116} = l6 Using Progression 1; //De aqui para arriba triangulo inf_der_int
//+
Transfinite Curve {15, 114111} = l7 Using Progression 1;
//+
Transfinite Curve {12111, 14} = l7 Using Progression 1; //De aqui para arriba cuadrado sup_iz_ext
//+
Transfinite Curve {114111, 10115} = l8 Using Progression 1;
//+
Transfinite Curve {111115, 13} = l8 Using Progression 1; //De aqui para arriba cuadrado sup_iz_medio
//+
Transfinite Curve {10115, 93} = l9 Using Progression 1;
//+
Transfinite Curve {1153, 12} = l9 Using Progression 1; //De aqui para arriba cuadrado sup_iz_int
//+
Transfinite Curve {8, 113112} = l10 Using Progression 1;
//+
Transfinite Curve {1126, 9} = l10 Using Progression 1; //De aqui para arriba cuadrado sup_der_ext
//+
Transfinite Curve {113112, 8116} = l11 Using Progression 1;
//+
Transfinite Curve {116112, 10} = l11 Using Progression 1; //De aqui para arriba cuadrado sup_der_medio
//+
Transfinite Curve {8116, 93} = l12 Using Progression 1; 
//+
Transfinite Curve {3116, 11} = l12 Using Progression 1; //De aqui para arriba cuadrado sup_der_int
