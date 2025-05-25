//-----------------------------------------------------------------------------
// 0) PARÁMETROS DE MALLA
//-----------------------------------------------------------------------------
h_max = 2.0;        // tamaño de elemento global [mm]
// para refinamiento local:
h_min = h_max/10;   // tamaño mínimo cerca del punto 7
r1    =  50.0;      // radio interior de refinamiento [mm]
r2    = 100.0;      // radio exterior de transición  [mm]

//-----------------------------------------------------------------------------
// 1) GEOMETRÍA
//-----------------------------------------------------------------------------
SetFactory("OpenCASCADE");

// Creo puntos (el cuarto parámetro es la characteristic length inicial)
Point(1) = {0.0,   0.0,   0.0,  h_max};
Point(2) = {800.0, 0.0,   0.0,  h_max};
Point(3) = {1000.0,200.0,  0.0,  h_max};
Point(4) = {1000.0,1000.0, 0.0,  h_max};
Point(5) = {0.0,   1000.0, 0.0,  h_max};
Point(6) = {0.0,   200.0,  0.0,  h_max};
Point(7) = {400.0, 200.0,  0.0,  h_max};
Point(8) = {400.0, 1000.0, 0.0,  h_max};
Point(9) = {400.0, 0.0,    0.0,  h_max};

// Líneas exteriores
Line(1) = {1,9};   Line(2)  = {9,2};
Line(3) = {2,3};   Line(4)  = {3,4};
Line(5) = {4,8};   Line(6)  = {8,5};
Line(7) = {5,6};   Line(8)  = {6,1};
// Líneas internas
Line(9)  = {6,7};  Line(10) = {7,3};
Line(11) = {9,7};  Line(12) = {7,8};

// Loops y superficies
Line Loop(1) = {1,11,8,9};
Line Loop(2) = {2,3,10,11};
Line Loop(3) = {10,4,5,12};
Line Loop(4) = {9,12,6,7};
Plane Surface(1) = {1};
Plane Surface(2) = {2};
Plane Surface(3) = {3};
Plane Surface(4) = {4};

// Transfinite (opcional, si quieres mallado estructurado)
m = 10;
Transfinite Curve {1, 9, 6}    = m;
Transfinite Curve {2,10,5}     = m;
Transfinite Curve {7,12,4}     = m;
Transfinite Curve {8,11,3}     = m/2;
Transfinite Surface {1,2,3,4};

// Física
Physical Surface(1) = {1};
Physical Surface(2) = {2};
Physical Surface(3) = {3};
Physical Surface(4) = {4};
Physical Line("Fuerza_Y_1") = {7};
Physical Line("Fuerza_Y_2") = {8};
Physical Line("Restriccion")  = {4};

//-----------------------------------------------------------------------------
// 2) REFINAMIENTO LOCAL POR CAMPO (ahora sobre la línea 3)
//-----------------------------------------------------------------------------
//Field[1] = Distance;
//Field[1].CurvesList = {3};   // usar la curva 3 → segmento entre punto 2 y 3
//Field[1].Sampling   = 100;

//Field[2] = Threshold;
//Field[2].InField   = 1;
//Field[2].SizeMin   = h_min;  // tamaño mínimo en la proximidad de la línea
//Field[2].SizeMax   = h_max;  // tamaño máximo fuera de la zona
//Field[2].DistMin   = r1;     // dentro de r1 desde la línea → h_min
//Field[2].DistMax   = r2;     // fuera de r2 desde la línea → h_max

//Background Field = 2;

//Comando para mallado global
// comenta el bloque Field[1]…Background Field
Mesh.CharacteristicLengthMax = h_max;
Mesh.CharacteristicLengthMin = h_max;


//-----------------------------------------------------------------------------
// 3) MESH
//-----------------------------------------------------------------------------
//Mesh 2;
