import numpy as np
import matplotlib.pyplot as plt
import gmsh
import os
import math
import sys

gmsh.initialize()
gmsh.model.add("T3")

gmsh.model.setCurrent("OpenCASCADE")

# Crear puntos
points = {
    1: (0.0, 0.0, 0.0),
    2: (800.0, 0.0, 0.0),
    3: (1000.0, 200.0, 0.0),
    4: (1200.0, 0.0, 0.0),
    5: (2000.0, 0.0, 0.0),
    6: (2000.0, 200.0, 0.0),
    7: (2000.0, 1000.0, 0.0),
    8: (1200.0, 1000.0, 0.0),
    9: (1000.0, 1000.0, 0.0),
    10: (800.0, 1000.0, 0.0),
    11: (0.0, 1000.0, 0.0),
    12: (0.0, 200.0, 0.0),
    111: (400.0, 200.0, 0.0),
    112: (1600.0, 200.0, 0.0),
    113: (1600.0, 1000.0, 0.0),
    114: (400.0, 1000.0, 0.0),
    115: (800.0, 200.0, 0.0),
    116: (1200.0, 200.0, 0.0),
    117: (400.0, 0.0, 0.0),
    118: (1600.0, 0.0, 0.0),
}

for tag, coords in points.items():
    gmsh.model.geo.addPoint(*coords, 1.0, tag)

# Crear líneas exteriores
lines = {
    1: (1, 117),
    2: (117, 2),
    3: (2, 3),
    4: (3, 4),
    5: (4, 118),
    6: (118, 5),
    7: (5, 6),
    8: (6, 7),
    9: (7, 113),
    10: (113, 8),
    11: (8, 9),
    12: (9, 10),
    13: (10, 114),
    14: (114, 11),
    15: (11, 12),
    16: (12, 1),
}

for tag, (start, end) in lines.items():
    gmsh.model.geo.addLine(start, end, tag)

# Crear líneas interiores
inner_lines = {
    12111: (12, 111),
    111115: (111, 115),
    1153: (115, 3),
    3116: (3, 116),
    116112: (116, 112),
    1126: (112, 6),
    114111: (114, 111),
    10115: (10, 115),
    93: (9, 3),
    8116: (8, 116),
    113112: (113, 112),
    1152: (115, 2),
    1164: (116, 4),
    111117: (111, 117),
    112118: (112, 118),
}

for tag, (start, end) in inner_lines.items():
    gmsh.model.geo.addLine(start, end, tag)

# Crear loop con las líneas exteriores (en orden)
loop_lines = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
gmsh.model.geo.addCurveLoop(loop_lines, 1)

# Crear superficie plana a partir del loop
gmsh.model.geo.addPlaneSurface([1], 1)

# Grupos físicos (Physical Surface y Lines)
gmsh.model.addPhysicalGroup(2, [1], 1)  # superficie física 1

gmsh.model.addPhysicalGroup(1, [7], name="F1")
gmsh.model.addPhysicalGroup(1, [5], name="F2")
gmsh.model.addPhysicalGroup(0, [8], 8)
gmsh.model.addPhysicalGroup(0, [3], 3)

# Curvas transfinita con número de divisiones (igual a los l1, l2, ...)
transfinite_lines = {
    (16, 111117): 10,
    (1, 12111): 10,
    (111117, 1152): 10,
    (2, 111115): 10,
    (1152, 3): 10,
    (3, 1153): 10,
    (7, 112118): 10,
    (6, 1126): 10,
    (112118, 1164): 10,
    (5, 116112): 10,
    (1164, 4): 10,
    (4, 3116): 10,
    (15, 114111): 10,
    (12111, 14): 10,
    (114111, 10115): 10,
    (111115, 13): 10,
    (10115, 93): 10,
    (1153, 12): 10,
    (8, 113112): 10,
    (1126, 9): 10,
    (113112, 8116): 10,
    (116112, 10): 10,
    (8116, 93): 10,
    (3116, 11): 10,
}

# Para asignar las curvas transfinite, necesitamos obtener el id real de cada curva.
# En GMSH Python API, las curvas están identificadas por su tag (las líneas ya están creadas).
# Pero aquí en el geo estaban usando pares (como 111117, 1152), que son líneas con esos tags.
# En Python ya creamos esas líneas con esos tags, así que podemos usar directamente esos tags.

for (line1, line2), divisions in transfinite_lines.items():
    # Ambas son líneas, usamos la función TransfiniteCurve para cada una por separado.
    gmsh.model.geo.mesh.setTransfiniteCurve(line1, divisions)
    gmsh.model.geo.mesh.setTransfiniteCurve(line2, divisions)

# Sincronizar geometría
gmsh.model.geo.synchronize()

# El script no genera malla ni guarda archivo, solo define geometría

if '-nopopup' not in sys.argv:
    gmsh.fltk.run()

gmsh.finalize()
