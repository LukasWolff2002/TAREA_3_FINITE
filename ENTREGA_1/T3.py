import numpy as np
import matplotlib.pyplot as plt
import gmsh
import os
import math
import sys

from fem.units import mm, cm, m, kgf, N, tf, kN, MPa, GPa
from fem import Node, Material, Membrane, CST, matrix_extract, matrix_replace, get_nodes_from_physical_id, get_line_load_global_vector
from fem.quad4 import Quad4

output_path=r'C:\Users\felip\OneDrive\Escritorio\202510\Finite Elements\TAREA_3_FINITE'  # Path to save the mesh file
mesh_name='TAREA_3_PARTE_1'  # Name of the mesh file
output_file=os.path.join(output_path, mesh_name + '.msh')  # Full path to the mesh file
if not os.path.exists(output_path):
    os.makedirs(output_path)  # Create the output directory if it doesn't exist

# General model parameters

# Definimos el material
steel=Material(name='steel',
             E=200*GPa,
             nu=0.3,
             rho=7.85*tf/m**3)
# Definimos una seccion de la membrana
Head=Membrane(name='Body',
              thickness = 2*cm,
              material=steel)

# Definimos los grupos fisicos de las partes del modelo
# Map the physical group id to a section
section_dictionary={201:Head}
                
load_dictionary={101:1,
                 102:2}

restrain_dictionary={102:['r', 'r']}

# Definimos carga de peso propio
self_weight=[0,0]

# Initialize the Gmsh API and create a new model.
gmsh.initialize()
gmsh.model.add("TAREA_3_PARTE_1")

###############################################################################
# PARAMETERS (from your .geo file)
###############################################################################

# Datos
cm = 0.01  # asumiendo que 1 unidad = 1 cm
x_origen = 0
y_origen = 0

# ---------------------------------------------------
# GEOMETRÍA: puntos
# ---------------------------------------------------
p1  = gmsh.model.geo.addPoint(x_origen,          y_origen,           0, 1)
p2  = gmsh.model.geo.addPoint(x_origen+80*cm,    y_origen,           0, 1)
p3  = gmsh.model.geo.addPoint(x_origen+100*cm,   y_origen+20*cm,     0, 1)
p4  = gmsh.model.geo.addPoint(x_origen+120*cm,   y_origen,           0, 1)
p5  = gmsh.model.geo.addPoint(x_origen+200*cm,   y_origen,           0, 1)
p6  = gmsh.model.geo.addPoint(x_origen+200*cm,   y_origen+20*cm,     0, 1)
p7  = gmsh.model.geo.addPoint(x_origen+200*cm,   y_origen+100*cm,    0, 1)
p8  = gmsh.model.geo.addPoint(x_origen+120*cm,   y_origen+100*cm,    0, 1)
p9  = gmsh.model.geo.addPoint(x_origen+100*cm,   y_origen+100*cm,    0, 1)
p10 = gmsh.model.geo.addPoint(x_origen+80*cm,    y_origen+100*cm,    0, 1)
p11 = gmsh.model.geo.addPoint(x_origen,          y_origen+100*cm,    0, 1)
p12 = gmsh.model.geo.addPoint(x_origen,          y_origen+20*cm,     0, 1)

# ---------------------------------------------------
# GEOMETRÍA: líneas
# ---------------------------------------------------
l1  = gmsh.model.geo.addLine(p1,  p2)
l2  = gmsh.model.geo.addLine(p2,  p3)   # lado izquierdo del notch
l3  = gmsh.model.geo.addLine(p3,  p4)   # lado derecho del notch
l4  = gmsh.model.geo.addLine(p4,  p5)
l5  = gmsh.model.geo.addLine(p5,  p6)
l6  = gmsh.model.geo.addLine(p6,  p7)
l7  = gmsh.model.geo.addLine(p7,  p8)
l8  = gmsh.model.geo.addLine(p8,  p9)
l9  = gmsh.model.geo.addLine(p9,  p10)
l10 = gmsh.model.geo.addLine(p10, p11)
l11 = gmsh.model.geo.addLine(p11, p12)
l12 = gmsh.model.geo.addLine(p12, p1)

# ---------------------------------------------------
# CURVE LOOP & SUPERFICIE
# ---------------------------------------------------
loop1    = gmsh.model.geo.addCurveLoop(
    [l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12]
)
surface1 = gmsh.model.geo.addPlaneSurface([loop1])

# ---------------------------------------------------
# MALLADO TRANSFINITO EN CURVAS (con progresión)
# ---------------------------------------------------
m = 50
# Coeficientes de progresión
coef_notch   = 0.3   # malla más fina en la punta
coef_exterior = 1.2  # ligera gradación en el contorno exterior

# Notch: lados p2→p3 y p3→p4
for tag in [l2, l3]:
    gmsh.model.geo.mesh.setTransfiniteCurve(tag, int(1.5*m)+1,
                                            scheme="Progression",
                                            coef=coef_notch)

# Bordes exteriores largos
for tag in [l1, l4, l5, l6, l7, l8, l9, l10, l11, l12]:
    gmsh.model.geo.mesh.setTransfiniteCurve(tag, m+1,
                                            scheme="Progression",
                                            coef=coef_exterior)

# ---------------------------------------------------
# SINCRONIZAR GEOMETRÍA
# ---------------------------------------------------
gmsh.model.geo.synchronize()

# ---------------------------------------------------
# MALLADO TRANSFINITO EN SUPERFICIE & RECOMBINE
# ---------------------------------------------------
# Define las 4 esquinas para la superficie transfinita
gmsh.model.geo.mesh.setTransfiniteSurface(surface1,
                                          cornerTags=[p1, p5, p7, p11])
# Fuerza quads puros
gmsh.model.mesh.setRecombine(2, surface1)

# ---------------------------------------------------
# OPCIONES DE CALIDAD Y OPTIMIZACIÓN
# ---------------------------------------------------
gmsh.option.setNumber("Mesh.RecombineAll", 1)
gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 1)   # Blossom
gmsh.option.setNumber("Mesh.Smoothing", 100)             # iteraciones Laplaciano
gmsh.option.setNumber("Mesh.Optimize", 1)
# Ajustes generales de tamaño (opcionalmente afínea a tus cm)
gmsh.option.setNumber("Mesh.CharacteristicLengthMin",   1)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 100)
# Algoritmo de triangulación auxiliar
gmsh.option.setNumber("Mesh.Algorithm", 8)               # Delaunay 3D (Netgen)
gmsh.option.setNumber("Mesh.SurfaceFaces", 1)

# ---------------------------------------------------
# GENERAR MALLA, OPTIMIZAR Y GUARDAR
# ---------------------------------------------------
gmsh.model.mesh.generate(2)
gmsh.model.mesh.optimize("Netgen")

output_file = "TAREA_3_PARTE_1_refinada.msh"
gmsh.write(output_file)

if '-nopopup' not in sys.argv:
    gmsh.fltk.run()

gmsh.finalize()
