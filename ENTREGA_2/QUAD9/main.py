import os
import numpy as np
import matplotlib.pyplot as plt
import gmsh
import meshio
import matplotlib.tri as mtri
import re

from nodes import Node
from material import Material
from membrane import Membrane
from Quad2D import Quad9
from solve import Solve
from graph import plot_results
from collections import defaultdict

def make_nodes_groups_quad9(output_file, title):
    mesh = meshio.read(output_file)
    
    tag_to_name = {v[0]: k for k, v in mesh.field_data.items()}
    grupos = defaultdict(dict)  # nombre_grupo: {id_nodo: Node}

    # Procesar elementos tipo quad9
    for cell_block, phys_tags in zip(mesh.cells, mesh.cell_data["gmsh:physical"]):
        if cell_block.type != "quad9":
            continue
        for quad, tag in zip(cell_block.data, phys_tags):
            nombre = tag_to_name.get(tag, str(tag))
            for node_id in quad:
                x, y = mesh.points[node_id][:2]
                if node_id not in grupos[nombre]:
                    grupos[nombre][node_id] = Node(node_id + 1, [x, y])

    # Procesar líneas tipo line3 para condiciones de borde
    for cell_block, phys_tags in zip(mesh.cells, mesh.cell_data["gmsh:physical"]):
        if cell_block.type != "line3":
            continue
        for line, tag in zip(cell_block.data, phys_tags):
            nombre = tag_to_name.get(tag, str(tag))
            for node_id in line:
                x, y = mesh.points[node_id][:2]
                restrain = [0, 0]
                if nombre == "Restriccion":
                    restrain = [1, 0]
                if np.isclose(x, 1000) and np.isclose(y, 1000):
                    restrain = [1, 1]
                if node_id not in grupos[nombre]:
                    grupos[nombre][node_id] = Node(node_id + 1, [x, y], restrain=restrain)
                else:
                    grupos[nombre][node_id].restrain = restrain  # Actualiza si ya existe

    # Convertir a listas
    grupos_final = {nombre: list(nodos.values()) for nombre, nodos in grupos.items()}

    # Visualizar (si está disponible)
    #Node.plot_nodes_por_grupo(grupos_final, title, show_ids=False, save=False)

    return grupos_final, mesh

def make_sections(grupos, thickness_dict, E, nu, gamma):
    
    sections = {}

    for group in thickness_dict:
        material = Material(E, nu, gamma)
        sections[group] = Membrane(thickness_dict[group], material)

    nodes_dict = {}
    for group in grupos:
        for node in grupos[group]:
            nodes_dict[node.index] = node

    return sections, nodes_dict

def make_quad9_elements(mesh, sections, nodes_dict):
    quads = mesh.cells_dict.get('quad9', [])
    tags = mesh.cell_data_dict["gmsh:physical"].get("quad9", [])
    elements = []
    used_nodes = set()
    nodos_faltantes = []
    errores_jacobiano = []

    for i in range(len(tags)):
        section_tag = str(tags[i])
        if section_tag not in sections:
            print(f"⚠️ Tag físico {section_tag} no tiene sección asociada. Elemento {i + 1} omitido.")
            continue

        section = sections[section_tag]
        node_ids = quads[i]

        try:
            nodos = [nodes_dict[node_id + 1] for node_id in node_ids]
        except KeyError as e:
            nodos_faltantes.append(node_ids)
            print(f"❌ Nodo no encontrado en nodes_dict: {e}")
            continue

        for nodo in nodos:
            used_nodes.add(nodo)

        # Intentamos crear el elemento y capturamos errores de Jacobiano
        try:
            element = Quad9(i + 1, nodos, section)
            elements.append(element)
        except ValueError as ve:
            print(f"❌ Error en el elemento {i + 1} con Jacobiano no positivo:")
            print(f"   Nodos: {[n.index for n in nodos]}")
            print(f"   Coordenadas:")
            for j, n in enumerate(nodos):
                print(f"     Nodo local {j}: ID {n.index}, coord = {n.coord}")
            errores_jacobiano.append(i + 1)
            continue

    if nodos_faltantes:
        print(f"❌ Se omitieron {len(nodos_faltantes)} elementos por nodos faltantes.")
    if errores_jacobiano:
        print(f"⚠️ Se omitieron {len(errores_jacobiano)} elementos por Jacobiano negativo.")

    return elements, list(used_nodes)

def apply_distributed_force(grupo_nodos, fuerza_total_x, estructura):
    nodos = grupo_nodos
    n = len(nodos)
    if n < 2:
        print("Se requieren al menos dos nodos para aplicar fuerza distribuida.")
        return

    # Calcular posiciones acumuladas según distancia entre nodos (longitud sobre la curva)
    posiciones = [0.0]
    for i in range(1, n):
        dx = nodos[i].coord[0] - nodos[i-1].coord[0]
        dy = nodos[i].coord[1] - nodos[i-1].coord[1]
        distancia = np.sqrt(dx**2 + dy**2)
        posiciones.append(posiciones[-1] + distancia)
    total_longitud = posiciones[-1]

    # Inicializar fuerzas nodales
    nodal_forces = {}

    # Aplicar fuerza proporcional al tramo entre posiciones adyacentes
    for i in range(n):
        if i == 0:
            # Primer nodo: mitad de la diferencia con siguiente nodo
            fuerza = (posiciones[1] - posiciones[0]) / total_longitud * fuerza_total_x * 0.5
        elif i == n-1:
            # Último nodo: mitad de la diferencia con nodo anterior
            fuerza = (posiciones[-1] - posiciones[-2]) / total_longitud * fuerza_total_x * 0.5
        else:
            # Nodo interno: mitad de tramo anterior + mitad de tramo siguiente
            fuerza = ((posiciones[i] - posiciones[i-1]) + (posiciones[i+1] - posiciones[i])) / total_longitud * fuerza_total_x * 0.5
        nodal_forces[nodos[i].index] = fuerza

    # Aplicar fuerzas en X
    for node in nodos:
        fx = nodal_forces[node.index]
        dof_x, dof_y = node.dofs
        estructura.apply_force(dof_x, fx)
        estructura.apply_force(dof_y, 0.0)
        #print(f"Nodo {node.index} ← Fx = {fx:.3f} N, Fy = 0.000 N, coordenadas y = {node.coord[1]:.3f}")

def apply_self_weight(elements, rho, estructura):
    """
    Aplica peso propio a cada elemento Quad2D como fuerza puntual centrada interpolada.
    """
    g = 9.81
    P = 0.0

    for element in elements:
        centroid = element.get_centroid()
        area = element.A
        t = element.thickness
        peso = area * t * rho * g
        P += peso

        f_local = element.apply_point_body_force(
            x=centroid[0], y=centroid[1], force_vector=[0, -peso]
        )

        for idx_local, dof_global in enumerate(element.index):
            estructura.apply_force(dof_global, f_local[idx_local])

    print(f"✅ Peso total aplicado: {P:.3f} N")
    return P

from matplotlib.colors import Normalize

def plot_elements_by_thickness(elements, title="Espesor por elemento", cmap="viridis"):
    """
    Genera un mapa de calor del espesor de los elementos (Quad2D).

    Parameters:
    - elements: lista de objetos Quad2D
    - title: título del gráfico
    - cmap: colormap (por defecto 'viridis')
    """
    xs, ys = [], []
    nodos_globales = {}
    tri_indices = []
    node_id_map = {}
    idx_counter = 0
    valores = []

    for elem in elements:
        coords = np.array([n.coord for n in elem.node_list])
        t = elem.thickness

        ids = []
        for node in elem.node_list:
            key = tuple(node.coord)
            if key not in node_id_map:
                node_id_map[key] = idx_counter
                nodos_globales[idx_counter] = node
                xs.append(node.coord[0])
                ys.append(node.coord[1])
                idx_counter += 1
            ids.append(node_id_map[key])

        if len(ids) == 4:
            tri_indices.append([ids[0], ids[1], ids[2]])
            tri_indices.append([ids[0], ids[2], ids[3]])
            valores.extend([t, t])

    if not tri_indices:
        print("❌ No se generaron triángulos.")
        return

    triang = mtri.Triangulation(xs, ys, tri_indices)
    norm = Normalize(vmin=min(valores), vmax=max(valores))

    fig, ax = plt.subplots(figsize=(10, 6))
    tpc = ax.tripcolor(triang, facecolors=valores, cmap=cmap, norm=norm, edgecolors='k')
    cbar = plt.colorbar(tpc, ax=ax)
    cbar.set_label("Espesor [mm]")

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.grid(True)

    os.makedirs("GRAFICOS", exist_ok=True)
    fig.savefig(f"GRAFICOS/{title.replace(' ', '_')}_espesor_post_topologic.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Gráfico guardado: GRAFICOS/{title.replace(' ', '_')}_espesor.png")

def compute_nodal_von_mises(elements, u_global):
    """
    Promedia los esfuerzos de Von Mises en los nodos a partir de los elementos vecinos.

    Args:
        elements (list): Lista de elementos (Quad2D, CST, etc.)
        u_global (ndarray): Vector de desplazamientos global.

    Returns:
        dict: Diccionario {node.index: von Mises promedio}
    """
    nodal_vm = {}  # node.index : [vm1, vm2, ...]

    for elem in elements:
        vm = elem.von_mises_stress(u_global)
        for node in elem.node_list:
            if node.index not in nodal_vm:
                nodal_vm[node.index] = []
            nodal_vm[node.index].append(vm)

    # Promediar los esfuerzos por nodo
    nodal_vm_avg = {node_index: np.mean(vms) for node_index, vms in nodal_vm.items()}
    return nodal_vm_avg

def plot_von_mises_field(nodes, elements, vm_nodal_dict, title, cmap='plasma'):
    node_id_to_index = {}
    xs, ys, vms = [], [], []

    # Asociar cada nodo con un índice para el triangulado
    for i, node in enumerate(nodes):
        node_id_to_index[node.index] = i
        xs.append(node.coord[0])
        ys.append(node.coord[1])

        vms.append(vm_nodal_dict.get(node.index, 0.0))  # Usar node.index aquí

    # Construir triangulación a partir de nodos de elementos
    triangles = []
    for elem in elements:
        n0 = node_id_to_index[elem.node_list[0].index]
        n1 = node_id_to_index[elem.node_list[1].index]
        n2 = node_id_to_index[elem.node_list[2].index]
        n3 = node_id_to_index[elem.node_list[3].index]

        # Dos triángulos: (n0, n1, n2) y (n0, n2, n3)
        triangles.append([n0, n1, n2])
        triangles.append([n0, n2, n3])


    triang = mtri.Triangulation(xs, ys, triangles)

    # Márgenes y proporciones
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_margin = (x_max - x_min) * 0.05
    y_margin = (y_max - y_min) * 0.05

    x_range = (x_max - x_min) + 2 * x_margin
    y_range = (y_max - y_min) + 2 * y_margin
    fixed_width = 8
    height = fixed_width * (y_range / x_range)

    fig, ax = plt.subplots(figsize=(fixed_width, height))
    tcf = ax.tricontourf(triang, vms, levels=20, cmap=cmap)

    cbar = fig.colorbar(tcf, ax=ax)
    cbar.set_label("Von Mises Stress (MPa)")

    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_ylim(y_min - y_margin, y_max + y_margin)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title("Von Mises stress field over elements")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True)

    # Guardar gráfico
    os.makedirs("GRAFICOS", exist_ok=True)
    fig.savefig(f"GRAFICOS/{title}_von_mises_post_topologic.png", dpi=300, bbox_inches='tight')
    plt.close()

def optimize_topology_iterative_n_extremes(P, grupos, elements, nodes, rho, estructure,
                                           num_iterations=5, num_elements=2,
                                           delta_t=0.2, t_min=0.2, t_max=10.0, E=0, nu=0):

    """
    Optimización topológica iterativa con propagación ultra-suavizada:
    - Aplica cambios principales a los N elementos extremos
    - Propaga ajustes suaves a través de una función Gaussiana acumulativa
    """
    import numpy as np
    import math
    from collections import defaultdict

    g = 9.81

    def gaussian_weight(level, sigma=2.0):
        return math.exp(-0.5 * (level / sigma) ** 2)

    def find_neighbors_recursive(start_indices, levels):
        neighbor_levels = defaultdict(set)
        current = set(start_indices)
        visited = set(start_indices)

        for level in range(1, levels + 1):
            next_neighbors = set()
            target_nodes = set(n for idx in current for n in elements[idx].node_list)

            for i, elem in enumerate(elements):
                if i in visited:
                    continue
                if any(n in target_nodes for n in elem.node_list):
                    neighbor_levels[level].add(i)
                    next_neighbors.add(i)

            visited.update(next_neighbors)
            current = next_neighbors

        return neighbor_levels

    def update_element_thickness(elem, delta, tag):
        t_old = elem.thickness
        t_new = np.clip(t_old + delta, t_min, t_max)

        # Crear nueva sección con el mismo material
        new_section = Membrane(t_new, elem.material)
        
        # ACTUALIZA el elemento completo
        elem.section = new_section
        elem.thickness = t_new
        elem.Kg, elem.A, elem.F_fe_global = elem.calculate_K0()
        
        updated_indices.add(elem.elementTag)


    for it in range(num_iterations):
        print(f"\n🔁 Iteración {it+1}/{num_iterations}")
        print(f"El peso original es: {P:.5f} N")

        estructure = Solve(nodes, elements)
        apply_self_weight(elements, rho, estructure)
        #===========================================
        #AGREGAR FUERZAS DISTRIBUIDAS
        nodos_fuerza = grupos["Fuerza_Y_1"]
        apply_distributed_force(nodos_fuerza, fuerza_total_x=-1200000, estructura=estructure)

        nodos_fuerza = grupos["Fuerza_Y_2"]
        apply_distributed_force(nodos_fuerza, fuerza_total_x=-300000, estructura=estructure)
        #===========================================

        estructure.solve()

        for node in estructure.nodes:
            node.structure = estructure

        von_mises = np.array([elem.von_mises_stress(estructure.u_global) for elem in elements])
        sorted_indices = np.argsort(von_mises)

        max_indices = sorted_indices[-num_elements:]
        min_indices = sorted_indices[:num_elements]

        updated_indices = set()

        # Aplicar cambio principal
        for idx in max_indices:
            update_element_thickness(elements[idx], +delta_t, "🔺 max")

        for idx in min_indices:
            update_element_thickness(elements[idx], -delta_t, "🔻 min")

        # Propagación ultra-suavizada
        sigma = 2.0
        levels = 6  # hasta vecinos de 6º orden

        max_neighbors_by_level = find_neighbors_recursive(max_indices, levels)
        min_neighbors_by_level = find_neighbors_recursive(min_indices, levels)

        for level in range(1, levels + 1):
            weight = gaussian_weight(level, sigma) * delta_t
            for idx in max_neighbors_by_level[level]:
                if elements[idx].elementTag in updated_indices:
                    continue
                update_element_thickness(elements[idx], +weight, f"⤴ nivel {level}")
            for idx in min_neighbors_by_level[level]:
                if elements[idx].elementTag in updated_indices:
                    continue
                update_element_thickness(elements[idx], -weight, f"⤵ nivel {level}")

        # Reportar peso
        peso_total = sum(
            (el.A) * (el.thickness) * rho * g
            for el in elements
        )
        print(f"⚖️ Peso total aproximado: {peso_total:.5f} N")

    
    return estructure

def main(title, output_file, self_weight=True, Topologic_Optimization=False): 


    E = 210e3  # MPa
    nu = 0.3
    rho = 7800e-9 # kg/mm³

    thickness_dict = {"1": 20, "2": 20, "3": 20, "4": 20}

    grupos, mesh = make_nodes_groups_quad9(output_file, "Quad9")
    sections, nodes_dict = make_sections(grupos, thickness_dict, E, nu, rho)
    elements, used_nodes = make_quad9_elements(mesh, sections, nodes_dict)

    estructure = Solve(used_nodes, elements)

    if self_weight:
        
        # Aplicar peso propio a los elementos
        Peso = apply_self_weight(elements, rho, estructure)

    nodos_fuerza = grupos["Fuerza_Y_1"]
    apply_distributed_force(nodos_fuerza, fuerza_total_x=-1200000, estructura=estructure)

    nodos_fuerza = grupos["Fuerza_Y_2"]
    apply_distributed_force(nodos_fuerza, fuerza_total_x=-300000, estructura=estructure)

    desplazamientos = estructure.solve()


    plot_results(
        estructure,
        elements,
        title=title,
        def_scale=1e4,
        force_scale=1,
        reaction_scale=1e-2,
        sigma_y_tension=250, 
        sigma_y_compression=250
    )

    if Topologic_Optimization:
        if not self_weight:
            raise ValueError("La optimización topológica requiere aplicar el peso propio.")
        estructure = optimize_topology_iterative_n_extremes(P=Peso,
                    grupos=grupos,
                    elements=elements,
                    nodes=used_nodes,
                    rho=rho,
                    estructure=estructure,
                    num_iterations=50,
                    num_elements=50,        
                    delta_t=2,
                    t_min=1,
                    t_max=40,
                    E=E,
                    nu=nu
                )
        
        plot_elements_by_thickness(estructure.elements, title)

        # Importante: guardar los desplazamientos en cada nodo
        for node in estructure.nodes:
            node.structure = estructure  # para acceder a u_global desde cada nodo


        vm_nodal = compute_nodal_von_mises(estructure.elements, estructure.u_global)
        plot_von_mises_field(estructure.nodes, estructure.elements, vm_nodal, title+'_topo')

import os

def verificar_paths(output_file, title):
    """
    Verifica existencia del archivo de malla y de los archivos esperados en la carpeta base de `title`.
    La carpeta base esperada es 'GRAFICOS/<folder1>/<folder2>' según el formato de `title`.
    """
    partes = title.split("/")  # ["Quad9", "2mm_local", "resultados"]
    if len(partes) < 2:
        print(f"❌ Formato de title inválido: '{title}'")
        return

    carpeta_base = os.path.join("GRAFICOS", partes[0], partes[1])
    print(f"\n🔍 Verificando: output_file='{output_file}'")
    print(f"📁 Carpeta base esperada: '{carpeta_base}'")

    # Verificar archivo de malla
    if os.path.isfile(output_file):
        print(f"✅ Archivo de malla encontrado: {output_file}")
    else:
        print(f"❌ Archivo de malla no encontrado: {output_file}")

    # Verificar carpeta base en GRAFICOS/
    if not os.path.isdir(carpeta_base):
        print(f"❌ Carpeta '{carpeta_base}' no existe.")
        return
    else:
        print(f"✅ Carpeta '{carpeta_base}' existe.")


if __name__ == "__main__":
    # Quad9 - Global
    verificar_paths("ENTREGA_2/QUAD9/GEOS_QUAD9_GLOBAL/M1_Q9_2mm.msh",     "Quad9/2mm_global/resultados")
    verificar_paths("ENTREGA_2/QUAD9/GEOS_QUAD9_GLOBAL/M1_Q9_1.75mm.msh",  "Quad9/1.75mm_global/resultados")
    verificar_paths("ENTREGA_2/QUAD9/GEOS_QUAD9_GLOBAL/M1_Q9_1.5mm.msh",   "Quad9/1.5mm_global/resultados")
    verificar_paths("ENTREGA_2/QUAD9/GEOS_QUAD9_GLOBAL/M1_Q9_1.25mm.msh",  "Quad9/1.25mm_global/resultados")

    # Quad9 - Local
    verificar_paths("ENTREGA_2/QUAD9/GEOS_QUAD9_LOCAL/M1_Q9_2mm.msh",      "Quad9/2mm_local/resultados")
    verificar_paths("ENTREGA_2/QUAD9/GEOS_QUAD9_LOCAL/M1_Q9_1.75mm.msh",   "Quad9/1.75mm_local/resultados")
    verificar_paths("ENTREGA_2/QUAD9/GEOS_QUAD9_LOCAL/M1_Q9_1.5mm.msh",    "Quad9/1.5mm_local/resultados")
    verificar_paths("ENTREGA_2/QUAD9/GEOS_QUAD9_LOCAL/M1_Q9_1.25mm.msh",   "Quad9/1.25mm_local/resultados")

    output_file = "ENTREGA_2/QUAD9/GEOS_QUAD9_GLOBAL/M1_Q9_2mm.msh"
    main(title="Quad9/2mm_global/resultados", output_file=output_file, self_weight=True, Topologic_Optimization=True)

    output_file = "ENTREGA_2/QUAD9/GEOS_QUAD9_GLOBAL/M1_Q9_1.75mm.msh"
    main(title="Quad9/1.75mm_global/resultados", output_file=output_file, self_weight=True)

    output_file = "ENTREGA_2/QUAD9/GEOS_QUAD9_GLOBAL/M1_Q9_1.5mm.msh"
    main(title="Quad9/1.5mm_global/resultados", output_file=output_file, self_weight=True)

    output_file = "ENTREGA_2/QUAD9/GEOS_QUAD9_GLOBAL/M1_Q9_1.25mm.msh"
    main(title="Quad9/1.25mm_global/resultados", output_file=output_file, self_weight=True)

    output_file = "ENTREGA_2/QUAD9/GEOS_QUAD9_LOCAL/M1_Q9_2mm.msh"
    main(title="Quad9/2mm_local/resultados", output_file=output_file, self_weight=True)

    output_file = "ENTREGA_2/QUAD9/GEOS_QUAD9_LOCAL/M1_Q9_1.75mm.msh"
    main(title="Quad9/1.75mm_local/resultados", output_file=output_file, self_weight=True)

    output_file = "ENTREGA_2/QUAD9/GEOS_QUAD9_LOCAL/M1_Q9_1.5mm.msh"
    main(title="Quad9/1.5mm_local/resultados", output_file=output_file, self_weight=True)

    output_file = "ENTREGA_2/QUAD9/GEOS_QUAD9_LOCAL/M1_Q9_1.25mm.msh"
    main(title="Quad9/1.25mm_local/resultados", output_file=output_file, self_weight=True)
    
