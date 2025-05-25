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
from Quad2D import Quad2D
from solve import Solve
from graph import plot_results

def make_nodes_groups(output_file, title):
    mesh = meshio.read(output_file)
    
    # Traducimos los tags físicos a nombres
    tag_to_name = {v[0]: k for k, v in mesh.field_data.items()}
    grupos = {}

    # Elementos tipo "quad" → para el dominio estructural
    for cell_block, phys_tags in zip(mesh.cells, mesh.cell_data["gmsh:physical"]):
        if cell_block.type != "quad":
            continue
        for quad, tag in zip(cell_block.data, phys_tags):
            nombre = tag_to_name.get(tag, f"{tag}")
            if nombre not in grupos:
                grupos[nombre] = []
            for node_id in quad:
                x, y = mesh.points[node_id][:2]
                grupos[nombre].append(Node(node_id + 1, [x, y]))

    # Elementos tipo "line" → condiciones de borde
    for cell_block, phys_tags in zip(mesh.cells, mesh.cell_data["gmsh:physical"]):
        if cell_block.type != "line":
            continue
        for line, tag in zip(cell_block.data, phys_tags):
            nombre = tag_to_name.get(tag, f"{tag}")
            if nombre not in grupos:
                grupos[nombre] = []
            for node_id in line:
                x, y = mesh.points[node_id][:2]
                restrain = [0, 0]
                if nombre in ["Restriccion"]:
                    restrain = [1, 0]
                if x == 1000 and y == 1000:
                    restrain = [1, 1]
                grupos[nombre].append(Node(node_id + 1, [x, y], restrain=restrain))

    # Eliminar nodos duplicados por grupo (según id)
    for nombre in grupos:
        nodos_unicos = {}
        for nodo in grupos[nombre]:
            nodos_unicos[nodo.index] = nodo
        grupos[nombre] = list(nodos_unicos.values())

    # Visualización opcional
    #Node.plot_nodes_por_grupo(grupos, title, show_ids=False, save=False)

    return grupos, mesh

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

def make_quad2d_elements(mesh, sections, nodes_dict):
    quads = mesh.cells_dict.get('quad', [])
    tags = mesh.cell_data_dict["gmsh:physical"].get("quad", [])
    elements = []
    used_nodes = set()
    nodos_faltantes = []

    for i in range(len(tags)):
        section_tag = str(tags[i])
        if section_tag not in sections:
            print(f"⚠️ Tag físico {section_tag} no tiene sección asociada. Elemento {i+1} omitido.")
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

        #print(nodos)

        element = Quad2D(i + 1, nodos, section)
        elements.append(element)

    if nodos_faltantes:
        print(f"❌ Se omitieron {len(nodos_faltantes)} elementos por nodos faltantes.")
    
    return elements, list(used_nodes)

def plot_all_elements(elements, title, show_ids=True):
    all_x = []
    all_y = []

    # Recopilar coordenadas de todos los nodos
    for elem in elements:
        coords = elem.xy  # accede directamente a las coordenadas
        coords = np.vstack([coords, coords[0]])  # cerrar el polígono
        all_x.extend(coords[:, 0])
        all_y.extend(coords[:, 1])

    # Márgenes y límites
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    x_margin = (x_max - x_min) * 0.05
    y_margin = (y_max - y_min) * 0.05

    x_range = (x_max - x_min) + 2 * x_margin
    y_range = (y_max - y_min) + 2 * y_margin

    fixed_width = 8
    aspect_ratio = y_range / x_range
    height = fixed_width * aspect_ratio

    fig, ax = plt.subplots(figsize=(fixed_width, height))

    for elem in elements:
        coords = elem.xy
        coords = np.vstack([coords, coords[0]])  # cerrar el polígono

        ax.plot(coords[:, 0], coords[:, 1], 'k-', linewidth=1)

        if show_ids:
            for nodo, (x, y) in zip(elem.node_list, coords[:-1]):
                ax.text(x, y, f'N{nodo.index}', color='black', fontsize=6, ha='center', va='center')

    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_ylim(y_min - y_margin, y_max + y_margin)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("All Quad2D elements")
    ax.grid(True)

    plt.show()
    
def apply_distributed_force(grupo_nodos, fuerza_total_y, estructura):
    """
    Aplica una fuerza distribuida vertical (por ejemplo, peso) sobre una línea formada por nodos.
    La fuerza se reparte proporcionalmente a la longitud de los tramos y se descompone en x e y.
    """

    nodos = grupo_nodos
    n = len(nodos)
    if n < 2:
        print("Se requieren al menos dos nodos para aplicar fuerza distribuida.")
        return

    # Paso 1: calcular longitud total
    longitudes = []
    total_length = 0
    for i in range(n - 1):
        dx = nodos[i+1].coord[0] - nodos[i].coord[0]
        dy = nodos[i+1].coord[1] - nodos[i].coord[1]
        L = np.sqrt(dx**2 + dy**2)
        longitudes.append(L)
        total_length += L

    q_lineal = fuerza_total_y / total_length  # fuerza por metro

    # Paso 2: inicializar diccionario de fuerzas
    nodal_forces = {node.index: np.array([0.0, 0.0]) for node in nodos}

    for i in range(n - 1):
        ni = nodos[i]
        nj = nodos[i + 1]
        xi, yi = ni.coord
        xj, yj = nj.coord

        dx = xj - xi
        dy = yj - yi
        L = longitudes[i]

        # Vector perpendicular hacia "abajo"
        vx = dx / L
        vy = dy / L
        nx = -vy
        ny = vx

        Fi = q_lineal * L
        fx = Fi * nx
        fy = Fi * ny

        nodal_forces[ni.index] += np.array([fx / 2, fy / 2])
        nodal_forces[nj.index] += np.array([fx / 2, fy / 2])

    # Paso 3: aplicar fuerzas al sistema
    for node in nodos:
        fx, fy = nodal_forces[node.index]
        dof_x, dof_y = node.dofs
        #estructura.apply_force(dof_x, fx)
        estructura.apply_force(dof_x, fx)
        #print(f"Nodo {node.index} ← Fx = {fx:.3f} N, Fy = {fy:.3f} N")

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
    tpc = ax.tripcolor(triang, facecolors=valores, cmap=cmap, norm=norm, edgecolors='none')
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
        apply_distributed_force(nodos_fuerza, fuerza_total_y=-120000, estructura=estructure)

        nodos_fuerza = grupos["Fuerza_Y_2"]
        apply_distributed_force(nodos_fuerza, fuerza_total_y=-30000, estructura=estructure)
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

    grupos, mesh = make_nodes_groups(output_file, "test")
    sections, nodes_dict = make_sections(grupos, thickness_dict=thickness_dict,E=E, nu=nu, gamma=rho)
    elements, used_nodes = make_quad2d_elements(mesh, sections, nodes_dict)

    #plot_all_elements(elements, "All Quad2D elements", show_ids=True)

    estructure = Solve(used_nodes, elements)

    if self_weight:
        
        # Aplicar peso propio a los elementos
        Peso = apply_self_weight(elements, rho, estructure)


    nodos_fuerza = grupos["Fuerza_Y_1"]
    apply_distributed_force(nodos_fuerza, fuerza_total_y=-120000, estructura=estructure)

    nodos_fuerza = grupos["Fuerza_Y_2"]
    apply_distributed_force(nodos_fuerza, fuerza_total_y=-30000, estructura=estructure)

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
                    num_elements=100,        
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
        

if __name__ == "__main__":
    output_file = "ENTREGA_2/QUAD4/GEOS_QUAD4/M1_Q4_2mm.msh"
    main(title="Quad4/2mm_global/resultados", output_file=output_file, self_weight=True, Topologic_Optimization=True)

    #output_file = "ENTREGA_2/QUAD4/GEOS_QUAD4/M1_Q4_1.75mm.msh"
    #main(title="Quad4/1.75mm_global/resultados", output_file=output_file, self_weight=True)

    #output_file = "ENTREGA_2/QUAD4/GEOS_QUAD4/M1_Q4_1.5mm.msh"
    #main(title="Quad4/1.5mm_global/resultados", output_file=output_file, self_weight=True)

    #output_file = "ENTREGA_2/QUAD4/GEOS_QUAD4/M1_Q4_1.25mm.msh"
    #main(title="Quad4/1.25mm_global/resultados", output_file=output_file, self_weight=True)
