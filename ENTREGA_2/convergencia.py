import matplotlib.pyplot as plt

# Datos
h = [20, 15, 12.5, 10]
quad4_global = [236.1386, 260.264692, 284.5515745, 313.7866684]
quad4_local = [282.7483, 320.6951946, 360.8859064, 409.9212636]
quad9_global = [240.8620354, 265.563572, 288.5202264, 317.2676615]
quad9_local = [268.6489693, 305.4030124, 335.686647, 373.4007195]

plt.figure(figsize=(8,6))
plt.plot(h, quad4_global, marker='o', label='Quad 4 Global')
plt.plot(h, quad4_local, marker='s', label='Quad 4 Local')
plt.plot(h, quad9_global, marker='^', label='Quad 9 Global')
plt.plot(h, quad9_local, marker='d', label='Quad 9 Local')

plt.xlabel('h (mm)')
plt.ylabel('Max Von Mises Stress (MPa)')
#plt.title('Curvas vs h (ejes X e Y logarítmicos)')
plt.xscale('log')
plt.yscale('log')
plt.gca().invert_xaxis()  # Para que h decrezca a la derecha
plt.legend()
plt.grid(True, which="both", ls="--")
plt.savefig('GRAFICOS/convergencia.png', dpi=300)
