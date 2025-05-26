import subprocess

print("🔧 Ejecutando QUAD4...")
subprocess.run(["python", "ENTREGA_2/QUAD4/main.py"], check=True)

print("✅ QUAD4 terminado. Ejecutando QUAD9...")
subprocess.run(["python", "ENTREGA_2/QUAD9/main.py"], check=True)

print("🏁 Todo finalizado.")
