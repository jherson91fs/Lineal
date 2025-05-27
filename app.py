from flask import Flask, render_template, request
import matplotlib
matplotlib.use('Agg')
import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import plotly.graph_objs as go
import plotly.io as pio
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join('static')

def parse_func_objetivo(texto, num_vars):
    texto = texto.replace('-', '+-')
    partes = texto.split('+')
    coef = {'x':0, 'y':0, 'z':0}
    for parte in partes:
        parte = parte.strip()
        if 'x' in parte:
            num = parte.replace('x','') or '1'
            coef['x'] = float(num)
        elif 'y' in parte:
            num = parte.replace('y','') or '1'
            coef['y'] = float(num)
        elif 'z' in parte:
            num = parte.replace('z','') or '1'
            coef['z'] = float(num)
    return [coef['x'], coef['y'], coef['z']][:num_vars]

def parse_restriccion(texto, num_vars):
    texto = texto.replace('-', '+-')
    if '<=' in texto:
        izq, der = texto.split('<=')
        tipo = '<='
    elif '>=' in texto:
        izq, der = texto.split('>=')
        tipo = '>='
    elif '=' in texto:
        izq, der = texto.split('=')
        tipo = '='
    else:
        raise ValueError("Restricción debe contener <=, >= o =")

    coef = [0,0,0]
    partes = izq.split('+')
    for parte in partes:
        parte = parte.strip()
        if 'x' in parte:
            num = parte.replace('x','') or '1'
            coef[0] = float(num)
        elif 'y' in parte:
            num = parte.replace('y','') or '1'
            coef[1] = float(num)
        elif 'z' in parte:
            num = parte.replace('z','') or '1'
            coef[2] = float(num)
    return coef[:num_vars], float(der.strip()), tipo

def calcular_vertices_2d(A_ub, b_ub):
    vertices = []
    n = len(A_ub)
    for i in range(n):
        for j in range(i+1, n):
            det = A_ub[i][0]*A_ub[j][1] - A_ub[j][0]*A_ub[i][1]
            if abs(det) < 1e-10:
                continue
            x = (b_ub[i]*A_ub[j][1] - b_ub[j]*A_ub[i][1])/det
            y = (A_ub[i][0]*b_ub[j] - A_ub[j][0]*b_ub[i])/det
            p = np.array([x,y])
            if np.all(np.dot(A_ub, p) <= np.array(b_ub)+1e-5) and np.all(p >= -1e-5):
                vertices.append(p)
    return np.array(vertices)

def graficar_2d(A, b, vertices, res, tipo):
    fig, ax = plt.subplots(figsize=(8, 6))
    x_vals = np.linspace(0, max(vertices[:, 0]) * 1.5, 400)

    # Dibujar líneas de restricción
    for i, (coef, val) in enumerate(zip(A, b)):
        a, c = coef[0], coef[1]
        color = f'C{i}'  # Colores únicos por restricción
        if abs(c) > 1e-10:
            y_vals = (val - a * x_vals) / c
            y_vals = np.clip(y_vals, 0, 1e9)
            ax.plot(x_vals, y_vals, linestyle='--', color=color, label=f'Restricción {i+1}')
        else:
            x_line = val / a
            ax.axvline(x=x_line, linestyle='--', color=color, label=f'Restricción {i+1}')

    # Dibujar región factible sombreada
        # Dibujar región factible sombreada y vértices
    if len(vertices) > 0:
        ordenados = vertices[np.lexsort((vertices[:, 1], vertices[:, 0]))]
        poligono = Polygon(ordenados, closed=True, fill=True, facecolor='lightgreen',
                           edgecolor='green', alpha=0.5, label='Región factible')
        ax.add_patch(poligono)

        # Dibujar los puntos (vértices) con marcadores visibles
        for i, punto in enumerate(ordenados):
            ax.plot(punto[0], punto[1], 'ko', markersize=6)
            ax.annotate(f'V{i+1}\n({punto[0]:.2f}, {punto[1]:.2f})',
                        (punto[0], punto[1]), textcoords='offset points',
                        xytext=(0,5), ha='center', fontsize=8, color='gray')


    # Dibujar punto óptimo
    valor = res.fun if tipo == 'min' else -res.fun
    ax.plot(res.x[0], res.x[1], 'ro', markersize=10, label='Solución óptima')
    ax.annotate(f'G={valor:.2f}\n({res.x[0]:.2f}, {res.x[1]:.2f})',
                (res.x[0], res.x[1]), textcoords='offset points', xytext=(10, -40),
                bbox=dict(boxstyle='round,pad=0.4', fc='yellow', alpha=0.9),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    # Centrar la vista en la región factible + solución
    margen = 1
    x_min = max(0, min(vertices[:, 0].min(), res.x[0]) - margen)
    x_max = max(vertices[:, 0].max(), res.x[0]) + margen
    y_min = max(0, min(vertices[:, 1].min(), res.x[1]) - margen)
    y_max = max(vertices[:, 1].max(), res.x[1]) + margen

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Solución Gráfica 2D")
    ax.grid(True)
    ax.legend(loc='upper right', fontsize='small', frameon=True)

    output = os.path.join(app.config['UPLOAD_FOLDER'], 'grafico2d.png')
    plt.savefig(output, bbox_inches='tight')
    plt.close()
    return output



def graficar_2d_proyecciones(res, tipo):
    x, y, z = res.x
    valor = res.fun if tipo == 'min' else -res.fun

    fig_xy, ax1 = plt.subplots()
    ax1.plot(x, y, 'ro')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title(f'Proyección XY\nG={valor:.2f}')
    ax1.grid(True)
    path_xy = os.path.join(app.config['UPLOAD_FOLDER'], 'proyeccion_xy.png')
    fig_xy.savefig(path_xy)
    plt.close(fig_xy)

    fig_xz, ax2 = plt.subplots()
    ax2.plot(x, z, 'go')
    ax2.set_xlabel('x')
    ax2.set_ylabel('z')
    ax2.set_title(f'Proyección XZ\nG={valor:.2f}')
    ax2.grid(True)
    path_xz = os.path.join(app.config['UPLOAD_FOLDER'], 'proyeccion_xz.png')
    fig_xz.savefig(path_xz)
    plt.close(fig_xz)

    fig_yz, ax3 = plt.subplots()
    ax3.plot(y, z, 'bo')
    ax3.set_xlabel('y')
    ax3.set_ylabel('z')
    ax3.set_title(f'Proyección YZ\nG={valor:.2f}')
    ax3.grid(True)
    path_yz = os.path.join(app.config['UPLOAD_FOLDER'], 'proyeccion_yz.png')
    fig_yz.savefig(path_yz)
    plt.close(fig_yz)

    return path_xy, path_xz, path_yz


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/resolver', methods=['POST'])
def resolver():
    try:
        log_procesos = []

        num_vars = int(request.form.get('variables'))
        if num_vars not in [2, 3]:
            return "Número de variables debe ser 2 o 3", 400
        log_procesos.append(f"Número de variables: {num_vars}")

        objetivo = request.form.get('objetivo')
        if not objetivo:
            return "Función objetivo es requerida.", 400
        log_procesos.append(f"Función objetivo: {objetivo}")

        tipo = request.form.get('tipo_opt', 'max')
        log_procesos.append(f"Tipo de optimización: {'Minimizar' if tipo == 'min' else 'Maximizar'}")

        # Restricciones dinámicas
        restricciones = []
        i = 1
        while True:
            restriccion = request.form.get(f'restriccion{i}')
            if restriccion:
                restricciones.append(restriccion)
                i += 1
            else:
                break
        log_procesos.append(f"Restricciones: {restricciones}")

        f_obj = parse_func_objetivo(objetivo, num_vars)
        log_procesos.append(f"Coeficientes función objetivo: {f_obj}")

        A_ub, b_ub = [], []
        for r in restricciones:
            coef, val, tipo_r = parse_restriccion(r, num_vars)
            log_procesos.append(f"Restricción '{r}' parseada como coef={coef}, val={val}, tipo={tipo_r}")
            if tipo_r == '<=':
                A_ub.append(coef)
                b_ub.append(val)
            elif tipo_r == '>=':
                A_ub.append([-c for c in coef])
                b_ub.append(-val)
            elif tipo_r == '=':
                A_ub.append(coef)
                b_ub.append(val)
                A_ub.append([-c for c in coef])
                b_ub.append(-val)

        c = f_obj if tipo == 'min' else [-c for c in f_obj]
        res = linprog(c=c, A_ub=np.array(A_ub), b_ub=np.array(b_ub), method='highs')
        log_procesos.append("Ejecutando linprog...")

        if not res.success:
            log_procesos.append("No se encontró solución óptima.")
            return render_template('resultado.html', resultado="No se encontró solución óptima.",
                                   procesos='\n'.join(log_procesos), imagen=None)

        valor = res.fun if tipo == 'min' else -res.fun
        resultado = f"{'Mínimo' if tipo == 'min' else 'Máximo'} G = {valor:.2f}, Variables = {res.x.round(2)}"
        log_procesos.append(f"Resultado: {resultado}")

        if num_vars == 2:
            vertices = calcular_vertices_2d(np.array(A_ub), np.array(b_ub))
            grafico_path = graficar_2d(np.array(A_ub), np.array(b_ub), vertices, res, tipo)
            log_procesos.append("Generando gráfico 2D.")
            return render_template('resultado.html', resultado=resultado, imagen=grafico_path,
                                   procesos='\n'.join(log_procesos))
        else:
            img_xy, img_xz, img_yz = graficar_2d_proyecciones(res, tipo)
            return render_template('resultado_3d.html', resultado=resultado,
                       img_xy=img_xy, img_xz=img_xz, img_yz=img_yz,
                       procesos='\n'.join(log_procesos))


    except Exception as e:
        return f"Error procesando la solicitud: {e}", 500

if __name__ == '__main__':
    app.run(debug=True)
