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

def graficar_2d(A, b, vertices, res):
    fig, ax = plt.subplots(figsize=(8,6))
    x_vals = np.linspace(0, max(vertices[:,0])*1.5, 400)
    for coef, val in zip(A, b):
        a, c = coef[0], coef[1]
        if abs(c) > 1e-10:
            y_vals = (val - a*x_vals)/c
            y_vals = np.clip(y_vals, 0, 1e9)
            ax.plot(x_vals, y_vals, linestyle='--')
        else:
            x_line = val/a
            ax.axvline(x=x_line, linestyle='--')

    poligono = Polygon(vertices, closed=True, fill=True, facecolor='lightgreen', alpha=0.4)
    ax.add_patch(poligono)

    ax.plot(res.x[0], res.x[1], 'ro', markersize=10)
    ax.annotate(f'G={-res.fun:.2f}\n({res.x[0]:.2f}, {res.x[1]:.2f})',
                (res.x[0], res.x[1]), textcoords='offset points', xytext=(15,-30))

    ax.set_xlim(left=0, right=max(vertices[:,0].max(), res.x[0])*1.2)
    ax.set_ylim(bottom=0, top=max(vertices[:,1].max(), res.x[1])*1.2)
    ax.set_title("Solución Gráfica 2D")
    ax.grid(True)

    output = os.path.join(app.config['UPLOAD_FOLDER'], 'grafico2d.png')
    plt.savefig(output)
    plt.close()
    return output

def graficar_3d(A, b, res):
    x = np.linspace(0, max(res.x[0]*2,10), 30)
    y = np.linspace(0, max(res.x[1]*2,10), 30)
    X, Y = np.meshgrid(x, y)
    surfaces = []
    for coef, val in zip(A, b):
        a, b_, c_ = coef
        if abs(c_) < 1e-6:
            continue
        Z = (val - a*X - b_*Y)/c_
        surfaces.append(go.Surface(z=Z, x=X, y=Y, opacity=0.4, colorscale='Viridis', showscale=False))

    punto = res.x
    punto_trace = go.Scatter3d(x=[punto[0]], y=[punto[1]], z=[punto[2]], mode='markers+text',
                               marker=dict(size=7, color='red'), text=[f'G={-res.fun:.2f}'])

    fig = go.Figure(data=surfaces + [punto_trace])
    fig.update_layout(scene=dict(xaxis_title='x', yaxis_title='y', zaxis_title='z'),
                      title='Solución Gráfica 3D', width=800, height=600)

    output = os.path.join(app.config['UPLOAD_FOLDER'], 'grafico3d.html')
    pio.write_html(fig, file=output, auto_open=False)
    return output

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
            grafico_path = graficar_2d(np.array(A_ub), np.array(b_ub), vertices, res)
            log_procesos.append("Generando gráfico 2D.")
            return render_template('resultado.html', resultado=resultado, imagen=grafico_path,
                                   procesos='\n'.join(log_procesos))
        else:
            grafico_path = graficar_3d(np.array(A_ub), np.array(b_ub), res)
            log_procesos.append("Generando gráfico 3D.")
            return render_template('resultado_3d.html', resultado=resultado, html_path=grafico_path,
                                   procesos='\n'.join(log_procesos))

    except Exception as e:
        return f"Error procesando la solicitud: {e}", 500

if __name__ == '__main__':
    app.run(debug=True)
