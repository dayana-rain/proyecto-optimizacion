import streamlit as st
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy.optimize import line_search
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application, convert_xor

# Configuración visual de la página web
st.set_page_config(page_title="OptiWeb - Métodos de Optimización", layout="wide")
st.title("Aplicación Web de Optimización Numérica")
st.caption("Proyecto Final — Métodos de Optimización")

# --- PANEL DE ENTRADAS ORDENADO (SIDEBAR) ---
st.sidebar.header("Configuración del Sistema")

# Grupo 1: Configuración de Variables y Función
with st.sidebar.expander("1. Variables y Función Objetivo", expanded=True):
    num_vars = st.number_input("Número de variables", min_value=1, max_value=20, value=2)
    vars_symbols = sp.symbols(f'x1:{num_vars+1}')
    st.info(f"Variables habilitadas: {', '.join([str(v) for v in vars_symbols])}")
    func_str = st.text_input("Función objetivo (ej: x1^2 + 2*x2^2", value="x1^2 + 2*x2^2")

# Grupo 2: Configuración del Algoritmo
with st.sidebar.expander("2. Algoritmo de Optimizacion", expanded=True):
    modo_comparativo = st.checkbox("Habilitar Modo Comparativo", value=False)
    
    if not modo_comparativo:
        metodo = st.selectbox("Metodo a ejecutar", ["Gradiente", "Gradiente Conjugado", "Newton"])
        metodos_seleccionados = [metodo]
    else:
        # Implementacion del checklist con casillas independientes
        st.write("Selecciona los metodos a comparar:")
        check_gradiente = st.checkbox("Gradiente", value=True)
        check_conjugado = st.checkbox("Gradiente Conjugado", value=True)
        check_newton = st.checkbox("Newton", value=False)
        
        # Construccion de la lista basada en la seleccion del usuario
        metodos_seleccionados = []
        if check_gradiente:
            metodos_seleccionados.append("Gradiente")
        if check_conjugado:
            metodos_seleccionados.append("Gradiente Conjugado")
        if check_newton:
            metodos_seleccionados.append("Newton")
            
        if len(metodos_seleccionados) < 2:
            st.warning("Aviso: Selecciona al menos 2 metodos para poder realizar una comparativa.")
        metodo = None
        
    start_str = st.text_input("Punto de partida (separado por comas)", value=", ".join(["1.0"] * num_vars))
    max_iter = st.number_input("Iteraciones maximas", min_value=1, max_value=1000, value=100)
    tol = st.number_input("Tolerancia (Epsilon)", min_value=1e-7, max_value=1e-1, value=1e-5, format="%.7f")
    
# Grupo 3: Parámetros Avanzados de Wolfe
with st.sidebar.expander("3. Condiciones de Búsqueda de Línea (Wolfe)", expanded=False):
    c1 = st.number_input("c1 (Armijo)", min_value=1e-4, max_value=0.3, value=1e-4, format="%.4f")
    c2 = st.number_input("c2 (Curvatura)", min_value=0.1, max_value=0.9, value=0.9, format="%.2f")

# --- PROCESAMIENTO MATEMÁTICO CONTRA ERRORES ---
try:
    transformations = standard_transformations + (implicit_multiplication_application, convert_xor)
    local_dict = {str(s): s for s in vars_symbols}
    
    try:
        f_expr = parse_expr(func_str, transformations=transformations, local_dict=local_dict)
    except Exception:
        st.error("**Error en la función objetivo:** Expresión matemática inválida o con errores de sintaxis.")
        st.stop()
        
    try:
        grad_expr = [sp.diff(f_expr, v) for v in vars_symbols]
        hessian_expr = [[sp.diff(g, v) for v in vars_symbols] for g in grad_expr]
    except Exception:
        st.error("**Error de diferenciación:** No se pudieron computar las derivadas simbólicas de este modelo.")
        st.stop()

    f_num = sp.lambdify(vars_symbols, f_expr, 'numpy')
    grad_num = sp.lambdify(vars_symbols, grad_expr, 'numpy')
    hessian_num = sp.lambdify(vars_symbols, hessian_expr, 'numpy')

    def f(x): return float(f_num(*x))
    def grad(x): return np.array(grad_num(*x), dtype=float).flatten()
    def hessian(x): return np.array(hessian_num(*x), dtype=float).reshape(num_vars, num_vars)

    try:
        x0 = np.array([float(x.strip()) for x in start_str.split(",")], dtype=float)
    except Exception:
        st.error("**Error en punto de partida:** Utiliza únicamente números separados por comas.")
        st.stop()
        
    if len(x0) != num_vars:
        st.error(f"**Inconsistencia de dimensiones:** Definiste {num_vars} variables pero ingresaste {len(x0)} coordenadas iniciales.")
        st.stop()
        
except Exception as e:
    st.error(f"Error crítico en el motor matemático: {e}")
    st.stop()

# --- NÚCLEO DEL ALGORITMO ---
def optimizar(metodo, x0, max_iter, tol, c1, c2):
    x = x0.copy()
    historial_error = []
    historial_tabla = []
    historial_x = [x0.copy()]
    iteraciones = 0
    criterio = "Máximo de iteraciones alcanzado por límite preventivo."
    d = -grad(x)
    
    for k in range(max_iter):
        g = grad(x)
        error_actual = np.linalg.norm(g)
        historial_error.append(error_actual)
        historial_tabla.append({
            "Iteración": k + 1, 
            "Punto de Inspección (x)": str(np.round(x, 4)), 
            "f(x)": round(f(x), 6), 
            "||∇f|| (Norma Gradiente)": error_actual
        })
        
        if error_actual < tol:
            criterio = f"Convergencia alcanzada exitosamente (Norma del Gradiente < {tol})."
            break
            
        if metodo == "Gradiente":
            d = -g
        elif metodo == "Gradiente Conjugado" and k > 0:
            g_anterior = grad(historial_x[-2])
            beta = np.dot(g, g) / (np.dot(g_anterior, g_anterior) + 1e-10)
            d = -g + beta * d
        elif metodo == "Newton":
            H = hessian(x)
            try:
                H = (H + H.T) / 2.0
                if num_vars > 1:
                    vals = np.linalg.eigvals(H)
                    min_vac = np.min(vals)
                    if min_vac <= 1e-5:
                        H += (abs(min_vac) + 0.5) * np.eye(num_vars)
                else:
                    if H[0,0] <= 1e-5:
                        H[0,0] = 0.5
                
                d = np.linalg.solve(H, -g)
                if np.dot(d, g) >= 0:
                    d = -g
            except Exception:
                d = -g
        
        res_wolfe = line_search(f, grad, x, d, c1=c1, c2=c2)
        alpha = res_wolfe[0]
        if alpha is None:
            alpha = 0.1
            for _ in range(5):
                if f(x + alpha * d) < f(x):
                    break
                alpha *= 0.5
        
        x = x + alpha * d 
        historial_x.append(x.copy())
        iteraciones += 1
        
    return x, f(x), iteraciones, error_actual, criterio, historial_error, historial_tabla

# --- RENDERIZADO VISUAL CONDICIONAL ---
if st.sidebar.button("Ejecutar Optimizacion", use_container_width=True):
    
    # Validacion previa en modo comparativo
    if modo_comparativo and len(metodos_seleccionados) < 2:
        st.error("Error: Debes seleccionar al menos 2 metodos en el panel izquierdo para ejecutar la comparacion.")
        st.stop()

    # -------------------------------------------------------------
    # FLUJO A: MODO COMPARATIVO SELECCIONABLE (2 O 3 METODOS)
    # -------------------------------------------------------------
    if modo_comparativo:
        resultados_comp = {}
        
        for m in metodos_seleccionados:
            resultados_comp[m] = optimizar(m, x0, max_iter, tol, c1, c2)
            
        tab1, tab2, tab3 = st.tabs(["Comparativa Global", "Analisis Simbolico", "Historiales por Metodo"])
        
        with tab1:
            st.subheader("Analisis Comparativo de Rendimiento")
            st.write("Resultados consolidados al evaluar la misma ecuacion matematica bajo el punto de inicio indicado:")
            
            tabla_resumen = []
            for m in metodos_seleccionados:
                x_min, f_min, iters, err_final, criterio, _, _ = resultados_comp[m]
                tabla_resumen.append({
                    "Metodo": m,
                    "Minimo Encontrado (x*)": str(np.round(x_min, 4)),
                    "Evaluacion f(x*)": round(f_min, 6),
                    "Iteraciones": iters,
                    "Error Final (||Grad f||)": f"{err_final:.2e}",
                    "Resultado": "Convergencia" if "Convergencia" in criterio else "Limite Alcanzado"
                })
            st.dataframe(tabla_resumen, use_container_width=True, hide_index=True)
            st.markdown("---")
            
            fig, ax = plt.subplots(figsize=(7, 3.8))
            fig.patch.set_facecolor('#FFFDFE')
            ax.set_facecolor('#FFFDFE')
            
            colores_lineas = {
                "Gradiente": "#B83B6F",
                "Gradiente Conjugado": "#4A7BB0",
                "Newton": "#8A4F7D"
            }
            
            for m in metodos_seleccionados:
                errores_m = resultados_comp[m][5]
                ax.plot(range(1, len(errores_m) + 1), errores_m, marker='o', markersize=3,
                        color=colores_lineas[m], linewidth=1.8, label=f'Ruta {m}')
                
            ax.axhline(y=tol, color='#611833', linestyle='--', alpha=0.8, 
                       linewidth=1.5, label=f'Tolerancia ({tol})')
            
            ax.set_yscale('log')
            ax.grid(True, which="both", linestyle=":", alpha=0.4, color='#999999')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#2E111D')
            ax.spines['bottom'].set_color('#2E111D')
            ax.tick_params(colors='#2E111D', labelsize=9)
            
            ax.set_xlabel("Iteraciones", fontsize=10, fontweight='bold', color='#2E111D', labelpad=6)
            ax.set_ylabel("Magnitud del Error (Log)", fontsize=10, fontweight='bold', color='#2E111D', labelpad=6)
            ax.set_title("Comparativa de Velocidad de Convergencia", fontsize=11, fontweight='bold', color='#2E111D', pad=12)
            ax.legend(frameon=False, loc='upper right', fontsize=9, labelcolor='#2E111D')
            
            fig.tight_layout()
            _, col_plot, _ = st.columns([1, 4, 1])
            with col_plot:
                st.pyplot(fig)
                
        with tab2:
            st.subheader("Modelamiento Analitico Desarrollado por SymPy")
            col_sym1, col_sym2 = st.columns(2)
            with col_sym1:
                st.markdown("#### Vector Gradiente Analitico (Grad f)")
                st.latex(sp.latex(grad_expr))
            with col_sym2:
                st.markdown("#### Matriz Hessiana Simbolica (H)")
                st.latex(sp.latex(sp.Matrix(hessian_expr)))
                
        with tab3:
            st.subheader("Bitacoras de Iteracion Individuales")
            st.write("Expande el metodo que desees para revisar su historial de saltos numercos:")
            for m in metodos_seleccionados:
                with st.expander(f"Ver tabla detallada de: {m}"):
                    st.dataframe(resultados_comp[m][6], use_container_width=True)

    # -------------------------------------------------------------
    # FLUJO B: MODO INDIVIDUAL TRADICIONAL (1 METODO)
    # -------------------------------------------------------------
    else:
        x_min, f_min, iters, err_final, criterio, errores, tabla_pasos = optimizar(metodo, x0, max_iter, tol, c1, c2)
        
        tab1, tab2, tab3 = st.tabs(["Resumen y Convergencia", "Analisis Simbolico", "Historial Paso a Paso"])
        
        with tab1:
            st.subheader(f"Análisis de Desempeño: Metodo de {metodo}")
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1:
                st.metric(label="Minimo Encontrado (x*)", value=str(np.round(x_min, 4)))
            with col_m2:
                st.metric(label="Evaluacion f(x*)", value=f"{f_min:.6f}")
            with col_m3:
                st.metric(label="Iteraciones Totales", value=iters)
            with col_m4:
                st.metric(label="Error Final (||Grad f||)", value=f"{err_final:.2e}")
                
            st.info(f"Condicion de Finalizacion: {criterio}")
            st.markdown("---")
            
            fig, ax = plt.subplots(figsize=(7, 3.8))
            fig.patch.set_facecolor('#FFFDFE')
            ax.set_facecolor('#FFFDFE')
            
            ax.plot(range(1, len(errores) + 1), errores, marker='o', markersize=4, 
                    color='#B83B6F', linewidth=2, label='Historial de Error (||Grad f||)')
            ax.axhline(y=tol, color='#611833', linestyle='--', alpha=0.8, linewidth=1.5, label=f'Tolerancia ({tol})')
            
            ax.set_yscale('log')
            ax.grid(True, which="both", linestyle=":", alpha=0.5, color='#B83B6F')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#2E111D')
            ax.spines['bottom'].set_color('#2E111D')
            ax.tick_params(colors='#2E111D', labelsize=9)
            
            ax.set_xlabel("Iteraciones Realizadas", fontsize=10, fontweight='bold', color='#2E111D', labelpad=6)
            ax.set_ylabel("Magnitud del Error (Escala Log)", fontsize=10, fontweight='bold', color='#2E111D', labelpad=6)
            ax.set_title("Trayectoria de Descenso hacia el Optimo", fontsize=11, fontweight='bold', color='#2E111D', pad=12)
            ax.legend(frameon=False, loc='upper right', fontsize=9, labelcolor='#2E111D')
            
            fig.tight_layout()
            _, col_plot, _ = st.columns([1, 4, 1])
            with col_plot:
                st.pyplot(fig)

        with tab2:
            st.subheader("Modelamiento Analitico Desarrollado por SymPy")
            col_sym1, col_sym2 = st.columns(2)
            with col_sym1:
                st.markdown("#### Vector Gradiente Analitico (Grad f)")
                st.latex(sp.latex(grad_expr))
            with col_sym2:
                st.markdown("#### Matriz Hessiana Simbolica (H)")
                st.latex(sp.latex(sp.Matrix(hessian_expr)))

        with tab3:
            st.subheader("Bitacora Detallada de Optimizacion Numerica")
            st.dataframe(tabla_pasos, use_container_width=True)

else:
    st.info("Modifica las variables en el panel izquierdo y haz clic en 'Ejecutar Optimizacion'.")
