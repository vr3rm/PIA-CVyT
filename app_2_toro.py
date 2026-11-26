import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.integrate import quad

# Configuración general de la página
st.set_page_config(page_title="Longitudes sobre Superficie Toroidal", layout="wide")
st.title("Medición Numérica de Longitudes sobre una Superficie Toroidal")


st.sidebar.header("Parámetros del Toro y Simulación")

R = st.sidebar.number_input("Radio mayor (R)", min_value=0.1, value=5.0, step=0.5)
r = st.sidebar.number_input("Radio menor (r)", min_value=0.1, value=2.0, step=0.5)

# Validamos la condición R > r > 0
if not (R > r > 0):
    st.error("Condición inválida: debe cumplirse estrictamente que R > r > 0.")
    st.stop()

trayectorias_disponibles = {
    "A: Circunferencia exterior": "A",
    "B: Circunferencia meridiana": "B",
    "C: Curva toroidal cerrada (2, 3)": "C",
}

opcion_trayectoria = st.sidebar.selectbox(
    "Selecciona la trayectoria principal:", list(trayectorias_disponibles.keys())
)
cod_trayectoria = trayectorias_disponibles[opcion_trayectoria]

metodo_seleccionado = st.sidebar.selectbox(
    "Selecciona el método de integración:", ["Trapecio", "Simpson"]
)

# Parámetros fijos de integración
A_LIM, B_LIM = 0.0, 2.0 * np.pi
LISTA_N = [8, 16, 32, 64, 128]

def superficie_toro(theta, phi, R, r):
    """Calcula las coordenadas cartesianas (X, Y, Z) del toro."""
    X = (R + r * np.cos(theta)) * np.cos(phi)
    Y = (R + r * np.cos(theta)) * np.sin(phi)
    Z = r * np.sin(theta)
    return X, Y, Z


def obtener_trayectoria(codigo, lam):
    """Devuelve theta(lam), phi(lam) y sus derivadas respecto a lam."""
    if codigo == "A":
        # Circunferencia exterior: theta = 0, phi = lam
        theta = np.zeros_like(lam, dtype=float)
        phi = np.array(lam, dtype=float)
        dtheta = np.zeros_like(lam, dtype=float)
        dphi = np.ones_like(lam, dtype=float)
    elif codigo == "B":
        # Circunferencia meridiana: theta = lam, phi = 0
        theta = np.array(lam, dtype=float)
        phi = np.zeros_like(lam, dtype=float)
        dtheta = np.ones_like(lam, dtype=float)
        dphi = np.zeros_like(lam, dtype=float)
    elif codigo == "C":
        # Curva toroidal cerrada (2, 3): theta = 2*lam, phi = 3*lam
        theta = 2.0 * np.array(lam, dtype=float)
        phi = 3.0 * np.array(lam, dtype=float)
        dtheta = 2.0 * np.ones_like(lam, dtype=float)
        dphi = 3.0 * np.ones_like(lam, dtype=float)
    else:
        raise ValueError("Código de trayectoria no reconocido.")

    return theta, phi, dtheta, dphi

def integrando_extrinseco(lam, codigo, R, r):
    """Evalúa el elemento de arco en coordenadas cartesianas (métrica extrínseca)."""
    theta, phi, dtheta, dphi = obtener_trayectoria(codigo, lam)

    # Derivadas cartesianas respecto a lambda mediante regla de la cadena
    dX = -r * np.sin(theta) * dtheta * np.cos(phi) - (R + r * np.cos(theta)) * np.sin(phi) * dphi
    dY = -r * np.sin(theta) * dtheta * np.sin(phi) + (R + r * np.cos(theta)) * np.cos(phi) * dphi
    dZ = r * np.cos(theta) * dtheta

    return np.sqrt(dX**2 + dY**2 + dZ**2)


def integrando_intrinseco(lam, codigo, R, r):
    """Evalúa el elemento de arco mediante el tensor métrico inducido (métrica intrínseca)."""
    theta, _, dtheta, dphi = obtener_trayectoria(codigo, lam)

    g_theta_theta = r**2
    g_phi_phi = (R + r * np.cos(theta)) ** 2

    return np.sqrt(g_theta_theta * (dtheta**2) + g_phi_phi * (dphi**2))

def integrar_trapecio(func, a, b, N, *args):
    """Regla compuesta del trapecio para N subintervalos."""
    h = (b - a) / N
    x = np.linspace(a, b, N + 1)
    y = func(x, *args)
    return (h / 2.0) * (y[0] + 2.0 * np.sum(y[1:-1]) + y[-1])


def integrar_simpson(func, a, b, N, *args):
    """Regla compuesta de Simpson 1/3 para N subintervalos (N debe ser par)."""
    if N % 2 != 0:
        raise ValueError("El método de Simpson requiere un número par de subintervalos N.")
    h = (b - a) / N
    x = np.linspace(a, b, N + 1)
    y = func(x, *args)
    return (h / 3.0) * (y[0] + 4.0 * np.sum(y[1:-1:2]) + 2.0 * np.sum(y[2:-2:2]) + y[-1])


def obtener_referencia(codigo, R, r):
    """Devuelve la longitud analítica (A y B) o de alta precisión (C)."""
    if codigo == "A":
        return 2.0 * np.pi * (R + r)
    elif codigo == "B":
        return 2.0 * np.pi * r
    elif codigo == "C":
        valor, _ = quad(
            integrando_intrinseco, A_LIM, B_LIM, args=(codigo, R, r), epsabs=1e-14, epsrel=1e-14
        )
        return valor

def generar_tabla_resultados(codigo, metodo, R, r):
    """Genera un DataFrame con longitudes y errores para N = 8, 16, 32, 64, 128."""
    integrador = integrar_trapecio if metodo == "Trapecio" else integrar_simpson
    L_ref = obtener_referencia(codigo, R, r)
    filas = []

    for N in LISTA_N:
        L_ext = integrador(integrando_extrinseco, A_LIM, B_LIM, N, codigo, R, r)
        L_int = integrador(integrando_intrinseco, A_LIM, B_LIM, N, codigo, R, r)

        E_coord = abs(L_ext - L_int)
        E_abs_ext = abs(L_ext - L_ref)
        E_abs_int = abs(L_int - L_ref)
        E_rel_ext = E_abs_ext / L_ref
        E_rel_int = E_abs_int / L_ref

        filas.append(
            {
                "N": N,
                "L_ext": L_ext,
                "L_int": L_int,
                "E_coord": E_coord,
                "E_abs,ext": E_abs_ext,
                "E_abs,int": E_abs_int,
                "E_rel,ext": E_rel_ext,
                "E_rel,int": E_rel_int,
            }
        )

    return pd.DataFrame(filas), L_ref

def crear_figura_3d(R, r, codigo_activo):
    """Construye la visualización 3D del toro y las tres trayectorias."""
    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(0, 2 * np.pi, 60)
    THETA, PHI = np.meshgrid(u, v)
    X_toro, Y_toro, Z_toro = superficie_toro(THETA, PHI, R, r)

    fig = go.Figure()

    # Superficie del toro
    fig.add_trace(
        go.Surface(
            x=X_toro,
            y=Y_toro,
            z=Z_toro,
            colorscale="Blues",
            opacity=0.35,
            showscale=False,
            name="Superficie del Toro",
        )
    )

    # Trazado de las tres trayectorias
    lam_plot = np.linspace(0, 2 * np.pi, 400)
    estilos = {
        "A": ("A: Circunferencia exterior", "red"),
        "B": ("B: Circunferencia meridiana", "green"),
        "C": ("C: Curva toroidal (2, 3)", "purple"),
    }

    for cod, (nombre, color) in estilos.items():
        th, ph, _, _ = obtener_trayectoria(cod, lam_plot)
        xt, yt, zt = superficie_toro(th, ph, R, r)
        grosor = 8 if cod == codigo_activo else 4
        fig.add_trace(
            go.Scatter3d(
                x=xt,
                y=yt,
                z=zt,
                mode="lines",
                line=dict(color=color, width=grosor),
                name=nombre,
            )
        )

    fig.update_layout(
        title=f"Visualización 3D del Toro (R = {R}, r = {r})",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        height=550,
    )
    return fig


def crear_grafica_convergencia(df_trap, df_simp, titulo_trayectoria):
    """Genera la gráfica log-log de convergencia de errores entre Trapecio y Simpson."""
    fig = go.Figure()

    
    err_trap = np.maximum(df_trap["E_abs,int"], 1e-16)
    err_simp = np.maximum(df_simp["E_abs,int"], 1e-16)

    fig.add_trace(
        go.Scatter(
            x=df_trap["N"],
            y=err_trap,
            mode="lines+markers",
            name="Método del Trapecio",
            line=dict(width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_simp["N"],
            y=err_simp,
            mode="lines+markers",
            name="Método de Simpson",
            line=dict(width=3, dash="dash"),
        )
    )

    fig.update_layout(
        title=f"Convergencia Numérica - {titulo_trayectoria}",
        xaxis=dict(title="Número de subintervalos (N)", type="log", tickvals=LISTA_N),
        yaxis=dict(title="Error Absoluto |L_num - L_ref|", type="log", exponentformat="e"),
        height=450,
    )
    return fig


# --- renderizado PRINCIPAL de la app ---
col_3d, col_datos = st.columns([1.1, 1.2])

with col_3d:
    st.plotly_chart(crear_figura_3d(R, r, cod_trayectoria), use_container_width=True)

with col_datos:
    df_actual, L_ref_actual = generar_tabla_resultados(cod_trayectoria, metodo_seleccionado, R, r)
    st.subheader(f"Tabla de Resultados: {opcion_trayectoria} ({metodo_seleccionado})")
    st.metric("Longitud de referencia (L_ref)", f"{L_ref_actual:.12f}")

    formato_columnas = {
        "L_ext": "{:.10f}",
        "L_int": "{:.10f}",
        "E_coord": "{:.3e}",
        "E_abs,ext": "{:.3e}",
        "E_abs,int": "{:.3e}",
        "E_rel,ext": "{:.3e}",
        "E_rel,int": "{:.3e}",
    }
    st.dataframe(df_actual.style.format(formato_columnas), use_container_width=True, hide_index=True)

st.divider()

# Gráfica de convergencia y comparación
df_trap_act, _ = generar_tabla_resultados(cod_trayectoria, "Trapecio", R, r)
df_simp_act, _ = generar_tabla_resultados(cod_trayectoria, "Simpson", R, r)
st.plotly_chart(
    crear_grafica_convergencia(df_trap_act, df_simp_act, opcion_trayectoria),
    use_container_width=True,
)

# Sección con todas las tablas generadas
with st.expander("Ver todas las tablas comparativas generadas automáticamente"):
    for nombre_t, cod_t in trayectorias_disponibles.items():
        st.markdown(f"### {nombre_t}")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Método del Trapecio**")
            df_t, _ = generar_tabla_resultados(cod_t, "Trapecio", R, r)
            st.dataframe(df_t.style.format(formato_columnas), use_container_width=True, hide_index=True)
        with c2:
            st.markdown("**Método de Simpson**")
            df_s, _ = generar_tabla_resultados(cod_t, "Simpson", R, r)
            st.dataframe(df_s.style.format(formato_columnas), use_container_width=True, hide_index=True)
