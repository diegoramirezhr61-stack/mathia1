from flask import Flask, render_template, request, jsonify, session
from groq import Groq

import os
import io
import base64

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from sympy import *
from sympy import (
    symbols,
    sympify,
    diff,
    integrate,
    limit,
    oo,
    latex
)
# ================================
# APP
# ================================

app = Flask(__name__)
app.secret_key = "mathia_secret"

# ================================
# GROQ
# ================================

GROQ_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_KEY:
    raise Exception("Falta GROQ_API_KEY")

cliente = Groq(api_key=GROQ_KEY)

# ================================
# SESSION
# ================================

def init_session():

    if "memoria_chat" not in session:
        session["memoria_chat"] = []

    if "contador_preguntas" not in session:
        session["contador_preguntas"] = 0

    if "historial_graficas" not in session:
        session["historial_graficas"] = []

# ================================
# IA GROQ MEJORADA
# ================================

def preguntar_ia(prompt):

    init_session()

    mensajes = [

        {
            "role": "system",

            "content": """
Eres MathIA, una inteligencia artificial matemática avanzada.

Tu trabajo es resolver ejercicios matemáticos
paso a paso de manera clara y profesional.

SIEMPRE:

- Explica detalladamente.
- Usa pasos numerados.
- Usa LaTeX.
- Resuelve como profesor universitario.
- Si es cálculo, explica derivadas e integrales.
- Si es límite, aplica propiedades.
- Si es continuidad, analiza dominio.
- Si es límite al infinito, analiza comportamiento.
- Si es integración por sustitución, explica el cambio de variable.
- Si es integral definida, resuelve evaluando límites.
- Nunca respondas corto.
- Nunca digas solo el resultado.

FORMATO:

1. Interpretación del ejercicio
2. Desarrollo paso a paso
3. Resultado final

Ejemplos:

\\[
\\int_1^3 x^2 dx
\\]

\\[
\\lim_{x \\to \\infty} \\frac{1}{x}
\\]

\\[
f'(x)=2x
\\]

Habla SIEMPRE en español.
"""
        }

    ]

    mensajes.extend(session["memoria_chat"])

    mensajes.append({
        "role": "user",
        "content": prompt
    })

    respuesta = cliente.chat.completions.create(

        model="llama-3.1-8b-instant",

        messages=mensajes,

        temperature=0.2,

        max_tokens=1500
    )

    texto = respuesta.choices[0].message.content

    session["memoria_chat"].append({
        "role": "user",
        "content": prompt
    })

    session["memoria_chat"].append({
        "role": "assistant",
        "content": texto
    })

    session["memoria_chat"] = session["memoria_chat"][-12:]

    return texto
# ================================
# NORMALIZAR FUNCION
# ================================

def normalizar_funcion(texto):

    texto = texto.lower()

    borrar = [
        "grafica",
        "gráfica",
        "plot",
        "dibujar",
        "dibújame"
    ]

    for palabra in borrar:
        texto = texto.replace(palabra, "")

    texto = texto.strip()

    texto = texto.replace("x al cuadrado", "x^2")
    texto = texto.replace("x al cubo", "x^3")

    texto = texto.replace("al cuadrado", "^2")
    texto = texto.replace("al cubo", "^3")

    texto = texto.replace("^", "**")

    texto = texto.replace("²", "**2")
    texto = texto.replace("³", "**3")

    return texto

# ================================
# GRAFICAS CORREGIDAS
# ================================

def generar_grafica(funcion):

    try:

        print("FUNCION ORIGINAL:", funcion)

        # limpiar texto
        funcion = funcion.lower()

        funcion = funcion.replace("grafica", "")
        funcion = funcion.replace("gráfica", "")
        funcion = funcion.replace("plot", "")
        funcion = funcion.replace("dibujar", "")

        funcion = funcion.strip()

        # potencia
        funcion = funcion.replace("^", "**")

        print("FUNCION LIMPIA:", funcion)

        # variable
        x = symbols("x")

        # convertir expresion
        expr = sympify(funcion)

        # convertir a numpy
        f = lambdify(x, expr, "numpy")

        # valores x
        xs = np.linspace(-10, 10, 500)

        # valores y
        ys = f(xs)

        # evitar errores infinitos
        ys = np.nan_to_num(
            ys,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        # figura
        plt.figure(figsize=(7,5))

        plt.plot(xs, ys)

        plt.axhline(0, linewidth=1)

        plt.axvline(0, linewidth=1)

        plt.grid(True)

        plt.title(f"f(x) = {funcion}")

        # guardar
        img = io.BytesIO()

        plt.savefig(
            img,
            format="png",
            bbox_inches="tight"
        )

        plt.close()

        img.seek(0)

        grafica_base64 = base64.b64encode(
            img.getvalue()
        ).decode("utf-8")

        return grafica_base64

    except Exception as e:

        print("ERROR GRAFICA:", e)

        return None

# ================================
# VARIABLE GLOBAL
# ================================

x = symbols("x")

def resolver_derivada(expresion):

    try:

        expresion = expresion.strip()

        expresion = expresion.replace("^", "**")

        expr = sympify(expresion)

        resultado = diff(expr, x)

        prompt = f"""
Resuelve paso a paso esta derivada.

Función:

{expresion}

La derivada correcta es:

{resultado}

Explica detalladamente el procedimiento
usando LaTeX y pasos numerados.
"""

        explicacion = preguntar_ia(prompt)

        return explicacion

    except Exception as e:

        print("ERROR DERIVADA:", e)

        return f"Error en derivada: {e}"

# ================================
# INTEGRALES REALES
# ================================

def resolver_integral(expresion):

    try:

        expresion = expresion.strip()

        expresion = expresion.replace("^", "**")

        expr = sympify(expresion)

        resultado = integrate(expr, x)

        prompt = f"""
Resuelve paso a paso esta integral.

Integral:

{expresion}

El resultado correcto es:

{resultado}

Explica detalladamente usando LaTeX.
"""

        explicacion = preguntar_ia(prompt)

        return explicacion

    except Exception as e:

        print("ERROR INTEGRAL:", e)

        return f"Error en integral: {e}"

# ================================
# LIMITES REALES
# ================================

def resolver_limite(expresion, valor):

    try:

        expresion = expresion.strip()

        expresion = expresion.replace("^", "**")

        expr = sympify(expresion)

        if valor == "infinito":
            valor_sympy = oo
        else:
            valor_sympy = float(valor)

        resultado = limit(expr, x, valor_sympy)

        prompt = f"""
Resuelve paso a paso este límite.

Límite:

{expresion}

El resultado correcto es:

{resultado}

Explica usando propiedades de límites
y LaTeX.
"""

        explicacion = preguntar_ia(prompt)

        return explicacion

    except Exception as e:

        print("ERROR LIMITE:", e)

        return f"Error en límite: {e}"
    
# ================================
# ROUTER INTELIGENTE
# ================================

def router(pregunta):

    texto = pregunta.lower()

    # =========================
    # GRAFICAS
    # =========================

    if any(x in texto for x in [

        "grafica",
        "gráfica",
        "graficar",
        "plot",
        "dibujar"

    ]):

        funcion = texto

        grafica = generar_grafica(funcion)

        return {
            "respuesta": "Gráfica generada correctamente.",
            "grafica": grafica
        }

    # =========================
    # DERIVADAS
    # =========================

    if texto.startswith("derivada de"):

        expr = texto.replace(
            "derivada de",
            ""
        ).strip()

        print("DERIVADA:", expr)

        return {
            "respuesta": resolver_derivada(expr),
            "grafica": None
        }

    elif texto.startswith("deriva"):

        expr = texto.replace(
            "deriva",
            ""
        ).strip()

        print("DERIVADA:", expr)

        return {
            "respuesta": resolver_derivada(expr),
            "grafica": None
        }

    # =========================
    # INTEGRALES
    # =========================

    if "integral" in texto:

        expr = texto

        expr = expr.replace(
            "integral de",
            ""
        )

        expr = expr.replace(
            "integral",
            ""
        )

        expr = expr.strip()

        return {
            "respuesta": resolver_integral(expr),
            "grafica": None
        }

    # =========================
    # LIMITES
    # =========================

    if "limite" in texto or "límite" in texto:

        try:

            expr = texto

            expr = expr.replace(
                "limite de",
                ""
            )

            expr = expr.replace(
                "límite de",
                ""
            )

            expr = expr.strip()

            return {
                "respuesta": resolver_limite(
                    expr,
                    "infinito"
                ),
                "grafica": None
            }

        except Exception as e:

            print("ERROR LIMITE:", e)

            return {
                "respuesta": "Error resolviendo límite.",
                "grafica": None
            }

    # =========================
    # IA NORMAL
    # =========================

    return {
        "respuesta": preguntar_ia(pregunta),
        "grafica": None
    }
# ================================
# HOME
# ================================

@app.route("/")
def home():

    init_session()

    return render_template("index.html")

# ================================
# PREGUNTAR
# ================================

@app.route("/preguntar", methods=["POST"])
def preguntar():

    init_session()

    try:

        data = request.get_json()

        pregunta = data.get("pregunta", "")

        resultado = router(pregunta)

        session["contador_preguntas"] += 1

        return jsonify(resultado)

    except Exception as e:

        print("ERROR GENERAL:", e)

        return jsonify({
            "respuesta": "Error en servidor",
            "grafica": None
        })

# ================================
# DASHBOARD
# ================================

@app.route("/dashboard")
def dashboard():

    init_session()

    return jsonify({

        "preguntas_totales":
        session["contador_preguntas"],

        "graficas_generadas":
        len(session["historial_graficas"]),

        "mensajes_memoria":
        len(session["memoria_chat"])
    })

# ================================
# RUN
# ================================

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 10000))

    app.run(
        host="0.0.0.0",
        port=port
    )
