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
# ROUTER MATEMÁTICO
# ================================

def router(pregunta):

    texto = pregunta.lower()

    # =========================
    # GRAFICAS
    # =========================

    if any(x in texto for x in [
    "grafica",
    "gráfica",
    "plot",
    "dibujar",
    "graficar"
]):

        funcion = texto

        funcion = funcion.replace("grafica", "")
        funcion = funcion.replace("gráfica", "")
        funcion = funcion.replace("plot", "")
        funcion = funcion.replace("dibujar", "")

        grafica = generar_grafica(funcion)

        return {
            "respuesta": "Gráfica generada correctamente.",
            "grafica": grafica
        }

    # =========================
    # LIMITES
    # =========================

    if "limite" in texto or "límite" in texto:

        prompt = f"""
Resuelve este límite paso a paso:

{pregunta}

Usa LaTeX y explica cada paso.
"""

        return {
            "respuesta": preguntar_ia(prompt),
            "grafica": None
        }

    # =========================
    # DERIVADAS
    # =========================

    if any(x in texto for x in [

        "deriva",
        "derivada",
        "f'",
        "dy/dx"

    ]):

        prompt = f"""
Resuelve esta derivada paso a paso:

{pregunta}

Usa reglas de derivación y LaTeX.
"""

        return {
            "respuesta": preguntar_ia(prompt),
            "grafica": None
        }

    # =========================
    # INTEGRALES
    # =========================

    if any(x in texto for x in [

        "integral",
        "∫"

    ]):

        prompt = f"""
Resuelve esta integral paso a paso:

{pregunta}

Si es sustitución explícalo.
Usa LaTeX.
"""

        return {
            "respuesta": preguntar_ia(prompt),
            "grafica": None
        }

    # =========================
    # CONTINUIDAD
    # =========================

    if "continuidad" in texto or "continua" in texto:

        prompt = f"""
Analiza la continuidad de la función:

{pregunta}

Explica dominio, límites laterales y conclusión.
"""

        return {
            "respuesta": preguntar_ia(prompt),
            "grafica": None
        }

    # =========================
    # NORMAL
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
