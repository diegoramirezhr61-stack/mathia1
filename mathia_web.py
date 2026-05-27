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
# IA
# ================================

def preguntar_ia(prompt):

    init_session()

    mensajes = [
        {
            "role": "system",
            "content": """
Eres MathIA.

Explicas matemáticas paso a paso.

Usa LaTeX para las fórmulas matemáticas.
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

        temperature=0.3,

        max_tokens=800
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

    session["memoria_chat"] = session["memoria_chat"][-10:]

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
# GRAFICAS
# ================================

def generar_grafica(funcion):

    try:

        if not funcion:
            funcion = "x**2"

        funcion = funcion.replace("^", "**")

        x = symbols("x")

        expr = sympify(funcion)

        f = lambdify(x, expr, "numpy")

        xs = np.linspace(-10, 10, 400)

        ys = f(xs)

        if np.any(np.isnan(ys)) or np.any(np.isinf(ys)):
            return None

        plt.figure(figsize=(7,5))

        plt.plot(xs, ys)

        plt.axhline(0)
        plt.axvline(0)

        plt.grid()

        img = io.BytesIO()

        plt.savefig(img, format="png")

        plt.close()

        img.seek(0)

        return base64.b64encode(
            img.getvalue()
        ).decode()

    except Exception as e:

        print("ERROR GRAFICA:", e)

        return None

# ================================
# ROUTER
# ================================

def router(pregunta):

    texto = pregunta.lower()

    # 📊 GRAFICAS

    if any(x in texto for x in [
        "grafica",
        "gráfica",
        "plot",
        "dibujar"
    ]):

        funcion = normalizar_funcion(pregunta)

        grafica = generar_grafica(funcion)

        session["historial_graficas"].append(grafica)

        session["historial_graficas"] = (
            session["historial_graficas"][-10:]
        )

        return {
            "respuesta": "Gráfica generada correctamente.",
            "grafica": grafica
        }

    # 💬 IA NORMAL

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
