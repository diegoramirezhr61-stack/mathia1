from flask import Flask, render_template, request, jsonify
from groq import Groq
import os
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sympy import symbols, sympify, lambdify
import io
import base64

app = Flask(__name__)

# =========================================================
# GROQ
# =========================================================

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise Exception("Falta GROQ_API_KEY en variables de entorno")

cliente = Groq(api_key=api_key)

# =========================================================
# IA
# =========================================================

def preguntar_ia(prompt):

    try:

        respuesta = cliente.chat.completions.create(

            model="llama-3.1-8b-instant",

            messages=[
                {
                    "role": "system",
                    "content": (
                        "Eres MathIA v5.0, un asistente matemático experto. "
                        "Hablas español y explicas paso a paso."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            temperature=0.3,
            max_tokens=800
        )

        return respuesta.choices[0].message.content

    except Exception as e:
        return f"Error: {e}"
def generar_grafica(funcion_str):

    try:

        x = symbols('x')

        expresion = sympify(funcion_str)

        funcion = lambdify(x, expresion, "numpy")

        valores_x = np.linspace(-10, 10, 400)

        valores_y = funcion(valores_x)

        plt.figure(figsize=(5,5))

        plt.plot(valores_x, valores_y)

        plt.axhline(0, color='black')

        plt.axvline(0, color='black')

        plt.grid(True)

        img = io.BytesIO()

        plt.savefig(img, format='png')

        img.seek(0)

        grafica_base64 = base64.b64encode(img.getvalue()).decode()

        plt.close()

        return grafica_base64

    except Exception as e:

        return None
# =========================================================
# RUTAS
# =========================================================

@app.route("/")
def inicio():
    return render_template("index.html")

# =========================================================
# RUTA PARA PREGUNTAR
# =========================================================

@app.route("/preguntar", methods=["POST"])
def preguntar():

    datos = request.get_json()

    pregunta = datos.get("pregunta", "")

    respuesta = preguntar_ia(pregunta)

    grafica = None

    texto = pregunta.lower()

    if "grafica" in texto or "gráfica" in texto:

        try:

            funcion = (
                texto
                .replace("grafica", "")
                .replace("gráfica", "")
                .strip()
            )

            funcion = funcion.replace("^", "**")

            grafica = generar_grafica(funcion)

        except:
            grafica = None

    return jsonify({
        "respuesta": respuesta,
        "grafica": grafica
    })

# =========================================================
# INICIO
# =========================================================

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port
    )
