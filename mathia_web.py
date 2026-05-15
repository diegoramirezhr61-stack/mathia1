from flask import Flask, render_template, request, jsonify, session
from groq import Groq
import os
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sympy import *
import io
import base64
import PyPDF2

app = Flask(__name__)
app.secret_key = "mathia_secret"

# =========================
# GROQ
# =========================

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise Exception("Falta GROQ_API_KEY")

cliente = Groq(api_key=api_key)

# =========================
# SESSION INIT
# =========================

def init_session():
    if "memoria_chat" not in session:
        session["memoria_chat"] = []

    if "pdf_chunks" not in session:
        session["pdf_chunks"] = []

    if "modo_pdf" not in session:
        session["modo_pdf"] = False

    if "historial_graficas" not in session:
        session["historial_graficas"] = []

    if "contador_preguntas" not in session:
        session["contador_preguntas"] = 0


# =========================
# PDF
# =========================

def leer_pdf(archivo):
    try:
        lector = PyPDF2.PdfReader(archivo)
        texto = ""
        for p in lector.pages:
            t = p.extract_text()
            if t:
                texto += t + "\n"
        return texto.strip()
    except Exception as e:
        print(f"Error leyendo PDF: {e}")
        return ""


def dividir_texto(texto, max_chars=800):
    palabras = texto.split()
    chunks = []
    actual = ""

    for p in palabras:
        if len(actual) + len(p) < max_chars:
            actual += " " + p
        else:
            chunks.append(actual.strip())
            actual = p

    if actual:
        chunks.append(actual.strip())

    return chunks


# =========================
# NORMALIZAR FUNCIÓN
# =========================

def normalizar_funcion(texto):
    texto = texto.lower()

    borrar = ["grafica", "gráfica", "plot", "dibujar", "dibújame"]
    for p in borrar:
        texto = texto.replace(p, "")

    texto = texto.strip()

    # lenguaje natural → matemático
    texto = texto.replace("x al cuadrado", "x^2")
    texto = texto.replace("x al cubo", "x^3")
    texto = texto.replace("al cuadrado", "^2")
    texto = texto.replace("al cubo", "^3")

    texto = texto.replace("seno", "sin")
    texto = texto.replace("coseno", "cos")
    texto = texto.replace("tangente", "tan")

    texto = texto.replace("^", "**")
    texto = texto.replace("²", "**2")
    texto = texto.replace("³", "**3")

    return texto


# =========================
# IA
# =========================

def preguntar_ia(prompt):
    init_session()

    mensajes = [
        {
            "role": "system",
            "content": "Eres MathIA. Explicas matemáticas paso a paso."
        }
    ]

    mensajes.extend(session["memoria_chat"])

    contenido = prompt

    mensajes.append({"role": "user", "content": contenido})

    respuesta = cliente.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=mensajes,
        temperature=0.3,
        max_tokens=800
    )

    texto = respuesta.choices[0].message.content

    session["memoria_chat"].append({"role": "user", "content": prompt})
    session["memoria_chat"].append({"role": "assistant", "content": texto})

    session["memoria_chat"] = session["memoria_chat"][-10:]

    return texto


# =========================
# GRAFICA SEGURA (FIX REAL)
# =========================

def generar_grafica(funcion):
    try:
        if not funcion:
            funcion = "x**2"

        funcion = funcion.replace("^", "**")

        x = symbols("x")

        expr = sympify(funcion, evaluate=False)
        f = lambdify(x, expr, "numpy")

        xs = np.linspace(-10, 10, 400)
        ys = f(xs)

        plt.figure()
        plt.plot(xs, ys)
        plt.axhline(0)
        plt.axvline(0)
        plt.grid()

        img = io.BytesIO()
        plt.savefig(img, format="png")
        plt.close()

        img.seek(0)
        return base64.b64encode(img.getvalue()).decode()

    except Exception as e:
        print("ERROR GRAFICA:", e)
        return None


# =========================
# RUTAS
# =========================

@app.route("/")
def home():
    init_session()
    return render_template("index.html")


@app.route("/preguntar", methods=["POST"])
def preguntar():
    init_session()

    try:
        data = request.get_json()
        pregunta = data.get("pregunta", "")

        texto = pregunta.lower()
        palabras = ["grafica", "gráfica", "plot", "dibujar", "dibújame"]

        es_grafica = any(p in texto for p in palabras)

        grafica = None

        # =========================
        # 🔥 SI ES GRÁFICA → NO IA
        # =========================
        if es_grafica:

            funcion = normalizar_funcion(pregunta)

            funcion = funcion.replace("grafica", "")
            funcion = funcion.replace("gráfica", "")
            funcion = funcion.strip()

            if not funcion:
                funcion = "x**2"

            grafica = generar_grafica(funcion)

            return jsonify({
                "respuesta": "Gráfica generada correctamente.",
                "grafica": grafica
            })

        # =========================
        # 🔥 SI NO ES GRÁFICA → IA
        # =========================
        respuesta = preguntar_ia(pregunta)

        return jsonify({
            "respuesta": respuesta,
            "grafica": None
        })

    except Exception as e:
        print("ERROR GENERAL:", e)
        return jsonify({
            "respuesta": "Error en servidor",
            "grafica": None
        })

        # =========================
        # IA SIEMPRE RESPONDE
        # =========================
        respuesta = preguntar_ia(pregunta)

        grafica = None

        # =========================
        # GRAFICA (NO BLOQUEA)
        # =========================
        if es_grafica:
            try:
                funcion = normalizar_funcion(pregunta)

                for p in palabras:
                    funcion = funcion.replace(p, "")

                funcion = funcion.strip()

                if not funcion:
                    funcion = "x**2"

                grafica = generar_grafica(funcion)

                if grafica:
                    session["historial_graficas"].append(grafica)
                    session["historial_graficas"] = session["historial_graficas"][-10:]

            except Exception as e:
                print("ERROR GRAFICA:", e)
                grafica = None

        return jsonify({
            "respuesta": respuesta,
            "grafica": grafica
        })

    except Exception as e:
        print("ERROR GENERAL:", e)
        return jsonify({
            "respuesta": "Error en servidor",
            "grafica": None
        })


@app.route("/dashboard")
def dashboard():
    init_session()

    return jsonify({
        "preguntas_totales": session["contador_preguntas"],
        "graficas_generadas": len(session["historial_graficas"]),
        "memoria_chat": len(session["memoria_chat"])
    })


# =========================
# RUN
# =========================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))

    app.run(
        host="0.0.0.0",
        port=port
    )
