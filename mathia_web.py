from flask import Flask, render_template, request, jsonify, session
from groq import Groq
import google.generativeai as genai

import os
import io
import base64

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from sympy import *

from PIL import Image
import PyPDF2

# ================================
# APP
# ================================

app = Flask(__name__)
app.secret_key = "mathia_secret"

# ================================
# APIs
# ================================

GROQ_KEY = os.getenv("GROQ_API_KEY")
GEMINI_KEY = os.getenv("GEMINI_API_KEY")

if not GROQ_KEY:
    raise Exception("Falta GROQ_API_KEY")

if not GEMINI_KEY:
    raise Exception("Falta GEMINI_API_KEY")

cliente = Groq(api_key=GROQ_KEY)

genai.configure(api_key=GEMINI_KEY)

modelo_gemini = genai.GenerativeModel("gemini-1.5-flash")

# ================================
# SESSION
# ================================

def init_session():
    if "memoria_chat" not in session:
        session["memoria_chat"] = []

    if "pdf_chunks" not in session:
        session["pdf_chunks"] = []

    if "contador_preguntas" not in session:
        session["contador_preguntas"] = 0

    if "historial_graficas" not in session:
        session["historial_graficas"] = []

# ================================
# PDF
# ================================

def leer_pdf(archivo):
    try:
        lector = PyPDF2.PdfReader(archivo)
        texto = ""

        for pagina in lector.pages:
            contenido = pagina.extract_text()
            if contenido:
                texto += contenido + "\n"

        return texto.strip()

    except Exception as e:
        print("ERROR PDF:", e)
        return ""

def dividir_texto(texto, max_chars=800):
    palabras = texto.split()
    chunks = []
    actual = ""

    for palabra in palabras:
        if len(actual) + len(palabra) < max_chars:
            actual += " " + palabra
        else:
            chunks.append(actual.strip())
            actual = palabra

    if actual:
        chunks.append(actual.strip())

    return chunks

# ================================
# IA GROQ
# ================================

def preguntar_ia(prompt):

    init_session()

    mensajes = [
        {
            "role": "system",
            "content": "Eres MathIA, explicas matemáticas paso a paso con LaTeX."
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

    session["memoria_chat"].append({"role": "user", "content": prompt})
    session["memoria_chat"].append({"role": "assistant", "content": texto})

    session["memoria_chat"] = session["memoria_chat"][-10:]

    return texto

# ================================
# GEMINI VISION (FIX REAL)
# ================================

def analizar_imagen_con_ia(imagen):

    try:
        img_bytes = imagen.read()

        response = modelo_gemini.generate_content([
            {
                "role": "user",
                "parts": [
                    "Resuelve el ejercicio matemático paso a paso usando LaTeX.",
                    {
                        "mime_type": "image/png",
                        "data": img_bytes
                    }
                ]
            }
        ])

        return response.text

    except Exception as e:
        print("ERROR GEMINI:", e)
        return "No pude analizar la imagen."

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

        plt.figure(figsize=(7,5))
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

# ================================
# ROUTER (CEREBRO)
# ================================

def router(pregunta, imagen=None):

    texto = pregunta.lower()

    # 🖼️ Imagen
    if imagen:
        return {
            "respuesta": analizar_imagen_con_ia(imagen),
            "grafica": None
        }

    # 📊 Gráficas
    if any(x in texto for x in ["grafica", "gráfica", "plot", "dibujar"]):

        funcion = pregunta.lower()
        funcion = funcion.replace("grafica", "").replace("gráfica", "")

        grafica = generar_grafica(funcion)

        return {
            "respuesta": "Gráfica generada correctamente.",
            "grafica": grafica
        }

    # 📄 PDF contexto
    if session.get("pdf_chunks"):
        contexto = " ".join(session["pdf_chunks"][:3])
        pregunta = f"{pregunta}\nContexto PDF:\n{contexto}"

    # 💬 IA normal
    return {
        "respuesta": preguntar_ia(pregunta),
        "grafica": None
    }

# ================================
# ROUTES
# ================================

@app.route("/")
def home():
    init_session()
    return render_template("index.html")

@app.route("/preguntar", methods=["POST"])
def preguntar():

    init_session()

    data = request.get_json()
    pregunta = data.get("pregunta", "")

    resultado = router(pregunta)

    session["contador_preguntas"] += 1

    return jsonify(resultado)

@app.route("/resolver_imagen", methods=["POST"])
def resolver_imagen():

    try:
        if "imagen" not in request.files:
            return jsonify({"respuesta": "No se recibió imagen"})

        imagen = request.files["imagen"]

        respuesta = router("", imagen=imagen)

        return jsonify({
            "texto_detectado": "Imagen procesada",
            "respuesta": respuesta["respuesta"]
        })

    except Exception as e:
        print("ERROR IMAGEN:", e)
        return jsonify({"respuesta": "Error analizando imagen"})

@app.route("/subir_pdf", methods=["POST"])
def subir_pdf():

    init_session()

    try:
        if "pdf" not in request.files:
            return jsonify({"mensaje": "No PDF"})

        archivo = request.files["pdf"]
        texto = leer_pdf(archivo)

        chunks = dividir_texto(texto)
        session["pdf_chunks"] = chunks

        return jsonify({"mensaje": f"PDF cargado: {len(chunks)} fragmentos"})

    except Exception as e:
        print("ERROR PDF:", e)
        return jsonify({"mensaje": "Error PDF"})

@app.route("/dashboard")
def dashboard():

    init_session()

    return jsonify({
        "preguntas": session["contador_preguntas"],
        "graficas": len(session["historial_graficas"]),
        "memoria": len(session["memoria_chat"])
    })

# ================================
# RUN
# ================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))

    app.run(host="0.0.0.0", port=port)
