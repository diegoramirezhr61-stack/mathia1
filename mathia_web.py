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
import io
import base64
import PyPDF2
import pytesseract

from PIL import Image
import cv2


# ========================================
# TESSERACT WINDOWS
# ========================================

if os.name == "nt":
    pytesseract.pytesseract.tesseract_cmd = (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    )
# ========================================
# APP
# ========================================

app = Flask(__name__)

app.secret_key = "mathia_secret"

# ========================================
# GROQ
# ========================================

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise Exception("Falta GROQ_API_KEY")

cliente = Groq(api_key=api_key)

# ========================================
# SESSION
# ========================================

def init_session():

    if "memoria_chat" not in session:
        session["memoria_chat"] = []

    if "pdf_chunks" not in session:
        session["pdf_chunks"] = []

    if "contador_preguntas" not in session:
        session["contador_preguntas"] = 0

    if "historial_graficas" not in session:
        session["historial_graficas"] = []

# ========================================
# LEER PDF
# ========================================

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

# ========================================
# DIVIDIR TEXTO
# ========================================

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

# ========================================
# NORMALIZAR FUNCION
# ========================================

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

    texto = texto.replace("seno", "sin")
    texto = texto.replace("coseno", "cos")
    texto = texto.replace("tangente", "tan")

    texto = texto.replace("^", "**")

    texto = texto.replace("²", "**2")
    texto = texto.replace("³", "**3")

    return texto

# ========================================
# IA
# ========================================

def preguntar_ia(prompt):

    init_session()

    mensajes = [
        {
            "role": "system",
            "content": """
Eres MathIA.

Explicas matemáticas paso a paso.

IMPORTANTE:
Cuando escribas fórmulas matemáticas usa formato LaTeX.

Ejemplos:

$x^2$

$\\frac{a}{b}$

$\\int x^2 dx$

Usa SIEMPRE símbolos matemáticos bien formateados.
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

# ========================================
# GRAFICAS
# ========================================

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

def leer_imagen_matematica(imagen):

    try:

        # =========================
        # ABRIR IMAGEN
        # =========================

        img = Image.open(imagen).convert("L")

        # PIL -> NUMPY
        img = np.array(img)

        # =========================
        # MEJORAR OCR
        # =========================

        img = cv2.resize(
            img,
            None,
            fx=2,
            fy=2,
            interpolation=cv2.INTER_CUBIC
        )

        img = cv2.GaussianBlur(
            img,
            (3,3),
            0
        )

        img = cv2.threshold(
            img,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )[1]

        # =========================
        # OCR
        # =========================

        texto = pytesseract.image_to_string(
            img,
            config='--psm 6'
        )

        print("TEXTO OCR:", texto)

        return texto.strip()

    except Exception as e:

        print("ERROR OCR:", e)

        return ""

# ========================================
# HOME
# ========================================

@app.route("/")
def home():

    init_session()

    return render_template("index.html")

# ========================================
# PREGUNTAR
# ========================================

@app.route("/preguntar", methods=["POST"])
def preguntar():

    init_session()

    try:

        data = request.get_json()

        pregunta = data.get("pregunta", "")

        texto = pregunta.lower()

        palabras = [
            "grafica",
            "gráfica",
            "plot",
            "dibujar",
            "dibújame"
        ]

        es_grafica = any(
            p in texto for p in palabras
        )

        grafica = None

        # ========================================
        # GRAFICA
        # ========================================

        if es_grafica:

            funcion = normalizar_funcion(pregunta)

            if not funcion:
                funcion = "x**2"

            grafica = generar_grafica(funcion)

            session["historial_graficas"].append(
                grafica
            )

            session["historial_graficas"] = (
                session["historial_graficas"][-10:]
            )

            return jsonify({
                "respuesta": "Gráfica generada correctamente.",
                "grafica": grafica
            })

        # ========================================
        # IA
        # ========================================

        respuesta = preguntar_ia(pregunta)

        session["contador_preguntas"] += 1

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

# ========================================
# SUBIR PDF
# ========================================

@app.route("/subir_pdf", methods=["POST"])
def subir_pdf():

    init_session()

    try:

        if "pdf" not in request.files:

            return jsonify({
                "mensaje": "No se recibió PDF"
            })

        archivo = request.files["pdf"]

        texto = leer_pdf(archivo)

        if not texto:

            return jsonify({
                "mensaje": "No pude leer el PDF"
            })

        chunks = dividir_texto(texto)

        session["pdf_chunks"] = chunks

        return jsonify({
            "mensaje": f"PDF cargado correctamente. {len(chunks)} fragmentos detectados."
        })

    except Exception as e:

        print("ERROR PDF:", e)

        return jsonify({
            "mensaje": "Error procesando PDF"
        })

# ========================================
# RESOLVER IMAGEN
# ========================================

@app.route("/resolver_imagen", methods=["POST"])
def resolver_imagen():

    try:

        if "imagen" not in request.files:

            return jsonify({
                "respuesta": "No se recibió imagen"
            })

        imagen = request.files["imagen"]

        texto = leer_imagen_matematica(imagen)

        if not texto:

            return jsonify({
                "respuesta": "No pude leer la imagen"
            })

        prompt = f"""
Resuelve paso a paso este ejercicio matemático:

{texto}
"""

        respuesta = preguntar_ia(prompt)

        return jsonify({

            "texto_detectado": texto,

            "respuesta": respuesta
        })

    except Exception as e:

        print("ERROR IMAGEN:", e)

        return jsonify({
            "respuesta": "Error analizando imagen"
        })

# ========================================
# DASHBOARD
# ========================================

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

# ========================================
# RUN
# ========================================

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 10000))

    app.run(
        host="0.0.0.0",
        port=port
    )
