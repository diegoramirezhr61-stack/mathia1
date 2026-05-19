from flask import Flask, render_template, request, jsonify, session
from groq import Groq
import os

# =========================
# MATPLOTLIB
# =========================

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

# =========================
# SYMPY
# =========================

from sympy import *

# =========================
# UTILS
# =========================

import io
import base64
import PyPDF2

# =========================
# OCR
# =========================

import pytesseract
from PIL import Image
import easyocr

# =========================
# APP
# =========================

app = Flask(__name__)
app.secret_key = "mathia_secret"

# =========================
# TESSERACT WINDOWS
# =========================

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

# =========================
# EASY OCR
# =========================

lector_ocr = easyocr.Reader(['es', 'en'])

# =========================
# GROQ
# =========================

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise Exception("Falta GROQ_API_KEY")

cliente = Groq(api_key=api_key)

# =========================
# SESSION
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

    for p in palabras:

        if len(actual) + len(p) < max_chars:

            actual += " " + p

        else:

            chunks.append(actual.strip())

            actual = p

    if actual:
        chunks.append(actual.strip())

    return chunks

def buscar_chunks(pregunta, chunks, top_k=3):

    pregunta = pregunta.lower()

    palabras = set(pregunta.split())

    scores = []

    for c in chunks:

        score = sum(
            1 for p in palabras
            if p in c.lower()
        )

        scores.append((score, c))

    scores.sort(
        reverse=True,
        key=lambda x: x[0]
    )

    return [c[1] for c in scores[:top_k]]

# =========================
# NORMALIZAR FUNCION
# =========================

def normalizar_funcion(texto):

    texto = texto.lower()

    borrar = [
        "grafica",
        "gráfica",
        "plot",
        "dibujar",
        "dibújame"
    ]

    for p in borrar:
        texto = texto.replace(p, "")

    texto = texto.strip()

    # lenguaje natural
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

    # =========================
    # CONTEXTO PDF
    # =========================

    if session["modo_pdf"] and session["pdf_chunks"]:

        relevantes = buscar_chunks(
            prompt,
            session["pdf_chunks"]
        )

        contexto = "\n\n".join(relevantes)

        contenido = f"""
DOCUMENTO:

{contexto}

PREGUNTA:

{prompt}
"""

    else:

        contenido = prompt

    mensajes.append({
        "role": "user",
        "content": contenido
    })

    respuesta = cliente.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=mensajes,
        temperature=0.3,
        max_tokens=800
    )

    texto = respuesta.choices[0].message.content

    # memoria
    session["memoria_chat"].append({
        "role": "user",
        "content": prompt
    })

    session["memoria_chat"].append({
        "role": "assistant",
        "content": texto
    })

    session["memoria_chat"] = (
        session["memoria_chat"][-10:]
    )

    return texto

# =========================
# GRAFICAS
# =========================

def generar_grafica(funcion):

    try:

        if not funcion:
            funcion = "x**2"

        # seguridad
        if "__" in funcion or "import" in funcion:
            funcion = "x**2"

        funcion = funcion.replace("^", "**")

        x = symbols("x")

        expr = sympify(funcion)

        f = lambdify(x, expr, "numpy")

        xs = np.linspace(-10, 10, 500)

        ys = f(xs)

        plt.figure(figsize=(6,4))

        plt.plot(xs, ys)

        plt.axhline(0)
        plt.axvline(0)

        plt.grid(True)

        img = io.BytesIO()

        plt.savefig(
            img,
            format="png",
            bbox_inches="tight"
        )

        plt.close()

        img.seek(0)

        return base64.b64encode(
            img.getvalue()
        ).decode()

    except Exception as e:

        print("ERROR GRAFICA:", e)

        return None

# =========================
# OCR
# =========================

def leer_imagen_matematica(imagen):

    try:

        img = Image.open(imagen)

        # =========================
        # TESSERACT
        # =========================

        texto = pytesseract.image_to_string(
            img,
            lang='eng'
        )

        # =========================
        # EASYOCR FALLBACK
        # =========================

        if len(texto.strip()) < 2:

            resultado = lector_ocr.readtext(
                np.array(img)
            )

            texto = " ".join(
                [x[1] for x in resultado]
            )

        return texto.strip()

    except Exception as e:

        print("ERROR OCR:", e)

        return ""

# =========================
# HOME
# =========================

@app.route("/")
def home():

    init_session()

    return render_template("index.html")

# =========================
# SUBIR PDF
# =========================

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

        if len(texto) < 20:

            return jsonify({
                "mensaje": "PDF vacío"
            })

        session["pdf_chunks"] = dividir_texto(texto)

        session["modo_pdf"] = True

        return jsonify({
            "mensaje": "PDF cargado correctamente"
        })

    except Exception as e:

        print("ERROR PDF:", e)

        return jsonify({
            "mensaje": "Error procesando PDF"
        })

# =========================
# PREGUNTAR
# =========================

@app.route("/preguntar", methods=["POST"])
def preguntar():

    init_session()

    try:

        data = request.get_json()

        pregunta = data.get("pregunta", "")

        session["contador_preguntas"] += 1

        texto = pregunta.lower()

        palabras_grafica = [
            "grafica",
            "gráfica",
            "plot",
            "dibujar",
            "dibújame"
        ]

        es_grafica = any(
            p in texto
            for p in palabras_grafica
        )

        # =========================
        # GRAFICA
        # =========================

        if es_grafica:

            funcion = normalizar_funcion(
                pregunta
            )

            grafica = generar_grafica(
                funcion
            )

            return jsonify({
                "respuesta": f"""
He generado la gráfica de:

${funcion.replace('**','^')}$
""",
                "grafica": grafica
            })

        # =========================
        # IA NORMAL
        # =========================

        respuesta = preguntar_ia(
            pregunta
        )

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
# ANALIZAR IMAGEN
# =========================

@app.route("/resolver_imagen", methods=["POST"])
def resolver_imagen():

    try:

        if "imagen" not in request.files:

            return jsonify({
                "respuesta": "No se recibió imagen"
            })

        imagen = request.files["imagen"]

        texto = leer_imagen_matematica(
            imagen
        )

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

# =========================
# DASHBOARD
# =========================

@app.route("/dashboard")
def dashboard():

    init_session()

    return jsonify({
        "preguntas_totales":
            session["contador_preguntas"],

        "graficas_generadas":
            len(session["historial_graficas"]),

        "memoria_chat":
            len(session["memoria_chat"])
    })

# =========================
# RUN
# =========================

if __name__ == "__main__":

    port = int(
        os.environ.get("PORT", 10000)
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=True
    )
