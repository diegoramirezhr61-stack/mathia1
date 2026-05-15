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

# =========================================================
# GROQ
# =========================================================

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise Exception("Falta GROQ_API_KEY en variables de entorno")

cliente = Groq(api_key=api_key)

# =========================================================
# PDF READER
# =========================================================

def leer_pdf(archivo):
    texto = ""
    try:
        lector = PyPDF2.PdfReader(archivo)

        for pagina in lector.pages:
            contenido = pagina.extract_text()
            if contenido:
                texto += contenido + "\n"

    except Exception:
        return ""

    return texto.strip()

# =========================================================
# CHUNKS SYSTEM
# =========================================================

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


def buscar_chunks(pregunta, chunks, top_k=3):
    pregunta = pregunta.lower()
    palabras = set(pregunta.split())

    scored = []

    for chunk in chunks:
        score = 0
        for palabra in palabras:
            if palabra in chunk.lower():
                score += 1
        scored.append((score, chunk))

    scored.sort(reverse=True, key=lambda x: x[0])

    return [c[1] for c in scored[:top_k]]

# =========================================================
# IA
# =========================================================

def preguntar_ia(prompt):

    try:

        if "memoria_chat" not in session:
            session["memoria_chat"] = []

        modo_pdf = session.get("modo_pdf", False)
        chunks = session.get("pdf_chunks", [])

        mensajes = [
            {
                "role": "system",
                "content": "Eres MathIA. Respondes en español claro y explicas paso a paso."
            }
        ]

        mensajes.extend(session["memoria_chat"])

        if modo_pdf and chunks:

            relevantes = buscar_chunks(prompt, chunks)
            contexto = "\n\n".join(relevantes)

            contenido = f"""
RESPONDE SOLO CON BASE EN ESTE CONTEXTO DEL DOCUMENTO:

{contexto}

PREGUNTA:
{prompt}
"""
        else:
            contenido = prompt

        mensajes.append({"role": "user", "content": contenido})

        respuesta = cliente.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=mensajes,
            temperature=0.3,
            max_tokens=800
        )

        texto_respuesta = respuesta.choices[0].message.content

        session["memoria_chat"].append({"role": "user", "content": prompt})
        session["memoria_chat"].append({"role": "assistant", "content": texto_respuesta})

        session["memoria_chat"] = session["memoria_chat"][-10:]

        return texto_respuesta

    except Exception as e:
        return f"Error IA: {e}"

# =========================================================
# NORMALIZAR FUNCIÓN (GRÁFICAS)
# =========================================================

def normalizar_funcion(texto):

    texto = texto.lower()

    palabras_grafica = ["grafica", "gráfica", "plot", "dibujar", "dibújame"]

    for p in palabras_grafica:
        texto = texto.replace(p, "")

    texto = texto.strip()

    # lenguaje natural
    texto = texto.replace("x al cuadrado", "x**2")
    texto = texto.replace("x al cubo", "x**3")
    texto = texto.replace("cuadrado", "**2")
    texto = texto.replace("cubo", "**3")

    texto = texto.replace("seno", "sin(x)")
    texto = texto.replace("coseno", "cos(x)")
    texto = texto.replace("tangente", "tan(x)")

    texto = texto.replace("^", "**")

    if not texto:
        texto = "x**2"

    return texto

# =========================================================
# PDF UPLOAD
# =========================================================

@app.route("/subir_pdf", methods=["POST"])
def subir_pdf():

    if "pdf" not in request.files:
        return jsonify({"mensaje": "No se envió PDF"})

    archivo = request.files["pdf"]
    texto = leer_pdf(archivo)

    if len(texto) < 20:
        return jsonify({"mensaje": "⚠ PDF vacío o escaneado"})

    session["pdf_chunks"] = dividir_texto(texto)

    return jsonify({
        "mensaje": f"PDF procesado en {len(session['pdf_chunks'])} bloques"
    })

# =========================================================
# MODO PDF
# =========================================================

@app.route("/modo_pdf", methods=["POST"])
def modo_pdf():

    data = request.get_json()
    session["modo_pdf"] = data.get("estado", False)

    return jsonify({"modo_pdf": session["modo_pdf"]})

# =========================================================
# PREGUNTAR + GRÁFICAS
# =========================================================

@app.route("/preguntar", methods=["POST"])
def preguntar():

    datos = request.get_json()
    pregunta = datos.get("pregunta", "")

    respuesta = preguntar_ia(pregunta)

    grafica = None
    texto = pregunta.lower()

    palabras_grafica = ["grafica", "gráfica", "plot", "dibujar", "dibújame"]

    if any(p in texto for p in palabras_grafica):

        try:
            funcion = normalizar_funcion(pregunta)

            x = symbols('x')
            expr = sympify(funcion)
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
            img.seek(0)

            grafica = base64.b64encode(img.getvalue()).decode()
            plt.close()

        except Exception as e:
            print("Error grafica:", e)
            grafica = None

    return jsonify({
        "respuesta": respuesta,
        "grafica": grafica
    })

# =========================================================
# INICIO
# =========================================================

@app.route("/")
def inicio():
    return render_template("index.html")

# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port
    )
