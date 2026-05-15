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
# IA
# =========================================================

def preguntar_ia(prompt, texto_pdf=""):

    try:

        if "memoria_chat" not in session:
            session["memoria_chat"] = []

        mensajes = [
            {
                "role": "system",
                "content": (
                    "Eres MathIA v5.0, asistente matemático experto. "
                    "Explicas paso a paso en español."
                )
            }
        ]

        mensajes.extend(session["memoria_chat"])

        mensaje_usuario = {
            "role": "user",
            "content": f"""
DOCUMENTO PDF (si existe):

{texto_pdf[:4000]}

PREGUNTA:
{prompt}
"""
        }

        mensajes.append(mensaje_usuario)

        respuesta = cliente.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=mensajes,
            temperature=0.3,
            max_tokens=800
        )

        texto_respuesta = respuesta.choices[0].message.content

        session["memoria_chat"].append({
            "role": "user",
            "content": prompt
        })

        session["memoria_chat"].append({
            "role": "assistant",
            "content": texto_respuesta
        })

        session["memoria_chat"] = session["memoria_chat"][-10:]

        return texto_respuesta

    except Exception as e:
        return f"Error IA: {e}"

# =========================================================
# PDF (MEJORADO)
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
# VALIDACIÓN PDF
# =========================================================

def validar_pdf(texto):

    if not texto:
        return False

    if len(texto.strip()) < 20:
        return False

    return True

# =========================================================
# RUTAS
# =========================================================

@app.route("/")
def inicio():
    return render_template("index.html")

@app.route("/subir_pdf", methods=["POST"])
def subir_pdf():

    if "pdf" not in request.files:
        return jsonify({"mensaje": "No se envió PDF"})

    archivo = request.files["pdf"]

    texto = leer_pdf(archivo)

    # guardar en sesión (NO global)
    session["texto_pdf"] = texto

    if not validar_pdf(texto):
        return jsonify({
            "mensaje": "⚠ PDF vacío o escaneado (usa OCR)"
        })

    return jsonify({
        "mensaje": "PDF cargado correctamente"
    })

# =========================================================
# PREGUNTAR
# =========================================================

@app.route("/preguntar", methods=["POST"])
def preguntar():

    datos = request.get_json()
    pregunta = datos.get("pregunta", "")

    texto_pdf = session.get("texto_pdf", "")

    # MATEMÁTICA IA
    respuesta = preguntar_ia(pregunta, texto_pdf)

    # GRÁFICAS
    grafica = None

    texto = pregunta.lower()

    if "grafica" in texto or "gráfica" in texto:

        try:
            funcion = texto.replace("grafica", "").replace("gráfica", "").strip()
            funcion = funcion.replace("^", "**")

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

        except:
            grafica = None

    return jsonify({
        "respuesta": respuesta,
        "grafica": grafica
    })

# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port
    )
