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
# PDF LECTOR
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
# IA
# =========================================================

def preguntar_ia(prompt):

    try:

        if "memoria_chat" not in session:
            session["memoria_chat"] = []

        texto_pdf = session.get("texto_pdf", "")
        modo_pdf = session.get("modo_pdf", False)

        mensajes = [
            {
                "role": "system",
                "content": (
                    "Eres MathIA. Respondes en español de forma clara."
                )
            }
        ]

        mensajes.extend(session["memoria_chat"])

        # =========================
        # MODO PDF ACTIVADO
        # =========================
        if modo_pdf:

            contenido = f"""
ESTÁS EN MODO PDF.
Reglas:
- SOLO usa este documento
- Si no está la respuesta, di: "No está en el documento"

DOCUMENTO:
{texto_pdf[:4000]}

PREGUNTA:
{prompt}
"""

        # =========================
        # MODO NORMAL
        # =========================
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
# PDF UPLOAD
# =========================================================

@app.route("/subir_pdf", methods=["POST"])
def subir_pdf():

    if "pdf" not in request.files:
        return jsonify({"mensaje": "No se envió PDF"})

    archivo = request.files["pdf"]

    texto = leer_pdf(archivo)

    session["texto_pdf"] = texto

    if len(texto) < 20:
        return jsonify({
            "mensaje": "⚠ PDF vacío o escaneado"
        })

    return jsonify({
        "mensaje": "PDF cargado correctamente"
    })

# =========================================================
# MODO PDF ON/OFF
# =========================================================

@app.route("/modo_pdf", methods=["POST"])
def modo_pdf():

    data = request.get_json()
    estado = data.get("estado", False)

    session["modo_pdf"] = estado

    return jsonify({
        "modo_pdf": session["modo_pdf"]
    })

# =========================================================
# PREGUNTAR
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
