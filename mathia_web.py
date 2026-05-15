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
# INIT SESSION
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
    texto = ""
    try:
        lector = PyPDF2.PdfReader(archivo)
        for pagina in lector.pages:
            contenido = pagina.extract_text()
            if contenido:
                texto += contenido + "\n"
    except:
        return ""
    return texto.strip()


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

    scored = []
    for c in chunks:
        score = sum(1 for p in palabras if p in c.lower())
        scored.append((score, c))

    scored.sort(reverse=True, key=lambda x: x[0])
    return [c[1] for c in scored[:top_k]]


# =========================
# NORMALIZAR FUNCION
# =========================

def normalizar_funcion(texto):
    texto = texto.lower()

    palabras = ["grafica", "gráfica", "plot", "dibujar", "dibújame"]
    for p in palabras:
        texto = texto.replace(p, "")

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


# =========================
# IA
# =========================

def preguntar_ia(prompt):
    init_session()

    modo_pdf = session["modo_pdf"]
    chunks = session["pdf_chunks"]

    mensajes = [
        {
            "role": "system",
            "content": "Eres MathIA. Explicas matemáticas paso a paso."
        }
    ]

    mensajes.extend(session["memoria_chat"])

    if modo_pdf and chunks:
        relevantes = buscar_chunks(prompt, chunks)
        contexto = "\n\n".join(relevantes)

        contenido = f"""
DOCUMENTO:
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

    texto = respuesta.choices[0].message.content

    session["memoria_chat"].append({"role": "user", "content": prompt})
    session["memoria_chat"].append({"role": "assistant", "content": texto})

    session["memoria_chat"] = session["memoria_chat"][-10:]

    return texto


# =========================
# GRAFICA
# =========================

def generar_grafica(funcion):
    try:
        x = symbols("x")
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

        return base64.b64encode(img.getvalue()).decode()

    except Exception as e:
        print("ERROR GRAFICA:", e)
        return None


# =========================
# ROUTES
# =========================

@app.route("/")
def home():
    init_session()
    return render_template("index.html")


@app.route("/subir_pdf", methods=["POST"])
def subir_pdf():
    init_session()

    if "pdf" not in request.files:
        return jsonify({"mensaje": "No PDF"})

    archivo = request.files["pdf"]
    texto = leer_pdf(archivo)

    if len(texto) < 20:
        return jsonify({"mensaje": "PDF vacío"})

    session["pdf_chunks"] = dividir_texto(texto)

    return jsonify({
        "mensaje": "PDF cargado",
        "chunks": len(session["pdf_chunks"])
    })


@app.route("/preguntar", methods=["POST"])
def preguntar():
    init_session()

    try:
        datos = request.get_json()
        pregunta = datos.get("pregunta", "")

        session["contador_preguntas"] += 1

        texto = pregunta.lower()
        palabras_grafica = ["grafica", "gráfica", "plot", "dibujar", "dibújame"]

        es_grafica = any(p in texto for p in palabras_grafica)

        grafica = None

        # =========================
        # IA SIEMPRE RESPONDE
        # =========================
        respuesta = preguntar_ia(pregunta)

        # =========================
        # GRAFICA SEGURA
        # =========================
        if es_grafica:
            try:
                funcion = normalizar_funcion(pregunta)

                funcion = funcion.lower()
                for p in palabras_grafica:
                    funcion = funcion.replace(p, "")

                funcion = funcion.strip()

                if funcion == "" or len(funcion) < 2:
                    funcion = "x**2"

                if "import" in funcion or "__" in funcion:
                    funcion = "x**2"

                x = symbols("x")
                expr = sympify(funcion, evaluate=True)
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
    app.run(host="0.0.0.0", port=5000)
