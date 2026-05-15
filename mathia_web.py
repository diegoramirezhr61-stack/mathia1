from flask import Flask, render_template, request, jsonify
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
texto_pdf_global = ""
memoria_chat = []
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

    global texto_pdf_global
    global memoria_chat

    try:

        mensajes = [

            {
                "role": "system",
                "content": (
                    "Eres MathIA v5.0, un asistente matemático experto. "
                    "Hablas español y explicas paso a paso."
                )
            }

        ]

        # =========================
        # MEMORIA
        # =========================

        mensajes.extend(memoria_chat)

        # =========================
        # MENSAJE ACTUAL
        # =========================

        mensaje_usuario = {
            "role": "user",
            "content": f"""
DOCUMENTO PDF:

{texto_pdf_global[:4000]}

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

        texto_respuesta = (
            respuesta
            .choices[0]
            .message
            .content
        )

        # =========================
        # GUARDAR MEMORIA
        # =========================

        memoria_chat.append({
            "role": "user",
            "content": prompt
        })

        memoria_chat.append({
            "role": "assistant",
            "content": texto_respuesta
        })

        # Limitar memoria
        memoria_chat = memoria_chat[-10:]

        return texto_respuesta

    except Exception as e:

        return f"Error IA: {e}"

# =========================================================
# GRÁFICAS
# =========================================================

def generar_grafica(funcion_str):

    try:

        x = symbols('x')

        expresion = sympify(funcion_str)

        funcion = lambdify(x, expresion, "numpy")

        valores_x = np.linspace(-10, 10, 400)

        valores_y = funcion(valores_x)

        plt.figure(figsize=(5, 5))

        plt.plot(valores_x, valores_y)

        plt.axhline(0, color='black')

        plt.axvline(0, color='black')

        plt.grid(True)

        img = io.BytesIO()

        plt.savefig(img, format='png')

        img.seek(0)

        grafica_base64 = base64.b64encode(
            img.getvalue()
        ).decode()

        plt.close()

        return grafica_base64

    except Exception as e:

        return None

# =========================================================
# MATEMÁTICAS
# =========================================================

def resolver_matematica(texto):

    x = symbols('x')

    texto = texto.lower()

    try:

        # =====================================
        # DERIVADAS
        # =====================================

        if texto.startswith("derivada de"):

            expr = (
                texto
                .replace("derivada de", "")
                .strip()
            )

            expr = expr.replace("^", "**")

            expr = sympify(expr)

            resultado = diff(expr, x)

            return f"Derivada:\n{resultado}"

        # =====================================
        # INTEGRALES
        # =====================================

        elif texto.startswith("integral de"):

            expr = (
                texto
                .replace("integral de", "")
                .strip()
            )

            expr = expr.replace("^", "**")

            expr = sympify(expr)

            resultado = integrate(expr, x)

            return f"Integral:\n{resultado}"

        # =====================================
        # MATRICES
        # =====================================

        elif "matriz" in texto:

            M = Matrix([
                [1, 2],
                [3, 4]
            ])

            return (
                f"Matriz:\n{M}\n\n"
                f"Determinante:\n{M.det()}\n\n"
                f"Inversa:\n{M.inv()}"
            )

        return None

    except Exception as e:

        return f"Error matemático: {e}"
    
def leer_pdf(archivo):

    texto = ""

    lector = PyPDF2.PdfReader(archivo)

    for pagina in lector.pages:

        contenido = pagina.extract_text()

        if contenido:
            texto += contenido + "\n"

    return texto

# =========================================================
# RUTAS
# =========================================================

@app.route("/")
def inicio():

    return render_template("index.html")

@app.route("/subir_pdf", methods=["POST"])
def subir_pdf():

    global texto_pdf_global

    if "pdf" not in request.files:

        return jsonify({
            "mensaje": "No se envió PDF"
        })

    archivo = request.files["pdf"]

    texto_pdf_global = leer_pdf(archivo)

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

    # =====================================
    # MATEMÁTICAS
    # =====================================

    respuesta_math = resolver_matematica(pregunta)

    if respuesta_math:
        respuesta = respuesta_math
    else:
        respuesta = preguntar_ia(pregunta)

    # =====================================
    # GRÁFICAS
    # =====================================

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
