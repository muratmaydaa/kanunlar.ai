import os
from flask import Flask, render_template, request, jsonify, Response
from dotenv import load_dotenv
from openai import OpenAI
from utils.vector_db import load_vector_db, get_relevant_documents

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

app = Flask(__name__)

@app.route("/")
def chat_page():
    return render_template("chat.html")

@app.route("/get_response", methods=["POST"])
def get_response():
    user_message = request.json["message"]
    return jsonify({"message": user_message})

@app.route("/get_response_stream", methods=["GET"])
def get_response_stream():
    user_message = request.args.get("message")

    vector_db = load_vector_db()
    relevant_docs = get_relevant_documents(user_message, vector_db)

    context = "\n".join([node.node.get_content() for node in relevant_docs])
    #  Maddeler halinde yazarken sayılardan sonra nokta yerine parantez kullan.
    prompt = f"""
        Sen bir iş güvenliği asistanısın. Adın Mevzuat Asistanı. Aşağıdaki bilgilere dayanarak, kullanıcı sorgusuna en uygun cevabı maddeler halinde yaz.
        
        Bilgi:
        {context}

        Kullanıcı Sorgusu: {user_message}

        Cevap:
    """

    def generate():
        try:
            stream = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=750,
                temperature=0.7,
                stream=True,
            )

            for chunk in stream:
               if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    text = chunk.choices[0].delta.content
                    
                    parts = []
                    current_part = ""
                    for char in text:
                         current_part += char
                         if char in ['.', ':', '?', '\n']:
                            parts.append(current_part+'\n')
                            current_part="\n"
                    if current_part:
                        parts.append(current_part)



                    for part in parts:
                         yield f"data: {part}\n\n"
            yield f"data: [DONE]\n\n"


        except Exception as e:
            yield f"data: Bir hata oluştu: {str(e)}\n\n"

    return Response(generate(), mimetype='text/event-stream')

if __name__ == "__main__":
    app.run(debug=True)