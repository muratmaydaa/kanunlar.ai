import os
from flask import Flask, render_template, request, redirect, url_for
from utils.vector_db import update_vector_db

app_admin = Flask(__name__)


@app_admin.route("/admin", methods=["GET", "POST"])
def admin_panel():
    if request.method == "POST":
        if "update_db" in request.form:
            update_vector_db()
            return redirect(url_for("admin_panel", message="Vektör veritabanı güncellendi."))
        
        if "add_doc" in request.files:
             uploaded_file=request.files["add_doc"]
             if uploaded_file.filename!="":
                
                file_path=os.path.join("data/documents",uploaded_file.filename)
                uploaded_file.save(file_path)

    message = request.args.get("message")  # Mesajı al
    return render_template("admin.html", message=message)
    

if __name__ == "__main__":
    app_admin.run(debug=True, port=5001) # Farklı bir portta çalıştır