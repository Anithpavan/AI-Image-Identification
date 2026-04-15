from flask import Flask, render_template, request, redirect, url_for
from predict import predict_animal   # still named predict_animal for compatibility
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads"

# ==========================
# HOME ROUTE
# ==========================
@app.route("/")
def home():
    return render_template("index.html")

# ==========================
# PREDICTION ROUTE
# ==========================
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return redirect(request.url)
    
    file = request.files["file"]
    if file.filename == "":
        return redirect(request.url)
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
        file.save(filepath)

        # Get prediction and details
        bird_species, details = predict_animal(filepath)
        
        return render_template(
            "result.html",
            image_path=filepath,
            bird_species=bird_species,
            details=details
        )

# ==========================
# RUN FLASK APP
# ==========================
if __name__ == "__main__":
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    print("🚀 Starting Bird Species Detector Flask App...")
    app.run(debug=True)
