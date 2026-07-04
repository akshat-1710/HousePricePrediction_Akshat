from flask import Flask, render_template, request
import joblib
import numpy as np

# ==========================
# Load Trained ML Model
# ==========================
model = joblib.load("house_price_model.pkl")

# ==========================
# Create Flask App
# ==========================
app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None

    if request.method == "POST":

        # Numerical Inputs
        area = float(request.form["area"])
        bedrooms = int(request.form["bedrooms"])
        bathrooms = int(request.form["bathrooms"])
        stories = int(request.form["stories"])
        parking = int(request.form["parking"])

        # Yes / No Inputs
        mainroad = 1 if request.form["mainroad"] == "yes" else 0
        guestroom = 1 if request.form["guestroom"] == "yes" else 0
        basement = 1 if request.form["basement"] == "yes" else 0
        hotwaterheating = 1 if request.form["hotwaterheating"] == "yes" else 0
        airconditioning = 1 if request.form["airconditioning"] == "yes" else 0
        prefarea = 1 if request.form["prefarea"] == "yes" else 0

        # Furnishing Status Encoding
        furnishing = request.form["furnishingstatus"]

        semi = 1 if furnishing == "semi-furnished" else 0
        unfurnished = 1 if furnishing == "unfurnished" else 0

        # Create Feature Array
        features = np.array([[
            area,
            bedrooms,
            bathrooms,
            stories,
            parking,
            mainroad,
            guestroom,
            basement,
            hotwaterheating,
            airconditioning,
            prefarea,
            semi,
            unfurnished
        ]])

        # Predict House Price
        prediction = round(float(model.predict(features)[0]), 2)

    return render_template(
        "index.html",
        prediction=prediction
    )


if __name__ == "__main__":
    app.run(debug=True)