
from flask import Flask, render_template, request
import pickle
import math
from pathlib import Path

app = Flask(__name__)

# Find the folder where app.py is located
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "SLR_MODEL.pkl"


# Load the trained machine learning model
try:
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)

    print("Salary prediction model loaded successfully!")

except (OSError, pickle.UnpicklingError, EOFError) as error:
    model = None
    print("Error loading model:", error)


# Home page
@app.route("/")
def home():
    return render_template(
        "index.html",
        prediction_text=None,
        experience=None,
        error_message=None
    )


# Salary prediction route
@app.route("/predict", methods=["POST"])
def predict():
    prediction_text = None
    error_message = None
    experience = request.form.get("experience", "").strip()

    try:
        # Check whether the model was loaded
        if model is None:
            raise RuntimeError(
                "The model could not be loaded. Check SLR_MODEL.pkl."
            )

        # Check empty input
        if not experience:
            raise ValueError("Please enter your years of experience.")

        # Convert input into a number
        years = float(experience)

        # Validate the input
        if not math.isfinite(years):
            raise ValueError("Please enter a valid number.")

        if years < 0 or years > 100:
            raise ValueError(
                "Experience must be between 0 and 100 years."
            )

        # Machine learning models expect a 2D input
        input_data = [[years]]

        # Predict the salary
        result = model.predict(input_data)[0]

        if not math.isfinite(float(result)):
            raise ValueError("The model returned an invalid result.")

        # Display the result with two decimal places
        prediction_text = f"{float(result):,.2f}"

    except ValueError as error:
        error_message = str(error)

    except RuntimeError as error:
        error_message = str(error)

    except Exception:
        app.logger.exception("Salary prediction failed.")
        error_message = (
            "Unable to predict salary. Please check the model and try again."
        )

    return render_template(
        "index.html",
        prediction_text=prediction_text,
        experience=experience,
        error_message=error_message
    )


if __name__ == "__main__":
    app.run(debug=True)
