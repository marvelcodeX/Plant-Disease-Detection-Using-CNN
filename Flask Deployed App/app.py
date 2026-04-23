import json
import os
import sqlite3
from functools import wraps
from uuid import uuid4

from flask import Flask, flash, g, redirect, render_template, request, session, url_for
from PIL import Image
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

import CNN


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "plant_disease_model_1_latest.pt")
DISEASE_INFO_PATH = os.path.join(BASE_DIR, "disease_info.csv")
SUPPLEMENT_INFO_PATH = os.path.join(BASE_DIR, "supplement_info.csv")
DATABASE_PATH = os.path.join(BASE_DIR, "plant_disease.db")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")

HEALTHY_CLASS_IDS = {3, 5, 7, 11, 15, 18, 20, 23, 24, 25, 28, 38}


disease_info = pd.read_csv(DISEASE_INFO_PATH, encoding="cp1252")
supplement_info = pd.read_csv(SUPPLEMENT_INFO_PATH, encoding="cp1252")

model = CNN.CNN(39)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
model.eval()


app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "change-this-secret-key")
app.config["DATABASE"] = DATABASE_PATH
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)


def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(app.config["DATABASE"])
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(error):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    with sqlite3.connect(app.config["DATABASE"]) as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS scan_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                image_path TEXT NOT NULL,
                predicted_index INTEGER NOT NULL,
                predicted_title TEXT NOT NULL,
                predicted_confidence REAL NOT NULL,
                is_healthy INTEGER NOT NULL,
                top_predictions_json TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            );
            """
        )


@app.before_request
def load_logged_in_user():
    user_id = session.get("user_id")
    if user_id is None:
        g.user = None
        return

    user = get_db().execute(
        "SELECT id, username, created_at FROM users WHERE id = ?",
        (user_id,),
    ).fetchone()

    if user is None:
        session.clear()
        g.user = None
    else:
        g.user = user


@app.context_processor
def inject_current_user():
    return {"current_user": g.get("user")}


def login_required(view):
    @wraps(view)
    def wrapped_view(*args, **kwargs):
        if g.user is None:
            flash("Please log in to access that page.", "warning")
            return redirect(url_for("login", next=request.path))
        return view(*args, **kwargs)

    return wrapped_view


def build_prediction_details(index, confidence):
    disease_row = disease_info.iloc[index]
    supplement_row = supplement_info.iloc[index]

    return {
        "index": index,
        "title": disease_row["disease_name"],
        "desc": disease_row["description"],
        "prevent": disease_row["Possible Steps"],
        "image_url": disease_row["image_url"],
        "confidence": round(float(confidence) * 100, 2),
        "is_healthy": index in HEALTHY_CLASS_IDS,
        "sname": supplement_row["supplement name"],
    }


def prediction(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224))
    input_data = TF.to_tensor(image)
    input_data = input_data.view((-1, 3, 224, 224))

    with torch.no_grad():
        output = model(input_data)
        probabilities = torch.softmax(output, dim=1).squeeze(0)
        top_probabilities, top_indices = torch.topk(probabilities, k=min(3, probabilities.shape[0]))

    top_predictions = []
    for rank, (idx, prob) in enumerate(zip(top_indices.tolist(), top_probabilities.tolist()), start=1):
        prediction_details = build_prediction_details(int(idx), prob)
        prediction_details["rank"] = rank
        top_predictions.append(prediction_details)

    return {
        "primary_prediction": top_predictions[0],
        "top_predictions": top_predictions,
    }


def save_scan_history(user_id, image_path, primary_prediction, top_predictions):
    serializable_predictions = []
    for item in top_predictions:
        serializable_predictions.append(
            {
                "rank": item["rank"],
                "title": item["title"],
                "confidence": item["confidence"],
                "is_healthy": item["is_healthy"],
            }
        )

    db = get_db()
    db.execute(
        """
        INSERT INTO scan_history (
            user_id,
            image_path,
            predicted_index,
            predicted_title,
            predicted_confidence,
            is_healthy,
            top_predictions_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            user_id,
            image_path,
            primary_prediction["index"],
            primary_prediction["title"],
            primary_prediction["confidence"],
            int(primary_prediction["is_healthy"]),
            json.dumps(serializable_predictions),
        ),
    )
    db.commit()


@app.route("/")
def home_page():
    return render_template("home.html")


@app.route("/contact")
def contact():
    return render_template("contact-us.html")


@app.route("/index")
@login_required
def ai_engine_page():
    return render_template("index.html")


@app.route("/mobile-device")
def mobile_device_detected_page():
    return redirect(url_for("home_page"))


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if g.user is not None:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        if not username:
            flash("Username is required.", "danger")
        elif len(username) < 3:
            flash("Username must be at least 3 characters long.", "danger")
        elif not password:
            flash("Password is required.", "danger")
        elif len(password) < 6:
            flash("Password must be at least 6 characters long.", "danger")
        elif password != confirm_password:
            flash("Passwords do not match.", "danger")
        else:
            db = get_db()
            existing_user = db.execute(
                "SELECT id FROM users WHERE username = ?",
                (username,),
            ).fetchone()

            if existing_user is not None:
                flash("That username is already taken.", "danger")
            else:
                cursor = db.execute(
                    "INSERT INTO users (username, password_hash) VALUES (?, ?)",
                    (username, generate_password_hash(password)),
                )
                db.commit()
                session.clear()
                session["user_id"] = cursor.lastrowid
                flash("Account created successfully.", "success")
                return redirect(url_for("dashboard"))

    return render_template("signup.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if g.user is not None:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        user = get_db().execute(
            "SELECT * FROM users WHERE username = ?",
            (username,),
        ).fetchone()

        if user is None or not check_password_hash(user["password_hash"], password):
            flash("Invalid username or password.", "danger")
        else:
            session.clear()
            session["user_id"] = user["id"]
            flash("Logged in successfully.", "success")

            next_page = request.values.get("next")
            if next_page and next_page.startswith("/"):
                return redirect(next_page)
            return redirect(url_for("dashboard"))

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("home_page"))


@app.route("/submit", methods=["POST"])
@login_required
def submit():
    image = request.files.get("image")

    if image is None or image.filename == "":
        flash("Please choose an image before submitting.", "warning")
        return redirect(url_for("ai_engine_page"))

    safe_name = secure_filename(image.filename) or "leaf_image.jpg"
    unique_filename = f"{uuid4().hex}_{safe_name}"
    saved_file_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_filename)
    image.save(saved_file_path)

    prediction_result = prediction(saved_file_path)
    primary_prediction = prediction_result["primary_prediction"]
    relative_image_path = f"uploads/{unique_filename}"

    save_scan_history(
        user_id=g.user["id"],
        image_path=relative_image_path,
        primary_prediction=primary_prediction,
        top_predictions=prediction_result["top_predictions"],
    )

    return render_template(
        "submit.html",
        title=primary_prediction["title"],
        desc=primary_prediction["desc"],
        prevent=primary_prediction["prevent"],
        image_url=primary_prediction["image_url"],
        pred=primary_prediction["index"],
        sname=primary_prediction["sname"],
        confidence=primary_prediction["confidence"],
        is_healthy=primary_prediction["is_healthy"],
        top_predictions=prediction_result["top_predictions"],
        uploaded_image=relative_image_path,
    )


@app.route("/dashboard")
@login_required
def dashboard():
    db = get_db()
    history_rows = db.execute(
        """
        SELECT
            id,
            image_path,
            predicted_title,
            predicted_confidence,
            is_healthy,
            top_predictions_json,
            created_at
        FROM scan_history
        WHERE user_id = ?
        ORDER BY datetime(created_at) DESC, id DESC
        """,
        (g.user["id"],),
    ).fetchall()

    history = []
    for row in history_rows:
        item = dict(row)
        item["top_predictions"] = json.loads(item["top_predictions_json"])
        history.append(item)

    stats_row = db.execute(
        """
        SELECT
            COUNT(*) AS total_scans,
            COALESCE(SUM(CASE WHEN is_healthy = 1 THEN 1 ELSE 0 END), 0) AS healthy_count,
            COALESCE(AVG(predicted_confidence), 0) AS average_confidence
        FROM scan_history
        WHERE user_id = ?
        """,
        (g.user["id"],),
    ).fetchone()

    disease_breakdown = db.execute(
        """
        SELECT predicted_title, COUNT(*) AS total
        FROM scan_history
        WHERE user_id = ?
        GROUP BY predicted_title
        ORDER BY total DESC, predicted_title ASC
        LIMIT 5
        """,
        (g.user["id"],),
    ).fetchall()

    total_scans = stats_row["total_scans"]
    healthy_count = stats_row["healthy_count"]

    return render_template(
        "dashboard.html",
        total_scans=total_scans,
        healthy_count=healthy_count,
        diseased_count=total_scans - healthy_count,
        average_confidence=round(stats_row["average_confidence"], 2),
        recent_history=history[:10],
        disease_breakdown=disease_breakdown,
    )


init_db()


if __name__ == "__main__":
    app.run(debug=True)
