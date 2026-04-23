# Flask App Notes

This folder contains the runnable Flask application for the Plant Disease Detection project.

Required local-only files:

- `plant_disease_model_1_latest.pt`
- `plant_disease.db` after the app starts
- uploaded images under `static/uploads/`

Only the source files, templates, CSV metadata, requirements, and `static/uploads/.gitkeep` should be committed. The model checkpoint, database, and uploaded images are ignored by the root `.gitignore`.

Run locally from this folder:

```bash
pip install -r requirements.txt
export SECRET_KEY="replace-with-a-secure-local-secret"
python app.py
```

See the root `README.md` for full setup and GitHub publishing notes.
