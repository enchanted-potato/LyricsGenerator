# LYRICS GENERATOR

### Setup

1. **Install dependencies:**
   ```bash
   poetry install
   ```

2. **Set up environment variables:**
   - Copy `.env.example` to `.env`:
     ```bash
     cp .env.example .env
     ```
   - Edit `.env` and add your:
     - `GENIUS_API_KEY`: Get from https://genius.com/api-clients
     - `FLASK_SECRET_KEY`: Generate with `python -c "import secrets; print(secrets.token_hex(32))"`

### Running the Application

Run the following from the project root directory:
```bash
poetry run python web-app/app.py
```

Or activate the Poetry shell first:
```bash
poetry shell
python web-app/app.py
```