# Python Environment Setup Guide

This guide will walk you step-by-step through:

* Installing Python 3.11
* Creating a virtual environment
* Installing dependencies from `environment/requirements.txt`
* Running the ML pipeline via `pipeline/run_inference.py`

---

## 1. **Verify Python 3.11 Installation**

Make sure Python 3.11 is installed on your system.

```bash
python3.11 --version
```

If it shows something like `Python 3.11.x`, you're good.

If not, download and install Python 3.11 from the official site:

* [https://www.python.org/downloads/release/python-3110/](https://www.python.org/downloads/release/python-3110/)

---

## 2. **Create a Virtual Environment (Python 3.11)**

Navigate to your project folder:

```bash
cd /path/to/your/project
```

Create a new environment:

```bash
python3.11 -m venv env
```

Activate the environment:

* **Mac/Linux:**

  ```bash
  source env/bin/activate
  ```
* **Windows:**

  ```bash
  env\Scripts\activate
  ```

You should now see `(env)` before your terminal prompt.

---

## 3. **Install Platform Requirements**

Run the following command from the root of your project:

```bash
# Windows:
pip install -r environment/requirements-windows.txt
# macOS:
pip install -r environment/requirements-macos.txt
```

This will install all required packages for your ML pipeline.

---

## 4. **Run FastAPI and Streamlit (PowerShell)**

From the project root, configure the shared persistent database and keep phone
notifications disabled by default:

```powershell
$env:SMARTVISION_DATABASE_PATH = "C:\weapon detection\wepondetection\data\incidents.db"
$env:SMARTVISION_ALERTS_ENABLED = "false"
$env:SMARTVISION_NTFY_TOPIC = ""
```

Start the backend and dashboard in separate PowerShell windows:

```powershell
.\.venv\Scripts\python.exe -m uvicorn server.main:app --host 127.0.0.1 --port 8000
```

```powershell
.\.venv\Scripts\python.exe -m streamlit run pipeline\app.py --server.port 8501
```

The backend stores history in the project-root `data\incidents.db` by default;
the dashboard reads it through the FastAPI incidents API.

To opt in to ntfy after validating and approving the detector and verifying
the endpoint, set these in both service environments before starting them:

```powershell
$env:SMARTVISION_ALERTS_ENABLED = "true"
$env:SMARTVISION_ALERT_ENDPOINT_VERIFIED = "true"
$env:SMARTVISION_NTFY_TOPIC = "<your-private-topic>"
```

Then use **Send safe test notification** under Model and system health to test
delivery without inserting a fake incident. Never commit the topic or place it
in source code.

## 5. **Run the ML Pipeline**

Navigate to the pipeline directory:

```bash
cd pipeline
```

Run the pipeline script:

```bash
python run_inference.py
```

This will start your machine learning inference pipeline.

---

## 6. **Deactivate the Environment (Optional)**

When you're done:

```bash
deactivate
```
