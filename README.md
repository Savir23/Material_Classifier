## Install Python requirements (Windows cmd)

If you have Python and pip installed, open a Windows `cmd` prompt and run:

```
cd C:\Users\savir\MaterialClassifier
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Notes:
- Run the `cd` command to change to the project folder if you're not already there.
- If `python` is not on your PATH, replace `python` with the full path to your Python executable (for example `C:\Python310\python.exe`).
- Consider using a virtual environment before installing dependencies:

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

