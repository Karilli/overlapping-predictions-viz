# Requirements
You need linux OS and python3 (interpreter, pip, venv).

# Create venv
```bash
cd path/to/project/folder
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

# generate data
Run data generation, it should take at most 60s total. (Or run the notebook manually, if you wish.)
```bash
jupyter nbconvert --to python generate_data.ipynb
python3 generate_data.py
rm generate_data.py
```
It should create a new folder `data` with a single file `model.pkl`.

# run the app
```bash
python3 app.py
```
Open the app in browser: http://127.0.0.1:8050.