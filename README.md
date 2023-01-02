# Idea Matcher


to create a virtual envirenment (you can skip this)
```
python -m venv myenv
myenv\Scripts\activate.bat
```

install requirements
```
pip install -r requirements.txt
```

how to run 
```
streamlit run app.py
```


build thedocker image
```
docker build -t streamlit-app .
```

run the docker image
```
docker run -p 8501:8501 streamlit-app
```

> you will find your app in your browser at http://localhost:8501/