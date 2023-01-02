# Idea Matcher


to create a virtual envirenment (you can skip this)
```
python -m venv myenv
myenv\Scripts\activate.bat
```

### installing the requirements
```
pip install -r requirements.txt
```

### running the app 
```
streamlit run app.py
```


### build the docker image
```
docker build -t streamlit-app .
```

### run the docker image
```
docker run -d -p 8501:8501 streamlit-app
```

### how to stop the docker image 
find `CONTAINER_ID` with the first command then stop your container
```
docker container ls
docker stop [CONTAINER_ID]
```



* you will find your app in your browser at http://localhost:8501/

* note that the app will take significant time to load the first time it is launched because it needs to download the api which is 1.6Gb 