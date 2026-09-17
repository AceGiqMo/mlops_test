# How to run
```bash
git clone https://github.com/AceGiqMo/mlops_test.git
docker compose up -d --build

docker compose exec airflow airflow users create \
  --username acegiqmo \
  --firstname AceGiqMo \
  --lastname Ace \
  --role Admin \
  --email ace.giqmo@example.com \
  --password admin
  
docker compose exec airflow airflow dags trigger 
```

Now you can open **Apache Airflow UI** by the address `http://localhost:8080` and wait until the DAG
will finish the execution

After that you open the web app made with **Streamlit**, accessed by the address `http://localhost:8501`.
Here you can enter your **input** and get **prediction**