venv-create:
	python3 -m venv myenv

activate-venv:
	source myenv/bin/activate

update-pip:
	pip install --upgrade pipenv

run-app:
	streamlit run main.py

start-qdrant:
	docker run -d \
  --name qdrant \
  -p 6333:6333 \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  qdrant/qdrant