venv-create:
	python3 -m venv myenv

activate-venv:
	source myenv/bin/activate

update-pip:
	pip install --upgrade pipenv

run-app:
	streamlit run main.py