install:
	pip install --upgrade pip && pip install -r requirements.txt

train:
	python model.py

evaluate:
	python evaluate.py

all: install train evaluate