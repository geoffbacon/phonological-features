SHELL = /bin/bash

# Development-related targets

# list dependencies
requirements:
	pipreqs --force --savepath requirements.txt src

# install dependencies
install:
	python -m pip install -r requirements.txt
	CFLAGS="-stdlib=libc++" python -m pip install fairseq
	conda install gensim

# remove Python file artifacts
clean:
	@find . -name '*.pyc' -exec rm -f {} +
	@find . -name '*.pyo' -exec rm -f {} +
	@find . -name '*~' -exec rm -f {} +
	@find . -name '__pycache__' -exec rm -rf {} +
	@find . -name '.ipynb_checkpoints' -exec rm -rf {} +

# format according to style guide
format:
	black src
	isort -rc src

# check style
lint: format
	pylint --exit-zero --jobs=0 --output-format=colorized src
	pycodestyle --show-source src
	pydocstyle src

# Data-related targets

# download and preprocess the phoneme-level data
wikipron:
	./src/wikipron.sh
	python src/wikipron.py
	python src/features.py

# train all phoneme-level models
train-phonemes:
	nohup python -u src/embeddings.py train-phonemes > logs/nohup.out
	nohup python -u src/rnn.py train-phonemes > logs/nohup.out &

.PHONY: requirements install clean format lint wikipron train-phonemes