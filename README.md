# Cat2Ax

Cat2Ax is an approach for the extraction of axioms and assertions from Wikipedia categories.
In this package you can also find implementations of the two most closely related approaches Catriple and C-DF.

## Configuration
### Prerequisites
- Python 3
- pipenv (https://pipenv.readthedocs.io/en/latest/)

### Setup

- Create and initialise a virtual environment with pipenv (run in terminal):
```
pipenv install
```

- Download the spacy corpus (run in terminal):
```
pipenv run python -m spacy download en_core_web_lg
```

- Download the wordnet corpus of nltk (run in python):
```
import nltk
nltk.download('wordnet')
```

### Basic Configuration Options

Use `config.yaml` for configuration of the application.

## Usage

- Run the application with pipenv:
```
pipenv run {cat2ax.py | catriple.py | cdf.py}
```

## License
MIT.
https://opensource.org/licenses/MIT