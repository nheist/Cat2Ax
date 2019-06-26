# Cat2Ax

Cat2Ax is an approach for the extraction of axioms and assertions from Wikipedia categories.
In this package you can also find implementations of the two most closely related approaches Catriple and C-DF.

## Configuration
### Prerequisites
- Python 3
- pipenv (https://pipenv.readthedocs.io/en/latest/)

### System Requirements
- You need a machine with at least XX GB of RAM as we load most of DBpedia in memory to speed up the extraction
  - If that is not possible for you and you nevertheless want to run the extraction, you can change the functionalities in `impl.category.store` and `impl.dbpedia.store` to use a database instead 
- During the first execution of an extraction you need a stable internet connection as the required DBpedia files are downloaded automatically 

### Setup

- Navigate to the source directory of the project

- Create and initialize a virtual environment with pipenv (run in terminal):
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

You can configure the application-specific parameters as well as logging- and file-related parameters in `config.yaml`. 

## Usage

- Run one of the extraction methods with pipenv:
```
pipenv run {cat2ax.py | catriple.py | cdf.py}
```
(All the required resources - e.g. DBpedia files - will be downloaded automatically and on-demand)

- If you want to extract the type lexicalisations (not necessary as we provide them as cache file), run the following:
```
pipenv run type_lexicalisations.py
```
(Note that the extraction of type lexicalisations might take several days!)

## Results

If you don't want to run the extraction yourself, you can find the results [here](http://data.dws.informatik.uni-mannheim.de/Cat2Ax/).

## Documentation

TODO: ARCHITECTURE

## License
MIT.
https://opensource.org/licenses/MIT