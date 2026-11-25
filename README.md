# Movie-Recommendation-System


**Content-based movie recommender using TF-IDF and semantic embeddings, with genre-diverse re-ranking**

A recommendation prototype built on the TMDB 5000 Movies dataset. Given a movie title, it returns similar movies by comparing their genres, keywords, cast, director and plot overview. It starts from a classic TF-IDF baseline and then upgrades to semantic embeddings from a pre-trained Sentence-Transformers model, so movies with similar *meaning* match even when they share few exact words.

> **Note:** Academic project built in Google Colab. It uses content-based similarity only and has not been formally evaluated (see [Limitations](#limitations)).

**Tech:** Python · Pandas · NumPy · scikit-learn · Sentence-Transformers (`all-MiniLM-L6-v2`)

---

## Table of Contents

1. [Highlights](#highlights)
2. [How It Works](#how-it-works)
3. [Dataset](#dataset)
4. [Quick Start](#quick-start)
5. [Usage](#usage)
6. [Project Structure](#project-structure)
7. [Limitations](#limitations)
8. [Future Work](#future-work)

---

## Highlights

- **Two approaches compared:** a TF-IDF + cosine similarity baseline, and a semantic version using `all-MiniLM-L6-v2` embeddings
- **Richer movie profile:** genres, keywords, top 3 cast members, director and plot overview are combined into one text field
- **Diversity-aware re-ranking:** the top 50 most similar movies are filtered so the final list introduces new genres instead of 10 near-duplicates
- **Robust parsing:** the TMDB JSON-style columns are cleaned with safe parsing that skips malformed rows

## How It Works

1. **Load and merge** the TMDB movies and credits files on `title`.
2. **Clean the data:** convert the genres, keywords, cast and crew columns from JSON strings into lists of names (top 3 cast, director only) and fill missing overviews.
3. **Build a text profile** for each movie by joining genres, keywords, cast, director and overview.
4. **Baseline:** vectorize the profiles with TF-IDF (English stop words removed) and compute cosine similarity.
5. **Semantic version:** encode the profiles with the `all-MiniLM-L6-v2` Sentence-Transformers model and compute cosine similarity on the embeddings.
6. **Recommend:** take the 50 most similar movies, then keep those that add new genres until 10 are selected.

## Dataset

[TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) (Kaggle). It needs two files:

- `tmdb_5000_movies.csv`
- `tmdb_5000_credits.csv`

The data files are not included in this repository. Download them and place them next to the notebook.

## Quick Start

```bash
git clone https://github.com/LakshmiAiswarya/movie-recommendation-system.git
cd movie-recommendation-system
pip install -r requirements.txt
```

1. Download the two CSV files from Kaggle (link above) into the project folder.
2. Open `movie_recommendation.ipynb` in Jupyter or Google Colab.
3. If running locally, change the file paths from `/content/...` to your local folder.
4. Run all cells. The first run downloads the embedding model (about 90 MB).

## Usage

```python
movie_name = "The Dark Knight"
print(get_semantic_recommendations(movie_name))
```

This returns 10 recommended titles that are similar to the chosen movie and cover a range of genres. The baseline functions `get_recommendations()` and `get_diverse_recommendations()` work the same way using TF-IDF.

## Project Structure

```
.
├── movie_recommendation.ipynb   # Full notebook: baseline + semantic recommender
├── requirements.txt             # Python dependencies
└── README.md
```

## Limitations

- **Content-based only:** recommendations use movie metadata, not user ratings or viewing behaviour, so there is no collaborative filtering.
- **No formal evaluation yet:** accuracy metrics such as Precision, Recall, F1 or RMSE are not computed, and quality was judged by inspecting results.
- **Title lookup is exact:** a movie must be spelled exactly as in the dataset, and duplicate titles keep only the first entry.
- **Simple diversity rule:** diversity is based on genre overlap only, so it can sometimes rank a less similar movie above a closer one.
- **Personalized recommendations** (based on a watch history) are drafted in the code but not enabled.
- The dataset covers about 5,000 movies, so recommendations are limited to that catalogue.

## Future Work

- Add collaborative filtering with a ratings dataset (for example MovieLens) and combine it with the content-based scores in a true hybrid recommender
- Evaluate the system with Precision@K, Recall@K and NDCG on held-out data
- Enable personalized recommendations from a user's watch history
- Add fuzzy title search and a simple web interface (for example Streamlit)

## Author

**Thanubuddu Lakshmi Aiswarya**: B.Tech AI & ML, Vignan's Foundation for Science, Technology & Research
