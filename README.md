# Romanian Political Speech Analysis

A Streamlit dashboard and NLP pipeline for analyzing Romanian presidential speech and parliamentary debate data using TF-IDF.

## Project Contents

- `dashboard.py` — Streamlit dashboard for interactive visualization and analysis.
- `tfidf_matrix_presidents.csv` — cached TF-IDF matrix for presidential speeches.
- `tfidf_matrix_parliamentary.csv` — cached TF-IDF matrix for parliamentary speeches.
The following files are NOT included in this repo due to size. Download them and place them in the same folder as `dashboard.py`:
- `ParlEE_RO_plenary_speeches.csv` — download from [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/VOPK0E)
- `iliescu.txt`, `constantinescu.txt`, `basescu.txt`, `iohannis.txt` — download from [GitHub](https://github.com/grrrrah/RomanianPresidentialDiscourses)
  
## Dataset Citations

- ParLEE plenary speeches V2 dataset: Sylvester, C. (Creator), Greene, Z. (Creator), Ershova, A. (Contributor), Khokhlova, A. (Contributor), Yordanova, N. (Creator) (21 Feb 2023). ParlEE plenary speeches V2 data set: Annotated full-text of 15.1 million sentence-level plenary speeches of six EU legislative chambers. Harvard Dataverse. https://doi.org/10.7910/DVN/VOPK0E
- Presidential speeches repository: https://github.com/grrrrah/RomanianPresidentialDiscourses

## Prerequisites

- Python 3.8+ installed
- `pip` available via `python -m pip`

## Dependencies

Install required packages with:

```powershell
python -m pip install streamlit pandas matplotlib seaborn scikit-learn wordcloud spacy nltk
python -m spacy download ro_core_news_sm
```

> If `streamlit` is not available on your PATH, use `python -m streamlit`.

## Running the Dashboard

From the project directory:

```powershell
python -m streamlit run dashboard.py
```

If you prefer a virtual environment:

```powershell
python -m venv .venv
# In PowerShell
.\.venv\Scripts\Activate.ps1
# or in CMD
.\.venv\Scripts\activate.bat
python -m pip install streamlit pandas matplotlib seaborn scikit-learn wordcloud spacy nltk
python -m spacy download ro_core_news_sm
python -m streamlit run dashboard.py
```

## Notes

- Run the dashboard from the project directory (where `dashboard.py` lives).
- The app expects data files to be in the **same directory**.
- The dashboard uses cached TF-IDF CSV files if present; if they are missing, it will show warnings and require the caches to be generated before visualizing TF-IDF data.
- You can avoid recomputing TF-IDF by downloading the precomputed CSV cache files (`tfidf_matrix_presidents.csv` and `tfidf_matrix_parliamentary.csv`) and placing them in the project directory. However, the full parliamentary dataset file `ParlEE_RO_plenary_speeches.csv` is still required for other analyses and cannot be replaced by the cached CSVs.

## What the Dashboard Shows

- Overview metrics for parliamentary speech data
- Gender-based topic analysis
- Presidential TF-IDF analysis
- Parliamentary TF-IDF analysis by presidential term
- President vs Parliament vocabulary comparison

## What is TF-IDF?

TF-IDF stands for Term Frequency-Inverse Document Frequency. It is a text analysis technique that highlights words that are important or distinctive in one document compared to a collection of documents. 
In this project, TF-IDF is used to find the most characteristic words for each president and for parliamentary speech groups.

## Troubleshooting

- `The term 'streamlit' is not recognized` — use `python -m streamlit run dashboard.py`
- `Execution policies` when activating venv in PowerShell — either use the `.bat` activate script or run PowerShell as administrator and adjust policy to `RemoteSigned` temporarily.

