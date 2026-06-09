"""
Romanian Political Speech Analysis — Streamlit Dashboard
Run with: python -m streamlit run "D:\vscode projects\data stuff\tfidf\dashboard.py"
"""

import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import spacy
from nltk.corpus import stopwords

# ==================== 0. Page config — must be FIRST streamlit call ====================
st.set_page_config(
    page_title = "🇷🇴 Romanian Political Speeches Analysis",
    page_icon  = "🇷🇴",
    layout     = "wide",                   # use the full browser width
    initial_sidebar_state = "expanded"     # sidebar open by default
)

# ==================== 1. Set working folder ====================
os.chdir(os.path.dirname(os.path.abspath(__file__)))  # automatically changes to where the file is on your computer

# ==================== 2. File name constants ====================
# Centralise all file names so we only need to change them in one place
PRESIDENT_FILES = {
    "Iliescu":        "iliescu.txt",
    "Constantinescu": "constantinescu.txt",
    "Basescu":        "basescu.txt",
    "Iohannis":       "iohannis.txt",
}
PARLIAMENT_CSV        = "ParlEE_RO_plenary_speeches.csv"
PRESIDENT_CACHE_FILE  = "tfidf_matrix_presidents.csv"
PARLIAMENT_CACHE_FILE = "tfidf_matrix_parliamentary.csv"

# ==================== 3. Policy area codes ====================
# Maps the numeric policyarea column to a human-readable label
POLICY_AREA_CODES = {
    1: "Macroeconomics",           2: "Civil Rights & Liberties",
    3: "Health",                    4: "Agriculture",
    5: "Labour & Employment",       6: "Education",
    7: "Environment",               8: "Energy",
    10: "Transportation",           12: "Law & Crime",
    13: "Social Welfare",           14: "Housing",
    15: "Commerce & Banking",       16: "Defence",
    17: "Technology",               18: "Foreign Trade",
    19: "Foreign Affairs",          20: "Government Operations",
    21: "Public Lands",             23: "Culture & Entertainment",
    24: "Local Government",         27: "Immigration",
    98: "Non-domestic",             99: "Other / Uncodeable"
}

# One colour per president — used consistently across every chart
PRESIDENT_COLORS = {
    "Iliescu":        "red",
    "Constantinescu": "gold",
    "Basescu":        "darkgreen",
    "Iohannis":       "royalblue"
}


# ====================================================================================
# DATA LOADING FUNCTIONS
# @st.cache_data    — caches the return value so the function only runs once per session
# @st.cache_resource — caches a non-serialisable object (like a spaCy model) once per session
# ====================================================================================

@st.cache_resource(show_spinner="Loading spaCy Romanian model...")
def get_spacy_nlp(model_name="ro_core_news_sm"):
    """Load the Romanian spaCy model. Raises a clean error if it is not installed."""
    try:
        nlp = spacy.load(model_name, disable=["parser", "ner"])   # disable unused components for speed
    except OSError as exc:
        st.error(
            f"spaCy model '{model_name}' not found.\n\n"
            f"Run this in your terminal:\n\n"
            f"    python -m spacy download {model_name}"
        )
        raise SystemExit(1) from exc
    nlp.max_length = 1000000    # allow texts up to 1M characters per chunk
    return nlp


@st.cache_data(show_spinner="Building stopword list...")
def load_stopwords_cached():
    """Combine NLTK, spaCy, and custom Romanian stopwords into one list."""
    nlp = get_spacy_nlp()                       # reuse the cached model
    custom_remove = [
        "doamnelor domn", "domn deputat", "multumesc domn",
        "domn presedint", "multumi domn"
    ]
    ro_stopwords = list(set(
        stopwords.words("romanian") +           # NLTK Romanian list
        list(nlp.Defaults.stop_words) +         # spaCy Romanian list (more comprehensive)
        custom_remove                           # hand-crafted list for parliamentary phrases
    ))
    return ro_stopwords, custom_remove


@st.cache_data(show_spinner="Loading parliamentary speeches CSV...")
def load_parliamentary_speeches():
    """
    Read the ParlEE CSV and add three derived columns:
    - gender   : inferred from the speaker's first name
    - president: the Romanian president in office on that date
    - policy_name: human-readable label for the policyarea code
    """
    speeches = pd.read_csv(PARLIAMENT_CSV)

    # convert the date string to a proper datetime so we can compare dates
    speeches["date"] = pd.to_datetime(speeches["date"], format="%d/%m/%Y")

    # add gender column using the name heuristic
    speeches["gender"] = speeches["speaker"].apply(name_to_gender)

    # add president column based on when each speech was made
    speeches["president"] = speeches["date"].apply(find_president)

    # add readable policy area name
    speeches["policy_name"] = speeches["policyarea"].map(POLICY_AREA_CODES)

    return speeches


@st.cache_data(show_spinner="Loading presidential speech files...")
def load_presidential_speeches():
    """Read each president's txt file into a dictionary {name: full_text}."""
    president_texts = {}
    for president_name, file_name in PRESIDENT_FILES.items():
        with open(file_name, encoding="utf-8") as f:
            president_texts[president_name] = f.read()
    return president_texts


@st.cache_data(show_spinner="Loading presidential TF-IDF matrix...")
def load_president_tfidf():
    """
    Load the presidential TF-IDF matrix from cache if it exists.
    Returns None if the cache file has not been created yet —
    the user will need to run the main script first.
    """
    if os.path.exists(PRESIDENT_CACHE_FILE):
        return pd.read_csv(PRESIDENT_CACHE_FILE, index_col=0)   # index_col=0 restores president names as row labels
    return None


@st.cache_data(show_spinner="Loading parliamentary TF-IDF matrix...")
def load_parliament_tfidf():
    """
    Load the parliamentary TF-IDF matrix from cache if it exists.
    Returns None if not yet computed.
    """
    if os.path.exists(PARLIAMENT_CACHE_FILE):
        return pd.read_csv(PARLIAMENT_CACHE_FILE, index_col=0)
    return None


# ====================================================================================
# HELPER / UTILITY FUNCTIONS
# ====================================================================================

def name_to_gender(name):
    """
    Guess gender from the first name using a Romanian heuristic:
    female names almost always end in 'a', with a small exclusion list
    of male names that also end in 'a'.
    """
    male_names_end_in_a = ["mihaita", "mircea", "mircia", "stelica",
                            "horia", "luca", "toma", "costica"]
    if pd.isna(name):
        return None
    first_name = str(name).strip().split()[0].lower()    # take only the first name, lowercase
    return "F" if first_name.endswith("a") and first_name not in male_names_end_in_a else "M"


def find_president(date):
    """Return the Romanian president in office on a given date (for 2009-2018 range)."""
    if date < pd.Timestamp("2014-12-21"):
        return "Basescu"      # Traian Băsescu served until 21 Dec 2014
    else:
        return "Iohannis"     # Klaus Iohannis took office on 21 Dec 2014


def get_top_words(tfidf_df, n=20):
    """Return the top n highest TF-IDF words for each row (president/group) in the matrix."""
    return {group: tfidf_df.loc[group].nlargest(n) for group in tfidf_df.index}


# ====================================================================================
# PLOTTING FUNCTIONS — each returns a matplotlib figure so Streamlit can display it
# ====================================================================================

def plot_barchart(topwords_per_group, plot_title):
    """Draw a 2-column grid of horizontal bar charts, one panel per president/group."""
    group_names = list(topwords_per_group.keys())
    colors      = ["red", "gold", "darkgreen", "royalblue", "purple", "teal"]
    rows        = (len(group_names) + 1) // 2   # ceiling division — 4 groups = 2 rows

    fig, axes = plt.subplots(rows, 2, figsize=(12, 5 * rows))
    axes = axes.flatten()    # turn 2D array into a flat list so we can index with a single number

    for i, group_name in enumerate(group_names):
        words  = topwords_per_group[group_name].index.tolist()   # the word labels on y axis
        scores = topwords_per_group[group_name].values           # the TF-IDF scores on x axis
        axes[i].barh(words, scores, color=colors[i % len(colors)])
        axes[i].set_title(group_name, fontsize=13, fontweight="bold")
        axes[i].set_xlabel("TF-IDF score")
        axes[i].invert_yaxis()    # highest score appears at the top, not the bottom

    for j in range(len(group_names), len(axes)):
        axes[j].axis("off")       # hide any empty subplot panels

    plt.suptitle(plot_title, fontsize=15)
    plt.tight_layout()
    return fig


def plot_wordclouds(tfidf_df, ro_stopwords, plot_title):
    """Draw a 2-column grid of word clouds, one per president/group."""
    group_names = list(tfidf_df.index)
    rows        = (len(group_names) + 1) // 2

    fig, axes = plt.subplots(rows, 2, figsize=(15, 5 * rows))
    axes = axes.flatten()

    for i, group_name in enumerate(group_names):
        word_scores = tfidf_df.loc[group_name].nlargest(100).to_dict()
        # .nlargest(100)  — keep only the 100 most distinctive words
        # .to_dict()      — convert to {word: score} format that WordCloud expects

        wc = WordCloud(
            width=750, height=350,
            background_color="white",
            colormap="Dark2",
            stopwords=set(ro_stopwords),     # also filter stopwords inside the cloud
        ).generate_from_frequencies(word_scores)
        # generate_from_frequencies: bigger word = higher TF-IDF score

        axes[i].imshow(wc, interpolation="bilinear")   # render the cloud as an image
        axes[i].axis("off")                            # hide axes — we just want the image
        axes[i].set_title(group_name, fontsize=15, fontweight="bold")

    for j in range(len(group_names), len(axes)):
        axes[j].axis("off")

    plt.suptitle(plot_title, fontsize=15)
    plt.tight_layout()
    return fig


def plot_topics_by_gender(speeches):
    """
    Two side-by-side bar charts showing what proportion of each gender's
    speeches falls into each policy area.
    """
    sns.set_theme(style="whitegrid")

    # crosstab counts speeches per (policyarea, gender), then normalize per column
    # so the result is a percentage of each gender's total speeches
    topic_gender = pd.crosstab(
        speeches["policyarea"], speeches["gender"], normalize="columns"
    ) * 100

    # map numeric codes to readable names for the y axis labels
    topic_gender.index = topic_gender.index.map(POLICY_AREA_CODES)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # men — left panel
    topic_gender["M"].sort_values().plot(kind="barh", ax=ax1, color="#1f77b4")
    ax1.set_title("Topics Discussed by Men", fontsize=13, fontweight="bold")
    ax1.set_xlabel("% of speeches within gender")

    # women — right panel
    topic_gender["F"].sort_values().plot(kind="barh", ax=ax2, color="#e377c2")
    ax2.set_title("Topics Discussed by Women", fontsize=13, fontweight="bold")
    ax2.set_xlabel("% of speeches within gender")

    fig.suptitle("Policy Topics: Men vs Women", fontsize=15, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_gender_by_party(speeches):
    """Bar chart showing the count of male vs female speeches per party."""
    # only keep the top 8 parties by number of speeches to avoid clutter
    top_parties = speeches["party"].value_counts().head(8).index
    filtered_parties= speeches[speeches["party"].isin(top_parties)]

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.countplot(
        data    = filtered_parties,
        x       = "party",
        hue     = "gender",
        palette = {"M": "#1f77b4", "F": "#e377c2"},   #keeping the same color for male vs female for all parties
        ax      = ax # for axes
    )
    ax.set_title("Gender Distribution Across Parties", fontsize=15, fontweight="bold")
    ax.set_xlabel("Party")
    ax.set_ylabel("Number of speeches")
    ax.legend(title="Gender", labels=["Male", "Female"])
    plt.tight_layout()
    return fig


def plot_president_vs_parliament(tfidf_president, tfidf_parliament, presidents_to_compare):
    """
    For each selected president: left panel shows their own rhetoric,
    right panel shows what parliament focused on during their term.
    """
    fig, axes = plt.subplots(
        nrows = len(presidents_to_compare),
        ncols = 2,
        figsize = (16, 6 * len(presidents_to_compare))
    )
    # if only one president is selected axes is 1D — wrap it so we can always index axes[i][j]
    if len(presidents_to_compare) == 1:
        axes = [axes]

    fig.suptitle("Discourse Comparison: President vs Parliament",
                 fontsize=18, fontweight="bold")

    for i, president_name in enumerate(presidents_to_compare):
        try:
            # get top 12 terms from each matrix for this president
            top_pres = tfidf_president.loc[president_name].nlargest(12).sort_values(ascending=True)
            top_parl = tfidf_parliament.loc[president_name].nlargest(12).sort_values(ascending=True)
        except KeyError:
            st.warning(f"'{president_name}' not found in one of the TF-IDF matrices.")
            continue

        # left: president's distinctive terms
        top_pres.plot(kind="barh", color=PRESIDENT_COLORS.get(president_name, "grey"),
                      ax=axes[i][0], edgecolor="black")
        axes[i][0].set_title(f"President {president_name} — speeches", fontsize=13)
        axes[i][0].set_xlabel("TF-IDF score")

        # right: parliament's distinctive terms during that president's term
        top_parl.plot(kind="barh", color="#1f77b4",
                      ax=axes[i][1], edgecolor="black")
        axes[i][1].set_title(f"Parliament — during {president_name}'s term", fontsize=13)
        axes[i][1].set_xlabel("TF-IDF score")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig

# ====================================================================================
# TF-IDF FUNCTIONS - in case the CSV cache files don't exist 
# ====================================================================================
def lemmatize_ro(text, nlp, custom_remove, chunk_size=500000):
    lemmatized_chunks = []
    total_chunks = (len(text) // chunk_size) + 1

    print(f"Text length: {len(text)} characters")
    print(f"Splitting into about {total_chunks} chunks...")

    for chunk_number, start_position in enumerate(range(0, len(text), chunk_size), start=1):
        end_position = start_position + chunk_size
        text_chunk = text[start_position:end_position]

        print(f"Processing chunk {chunk_number}/{total_chunks}...")

        doc = nlp(text_chunk)

        chunk_lemmas = [
            token.lemma_.lower().strip()
            for token in doc
            if not token.is_stop
            and not token.is_punct
            and not token.is_space
            and token.lemma_.strip()
            and token.lemma_.lower() not in custom_remove
        ]

        lemmatized_chunks.append(" ".join(chunk_lemmas))

    return " ".join(lemmatized_chunks)

def lemmatize_all(text_dictionary, nlp, custom_remove):
    print("Lemmatizing speeches — this may take a few minutes...")
    result = {}

    for group_name, text in text_dictionary.items():
        print(f"  Processing {group_name}...")
        result[group_name] = lemmatize_ro(text, nlp, custom_remove)

    print("Lemmatization done.")
    return result

def compute_tfidf(lemmatized_texts, ro_stopwords):
    vectorizer = TfidfVectorizer(
        stop_words=ro_stopwords,   # remove stopwords
        max_features=1000,         # keep the top 1000 features
        ngram_range=(2, 2),        # use two-word phrases only
        min_df=1                   # a term can appear in only one document and still be kept
    )

    tf_idf_matrix = vectorizer.fit_transform(lemmatized_texts.values())
    group_names = list(lemmatized_texts.keys())
    vocabulary = vectorizer.get_feature_names_out()

    tfidf_df = pd.DataFrame(
        tf_idf_matrix.toarray(),
        index=group_names,
        columns=vocabulary
    )

    return tfidf_df

def build_parliament_texts_by_president(speeches):
    print("Building grouped parliamentary texts by president...")
    parliament_texts = {}

    for president_name in speeches["president"].dropna().unique():
        print(f"  Grouping speeches for {president_name}...")
        parliament_texts[president_name] = " ".join(
            speeches.loc[speeches["president"] == president_name, "text"]
            .dropna()
            .astype(str)
            .tolist()
        )

    print("Grouped parliamentary texts built.")
    return parliament_texts

# ====================================================================================
# STREAMLIT DASHBOARD FUNCTION
# ====================================================================================

def generate_dashboard():

    st.sidebar.title("🇷🇴 Romanian Political Speech")
    st.sidebar.markdown("---")
    # ====================================================================================
    # STREAMLIT SIDEBAR — navigation and filters
    # ====================================================================================
    # main navigation — user picks which analysis to view
    page = st.sidebar.radio("📂 Select analysis", [
        "📊 Overview",
        "👥 Gender Analysis",
        "🎤 Presidential TF-IDF",
        "🏛️ Parliamentary TF-IDF",
        "🔀 President vs Parliament"
    ])
    #These are the things that will always show in the sidebar, regardless of which page is selected
    st.sidebar.markdown("---")
    st.sidebar.caption("Data: ParlEE v2 & presidential speech txt files")
    st.sidebar.caption("[ParLEE v2 Dataset Link  - harvard.edu](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/VOPK0E)")
    st.sidebar.caption("Sylvester, C. (Creator), Greene, Z. (Creator), Ershova, A. (Contributor), Khokhlova, A. (Contributor), Yordanova, N. (Creator) (21 Feb 2023)." \
    " ParlEE plenary speeches V2 data set: Annotated full-text of 15.1 million sentence-level plenary speeches of six EU legislative chambers. Harvard Dataverse. 10.7910/DVN/VOPK0E")
    st.sidebar.caption("[Presidential speeches GitHub repo](https://github.com/grrrrah/RomanianPresidentialDiscourses)")

    # ====================================================================================
    # LOAD SHARED DATA — loaded once, reused across all pages via Streamlit cache
    # ====================================================================================

    speeches        = load_parliamentary_speeches()   # 1.1M row DataFrame
    ro_stopwords, custom_remove = load_stopwords_cached()

    # ====================================================================================
    # COMPUTE CACHE IF MISSING
    # ====================================================================================

    if not os.path.exists(PRESIDENT_CACHE_FILE) or not os.path.exists(PARLIAMENT_CACHE_FILE):
        st.sidebar.warning("⚠️ TF-IDF matrices not found")
        if st.sidebar.button("⚙️ Compute TF-IDF matrices"):
            
            # load spaCy
            with st.spinner("Loading spaCy model..."):
                nlp = get_spacy_nlp()
            ro_stopwords_compute, custom_remove_compute = load_stopwords_cached()

            # presidential TF-IDF
            if not os.path.exists(PRESIDENT_CACHE_FILE):
                with st.spinner("Computing presidential TF-IDF — this will take a few minutes..."):
                    president_texts   = load_presidential_speeches()
                    pres_lemmatized   = lemmatize_all(president_texts, nlp, custom_remove_compute)
                    tfidf_pres        = compute_tfidf(pres_lemmatized, ro_stopwords_compute)
                    tfidf_pres.to_csv(PRESIDENT_CACHE_FILE)
                st.success("Presidential TF-IDF saved.")

            # parliamentary TF-IDF
            if not os.path.exists(PARLIAMENT_CACHE_FILE):
                with st.spinner("Computing parliamentary TF-IDF — this will take a while..."):
                    parl_texts      = build_parliament_texts_by_president(speeches)
                    parl_lemmatized = lemmatize_all(parl_texts, nlp, custom_remove_compute)
                    tfidf_parl      = compute_tfidf(parl_lemmatized, ro_stopwords_compute)
                    tfidf_parl.to_csv(PARLIAMENT_CACHE_FILE)
                st.success("Parliamentary TF-IDF saved.")

            st.cache_data.clear()   # clear cache so the new files get picked up
            st.rerun()              # rerun so the pages load the newly computed matrices
    # ====================================================================================
    # PAGE 1 — OVERVIEW
    # ====================================================================================

    if page == "📊 Overview":
        st.title("Romanian Political Speech Analysis")
        st.markdown(
            "Exploring TF-IDF, gender patterns, and discourse comparison across "
            "**presidential speeches** and **1.1M parliamentary sentences** (2009–2019)."
        )
        st.markdown("---")

        # summary metrics across the top of the page
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total sentences",   f"{len(speeches):,}")
        c2.metric("Unique speakers",   speeches["speaker"].nunique())
        c3.metric("Parties",           speeches["party"].nunique())
        c4.metric("From",              str(speeches["date"].min().date()))
        c5.metric("To",                str(speeches["date"].max().date()))

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Speeches per party")
            party_counts = speeches["party"].value_counts().head(10)
            fig, ax = plt.subplots(figsize=(8, 4))
            party_counts.plot(kind="bar", ax=ax, color="steelblue")
            ax.set_xlabel("Party")
            ax.set_ylabel("Number of sentences")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            st.pyplot(fig)    # st.pyplot() renders a matplotlib figure inline

        with col2:
            st.subheader("Speeches per president - 2 terms each")
            pres_counts = speeches["president"].value_counts()
            fig, ax = plt.subplots(figsize=(8, 4))
            pres_counts.plot(kind="bar", ax=ax,
                            color=[PRESIDENT_COLORS.get(p, "grey") for p in pres_counts.index])
            ax.set_xlabel("President")
            ax.set_ylabel("Number of sentences")
            plt.xticks(rotation=0)
            plt.tight_layout()
            st.pyplot(fig)

        st.subheader("Gender split overall")
        gender_counts = speeches["gender"].value_counts()
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.pie(gender_counts, labels=["Male", "Female"],
            colors=["#1f77b4", "#e377c2"], autopct="%1.1f%%", startangle=90)
        ax.set_title("Proportion of speeches by gender")
        st.pyplot(fig)


    # ====================================================================================
    # PAGE 2 — GENDER ANALYSIS
    # ====================================================================================

    elif page == "👥 Gender Analysis":
        st.title("Gender Analysis")
        st.markdown("Do men and women MPs speak about different topics? Are some parties more gender-balanced?")
        st.markdown("‼️ The dataset does not feature the gnder of the speaker. This was extarcted using the Romanian heuristic that most female names end in 'a', with a small list of exceptions what were handeled. ‼️")
        st.markdown("---")

        # --- sidebar filter: let the user choose which parties to include ---
        all_parties   = speeches["party"].dropna().unique().tolist()
        chosen_parties = st.sidebar.multiselect(
            "Filter by party",
            options  = all_parties,
            default  = all_parties      # all selected by default
        )
        filtered_speeches = speeches[speeches["party"].isin(chosen_parties)]

        # --- plot 1: topics by gender ---
        st.subheader("Policy areas discussed by each gender")
        st.caption("Percentage of each gender's total speeches falling into each policy area")
        fig = plot_topics_by_gender(filtered_speeches)
        st.pyplot(fig)

        st.markdown("---")

        # --- plot 2: gender per party ---
        st.subheader("Gender distribution across parties")
        st.caption("Number of speeches by male vs female speakers, per party")
        fig = plot_gender_by_party(filtered_speeches)
        st.pyplot(fig)

        st.markdown("---")

        # --- EU vs domestic focus ---
        st.subheader("EU vs domestic focus by gender")
        eu_gender = (
            filtered_speeches[filtered_speeches["gender"].notna()]
            .groupby("gender")["eu"].mean()
            .reset_index()
        )
        eu_gender["gender"] = eu_gender["gender"].map({"F": "Women", "M": "Men"})   # readable labels
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(eu_gender["gender"], eu_gender["eu"],
            color=["#e377c2", "#1f77b4"])
        ax.set_ylabel("Proportion of EU-focused speeches")
        ax.set_title("EU focus by gender")
        plt.tight_layout()
        st.pyplot(fig)


    # ====================================================================================
    # PAGE 3 — PRESIDENTIAL TF-IDF
    # ====================================================================================

    elif page == "🎤 Presidential TF-IDF":
        st.title("Presidential Speech TF-IDF")
        st.markdown(
            "Most distinctive words per president in their **own speeches**. "
            "Higher TF-IDF = more unique to that president compared to the others."
        )
        st.markdown("---")

        tfidf_president = load_president_tfidf()   # loads from cache CSV

        if tfidf_president is None:
            # guide the user if the cache doesn't exist yet
            st.warning(
                "Presidential TF-IDF matrix not found.\n\n"
                "Run the main `presidential_speeches.py` script first to generate "
                f"`{PRESIDENT_CACHE_FILE}`."
            )
        else:
            # sidebar controls for this page
            n_words = st.sidebar.slider(
                "Number of top words to show",
                min_value = 5, max_value = 20, value = 10
            )
            # let the user pick which presidents to show
            available_presidents = tfidf_president.index.tolist()
            chosen_presidents    = st.sidebar.multiselect(
                "Presidents to show",
                options = available_presidents,
                default = available_presidents
            )

            filtered_tfidf = tfidf_president.loc[chosen_presidents]   # filter to selected presidents
            topwords       = get_top_words(filtered_tfidf, n=n_words)

            # tabs split bar charts and word clouds onto separate sub-pages
            tab1, tab2, tab3 = st.tabs(["📊 Bar charts", "☁️ Word clouds", "📃 Some Interpretations"])
            with tab1:
                fig = plot_barchart(topwords, "Most distinctive words per Romanian president")
                st.pyplot(fig)
            with tab2:
                fig = plot_wordclouds(filtered_tfidf, ro_stopwords,
                                    "Presidential speech word clouds")
                st.pyplot(fig)
            with tab3:
                st.markdown(
                    "The TF-IDF analysis reveals the unique rhetorical focuses of each Romanian president. \n" \
                    "There is a constant mention of the term **European Union**, highlighting the importance of EU integration across all presidencies, but each president also has their own distinctive themes that set them apart from the others."

                )
                st.markdown("---")   
                st.markdown(
                    "For example, Iliescu's top words include mostly stuff about **economy** and **economic growth**, showing what Romania had to focus on after the Revolution of 1989"
                )
                st.markdown("---")   
                st.markdown(
                    "Constantinescu's distinctive terms are more about **NGOs**, **organised crime** and **reform**, reflecting his focus on anti-corruption and prerequites of EU integration during his term.")
                st.markdown("---")   
                st.markdown(
                    "Băsescu's top words include **january 2007** - a distinctive date for Romania as that is when we joined the EU and **Omar Hayssam** - which was a terrorist that caused notable indignationin the Romanian public space")
                st.markdown("---")   
                st.markdown(
                    "Iohannis is dictinctive through his overuse of the term **ladies and gentelmen** and **Educated Romania**, which was the name of one of his most important projects")
    # ====================================================================================
    # PAGE 4 — PARLIAMENTARY TF-IDF
    # ====================================================================================

    elif page == "🏛️ Parliamentary TF-IDF":
        st.title("Parliamentary Speech TF-IDF")
        st.markdown(
            "Most distinctive words in parliamentary debates during each presidential term. "
            "Grouped by who was president at the time of each speech."
        )
        st.markdown("---")

        tfidf_parliament = load_parliament_tfidf()

        if tfidf_parliament is None:
            st.warning(
                "Parliamentary TF-IDF matrix not found.\n\n"
                "Run the main script first to generate "
                f"`{PARLIAMENT_CACHE_FILE}`."
            )
        else:
            n_words = st.sidebar.slider(
                "Number of top words to show",
                min_value = 5, max_value = 30, value = 20
            )
            topwords = get_top_words(tfidf_parliament, n=n_words)

            tab1, tab2 = st.tabs(["📊 Bar charts", "☁️ Word clouds"])
            with tab1:
                fig = plot_barchart(topwords, "Parliamentary discourse per presidential term")
                st.pyplot(fig)
            with tab2:
                fig = plot_wordclouds(tfidf_parliament, ro_stopwords,
                                    "Parliamentary word clouds per term")
                st.pyplot(fig)


    # ====================================================================================
    # PAGE 5 — PRESIDENT VS PARLIAMENT COMPARISON
    # ====================================================================================

    elif page == "🔀 President vs Parliament":
        st.title("President vs Parliament -- Discourse Comparison")
        st.markdown(
            "Did the president's rhetoric filter into parliament, or did they talk about "
            "completely different things? Compare top TF-IDF terms side by side."
        )
        st.markdown("---")

        tfidf_president  = load_president_tfidf()
        tfidf_parliament = load_parliament_tfidf()

        if tfidf_president is None or tfidf_parliament is None:
            st.warning(
                "One or both TF-IDF cache files are missing. "
                "Run the main script first to generate both CSV files."
            )
        else:
            # only compare presidents that exist in BOTH matrices
            common_presidents = list(
                set(tfidf_president.index) & set(tfidf_parliament.index)
            )

            # multiselect so user can choose one or both
            presidents_to_compare = st.sidebar.multiselect(
                "Select presidents to compare",
                options = common_presidents,
                default = common_presidents
            )

            if not presidents_to_compare:
                st.info("Select at least one president from the sidebar.")
            else:
                fig = plot_president_vs_parliament(
                    tfidf_president, tfidf_parliament, presidents_to_compare
                )
                st.pyplot(fig)

                st.markdown("---")
                st.subheader("Vocabulary overlap")
                st.caption("Words appearing in the top 20 of BOTH the president's speeches and parliament during their term")

                for pres in presidents_to_compare:
                    try:
                        # find words that are in the top 20 for both sources
                        pres_top20 = set(tfidf_president.loc[pres].nlargest(20).index)
                        parl_top20 = set(tfidf_parliament.loc[pres].nlargest(20).index)
                        overlap    = pres_top20 & parl_top20

                        if overlap:
                            st.markdown(f"**{pres}:** {', '.join(sorted(overlap))}")
                        else:
                            st.markdown(f"**{pres}:** No overlap in top 20 words")
                    except KeyError:
                        pass   # skip silently if president not found



generate_dashboard()
