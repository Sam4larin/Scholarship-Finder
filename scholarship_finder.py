import streamlit as st
import requests
import pandas as pd
import re
import itertools
from datetime import datetime
from bs4 import BeautifulSoup
from ddgs import DDGS
from sentence_transformers import SentenceTransformer, util

# ------------------------------
# Load embedding model
# ------------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# Predefined academic fields (can be expanded or loaded from CSV)
ACADEMIC_FIELDS = [
    "computer science", "artificial intelligence", "machine learning",
    "deep learning", "data science", "business", "management",
    "entrepreneurship", "finance", "economics", "engineering",
    "project management", "medicine", "law", "psychology", "education",
    "chemistry", "biology", "physics", "mathematics", "statistics",
    "public health", "sociology", "political science", "philosophy"
]

# ------------------------------
# Utility Functions
# ------------------------------

def normalize_gpa(user_gpa, user_scale):
    """Convert GPA to % scale"""
    return (user_gpa / user_scale) * 100 if user_gpa and user_scale else None

def extract_deadline_from_text(text):
    """Extract deadline dates from text"""
    text = text.replace("\n", " ")
    patterns = [
        r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}",
        r"\d{1,2}\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}",
        r"\d{1,2}/\d{1,2}/\d{4}"
    ]
    for pat in patterns:
        match = re.search(pat, text, re.IGNORECASE)
        if match:
            date_str = match.group(0)
            for fmt in ["%B %d, %Y", "%d %B %Y", "%m/%d/%Y", "%d/%m/%Y"]:
                try:
                    return datetime.strptime(date_str, fmt).date()
                except:
                    continue
    return None

def extract_deadline_from_page(url):
    """Visit scholarship page and extract deadline"""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code != 200:
            return None
        text = BeautifulSoup(resp.text, "html.parser").get_text(separator=" ")
        return extract_deadline_from_text(text)
    except:
        return None

def expand_programs(user_programs, top_n=5):
    """Expand user programs with semantically similar fields"""
    embeddings_catalog = model.encode(ACADEMIC_FIELDS, convert_to_tensor=True)
    expanded = []
    for prog in user_programs:
        query_emb = model.encode(prog, convert_to_tensor=True)
        scores = util.pytorch_cos_sim(query_emb, embeddings_catalog)[0]
        top_indices = scores.topk(top_n).indices.tolist()
        expanded.extend([ACADEMIC_FIELDS[i] for i in top_indices])
    return list(set(user_programs + expanded))

def fetch_scholarships(queries, max_results=10):
    """Search scholarships via DuckDuckGo"""
    results = []
    with DDGS() as ddgs:
        for program, country in queries:
            q = f"{program} scholarships {country}"
            try:
                for r in ddgs.text(q, max_results=max_results):
                    if "scholarship" not in r.get("title", "").lower():
                        continue
                    results.append({
                        "Title": r.get("title"),
                        "Link": r.get("href"),
                        "Snippet": r.get("body"),
                        "Program": program,
                        "Country": country
                    })
            except Exception:
                continue
    return pd.DataFrame(results).drop_duplicates(subset="Link")

def enrich_scholarship_info(row):
    """Fetch more info from scholarship webpage"""
    url = row["Link"]
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code != 200:
            return row
        soup = BeautifulSoup(resp.text, "html.parser")
        text = soup.get_text(separator=" ")

        # Extract deadline
        row["Deadline"] = extract_deadline_from_text(text)

        # Extract amount (simple regex for $ or €)
        amt_match = re.search(r"(\$|€|£)\s?\d+[,\d]*", text)
        row["Amount"] = amt_match.group(0) if amt_match else None

        # Provider (first h1/h2 or title tag)
        row["Provider"] = soup.title.string if soup.title else None

        return row
    except:
        return row

def filter_and_rank(df, profile, preference):
    """Apply filtering and ranking"""
    if df.empty:
        return df

    # Enrich with webpage info
    df = df.apply(enrich_scholarship_info, axis=1)

    # Filter expired
    today = datetime.today().date()
    df = df[df["Deadline"].isnull() | (df["Deadline"] >= today)]

    # Compute Score
    df["Score"] = 0

    # Program fit
    for prog in profile["programs"]:
        df.loc[df["Snippet"].str.contains(prog, case=False, na=False), "Score"] += 2

    # Degree fit
    df.loc[df["Snippet"].str.contains(profile["degree"], case=False, na=False), "Score"] += 2

    # GPA fit (if GPA requirement mentioned, crude filter)
    if profile["gpa_norm"]:
        df["GPA_Fit"] = True
        df.loc[df["Snippet"].str.contains("gpa", case=False, na=False), "Score"] += 1

    # Nationality/Ethnicity
    if profile["nationality"]:
        df.loc[df["Snippet"].str.contains(profile["nationality"], case=False, na=False), "Score"] += 1
    if profile["ethnicity"]:
        df.loc[df["Snippet"].str.contains(profile["ethnicity"], case=False, na=False), "Score"] += 1

    # Sorting
    if preference == "Best Fit":
        df = df.sort_values(by="Score", ascending=False)
    elif preference == "Highest Chance":
        df = df.sort_values(by=["Score"], ascending=False)
    elif preference == "Soonest Deadline":
        df["DaysLeft"] = df["Deadline"].apply(lambda d: (d - today).days if d else None)
        df = df.sort_values(by=["DaysLeft", "Score"], ascending=[True, False])

    return df

# ------------------------------
# Streamlit App
# ------------------------------
st.title("🎓 Scholarship Finder with Profile Matching")

with st.sidebar:
    st.header("User Profile")

    degree = st.selectbox("Degree of Study", ["Bachelor", "Master", "PhD"])
    gpa = st.number_input("Your CGPA", min_value=0.0, max_value=5.0, step=0.1)
    gpa_scale = st.selectbox("CGPA Scale", [4.0, 5.0])
    normalized_gpa = normalize_gpa(gpa, gpa_scale)

    programs_input = st.text_input("Programs of Interest (comma separated)", "machine learning, business")
    programs = [p.strip() for p in programs_input.split(",") if p.strip()]
    expanded_programs = expand_programs(programs)

    countries = st.multiselect("Countries to Study", ["Global", "USA", "UK", "Canada", "Germany", "Australia"])
    if "Global" in countries or not countries:
        countries = [""]  # search all

    nationality = st.text_input("Country of Nationality")
    residence = st.text_input("Country of Residence")
    ethnicity = st.selectbox("Ethnicity (optional)", ["", "African", "Asian", "European", "Latino", "Indigenous", "Middle Eastern"])

    university = st.text_input("University (optional)")

    preference = st.selectbox("Sort Results By", ["Best Fit", "Highest Chance", "Soonest Deadline"])

    search_btn = st.button("🔎 Search Scholarships")


if search_btn:
    st.write("⏳ Searching scholarships... please wait")

    # Build profile, only add optional fields if provided
    profile = {
        "degree": degree,
        "programs": expanded_programs,
        "nationality": nationality if nationality else None,
        "residence": residence if residence else None,
        "ethnicity": ethnicity if ethnicity else None,
        "university": university if university else None,
        "gpa_norm": normalized_gpa if gpa > 0 else None
    }

    # Query construction: only add optional fields if provided
    queries = []
    for program in expanded_programs:
        for country in countries:
            query_parts = [program, "scholarships", degree]
            if country:
                query_parts.append(country)
            if university:
                query_parts.append(university)
                query_parts.append("university scholarships")
            # CGPA and ethnicity/gender only if provided
            if profile["gpa_norm"]:
                query_parts.append(f"CGPA {gpa}")
            if ethnicity:
                query_parts.append(ethnicity)
            queries.append((" ".join([str(x) for x in query_parts if x]), country))

    # Fetch more results for better coverage
    df = pd.DataFrame()
    for q, country in queries:
        temp_df = fetch_scholarships([(q, country)], max_results=10)
        df = pd.concat([df, temp_df], ignore_index=True)
    results = filter_and_rank(df, profile, preference)

    if results.empty:
        st.warning("No scholarships found. Try broadening your filters.")
    else:
        st.success(f"Found {len(results)} scholarships")

        # Format links as clickable
        results["Link"] = results["Link"].apply(lambda x: f"[🔗 Visit]({x})" if x else "")

        # Display results in a table
        st.write("### Scholarship Results")
        st.write(results[["Title", "Provider", "Amount", "Deadline", "Link", "Score"]].to_markdown(index=False), unsafe_allow_html=True)

        # Download buttons
        st.write("### Download Results")
        csv = results.to_csv(index=False)
        st.download_button("Download as CSV", csv, "scholarships.csv", "text/csv")
        import io
        excel_buffer = io.BytesIO()
        results.to_excel(excel_buffer, index=False, engine='openpyxl')
        excel_buffer.seek(0)
        st.download_button("Download as Excel", excel_buffer, "scholarships.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
