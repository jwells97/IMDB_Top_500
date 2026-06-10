# Install dependencies as needed:
# pip install kagglehub[pandas-datasets] matplotlib seaborn adjustText
import kagglehub
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from adjustText import adjust_text
from kagglehub import KaggleDatasetAdapter

# Set the path to the file you'd like to load
file_path = "Top 500 Movies Ranked by Combined Critics and Audience Scores.csv"

# Load the latest version
df = kagglehub.dataset_load(
    KaggleDatasetAdapter.PANDAS,
    "prashant0kumar7/top-500-movies-ranked-by-combined-critics",
    file_path,
)

sns.set_theme(style="whitegrid")

# 1. Rating distributions
rating_columns = [
    ("Audience_Rating", "Audience Rating"),
    ("Critic_Rating_RT", "Critic Rating (RT)"),
    ("IMDb_10", "IMDb Rating (out of 10)"),
    ("Custom_Score", "Custom Score"),
]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("Rating Distributions", fontsize=14)

for ax, (column, label) in zip(axes.flat, rating_columns):
    sns.histplot(df[column].dropna(), kde=True, ax=ax, color="steelblue")
    ax.set_title(label)
    ax.set_xlabel(label)
    ax.set_ylabel("Count")

plt.tight_layout()
plt.savefig("rating_distributions.png", dpi=150)
plt.show()

# 2. Movies per decade
df["Decade"] = (df["Year"] // 10) * 10
decade_counts = df["Decade"].value_counts().sort_index()

plt.figure(figsize=(10, 5))
sns.barplot(
    x=decade_counts.index.astype(str),
    y=decade_counts.values,
    color="steelblue",
)
plt.title("Movies per Decade")
plt.xlabel("Decade")
plt.ylabel("Number of Movies")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("movies_per_decade.png", dpi=150)
plt.show()

# 3. Top directors on the list
top_directors = df["Director"].value_counts().head(15)

plt.figure(figsize=(10, 6))
sns.barplot(
    x=top_directors.values,
    y=top_directors.index,
    color="steelblue",
)
plt.title("Top 15 Directors on the List")
plt.xlabel("Number of Movies")
plt.ylabel("Director")
plt.tight_layout()
plt.savefig("top_directors.png", dpi=150)
plt.show()

print("Saved charts:")
print("- rating_distributions.png")
print("- movies_per_decade.png")
print("- top_directors.png")

# Step 1: get the names of the top 5 directors
top_5_directors = df["Director"].value_counts().head(15).index

# Step 2: keep only rows where Director is in that list
top_director_movies = df[df["Director"].isin(top_5_directors)]

# Step 3: group and print the movies by director
for director in top_5_directors:
    print(f"\n{director}:")
    movies = (
        df[df["Director"] == director][["Title", "Year", "Audience_Rating", "Custom_Score"]]
        .sort_values("Audience_Rating", ascending=False)
    )
    print(movies.to_string(index=False))

# 4. Critics vs. audience scatter plot
ratings = df.dropna(subset=["Critic_Rating_RT", "Audience_Rating"]).copy()
ratings["Rating_Gap"] = ratings["Audience_Rating"] - ratings["Critic_Rating_RT"]

plt.figure(figsize=(10, 8))
sns.scatterplot(
    data=ratings,
    x="Critic_Rating_RT",
    y="Audience_Rating",
    alpha=0.6,
    color="steelblue",
)
plt.axline((70, 70), slope=1, color="gray", linestyle="--", label="Perfect agreement")
plt.xlabel("Critic Rating (Rotten Tomatoes)")
plt.ylabel("Audience Rating")
plt.title("Critics vs. Audience Ratings")
plt.legend()

# Label disagreements and perfect-agreement movies; adjust_text prevents overlap
outliers = pd.concat([
    ratings.nlargest(4, "Rating_Gap"),
    ratings.nsmallest(4, "Rating_Gap"),
])
perfect_agreement = ratings[ratings["Rating_Gap"] == 0].sort_values("Critic_Rating_RT")

texts = []
label_points = []

for _, movie in pd.concat([outliers, perfect_agreement]).iterrows():
    x = movie["Critic_Rating_RT"]
    y = movie["Audience_Rating"]
    label_points.append((x, y))
    texts.append(
        plt.text(
            x,
            y,
            movie["Title"],
            fontsize=7,
            alpha=0.85,
        )
    )

adjust_text(
    texts,
    x=[point[0] for point in label_points],
    y=[point[1] for point in label_points],
    arrowprops=dict(arrowstyle="-", color="gray", lw=0.5, alpha=0.6),
    expand=(1.5, 2.0),
)

plt.tight_layout()
plt.savefig("critics_vs_audience.png", dpi=150)
plt.show()

# 5. Average score by genre (requires splitting multi-genre labels)
genre_df = (
    df.assign(Genre=df["Genre"].str.split(", "))
    .explode("Genre")
    .assign(Genre=lambda d: d["Genre"].str.strip())
)
genre_scores = (
    genre_df.groupby("Genre")["Custom_Score"]
    .mean()
    .sort_values(ascending=True)
)

plt.figure(figsize=(10, 7))
sns.barplot(
    x=genre_scores.values,
    y=genre_scores.index,
    color="steelblue",
)
plt.title("Average Custom Score by Genre")
plt.xlabel("Average Custom Score")
plt.ylabel("Genre")
plt.tight_layout()
plt.savefig("genre_scores.png", dpi=150)
plt.show()

print("Saved charts:")
print("- critics_vs_audience.png")
print("- genre_scores.png")
