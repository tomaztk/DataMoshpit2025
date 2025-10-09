# DataMoshpit2025
Demo for DataMoshpit - Psycho-social Metal 🤘 Analysis

## Theory behind

Trait data: There is no scientific dataset that has measured Big Five or psychosocial values for each band individually. What we can do is approximate band profiles from their subgenre tags, because the psychological research gives us associations at the genre family level (e.g., “extreme metal” → higher nonconformity/uniqueness, “progressive” → higher openness, etc.).

ToDO:

1. Loads the full band roster (e.g. from Kaggle’s Metal Archives dataset  URL: https://www.kaggle.com/datasets/guimacrlh/every-metal-archives-band-october-2024/data  or via API : https://www.metal-api.dev/index.html )
2. Normalizes the genre text.
3. Uses a subgenre → trait mapping dictionary (like the one we’ve started, but expanded).
4. Outputs a CSV: Band, Country, Genre, Openness, Nonconformity, Authority_skepticism, Extraversion, Emotion_regulation, Sensation_seeking


| Band      | Genre             | Openness | Nonconformity | Authority skepticism | Extraversion | Emotion regulation | Sensation-seeking |
| --------- | ----------------- | -------- | ------------- | -------------------- | ------------ | ------------------ | ----------------- |
| Slayer    | Thrash Metal      | 0.68     | 0.60          | 0.60                 | 0.42         | 0.60               | 0.70              |
| Opeth     | Progressive Metal | 0.85     | 0.60          | 0.50                 | 0.40         | 0.60               | 0.60              |
| Nightwish | Symphonic Metal   | 0.70     | 0.50          | 0.30                 | 0.60         | 0.55               | 0.50              |
| Behemoth  | Blackened Death   | 0.70     | 0.75          | 0.70                 | 0.30         | 0.65               | 0.75              |
| Sabaton   | Power Metal       | 0.65     | 0.45          | 0.30                 | 0.70         | 0.50               | 0.55              |


## Local run

1. Questionnaire on app 
 run metal_recommender_app.py and make sure to have installed streamlit:

```(Python)
pip install streamlit 
pip install pandas 
pip install numpy
streamlit run metal_recommender_app.py
```
2. go to folder and run:

```{Python}
streamlit run metal_recommender_app.py
```

open URL: http://localhost:8501/


## Deployment on Azure Web app (stat)

1. Have Resource group created and Web App created
2. Run Github Actions and change YAML file
3. Have permissions on on resource group added
4. Add role assignment (role = Contributor) for your Web app