# 🎬 Hybrid Movie Recommender System

Welcome to the **Hybrid Movie Recommender System**, a sophisticated platform that provides personalized movie recommendations by leveraging both Collaborative Filtering (CF) and Content-Based Filtering (CBF) using a hybrid LightFM model. This project integrates a Streamlit web application for an interactive user experience and utilizes NCSU's High-Performance Computing (HPC) resources to handle computationally intensive tasks efficiently.

## Table of Contents

- [🎬 Hybrid Movie Recommender System](#-hybrid-movie-recommender-system)
  - [Table of Contents](#table-of-contents)
  - [🚀 Project Overview](#-project-overview)
  - [🔍 Features](#-features)
  - [🛠️ Technologies Used](#️-technologies-used)
  - [📈 Model Optimization](#-model-optimization)
  - [💻 Environment Setup](#-environment-setup)
    - [Using Conda](#using-conda)
    - [Configuration](#configuration)
  - [🔧 Running the Application](#-running-the-application)
    - [Local Deployment](#local-deployment)
    - [HPC Deployment](#hpc-deployment)
  - [📝 Scripts Overview](#-scripts-overview)
  - [📊 User Feedback Mechanism](#-user-feedback-mechanism)
  - [📦 Repository Structure](#-repository-structure)
  - [📚 Additional Resources](#-additional-resources)
  - [📜 License](#-license)
  - [📬 Contact](#-contact)

---

## 🚀 Project Overview

The **Hybrid Movie Recommender System** aims to provide users with personalized movie suggestions by combining the strengths of both Collaborative Filtering and Content-Based Filtering. By utilizing a hybrid LightFM model, the system offers accurate and relevant recommendations based on user preferences and movie metadata.

## 🔍 Features

- **Personalized Recommendations:** Tailored movie suggestions based on user-selected favorite movies.
- **Interactive Web Interface:** User-friendly Streamlit application for seamless interaction.
- **Efficient Model Training:** Optimized model training using reduced sparsity to enhance computational efficiency.
- **Scalable Deployment:** Leveraging NCSU's HPC resources for handling large-scale computations.
- **User Feedback:** Mechanism for users to provide feedback on recommendations, enabling continuous improvement.

## 🛠️ Technologies Used

- **Programming Language:** Python 3.9
- **Libraries & Frameworks:** 
  - [Streamlit](https://streamlit.io/) for the web application.
  - [LightFM](https://making.lyst.com/lightfm/docs/home.html) for the hybrid recommendation model.
  - [Pandas](https://pandas.pydata.org/) for data manipulation.
  - [Scikit-learn](https://scikit-learn.org/stable/) for PCA and other preprocessing tasks.
  - [Joblib](https://joblib.readthedocs.io/en/latest/) for model serialization.
  - [PyYAML](https://pyyaml.org/) for configuration management.
  - [Requests](https://docs.python-requests.org/en/latest/) for API interactions.
- **Deployment:** NCSU High-Performance Computing (HPC) cluster and Streamlit Cloud.

## 📈 Model Optimization

To enhance the computational efficiency of our **Hybrid LightFM Model**, we reduced the sparsity of user metadata. High sparsity can lead to increased computational overhead and longer training times. Here's how we achieved this:

1. **Dimensionality Reduction with PCA:**
   - Applied Principal Component Analysis (PCA) to reduce the dimensionality of item features.
   - Selected a suitable number of components (`n_components`) to capture significant variance in the data.

2. **Introducing Sparsity:**
   - Implemented a thresholding mechanism post-PCA to set values below a certain threshold to zero.
   - Achieved a desired sparsity level (e.g., 15%) to balance between model performance and computational efficiency.

3. **Benefits:**
   - **Reduced Computational Load:** Lower sparsity leads to fewer non-zero entries, accelerating matrix operations.
   - **Faster Training:** Decreased model training time without significant loss in recommendation quality.
   - **Memory Efficiency:** Lower memory consumption due to reduced data dimensionality.

The script responsible for this optimization is [`reduce_item_features.py`](src/scripts/reduce_item_features.py).

## 💻 Environment Setup

### Using Conda

We utilize Conda for environment management to ensure reproducibility and ease of dependency handling.

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/your-username/hybrid-movie-recommender.git
   cd hybrid-movie-recommender
   ```

2. **Create the Conda Environment:**

   Ensure you have Conda installed. Then, create the environment using the provided `environment.yml`:

   ```bash
   conda env create -f environment.yml
   ```

3. **Activate the Environment:**

   ```bash
   conda activate movie-env
   ```

4. **Verify Installation:**

   List installed packages to ensure all dependencies are correctly installed:

   ```bash
   conda list
   ```

### Configuration

1. **Configuration File (`config.yml`):**

   The `config.yml` file contains essential configuration parameters such as file paths and API keys. Ensure it's correctly set up before running the application.

   ```yaml
   imdb_datasets:
     title_basics: 'data/raw/title.basics.tsv.gz'
     title_ratings: 'data/raw/title.ratings.tsv.gz'
     title_akas: 'data/raw/title.akas.tsv.gz'  # Optional
     ratings_movies_imdb_merged: 'data/processed/ratings_movies_imdb_merged.csv'

   include_akas: false  # Set to true to include alternative titles

   collaborative_filtering:
     interaction_matrix_path: 'data/processed/interaction_matrix.npz'
     model_path: 'data/processed/lightfm_model.joblib'
     user_mapping: 'data/processed/user_mapping.csv'
     movie_mapping: 'data/processed/movie_mapping.csv'
     item_features_path: 'data/processed/item_features_reduced.npz'
     pca_components: 100
     pca_desired_sparsity: 0.15  # 15% non-zero entries

   web_app:
     title_to_index_path: 'data/processed/title_to_index.pkl'

   tmdb:
     api_key: 'your_tmdb_api_key_here'  # Replace with your TMDb API key

   hybrid_recommendations:
     output_path: 'data/processed/hybrid_recommendations.csv'
   ```

2. **Obtaining TMDb API Key (Optional):**

   To display movie posters in the web application, obtain an API key from [The Movie Database (TMDb)](https://www.themoviedb.org/documentation/api) and add it to the `tmdb` section in `config.yml`.

## 🔧 Running the Application

### Local Deployment

1. **Activate the Conda Environment:**

   ```bash
   conda activate movie-env
   ```

2. **Navigate to the Web App Directory:**

   ```bash
   cd web_app/
   ```

3. **Launch the Streamlit App:**

   ```bash
   streamlit run app.py
   ```

4. **Access the Application:**

   Open your web browser and navigate to the URL provided by Streamlit (typically `http://localhost:8501`).

### HPC Deployment

For training models and performing computationally intensive tasks, we utilize NCSU's HPC resources. Here's how to submit jobs to the HPC cluster.

1. **Batch Submission Script (`cfd_train_model.bsub`):**

   ```bash
   #!/bin/bash
   #BSUB -n 8
   #BSUB -R "span[hosts=1]"
   #BSUB -W 5:30
   #BSUB -q shared_memory
   #BSUB -J cfd_train_model
   #BSUB -o out.%J
   #BSUB -e err.%J

   # Load Conda environment
   source /usr/local/apps/miniconda20240526/etc/profile.d/conda.sh

   # Activate the movie-env environment
   conda activate /share/blondin/jrfloren/conda_envs/movie-env

   # Run the model training script and redirect logs
   python /share/blondin/jrfloren/movie-recommender/src/scripts/cf_train_model.py > /share/blondin/jrfloren/movie-recommender/data/processed/cf_train_model_log.txt

   # Deactivate the Conda environment
   conda deactivate
   ```

2. **Submitting the Job:**

   Save the above script as `cfd_train_model.bsub` and submit it using the `bsub` command:

   ```bash
   bsub < cfd_train_model.bsub
   ```

3. **Monitoring the Job:**

   - **Output Logs:** `out.<JOB_ID>`
   - **Error Logs:** `err.<JOB_ID>`
   - **Training Logs:** `data/processed/cf_train_model_log.txt`

   Use commands like `bjobs`, `bpeek`, or `bhist` to monitor job status.

## 📝 Scripts Overview

The project comprises several scripts organized for clarity and efficiency. Below is an overview of key scripts:

- **Data Loading & Preprocessing:**
  - `src/scripts/data_loading.py`: Handles loading and initial preprocessing of raw data.
  - `src/scripts/preprocessing.py`: Performs data cleaning, feature engineering, and preparation for modeling.
  - `src/scripts/create_id_mappings.py`: Creates mappings between user/movie IDs and their indices.
  - `src/scripts/create_title_to_index.py`: Generates a mapping from movie titles to their respective indices.

- **Model Training & Evaluation:**
  - `src/scripts/cf_train_model.py`: Trains the Collaborative Filtering model using LightFM.
  - `src/scripts/cf_train_model_diagnostic.py`: Evaluates the trained model and generates diagnostic metrics.
  - `src/scripts/content_based_filtering.py`: Implements Content-Based Filtering logic.

- **Model Optimization:**
  - `src/scripts/reduce_item_features.py`: Reduces the sparsity of item features using PCA and thresholding to enhance computational efficiency.

- **Recommendation Generation:**
  - `src/scripts/cf_generate_recommendations.py`: Generates movie recommendations based on user input and the trained model.

- **Exploratory Data Analysis:**
  - `src/scripts/eda.py`: Conducts exploratory data analysis to understand data distributions and insights.
  - `src/scripts/imdb_eda.py`: Performs EDA specifically on IMDb datasets.

## 📊 User Feedback Mechanism

To continually improve the recommendation system, we have integrated a user feedback mechanism allowing users to "Like" or "Dislike" recommended movies.

1. **Feedback Buttons:**

   In the Streamlit application, each recommended movie displays "Like" and "Dislike" buttons.

2. **Logging Feedback:**

   User feedback is logged into a CSV file (`data/processed/user_feedback.csv`) with details such as the movie title, type of feedback, and timestamp. This data can be utilized for future model refinements.

   ```python
   def log_feedback(movie, feedback, log_path='data/processed/user_feedback.csv'):
       timestamp = datetime.datetime.now().isoformat()
       log_entry = {'movie': movie, 'feedback': feedback, 'timestamp': timestamp}
       feedback_df = pd.DataFrame([log_entry])

       # Ensure the directory exists
       os.makedirs(os.path.dirname(log_path), exist_ok=True)

       if not os.path.exists(log_path):
           feedback_df.to_csv(log_path, index=False)
       else:
           feedback_df.to_csv(log_path, mode='a', header=False, index=False)
   ```

## 📦 Repository Structure

```
hybrid-movie-recommender/
├── config.yml
├── data/
│   ├── raw/
│   └── processed/
├── environment.yml
├── Makefile
├── models/
│   └── lightfm_model.joblib
├── notebooks/
├── README.md
├── requirements.txt
├── running_jobs/
├── src/
│   ├── data_loading.py
│   ├── preprocessing.py
│   ├── preprocess.py
│   ├── scripts/
│   │   ├── 1
│   │   ├── cf_generate_recommendations.py
│   │   ├── cf_prepare_data.py
│   │   ├── cf_train_model.py
│   │   ├── cf_train_model_diagnostic.py
│   │   ├── content_based_filtering.py
│   │   ├── create_id_mappings.py
│   │   ├── create_title_to_index.py
│   │   ├── eda.py
│   │   ├── imdb_eda.py
│   │   ├── reduce_item_features.py
│   │   └── ...
│   └── __pycache__/
├── tests/
└── web_app/
    └── app.py
```

## 📚 Additional Resources

- **LightFM Documentation:** [https://making.lyst.com/lightfm/docs/home.html](https://making.lyst.com/lightfm/docs/home.html)
- **Streamlit Documentation:** [https://docs.streamlit.io/](https://docs.streamlit.io/)
- **NCSU HPC Documentation:** [NCSU HPC Resources](https://hpc.ncsu.edu/)
- **TMDb API Documentation:** [https://developers.themoviedb.org/3](https://developers.themoviedb.org/3)

## 📜 License

This project is licensed under the [MIT License](LICENSE).

## 📬 Contact

For any questions, suggestions, or feedback, please reach out to:

**Justin R Florence**  
Email: [jrfloren@ncsu.edu](mailto:jrfloren@ncsu.edu)  
GitHub: [https://github.com/justinflorence](https://github.com/justinflorence)

---

**Happy Coding! 🚀**

---

# Additional Notes for the README

1. **Replace Placeholders:**
   - **Repository URL:** Update `https://github.com/your-username/hybrid-movie-recommender.git` with your actual GitHub repository URL.
   - **GitHub Username:** Replace `your-username` with your actual GitHub username.
   - **Contact Information:** Update the contact section with your actual details.
   - **License:** Ensure you have a `LICENSE` file in your repository or adjust the license section accordingly.

2. **Scripts and File Paths:**
   - Ensure that all file paths mentioned in the README (like `config.yml`, script locations, etc.) accurately reflect your repository structure.
   - Verify that all scripts have appropriate permissions and are executable where necessary.

3. **Environment Variables and Secrets:**
   - For security, especially concerning API keys, consider using environment variables or Streamlit's [Secrets Management](https://docs.streamlit.io/en/stable/cloud/deploy-streamlit-app.html#manage-secrets) feature instead of hardcoding them in `config.yml`.

4. **Continuous Integration (CI):**
   - Consider setting up CI pipelines (e.g., GitHub Actions) for automated testing, linting, and deployment to ensure code quality and streamline development workflows.

5. **Deployment on Streamlit Cloud:**
   - The README mentions deploying on Streamlit Cloud. Ensure you follow Streamlit's [deployment guidelines](https://docs.streamlit.io/en/stable/deploy_streamlit_app.html) for a smooth deployment process.

6. **Feedback Mechanism Enhancement:**
   - While the current feedback mechanism logs user interactions, integrating real-time feedback analysis and model retraining could further enhance recommendation accuracy.

7. **Data Privacy Compliance:**
   - If your application collects user data, ensure compliance with relevant data privacy laws (e.g., GDPR, CCPA) and clearly state your data handling practices in the README or a separate privacy policy.

8. **Future Enhancements Roadmap:**
   - Consider adding a roadmap section or a `ROADMAP.md` file to outline future features and improvements, providing transparency and encouraging community contributions.

