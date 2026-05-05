{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyNhV/EoMLFN1uIo3YC4OgyZ",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/fanisam/Praktikum_AI/blob/main/apss2.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import streamlit as st\n",
        "import pandas as pd\n",
        "import numpy as np\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.ensemble import RandomForestClassifier\n",
        "\n",
        "@st.cache_data\n",
        "def load_data():\n",
        "    df = pd.read_csv(\"Mall_Customers.csv\")\n",
        "    return df\n",
        "\n",
        "@st.cache_data\n",
        "def preprocess(df):\n",
        "    # Encode gender\n",
        "    df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})\n",
        "    return df\n",
        "\n",
        "@st.cache_data\n",
        "def train_model(df):\n",
        "    # Kita gunakan Spending Score sebagai target (klasifikasi sederhana)\n",
        "    df['Spending_Category'] = pd.cut(df['Spending Score (1-100)'],\n",
        "                                    bins=[0, 40, 70, 100],\n",
        "                                    labels=[\"Low\", \"Medium\", \"High\"])\n",
        "\n",
        "    X = df[['Gender', 'Age', 'Annual Income (k$)']]\n",
        "    y = df['Spending_Category']\n",
        "\n",
        "    X_train, X_test, y_train, y_test = train_test_split(\n",
        "        X, y, test_size=0.2, random_state=42\n",
        "    )\n",
        "\n",
        "    model = RandomForestClassifier()\n",
        "    model.fit(X_train, y_train)\n",
        "    accuracy = model.score(X_test, y_test)\n",
        "\n",
        "    return model, accuracy\n",
        "\n",
        "def main():\n",
        "    st.set_page_config(page_title=\"Mall Customer App\", layout=\"centered\")\n",
        "    st.title(\"🛍️ Klasifikasi Customer Mall\")\n",
        "    st.write(\"Masukkan data pelanggan:\")\n",
        "\n",
        "    df = load_data()\n",
        "    df = preprocess(df)\n",
        "    model, accuracy = train_model(df)\n",
        "\n",
        "    # Input user\n",
        "    gender = st.selectbox(\"Gender\", [\"Male\", \"Female\"])\n",
        "    age = st.slider(\"Age\", int(df.Age.min()), int(df.Age.max()), int(df.Age.mean()))\n",
        "    income = st.slider(\"Annual Income (k$)\",\n",
        "                       int(df['Annual Income (k$)'].min()),\n",
        "                       int(df['Annual Income (k$)'].max()),\n",
        "                       int(df['Annual Income (k$)'].mean()))\n",
        "\n",
        "    gender_val = 0 if gender == \"Male\" else 1\n",
        "\n",
        "    if st.button(\"Prediksi\"):\n",
        "        input_data = np.array([[gender_val, age, income]])\n",
        "        prediction = model.predict(input_data)[0]\n",
        "\n",
        "        st.success(f\"🧾 Kategori Customer: **{prediction}**\")\n",
        "        st.info(f\"Akurasi model: {accuracy*100:.2f}%\")\n",
        "\n",
        "    if st.checkbox(\"Tampilkan Dataset\"):\n",
        "        st.dataframe(df)\n",
        "\n",
        "if __name__ == \"__main__\":\n",
        "    main()"
      ],
      "metadata": {
        "id": "KQB5tUFYFutv",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 383
        },
        "outputId": "54440ffd-d482-4900-f181-316ab18df358"
      },
      "execution_count": 1,
      "outputs": [
        {
          "output_type": "error",
          "ename": "ModuleNotFoundError",
          "evalue": "No module named 'streamlit'",
          "traceback": [
            "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
            "\u001b[0;31mModuleNotFoundError\u001b[0m                       Traceback (most recent call last)",
            "\u001b[0;32m/tmp/ipykernel_3494/3894933356.py\u001b[0m in \u001b[0;36m<cell line: 0>\u001b[0;34m()\u001b[0m\n\u001b[0;32m----> 1\u001b[0;31m \u001b[0;32mimport\u001b[0m \u001b[0mstreamlit\u001b[0m \u001b[0;32mas\u001b[0m \u001b[0mst\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m      2\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mpandas\u001b[0m \u001b[0;32mas\u001b[0m \u001b[0mpd\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      3\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mnumpy\u001b[0m \u001b[0;32mas\u001b[0m \u001b[0mnp\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      4\u001b[0m \u001b[0;32mfrom\u001b[0m \u001b[0msklearn\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mmodel_selection\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mtrain_test_split\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      5\u001b[0m \u001b[0;32mfrom\u001b[0m \u001b[0msklearn\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mensemble\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mRandomForestClassifier\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;31mModuleNotFoundError\u001b[0m: No module named 'streamlit'",
            "",
            "\u001b[0;31m---------------------------------------------------------------------------\u001b[0;32m\nNOTE: If your import is failing due to a missing package, you can\nmanually install dependencies using either !pip or !apt.\n\nTo view examples of installing some common dependencies, click the\n\"Open Examples\" button below.\n\u001b[0;31m---------------------------------------------------------------------------\u001b[0m\n"
          ],
          "errorDetails": {
            "actions": [
              {
                "action": "open_url",
                "actionText": "Open Examples",
                "url": "/notebooks/snippets/importing_libraries.ipynb"
              }
            ]
          }
        }
      ]
    }
  ]
}