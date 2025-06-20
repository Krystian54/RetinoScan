import os
import csv
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
from PIL import Image
from gui_functions import make_predictions
import pandas as pd


# ścieżki do modelów
ResNet_PATH = 'data/models/resnet50.pth'
DenseNet_PATH = 'data/models/densenet121.pth'
EfficientNet_PATH = 'data/models/efficientnet_b5.pth'
ViT_PATH = 'data/models/ViT_B_16.pth'
ConvNeXt_PATH = 'data/models/convnext_base.pth'

models_labels = ["ResNet", "DenseNet", "EfficientNet", "ViT", "ConvNeXt", "Inny"]
models_paths = {
    "ResNet": ResNet_PATH,
    "DenseNet": DenseNet_PATH,
    "EfficientNet": EfficientNet_PATH,
    "ViT": ViT_PATH,
    "ConvNeXt": ConvNeXt_PATH
}

# możliwe diagnozy
diagnosis = [
    '0 - Brak retinopatii cukrzycowej',
    '1 - Łagodna retinopatia cukrzycowa',
    '2 - Umiarkowana retinopatia cukrzycowa',
    '3 - Ciężka retinopatia cukrzycowa',
    '4 - Proliferacyjna retinopatia cukrzycowa'
]

# główna konfiguracja okna
st.set_page_config(page_title="RetinoScan", layout="wide")
st.title("RetinoScan")
st.write("Aplikacja do automatycznego wykrywania retinopatii cukrzycowej przy użyciu głębokich sieci neuronowych")

# podział na kolumny
col1, gap, col2 = st.columns([0.45,0.05, 0.5])

with col1:
    # ścieżki
    input_folder = st.text_input(label="**Plik/folder wejściowy:**", value="data/test_images/")
    output_folder = st.text_input("**Folder wyjściowy:**", value="results")

    # folder wyjściowy
    output_path = os.path.join(output_folder, "predictions.csv")
    os.makedirs(output_folder, exist_ok=True)

    # wybór modelu
    selected_model = st.selectbox("**Wybór modelu:**", models_labels)

    model_path = None

    if selected_model != "Inny":
        model_path = models_paths[selected_model]
    else:
        model_path_input = st.text_input("Podaj pełną ścieżkę do modelu (.pth):")
        if (model_path_input and os.path.exists(model_path_input)
                            and model_path_input.endswith(".pth")):
            model_path = model_path_input
        else:
            st.warning("Nieprawidłowa ścieżka. Przywrócono domyślny model ResNet.")
            selected_model_name = "ResNet"
            model_path = models_paths["ResNet"]

    # ustawienia przycisku
    st.markdown("""
        <style>
        div.stButton > button:first-child {
            background-color: #4CAF50;
            color: white;
            height: 3em;
            width: 10em;
            border-radius: 10px;
            font-size: 16px;
            font-weight: bold;
        }
        </style>
    """, unsafe_allow_html=True)

    # dokonanie predykcji
    if st.button("START"):
        try:
            make_predictions(model_path, input_folder, output_path)
            st.success("Algorytm zakończony sukcesem!")
        except:
            st.error(f"Błąd podczas dokonywania predykcji!")

    # odczyt danych z pliku csv
    if os.path.exists(output_path):
        with open(output_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            data = list(reader)

            if len(data) > 1:
                headers = data[0]

                # doddanie jednostki do tabeli
                headers[2] = headers[2] + " [%]"
                headers[3] = headers[3] + " [%]"

                rows = data[1:]

                # zaokrąglenie
                rows_rounded = []

                for row in rows:
                    new_row = row[:2]
                    
                    prob = float(row[2])
                    new_row.append(f"{round(prob * 100, 2)}")

                    raw_array = row[3].replace('[', '').replace(']', '')
                    float_list = [float(x) for x in raw_array.split()]
                    float_list_percentage = [round(x * 100, 2) for x in float_list]

                    string_list_percentage = "[" + " ".join(str(x) for x in float_list_percentage) + "]"

                    new_row.append(string_list_percentage)

                    rows_rounded.append(new_row)
                
                # tabela z wynikami
                st.write("**Wyniki (wybór wiersza do podglądu):**")
                gb = GridOptionsBuilder.from_dataframe(pd.DataFrame(rows_rounded, columns=headers))
                gb.configure_selection('single', use_checkbox=False)
                grid_options = gb.build()

                grid_response = AgGrid(
                    pd.DataFrame(rows_rounded, columns=headers),
                    gridOptions=grid_options,
                    update_mode=GridUpdateMode.SELECTION_CHANGED,
                    height=300,
                    fit_columns_on_grid_load=True
                )

                selected_row = grid_response['selected_rows']

with col2:
    # wyświetlenie obrazu
    try:
        if selected_row is not None:
            row = selected_row.iloc[0]
            selected_file = row[headers[0]]
            try:
                pred_idx = int(row[headers[1]])
                confidence = row[headers[2]]

                # konwersja na float
                list_of_probabilities = row[headers[3]].strip("[]")
                list_of_probabilities_float = [float(x) for x in list_of_probabilities.split()]
                list_of_probabilities_percent = [f"{x}%" for x in list_of_probabilities_float]

                st.markdown(f"**Nazwa pliku:** {selected_file}")
                
                # kolor czerwony jeżeli ryzyko choroby
                if pred_idx == 0:
                    st.markdown(f"**Przewidywana klasa:** <span style='color:green;'>{diagnosis[pred_idx]}</span>", unsafe_allow_html=True)
                else:
                    st.markdown(f"**Przewidywana klasa:** <span style='color:red;'>{diagnosis[pred_idx]}</span>", unsafe_allow_html=True)

                st.markdown(f"**Prawdopodobieństwo:** {confidence} %")
                st.markdown(f"**Prawdopodobieństwa klas:** [" + ", ".join(list_of_probabilities_percent) + "]")

                # obsługa pojedynczego zdjęcia
                if os.path.isdir(input_folder):

                    image_path = os.path.join(input_folder, selected_file)
                    image = Image.open(image_path)
                    st.image(image, width=600)

                else:

                    image_path = input_folder
                    image = Image.open(image_path)
                    st.image(image, width=600)

            except:
                st.error(f"Błąd podczas wyświetlania obrazu!")
    except:
        pass

# streamlit run gui.py
