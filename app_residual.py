
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import tensorflow as tf
import pickle
from io import BytesIO

# --- Загрузка моделей ---
@st.cache_resource
def load_model_3d():
    with open('best_model_interpolated_features.pkl', 'rb') as f:
        return pickle.load(f)

@st.cache_resource
def load_model_10d():
    return tf.keras.models.load_model('Model_dense_ostatki_10days.keras')

@st.cache_resource
def load_model_37d():
    return tf.keras.models.load_model('Model_transformer_ostatki_37days_realistic.keras')

@st.cache_resource
def load_scaler_3d():
    with open('scaler_interp_3days.pkl', 'rb') as f:
        return pickle.load(f)

@st.cache_resource
def load_scaler_10d():
    with open('scaler_interp_10days.pkl', 'rb') as f:
        return pickle.load(f)

@st.cache_resource
def load_scaler_37d():
    with open('scaler_interp_37days.pkl', 'rb') as f:
        return pickle.load(f)

# Загрузка моделей и скалеров
model_3d = load_model_3d()
model_10d = load_model_10d()
model_37d = load_model_37d()

scaler_10d = load_scaler_10d()
scaler_37d = load_scaler_37d()

# --- Категории филиалов ---
branch_categories = {
    'Абаканский ПУ': 0, 'Абалаково': 1, 'Восточный': 2, 'Западный': 3,
    'Красноярский участок': 4, 'Курагино': 5, 'Минусинск': 6,
    'Рыбинск': 8, 'Ужурский ПУ': 9, 'Юго-Восточ': 10
}

# --- Интерфейс ---
st.title("📦 Прогноз остатков за период")

uploaded_file = st.file_uploader("Перетащите сюда файл Excel/csv с данными обеспеченности на дату", type=['xlsx', 'csv'])

# --- Просмотр загруженных данных ---
if uploaded_file:
    if uploaded_file.name.endswith('.xlsx'):
        df_preview = pd.read_excel(uploaded_file)
    else:
        df_preview = pd.read_csv(uploaded_file)
    st.write("### 📄 Данные (первые 10 строк):")
    st.dataframe(df_preview.head(10))

# --- Выбор модели ---
model_choice = st.radio(
    "Выберите период прогноза:",
    options=["3 дня", "10 дней", "37 дней"],
    horizontal=True
)

# --- Загрузка данных ---
df = None
if uploaded_file:
    try:
        if uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)
        st.success("✅ Данные загружены!")
    except Exception as e:
        st.error(f"❌ Ошибка: {e}")
        st.stop()

# --- Подготовка признаков ---
if df is not None:
    df['Дата'] = pd.to_datetime(df['Дата'])
    df['Месяц'] = df['Дата'].dt.month

    # ФНБ
    fnb_cols = ['ФНБ рекорд трейдинг', 'ФНБ солид-Смоленск', 'ФНБ импория']
    for col in fnb_cols:
        df[col] = df[col].fillna(0)
    df['ФНБ общее'] = df[fnb_cols].sum(axis=1)
    df = df.drop(columns=fnb_cols)

    # Категория филиала
    df['Филиал_категория'] = df['Филиал'].map(branch_categories)
    if df['Филиал_категория'].isna().any():
        # st.warning("⚠️ Обнаружены неизвестные филиалы — заполнены как -1")
        df['Филиал_категория'] = df['Филиал_категория'].fillna(-1).astype(int)

# --- Предсказание ---
if st.button("Предсказать остатки"):
    if df is None:
        st.warning("⚠️ Загрузите данные!")
    else:
        try:
            # --- Выбор и отбор признаков ---
            feature_columns = {
                "3 дня": [
                    'Ост. НБ без хр', 'Ост. АЗС', 'хр. по талонам', 'Сумма остатков',
                    'В пути ГП-РП 95', 'ГП-РП на дату', 'ФНБ  в пути', 'В пути Биржа 95',
                    'МО', 'в пути ж/д для КНП', 'в пути ж/д общая', 'Цена НК Роснефть',
                    'Цена на нефть (Brent)', 'Филиал_категория', 'Месяц'
                ],
                "10 дней": [
                    'Ост. НБ без хр', 'Ост. АЗС', 'хр. по талонам', 'Сумма остатков',
                    'В пути ГП-РП 95', 'ГП-РП на дату', 'ФНБ  в пути', 'В пути Биржа 95',
                    'МО', 'в пути ж/д для КНП', 'в пути ж/д общая', 'Филиал_категория', 'Месяц'
                ],
                "37 дней": [
                    'Ост. НБ без хр', 'хр. по талонам', 'Сумма остатков', 'В пути ГП-РП 95',
                    'ГП-РП на дату', 'ФНБ  в пути', 'В пути Биржа 95', 'МО',
                    'в пути ж/д для КНП', 'в пути ж/д общая', 'закупка биржа', 'Цена биржа',
                    'ФНБ общее', 'Филиал_категория', 'Месяц'
                ]
            }

            selected_features = feature_columns[model_choice]

            # Проверка колонок
            missing_cols = [col for col in selected_features if col not in df.columns]
            if missing_cols:
                st.error(f"❌ Не хватает колонок: {missing_cols}")
                st.stop()

            X = df[selected_features].fillna(0)

            # --- Масштабирование и предсказание ---
            with st.spinner("🔄 Выполняю предсказание... Это может занять несколько секунд."):
                if model_choice == "3 дня":
                    predictions = model_3d.predict(X)
                elif model_choice == "10 дней":
                    X_scaled = scaler_10d.transform(X)
                    predictions = model_10d.predict(X_scaled)
                else:  # 37 дней
                    X_scaled = scaler_37d.transform(X)
                    X_scaled = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])
                    predictions = model_37d.predict(X_scaled)

            # --- Результат ---
            period_num = {"3 дня": 3, "10 дней": 10, "37 дней": 37}[model_choice]
            results_df = df[['Филиал', 'Дата', 'Ост. НБ без хр', 'Ост. АЗС', 'Сумма остатков']].copy()
            results_df[f'Прогноз остатков, {period_num} дн.'] = predictions.flatten()

            # --- Вывод ---
            st.write("### 📊 Результаты прогноза:")
            st.dataframe(results_df.style.format({
                "Дата": lambda x: x.strftime("%Y-%m-%d") if pd.notna(x) else "",
                f'Прогноз остатков, {period_num} дн.': "{:.1f}"
            }))

            # --- График ---
            st.write("### 📈 Визуализация (первые 20):")
            fig, ax = plt.subplots(figsize=(12, 5))
            sns.barplot(
                data=results_df.head(20),
                x=results_df['Филиал'][:20],
                y=f'Прогноз остатков, {period_num} дн.',
                ax=ax,
                color="skyblue"
            )
            ax.set_title(f"Прогноз остатков на {period_num} дней")
            ax.set_xlabel("Филиал (индекс)")
            ax.set_ylabel("Остатки")
            ax.tick_params(axis='x', rotation=45)
            st.pyplot(fig)
            plt.clf()

            # --- Статистика ---
            mean_pred = predictions.mean()
            st.info(f"🧮 Средний прогнозируемый остаток: **{mean_pred:.1f}** тн.")

            # --- Кнопка скачивания ---
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Скачать результаты (CSV)",
                data=csv,
                file_name=f"прогноз_остатков_{period_num}_дней.csv",
                mime="text/csv"
            )

            # --- Кнопка скачивания (Excel) ---
            # Используем BytesIO для сохранения в Excel
            excel_buffer = BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                results_df.to_excel(writer, index=False, sheet_name='Прогноз')
            excel_buffer.seek(0)  # Возвращаем указатель в начало

            st.download_button(
                label="📥 Скачать результаты (XLSX)",
                data=excel_buffer,
                file_name=f"прогноз_остатков_{period_num}_дней.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            
        except Exception as e:
            st.error(f"❌ Ошибка при выполнении предсказания: {e}")

































