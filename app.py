import streamlit as st
import pandas as pd
import numpy as np
import requests
from geopy.geocoders import Nominatim
from datetime import date, timedelta
import plotly.express as px
import plotly.graph_objects as go
import os

# ML бібліотеки
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.feature_selection import SelectFromModel

# Налаштування сторінки Streamlit
st.set_page_config(page_title="Прогноз опадів ML", page_icon="☔", layout="wide")

# ДОПОМІЖНІ ФУНКЦІЇ

@st.cache_data
def get_coordinates(city_name):
    """Отримання координат за назвою міста за допомогою Geopy"""
    try:
        geolocator = Nominatim(user_agent="weather_ml_app_ua")
        location = geolocator.geocode(city_name)
        if location:
            return location.latitude, location.longitude
        return None, None
    except Exception as e:
        st.error(f"Помилка геокодування: {e}")
        return None, None

@st.cache_data
def fetch_weather_data(lat, lon, start_date, end_date, is_forecast=False):
    """Завантаження даних з Open-Meteo та перевірка на пропуски"""
    if is_forecast:
        url = "https://api.open-meteo.com/v1/forecast"
    else:
        url = "https://archive-api.open-meteo.com/v1/archive"
        
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": [
            "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
            "apparent_temperature_max", "apparent_temperature_min",
            "wind_speed_10m_max", "wind_gusts_10m_max", "wind_direction_10m_dominant",
            "shortwave_radiation_sum", "sunshine_duration", "daylight_duration",
            "et0_fao_evapotranspiration", "precipitation_sum"
        ],
        "timezone": "auto"
    }
    
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        if 'daily' in data:
            df = pd.DataFrame(data['daily'])
            df['time'] = pd.to_datetime(df['time'])
            
            # Перевірка та обробка пропусків
            missing_count = df.isnull().sum().sum()
            if missing_count > 0:
                cols_to_interpolate = df.columns.drop('time')
                df[cols_to_interpolate] = df[cols_to_interpolate].apply(pd.to_numeric, errors='coerce')
                df[cols_to_interpolate] = df[cols_to_interpolate].interpolate(method='linear').ffill().bfill()
                
            return df, missing_count
    return None, 0

def engineer_features(df):
    """Генерація додаткових часових ознак (Feature Engineering)"""
    df_engineered = df.copy()
    df_engineered['month'] = df_engineered['time'].dt.month
    df_engineered['day_of_year'] = df_engineered['time'].dt.dayofyear
    return df_engineered


# ДЛЯ ІНТЕРФЕЙСУ

st.title("☔ Сервіс для прогнозування опадів")
st.markdown("Проєкт виконав: Гуленко Назар, група ІДС-501.")

# Збереження стану сесії
if 'current_step' not in st.session_state:
    st.session_state.current_step = "🔴 Завантаження даних (Load data)"
if 'historical_df' not in st.session_state:
    st.session_state.historical_df = None
if 'best_model' not in st.session_state:
    st.session_state.best_model = None
if 'best_model_name' not in st.session_state:
    st.session_state.best_model_name = ""
if 'feature_cols' not in st.session_state:
    st.session_state.feature_cols = None
if 'lat' not in st.session_state:
    st.session_state.lat = None
if 'lon' not in st.session_state:
    st.session_state.lon = None

# Створюємо список кроків
steps = [
    "🔴 Завантаження даних (Load data)", 
    "🔴 Навчання ML моделі (ML model training)", 
    "🔴 Прогноз (Forecast)"
]

# Навігація через бічну панель (Sidebar), прив'язана до стану сесії
step = st.sidebar.radio("Навігація по проєкту:", steps, key="navigation_radio", 
                        index=steps.index(st.session_state.current_step))

# Оновлюємо стан відповідно до вибору користувача вручну
st.session_state.current_step = step


# 1.ОТРИМАННЯ ДАНИХ

if st.session_state.current_step == "🔴 Завантаження даних (Load data)":
    st.header("Отримання історичних метеоданих про погоду")
    
    col1, col2 = st.columns(2)
    with col1:
        location_type = st.radio("Як задати локацію?", ["Назва міста", "Координати"])
        if location_type == "Назва міста":
            city = st.text_input("Введіть назву населеного пункту (напр. Kyiv, Lviv)", "Kyiv")
        else:
            lat_input = st.number_input("Широта (Latitude)", value=50.45)
            lon_input = st.number_input("Довгота (Longitude)", value=30.52)
            
    with col2:
        default_start = date.today() - timedelta(days=365*2)
        default_end = date.today() - timedelta(days=7) 
        start_d = st.date_input("Дата початку", default_start)
        end_d = st.date_input("Дата кінця", default_end)
        
    if st.button("Завантажити дані", type="primary"):
        with st.spinner("Отримання даних..."):
            if location_type == "Назва міста":
                lat, lon = get_coordinates(city)
                if lat is None:
                    st.error("Не вдалося знайти координати міста. Спробуйте ввести координати вручну.")
                    st.stop()
            else:
                lat, lon = lat_input, lon_input
            
            st.session_state.lat = lat
            st.session_state.lon = lon
            
            df, missing_count = fetch_weather_data(lat, lon, start_d, end_d, is_forecast=False)
            
            if df is not None:
                st.session_state.historical_df = df
                
                # Збереження у CSV форматі
                file_name = 'weather_daily.csv'
                df.to_csv(file_name, index=False)
                
                # Повідомляємо про успіх
                st.success(f"✅ Успішно завантажено {len(df)} записів та збережено у CSV!")
                
                # ЗМІНА СТАНУ ДЛЯ АВТОПЕРЕХОДУ
                st.session_state.current_step = "🔴 Навчання ML моделі (ML model training)"
                # Перезавантажуємо сторінку, щоб застосувати новий крок
                st.rerun()
            else:
                st.error("Помилка при завантаженні даних")

    if st.session_state.historical_df is not None:
        st.dataframe(st.session_state.historical_df.tail())
        csv = st.session_state.historical_df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Завантажити CSV", data=csv, file_name='weather_daily.csv', mime='text/csv')

# 2.МОДЕЛІ ТА КРОС-ВАЛІДАЦІЯ

elif st.session_state.current_step == "🔴 Навчання ML моделі (ML model training)":
    st.header("Навчання, підбір гіперпараметрів та оцінка")
    
    # Інформаційний блок про архітектуру розбиття
    st.write("### Архітектура валідації для часових рядів (Time-Series Splitting)")
    st.info("""
    Оскільки метеодані є часовим рядом, звичайне випадкове перемішування призведе до **витоку даних з майбутнього**. Тому буде використано хронологічну архітектуру:
    1. **Test Set - 20% вибірки**: Хронологічна частина даних, яка відкладається без перемішування (`shuffle=False`).
    2. **Train Set - перші 80% вибірки**: Передається у GridSearchCV.
    3. **Крос-валідація**: Модель вчиться на минулому і перевіряється на наступному періоді, ніколи не заглядаючи в майбутнє.
    """)
    
    if st.session_state.historical_df is None:
        st.warning("Спочатку завантажте дані на першому кроці.")
    else:
        df = st.session_state.historical_df.copy()
        
        # Формування цільової змінної
        df['target'] = (df['precipitation_sum'] > 0).astype(int)
        cols_to_drop = ['time', 'precipitation_sum', 'target']
        
        # Feature Engineering
        df = engineer_features(df)
        X = df.drop(columns=cols_to_drop)
        y = df['target']
        st.session_state.feature_cols = X.columns.tolist()

        # Використовуємо форму для того, щоб користувач міг запустити навчання, 
        # а результати та кнопка переходу не зникали при натисканні
        if st.button("Запустити TimeSeries CV та навчання", type="primary"):
            with st.spinner("Виконується пошук оптимальних параметрів..."):
                
                # хронологічне розбиття даних та моделі
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
                
                models_and_params = {
                    "Random Forest": {
                        "pipeline": Pipeline([
                            ('imputer', SimpleImputer(strategy='median')),
                            ('scaler', StandardScaler()),
                            ('feature_selection', SelectFromModel(RandomForestClassifier(n_estimators=50, random_state=123))),
                            ('classifier', RandomForestClassifier(class_weight='balanced', random_state=123))
                        ]),
                        "params": {
                            'classifier__n_estimators': [50, 100, 200],
                            'classifier__max_depth': [None, 10],
                            'classifier__min_samples_split': [2, 5]
                        }
                    },
                    "Logistic Regression": {
                        "pipeline": Pipeline([
                            ('imputer', SimpleImputer(strategy='median')),
                            ('scaler', StandardScaler()),
                            ('feature_selection', SelectFromModel(LogisticRegression(penalty='l1', solver='liblinear', random_state=123))),
                            ('classifier', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=123))
                        ]),
                        "params": {
                            'classifier__C': [0.01, 0.1, 1.0, 10.0]
                        }
                    }
                }
                
                results = []
                best_f1 = 0
                best_pipeline = None
                best_name = ""
                best_params_display = {}
                
                # Використання TimeSeriesSplit замість звичайного KFold
                tscv = TimeSeriesSplit(n_splits=5)
                
                # Навчання з крос-валідацією
                for name, config in models_and_params.items():
                    grid_search = GridSearchCV(
                        estimator=config["pipeline"],
                        param_grid=config["params"],
                        cv=tscv,            
                        scoring='f1',       
                        n_jobs=-1           
                    )
                    
                    grid_search.fit(X_train, y_train)
                    current_best_model = grid_search.best_estimator_
                    
                    # Перевірка моделі на відкладеній тестовій вибірці
                    y_pred = current_best_model.predict(X_test)
                    y_proba = current_best_model.predict_proba(X_test)[:, 1]
                    
                    cv_score = grid_search.best_score_ # Результат на етапі валідації
                    acc = accuracy_score(y_test, y_pred)
                    prec = precision_score(y_test, y_pred, zero_division=0)
                    rec = recall_score(y_test, y_pred, zero_division=0)
                    f1 = f1_score(y_test, y_pred, zero_division=0)
                    roc_auc = roc_auc_score(y_test, y_proba)
                    
                    results.append({
                        "Модель": name, 
                        "CV F1-Score": cv_score, 
                        "Test F1-Score": f1,
                        "Test Accuracy": acc, 
                        "Test Precision": prec, 
                        "Test Recall": rec, 
                        "Test ROC-AUC": roc_auc
                    })
                    
                    clean_params = {k.replace('classifier__', ''): v for k, v in grid_search.best_params_.items()}
                    best_params_display[name] = clean_params
                    
                    if f1 > best_f1:
                        best_f1 = f1
                        best_name = name
                        best_pipeline = current_best_model
                        
                # Зберігаємо все необхідне в сесію, щоб воно не зникло після перезапуску
                st.session_state.best_model = best_pipeline
                st.session_state.best_model_name = best_name
                st.session_state.training_results = results
                st.session_state.best_params_display = best_params_display
                st.session_state.y_test = y_test
                st.session_state.best_y_pred = best_pipeline.predict(X_test)

                st.success(f"🎉 Навчання завершено! Найкраща модель: **{best_name}**")
                
                # Дозволяємо відображення кнопки переходу
                st.session_state.current_step = "🔴 Прогноз (Forecast)"
                st.rerun()

        # Поза блоком кнопки перевіряємо, чи є вже навчена модель у сесії, 
        # щоб відобразити результати минулого запуску
        if st.session_state.best_model is not None:
            st.write("### 📊 Метрики на незалежній Test Set (Хронологічні останні 20%)")
            df_results = pd.DataFrame(st.session_state.training_results)
            st.dataframe(df_results.style.highlight_max(
                subset=['CV F1-Score', 'Test F1-Score', 'Test Accuracy', 'Test Precision', 'Test Recall', 'Test ROC-AUC'], 
                color='lightgreen'), 
                use_container_width=True)
            
            st.write("### ⚙️ Знайдені найкращі гіперпараметри")
            for m_name, p in st.session_state.best_params_display.items():
                st.markdown(f"- **{m_name}:** {p}")

            # Матриця помилок
            st.divider()
            st.write(f"### 🧮 Матриця помилок (Confusion Matrix) для {st.session_state.best_model_name}")
            cm = confusion_matrix(st.session_state.y_test, st.session_state.best_y_pred)
            fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale='Blues',
                               labels=dict(x="Прогноз моделі", y="Фактично", color="Кількість"),
                               x=['Без опадів (0)', 'З опадами (1)'],
                               y=['Без опадів (0)', 'З опадами (1)'])
            fig_cm.update_layout(xaxis_title="Прогноз моделі", yaxis_title="Фактично", font=dict(size=14))
            st.plotly_chart(fig_cm, use_container_width=True)

            # Відбір ознак
            st.divider()
            st.write("### 🔍 Відбір ознак (Feature Selection)")
            pipeline = st.session_state.best_model
            selector = pipeline.named_steps['feature_selection']
            classifier = pipeline.named_steps['classifier']
            
            feature_names = np.array(st.session_state.feature_cols)
            selected_mask = selector.get_support()
            selected_features = feature_names[selected_mask]
            dropped_features = feature_names[~selected_mask]
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Всього ознак на вході", len(feature_names))
            c2.metric("Відібрано алгоритмом", len(selected_features))
            c3.metric("Відкинуто", len(dropped_features))
            
            if len(dropped_features) > 0:
                st.info(f"**Відкинуто:** {', '.join(dropped_features)}")
            
            st.write("#### Важливість відібраних ознак для фінальної моделі:")
            if "Random Forest" in st.session_state.best_model_name:
                importances = classifier.feature_importances_
            else:
                importances = np.abs(classifier.coef_[0]) 
                
            feat_imp_df = pd.DataFrame({'Ознака': selected_features, 'Вплив на прогноз': importances}).sort_values(by='Вплив на прогноз', ascending=True)
            fig = px.bar(feat_imp_df, x='Вплив на прогноз', y='Ознака', orientation='h', color='Вплив на прогноз', color_continuous_scale='Viridis')
            st.plotly_chart(fig, use_container_width=True)
            
            # Якщо модель щойно навчилася — показуємо кнопку для переходу 
            if st.session_state.get('just_trained', False):
                st.session_state.just_trained = False # скидаємо прапорець
                st.success("Перенаправлення на сторінку прогнозування...")
                

# 3.ПРОГНОЗУВАННЯ

elif st.session_state.current_step == "🔴 Прогноз (Forecast)":
    st.header("Прогноз опадів на наступні дні")
    
    if st.session_state.best_model is None:
        st.warning("Спочатку навчіть модель у вкладці 2.")
    elif st.session_state.lat is None:
        st.warning("Спочатку завантажте локацію у вкладці 1.")
    else:
        forecast_days = st.slider("Кількість днів для прогнозу", min_value=1, max_value=7, value=7)
        
        if st.button("Отримати прогноз", type="primary"):
            with st.spinner("Отримання прогнозу..."):
                start_f = date.today()
                end_f = start_f + timedelta(days=forecast_days - 1)
                
                df_forecast, missing_f = fetch_weather_data(st.session_state.lat, st.session_state.lon, start_f, end_f, is_forecast=True)
                
                if df_forecast is not None:
                    df_features = engineer_features(df_forecast)
                    X_forecast = df_features[st.session_state.feature_cols]
                    
                    model = st.session_state.best_model
                    predictions = model.predict(X_forecast)
                    probabilities = model.predict_proba(X_forecast)[:, 1]
                    
                    results_display = pd.DataFrame({
                        "Дата": df_forecast['time'].dt.date,
                        "Макс. Темп. (°C)": df_forecast['temperature_2m_max'],
                        "Вітер (км/год)": df_forecast['wind_speed_10m_max'],
                        "Очікуються опади?": ["🌧️ ТАК" if p == 1 else "☀️ НІ" for p in predictions],
                        "Ймовірність опадів": [f"{prob*100:.1f}%" for prob in probabilities]
                    })
                    
                    st.write("### Результат прогнозування")
                    
                    def color_rain(val):
                        if 'ТАК' in str(val):
                            return 'background-color: #d32f2f; color: white;' 
                        elif 'НІ' in str(val):
                            return 'background-color: #388e3c; color: white;'
                        return ''
                    
                    st.dataframe(results_display.style.map(color_rain, subset=['Очікуються опади?']), use_container_width=True)
                    
                    st.write("### Візуалізація прогнозу")
                    fig2 = go.Figure()
                    
                    fig2.add_trace(go.Bar(
                        x=results_display['Дата'], y=probabilities * 100, name="Ймовірність опадів (%)",
                        marker_color=['#1f77b4' if p == 1 else '#b0bec5' for p in predictions],
                        opacity=0.85
                    ))
                    
                    fig2.add_trace(go.Scatter(
                        x=results_display['Дата'], y=results_display['Макс. Темп. (°C)'],
                        mode='lines+markers', name='Температура (°C)', yaxis='y2', 
                        line=dict(color='#d32f2f', width=2.5),
                        marker=dict(size=8, symbol='square', line=dict(color='white', width=1))
                    ))
                    
                    fig2.update_layout(
                        title="Динаміка температури та ймовірності опадів",
                        template="simple_white",
                        yaxis=dict(title="Ймовірність (%)", range=[0, 105], showgrid=True, gridcolor="#a5a4a4"),
                        yaxis2=dict(title="Температура (°C)", overlaying='y', side='right', showgrid=False),
                        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.9)', bordercolor='black', borderwidth=1),
                        hovermode="x unified",
                        margin=dict(l=40, r=40, t=50, b=40)
                    )
                    st.plotly_chart(fig2, use_container_width=True)
                else:
                    st.error("Помилка отримання даних прогнозу")
