# calorie-prediction
A machine learning project for predicting calories burned during workouts using Random Forest Regression and statistical analysis.

---

## 🧠 Regression Exercise – Calorie Consumption Prediction

### 📌 Problem Statement

This project involves a dataset named `calorie.csv`, which contains information about individuals' physical characteristics, fitness indicators, and workout-related metrics. The primary goal is to **predict the number of calories burned** by individuals during various workout sessions using regression techniques.



### 🧾 Dataset Features:

- `Age`: Age of the individual  
- `Gender`: Gender  
- `Weight (kg)`: Body weight in kilograms  
- `Height (m)`: Height in meters  
- `Max_BPM`: Maximum heart rate during workout  
- `Avg_BPM`: Average heart rate  
- `Resting_BPM`: Resting heart rate  
- `Session_Duration (hours)`: Duration of the workout session (in hours)  
- `Calories_Burned`: Target variable – Calories burned during workout  
- `Workout_Type`: Type of exercise performed  
- `Fat_Percentage`: Body fat percentage  
- `Water_Intake (liters)`: Water consumption (in liters)  
- `Workout_Frequency (days/week)`: Weekly workout frequency  
- `Experience_Level`: Fitness experience level (e.g., Beginner, Intermediate, Advanced)  
- `BMI`: Body Mass Index



### 🧹 (A) Data Preprocessing

The dataset is preprocessed by handling missing values, encoding categorical variables, and preparing it for machine learning algorithms.



### 🤖 (B) Model Training & Validation

1. **Train-Test Split**  
   - 70% for training & cross-validation  
   - 30% for final testing  

2. **Cross-Validation**  
   - 5-Fold Cross-Validation is applied to the training data to ensure robust evaluation.

3. **Modeling**  
   - A **Random Forest Regressor** is trained to predict calories burned.

4. **Evaluation Metrics**  
   - **MAE (Mean Absolute Error)**  
   - **R² (R-squared Score)**



### 📊 (C) Feature Analysis

- The feature importances extracted from the trained Random Forest model are visualized using a bar chart.
- The most influential features in calorie prediction are analyzed and discussed.



### 📈 (D) Statistical Summary with `statsmodels`

For deeper statistical insight, we use the `statsmodels` library to fit a linear regression model. The summary includes:
- Regression coefficients  
- p-values  
- Standard errors  
- Additional statistical indicators



### 📝 (E) Final Report Includes:

1. A summary of preprocessing steps  
2. 5-Fold cross-validation results  
3. Feature importance plot and insights  
4. Model performance on test data  
5. Overall conclusion on model effectiveness and key predictors  
6. Statistical summary using `statsmodels`

---

<div dir="rtl">
## 📊 تمرین رگرسیون – پیش‌بینی کالری مصرفی

### 🧩 شرح مسئله

در این پروژه، یک مجموعه‌داده با نام `calorie.csv` در اختیار دارید که شامل اطلاعاتی درباره ویژگی‌های فیزیکی افراد، شاخص‌های مرتبط با ورزش و سلامت است. هدف نهایی، **پیش‌بینی میزان کالری مصرف‌شده** توسط افراد طی جلسات مختلف تمرینی با استفاده از تکنیک‌های یادگیری ماشین است.



### 📁 ویژگی‌های موجود در داده‌ها

- `Age`: سن  
- `Gender`: جنسیت  
- `Weight (kg)`: وزن (کیلوگرم)  
- `Height (m)`: قد (متر)  
- `Max_BPM`: حداکثر ضربان قلب در طول تمرین  
- `Avg_BPM`: میانگین ضربان قلب  
- `Resting_BPM`: ضربان قلب در حالت استراحت  
- `Session_Duration (hours)`: مدت زمان جلسه تمرینی (به ساعت)  
- `Calories_Burned`: مقدار کالری مصرف‌شده (متغیر هدف)  
- `Workout_Type`: نوع تمرین  
- `Fat_Percentage`: درصد چربی بدن  
- `Water_Intake (liters)`: میزان مصرف آب (لیتر)  
- `Workout_Frequency (days/week)`: تعداد دفعات تمرین در هفته  
- `Experience_Level`: سطح تجربه ورزشی (مبتدی، متوسط، حرفه‌ای)  
- `BMI`: شاخص توده بدنی  



### 🧹 (الف) پیش‌پردازش داده‌ها

در این مرحله، داده‌ها از نظر مقادیر گمشده بررسی و پاک‌سازی شده، داده‌های متنی به عددی تبدیل شده و ساختار آن‌ها برای آموزش مدل آماده می‌شود.



### 🤖 (ب) آموزش مدل و اعتبارسنجی


<p dir="rtl">

۱. <strong>تقسیم داده‌ها</strong>  
&nbsp;&nbsp;&nbsp;&nbsp;- ۷۰٪ برای آموزش و اعتبارسنجی  
&nbsp;&nbsp;&nbsp;&nbsp;- ۳۰٪ برای ارزیابی نهایی مدل (تست)

۲. <strong>اعتبارسنجی متقابل پنج‌بخشی (5-Fold Cross Validation)</strong>  
&nbsp;&nbsp;&nbsp;&nbsp;- برای جلوگیری از بیش‌برازش و افزایش دقت مدل

۳. <strong>مدل‌سازی با الگوریتم Random Forest Regressor</strong>  
&nbsp;&nbsp;&nbsp;&nbsp;- آموزش مدل جهت پیش‌بینی کالری سوزانده‌شده

۴. <strong>ارزیابی عملکرد مدل با معیارهای زیر:</strong>  
&nbsp;&nbsp;&nbsp;&nbsp;- میانگین خطای مطلق (MAE)  
&nbsp;&nbsp;&nbsp;&nbsp;- ضریب تعیین (R²)

</p>




### 📊 (ج) تحلیل ویژگی‌ها (Feature Importance)

- پس از آموزش مدل، میزان اهمیت هر ویژگی استخراج شده و در قالب نمودار میله‌ای نمایش داده می‌شود.
- سپس ویژگی‌هایی که بیشترین تأثیر را بر پیش‌بینی کالری دارند تحلیل می‌شوند.



### 📈 (د) خلاصه آماری مدل با استفاده از کتابخانه `statsmodels`

برای تحلیل دقیق‌تر و آماری مدل، از رگرسیون خطی در کتابخانه `statsmodels` استفاده شده و خلاصه‌ای از مدل شامل موارد زیر ارائه می‌شود:

- ضرایب مدل  
- مقادیر p-value  
- خطای استاندارد ضرایب  
- شاخص‌های آماری دیگر مانند R²، F-statistic و...



### 📝 (هـ) گزارش نهایی

در گزارش نهایی (حداقل ۳ صفحه)، موارد زیر باید گنجانده شوند:

1. خلاصه‌ای از مراحل پیش‌پردازش داده‌ها  
2. نتایج اعتبارسنجی متقاطع ۵ بخشی  
3. نمودار اهمیت ویژگی‌ها و تحلیل آن  
4. عملکرد مدل روی داده‌های تست  
5. نتیجه‌گیری کلی درباره دقت مدل و ویژگی‌های مؤثر  
6. خلاصه آماری مدل با استفاده از کتابخانه `statsmodels`

</p>
