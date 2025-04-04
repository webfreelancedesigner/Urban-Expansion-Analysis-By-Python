# Urban-Expansion-Analysis-By-Python
تحليل النمو الحضري وانعكاساته البيئية باستعمال بايثون
# تحليل النمو الحضري وانعكاساته البيئية

مشروع لتحليل العلاقة بين التوسع الحضري ومؤشرات جودة البيئة باستخدام Python وبيانات الاستشعار عن بعد.

## 📌 المميزات

- تحليل العلاقة بين المساحة المبنية والكثافة السكانية
- حساب مؤشرات NDVI (الغطاء النباتي) وNDBI (المناطق المبنية)
- إنشاء مخططات بيانية تدعم اللغة العربية
- حساب الإحصائيات الأساسية (الارتباط، الانحدار الخطي)

## 📦 المتطلبات

- Python 3.7+
- المكتبات المطلوبة (سيتم تثبيتها تلقائياً):
- pandas, matplotlib, numpy, scikit-learn, scipy, opencv-python, arabic-reshaper, python-bidi


## 🛠️ الإعداد

1. استنساخ المستودع:
 ```bash
 git clone https://github.com/yourusername/urban-growth-analysis.git
 cd urban-growth-analysis
إنشاء بيئة افتراضية (اختياري):

python -m venv venv
source venv/bin/activate  # لنظام Linux/Mac
venv\Scripts\activate    # لنظام Windows

pip install -r requirements.txt

🏃‍♂️ كيفية التشغيل
تأكد من وجود ملفات البيانات في المسارات الصحيحة:

data/Demographic_and_Spatial_data.csv

data/ndbi_images/

data/ndvi_images/

تشغيل البرنامج الرئيسي:

python main.py

سيتولى البرنامج:

تحليل البيانات

إنشاء المخططات البيانية

حفظ النتائج في مجلد output/

urban-growth-analysis/
├── data/                   # ملفات البيانات
│   ├── Demographic_and_Spatial_data.csv
│   ├── ndbi_images/        # صور NDBI
│   └── ndvi_images/        # صور NDVI
├── output/                 # المخرجات (تولد تلقائياً)
│   ├── NDBI_NDVI_Analysis_AR.png
│   └── Urban_BuildUp_Analysis_AR.png
├── src/
│   ├── main.py             # البرنامج الرئيسي
│   └── analysis_functions.py # دوال التحليل
├── requirements.txt        # متطلبات المشروع
└── README.md              # هذا الملف

📊 أمثلة المخرجات
Urban Growth Analysis
NDBI/NDVI Analysis
