import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from arabic_reshaper import arabic_reshaper
from bidi.algorithm import get_display
import os

# 1. إعداد البيئة للغة العربية
plt.rcParams.update({
    'font.family': 'Times New Roman',  # خط يدعم العربية
    'font.size': 12,
    'axes.unicode_minus': False,
})

# 2. دالة معالجة النص العربي
def format_arabic(text):
    try:
        reshaped_text = arabic_reshaper.reshape(text)
        return get_display(reshaped_text)
    except Exception as e:
        print(f"خطأ في معالجة النص العربي: {e}")
        return text

# ----------------------------
# الجزء 1: تحليل العلاقة بين السكان الحضريين والمساحة المبنية
# ----------------------------

def analyze_urban_build_up():
    # تحميل البيانات الديموغرافية والمكانية
    try:
        demographic_file = 'Demographic and Spatial data - Data.csv'
        df_demo = pd.read_csv(demographic_file, encoding='utf-8-sig')
        
        # تحويل النسب المئوية إلى قيم رقمية
        for col in ['Non-Build-Up areas (%)', 'Build-Up areas (%)', 'Urban Percentage %', 'Rural Percentage %']:
            df_demo[col] = df_demo[col].str.rstrip('%').astype(float)
        
        # رسم العلاقة بين النسبة الحضرية والمساحة المبنية
        plt.figure(figsize=(12, 5))
        
        # المخطط الخطي
        plt.subplot(1, 2, 1)
        plt.plot(df_demo['Years'], df_demo['Urban Percentage %'], 
                label=format_arabic('نسبة سكان حضريين'), marker='o', linestyle='-')
        plt.plot(df_demo['Years'], df_demo['Build-Up areas (%)'], 
                label=format_arabic('نسبة مساحة مبنية'), marker='s', linestyle='--')
        plt.xlabel(format_arabic('السنة'))
        plt.ylabel(format_arabic('النسبة المئوية (%)'))
        plt.title(format_arabic('تطور النسبة الحضرية والمساحة المبنية'))
        plt.legend()
        plt.grid(True)
        
        # حساب الانحدار الخطي
        X_build = df_demo['Build-Up areas (%)'].values.reshape(-1, 1)
        y_urban = df_demo['Urban Percentage %'].values
        model_build = LinearRegression()
        model_build.fit(X_build, y_urban)
        y_pred_build = model_build.predict(X_build)
        r2_build = r2_score(y_urban, y_pred_build)
        corr_build, _ = pearsonr(df_demo['Build-Up areas (%)'], df_demo['Urban Percentage %'])
        
        # مخطط التشتت مع خط الانحدار
        plt.subplot(1, 2, 2)
        plt.scatter(X_build, y_urban, color='blue', label=format_arabic('بيانات فعلية'))
        plt.plot(X_build, y_pred_build, color='red', 
                label=format_arabic(f'خط الانحدار (R² = {r2_build:.2f}'))
        plt.xlabel(format_arabic('المساحة المبنية (%)'))
        plt.ylabel(format_arabic('النسبة الحضرية (%)'))
        plt.title(format_arabic('العلاقة بين المساحة المبنية والنسبة الحضرية'))
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig("Urban_BuildUp_Analysis_AR.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        print('\n' + format_arabic('📊 تحليل العلاقة بين السكان الحضريين والمساحة المبنية:'))
        print(format_arabic(f'معامل الارتباط (r): {corr_build:.4f}'))
        print(format_arabic(f'معامل التحديد (R²): {r2_build:.4f}'))
        print(format_arabic(f'معادلة الانحدار: النسبة_الحضرية = {model_build.coef_[0]:.4f} * المساحة_المبنية + {model_build.intercept_:.4f}'))
    
    except Exception as e:
        print(format_arabic(f'حدث خطأ في تحليل البيانات: {str(e)}'))

# ----------------------------
# الجزء 2: تحليل NDVI وNDBI
# ----------------------------

def analyze_ndbi_ndvi():
    try:
        # تهيئة المسارات والمتغيرات
        ndbi_folder = "ndbi_images"
        ndvi_folder = "ndvi_images"
        available_years = [2000, 2004, 2006, 2011, 2014, 2024]
        valid_years = []
        ndbi_values, ndvi_values = [], []
        
        # دالة لاستخراج متوسط شدة الصورة
        def extract_intensity(image_path):
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            return np.mean(image) if image is not None else None
        
        # معالجة الصور
        for year in available_years:
            ndbi_path = os.path.join(ndbi_folder, f"NDBI_{year}.png")
            ndvi_path = os.path.join(ndvi_folder, f"NDVI_{year}.png")
            
            if os.path.exists(ndbi_path) and os.path.exists(ndvi_path):
                ndbi_avg = extract_intensity(ndbi_path)
                ndvi_avg = extract_intensity(ndvi_path)
                if ndbi_avg is not None and ndvi_avg is not None:
                    valid_years.append(year)
                    ndbi_values.append(ndbi_avg)
                    ndvi_values.append(ndvi_avg)
        
        if not valid_years:
            print(format_arabic("لا توجد بيانات كافية للتحليل"))
            return
        
        # عرض البيانات المستخرجة
        print("\n" + format_arabic("📋 البيانات المستخرجة:"))
        print(format_arabic("السنة | NDBI | NDVI"))
        for i in range(len(valid_years)):
            print(f"{valid_years[i]} | {ndbi_values[i]:.2f} | {ndvi_values[i]:.2f}")
        
        # رسم المخططات
        plt.figure(figsize=(12, 5))
        
        # المخطط الخطي
        plt.subplot(1, 2, 1)
        plt.plot(valid_years, ndbi_values, marker='o', linestyle='-', 
                label=format_arabic('NDBI'))
        plt.plot(valid_years, ndvi_values, marker='s', linestyle='--', 
                label=format_arabic('NDVI'))
        plt.xlabel(format_arabic('السنة'))
        plt.ylabel(format_arabic('الشدة (مقياس الرمادي)'))
        plt.title(format_arabic('تطور مؤشري NDBI وNDVI عبر الزمن'))
        plt.legend()
        plt.grid(True)
        
        # حساب الانحدار الخطي
        X_ndbi = np.array(ndbi_values).reshape(-1, 1)
        y_ndvi = np.array(ndvi_values)
        model_nd = LinearRegression()
        model_nd.fit(X_ndbi, y_ndvi)
        y_pred_nd = model_nd.predict(X_ndbi)
        r2_nd = r2_score(y_ndvi, y_pred_nd)
        corr_nd, _ = pearsonr(ndbi_values, ndvi_values)
        
        # مخطط التشتت مع خط الانحدار
        plt.subplot(1, 2, 2)
        plt.scatter(X_ndbi, y_ndvi, color='green', label=format_arabic('بيانات فعلية'))
        plt.plot(X_ndbi, y_pred_nd, color='orange', 
                label=format_arabic(f'خط الانحدار (R² = {r2_nd:.2f}'))
        plt.xlabel(format_arabic('قيم NDBI'))
        plt.ylabel(format_arabic('قيم NDVI'))
        plt.title(format_arabic('العلاقة بين NDBI وNDVI'))
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig("NDBI_NDVI_Analysis_AR.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        print('\n' + format_arabic('📊 تحليل العلاقة بين NDBI وNDVI:'))
        print(format_arabic(f'معامل الارتباط (r): {corr_nd:.4f}'))
        print(format_arabic(f'معامل التحديد (R²): {r2_nd:.4f}'))
        print(format_arabic(f'معادلة الانحدار: NDVI = {model_nd.coef_[0]:.4f} * NDBI + {model_nd.intercept_:.4f}'))
    
    except Exception as e:
        print(format_arabic(f'حدث خطأ في تحليل الصور: {str(e)}'))

# ----------------------------
# تنفيذ التحليلات
# ----------------------------

if __name__ == "__main__":
    print("="*50)
    print(format_arabic("تحليل النمو الحضري وانعكاساته البيئية"))
    print("="*50)
    
    analyze_urban_build_up()
    analyze_ndbi_ndvi()
