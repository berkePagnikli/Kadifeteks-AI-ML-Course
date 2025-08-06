"""
Üretim Hattından Gelen Veri Setini Analiz Etme
(Production Line Dataset Analysis)

Bu proje NumPy ve Pandas kullanarak gerçek üretim verilerini analiz eder:
- Veri okuma ve temizleme
- İstatistiksel analiz
- Trend analizi  
- Görselleştirme hazırlığı
- Performans metrikleri

This project demonstrates advanced data analysis using NumPy and Pandas
for production line dataset analysis and insights generation.
"""

import numpy as np
import pandas as pd
import datetime
import random
from collections import defaultdict

print("=" * 70)
print("ÜRETİM HATTI VERİ ANALİZİ - NUMPY & PANDAS PROJESİ")
print("PRODUCTION LINE DATA ANALYSIS - NUMPY & PANDAS PROJECT")
print("=" * 70)

# =====================================
# PART 1: VERİ SETİ OLUŞTURMA
# =====================================
print("\n1. VERİ SETİ OLUŞTURMA (Dataset Creation)")
print("-" * 50)

def create_production_dataset(num_records=1000):
    """
    Gerçekçi üretim hattı verisi oluşturur
    """
    np.random.seed(42)  # Tekrarlanabilir sonuçlar için
    random.seed(42)
    
    # Tarih aralığı oluştur (son 90 gün)
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=90)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    data = []
    
    for i in range(num_records):
        # Rastgele tarih seç
        production_date = random.choice(date_range)
        
        # Haftasonu etkisi (daha az üretim)
        is_weekend = production_date.weekday() >= 5
        base_efficiency = 0.7 if is_weekend else 0.9
        
        # Vardiya etkisi
        shift = random.choice(['Gündüz', 'Gece', 'Hafta sonu'])
        shift_efficiency = {'Gündüz': 1.0, 'Gece': 0.85, 'Hafta sonu': 0.75}
        
        # Makine performansı (normale yakın dağılım)
        machine_efficiency = np.random.normal(base_efficiency * shift_efficiency[shift], 0.1)
        machine_efficiency = np.clip(machine_efficiency, 0.3, 1.0)
        
        # Üretim metrikleri
        target_production = 100  # Günlük hedef
        actual_production = int(target_production * machine_efficiency + np.random.normal(0, 5))
        actual_production = max(0, actual_production)
        
        # Kalite metrikleri
        base_quality = 95
        quality_score = base_quality + np.random.normal(0, 3) - (5 if is_weekend else 0)
        quality_score = np.clip(quality_score, 70, 100)
        
        # Hata sayısı (kalite ile ters orantılı)
        defect_rate = np.random.poisson(max(0, (100 - quality_score) / 10))
        
        # Enerji tüketimi
        energy_consumption = actual_production * np.random.uniform(1.8, 2.2) + np.random.normal(0, 5)
        energy_consumption = max(0, energy_consumption)
        
        # Makine durma süresi
        downtime = np.random.exponential(2) if random.random() < 0.15 else 0
        downtime = min(downtime, 8)  # Maksimum 8 saat
        
        record = {
            'tarih': production_date,
            'vardiya': shift,
            'makine_id': f"M{random.randint(1, 10):02d}",
            'operatör': random.choice(['Ali', 'Ayşe', 'Mehmet', 'Fatma', 'Can', 'Elif', 'Burak', 'Zeynep']),
            'hedef_üretim': target_production,
            'gerçek_üretim': actual_production,
            'kalite_skoru': round(quality_score, 2),
            'hata_sayısı': defect_rate,
            'enerji_tüketimi': round(energy_consumption, 2),
            'makine_durma_süresi': round(downtime, 2),
            'üretim_hızı': round(actual_production / max(1, 8 - downtime), 2),
            'verimlilik': round((actual_production / target_production) * 100, 2),
            'maliyet': round(energy_consumption * 0.15 + defect_rate * 50 + downtime * 200, 2)
        }
        
        data.append(record)
    
    return pd.DataFrame(data)

# Veri setini oluştur
df = create_production_dataset(1000)
print(f"Oluşturulan veri seti: {df.shape[0]} satır, {df.shape[1]} sütun")
print("\nVeri seti ilk 5 satır:")
print(df.head())

print("\nVeri seti bilgileri:")
print(df.info())

# =====================================
# PART 2: VERİ KEŞFI VE TEMİZLEME
# =====================================
print("\n\n2. VERİ KEŞFİ VE TEMİZLEME (Data Exploration and Cleaning)")
print("-" * 50)

# Temel istatistikler
print("\n2.1 Temel İstatistikler:")
print(df.describe())

# Eksik veri kontrolü
print("\n2.2 Eksik Veri Kontrolü:")
missing_data = df.isnull().sum()
print(missing_data[missing_data > 0])
if missing_data.sum() == 0:
    print("Eksik veri bulunamadı! ✓")

# Aykırı değer tespiti
print("\n2.3 Aykırı Değer Analizi:")
numerical_columns = ['gerçek_üretim', 'kalite_skoru', 'enerji_tüketimi', 'verimlilik']

outliers_info = {}
for col in numerical_columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    outliers_info[col] = len(outliers)
    
    print(f"{col}: {len(outliers)} aykırı değer tespit edildi")

# Veri tipleri optimizasyonu
print("\n2.4 Veri Tipi Optimizasyonu:")
original_memory = df.memory_usage(deep=True).sum()
print(f"Orijinal hafıza kullanımı: {original_memory / 1024:.2f} KB")

# Tarihleri datetime'a çevir
df['tarih'] = pd.to_datetime(df['tarih'])

# Kategorik verileri category tipine çevir
categorical_columns = ['vardiya', 'makine_id', 'operatör']
for col in categorical_columns:
    df[col] = df[col].astype('category')

optimized_memory = df.memory_usage(deep=True).sum()
print(f"Optimize edilmiş hafıza kullanımı: {optimized_memory / 1024:.2f} KB")
print(f"Hafıza tasarrufu: %{((original_memory - optimized_memory) / original_memory * 100):.1f}")

# =====================================
# PART 3: NUMPY İLE İLERİ ANALİZ
# =====================================
print("\n\n3. NUMPY İLE İLERİ ANALİZ (Advanced Analysis with NumPy)")
print("-" * 50)

# NumPy array'lerine dönüştür
production_array = df['gerçek_üretim'].values
quality_array = df['kalite_skoru'].values
efficiency_array = df['verimlilik'].values
energy_array = df['enerji_tüketimi'].values

print("\n3.1 Üretim Performansı Analizi:")
print(f"Ortalama üretim: {np.mean(production_array):.2f}")
print(f"Medyan üretim: {np.median(production_array):.2f}")
print(f"Standart sapma: {np.std(production_array):.2f}")
print(f"Minimum üretim: {np.min(production_array)}")
print(f"Maksimum üretim: {np.max(production_array)}")

# Yüzdelik dilimler
percentiles = np.percentile(production_array, [25, 50, 75, 90, 95])
print(f"Yüzdelik dilimler (25, 50, 75, 90, 95): {percentiles}")

print("\n3.2 Kalite-Üretim Korelasyonu:")
correlation = np.corrcoef(production_array, quality_array)[0, 1]
print(f"Üretim-Kalite korelasyonu: {correlation:.3f}")

# Kalite kategorileri
quality_categories = np.where(quality_array >= 95, 'Yüksek',
                             np.where(quality_array >= 85, 'Orta', 'Düşük'))
unique_categories, counts = np.unique(quality_categories, return_counts=True)
print("\nKalite kategorisi dağılımı:")
for cat, count in zip(unique_categories, counts):
    print(f"  {cat}: {count} ({count/len(quality_array)*100:.1f}%)")

print("\n3.3 Verimlilik Analizi:")
high_efficiency = efficiency_array[efficiency_array >= 90]
low_efficiency = efficiency_array[efficiency_array < 70]

print(f"Yüksek verimlilik (≥90%) gün sayısı: {len(high_efficiency)}")
print(f"Düşük verimlilik (<70%) gün sayısı: {len(low_efficiency)}")
print(f"Ortalama yüksek verimlilik günü üretimi: {np.mean(production_array[efficiency_array >= 90]):.2f}")
print(f"Ortalama düşük verimlilik günü üretimi: {np.mean(production_array[efficiency_array < 70]):.2f}")

# =====================================
# PART 4: PANDAS İLE GROUPBY ANALİZİ
# =====================================
print("\n\n4. PANDAS İLE GROUPBY ANALİZİ (GroupBy Analysis with Pandas)")
print("-" * 50)

print("\n4.1 Vardiya Bazlı Analiz:")
shift_analysis = df.groupby('vardiya').agg({
    'gerçek_üretim': ['mean', 'std', 'sum'],
    'kalite_skoru': ['mean', 'min', 'max'],
    'verimlilik': 'mean',
    'enerji_tüketimi': 'mean',
    'hata_sayısı': 'sum'
}).round(2)

print(shift_analysis)

print("\n4.2 Makine Bazlı Performans:")
machine_performance = df.groupby('makine_id').agg({
    'gerçek_üretim': 'sum',
    'kalite_skoru': 'mean',
    'makine_durma_süresi': 'sum',
    'verimlilik': 'mean',
    'maliyet': 'sum'
}).round(2)

# En iyi ve en kötü makineler
best_machine = machine_performance.loc[machine_performance['verimlilik'].idxmax()]
worst_machine = machine_performance.loc[machine_performance['verimlilik'].idxmin()]

print(f"En verimli makine: {machine_performance['verimlilik'].idxmax()}")
print(f"  Verimlilik: {best_machine['verimlilik']:.2f}%")
print(f"  Toplam üretim: {best_machine['gerçek_üretim']}")
print(f"  Ortalama kalite: {best_machine['kalite_skoru']:.2f}")

print(f"\nEn az verimli makine: {machine_performance['verimlilik'].idxmin()}")
print(f"  Verimlilik: {worst_machine['verimlilik']:.2f}%")
print(f"  Toplam üretim: {worst_machine['gerçek_üretim']}")
print(f"  Ortalama kalite: {worst_machine['kalite_skoru']:.2f}")

print("\n4.3 Operatör Performansı:")
operator_performance = df.groupby('operatör').agg({
    'gerçek_üretim': ['mean', 'sum'],
    'kalite_skoru': 'mean',
    'verimlilik': 'mean',
    'hata_sayısı': 'mean'
}).round(2)

# En iyi operatörü bul
operator_performance.columns = ['_'.join(col).strip() for col in operator_performance.columns]
top_operator = operator_performance.loc[operator_performance['verimlilik_mean'].idxmax()]

print(f"En verimli operatör: {operator_performance['verimlilik_mean'].idxmax()}")
print(f"  Ortalama verimlilik: {top_operator['verimlilik_mean']:.2f}%")
print(f"  Ortalama kalite: {top_operator['kalite_skoru_mean']:.2f}")

# =====================================
# PART 5: ZAMAN SERİSİ ANALİZİ
# =====================================
print("\n\n5. ZAMAN SERİSİ ANALİZİ (Time Series Analysis)")
print("-" * 50)

# Günlük agregasyon
daily_data = df.groupby('tarih').agg({
    'gerçek_üretim': 'sum',
    'kalite_skoru': 'mean',
    'verimlilik': 'mean',
    'enerji_tüketimi': 'sum',
    'maliyet': 'sum'
}).round(2)

print("\n5.1 Günlük Trend Analizi:")
print(f"İlk günkü toplam üretim: {daily_data['gerçek_üretim'].iloc[0]}")
print(f"Son günkü toplam üretim: {daily_data['gerçek_üretim'].iloc[-1]}")

# 7 günlük hareketli ortalama
daily_data['üretim_7gün_ort'] = daily_data['gerçek_üretim'].rolling(window=7).mean()
daily_data['kalite_7gün_ort'] = daily_data['kalite_skoru'].rolling(window=7).mean()

print(f"Son 7 günün ortalama üretimi: {daily_data['üretim_7gün_ort'].iloc[-1]:.2f}")
print(f"Son 7 günün ortalama kalitesi: {daily_data['kalite_7gün_ort'].iloc[-1]:.2f}")

# Hafta içi vs hafta sonu
df['gün_tipi'] = df['tarih'].dt.weekday.apply(lambda x: 'Hafta_sonu' if x >= 5 else 'Hafta_içi')
day_type_analysis = df.groupby('gün_tipi').agg({
    'gerçek_üretim': 'mean',
    'kalite_skoru': 'mean',
    'verimlilik': 'mean'
}).round(2)

print("\n5.2 Hafta İçi vs Hafta Sonu:")
print(day_type_analysis)

# =====================================
# PART 6: PERFORMANS METRİKLERİ
# =====================================
print("\n\n6. PERFORMANS METRİKLERİ (Performance Metrics)")
print("-" * 50)

print("\n6.1 Genel Performans KPI'ları:")

# OEE (Overall Equipment Effectiveness) hesaplama
df['availability'] = (480 - df['makine_durma_süresi'] * 60) / 480  # 8 saatlik vardiya
df['performance'] = df['gerçek_üretim'] / df['hedef_üretim']
df['quality_rate'] = (df['gerçek_üretim'] - df['hata_sayısı']) / df['gerçek_üretim']
df['quality_rate'] = df['quality_rate'].fillna(0).clip(0, 1)
df['oee'] = df['availability'] * df['performance'] * df['quality_rate']

average_oee = df['oee'].mean() * 100
print(f"Ortalama OEE (Overall Equipment Effectiveness): {average_oee:.2f}%")

# Dünya standartları ile karşılaştırma
if average_oee >= 85:
    oee_status = "Dünya standartları (Mükemmel)"
elif average_oee >= 65:
    oee_status = "Kabul edilebilir"
else:
    oee_status = "Geliştirilmeli"

print(f"OEE Değerlendirmesi: {oee_status}")

# Diğer KPI'lar
print(f"Ortalama Kalite Skoru: {df['kalite_skoru'].mean():.2f}")
print(f"Ortalama Verimlilik: {df['verimlilik'].mean():.2f}%")
print(f"Toplam Üretim: {df['gerçek_üretim'].sum():,} birim")
print(f"Toplam Hata: {df['hata_sayısı'].sum()} birim")
print(f"Hata Oranı: {(df['hata_sayısı'].sum() / df['gerçek_üretim'].sum() * 100):.3f}%")

print("\n6.2 Maliyet Analizi:")
total_cost = df['maliyet'].sum()
cost_per_unit = total_cost / df['gerçek_üretim'].sum()
print(f"Toplam Maliyet: {total_cost:,.2f} TL")
print(f"Birim Maliyet: {cost_per_unit:.2f} TL/birim")

# Maliyet dağılımı
energy_cost = (df['enerji_tüketimi'] * 0.15).sum()
defect_cost = (df['hata_sayısı'] * 50).sum()
downtime_cost = (df['makine_durma_süresi'] * 200).sum()

print(f"Enerji Maliyeti: {energy_cost:,.2f} TL ({energy_cost/total_cost*100:.1f}%)")
print(f"Hata Maliyeti: {defect_cost:,.2f} TL ({defect_cost/total_cost*100:.1f}%)")
print(f"Durma Maliyeti: {downtime_cost:,.2f} TL ({downtime_cost/total_cost*100:.1f}%)")

# =====================================
# PART 7: NUMPY İLE İSTATİSTİKSEL TESTLER
# =====================================
print("\n\n7. İSTATİSTİKSEL ANALİZ (Statistical Analysis)")
print("-" * 50)

print("\n7.1 Vardiya Performans Karşılaştırması:")

# Vardiyalar arası üretim farkı
day_shift = df[df['vardiya'] == 'Gündüz']['gerçek_üretim'].values
night_shift = df[df['vardiya'] == 'Gece']['gerçek_üretim'].values

print(f"Gündüz vardiyası ortalama üretim: {np.mean(day_shift):.2f}")
print(f"Gece vardiyası ortalama üretim: {np.mean(night_shift):.2f}")
print(f"Fark: {np.mean(day_shift) - np.mean(night_shift):.2f} birim")

# Güven aralığı hesaplama
def confidence_interval(data, confidence=0.95):
    mean = np.mean(data)
    std_err = np.std(data) / np.sqrt(len(data))
    margin = 1.96 * std_err  # %95 güven aralığı için
    return mean - margin, mean + margin

day_ci = confidence_interval(day_shift)
night_ci = confidence_interval(night_shift)

print(f"Gündüz vardiyası %95 güven aralığı: [{day_ci[0]:.2f}, {day_ci[1]:.2f}]")
print(f"Gece vardiyası %95 güven aralığı: [{night_ci[0]:.2f}, {night_ci[1]:.2f}]")

print("\n7.2 Kalite Kontrolü İstatistikleri:")
# Kontrol limitleri (3-sigma)
quality_mean = np.mean(quality_array)
quality_std = np.std(quality_array)
ucl = quality_mean + 3 * quality_std  # Upper Control Limit
lcl = quality_mean - 3 * quality_std  # Lower Control Limit

print(f"Kalite ortalaması: {quality_mean:.2f}")
print(f"Üst kontrol limiti (UCL): {ucl:.2f}")
print(f"Alt kontrol limiti (LCL): {lcl:.2f}")

# Kontrol dışı noktalar
out_of_control = np.sum((quality_array > ucl) | (quality_array < lcl))
print(f"Kontrol dışı nokta sayısı: {out_of_control} ({out_of_control/len(quality_array)*100:.2f}%)")

# =====================================
# PART 8: GELECEK TAHMİNİ
# =====================================
print("\n\n8. GELECEK TAHMİNİ (Future Predictions)")
print("-" * 50)

print("\n8.1 Basit Trend Tahmini:")
# Son 30 günün verisi ile lineer trend
recent_data = daily_data.tail(30).reset_index()
recent_data['gün_sayısı'] = range(len(recent_data))

# Numpy ile lineer regresyon
X = recent_data['gün_sayısı'].values
y = recent_data['gerçek_üretim'].values

# y = mx + b formatında
n = len(X)
sum_x = np.sum(X)
sum_y = np.sum(y)
sum_xy = np.sum(X * y)
sum_x2 = np.sum(X ** 2)

# Slope (eğim) ve intercept (kesim) hesaplama
slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
intercept = (sum_y - slope * sum_x) / n

print(f"Trend eğimi: {slope:.2f} birim/gün")
if slope > 0:
    print("Üretim artış trendinde ✓")
else:
    print("Üretim azalış trendinde ⚠")

# Gelecek 7 günün tahmini
future_days = 7
future_predictions = []
for i in range(1, future_days + 1):
    prediction = slope * (len(recent_data) + i) + intercept
    future_predictions.append(prediction)

print(f"\nGelecek {future_days} günün tahmini üretimi:")
for i, pred in enumerate(future_predictions, 1):
    print(f"  Gün {i}: {pred:.0f} birim")

total_predicted = sum(future_predictions)
print(f"Toplam tahmini haftalık üretim: {total_predicted:.0f} birim")

# =====================================
# PART 9: ACTIONABLE INSIGHTS
# =====================================
print("\n\n9. EYLEMSEL ÖNGÖRÜLER (Actionable Insights)")
print("-" * 50)

def generate_insights(df):
    insights = []
    
    # Verimlilik analizi
    low_efficiency_threshold = df['verimlilik'].quantile(0.25)
    low_efficiency_count = (df['verimlilik'] < low_efficiency_threshold).sum()
    if low_efficiency_count > len(df) * 0.3:
        insights.append(f"⚠ Düşük verimlilik problemi: {low_efficiency_count} kayıt (%{low_efficiency_count/len(df)*100:.1f})")
    
    # Makine durma süreleri
    high_downtime = df[df['makine_durma_süresi'] > 2]
    if len(high_downtime) > 0:
        worst_machines = high_downtime.groupby('makine_id')['makine_durma_süresi'].sum().nlargest(3)
        insights.append(f"🔧 En fazla duran makineler: {', '.join(worst_machines.index.tolist())}")
    
    # Kalite sorunları
    low_quality_count = (df['kalite_skoru'] < 85).sum()
    if low_quality_count > 0:
        insights.append(f"📉 Düşük kalite uyarısı: {low_quality_count} kayıt kalite standardının altında")
    
    # Enerji verimliliği
    high_energy_per_unit = df['enerji_tüketimi'] / df['gerçek_üretim']
    inefficient_energy = (high_energy_per_unit > high_energy_per_unit.quantile(0.8)).sum()
    if inefficient_energy > 0:
        insights.append(f"⚡ Enerji verimliliği uyarısı: {inefficient_energy} kayıt yüksek enerji tüketimi")
    
    # Operatör performansı
    operator_variance = df.groupby('operatör')['verimlilik'].std()
    inconsistent_operators = operator_variance[operator_variance > 15].index.tolist()
    if inconsistent_operators:
        insights.append(f"👥 Tutarsız performans: {', '.join(inconsistent_operators)} operatörleri")
    
    return insights

insights = generate_insights(df)
print("Tespit Edilen Ana Sorunlar:")
for insight in insights:
    print(f"  {insight}")

print("\n9.2 Önerilen Eylemler:")
recommendations = [
    "1. Düşük verimli makinelerde bakım planı uygulayın",
    "2. En iyi performanslı operatörlerin deneyimlerini paylaştırın", 
    "3. Gece vardiyası için ek kalite kontrol önlemleri alın",
    "4. Enerji tüketimi yüksek olan süreçleri optimize edin",
    "5. Makine duruş sürelerini azaltmak için preventif bakım planı yapın"
]

for rec in recommendations:
    print(f"  {rec}")

# =====================================
# PART 10: RAPOR ÖZET
# =====================================
print("\n\n10. RAPOR ÖZETİ (Report Summary)")
print("-" * 50)

print(f"""
ÜRETIM HATTI PERFORMANS RAPORU
===============================

VERİ SETİ BİLGİLERİ:
• Analiz edilen kayıt sayısı: {len(df):,}
• Tarih aralığı: {df['tarih'].min().strftime('%d.%m.%Y')} - {df['tarih'].max().strftime('%d.%m.%Y')}
• Toplam makine sayısı: {df['makine_id'].nunique()}
• Toplam operatör sayısı: {df['operatör'].nunique()}

ANA PERFORMANS GÖSTERGELERİ:
• Ortalama OEE: %{average_oee:.1f} ({oee_status})
• Toplam üretim: {df['gerçek_üretim'].sum():,} birim
• Ortalama kalite skoru: {df['kalite_skoru'].mean():.1f}/100
• Ortalama verimlilik: %{df['verimlilik'].mean():.1f}
• Toplam maliyet: {total_cost:,.0f} TL
• Birim maliyet: {cost_per_unit:.2f} TL

TREND ANALİZİ:
• Günlük üretim trendi: {slope:.1f} birim/gün {'(Artan)' if slope > 0 else '(Azalan)'}
• Gelecek hafta tahmini: {total_predicted:.0f} birim

KRİTİK BULGULAR:
{chr(10).join([f'• {insight}' for insight in insights]) if insights else '• Kritik sorun tespit edilmedi ✓'}

Bu analiz NumPy ve Pandas kullanılarak gerçekleştirilmiştir.
Kullanılan teknikler:
- Veri keşfi ve temizleme
- İstatistiksel analiz ve korelasyon
- Zaman serisi analizi
- Performans metrikleri (OEE)
- Trend analizi ve tahmin
- Actionable insights üretimi
""")

print("\n" + "=" * 70)
print("VERİ ANALİZİ TAMAMLANDI - ANALYSIS COMPLETED")
print("=" * 70)