"""
Üretim Verilerini İşleyen Bir Script (Production Data Processing Script)

Bu proje, bir üretim hattından gelen verileri işleyen kapsamlı bir script'tir.
Temel Python veri yapıları (lists, dicts, sets, tuples) kullanarak:
- Üretim verilerini kaydetme
- Kalite kontrol analizi
- Performans raporu oluşturma
- Hatalı ürünleri tespit etme
"""

import datetime
import random
from collections import defaultdict

# =====================================
# PART 1: VERİ YAPILARI TANIMLAMA
# =====================================

# Üretim hatları (Production lines) - Tuple (değişmez)
PRODUCTION_LINES = ("Hat-A", "Hat-B", "Hat-C", "Hat-D")
print(f"Üretim Hatları: {PRODUCTION_LINES}")

# Ürün tipleri (Product types) - Set (benzersiz değerler)
PRODUCT_TYPES = {"Laptop", "Tablet", "Smartphone", "Desktop"}
print(f"Ürün Tipleri: {PRODUCT_TYPES}")

# Kalite standartları (Quality standards) - Dictionary
QUALITY_STANDARDS = {
    "Laptop": {"min_score": 85, "max_defects": 2},
    "Tablet": {"min_score": 80, "max_defects": 3},
    "Smartphone": {"min_score": 90, "max_defects": 1},
    "Desktop": {"min_score": 82, "max_defects": 2}
}
print(f"Kalite Standartları: {QUALITY_STANDARDS}")

# =====================================
# PART 2: ÜRETİM VERİSİ OLUŞTURMA
# =====================================

def generate_production_data(num_products=50):
    """
    Rastgele üretim verisi oluşturur
    Returns: List of dictionaries containing production data
    """
    production_data = []
    
    for i in range(num_products):
        product = {
            "id": f"PRD-{i+1:04d}",
            "type": random.choice(list(PRODUCT_TYPES)),
            "production_line": random.choice(PRODUCTION_LINES),
            "production_date": datetime.date.today() - datetime.timedelta(days=random.randint(0, 30)),
            "quality_score": random.randint(70, 100),
            "defect_count": random.randint(0, 5),
            "production_time": random.randint(30, 180),  # dakika
            "operator": random.choice(["Ali", "Ayşe", "Mehmet", "Fatma", "Can"])
        }
        production_data.append(product)
    
    return production_data

# Üretim verisi oluştur
production_records = generate_production_data(50)
print(f"Toplam {len(production_records)} adet üretim kaydı oluşturuldu.")
print("\nÖrnek Kayıtlar:")
for i in range(3):
    record = production_records[i]
    print(f"  {record['id']}: {record['type']} - Kalite: {record['quality_score']}")

# =====================================
# PART 3: KALİTE KONTROL ANALİZİ
# =====================================

def quality_control_analysis(data):
    """
    Kalite kontrol analizi yapar
    """
    results = {
        "passed": [],      # Kaliteyi geçen ürünler
        "failed": [],      # Kaliteyi geçemeyen ürünler
        "total_checked": len(data)
    }
    
    for product in data:
        product_type = product["type"]
        standards = QUALITY_STANDARDS[product_type]
        
        # Kalite kontrolü
        quality_check = (
            product["quality_score"] >= standards["min_score"] and
            product["defect_count"] <= standards["max_defects"]
        )
        
        if quality_check:
            results["passed"].append(product)
        else:
            results["failed"].append(product)
    
    return results

# Kalite kontrol analizi yap
qc_results = quality_control_analysis(production_records)

print(f"Toplam Kontrol Edilen: {qc_results['total_checked']}")
print(f"Kaliteyi Geçen: {len(qc_results['passed'])} (%{len(qc_results['passed'])/qc_results['total_checked']*100:.1f})")
print(f"Kaliteyi Geçemeyen: {len(qc_results['failed'])} (%{len(qc_results['failed'])/qc_results['total_checked']*100:.1f})")

print("\nKaliteyi Geçemeyen Ürünler:")
for product in qc_results["failed"][:5]:  # İlk 5 başarısız ürün
    print(f"  {product['id']}: {product['type']} - Skor: {product['quality_score']}, Hata: {product['defect_count']}")

# =====================================
# PART 4: ÜRETİM HATTI ANALİZİ
# =====================================

def production_line_analysis(data):
    """
    Üretim hatlarının performansını analiz eder
    """
    # defaultdict kullanarak hat bazında verileri grupla
    line_stats = defaultdict(lambda: {
        "total_products": 0,
        "total_time": 0,
        "quality_scores": [],
        "defects": [],
        "product_types": set()
    })
    
    for product in data:
        line = product["production_line"]
        line_stats[line]["total_products"] += 1
        line_stats[line]["total_time"] += product["production_time"]
        line_stats[line]["quality_scores"].append(product["quality_score"])
        line_stats[line]["defects"].append(product["defect_count"])
        line_stats[line]["product_types"].add(product["type"])
    
    # İstatistikleri hesapla
    analysis_results = {}
    for line, stats in line_stats.items():
        avg_quality = sum(stats["quality_scores"]) / len(stats["quality_scores"])
        avg_time = stats["total_time"] / stats["total_products"]
        avg_defects = sum(stats["defects"]) / len(stats["defects"])
        
        analysis_results[line] = {
            "total_products": stats["total_products"],
            "avg_quality_score": round(avg_quality, 2),
            "avg_production_time": round(avg_time, 2),
            "avg_defects": round(avg_defects, 2),
            "product_variety": len(stats["product_types"]),
            "product_types": list(stats["product_types"])
        }
    
    return analysis_results

# Üretim hattı analizi
line_analysis = production_line_analysis(production_records)

print("Üretim Hattı Performans Raporu:")
for line, stats in line_analysis.items():
    print(f"\n{line}:")
    print(f"  Toplam Ürün: {stats['total_products']}")
    print(f"  Ortalama Kalite: {stats['avg_quality_score']}")
    print(f"  Ortalama Üretim Süresi: {stats['avg_production_time']} dk")
    print(f"  Ortalama Hata Sayısı: {stats['avg_defects']}")
    print(f"  Ürün Çeşitliliği: {stats['product_variety']} tip")
    print(f"  Üretilen Tipler: {', '.join(stats['product_types'])}")

# =====================================
# PART 5: OPERATÖR PERFORMANSI
# =====================================

def operator_performance_analysis(data):
    """
    Operatörlerin performansını analiz eder
    """
    operator_stats = {}
    
    for product in data:
        operator = product["operator"]
        
        if operator not in operator_stats:
            operator_stats[operator] = {
                "products": [],
                "total_time": 0,
                "quality_scores": [],
                "defect_counts": []
            }
        
        operator_stats[operator]["products"].append(product["id"])
        operator_stats[operator]["total_time"] += product["production_time"]
        operator_stats[operator]["quality_scores"].append(product["quality_score"])
        operator_stats[operator]["defect_counts"].append(product["defect_count"])
    
    # Performans metriklerini hesapla
    performance_report = {}
    for operator, stats in operator_stats.items():
        total_products = len(stats["products"])
        avg_quality = sum(stats["quality_scores"]) / total_products
        avg_time = stats["total_time"] / total_products
        avg_defects = sum(stats["defect_counts"]) / total_products
        
        performance_report[operator] = {
            "total_products": total_products,
            "avg_quality": round(avg_quality, 2),
            "avg_time": round(avg_time, 2),
            "avg_defects": round(avg_defects, 2),
            "efficiency_score": round((avg_quality - avg_defects) / avg_time * 100, 2)
        }
    
    return performance_report

# Operatör performans analizi
operator_performance = operator_performance_analysis(production_records)

print("Operatör Performans Raporu:")
# Operatörleri verimlilik skoruna göre sırala
sorted_operators = sorted(operator_performance.items(), 
                         key=lambda x: x[1]["efficiency_score"], reverse=True)

for operator, stats in sorted_operators:
    print(f"\n{operator}:")
    print(f"  Toplam Ürün: {stats['total_products']}")
    print(f"  Ortalama Kalite: {stats['avg_quality']}")
    print(f"  Ortalama Süre: {stats['avg_time']} dk")
    print(f"  Ortalama Hata: {stats['avg_defects']}")
    print(f"  Verimlilik Skoru: {stats['efficiency_score']}")

# =====================================
# PART 6: ÜRÜN TİPİ ANALİZİ
# =====================================

def product_type_analysis(data):
    """
    Ürün tiplerinin performansını analiz eder
    """
    type_stats = {}
    
    for product in data:
        ptype = product["type"]
        
        if ptype not in type_stats:
            type_stats[ptype] = {
                "count": 0,
                "quality_scores": [],
                "defect_counts": [],
                "production_times": [],
                "production_lines": set()
            }
        
        type_stats[ptype]["count"] += 1
        type_stats[ptype]["quality_scores"].append(product["quality_score"])
        type_stats[ptype]["defect_counts"].append(product["defect_count"])
        type_stats[ptype]["production_times"].append(product["production_time"])
        type_stats[ptype]["production_lines"].add(product["production_line"])
    
    # Analiz sonuçları
    analysis = {}
    for ptype, stats in type_stats.items():
        analysis[ptype] = {
            "total_produced": stats["count"],
            "avg_quality": round(sum(stats["quality_scores"]) / len(stats["quality_scores"]), 2),
            "avg_defects": round(sum(stats["defect_counts"]) / len(stats["defect_counts"]), 2),
            "avg_time": round(sum(stats["production_times"]) / len(stats["production_times"]), 2),
            "production_lines": len(stats["production_lines"]),
            "success_rate": round(len([q for q in stats["quality_scores"] 
                                     if q >= QUALITY_STANDARDS[ptype]["min_score"]]) / len(stats["quality_scores"]) * 100, 2)
        }
    
    return analysis

# Ürün tipi analizi
product_analysis = product_type_analysis(production_records)

print("Ürün Tipi Performans Raporu:")
for ptype, stats in product_analysis.items():
    print(f"\n{ptype}:")
    print(f"  Toplam Üretilen: {stats['total_produced']}")
    print(f"  Ortalama Kalite: {stats['avg_quality']}")
    print(f"  Ortalama Hata: {stats['avg_defects']}")
    print(f"  Ortalama Süre: {stats['avg_time']} dk")
    print(f"  Başarı Oranı: {stats['success_rate']}%")
    print(f"  Üretim Hatları: {stats['production_lines']}")

# =====================================
# PART 7: HAFTALIK RAPOR
# =====================================

def generate_weekly_report(data):
    """
    Haftalık üretim raporu oluşturur
    """
    # Tarihe göre grupla
    daily_production = defaultdict(list)
    
    for product in data:
        day = product["production_date"]
        daily_production[day].append(product)
    
    print("Günlük Üretim Özeti:")
    total_weekly_production = 0
    total_weekly_quality = 0
    daily_summaries = []
    
    # Tarihleri sırala
    sorted_days = sorted(daily_production.keys())
    
    for day in sorted_days[-7:]:  # Son 7 gün
        products = daily_production[day]
        daily_count = len(products)
        daily_avg_quality = sum(p["quality_score"] for p in products) / daily_count
        daily_defects = sum(p["defect_count"] for p in products)
        
        daily_summaries.append({
            "date": day,
            "count": daily_count,
            "avg_quality": round(daily_avg_quality, 2),
            "total_defects": daily_defects
        })
        
        total_weekly_production += daily_count
        total_weekly_quality += daily_avg_quality
        
        print(f"  {day}: {daily_count} ürün, Kalite: {daily_avg_quality:.2f}, Hata: {daily_defects}")
    
    # Haftalık özet
    weekly_avg_quality = total_weekly_quality / len(daily_summaries) if daily_summaries else 0
    
    print(f"\nHaftalık Özet:")
    print(f"  Toplam Üretim: {total_weekly_production} ürün")
    print(f"  Ortalama Günlük Üretim: {total_weekly_production / len(daily_summaries):.1f} ürün")
    print(f"  Ortalama Kalite: {weekly_avg_quality:.2f}")
    
    return daily_summaries

weekly_report = generate_weekly_report(production_records)

# =====================================
# PART 8: HATA TESPİT SİSTEMİ
# =====================================

def detect_production_issues(data):
    """
    Üretimde potansiyel sorunları tespit eder
    """
    issues = {
        "low_quality_products": [],
        "high_defect_products": [],
        "slow_production": [],
        "operator_warnings": [],
        "line_warnings": []
    }
    
    # Genel istatistikler
    avg_quality = sum(p["quality_score"] for p in data) / len(data)
    avg_time = sum(p["production_time"] for p in data) / len(data)
    avg_defects = sum(p["defect_count"] for p in data) / len(data)
    
    # Hata tespiti
    for product in data:
        # Düşük kalite
        if product["quality_score"] < avg_quality - 10:
            issues["low_quality_products"].append(product["id"])
        
        # Yüksek hata sayısı
        if product["defect_count"] > avg_defects + 2:
            issues["high_defect_products"].append(product["id"])
        
        # Yavaş üretim
        if product["production_time"] > avg_time + 30:
            issues["slow_production"].append(product["id"])
    
    # Operatör uyarıları
    operator_quality = defaultdict(list)
    for product in data:
        operator_quality[product["operator"]].append(product["quality_score"])
    
    for operator, scores in operator_quality.items():
        avg_op_quality = sum(scores) / len(scores)
        if avg_op_quality < avg_quality - 5:
            issues["operator_warnings"].append(operator)
    
    # Hat uyarıları
    line_quality = defaultdict(list)
    for product in data:
        line_quality[product["production_line"]].append(product["quality_score"])
    
    for line, scores in line_quality.items():
        avg_line_quality = sum(scores) / len(scores)
        if avg_line_quality < avg_quality - 3:
            issues["line_warnings"].append(line)
    
    return issues

# Hata tespiti
production_issues = detect_production_issues(production_records)

print("Tespit Edilen Sorunlar:")
print(f"  Düşük Kaliteli Ürünler: {len(production_issues['low_quality_products'])}")
if production_issues["low_quality_products"]:
    print(f"    Örnekler: {', '.join(production_issues['low_quality_products'][:5])}")

print(f"  Yüksek Hatalı Ürünler: {len(production_issues['high_defect_products'])}")
if production_issues["high_defect_products"]:
    print(f"    Örnekler: {', '.join(production_issues['high_defect_products'][:5])}")

print(f"  Yavaş Üretilen Ürünler: {len(production_issues['slow_production'])}")
if production_issues["slow_production"]:
    print(f"    Örnekler: {', '.join(production_issues['slow_production'][:5])}")

print(f"  Dikkat Gereken Operatörler: {', '.join(production_issues['operator_warnings'])}")
print(f"  Dikkat Gereken Hatlar: {', '.join(production_issues['line_warnings'])}")