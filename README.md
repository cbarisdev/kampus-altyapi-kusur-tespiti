# Kampüs Altyapı Kusur Tespit Sistemi

YOLOv8 tabanlı web uygulaması. Kampüs içi çukur ve kırık kaldırım tespiti yapar,
hasar puanı hesaplar, geri bildirim alır ve CSV / Excel / PDF raporu üretir.

## Kurulum

```bash
pip install flask ultralytics opencv-python-headless pandas openpyxl reportlab Pillow
```

### PostgreSQL kullanmak istiyorsanız (isteğe bağlı)

```bash
pip install psycopg2-binary
```

Sonra ortam değişkeni tanımlayın:

```bash
# Linux / macOS
export DATABASE_URL=postgresql://kullanici:sifre@localhost:5432/kampusdb

# Windows (CMD)
set DATABASE_URL=postgresql://kullanici:sifre@localhost:5432/kampusdb
```

`DATABASE_URL` tanımlanmazsa sistem otomatik olarak SQLite kullanır (`kampus.db`).

---

## Çalıştırma

```bash
cd kampus_app
python start.py
```

Tarayıcıda → `http://localhost:5000`

---

## Yenilikler

| Özellik | Açıklama |
|---|---|
| **Conf Slider** | Header'daki kaydırıcı ile güven eşiğini 0.05–0.90 arasında ayarlayın |
| **Veritabanı** | Her analiz ve tespit PostgreSQL veya SQLite'a otomatik kaydedilir |
| **Geri Bildirim** | Her kart üzerinde "Yanlış Tespit" ve "Yanlış Durum" butonları; kayıtlar `feedbacks` tablosuna yazılır |
| **Geçmiş Panel** | Sağ alttaki "Geçmiş Oturumlar" butonu ile önceki analizlere bakın |
| **API** | `GET /history` → oturum geçmişi &nbsp;│&nbsp; `GET /feedbacks` → tüm geri bildirimler |

## Parametre Referansı

| Parametre | Varsayılan | Açıklama |
|---|---|---|
| Güven eşiği | 0.25 | Slider ile değiştirilebilir |
| Sınıf ağırlıkları | 1.0 / 1.0 | `app.py` içinde `CLASS_WEIGHTS` |
| Hasar formülü | Σ(conf² × ağırlık) × 10 | max 10 |
| Durum: Sağlam | puan = 0 | |
| Durum: Orta | 0 < puan < 5 | |
| Durum: Kritik | puan ≥ 5 | |
