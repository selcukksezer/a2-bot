# Gatherion A2 - Private Pixel Bot (Autonomous Gathering & Navigation)

AION 2 için bellek enjeksiyonu yapmayan, tamamen görüntü işleme (OpenCV) ve sanal donanım girdileriyle çalışan profesyonel ekran otomasyonu (Pixel Bot).

---

## 🌟 Temel Özellikler

1. **Belleğe (RAM) Dokunmaz / Anti-Cheat Dostu:**
   - Oyun dosyalarını veya belleği okumaz, modifiye etmez.
   - Ekranı `mss` ile 30-60 FPS hızında yakalar, şablon eşleştirme (`cv2.matchTemplate`) ile karar verir.
   - `PyDirectInput` scancode'ları ile insan benzeri mikro gecikmeli tuş basımları ve fare sapmaları uygular.

2. **Oyun Üstü Şeffaf HUD Overlay:**
   - PyQt6 tabanlı, sol üst köşede duran, fareyle sürüklenebilir durum paneli.
   - Anlık mod rozeti (`GATHERING`, `WALKING`, `PAUSED`, `FIGHTING`).
   - Toplanan kaynak sayısı, basılan `F` sayısı ve prompt eşleşme yüzdesi.
   - Canlı log akışı.

3. **F9 Radial / Context Menüsü:**
   - Klavyeden `F9`'a basıldığında doğrudan farenin bulunduğu noktada açılır:
     - `Resume (R)`: Kaldığı yerden devam eder.
     - `Auto record (A)`: Siz gezersiniz, kaynak yanına vardığınızda bot otomatik toplar.
     - `Manual record (N)`: Siz gezersiniz, `F8` basınca düğümü haritaya kaydeder.
     - `Route mode (M)`: Bot haritayı açar, sıradaki kaynağa tıklar, yürür ve toplar.
     - `Settings (S)`: Rota ve toplama ayarlarını açar.
     - `Exit (X)`: Uygulamayı kapatır.

4. **Akıllı Harita Navigasyonu (Route Mode):**
   - 3B karmaşık navigasyon yerine oyunun yerel harita üzerinden yürüme mekanizmasını (`M` tuşu ile harita tıklaması) kullanır.
   - Düğümler toplandıktan sonra **5 dakikalık dinlenme (cooldown)** süresine girer; yalnızca süresi dolmuş düğümler rotaya alınır.
   - 60 saniye içinde ulaşılamayan (takılma vb.) düğümleri otomatik atlar.

5. **Otomatik Savaş Koruması (Fight Back):**
   - Karakter yürürken veya toplarken bir yaratık saldırdığında hedef barını algılar, otomatik saldırı rotasyonuyla yaratığı keser ve ardından rotaya geri döner.

---

## 🚀 Başlangıç ve Kullanım

### 1. Gereksinimler
Tüm bağımlılıklar sisteme kurulmuştur:
```powershell
pip install -r requirements.txt
```

### 2. Botu Başlatma
Oyun açıkken (Borderless Window veya Windowed modunda):
```powershell
python main.py
```

### 3. Kısayollar
* **`F8`**: Karakter bir kaynağın yanındayken basıldığında haritayı saliselik açıp düğümü rotaya kaydeder ve sesli onay verir.
* **`F9`**: Menüyü açar veya botu duraklatır/devam ettirir.
* **`ESC`**: Acil durum durdurma tuşu.
