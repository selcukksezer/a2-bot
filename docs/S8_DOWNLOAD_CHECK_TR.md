# S8 Download — sadece kontrol kartı (flash yok)

**Amaç:** Odin/Start **yok**. Ekranda **KG STATUS / RMM STATE / OEM LOCK** oku; Magisk yolunun açılıp açılmadığına karar ver; güvenli çık.

## Kopyala-yapıştır (5 satır — ŞİMDİ, flash yok)

```
1) Telefonu kapat.
2) Vol− + Bixby + Power basılı tut → Download/uyarı.
3) Uyarıdaysa Vol+ → “Downloading…” (mavi). Odin AÇMA / Start YOK.
4) Oku: KG STATUS / RMM STATE + OEM LOCK → foto/yaz, bize gönder.
5) Çıkış: Vol− + Power (sisteme dön). Bu ziyarette flash yok.
```

| Ekranda | Anlam | Sonraki adım |
|---------|--------|--------------|
| **Prenormal** | Kapı kapalı | Magisk **YOK**. OEM ON + stok + internet; ~7 gün (`S8_RMM_WAIT_CHECKLIST.md`) |
| **Checking / Completed** + **OEM LOCK: OFF** | Kapı açık | Onayınla `S8_MAGISK_SEQUENCE_SAFE.md` → sonra `tools\after_magisk_hid.bat` |
| **OEM LOCK: ON** | Bootloader kilitli | Önce unlock (bu birimde genelde zaten OFF) |

## Giriş (check-only — ayrıntı)

1. Telefonu **kapat**.
2. **Vol− + Bixby + Power** basılı tut → Download / uyarı ekranı.
3. Uyarıdaysa **Vol+** ile devam et (“Downloading…” / mavi bar).
4. **Odin açma / Start basma.** `adb devices` boş olması normal.

## Ekranda oku (satır satır)

| Satır | Anlam | Ne yap |
|-------|--------|--------|
| **KG STATUS: Prenormal** veya **RMM STATE: Prenormal** | Kapı kapalı | Custom flash **YOK** → `docs/S8_RMM_WAIT_CHECKLIST.md` (~7 gün, OEM ON, stok) |
| **KG STATUS: Checking** veya **Completed** (Prenormal değil) | RMM/KG temiz sayılır | Aşağıdaki OEM LOCK ile birlikte değerlendir |
| **OEM LOCK: ON** | Bootloader kilitli | Unlock gerekir — bu telefonda genelde zaten OFF olmalı |
| **OEM LOCK: OFF** | Bootloader açık | Unlock tekrarlama |

### Magisk yolu (kilidi açan kombinasyon)

**KG Checking veya Completed** + **OEM LOCK: OFF** → Magisk/TWRP yolu **açılır** (yine de kullanıcı onayı + `docs/S8_MAGISK_SEQUENCE_SAFE.md`: önce multidisabler, sonra Magisk).

**Prenormal** varken veya OEM LOCK **ON** iken → flash **yok**.

## Güvenli çıkış

- **Vol− + Power** (veya uzun Power) → Download’dan çık, sisteme dön.
- Bu ziyarette **Start / AP flash yok**.

## Sonra

- Prenormal: bekleme checklist (`docs/S8_RMM_WAIT_CHECKLIST.md`) — ~**7 gün**, **OEM unlocking ON**, stok ROM, **internet** açık bırak. Kısa özet: `tools/check_rmm_ready.md`.
- **Tarih/saat hilesi birincil yol değil** (güvenilmez / riskli); asıl plan bekleme + Download’da Prenormal’in kalkması.
- Kapı açık + onay: güvenli Magisk sırası → root sonrası tek dosya: `tools\after_magisk_hid.bat`
