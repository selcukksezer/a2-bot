# -*- coding: utf-8 -*-
"""Tek "onay" modeli (P3-27, ultra tarama madde 27; K-21/K-22/K-23).

``done`` / ``human_approved`` / ``bulk_approved`` / ``trusted`` /
``hand_label`` bayrakları bugüne kadar her modülde ayrı yazımla okunup
yazılıyordu (aynı kural 28 karar noktasında 13 farklı yazımla — F072;
"insan" tanımı 5 farklı — F015). Bu modül TEK kanonik tanımdır: bayrak
anlamını değiştirmez, yalnızca yazıcı/okuyucu tekilliğini sağlar.

Kural: bu modül repo içinden HİÇBİR şey import etmez (saf dict mantığı),
o yüzden ``label_trust``, ``approved_pool``, ``atomic_review_sync`` ve
dialog hepsi buradan import edebilir — döngü riski yoktur. Yeni bir
onay kuralı yazılacaksa BURAYA yazılır; hiçbir modül kendi kopyasını
tutmaz. P1-10 (``fix/p1-10-persist-donusum`` @ e8abcd5) kendi
``box_has_human_provenance`` kopyasını ``approved_pool``'a bırakmıştı —
o dal v16'da YOK, birleşmede o kopya SİLİNİP bu modüldeki kanonik
fonksiyon import edilecek; iki tanım bir daha yan yana yaşamayacak.

Durum makinesi (kare düzeyi — ``done``):

    PENDING (``done`` None/eksik) --tekli onay--> OK (``done == "ok"``)
    PENDING --toplu onay--> OK (``done == "ok"`` + ``bulk_approved`` izi)
    PENDING --kutu kabulü (weak kalmadı)--> OK
    OK --demote / requeue / incelemeyi aç--> PENDING (``done = None``)
    OK --reddet--> SİLİNDİ (kullanıcı kararı 11 Kasım 05:19: reddet
        kalıcı silmedir; ``done == "no"`` yazıcısı kapsamda yoktur,
        reddedilen kare kuyruğa dönmez, geri yükleme yolu yapılmaz)

    v18/F15 (§7, kullanıcı kararı 11 Kasım 12:47 — "kural kazansın"):
    hand_label / tekli human_approved kutu içeren kare requeue'ya
    GİRMEZ — asla kuyruğa geri dönmez. Toplu onay (bulk_approved) bu
    korumaya girmez; toplu onaylı kare requeue'ya düşebilir. Demote
    edilmiş kutu da girer (demote daha yeni bir insan kararıdır).

    ``done``'un yazıcısı SADECE dialog akışlarıdır (aşağıdaki tablo);
    diskten okuyan tarama ``done``'u belleğe taşımaz — tarama her öğeyi
    ``"done": None`` kurar, diskteki ``done``/``human_approved`` kuyruk
    atlama kararında okunur (``is_frame_closed``), öğeye kopyalanmaz
    (F047). Bu madde bunu değiştirmez; yalnızca kuralı tek yerde toplar.

Kutu düzeyi (``trusted`` / ``weak`` üyeliği + köken bayrakları):

    trusted <=> kutu eğitim TXT'sine girmeye adaydır (floor + köken
        kuralları ``train_txt_boxes``'tadır; bu modül TXT emit kuralını
        değiştirmez).
    hand_label <=> kutu el çizimi / TXT el etiketidir (köken kanıtı).
        Demote ``hand_label``'ı ASLA False yazmaz (P0-8/K-12); yalnızca
        ``trusted``/``weak`` ayrımı + ``demoted_by_user`` değişir.
    human_approved (kutu) <=> kutuya tek tek bakılıp onaylandı.
    bulk_approved (kutu) <=> kutu toplu onayda eşik üstü işaretlendi,
        TERFİ ETMEDİ (B-03). Yalnızca izdir; OKUYUCUSU YOKTUR (aşağıda).

Yazıcı / okuyucu tablosu (v16 ağacı, grep ile çıkarıldı; satırlar bu
dalın HEAD'ine göredir — raporun 03068f8 satırları değil):

    Bayrak (düzey)        Anlamı / kural                YAZICILAR
    --------------------  ----------------------------  ----------------------
    done (kare)           None=bekleyen, "ok"=onaylı.   dialog bulk onay
                          "no" yazılmaz (reddet=sil).  (2 dal), _review_done,
                          _accept_box (weak kalmadıysa),
                          prune fallback, _demote (None),
                          label_trust requeue (None),
                          commit (passthrough, eski
                          meta'dan miras), atomic sync
                          write_synchronized_pair /
                          approve_sample (parametre)
    human_approved        Kare: açık insan onayı       dialog _review_done
    (kare)                (F-0828: done/bulk/çıplak     (True), bulk onay
                          approved SAYILMAZ).          (True — P1-10 kaldırır),
                          _persist türetimi (aşağıda),
                          atomic sync approve_sample /
                          write_synchronized_pair,
                          commit (passthrough), requeue
                          (False), atomic sync Case 2
                          salvage (kurtarılan bayrak)
    human_approved        Kutu: tek tek bakıldı.       canvas drag/release,
    (kutu)                                          _review_done canvas dalı,
                          _accept_box? (hayır —
                          hand_label yazar), atomic
                          sync approve_sample (True),
                          rubberband converter,
                          manual_tag_search, kalibrasyon
                          sihirbazı (el etiketiyle
                          birlikte)
    bulk_approved         Kutu/kare: toplu onay izi.   dialog bulk onay (kutu),
    (kutu; karede         Terfi DEĞİLDİR.             _review_done weak-only
    P1-10 ile gelir)                                  dalı (kutu), atomic sync
                          sanitize (kutu), label_trust
                          _load_state (kutu, kare onayı
                          varsa)
    trusted (kutu)        Kutu trusted listesinde;     tarama (TXT dalı +
                          TXT'ye aday.                reconcile), _accept_box,
                          canvas, _normalize_box_dict
                          (human bayrağı → trusted,
                          demote vetolu), approve_sample,
                          demote (False), bulk onay
                          (False bırakır), requeue
    hand_label (kutu)     Köken: el çizimi / TXT el    tarama TXT dalı,
                          etiketi. Demote değiştirmez. _accept_box, canvas
                          çizim/sürükleme, sınıf
                          değişimi, remap, _review_done
                          canvas dalı, label_trust
                          _load_state reconcile +
                          requeue kutu taşıyıcı,
                          multi_mix, manual_tag_search,
                          kalibrasyon sihirbazı,
                          rubberband converter

    Bayrak (düzey)        OKUYUCULAR (kanonik yol)
    --------------------  ------------------------------------------------
    done (kare)           _pending_count / _pending_train_blockers (yalnızca
                          done sayar — P0-4, persist türetimini varsaymaz),
                          tarama kuyruk atlama (is_frame_closed), sorter
                          is_approved, _review_done guard, bulk onay
                          "zaten done" atlaması, validator hazır-kapısı
    human_approved        Havuz/dondurma kapısı: approved_pool freeze gate +
    (kare)                _review_approval_stems (is_frame_human_approved);
                          atomic sync top_human_approved; tarama kuyruk
                          atlama (is_frame_closed içinde); sorter
    human_approved /      TEK kanonik okuma: box_has_human_provenance
    hand_label (kutu)     (dar tanım). Okuyucular: train_txt_boxes (TXT
                          emit), atomic sync _normalize_box_dict (demote
                          vetolu) + sanitize terfisi, _is_txt_hand_box
                          (yalnızca hand_label + TXT kaynağı — daha dar),
                          koruma predikatları (aşağıda)
    bulk_approved         OKUYUCU YOK — hiçbir kapı/filtre/sayaç okumaz.
                          Yalnızca iz (audit). Bu yüzden toplu onay kutusu
                          hiçbir yoldan eğitime girmez (K-22); karar P1-10'da.
    trusted (kutu)        train_txt_boxes, havuz add_approved_sample,
                          sync emit, _pool_add_approved, prune, _reject_box,
                          web önizleme, sayaçlar
    hand_label (kutu)     Koruma predikatları: _is_hand_label_protected_item
    (koruma)              (TXT kutu veya TXT-only disk), _is_txt_hand_box,
                          _rejected_stem_is_hand_protected, _is_requeued_hand_frame,
                          _stem_review_has_hand_boxes, _is_reject_protected_box

    AYRI TANIMLAR — birleştirilmedi (anlam farklı, davranış korunur):
    * GENİŞ koruma (``is_protected_box``, label_trust): hand_label /
      human_approved / ÇIPLAK approved / confirmed / gather seed /
      source içinde "hand/manual/human". Budama/temizlik korumasında
      kullanılır; dar tanımdan geniştir. Dar + genişin tek tanıma
      inmesi ANLAM kararı ister (çıplak approved sayılsın mı?) — bu
      madde vermez; açık nokta olarak raporda durur.
    * Kuyruk filtresi (``review_queue_sorter.is_approved``): done /
      human_approved'u "true" string'i ve meta alanı dahil toleranslı
      okur — yalnızca kuyruk filtreleme içindir, kapı değildir.
    * ``demoted_by_user``: demote kararı; koruma okuyucularında köken
      kanıtı olarak sayılır (P0-8), onay bayrağı değildir.
"""

from __future__ import annotations

from typing import Any, Dict, List

__all__ = [
    "box_has_human_provenance",
    "box_is_never_requeue",
    "is_frame_human_approved",
    "is_frame_done_ok",
    "is_frame_closed",
    "frame_has_provenance_box",
    "derive_persist_human_approved",
    "is_allowed_frame_transition",
    "FRAME_TRANSITIONS",
    "APPROVAL_FLAG_TABLE",
]

# ---------------------------------------------------------------------------
# Kanonik predikatlar — tek tanım, tek ev. Başka modül kendi kopyasını
# yazmaz; import eder. Kural değişirse SADECE burası değişir.
# ---------------------------------------------------------------------------


def box_has_human_provenance(box: Any) -> bool:
    """Kutu düzeyinde gerçek insan onayı var mı? (DAR tanım, E-04.)

    Yalnızca ``hand_label is True`` (el çizimi / TXT el etiketi) veya
    per-box ``human_approved is True`` (kutuya tek tek bakılıp onaylandı)
    sayılır. Kare düzeyi bayraklar (``done == "ok"``, ``bulk_approved``,
    çıplak ``approved``/``confirmed``) sayılmaz: toplu onay kutuya tek
    tek bakmadı.

    Bu, ``label_trust._is_explicitly_human_approved`` ile AYNI kuraldır —
    o fonksiyon artık bu fonksiyona delege eder (uyumluluk için adı
    korunur). P1-10'un ``approved_pool``'a bıraktığı
    ``box_has_human_provenance`` kopyası da bu tanıma birleşir: birleşmede
    o kopya silinip burası import edilecek.
    """
    if not isinstance(box, dict):
        return False
    return bool(box.get("hand_label") is True or box.get("human_approved") is True)


def box_is_never_requeue(box: Any) -> bool:
    """Kutu requeue'ya asla geri dönmez mi? (v18/F15 §7 kullanıcı kararı.)

    Yalnızca DAR insan kökeni (``box_has_human_provenance``: el çizimi /
    TXT el etiketi ``hand_label`` veya kutuya TEK TEK bakılıp verilen
    ``human_approved``) sayılır. Toplu onay izi (``bulk_approved``),
    çıplak ``approved``/``confirmed`` ve şablon (seed) izi sayılmaz —
    bu kutular requeue'ya düşebilir; toplu onaylı, tekli onaylı sayılmaz.

    Demote vetosu: ``demoted_by_user is True`` daha yeni, açık bir insan
    kararıdır ("şüpheli") — demote edilmiş kutu yeniden incelenebilmelidir,
    koruma kalkar (aksi halde demote edilen kutu kuyruğa bir daha hiç
    dönemez, şüphe hükmü denetlenemez olur). Demote zaten kutunun
    ``human_approved``'unu temizler (P0-8/K-12); ``hand_label`` durur ama
    veto bu predikatta açıkça okunur.
    """
    if not isinstance(box, dict):
        return False
    if box.get("demoted_by_user") is True:
        return False
    return box_has_human_provenance(box)


def is_frame_human_approved(payload: Any) -> bool:
    """Kare düzeyi açık insan onayı var mı? (F-0828, KATİ kural.)

    Yalnızca ``human_approved is True`` sayılır. ``done == "ok"``,
    ``bulk_approved`` ve çıplak ``approved`` pipeline bayrağıdır, insan
    onayı değildir ve tek başına hiçbir kapıyı açmaz. Havuz/dondurma
    kapısı (``approved_pool``) ve ``atomic_review_sync`` bu fonksiyonu
    okur.
    """
    if not isinstance(payload, dict):
        return False
    return bool(payload.get("human_approved") is True)


def is_frame_done_ok(payload: Any) -> bool:
    """Kare ``done == "ok"`` mu? (Kuyruk durumu; onay kanıtı DEĞİLDİR.)"""
    if not isinstance(payload, dict):
        return False
    return bool(payload.get("done") == "ok")


def is_frame_closed(payload: Any) -> bool:
    """Kare kuyruk için kapalı mı? (``done == "ok"`` VEYA açık onay.)

    Tarama kuyruk atlaması + ``_load_state`` bulk-işaret kuralının
    kapsayıcı okumasıdır: kuyruktan çıkmış kareye bir daha bakılmaz.
    Bu, insan onayı iddiası DEĞİLDİR — kapılar (havuz/dondurma) bunu
    okumaz, ``is_frame_human_approved`` okur. İki kuralın tek fonksiyonda
    karışmaması için ayrı tutulur.
    """
    if not isinstance(payload, dict):
        return False
    return bool(payload.get("done") == "ok" or payload.get("human_approved") is True)


def frame_has_provenance_box(payload: Any) -> bool:
    """Karenin trusted listesinde köken kanıtlı en az bir kutu var mı?

    Havuz kapısının P1-10 sonrası biçimi (kare bayrağı + en az bir kutu
    kanıtı) bu yardımcıya dayanır; bu madde kapı anlamını değiştirmez,
    yalnızca okuma yolunu tekilleştirir.
    """
    if not isinstance(payload, dict):
        return False
    trusted = payload.get("trusted") or []
    if not isinstance(trusted, list):
        return False
    return any(box_has_human_provenance(b) for b in trusted)


def derive_persist_human_approved(item: Any) -> bool:
    """``_persist_review_item``'ın kare düzeyi onay türetimi (K-21).

    MEVCUT kural (P1-10 öncesi, davranış korunur): ``done == "ok"``
    tek başına diskte ``human_approved=True`` üretir. Bu, K-21'in
    kaldırılmasını istediği dönüşümdür — ama kaldırma P1-10'un işidir;
    bu madde kuralı yalnızca tek yere toplar ki kaldırma tek satırda
    yapılsın. Bu fonksiyonu başka hiçbir yer çağırmaz; kuralı kopyalama.
    """
    if not isinstance(item, dict):
        return False
    return bool(item.get("human_approved") or item.get("done") == "ok")


# ---------------------------------------------------------------------------
# Durum makinesi — kare düzeyi ``done`` geçişleri.
# ---------------------------------------------------------------------------

#: İzin verilen kare ``done`` geçişleri (kaynak -> {hedefler}).
#: ``None`` = bekleyen (``done`` yok/None). ``"no"`` hedefi yoktur:
#: reddet kalıcı silmedir, durum yazılmaz (11 Kasım 05:19 kullanıcı kararı).
FRAME_TRANSITIONS: Dict[Any, frozenset] = {
    None: frozenset({None, "ok"}),
    "ok": frozenset({None, "ok"}),
}


def is_allowed_frame_transition(old_done: Any, new_done: Any) -> bool:
    """``done`` geçişi durum makinesine uygun mu? (Saf denetim; kapı değil.)

    ``done`` yalnızca ``None`` <-> ``"ok"`` arasında gidip gelir; başka
    değer (ör. ``"no"``, ``True``) makine dışıdır ve yazılmamalıdır.
    Davranış değiştirmez — yalnızca gelecekteki yazıcıların denetimi için
    tek kuraldır.
    """
    old = old_done if old_done in (None, "ok") else "?"
    new = new_done if new_done in (None, "ok") else "?"
    return new in FRAME_TRANSITIONS.get(old, frozenset())


# ---------------------------------------------------------------------------
# Kod-içi yazıcı/okuyucu tablosu (docstring tablosunun makine-okunur izdüşümü).
# ---------------------------------------------------------------------------

APPROVAL_FLAG_TABLE: List[Dict[str, str]] = [
    {
        "flag": "done",
        "level": "kare",
        "meaning": "None=bekleyen, 'ok'=onaylı; 'no' yazılmaz (reddet=sil)",
        "writers": "dialog: bulk onay (2 dal), _review_done, _accept_box, prune fallback, _demote (None); label_trust requeue (None) + commit passthrough; atomic sync write/approve",
        "readers": "_pending_count, _pending_train_blockers, tarama kuyruk atlama (is_frame_closed), sorter.is_approved, validator hazır-kapısı",
    },
    {
        "flag": "human_approved",
        "level": "kare",
        "meaning": "Açık insan onayı; done/bulk/çıplak approved sayılmaz (F-0828)",
        "writers": "dialog _review_done (True), bulk onay (True; P1-10 kaldırır), _persist türetimi (derive_persist_human_approved), atomic sync approve/write, commit passthrough, requeue (False)",
        "readers": "approved_pool freeze gate + _review_approval_stems (is_frame_human_approved), atomic sync top_human_approved, tarama atlama (is_frame_closed içinde)",
    },
    {
        "flag": "human_approved",
        "level": "kutu",
        "meaning": "Kutuya tek tek bakılıp onaylandı",
        "writers": "canvas drag/release, _review_done canvas dalı, atomic sync approve_sample, rubberband converter, manual_tag_search, kalibrasyon sihirbazı",
        "readers": "box_has_human_provenance (train_txt_boxes, atomic sync normalize/sanitize)",
    },
    {
        "flag": "bulk_approved",
        "level": "kutu (+karede P1-10 ile)",
        "meaning": "Toplu onay izi; terfi değildir (B-03)",
        "writers": "dialog bulk onay + _review_done weak-only, atomic sync sanitize, label_trust _load_state",
        "readers": "YOK — hiçbir kapı/filtre/sayaç okumaz (K-22)",
    },
    {
        "flag": "trusted",
        "level": "kutu (liste üyeliği)",
        "meaning": "Kutu trusted listesinde; TXT/havuz adayı",
        "writers": "tarama, _accept_box, canvas, _normalize_box_dict (demote vetolu), approve_sample, demote (False), bulk onay (False bırakır)",
        "readers": "train_txt_boxes, havuz add_approved_sample, sync emit, _pool_add_approved, prune, _reject_box, web önizleme",
    },
    {
        "flag": "hand_label",
        "level": "kutu (köken)",
        "meaning": "El çizimi / TXT el etiketi; demote değiştirmez (P0-8)",
        "writers": "tarama TXT dalı, _accept_box, canvas, sınıf değişimi, remap, _review_done canvas, label_trust reconcile + requeue, multi_mix, manual_tag_search, kalibrasyon, rubberband",
        "readers": "box_has_human_provenance; koruma predikatları (_is_hand_label_protected_item, _is_txt_hand_box, _rejected_stem_is_hand_protected, _is_requeued_hand_frame, _stem_review_has_hand_boxes)",
    },
]
