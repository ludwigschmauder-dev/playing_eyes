#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
from pathlib import Path
from dataclasses import dataclass
import pandas as pd
import streamlit as st
from io import StringIO

# --- Bilder ---
from PIL import Image

# ---------------- Grundkonstanten ----------------
CARD_TO_POINTS = {"K": 10, "6": 6, "3": 3}
IMAGES_DIR = Path(__file__).parent / "images"
DATA_DIR   = Path(__file__).parent / "data"

IMG_FILES = {
    "K": ["king.jpg", "king.jpeg", "king.png"],
    "6": ["six.jpg",  "six.jpeg",  "six.png"],
    "3": ["three.jpg","three.jpeg","three.png"],
}

BLUFF_PROBS = {"low":0.70,"medium":0.40,"high":0.10,"none":1.00}
SIGNAL_REF   = {"high":16.0,"medium":12.5,"low":7.5}
BELIEVE_OFFSET = 1.0

SIG_INT_TO_DE = {"low":"tief", "medium":"mittel", "high":"hoch"}
SIG_DE_TO_INT = {"tief":"low", "mittel":"medium", "hoch":"high"}

# ---------------- Utility ----------------
def categorize(points: int) -> str:
    """ Kategorien gemäß deiner Spezifikation. """
    if points == 16: return "high"      # K+6
    if points in (12, 13): return "medium"
    if points in (6, 9): return "low"
    if points == 20: return "none"      # K+K: keine wahre Kategorie -> muss bluffen
    return "none"

def label_category(cat: str) -> str:
    return {"high":"Hoch","medium":"Mittel","low":"Tief","none":"Überspielt"}.get(cat,"??")

def draw_two_cards():
    deck = ["K"]*4 + ["6"]*4 + ["3"]*4
    cards = random.sample(deck, 2)
    return cards, sum(CARD_TO_POINTS[c] for c in cards)

def canon_hand(cards):
    order = {"K":2, "6":1, "3":0}
    a, b = sorted(cards, key=lambda c: order[c], reverse=True)
    return f"{a.lower()},{b.lower()}"

def payoff(truthful, believed, sum_s, sum_r):
    """ Rückgabe: (delta_p1, delta_p2) """
    if truthful and believed:
        if sum_s > sum_r: return (1,-1)
        if sum_s < sum_r: return (-1,1)
        return (0,0)
    if (not truthful) and believed: return (1,-1)
    if truthful and (not believed): return (1,-1)
    return (-1,1)

# ---------------- Bilder laden ----------------
@st.cache_data(show_spinner=False)
def load_card_image(key: str, target_h: int = 180):
    for name in IMG_FILES[key]:
        p = IMAGES_DIR / name
        if p.exists():
            img = Image.open(p)
            # proportional auf Höhe target_h
            w, h = img.size
            if h != target_h:
                r = target_h / float(h)
                img = img.resize((int(w*r), target_h))
            return img
    return None  # Fallback: kein Bild vorhanden

# ---------------- BNE-Policies laden ----------------
@st.cache_data(show_spinner=False)
def load_policies():
    POLICY_P1, POLICY_P2 = {}, {}
    p1_path = DATA_DIR / "Bayes-Nash__gemischt____SP1_Signalpolitik.csv"
    p2_path = DATA_DIR / "Bayes-Nash__gemischt____SP2_Antwortpolitik.csv"
    try:
        df1 = pd.read_csv(p1_path)
        for _, row in df1.iterrows():
            h = str(row.get("SP1 Hand","")).strip().lower()
            if not h: continue
            POLICY_P1[h] = {
                "low": float(row.get("P(Signal=tief)", 0) or 0),
                "medium": float(row.get("P(Signal=mittel)", 0) or 0),
                "high": float(row.get("P(Signal=hoch)", 0) or 0),
            }
    except Exception:
        pass

    try:
        df2 = pd.read_csv(p2_path)
        for _, row in df2.iterrows():
            h = str(row.get("SP2 Hand","")).strip().lower()
            sig = str(row.get("Signal","")).strip().lower()
            if not h or not sig: continue
            POLICY_P2[(h, sig)] = float(row.get("P(glaubt)", 0) or 0)
    except Exception:
        pass

    return POLICY_P1, POLICY_P2

# ---------------- Computerpolitik (BNE + Fallback) ----------------
def bne_signal(cards, POLICY_P1):
    """BNE-Signal (falls vorhanden), sonst Heuristik."""
    h = canon_hand(cards)
    if h in POLICY_P1:
        probs = POLICY_P1[h]
        total = sum(probs.values()) or 1.0
        r = random.random() * total
        acc = 0.0
        for k, v in probs.items():
            acc += v
            if r <= acc:
                sig = k
                truthful = (categorize(sum(CARD_TO_POINTS[c] for c in cards)) == sig)
                return sig, truthful
    # Fallback Heuristik
    own_sum = sum(CARD_TO_POINTS[c] for c in cards)
    true_cat = categorize(own_sum)
    will_bluff = (true_cat == "none") or (random.random() < BLUFF_PROBS[true_cat])
    if not will_bluff:
        return true_cat, True
    options = [s for s in ["low","medium","high"] if s != true_cat]
    return random.choice(options), False

def bne_response(signal, cards, POLICY_P2):
    """BNE-Antwort (falls vorhanden), sonst Heuristik."""
    h = canon_hand(cards)
    key = (h, SIG_INT_TO_DE[signal])
    if key in POLICY_P2:
        return "believe" if random.random() < POLICY_P2[key] else "doubt"
    own_sum = sum(CARD_TO_POINTS[c] for c in cards)
    return "believe" if own_sum >= (SIGNAL_REF[signal] - BELIEVE_OFFSET) else "doubt"

def pc_choose_signal(cards, optimal_prob, POLICY_P1):
    """
    Mit Wahrscheinlichkeit optimal_prob BNE/Heuristik; sonst zufällig (50/50 Wahrheit/Bluff, falls möglich).
    Rückgabe: (signal, truthful, policy_tag)
    """
    if random.random() < optimal_prob:
        sig, truth = bne_signal(cards, POLICY_P1)
        return sig, truth, "optimal"
    true_cat = categorize(sum(CARD_TO_POINTS[c] for c in cards))
    if true_cat == "none":
        options = ["low","medium","high"]
        return random.choice(options), False, "random"
    if random.random() < 0.5:
        return true_cat, True, "random"
    options = [s for s in ["low","medium","high"] if s != true_cat]
    return random.choice(options), False, "random"

def pc_choose_response(signal, cards, optimal_prob, POLICY_P2):
    if random.random() < optimal_prob:
        return bne_response(signal, cards, POLICY_P2), "optimal"
    return ("believe" if random.random() < 0.5 else "doubt"), "random"

# ---------------- Session-State ----------------
def init_state():
    if "round" not in st.session_state:
        st.session_state.round = 0

    # Rollen-Score (für Spiellogik) – bleibt an P1/P2
    if "p1_pts" not in st.session_state:
        st.session_state.p1_pts = 20
    if "p2_pts" not in st.session_state:
        st.session_state.p2_pts = 20

    # Personen-Score (Anzeige) – bleibt an Du/PC
    if "human_pts" not in st.session_state:
        st.session_state.human_pts = 20
    if "pc_pts" not in st.session_state:
        st.session_state.pc_pts = 20

    # Wer ist diese Runde P1 (Signalisierer)? (wechselt jede Runde)
    if "human_is_p1" not in st.session_state:
        st.session_state.human_is_p1 = True

    # Schwierigkeit (Optimal-Wahrscheinlichkeit)
    if "optimal_prob" not in st.session_state:
        st.session_state.optimal_prob = 1.0  # 100%

    # Rundenstate
    keys = ["p1_cards","p2_cards","p1_sum","p2_sum","cur_sig","truth",
            "pc_p1_policy","pc_p2_policy","awaiting_response","finished"]
    for k in keys:
        if k not in st.session_state:
            st.session_state[k] = None if k not in ("awaiting_response","finished") else False

    # Logging: Liste von Dict-Zeilen
    if "logs" not in st.session_state:
        st.session_state.logs = []

# ---------------- Neue Runde ----------------
def new_round():
    st.session_state.round += 1
    # Rolle wechseln (außer bei der allerersten Runde)
    if st.session_state.round > 1:
        st.session_state.human_is_p1 = not st.session_state.human_is_p1

    # Karten austeilen
    st.session_state.p1_cards, st.session_state.p1_sum = draw_two_cards()
    st.session_state.p2_cards, st.session_state.p2_sum = draw_two_cards()

    # Reset Anzeige
    st.session_state.cur_sig = None
    st.session_state.truth = None
    st.session_state.pc_p1_policy = "human" if st.session_state.human_is_p1 else ""
    st.session_state.pc_p2_policy = "" if st.session_state.human_is_p1 else "human"
    st.session_state.awaiting_response = False
    st.session_state.finished = False

# ---------------- Auflösen & Loggen ----------------
def finish_round(believed, POLICY_P1, POLICY_P2):
    ds, dr = payoff(st.session_state.truth, believed,
                    st.session_state.p1_sum, st.session_state.p2_sum)

    # Rollenpunkte (P1/P2)
    st.session_state.p1_pts += ds
    st.session_state.p2_pts += dr

    # Personenpunkte (Du/PC) – korrekt mappen
    if st.session_state.human_is_p1:
        human_delta, pc_delta = ds, dr
    else:
        human_delta, pc_delta = dr, ds
    st.session_state.human_pts += human_delta
    st.session_state.pc_pts    += pc_delta

    # Log-Zeile erstellen
    log_row = {
        "round": st.session_state.round,
        "human_role": "P1" if st.session_state.human_is_p1 else "P2",
        "p1_cards": "/".join(st.session_state.p1_cards),
        "p2_cards": "/".join(st.session_state.p2_cards),
        "p1_sum": st.session_state.p1_sum,
        "p2_sum": st.session_state.p2_sum,
        "p1_category": label_category(categorize(st.session_state.p1_sum)),
        "p2_category": label_category(categorize(st.session_state.p2_sum)),
        "signal": label_category(st.session_state.cur_sig) if st.session_state.cur_sig else "",
        "truthful": 1 if (st.session_state.truth is True) else 0,
        "responder_believed": 1 if believed else 0,
        "pc_p1_policy": st.session_state.pc_p1_policy or "",
        "pc_p2_policy": st.session_state.pc_p2_policy or "",
        "delta_p1": ds, "delta_p2": dr,
        "p1_pts": st.session_state.p1_pts, "p2_pts": st.session_state.p2_pts,
        "human_pts": st.session_state.human_pts, "pc_pts": st.session_state.pc_pts,
        "optimal_prob": round(st.session_state.optimal_prob, 2),
    }
    st.session_state.logs.append(log_row)
    st.session_state.finished = True

# ---------------- Streamlit UI ----------------
def main():
    st.set_page_config(page_title="Playing Eyes – Bluff Game", layout="wide")
    init_state()
    POLICY_P1, POLICY_P2 = load_policies()

    # Sidebar: Schwierigkeit + Kontrolle
    st.sidebar.header("Einstellungen")
    col_diff = st.sidebar.columns([1,2,1])
    if col_diff[0].button("−", help="Schwierigkeit verringern (−25%)"):
        st.session_state.optimal_prob = max(0.0, round(st.session_state.optimal_prob - 0.25, 2))
    col_diff[1].markdown(f"**PC-Genauigkeit:** {int(st.session_state.optimal_prob*100)}%")
    if col_diff[2].button("+", help="Schwierigkeit erhöhen (+25%)"):
        st.session_state.optimal_prob = min(1.0, round(st.session_state.optimal_prob + 0.25, 2))

    if st.sidebar.button("Neues Spiel (Reset)"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        init_state()
        new_round()

    # Header
    c1, c2, c3 = st.columns([1,2,1])
    c1.markdown(f"### Runde: {st.session_state.round}")
    c2.markdown(f"**Diese Runde: Du bist Spieler {'1' if st.session_state.human_is_p1 else '2'}**")
    c3.metric("PC-Genauigkeit", f"{int(st.session_state.optimal_prob*100)}%")

    # Scores (Du vs PC – konstant!)
    s1, s2 = st.columns(2)
    s1.metric("Du", st.session_state.human_pts)
    s2.metric("PC", st.session_state.pc_pts)

    # Erste Runde automatisch starten
    if st.session_state.round == 0:
        new_round()

    # Panels: Links P1 (Signal) / Rechts P2 (Antwort)
    left, right = st.columns(2)

    # --- LINKES PANEL: Spieler 1 (Signal) ---
    with left:
        st.subheader("Spieler 1 (Signal)")
        # Anzeige wahre Kategorie (nur sichtbar, wenn Du in dieser Runde P1 bist)
        p1_cat = label_category(categorize(st.session_state.p1_sum))
        st.caption(
            f"Wahre Kategorie: {p1_cat if (st.session_state.human_is_p1 or st.session_state.finished) else '??'}"
        )

        # Signal-Buttons (nur wenn Mensch P1 & Runde nicht fertig)
        if st.session_state.human_is_p1 and not st.session_state.finished:
            btn_cols = st.columns(3)
            if btn_cols[0].button("Signal: Hoch (nur 16)"):
                st.session_state.cur_sig = "high"
                st.session_state.truth = (categorize(st.session_state.p1_sum) != "none" and
                                          categorize(st.session_state.p1_sum) == "high")
                # PC antwortet
                resp, pol = pc_choose_response("high", st.session_state.p2_cards,
                                               st.session_state.optimal_prob, POLICY_P2)
                st.session_state.pc_p2_policy = pol
                finish_round(resp == "believe", POLICY_P1, POLICY_P2)

            if btn_cols[1].button("Signal: Mittel (12–13)"):
                st.session_state.cur_sig = "medium"
                st.session_state.truth = (categorize(st.session_state.p1_sum) != "none" and
                                          categorize(st.session_state.p1_sum) == "medium")
                resp, pol = pc_choose_response("medium", st.session_state.p2_cards,
                                               st.session_state.optimal_prob, POLICY_P2)
                st.session_state.pc_p2_policy = pol
                finish_round(resp == "believe", POLICY_P1, POLICY_P2)

            if btn_cols[2].button("Signal: Tief (6–9)"):
                st.session_state.cur_sig = "low"
                st.session_state.truth = (categorize(st.session_state.p1_sum) != "none" and
                                          categorize(st.session_state.p1_sum) == "low")
                resp, pol = pc_choose_response("low", st.session_state.p2_cards,
                                               st.session_state.optimal_prob, POLICY_P2)
                st.session_state.pc_p2_policy = pol
                finish_round(resp == "believe", POLICY_P1, POLICY_P2)


            # Karten anzeigen: nach Abschluss IMMER offen; sonst nur, wenn du P1 bist
        reveal_p1 = st.session_state.finished or st.session_state.human_is_p1
        render_cards(st.session_state.p1_cards, reveal=reveal_p1)

        # Gezeigtes Signal (falls vorhanden)
        if st.session_state.cur_sig:
            st.info(f"Signal: **{label_category(st.session_state.cur_sig)}**")

        # Nach Abschluss: Zeige ggf. PC-Entscheidung (wenn PC Responder war)
        if st.session_state.finished and st.session_state.human_is_p1 and st.session_state.logs:
            believed = bool(st.session_state.logs[-1]["responder_believed"])
            st.success(f"PC-Antwort: **{'glaubt' if believed else 'zweifelt'}**")


    # --- RECHTES PANEL: Spieler 2 (Antwort) ---
    with right:
        st.subheader("Spieler 2 (Antwort)")
        p2_cat = label_category(categorize(st.session_state.p2_sum))
        st.caption(
            f"Wahre Kategorie: {p2_cat if ((not st.session_state.human_is_p1) or st.session_state.finished) else '??'}"
        )


        # Wenn PC P1 ist, signalisiert er sofort – Mensch antwortet:
        if (not st.session_state.human_is_p1) and (st.session_state.cur_sig is None):
            sig, truth, pol = pc_choose_signal(st.session_state.p1_cards,
                                               st.session_state.optimal_prob, POLICY_P1)
            st.session_state.cur_sig = sig
            st.session_state.truth = truth
            st.session_state.pc_p1_policy = pol
            st.info(f"Spieler 1 signalisiert: **{label_category(sig)}** – Deine Antwort?")


            # Wenn die Runde fertig ist und der PC P1 war: sein Signal + (Wahr/Bluff) noch mal klar anzeigen
        if st.session_state.finished and (not st.session_state.human_is_p1):
            sig_txt = label_category(st.session_state.cur_sig)
            p1_strat = "Wahr" if st.session_state.truth else "Bluff"
            st.info(f"PC-Signal: **{sig_txt}** ({p1_strat})")

        
        # Antwort-Buttons (nur wenn Mensch P2 & Runde nicht fertig)
        if (not st.session_state.human_is_p1) and (not st.session_state.finished):
            rcols = st.columns(2)
            if rcols[0].button("Glauben"):
                finish_round(True, POLICY_P1, POLICY_P2)
                st.rerun()  # << NEU: sofort neu rendern, damit links P1-Karten aufgedeckt werden
            if rcols[1].button("Zweifeln"):
                finish_round(False, POLICY_P1, POLICY_P2)
                st.rerun()  # << NEU




            # Kartenanzeige: nach Abschluss IMMER offen; sonst nur, wenn du P2 bist
        reveal_p2 = st.session_state.finished or (not st.session_state.human_is_p1)
        render_cards(st.session_state.p2_cards, reveal=reveal_p2)

        # Nach Abschluss: Zeige die Entscheidung der Person auf der rechten Seite
        if st.session_state.finished and st.session_state.logs:
            believed = bool(st.session_state.logs[-1]["responder_believed"])
            # Wenn du P2 warst, hast DU geantwortet -> zeige deine Antwort hier
            if not st.session_state.human_is_p1:
                st.success(f"Deine Antwort: **{'glauben' if believed else 'zweifeln'}**")
            else:
                # Du warst P1 -> PC war P2, seine Antwort siehst du links schon;
                # hier zusätzlich: (optional) nur ein Hinweis weglassen oder doppelt NICHT nötig
                pass


    st.markdown("---")

    # ---------------- Rundenergebnis ----------------
    if st.session_state.finished:
        sig_txt = label_category(st.session_state.cur_sig)
        p1_strat = "Wahr" if st.session_state.truth else "Bluff"

        # Wer hat geglaubt/gezweifelt? -> hängt davon ab, wer P2 ist.
        last_log = st.session_state.logs[-1]
        believed = bool(last_log["responder_believed"])
        p2_strat = "glaubt" if believed else "zweifelt"

        ds, dr = last_log["delta_p1"], last_log["delta_p2"]
        if ds > dr: round_winner = "Spieler 1"
        elif dr > ds: round_winner = "Spieler 2"
        else: round_winner = "Unentschieden"

        st.success(
            f"**Sieger:** {round_winner} | ΔP1: {ds:+d}, ΔP2: {dr:+d}"
        )

        # „Nächste Runde“-Button
        if st.button("Nächste Runde"):
            new_round()
            st.rerun()

    # ---------------- CSV-Download (Logs) ----------------
    st.markdown("### Runden-Log")
    if st.session_state.logs:
        df_log = pd.DataFrame(st.session_state.logs)
        st.dataframe(df_log, use_container_width=True, hide_index=True)
        csv_buf = StringIO()
        df_log.to_csv(csv_buf, index=False)
        st.download_button(
            "Log als CSV herunterladen",
            data=csv_buf.getvalue(),
            file_name="playing_eyes_log.csv",
            mime="text/csv"
        )
    else:
        st.info("Noch keine Runden geloggt. Spiele eine Runde 🙂")


def render_cards(cards, reveal: bool, caption=("Karte 1", "Karte 2")):
    """Zeigt 2 Kartenbilder, falls reveal=True; sonst '?? ??'."""
    if reveal:
        imgs = [img for img in (load_card_image(cards[0]), load_card_image(cards[1])) if img is not None]
        if imgs:
            st.image(imgs, caption=list(caption), width=100)
        else:
            st.write("**Karten:** (Bilder fehlen)")
    else:
        st.write("**Karten:** ??  ??")

if __name__ == "__main__":
    random.seed()
    main()
