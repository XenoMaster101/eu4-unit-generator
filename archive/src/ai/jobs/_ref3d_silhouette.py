"""The horse's layout read from its silhouette and its material map (Track M, SPEC M18); numpy and
OpenCV only (the AI environment's jobs: ref3d_parts; the landmarks reuse it).

Everything is worked out with the head on the image's left (a horse facing right is flipped first
and the results flipped back):
  facing   the end whose outer fifth reaches higher (the ears over the croup);
  opened   the horse mask opened by ``open_frac`` of its width: straps and reins gone;
  belly    from the ground up, the first row whose longest run of the opened mask is at least
           ``trunk_run`` of the width (a caparison's hem when there is one: the legs are what shows below);
  back     the opened mask's top over the middle fifth;
  tail     the largest hair region behind ``tail_from`` of the width;
  chest    the front-most opened pixel between the back and the belly's lower half; buttock: the
           rear-most one, the tail left out;
  head     the opened mask in front of ``muzzle + head_len`` x the width, above the belly;
  legs     below the belly, the tail left out, split front and back at the middle.
"""
import numpy as np


def _cv():
    import cv2
    return cv2


def _box(m):
    ys, xs = np.nonzero(m)
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1] if len(xs) else None


def _polys(m, eps_frac=0.004):
    cv2 = _cv()
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in sorted(cs, key=cv2.contourArea, reverse=True):
        if cv2.contourArea(c) < 0.02 * m.sum():
            continue
        c = cv2.approxPolyDP(c, eps_frac * max(m.shape), True)
        if len(c) >= 3:
            out.append(c.reshape(-1, 2).astype(int).tolist())
    return out


def _largest(m):
    cv2 = _cv()
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
    if n <= 1:
        return np.zeros_like(m)
    return lab == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))


def facing_left(fg):
    x0, y0, x1, y1 = _box(fg)
    w = x1 - x0
    left = fg[:, x0:x0 + w // 5]
    right = fg[:, x1 - w // 5:x1]
    top = lambda m: np.nonzero(m.any(axis=1))[0].min()  # noqa: E731
    return top(left) <= top(right)


def analyse(fg, mats, rgb, st):
    """-> (classes: object array, zones: {zone: [polygon]}, parts: {part: {...}}, layout summary)."""
    cv2 = _cv()
    left = facing_left(fg)
    flip = (lambda a: a) if left else (lambda a: a[:, ::-1])
    F = flip(fg)
    M = {k: flip(v) for k, v in mats.items()}
    R = flip(rgb)
    H, W = F.shape
    x0, y0, x1, y1 = _box(F)
    w, h = x1 - x0, y1 - y0
    r = max(1, int(st["open_frac"] * w))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    opened = cv2.morphologyEx(F.astype(np.uint8), cv2.MORPH_OPEN, k).astype(bool)
    opened = _largest(opened)
    # tail: the largest hair region behind tail_from of the width
    hair = M.get("hair", np.zeros_like(F))
    rear = np.zeros_like(F)
    rear[:, x0 + int(st["tail_from"] * w):] = True
    tail = _largest(hair & rear)
    if tail.sum() < st["tail_min"] * F.sum():
        tail = np.zeros_like(F)
    else:  # closed and grown a little: the buttock found on its holes was the tail's edge (horse 2: 11%)
        tail = cv2.morphologyEx(tail.astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4 * r + 1, 4 * r + 1)))
        tail = cv2.dilate(tail, k).astype(bool) & F
    body = opened & ~tail

    def longest_run(row):
        if not row.any():
            return 0
        d = np.diff(np.concatenate([[0], row.astype(np.int8), [0]]))
        s, e = np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]
        return int((e - s).max())

    runs = np.array([longest_run(body[y]) for y in range(H)])
    belly = y1
    for y in range(y1 - 1, y0, -1):
        if runs[y] >= st["trunk_run"] * w:
            belly = y
            break
    mid = slice(x0 + int(0.4 * w), x0 + int(0.6 * w))
    cols = body[:, mid].any(axis=1)
    back = int(np.nonzero(cols)[0].min())
    lo, hi = back + int(0.7 * (belly - back)), belly - int(0.05 * (belly - back))  # below the breast collar
    xs_front = [np.nonzero(body[y])[0].min() for y in range(lo, hi) if body[y].any()]
    chest = int(min(xs_front)) if xs_front else x0
    lo2, hi2 = back + int(0.2 * (belly - back)), back + int(0.7 * (belly - back))
    xs_rear = [np.nonzero(body[y])[0].max() for y in range(lo2, hi2) if body[y].any()]
    buttock = int(max(xs_rear)) if xs_rear else x1
    muzzle = int(np.nonzero(opened.any(axis=0))[0].min())
    head = opened.copy()
    head[:, muzzle + int(st["head_len"] * w):] = False
    head[belly:] = False
    head = _largest(head)
    legs = F & ~tail
    legs[:belly] = False
    xm = (chest + buttock) // 2
    fore, hind = legs.copy(), legs.copy()
    fore[:, xm:] = False
    hind[:, :xm] = False
    trunk = F.copy()
    trunk[:back] = False
    trunk[belly:] = False
    trunk[:, :chest] = False
    trunk[:, buttock:] = False
    neck = opened & ~head & ~tail
    neck[:, chest + int(0.05 * w):] = False
    neck[belly:] = False

    # classes
    cls = np.full(F.shape, "unclassified", dtype=object)
    name = {"metal": "scale", "gold": "trim", "cloth": "cloth", "leather": "leather", "coat": "coat", "hair": "coat",
            "plume": "cloth"}
    for m, c in name.items():
        if m in M:
            cls[M[m] & F] = c
    cls[tail & F] = "tail"
    plume_zone = np.zeros_like(F)
    hb = _box(head)
    if hb:
        pz = (max(0, hb[0] - int(0.05 * w)), max(0, hb[1] - int(0.15 * h)), hb[2] + int(0.05 * w), hb[1] + int(0.5 * (hb[3] - hb[1])))
        plume_zone[pz[1]:pz[3], pz[0]:pz[2]] = True
    plume = M.get("plume", np.zeros_like(F)) & F & plume_zone
    cls[plume] = "plume"
    hm = head & F
    head_metal = float((M.get("metal", np.zeros_like(F)) & hm).sum() / max(hm.sum(), 1))
    cls[hm & (cls == "scale")] = "chanfron"
    if head_metal >= st["chanfron_min"]:
        cls[hm & (cls == "trim")] = "chanfron"
    leg_zone = np.zeros_like(F)
    for part in (fore, hind):
        b = _box(part)
        if b:
            pad = int(0.02 * w)
            leg_zone[belly:y1, max(0, b[0] - pad):b[2] + pad] = True
    # below the belly (a caparison's hem) the legs are coat: lit grey coat read as metal (cataphract S
    # v013: the hind legs as scale); the master builds no leg armour
    cls[leg_zone & F & np.isin(cls, ("coat", "scale"))] = "leg"

    # parts
    nF = max(int(F.sum()), 1)

    def part(mask, present, **extra):
        b = _box(mask)
        cols = {}
        for c in ("cloth", "trim", "leather", "scale", "chanfron", "coat", "plume", "tail", "leg"):
            mm = mask & (cls == c)
            if mm.sum() >= 50:
                cols[c] = np.median(R[mm], axis=0).astype(int).tolist()
        return {"present": bool(present), "share": round(float(mask.sum() / nF), 4), "box": b, "colours": cols, **extra}

    parts = {}
    armour = (cls == "scale") & ~head
    zone = (trunk | neck) & F
    parts["armour"] = part(armour, armour.sum() >= st["armour_min"] * max(zone.sum(), 1),
                           of_body=round(float((armour & zone).sum() / max(zone.sum(), 1)), 4))
    ch = cls == "chanfron"
    parts["chanfron"] = part(ch, head_metal >= st["chanfron_min"], of_head=round(head_metal, 4))
    textile = np.isin(cls, ("cloth", "trim")) & ~head
    textile = cv2.morphologyEx(textile.astype(np.uint8), cv2.MORPH_CLOSE, k).astype(bool) & F
    n, lab_, stats, _ = cv2.connectedComponentsWithStats(textile.astype(np.uint8), 8)
    capar, sc = np.zeros_like(F), np.zeros_like(F)
    deep = belly - st["hem_reach"] * (belly - back)
    comps = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < st["cloth_min"] * nF:
            continue
        m = lab_ == i
        b = _box(m)
        hangs = b[3] >= deep
        long_ = (b[2] - b[0]) >= st["capar_span"] * (buttock - chest)
        kind = "caparison" if (hangs or long_) else "saddle_cloth"
        (capar if kind == "caparison" else sc)[m] = True
        comps.append({"kind": kind, "box": b, "share": round(float(m.sum() / nF), 4), "hangs_below_belly": bool(hangs)})
    parts["caparison"] = part(capar, capar.any(), pieces=[c for c in comps if c["kind"] == "caparison"])
    parts["saddle_cloth"] = part(sc, sc.any(), pieces=[c for c in comps if c["kind"] == "saddle_cloth"])
    leather = cls == "leather"
    seat = np.zeros_like(F)
    seat[:back + int(0.45 * (belly - back)), chest + int(0.25 * (buttock - chest)):chest + int(0.75 * (buttock - chest))] = True
    saddle = _largest(leather & seat)
    parts["saddle"] = part(saddle, saddle.sum() >= st["saddle_min"] * nF)
    br = leather & (head | neck)
    parts["bridle"] = part(br, br.sum() >= st["bridle_min"] * max(hm.sum(), 1))
    parts["plume"] = part(plume, plume.sum() >= st["plume_min"] * nF)
    coat = np.isin(cls, ("coat", "leg")) & zone
    parts["bare_coat"] = part(coat, coat.sum() >= st["bare_min"] * max(zone.sum(), 1),
                              of_body=round(float(coat.sum() / max(zone.sum(), 1)), 4))
    parts["tail"] = part(tail & F, tail.sum() >= st["tail_min"] * nF)

    # back to the image's frame
    if not left:
        cls = cls[:, ::-1]

        def fx(b):
            return None if b is None else [W - b[2], b[1], W - b[0], b[3]]
        for p in parts.values():
            p["box"] = fx(p["box"])
            for c in p.get("pieces", []):
                c["box"] = fx(c["box"])
    unflip = (lambda m: m) if left else (lambda m: m[:, ::-1])
    zones = {"head": _polys(unflip(head)), "legs": [q for m in (fore, hind) for q in _polys(unflip(_leg_rect(m, belly, y1, w)))],
             "tail": _polys(unflip(tail)), "trunk": _polys(unflip(trunk))}
    if plume.any():
        zones["plume"] = _polys(unflip(cv2.dilate(plume.astype(np.uint8), k).astype(bool)))
    X = (lambda x: int(x)) if left else (lambda x: int(W - 1 - x))
    summary = {"facing": "left" if left else "right", "box": _box(fg), "back_y": back, "belly_y": int(belly),
               "chest_x": X(chest), "buttock_x": X(buttock), "muzzle_x": X(muzzle), "head_box": None,
               "open_px": r}
    if hb:
        summary["head_box"] = [X(hb[0]), hb[1], X(hb[2]), hb[3]] if left else [X(hb[2] - 1), hb[1], X(hb[0]) + 1, hb[3]]
    # the ears from what can be an ear (not cloth, trim, plume or tail): cataphract S's plume, partly cloth, read as
    # ears; the trunk's landmarks from the horse without its hanging cloth (its chest and buttock were the caparison's)
    ear_px = F & ~np.isin(cls, ("cloth", "trim", "plume", "tail"))  # (coat only lost horse 2's ear under the bridle)
    bare = cv2.morphologyEx((F & ~np.isin(cls, ("cloth", "trim", "tail", "plume"))).astype(np.uint8), cv2.MORPH_OPEN, k).astype(bool) & ~tail
    lms = landmarks(F & ~plume, ear_px, bare, tail, head, belly, back, chest, buttock, (x0, y0, x1, y1), st)
    summary["landmarks"] = {k: [X(x), int(y)] for k, (x, y) in lms.items()}
    return cls, zones, parts, summary


def landmarks(F, ear_px, body, tail, head, belly, back, chest, buttock, box, st):
    """The template's landmarks a silhouette shows (head on the left; image px): the head's outline (muzzle,
    ears, poll, chin, throat), the trunk's (chest_front, buttock, croup), the feet (hooves on the ground, the
    lower foot of a pair the near leg; fetlocks ``fetlock_up`` x (ground - back) above), the tail's top and
    tip. Not the eye, nostril or bit ring: no outline shows them (the fitted body gives those)."""
    x0, y0, x1, y1 = box
    w = x1 - x0
    out = {}
    ys = lambda m, x: np.nonzero(m[:, x])[0]  # noqa: E731
    # head: the muzzle (front-most), ears (the two highest peaks of the top outline), poll (the dip behind them)
    hcols = np.nonzero(head.any(axis=0))[0]
    if len(hcols):
        top = np.array([ys(ear_px, x).min() if ear_px[:, x].any() else F.shape[0] for x in hcols])  # ears stand above the opened head
        xm = int(hcols.min())
        yfront = np.nonzero(head[:, xm])[0]
        out["muzzle_front"] = (xm, int(yfront.mean()))
        sep = max(3, int(0.012 * w))
        order = np.argsort(top)
        peaks = []
        for i in order:
            if all(abs(hcols[i] - hcols[j]) > sep for j in peaks):
                peaks.append(i)
            if len(peaks) == 2:
                break
        if peaks and top[peaks[0]] < out["muzzle_front"][1]:
            ea = (int(hcols[peaks[0]]), int(top[peaks[0]]))
            eb = (int(hcols[peaks[1]]), int(top[peaks[1]])) if len(peaks) > 1 and top[peaks[1]] < top[peaks[0]] + 0.06 * w else ea
            out["ear_tip_a"], out["ear_tip_b"] = ea, eb
            # the poll: under the rear ear, as low as the notch between the ears (a dip search behind them reached
            # the mane: horse 2, 8%; the top just behind the ear was the ear's own base)
            lo_, hi_ = sorted((ea[0], eb[0]))
            notch = top[(hcols >= lo_) & (hcols <= hi_)]
            ny = int(notch.max()) if len(notch) and hi_ - lo_ > sep else int(top[peaks[0]] + 0.04 * w)
            out["poll"] = (int(hi_), ny)
        # chin: the lowest point of the head's front half; throat: where the lower outline turns to the neck
        bot = np.array([ys(F & head, x).max() if (F[:, x] & head[:, x]).any() else -1 for x in hcols])
        half = hcols < hcols.min() + 0.5 * (hcols.max() - hcols.min())
        if half.any():
            j = int(np.argmax(np.where(half, bot, -1)))
            out["chin"] = (int(hcols[j]), int(bot[j]))
            rest = ~half & (bot >= 0)
            if rest.any():
                j = int(np.argmin(np.where(rest, bot, 1e9)))  # the highest point of the lower outline behind the jaw
                out["throat"] = (int(hcols[j]), int(bot[j]))
    # trunk
    band = range(back + int(0.7 * (belly - back)), belly - int(0.05 * (belly - back)))
    fr = [(int(np.nonzero(body[y])[0].min()), y) for y in band if body[y].any()]
    if fr:
        out["chest_front"] = min(fr)
    band = range(back + int(0.2 * (belly - back)), back + int(0.7 * (belly - back)))
    rr = [(int(np.nonzero(body[y])[0].max()), y) for y in band if body[y].any()]
    if rr:
        x, y = max(rr)
        out["buttock"] = (x, y)
    xs = np.arange(int(buttock - 0.25 * (buttock - chest)), buttock)
    tops = [(ys(body, x).min(), x) for x in xs if body[:, x].any()]
    if tops:
        y, x = min(tops)
        out["croup"] = (int(x), int(y))
    # feet: the columns whose bottom reaches the ground, in runs; per half (fore, hind) the two lowest runs
    legs = F & ~tail
    bottom = np.array([ys(legs, x).max() if legs[:, x].any() else -1 for x in range(F.shape[1])])
    ground = int(bottom.max())
    near_ground = bottom >= ground - st["foot_band"] * (ground - back)
    xm = (chest + buttock) // 2
    runs = []
    x = 0
    while x < F.shape[1]:
        if near_ground[x]:
            s_ = x
            while x < F.shape[1] and near_ground[x]:
                x += 1
            runs.append((s_, x))
        else:
            x += 1
    for name, sel in (("fore", lambda r: (r[0] + r[1]) / 2 < xm), ("hind", lambda r: (r[0] + r[1]) / 2 >= xm)):
        rs = [r for r in runs if sel(r) and r[1] - r[0] >= 0.01 * w]
        rs = sorted(rs, key=lambda r: -bottom[r[0]:r[1]].max())[:2]
        if not rs:
            continue
        feet = []
        for a_, b_ in rs:
            yb = int(bottom[a_:b_].max())
            feet.append(((a_ + b_) // 2, yb))
        if len(feet) == 1:
            feet.append(feet[0])  # one foot hides the other
        for side, (fx, fy) in zip(("near", "far"), feet):
            out[f"{name}_{side}_hoof"] = (int(fx), int(fy))
            yf = int(fy - st["fetlock_up"] * (ground - back))
            row = np.nonzero(legs[yf])[0]
            if len(row):
                # the leg's run at that height nearest the hoof
                d = np.diff(np.r_[0, legs[yf].astype(np.int8), 0])
                s0, e0 = np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]
                k = int(np.argmin(np.abs((s0 + e0) / 2 - fx)))
                out[f"{name}_{side}_fetlock"] = (int((s0[k] + e0[k]) // 2), yf)
    if tail.any():
        ty, tx = np.nonzero(tail)
        i = int(np.argmin(ty))
        out["tail_top_visible"] = (int(tx[i]), int(ty[i]))
        i = int(np.argmax(ty))
        out["tail_tip"] = (int(tx[i]), int(ty[i]))
    return out


def _leg_rect(m, belly, y1, w):
    b = _box(m)
    out = np.zeros_like(m)
    if b:
        pad = int(0.02 * w)
        out[belly:y1, max(0, b[0] - pad):b[2] + pad] = True
    return out
