#!/usr/bin/env python3
import math
import random
import signal
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple
from urllib.request import urlopen


def load_text_corpus(max_chars: int = 25000) -> str:
    urls = [
        "https://www.gutenberg.org/files/11/11-0.txt",
        "https://www.gutenberg.org/cache/epub/1661/pg1661.txt",
    ]
    for u in urls:
        try:
            with urlopen(u, timeout=5) as r:
                raw = r.read().decode("utf-8", errors="ignore")
            t = " ".join(raw.split())
            if len(t) > 1000:
                return t[:max_chars]
        except Exception:
            pass
    base = (
        "Mathematics describes structure. Evolution explores parameter spaces. "
        "Neural systems infer patterns from history and text. "
    )
    return (base * (max_chars // len(base) + 1))[:max_chars]


def build_char_vocab(text: str, max_vocab: int = 64) -> Tuple[List[str], Dict[str, int]]:
    chars = sorted(set(text))
    if len(chars) > max_vocab:
        freq = {c: text.count(c) for c in chars}
        chars = sorted(chars, key=lambda c: freq[c], reverse=True)[: max_vocab - 1] + ["?"]
    return chars, {c: i for i, c in enumerate(chars)}


def encode_char(c: str, stoi: Dict[str, int]) -> int:
    return stoi[c] if c in stoi else stoi.get("?", 0)


def sample_batch(text: str, stoi: Dict[str, int], n: int = 64):
    xs, ys = [], []
    for _ in range(n):
        i = random.randint(0, len(text) - 2)
        xs.append(encode_char(text[i], stoi))
        ys.append(encode_char(text[i + 1], stoi))
    return xs, ys


def sample_numeric_batch(n: int = 128):
    xs, ys = [], []
    for _ in range(n):
        x1 = random.uniform(-3, 3)
        x2 = random.uniform(-3, 3)
        x3 = random.uniform(-3, 3)
        y = math.sin(1.7 * x1) + 0.25 * (x2 ** 3) - 0.4 * x3 + 0.3 * math.cos(x1 * x3)
        xs.append([x1, x2, x3])
        ys.append(y)
    return xs, ys


def sample_timeseries_batch(n: int = 128):
    xs, ys = [], []
    for _ in range(n):
        t = random.uniform(0, 30)
        y = 0.65 * math.sin(0.45 * t) + 0.25 * math.sin(1.8 * t + 0.2) + 0.1 * math.cos(0.1 * (t ** 1.2))
        xs.append([t, math.sin(t), math.cos(t)])
        ys.append(y)
    return xs, ys


ARCHS = ["mlp", "poly", "fourier", "rbf", "gated"]
ACTS = ["tanh", "relu", "sin"]


def act(name: str, z: float) -> float:
    if name == "tanh":
        return math.tanh(z)
    if name == "relu":
        return z if z > 0 else 0.0
    return math.sin(z)


def sigmoid(z: float) -> float:
    if z >= 0:
        ez = math.exp(-z)
        return 1 / (1 + ez)
    ez = math.exp(z)
    return ez / (1 + ez)


@dataclass
class Individual:
    arch: str
    act: str
    hidden: int
    sigma: float
    offspring_k: int
    local_steps: int
    lr: float
    params: List[float]
    fitness: float = -1e18


PARAM_COUNTS = {
    "mlp": lambda d, h, v: d * h + h + h + 1 + v * h + v,
    "poly": lambda d, h, v: d * 3 + 3 + v * 3 + v,
    "fourier": lambda d, h, v: d * h + h + h + 1 + v * h + v,
    "rbf": lambda d, h, v: d * h + h + h + 1 + v * h + v,
    "gated": lambda d, h, v: d * h + h + d * h + h + h + 1 + v * h + v,
}


def randn(scale=1.0):
    return random.gauss(0.0, scale)


def rand_vector(n, scale=0.4):
    return [randn(scale) for _ in range(n)]


def rand_individual(d_in=3, vocab=32):
    arch = random.choice(ARCHS)
    hidden = random.randint(6, 24)
    n = PARAM_COUNTS[arch](d_in, hidden, vocab)
    return Individual(
        arch=arch,
        act=random.choice(ACTS),
        hidden=hidden,
        sigma=10 ** random.uniform(-2.2, -0.4),
        offspring_k=random.randint(2, 8),
        local_steps=random.randint(1, 4),
        lr=10 ** random.uniform(-2.5, -0.8),
        params=rand_vector(n, 0.4),
    )


def unpack(ind: Individual, d_in: int, vocab: int):
    p, h, i = ind.params, ind.hidden, 0

    def take(n):
        nonlocal i
        s = p[i : i + n]
        i += n
        return s

    def mat(rows, cols, arr):
        return [arr[r * cols : (r + 1) * cols] for r in range(rows)]

    if ind.arch == "poly":
        A = mat(d_in, 3, take(d_in * 3))
        c = take(3)
        Wtxt = mat(vocab, 3, take(vocab * 3))
        btxt = take(vocab)
        return {"A": A, "c": c, "Wtxt": Wtxt, "btxt": btxt}

    if ind.arch == "mlp":
        W1 = mat(d_in, h, take(d_in * h))
        b1 = take(h)
    elif ind.arch in ("fourier", "rbf"):
        W1 = mat(d_in, h, take(d_in * h))
        b1 = take(h)
    else:
        Wm = mat(d_in, h, take(d_in * h))
        bm = take(h)
        Wg = mat(d_in, h, take(d_in * h))
        bg = take(h)

    W2 = take(h)
    b2 = take(1)[0]
    Wtxt = mat(vocab, h, take(vocab * h))
    btxt = take(vocab)

    if ind.arch in ("mlp", "fourier", "rbf"):
        return {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "Wtxt": Wtxt, "btxt": btxt}
    return {"Wm": Wm, "bm": bm, "Wg": Wg, "bg": bg, "W2": W2, "b2": b2, "Wtxt": Wtxt, "btxt": btxt}


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def row_apply(x, W):
    # x:[d], W:[d][h] -> [h]
    h = len(W[0])
    out = [0.0] * h
    for j in range(h):
        s = 0.0
        for i in range(len(x)):
            s += x[i] * W[i][j]
        out[j] = s
    return out


def numeric_forward(ind: Individual, x: List[float], parts: dict) -> float:
    if ind.arch == "poly":
        z = [dot(x, [parts["A"][i][k] for i in range(3)]) + parts["c"][k] for k in range(3)]
        return z[0] + z[1] ** 2 + z[2] ** 3

    if ind.arch in ("mlp", "fourier", "rbf"):
        z = row_apply(x, parts["W1"])
        z = [zj + bj for zj, bj in zip(z, parts["b1"])]
        if ind.arch == "mlp":
            h = [act(ind.act, v) for v in z]
        elif ind.arch == "fourier":
            h = [math.sin(v) for v in z]
        else:
            h = [math.exp(-(v * v)) for v in z]
        return dot(h, parts["W2"]) + parts["b2"]

    m = row_apply(x, parts["Wm"])
    g = row_apply(x, parts["Wg"])
    m = [act(ind.act, v + b) for v, b in zip(m, parts["bm"])]
    g = [sigmoid(v + b) for v, b in zip(g, parts["bg"])]
    h = [mi * gi for mi, gi in zip(m, g)]
    return dot(h, parts["W2"]) + parts["b2"]


def text_logits(ind: Individual, x_idx: int, parts: dict, vocab: int) -> List[float]:
    one = [0.0] * vocab
    one[x_idx] = 1.0

    if ind.arch == "poly":
        z = [dot(one, [parts["Wtxt"][i][k] for i in range(vocab)]) for k in range(3)]
        feat = [z[0], z[1] ** 2, z[2] ** 3]
        return [dot(feat, [0.0, 0.0, 0.0]) + parts["btxt"][j] for j in range(vocab)]

    h0 = row_apply(one, parts["Wtxt"])
    if ind.arch == "mlp":
        h = [act(ind.act, v) for v in h0]
    elif ind.arch == "fourier":
        h = [math.sin(v) for v in h0]
    elif ind.arch == "rbf":
        h = [math.exp(-(v * v)) for v in h0]
    else:
        m = [act(ind.act, v) for v in h0]
        g = [sigmoid(v) for v in h0]
        h = [mi * gi for mi, gi in zip(m, g)]

    logits = []
    for j in range(vocab):
        col = [parts["Wtxt"][j][k] for k in range(len(h))]
        logits.append(dot(h, col) + parts["btxt"][j])
    return logits


def softmax_ce(logits_batch: List[List[float]], y_batch: List[int]) -> float:
    s = 0.0
    for logits, y in zip(logits_batch, y_batch):
        m = max(logits)
        exps = [math.exp(v - m) for v in logits]
        z = sum(exps) + 1e-12
        py = exps[y] / z
        s += -math.log(py + 1e-12)
    return s / len(y_batch)


def evaluate(ind: Individual, text: str, stoi: Dict[str, int], vocab: int) -> float:
    parts = unpack(ind, 3, vocab)

    xn, yn = sample_numeric_batch(80)
    mse_n = sum((numeric_forward(ind, x, parts) - y) ** 2 for x, y in zip(xn, yn)) / len(xn)

    xt, yt = sample_timeseries_batch(80)
    mse_t = sum((numeric_forward(ind, x, parts) - y) ** 2 for x, y in zip(xt, yt)) / len(xt)

    xs, ys = sample_batch(text, stoi, 50)
    logits = [text_logits(ind, xi, parts, vocab) for xi in xs]
    ce = softmax_ce(logits, ys)

    l2 = sum(v * v for v in ind.params)
    return -(0.5 * mse_n + 0.3 * mse_t + 0.2 * ce + 1e-4 * l2)


def clone(ind: Individual) -> Individual:
    return Individual(ind.arch, ind.act, ind.hidden, ind.sigma, ind.offspring_k, ind.local_steps, ind.lr, ind.params[:], ind.fitness)


def local_search(ind: Individual, text: str, stoi: Dict[str, int], vocab: int) -> Individual:
    best = clone(ind)
    best_fit = evaluate(best, text, stoi, vocab)
    for _ in range(ind.local_steps):
        cand = clone(best)
        cand.params = [v + randn(ind.lr) for v in cand.params]
        fit = evaluate(cand, text, stoi, vocab)
        if fit > best_fit:
            best, best_fit = cand, fit
    best.fitness = best_fit
    return best


def mutate(ind: Individual, vocab: int) -> Individual:
    c = clone(ind)
    if random.random() < 0.15:
        c.arch = random.choice(ARCHS)
    if random.random() < 0.2:
        c.act = random.choice(ACTS)
    if random.random() < 0.25:
        c.hidden = max(4, min(48, c.hidden + random.randint(-3, 3)))

    c.sigma = max(1e-3, min(1.2, c.sigma * math.exp(randn(0.15))))
    c.offspring_k = max(1, min(12, int(round(c.offspring_k + randn(0.8)))))
    c.local_steps = max(1, min(8, int(round(c.local_steps + randn(0.6)))))
    c.lr = max(1e-4, min(0.3, c.lr * math.exp(randn(0.2))))

    need = PARAM_COUNTS[c.arch](3, c.hidden, vocab)
    if len(c.params) != need:
        newp = rand_vector(need, 0.3)
        for i in range(min(need, len(c.params))):
            newp[i] = c.params[i]
        c.params = newp

    c.params = [v + randn(c.sigma) for v in c.params]
    return c


def crossover(a: Individual, b: Individual, vocab: int) -> Individual:
    child = clone(a if random.random() < 0.5 else b)
    child.arch = random.choice([a.arch, b.arch])
    child.act = random.choice([a.act, b.act])
    child.hidden = random.choice([a.hidden, b.hidden])

    need = PARAM_COUNTS[child.arch](3, child.hidden, vocab)
    child.params = [0.0] * need
    for i in range(need):
        src = a.params if (i % 2 == 0) else b.params
        child.params[i] = src[i % len(src)]

    child.sigma = max(1e-3, min(1.2, 0.5 * (a.sigma + b.sigma)))
    child.offspring_k = max(1, min(12, int(round(0.5 * (a.offspring_k + b.offspring_k)))))
    child.local_steps = max(1, min(8, int(round(0.5 * (a.local_steps + b.local_steps)))))
    child.lr = max(1e-4, min(0.3, 0.5 * (a.lr + b.lr)))
    return child


def formula(ind: Individual) -> str:
    if ind.arch == "mlp":
        return "f(x)=W2·" + ind.act + "(W1x+b1)+b2"
    if ind.arch == "poly":
        return "f(x)=z1+z2^2+z3^3, z=Ax+c"
    if ind.arch == "fourier":
        return "f(x)=W2·sin(Ωx+φ)+b2"
    if ind.arch == "rbf":
        return "f(x)=W2·exp(-(Ωx+φ)^2)+b2"
    return "f(x)=W2·(σ(Wgx+bg)⊙" + ind.act + "(Wmx+bm))+b2"


def evolve_forever(seed: int = 42):
    random.seed(seed)
    text = load_text_corpus()
    chars, stoi = build_char_vocab(text)
    vocab = len(chars)
    pop = [rand_individual(vocab=vocab) for _ in range(16)]

    stop = {"flag": False}

    def on_sigint(_s, _f):
        stop["flag"] = True

    signal.signal(signal.SIGINT, on_sigint)

    gen = 0
    print(f"START | vocab={vocab} | population={len(pop)}")
    while not stop["flag"]:
        gen += 1
        pop = [local_search(i, text, stoi, vocab) for i in pop]
        pop.sort(key=lambda z: z.fitness, reverse=True)
        elite = pop[: max(2, len(pop) // 4)]
        best = elite[0]

        print(
            f"gen={gen:06d} | best={best.fitness:.6f} | arch={best.arch:<7} | "
            f"h={best.hidden:<2d} | sigma={best.sigma:.4f} | lambda={best.offspring_k:<2d} | "
            f"steps={best.local_steps:<2d} | lr={best.lr:.5f}"
        )
        print("BEST_FORMULA:", formula(best))

        target = max(12, min(80, sum(e.offspring_k for e in elite)))
        new_pop = [clone(best), clone(random.choice(elite))]
        while len(new_pop) < target:
            if random.random() < 0.45:
                child = mutate(random.choice(elite), vocab)
            else:
                p1, p2 = random.sample(elite, 2)
                child = mutate(crossover(p1, p2, vocab), vocab)
            new_pop.append(child)
        pop = new_pop
        time.sleep(0.03)

    print("Stopped by user")


if __name__ == "__main__":
    evolve_forever()
