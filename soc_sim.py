"""SOC simulation: async consumption, network-coupled failure."""
import numpy as np
import networkx as nx


def run_simulation(N=500, k=8, p_rewire=0.1, T=15000, seed=0,
                   consume_period=10, respawn_delay=200,
                   drop_rate=None, initial_fruit=8):
    """
    Run one simulation. Returns a list of (tick, agent_id) death events.

    Parameters
    ----------
    N              : number of agents
    k              : mean degree of Watts-Strogatz graph
    p_rewire       : rewiring probability (small-world parameter)
    T              : total ticks
    seed           : RNG seed
    consume_period : each agent eats 1 apple + 1 orange every this many ticks
    respawn_delay  : dead agents respawn after this many ticks
    drop_rate      : mean fruits dropped per tick (Poisson). If None, auto-set
                     to 92% of population consumption rate.
    initial_fruit  : initial fruit endowment (randomized around this value)
    """
    if drop_rate is None:
        drop_rate = 2 * N / consume_period

    rng = np.random.default_rng(seed)
    G = nx.watts_strogatz_graph(N, k, p_rewire, seed=seed)
    neighbors = [list(G.neighbors(i)) for i in range(N)]

    apples  = rng.integers(1, 5, size=N).astype(float)
    oranges = rng.integers(1, 5, size=N).astype(float)
    alive = np.ones(N, dtype=bool)
    next_consume = rng.integers(1, consume_period + 1, size=N)
    respawn_at = np.full(N, -1, dtype=int)

    deaths = []

    for t in range(T):
        # --- Respawn dead agents whose delay has elapsed ---
        mask = (~alive) & (respawn_at == t)
        if mask.any():
            idx = np.where(mask)[0]
            apples[idx]  = rng.integers(initial_fruit // 2, initial_fruit * 2, size=len(idx))
            oranges[idx] = rng.integers(initial_fruit // 2, initial_fruit * 2, size=len(idx))
            alive[idx] = True
            next_consume[idx] = t + rng.integers(1, consume_period + 1, size=len(idx))

        # --- Slow drive: Poisson fruit drops onto random alive agents ---
        if t>0 and t%10==0:
            n_drops = rng.poisson(drop_rate*0.9)
            alive_ids = np.where(alive)[0]
            if n_drops > 0 and len(alive_ids) > 0:
                tgts = rng.choice(alive_ids, size=n_drops)
                is_apple = rng.random(n_drops) < 0.5
                for tgt, a in zip(tgts, is_apple):
                    if a:
                        apples[tgt] += 1
                    else:
                        oranges[tgt] += 1

            # --- Local trade: each agent tries to swap with best neighbor ---
            rng.shuffle(alive_ids)
            for i in alive_ids:
                if not alive[i]:
                    continue
                d_i = apples[i] - oranges[i]
                if abs(d_i) < 2:
                    continue
                nbrs = [j for j in neighbors[i] if alive[j]]
                if not nbrs:
                    continue
                best, best_abs = -1, 0
                for j in nbrs:
                    d_j = apples[j] - oranges[j]
                    if d_i * d_j < 0 and abs(d_j) > best_abs:
                        best, best_abs = j, abs(d_j)
                if best < 0:
                    continue
                j = best
                if d_i > 0:
                    if apples[i] >= 1 and oranges[j] >= 1:
                        apples[i] -= 1
                        oranges[i] += 1
                        oranges[j] -= 1
                        apples[j] += 1
                else:
                    if oranges[i] >= 1 and apples[j] >= 1:
                        oranges[i] -= 1
                        apples[i] += 1
                        apples[j] -= 1
                        oranges[j] += 1

        # --- Asynchronous consumption ---
        due = alive & (next_consume == t)
        for i in np.where(due)[0]:
            if apples[i] >= 1 and oranges[i] >= 1:
                apples[i] -= 1
                oranges[i] -= 1
                next_consume[i] = t + consume_period
            else:
                deaths.append((t, int(i)))
                alive[i] = False
                respawn_at[i] = t + respawn_delay
                apples[i] = 0
                oranges[i] = 0

    return deaths


def extract_cascades(deaths, window=1):
    """Group deaths where successive events are within `window` ticks."""
    if not deaths:
        return []
    deaths = sorted(deaths)
    cascades, cur, last = [], 1, deaths[0][0]
    for (t, _) in deaths[1:]:
        if t - last <= window:
            cur += 1
        else:
            cascades.append(cur)
            cur = 1
        last = t
    cascades.append(cur)
    return cascades


def fit_powerlaw(sizes, xmin=2):
    """MLE discrete power-law exponent (Clauset et al. 2009)."""
    sizes = np.array([s for s in sizes if s >= xmin])
    n = len(sizes)
    if n < 10:
        return None, n
    alpha = 1 + n / np.sum(np.log(sizes / (xmin - 0.5)))
    return alpha, n


def logbin_pdf(sizes, nbins=20):
    """Log-binned PDF for clean log-log plots."""
    sizes = np.array(sizes)
    sizes = sizes[sizes > 0]
    if len(sizes) == 0:
        return np.array([]), np.array([])
    bins = np.logspace(0, np.log10(sizes.max() + 1), nbins)
    hist, edges = np.histogram(sizes, bins=bins)
    widths = np.diff(edges)
    centers = np.sqrt(edges[:-1] * edges[1:])
    pdf = hist / (widths * len(sizes))
    m = hist > 0
    return centers[m], pdf[m];